# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import os
import time

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import torch
from torch.tensor import Tensor
from typing import Tuple, Dict


class MIT_humanoid(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_scale = self.cfg["env"]["learn"]["heightScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["smooth"] = self.cfg["env"]["learn"]["smoothRewardScale"]
        self.rew_scales["height"] = self.cfg["env"]["learn"]["heightRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.dt = sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        # * observation
        self.cfg["env"]["numObservations"] = 67 + 18*2 + 2  # 6+18*2 + 18*2
        self.cfg["env"]["numActions"] = 18

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs,
                                    3, dtype=torch.float, device=self.device,
                                    requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos,
                                                dtype=torch.float,
                                                device=self.device,
                                                requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state,
                                               device=self.device,
                                               requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx),
                                    device=self.device).repeat((self.num_envs,
                                                                1))
        self.actions = torch.zeros(self.num_envs,
                                   self.num_actions,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # action history
        self.actions_k1 = self.actions.clone()
        self.actions_k2 = self.actions.clone()

        # phase
        self.phase = torch.zeros((self.num_envs, 1), dtype=torch.float,
                                 device=self.device, requires_grad=False)
        # TODO pull out into cfg
        self.base_freq = torch.ones((self.num_envs, 1), dtype=torch.float,
                                    device=self.device, requires_grad=False)

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id,
                                      self.graphics_device_id,
                                      self.physics_engine,
                                      self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'],
                          int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "../../assets"
        asset_file = "urdf/mit_humanoid/humanoid_R_sf.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        # asset_options.flip_visual_attachments = False  # ! remove?
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        humanoid_asset = self.gym.load_asset(self.sim, asset_root,
                                             asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        # extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        # TODO switch between single foot and heel-toe
        extremity_name = "_foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long,
                                        device=self.device, requires_grad=False)
        # knee_names = [s for s in body_names if "THIGH" in s]
        # self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long,
        #                                 device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(humanoid_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.Kp
            dof_props['damping'][i] = self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.humanoid_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper,
                                          num_per_row)
            humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset,
                                                    start_pose, "humanoid", i,
                                                    1, 0)
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle,
                                              dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
            self.envs.append(env_ptr)
            self.humanoid_handles.append(humanoid_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], feet_names[i])
        # for i in range(len(knee_names)):
        #     self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], knee_names[i])
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                            self.humanoid_handles[0], "base")

    def pre_physics_step(self, actions):
        self.actions_k2 = self.actions_k1.clone()
        self.actions_k1 = self.actions.clone()
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1
        # self.phase = torch.fmod(self.phase + self.dt*self.base_freq, 1.)
        self.phase = torch.fmod(self.phase + self.dt*1., 1.)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_humanoid_reward(
            # tensors
            self.root_states,
            self.commands,
            self.torques,
            self.contact_forces,
            # self.knee_indices,
            self.progress_buf,
            self.phase,
            self.actions,
            self.actions_k1,
            self.actions_k2,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_humanoid_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.phase,
                                                        self.actions,
                                                        self.actions_k1,
                                                        self.actions_k2,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale,
                                                        self.height_scale
        )

    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]*positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                            gymtorch.unwrap_tensor(self.initial_root_states),
                            gymtorch.unwrap_tensor(env_ids_int32),
                            len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                            gymtorch.unwrap_tensor(self.dof_state),
                            gymtorch.unwrap_tensor(env_ids_int32),
                            len(env_ids_int32))

        # TODO clean up verbosity
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0],
                                                self.command_x_range[1],
                                                (len(env_ids), 1),
                                                device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0],
                                                self.command_y_range[1],
                                                (len(env_ids), 1),
                                                device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.phase[env_ids] = 0

#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_humanoid_reward(
    # tensors
    root_states: Tensor,
    commands: Tensor,
    torques: Tensor,
    contact_forces: Tensor,
    # knee_indices: Tensor,
    episode_lengths: Tensor,
    phase: Tensor,
    actions: Tensor,
    actions_k1: Tensor,
    actions_k2: Tensor,
    # Dict
    rew_scales: Dict[str, float],
    # other
    base_index: int,
    max_episode_length: int,
) -> Tuple[Tensor, Tensor]:  # (reward, reset, feet_in air, feet_air_time, episode sums)

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2]-base_lin_vel[:, :2]),
                              dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # smoothness
    rew_smooth = torch.sum(torch.square(actions-2*actions_k1+actions_k2)) * \
        rew_scales["smooth"]

    # height
    height_sq = torch.square(root_states[:, 2]-0.65)
    rew_height = torch.exp(-0.1*height_sq) * rew_scales["height"]

    # symmetry


    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + \
        rew_smooth + rew_height
    # total_reward = torch.clip(total_reward, 0., None)
    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    # reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths > max_episode_length  # no terminal reward for time-outs
    # * not clear about this...
    reset = reset | time_out

    return total_reward.detach(), reset


# @torch.jit.script
def compute_humanoid_observations(root_states: Tensor,
                                commands: Tensor,
                                dof_pos: Tensor,
                                default_dof_pos: Tensor,
                                dof_vel: Tensor,
                                gravity_vec: Tensor,
                                phase: Tensor,
                                actions: Tensor,
                                actions_k1: Tensor,
                                actions_k2: Tensor,
                                lin_vel_scale: float,
                                ang_vel_scale: float,
                                dof_pos_scale: float,
                                dof_vel_scale: float,
                                height_scale: float
                                ) -> Tensor:

    # base_position = root_states[:, 0:3]
    base_height = torch.atleast_2d(root_states[:, 2]).T * height_scale
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * \
        lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * \
        ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale,
                                            ang_vel_scale],
                                            requires_grad=False,
                                            device=commands.device)

    obs = torch.cat((base_height,
                     base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     torch.sin(phase*2*np.pi),
                     torch.cos(phase*2*np.pi),
                     actions,
                     actions_k1,
                     actions_k2
                     ), dim=-1)

    return obs
