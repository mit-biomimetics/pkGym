# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from gpugym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gpugym.envs import LeggedRobot

class Cartpole(LeggedRobot):

    def _custom_init(self, cfg):
        self.cfg = cfg

        # model_params = cfg['train']['params']['network']['mlp']['units']
        # dynamics_augmented = cfg['env']['do_dynamics_augmentation']
        # controls_augmented = cfg['env']['do_controls_augmentation']

        # self.logger.config = {
        #     "dynamics_augmented": dynamics_augmented,
        #     "controls_augmented": controls_augmented,
        #     # 'layer_neurons': model_params[-1],
        #     # 'num_layers': len(model_params),
        #     'pole_swing_rew': cfg['env']['hierarchical_reward_scaling']['pole_position']['weight'],
        #     'sub_reward_activation': cfg['env']['hierarchical_reward_scaling']['pole_position']['sub_reward_activation_space'],
        #     'cart_center_rew': cfg['env']['hierarchical_reward_scaling']['pole_position']['sub_reward']['cart_position']['weight']
        # }

        # if cfg['train']['params']['do_wandb']:
        # self.logger.init(project="augmented-cartpole-rl",
        #                   entity="liamack27",
        #                   config=self.logger.config,
        #                   name=exp_name)

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500  # 500

        # HANDLE AUGMENTATIONS
        do_dynamics_augment = cfg["env"]["do_dynamics_augmentation"]
        do_controls_augment = cfg["env"]["do_controls_augmentation"]
        if do_dynamics_augment and do_controls_augment:
            self.augmentations = cfg["env"]["both_augmentations"]
            self.augmentation_names = []
            self.augmentation_dofs = []
            self.augmentation_scales = []
            for augmentation_name, augmentation_dof, augmentation_scale in self.augmentations:
                self.augmentation_names.append(augmentation_name)  # str
                self.augmentation_dofs.append(augmentation_dof)  # str
                self.augmentation_scales.append(augmentation_scale)  # float
            self.num_augmentations = len(self.augmentations)
        if do_dynamics_augment:
            self.augmentations = cfg["env"]["dynamics_augmentations"]
            self.augmentation_names = []
            self.augmentation_dofs = []
            self.augmentation_scales = []
            for augmentation_name, augmentation_dof, augmentation_scale in self.augmentations:
                self.augmentation_names.append(augmentation_name)  # str
                self.augmentation_dofs.append(augmentation_dof)  # str
                self.augmentation_scales.append(augmentation_scale)  # float
            self.num_augmentations = len(self.augmentations)
        if do_controls_augment:
            self.augmentations = cfg["env"]["controls_augmentations"]
            self.augmentation_names = []
            self.augmentation_dofs = []
            self.augmentation_scales = []
            for augmentation_name, augmentation_dof, augmentation_scale in self.augmentations:
                self.augmentation_names.append(augmentation_name)  # str
                self.augmentation_dofs.append(augmentation_dof)  # str
                self.augmentation_scales.append(augmentation_scale)  # float
            self.num_augmentations = len(self.augmentations)
        else:
            self.augmentations = []
            self.num_augmentations = 0

        self.use_hierarchical_rewards = cfg['env']['hierarchical_rewards']
        self.reward_scaling = cfg['env']['reward_scaling']
        self.hierarchical_reward_scaling = cfg['env']['hierarchical_reward_scaling']

        self.dof_state_scaling = torch.tensor(cfg["env"]["dof_scaling"], device=sim_device)
        self.num_states = 4

        self.cfg["env"]["numObservations"] = self.num_states + self.num_augmentations
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        self.num_states = 4

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations, phase-dynamics
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        if self.cfg.init_state.is_single_traj:
            self.phase = torch.minimum(self.phase + self.dt / self.total_ref_time, torch.tensor(1))
        else:
            if (self.total_ref_time > 0.0):
                self.phase = torch.fmod(self.phase + self.dt / self.total_ref_time, 1)
            else:
                self.phase = torch.fmod(self.phase + self.dt, 1.)

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def get_augment_function(self, augment_name):
        if augment_name == 'SIN':
            return torch.sin
        elif augment_name == 'COS':
            return torch.cos
        elif augment_name == 'TAN':
            return torch.tan
        elif augment_name == 'SINSQR':
            return lambda _: (torch.sin(_))**2
        elif augment_name == 'COSSQR':
            return lambda _: (torch.cos(_))**2
        elif augment_name == 'TANSQR':
            return lambda _: (torch.tan(_))**2
        elif augment_name == 'SQR':
            return lambda _: _**2
        elif augment_name == 'CUB':
            return lambda _: _**3
        else:
            return 'FUNCTION NOT FOUND, WILL CAUSE CANT CALL STR ERROR'

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        pose.p.z = 2.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def vanilla_rewards(self):
        # retrieve environment observations from buffer
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]

        reward = torch.zeros_like(self.rew_buf)

        # POLE REWARDS
        pole_pos_raw_reward = torch.cos(pole_angle)  # Reward pole angle standing -> 1 when pole standing
        pole_vel_raw_reward = torch.exp(
            -(pole_vel) ** 2 / self.reward_scaling['pole_velocity']['space'])  # Disincentivize pole moving
        pole_pos_reward = self.reward_scaling['pole_position']['weight'] * pole_pos_raw_reward
        pole_vel_reward = self.reward_scaling['pole_velocity']['weight'] * pole_vel_raw_reward
        reward += pole_pos_reward
        reward += pole_vel_reward

        # CART REWARDS
        cart_pos_raw_reward = torch.exp(
            -(cart_pos) ** 2 / self.reward_scaling['cart_position']['space'])  # Penalize cart deviation from 0.0
        cart_vel_raw_reward = torch.exp(
            -(cart_vel) ** 2 / self.reward_scaling['cart_velocity']['space'])  # Penalize cart velocity
        cart_pos_reward = self.reward_scaling['cart_position']['weight'] * cart_pos_raw_reward
        cart_vel_reward = self.reward_scaling['cart_velocity']['weight'] * cart_vel_raw_reward
        reward += cart_pos_reward
        reward += cart_vel_reward

        # ACTUATION PENALTIES
        actuation_penalty_raw_reward = (self.actions_buf ** 2) / (self.cfg['env']['maxEffort'] ** 2)
        actuation_penalty_reward = self.reward_scaling['actuation']['weight'] * actuation_penalty_raw_reward
        reward += actuation_penalty_reward

        # adjust reward for reset agents
        reward = torch.where(torch.abs(cart_pos) > self.reset_dist,
                             torch.ones_like(reward) * self.reward_scaling['reset']['weight'], reward)

        return reward

    def sqrdexp(self, value, space):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(value)/(space))

    def hierarchical_rewards(self):
        '''
        This should be able to maintain the same swing up behavior regardless of reward ratios

          hierarchical_rewards: true
          hierarchical_reward_scaling:
            pole_position: # This is the cos(pole pos) \in (-1, 1]
              weight: 1.0
              sqrdexp_space: 1.0
              sub_reward_activation_space: 0.1
              sub_reward:
                cart_position: # \in (-3, 3)
                  weight: 1.0
                  sqrdexp_space: 3.0
            actuation:
              weight: -1e-3
              space: false
            reset:
              weight: -4.0
        '''
        # retrieve environment observations from buffer
        pole_pos = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]

        cos_pole_pos = torch.cos(pole_pos)

        reward = torch.zeros_like(self.rew_buf)
        reward_info = self.hierarchical_reward_scaling

        # POLE POSITION REWARD
        pole_pos_raw_reward = reward_info['pole_position']['weight'] * self.sqrdexp(pole_pos, reward_info['pole_position']['sqrdexp_space'])
        pole_pos_reward = pole_pos_raw_reward

        # POLE VELOCITY REWARD
        pole_vel_raw_reward = reward_info['pole_velocity']['weight'] * self.sqrdexp(pole_vel, reward_info['pole_velocity']['sqrdexp_space'])
        pole_vel_reward = pole_vel_raw_reward

        # CART POSITION REWARD
        parent_info = reward_info['pole_position']
        reward_activation = self.sqrdexp(cos_pole_pos, parent_info['sub_reward_activation_space'])
        cart_pos_raw_reward = parent_info['sub_reward']['cart_position']['weight'] * self.sqrdexp(cart_pos, parent_info['sub_reward']['cart_position']['sqrdexp_space'])
        cart_pos_reward = reward_activation * cart_pos_raw_reward

        # ACTUATION PENALTIES
        actuation_penalty_raw_reward = self.sqrdexp(self.actions_buf, self.cfg['env']['maxEffort'])
        actuation_penalty_reward = self.hierarchical_reward_scaling['actuation']['weight'] * actuation_penalty_raw_reward

        # ACCUMULATE REWARDS
        reward += pole_pos_reward
        self.cumulative_rewards[:, 1] += pole_pos_reward

        reward += pole_vel_reward
        self.cumulative_rewards[:, 2] += pole_vel_reward

        reward += cart_pos_reward
        self.cumulative_rewards[:, 3] += cart_pos_reward

        reward += actuation_penalty_reward
        self.cumulative_rewards[:, 4] += actuation_penalty_reward

        # HANDLE AGENT RESETS
        reward = torch.where(torch.abs(cart_pos) > self.reset_dist,
                             torch.ones_like(reward) * self.reward_scaling['reset']['weight'], reward)

        self.cumulative_rewards[:, 0] += reward

        return reward


    def compute_reward(self):
        # retrieve environment observations from buffer
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]

        reset = torch.where(torch.abs(cart_pos) > self.reset_dist, torch.ones_like(self.reset_buf), self.reset_buf)
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)

        if self.use_hierarchical_rewards:
            self.rew_buf[:] = self.hierarchical_rewards()
        else:
            self.rew_buf[:] = self.vanilla_rewards()
        self.reset_buf[:] = reset


    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        pole_angle = self.dof_pos[env_ids, 1].squeeze()  # Pole Angle
        pole_vel   = self.dof_vel[env_ids, 1].squeeze()  # Pole Vel
        cart_vel   = self.dof_vel[env_ids, 0].squeeze()  # Cart Vel
        cart_pos   = self.dof_pos[env_ids, 0].squeeze()  # Cart Pos

        self.trajectory_buf[env_ids, self.progress_buf[env_ids], :] = torch.vstack([cart_pos, pole_angle, cart_vel, pole_vel]).T

        if self.cfg['env']['observation_randomization']['randomize']:
            pole_angle += torch.normal(mean=0.0, std=1.0, size=pole_angle.shape, device=self.device) * self.cfg['env']['observation_randomization']['pole_pos_rand_scale']
            pole_vel   += torch.normal(mean=0.0, std=1.0, size=pole_angle.shape, device=self.device) * self.cfg['env']['observation_randomization']['pole_vel_rand_scale']
            cart_pos   += torch.normal(mean=0.0, std=1.0, size=pole_angle.shape, device=self.device) * self.cfg['env']['observation_randomization']['cart_pos_rand_scale']
            cart_vel   += torch.normal(mean=0.0, std=1.0, size=pole_angle.shape, device=self.device) * self.cfg['env']['observation_randomization']['cart_vel_rand_scale']


        self.obs_buf[env_ids, 0] = cart_pos
        self.obs_buf[env_ids, 1] = cart_vel
        self.obs_buf[env_ids, 2] = pole_angle
        self.obs_buf[env_ids, 3] = pole_vel

        for aug_idx in range(self.num_augmentations):
            # Get actual augmentation function
            augmentation_func = self.get_augment_function(self.augmentation_names[aug_idx])

            what_to_augment = self.augmentation_dofs[aug_idx]

            augmentation_scale = self.augmentation_scales[aug_idx]

            # Gets the value to apply from dof states
            if what_to_augment == 'pole angle':
                dof_vals = pole_angle
            elif what_to_augment == 'pole velocity':
                dof_vals = pole_vel
            elif what_to_augment == 'cart position':
                dof_vals = cart_pos
            elif what_to_augment == 'cart velocity':
                dof_vals = cart_vel
            else:
                print("There's no such thing to augment, yeet")

            # Computes the value of the input when processed
            augmented_values = augmentation_func(dof_vals)
            # Adds the augmented value to the observation buffer
            self.obs_buf[env_ids, self.num_states + aug_idx] = augmentation_scale*augmented_values

        # TODO: INJECT NOISE ONTO THE OBSERVATIONS
        self.obs_buf[env_ids]

        return self.obs_buf

    def reset_idx(self, env_ids):
        # Torch.rand -> tensor of vals in [0, 1]
        cart_positions  = 0.2 * (torch.rand((len(env_ids), 1), device=self.device) - 0.5)
        pole_positions  = 0.8 * torch.pi * (torch.rand((len(env_ids), 1), device=self.device) - 0.5)
        cart_velocities = 0.2 * (torch.rand((len(env_ids), 1), device=self.device) - 0.5)
        pole_velocities = 0.2 * (torch.rand((len(env_ids), 1), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = torch.hstack((cart_positions, pole_positions))  # positions[:]
        self.dof_vel[env_ids, :] = torch.hstack((cart_velocities, pole_velocities))  # velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # SEND THE REWARD INFORMATION OF THE FINISHED ENVIRONMENTS TO THE LOGGER
        self.logger.log({'rewards':                 torch.mean(self.cumulative_rewards[env_ids, 0], dim=-1),
                         'swing_up_reward':         torch.mean(self.cumulative_rewards[env_ids, 1], dim=-1),
                         'pole_velocity_reward':    torch.mean(self.cumulative_rewards[env_ids, 2], dim=-1),
                         'center_cart_reward':      torch.mean(self.cumulative_rewards[env_ids, 3], dim=-1),
                         'actuation_penalty':       torch.mean(self.cumulative_rewards[env_ids, 4], dim=-1)})

        # I ALSO NEED TO LABEL THE BEHAVIOR OF THE SYSTEMS BASED ON THEIR TRAJECTORIES
        # The labels I will attempt to use are:
        # 1 - failure: the episode terminated before the max episode time or the pole angle was greater than the threshold
        # 2 - pole angle success - the final pole angle was within the threshold
        # 3 - cart center success - the cart was centered within some distance threshold
        # 4 - pole stable - the pole maintained its upright position for the last x timesteps
        # They will be applied in a cascade such that:
        #   a failure can never be a pole angle success
        #   a cart center success must be a pole angle success

        # Count the failure rate: number of environments prematurely


        testing_play = False
        if testing_play:
            # Should be only 1 cartpole on the screen performing
            cart_pos_trajectory = self.trajectory_buf[0, 0:self.progress_buf[0], 0]
            pole_pos_trajectory = self.trajectory_buf[0, 0:self.progress_buf[0], 1]
            cart_vel_trajectory = self.trajectory_buf[0, 0:self.progress_buf[0], 2]
            pole_vel_trajectory = self.trajectory_buf[0, 0:self.progress_buf[0], 3]
            fig, axs = plt.subplots(2, 2)
            t = torch.tensor([_ for _ in range(self.progress_buf[0])])
            axs[0, 0].plot(t, cart_pos_trajectory.cpu())
            axs[0, 0].set_title('Cart Position [m]')

            axs[0, 1].plot(t, cart_vel_trajectory.cpu(), 'tab:orange')
            axs[0, 1].set_title('Cart Velocity [m/s]')

            axs[1, 0].plot(t, (pole_pos_trajectory/torch.pi).cpu(), 'tab:green')
            axs[1, 0].set_title('Pole Position [rad^-1]')

            axs[1, 1].plot(t, pole_vel_trajectory.cpu(), 'tab:red')
            axs[1, 1].set_title('Pole Velocity [rad/s]')

            # for ax in axs.flat:
            #     ax.set(xlabel='x-label', ylabel='y-label')

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            # for ax in axs.flat:
            #     ax.label_outer()
            fig.show()

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.trajectory_buf[env_ids, :, :] = torch.zeros(
            (env_ids.shape[0], self.max_episode_length, self.num_states), device=self.device, dtype=torch.float)

        self.cumulative_rewards[env_ids, :] = torch.zeros((env_ids.shape[0], self.num_rewards), device=self.device, dtype=torch.float)

    # def pre_physics_step(self, actions):
    #     actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
    #     actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
    #     self.actions_buf[:] = actions.to(self.device).squeeze() * self.max_push_effort  # actions_tensor[:, :]
    #     forces = gymtorch.unwrap_tensor(actions_tensor)
    #     self.gym.set_dof_actuation_force_tensor(self.sim, forces)
    #
    # def post_physics_step(self):
    #     self.progress_buf += 1
    #
    #     env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    #     if len(env_ids) > 0:
    #         self.reset_idx(env_ids)
    #
    #     self.compute_observations()
    #     self.compute_reward()

