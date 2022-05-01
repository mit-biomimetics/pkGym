from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from gpugym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gpugym.utils.gait_scheduler import gait_scheduler
from gpugym.envs import LeggedRobot
import pandas as pd

END_EFFECTOR = ["left_hand", "right_hand", "left_foot", "right_foot"]

import pandas as pd

class MIT_Humanoid(LeggedRobot):

    def _custom_init(self, cfg):
        # get end_effector IDs for forward kinematics
        body_ids = []
        for body_name in END_EFFECTOR:
            body_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
            body_ids.append(body_id)

        self.end_eff_ids = to_torch(body_ids, device=self.device, dtype=torch.long)


    # def _post_physics_step_callback(self):
    #     """ Callback called before computing terminations, rewards, and observations, phase-dynamics
    #         Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
    #     """
    #     env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
    #     self._resample_commands(env_ids)
    #     if self.cfg.commands.heading_command:
    #         forward = quat_apply(self.base_quat, self.forward_vec)
    #         heading = torch.atan2(forward[:, 1], forward[:, 0])
    #         self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    #     if self.cfg.terrain.measure_heights:
    #         self.measured_heights = self._get_heights()
    #     if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
    #         self._push_robots()


    def compute_observations(self):
        """ Computes observations
        """

        # base_z
        # base lin vel
        # base ang vel
        # projected gravity vec
        # commands
        # joint pos
        # joint vel
        # actions
        # actions (n-1, n-2)
        base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z
        dof_pos = (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos

        self.obs_buf = torch.cat((base_z,
                                  self.base_lin_vel*self.obs_scales.lin_vel,
                                  self.base_ang_vel*self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3]*self.commands_scale,
                                  dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel,
                                  self.actions,
                                  self.ctrl_hist
                                  ),
                                 dim=-1)
        # * add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # ! noise_scale_vec must be of correct order! Check def below
        # * add noise if needed
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                            * self.noise_scale_vec


    def _get_noise_scale_vec(self, cfg):
        '''
        Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        '''
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        ns_lvl = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel*ns_lvl*self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel*ns_lvl*self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity*ns_lvl
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:30] = noise_scales.dof_pos*ns_lvl*self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel*ns_lvl*self.obs_scales.dof_vel
        noise_vec[48:66] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[66:187] = noise_scales.height_measurements*ns_lvl \
                                * self.obs_scales.height_measurements
        return noise_vec


    def _custom_reset(self, env_ids):
        if self.cfg.init_state.penetration_check:
            temp_root_state = torch.clone(self.root_states)
            temp_dof_state = torch.clone(self.dof_state)

            self.gym.simulate(self.sim) #Need to one step the simulation to update dof !!THIS MAY BE A TERRIBLE IDEA!!

            #retrieve body states of every link in every environment. 
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
            rb_states = gymtorch.wrap_tensor(body_states)

            num_links = int(len(rb_states[:,0])/self.num_envs)

            #iterate through the env ids that are being reset (and only the ones being reset)
            for i in env_ids:
                max_penetration = 0 
                for j in range(num_links):
                    #check each body position 
                    link_height = rb_states[i*num_links + j, 2]
                    if (link_height < 0.1): #check if COM of rigid link is to close to ground 
                        #TODO: replace with exact measurement of toe/heel height
                        if (0.1 - link_height > max_penetration):
                            max_penetration = -link_height + 0.1

                #find max penetration and shift root state by that amount. 
                temp_root_state[i, 2] += max_penetration

            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(temp_root_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(temp_dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))       

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)


    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1) == 1
        return 1.*single_contact


    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity w. squared exp
        return self.sqrdexp(self.base_lin_vel[:, 2]  \
                            * self.cfg.normalization.obs_scales.lin_vel)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.base_yaw_rate_tracking)


    def _reward_orientation(self):
        # Penalize non flat base orientation
        error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)
        # return self.sqrdexp(self.projected_gravity[:, 2]+1.)


    def _reward_base_height(self):
        """
        Squared exponential saturating at base_height target
        """
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error *= self.obs_scales.base_z
        error = torch.clamp(error, max=0, min=None).flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)


    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)



    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(self.sqrdexp(self.dof_vel  \
                            / self.cfg.normalization.obs_scales.dof_vel), dim=1)



    def _reward_symm_legs(self):
        error = 0.
        for i in range(2, 5):
            error += self.sqrdexp((self.dof_pos[:, i]+self.dof_pos[:, i+9]) \
                        / self.cfg.normalization.obs_scales.dof_pos)
        for i in range(0, 2):
            error += self.sqrdexp((self.dof_pos[:, i]-self.dof_pos[:, i+9]) \
                        / self.cfg.normalization.obs_scales.dof_pos)
        return error


    def _reward_symm_arms(self):
        error = 0.
        for i in range(6, 8):
            error += self.sqrdexp((self.dof_pos[:, i]-self.dof_pos[:, i+9]) \
                        / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp((self.dof_pos[:, 5]+self.dof_pos[:, 14]) \
                        / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp((self.dof_pos[:, 8]+self.dof_pos[:, 17]) \
                        / self.cfg.normalization.obs_scales.dof_pos)
        return error

    def _reward_dof_near_home(self):
        jnt_scales = torch.tensor(self.cfg.rewards.joint_level_scaling, device=self.device)

        return torch.sum(jnt_scales*self.sqrdexp((self.dof_pos - self.default_dof_pos) \
            * self.cfg.normalization.obs_scales.dof_pos), dim=1)