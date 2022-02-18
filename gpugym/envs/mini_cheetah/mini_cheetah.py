from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from gpugym.utils.augmentor import Augmentor

from gpugym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gpugym.envs import LeggedRobot

class MiniCheetah(LeggedRobot):

    def _custom_init(self, cfg):
        # * init buffer for phase variable
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)

        # * additional buffer for last ctrl: whatever is actually used for PD control (which can be shifted compared to action)
        self.ctrl_hist = torch.zeros(self.num_envs, self.num_actions*3,
                                         dtype=torch.float, device=self.device,
                                         requires_grad=False)

        self.augmentor = Augmentor(cfg)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations, phase-dynamics
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        if (self.total_ref_time > 0.0):
            self.phase = torch.fmod(self.phase + self.dt/self.total_ref_time, 1)
        else:
            self.phase = torch.fmod(self.phase+self.dt, 1.)

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

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
        # phase
        base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z
        dof_pos = (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos

        # * update commanded action history buffer
        nact = self.num_actions
        self.ctrl_hist[:, 2*nact:] = self.ctrl_hist[:, nact:2*nact]
        self.ctrl_hist[:, nact:2*nact] = self.ctrl_hist[:, :nact]
        self.ctrl_hist[:, :nact] = self.actions*self.cfg.control.action_scale  + self.default_dof_pos

        self.obs_buf = torch.cat((base_z,
                                  self.base_lin_vel*self.obs_scales.lin_vel,
                                  self.base_ang_vel*self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3]*self.commands_scale,
                                  dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel,
                                  self.actions,
                                  self.ctrl_hist,
                                  torch.cos(self.phase*2*torch.pi),
                                  torch.sin(self.phase*2*torch.pi)),
                                 dim=-1)

        # ! noise_scale_vec must be of correct order! Check def below
        # * add noise if needed
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                            * self.noise_scale_vec

        # Add augmentations to the buffer - the method of adding noise required me to grab dof info from obs_buf
        noisy_body_lin_vel = self.obs_buf[:, 1:4] / self.obs_scales.lin_vel
        noisy_body_ang_vel = self.obs_buf[:, 4:7] / self.obs_scales.ang_vel
        noisy_dof_pos = self.obs_buf[:, 13:25]  # based on calculation that the dof pos stuff happens at these indices
        noisy_dof_vel_unscaled = self.obs_buf[:, 25:37] / self.obs_scales.dof_vel
        augmented_dofs_list = self.augmentor.apply_augmentations(noisy_body_lin_vel, noisy_body_ang_vel, noisy_dof_pos, noisy_dof_vel_unscaled)
        if len(augmented_dofs_list) > 0:
            augmented_dofs_tensor = torch.cat(augmented_dofs_list, dim=-1)
            # mins = torch.min(augmented_dofs_tensor, dim=0)
            # maxs = torch.max(augmented_dofs_tensor, dim=0)
            # When to start scaling stuff: (maxs.values > torch.ones_like(maxs.values)).any() evaluates to true
            self.obs_buf = torch.cat([self.obs_buf] + augmented_dofs_list, dim=-1)

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
        noise_vec[1:4] = noise_scales.lin_vel*ns_lvl*self.obs_scales.lin_vel
        noise_vec[4:7] = noise_scales.ang_vel*ns_lvl*self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity*ns_lvl
        noise_vec[10:13] = 0.  # commands
        noise_vec[13:25] = noise_scales.dof_pos*ns_lvl*self.obs_scales.dof_pos
        noise_vec[25:37] = noise_scales.dof_vel*ns_lvl*self.obs_scales.dof_vel
        noise_vec[37:49] = 0.  # previous actions
        noise_vec[49:85] = 0.
        noise_vec[85:87] = 0.  # phase # * could add noise, to make u_ff robust
        if self.cfg.terrain.measure_heights:
            noise_vec[66:187] = noise_scales.height_measurements*ns_lvl \
                                * self.obs_scales.height_measurements
        return noise_vec

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity w. squared exp
        return self.sqrdexp(self.base_lin_vel[:, 2]  \
                            * self.cfg.normalization.obs_scales.lin_vel)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        error = self.sqrdexp(self.base_ang_vel[:, :2] \
                             * self.cfg.normalization.obs_scales.ang_vel)
        return torch.sum(error, dim=1)

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

    def _reward_reference_traj(self):
        #tracking the reference trajectory
        ref_traj_idx = (torch.round(self.phase*self.pos_traj.size(dim=0)).squeeze(1)).long()
        pos_ref_frame = self.pos_traj.repeat(self.num_envs,1)[ref_traj_idx,:]
        vel_ref_frame = self.vel_traj.repeat(self.num_envs,1)[ref_traj_idx.long(),:]
        reward = 0.

        # todo needs to be redone: metrics in quaternion space are crap
        # base position error
        # base_pos_error = self.root_states[:,0:7] - pos_ref_frame[:, 1:8]
        # base_pos_error = torch.exp(-torch.sum(torch.square(base_pos_error), dim=1))
        # base_pos_error[:, 0:3] *= self.cfg.normalization.obs_scales.base_z
        # reward += self.sqrdexp(base_pos_error)
        #dof position error
        dof_pos_err = self.dof_pos - pos_ref_frame[:,8:]
        dof_pos_err *= self.cfg.normalization.obs_scales.dof_pos
        reward += torch.sum(self.sqrdexp(dof_pos_err), dim=1) \
                  * self.cfg.rewards.dof_pos_tracking

        # base velocity error
        # * might want this to be vector instead of element-wise
        base_vel_err = self.root_states[:,7:] - vel_ref_frame[:,1:7]
        base_vel_err[:, 1:4] *= self.cfg.normalization.obs_scales.lin_vel
        base_vel_err[:, 4:] *= self.cfg.normalization.obs_scales.ang_vel
        reward += torch.sum(self.sqrdexp(base_vel_err), dim=1) \
                  * self.cfg.rewards.base_vel_tracking

        # dof velocity error
        dof_vel_err = self.dof_pos - vel_ref_frame[:,7:]
        dof_vel_err *= self.cfg.normalization.obs_scales.dof_vel
        reward += torch.sum(self.sqrdexp(dof_vel_err), dim=1) \
                  * self.cfg.rewards.dof_vel_tracking
        # dof_vel_error =  torch.exp(-torch.sum(torch.square(dof_vel_error),dim=1))

        return reward

    def _reward_action_rate2(self):
        # Penalize changes in actions
        nact = self.num_actions
        dt2 = (self.dt*self.cfg.control.decimation)**2
        error = torch.square(self.ctrl_hist[:, :nact]  \
                             - 2*self.ctrl_hist[:, nact:2*nact]  \
                             + self.ctrl_hist[:, 2*nact:])/dt2
        # todo this tracking_sigma is not scaled (check)
        # error = torch.exp(-error/self.cfg.rewards.tracking_sigma)
        return torch.sum(error, dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return self.sqrdexp(self.dof_vel  \
                            / self.cfg.normalization.obs_scales.dof_vel)

    # def _reward_symm_legs(self):
    #     error = 0.
    #     for i in range(2, 5):
    #         error += self.sqrdexp((self.dof_pos[:, i]+self.dof_pos[:, i+9]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     for i in range(0, 2):
    #         error += self.sqrdexp((self.dof_pos[:, i]-self.dof_pos[:, i+9]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     return error

    # def _reward_symm_arms(self):
    #     error = 0.
    #     for i in range(6, 8):
    #         error += self.sqrdexp((self.dof_pos[:, i]-self.dof_pos[:, i+9]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     error += self.sqrdexp((self.dof_pos[:, 5]+self.dof_pos[:, 14]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     error += self.sqrdexp((self.dof_pos[:, 8]+self.dof_pos[:, 17]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     return error
