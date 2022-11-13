from time import time
import numpy as np
import os

from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gym.envs import LeggedRobot


class MiniCheetah(LeggedRobot):

    def _custom_init(self, cfg):
        # * init buffer for phase variable
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)

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
        base_z = self.root_states[:, 2].unsqueeze(1)*self.scales["base_z"]
        dof_pos = (self.dof_pos-self.default_dof_pos)*self.scales["dof_pos"]

        # * update commanded action history buffer
        control_type = self.cfg.control.control_type
        nact = self.num_actions
        self.ctrl_hist[:, 2 * nact:] = self.ctrl_hist[:, nact:2 * nact]
        self.ctrl_hist[:, nact:2 * nact] = self.ctrl_hist[:, :nact]
        self.ctrl_hist[:, :nact] = self.actions*self.cfg.control.action_scale + self.default_dof_pos

        self.obs_buf = torch.cat((base_z,
                                  self.base_lin_vel*self.scales["lin_vel"],
                                  self.base_ang_vel*self.scales["ang_vel"],
                                  self.projected_gravity,
                                  self.commands[:, :3]*self.commands_scale,
                                  dof_pos,
                                  self.dof_vel*self.scales["dof_vel"],
                                  self.ctrl_hist,
                                  torch.cos(self.phase*2*torch.pi),
                                  torch.sin(self.phase*2*torch.pi)),
                                 dim=-1)

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
            [torch.Tensor]: Vector of scales used to multiply a uniform
            distribution in [-1, 1]
        '''
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        ns_lvl = self.cfg.noise.noise_level
        noise_vec[1:4] = noise_scales.lin_vel * ns_lvl * self.scales["lin_vel"]
        noise_vec[4:7] = to_torch(noise_scales.ang_vel, device=self.device) \
                         * ns_lvl*self.scales["ang_vel"]
        noise_vec[7:10] = noise_scales.gravity*ns_lvl
        noise_vec[10:13] = 0.  # commands
        noise_vec[13:25] = noise_scales.dof_pos*ns_lvl*self.scales["dof_pos"]
        noise_vec[25:37] = noise_scales.dof_vel*ns_lvl*self.scales["dof_vel"]

        if self.cfg.terrain.measure_heights:
            noise_vec[66:187] = noise_scales.height_measurements*ns_lvl \
                                * self.scales["height_measurements"]
        return noise_vec

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity w. squared exp
        return self._sqrdexp(self.base_lin_vel[:, 2]
                            * self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        error = self._sqrdexp(self.base_ang_vel[:, :2]
                              * self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        error = torch.square(self.projected_gravity[:, :2]) \
                / self.cfg.reward_settings.tracking_sigma
        return torch.sum(torch.exp(-error), dim=1)
        # return self._sqrdexp(self.projected_gravity[:, 2]+1.)

    def _reward_min_base_height(self):
        """
        Squared exponential saturating at base_height target
        """
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height-self.cfg.reward_settings.base_height_target)
        error *= self.scales["base_height"]
        error = torch.clamp(error, max=0, min=None).flatten()
        return torch.exp(-torch.square(error)
                         / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.reward_settings.tracking_sigma)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(self._sqrdexp(self.dof_vel * self.scales["dof_vel"]),
                         dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(self._sqrdexp((self.dof_pos - self.default_dof_pos) \
               * self.scales["dof_pos"]), dim=1)
