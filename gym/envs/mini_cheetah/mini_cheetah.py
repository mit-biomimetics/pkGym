from time import time
import numpy as np
import os

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import to_torch

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gym.envs import LeggedRobot


class MiniCheetah(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.dof_pos_obs = torch.zeros_like(self.dof_pos, requires_grad=False)
        self.base_height = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                      device=self.device, requires_grad=False)

    def _post_physics_step(self):
        """ Callback called before computing terminations, rewards, and
         observations, phase-dynamics.
            Default behaviour: Compute ang vel command based on target and
             heading, compute measured terrain heights and randomly push robots
        """
        super()._post_physics_step()

        self.base_height = self.root_states[:, 2:3]

        nact = self.num_actions
        self.ctrl_hist[:, 2*nact:] = self.ctrl_hist[:, nact:2*nact]
        self.ctrl_hist[:, nact:2*nact] = self.ctrl_hist[:, :nact]
        self.ctrl_hist[:, :nact] = self.actions
        self.dof_pos_obs = (self.dof_pos - self.default_dof_pos) \
                            * self.scales["dof_pos_obs"]

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
               * self.scales["dof_pos_obs"]), dim=1)
