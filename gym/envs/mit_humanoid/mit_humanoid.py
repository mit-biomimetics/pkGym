from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from gym.utils.math import *
from gym.envs import LeggedRobot

END_EFFECTOR = ["left_hand", "right_hand", "left_toe", "left_heel", "right_toe", "right_heel"]

class MIT_Humanoid(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # get end_effector IDs for forward kinematics
        body_ids = []
        for body_name in END_EFFECTOR:
            body_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
            body_ids.append(body_id)
        self.end_eff_ids = to_torch(body_ids, device=self.device,
                                    dtype=torch.long)

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

        self.dof_pos_obs = (self.dof_pos - self.default_dof_pos)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity w. squared exp
        return self._sqrdexp(self.base_lin_vel[:, 2] *self.scales["base_lin_vel"])


    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square((self.commands[:, 2] - self.base_ang_vel[:, 2])*2/torch.pi)
        return self._sqrdexp(-ang_vel_error * self.scales['base_ang_vel'])


    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(self._sqrdexp(torch.square( 
                         self.projected_gravity[:, :2])), dim=1)

    def _reward_min_base_height(self):
        """
        Squared exponential saturating at base_height target
        """
        error = (self.base_height-self.cfg.reward_settings.base_height_target)
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)


    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        return torch.sum(self._sqrdexp(error), dim=1)


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(self._sqrdexp(self.dof_vel  
                            * self.scales['dof_vel']), dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(self._sqrdexp((self.dof_pos - self.default_dof_pos) 
                                        * self.scales['dof_pos_obs']), dim=1)
