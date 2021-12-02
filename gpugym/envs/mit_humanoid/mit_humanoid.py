# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from gpugym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gpugym.envs import LeggedRobot

class MIT_Humanoid(LeggedRobot):

    def custom_init(self, cfg):
        # * init buffer for phase variable
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)
        # self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float,
        #                                    device=self.device,
        #                                    requires_grad=False)



        # def pre_physics_step(self):
        # """
        # adjust actions
        # """

        # return 0


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

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
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
        # ! actions (n-1, n-2)
        # phase
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
                                  torch.cos(self.phase),
                                  torch.sin(self.phase)),
                                 dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # ! noise_scale_vec must be of correct order! Check def below
        # add noise if needed
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
        noise_vec[66:68] = 0.  # phase # * could add noise, to make u_ff robust
        if self.cfg.terrain.measure_heights:
            noise_vec[66:187] = noise_scales.height_measurements*ns_lvl \
                                * self.obs_scales.height_measurements
        return noise_vec

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1) == 1
        return 1.*single_contact

    def _reward_base_height(self):
        """ Squared exponential saturating at base_height target
        """
        base_height = self.root_states[:,2].unsqueeze(1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error *= self.obs_scales.base_z
        error = torch.clamp(error, max=0, min=None).flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= self.obs_scales.lin_vel
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_reference_traj(self):
        #tracking the reference trajectory
        ref_traj_idx = (torch.round(self.phase*self.pos_traj.size(dim=0)).squeeze(1)).long()
        pos_ref_frame = self.pos_traj.repeat(self.num_envs,1)[ref_traj_idx,:]
        vel_ref_frame = self.vel_traj.repeat(self.num_envs,1)[ref_traj_idx.long(),:]

        #base position error
        base_pos_error = self.root_states[:,0:7] - pos_ref_frame[:, 1:8]
        base_pos_error = torch.exp(-torch.sum(torch.square(base_pos_error), dim=1))

        #dof position error
        dof_pos_error = self.dof_pos - pos_ref_frame[:,8:]
        dof_pos_error = torch.exp(-torch.sum(torch.square(dof_pos_error), dim=1))

        #base velocity error
        base_vel_error = self.root_states[:,7:] - vel_ref_frame[:,1:7]
        base_vel_error = torch.exp(-torch.sum(torch.square(base_vel_error),dim=1))

        #dof velocity error
        dof_vel_error = self.dof_vel - vel_ref_frame[:,7:]
        dof_vel_error =  torch.exp(-torch.sum(torch.square(dof_vel_error),dim=1))

        base_pos_reward = self.cfg.rewards.base_pos_tracking * base_pos_error 
        base_vel_reward = self.cfg.rewards.base_vel_tracking * base_vel_error
        dof_pos_reward = self.cfg.rewards.dof_pos_tracking * dof_pos_error
        dof_vel_reward = self.cfg.rewards.dof_vel_tracking * dof_vel_error

        error = base_pos_reward + base_vel_reward + dof_pos_reward + dof_vel_reward

        return error
