# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# ARE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

from gym.envs.base.legged_robot import LeggedRobot
from gym import LEGGED_GYM_ROOT_DIR


class Anymal(LeggedRobot):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            net_file = self.cfg.control.actuator_net_file
            sea_network_path = net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(sea_network_path).to(self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.0
        self.sea_cell_state_per_env[:, env_ids] = 0.0

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        if self.cfg.control.use_actuator_network:
            self._init_sea()

    def _init_sea(self):
        num_sea_states = self.num_envs * self.num_actuators
        self.sea_input = torch.zeros(num_sea_states, 1, 2, device=self.device)
        self.sea_hidden_state = torch.zeros(2, num_sea_states, 8, device=self.device)
        self.sea_cell_state = torch.zeros(2, num_sea_states, 8, device=self.device)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            2, self.num_envs, self.num_actuators, 8
        )
        self.sea_cell_state_per_env = self.sea_cell_state.view(
            2, self.num_envs, self.num_actuators, 8
        )

    def _compute_torques(self):
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (
                    self.cfg.actions * self.cfg.control.action_scale
                    + self.default_dof_pos
                    - self.dof_pos
                ).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                (
                    torques,
                    (self.sea_hidden_state[:], self.sea_cell_state[:]),
                ) = self.actuator_network(
                    self.sea_input,
                    (self.sea_hidden_state, self.sea_cell_state),
                )
            return torques
        else:
            return super()._compute_torques()

    def _switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(
            -torch.square(torch.max(torch.zeros_like(c_vel), c_vel - 0.2)) / 0.1
        )

    def _reward_lin_vel_z(self):
        """
        Penalize z axis base linear velocity w. squared exp
        """

        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        """
        Penalize xy axes base angular velocity
        """
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        """
        Penalize non flat base orientation
        """
        error = (
            torch.square(self.projected_gravity[:, :2])
            / self.cfg.reward_settings.tracking_sigma
        )
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_min_base_height(self):
        """
        Squared exponential saturating at base_height target
        """
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales["base_height"]
        error = torch.clamp(error, max=0, min=None).flatten()
        return torch.exp(-torch.square(error) / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        """
        Tracking of linear velocity commands (xy axes)
        """

        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error / self.cfg.reward_settings.tracking_sigma)

    def _reward_dof_vel(self):
        """
        Penalize dof velocities
        """

        return torch.mean(self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_stand_still(self):
        """
        Penalize motion at zero commands
        """

        # * normalize angles so we care about being within 5 deg
        error = (self.dof_pos - self.default_dof_pos) / torch.pi * 36
        rew_pos = torch.mean(self._sqrdexp(error), dim=1)
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel + rew_pos - rew_base_vel) * self._switch()

    def _reward_dof_near_home(self):
        return torch.mean(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
            ),
            dim=1,
        )
