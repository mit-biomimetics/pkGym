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
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

from gym.envs.base.fixed_robot import FixedRobot


class Cartpole(FixedRobot):

    def _init_buffers(self):
        super()._init_buffers()
        n_envs = self.num_envs
        self.cart_vel_square = torch.zeros(n_envs, 1,
                                           dtype=torch.float,
                                           device=self.device)
        self.pole_vel_square = torch.zeros(n_envs, 1,
                                           dtype=torch.float,
                                           device=self.device)
        self.pole_trig_obs = torch.zeros(n_envs, 2,
                                         dtype=torch.float,
                                         device=self.device)
        self.cart_obs = torch.zeros(n_envs, 1,
                                    dtype=torch.float,
                                    device=self.device)

    def _post_physics_step(self):
        super()._post_physics_step()
        self.pole_trig_obs = torch.cat((torch.sin(self.dof_pos[:, 1:]),
                                        torch.cos(self.dof_pos[:, 1:])),
                                       dim=1)
        self.cart_obs = self.dof_pos[:, 0:1].square()
        self.cart_vel_square = self.dof_vel[:, 0:1].square()
        self.pole_vel_square = self.dof_vel[:, 1:2].square()

    def _compute_torques(self):
        return self.tau_ff

    def _reward_pole_pos(self):
        return torch.cos(self.dof_pos[:, 1])

    def _reward_pole_vel(self):
        pole_vel = self.dof_vel[:, 1]
        return -pole_vel.square()

    def _reward_cart_pos(self):
        cart_pos = self.dof_pos[:, 0]
        return -cart_pos.square()

    def _reward_upright_pole(self):
        return torch.cos(self.dof_pos[:, 1])

    def _reward_energy(self):
        m_cart = 1.
        m_pole = 1.
        l_pole = 2.
        kinetic_energy = (0.5*(m_cart + m_pole)*self.dof_vel[:, 0].square()
                          + 0.5*m_pole * l_pole**2 * self.dof_vel[:, 1].square()
                          + self.dof_vel[:, 0]*self.dof_vel[:, 1]*l_pole**2/2.
                          * torch.sin(self.dof_pos[:, 1]))
        potential_energy = m_pole*9.81*l_pole*torch.cos(self.dof_pos[:, 1])
        upright_energy = m_pole*9.81*l_pole

        energy_error = kinetic_energy + potential_energy - upright_energy
        reward = self._sqrdexp(energy_error/upright_energy) \
                 + self._sqrdexp(energy_error/upright_energy*10.)

        return reward

