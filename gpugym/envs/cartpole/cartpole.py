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
from gpugym.envs.base.fixed_robot import FixedRobot

class Cartpole(FixedRobot):

    def _custom_init(self, cfg):
        self.cfg = cfg

        self.reset_dist = self.cfg.env.reset_dist

        self.max_push_effort = self.cfg.env.max_effort
        self.max_episode_length = 500  # 500

        # HANDLE AUGMENTATIONS
        self.augmentations = self.cfg.env.augmentations

        # HANDLE HIERARCHICAL REWARDS
        self.reward_hierarchy = cfg.rewards.hierarchy


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations, phase-dynamics
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
        #     as_tuple=False).flatten()

        # if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
        #     self._push_robots()
        pass

    def _custom_termination(self):
        reset_indices = torch.absolute(self.dof_pos[:, 0]) > self.cfg.env.reset_dist
        self.reset_buf[:] = torch.where(reset_indices,
                                        torch.ones_like(self.reset_buf, device=self.device),
                                        torch.zeros_like(self.reset_buf, device=self.device))

    def sqrdexp(self, value, space):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(value)/(space))

    def _reward_pole_pos(self):
        pole_pos = self.dof_pos[:, 1]

        pole_pos_raw = self.sqrdexp(pole_pos, self.cfg.rewards.spaces.pole_pos)

        pole_pos_reward = self.cfg.rewards.scales.pole_pos * pole_pos_raw

        return pole_pos_reward.squeeze(dim=-1)

    def _reward_pole_vel(self):
        pole_vel = self.dof_vel[:, 1]

        pole_vel_raw = self.sqrdexp(pole_vel, self.cfg.rewards.spaces.pole_vel)

        pole_vel_reward = self.cfg.rewards.scales.pole_vel * pole_vel_raw

        return pole_vel_reward.squeeze(dim=-1)

    def _reward_cart_pos(self):
        cart_pos = self.dof_pos[:, 0]
        pole_pos = self.dof_pos[:, 1]

        pole_cart_raw = self.sqrdexp(cart_pos, self.cfg.rewards.spaces.cart_pos)

        if self.cfg.rewards.sub_spaces.pole_pos is not None:
            pole_pos_activation = self.sqrdexp(pole_pos, self.cfg.rewards.sub_spaces.pole_pos)

            cart_pos_reward = self.cfg.rewards.scales.cart_pos * pole_pos_activation * pole_cart_raw

        else:
            cart_pos_reward = self.cfg.rewards.scales.cart_pos * pole_cart_raw

        return cart_pos_reward.squeeze(dim=-1)

    def _reward_actuation(self):
        actuation = self.actions

        actuation_raw = self.sqrdexp(actuation, self.cfg.rewards.spaces.actuation)

        actuation_reward = self.cfg.rewards.scales.actuation * actuation_raw

        return actuation_reward.squeeze(dim=-1)

    def compute_observations(self, env_ids=None):

        cart_pos = self.dof_pos[:, 0].unsqueeze(dim=-1)
        pole_pos = self.dof_pos[:, 1].unsqueeze(dim=-1)
        cart_vel = self.dof_vel[:, 0].unsqueeze(dim=-1)
        pole_vel = self.dof_vel[:, 1].unsqueeze(dim=-1)

        add_noise = self.cfg.noise.add_noise
        if add_noise:
            cart_pos += torch.normal(size=(self.num_envs,), mean=0.0, std=self.cfg.noise.noise_scales.cart_pos, device=self.device).unsqueeze(dim=-1)
            pole_pos += torch.normal(size=(self.num_envs,), mean=0.0, std=self.cfg.noise.noise_scales.pole_pos, device=self.device).unsqueeze(dim=-1)
            cart_vel += torch.normal(size=(self.num_envs,), mean=0.0, std=self.cfg.noise.noise_scales.cart_vel, device=self.device).unsqueeze(dim=-1)
            pole_vel += torch.normal(size=(self.num_envs,), mean=0.0, std=self.cfg.noise.noise_scales.pole_vel, device=self.device).unsqueeze(dim=-1)

        observations = [cart_pos * self.cfg.normalization.obs_scales.cart_pos,
                        pole_pos * self.cfg.normalization.obs_scales.pole_pos,
                        cart_vel * self.cfg.normalization.obs_scales.cart_vel,
                        pole_vel * self.cfg.normalization.obs_scales.pole_vel,
                        self.actions]

        dof = {'cart_pos': cart_pos, 'pole_pos': pole_pos,
               'cart_vel': cart_vel, 'pole_vel': pole_vel}

        for func, name, scale in self.augmentations:
            new_ob = func(dof[name])
            scaled_ob = scale * new_ob
            observations.append(scaled_ob)

        cur_idx = 0
        self.obs_buf = torch.zeros(size=(self.num_envs, self.num_obs), device=self.device)
        for obs_idx, observation_data in enumerate(observations):
            obs_length = observation_data.shape[1]
            self.obs_buf[:, cur_idx: cur_idx + obs_length] = observation_data[:]
            cur_idx += obs_length
