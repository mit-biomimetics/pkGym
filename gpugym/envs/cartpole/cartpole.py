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
from gpugym.envs.base.fixed_robot import FixedRobot

class Cartpole(FixedRobot):

    def _custom_init(self, cfg):
        self.cfg = cfg

        self.reset_dist = self.cfg.env.reset_dist

        self.max_push_effort = self.cfg.env.max_effort
        self.max_episode_length = 500  # 500

        # HANDLE AUGMENTATIONS
        self.augmentations = self.cfg.env.augmentations
        self.augmentation_names = [augmentation[1] for augmentation in self.augmentations]
        self.augmentation_scales = [augmentation[2] for augmentation in self.augmentations]
        self.num_augmentations = len(self.augmentations)

        # HANDLE HIERARCHICAL REWARDS
        self.reward_hierarchy = cfg.rewards.hierarchy


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations, phase-dynamics
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()

        # if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
        #     self._push_robots()

    def sqrdexp(self, value, space):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(value)/(space))

    def _reward_pole_position(self):
        # retrieve environment observations from buffer
        pole_pos = self.obs_buf[:, 2]
        return 0

        # reward = torch.zeros_like(self.rew_buf)
        #
        # # POLE POSITION REWARD
        # pole_pos_raw_reward = reward_info['pole_position']['weight'] * self.sqrdexp(pole_pos,
        #                                                                             reward_info['pole_position'][
        #                                                                                 'sqrdexp_space'])
        # pole_pos_reward = pole_pos_raw_reward
        #
        # # ACCUMULATE REWARDS
        # reward += pole_pos_reward
        # self.cumulative_rewards[:, 1] += pole_pos_reward
        #
        # return reward

    def _reward_pole_velocity(self):
        # retrieve environment observations from buffer
        pole_vel = self.obs_buf[:, 3]
        return 0
        # reward = torch.zeros_like(self.rew_buf)
        #
        #
        # # POLE VELOCITY REWARD
        # pole_vel_raw_reward = reward_info['pole_velocity']['weight'] * self.sqrdexp(pole_vel,
        #                                                                             reward_info['pole_velocity'][
        #                                                                                 'sqrdexp_space'])
        # pole_vel_reward = pole_vel_raw_reward
        #
        # # ACCUMULATE REWARDS
        # reward += pole_vel_reward
        # self.cumulative_rewards[:, 2] += pole_vel_reward
        #
        # return reward

    def _reward_cart_position(self):
        # retrieve environment observations from buffer
        pole_pos = self.obs_buf[:, 2]
        cart_pos = self.obs_buf[:, 0]

        cos_pole_pos = torch.cos(pole_pos)
        return 0
        # reward = torch.zeros_like(self.rew_buf)
        #
        # # CART POSITION REWARD
        # parent_info = reward_info['pole_position']
        # reward_activation = self.sqrdexp(cos_pole_pos, parent_info['sub_reward_activation_space'])
        # cart_pos_raw_reward = parent_info['sub_reward']['cart_position']['weight'] * self.sqrdexp(cart_pos, parent_info[
        #     'sub_reward']['cart_position']['sqrdexp_space'])
        # cart_pos_reward = reward_activation * cart_pos_raw_reward
        #
        # reward += cart_pos_reward
        # self.cumulative_rewards[:, 3] += cart_pos_reward
        #
        # return reward

    def _reward_actuation(self):
        # retrieve environment observations from buffer
        return 0
        # reward = torch.zeros_like(self.rew_buf)
        # # ACTUATION PENALTIES
        # actuation_penalty_raw_reward = self.sqrdexp(self.actions_buf, self.cfg['env']['maxEffort'])
        # actuation_penalty_reward = self.hierarchical_reward_scaling['actuation'][
        #                                'weight'] * actuation_penalty_raw_reward
        # reward += actuation_penalty_reward
        #
        # return reward

    def _reward_termination(self):
        # retrieve environment observations from buffer
        cart_pos = self.obs_buf[:, 0]
        return 0
        # reward = torch.zeros_like(self.rew_buf)
        #
        # # HANDLE AGENT RESETS
        # reward = torch.where(torch.abs(cart_pos) > self.reset_dist,
        #                      torch.ones_like(reward) * self.reward_scaling['reset']['weight'], reward)
        #
        # self.cumulative_rewards[:, 0] += reward
        #
        # return reward

    def compute_observations(self, env_ids=None):

        base_observations = [self.dof_pos * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.actions]

        cur_idx = 0
        for observation_data in base_observations:
            obs_length = observation_data.shape[1]
            self.obs_buf[:, cur_idx: cur_idx + obs_length] = observation_data[:]
            cur_idx += obs_length
