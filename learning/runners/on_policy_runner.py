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

import os
import torch
from learning.algorithms import PPO
from learning.modules import ActorCritic
from learning.env import VecEnv
from learning.utils import remove_zero_weighted_rewards

from learning.utils import Logger

logger = Logger()


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 device='cpu'):

        self.device = device
        self.env = env
        self.parse_train_cfg(train_cfg)

        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])
        actor_critic = ActorCritic(num_actor_obs,
                                   num_critic_obs,
                                   num_actions,
                                   **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic,
                                  device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.tot_timesteps = 0
        self.it = 0

        # * init storage and model
        self.init_storage()
        self.log_dir = train_cfg["log_dir"]

        logger.initialize(self.env.num_envs,
                          self.env.dt,
                          self.cfg['max_iterations'],
                          self.device)

    def parse_train_cfg(self, train_cfg):
        self.cfg = train_cfg['runner']
        self.alg_cfg = train_cfg['algorithm']
        remove_zero_weighted_rewards(train_cfg['policy']['reward']['weights'])
        self.policy_cfg = train_cfg['policy']

    def init_storage(self):
        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              actor_obs_shape=[num_actor_obs],
                              critic_obs_shape=[num_critic_obs],
                              action_shape=[num_actions])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):

        # * unpack out of config
        reward_weights = self.policy_cfg['reward']['weights']
        termination_weight = self.policy_cfg['reward']['termination_weight']
        rewards_dict = {}
        total_rewards = torch.zeros(self.env.num_envs, device=self.device)

        # * set up logger
        logger.register_rewards(list(reward_weights.keys()))
        logger.register_rewards(list(termination_weight.keys()))
        logger.register_rewards(['total_rewards'])

        logger.register_category('algorithm',
                                 self.alg,
                                 ['mean_value_loss',
                                  'mean_surrogate_loss'])

        logger.attach_torch_obj_to_wandb((self.alg.actor_critic.actor,
                                          self.alg.actor_critic.critic))

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length))

        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
        self.alg.actor_critic.train()
        self.num_learning_iterations = num_learning_iterations
        tot_iter = self.it + num_learning_iterations

        self.save()

        logger.tic('runtime')
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic('iteration')
            logger.tic('collection')
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    self.set_actions(actions)

                    self.env.step()

                    actor_obs = self.get_noisy_obs(
                        self.policy_cfg['actor_obs'],
                        self.policy_cfg['noise'])
                    critic_obs = self.get_obs(self.policy_cfg['critic_obs'])
                    # * get time_outs
                    timed_out = self.get_timed_out()
                    terminated = self.get_terminated()
                    dones = timed_out | terminated

                    rewards_dict.update(self.get_rewards(termination_weight,
                                                         mask=terminated))
                    rewards_dict.update(self.get_rewards(reward_weights,
                                                         modifier=self.env.dt,
                                                         mask=~terminated))

                    total_rewards = torch.stack(
                        tuple(rewards_dict.values())).sum(dim=0)

                    logger.log_rewards(rewards_dict)
                    logger.log_rewards({'total_rewards': total_rewards})
                    logger.finish_step(dones)

                    self.alg.process_env_step(total_rewards,
                                              dones,
                                              timed_out)
                self.alg.compute_returns(critic_obs)
            logger.toc('collection')

            logger.tic('learning')
            self.alg.update()
            logger.toc('learning')
            logger.log_category()

            logger.finish_iteration()
            logger.toc('iteration')
            logger.toc('runtime')
            logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save()
        self.save()

    def get_noise(self, obs_list, noise_dict):
        noise_vec = torch.zeros(self.get_obs_size(obs_list),
                                device=self.device)
        obs_index = 0
        for obs in obs_list:
            obs_size = self.get_obs_size([obs])
            if obs in noise_dict.keys():
                noise_tensor = torch.ones(obs_size).to(self.device) \
                    * torch.tensor(noise_dict[obs]).to(self.device)
                if obs in self.env.scales.keys():
                    noise_tensor /= self.env.scales[obs]
                noise_vec[obs_index:obs_index + obs_size] = noise_tensor
            obs_index += obs_size
        return noise_vec * torch.randn(self.env.num_envs, len(noise_vec),
                                       device=self.device)

    def get_noisy_obs(self, obs_list, noise_dict):
        observation = self.get_obs(obs_list)
        return observation + self.get_noise(obs_list, noise_dict)

    def get_obs(self, obs_list):
        observation = self.env.get_states(obs_list).to(self.device)
        return observation

    def set_actions(self, actions):
        if self.policy_cfg['disable_actions']:
            return
        if hasattr(self.env.cfg.scaling, "clip_actions"):
            actions = torch.clip(actions,
                                 -self.env.cfg.scaling.clip_actions,
                                 self.env.cfg.scaling.clip_actions)
        self.env.set_states(self.policy_cfg["actions"], actions)

    def get_timed_out(self):
        return self.env.get_states(['timed_out']).to(self.device)

    def get_terminated(self):
        return self.env.get_states(['terminated']).to(self.device)

    def get_obs_size(self, obs_list):
        # todo make unit-test to assert len(shape)==1 always
        return self.get_obs(obs_list)[0].shape[0]

    def get_action_size(self, action_list):
        return self.env.get_states(action_list)[0].shape[0]

    def get_rewards(self, reward_weights, modifier=1, mask=None):
        rewards_dict = {}
        if mask is None:
            mask = 1.0
        for name, weight in reward_weights.items():
            rewards_dict[name] = mask * self._get_reward({name: weight},
                                                         modifier)
        return rewards_dict

    def _get_reward(self, name_weight, modifier=1):
        return modifier * self.env.compute_reward(name_weight).to(self.device)

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, 'model_{}.pt'.format(self.it))
        torch.save({'model_state_dict': self.alg.actor_critic.state_dict(),
                    'optimizer_state_dict': self.alg.optimizer.state_dict(),
                    'iter': self.it}, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(
                loaded_dict['optimizer_state_dict'])
        self.it = loaded_dict['iter']

    def switch_to_eval(self):
        self.alg.actor_critic.eval()

    def get_inference_actions(self):
        obs = self.get_noisy_obs(self.policy_cfg['actor_obs'],
                                 self.policy_cfg['noise'])
        return self.alg.actor_critic.actor.act_inference(obs)

    def export(self, path):
        self.alg.actor_critic.export_policy(path)
