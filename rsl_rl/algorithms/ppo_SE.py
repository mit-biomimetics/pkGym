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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, StateEstimator
from rsl_rl.storage import RolloutStorage, LongTermStorage

class PPO_SE:
    actor_critic: ActorCritic
    state_estimator: StateEstimator
    def __init__(self,
                 actor_critic,
                 state_estimator,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 storage_size=4000,
                 priv_obs_only=False
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(),
                                    lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # SE components
        self.state_estimator = state_estimator
        self.state_estimator.to(self.device)
        self.SE_optimizer = optim.Adam(self.state_estimator.parameters(),
                                    lr=learning_rate)
        self.SE_loss_fn = nn.MSELoss()
         # * will only need once we use the LT storage
         # * for now, using batch from actor_critic
        # self.SE_epochs = epochs
        # self.SE_batch_size = batch_size
        # self.SE_n_batch_samples = n_batch_samples

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # ppo plus params
        self.LT_storage_size = storage_size
        self.LT_priv_obs_only = priv_obs_only

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                        critic_obs_shape, action_shape, se_shape):

        self.storage = LongTermStorage(num_envs, num_transitions_per_env,
                                        self.LT_storage_size,
                                        actor_obs_shape,
                                        critic_obs_shape,
                                        action_shape,
                                        se_shape,
                                        self.LT_priv_obs_only,
                                        self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, new_actor_obs, new_critic_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.transition.SE_targets = infos['SE_targets']
        # Record the transition

        self.storage.rollout.add_transitions(self.transition)
        # * add transitions that are not time-outs to LT storage

        keep = ~infos['time_outs'].to(self.device)
        self.storage.add_LT_transitions(self.transition.observations[keep, :],
                            self.transition.critic_observations[keep, :],
                            self.transition.actions[keep, :],
                            new_actor_obs[keep, :],
                            new_critic_obs[keep, :],
                            self.transition.rewards[keep].unsqueeze(1))
                            # possibly add dones (mark failure terminations)
        # if self.LT_crit_obs_only:
        # start = self.storage.data_count
        # keep = ~infos['time_outs'].to(self.device)
        # n_keep = sum(~infos['time_outs'].to(self.device))
        # self.storage.actor_obs[start:start+n_keep, :] = \ 
        #                         self.transition.observations[keep, :]
        # self.storage.critic_obs[start:start+n_keep, :] = \
        #                         self.transition.critic_observations[keep, :]
        # self.storage.actions[start:start+n_keep, :] = \
        #                         self.transition.actions[keep, :]
        # self.storage.next_actor_obs[start:start+n_keep, :] = \
        #                         new_actor_obs[keep, :]
        # self.storage.next_critic_obs[start:start+n_keep, :] = \
        #                         new_critic_obs[keep, :]
        # self.storage.data_count += n_keep
        # * clear and reset
        self.transition.clear()
        self.actor_critic.reset(dones)  # does NOTHING


    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.rollout.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.rollout.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.rollout.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, SE_target_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch,
                                        hidden_states=hid_states_batch[0])
                actions_log_prob_batch = \
                        self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch,
                                            masks=masks_batch,
                                            hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch \
                            - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) \
                                    * torch.clamp(ratio, 1.0 - self.clip_param,
                                                    1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss \
                        - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                            self.max_grad_norm)
                self.optimizer.step()
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                # * do a pass on the state-estimator too
                SE_prediction = self.state_estimator.evaluate(obs_batch,
                                        masks=masks_batch,
                                        hidden_states=hid_states_batch[0])
                SE_loss = self.SE_loss_fn(SE_prediction, SE_target_batch)
                self.SE_optimizer.zero_grad()
                SE_loss.backward()
                self.SE_optimizer.step()



        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.rollout.clear()

        return mean_value_loss, mean_surrogate_loss

    # def update_LT_storage(self):
    #     # for now, just randomly select a mix of previous and new obs

    #     # todo if all zeros (or anyway not yet initialized), then just copy over
    #     # todo: actually, use a step-var to keep track of how full it is

    #     n_LT = self.storage.data_count
    #     n_LT_max = self.storage.LT_storage_size
    #     n_new = self.storage.observations.flatten(end_dim=1).shape[0]

    #     if n_LT >= n_LT_max:
    #         all_obs = torch.cat((self.LT_storage.obs,
    #                             self.storage.observations.flatten(end_dim=1)),
    #                             dim=0)
    #         indices = torch.randperm(n_LT + n_new)[:n_LT]
    #         self.LT_storage.obs = all_obs[indices, :]
    #     elif n_LT+n_new >= n_LT_max:  # keep a random set
    #         all_obs = torch.cat((self.LT_storage.obs[:n_LT, :],
    #                             self.storage.observations.flatten(end_dim=1)),
    #                             dim=0)
    #         indices = torch.randperm(n_LT + n_new)[:n_LT_max]
    #         self.LT_storage.obs = all_obs[indices, :]
    #         self.LT_storage.data_count = n_LT_max
    #     else:  # just fill
    #         self.LT_storage.obs[self.LT_storage.data_count:self.LT_storage.data_count+n_new, :] = self.storage.observations.flatten(end_dim=1)
    #         self.LT_storage.data_count += n_new