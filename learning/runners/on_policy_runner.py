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

import time
import os
import shutil
from collections import deque
from statistics import mean

import wandb
import torch
import numpy as np
from isaacgym.torch_utils import torch_rand_float

from learning.algorithms import PPO, StateEstimator
from learning.modules import ActorCritic
from learning.modules import StateEstimatorNN
from learning.env import VecEnv
from gym.utils.helpers import class_to_dict
from learning.utils import remove_zero_weighted_rewards
from learning.utils import Logger

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.device = device
        self.env = env
        self.parse_train_cfg(train_cfg)

        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        if self.cfg["SE_learner"] == "modular_SE":    # if using SE
            num_actor_obs += self.get_obs_size(self.se_cfg["targets"])
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])
        actor_critic: ActorCritic = actor_critic_class(num_actor_obs,
                                            num_critic_obs,
                                            num_actions,
                                            **self.policy_cfg).to(self.device)

        # ! this is hardcoded
        if self.cfg["algorithm_class_name"] == "PPO":
            alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
            self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        else:
            raise("No idea what algorithm you want from me here.")

        if self.cfg["SE_learner"] == "modular_SE":
            num_SE_obs = self.get_obs_size(self.se_cfg["obs"])
            num_SE_outputs = self.get_obs_size(self.se_cfg["targets"])
            state_estimator_nn = StateEstimatorNN(num_SE_obs,
                                                  num_SE_outputs,
                                                  **self.se_cfg["neural_net"])
            state_estimator_nn.to(self.device)
            self.state_estimator = StateEstimator(state_estimator_nn,
                                                  device=self.device,
                                                  **self.se_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.init_storage()

        # Log
        self.log_dir = log_dir
        self.SE_path = os.path.join(self.log_dir, 'SE')   # log_dir for SE
        self.wandb = None
        self.do_wandb = False
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.current_episode_rewards = {name: torch.zeros(self.env.num_envs, dtype=torch.float,
                                               device=self.device,
                                               requires_grad=False)
                        for name in self.policy_cfg["reward"]["weights"].keys()}
        self.current_episode_rewards.update({name: torch.zeros(self.env.num_envs,
                                               dtype=torch.float,
                                               device=self.device,
                                               requires_grad=False)
                        for name in self.policy_cfg["reward"]["termination_weight"].keys()})      
        self.cur_episode_length = torch.zeros(self.env.num_envs,
                                         dtype=torch.float, device=self.device)
        self.rewbuffer = {name:  deque(maxlen=100)
                        for name in  self.current_episode_rewards.keys()}   
        self.lenbuffer = deque(maxlen=100)

        self.mean_episode_length = 0.
        self.mean_rewards = {"Episode/"+name:  0.
                        for name in  self.current_episode_rewards.keys()} 
        self.total_mean_reward = 0.

        self.logger = Logger(log_dir)


    def parse_train_cfg(self, train_cfg):
        self.cfg = train_cfg['runner']
        self.alg_cfg = train_cfg['algorithm']
        remove_zero_weighted_rewards(train_cfg['policy']['reward']['weights'])
        self.policy_cfg = train_cfg['policy']

        if 'state_estimator' in train_cfg:
            self.se_cfg = train_cfg['state_estimator']
        else:
            self.se_cfg = None

    def configure_local_files(self, save_paths):
        # copy the relevant source files to the local logs for records
        save_dir = self.log_dir+'/files/'
        for save_path in save_paths:
            if save_path['type'] == 'file':
                os.makedirs(save_dir+save_path['target_dir'], exist_ok=True)
                shutil.copy2(save_path['source_file'], 
                              save_dir+save_path['target_dir'])
            elif save_path['type'] == 'dir':
                shutil.copytree(
                    save_path['source_dir'], save_dir+save_path['target_dir'],
                    ignore=shutil.ignore_patterns(*save_path['ignore_patterns']))
            else:
                print('WARNING: uncaught save path type:', save_path['type'])

    def init_storage(self):
        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        if self.cfg["SE_learner"] == "modular_SE":
            num_SE_obs = self.get_obs_size(self.se_cfg["obs"])
            num_SE_outputs = self.get_obs_size(self.se_cfg["targets"])
            self.state_estimator.init_storage(self.env.num_envs,
                                              self.num_steps_per_env,
                                              [num_SE_obs],
                                              [num_SE_outputs])
            num_actor_obs += num_SE_outputs
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              actor_obs_shape=[num_actor_obs],
                              critic_obs_shape=[num_critic_obs],
                              action_shape=[num_actions])

    def configure_wandb(self, wandb_in, log_freq=100, log_graph=True):
        self.wandb = wandb_in
        self.do_wandb = True
        self.wandb.watch((self.alg.actor_critic.actor,
                          self.alg.actor_critic.critic),
                         log_freq=log_freq,
                         log_graph=log_graph)

    def reset_learn(self):
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        if self.cfg["SE_learner"] == "modular_SE":
            SE_obs = self.get_obs(self.se_cfg["obs"])
            SE_estimate = self.state_estimator.predict(SE_obs)
            actor_obs = torch.cat((SE_estimate, actor_obs), dim=1)
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])

        self.alg.actor_critic.train()
        self.num_learning_iterations = num_learning_iterations
        self.tot_iter = self.current_learning_iteration + num_learning_iterations
        for self.it in range(self.current_learning_iteration, self.tot_iter):
            start = time.time()
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    self.set_actions(actions)
                    self.env.step()

                    actor_obs = self.get_noisy_obs(self.policy_cfg["actor_obs"],
                                                   self.policy_cfg["noise"])
                    critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
                    dones = self.get_dones()
                    rewards = self.get_and_log_rewards(self.policy_cfg["reward"]["weights"],
                                                       modifier=self.env.dt)
                    rewards += self.get_and_log_rewards(self.policy_cfg["reward"]["termination_weight"])

                    if self.cfg["SE_learner"] == "modular_SE":
                        SE_obs = self.get_obs(self.se_cfg["obs"])
                        SE_estimate = self.state_estimator.predict(SE_obs)
                        actor_obs = torch.cat((SE_estimate, actor_obs), dim=1)
                        SE_targets = self.get_obs(self.se_cfg["targets"])
                        self.state_estimator.process_env_step(SE_obs,
                                                              SE_targets)
                    if self.cfg["algorithm_class_name"] == "PPO":
                        timed_out = self.get_timed_out()
                        self.alg.process_env_step(rewards, dones, timed_out)

                    self.update_episode_buf(dones)

                stop = time.time()
                self.collection_time = stop - start
                # * Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            self.mean_value_loss, self.mean_surrogate_loss = self.alg.update()
            if self.cfg["SE_learner"] == "modular_SE":
                SE_loss = self.state_estimator.update()
            stop = time.time()
            self.learn_time = stop - start
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += self.collection_time + self.learn_time
            self.log()
            self.logger.log_to_wandb()
            self.logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.it)))
                if self.cfg["SE_learner"] == "modular_SE": 
                    if not os.path.exists(self.SE_path):
                        os.makedirs(self.SE_path)
                    self.save_SE(os.path.join(self.SE_path, 'SE_{}.pt'.format(self.it)))

        self.current_learning_iteration += num_learning_iterations
        # * save
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        if self.cfg["SE_learner"] == "modular_SE": 
            self.save_SE(os.path.join(self.SE_path, 'SE_{}.pt'.format(self.current_learning_iteration)))

    def get_noise(self, obs_list, noise_dict):
        noise_vec = torch.zeros(self.get_obs_size(obs_list), device=self.device)
        obs_index = 0
        for obs in obs_list:
            obs_size = self.get_obs_size([obs])
            if obs in noise_dict.keys():
                noise_tensor = torch.ones(obs_size).to(self.device) * noise_dict[obs]
                if obs in self.env.scales.keys():
                    noise_tensor *= self.env.scales[obs]
                noise_vec[obs_index:obs_index+obs_size] = noise_tensor
            obs_index += obs_size
        return torch_rand_float(-1., 1., (self.env.num_envs, len(noise_vec)), 
                                            self.device) * noise_vec
    
    def get_noisy_obs(self, obs_list, noise_dict):
        observation = self.get_obs(obs_list)
        return observation + self.get_noise(obs_list, noise_dict)

    def get_obs(self, obs_list):
        observation = self.env.get_states(obs_list).to(self.device)
        return observation

    def set_actions(self,actions):
        self.env.set_states(self.policy_cfg["actions"], actions)

    def get_timed_out(self):
        return self.env.get_states(["timed_out"]).to(self.device)

    def get_obs_size(self, obs_list):
        # todo make unit-test to assert len(shape)==1 always
        return self.get_obs(obs_list)[0].shape[0]

    def get_action_size(self, action_list):
        return self.env.get_states(action_list)[0].shape[0]

    def get_rewards(self, reward_weights, modifier=1):
        return self.env.compute_reward(reward_weights, modifier).to(self.device)

    def get_and_log_rewards(self, reward_weights,
                            modifier=1):
        '''
        Computes each reward on the fly and returns.
        Also takes care of logging...
        '''
        total_rewards = torch.zeros(self.env.num_envs,
                                    device=self.device, dtype=torch.float)
        for name, weight in reward_weights.items():
            reward = self.env.compute_reward({name: weight},
                                             modifier).to(self.device)
            total_rewards += reward
            self.log_current_reward(name, reward)
        return total_rewards

    def log_current_reward(self, name, reward):
        if name in self.current_episode_rewards.keys():
            self.current_episode_rewards[name] += reward  
    
    def update_episode_buf(self, dones):
        self.cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        for name in self.current_episode_rewards.keys():
            self.rewbuffer[name].extend(self.current_episode_rewards[name]
                                        [new_ids][:, 0].cpu().numpy().tolist())
            self.current_episode_rewards[name][new_ids] = 0.
        self.lenbuffer.extend(self.cur_episode_length[new_ids]
                              [:, 0].cpu().numpy().tolist())
        self.cur_episode_length[new_ids] = 0
        if (len(self.lenbuffer) > 0):
            self.calculate_reward_avg()

    def calculate_reward_avg(self):
        self.mean_episode_length = mean(self.lenbuffer)
        self.mean_rewards = {"Episode/"+name:  mean(self.rewbuffer[name])
                        for name in  self.current_episode_rewards.keys()} 
        self.total_mean_reward = mean(list(self.mean_rewards.values()))


    def log(self):
        fps = int(self.num_steps_per_env * self.env.num_envs \
                  / (self.collection_time+self.learn_time))
        mean_noise_std = self.alg.actor_critic.std.mean().item()
        self.logger.add_log(self.mean_rewards)
        self.logger.add_log({
                             "Loss/value_function" : self.mean_value_loss,
                             "Loss/surrogate" : self.mean_surrogate_loss,
                             "Loss/learning_rate": self.alg.learning_rate,
                             "Policy/mean_noise_std" : mean_noise_std,
                             "Perf/total_fps": fps,
                             "Perf/collection_time": self.collection_time,
                             "Perf/learning_time" : self.learn_time,
                             "Train/mean_reward" : self.total_mean_reward,
                             "Train/mean_episode_length" : self.mean_episode_length,
                             "Train/total_timesteps": self.tot_timesteps,
                             "Train/iteration_time": self.collection_time+self.learn_time,
                             "Train/time": self.tot_time,
                             })
        self.logger.update_iterations(self.it, self.tot_iter, 
                                      self.num_learning_iterations)
                        
        #TODO: iterate through the config for any extra things you might want to log

    def get_dones(self):
        return self.env.reset_buf.to(self.device)

    def get_infos(self):
        return self.env.extras


    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def save_SE(self, path, infos=None):
        torch.save({
            'model_state_dict': self.state_estimator.state_estimator.state_dict(),
            'optimizer_state_dict': self.state_estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']

        if self.cfg["SE_learner"] == "modular_SE": 
            SE_path = path.replace("/model_", "/SE/SE_")
            SEloaded_dict = torch.load(SE_path)
            self.state_estimator.state_estimator.load_state_dict(SEloaded_dict['model_state_dict'])

        return loaded_dict['infos']


    def get_state_estimator(self, device=None):
        self.state_estimator.state_estimator.eval()
        if device is not None:
            self.state_estimator.state_estimator.to(device)
        return self.state_estimator


    def get_inference_policy(self, device=None):
        # switch to evaluation mode (dropout for example)
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.actor.act_inference


    def export(self, path):
        self.alg.actor_critic.export_policy(path)
        if self.cfg["SE_learner"] is not None:
            self.state_estimator.export(path)