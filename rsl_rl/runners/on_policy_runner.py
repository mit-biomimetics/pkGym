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
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import wandb
import torch

from rsl_rl.algorithms import PPO, StateEstimator
from rsl_rl.modules import ActorCritic
from rsl_rl.modules import StateEstimatorNN
from rsl_rl.env import VecEnv

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.parse_train_cfg(train_cfg)
        self.device = device
        self.env = env

        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        num_actor_obs = self.env.num_obs
        if self.cfg["SE_learner"] == "modular_SE":    # if using SE
            num_actor_obs += self.se_nn_cfg['num_outputs']
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = num_actor_obs
        actor_critic: ActorCritic = actor_critic_class(num_actor_obs,
                                            num_critic_obs,
                                            self.env.num_actions,
                                            **self.policy_cfg).to(self.device)

        # ! this is hardcoded
        if self.cfg["algorithm_class_name"] == "PPO":
            alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
            self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        else:
            raise("No idea what algorithm you want from me here.")

        if self.cfg["SE_learner"] == "modular_SE": 
            self.state_estimator_nn = StateEstimatorNN(self.env.num_se_obs,
                                                       **self.se_nn_cfg)
            self.state_estimator_nn.to(self.device)
            self.state_estimator = StateEstimator(self.state_estimator_nn ,device=self.device, **self.se_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.init_storage()

        # Log
        self.log_dir = log_dir
        self.SE_path = os.path.join(self.log_dir, 'SE')   # log_dir for SE
        self.wandb = None
        self.do_wandb = False
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()


    def parse_train_cfg(self, train_cfg):
        self.cfg=train_cfg['runner']
        self.alg_cfg = train_cfg['algorithm']
        self.policy_cfg = train_cfg['policy']
        if 'state_estimator' in train_cfg:
            self.se_cfg = train_cfg['state_estimator']
            self.se_nn_cfg = train_cfg['state_estimator_nn']
        else:
            self.se_cfg = None


    def init_storage(self):
        actor_obs_shape = self.env.num_obs
        if self.cfg["SE_learner"] == "modular_SE":
            self.state_estimator.init_storage(self.env.num_envs,
                                              self.num_steps_per_env,
                                              [self.env.num_se_obs],
                                              [self.se_nn_cfg['num_outputs']])
            actor_obs_shape += self.se_nn_cfg['num_outputs']
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              actor_obs_shape=[actor_obs_shape],
                              critic_obs_shape=[self.env.num_privileged_obs],
                              action_shape=[self.env.num_actions])


    def configure_wandb(self, wandb):
        self.wandb = wandb
        self.do_wandb = True
        self.wandb.watch((self.alg.actor_critic.actor, self.alg.actor_critic.critic), log_freq=100, log_graph=True)


    def reset_learn(self):
        self.current_learning_iteration = 0


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        # actor_obs = self.env.get_observations()
        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        if self.cfg["SE_learner"] == "modular_SE":
            SE_obs = self.get_obs(self.se_cfg["obs"])
            SE_estimate = self.state_estimator.predict(SE_obs)
            actor_obs = torch.cat((SE_estimate, actor_obs), dim=1)
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])

        self.alg.actor_critic.train()

        ep_infos = []
        success_counts_infos = []
        episode_counts_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)                        # compute SE prediction and actions and values
                    # * step simulation
                    # actor_obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    self.env.step(actions)
                    actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
                    critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
                    dones = self.get_dones()
                    infos = self.get_infos()
                    rewards = self.get_rewards()
                    if self.cfg["SE_learner"] == "modular_SE":
                        SE_obs = self.get_obs(self.se_cfg["obs"])
                        SE_estimate = self.state_estimator.predict(SE_obs)
                        actor_obs = torch.cat((SE_estimate, actor_obs), dim=1)

                        SE_targets = self.get_obs(self.se_cfg["targets"])
                        self.state_estimator.process_env_step(SE_obs,
                                                              SE_targets)

                    if self.cfg["algorithm_class_name"] == "PPO":
                        self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # * Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'success counts' in infos and 'episode counts' in infos:
                            success_counts_infos.append(infos['success counts'])
                            episode_counts_infos.append(infos['episode counts'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # * Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            if self.cfg["SE_learner"] == "modular_SE":
                SE_loss = self.state_estimator.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                if self.cfg["SE_learner"] == "modular_SE": 
                    if not os.path.exists(self.SE_path):
                        os.makedirs(self.SE_path)
                    self.save_SE(os.path.join(self.SE_path, 'SE_{}.pt'.format(it)))
            ep_infos.clear()
        self.current_learning_iteration += num_learning_iterations
        # * save
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        if self.cfg["SE_learner"] == "modular_SE": 
            self.save_SE(os.path.join(self.SE_path, 'SE_{}.pt'.format(self.current_learning_iteration)))


    def get_obs(self, obs_list):
        return self.env.get_obs(obs_list).to(self.device)


    def get_rewards(self):
        # TODO change this (on gpugym side) to actually compute a reward tensor on
        # the fly, and return in. And get rid of the need for a buffer.
        # This means moving prepare rewards etc. to on_policy_runner
        return self.env.rew_buf.to(self.device)



    def get_dones(self):
        return self.env.reset_buf.to(self.device)


    def get_infos(self):
        return self.env.extras


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        # Craft logging info database
        wandb_to_log = {'Episode/Total_reward': 0.0}

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                wandb_to_log['Episode/' + key] = value
                wandb_to_log['Episode/Total_reward'] += value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        if locs['success_counts_infos'] and locs['episode_counts_infos']:
            # This counts all the agent resets that occurred during the logging period
            total_tries = torch.sum(torch.tensor([logged_info['total_reset'] for logged_info in locs['episode_counts_infos']]))
            for key in locs['success_counts_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for success_counts_info in locs['success_counts_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(success_counts_info[key], torch.Tensor):
                        success_counts_info[key] = torch.Tensor([success_counts_info[key]])
                    if len(success_counts_info[key].shape) == 0:
                        success_counts_info[key] = success_counts_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, success_counts_info[key].to(self.device)))
                success_count = torch.sum(infotensor)
                success_rate = success_count / total_tries
                self.writer.add_scalar('Success Rates/' + key, success_rate, locs['it'])
                wandb_to_log['Success Rates/' + key] = success_rate
                ep_string += f"""{f'Mean Success Rate {key}:':>{pad}} {success_rate*100:.1f}%\n"""

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        if self.cfg["SE_learner"] == "modular_SE": 
            self.writer.add_scalar('Loss/SE_loss', locs['SE_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        if self.do_wandb:
            wandb_to_log['Loss/value_function'] = locs['mean_value_loss']
            wandb_to_log['Loss/surrogate'] = locs['mean_surrogate_loss']
            wandb_to_log['Loss/learning_rate'] = self.alg.learning_rate
            wandb_to_log['Policy/mean_noise_std'] = mean_std.item()
            wandb_to_log['Perf/total_fps'] = fps
            wandb_to_log['Perf/collection time'] = locs['collection_time']
            wandb_to_log['Perf/learning_time'] = locs['learn_time']
            self.wandb.log(wandb_to_log, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

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
        if self.se_cfg is not None:
            self.state_estimator.export(path)