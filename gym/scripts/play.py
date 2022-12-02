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

from gym import LEGGED_GYM_ROOT_DIR
import os
import copy
import isaacgym
from gym.envs import *
from gym.utils import  get_args, task_registry, Logger, class_to_dict
from gym.utils import KeyboardInterface, GamepadInterface
import numpy as np
import torch
import isaacgym


def play(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    env, env_cfg = task_registry.make_env(name=args.task, env_cfg=env_cfg)

    task_registry.prepare_sim()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env,
                                                          name=args.task,
                                                          args=args,
                                                          train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    if train_cfg.runner.SE_learner == "modular_SE":
        SE_ON = True
        state_estimator = ppo_runner.get_state_estimator(device=env.device)
    else:
        SE_ON = False

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name, 'exported')
        ppo_runner.export(path)

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    # interface = GamepadInterface(env)
    interface = KeyboardInterface(env)

    obs = env.get_states(train_cfg.policy.actor_obs)
    for i in range(10*int(env.max_episode_length)):
        # * handle interface
        # * handle state-estimation
        if SE_ON:
            SE_obs = env.get_states(train_cfg.state_estimator.obs)
            SE_prediction = state_estimator.predict(SE_obs)
            obs = torch.cat((SE_prediction.detach(), obs.detach()), dim=1)

        actions = policy(obs.detach())
        interface.update(env)
        env.set_states(train_cfg.policy.actions, actions)
        env.step()
        obs = env.get_states(train_cfg.policy.actor_obs)

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
