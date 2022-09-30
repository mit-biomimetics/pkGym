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

from gpugym import LEGGED_GYM_ROOT_DIR
import os
import copy
import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from gpugym.utils import KeyboardInterface, GamepadInterface
import numpy as np
import torch
from isaacgym import gymapi

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)

    # * prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # * load policy
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
    # * export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name,
                            'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        if SE_ON:
            se_path = os.path.join(path, 'se_1.jit')
            se_model = copy.deepcopy(state_estimator.state_estimator).to('cpu')
            traced_script_module = torch.jit.script(se_model)
            traced_script_module.save(se_path)

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    # interface = GamepadInterface(env)
    interface = KeyboardInterface(env)

    for i in range(10*int(env.max_episode_length)):
        # * handle interface
        # * handle state-estimation
        if SE_ON:
            se_obs = env.get_se_observations()
            se_prediction = state_estimator.predict(se_obs)
            obs = torch.cat((se_prediction.detach(), obs.detach()), dim=1)

        actions = policy(obs.detach())
        interface.update(env)
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
