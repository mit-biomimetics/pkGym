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

import numpy as np
import os
from datetime import datetime

import isaacgym
from gpugym.envs import *
from gpugym.utils import get_args, task_registry
from gpugym import LEGGED_GYM_ROOT_DIR
import torch

import wandb

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    do_kin = env_cfg.env.obs_augmentations.add_kinematics_augmentations
    do_jac = env_cfg.env.obs_augmentations.add_jacobian_augmentations
    do_cen = env_cfg.env.obs_augmentations.add_centripetal_augmentations
    do_cor = env_cfg.env.obs_augmentations.add_coriolis_augmentations

    wandb.config = {
              'num+observations': env_cfg.env.num_observations,
              'kinematic_aug': do_kin,
              'jacobian_aug': do_jac,
              'centripetal_aug': do_cen,
              'coriolos_aug': do_cor,
              'nn_shape': train_cfg.policy.actor_hidden_dims
        }

    experiment_name = f'{args.task}' + f'{"_Kin" if do_kin else ""}' + \
        f'{"_Jac" if do_jac else ""}' + \
        f'{"_Cen" if do_cen else ""}' + \
        f'{"_Cor" if do_cor else ""}'

    wandb.init(project=args.wandb_project,
                entity=args.wandb_entity,
                config=wandb.config,
                name=experiment_name)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    wandb.tensorboard.patch(root_logdir=log_dir)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
