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

import importlib
from gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.base_task import BaseTask
from .base.legged_robot import LeggedRobot
from .mit_humanoid.mit_humanoid import MIT_Humanoid
from .mit_humanoid.mit_humanoid_config import (MITHumanoidCfg,
                                               MITHumanoidRunnerCfg)
from .mini_cheetah.mini_cheetah import MiniCheetah
from .mini_cheetah.mini_cheetah_config import (MiniCheetahCfg,
                                               MiniCheetahRunnerCfg)
from .cartpole.cartpole import Cartpole
from .cartpole.cartpole_config import CartpoleCfg, CartpoleRunnerCfg
from .mini_cheetah.mini_cheetah_ref import MiniCheetahRef
from .mini_cheetah.mini_cheetah_ref_config import MiniCheetahRefCfg, MiniCheetahRefRunnerCfg
from gym.utils.task_registry import task_registry

# To add a new env:
# 1. add the base env and env class name and location to the class dict
# 2. add the config name and location to the config dict
# 3. add the runner confg name and location to the runner config dict
# 3. register the task experiment name to the env/config/ppo classes

# from y import x where {y:x}
class_dict = {
    'LeggedRobot': '.base.legged_robot',
    'FixedRobot': '.base.fixed_robot',
    'Cartpole': '.cartpole.cartpole',
    'MiniCheetah': '.mini_cheetah.mini_cheetah',
    'MiniCheetahRef': '.mini_cheetah.mini_cheetah_ref',
    'MIT_Humanoid': '.mit_humanoid.mit_humanoid'
}

config_dict = {
    'CartpoleCfg': '.cartpole.cartpole_config',
    'MiniCheetahCfg': '.mini_cheetah.mini_cheetah_config',
    'MITHumanoidCfg': '.mit_humanoid.mit_humanoid_config',
    'MiniCheetahRefCfg': '.mini_cheetah.mini_cheetah_ref_config'
}

runner_config_dict = {
    'CartpoleRunnerCfg': '.cartpole.cartpole_config',
    'MiniCheetahRunnerCfg': '.mini_cheetah.mini_cheetah_config',
    'MITHumanoidRunnerCfg': '.mit_humanoid.mit_humanoid_config',
    'MiniCheetahRefRunnerCfg': '.mini_cheetah.mini_cheetah_ref_config'
}

task_dict = {
    'humanoid': ['MIT_Humanoid', 'MITHumanoidCfg', 'MITHumanoidRunnerCfg'],
    'mini_cheetah': ['MiniCheetah', 'MiniCheetahCfg', 'MiniCheetahRunnerCfg'],
    'mini_cheetah_ref':
        ['MiniCheetahRef', 'MiniCheetahRefCfg', 'MiniCheetahRefRunnerCfg'],
    'cartpole': ['Cartpole', 'CartpoleCfg', 'CartpoleRunnerCfg']
}


for class_name, class_location in class_dict.items():
    locals()[class_name] = getattr(
        importlib.import_module(class_location, __name__), class_name)
for config_name, config_location in config_dict.items():
    locals()[config_name] = getattr(
        importlib.import_module(config_location, __name__), config_name)
for runner_config_name, runner_config_location in runner_config_dict.items():
    locals()[runner_config_name] = getattr(
        importlib.import_module(
            runner_config_location, __name__), runner_config_name)

for task_name, class_list in task_dict.items():
    task_registry.register(task_name,
                           locals()[class_list[0]],
                           locals()[class_list[1]](),
                           locals()[class_list[2]]())
