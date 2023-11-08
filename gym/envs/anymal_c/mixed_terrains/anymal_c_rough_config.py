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
# ARE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gym.envs import AnymalCFlatCfg, AnymalCFlatCfgPPO


class AnymalCRoughCfg(AnymalCFlatCfg):
    class env(AnymalCFlatCfg.env):
        num_observations = 48

    class terrain(AnymalCFlatCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = False

    class asset(AnymalCFlatCfg.asset):
        # 1 to disable, 0 to enable...bitwise filter
        self_collisions = 1

    class commands(AnymalCFlatCfg.commands):
        resampling_time = 4.0

        class ranges(AnymalCFlatCfg.commands.ranges):
            yaw_vel = 1.5

    class domain_rand(AnymalCFlatCfg.domain_rand):
        friction_range = [0.0, 1.5]


class AnymalCRoughCCfgPPO(AnymalCFlatCfgPPO):
    class policy(AnymalCFlatCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"

    class algorithm(AnymalCFlatCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(AnymalCFlatCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_anymal_c"
        load_run = -1
        max_iterations = 300
