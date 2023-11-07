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

from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 0.5


class AnymalCFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": -0.0,
            "RH_HAA": -0.0,
            "LF_HFE": 0.4,
            "LH_HFE": -0.4,
            "RF_HFE": 0.4,
            "RH_HFE": -0.4,
            "LF_KFE": -0.8,
            "LH_KFE": 0.8,
            "RF_KFE": -0.8,
            "RH_KFE": 0.8,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {"HAA": 80.0, "HFE": 80.0, "KFE": 80.0}  # [N*m/rad]
        damping = {"HAA": 2.0, "HFE": 2.0, "KFE": 2.0}  # [N*m*s/rad]
        use_actuator_network = False
        actuator_net_file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/" + "actuator_nets/anydrive_v3_lstm.pt"
        )
        ctrl_frequency = 80
        desired_sim_frequency = 400

    class commands:
        resampling_time = 10.0  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 3.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 2  # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 1
        max_push_vel_xy = 0.5

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" + "anymal_c/urdf/anymal_c.urdf"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK"]
        terminate_after_contacts_on = ["BASE", "THIGH"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 3.14 * (BASE_HEIGHT_REF / 9.81) ** 0.5
        base_lin_vel = 1.0
        commands = 1
        dof_vel = 100.0  # ought to be roughly max expected speed.
        base_height = BASE_HEIGHT_REF
        dof_pos = 4 * [0.1, 1.0, 2]  # hip-abad, hip-pitch, knee
        dof_pos_obs = dof_pos
        # Action scales
        dof_pos_target = dof_pos
        tau_ff = 4 * [80, 80, 80]  # hip-abad, hip-pitch, knee


class AnymalCFlatRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"

        actor_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_history",
        ]

        critic_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_history",
        ]

        actions = ["dof_pos_target"]

        class noise:
            dof_pos_obs = 0.005  # can be made very low
            dof_vel = 0.005
            base_ang_vel = 0.05  # 0.027, 0.14, 0.37
            projected_gravity = 0.02

        class reward(LeggedRobotRunnerCfg.policy.reward):
            class weights(LeggedRobotRunnerCfg.policy.reward.weights):
                tracking_lin_vel = 3.0
                tracking_ang_vel = 1.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.0
                orientation = 1.0
                torques = 5.0e-6
                dof_vel = 1.0
                base_height = 0.2
                action_rate = 0.001  # -0.01
                action_rate2 = 0.0001  # -0.001
                stand_still = 0.5
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.1

            class termination_weight:
                termination = 1.0

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # training params, set to be same as mini_cheetah for right now
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        # mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 4
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        discount_horizon = 2.0  # [s]
        GAE_bootstrap_horizon = 0.5  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "flat_anymal_c"
        algorithm_class_name = "PPO"
        max_iterations = 1000  # number of policy updates
        num_steps_per_env = 24
