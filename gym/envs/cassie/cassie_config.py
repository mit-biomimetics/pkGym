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


from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CassieRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48  # 169
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.5]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "hip_abduction_left": 0.2,  # -0.3927 : 0.2618
            "hip_rotation_left": 0.3,  # -0.3927 : 0.3927"
            "hip_flexion_left": 1.0,  # -0.8727 : 1.3963
            "thigh_joint_left": -0.8,  # -2.8623 : -0.6458
            "ankle_joint_left": 1.57,  # 0.6458  : 2.8623
            "toe_joint_left": -0.57,  # -2.4435  : -0.5236
            "hip_abduction_right": -0.3,
            "hip_rotation_right": -0.25,
            "hip_flexion_right": 1.0,
            "thigh_joint_right": -2.8,
            "ankle_joint_right": 1.57,
            "toe_joint_right": -2.17,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            "hip_abduction": 100.0,
            "hip_rotation": 100.0,
            "hip_flexion": 200.0,
            "thigh_joint": 200.0,
            "ankle_joint": 200.0,
            "toe_joint": 40.0,
        }  # [N*m/rad]
        damping = {
            "hip_abduction": 3.0,
            "hip_rotation": 3.0,
            "hip_flexion": 6.0,
            "thigh_joint": 6.0,
            "ankle_joint": 6.0,
            "toe_joint": 1.0,
        }  # [N*m*s/rad]     # [N*m*s/rad]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf"
        foot_name = "toe"
        terminate_after_contacts_on = ["pelvis"]
        flip_visual_attachments = False
        disable_gravity = True
        self_collisions = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.0

        class weights(LeggedRobotCfg.rewards.weights):
            termination = -200.0
            tracking_ang_vel = 1.0
            torques = -5.0e-6
            lin_vel_z = -0.5
            feet_air_time = 5.0
            dof_pos_limits = -1.0
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.0


class CassieRoughCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_cassie"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
