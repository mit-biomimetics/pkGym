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

from gpugym.envs.base.legged_robot_config import LeggedRobotCfg
from gpugym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class MITHumanoidCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 409
        num_observations = 66 # 187  # ! why? should be 66...
        num_actions = 18

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False
        # ! What's this?
        # measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 1mx1m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # terrain_types: [smooth slope, stairs up, stairs down]
        # terrain_proportions = [0.1, 0.25, 0.2]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [1.0, 5.0] # min max [m/s]
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.72]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw': 0.,
            'left_hip_abad': 0.,
            'left_hip_pitch': 0.,
            'left_knee': 0.,
            'left_ankle': 0.,
            'left_shoulder_pitch': 0.,
            'left_shoulder_abad': 0.,
            'left_shoulder_yaw': 0.,
            'left_elbow': 0.,
            'right_hip_yaw': 0.,
            'right_hip_abad': 0.,
            'right_hip_pitch': 0.,
            'right_knee': 0.,
            'right_ankle': 0.,
            'right_shoulder_pitch': 0.,
            'right_shoulder_abad': 0.,
            'right_shoulder_yaw': 0.,
            'right_elbow': 0.
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 40.,
                     'hip_abad': 40.,
                     'hip_pitch': 40.,
                     'knee': 40.,
                     'ankle': 40.,
                     'shoulder_pitch': 20.,
                     'shoulder_abad': 20.,
                     'shoulder_yaw': 20.,
                     'elbow': 20.,
                    }  # [N*m/rad]
        damping = {'hip_yaw': 1.,
                   'hip_abad': 1.,
                   'hip_pitch': 1.,
                   'knee': 1.,
                   'ankle': 1.,
                   'shoulder_pitch': 1.,
                   'shoulder_abad': 1.,
                   'shoulder_yaw': 1.,
                   'elbow': 1.,
                    }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 3.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6  # ! substeps?

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mit_humanoid/urdf/humanoid_R_sf.urdf'
        foot_name = 'foot'
        penalize_contacts_on = ["upper_leg", "lower_leg", "upper_arm", "lower_arm"]
        terminate_after_contacts_on = ['base']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        # soft_dof_pos_limit = 0.95
        # soft_dof_vel_limit = 0.9
        # soft_torque_limit = 0.9
        max_contact_force = 600.
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -20.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            lin_vel_z = -0.05
            ang_vel_xy = -0.0
            orientation = 0.
            torques = -5.e-5
            dof_vel = -0.
            dof_acc = 0.
            base_height = 0.
            feet_air_time = 1.  # rewards keeping feet in the air
            collision = -1.
            feet_stumble = -0.
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -1.
            no_fly = 0.25
            feet_contact_forces = -0.

        class normalization(LeggedRobotCfg.normalization):
            class obs_scales(LeggedRobotCfg.normalization.obs_scales):
                lin_vel = 1.0
                ang_vel = 1.0
                dof_pos = 1.0
                dof_vel = 1.0
                height_measurements = 1.0
            clip_observations = 100000000000000.
            clip_actions = 100000000000000.

        class noise(LeggedRobotCfg.noise):
            add_noise = False
            noise_level = 1.0 # scales other values
            class noise_scales(LeggedRobotCfg.noise.noise_scales):
                dof_pos = 0.01
                dof_vel = 1.5
                lin_vel = 0.1
                ang_vel = 0.2
                gravity = 0.05
                height_measurements = 0.1

        class sim(LeggedRobotCfg.sim):
            dt = 0.005
            substeps = 4


class MITHumanoidCfgPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'MIT_Humanoid'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 48
        max_iterations = 15000

        save_interval = 500