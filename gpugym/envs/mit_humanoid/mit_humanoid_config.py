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
        num_envs = 3000
        num_observations = 67+3*18 # 187
        num_actions = 18
        episode_length_s = 10  # episode length in seconds


    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0., 0.] # min max [m/s]
            lin_vel_y = [0., 0]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [0, 0]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = "reset_to_basic" # default setup chooses how the initial conditions are chosen.
                                # "reset_to_basic" = a single position with some randomized noise on top. 
                                # "reset_to_range" = a range of joint positions and velocities.
                                #  "reset_to_traj" = feed in a trajectory to sample from. 
        penetration_check = False  # disable to not check for penetration on initial conds.

        #default for normalization and basic initialization 
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw': 0.,
            'left_hip_abad': 0.,
            'left_hip_pitch': -0.4,
            'left_knee': 0.77,  
            'left_ankle': -0.37,
            'left_shoulder_pitch': 0.,
            'left_shoulder_abad': 0.,
            'left_shoulder_yaw': 0.,
            'left_elbow': 0.,
            'right_hip_yaw': 0.,
            'right_hip_abad': 0.,
            'right_hip_pitch': -0.4,
            'right_knee': 0.77,  # 0.785,
            'right_ankle': -0.37,
            'right_shoulder_pitch': 0.,
            'right_shoulder_abad': 0.,
            'right_shoulder_yaw': 0.,
            'right_elbow': 0.
        }

        #default COM for basic initialization 
        pos = [0.0, 0.0, 0.66]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # initialization for random range setup
        dof_pos_high = [0.,0.,0.,0.75, 0., 0., 0.,0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0. ] #DOF dimensions
        dof_pos_low =  [0.,0.,0.,0., 0., 0., 0.,0.,0.,   0., 0., 0., 0.,0., 0., 0., 0., 0. ]
        dof_vel_high = [0.,0.,0.,0.0, 0., 0., 0.,0.,0.0, 0., 0., 0., 0.,0., 0., 0., 0., 0. ]
        dof_vel_low = [0.,0.,0.,0.0, 0., 0., 0.,0.,0.0, 0., 0., 0., 0.,0., 0., 0., 0., 0. ]

        com_pos_high = [0.,0.,1., 0., 0.5, 0.] #COM dimensions, in euler angles because randomizing in quat is confusing
        com_pos_low = [0.,0.,1., 0., -0.5, 0.] #x, y ,z, roll, pitch, yaw
        com_vel_high = [0.,0.,0., 0., 0.0, 0.]
        com_vel_low = [0.,0.,0.,0.,0.,0.]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100.,
                     'hip_abad': 100.,
                     'hip_pitch': 100.,
                     'knee': 100.,
                     'ankle': 100.,
                     'shoulder_pitch': 20.,
                     'shoulder_abad': 20.,
                     'shoulder_yaw': 20.,
                     'elbow': 20.,
                    }  # [N*m/rad]
        damping = {'hip_yaw': 2.,
                   'hip_abad': 2.,
                   'hip_pitch': 2.,
                   'knee': 2.,
                   'ankle': 2.,
                   'shoulder_pitch': 0.5,
                   'shoulder_abad': 0.5,
                   'shoulder_yaw': 0.5,
                   'elbow': 5,
                    }  # [N*m*s/rad]     # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # ! substeps?

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 2
        max_push_vel_xy = 0.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mit_humanoid/urdf/humanoid_R_sf.urdf'
        foot_name = 'foot'
        penalize_contacts_on = ['base', 'arm']
        terminate_after_contacts_on = ['base' ]
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disagble, 0 to enable...bitwise filter
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        disable_gravity = False
        disable_actions = False
        disable_motors = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 1000.

        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        base_height_target = 0.6565
        tracking_sigma = 0.25

        swing_height_target = 0.1

        # this is scaling sqrd-exp width
        base_yaw_rate_tracking = 0.7
        swing_height_tracking = 0.08

        joint_level_scaling = [1.0,  # left_hip_yaw
                               2.0,  # left_hip_abad
                               1.0,  # left_hip_pitch
                               1.0,  # left_knee
                               1.0,  # left_ankle
                               1.0,  # left_shoulder_pitch
                               1.0,  # left_shoulder_abad
                               1.0,  # left_shoulder_yaw
                               1.0,  # left_elbow
                               1.0,  # right_hip_yaw
                               2.0,  # right_hip_abad
                               1.0,  # right_hip_pitch
                               1.0,  # right_knee
                               1.0,  # right_ankle
                               1.0,  # right_shoulder_pitch
                               1.0,  # right_shoulder_abad
                               1.0,  # right_shoulder_yaw
                               1.0]  # right_elbow

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -1.
            tracking_lin_vel = 0.0001
            tracking_ang_vel = 0.02
            lin_vel_z = 0.5
            ang_vel_xy = -0.1
            orientation = 0.1
            torques = -5.e-7
            dof_vel = 0.01
            base_height = 1.
            dof_near_home = 1.
            action_rate = -0.01
            action_rate2 = -0.001
            collision = -1.


            stand_still = -0.
            dof_pos_limits = -0.0
            no_fly = 0.0
            feet_contact_forces = -0.
            symm_legs = 0.#0.000002
            symm_arms = 0.0
            feet_air_time = 0.0  # rewards keeping feet in the air

    class normalization(LeggedRobotCfg.normalization):
            class obs_scales(LeggedRobotCfg.normalization.obs_scales):
                # * helper fcts
                # * dimensionless time: sqrt(L/g) or sqrt(I/[mgL]), with I=I0+mL^2
                dimless_time = (0.7/9.81)**0.5
                v_leg = 0.72
                # lin_vel = 1/v_leg*dimless_time
                base_z = 1./0.6565
                lin_vel =  1./v_leg  # virtual leg lengths per second
                # ang_vel = 0.25
                ang_vel = 1./3.14*dimless_time
                dof_pos = 1./3.14
                dof_vel = 0.05  # ought to be roughly max expected speed.

                height_measurements = 1./0.6565
            # clip_observations = 100.
            clip_actions = 1000.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.1  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.002
        substeps = 1
        gravity = [0., 0., -9.81]

class MITHumanoidCfgPPO(LeggedRobotCfgPPO):

    do_wandb = True

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 25
        max_iterations = 1000
        run_name = 'Standing'
        experiment_name = 'MIT_Humanoid_Stand'
        save_interval = 50

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
