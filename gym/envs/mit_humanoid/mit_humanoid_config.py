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

from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg

class MITHumanoidCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 6000
        num_observations = 49+3*18  # 121
        num_actions = 18
        episode_length_s = 100  # episode length in seconds
        num_privileged_obs = num_observations

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, yaw_vel, heading (in heading mode yaw_vel is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0., 4.]  # min max [m/s]
            lin_vel_y = 0.  # max [m/s]
            yaw_vel = 0.  # max [rad/s]
            heading = 0.

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = "reset_to_range" # default setup chooses how the initial conditions are chosen.
                                # "reset_to_basic" = a single position with some randomized noise on top. 
                                # "reset_to_range" = a range of joint positions and velocities.
                                #  "reset_to_traj" = feed in a trajectory to sample from. 

        #default for normalization and basic initialization 
        # -0.27, 0.638, -0.35;
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw': 0.,
            'left_hip_abad': 0.,
            'left_hip_pitch': -0.4,
            'left_knee': 0.9,  # 0.785,
            'left_ankle': -0.45,
            'left_shoulder_pitch': 0.,
            'left_shoulder_abad': 0.,
            'left_shoulder_yaw': 0.,
            'left_elbow': 0.,
            'right_hip_yaw': 0.,
            'right_hip_abad': 0.,
            'right_hip_pitch': -0.4,
            'right_knee': 0.9,  # 0.785,
            'right_ankle': -0.45,
            'right_shoulder_pitch': 0.,
            'right_shoulder_abad': 0.,
            'right_shoulder_yaw': 0.,
            'right_elbow': 0.
        }

        #default COM for basic initialization 
        pos = [0.0, 0.0, 0.66]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # initialization for random range setup

        dof_pos_range = {'hip_yaw': [0., 0.],
                         'hip_abad': [0., 0.],
                         'hip_pitch': [-0.29, -0.25],
                         'knee': [0.67, 0.71],
                         'ankle': [-0.43, -0.39],
                         'shoulder_pitch': [0., 0.],
                         'shoulder_abad': [0., 0.],
                         'shoulder_yaw': [0., 0.],
                         'elbow': [0., 0.]
                         }
        
        dof_vel_range = {'hip_yaw': [-0., 0.1],
                         'hip_abad': [-0., 0.1],
                         'hip_pitch': [-0.1, -0.1],
                         'knee': [-0.05, 0.05],
                         'ankle': [-0.05, 0.05],
                         'shoulder_pitch': [0., 0.],
                         'shoulder_abad': [0., 0.],
                         'shoulder_yaw': [0., 0.],
                         'elbow': [0., 0.]
                         }

        root_pos_range = [[0., 0.],  # x
                          [0., 0.],  # y
                          [0.7, 0.72],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw

        root_vel_range = [[-0.1, 0.1],  # x
                          [-0.1, 0.1],  # y
                          [-0.1, 0.1],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 30.,
                     'hip_abad': 30.,
                     'hip_pitch': 30.,
                     'knee': 30.,
                     'ankle': 30.,
                     'shoulder_pitch': 40.,
                     'shoulder_abad': 40.,
                     'shoulder_yaw': 40.,
                     'elbow': 40.,
                    }  # [N*m/rad]
        damping = {'hip_yaw': 5.,
                   'hip_abad': 5.,
                   'hip_pitch': 5.,
                   'knee': 5.,
                   'ankle': 5.,
                   'shoulder_pitch': 5.,
                   'shoulder_abad': 5.,
                   'shoulder_yaw': 5.,
                   'elbow': 5.,
                    }  # [N*m*s/rad]     # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.
        # * exponential average decay for action scale
        exp_avg_decay = None  # set to None to disable
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # ! substeps?

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 7
        max_push_vel_xy = 0.1

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mit_humanoid/urdf/humanoid_R_sf.urdf'
        foot_name = 'foot'
        penalize_contacts_on = ['base', 'arm']
        terminate_after_contacts_on = ['base']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disagble, 0 to enable...bitwise filter
        collapse_fixed_joints = False
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        disable_gravity = False
        disable_actions = False
        disable_motors = False

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 1500.

        base_height_target = 0.65
        tracking_sigma = 0.5

    class scaling(LeggedRobotCfg.scaling):
        # * dimensionless time: sqrt(L/g) or sqrt(I/[mgL]), with I=I0+mL^2
        dimless_time = (0.7/9.81)**0.5
        virtual_leg = 0.65
        base_height = 1./virtual_leg
        base_lin_vel = 1./virtual_leg 
        base_ang_vel = 1./3.14*dimless_time
        dof_pos_obs = 1./3.14
        dof_vel = 0.05  # ought to be roughly max expected speed.
        height_measurements = 1./virtual_leg

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.25  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            base_height = 0.05
            dof_pos_obs = 0.0
            dof_vel = 0.0
            base_lin_vel = 0.1
            base_ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]
        class physx:
            max_depenetration_velocity = 50.0

class MITHumanoidRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

        actor_obs = ["base_height",
                     "base_lin_vel",
                     "base_ang_vel",
                     "projected_gravity",
                     "commands",
                     "dof_pos_obs",
                     "dof_vel"]

        critic_obs = actor_obs

        class reward:
            make_PBRS = []
            class weights:
                tracking_ang_vel = 0.5  
                tracking_lin_vel = 0.5  
                orientation = 1.5 
                torques = 5.e-6 
                min_base_height = 1.5  
                action_rate = 0.1
                action_rate2 = 0.001 
                lin_vel_z = 0. 
                ang_vel_xy = 0.0 
                dof_vel = 0.0 
                stand_still = 0. 
                dof_pos_limits = 0.0
                dof_near_home = 0.5  
            class termination_weight:
                termination = 15
    class algorithm(LeggedRobotRunnerCfg.algorithm):
         # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.999
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 1000
        run_name = 'Standing'
        experiment_name = 'Humanoid'
        save_interval = 50
    