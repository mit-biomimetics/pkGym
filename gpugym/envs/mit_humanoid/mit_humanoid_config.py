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
<<<<<<< HEAD
    class env(LeggedRobotCfg.env):
        num_envs = 10
        num_observations = 67+2  # 187  # ! why? should be 66...
=======
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 67+2  # 169
>>>>>>> dev
        num_actions = 18

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
            lin_vel_x = [0.0, 3.0] # min max [m/s]
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [-1.14, 1.14]

    class init_state(LeggedRobotCfg.init_state):
<<<<<<< HEAD
        default_setup = "Trajectory" #default setup chooses how the initial conditions are chosen. 
                                #"Basic" = a single position with some randomized noise on top. "Range" = a range of joint positions and velocities. "Trajectory" = feed in a trajectory to sample from. 

        #default for normalization and basic initialization 
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw': 0.,
            'left_hip_abad': 0.,
            'left_hip_pitch': 0.,
            'left_knee': 0.,  # 0.785,
            'left_ankle': 0.,
            'left_shoulder_pitch': 0.,
            'left_shoulder_abad': 0.,
            'left_shoulder_yaw': 0.,
            'left_elbow': 0.,
            'right_hip_yaw': 0.,
            'right_hip_abad': 0.,
            'right_hip_pitch': 0.,
            'right_knee': 0.,  # 0.785,
            'right_ankle': 0.,
            'right_shoulder_pitch': 0.,
            'right_shoulder_abad': 0.,
            'right_shoulder_yaw': 0.,
            'right_elbow': 0.
=======
        pos = [0.0, 0.0, 0.71]  # x,y,z [m]
        #rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]

        lin_vel = [0., 0.0, 0.0]

        default_joint_angles = {
            'left_hip_yaw': 0.,  # * -6.28 | 6.28
            'left_hip_abad': 0.,  # * -6.28 | 6.28
            'left_hip_pitch': 0.,  # * -6.28 | 6.28
            'left_knee':  0.0,  # * 0. | 3.
            'left_ankle': 0.,  # * -1.5 | 1.5
            'left_shoulder_pitch': 0.,  # * -6.28 | 6.28
            'left_shoulder_abad': 0.,  # * -6.28 | 6.28
            'left_shoulder_yaw': 0.,  # * -6.28 | 6.28
            'left_elbow': 0.,  # * -6.28 | 6.28

            'right_hip_yaw': 0.,  # * -6.28 | 6.28
            'right_hip_abad': 0.,  # * -6.28 | 6.28
            'right_hip_pitch': 0.,  # * -6.28 | 6.28
            'right_knee': 0.0,  # * 0. | 3.
            'right_ankle': 0.,  # * -1.5 | 1.5
            'right_shoulder_pitch': 0.0,  # * -6.28 | 6.28
            'right_shoulder_abad': 0.,  # * -6.28 | 6.28
            'right_shoulder_yaw': 0.,  # * -6.28 | 6.28
            'right_elbow': 0.  # * -6.28 | 6.28
>>>>>>> dev
        }

        #default COM for basic initialization 
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        #initialization for random range setup
        dof_pos_high = [0.,0.,0.,0.75, 0., 0., 0.,0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0. ] #DOF dimensions
        dof_pos_low =  [0.,0.,0.,0.,   0., 0., 0.,0.,0.,   0., 0., 0., 0.,0., 0., 0., 0., 0. ]
        dof_vel_high = [0.,0.,0.,0.0, 0., 0., 0.,0.,0.0, 0., 0., 0., 0.,0., 0., 0., 0., 0. ]
        dof_vel_low = [0.,0.,0.,0.0, 0., 0., 0.,0.,0.0, 0., 0., 0., 0.,0., 0., 0., 0., 0. ]

        com_pos_high = [0.,0.,1., 0., 0.5, 0.] #COM dimensions, in euler angles because randomizing in quat is confusing
        com_pos_low = [0.,0.,1., 0., -0.5, 0.] #x, y ,z, roll, pitch, yaw
        com_vel_high = [0.,0.,0., 0., 0.0, 0.]
        com_vel_low = [0.,0.,0.,0.,0.,0.]

        #initialization for traj sampling setup
        ref_traj = "../../resources/robots/mit_humanoid/trajectories/humanoid3d_roll.csv"
        ref_type = "Pos" #Pos, PosVel

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 40.,
                     'hip_abad': 40.,
                     'hip_pitch': 60.,
                     'knee': 60.,
                     'ankle': 40.,
                     'shoulder_pitch': 20.,
                     'shoulder_abad': 20.,
                     'shoulder_yaw': 20.,
                     'elbow': 20.,
                    }  # [N*m/rad]
        damping = {'hip_yaw': 1,
                   'hip_abad': 1,
                   'hip_pitch': 1,
                   'knee': 1,
                   'ankle': 1,
                   'shoulder_pitch': 1,
                   'shoulder_abad': 1,
                   'shoulder_yaw': 1,
                   'elbow': 1,
                    }  # [N*m*s/rad]     # [N*m*s/rad]
        # stiffness = {}
        # damping = {}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1  # ! substeps?

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
        penalize_contacts_on = ['base']
        terminate_after_contacts_on = ['base']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        disable_gravity = True


    class rewards(LeggedRobotCfg.rewards):
        # soft_dof_pos_limit = 0.95
        # soft_dof_vel_limit = 0.9
        # soft_torque_limit = 0.9
        max_contact_force = 600.
        
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        base_height_target = 0.55
        tracking_sigma = 0.25

        #reference traj tracking
        base_pos_tracking = 0.0
        base_vel_tracking = 0.1
        dof_pos_tracking = 0.1
        dof_vel_tracking = 0.1

        class scales(LeggedRobotCfg.rewards.scales):
            reference_traj = 1.0
            termination = -10.
            tracking_lin_vel = 5.0
            tracking_ang_vel = 0.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.0
            orientation = -1.25
            torques = -5.e-8
            dof_vel = 0.0
            dof_acc = 0.
            base_height = 1.5
            feet_air_time = 0.0  # rewards keeping feet in the air
            collision = -0.
            feet_stumble = -0.
            action_rate = 0.  # -0.01
            stand_still = -0.
            dof_pos_limits = -1.
            no_fly = 0.0
            feet_contact_forces = -0.

    class normalization(LeggedRobotCfg.normalization):
            class obs_scales(LeggedRobotCfg.normalization.obs_scales):
                # * helper fcts
                # * dimensionless time: sqrt(L/g) or sqrt(I/[mgL]), with I=I0+mL^2
                dimless_time = (0.7/9.81)**0.5
                v_leg = 0.72
                # lin_vel = 1/v_leg*dimless_time
                base_z = 1./0.72
                lin_vel =  1./v_leg  # virtual leg lengths per second
                # ang_vel = 0.25
                ang_vel = 1./3.14*dimless_time
                dof_pos = 1./3.14
                dof_vel = 0.05  # ought to be roughly max expected speed.

                height_measurements = 1./0.72
            # clip_observations = 100.
            clip_actions = 1000.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 0.1  # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025
        substeps = 1
        gravity = [0., 0., 0.]


class MITHumanoidCfgPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'MIT_Humanoid'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 20
        max_iterations = 15000

        save_interval = 100
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
