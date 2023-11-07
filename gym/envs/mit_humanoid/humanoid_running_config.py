"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)


class HumanoidRunningCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actuators = 18
        episode_length_s = 5  # 100

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = "plane"
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = "reset_to_range"
        pos = [0.0, 0.0, 0.75]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = {
            "hip_yaw": 0.0,
            "hip_abad": 0.0,
            "hip_pitch": -0.2,
            "knee": 0.25,  # 0.6
            "ankle": 0.0,
            "shoulder_pitch": 0.0,
            "shoulder_abad": 0.0,
            "shoulder_yaw": 0.0,
            "elbow": -1.25,
        }

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.7, 0.72],  # z
            [-torch.pi / 10, torch.pi / 10],  # roll
            [-torch.pi / 10, torch.pi / 10],  # pitch
            [-torch.pi / 10, torch.pi / 10],  # yaw
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-0.5, 2.5],  # x
            [-0.5, 0.5],  # y
            [-0.5, 0.5],  # z
            [-0.5, 0.5],  # roll
            [-0.5, 0.5],  # pitch
            [-0.5, 0.5],  # yaw
        ]

        dof_pos_range = {
            "hip_yaw": [-0.1, 0.1],
            "hip_abad": [-0.2, 0.2],
            "hip_pitch": [-0.2, 0.2],
            "knee": [0.6, 0.7],
            "ankle": [-0.3, 0.3],
            "shoulder_pitch": [0.0, 0.0],
            "shoulder_abad": [0.0, 0.0],
            "shoulder_yaw": [0.0, 0.0],
            "elbow": [0.0, 0.0],
        }

        dof_vel_range = {
            "hip_yaw": [-0.1, 0.1],
            "hip_abad": [-0.1, 0.1],
            "hip_pitch": [-0.1, 0.1],
            "knee": [-0.1, 0.1],
            "ankle": [-0.1, 0.1],
            "shoulder_pitch": [0.0, 0.0],
            "shoulder_abad": [0.0, 0.0],
            "shoulder_yaw": [0.0, 0.0],
            "elbow": [0.0, 0.0],
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        stiffness = {
            "hip_yaw": 30.0,
            "hip_abad": 30.0,
            "hip_pitch": 30.0,
            "knee": 30.0,
            "ankle": 30.0,
            "shoulder_pitch": 30.0,
            "shoulder_abad": 30.0,
            "shoulder_yaw": 30.0,
            "elbow": 50.0,
        }  # [N*m/rad]
        damping = {
            "hip_yaw": 5.0,
            "hip_abad": 5.0,
            "hip_pitch": 5.0,
            "knee": 5.0,
            "ankle": 5.0,
            "shoulder_pitch": 5.0,
            "shoulder_abad": 5.0,
            "shoulder_yaw": 5.0,
            "elbow": 1.0,
        }  # [N*m*s/rad]
        ctrl_frequency = 100
        desired_sim_frequency = 800

    class commands(LeggedRobotCfg.commands):
        resampling_time = 5.0

        class ranges:
            # TRAINING COMMAND RANGES #
            lin_vel_x = [0, 4.5]  # min max [m/s]
            lin_vel_y = 0.75  # min max [m/s]
            yaw_vel = 4.0  # min max [rad/s]
            # # PLAY COMMAND RANGES #
            # lin_vel_x = [3., 3.]    # min max [m/s]
            # lin_vel_y = 0.    # min max [m/s]
            # yaw_vel = 0.      # min max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 2.5
        max_push_vel_xy = 0.5
        push_box_dims = [0.1, 0.2, 0.3]  # x,y,z [m]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]

        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]

    class asset(LeggedRobotCfg.asset):
        # file = ('{LEGGED_GYM_ROOT_DIR}/resources/robots/rom/urdf/'
        #         +'humanoid_fixed_arms_full.urdf')
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mit_humanoid/urdf/humanoid_F_sf.urdf"
        )
        keypoints = ["base"]
        end_effectors = ["left_foot", "right_foot"]
        # end_effector_names = ['left_toe', 'left_heel',
        #                       'right_toe', 'right_heel']
        foot_name = "foot"
        terminate_after_contacts_on = [
            "base",
            "left_upper_leg",
            "left_lower_leg",
            "right_upper_leg",
            "right_lower_leg",
            "left_upper_arm",
            "right_upper_arm",
            "left_lower_arm",
            "right_lower_arm",
            "left_hand",
            "right_hand",
        ]

        fix_base_link = False
        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 1500.0

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False  # ! zap?
        base_height_target = 0.62
        tracking_sigma = 0.5

    class scaling(LeggedRobotCfg.scaling):
        base_height = 0.6565
        base_lin_vel = 1.0
        base_ang_vel = 1.0
        dof_pos = 2 * [0.5, 1, 3, 2, 2] + 2 * [2, 1, 0.5, 2.0]
        dof_vel = 1.0
        dof_pos_target = dof_pos
        clip_actions = 1000.0


class HumanoidRunningRunnerCfg(LeggedRobotRunnerCfg):
    do_wandb = True
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = "elu"
        normalize_obs = True  # True, False

        actor_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "phase_sin",
            "phase_cos",
            "dof_pos_legs",
            "dof_vel_legs",
            "in_contact",
        ]

        critic_obs = actor_obs

        actions = ["dof_pos_target_legs"]

        add_noise = True
        noise_level = 1.0  # scales other values

        class noise:
            base_height = 0.05
            base_lin_vel = 0.1
            base_ang_vel = 0.05
            projected_gravity = 0.05
            dof_pos = 0.005
            dof_vel = 0.01
            in_contact = 0.1

        class reward:
            class weights:
                # * Behavioral rewards * #
                action_rate = 1.0e-3
                action_rate2 = 1.0e-4
                tracking_lin_vel = 5.0
                tracking_ang_vel = 5.0
                torques = 1e-4
                # dof_pos_limits = 10
                torque_limits = 1e-2
                dof_vel = 1e-4

                # * Shaping rewards * #
                base_height = 0.1
                orientation = 1.0
                hip_yaw_zero = 2.0
                hip_abad_symmetry = 0.2

            class termination_weight:
                termination = 1

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.0e-5
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.999
        lam = 0.99
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 1000
        run_name = "ICRA2023"
        experiment_name = "HumanoidLocomotion"
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False
