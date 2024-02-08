from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 0.3


class MiniCheetahCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 12
        episode_length_s = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            "haa": 0.0,
            "hfe": -0.785398,
            "kfe": 1.596976,
        }

        # * reset setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup
        dof_pos_range = {
            "haa": [-0.01, 0.01],
            "hfe": [-0.785398, -0.785398],
            "kfe": [1.596976, 1.596976],
        }
        dof_vel_range = {"haa": [0.0, 0.0], "hfe": [0.0, 0.0], "kfe": [0.0, 0.0]}
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.35, 0.35],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [-0.5, 2.0],  # x
            [0.0, 0.0],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(LeggedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {"haa": 20.0, "hfe": 20.0, "kfe": 20.0}
        damping = {"haa": 0.5, "hfe": 0.5, "kfe": 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 500

    class commands:
        # * time before command are changed[s]
        resampling_time = 3.0

        class ranges:
            lin_vel_x = [-2.0, 3.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 3  # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 15
        max_push_vel_xy = 0.05
        push_box_dims = [0.3, 0.1, 0.1]  # x,y,z [m]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.0]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/urdf/mini_cheetah_simple.urdf"
        )
        foot_name = "foot"
        penalize_contacts_on = ["shank"]
        terminate_after_contacts_on = ["base"]
        end_effector_names = ["foot"]
        collapse_fixed_joints = False
        self_collisions = 1
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False
        joint_damping = 0.1
        rotor_inertia = [0.002268, 0.002268, 0.005484] * 4

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 0.3
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = 4 * [2.0, 2.0, 4.0]
        base_height = 0.3
        dof_pos = 4 * [0.2, 0.3, 0.3]
        dof_pos_obs = dof_pos
        dof_pos_target = 4 * [0.2, 0.3, 0.3]
        tau_ff = 4 * [18, 18, 28]
        commands = [3, 1, 3]


class MiniCheetahRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [128, 64]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"

        actor_obs = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        critic_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]

        actions = ["dof_pos_target"]
        add_noise = True

        class noise:
            scale = 1.0
            dof_pos_obs = 0.01
            base_ang_vel = 0.01
            dof_pos = 0.005
            dof_vel = 0.005
            lin_vel = 0.05
            ang_vel = [0.3, 0.15, 0.4]
            gravity_vec = 0.1

        class reward(LeggedRobotRunnerCfg.policy.reward):
            class weights(LeggedRobotRunnerCfg.policy.reward.weights):
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.01
                orientation = 1.0
                torques = 5.0e-7
                dof_vel = 0.0
                min_base_height = 1.5
                action_rate = 0.01
                action_rate2 = 0.001
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0

            class termination_weight:
                termination = 0.01

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # * training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.02
        num_learning_epochs = 4
        # * mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 8
        learning_rate = 1.0e-5
        schedule = "adaptive"  # can be adaptive or fixed
        discount_horizon = 1.0  # [s]
        GAE_bootstrap_horizon = 2.0  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "mini_cheetah"
        max_iterations = 500
        algorithm_class_name = "PPO"
        num_steps_per_env = 32
