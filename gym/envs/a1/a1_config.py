from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 0.33


class A1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 12
        episode_length_s = 5.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

        reset_mode = "reset_to_basic"
        # reset setup chooses how the initial conditions are chosen.
        # "reset_to_basic" = a single position
        # "reset_to_range" = uniformly random from a range defined below

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup

        dof_pos_range = {
            "haa": [-0.05, 0.05],
            "hfe": [-0.85, -0.6],
            "kfe": [-1.45, 1.72],
        }

        dof_vel_range = {
            "haa": [0.0, 0.0],
            "hfe": [0.0, 0.0],
            "kfe": [0.0, 0.0],
        }

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.37, 0.4],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],
        ]  # yaw
        root_vel_range = [
            [-0.05, 0.05],  # x
            [0.0, 0.0],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],
        ]  # yaw

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {"joint": 20.0}  # [N*m/rad]
        damping = {"joint": 0.5}  # [N*m*s/rad]

        ctrl_frequency = 100
        desired_sim_frequency = 500

    class commands:
        resampling_time = 10.0  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 1  # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 1
        max_push_vel_xy = 0.5

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base", "thigh", "hip"]
        # merge bodies connected by fixed joints.
        collapse_fixed_joints = False
        # added blindly from the AnymalCFlatCFG.
        # 1 to disable, 0 to enable...bitwise filter
        self_collisions = 1
        flip_visual_attachments = True
        disable_gravity = False
        disable_motors = False  # all torques set to 0

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
        tau_ff = 4 * [18, 18, 28]  # hip-abad, hip-pitch, knee


class A1RunnerCfg(LeggedRobotRunnerCfg):
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # activation can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        actor_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_history",
            "commands",
        ]

        critic_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_history",
            "commands",
        ]

        actions = ["dof_pos_target"]

        class noise:
            dof_pos_obs = 0.005  # can be made very low
            dof_vel = 0.005
            base_ang_vel = 0.05
            projected_gravity = 0.02

        class reward(LeggedRobotRunnerCfg.policy.reward):
            class weights(LeggedRobotRunnerCfg.policy.reward.weights):
                tracking_lin_vel = 1.0
                tracking_ang_vel = 1.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.0
                orientation = 1.0
                torques = 5.0e-7
                dof_vel = 0.0
                base_height = 1.0
                action_rate = 0.001
                action_rate2 = 0.0001
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 1.0

            class termination_weight:
                termination = 1.0

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        # mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 4
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        discount_horizon = 0.5  # [s]
        GAE_bootstrap_horizon = 0.2  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "a1"
        max_iterations = 500  # number of policy updates
        algorithm_class_name = "PPO"
        # per iteration
        # (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
        num_steps_per_env = 24
