
from gym.envs.mini_cheetah.mini_cheetah_config \
    import MiniCheetahCfg, MiniCheetahRunnerCfg

BASE_HEIGHT_REF = 0.35

class MiniCheetahRefCfg(MiniCheetahCfg):
    class env(MiniCheetahCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 15.

    class terrain(MiniCheetahCfg.terrain):
        pass

    class init_state(MiniCheetahCfg.init_state):

        ref_traj = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/trajectories/single_leg.csv"
        ref_type = "Pos"

    class control(MiniCheetahCfg.control):
        # PD Drive parameters:
        stiffness = {'haa': 20., 'hfe': 20., 'kfe': 20.}
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}
        gait_freq = 3.5 #
        dof_pos_decay = 0.15  # set to None to disable
        ctrl_frequency = 100
        desired_sim_frequency = 500

    class commands:
        resampling_time = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 4.0]  # min max [m/s]
            lin_vel_y = 1.  # max [m/s]
            yaw_vel = 1    # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 2
        max_push_vel_xy = 0.1

    class domain_rand(MiniCheetahCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.75, 1.05]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]

    class asset(MiniCheetahCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah_simple.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["shank"]
        terminate_after_contacts_on = ["base", "thigh"]
        collapse_fixed_joints = False
        fix_base_link = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False  # all torques set to 0

    class reward_settings(MiniCheetahCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.

        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.3

    class scaling(MiniCheetahCfg.scaling):
        pass

class MiniCheetahRefRunnerCfg(MiniCheetahRunnerCfg):
    seed = -1
    class policy( MiniCheetahRunnerCfg.policy ):
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

        actor_obs = ["base_height",
                     "base_lin_vel",
                     "base_ang_vel",
                     "projected_gravity",
                     "commands",
                     "dof_pos_obs",
                     "dof_vel",
                     "dof_pos_history",
                     "phase_obs"
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
                      "phase_obs"
                      ]

        actions = ["dof_pos_target"]

        class noise:
            # dof_pos_obs = 0.005  # can be made very low
            # dof_vel = 0.005
            # ang_vel = [0.1, 0.1, 0.1]  # 0.027, 0.14, 0.37
            projected_gravity = 0.0
        class reward:
            make_PBRS = []
            class weights:
                tracking_lin_vel = 3.0
                tracking_ang_vel = 1.0
                lin_vel_z = 0.1
                ang_vel_xy = 0.0
                orientation = 1.5
                torques = 5.e-7
                dof_vel = 0.
                min_base_height = 1.
                collision = 0.25
                action_rate = 0.01  # -0.01
                action_rate2 = 0.001  # -0.001
                stand_still = 0.5
                dof_pos_limits = 0.
                feet_contact_forces = 0.
                dof_near_home = 0.
                reference_traj = 0.5
                swing_grf = 0.75
                stance_grf = 1.5
            class termination_weight:
                termination = 15

    class algorithm( MiniCheetahRunnerCfg.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 6
        num_mini_batches = 6  # mini batch size = num_envs*nsteps/nminibatches
        learning_rate = 5.e-5
        schedule = 'adaptive'  # can be adaptive, fixed
        discount_horizon = 0.75  # [s]
        GAE_bootstrap_horizon = 0.2  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.
    class state_estimator:
        num_learning_epochs = 10
        num_mini_batches = 1  # mini batch size = num_envs*nsteps / nminibatches
        obs = ["base_ang_vel",
               "projected_gravity",
               "dof_pos",
               "dof_vel"]

        targets = ["base_height",
                   "base_lin_vel"]

        states_to_write_to = targets
        # * neural network params
        class neural_net:
            activation = "elu"
            hidden_dims = [256, 128, 64]  # None will default to 256, 128
            # dropouts: randomly zeros output of a node.
            # specify the probability of a dropout, 0 means no dropouts.
            # Done per layer, including initial layer (input-first, no last-output)
            # len(dropouts) == len(hidden_dims)
            dropouts = [0.1, 0.1, 0.1]

    class runner(MiniCheetahRunnerCfg.runner):
        run_name = ''
        experiment_name = 'mini_cheetah_ref'
        max_iterations = 1000  # number of policy updates
        SE_learner = 'modular_SE'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32
