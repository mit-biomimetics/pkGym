from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO
import torch

class CartpoleCfg(FixedRobotCfg):

    class env(FixedRobotCfg.env):
        num_envs = 1024
        num_actions = 1  # 1 for the cart force
        episode_length_s = 10
    class terrain(FixedRobotCfg.terrain):
        # curriculum = False
        # mesh_type = 'plane'
        # measure_heights = False
        pass

    class init_state(FixedRobotCfg.init_state):

        default_joint_angles = {"slider_to_cart": 0.,
                                "cart_to_pole": 0.}

        reset_mode = "reset_to_range" 
        # default setup chooses how the initial conditions are chosen. 
        # "reset_to_basic" = a single position
        # "reset_to_range" = uniformly random from a range defined below

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'slider_to_cart': [-2.5, 2.5],
                         'cart_to_pole': [-torch.pi, torch.pi]}
        dof_vel_range = {'slider_to_cart': [-0.1, 0.1],
                         'cart_to_pole': [-0.1, 0.1]}

    class control(FixedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'slider_to_cart': 10.}  # [N*m/rad]
        damping = {'slider_to_cart': 0.5}  # [N*m*s/rad]

        control_type = "T"

        # for each dof: 1 if actuated, 0 if passive
        # Empty implies no chance in the _compute_torques step
        actuated_joints_mask = [1,  # slider_to_cart
                                0]  # cart_to_pole


        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 4.0

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset(FixedRobotCfg.asset):
        # Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cartpole/urdf/cartpole.urdf"
        flip_visual_attachments = False

        # Toggles to keep
        disable_gravity = False
        disable_actions = False  # disable neural networks
        disable_motors = False  # all torques set to 0

    class reward_settings(FixedRobotCfg.reward_settings):
        pass

    class scaling(FixedRobotCfg.scaling):
        dof_pos = [1/3., 1/torch.pi]
        dof_vel = [1/20., 1/(4*torch.pi)]

    class noise(FixedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            noise = 0.1  # implement as needed, also in your robot class
            cart_pos = 0.001
            pole_pos = 0.001
            cart_vel = 0.010
            pole_vel = 0.010
            actuation = 0.00

    class sim:
        dt = 0.001 # 1/60 s
        substeps = 2
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 4  # 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.02  # 0.01  # [m]
            rest_offset = 0.001  # 0.0  # [m]
            bounce_threshold_velocity = 0.2  # 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 100.0  # 10.0
            max_gpu_contact_pairs = 1024*1024  # 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 2.0  # 5
            contact_collection = 0  # 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class CartpoleRunnerCfg(FixedRobotCfgPPO):
    # We need random experiments to run
    seed = -1
    runner_class_name = 'OnPolicyRunner'

    do_wandb = True
    class wandb:
        what_to_log = {}

    class policy(FixedRobotCfgPPO.policy):
        init_noise_std = 1.0
        num_layers = 2
        num_units = 64
        actor_hidden_dims = [num_units] * num_layers
        critic_hidden_dims = [num_units] * num_layers
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        actor_obs = ["dof_pos",
                     "dof_vel"
                     ]

        critic_obs = actor_obs

        class reward:
            make_PBRS = []

            class weights:
                pole_pos = 5
                pole_vel = 0.025
                cart_pos = 1
                torques = 0.1
                dof_vel = 0.0
                collision = 0.0
                upright_pole = 10
            class termination_weight:
                termination = 0.

    class algorithm(FixedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.999
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(FixedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'cartpole'

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
