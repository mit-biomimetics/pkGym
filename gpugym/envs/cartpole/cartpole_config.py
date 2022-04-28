from gpugym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO
import torch

class obs_augmentations:

    SIN = torch.sin
    COS = torch.cos
    SINSQR = lambda x: torch.square(torch.sin(x))
    COSSQR = lambda x: torch.square(torch.cos(x))
    SQR = torch.square
    CUB = lambda x: x**3

    do_dynamics_augmentation = True
    do_controls_augmentation = True
    # do_dynamics_augmentation = False
    # do_controls_augmentation = False

    dynamics_augmentations = [(SIN, "pole_pos", 1.0),
                              (COS, "pole_pos", 1.0),
                              (SQR, "pole_vel", 0.01)]

    controls_augmentations = [(COS,    "pole_pos", 1.0),
                              (COSSQR, "pole_pos", 1.0),
                              (CUB,    "pole_vel", 0.005)]

    combined_augmentations = [(SIN,    "pole_pos", 1.0),
                              (COS,    "pole_pos", 1.0),
                              (COSSQR, "pole_pos", 1.0),
                              (SQR,    "pole_vel", 0.01),
                              (CUB,    "pole_vel", 0.005)]

    if do_dynamics_augmentation and do_controls_augmentation:
        augmentations_list = combined_augmentations
    elif do_dynamics_augmentation:
        augmentations_list = dynamics_augmentations
    elif do_controls_augmentation:
        augmentations_list = controls_augmentations
    else:
        augmentations_list = []

class underactuation:
    max_effort = 4.0

class CartpoleCfg(FixedRobotCfg):

    class env(FixedRobotCfg.env):
        num_envs = 4096 # 1096
        num_actions = 1  # 1 for the cart force
        max_effort = underactuation.max_effort
        reset_dist = 3.0

        do_dynamics_augmentation = obs_augmentations.do_dynamics_augmentation
        do_controls_augmentation = obs_augmentations.do_controls_augmentation
        augmentations = obs_augmentations.augmentations_list
        num_observations = 5 + len(augmentations)


    class terrain(FixedRobotCfg.terrain):
        # curriculum = False
        # mesh_type = 'plane'
        # measure_heights = False
        pass

    class init_state(FixedRobotCfg.init_state):
        """
        Initial States of the Cartpole where the middle is cart 0 and up is pole 0
        """
        reset_mode = "reset_to_range" 
        # default setup chooses how the initial conditions are chosen. 
        # "reset_to_basic" = a single position
        # "reset_to_range" = uniformly random from a range defined below
        # "reset_to_storage" = reset from a storage of initial conditions

        dof_pos_high = [2.5, torch.pi]  # DOF dimensionspp
        dof_pos_low = [-2.5, -torch.pi]
        dof_vel_high = [0.1, 0.1]
        dof_vel_low = [-0.1, -0.1]

        default_joint_angles = {  # target angles when action = 0.0
            "slider_to_cart": 0.,
            "cart_to_pole": 0.}

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
        action_scale = underactuation.max_effort

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1


    class domain_rand(FixedRobotCfg.domain_rand):
        pass


    class asset(FixedRobotCfg.asset):
        # Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cartpole/urdf/cartpole.urdf"
        flip_visual_attachments = False
        fix_base_link = True

        # Toggles to keep
        disable_gravity = False
        disable_actions = False  # disable neural networks
        disable_motors = False  # all torques set to 0

    class rewards(FixedRobotCfg.rewards):
        hierarchy = {'pole position': {'cart position'}, 'pole velocity': None, 'actuation': None}

        class spaces:
            # Space, sub_space
            pole_pos = torch.pi
            pole_vel = 14.0
            cart_pos = 3.0
            cart_vel = None  # (None, None, None)
            actuation = underactuation.max_effort
            termination = 3.0

        class sub_spaces:
            pole_pos = 0.5 * torch.pi
            pole_vel = None
            cart_pos = None
            cart_vel = None  # (None, None, None)
            actuation = None  # Assuming max effort is 10.0
            termination = None

        class scales:
            termination = -20.0
            pole_pos = 5
            pole_vel = 0.01
            cart_pos = 5
            actuation = 1e-5

            # Unused rewards
            torques = 0.0
            dof_vel = 0.0
            collision = 0.0
            action_rate = 0.0
            action_rate2 = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0

    class success_metrics:
        class thresholds:
            pole_pos = 0.2 * torch.pi  # [rad] - pole should be centered about the top of its range
            pole_vel = 0.3 * torch.pi  # [rad/s] - the pole should not be moving
            cart_pos = 0.3  # [m] - the cart should be centered in its range

    class normalization(FixedRobotCfg.normalization):
        clip_observations = 5.0
        clip_actions = 1.0

        class obs_scales:
            # Used to be...
            cart_pos = 1/torch.pi
            pole_pos = 1/3.0
            cart_vel = 1/20.0
            pole_vel = 1/(4*torch.pi)
            # sin_pole_pos = 1.0
            # cos_pole_pos = 1.0
            # sqr_pole_vel = 0.01
            # cos_sqr_pole_pos = 1.0
            # cub_pole_vel = 0.005


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
        # These values now match the original cartpole example experiment when using RL_games
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


class CartpoleCfgPPO(FixedRobotCfgPPO):
    # We need random experiments to run
    seed = -1
    runner_class_name = 'OnPolicyRunner'

    do_wandb = True
    class wandb:
        what_to_log = {}
        what_to_log['num_layers'] = ['train_cfg', 'policy', 'num_layers']
        what_to_log['num_units'] = ['train_cfg', 'policy', 'num_layers']

        what_to_log['activation_sub_space'] = ['env_cfg', 'rewards', 'sub_spaces', 'pole_pos']
        what_to_log['pole_pos_rew'] = ['env_cfg', 'rewards', 'scales', 'pole_pos']

        what_to_log['cart_pos_rew'] = ['env_cfg', 'rewards', 'scales', 'cart_pos']
        what_to_log['do_dynamics_augmentation'] = ['env_cfg', 'env', 'do_dynamics_augmentation']
        what_to_log['do_controls_augmentation'] = ['env_cfg', 'env', 'do_controls_augmentation']

    # TODO COME BACK TO THIS AND MAKE SURE VALUES ARE THE SAME AS BEFORE IF ITS NOT WORKING
    class policy(FixedRobotCfgPPO.policy):
        init_noise_std = 1.0
        num_layers = 1
        num_units = 10
        actor_hidden_dims = [num_units] * num_layers
        critic_hidden_dims = [num_units] * num_layers
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

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
