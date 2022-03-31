from gpugym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO
import torch

class obs_augmentations:

    SIN = torch.sin
    COS = torch.cos
    SINSQR = lambda x: torch.square(torch.sin(x))
    COSSQR = lambda x: torch.square(torch.cos(x))
    SQR = torch.square
    CUB = lambda x: x**3

    do_dynamics_augmentation = False
    do_controls_augmentation = False

    dynamics_augmentations = {(SIN, "pole angle", 1.0),
                              (COS, "pole angle", 1.0),
                              (SQR, "pole velocity", 0.01)}

    controls_augmentations = {(COS, "pole angle", 1.0),
                              (COSSQR, "pole angle", 1.0),
                              (CUB, "pole velocity", 0.005)}

    final_augmentations = set()
    final_augmentations |= dynamics_augmentations if do_dynamics_augmentation else set()
    final_augmentations |= controls_augmentations if do_controls_augmentation else set()
    augmentations_list = list(final_augmentations) # todone: Ensure this always happens in the same order - done in Pyconsole

class CartpoleCfg(FixedRobotCfg):

    class env(FixedRobotCfg.env):
        num_actions = 1  # 1 for the cart force
        max_effort = 10.0
        reset_dist = 3.0
        augmentations = obs_augmentations.augmentations_list
        num_observations = 4 + len(augmentations)


    class terrain(FixedRobotCfg.terrain):
        # curriculum = False
        # mesh_type = 'plane'
        # measure_heights = False
        pass

    class init_state(FixedRobotCfg.init_state):
        """
        Initial States of the Cartpole where the middle is cart 0 and up is pole 0
        """
        default_setup = "Basic"  # default setup chooses how the initial conditions are chosen.
        # "Basic" = a single position with some randomized noise on top.
        # "Range" = a range of joint positions and velocities.
        #  "Trajectory" = feed in a trajectory to sample from.

        dof_pos_high = [0., 0.]  # DOF dimensions
        dof_pos_low = [0., 0.]
        dof_vel_high = [0., 0.]
        dof_vel_low = [0., 0.]

        default_joint_angles = {  # target angles when action = 0.0
            "cart": 0.,
            "pole": 0.}

    class control(FixedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'cart': 10.}  # [N*m/rad]
        damping = {'cart': 0.5}  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5


    class domain_rand(FixedRobotCfg.domain_rand):
        pass

    class asset(FixedRobotCfg.asset):
        # Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cartpole/urdf/cartpole.urdf"
        flip_visual_attachments = False

        # Toggles to keep
        disable_gravity = False
        disable_actions = False  # disable neural networks
        disable_motors = False  # all torques set to 0

    class rewards(FixedRobotCfg.rewards):
        hierarchy = {'pole position': {'cart position'}, 'pole velocity': None, 'actuation': None}

        # Space, sub_space
        pole_position = (torch.pi, 0.1*torch.pi)
        pole_velocity = (14.0, None)
        cart_position = (3.0, None)
        cart_velocity = None  # (None, None, None)
        actuation = (10.0, None)  # Assuming max effort is 10.0
        termination = (3.0, None)

        class scales:
            termination = -20.0
            pole_position = 1.0
            pole_velocity = 0.01
            cart_position = 1.0
            actuation = 1e-5

            # Unused rewards
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            collision = 0.0
            action_rate = 0.0
            dof_pos_limits = 0.0

    class normalization(FixedRobotCfg.normalization):
        clip_observations = 5.0
        clip_actions = 1.0

        class obs_scales:
            # Used to be...
            dof_pos = 1/3
            dof_vel = 0.1
            # cart_pos = 1.0
            # pole_pos = 1.0
            # cart_vel = 1.0
            # pole_vel = 1.0
            # sin_pole_pos = 1.0
            # cos_pole_pos = 1.0
            # sqr_pole_vel = 0.01
            # cos_sqr_pole_pos = 1.0
            # cub_pole_vel = 0.005


    class noise(FixedRobotCfg.noise):
        # add_noise = False
        # noise_level = 0.1  # scales other values
        #
        # class noise_scales(FixedRobotCfg.noise.noise_scales):
        #     dof_pos = 0.01
        #     dof_vel = 0.01
        #     lin_vel = 0.1
        #     ang_vel = 0.2
        #     gravity = 0.05
        #     height_measurements = 0.1

        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            noise = 0.1  # implement as needed, also in your robot class

    class sim:
        # These values now match the original cartpole example experiment when using RL_games
        dt = 0.0166 # 1/60 s
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


    # TODO COME BACK TO THIS AND MAKE SURE VALUES ARE THE SAME AS BEFORE IF ITS NOT WORKING
    class policy(FixedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [64, 64, 64]
        critic_hidden_dims = [64, 64, 64]
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
        gamma = 0.99
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
