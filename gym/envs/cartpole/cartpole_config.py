import torch

from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO


class CartpoleCfg(FixedRobotCfg):

    class env(FixedRobotCfg.env):
        num_envs = 1024
        num_actuators = 1  # 1 for the cart force
        episode_length_s = 10

    class terrain(FixedRobotCfg.terrain):
        pass

    class init_state(FixedRobotCfg.init_state):

        default_joint_angles = {"slider_to_cart": 0.,
                                "cart_to_pole": 0.}

        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * initial conditions for reset_to_range
        dof_pos_range = {'slider_to_cart': [-2.5, 2.5],
                         'cart_to_pole': [-torch.pi, torch.pi]}
        dof_vel_range = {'slider_to_cart': [-0.1, 0.1],
                         'cart_to_pole': [-0.1, 0.1]}

    class control(FixedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {'slider_to_cart': 10.}  # [N*m/rad]
        damping = {'slider_to_cart': 0.5}  # [N*m*s/rad]

        actuated_joints_mask = [1,  # slider_to_cart
                                0]  # cart_to_pole

        ctrl_frequency = 100
        desired_sim_frequency = 200

    class asset(FixedRobotCfg.asset):
        # * Things that differ
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "cartpole/urdf/cartpole.urdf")
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False  # all torques set to 0

    class reward_settings(FixedRobotCfg.reward_settings):
        pass

    class scaling(FixedRobotCfg.scaling):
        dof_pos = [1/3., 1/torch.pi]
        dof_vel = [1/20., 1/(4*torch.pi)]

        # * Action scales
        tau_ff = 10
        dof_pos_target = 4


class CartpoleRunnerCfg(FixedRobotCfgPPO):
    # We need random experiments to run
    seed = -1
    runner_class_name = 'OnPolicyRunner'

    class policy(FixedRobotCfgPPO.policy):
        init_noise_std = 1.0
        num_layers = 2
        num_units = 64
        actor_hidden_dims = [num_units] * num_layers
        critic_hidden_dims = [num_units] * num_layers
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

        actor_obs = [
            "dof_pos",
            "dof_vel"]
        critic_obs = actor_obs

        actions = ["tau_ff"]

        class noise:
            noise = 0.1  # implement as needed, also in your robot class
            cart_pos = 0.001
            pole_pos = 0.001
            cart_vel = 0.010
            pole_vel = 0.010
            actuation = 0.00

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
        # * mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'  # could be adaptive, fixed
        discount_horizon = 1.  # [s]
        GAE_bootstrap_horizon = 0.2  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(FixedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 500  # number of policy updates

        # * logging
        # * check for potential saves every this many iterations
        save_interval = 50
        run_name = ''
        experiment_name = 'cartpole'

        # * load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
