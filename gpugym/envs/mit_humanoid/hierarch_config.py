
# from .base_config import BaseConfig
from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from gpugym.envs import MITHumanoidCfg, MITHumanoidCfgPPO

class HierarchCfg(MITHumanoidCfg):
    class env:
        num_envs = 4096
        num_observations = 50
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 500
        # env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        num_privileged_obs = None

    class control:
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class domain_rand:
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -1.

            reference_traj = 0.0 

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.  # ! may want to turn this off
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

        #ref traj tracking
        base_pos_tracking = 0.0
        base_vel_tracking = 0.0
        dof_pos_tracking = 0.0
        dof_vel_tracking = 0.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]


    class sim(MITHumanoidCfg.sim):
        gravity = [0., 0., -9.81]

class HierarchCfgPPO(LeggedRobotCfgPPO):
    seed = 2
    runner_class_name = 'OnPolicyRunnerHierarch'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 10 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
