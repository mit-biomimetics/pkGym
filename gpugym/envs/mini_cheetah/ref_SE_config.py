
from gpugym.envs.mini_cheetah.mini_cheetah_config import MiniCheetahCfg, MiniCheetahCfgPPO

BASE_HEIGHT_REF = 0.32

class SERefCfg(MiniCheetahCfg):
    class env(MiniCheetahCfg.env):
        num_envs = 4096
        num_actions = 12
        num_observations = 71
        num_privileged_obs = 71  # same but without noise
        episode_length_s = 15.

        # * if learn state-estimator is set to True, also set the config
        #  below for `state_estimator_nn` and `state_estimator`
        learn_SE = True
        num_se_obs = 30
    class terrain(MiniCheetahCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
    class init_state(MiniCheetahCfg.init_state):
        """
        Initial States of the Mini Cheetah
        From Robot-Software/systems/quadruped/state_machine/FSM_State_RecoveryStand.cpp, line 38
        Ab/ad: 0˚, hip: -45˚, knee: 91.5˚
        Default pose is around 0.27
        """

        default_joint_angles = {
            "lf_haa": 0.0,
            "lh_haa": 0.0,
            "rf_haa": 0.0,
            "rh_haa": 0.0,

            "lf_hfe": -0.785398,
            "lh_hfe": -0.785398,
            "rf_hfe": -0.785398,
            "rh_hfe": -0.785398,

            "lf_kfe": 1.596976,
            "lh_kfe": 1.596976,
            "rf_kfe": 1.596976,
            "rh_kfe": 1.596976,
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

        dof_pos_range = {'haa': [-0.05, 0.05],
                        'hfe': [-0.85, -0.75],
                        'kfe': [-1.55, 1.65]}

        dof_vel_range = {'haa': [-0.1, 0.1],
                        'hfe': [-0.1, 0.1],
                        'kfe': [-0.1, 0.1]}

        root_pos_range = [[0., 0.],  # x
                          [0., 0.],  # y
                          [0.33, 0.35],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw
        root_vel_range = [[-0.5, 1.0],  # x
                          [-0.01, 0.01],  # y
                          [-0.2, 0.2],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw

        ref_traj = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/trajectories/single_leg.csv"
        ref_type = "Pos"

    class control(MiniCheetahCfg.control):
        # PD Drive parameters:
        stiffness = {'haa': 20., 'hfe': 20., 'kfe': 20.}
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}
        gait_freq = 4. #
        # Control type: "P" (position + damping) or "Td" (torque + damping)
        control_type = "P"
        action_scale = 0.75
        exp_avg_decay = 0.15  # set to None to disable
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class domain_rand(MiniCheetahCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.75, 1.05]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        # on ground planes the friction combination mode is averaging,
        # i.e total friction = (foot_friction + 1.)/2.
        friction_range = [0., 1.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.2

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
        disable_actions = False  # neural networks output set to 0
        disable_motors = False  # all torques set to 0

    class rewards(MiniCheetahCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.

        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.3
        make_PBRS = ["base_height",
                     "reference_traj",
                     "rew_orientation"
                     ]
        class weights(MiniCheetahCfg.rewards.weights):
            termination = -15.
            tracking_lin_vel = 4.0
            tracking_ang_vel = 1.0
            lin_vel_z = 0.6
            ang_vel_xy = 0.0
            orientation = 1.75
            torques = -5.e-7
            dof_vel = 0.
            base_height = 1.5
            feet_air_time = 0.  # rewards keeping feet in the air
            collision = -0.25
            action_rate = -0.01  # -0.01
            action_rate2 = -0.001  # -0.001
            stand_still = 0.5
            dof_pos_limits = 0.
            feet_contact_forces = 0.
            dof_near_home = 0.
            reference_traj = 0.5
            swing_grf = -0.75
            stance_grf = 1.5

    class commands(MiniCheetahCfg.commands):
        heading_command = False
        resampling_time = 4.
        curriculum = True
        max_curriculum_x = 4.
        max_curriculum_ang = 2.5
        class ranges(MiniCheetahCfg.commands.ranges):
            lin_vel_x = [-1., 2.] # min max [m/s]
            lin_vel_y = 1.   # max [m/s]
            yaw_vel = 3.14/2.    # max [rad/s]
            heading = 0.

    class normalization(MiniCheetahCfg.normalization):
            class obs_scales(MiniCheetahCfg.normalization.obs_scales): 
                # * helper fcts
                # * dimensionless time: sqrt(L/g) or sqrt(I/[mgL]), with I=I0+mL^2
                v_leg = BASE_HEIGHT_REF
                dimless_time = (v_leg/9.81)**0.5
                base_z = 1./v_leg
                lin_vel = 1./v_leg  # virtual leg lengths per second
                ang_vel = 1./3.14*dimless_time
                dof_pos = [10., 1., 0.5]  # [50, 10., 5.]
                dof_vel = 0.01 # 0.05  # ought to be roughly max expected speed.
                height_measurements = 1./v_leg
            clip_actions = 100.

    class noise(MiniCheetahCfg.noise):
        add_noise = True
        noise_level = 1.  # scales other values

        class noise_scales(MiniCheetahCfg.noise.noise_scales):
            dof_pos = 0.005  # can be made very low
            dof_vel = 0.005
            ang_vel = [0.3, 0.15, 0.4]  # 0.027, 0.14, 0.37
            gravity = 0.05

    class sim:
        dt =  0.001
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]

class SERefCfgPPO(MiniCheetahCfgPPO):
    seed = -1
    do_wandb = True

    class wandb:
        what_to_log = {}

    class policy( MiniCheetahCfgPPO.policy ):
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

    class algorithm( MiniCheetahCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 6
        num_mini_batches = 6  # mini batch size = num_envs*nsteps/nminibatches
        learning_rate = 5.e-5
        schedule = 'adaptive'  # can be adaptive, fixed
        gamma = 0.99
        lam = 0.99
        desired_kl = 0.01
        max_grad_norm = 1.

    class state_estimator_nn:
        # how many quantities we are estimating:
        num_outputs = 4
        hidden_dims = [256, 128, 64]  # None will default to 256, 128
        # dropouts: randomly zeros output of a node.
        # specify the probability of a dropout, 0 means no dropouts.
        # Done per layer, including initial layer (input-first, no last-output)
        # len(dropouts) == len(hidden_dims)
        dropouts = [0.1, 0.1, 0.1]

    class state_estimator:
        num_learning_epochs = 10
        num_mini_batches = 1  # mini batch size = num_envs*nsteps / nminibatches


    class runner(MiniCheetahCfgPPO.runner):
        run_name = ''
        experiment_name = 'se_ref_D'
        max_iterations = 1000  # number of policy updates
        SE_learner = 'modular_SE'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32 # per iteration (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
