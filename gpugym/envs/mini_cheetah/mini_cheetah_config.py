
from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

BASE_HEIGHT_REF = 0.33

class MiniCheetahCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actions = 12
        num_observations = 75
        episode_length_s = 10.

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'  # added blindly from the AnymalCFlatCFG.
        # 'trimesh' # from Nikita: use a triangle mesh instead of a height field
        measure_heights = False  # added blindly from the AnymalCFlatCFG TODO: why this?

    class init_state(LeggedRobotCfg.init_state):
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
                        'hfe': [-0.85, -0.6],
                        'kfe': [-1.45, 1.72]}

        dof_vel_range = {'haa': [0., 0.],
                        'hfe': [0., 0.],
                        'kfe': [0., 0.]}

        root_pos_range = [[0., 0.],  # x
                          [0., 0.],  # y
                          [0.37, 0.4],  # z
                          [0., 0.],  # roll
                          [0., 0.],  # pitch
                          [0., 0.]]  # yaw
        root_vel_range = [[-0.05, 0.05],  # x
                          [0., 0.],  # y
                          [-0.05, 0.05],  # z
                          [0., 0.],  # roll
                          [0., 0.],  # pitch
                          [0., 0.]]  # yaw
                          
        # TODO: add new traj
        ref_traj = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/trajectories/single_leg.csv"
        ref_type = "Pos"

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'haa': 20., 'hfe': 20., 'kfe': 20.}
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}

        # Control type
        control_type = "P"  # "Td"

        # action scale: target angle = actionScale * action + defaultAngle
        if control_type == "T":
            action_scale = 20 * 0.5
        elif control_type == "Td":
            action_scale = 4.0 # 1e-2 # stiffness['haa'] * 0.5
        else:
            action_scale = 0.25 # 0.5

        # decimation: Number of control action updates @ sim DT per policy DT
        exp_avg_decay = 0.35  # set to None to disable
        decimation = 5

        use_actuator_network = False
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.75, 1.05]
        randomize_base_mass = False
        added_mass_range = [-2., 2.]
        friction_range = [0., 1.0] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.05

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah_simple.urdf"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base", "thigh"]
        collapse_fixed_joints = False  # merge bodies connected by fixed joints.
        self_collisions = 1  # added blindly from the AnymalCFlatCFG.  1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False
        disable_actions = False  # disable neural networks
        disable_motors = False  # all torques set to 0

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.

        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25
        class weights(LeggedRobotCfg.rewards.weights):
            termination = -1.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            lin_vel_z = -0.
            ang_vel_xy = 0.0
            orientation = 1.0
            torques = -5.e-7
            dof_vel = 0.
            base_height = 1.
            feet_air_time = 0.  # rewards keeping feet in the air
            collision = -0.
            action_rate = -0.001  # -0.01
            action_rate2 = -0.0001  # -0.001
            stand_still = 0.
            dof_pos_limits = 0.
            feet_contact_forces = 0.
            dof_near_home = 1.
            # symm_legs = 0.0
            # symm_arms = 0.0

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.
        curriculum = True
        max_curriculum = 3.
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0., 1.]  # min max [m/s]
            lin_vel_y = 0.   # max [m/s]
            yaw_vel = 0.  # max [rad/s]
            heading = 0.

    class normalization(LeggedRobotCfg.normalization):
            class obs_scales(LeggedRobotCfg.normalization.obs_scales):
                # * helper fcts
                # * dimensionless time: sqrt(L/g) or sqrt(I/[mgL]), with I=I0+mL^2
                v_leg = BASE_HEIGHT_REF
                dimless_time = (v_leg/9.81)**0.5
                # lin_vel = 1/v_leg*dimless_time
                base_z = 1./v_leg
                lin_vel = 1./v_leg  # virtual leg lengths per second
                # ang_vel = 0.25
                ang_vel = 1./3.14*dimless_time
                dof_pos = 1./3.14
                dof_vel = 0.01 # 0.05  # ought to be roughly max expected speed.

                action_scale = 1e-3

                height_measurements = 1./v_leg
            # clip_observations = 100.
            clip_actions = 1000.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.005  # can be made very low
            dof_vel = 0.005
            ang_vel = [0.3, 0.15, 0.4]  # 0.027, 0.14, 0.37
            gravity = 0.02
            # height_measurements = 0.1
    
    class sim:
        dt =  0.002
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]

class MiniCheetahCfgPPO(LeggedRobotCfgPPO):
    seed = -1
    do_wandb = False
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.


    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'mini_cheetah'
        max_iterations = 1000  # number of policy updates
        SE_learner = 'none'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
