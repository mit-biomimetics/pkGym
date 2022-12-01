
from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg

BASE_HEIGHT_REF = 0.33

class MiniCheetahCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 12
        episode_length_s = 10.

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

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

        dof_pos_decay = 0.35  # set to None to disable

        ctrl_frequency = 100
        desired_sim_frequency = 200

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
        disable_motors = False  # all torques set to 0

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 3.14*(BASE_HEIGHT_REF/9.81)**0.5
        base_lin_vel = 1.
        commands = 1
        dof_vel = 100.  # ought to be roughly max expected speed.
        base_height = BASE_HEIGHT_REF
        dof_pos = 4*[0.1, 1., 2]  # hip-abad, hip-pitch, knee
        dof_pos_obs = dof_pos
        # Action scales
        dof_pos_target = dof_pos
        tau_ff = 4*[18, 18, 28]  # hip-abad, hip-pitch, knee


class MiniCheetahRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    class wandb:
        what_to_log = {}

    class policy( LeggedRobotRunnerCfg.policy ):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        actor_obs = ["base_height",
                     "base_lin_vel",
                     "base_ang_vel",
                     "projected_gravity",
                     "commands",
                     "dof_pos_obs",
                     "dof_vel",
                     "dof_pos_history",
                     ]

        critic_obs = ["base_height",
                      "base_lin_vel",
                      "base_ang_vel",
                      "projected_gravity",
                      "commands",
                      "dof_pos_obs",
                      "dof_vel",
                      "dof_pos_history",
                      ]

        actions = ["dof_pos_target"]
        class noise:
            dof_pos_obs = 0.005  # can be made very low
            dof_vel = 0.005
            base_ang_vel = 0.05  # 0.027, 0.14, 0.37
            projected_gravity = 0.02

        class reward(LeggedRobotRunnerCfg.policy.reward):
            make_PBRS = []
            class weights(LeggedRobotRunnerCfg.policy.reward.weights):
                tracking_lin_vel = 1.0
                tracking_ang_vel = 1.0
                lin_vel_z = 0.
                ang_vel_xy = 0.0
                orientation = 1.0
                torques = 5.e-7
                dof_vel = 0.
                base_height = 1.
                action_rate = 0.001  # -0.01
                action_rate2 = 0.0001  # -0.001
                stand_still = 0.
                dof_pos_limits = 0.
                feet_contact_forces = 0.
                dof_near_home = 1.
            class termination_weight:
                termination = 1.


    class algorithm( LeggedRobotRunnerCfg.algorithm):
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


    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ''
        experiment_name = 'mini_cheetah'
        max_iterations = 1000  # number of policy updates
        SE_learner = None
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
