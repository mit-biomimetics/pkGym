from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CartpoleCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 12  # 12 for the 12 actuated DoFs of the mini cheetah
        num_observations = 87  # added blindly from the AnymalCFlatCFG TODO: why this number?

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
        default_setup = "Basic"  # default setup chooses how the initial conditions are chosen.
        # "Basic" = a single position with some randomized noise on top.
        # "Range" = a range of joint positions and velocities.
        #  "Trajectory" = feed in a trajectory to sample from.

        default_joint_angles = {
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": 0.0,
            "RH_HAA": 0.0,

            "LF_HFE": -0.785398,
            "LH_HFE": -0.785398,
            "RF_HFE": -0.785398,
            "RH_HFE": -0.785398,

            "LF_KFE": 1.596976,
            "LH_KFE": 1.596976,
            "RF_KFE": 1.596976,
            "RH_KFE": 1.596976,
        }

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup
        dof_pos_high = [0., 0., 0., 0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]  # DOF dimensions
        dof_pos_low = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        dof_vel_high = [0., 0., 0., 0.0, 0., 0., 0., 0., 0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        dof_vel_low = [0., 0., 0., 0.0, 0., 0., 0., 0., 0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        com_pos_high = [0., 0., 1., 0., 0.5,
                        0.]  # COM dimensions, in euler angles because randomizing in quat is confusing
        com_pos_low = [0., 0., 1., 0., -0.5, 0.]  # x, y ,z, roll, pitch, yaw
        com_vel_high = [0., 0., 0., 0., 0.0, 0.]
        com_vel_low = [0., 0., 0., 0., 0., 0.]

        # * initialization for trajectory (needs trajectory)
        # ref_traj = "../../resources/robots/mit_humanoid/trajectories/humanoid3d_walk.csv"
        # ref_type = "Pos" #Pos, PosVel

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}  # [N*m*s/rad]

        # requires reference trajectory to be loaded
        # TODO: ignore if no ref traj is loaded
        nominal_pos = False
        nominal_vel = False

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5

        use_actuator_network = False
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        friction_range = [0.,
                          1.5]  # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
        foot_name = "FOOT"  # TODO: fix this!
        penalize_contacts_on = ["SHANK", "THIGH"]  # TODO: fix this!
        terminate_after_contacts_on = ["base"]
        initial_penetration_check = False
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

        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        base_height_target = 0.33
        tracking_sigma = 0.25

        # reference traj tracking
        base_pos_tracking = 0.
        base_vel_tracking = 0.
        dof_pos_tracking = 1.
        dof_vel_tracking = 0.

        class scales(LeggedRobotCfg.rewards.scales):
            reference_traj = 0.0
            termination = -1.
            tracking_lin_vel = 1.
            tracking_ang_vel = 0.1
            lin_vel_z = -0.
            ang_vel_xy = -0.0
            orientation = 0.1
            torques = -5.e-7
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.25
            feet_air_time = 2.5  # rewards keeping feet in the air
            collision = -1.
            action_rate = -0.01  # -0.01
            action_rate2 = -0.001
            stand_still = -0.
            dof_pos_limits = -0.25
            feet_contact_forces = -0.15
            # symm_legs = 0.0
            # symm_arms = 0.0

    class commands(LeggedRobotCfg.commands):
        heading_command = True
        resampling_time = 4.

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.5, 2.0]  # min max [m/s]
            lin_vel_y = [0., 0]  # min max [m/s]
            ang_vel_yaw = [0., 0.]  # min max [rad/s]
            heading = [-0.5, 0.5]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            # * helper fcts
            # * dimensionless time: sqrt(L/g) or sqrt(I/[mgL]), with I=I0+mL^2
            v_leg = 0.33
            dimless_time = (v_leg / 9.81) ** 0.5
            # lin_vel = 1/v_leg*dimless_time
            base_z = 1. / v_leg
            lin_vel = 1. / v_leg  # virtual leg lengths per second
            # ang_vel = 0.25
            ang_vel = 1. / 3.14 * dimless_time
            dof_pos = 1. / 3.14
            dof_vel = 0.05  # ought to be roughly max expected speed.

            height_measurements = 1. / v_leg

        # clip_observations = 100.
        clip_actions = 1000.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 0.1  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt = 0.002
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]


class CartpoleCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'mini_cheetah'