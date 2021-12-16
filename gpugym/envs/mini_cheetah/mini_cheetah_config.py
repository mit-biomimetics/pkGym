
from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MiniCheetahCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 12  # 12 for the 12 actuated DoFs of the mini cheetah
        num_observations = 48  # added blindly from the AnymalCFlatCFG TODO: why this number?

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # added blindly from the AnymalCFlatCFG.
        # 'trimesh' # from Nikita: use a triangle mesh instead of a height field
        measure_heights = False  # added blindly from the AnymalCFlatCFG TODO: why this?

    class init_state(LeggedRobotCfg.init_state):
        """
        Initial States of the Mini Cheetah
        From Robot-Software/systems/quadruped/state_machine/FSM_State_RecoveryStand.cpp, line 38
        Ab/ad: 0˚, hip: -45˚, knee: 91.5˚
        Default pose is around 0.3
        """
        pos = [0.0, 0.0, 0.32]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
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

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
        foot_name = "FOOT"  # TODO: fix this!
        penalize_contacts_on = ["SHANK", "THIGH"]  # TODO: fix this!
        # penalize_contacts_on = ['base']
        terminate_after_contacts_on = ["base"]
        self_collisions = 1   # added blindly from the AnymalCFlatCFG.  1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False # False means there is gravity

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.
            # feet_contact_forces = -0.01

    # added from AnymalCRoughCfg
    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(LeggedRobotCfg.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

class MiniCheetahCfgPPO(LeggedRobotCfgPPO):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'mini_cheetah'
