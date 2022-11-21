# todo: test if these imports work/are needed
from gym.envs import class_dict, config_dict
from gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import os


# check if enable_local_saving is true in the training_config
def use_local_saving(train_cfg):
    if hasattr(train_cfg, 'logging') and \
       hasattr(train_cfg.logging, 'enable_local_saving'):
        enable_local_saving = train_cfg.logging.enable_local_saving
    else:
        enable_local_saving = False
    return enable_local_saving


# create a save_paths object to dictate the code to locally save for a run
def get_save_local_paths(env, env_cfg):
    # class_source = os.path.join(
    #     LEGGED_GYM_ENVS_DIR,
    #     class_dict[type(env).__name__][1:].replace('.', '/')+'.py')
    # class_target = os.path.join(
    #     'envs', *class_dict[type(env).__name__].split('.')[:-1])

    # config_source = os.path.join(
    #     LEGGED_GYM_ENVS_DIR,
    #     config_dict[type(env_cfg).__name__][1:].replace('.', '/')+'.py')
    # config_target = os.path.join(
    #     'envs',
    #     *config_dict[type(env_cfg).__name__].split('.')[:-1])

    # envs_source = os.path.join(
    #     LEGGED_GYM_ROOT_DIR, 'gym', 'envs', '__init__.py')
    # envs_target = os.path.join('envs')

    runners_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'learning', 'runners')
    runners_target = os.path.join('learning', 'runners')

    envs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym', 'envs')
    envs_target = os.path.join('envs')

    scripts_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym', 'scripts')
    scripts_target = os.path.join('scripts')

    utils_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym', 'utils')
    utils_target = os.path.join('utils')

    # list of things to copy
    # source paths need the full path and target are relative to log_dir
    save_paths = [
        # {'type': 'file', 'source_file': class_source,
        #                  'target_dir': class_target},
        # {'type': 'file', 'source_file': config_source,
        #                  'target_dir': config_target},
        # {'type': 'file', 'source_file': envs_source,
        #                  'target_dir': envs_target},
        {'type': 'dir', 'source_dir': runners_dir,
                        'target_dir': runners_target,
            'ignore_patterns': ['__pycache__*']},
        {'type': 'dir', 'source_dir': envs_dir,
                        'target_dir': envs_target,
            'ignore_patterns': ['__pycache__*']},
        {'type': 'dir', 'source_dir': scripts_dir,
                        'target_dir': scripts_target,
            'ignore_patterns': ['__pycache__*', 'wandb*']},
        {'type': 'dir', 'source_dir': utils_dir,
                        'target_dir': utils_target,
            'ignore_patterns': ['__pycache__*']}
    ]

    return save_paths
