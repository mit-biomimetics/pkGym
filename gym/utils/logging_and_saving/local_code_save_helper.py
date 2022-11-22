import os
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils.logging_and_saving import wandb_helper


# configure local and cloud code saving and logging
def log_and_save(env, env_cfg, train_cfg, runner, args):
    # setup local code saving if enabled
    if check_local_saving_flag(train_cfg):
        save_paths = get_local_save_paths(env, env_cfg)
        runner.configure_local_files(save_paths)

    # setup WandB if enabled
    if wandb_helper.is_wandb_enabled(args):
        wandb_helper.setup_wandb(runner, args)


# check if enable_local_saving is set to true in the training_config
def check_local_saving_flag(train_cfg):
    if hasattr(train_cfg, 'logging') and \
       hasattr(train_cfg.logging, 'enable_local_saving'):
        enable_local_saving = train_cfg.logging.enable_local_saving
    else:
        enable_local_saving = False
    return enable_local_saving


# create a save_paths object for saving code locally
def get_local_save_paths(env, env_cfg):

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
