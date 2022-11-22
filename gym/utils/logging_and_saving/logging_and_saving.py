
import wandb
from gym.utils.logging_and_saving \
    import wandb_helper, local_code_save_helper


# configure local and cloud code saving and logging
def log_and_save(env, env_cfg, train_cfg, runner, args):
    # setup local code saving if enabled
    if local_code_save_helper.check_local_saving_flag(train_cfg):
        save_paths = local_code_save_helper.get_local_save_paths(env, env_cfg)
        runner.configure_local_files(save_paths)

    # setup WandB if enabled
    if wandb_helper.is_wandb_enabled(args):
        wandb_helper.wandb_setup(runner, args)


# close WandB process after training has finished
def wandb_close(train_cfg, args):
    # close WandB after learning is done
    if wandb_helper.is_wandb_enabled(args):
        wandb.finish()
