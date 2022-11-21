
import wandb
from gym.utils.logging_and_saving \
    import wandb_helper, local_code_save_helper


# configure local and cloud code saving and logging
def log_and_save(env, env_cfg, train_cfg, runner, args):
    # setup local code saving if enabled
    if local_code_save_helper.use_local_saving(train_cfg):
        save_paths = local_code_save_helper.get_save_local_paths(env, env_cfg)
        runner.configure_local_files(save_paths)

    # setup WandB if enabled
    if wandb_helper.use_wandb(train_cfg, args):
        wandb_helper.wandb_setup(runner, train_cfg, env_cfg, args)


# close WandB process after training has finished
def wandb_close(train_cfg, args):
    # close WandB after learning is done
    if wandb_helper.use_wandb(train_cfg, args):
        wandb.finish()
