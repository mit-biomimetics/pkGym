import os
import wandb
from gym import LEGGED_GYM_ROOT_DIR


def is_wandb_enabled(args):
    """Return true if entity and project are in the commandline args"""

    enable_wandb = True

    # check if an entity and project are defined on the command line,
    if None in (args.wandb_project, args.wandb_entity):
        enable_wandb = False

    return enable_wandb


def setup_wandb(policy_runner, args):
    experiment_name = f'{args.task}'

    wandb.config = {}


    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               dir=os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'),
               config=wandb.config,
               name=experiment_name)

    wandb.run.log_code(
        os.path.join(LEGGED_GYM_ROOT_DIR, 'gym'))

    policy_runner.configure_wandb(wandb)


def close_wandb(args):
    """Close WandB process after training has finished"""

    if is_wandb_enabled(args):
        wandb.finish()
