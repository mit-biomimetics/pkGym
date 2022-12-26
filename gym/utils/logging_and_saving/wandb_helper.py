import os
import wandb
from gym import LEGGED_GYM_ROOT_DIR


# may need to totally change this function to return the project and entity
# check if the args are set right, if so return those
# otherwise check train_cfg if true, then load json entity and project
# if neither, wandb should be off
def is_wandb_enabled(args, is_sweep=False):
    """Return true if entity and project are in the commandline args"""

    enable_wandb = True

    # # short-circuit turning WandB on if this is a sweep
    if is_sweep:
        return enable_wandb

    # check if an entity and project are defined on the command line,
    if None in (args.wandb_project, args.wandb_entity):
        enable_wandb = False

    return enable_wandb


def get_wandb_project_and_entity(args, train_cfg):
    wandb_project_and_entity = None

    if hasattr(train_cfg, 'wandb_settings') and \
       hasattr(train_cfg.wandb_settings, 'enable_wandb'):
        # load json for
        pass

    if None not in (args.wandb_project, args.wandb_entity):
        wandb_project_and_entity = {
            'project': args.wandb_project,
            'entity': args.wandb_entity
        }

    return wandb_project_and_entity


def setup_wandb(policy_runner, args, is_sweep=False):
    experiment_name = f'{args.task}'

    wandb.config = {}

    if is_sweep:
        print('Received is_sweep is true. Preparing WandB for sweeps.')

        wandb.init(dir=os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'),
                   config=wandb.config,
                   name=experiment_name)
    else:
        print(f'Received WandB project name: {args.wandb_project}\n' +
              'Received WandB entitiy name: {args.wandb_entity}\n')

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
