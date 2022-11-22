import wandb


# return true if an entity and project are given in the commandline args
def is_wandb_enabled(args):

    enable_wandb = True

    # check if an entity and project are defined on the command line,
    # check (later) if there is a local yaml, otherwise do not log wandb
    if None in (args.wandb_project, args.wandb_entity):
        print('WARNING: WandB flag set to True, but no project or entity \
                specified in the arguments. Setting WandB to False.')
        enable_wandb = False

    return enable_wandb


def wandb_setup(ppo_runner, args):
    experiment_name = f'{args.task}'

    wandb.config = {}

    print(f'Received WandB project name: {args.wandb_project}\n \
            Received WandB entitiy name: {args.wandb_entity}\n')

    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               config=wandb.config,
               name=experiment_name)

    # todo: need to make this not based on where the script is run, just caught this bug
    wandb.run.log_code('..')

    ppo_runner.configure_wandb(wandb)
