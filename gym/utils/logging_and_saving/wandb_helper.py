import wandb


# return true is the wandb flag is set to true in the training_config
# and an entity and project are given in the commandline args
def use_wandb(train_cfg, args):
    # Check if we specified that we want to use wandb
    if hasattr(train_cfg, 'logging') and \
       hasattr(train_cfg.logging, 'enable_wandb'):
        enable_wandb = train_cfg.logging.enable_wandb
    else:
        enable_wandb = False
    # Do the logging only if wandb requirements have been fully specified
    if enable_wandb:
        if None in (args.wandb_project, args.wandb_entity):
            print('WARNING: WandB flag set to True, but no project or entity \
                  specified in the arguments. Setting WandB to False.')
            enable_wandb = False

    return enable_wandb


def wandb_setup(ppo_runner, train_cfg, env_cfg, args):
    experiment_name = f'{args.task}'

    wandb.config = {}

    if hasattr(train_cfg, 'wandb'):
        what_to_log = train_cfg.wandb.what_to_log
        craft_log_config(env_cfg, train_cfg, wandb.config, what_to_log)

    print(f'Received WandB project name: {args.wandb_project}\n \
          Received WandB entitiy name: {args.wandb_entity}\n')

    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               config=wandb.config,
               name=experiment_name)

    # if we want to follow the same save specific source files like local
    # saving, we need to rewrite the wandb log_code fn - deemed not critical
    # for now artifacts .py files from the parent dir in the cloud save
    wandb.run.log_code('..')

    ppo_runner.configure_wandb(wandb)
