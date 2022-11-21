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


def recursive_value_find(cfg, location):
    if len(location) == 1:
        return getattr(cfg, location[0])

    if hasattr(cfg, location[0]):
        return recursive_value_find(getattr(cfg, location[0]), location[1:])
    else:
        raise Exception(
            f"I couldn't find the value {location[0]} that you specified")


def craft_log_config(env_cfg, train_cfg, wandb_cfg, what_to_log):
    for log_key in what_to_log:
        location = what_to_log[log_key]
        if location[0] == 'train_cfg':
            wandb_cfg[log_key] = recursive_value_find(train_cfg, location[1:])
        elif location[0] == 'env_cfg':
            wandb_cfg[log_key] = recursive_value_find(env_cfg, location[1:])
        else:
            raise Exception(
                f"You didn't specify a valid cfg file in location: {location}")
