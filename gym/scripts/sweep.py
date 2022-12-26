import wandb
import isaacgym
from gym.envs import *
from gym.utils import get_args, task_registry
from gym.utils.logging_and_saving import local_code_save_helper, wandb_helper


# todo: feature upgrade, move this to yaml/json
def configure_sweep():
    """Configure the sweep instructions including strategy,
       goal metric, and parameters to sweep over.
       Returns a dictionary to pass to WandB"""

    # set the sweep strategy
    sweep_config = {
        'method': 'random'
    }

    # set the metric objective for the sweeps
    metric = {
        'name': 'Train/mean_reward',
        'goal': 'maximize'
    }

    # set the number of sweeps to perform
    count = 5

    # set the sweep parameters to change
    # can be ranges or lists of values
    parameters_dict = {
        'train_cfg.runner.max_iterations': {
            'values': [10]
        },
        'env_cfg.env.episode_length_s': {
            'min': 2,
            'max': 10
        }
    }

    sweep_config['metric'] = metric
    sweep_config['count'] = count
    sweep_config['parameters'] = parameters_dict

    return sweep_config


def set_wandb_sweep_cfg_values(env_cfg, train_cfg, sweep_dict):
    parameters_dict = sweep_dict['parameters']

    for key, value in parameters_dict.items():
        print('Setting: ' + key + ' = ' + str(value))
        locs = key.split('.')

        if locs[0] == 'train_cfg':
            attr = train_cfg
        elif locs[0] == 'env_cfg':
            attr = env_cfg
        else:
            print('Unrecognized cfg: ' + locs[0])
            break

        for loc in locs[1:-1]:
            attr = getattr(attr, loc)

        setattr(attr, locs[-1], value)
        print('set ' + locs[-1] + ' to ' + str(getattr(attr, locs[-1])))


def sweep_wandb():
    args = get_args()
    # * prepare environment
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_sim()
    env, env_cfg = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    # * then make env
    policy_runner, train_cfg = \
        task_registry.make_alg_runner(env=env, name=args.task, args=args)
    task_registry.prepare_sim()  # this gets called in legged too for some reason?

    local_code_save_helper.log_and_save(
        env, env_cfg, train_cfg, policy_runner, args, is_sweep=True)

    parameter_dict = wandb.config
    # todo: fix this mismatch
    sweep_dict = {
        'parameters': parameter_dict
    }

    # update the config settings based off the sweep_dict
    set_wandb_sweep_cfg_values(env_cfg, train_cfg, sweep_dict)

    policy_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True)

    wandb_helper.close_wandb(args)

    task_registry.destroy_sim()


def start_sweeps(args):
    sweep_config = configure_sweep()
    # make one gym for all sweeps - each sweep makes its own sim
    task_registry.make_gym()
    # todo: add the project name from args
    sweep_id = wandb.sweep(
        sweep_config, entity="ajm4", project="wandb-sweeps-testing")
    wandb.agent(sweep_id, sweep_wandb, count=sweep_config['count'])


if __name__ == '__main__':
    args = get_args()
    start_sweeps(args)
