import os
import json
import wandb
from torch.multiprocessing import Process
from torch.multiprocessing import set_start_method

import isaacgym

from gym.envs import *
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry
from gym.utils.logging_and_saving import wandb_singleton
from gym.scripts.train import train, setup


def load_sweep_config(file_name):
    return json.load(open(os.path.join(
        LEGGED_GYM_ROOT_DIR, 'gym', 'scripts',
        'sweep_configs', file_name)))


def set_wandb_sweep_cfg_values(env_cfg, train_cfg, parameters_dict):
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


def train_with_sweep_cfg():
    env_cfg, train_cfg, policy_runner = setup()

    # * update the config settings based off the sweep_dict
    parameter_dict = wandb.config
    set_wandb_sweep_cfg_values(env_cfg, train_cfg, parameter_dict)

    train(train_cfg=train_cfg, policy_runner=policy_runner)


def sweep_wandb_mp():
    ''' start a new process for each train function '''

    p = Process(target=train_with_sweep_cfg, args=())
    p.start()
    p.join()
    p.kill()


def start_sweeps(args):
    # * required for multiprocessing CUDA workloads
    set_start_method('spawn')

    # * load sweep_config from JSON file
    if args.wandb_sweep_config is not None:
        sweep_config = load_sweep_config(args.wandb_sweep_config)
    else:
        sweep_config = load_sweep_config('sweep_config_example.json')
    # * set sweep_id if you have a previous id to use
    sweep_id = None
    if args.wandb_sweep_id is not None:
        sweep_id = args.wandb_sweep_id

    _, train_cfg = task_registry.create_cfgs(args)

    wandb_helper = wandb_singleton.WandbSingleton()
    wandb_helper.set_wandb_values(args, train_cfg)

    if wandb_helper.get_project_name() is not None and \
       wandb_helper.get_entity_name() is not None:
        if sweep_id is None:
            sweep_id = wandb.sweep(
                sweep_config,
                entity=wandb_helper.get_entity_name(),
                project=wandb_helper.get_project_name())
        wandb.agent(
            sweep_id,
            sweep_wandb_mp,
            entity=wandb_helper.get_entity_name(),
            project=wandb_helper.get_project_name(),
            count=sweep_config['run_cap'])
    else:
        print('ERROR: No WandB project and entity provided for sweeping')


if __name__ == '__main__':
    args = get_args()
    start_sweeps(args)
