import os
import json
import wandb
import isaacgym
from gym.envs import *
from gym.utils import get_args, task_registry
from gym.utils.logging_and_saving \
    import local_code_save_helper, wandb_singleton
from torch.multiprocessing import Process
from torch.multiprocessing import set_start_method
from gym import LEGGED_GYM_ROOT_DIR


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


def train():
    args = get_args()
    wandb_helper = wandb_singleton.WandbSingleton()

    # * prepare environment
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    env, env_cfg = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    # * then make env
    policy_runner, train_cfg = \
        task_registry.make_alg_runner(env=env, name=args.task, args=args)
    task_registry.prepare_sim()

    local_code_save_helper.log_and_save(
        env, env_cfg, train_cfg, policy_runner)
    wandb_helper.set_wandb_values(args, train_cfg)
    if wandb_helper.is_wandb_enabled():
        wandb_helper.setup_wandb(policy_runner, is_sweep=True)

    parameter_dict = wandb.config

    # * update the config settings based off the sweep_dict
    set_wandb_sweep_cfg_values(env_cfg, train_cfg, parameter_dict)

    policy_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True)

    wandb_helper.close_wandb()


def sweep_wandb_mp():
    ''' start a new process for each train function '''

    p = Process(target=train, args=())
    p.start()
    p.join()
    p.kill()


def start_sweeps(args):
    # * required for multiprocessing CUDA workloads
    set_start_method('spawn')

    # * load sweep_config from JSON file
    sweep_config = load_sweep_config('sweep_config_example.json')

    _, train_cfg = task_registry.create_cfgs(args)

    wandb_helper = wandb_singleton.WandbSingleton()
    wandb_helper.set_wandb_values(args, train_cfg)

    if wandb_helper.get_project_name() is not None and \
       wandb_helper.get_entity_name() is not None:
        sweep_id = wandb.sweep(
            sweep_config,
            entity=wandb_helper.get_entity_name(),
            project=wandb_helper.get_project_name())
        wandb.agent(sweep_id, sweep_wandb_mp, count=sweep_config['run_cap'])
    else:
        print('ERROR: No WandB project and entity provided for sweeping')


if __name__ == '__main__':
    args = get_args()
    start_sweeps(args)
