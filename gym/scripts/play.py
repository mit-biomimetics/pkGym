from gym import LEGGED_GYM_ROOT_DIR
import os
import copy
import isaacgym
from gym.envs import *
from gym.utils import  get_args, task_registry, Logger, class_to_dict
from gym.utils import KeyboardInterface, GamepadInterface
import numpy as np
import torch
import isaacgym


def play(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    task_registry.make_gym_and_sim()
    env, env_cfg = task_registry.make_env(name=args.task, env_cfg=env_cfg)

    task_registry.prepare_sim()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env,
                                                          name=args.task,
                                                          args=args,
                                                          train_cfg=train_cfg)

    # switch to evaluation mode (dropout for example)
    ppo_runner.switch_to_eval()

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name, 'exported')
        ppo_runner.export(path)

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    # interface = GamepadInterface(env)
    interface = KeyboardInterface(env)
    for i in range(10*int(env.max_episode_length)):
        actions = ppo_runner.get_inference_actions()
        interface.update(env)
        env.set_states(train_cfg.policy.actions, actions)
        env.step()

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    with torch.no_grad():
        play(args)
