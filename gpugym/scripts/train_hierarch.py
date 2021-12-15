import numpy as np
import os
from datetime import datetime

import isaacgym
from gpugym.envs import *
from gpugym.utils import get_args, task_registry, set_seed
import torch

def train(args):

    # * create envB
    env_B, env_B_cfg = task_registry.make_env(name='mit_humanoid', args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env_B, name=args.task, args=args)

    # * create policyB
    ppo_runner_B, _ = task_registry.make_alg_runner(env=env_B,
                                                        name=args.task,
                                                        args=args,
                                                        train_cfg=train_cfg)
    policy_B = ppo_runner_B.get_inference_policy(device=env_B.device)

    # * env, env_cfg = task_registry.make_env(name=args.task, args=args)
    if args.task in task_registry.task_classes:
       task_class = task_registry.get_task_class('hierarch')
    env_cfg, _ = task_registry.get_cfgs('hierarch')
    set_seed(env_cfg.seed)
    # sim_params = {"sim": class_to_dict(env_cfg.sim)}
    # sim_params = parse_sim_params(args, sim_params)
    # env = task_class(   cfg=env_cfg,
    #                     sim_params=sim_params,
    #                     physics_engine=args.physics_engine,
    #                     sim_device=args.sim_device,
    #                     headless=args.headless)
    env_A = task_class(cfg=env_cfg, env_B=env_B, policy_B=policy_B)
    # This is hierarch.__init__()

    # todo scale train_cfg.episode length to account

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env_A,
                                                          name=args.task,
                                                          args=args)

    # * we want to think of "num_steps_per_env" in terms of policy_A evals
    ppo_runner.cfg['num_steps_per_env'] *= env_A.max_steps_on_B
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=False)

if __name__ == '__main__':
    args = get_args()
    train(args)