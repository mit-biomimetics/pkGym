import os

from gym.envs import __init__  # noqa: F401
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry

import torch


def setup_and_export(args):
    args.original_cfg = True
    args.resume = True
    args.headless = True

    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(args.task, env_cfg)
    runner = task_registry.make_alg_runner(env, train_cfg)

    runner.switch_to_eval()
    path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
    )
    runner.export(path)


if __name__ == "__main__":
    args = get_args()
    with torch.no_grad():
        setup_and_export(args)
