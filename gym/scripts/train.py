import isaacgym
from gym.envs import *
from gym.utils import get_args, task_registry
from gym.utils.logging_and_saving import local_code_save_helper, wandb_helper


def train(args):
    # * prepare environment
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    env, env_cfg = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    # * then make env
    policy_runner, train_cfg = \
        task_registry.make_alg_runner(env=env, name=args.task, args=args)
    task_registry.prepare_sim()

    local_code_save_helper.log_and_save(
        env, env_cfg, train_cfg, policy_runner, args)

    policy_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True)

    wandb_helper.close_wandb(args)


if __name__ == '__main__':
    args = get_args()
    train(args)
