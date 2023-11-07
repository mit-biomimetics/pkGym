from gym.envs import __init__  # noqa: F401
from gym.utils import get_args, task_registry
import torch


def setup():
    custom_parameter = [
        {
            "name": "--output_tensor_file",
            "type": str,
            "help": "Path to save the output tensor.",
        }
    ]
    args = get_args(custom_parameter)
    args.task = "mini_cheetah"
    args.seed = 0
    # args.max_iterations = 50
    # args.save_interval = 50
    args.num_envs = 1000
    args.headless = True
    args.disable_wandb = True

    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(args.task, env_cfg)
    runner = task_registry.make_alg_runner(env, train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    return env, runner, args


def _run_training_test(output_tensor_file, env, runner):
    # * take a full iteration step
    runner.learn(num_learning_iterations=5)

    # * get the test values after learning
    actions = runner.env.get_states(runner.policy_cfg["actions"])

    # * return the values to the parent for assertion
    actions.detach().cpu()
    torch.save(actions, output_tensor_file)


def worker():
    env, runner, args = setup()
    _run_training_test(args.output_tensor_file, env, runner)


if __name__ == "__main__":
    worker()
