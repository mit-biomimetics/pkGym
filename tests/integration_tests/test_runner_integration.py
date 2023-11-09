import os
import torch

from gym.envs import __init__  # noqa: F401
from gym.utils import task_registry


def run_integration_test(args):
    # * do initial setup
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    train_cfg.runner.save_interval = args.save_interval
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    policy_runner = task_registry.make_alg_runner(env=env, train_cfg=train_cfg)

    # * get the default test values before learning
    default_actions = policy_runner.env.get_states(train_cfg.policy.actions)

    # * take a full iteration step
    policy_runner.learn()

    # * get the test values after learning
    actions = policy_runner.env.get_states(train_cfg.policy.actions)

    # * return the values for assertion
    it = policy_runner.it
    log_dir = policy_runner.log_dir
    return actions.cpu().clone(), default_actions.cpu().clone(), it, log_dir


class TestDefaultIntegration:
    def test_default_integration_settings(self, args):
        # * simulation settings for the test
        args.task = "mini_cheetah"
        args.max_iterations = 8
        args.save_interval = 3
        args.num_envs = 16
        args.headless = True
        args.disable_wandb = True

        actions, default_actions, it, log_dir = run_integration_test(args)

        assert (
            torch.equal(actions, default_actions) is False
        ), "Actions were not updated from default"

        assert it == args.max_iterations, "Iteration update incorrect"

        # * check correct saving
        model_0_path = os.path.join(log_dir, "model_0.pt")
        model_1_path = os.path.join(log_dir, "model_1.pt")
        model_2_path = os.path.join(log_dir, "model_2.pt")
        model_3_path = os.path.join(log_dir, "model_3.pt")
        model_4_path = os.path.join(log_dir, "model_4.pt")
        model_5_path = os.path.join(log_dir, "model_5.pt")
        model_6_path = os.path.join(log_dir, "model_6.pt")
        model_7_path = os.path.join(log_dir, "model_7.pt")
        model_8_path = os.path.join(log_dir, "model_8.pt")

        assert os.path.exists(model_0_path), (
            f"{model_0_path} " "(pre-iteration) was not saved"
        )
        assert not os.path.exists(model_1_path), f"{model_1_path} was saved"
        assert not os.path.exists(model_2_path), f"{model_2_path} was saved"
        assert os.path.exists(model_3_path), f"{model_3_path} was not saved"
        assert not os.path.exists(model_4_path), f"{model_4_path} was saved"
        assert not os.path.exists(model_5_path), f"{model_5_path} was saved"
        assert os.path.exists(model_6_path), f"{model_6_path} was not saved"
        assert not os.path.exists(model_7_path), f"{model_7_path} was saved"
        assert os.path.exists(model_8_path), (
            f"{model_5_path}" "(last iteration) was not saved"
        )
