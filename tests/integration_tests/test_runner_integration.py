import os
import torch

from gym.envs import __init__  # noqa: F401
from gym.utils import task_registry


def learn_policy(args):
    # * do initial setup
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    train_cfg.runner.save_interval = args.save_interval
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    runner = task_registry.make_alg_runner(env=env, train_cfg=train_cfg)

    # * take a full iteration step
    runner.learn()

    return runner


class TestDefaultIntegration:
    def test_default_integration_settings(self, args):
        # * simulation settings for the test
        args.task = "mini_cheetah"
        args.max_iterations = 8
        args.save_interval = 3
        args.num_envs = 16
        args.headless = True
        args.disable_wandb = True

        # runner, actions, default_actions, it, log_dir = run_integration_test(args)
        runner = learn_policy(args)
        runner.switch_to_eval()

        with torch.no_grad():
            actions = runner.get_inference_actions()
            deployed_actions = runner.env.get_states(runner.policy_cfg["actions"])
        assert (
            torch.equal(actions, torch.zeros_like(actions)) is False
        ), "Policy returning all zeros"
        assert (
            torch.equal(deployed_actions, torch.zeros_like(deployed_actions)) is False
        ), "Actions not written to environment"

        assert runner.it == args.max_iterations, "Iteration update incorrect"

        # * check correct saving
        model_0_path = os.path.join(runner.log_dir, "model_0.pt")
        model_1_path = os.path.join(runner.log_dir, "model_1.pt")
        model_2_path = os.path.join(runner.log_dir, "model_2.pt")
        model_3_path = os.path.join(runner.log_dir, "model_3.pt")
        model_4_path = os.path.join(runner.log_dir, "model_4.pt")
        model_5_path = os.path.join(runner.log_dir, "model_5.pt")
        model_6_path = os.path.join(runner.log_dir, "model_6.pt")
        model_7_path = os.path.join(runner.log_dir, "model_7.pt")
        model_8_path = os.path.join(runner.log_dir, "model_8.pt")

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

        obs = torch.randn_like(runner.get_obs(runner.policy_cfg["actor_obs"]))
        actions_first = runner.alg.actor_critic.act_inference(obs).cpu().clone()
        runner.load(model_8_path)
        actions_loaded = runner.alg.actor_critic.act_inference(obs).cpu().clone()

        assert torch.equal(actions_first, actions_loaded), "Model loading failed"
