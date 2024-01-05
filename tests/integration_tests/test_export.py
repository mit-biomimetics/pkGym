import os
import onnx
import onnxruntime

from gym.envs import __init__  # noqa: F401
from gym.utils import task_registry
from learning.modules import ActorCritic
from gym.utils.helpers import get_load_path

import torch


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


def export_policy():
    return None


class TestONNX:
    def test_onnx_vs_pt(self, args):
        args.task = "mini_cheetah"
        args.max_iterations = 10
        args.save_interval = 10
        args.headless = True
        args.disable_wandb = True

        runner = learn_policy(args)
        runner.switch_to_eval()
        # load the saved policy
        num_actor_obs = runner.get_obs_size(runner.policy_cfg["actor_obs"])
        num_critic_obs = runner.get_obs_size(runner.policy_cfg["critic_obs"])
        num_actions = runner.get_action_size(runner.policy_cfg["actions"])
        actor_critic = ActorCritic(
            num_actor_obs, num_critic_obs, num_actions, **runner.policy_cfg
        ).to(runner.device)
        resume_path = get_load_path(
            name=runner.cfg["experiment_name"],
            load_run=runner.cfg["load_run"],
            checkpoint=runner.cfg["checkpoint"],
        )
        loaded_dict = torch.load(resume_path)
        actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        # eport the saved policy
        export_path = "exported_policy"
        actor_critic.export_policy(export_path)
        # load the exported onnx
        path_to_onnx = os.path.join(export_path, "policy.onnx")
        onnx_model = onnx.load(path_to_onnx)
        onnx.checker.check_model(onnx_model)

        # compare the runner policy with exported onnx output
        ort_session = onnxruntime.InferenceSession(
            path_to_onnx, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        # compute torch output
        with torch.no_grad():
            test_input = runner.get_obs(runner.policy_cfg["actor_obs"])[0:1]
            torch_out = runner.alg.actor_critic.actor.act_inference(test_input)
        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_input)}
        ort_out = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        print(torch_out)
        print(ort_out)
        torch.testing.assert_allclose(
            to_numpy(torch_out), ort_out[0], rtol=1e-03, atol=1e-05
        )
