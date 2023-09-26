import os
import isaacgym
import torch

from gym.envs import __init__
from gym.utils.logging_and_saving import local_code_save_helper
from gym.utils import task_registry

from multiprocessing import Process, Manager


def _run_training_test(return_queue, args):
    # * do initial setup
    env_cfg, train_cfg = task_registry.create_cfgs(args)

    task_registry.make_gym_and_sim()
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    policy_runner = task_registry.make_alg_runner(env=env, train_cfg=train_cfg)

    local_code_save_helper.log_and_save(env, env_cfg, train_cfg, policy_runner)

    # * take a full iteration step
    policy_runner.learn(num_learning_iterations=args.max_iterations)

    # * get the test values after learning
    actions = policy_runner.env.get_states(train_cfg.policy.actions)

    # * return the values to the parent for assertion
    return_queue.put(actions[0, :].cpu().clone())


class TestEnvTraining():
    def test_env_training_settings(self, args):
        # * simulation settings for the test
        args.task = 'mini_cheetah'
        args.seed = 0
        args.max_iterations = 20
        args.save_interval = 100
        args.num_envs = 1000
        args.headless = True
        args.disable_wandb = True

        # * create a queue to return the values for assertion
        manager = Manager()
        return_queue = manager.Queue()

        # * spin up a child process to run the simulation iteration
        test_proc = Process(
            target = _run_training_test, args=(return_queue, args))
        test_proc.start()
        test_proc.join()

        # * extract the values to test from the child's return queue
        actions = return_queue.get()
        unit_test_file = os.path.dirname(__file__)
        expected_tensor_file = "actions_expected.pt"
        tensor_file_path = os.path.join(unit_test_file, expected_tensor_file)
        actions_expected = torch.load(tensor_file_path)

        # * assert the returned values are as expected
        assert torch.all(torch.eq(actions, actions_expected))

        # * kill the child process now that we're done with it
        test_proc.kill()