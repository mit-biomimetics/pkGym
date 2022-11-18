import unittest
from gym.envs import *
from gym.utils import get_args, task_registry
import torch
import isaacgym
from gym.utils import class_to_dict

class TestHumanoidRewardDimensions(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        args = get_args()
        args.headless = True
        task_registry.make_gym()
        self.env_list = []
        # * activate all rewards
        for env_name in task_registry.task_classes.keys():
            args.task = env_name  # todo adjust make_sim args input
            env_cfg, train_cfg = task_registry.create_cfgs(args)
            task_registry.update_sim_cfg(args)
            task_registry.make_sim()
            env, _ = task_registry.make_env(name=env_name,
                                            env_cfg=env_cfg)
            self.env_list.append(env)
            task_registry._gym.destroy_sim(task_registry._sim)

    def test_all_rewards_have_right_shape(self):
        for env in self.env_list:
            for item in dir(env):
                if "_reward_" in item:
                    reward_name = item.replace("_reward_", "")
                    self.assertEqual(len(env._eval_reward(reward_name).shape),
                                     1,
                                     f"Wrong shape for {reward_name}")


if __name__ == '__main__':
    unittest.main()
