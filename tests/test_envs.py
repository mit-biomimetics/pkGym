import unittest
from gym.envs import *
from gym.utils import get_args, task_registry
import torch
import isaacgym
from gym.utils import class_to_dict

class TestEnvs(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.args = get_args()
        self.args.headless = True
        task_registry.make_gym()
        self.env_names = list(task_registry.task_classes.keys())

    def test_one_step(self):
        for env_name in self.env_names:
            self.args.task = env_name
            env_cfg, train_cfg = task_registry.create_cfgs(self.args)
            task_registry.update_sim_cfg(self.args)
            task_registry.make_sim()

            try:
                env, _ = task_registry.make_env(name=env_name,env_cfg=env_cfg)
                env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False))
            except:
                print(f'{env_name}.step() is NOT WORKING!')
                
            try:
                self.assertEqual(len(env.extras), 0, "env.extras has been modified!")
            except:
                print(f'{env_name}.extras is NOT EMPTY!')

            task_registry._gym.destroy_sim(task_registry._sim)


if __name__ == '__main__':
    unittest.main()