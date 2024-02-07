from learning.utils import Logger
from learning.utils import PotentialBasedRewardShaping

import torch


# let's create a simple dummy environment
class MyTask:
    def __init__(self):
        super().__init__()

    def compute_reward(self, reward_weights):
        reward = torch.zeros(1, device="cpu", dtype=torch.float)
        for name, weight in reward_weights.items():
            reward += weight * self._eval_reward(name)
        return reward

    def _eval_reward(self, name):
        return eval("self._reward_" + name + "()")

    def _reward_first(self):
        return torch.ones(1)

    def _reward_second(self):
        return torch.ones(1)


def test_PBRS():
    logger = Logger()
    logger.initialize(num_envs=1, episode_dt=0.1, total_iterations=100, device="cpu")
    self_env = MyTask()

    # let's create a dummy policy_cfg with just the reward weights
    policy_cfg = {"reward": {"weights": {"dummy": 1}}}
    policy_cfg["reward"]["pbrs_weights"] = {"first": 1, "second": 2}

    # and a dummy env

    PBRS = PotentialBasedRewardShaping(
        policy_cfg["reward"]["pbrs_weights"], device="cpu"
    )
    assert PBRS.get_reward_keys() == ["PBRS_first", "PBRS_second"]

    rewards_dict = {}
    logger.register_rewards(PBRS.get_reward_keys())

    # get new actions

    PBRS.pre_step(self_env)
    assert PBRS.prestep_counter > PBRS.poststep_counter
    # env step, get obs, rewards, dones etc.

    dones = torch.zeros(1, dtype=torch.bool)

    # handle standard rewards, and in addition...
    rewards_dict.update(PBRS.post_step(self_env, dones))
    assert PBRS.prestep_counter == PBRS.poststep_counter

    logger.log_rewards(rewards_dict)
    logger.finish_step(dones)
