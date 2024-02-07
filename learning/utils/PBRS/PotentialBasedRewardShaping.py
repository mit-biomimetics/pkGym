from learning.utils import set_discount_from_horizon


class PotentialBasedRewardShaping:
    def __init__(self, weights, device="cpu"):
        self.rewards_dict = {}
        self.weights = weights
        self.discount = 1.0
        self.prestep_counter = 0
        self.poststep_counter = 0
        self.prestep = {}
        self.device = device
        return

    def set_discount(self, horizon, dt):
        self.discount = set_discount_from_horizon(dt, horizon)

    def pre_step(self, env):
        for name, weight in self.weights.items():
            reward = env.compute_reward({name: weight}).to(self.device)
            self.prestep.update({name: reward})
        self.prestep_counter += 1

    def post_step(self, env, mask=None):
        if mask is None:
            mask = 1.0

        self.poststep_counter += 1
        assert self.prestep_counter == self.poststep_counter, "PBRS rewards out of sync"

        for name, weight in self.weights.items():
            reward = env.compute_reward({name: weight}).to(self.device)
            self.rewards_dict.update(
                {"PBRS_" + name: (self.discount * reward - self.prestep[name]) * mask}
            )

        return self.rewards_dict

    def get_rewards_dict(self):
        return self.rewards_dict

    def get_reward_keys(self):
        return list("PBRS_" + weight for weight in self.weights.keys())
