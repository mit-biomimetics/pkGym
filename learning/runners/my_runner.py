import torch

from learning.utils import Logger
from learning.utils import PotentialBasedRewardShaping
from learning.utils import remove_zero_weighted_rewards

from .on_policy_runner import OnPolicyRunner

logger = Logger()


class MyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

    def learn(self):
        self.set_up_logger()
        remove_zero_weighted_rewards(self.policy_cfg["reward"]["pbrs_weights"])
        PBRS = PotentialBasedRewardShaping(
            self.policy_cfg["reward"]["pbrs_weights"], self.device
        )
        logger.register_rewards(PBRS.get_reward_keys())

        rewards_dict = {}

        self.alg.actor_critic.train()
        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
        tot_iter = self.it + self.num_learning_iterations

        self.save()

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic("iteration")
            logger.tic("collection")
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    self.set_actions(
                        self.policy_cfg["actions"],
                        actions,
                        self.policy_cfg["disable_actions"],
                    )

                    PBRS.pre_step(self.env)
                    self.env.step()

                    actor_obs = self.get_noisy_obs(
                        self.policy_cfg["actor_obs"], self.policy_cfg["noise"]
                    )
                    critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
                    # * get time_outs
                    timed_out = self.get_timed_out()
                    terminated = self.get_terminated()
                    dones = timed_out | terminated

                    self.update_rewards(rewards_dict, terminated)
                    rewards_dict.update(PBRS.post_step(self.env, dones))
                    total_rewards = torch.stack(tuple(rewards_dict.values())).sum(dim=0)

                    logger.log_rewards(rewards_dict)
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(dones)

                    self.alg.process_env_step(total_rewards, dones, timed_out)
                self.alg.compute_returns(critic_obs)
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update()
            logger.toc("learning")
            logger.log_category()

            logger.finish_iteration()
            logger.toc("iteration")
            logger.toc("runtime")
            logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save()
        self.save()

    def update_rewards(self, rewards_dict, terminated):
        rewards_dict.update(
            self.get_rewards(
                self.policy_cfg["reward"]["termination_weight"], mask=terminated
            )
        )
        rewards_dict.update(
            self.get_rewards(
                self.policy_cfg["reward"]["weights"],
                modifier=self.env.dt,
                mask=~terminated,
            )
        )

    def set_up_logger(self):
        logger.register_rewards(list(self.policy_cfg["reward"]["weights"].keys()))
        logger.register_rewards(
            list(self.policy_cfg["reward"]["termination_weight"].keys())
        )
        logger.register_rewards(["total_rewards"])
        logger.register_category(
            "algorithm", self.alg, ["mean_value_loss", "mean_surrogate_loss"]
        )

        logger.attach_torch_obj_to_wandb(
            (self.alg.actor_critic.actor, self.alg.actor_critic.critic)
        )
