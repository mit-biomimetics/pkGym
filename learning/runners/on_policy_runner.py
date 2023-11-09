import os
import torch
from learning.env import VecEnv

from learning.utils import Logger

from .BaseRunner import BaseRunner

logger = Logger()


class OnPolicyRunner(BaseRunner):
    def __init__(self, env: VecEnv, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # * unpack out of config
        reward_weights = self.policy_cfg["reward"]["weights"]
        termination_weight = self.policy_cfg["reward"]["termination_weight"]
        rewards_dict = {}
        total_rewards = torch.zeros(self.env.num_envs, device=self.device)

        # * set up logger
        logger.register_rewards(list(reward_weights.keys()))
        logger.register_rewards(list(termination_weight.keys()))
        logger.register_rewards(["total_rewards"])

        logger.register_category(
            "algorithm", self.alg, ["mean_value_loss", "mean_surrogate_loss"]
        )

        logger.attach_torch_obj_to_wandb(
            (self.alg.actor_critic.actor, self.alg.actor_critic.critic)
        )

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
        self.alg.actor_critic.train()
        self.num_learning_iterations = num_learning_iterations
        tot_iter = self.it + num_learning_iterations

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

                    self.env.step()

                    actor_obs = self.get_noisy_obs(
                        self.policy_cfg["actor_obs"], self.policy_cfg["noise"]
                    )
                    critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
                    # * get time_outs
                    timed_out = self.get_timed_out()
                    terminated = self.get_terminated()
                    dones = timed_out | terminated

                    rewards_dict.update(
                        self.get_rewards(termination_weight, mask=terminated)
                    )
                    rewards_dict.update(
                        self.get_rewards(
                            reward_weights,
                            modifier=self.env.dt,
                            mask=~terminated,
                        )
                    )

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

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.it,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.it = loaded_dict["iter"]

    def switch_to_eval(self):
        self.alg.actor_critic.eval()

    def get_inference_actions(self):
        obs = self.get_noisy_obs(self.policy_cfg["actor_obs"], self.policy_cfg["noise"])
        return self.alg.actor_critic.actor.act_inference(obs)

    def export(self, path):
        self.alg.actor_critic.export_policy(path)
