import os
import torch

from learning.utils import Logger

from .BaseRunner import BaseRunner

logger = Logger()


class OnPolicyRunner(BaseRunner):
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

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        torch.save(
            {
                "actor_state_dict": self.alg.actor_critic.actor.state_dict(),
                "critic_state_dict": self.alg.actor_critic.critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.it,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.actor_critic.critic.load_state_dict(loaded_dict["critic_state_dict"])
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

    def init_storage(self):
        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            actor_obs_shape=[num_actor_obs],
            critic_obs_shape=[num_critic_obs],
            action_shape=[num_actions],
        )
