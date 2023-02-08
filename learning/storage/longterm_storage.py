# ! This file is deprecated and will soon be replaced (in the works).
# temporarily left here for reference.

# * keep track of obs across entire training run
# for off-policy algorithms, initial conditions
# todo prune nearby
# todo add privileged observatiosn too?
import torch

from algorithms.storage import RolloutStorage


class LongTermStorage:

    # * initialization with State Estimator
    def __init__(self, num_envs, num_transitions_per_env, LT_storage_size,
                 actor_obs_shape, critic_obs_shape,
                 actions_shape, se_shape=None,
                 priv_obs_only=True,
                 device='cpu'):

        self.device = device
        self.actor_obs_shape = actor_obs_shape
        # self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape
        # * start rollout storage
        self.rollout = RolloutStorage(num_envs, num_transitions_per_env,
                                      actor_obs_shape, critic_obs_shape,
                                      actions_shape, se_shape, self.device)

        # * initialize long-term storage
        self.data_count = 0  # amount of data stored
        # * if not priv_obs_only:
        self.actor_obs = torch.zeros(
            LT_storage_size, *actor_obs_shape, device=self.device)
        self.next_actor_obs = torch.zeros(
            LT_storage_size, *actor_obs_shape, device=self.device)

        if critic_obs_shape[0] is None:
            self.critic_obs_shape = actor_obs_shape
            self.critic_obs = torch.zeros(
                LT_storage_size, *actor_obs_shape, device=self.device)
            self.next_critic_obs = torch.zeros(
                LT_storage_size, *actor_obs_shape, device=self.device)
        else:
            self.critic_obs_shape = critic_obs_shape
            self.critic_obs = torch.zeros(
                LT_storage_size, *critic_obs_shape, device=self.device)
            self.next_critic_obs = torch.zeros(
                LT_storage_size, *critic_obs_shape, device=self.device)

        self.rewards = torch.zeros(
            LT_storage_size, 1, device=self.device)
        self.actions = torch.zeros(
            LT_storage_size, *actions_shape, device=self.device)
        self.dones = torch.zeros(
            LT_storage_size, 1, device=self.device).byte()
        # * trackers
        self.LT_storage_size = LT_storage_size
        # self.num_envs = num_envs
        self.step = 0

    def add_LT_transitions(self, actor_obs, critic_obs, actions,
                           next_actor_obs, next_critic_obs, rewards):
        n_add = actor_obs.shape[0]  # how many are being added, same for all
        count = self.data_count
        if count > self.LT_storage_size:
            raise ("Overflow LT storage.")
        elif count == self.LT_storage_size:
            # * already in overflow, replace randomly
            indices = torch.randint(count, n_add)
            self.actor_obs[indices, :] = actor_obs
            self.critic_obs[indices, :] = critic_obs
            self.actions[indices, :] = actions
            self.next_actor_obs[indices, :] = next_actor_obs
            self.next_critic_obs[indices, :] = next_critic_obs
            self.rewards[indices] = rewards
        elif count+n_add >= self.LT_storage_size:
            # * will overflow, keep random set of incoming
            # * only keep as many as will fit
            indices = torch.randperm(n_add)[:self.LT_storage_size - count]
            self.actor_obs[count:, :] = actor_obs[indices, :]
            self.critic_obs[count:, :] = critic_obs[indices, :]
            self.actions[count:, :] = actions[indices, :]
            self.next_actor_obs[count:, :] = next_actor_obs[indices, :]
            self.next_critic_obs[count:, :] = next_critic_obs[indices, :]
            self.rewards[count:] = rewards[indices, :]

            indices = torch.randperm(count + n_add)[:self.LT_storage_size]
            temp = torch.cat((self.actor_obs[:count, :],
                              self.rollout.observations.flatten(end_dim=1)),
                             dim=0)
            self.actor_obs = temp[indices, :]
            count = self.LT_storage_size
        else:
            # * just fill
            self.actor_obs[count:count+n_add, :] = actor_obs
            self.critic_obs[count:count+n_add, :] = critic_obs
            self.actions[count:count+n_add, :] = actions
            self.next_actor_obs[count:count+n_add, :] = next_actor_obs
            self.next_critic_obs[count:count+n_add, :] = next_critic_obs
            self.rewards[count:count+n_add] = rewards

            self.data_count += n_add

    def clear(self):
        self.__init__()

    def clear_rollout_storage(self):
        self.rollout.clear()

    # def add_transitions(self, transition):
    #     self.rollout.add_transitions(transition)
    #     # * anything for LT? not yet...

    # def compute_returns(self, last_values, gamma, lam):
    #     advantage = 0
    #     for step in reversed(range(self.num_transitions_per_env)):
    #         if step == self.num_transitions_per_env - 1:
    #             next_values = last_values
    #         else:
    #             next_values = self.values[step + 1]
    #         # next_is_not_terminal will zero the rest of the return if false
    #         next_is_not_terminal = 1.0 - self.dones[step].float()
    #         delta = self.rewards[step] + next_is_not_terminal * gamma \
    #                     * next_values - self.values[step]
    #         advantage = (
    #             delta + next_is_not_terminal * gamma * lam * advantage)
    #         self.returns[step] = advantage + self.values[step]

    #     # Compute and normalize the advantages
    #     self.advantages = self.returns - self.values
    #    self.advantages = (
    #        (self.advantages - self.advantages.mean())
    #        / (self.advantages.std() + 1e-8))

    def compute_advantages(self, gamma, lam):
        """
        to be done
        """
        # For off-policy, this will need to be redone each time.
        # V(s) - (r + gamma * V(s'))
        # and Q(s, a) = r + gamma * V(s')
        raise NotImplementedError

    # def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    #     # ! this will be different
    #     batch_size = self.LT_storage_size
    #     mini_batch_size = batch_size // num_mini_batches
    #     indices = torch.randperm(num_mini_batches*mini_batch_size,
    #                                 requires_grad=False,
    #                                 device=self.device)

    #     obs = self.obs.flatten(0, 1)
    #     if self.critic_obs_shape is not None:
    #         priv_obs = self.critic_obs_shape.flatten(0, 1)
    #     else:
    #         priv_obs = obs

    #     actions = self.actions.flatten(0, 1)
    #     values = self.values.flatten(0, 1)
    #     q_values = self.q_values.flatten(0, 1)
    #     # returns = self.returns.flatten(0, 1)
    #     # old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    #     # advantages = self.advantages.flatten(0, 1)
    #     # old_mu = self.mu.flatten(0, 1)
    #     # old_sigma = self.sigma.flatten(0, 1)

    #     for epoch in range(num_epochs):
    #         for i in range(num_mini_batches):

    #             start = i*mini_batch_size
    #             end = (i+1)*mini_batch_size
    #             batch_idx = indices[start:end]

    #             obs_batch = obs[batch_idx]
    #             critic_obs_batch = priv_obs[batch_idx]
    #             actions_batch = actions[batch_idx]
    #             target_values_batch = values[batch_idx]
    #             target_q_values_batch = q_values[batch_idx]
    #             # returns_batch = returns[batch_idx]
    #             # old_actions_log_prob_batch = \
    #             #     old_actions_log_prob[batch_idx]
    #             # advantages_batch = advantages[batch_idx]
    #             # old_mu_batch = old_mu[batch_idx]
    #             # old_sigma_batch = old_sigma[batch_idx]
    #             yield obs_batch, critic_obs_batch, actions_batch, \
    #                     target_values_batch, target_q_values_batch
    #                     # , advantages_batch, returns_batch, \
    #                     # old_actions_log_prob_batch, old_mu_batch, \
    #                     # old_sigma_batch
