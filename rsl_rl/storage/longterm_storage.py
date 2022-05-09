
# * keep track of obs across entire training run
# for off-policy algorithms, initial conditions
# todo prune nearby
# todo add privileged observatiosn too?
import torch
import numpy as np

# from rsl_rl.utils import split_and_pad_trajectories
# recurrent only

class LongTermStorage:
    # ! probably don't use this, pull in directly from RolloutStorage
    class Transition:
        def __init__(self):
            self.obs = None
            self.priv_obs = None
            self.next_obs = None
            self.next_priv_obs = None
            self.actions = None
            self.rewards = None
            self.dones = None  # probably not needed
            # self.values = None
            self.actions_log_prob = None
            # self.action_mean = None
            # self.action_sigma = None
            # self.hidden_states = None

        def clear(self):
            self.__init__()

    # def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):
    def __init__(self, max_storage, obs_shape, privileged_obs_shape,
                    actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.data_count = 0  # amount of data stored
        # Core
        self.obs = torch.zeros(max_storage, *obs_shape,
                                        device=self.device)
        self.next_obs = torch.zeros(max_storage, *obs_shape,
                                        device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_obs = torch.zeros(max_storage,
                                    *privileged_obs_shape, device=self.device)
            self.next_priv_obs = torch.zeros(max_storage,
                                    *privileged_obs_shape, device=self.device)
        else:
            self.privileged_obs = None
        self.rewards = torch.zeros(max_storage, 1,
                                   device=self.device)
        self.actions = torch.zeros(max_storage, *actions_shape,
                                   device=self.device)
        self.dones = torch.zeros(max_storage, 1,
                                 device=self.device).byte()

        # For PPO
        # * probably don't need
        # self.actions_log_prob = torch.zeros(max_storage, 1,
        #                                     device=self.device)
        # self.values = torch.zeros(max_storage, 1, device=self.device)
        # self.returns = torch.zeros(max_storage, 1,
        #                             device=self.device)
        # self.advantages = torch.zeros(max_storage, 1,
        #                                 device=self.device)
        # self.mu = torch.zeros(max_storage, *actions_shape,
        #                         device=self.device)
        # self.sigma = torch.zeros(max_storage, *actions_shape,
        #                         device=self.device)

        # trackers
        self.max_storage = max_storage
        # self.num_envs = num_envs
        self.step = 0

    # def add_transition(self, transition: Transition):
    #     self.observations[self.data_count]

    # def add_transitions(self, transition: Transition):
    #     if self.data_count >= self.max_storage:
    #         raise AssertionError("Rollout buffer overflow")
    #         # ! probably change message
    #     self.obs[self.step].copy_(transition.obs)
    #     if self.privileged_obs is not None:
    #         self.privileged_obs[self.step].copy_(transition.priv_obs)

    #     self.actions[self.step].copy_(transition.actions)
    #     self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    #     self.dones[self.step].copy_(transition.dones.view(-1, 1))
    #     # ! values need to re-evaluated for each update step
    #     # self.values[self.step].copy_(transition.values)
    #     # ! this data is used off-policy
    #     # self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    #     # ! ppo stuff
    #     # self.mu[self.step].copy_(transition.action_mean)
    #     # self.sigma[self.step].copy_(transition.action_sigma)
    #     self.step += 1  # probably not needed

    #     self.data_count += 1

    def clear(self):
        self.step = 0

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
    #         advantage = delta + next_is_not_terminal * gamma * lam * advantage
    #         self.returns[step] = advantage + self.values[step]

    #     # Compute and normalize the advantages
    #     self.advantages = self.returns - self.values
    #     self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_advantages(self, gamma, lam):
        """
        to be done
        """
        # For off-policy, this will need to be redone each time.
        # V(s) - (r + gamma * V(s'))
        # and Q(s, a) = r + gamma * V(s')
        return None

    # ! probably don't need this anymore
    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1],
                                    dtype=torch.int64),
                                    flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # ! this will be different
        batch_size = self.max_storage
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size,
                                    requires_grad=False,
                                    device=self.device)

        obs = self.obs.flatten(0, 1)
        if self.privileged_obs is not None:
            priv_obs = self.privileged_obs.flatten(0, 1)
        else:
            priv_obs = obs

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        q_values = self.q_values.flatten(0, 1)
        # returns = self.returns.flatten(0, 1)
        # old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        # advantages = self.advantages.flatten(0, 1)
        # old_mu = self.mu.flatten(0, 1)
        # old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = obs[batch_idx]
                critic_obs_batch = priv_obs[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                target_q_values_batch = q_values[batch_idx]
                # returns_batch = returns[batch_idx]
                # old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                # advantages_batch = advantages[batch_idx]
                # old_mu_batch = old_mu[batch_idx]
                # old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_obs_batch, actions_batch, \
                        target_values_batch, target_q_values_batch
                        # , advantages_batch, returns_batch, \
                        # old_actions_log_prob_batch, old_mu_batch, \
                        # old_sigma_batch

    
    # def add_data(self, new_transitions):
