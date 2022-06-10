from .Storage_base import LTStorageBase, RolloutStorageBase, TransitionBase

import torch
import numpy as np

class TransitionSE(TransitionBase):
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        self.dones = None
        self.values = None
        self.hidden_states = None
        self.SE_prediciton = None   # TODO: not sure whether we need this
        self.SE_targets = None

    def clear(self):
        self.__init__()


class RolloutSE(RolloutStorageBase):
    """ This class stores rollout data for each update.
    i.e. all the transition data in one ITERATION. It will be used to update the policy
    """

    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, se_shape, device='cpu'):
        
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

        self.actor_obs_shape = actor_obs_shape    # raw states for actor
        # self.critic_obs_shape = critic_obs_shape  # privileged raw states for critic, TODO: do we need this for SE? delete for now
        self.se_shape = se_shape                  # SE prediction states
    
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device)
        # if critic_obs_shape[0] is not None:
        #     self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        # else:
        #     self.privileged_observations = None
        self.SE_targets = torch.zeros(num_transitions_per_env, num_envs, *se_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

    def add_transitions(self, transition: TransitionSE):
        """ Add current transition to LT storage
        Store variables according to the __init__ 
        """
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        # if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.SE_targets[self.step].copy_(transition.SE_targets)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """ Generate mini batch for learning
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        SE_targets = self.SE_targets.flatten(0, 1)
        # if self.privileged_observations is not None:
        #     critic_observations = self.privileged_observations.flatten(0, 1)
        # else:
        #     critic_observations = observations

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                # critic_observations_batch = critic_observations[batch_idx]
                SE_target_batch = SE_targets[batch_idx]
                yield obs_batch, SE_target_batch, (None, None), None  # hid_states and mask

class LTStorageSE(LTStorageBase):
    """ This class stores LONGTERM data needed for certain algorithm
    1. Define transition (step): should be complete data and parameters for single step
    1. Define the full buffer to store the data
    2. Add data from transition
    """

    def __init__(self, num_envs, num_transitions_per_env, LT_storage_size,
                 actor_obs_shape, critic_obs_shape, se_shape=None,
                 device='cpu'):
        
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env  # num of steps for envs
        self.device = device
        self.LT_storage_size = LT_storage_size
        
        self.actor_obs_shape = actor_obs_shape
        self.critic_obs_shape = critic_obs_shape
        
        # LT storage data you want to store
        self.actor_obs = torch.zeros(LT_storage_size, *actor_obs_shape,
                                device=self.device)
        self.dones = torch.zeros(LT_storage_size, 1,
                                    device=self.device).byte()
        # other parameters
        self.data_count = 0
    
    # TODO: add transisions and batch later
    def add_transitions(self, transition: TransitionBase):
        """ Add current transition to LT storage
        Store variables according to the __init__ 
        """
        pass

    def mini_batch_generator(self):
        """ Generate mini batch for learning
        """
        pass
