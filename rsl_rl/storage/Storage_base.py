import torch
import numpy as np

""" Store data for different loop
Design consideration: change the original "has-a" relationship of LT->rollout->transition to parallel.
Transition stores all the needed data for every step and will be cleared/updated every step and passed to rollout and longterm
Rollout store data for every iteration, generate batch for learning
Longterm stores needed data from transition, and probably also generate batches
"""

class TransitionBase:
    """ Transition storage class.
    i.e. store data for each STEP of ALL agents
    """
    def __init__(self):
        """ Define all the data you need to store in __init__
        """
        raise NotImplementedError
    
    def clear(self):
        self.__init__()

class RolloutStorageBase:
    """ This class stores rollout data for each update.
    i.e. all the transition data in one ITERATION. It will be used to update the policy
    """

    def __init__(self, num_envs, num_transitions_per_env, device='cpu'):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions(self, transition: TransitionBase):
        """ Add current transition to LT storage
        Store variables according to the __init__ 
        """
        self.step += 1
        raise NotImplementedError

    def clear(self):
        self.step = 0

    def mini_batch_generator(self):
        """ Generate mini batch for learning
        """
        raise NotImplementedError

class LTStorageBase:
    """ This class stores LONGTERM data needed for certain algorithm
    1. Define transition (step): should be complete data and parameters for single step
    1. Define the full buffer to store the data
    2. Add data from transition
    """

    def __init__(self, num_envs, num_transitions_per_env, LT_storage_size, device='cpu'):
        
        self.device = device
        self.LT_storage_size = LT_storage_size
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env  # num of steps for envs

        # LT storage data you want to store
        
        # other parameters

    def add_transitions(self, transition: TransitionBase):
        """ Add current transition to LT storage
        Store variables according to the __init__ 
        """
        raise NotImplementedError

    def mini_batch_generator(self):
        """ Generate mini batch for learning
        """
        raise NotImplementedError

