from .Storage_base import LTStorageBase, RolloutStorageBase

class LTStorageSE(LTStorageBase):

    class Transition:                   # TODO: make sure this inherit can be redefined
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.dones = None
            self.values = None
            self.hidden_states = None
            self.SE_prediciton = None   # TODO: not sure whether we need this
            self.SE_targets = None

    def __init__(self, num_envs, num_transitions_per_env, LT_storage_size, device='cpu'):
        super().__init__(num_envs, num_transitions_per_env, LT_storage_size, device)
        self.transition = self.Transition()


    def LT_add_transitions(self, transition: Transition):
        
        pass

