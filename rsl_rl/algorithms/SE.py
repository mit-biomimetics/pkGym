
from rsl_rl.storage.Storage_SE import TransitionSE
from rsl_rl.storage.rollout_storage import RolloutStorage
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import StateEstimator
from rsl_rl.storage import LTStorageSE, RolloutSE

class StateEstimatorMod:
    """ This class provides a learned state estimator.
    predict() function provides state estimation for RL given the observation
    update() function optimize for the nn params
    process_env_step() function store values in long term storage

    @parameters:
    Storage size for LTStorageSE
    SE network (see modules)
    
    TODO: maybe use LSTM later?

    """
    state_estimator: StateEstimator
    def __init__(self,
                 state_estimator,    # network
                 learning_rate=1e-3,
                 storage_size = 4000,
                 num_mini_batches = 1,
                 num_learning_epochs = 1,
                 device='cpu',
                 ):

        # general parameters
        self.device = device
        self.learning_rate = learning_rate
        self.SE_LT_size = storage_size
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs

        # SE storage
        self.transition = None
        self.rollout    = None
        self.storage    = None  # initialized later

        # SE network and optimizer
        self.state_estimator = state_estimator
        self.state_estimator.to(self.device)
        self.SE_optimizer = optim.Adam(self.state_estimator.parameters(),
                                    lr=learning_rate)
        self.SE_loss_fn = nn.MSELoss()


    def init_storage(self, num_envs, num_transitions_per_env, raw_obs_shape, se_shape):

        # TODO: implement this function, do we need action?? is it torque??
        self.transition = TransitionSE()
        self.rollout    = RolloutSE(num_envs, num_transitions_per_env, raw_obs_shape,
                        se_shape, device=self.device)
        self.LTstorage  = LTStorageSE(num_envs, num_transitions_per_env, self.SE_LT_size,
                                        raw_obs_shape,
                                        se_shape,
                                        # self.LT_priv_obs_only,
                                        self.device)


    def predict(self, obs, critic_obs):
        """ Predicte the estimated states
        return: cat(predicted, raw_states)
        """
        # Compute the predicted states
        SE_prediction = self.state_estimator.evaluate(obs)
        actor_obs = torch.cat((SE_prediction, obs), dim=1)

        # Store transition values
        self.transition.observations = obs                 # only raw state observation
        self.transition.critic_observations = critic_obs   # privilege raw states observation
        self.transition.SE_prediction = SE_prediction
        return actor_obs

    def process_env_step(self, dones, infos, new_actor_obs, new_critic_obs):

        # Record the transition
        self.transition.dones = dones
        self.transition.SE_targets = infos['SE_targets']

        # TODO: implement function, check delete new_actor_obs and new_critic_obs is ok
        # Store transitions to rollout and longterm
        self.rollout.add_transitions(self.transition)
        self.LTstorage.add_transitions(self.transition)

        self.transition.clear()

    def update(self):

        generator = self.rollout.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        mean_SE_loss = 0
        for obs_batch, SE_target_batch, hid_states_batch, masks_batch in generator:

            # * plug state-estimator into policy
            SE_prediction_batch = self.state_estimator.evaluate(obs_batch,
                                    masks=masks_batch,
                                    hidden_states=hid_states_batch[0])
                                    # TODO: what is mask and hid_states

            # * do a pass on the state-estimator too
            SE_loss = self.SE_loss_fn(SE_prediction_batch, SE_target_batch)
            self.SE_optimizer.zero_grad()
            SE_loss.backward()
            self.SE_optimizer.step()
            mean_SE_loss += SE_loss.item()  # TODO: check is it the right way to record loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_SE_loss /= num_updates
        self.rollout.clear()

        return mean_SE_loss        
