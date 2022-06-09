
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import StateEstimator
from rsl_rl.storage import LTStorageSE

class SE:
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
                 device='cpu',
                 ):

        self.device = device
        self.learning_rate = learning_rate
        self.storage = None  # initialized later
        self.transition = LTStorageSE.Transition()

        # SE network and optimizer
        self.state_estimator = state_estimator
        self.state_estimator.to(self.device)
        self.SE_optimizer = optim.Adam(self.state_estimator.parameters(),
                                    lr=learning_rate)
        self.SE_loss_fn = nn.MSELoss()


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                        critic_obs_shape, action_shape, se_shape):
        # TODO: implement this function
        self.storage = LTStorageSE(num_envs, num_transitions_per_env,
                                        self.LT_storage_size,
                                        se_shape,
                                        self.device)


    def predict(self, obs, critic_obs):
        # Compute the predicted states
        SE_prediction = self.state_estimator.evaluate(obs)
        actor_obs = torch.cat((SE_prediction, obs), dim=1)

        # Store transition values
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.SE_prediction = SE_prediction
        return actor_obs

    def process_env_step(self, dones, infos, new_actor_obs, new_critic_obs):

        self.transition.dones = dones
        self.transition.SE_targets = infos['SE_targets']
        # Record the transition

        self.storage.LT_add_transitions(self.transition)  # TODO: not implemented yet
        # * add transitions that are not time-outs to LT storage

        keep = ~infos['time_outs'].to(self.device)
        self.storage.add_LT_transitions(self.transition.observations[keep, :],
                            self.transition.critic_observations[keep, :],
                            self.transition.actions[keep, :],
                            new_actor_obs[keep, :],
                            new_critic_obs[keep, :],
                            self.transition.rewards[keep].unsqueeze(1))
                            # possibly add dones (mark failure terminations)

        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        # last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.rollout.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.rollout.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.rollout.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, SE_target_batch, hid_states_batch, masks_batch in generator:

                # * plug state-estimator into policy
                SE_prediction_batch = self.state_estimator.evaluate(obs_batch,
                                        masks=masks_batch,
                                        hidden_states=hid_states_batch[0])
                actor_obs_batch = torch.cat((SE_prediction_batch.detach(),
                                            obs_batch), dim=1)

                # * do a pass on the state-estimator too
                SE_loss = self.SE_loss_fn(SE_prediction_batch, SE_target_batch)
                self.SE_optimizer.zero_grad()
                SE_loss.backward()
                self.SE_optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.rollout.clear()

        return mean_value_loss, mean_surrogate_loss        
