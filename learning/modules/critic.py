import torch.nn as nn
from .utils import create_MLP


class Critic(nn.Module):
    def __init__(self,
                 num_obs,
                 hidden_dims,
                 activation="elu",
                 **kwargs):

        if kwargs:
            print("Critic.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.NN = create_MLP(num_obs, 1, hidden_dims, activation)

    def evaluate(self, critic_observations):
        return self.NN(critic_observations).squeeze()
