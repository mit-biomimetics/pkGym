import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class StateEstimatorNN(nn.Module):
    is_recurrent = False
    def __init__(self,  num_inputs, num_outputs, hidden_dims=[256, 128],
                 activation='elu', dropouts=None, **kwargs):
        if kwargs:
            print("StateEstimator.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(StateEstimatorNN, self).__init__()

        activation = get_activation(activation)
        if dropouts is None:
            dropouts = [0]*len(hidden_dims)

        # Policy
        layers = []
        if not hidden_dims:  # handle no hidden layers
            layers.append(nn.Linear(num_inputs, num_outputs))
        else:
            layers.append(nn.Linear(num_inputs, hidden_dims[0]))
            if dropouts[0] > 0:
                        layers.append(nn.Dropout(p=dropouts[0]))
            layers.append(activation)
            for l in range(len(hidden_dims)):
                if l == len(hidden_dims) - 1:
                    layers.append(nn.Linear(hidden_dims[l], num_outputs))
                else:
                    layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                    if dropouts[l+1] > 0:
                        layers.append(nn.Dropout(p=dropouts[l+1]))
                    layers.append(activation)
        self.estimator = nn.Sequential(*layers)

        print(f"State Estimator MLP: {self.estimator}")

        # Action noise
        # disable args validation for speedup
        Normal.set_default_validate_args = True

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def evaluate(self, observations, **kwargs):
        outputs = self.estimator(observations)
        return outputs

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
