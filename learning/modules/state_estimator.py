import torch
import torch.nn as nn
from .utils import create_MLP


class StateEstimatorNN(nn.Module):
    """ Set up a neural network for state-estimation

    Keyword arguments:
    num_inputs -- the number of observations used
    num_outputs -- the number of estimated states
    hidden_dims -- list of hidden layers and their sizes (default [256, 128])
    activation -- activation type (default 'elu')
    dropouts -- list of dropout rate for each layer (default None)
    """
    def __init__(self,  num_inputs, num_outputs, hidden_dims=[256, 128],
                 activation='elu', dropouts=None, **kwargs):
        if kwargs:
            print("StateEstimator.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.NN = create_MLP(num_inputs, num_outputs, hidden_dims,
                             activation, dropouts)
        print(f"State Estimator MLP: {self.NN}")

    def evaluate(self, observations):
        return self.NN(observations)

    def export(self, path):
        import os
        import copy
        os.makedirs(path, exist_ok=True)
        path_ts = os.path.join(path, 'SE.pt')
        path_onnx = os.path.join(path, 'SE.onnx')
        model = copy.deepcopy(self.NN).to('cpu')
        model.eval()
        input = torch.rand(self.num_inputs,)
        model_traced = torch.jit.trace(model, input)
        torch.jit.save(model_traced, path_ts)
        torch.onnx.export(model_traced, input, path_onnx)
