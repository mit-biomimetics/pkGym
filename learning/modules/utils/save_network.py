import torch
import os
import copy

def save_network(network, network_name, num_inputs, path):
    """
    Thsi function traces and exports the given network module in .pt and 
    .onnx file formats. These can be used for evaluation on other systems
    without needing a Pytorch environment.

    :param network:         PyTorch neural network module
    :param network_name:    (string) Network will be saved with this name
    :param num_inputs:      (int) Number of inputs to the network module
    :path:                  (string) Network will be saved to this location
    """

    os.makedirs(path, exist_ok=True)
    path_TS = os.path.join(path, network_name + '.pt')   # TorchScript path
    path_onnx = os.path.join(path, network_name + '.onnx')   # ONNX path
    model = copy.deepcopy(network).to('cpu')
    # To trace model, must be evaluated once with arbitrary input
    model.eval()
    dummy_input = torch.rand(num_inputs,)
    model_traced = torch.jit.trace(model, dummy_input)
    torch.jit.save(model_traced, path_TS)
    torch.onnx.export(model_traced, dummy_input, path_onnx)