import torch
import numpy as np


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower


# @ torch.jit.script
def exp_avg_filter(x, avg, alpha=0.8):
    """
    Simple exponential average filter
    """
    avg = alpha * x + (1 - alpha) * avg
    return avg
