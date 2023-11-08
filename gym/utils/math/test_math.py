import torch
import numpy as np
from simple_math import wrap_to_pi, torch_rand_sqrt_float, exp_avg_filter


def test_wrap_to_pi():
    angles = np.array([0, np.pi, 2 * np.pi, 3 * np.pi])
    wrapped_angles = wrap_to_pi(angles)
    assert np.allclose(wrapped_angles, np.array([0, np.pi, 0, np.pi]))


def test_torch_rand_sqrt_float():
    lower = -1.0
    upper = 1.0
    shape = (10,)
    device = torch.device("cpu")
    samples = torch_rand_sqrt_float(lower, upper, shape, device)
    assert samples.shape == shape
    assert torch.all(samples >= lower)
    assert torch.all(samples <= upper)


def test_exp_avg_filter():
    x = torch.tensor(5.0)
    avg = torch.tensor(1.0)
    test_alpha = [0.0, 0.5, 1.0]
    for alpha in test_alpha:
        new_avg = exp_avg_filter(x, avg, alpha)
        assert new_avg == alpha * x + (1 - alpha) * avg
