import torch
import random
from learning.modules.utils.normalize import RunningMeanStd


def test_update_moments():
    num_items = random.randint(1, 10)
    num_envs = 5
    rms = RunningMeanStd(num_items)
    rms.count = 10 * torch.ones(())
    rms.running_mean = 0.95 * torch.ones(num_envs, num_items)
    rms.running_var = 0.2 * torch.ones(num_envs, num_items)
    batch_mean = 0.35 * torch.ones(num_envs, num_items)
    batch_var = 0.1 * torch.ones(num_envs, num_items)
    batch_count = num_envs
    new_mean, new_var, tot_count = rms._update_mean_var_from_moments(
        rms.running_mean,
        rms.running_var,
        rms.count,
        batch_mean,
        batch_var,
        batch_count,
    )
    ex_new_mean = torch.ones(num_envs, num_items) * 0.75
    ex_new_var = torch.ones(num_envs, num_items) * 3.7 / 14.0
    assert torch.equal(new_mean, ex_new_mean), "Updated mean doesn't match expectations"
    assert torch.equal(new_var, ex_new_var), "Updated var doesn't match expectations"
    assert tot_count == 15, "Updated count doesn't match expectations"


def test_normalize():
    num_items = 2
    num_envs = 2
    rms = RunningMeanStd(num_items)
    rms.training = True
    normalized = rms(torch.ones(num_envs, num_items))
    ex_normalized = (
        (1.0 / 3.0) / (5.0 / 6.0 + rms.epsilon) ** (1.0 / 2.0)
    ) * torch.ones_like(normalized)
    assert torch.all(
        torch.isclose(normalized, ex_normalized, rtol=1e-05, atol=1e-08)
    ).item(), "The normalized output doesn't match expectation"
