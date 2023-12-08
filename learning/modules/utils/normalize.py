import torch
import torch.nn as nn


def get_mean_var_with_masks(values, masks):
    """
    Applies a mask to the input values and calculates
    the mean and variance over the valid entries, as specified
    by the mask.
    """
    sum_mask = masks.sum()
    masked_vals = values * masks
    values_mean = masked_vals.sum() / sum_mask
    min_sqr = (((masked_vals) ** 2) / sum_mask).sum() - (
        (masked_vals / sum_mask).sum()
    ) ** 2
    values_var = min_sqr * sum_mask / (sum_mask - 1)
    return values_mean, values_var


class RunningMeanStd(nn.Module):
    """
    Keeps a running mean to normalize tensor of choice.
    """

    def __init__(self, num_items, axis=0, epsilon=1e-05):
        super(RunningMeanStd, self).__init__()
        self.num_items = num_items
        self.axis = axis
        self.epsilon = epsilon

        self.register_buffer(
            "running_mean", torch.zeros(num_items, dtype=torch.float64)
        )
        self.register_buffer("running_var", torch.ones(num_items, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_from_moments(
        self,
        running_mean,
        running_var,
        running_count,
        batch_mean,
        batch_var,
        batch_count,
    ):
        """
        Implements parallel algorithm for combining arbitrary sets A and B
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        #Parallel_algorithm
        """
        tot_count = running_count + batch_count
        delta = batch_mean - running_mean

        if running_count > 1e2 and abs(running_count - batch_count) < 10:
            new_mean = (
                (running_count * running_var) * (batch_count * batch_var)
            ) / tot_count
        else:
            new_mean = running_mean + (delta * batch_count) / tot_count
        m_a = running_var * running_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2 * running_count * batch_count) / tot_count
        new_var = M2 / (tot_count - 1)
        return new_mean, new_var, tot_count

    def forward(self, input, mask=None):
        """
        Returns the normalized version of the input.
        """
        assert (
            input.size()[1] == self.num_items
        ), f"input doesn't match expected size {self.num_items}"
        if self.training:
            if mask is not None:
                mean, var = get_mean_var_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)
                var = input.var(self.axis)
            (
                self.running_mean,
                self.running_var,
                self.count,
            ) = self._update_mean_var_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size()[0],
            )

        current_mean = self.running_mean
        current_var = self.running_var

        y = (input - current_mean.float()) / torch.sqrt(
            current_var.float() + self.epsilon
        )
        return y
