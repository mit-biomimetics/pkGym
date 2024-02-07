import torch
import sys
import numpy as np
from gym.utils.helpers import class_to_dict


class TaskSkeleton:
    def __init__(self, num_envs=1, max_episode_length=1.0, device="cpu"):
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        return None

    def get_states(self, obs_list):
        return torch.cat([self.get_state(obs) for obs in obs_list], dim=-1)

    def get_state(self, name):
        if name in self.scales.keys():
            return getattr(self, name) / self.scales[name]
        else:
            return getattr(self, name)

    def set_states(self, state_list, values):
        idx = 0
        for state in state_list:
            state_dim = getattr(self, state).shape[1]
            self.set_state(state, values[:, idx : idx + state_dim])
            idx += state_dim
        assert idx == values.shape[1], "Actions don't equal tensor shapes"

    def set_state(self, name, value):
        try:
            if name in self.scales.keys():
                setattr(self, name, value * self.scales[name])
            else:
                setattr(self, name, value)
        except AttributeError:
            print("Value for " + name + " does not match tensor shape")

    def _reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.step()

    def _reset_buffers(self):
        self.to_be_reset[:] = False
        self.terminated[:] = False
        self.timed_out[:] = False

    def compute_reward(self, reward_weights):
        """Compute and return a torch tensor of rewards
        reward_weights: dict with keys matching reward names, and values
            matching weights
        """
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for name, weight in reward_weights.items():
            reward += weight * self._eval_reward(name)
        return reward

    def _eval_reward(self, name):
        return eval("self._reward_" + name + "()")

    def _check_terminations_and_timeouts(self):
        """Check if environments need to be reset"""
        contact_forces = self.contact_forces[:, self.termination_contact_indices, :]
        self.terminated = torch.any(torch.norm(contact_forces, dim=-1) > 1.0, dim=1)
        self.timed_out = self.episode_length_buf >= self.max_episode_length
        self.to_be_reset = self.timed_out | self.terminated

    def step(self, actions):
        raise NotImplementedError

    def check_exit(self):
        if self.exit:
            sys.exit()

    def _parse_cfg(self, cfg=None):
        self.dt = self.cfg.control.ctrl_dt
        self.scales = class_to_dict(self.cfg.scaling, self.device)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
