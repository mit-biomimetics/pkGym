import torch


class EpisodicLogs:
    def __init__(self, num_envs, episode_dt, window=100, device="cpu"):
        self.log_items = {}
        self.episode_times = torch.zeros(num_envs, requires_grad=False, device=device)
        self.num_envs = num_envs
        self.window_size = window
        self.finished_episodes = {
            "time": torch.zeros(self.window_size, requires_grad=False, device=device)
        }
        self.finished_episode_count = 0
        self.episode_dt = episode_dt
        self.device = device
        self.average_episode_time = torch.zeros(
            self.window_size, requires_grad=False, device=device
        )

    def add_buffer(self, variable_names):
        assert isinstance(variable_names, list), "variable_names must be a list"
        for key in variable_names:
            self.log_items[key] = torch.zeros(
                self.num_envs, requires_grad=False, device=self.device
            )
            self.finished_episodes[key] = torch.zeros(
                self.window_size, requires_grad=False, device=self.device
            )
        return None

    def add_step(self, data_dict):
        for key, value in data_dict.items():
            self.log_items[key] += value

    def finish_step(self, dones):
        self.episode_times += self.episode_dt
        if dones.any():
            self.update_averages_buffer(dones)

    def update_averages_buffer(self, dones):
        def handle_overflow_entire_buffer(m, n, idx):
            if m >= self.window_size:
                n = 0
                m = self.window_size
                idx = idx[-m:]
            return m, n, idx

        idx = dones.nonzero().squeeze(dim=1)
        idx_reset = idx
        m = len(idx)
        assert m > 0
        n = self.finished_episode_count % self.window_size
        m, n, idx = handle_overflow_entire_buffer(m, n, idx)

        if n + m <= self.window_size:
            self.update_without_overflow(n, m, idx, idx_reset)
        else:
            self.update_with_overflow(n, m, idx, idx_reset)

        self.finished_episode_count += m

    def update_without_overflow(self, n, m, idx, idx_reset):
        for key, tensor in self.log_items.items():
            self.finished_episodes[key][n : n + m] = (
                tensor[idx] / self.episode_times[idx]
            )

            self.log_items[key][idx_reset] = 0.0
        self.finished_episodes["time"][n : n + m] = self.episode_times[idx]
        self.episode_times[idx_reset] = 0.0

    def update_with_overflow(self, n, m, idx, idx_reset):
        k = n + m - self.window_size

        for key, tensor in self.log_items.items():
            self.finished_episodes[key][n:] = (
                tensor[idx[k:]] / self.episode_times[idx[k:]]
            )
            self.finished_episodes[key][:k] = (
                tensor[idx[:k]] / self.episode_times[idx[:k]]
            )
            self.log_items[key][idx_reset] = 0.0

        self.finished_episodes["time"][n:] = self.episode_times[idx[k:]]
        self.finished_episodes["time"][:k] = self.episode_times[idx[:k]]
        self.episode_times[idx_reset] = 0.0

    def get_average_rewards(self):
        averages = {}
        for key in self.log_items.keys():
            averages[key] = torch.sum(self.finished_episodes[key])
            if self.finished_episode_count < self.window_size:
                averages[key] /= self.finished_episode_count
            else:
                averages[key] /= self.window_size
        return averages

    def get_average_time(self):
        if self.finished_episode_count < self.window_size:
            return self.finished_episodes["time"].sum() / self.finished_episode_count
        else:
            return self.finished_episodes["time"].mean()
