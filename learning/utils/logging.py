import os
import wandb
import torch
import shutil
import fnmatch
from collections import deque
from statistics import mean
import numbers


class Logger:
    def __init__(self, train_cfg, env):
        self.log_dir = train_cfg["log_dir"]
        self.device = train_cfg["runner"]["device"]
        self.avg_window = 100
        self.log = {}
        self.it = 0
        self.tot_iter = 0
        self.learning_iter = 0
        self.mean_episode_length = 0.
        self.total_mean_reward = 0.
        self.max_episode_length_s = env.max_episode_length_s
        self.train_cfg = train_cfg

        reward_keys_to_log = \
            list(train_cfg["policy"]["reward"]["weights"].keys()) \
            + list(train_cfg["policy"]["reward"]["termination_weight"].keys())

        self.extra_per_it_logs = train_cfg["extra_logs"]["per_iter"]
        self.extra_episode_mean_logs = train_cfg["extra_logs"]["episode_mean"]

        self.initialize_buffers(env.num_envs, reward_keys_to_log)
        self.initialize_extra_buffers(env.num_envs,
                                      self.extra_episode_mean_logs)

    def initialize_buffers(self, num_envs, reward_keys):
        self.current_episode_return = {
            name: torch.zeros(
                num_envs, dtype=torch.float,
                device=self.device, requires_grad=False)
            for name in reward_keys}
        self.current_episode_length = torch.zeros(
            num_envs,
            dtype=torch.float, device=self.device)
        self.avg_return_buffer = {
            name:  deque(maxlen=self.avg_window)
            for name in reward_keys}
        self.avg_length_buffer = deque(maxlen=self.avg_window)
        self.mean_rewards = {"Episode/"+name:  0. for name in reward_keys}

    def initialize_extra_buffers(self, num_envs, extra_keys):
        self.current_episode_log = {
            name: torch.zeros(
                num_envs, dtype=torch.float,
                device=self.device, requires_grad=False)
            for name in extra_keys}
        self.avg_log_buffer = {
            name:  deque(maxlen=self.avg_window)
            for name in extra_keys}
        self.mean_logs = {"Episode/Log/"+name:  0. for name in extra_keys}

    def log_to_wandb(self):
        wandb.log(self.log)

    def add_log(self, log_dict):
        self.log.update(log_dict)

    def update_iterations(self, it, tot_iter, learning_iter):
        self.it = it
        self.tot_iter = tot_iter
        self.learning_iter = learning_iter

    def log_step(self, env, rewards_dict, dones):
        for name, rewards in rewards_dict.items():
            self.log_current_reward(name, rewards)
        for name in self.extra_episode_mean_logs:
            log = env.compute_log(name)
            scaled = env.dt * log
            self.current_episode_log[name] += scaled
        self.update_episode_buffer(dones)

    def log_iteration(self, runner):
        runner_logs = self.get_runner_logs(runner)
        self.add_log(runner_logs)
        self.add_log(self.mean_rewards)
        self.add_log({
            'Train/mean_reward': self.total_mean_reward,
            'Train/mean_episode_length': self.mean_episode_length,
            })
        self.add_log(self.mean_logs)
        for log in self.extra_per_it_logs:
            value = runner.env.compute_log(log)
            if torch.is_tensor(value):
                value = value.item()
            if not isinstance(value, numbers.Number):
                raise ValueError(
                    f"Extra log {log} is not a number, but {type(value)}")
            else:
                self.add_log({
                    ('Log/' + log): value
                })

        self.update_iterations(runner.it, runner.tot_iter,
                               runner.num_learning_iterations)

        if wandb.run is not None:
            self.log_to_wandb()
        self.print_to_terminal()

    def log_current_reward(self, name, reward):
        if name in self.current_episode_return.keys():
            self.current_episode_return[name] += reward

    def update_episode_buffer(self, dones):
        self.current_episode_length += 1
        terminated_ids = torch.where(dones == True)[0]
        self.update_avg(self.avg_return_buffer,
                        self.current_episode_return,
                        terminated_ids)
        self.update_avg(self.avg_log_buffer,
                        self.current_episode_log,
                        terminated_ids)
        self.avg_length_buffer.extend(
            self.current_episode_length[terminated_ids].cpu().numpy().tolist())
        self.current_episode_length[terminated_ids] = 0
        if (len(self.avg_length_buffer) > 0):
            self.calculate_reward_avg()
            self.mean_logs = {
                "Episode/Log/"+name:
                mean(self.avg_log_buffer[name]) / self.max_episode_length_s
                for name in self.current_episode_log.keys()}

    def update_avg(self, avg_buffer, current_buffer, terminated_ids):
        for name in current_buffer.keys():
            avg_buffer[name].extend(
                current_buffer[name][terminated_ids].cpu().numpy().tolist())
            current_buffer[name][terminated_ids] = 0.

    def calculate_reward_avg(self):
        self.mean_episode_length = mean(self.avg_length_buffer)
        self.mean_rewards = {
            "Episode/"+name:
                mean(self.avg_return_buffer[name]) / self.max_episode_length_s
            for name in self.current_episode_return.keys()}
        self.total_mean_reward = sum(list(self.mean_rewards.values()))

    def print_to_terminal(self):
        width = 80
        pad = 35
        str = f" \033[1m Learning iteration {self.it}/{self.tot_iter} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {
                self.log['Perf/total_fps']:.0f} steps/s (collection: {
                self.log['Perf/collection_time']:.3f}s, learning {
                self.log['Perf/learning_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {
                self.log['Loss/value_function']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {
                self.log['Loss/surrogate']:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {
                self.log['Policy/mean_noise_std']:.2f}\n"""
            f"""{'Mean episode length:':>{pad}} {
                self.log['Train/mean_episode_length']:.2f}\n"""
            f"""{'Mean reward:':>{pad}} {self.total_mean_reward:.2f}\n""")
        log_string += (f"""{'-' * width}\n""")
        for key, value in self.mean_rewards.items():
            log_string += f"""{
                f'Mean episode {key[8:]}:':>{pad}} {value:.4f}\n"""
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {
                self.log['Train/total_timesteps']}\n"""
            f"""{'Iteration time:':>{pad}} {
                self.log['Train/iteration_time']:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.log['Train/time']:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.log['Train/time'] / (self.it + 1) * (
                self.learning_iter - self.it):.1f}s\n""")

        print(log_string)

    def configure_local_files(self, save_paths):
        # create ignore patterns dynamically based on include patterns
        def create_ignored_pattern_except(*patterns):
            def _ignore_patterns(path, names):
                keep = set(name for pattern in patterns for name in
                           fnmatch.filter(names, pattern))
                ignore = set(name for name in names if name not in keep and
                             not os.path.isdir(os.path.join(path, name)))
                return ignore
            return _ignore_patterns

        def remove_empty_folders(path, removeRoot=True):
            if not os.path.isdir(path):
                return
            # remove empty subfolders
            files = os.listdir(path)
            if len(files):
                for f in files:
                    fullpath = os.path.join(path, f)
                    if os.path.isdir(fullpath):
                        remove_empty_folders(fullpath)
            # if folder empty, delete it
            files = os.listdir(path)
            if len(files) == 0 and removeRoot:
                os.rmdir(path)

        # copy the relevant source files to the local logs for records
        save_dir = self.log_dir+'/files/'
        for save_path in save_paths:
            if save_path['type'] == 'file':
                os.makedirs(save_dir+save_path['target_dir'],
                            exist_ok=True)
                shutil.copy2(save_path['source_file'],
                             save_dir+save_path['target_dir'])
            elif save_path['type'] == 'dir':
                shutil.copytree(
                    save_path['source_dir'],
                    save_dir+save_path['target_dir'],
                    ignore=create_ignored_pattern_except(
                        *save_path['include_patterns']))
            else:
                print('WARNING: uncaught save path type:', save_path['type'])
        remove_empty_folders(save_dir)

    def get_runner_logs(self, runner):
        fps = int(runner.num_steps_per_env * runner.env.num_envs
                  / (runner.collection_time+runner.learn_time))
        mean_noise_std = runner.alg.actor_critic.std.mean().item()
        runner_logs = {
            'Loss/value_function': runner.mean_value_loss,
            'Loss/surrogate': runner.mean_surrogate_loss,
            'Loss/learning_rate': runner.alg.learning_rate,
            'Policy/mean_noise_std': mean_noise_std,
            'Perf/total_fps': fps,
            'Perf/collection_time': runner.collection_time,
            'Perf/learning_time': runner.learn_time,
            'Train/total_timesteps': runner.tot_timesteps,
            'Train/iteration_time': runner.collection_time+runner.learn_time,
            'Train/time': runner.tot_time,
        }
        return runner_logs
