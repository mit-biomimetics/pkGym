import wandb

from .EpisodicLogs import EpisodicLogs
from .PerIterationLogs import PerIterationLogs
from .TimeKeeper import TimeKeeper


class Logger:
    def __init__(self):
        self.initialized = False

    def initialize(self,
                   num_envs=1,
                   episode_dt=0.1,
                   total_iterations=100,
                   device='cpu'):

        self.device = device

        self.reward_logs = EpisodicLogs(num_envs, episode_dt, device=device)

        self.iteration_logs = PerIterationLogs()

        self.iteration_counter = 0
        self.tot_iter = total_iterations

        self.timer = TimeKeeper()
        self.num_envs = num_envs
        self.step_counter = 0
        self.algorithm_logs = {}
        self.initialized = True

    def register_category(self, category, target, attribute_list):
        self.iteration_logs.register_items(category, target, attribute_list)

    def register_rewards(self, reward_names):
        self.reward_logs.add_buffer(reward_names)

    def log_rewards(self, rewards_dict):
        self.reward_logs.add_step(rewards_dict)

    def finish_step(self, dones):
        self.reward_logs.finish_step(dones)
        self.step_counter += 1

    def finish_iteration(self):
        self.iteration_counter += 1
        if wandb.run is not None:
            self.log_to_wandb()
        return None

    def estimate_ETA(self, times=['runtime'], mode='total'):
        if mode == 'total':
            total_time = sum(self.timer.get_time(part) for part in times)
            iter_time = total_time / self.iteration_counter
        elif mode == 'iteration':
            # sum all the times in times
            iter_time = sum(self.timer.get_time(part) for part in times)
        else:
            iter_time = 0.
        return iter_time * (self.tot_iter - self.iteration_counter)

    def estimate_steps_per_second(self):
        return ((self.step_counter * self.num_envs / self.iteration_counter)
                / (self.timer.get_time('collection')))

    def print_to_terminal(self):
        width = 80
        pad = 35

        log_string = ""

        def format_log_entry(key, val, append=''):
            """Helper function to format a single log entry."""
            nonlocal log_string
            log_string += f"""{key:>{pad}}: {val:.2f} {append}\n"""

        def separator(subtitle="", marker='-'):
            nonlocal log_string
            subtitle_length = len(subtitle)
            if subtitle_length > 0:
                subtitle_length += 1  # Add 1 for the space after subtitle

            dashes_each_side = (width - subtitle_length) // 2
            log_string += "\n"
            log_string += (f"{marker * dashes_each_side} {subtitle} "
                           f"{marker * dashes_each_side}\n")

        separator(f"Iteration {self.iteration_counter}/{self.tot_iter}", '#')

        separator('Rewards')
        averages = self.reward_logs.get_average_rewards()

        for key, val in averages.items():
            format_log_entry(key, val)

        separator('Other Agent Metrics')
        format_log_entry('average episode time',
                         self.reward_logs.get_average_time())

        separator('Algorithm Performance')
        for key, val in self.iteration_logs.get_all_logs('algorithm').items():
            format_log_entry(key, val)

        separator('Timings')
        format_log_entry('steps/s', self.estimate_steps_per_second())
        tot_t = self.timer.get_time('iteration')
        col_time = self.timer.get_time('collection')
        learn_time = self.timer.get_time('learning')
        time_string = f"(sim: {col_time:.2f}" \
                      f", learn:{learn_time:.2f})"
        format_log_entry('total time', tot_t, time_string)

        format_log_entry('ETA', self.estimate_ETA(['runtime']))
        print(log_string)

    def log_category(self, category='algorithm'):
        self.iteration_logs.log(category)

    def log_to_wandb(self):
        def prepend_to_keys(section, dictionary):
            return {section + '/' + key: val
                    for key, val in dictionary.items()}

        averages = prepend_to_keys('rewards',
                                   self.reward_logs.get_average_rewards())

        algorithm_logs = prepend_to_keys(
            'algorithm', self.iteration_logs.get_all_logs('algorithm'))
        wandb.log({**averages, **algorithm_logs})

    def tic(self, category='default'):
        self.timer.tic(category)

    def toc(self, category='default'):
        self.timer.toc(category)

    def get_time(self, category='default'):
        return self.timer.get_time(category)

    def attach_torch_obj_to_wandb(self,
                                  obj_tuple,
                                  log_freq=100,
                                  log_graph=True):
        if wandb.run is None:
            return
        if type(obj_tuple) is not tuple:
            obj_tuple = (obj_tuple,)
        wandb.watch(obj_tuple, log_freq=log_freq, log_graph=log_graph)
