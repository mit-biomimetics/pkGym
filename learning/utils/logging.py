import pprint
import wandb


class Logger:
    def __init__(self, path):
        self.path = path
        self.log = {}
        self.it = 0
        self.tot_iter = 0
        self.learning_iter = 0

    def log_to_wandb(self):
        wandb.log(self.log)

    def add_log(self, log_dict):
        self.log.update(log_dict)

    def update_iterations(self, it, tot_iter, learning_iter):
        self.it = it
        self.tot_iter = tot_iter
        self.learning_iter = learning_iter

    def print_to_terminal(self):   
        width=80
        pad=35
        str = f" \033[1m Learning iteration {self.it}/{self.tot_iter} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {self.log['Perf/total_fps']:.0f} steps/s (collection: {self.log[
                        'Perf/collection_time']:.3f}s, learning {self.log['Perf/learning_time']:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {self.log['Loss/value_function']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {self.log['Loss/surrogate']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {self.log['Policy/mean_noise_std']:.2f}\n"""
                        f"""{'Mean reward:':>{pad}} {self.log['Train/mean_reward']:.2f}\n"""
                        f"""{'Mean episode length:':>{pad}} {self.log['Train/mean_episode_length']:.2f}\n""")

        for key, value in self.log.items():
            if "Episode/" in key:
                log_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.log['Train/total_timesteps']}\n"""
                       f"""{'Iteration time:':>{pad}} {self.log['Train/iteration_time']:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.log['Train/time']:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.log['Train/time'] / (self.it + 1) * (
                               self.learning_iter - self.it):.1f}s\n""")

        print(log_string)
