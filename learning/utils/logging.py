import torch

class Logger:
    def __init__(self, path):
        self.path = path
        self.log = {}

    def log_to_wandb(self):
        print("empty")
    
    def add_log(self, log_dict):
        self.log.update(log_dict)

    def print_to_terminal(self):
        print("Bingo!")

