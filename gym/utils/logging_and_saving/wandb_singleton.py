import os
import json
import wandb
from gym import LEGGED_GYM_ROOT_DIR


class WandbSingleton(object):
    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(WandbSingleton, self).__new__(self)
            self.entity_name = None
            self.project_name = None
            self.experiment_name = ''
            self.enabled = False

        return self.instance

    def set_wandb_values(self, args, train_cfg=None):
        # build the path for the wandb_config.json file
        wandb_config_path = os.path.join(
                LEGGED_GYM_ROOT_DIR, 'user', 'wandb_config.json')

        # load entity and project name from JSON if it exists
        if os.path.exists(wandb_config_path):
            json_data = json.load(open(wandb_config_path))
            print('Loaded WandB entity and project from JSON.')
            self.entity_name = json_data['entity']
            self.project_name = json_data['project']
        # override entity and project by commandline args if provided
        if args.wandb_entity is not None:
            print('Received WandB entity from arguments.')
            self.entity_name = args.wandb_entity
        if args.wandb_project is not None:
            print('Received WandB project from arguments.')
            self.project_name = args.wandb_project
        # assume WandB is off if entity or project is None and short-circuit
        if args.task is not None:
            self.experiment_name = f'{args.task}'

        if (self.entity_name is None or self.project_name is None
           or args.disable_wandb):
            self.enabled = False
        else:
            print(f'Setting WandB project name: {self.project_name}\n' +
                  f'Setting WandB entitiy name: {self.entity_name}\n')
            self.enabled = True

    def is_wandb_enabled(self):
        return self.enabled

    def get_entity_name(self):
        return self.entity_name

    def get_project_name(self):
        return self.project_name

    def setup_wandb(self, policy_runner, train_cfg, args, is_sweep=False):
        self.set_wandb_values(args, train_cfg)

        # short-circuit if the values say WandB is turned off
        if not self.is_wandb_enabled():
            print('WARNING: WandB is disabled and will not save or log.')
            return

        wandb.config = {}

        if is_sweep:
            wandb.init(dir=os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'),
                       config=wandb.config,
                       name=self.experiment_name)
        else:
            wandb.init(project=self.project_name,
                       entity=self.entity_name,
                       dir=os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'),
                       config=wandb.config,
                       name=self.experiment_name)

        wandb.run.log_code(
            os.path.join(LEGGED_GYM_ROOT_DIR, 'gym'))

        policy_runner.attach_to_wandb(wandb)

    def close_wandb(self):
        if self.enabled:
            wandb.finish()