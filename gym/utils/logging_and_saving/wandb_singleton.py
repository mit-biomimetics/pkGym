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
        # first priority for commandline args
        if args.wandb_project is not None and args.wandb_entity is not None:
            print('Recevied WandB entity and project from arguments.')
            self.entity_name = args.wandb_entity
            self.project_name = args.wandb_project
        # second priority for JSON
        elif train_cfg is not None and \
            hasattr(train_cfg, 'wandb_settings') and \
            hasattr(train_cfg.wandb_settings, 'enable_wandb') \
                and train_cfg.wandb_settings.enable_wandb:
            json_data = json.load(open(os.path.join(
                LEGGED_GYM_ROOT_DIR, 'gym', 'user', 'wandb_config.json')))
            print('Loaded WandB entity and project from JSON.')
            self.entity_name = json_data['entity']
            self.project_name = json_data['project']
        # assume WandB is off and give a warning
        else:
            print('WARNING: WandB is disabled and will not save or log.')
            return

        if args.task is not None:
            self.experiment_name = f'{args.task}'

        print(f'Setting WandB project name: {self.project_name}\n' +
              f'Setting WandB entitiy name: {self.entity_name}\n')
        self.enabled = True

    def is_wandb_enabled(self):
        return self.enabled

    def get_entity_name(self):
        return self.entity_name

    def get_project_name(self):
        return self.project_name

    def setup_wandb(self, policy_runner, is_sweep=False):

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

        policy_runner.configure_wandb(wandb)

    def close_wandb(self):
        if self.enabled:
            wandb.finish()
