from gym.utils.logging_and_saving import local_code_save_helper
from gym.utils.logging_and_saving import wandb_singleton
from gym.utils import get_args

class TestTaskRegistry:
    def test_naming_sync(self, task_registry, env_cfg, train_cfg, args):
        args = get_args()
        env_cfg, train_cfg = task_registry.create_cfgs(args)
        wandb_helper = wandb_singleton.WandbSingleton()
        task_registry.set_log_dir_name(train_cfg)
        wandb_helper.setup_wandb(env_cfg=env_cfg,
                                 train_cfg=train_cfg,
                                 args=args)
        assert wandb_helper.experiment_name == train_cfg.log_dir.split('/')[-1]
