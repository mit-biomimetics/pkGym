import os

from gym.envs import __init__
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface

# torch needs to be imported after isaacgym imports in local source
import torch

def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.env.episode_length_s = 9999
    task_registry.make_gym_and_sim()
    env, env_cfg = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    env.cfg.init_state.reset_mode = "reset_to_basic"
    task_registry.prepare_sim()
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env,
                                                      name=args.task,
                                                      args=args,
                                                      train_cfg=train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name, 'exported')
        runner.export(path)
    return env, runner, train_cfg


def play(env, runner, train_cfg):

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")
    if COMMANDS_INTERFACE:
        # interface = GamepadInterface(env)
        interface = KeyboardInterface(env)
    for i in range(10*int(env.max_episode_length)):
        actions = runner.get_inference_actions()
        if COMMANDS_INTERFACE:
            interface.update(env)
        env.set_states(train_cfg.policy.actions, actions)
        env.step()


if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)
