import importlib

from gym.utils.task_registry import task_registry

# * To add a new env:
# * 1. add the base env and env class name and location to the class dict
# * 2. add the config name and location to the config dict
# * 3. add the runner confg name and location to the runner config dict
# * 3. register the task experiment name to the env/config/ppo classes

# * from y import x where {y:x}
class_dict = {
    'LeggedRobot': '.base.legged_robot',
    'FixedRobot': '.base.fixed_robot',
    'Cartpole': '.cartpole.cartpole',
    'MiniCheetah': '.mini_cheetah.mini_cheetah',
    'MiniCheetahRef': '.mini_cheetah.mini_cheetah_ref',
    'MIT_Humanoid': '.mit_humanoid.mit_humanoid',
    "Anymal": '.anymal_c.anymal',
    'A1': '.a1.a1',
    'HumanoidRunning': '.mit_humanoid.humanoid_running',
}

config_dict = {
    'CartpoleCfg': '.cartpole.cartpole_config',
    'MiniCheetahCfg': '.mini_cheetah.mini_cheetah_config',
    'MiniCheetahRefCfg': '.mini_cheetah.mini_cheetah_ref_config',
    'MITHumanoidCfg': '.mit_humanoid.mit_humanoid_config',
    'A1Cfg': '.a1.a1_config',
    'AnymalCFlatCfg': '.anymal_c.flat.anymal_c_flat_config',
    'HumanoidRunningCfg': '.mit_humanoid.humanoid_running_config',
}

runner_config_dict = {
    'CartpoleRunnerCfg': '.cartpole.cartpole_config',
    'MiniCheetahRunnerCfg': '.mini_cheetah.mini_cheetah_config',
    'MiniCheetahRefRunnerCfg': '.mini_cheetah.mini_cheetah_ref_config',
    'MITHumanoidRunnerCfg': '.mit_humanoid.mit_humanoid_config',
    'A1RunnerCfg': '.a1.a1_config',
    'AnymalCFlatRunnerCfg': '.anymal_c.flat.anymal_c_flat_config',
    'HumanoidRunningRunnerCfg': '.mit_humanoid.humanoid_running_config',
}

task_dict = {
    'cartpole': ['Cartpole', 'CartpoleCfg', 'CartpoleRunnerCfg'],
    'mini_cheetah': ['MiniCheetah', 'MiniCheetahCfg', 'MiniCheetahRunnerCfg'],
    'mini_cheetah_ref':
        ['MiniCheetahRef', 'MiniCheetahRefCfg', 'MiniCheetahRefRunnerCfg'],
    'humanoid': ['MIT_Humanoid', 'MITHumanoidCfg', 'MITHumanoidRunnerCfg'],
    'humanoid_running': ['HumanoidRunning',
                         'HumanoidRunningCfg',
                         'HumanoidRunningRunnerCfg'],
    'a1': ['A1', 'A1Cfg', 'A1RunnerCfg'],
    'flat_anymal_c': ['Anymal', 'AnymalCFlatCfg', 'AnymalCFlatRunnerCfg'],
}

for class_name, class_location in class_dict.items():
    locals()[class_name] = getattr(
        importlib.import_module(class_location, __name__), class_name)
for config_name, config_location in config_dict.items():
    locals()[config_name] = getattr(
        importlib.import_module(config_location, __name__), config_name)
for runner_config_name, runner_config_location in runner_config_dict.items():
    locals()[runner_config_name] = getattr(
        importlib.import_module(
            runner_config_location, __name__),
        runner_config_name)

for task_name, class_list in task_dict.items():
    task_registry.register(
        task_name,
        locals()[class_list[0]],
        locals()[class_list[1]](),
        locals()[class_list[2]]())
