import isaacgym
import torch
import pytest
from gym.envs import *
from gym.utils import get_args, task_registry

environment_dict = {}
environment_list = []


def pytest_sessionstart(session):
    args = get_args()
    args.headless = True
    task_registry.make_gym()
    for env_name in task_registry.task_classes.keys():
        print(env_name)
        args.task = env_name  # todo adjust make_sim args input
        env_cfg, train_cfg = task_registry.create_cfgs(args)
        task_registry.update_sim_cfg(args)
        task_registry.make_sim()
        env, _ = task_registry.make_env(name=env_name,
                                        env_cfg=env_cfg)
        environment_dict[env_name] = env
        environment_list.append(environment_dict[env_name])
        task_registry._gym.destroy_sim(task_registry._sim)
    pass


@pytest.fixture
def env_dict():
    return environment_dict


@pytest.fixture
def env_list():
    return environment_list


@pytest.fixture
def mini_cheetah():
    return environment_dict['mini_cheetah']


@pytest.fixture
def humanoid():
    return environment_dict['humanoid']


@pytest.fixture
def cartpole():
    return environment_dict['cartpole']


@pytest.fixture
def flat_anymal_c():
    return environment_dict['flat_anymal_c']


@pytest.fixture
def a1():
    return environment_dict['a1']


@pytest.fixture
def humanoid_running():
    return environment_dict['humanoid_running']