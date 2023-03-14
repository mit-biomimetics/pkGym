import isaacgym
import torch
import pytest
from gym.envs import *
from gym.utils import get_args
from gym.utils import task_registry as _task_registry

environment_dict = {}
environment_list = []


def pytest_sessionstart(session):
    args = get_args()
    args.headless = True
    _task_registry.make_gym()
    for env_name in _task_registry.task_classes.keys():
        print(env_name)
        args.task = env_name  # todo adjust make_sim args input
        env_cfg, train_cfg = _task_registry.create_cfgs(args)
        _task_registry.update_sim_cfg(args)
        _task_registry.make_sim()
        env, _ = _task_registry.make_env(name=env_name,
                                         env_cfg=env_cfg)
        environment_dict[env_name] = env
        environment_list.append(environment_dict[env_name])
        _task_registry._gym.destroy_sim(_task_registry._sim)
    pass


@pytest.fixture
def task_registry():
    return _task_registry


@pytest.fixture
def env_cfg():
    return env_cfg


@pytest.fixture
def train_cfg():
    return train_cfg


@pytest.fixture
def args():
    return args


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