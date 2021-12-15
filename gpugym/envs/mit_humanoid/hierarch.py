from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict

from torch._C import _InferenceMode
from gpugym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gpugym.envs import MIT_Humanoid
from .hierarch_config import HierarchCfg
from gpugym.utils import get_args, task_registry  # todo pull this out

# Base class for an upper level controller
class Hierarch():

    def __init__(self, cfg: HierarchCfg, env_B, policy_B):
    #, sim_params, physics_engine,
     #            sim_device, headless)
        """
        Parses the provided config file, set things up.
        """
        self.cfg = cfg
        self.env_B = env_B
        # self.sim_params = sim_params
        # self.height_samples = None
        # self.debug_viz = False  # ! What is this?
        self._parse_cfg(self.cfg)  # TODO define this

        # * ***** Pulled from base_task
        self.num_envs = self.env_B.num_envs  # cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        self.device = env_B.device
        # * *****

        # * load in or evaluate from trajectory generated
        # self.max_time_on_B = 1
        # self.max_episode_length_s = self.cfg.env.episode_length_s
        # self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # * initialize an mit_humanoid env
        # * load in instead
        # self.env_B, self.envB_cfg = task_registry.make_env(name='mit_humanoid',
        #                                                    args=args,
        #                                                    env_cfg=env_cfg)
        # self.env_B.__init__(self, self.cfg, sim_params, physics_engine,
                            #  sim_device, headless)
        # ! self.env_B = env_B
        # * initialize a tracking PPO policy
        # * load in instead
        # ppo_runner_B, _ = task_registry.make_alg_runner(env=self.env_B,
        #                                                 name=args.task,
        #                                                 args=args,
        #                                                 train_cfg=train_cfg)
        # self.policy_B = ppo_runner_B.get_inference_policy(device=self.device)
        self.policy_B = policy_B
        # * initialize last_action, toggle for new_traj
        self._init_buffers()
        
        # * for now no privilege
        # TODO put this into cfg, and implement
        self.num_privileged_obs = None

    def _init_buffers(self):

        self.last_actions = torch.zeros(self.num_envs, self.num_actions,
                                        dtype=torch.float, device=self.device,
                                        requires_grad=False)

        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                        dtype=torch.float, device=self.device,
                                        requires_grad=False)

        # self.new_traj = torch.bool(self.num_envs, device=self.device)
        self.new_traj = torch.zeros(self.num_envs, dtype=torch.bool,
                                    device=self.device,
                                    requires_grad=False)
        self.env_time = torch.ones(self.num_envs,
                                   device=self.device,
                                   dtype=torch.long,
                                   requires_grad=False)
        self.time_on_B = torch.ones(self.num_envs, 1,
                                   device=self.device,
                                   dtype=torch.float,
                                   requires_grad=False)
        self.rew_buf = torch.zeros(self.env_B.num_envs,
                                   device=self.device,
                                   dtype=torch.float,
                                   requires_grad=False)
        # torch.zeros(self.num_envs, device=self.device,
                                #    dtype=torch.float)
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None


        # * ***** from base_task
        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Reimplementing
        """

        # * if new_traj =True
        # * clip action, and adjust.
        clip_actions = self.cfg.normalization.clip_actions
        self.actions[self.new_traj] = torch.clip(actions[self.new_traj],
                                                 -clip_actions,
                                                 clip_actions).to(self.device)
        # * save action to buffer
        self.last_actions[self.new_traj] = self.actions[self.new_traj]
        # * toggle new_traj to false
        self.new_traj[:] = False  # can just set all to false, yeah?

        # * get observations from env_B
        obs_B = self.env_B.get_observations()
        # * evaluate policy B

        # ! do whatever you need to the observations of env_B
        actions_B = self.policy_B(obs_B.detach())
        # ! do whatever you need to the actions of env_B

        # for a test, let's just replace them with actions_A
        actions_B = self.actions[:self.num_envs, :self.env_B.num_actions]

        # * step all sub_environments
        self.env_B.step(actions_B)
        # * get observation
        # obs_B = self.env_B.get_observations()
        # * save into buffer
        # TODO
        # * check if finished tracking time
        self.time_on_B += 1 # step all of them one timestep
        finished_on_B = self.time_on_B > self.dt
        terminated = self.env_B.reset_buf
        reset_ids = finished_on_B.reshape(-1) | terminated.reshape(-1)
        # * Keep a counter for each env.
            # * if not: new_traj = False
            # * if yes: new_traj = True
        self.new_traj[reset_ids] = True
        self.time_on_B[reset_ids] = 1.  # reset
        # * check if env_B terminated: # ! this happens automatically...
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # reset_buf is only reset during post_physics, so we can use this to check if the env was reset this time-step
        # * if yes, new_traj = True

        # * Compute rewards
        # self.compute_reward()  # just pick up reward from env_B
        self.rew_buf[reset_ids] = 0.  # ! check, this is a bit sloppy
        self.rew_buf += self.env_B.rew_buf  # ! need to account for discounting
        # * compute observations
        self.compute_observations()
        # * compute privileged_obs_buf
        # * return reset buf
        self.reset_buf = reset_ids
        # ? this is actually not necessary? sends to alg through extras...
        self.time_out_buf = self.env_B.time_out_buf 
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # * only update policy if we've terminated B (new update on A)
        # if self.cfg.env.toggle_updates: # TODO
            # self.extras["update_flag"] = self.new_traj

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, \
            self.reset_buf, self.extras

    def reset(self):
        # TODO: what do we reset here?
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        # self.env_B.reset()  # this takes a step with 0 action, to get all obs
        return obs, privileged_obs  # priv = None

    def compute_observations(self):
        time = self.time_on_B # needs to be of shape [num_env, 1]
        base_pos = self.env_B.root_states[:, :7]
        # TODO scaling
        base_lin_vel = self.env_B.base_lin_vel
        base_ang_vel = self.env_B.base_ang_vel
        robot_configs = self.env_B.dof_pos
        robot_joint_vels = self.env_B.dof_vel
        # TODO: consider grabbing this from env_B.obs_buf
        # ... but then can't change that as freely...
        self.obs_buf = torch.cat((time,
                                  base_pos,
                                  base_lin_vel,
                                  base_ang_vel,
                                  robot_configs,
                                  robot_joint_vels),
                                  dim=-1)


    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.env_B.dt
        # self.obs_scales = self.cfg.normalization.obs_scales
        # self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
        #     self.cfg.terrain.curriculum = False
        self.max_steps_on_B = np.ceil(self.dt / self.env_B.dt)
        self.max_episode_length_s = self.cfg.env.episode_length_s  # ? what should this be?
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        return 0.

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf