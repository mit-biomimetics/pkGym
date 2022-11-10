from gym import LEGGED_GYM_ROOT_DIR

from time import time
import os

from isaacgym import gymtorch, gymapi, gymutil
import torch

from isaacgym.torch_utils import torch_rand_float, to_torch
from gym.utils.math import exp_avg_filter

from gym.envs import MiniCheetah

import pandas as pd

class MiniCheetahRef(MiniCheetah):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # * reference traj
        csv_path = cfg.init_state.ref_traj.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.leg_ref = to_torch(pd.read_csv(csv_path).to_numpy(),
                                device=sim_device)
        self.omega = 2*torch.pi*cfg.control.gait_freq
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


    def _init_buffers(self):
        super()._init_buffers()
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)
        self.phase_obs = torch.zeros(self.num_envs, 2, dtype=torch.float,
                                     device=self.device, requires_grad=False)
        self.dof_pos_obs = torch.zeros_like(self.dof_pos, requires_grad=False)

        self.base_height = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)


    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        self.action_avg[env_ids] = 0.
        self.phase[env_ids] = torch_rand_float(0, torch.pi,
                                               shape=self.phase[env_ids].shape,
                                               device=self.device)


    def post_physics_step(self):
        """ Callback called before computing terminations, rewards, and
         observations, phase-dynamics.
            Default behaviour: Compute ang vel command based on target and
             heading, compute measured terrain heights and randomly push robots
        """
        super().post_physics_step()

        self.phase = torch.fmod(self.phase+self.dt*self.omega, 2*torch.pi)
        self.base_height = self.root_states[:, 2].unsqueeze(1)

        nact = self.num_actions
        self.ctrl_hist[:, 2*nact:] = self.ctrl_hist[:, nact:2*nact]
        self.ctrl_hist[:, nact:2*nact] = self.ctrl_hist[:, :nact]
        self.ctrl_hist[:, :nact] = self.actions

        # ? unsqueeze
        self.phase_obs = torch.cat([torch.cos(self.phase),
                                    torch.sin(self.phase)], dim=-1)

        self.dof_pos_obs = (self.dof_pos - self.default_dof_pos) \
                            * self.scales["dof_pos"]


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        super()._resample_commands(env_ids)
        # * with 10% chance, reset to 0 commands
        self.commands[env_ids, :3] *= (torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1) < 0.9).unsqueeze(1)


    def switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(-torch.square(torch.max(torch.zeros_like(c_vel),
                                                 c_vel-0.1))/0.1)


    def _reward_swing_grf(self):
        # Reward non-zero grf during swing (0 to pi)
        in_contact = torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        ph_off = torch.lt(self.phase, torch.pi)
        rew = in_contact*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return -torch.sum(rew.float(), dim=1)*(1-self.switch())


    def _reward_stance_grf(self):
        # Reward non-zero grf during stance (pi to 2pi)
        in_contact = torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        ph_off = torch.gt(self.phase, torch.pi)  # should this be in swing?
        rew = in_contact*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)

        return torch.sum(rew.float(), dim=1)*(1-self.switch())


    def _reward_reference_traj(self):
        # REWARDS EACH LEG INDIVIDUALLY BASED ON ITS POSITION IN THE CYCLE
        # dof position error
        error = self.get_ref() + self.default_dof_pos - self.dof_pos
        reward = torch.sum(self.sqrdexp(error) - torch.abs(error)*0.2, dim=1)/12.  # normalize by n_dof
        # * only when commanded velocity is higher
        return reward*(1-self.switch())


    def get_ref(self):
        leg_frame = torch.zeros_like(self.torques)
        # offset by half cycle (trot)
        ph_off = torch.fmod(self.phase+torch.pi, 2*torch.pi)
        phd_idx = (torch.round(self.phase* \
                            (self.leg_ref.size(dim=0)/(2*torch.pi)-1))).long()
        pho_idx = (torch.round(ph_off* \
                            (self.leg_ref.size(dim=0)/(2*torch.pi)-1))).long()
        leg_frame[:, 0:3] += self.leg_ref[phd_idx.squeeze(), :]
        leg_frame[:, 3:6] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 6:9] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 9:12] += self.leg_ref[phd_idx.squeeze(), :]
        return leg_frame


    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(self.sqrdexp((self.dof_pos - self.default_dof_pos)/torch.pi*36), dim=1)
        rew_vel = torch.mean(self.sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel+rew_pos-rew_base_vel)*self.switch()


    def _reward_tracking_lin_vel(self):
        # Tracking linear velocity commands (xy axes)
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)*(1-self.switch())
