from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from gpugym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gpugym.envs import LeggedRobot

from gpugym import LEGGED_GYM_ROOT_DIR
import pandas as pd

class MiniCheetah(LeggedRobot):

    def _custom_init(self, cfg):
        # * init buffer for phase variable
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)
        
        self.num_states = 13 + 2*self.num_dof + 1
        # self.SE_targets = torch.zeros(self.num_envs,
        #                         self.cfg.env.num_se_targets,
        #                         dtype=torch.float,
        #                         device=self.device, requires_grad=False)
        self.obs_scales.dof_pos = torch.tile(to_torch(self.obs_scales.dof_pos), (4,))

        # * reference traj
        csv_path = self.cfg.init_state.ref_traj.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.leg_ref = to_torch(pd.read_csv(csv_path).to_numpy())  # ! check that this works out

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations, phase-dynamics
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.phase = torch.fmod(self.phase+self.dt, 1.)

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def compute_observations(self):
        """ Computes observations
        """

        # base_z
        # base lin vel
        # base ang vel
        # projected gravity vec
        # commands
        # joint pos
        # joint vel
        # actions
        # actions (n-1, n-2)
        # phase
        base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z
        dof_pos = (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos

        # * update commanded action history buffer
        control_type = self.cfg.control.control_type
        if control_type in ['T', 'Td']:
            ndof = self.num_dof
            self.ctrl_hist[:, 2 * ndof:] = self.ctrl_hist[:, ndof:2 * ndof]
            self.ctrl_hist[:, ndof:2 * ndof] = self.ctrl_hist[:, :ndof]
            # self.ctrl_hist[:, :nact] = self.actions*self.obs_scales.action_scale
            self.ctrl_hist[:, :ndof] = self.dof_vel * self.obs_scales.dof_vel
        else:
            nact = self.num_actions
            self.ctrl_hist[:, 2 * nact:] = self.ctrl_hist[:, nact:2 * nact]
            self.ctrl_hist[:, nact:2 * nact] = self.ctrl_hist[:, :nact]
            self.ctrl_hist[:, :nact] = self.actions*self.cfg.control.action_scale + self.default_dof_pos

        # self.obs_buf = torch.cat((base_z,
        #                           self.base_lin_vel*self.obs_scales.lin_vel,
        #                           self.base_ang_vel*self.obs_scales.ang_vel,
        #                           self.projected_gravity,
        #                           self.commands[:, :3]*self.commands_scale,
        #                           dof_pos,
        #                           self.dof_vel*self.obs_scales.dof_vel,
        #                           self.actions,
        #                           self.ctrl_hist,
        #                           torch.cos(self.phase*2*torch.pi),
        #                           torch.sin(self.phase*2*torch.pi)),
        #                          dim=-1)

        self.obs_buf = torch.cat((self.base_ang_vel*self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3]*self.commands_scale,
                                  dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel,
                                  self.ctrl_hist,
                                  torch.cos(self.phase*2*torch.pi),
                                  torch.sin(self.phase*2*torch.pi)),
                                 dim=-1)

        # ! noise_scale_vec must be of correct order! Check def below
        # * add noise if needed
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                            * self.noise_scale_vec

        if self.cfg.env.num_se_targets:
            self.extras["SE_targets"] = torch.cat((base_z,
                                  self.base_lin_vel*self.obs_scales.lin_vel),
                                  dim=-1)

    def _get_noise_scale_vec(self, cfg):
        '''
        Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        '''
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        ns_lvl = self.cfg.noise.noise_level
        # noise_vec[1:4] = noise_scales.lin_vel*ns_lvl*self.obs_scales.lin_vel
        noise_vec[0:3] = to_torch(noise_scales.ang_vel)*ns_lvl*self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity*ns_lvl
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:21] = noise_scales.dof_pos*ns_lvl*self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel*ns_lvl*self.obs_scales.dof_vel

        if self.cfg.terrain.measure_heights:
            noise_vec[66:187] = noise_scales.height_measurements*ns_lvl \
                                * self.obs_scales.height_measurements
        return noise_vec


    def update_X0(self, X0, from_obs=False):
        """
        Needs to match any scaling or other changes made in observations
        """

        if not from_obs:
            self.X0_conds = X0
            return
        # * otherwise, unscale and handle obs to get back to states

        n_new = min(X0.shape[0], self.X0_conds.shape[0])
        base_z = X0[:n_new, 0:1]
        base_lin_vel = X0[:n_new, 1:4]/self.obs_scales.lin_vel
        base_ang_vel = X0[:n_new, 4:7]/self.obs_scales.ang_vel
        prj_grv = X0[:n_new, 7:10]  # ! I think? Check...
        dof_pos = X0[:n_new, 10:22]
        dof_vel = X0[:n_new, 34:46]/self.obs_scales.dof_vel
        phase = torch.atan2(X0[:n_new, 46:47], X0[:n_new, 47:48])

        # from prj_grv to roll and pitch
        pitch = torch.atan2(-prj_grv[:, 1], prj_grv[:, 2])
        roll = torch.atan2(prj_grv[:, 0], prj_grv[:, 2]/torch.cos(pitch))

        base_quat = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
        base_pos = torch.cat((torch.zeros_like(base_z),
                            torch.zeros_like(base_z),
                            base_z), dim=1)
        self.X0_conds[:n_new, :3] = base_pos
        self.X0_conds[:n_new, 3:7] = base_quat
        self.X0_conds[:n_new, 7:10] = base_lin_vel
        self.X0_conds[:n_new, 10:13] = base_ang_vel
        self.X0_conds[:n_new, 13:25] = dof_pos
        self.X0_conds[:n_new, 25:37] = dof_vel
        self.X0_conds[:n_new, 37:38] = phase

    
    def reset_to_storage(self, env_ids):
        # also reset phase
        # # * with replacement
        # # more general because it allows buffer to be less than envs
        idx = torch.randint(self.X0_conds.shape[0], (len(env_ids),))
        self.root_states[env_ids] = self.X0_conds[idx, :13]
        # self.dof_pos[env_ids] = torch.zeros_like(self.dof_pos[env_ids])
        self.dof_pos[env_ids] = self.X0_conds[idx, 13:13+self.num_dof]
        # self.dof_vel[env_ids] = torch.zeros_like(self.dof_vel[env_ids])
        self.dof_vel[env_ids] = self.X0_conds[idx,
                                            13+self.num_dof:13+2*self.num_dof]
        self.phase[env_ids] = self.X0_conds[idx, 37:38]  # keep it vertical (or unsqueeze)



    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity w. squared exp
        return self.sqrdexp(self.base_lin_vel[:, 2]  \
                            * self.cfg.normalization.obs_scales.lin_vel)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        error = self.sqrdexp(self.base_ang_vel[:, :2] \
                             * self.cfg.normalization.obs_scales.ang_vel)
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        error = torch.square(self.projected_gravity[:, :2])/self.cfg.rewards.tracking_sigma
        return torch.sum(torch.exp(-error), dim=1)
        # return self.sqrdexp(self.projected_gravity[:, 2]+1.)

    def _reward_base_height(self):
        """
        Squared exponential saturating at base_height target
        """
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error *= self.obs_scales.base_z
        error = torch.clamp(error, max=0, min=None).flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(self.sqrdexp(self.dof_vel * self.cfg.normalization.obs_scales.dof_vel), dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(self.sqrdexp((self.dof_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos), dim=1)

    # def _reward_symm_legs(self):
    #     error = 0.
    #     for i in range(2, 5):
    #         error += self.sqrdexp((self.dof_pos[:, i]+self.dof_pos[:, i+9]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     for i in range(0, 2):
    #         error += self.sqrdexp((self.dof_pos[:, i]-self.dof_pos[:, i+9]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     return error

    # def _reward_symm_arms(self):
    #     error = 0.
    #     for i in range(6, 8):
    #         error += self.sqrdexp((self.dof_pos[:, i]-self.dof_pos[:, i+9]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     error += self.sqrdexp((self.dof_pos[:, 5]+self.dof_pos[:, 14]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     error += self.sqrdexp((self.dof_pos[:, 8]+self.dof_pos[:, 17]) \
    #                     / self.cfg.normalization.obs_scales.dof_pos)
    #     return error
    def _reward_reference_traj(self):
        # REWARDS EACH LEG INDIVIDUALLY BASED ON ITS POSITION IN THE CYCLE
        # dof position error
        error = self.get_ref() + self.default_dof_pos - self.dof_pos
        # print(self.get_ref() + self.default_dof_pos)
        error *= self.obs_scales.dof_pos
        reward = torch.sum(self.sqrdexp(error) - torch.abs(error)*0.2, dim=1)/12.  # normalize by n_dof
        # * only when commanded velocity is higher
        return reward*(1-self.switch())


######################### added from mini_cheetah_ref.py ####################
    def _reward_stance_grf(self):
        # Reward non-zero grf during stance (pi to 2pi)
        grf = torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        ph_off = torch.gt(self.phase, torch.pi)  # should this be in swing?
        rew = grf*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1).int()

        return torch.sum(rew, dim=1)*(1-self.switch())

    def _reward_swing_grf(self):
        # Reward non-zero grf during swing (0 to pi)
        grf = torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        ph_off = torch.lt(self.phase, torch.pi)  # should this be in swing?
        rew = grf*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1).int()
        return torch.sum(rew, dim=1)*(1-self.switch())

    def get_ref(self):
        leg_frame = torch.zeros_like(self.torques)
        # offest by half cycle (trot)
        ph_off = torch.fmod(self.phase+torch.pi, 2*torch.pi)
        phd_idx = (torch.round(self.phase* \
                            (self.leg_ref.size(dim=0)/(2*np.pi)-1))).long()
        pho_idx = (torch.round(ph_off* \
                            (self.leg_ref.size(dim=0)/(2*np.pi)-1))).long()
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

    def switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(-torch.square(torch.max(torch.zeros_like(c_vel), c_vel-0.1))/0.1)