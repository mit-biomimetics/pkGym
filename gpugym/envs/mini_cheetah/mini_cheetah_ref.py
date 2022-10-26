from gpugym import LEGGED_GYM_ROOT_DIR

from time import time
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from gpugym.utils.math import *
from gpugym.envs import LeggedRobot, MiniCheetah

import pandas as pd
import numpy as np

class MiniCheetahRef(MiniCheetah):

    def _custom_init(self, cfg):
        # * init buffer for phase variable
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)
        
        self.omega = 2*torch.pi*cfg.control.gait_freq
        
        self.obs_scales.dof_pos = torch.tile(to_torch(self.obs_scales.dof_pos,
                                                      device=self.device),
                                             (4,))
        if cfg.env.num_se_obs:
            self.num_se_obs = cfg.env.num_se_obs
            self.se_obs_buf = torch.zeros(self.num_envs, cfg.env.num_se_obs,
                                          device=self.device,
                                          dtype=torch.float)
        self.se_noise_scale_vec = self._get_se_noise_scale_vec(self.cfg)

        # * reference traj
        csv_path = self.cfg.init_state.ref_traj.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        # todo check that this works out
        self.leg_ref = to_torch(pd.read_csv(csv_path).to_numpy(),
                                device=self.device)


    def _custom_reset(self, env_ids):
        self.action_avg[env_ids] = 0.


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and
         observations, phase-dynamics.
            Default behaviour: Compute ang vel command based on target and
             heading, compute measured terrain heights and randomly push robots
        """
        self.phase = torch.fmod(self.phase+self.dt*self.omega, 2*torch.pi)

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to
             a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs,
            even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller

        if self.cfg.control.exp_avg_decay:
            self.action_avg = exp_avg_filter(self.actions, self.action_avg,
                                            self.cfg.control.exp_avg_decay)
            actions = self.action_avg

        if self.cfg.control.control_type=="P":

            torques = self.p_gains*(actions * self.cfg.control.action_scale \
                                    + self.default_dof_pos \
                                    # + self.get_ref() \
                                    - self.dof_pos) \
                    - self.d_gains*self.dof_vel

        elif self.cfg.control.control_type=="T":
            torques = actions * self.cfg.control.action_scale

        elif self.cfg.control.control_type=="Td":
            torques = actions * self.cfg.control.action_scale \
                        - self.d_gains*self.dof_vel

        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def get_se_observations(self):
        return self.se_obs_buf


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
        dof_pos = (self.dof_pos-self.default_dof_pos) \
                   * self.obs_scales.dof_pos

        # * update commanded action history buffer

        nact = self.num_actions
        self.ctrl_hist[:, 2*nact:] = self.ctrl_hist[:, nact:2*nact]
        self.ctrl_hist[:, nact:2*nact] = self.ctrl_hist[:, :nact]
        self.ctrl_hist[:, :nact] = self.action_avg

        self.obs_buf = torch.cat((self.base_ang_vel*self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3]*self.commands_scale,
                                  dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel,
                                  self.ctrl_hist,
                                  torch.cos(self.phase),
                                  torch.sin(self.phase)),
                                  dim=-1)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.base_ang_vel*self.obs_scales.ang_vel,
                                                self.projected_gravity,
                                                self.commands[:, :3]*self.commands_scale,
                                                dof_pos,
                                                self.dof_vel*self.obs_scales.dof_vel,
                                                self.ctrl_hist,
                                                torch.cos(self.phase),
                                                torch.sin(self.phase)),
                                                dim=-1)

        # * SE observations
        self.se_obs_buf = torch.cat((self.base_ang_vel*self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel),
                                  dim=-1)

        # ! noise_scale_vec must be of correct order! Check def below
        # * add noise if needed
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                            * self.noise_scale_vec
            self.se_obs_buf += (2*torch.rand_like(self.se_obs_buf) - 1) \
                                * self.se_noise_scale_vec

        if self.cfg.env.learn_SE:
            # * store (privileged) targets for learning the state-estimator
            # We are piggy-backing this into the `extras`, which is originally
            # used just to send back and forth information for rsl_rl to print
            # statistics (e.g. average return of each reward function).
            self.extras["SE_targets"] = torch.cat((base_z,
                                self.base_lin_vel * self.obs_scales.lin_vel),
                                dim=-1)
            # self.extras["SE_targets"] = torch.cat((base_z,
            #                     self.base_lin_vel * self.obs_scales.lin_vel,
            #                     # self.contact_forces[:,self.feet_indices,:].view(-1, 12)),
            #                     self.contact_forces[:,self.feet_indices,2].view(-1, 4)),  # grf-z only
            #                     dim=-1)

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
        noise_vec[0:3] = to_torch(noise_scales.ang_vel, device=self.device) \
                         * ns_lvl*self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity*ns_lvl
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:21] = noise_scales.dof_pos*ns_lvl \
            *self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel*ns_lvl*self.obs_scales.dof_vel

        if self.cfg.terrain.measure_heights:
            noise_vec[66:187] = noise_scales.height_measurements*ns_lvl \
                                * self.obs_scales.height_measurements
        return noise_vec

    def _get_se_noise_scale_vec(self, cfg):
        '''
        Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
            Default is set to the same as _get_noise_scale_vec
        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        '''
        noise_vec = torch.zeros_like(self.se_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        ns_lvl = self.cfg.noise.noise_level
        noise_vec[0:3] = to_torch(noise_scales.ang_vel, device=self.device) \
                         * ns_lvl*self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity*ns_lvl
        noise_vec[6:18] = noise_scales.dof_pos*ns_lvl*self.obs_scales.dof_pos
        noise_vec[18:30] = noise_scales.dof_vel*ns_lvl*self.obs_scales.dof_vel
        return noise_vec

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum_x, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum_x)
            # also increase heading if it is good
            if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]:
                yaw_cmd = self.command_ranges["yaw_vel"]
                self.command_ranges["yaw_vel"] = np.clip(yaw_cmd + 0.15,
                                        0.,
                                        self.cfg.commands.max_curriculum_ang)


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(-self.command_ranges["lin_vel_y"],
                                                     self.command_ranges["lin_vel_y"],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"],
                                                         self.command_ranges["heading"],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(-self.command_ranges["yaw_vel"],
                                                         self.command_ranges["yaw_vel"],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

        # * with 10% chance, reset to 0 commands
            self.commands[env_ids, :3] *= (torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1) < 0.9).unsqueeze(1)
        # * set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

    
    def switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(-torch.square(torch.max(torch.zeros_like(c_vel), c_vel-0.1))/0.1)


    def _reward_swing_grf(self):
        # Reward non-zero grf during swing (0 to pi)
        grf = torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        ph_off = torch.lt(self.phase, torch.pi)  # should this be in swing?
        rew = grf*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1).int()
        return torch.sum(rew, dim=1)*(1-self.switch())
    
    def _reward_stance_grf(self):
        # Reward non-zero grf during stance (pi to 2pi)
        grf = torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        ph_off = torch.gt(self.phase, torch.pi)  # should this be in swing?
        rew = grf*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1).int()

        return torch.sum(rew, dim=1)*(1-self.switch())

    def _reward_reference_traj(self):
        # REWARDS EACH LEG INDIVIDUALLY BASED ON ITS POSITION IN THE CYCLE
        # dof position error
        error = self.get_ref() + self.default_dof_pos - self.dof_pos
        error *= self.obs_scales.dof_pos
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
