import torch

from isaacgym.torch_utils import quat_rotate_inverse
from gym.envs import LeggedRobot


class HumanoidRunning(LeggedRobot):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        self.omega = 2 * torch.pi * cfg.control.gait_freq
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        # * get the body_name to body_index dict
        body_dict = self.gym.get_actor_rigid_body_dict(
            self.envs[0], self.actor_handles[0]
        )
        # * extract a list of body_names where the index is the id number
        body_names = [
            body_tuple[0]
            for body_tuple in sorted(
                body_dict.items(), key=lambda body_tuple: body_tuple[1]
            )
        ]
        # * construct a list of id numbers corresponding to end_effectors
        self.end_effector_ids = []
        for end_effector_name in self.cfg.asset.end_effector_names:
            self.end_effector_ids.extend(
                [
                    body_names.index(body_name)
                    for body_name in body_names
                    if end_effector_name in body_name
                ]
            )

        # * end_effector_pos is world-frame and converted to env_origin
        self.end_effector_pos = self._rigid_body_pos[
            :, self.end_effector_ids
        ] - self.env_origins.unsqueeze(dim=1).expand(
            self.num_envs, len(self.end_effector_ids), 3
        )
        self.end_effector_quat = self._rigid_body_quat[:, self.end_effector_ids]

        self.end_effector_lin_vel = torch.zeros(
            self.num_envs,
            len(self.end_effector_ids),
            3,
            dtype=torch.float,
            device=self.device,
        )
        self.end_effector_ang_vel = torch.zeros(
            self.num_envs,
            len(self.end_effector_ids),
            3,
            dtype=torch.float,
            device=self.device,
        )

        # * end_effector vels are body-relative like body vels above
        for index in range(len(self.end_effector_ids)):
            self.end_effector_lin_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_lin_vel[:, self.end_effector_ids][:, index, :],
            )
            self.end_effector_ang_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_ang_vel[:, self.end_effector_ids][:, index, :],
            )

        # * separate legs and arms
        self.dof_pos_target_legs = torch.zeros(
            self.num_envs, 10, dtype=torch.float, device=self.device
        )
        self.dof_pos_target_arms = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )
        self.dof_pos_legs = torch.zeros(
            self.num_envs, 10, dtype=torch.float, device=self.device
        )
        self.dof_pos_arms = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )
        self.dof_vel_legs = torch.zeros(
            self.num_envs, 10, dtype=torch.float, device=self.device
        )
        self.dof_vel_arms = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )

        # * other
        self.base_pos = self.root_states[:, 0:3]
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.phase_obs = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.dof_pos_target[:, :10] = self.dof_pos_target_legs
        self.dof_pos_target[:, 10:] = self.dof_pos_target_arms

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        if self.cfg.commands.resampling_time == -1:
            self.commands[env_ids, :] = 0.0
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device
        )

    def _post_physx_step(self):
        """Update all states that are not handled in PhysX"""
        super()._post_physx_step()
        self.phase = (self.phase + self.dt * self.omega).fmod(2 * torch.pi)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.phase_obs = torch.cat(
            (torch.sin(self.phase), torch.cos(self.phase)), dim=1
        )

        self.in_contact = self.contact_forces[:, self.end_effector_ids, 2].gt(0.0)

        self.dof_pos_legs = self.dof_pos[:, :10]
        self.dof_pos_arms = self.dof_pos[:, 10:]
        self.dof_vel_legs = self.dof_vel[:, :10]
        self.dof_vel_arms = self.dof_vel[:, 10:]

        # * end_effector_pos is world-frame and converted to env_origin
        self.end_effector_pos = self._rigid_body_pos[
            :, self.end_effector_ids
        ] - self.env_origins.unsqueeze(dim=1).expand(
            self.num_envs, len(self.end_effector_ids), 3
        )
        self.end_effector_quat = self._rigid_body_quat[:, self.end_effector_ids]

        # * end_effector vels are body-relative like body vels above
        for index in range(len(self.end_effector_ids)):
            self.end_effector_lin_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_lin_vel[:, self.end_effector_ids][:, index, :],
            )
            self.end_effector_ang_vel[:, index, :] = quat_rotate_inverse(
                self.base_quat,
                self._rigid_body_ang_vel[:, self.end_effector_ids][:, index, :],
            )

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        select = torch.norm(self.commands[:, 0:2], dim=-1, keepdim=True) < 0.5
        self.commands[:, 0:2] = torch.where(
            select, 0.0 * self.commands[:, 0:2], self.commands[:, 0:2]
        )
        select = torch.abs(self.commands[:, 2:3]) < 0.5
        self.commands[:, 2:3] = torch.where(
            select, 0.0 * self.commands[:, 2:3], self.commands[:, 2:3]
        )

    def _check_terminations_and_timeouts(self):
        """Check if environments need to be reset"""
        super()._check_terminations_and_timeouts()

        # * Termination for velocities, orientation, and low height
        self.terminated |= (self.base_lin_vel.norm(dim=-1, keepdim=True) > 10).any(
            dim=1
        )
        self.terminated |= (self.base_ang_vel.norm(dim=-1, keepdim=True) > 5).any(dim=1)
        self.terminated |= (self.projected_gravity[:, 0:1].abs() > 0.7).any(dim=1)
        self.terminated |= (self.projected_gravity[:, 1:2].abs() > 0.7).any(dim=1)
        self.terminated |= (self.base_pos[:, 2:3] < 0.3).any(dim=1)

        self.to_be_reset = self.timed_out | self.terminated

    # ########################## REWARDS ######################## #

    # * Task rewards * #

    def _reward_tracking_lin_vel(self):
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 2.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return self._sqrdexp(error).sum(dim=1)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        ang_vel_error /= self.scales["base_ang_vel"]
        return self._sqrdexp(ang_vel_error)

    # * Shaping rewards * #

    def _reward_base_height(self):
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales["base_height"]
        error = error.flatten()
        return self._sqrdexp(error)

    def _reward_orientation(self):
        return self._sqrdexp(self.projected_gravity[:, 2] + 1)

    def _reward_joint_regularization_legs(self):
        # * Reward joint poses and symmetry
        reward = self._reward_hip_yaw_zero()
        reward += self._reward_hip_abad_symmetry()
        reward += self._reward_hip_pitch_symmetry()
        return reward / 3.0

    def _reward_hip_yaw_zero(self):
        error = self.dof_pos[:, 0] - self.default_dof_pos[:, 0]
        reward = self._sqrdexp(error / self.scales["dof_pos"][0]) / 2.0
        error = self.dof_pos[:, 5] - self.default_dof_pos[:, 5]
        reward += self._sqrdexp(error / self.scales["dof_pos"][5]) / 2.0
        return reward

    def _reward_hip_abad_symmetry(self):
        error = (
            self.dof_pos[:, 1] / self.scales["dof_pos"][1]
            - self.dof_pos[:, 6] / self.scales["dof_pos"][6]
        )
        return self._sqrdexp(error)

    def _reward_hip_pitch_symmetry(self):
        error = (
            self.dof_pos[:, 2] / self.scales["dof_pos"][2]
            + self.dof_pos[:, 7] / self.scales["dof_pos"][7]
        )
        return self._sqrdexp(error)

    def _reward_joint_regularization_arms(self):
        reward = 0
        reward += self._reward_arm_yaw_symmetry()
        reward += self._reward_arm_yaw_zero()
        reward += self._reward_arm_abad_zero()
        reward += self._reward_arm_abad_symmetry()
        reward += self._reward_arm_pitch_symmetry()
        reward += self._reward_arm_pitch_zero()
        reward += self._reward_elbow_zero()
        return reward / 6.0

    def _reward_arm_pitch_symmetry(self):
        error = (
            self.dof_pos[:, 10] / self.scales["dof_pos"][10]
            + self.dof_pos[:, 14] / self.scales["dof_pos"][14]
        )
        return self._sqrdexp(error)

    def _reward_arm_pitch_zero(self):
        error = self.dof_pos[:, 10] - self.default_dof_pos[:, 10]
        reward = self._sqrdexp(error / self.scales["dof_pos"][10])
        error = self.dof_pos[:, 14] - self.default_dof_pos[:, 14]
        reward += self._sqrdexp(error / self.scales["dof_pos"][14])
        return reward / 2.0

    def _reward_elbow_symmetry(self):
        error = (
            self.dof_pos[:, 13] / self.scales["dof_pos"][13]
            + self.dof_pos[:, 17] / self.scales["dof_pos"][17]
        )
        return self._sqrdexp(error)

    def _reward_elbow_zero(self):
        error = self.dof_pos[:, 13] - self.default_dof_pos[:, 13]
        reward = self._sqrdexp(error / self.scales["dof_pos"][13])
        error = self.dof_pos[:, 17] - self.default_dof_pos[:, 17]
        reward += self._sqrdexp(error / self.scales["dof_pos"][17])
        return reward / 2.0

    def _reward_arm_yaw_symmetry(self):
        error = (
            self.dof_pos[:, 12] / self.scales["dof_pos"][12]
            - self.dof_pos[:, 16] / self.scales["dof_pos"][16]
        )
        return self._sqrdexp(error)

    def _reward_arm_yaw_zero(self):
        error = self.dof_pos[:, 12] - self.default_dof_pos[:, 12]
        reward = self._sqrdexp(error / self.scales["dof_pos"][12])
        error = self.dof_pos[:, 16] - self.default_dof_pos[:, 16]
        reward += self._sqrdexp(error / self.scales["dof_pos"][16])
        return reward / 2.0

    def _reward_arm_abad_symmetry(self):
        error = (
            self.dof_pos[:, 11] / self.scales["dof_pos"][11]
            - self.dof_pos[:, 15] / self.scales["dof_pos"][15]
        )
        return self._sqrdexp(error)

    def _reward_arm_abad_zero(self):
        error = self.dof_pos[:, 11] - self.default_dof_pos[:, 11]
        reward = self._sqrdexp(error / self.scales["dof_pos"][11])
        error = self.dof_pos[:, 15] - self.default_dof_pos[:, 15]
        reward += self._sqrdexp(error / self.scales["dof_pos"][15])
        return reward / 2.0
