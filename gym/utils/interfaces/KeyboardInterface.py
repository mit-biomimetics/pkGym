from isaacgym import gymapi
import torch


class KeyboardInterface:
    def __init__(self, env):
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "forward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "left")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "right")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "back")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "yaw_left")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_E, "yaw_right")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_ESCAPE, "QUIT")
        env.gym.subscribe_viewer_keyboard_event(
            env.viewer, gymapi.KEY_SPACE, "space_shoot"
        )
        env.gym.subscribe_viewer_mouse_event(
            env.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot"
        )
        print("______________________________________________________________")
        print("Using keyboard interface, overriding default comand settings")
        print("commands are in 1/5 increments of max.")
        print("WASD: forward, strafe left, " "backward, strafe right")
        print("QE: yaw left/right")
        print("R: reset environments")
        print("ESC: quit")
        print("______________________________________________________________")

        env.commands[:] = 0.0
        env.cfg.commands.resampling_time = env.max_episode_length_s + 1
        self.max_vel_backward = -1.0
        self.max_vel_forward = 4.0
        self.increment_x = (self.max_vel_forward - self.max_vel_backward) * 0.1

        self.max_vel_sideways = 1.0
        self.increment_y = self.max_vel_sideways * 0.2

        self.max_vel_yaw = 2.0
        self.increment_yaw = self.max_vel_yaw * 0.2

    def update(self, env):
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.value == 0:
                continue
            if evt.action == "forward":
                env.commands[:, 0] = torch.clamp(
                    env.commands[:, 0] + self.increment_x,
                    max=self.max_vel_forward,
                )
            elif evt.action == "back":
                env.commands[:, 0] = torch.clamp(
                    env.commands[:, 0] - self.increment_x,
                    min=self.max_vel_backward,
                )
            elif evt.action == "left":
                env.commands[:, 1] = torch.clamp(
                    env.commands[:, 1] + self.increment_y,
                    min=-self.max_vel_sideways,
                )
            elif evt.action == "right":
                env.commands[:, 1] = torch.clamp(
                    env.commands[:, 1] - self.increment_y,
                    max=self.max_vel_sideways,
                )
            elif evt.action == "yaw_right":
                env.commands[:, 2] = torch.clamp(
                    env.commands[:, 2] - self.increment_yaw,
                    min=-self.max_vel_yaw,
                )
            elif evt.action == "yaw_left":
                env.commands[:, 2] = torch.clamp(
                    env.commands[:, 2] + self.increment_yaw,
                    max=self.max_vel_yaw,
                )
            elif evt.action == "QUIT":
                env.exit = True
            elif evt.action == "RESET":
                env.timed_out[:] = True
                env.reset()
            elif (
                evt.action == "space_shoot" or evt.action == "mouse_shoot"
            ) and evt.value > 0:
                env.shoot()
