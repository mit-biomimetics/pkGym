import sys
from isaacgym import gymapi
from isaacgym import gymutil
import torch
from gym.envs.base.task_skeleton import TaskSkeleton


# * Base class for RL tasks
class BaseTask(TaskSkeleton):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        self.gym = gym
        self.sim = sim
        self.sim_params = sim_params
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # * env device is GPU only if sim is on GPU and use_gpu_pipeline=True,
        # * otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # * graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_actuators = cfg.env.num_actuators

        # * optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.to_be_reset = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.terminated = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.timed_out = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.exit = False

        # * if running with a viewer, set up keyboard shortcuts and camera
        if self.headless is False:
            # * subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

    def _render(self, sync_frame_time=True):
        if self.viewer:
            # * check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # * check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # * fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # * step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
