# gpuGym - a clone of legged_gym #
This repository is a port of legged_gym from the good folk over at RSL.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.

---

### Useful Links ###
Project website: https://leggedrobotics.github.io/legged_gym/
Paper: https://arxiv.org/abs/2109.11978

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym (extract the zip package, copy the isaacgym folder within the package whereever you want it to live - I prefer in the directory with my virtual enviornment)
   - `cd isaacgym_lib/python && pip install -e .` to install the requirements
   - Try running an example `cd examples && python 1080_balls_of_solitude.py` (you need to execute the examples from the examples directory)
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && pip install -e .` 
5. Install gpuGym
    - Clone this repository
    - `cd gpuGym && pip install -e .`
6. Install WandB for experiment tracking - follow [this guide](https://docs.wandb.ai/quickstart)
    - `pip3 install wandb`

---

### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one containing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

---

### Use ###
1. Train:  
  ```python issacgym_anymal/scripts/train.py --task=anymal_c_flat```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python issacgym_anymal/scripts/play.py --task=anymal_c_flat```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

---

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resourses/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!

---

### Jenny's gpuGym Weights crash course ###
https://hackmd.io/@yHrQmxajTZOYt87bbz6YRg/SJbscyN3t

---

### Weights and Biases Integration ###

#### What is WandB? ####

"[WandB](https://wandb.ai/site) is a central dashboard to keep track of your hyperparameters, system metrics, and predictions so you can compare models live, and share your findings."

##### Why do we use it? #####

WandB's logging is the state of the art tool for deploying any system that trains neural networks. It offers an attractive interface for figure creation, and easy to use API, and the ability to automatically sweep hyperparameters.

##### How do we use it? #####

Once you've completed the WandB quickstart (installing and logging in), there are three required steps to tracking your experiments with WandB:
1. Within the `env/[your system]/[your system]_config.py` file, add `do_wandb=True` to the `[your system]CfgPPO` class structure
2. Add `--wannd_project=[your project name]` to your python run arguments
3. Add `--wandb_entity=[your wandb username]` to your python run arguments

Example:
1. `class MITHumanoidCfgPPO(LeggedRobotCfgPPO): [\newline]
    do_wandb = True`
3. ```python /isaacgym3/python/gpuGym/gpugym/scripts/train.py --task=mit_humanoid --wandb_entity=sheim --wandb_project=humanoid```

---

### WandB Major Features ###

##### Configuring Experiment Parameters #####

You may also elect to send information about specific your experiment runs to the WandB dashboard.
For example, if your experiment varys the network size, some reward scale, and the network output type, 
you would tell WandB this by adding `num_layers`, `termination rewards`, `control type` to the experiment confiugurtation dictionary.

To do this, create a `wandb` class in the same configuration file where you set `do_wandb=True`. 
Within this class, create a dictionary called `what_to_log`.
This dictionary takes the name of the value you'd like to log as the key and a list of where you will find that attribute as the value.


An Example of this class is as follows:
```python
class [System]CfgPPO(FixedRobotCfgPPO):
    do_wandb = True
    class wandb:
        what_to_log = {}
        # How to fill out `what_to_log`
        #   key: the name that will appear on the Weights and Biases Dashboard
        #   value: list of strings -
        #       value[0] = Choose one of {train_cfg: [System]CfgPPO, env_cfg: [System]Cfg}
        #       value[1, ..., n-1] = nested classes where desired value is stored
        #       value[n] = The actual attribute storing the value describing the experiment
        
        # [System]CfgPPO.policy.num_layers
        what_to_log['Number of Layers'] = ['train_cfg', 'policy', 'num_layers']

        # [System]Cfg.rewards.sub_spaces.actuation_penalty
        what_to_log['Actuation Reward Scale'] = ['env_cfg', 'rewards', 'scales', 'actuation']
```

##### Success Metric Logging #####

You may also care to log some quantitative information about your agents' behavior that is not clear from reading the rewards.
To track this information, you may use WandB to log success rates that you design and tune.
The current implementation collects this information when each agent is reset, though this can be easily changed.

The workflow to track success metrics has two main steps:
1. [optional] In the `env/[your system]/[your system]_config.py` file, add a class called `metrics` to the `[your system]Cfg` config class structure.
   1. In the `metrics` class, create tunable success `thresholds` for easier experiment tracking.
2. Create a `_custom_reset_logging(self, env_ids)` method within your `[system]` class in your `[system].py` file.
   1. Collect all relevant system information from the buffers. I.e. dof_states, global location, etc..
   2. Implement your success criteria and name it `[your_success_metric]`.
   3. Count the agents which satisfy the criteria.
   4. Set the success counts by `self.extras[success counts][your_success_metric] = success_count`
   5. Set the total environments seen by `self.extras[episode counts][total_reset] = env_ids.shape[0]`

An example of this full workflow is here:

In the `[your system]_config.py` file:
```python
class [your system]Cfg(FixedRobotCfg):
        class success_metrics:
            class thresholds:
                dof_pos = [some number]  # [units] - description of the threshold
                dof_vel = [some number]  # [units/s] - description of the threshold
```

In the `[your system].py` file:
```python

class [your system](FixedRobot):
    
    def _custom_reset_logging(self, env_ids):
        # Collect all relevant information to the success metric
        # Note: if you need trajectories of states, store that in a new state trajectories buffer for processing here
        dof_pos = self.dof_pos[env_ids, 0].unsqueeze(dim=-1)
        dof_vel = self.dof_vel[env_ids, 1].unsqueeze(dim=-1)

        # Collect success metrics that you define in env_config
        dof_pos_threshold = self.cfg.success_metrics.thresholds.dof_pos
        dof_vel_threshold = self.cfg.success_metrics.thresholds.dof_vel

        # dof stable ::= Both the position and velocity of the dof are near zero
        dof_pos_zero_success = torch.where(torch.absolute(dof_pos) < dof_pos_threshold, torch.ones_like(dof_pos), torch.zeros_like(dof_pos))
        dof_stable_success = torch.where(torch.absolute(dof_vel) < dof_vel_threshold, dof_pos_zero_success, torch.zeros_like(dof_pos))

        # Count all successes
        dof_pos_zero_count = torch.sum(dof_pos_zero_success)
        dof_stable_count = torch.sum(dof_stable_success)

        # Add these successes to the central logging area for GPU_rl to process
        self.extras["success counts"]['dof_pos_zero'] = dof_pos_zero_count
        self.extras["success counts"]['dof_stable'] = dof_stable_count
        self.extras["episode counts"]['total_reset'] =  env_ids.shape[0]

```

---

### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`
2. If you get the following error: `RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error?`, try [restarting your computer](https://discuss.pytorch.org/t/solved-torch-cant-access-cuda-runtimeerror-unexpected-error-from-cudagetdevicecount-and-error-101-invalid-device-ordinal/115004).

---

### Known Issues ###
1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesireable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from trhe reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```
