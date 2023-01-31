# gpuGym - a clone of legged_gym #
This repository is a port of legged_gym from the good folk over at RSL.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.

---
### Useful Links
Project website: https://leggedrobotics.github.io/legged_gym/
Paper: https://arxiv.org/abs/2109.11978

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Clone and initialize this repo
   - clone `gpu_gym`
3. Install GPU Gym Requirements:
```bash
pip install -r requirements.txt
```
4. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (Preview 3 should still work) from https://developer.nvidia.com/isaac-gym
     - Extract the zip package
     - Copy the `isaacgym` folder, and place it in a new location
   - Install `issacgym` requirements
   ```bash
   cd <issacgym_python_location>
   pip install -e .
   ```
5. Run an example to validate
    - Run the following command from within isaacgym
   ```bash
   cd <issacgym_location>/python/examples
   python 1080_balls_of_solitude.py
   ```
   - For troubleshooting check docs `isaacgym/docs/index.html`
6. Install gpuGym
    ```bash
    pip install -e .
    ```
7. Use WandB for experiment tracking - follow [this guide](https://docs.wandb.ai/quickstart)

---

### CUDA Installation for Ubuntu 22.04 and above
#### Inspired by: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

1. Ensure Kernel Headers and Dev packages are installed
```bash
sudo apt-get install linux-headers-$(uname -r)
```

2. Install nvidia-cuda-toolkit
```bash
sudo apt install nvidia-cuda-toolkit
```

3. Remove outdated Signing Key
```bash
sudo apt-key del 7fa2af80
```

4. Install CUDA
```bash
# ubuntu2004 or ubuntu2204 or newer.
DISTRO=ubuntu2204
# Likely what you want, but check if you need otherse
ARCH=x86_64
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
```

5. Reboot, and you're good.
```bash
sudo reboot
```

6. Use these commands to check your installation
```bash
nvidia-smi
nvcc --version
```

**Troubleshooting Docs**
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation

---
### Use ###
1. Train:  
  ```python gym/scripts/train.py --task=mini_cheetah_ref```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `<gpuGym>/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
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
```python <gpuGym>/scripts/play.py --task=mini_cheetah_ref```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `--load_run` and `--checkpoint`.

----

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resourses/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `<gpuGym>/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!

---

### Jenny's gpuGym Weights crash course ###
https://hackmd.io/@yHrQmxajTZOYt87bbz6YRg/SJbscyN3t

---

### Weights and Biases Integration ###

#### What is WandB? ####

WandB is a machine learning dashboard to track hyperparameters, metrics, logs and source code to compare models live, store all training data in one place accessible by any web browser, and share your results easily with others. WandB is integrated with both normal single run training and sweeping through the train.py and sweep.py scripts. By default, WandB is disabled and no logging of training progress is made while a network will be trained. When in this mode, networks and source code will still be saved locally to the `logs` directory. 

#### Using WandB ####

To enable WandB, two pieces of information must be given to the script. The first is the WandB entity name, which is the username of the account to log to, and the second is a WandB project name, which is the name that all data will be saved under during training in the WandB console. 

There are two ways to pass this information to the scripts. The first is by command line argument using the flags `--wandb_entity=todo` and `--wandb_project=todo`, replacing “todo” with the relevant information for your run. The second method is by JSON config file. An example config file, `gym/user/wandb_config_example.json` is included in the directory but must be renamed to `wandb_config.json` for the scripts to read it. By default `wandb_config.json` is ignored by GIT via the `.gitignore` file to not pollute the repository. 

The scripts will first look for a command line argument and then check if there is a JSON config file. The command line settings are the highest priority and will override the JSON config.

#### Using Sweeps ####

A sweep allows for training multiple runs with different hyperparameter settings automatically. Sweeps are controlled by WandB, so creating an account and enabling it for your setup is required.

There are two main facilitators of a sweep: the sweep controller and sweeping agents. A single controller is created for each sweep and selects the hyperparameters to test for each individual run. The sweep controller is created by the sweep.py script on the start of a sweep but exists in the cloud through WandB’s servers. The controller consumes a sweep config, that is specified in a sweep_config.json file, and follows the prescribed behavior to pass parameters to the sweeping agents. A sweeping agent is an individual run of training that receives its settings from the sweep controller. When the sweep.py script is run, a single sweep controller is made that returns a sweep_id for the sweep and then an agent is created as another process that consumes the sweep_id and receives its settings from the controller before performing the training run. Once completed an agent will fully shut down and the sweep.py script will create another agent to perform the next run (if there are still new runs to complete). 

In addition to running agents sequentially on a single desktop, other computers connected to WandB can also control agents to complete the workload. Using the sweep_id that the sweep controller creates, any other computer connected to your WandB can use that ID to create more agents to run in parallel. Using the command line argument `--wandb_sweep_id=todo`, the sweep.py script will not create a new sweep controller and instead communicate with the first controller to request parameters for another agent to train. This can be done with multiple computers to parallelize sweeping across many systems. Note: multiple agents can be trained simultaneously on the same machine (VRAM permitting) but in general this doesn’t improve performance much over running sequentially as processing speed is the main limitation on a single machine.

If you would like to create and multiple sweep_config.json files, you can name them however you would like using the `--wandb_sweep_config=todo` command line argument to select which file to find the JSON object to define the sweep.

#### CLI Examples ####
Manually setting WandB project and entity:  
```python gym/scripts/train.py --task=a1 --wandb_entity=ajm4 --wandb_project=wandb_test_project```

Using a wandb_config.json file:  
```python gym/scripts/sweep.py --task=mini_cheetah --headless```

Selecting a config file name:  
```python gym/scripts/sweep.py --task=mini_cheetah --headless --wandb_sweep_config=sweep_config_example.json```

Using the entity name in the wandb_config.json but overriding the project name:  
```python gym/scripts/train.py --task=a1 --wandb_project=wandb_test_project```

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
