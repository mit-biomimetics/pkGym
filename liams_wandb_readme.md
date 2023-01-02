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