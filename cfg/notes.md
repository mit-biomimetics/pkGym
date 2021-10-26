# IsaacGym Notes

### add new observation

1. increase observation size appropriately in `self.cfg["env"]["numObservations"] =`
2. in `compute_observations(self)`, pass everything needed to compute the new observation (including scaling factor)
3. in `compute_humanoid_observations(...)` calculate and pass out the actual new observation

### adding new reward
1. pass everything needed in `compute_reward`
2. implement in `compute_humanoid_reward`
3. add weight in `MIT_humanoid.yaml`, and read in initialization (e.g. `self.rew_scales["height"] = self.cfg["env"]["learn"]["heightRewardScale"]`)
4. **NOTE** everything used to compute a reward should be observable to the agent (or you lose the Markov property). Easiest is to pass it out as an observation.