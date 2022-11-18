class SimCfg:
    dt = 0.001  # calculated based on env.config.control.desired_sim_frequency
    substeps = 1
    gravity = [0., 0. , -9.81]  # [m/s^2]
    up_axis = 1  # 0 is y, 1 is z

    class physx:
        num_threads = 10
        solver_type = 1  # 0: pgs, 1: tgs (more robust, more expensive)
        num_position_iterations = 4
        num_velocity_iterations = 0
        contact_offset = 0.01
        rest_offset = 0.0
        bounce_threshold_velocity = 0.5 #0.5 [m/s]
        contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
        max_depenetration_velocity = 10.0
        max_gpu_contact_pairs = 2**23  #2**24 needed for 8000 envs or more
        default_buffer_size_multiplier = 5
