class TestEnvironment:
    def test_all_rewards_have_right_shape(self, env_list):
        def generate_message_shape(env, reward_name):
            name = env.__class__.__name__
            message = f"Wrong shape for {reward_name} in {name}"
            return message

        def generate_message_size(env, reward_name):
            name = env.__class__.__name__
            message = f"Wrong size for {reward_name} in {name}"
            return message

        for env in env_list:
            for item in dir(env):
                if "_reward_" in item:
                    reward_name = item.replace("_reward_", "")
                    reward = env._eval_reward(reward_name)
                    assert len(reward.shape) == 1, generate_message_shape(
                        env, reward_name
                    )
                    assert reward.shape[0] == env.num_envs, generate_message_size(
                        env, reward_name
                    )
