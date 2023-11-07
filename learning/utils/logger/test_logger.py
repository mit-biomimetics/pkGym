from .Logger import Logger
import torch


class MockObject:
    def __init__(self):
        self.a = 0
        self.b = 10


def all_rewards_registered(logger, reward_names):
    for key in reward_names:
        assert (
            key in logger.reward_logs.log_items.keys()
        ), "key not registered in logger."


def only_rewards_registered(logger, reward_names):
    for key in logger.reward_logs.log_items.keys():
        assert key in reward_names, "erroneous key in logger."


def only_category_registered(logger, categories):
    for key in logger.iteration_logs.logs.keys():
        assert key in categories, "erroneous key in logger."


def both_target_and_log_set_up(logger):
    for key in logger.iteration_logs.logs.keys():
        assert (
            key in logger.iteration_logs.targets.keys()
        ), "target not registered in PerIteration."
    for key in logger.iteration_logs.targets.keys():
        assert (
            key in logger.iteration_logs.logs.keys()
        ), "log not registered in PerIteration."


def test_logger_setup():
    mocky = MockObject()

    logger = Logger()

    assert not logger.initialized, "Logger should not be initialized."

    logger.initialize(num_envs=5, episode_dt=0.1, total_iterations=1000, device="cpu")

    assert logger.initialized, "Logger should be initialized."

    logger.register_rewards(["first", "second"])
    logger.register_rewards(reward_names=["third"])

    all_rewards_registered(logger, ["first", "second", "third"])
    only_rewards_registered(logger, ["first", "second", "third"])

    logger.register_category("first", mocky, ["a"])
    logger.register_category(
        category="alg_performance", target=mocky, attribute_list=["a", "b"]
    )

    only_category_registered(logger, ["first", "alg_performance"])
    both_target_and_log_set_up(logger)


def check_episode_count(logger, expected_count=1):
    assert logger.reward_logs.finished_episode_count == expected_count, (
        f"Episode count not correct. Expected {expected_count}"
        f", got {logger.reward_logs.finished_episode_count}"
    )


def check_average_time(logger, expected_time):
    avg_time = logger.reward_logs.get_average_time()
    assert (
        abs(avg_time.item() - expected_time) < 1e-5
    ), f"Average time {avg_time} is not close to {expected_time}"


def check_average_reward(logger, reward_name, expected_average):
    avg_reward = logger.reward_logs.get_average_rewards()[reward_name]
    assert (
        abs(avg_reward.item() - expected_average) < 1e-5
    ), f"Average reward {avg_reward} is not close to {expected_average}"


def test_logging_rewards():
    logger = Logger()
    episode_dt = 0.1
    logger.initialize(
        num_envs=3, episode_dt=episode_dt, total_iterations=1000, device="cpu"
    )
    logger.register_rewards(["first", "second", "third"])

    reward_dict_1 = {
        "first": torch.tensor([5.0, 5.0, 5.0]) * episode_dt,
        "second": torch.tensor([3.0, 0.0, 2.0]) * episode_dt,
    }
    reward_dict_2 = {"third": torch.tensor([1.0, 2.0, -2.0]) * episode_dt}
    dones = torch.tensor([False, False, False])

    for _ in range(9):
        logger.log_rewards(reward_dict_1)
        logger.log_rewards(reward_dict_2)
        logger.finish_step(dones)
    else:
        dones[1:] = True
        logger.log_rewards(reward_dict_1)
        logger.log_rewards(reward_dict_2)
        logger.finish_step(dones)

    check_episode_count(logger, expected_count=2)

    check_average_time(logger, expected_time=1.0)

    check_average_reward(logger, "first", expected_average=5.0)
    check_average_reward(logger, "second", expected_average=1.0)
    check_average_reward(logger, "third", expected_average=0.0)


def test_logging_iteration():
    mocky = MockObject()
    logger = Logger()
    logger.initialize(num_envs=2, episode_dt=0.1, total_iterations=1000, device="cpu")

    logger.register_category("performanceA", mocky, ["a"])
    logger.register_category("performanceAB", mocky, ["a", "b"])

    for _ in range(10):
        logger.log_category("performanceA")
        logger.log_category("performanceAB")
        logger.finish_iteration()

    for key, val in logger.iteration_logs.logs["performanceA"].items():
        assert key in ["a"], "Erroneous key in logger."
        assert val == 0, "Value not logged correctly."

    for key, val in logger.iteration_logs.logs["performanceAB"].items():
        assert key in ["a", "b"], "Erroneous key in logger."
        if key == "a":
            assert val == 0, "Value not logged correctly."
        elif key == "b":
            assert val == 10, "Value not logged correctly."
        else:
            raise ValueError("Erroneous key in logger.")


def test_timer():
    import time

    trial_time = 0.2

    logger = Logger()
    logger.initialize(total_iterations=1000, device="cpu")
    logger.tic("first_step")
    assert logger.get_time("first_step") == -1, "Timer not initialized correctly."

    time.sleep(trial_time)
    logger.toc("first_step")
    assert logger.get_time("first_step") > 0, "Timer not working correctly."

    logger.tic("second_step")
    time.sleep(trial_time)
    logger.toc("second_step")

    logger.finish_iteration()

    a = logger.get_time("first_step")
    b = logger.get_time("second_step")

    ETA = logger.estimate_ETA(["first_step"], mode="iteration")
    # print(ETA)
    expected_ETA = a * (1000 - 1)
    assert abs(ETA - expected_ETA) < 1e-5, f"ETA {ETA} is not close to {expected_ETA}"

    ETA2 = logger.estimate_ETA(["first_step"], mode="total")
    expected_ETA2 = a * (1000 - 1)
    assert (
        abs(ETA2 - expected_ETA2) < 1e-5
    ), f"ETA {ETA2} is not close to {expected_ETA2}"

    assert (a + b) >= 2 * trial_time, "Timer not working correctly."
