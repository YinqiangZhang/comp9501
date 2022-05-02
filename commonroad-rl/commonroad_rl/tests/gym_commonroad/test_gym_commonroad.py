__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Module tests of the module gym_commonroad
"""
import os
import random
import timeit
import numpy as np

from commonroad_rl.gym_commonroad import *
from commonroad_rl.tests.common.marker import *

from stable_baselines.common.env_checker import check_env

from commonroad_rl.tests.common.non_functional import function_to_string
from commonroad_rl.tests.common.path import resource_root, output_root

env_id = "commonroad-v0"
resource_path = resource_root("test_gym_commonroad")
meta_scenario_path = os.path.join(resource_path, "pickles", "meta_scenario")
problem_path = os.path.join(resource_path, "pickles", "problem")

output_path = output_root("test_gym_commonroad")
visualization_path = os.path.join(output_path, "visualization")


@pytest.mark.parametrize(
    ("num_of_checks", "test_env", "play"),
    [(15, False, False), (15, False, True), (15, True, False), (15, True, True)],
)
@module_test
@functional
def test_check_env(num_of_checks, test_env, play):

    # Run more circles of checking to search for sporadic issues
    for idx in range(num_of_checks):
        print(f"Checking progress: {idx + 1}/{num_of_checks}")
        env = gym.make(
            env_id,
            meta_scenario_path=meta_scenario_path,
            train_reset_config_path=problem_path,
            test_reset_config_path=problem_path,
            visualization_path=visualization_path,
            test_env=False,
            play=False,
        )
        check_env(env)


@pytest.mark.parametrize(
    "reward_type", ["dense_reward", "sparse_reward", "hybrid_reward"]
)
@module_test
@functional
def test_step(reward_type):
    env = gym.make(
        env_id,
        meta_scenario_path=meta_scenario_path,
        train_reset_config_path=problem_path,
        test_reset_config_path=problem_path,
        visualization_path=visualization_path,
        reward_type=reward_type,
    )
    env.reset()

    i = 1
    # while not done:
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # TODO: define reference format and assert
        print(f"step {i}, {info}")


@pytest.mark.parametrize(
    "reward_type", ["dense_reward", "sparse_reward", "hybrid_reward"]
)
@module_test
@functional
def test_observation_order(reward_type):
    env = gym.make(
        "commonroad-v0",
        meta_scenario_path=meta_scenario_path,
        train_reset_config_path=problem_path,
        test_reset_config_path=problem_path,
        flatten_observation=False,
    )

    # set random seed to make the env choose the same planning problem
    random.seed(0)
    obs_dict = env.reset()

    # collect observation in other format
    env = gym.make(
        "commonroad-v0",
        meta_scenario_path=meta_scenario_path,
        train_reset_config_path=problem_path,
        test_reset_config_path=problem_path,
        flatten_observation=True,
    )

    # seed needs to be reset before function call
    random.seed(0)
    obs_flatten = env.reset()
    obs_flatten_exp = np.zeros(env.observation_space_size)

    # flatten the dictionary observation
    index = 0
    for obs_dict_value in obs_dict.values():
        size = np.prod(obs_dict_value.shape)
        obs_flatten_exp[index: index + size] = obs_dict_value.flat
        index += size

    print(obs_flatten)
    print(obs_flatten_exp)
    # compare 2 observation
    assert np.allclose(
        obs_flatten_exp, obs_flatten
    ), "Two observations don't have the same order"


@pytest.mark.parametrize(
    "reward_type", ["dense_reward", "sparse_reward", "hybrid_reward"]
)
@module_test
@nonfunctional
def test_step_time(reward_type):
    # Define reference time
    reference_time = 15.0

    def measurement_setup():
        import gym
        import numpy as np

        env = gym.make(
            "{env_id}",
            meta_scenario_path="{meta_scenario_path}",
            train_reset_config_path="{problem_path}",
            test_reset_config_path="{problem_path}",
            visualization_path="{visualization_path}",
            reward_type="{reward_type}",
        )
        env.reset()
        action = np.array([0.0, 0.0])

    def measurement_code(env, action):
        env.step((action))

    setup_str = function_to_string(measurement_setup)
    code_str = function_to_string(measurement_code)

    times = timeit.repeat(setup=setup_str, stmt=code_str, repeat=1, number=10000)
    average_time = np.average(times)

    # TODO: Set exclusive CPU usage for this thread, because other processes influence the result
    # assert average_time < reference_time, f"The step is too slow, average time was {average_time}"
