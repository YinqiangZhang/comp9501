__author__ = "Mingyang Wang"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Mingyang Wang"
__email__ = "mingyang.wang@tum.de"
__status__ = "Integration"

"""
Unit tests of the module gym_commonroad.commonroad_env
"""
import os
import gym
import pytest
import numpy as np

from commonroad.scenario.trajectory import State
from commonroad.common.solution import VehicleType, VehicleModel

from commonroad_rl.tests.common.marker import *
from commonroad_rl.gym_commonroad.vehicle import Vehicle
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.tests.common.path import resource_root


env_id = "commonroad-v0"
resource_path = resource_root("test_commonroad_env")
meta_scenario_path = os.path.join(resource_path, "pickles", "meta_scenario")
problem_path = os.path.join(resource_path, "pickles", "problem")

# Test results in strict and non-strict mode; For non-strict check, give different check radius and see the difference
@pytest.mark.parametrize(
    ("strict_off_road_check", "non_strict_check_circle_radius", "action", "expected_is_off_road_list"),
    [
        (True, 0, np.array([0.0, 0.0]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (True, 0,  np.array([0.0, 0.1]), [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        (False, 0.1, np.array([0.0, 0.1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (False, 0.5, np.array([0.0, 0.1]), [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
        (False, 0.8, np.array([0.0, 0.1]), [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    ],
)
@unit_test
@functional
def test_check_off_road(strict_off_road_check, non_strict_check_circle_radius, action, expected_is_off_road_list):

    vehicle_params = {"vehicle_type": VehicleType.BMW_320i,
                      "vehicle_model": VehicleModel.PM}

    env = CommonroadEnv(
        meta_scenario_path=meta_scenario_path,
        train_reset_config_path=problem_path,
        test_reset_config_path=problem_path,
        strict_off_road_check=strict_off_road_check,
        non_strict_check_circle_radius = non_strict_check_circle_radius,
        vehicle_params = vehicle_params,
    )
    env.reset()
    result_list = []

    for i in range(40):
        observation, reward, done, info = env.step(action)
        if i >= 30:
            result_list.append(info["is_off_road"])
        # print(env.ego_vehicle.state.position)
        env.render()

    assert result_list == expected_is_off_road_list
