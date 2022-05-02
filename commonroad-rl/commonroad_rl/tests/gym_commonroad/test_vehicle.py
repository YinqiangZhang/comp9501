__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Unit tests of the module gym_commonroad.vehicle
"""
import os

import pytest
import time
import numpy as np
from commonroad.common.solution import VehicleType, VehicleModel
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import State

from commonroad_rl.gym_commonroad import *
from commonroad_rl.gym_commonroad.vehicle import Vehicle
from commonroad_rl.tests.common.marker import *

import inspect

from stable_baselines.common.env_checker import check_env
import timeit


@pytest.mark.parametrize(
    ("steering_angle", "velocity", "expected_orientation"),
    [
        (0, 0, 0),
        (0, 30, 0),
        (0.5, 10, 2.1183441714035096),
        (0.5, 30, 3 * 2.1183441714035096 - 2 * np.pi),
        (-0.5, 10, -2.1183441714035096),
        (-0.5, 30, -3 * 2.1183441714035096 + 2 * np.pi),
    ],
)
@unit_test
@functional
def test_valid_vehicle_orientation(steering_angle, velocity, expected_orientation):
    vehicle_params = {
        "vehicle_type": VehicleType.BMW_320i,
        "vehicle_model": VehicleModel.KS,
    }

    dummy_state = {
        "position": np.array([0.0, 0.0]),
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0.0,
        "orientation": 0.0,
    }
    initial_state = State(
        **dummy_state, steering_angle=steering_angle, velocity=velocity
    )
    dt = 1.0

    # Not to do anything, just continue the way with the given velocity
    action = np.array([0.0, 0.0])

    vehicle = Vehicle.create_vehicle(vehicle_params)
    vehicle.reset(initial_state, dt)

    vehicle.step(action)
    resulting_orientation = vehicle.state.orientation
    assert resulting_orientation == expected_orientation


@pytest.mark.parametrize(
    ("action", "velocity", "expected_position"),
    [
        (
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        ),
        (
            np.array([0.0, 0.0]),
            np.array([10.0, 0.5]),
            np.array([10.0, 0.5]),
        ),
        (
            np.array([0.5, 0.5]),
            np.array([0.0, 0.0]),
            np.array([2.15625, 2.15625]),
        ),
        (
            np.array([0.5, 0.5]),
            np.array([10.0, 0.5]),
            np.array([12.15625, 2.65625]),
        ),
        (
            np.array([0.5, 0]),
            np.array([10.0, 0.5]),
            np.array([12.15625, 0.5]),
        ),
        (
            np.array([0.0, 0.5]),
            np.array([10.0, 0.5]),
            np.array([10.0, 2.65625]),
        ),
    ],
)
@unit_test
@functional
def test_PM_vehicle_model(action, velocity, expected_position):
    """
    Tests the Point Mass vehicle model
    """
    vehicle_params = {
        "vehicle_type": VehicleType.BMW_320i,
        "vehicle_model": VehicleModel.PM,
    }

    dummy_state = {
        "position": np.array([0.0, 0.0]),
        "time_step": 0,
    }

    initial_state = State(**dummy_state, velocity=velocity[0], velocity_y=velocity[1])

    dt = 1.0

    vehicle = Vehicle.create_vehicle(vehicle_params)
    vehicle.reset(initial_state, dt)

    vehicle.step(action)
    resulting_position = vehicle.state.position
    assert np.allclose(resulting_position, expected_position)
