__author__ = "Brian Liao"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Brian Liao"
__email__ = "brian.liao@tum.de"
__status__ = "Integration"

"""
Unit test script for surrounding observations
"""

import pytest
import os
import pickle
import random
import matplotlib.pyplot as plt
import commonroad_dc.pycrcc as pycrcc

from commonroad_rl.gym_commonroad.feature_extraction.surroundings import *
from commonroad_rl.gym_commonroad.utils.scenario import *
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)
from commonroad_rl.gym_commonroad.vehicle import Vehicle

from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import *
from commonroad_rl.tests.common.evaluation import *


@pytest.mark.parametrize(
    ("filename", "time_step_gt", "ego_vehicle_state_gt", "p_rel_gt", "v_rel_gt"),
    [
        (
            "DEU_AAH-4_1000_T-1.pickle",
            50,
            {
                "position": [127.50756356637483, -50.69294785562317],
                "orientation": 4.298126916546023,
                "velocity": 8.343911610829114,
            },
            [60.0, 17.50689256668268, 60.0, 60.0, 60.0, 60.0],
            [0.0, -0.3290005752341578, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "DEU_AAH-4_1000_T-1.pickle",
            51,
            {
                "position": [127.39433877716928, -50.9499165494171],
                "orientation": 4.296344007588243,
                "velocity": 8.558071157124918,
            },
            [60.0, 17.431754173281178, 60.0, 60.0, 60.0, 60.0],
            [0.0, -0.0348669273743365, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "DEU_AAH-4_1000_T-1.pickle",
            52,
            {
                "position": [127.2789061588074, -51.210826963166035],
                "orientation": 4.294731018487505,
                "velocity": 8.560948563466251,
            },
            [60.0, 17.359412425745013, 60.0, 60.0, 60.0, 60.0],
            [0.0, 0.05249202534491637, 0.0, 0.0, 0.0, 0.0],
        ),
    ],
)
@unit_test
@functional
def test_get_surrounding_obstacles_lane_rect(
    filename, time_step_gt, ego_vehicle_state_gt, p_rel_gt, v_rel_gt
):
    """
    Tests the `src/gym_commonroad/feature_extraction/surroundings_circle.py::get_surrounding_obstables()` method at the initial step
    """
    # Load meta scenario and problem dict
    meta_scenario_reset_dict_path = os.path.join(
        resource_root("test_surroundings/meta_scenario"),
        "meta_scenario_reset_dict.pickle",
    )
    with open(meta_scenario_reset_dict_path, "rb") as f:
        meta_scenario_reset_dict = pickle.load(f)
    problem_meta_scenario_dict_path = os.path.join(
        resource_root("test_surroundings/meta_scenario"),
        "problem_meta_scenario_dict.pickle",
    )
    with open(problem_meta_scenario_dict_path, "rb") as f:
        problem_meta_scenario_dict = pickle.load(f)

    # Load scenarios and problems
    fn = os.path.join(resource_root("test_surroundings/problem"), filename)
    with open(fn, "rb") as f:
        problem_dict = pickle.load(f)
    benchmark_id = filename.split(".")[0]

    # Set scenario and problem
    meta_scenario = problem_meta_scenario_dict[benchmark_id]
    meta_scenario_id = meta_scenario.benchmark_id
    obstacle_list = problem_dict["obstacle"]
    scenario = restore_scenario(meta_scenario, obstacle_list)
    planning_problem: PlanningProblem = random.choice(
        list(problem_dict["planning_problem_set"].planning_problem_dict.values())
    )

    reset_config = meta_scenario_reset_dict[meta_scenario_id]
    obstacle_lanelet_id_dict = problem_dict["obstacle_lanelet_id_dict"]
    connected_lanelet_dict = reset_config["connected_lanelet_dict"]

    # Create ground truth ego vehicle state
    ego_vehicle = Vehicle.create_vehicle({"vehicle_type": 2, "vehicle_model": 2})
    ego_vehicle.reset(planning_problem.initial_state, scenario.dt)
    ego_vehicle_state = ego_vehicle.state
    ego_vehicle_state.position = ego_vehicle_state_gt["position"]
    ego_vehicle_state.velocity = ego_vehicle_state_gt["velocity"]
    ego_vehicle_state.orientation = ego_vehicle_state_gt["orientation"]

    lanelet_polygons = [
        (l.lanelet_id, l.convert_to_polygon())
        for l in scenario.lanelet_network.lanelets
    ]
    lanelet_polygons_sg = pycrcc.ShapeGroup()
    for l_id, poly in lanelet_polygons:
        lanelet_polygons_sg.add_shape(create_collision_object(poly))

    ego_vehicle_lanelet_id = sorted_lanelets_by_state_realtime(
        scenario, ego_vehicle_state, lanelet_polygons, lanelet_polygons_sg
    )[0]
    ego_vehicle_lanelet = scenario.lanelet_network.find_lanelet_by_id(
        ego_vehicle_lanelet_id
    )

    max_lane_merge_range = 1000.0
    curvi_cosy, _ = get_local_curvi_cosy(
        scenario, ego_vehicle_lanelet_id, None, max_lane_merge_range
    )

    surrounding_dict = dict()
    lanelet_dict, all_lanelets_set = get_nearby_lanelet_id(
        connected_lanelet_dict, ego_vehicle_lanelet
    )
    surrounding_dict["lanelet_dict"] = lanelet_dict
    surrounding_dict["all_lanelets_set"] = all_lanelets_set

    # Test the method
    dummy_rel_vol = 0.0
    dummy_rel_pos = 60.0
    lane_rect_sensor_range_length = 100.0
    lane_rect_sensor_range_width = 7.0
    (
        v_rel,
        p_rel,
        detected_obstacles,
        surrounding_area,
    ) = get_surrounding_obstacles_lane_rect(
        scenario.dynamic_obstacles,
        scenario.static_obstacles,
        obstacle_lanelet_id_dict,
        surrounding_dict["all_lanelets_set"],
        curvi_cosy,
        surrounding_dict["lanelet_dict"],
        ego_vehicle_state,
        time_step_gt,
        dummy_rel_vol,
        dummy_rel_pos,
        lane_rect_sensor_range_length,
        lane_rect_sensor_range_width,
    )

    # Check against ground truth
    assert np.allclose(np.array(v_rel), np.array(v_rel_gt))
    assert np.allclose(np.array(p_rel), np.array(p_rel_gt))


@pytest.mark.parametrize(
    ("filename", "time_step_gt", "ego_vehicle_state_gt", "p_rel_gt", "v_rel_gt"),
    [
        (
            "DEU_AAH-4_1000_T-1.pickle",
            43,
            {
                "position": [128.2222609335789, -49.022624022869934],
                "orientation": 4.319904886182895,
                "velocity": 7.082343029302184,
            },
            [60.0, 18.20398695801839, 60.0, 60.0, 47.40902350651531, 60.0],
            [0.0, -2.025488953984791, 0.0, 0.0, 2.6653220542271576, 0.0],
        ),
        (
            "DEU_AAH-4_1000_T-1.pickle",
            44,
            {
                "position": [128.13057639473206, -49.24300781209363],
                "orientation": 4.315381265190625,
                "velocity": 7.291671591323439,
            },
            [60.0, 18.0687671267808, 60.0, 60.0, 47.60227218657741, 60.0],
            [0.0, -1.7668531163933068, 0.0, 0.0, 2.529247893570327, 0.0],
        ),
        (
            "DEU_AAH-4_1000_T-1.pickle",
            45,
            {
                "position": [128.03441395825942, -49.47140451195109],
                "orientation": 4.31163016190567,
                "velocity": 7.668769571559476,
            },
            [60.0, 17.945934452223746, 60.0, 60.0, 47.80335935977887, 60.0],
            [0.0, -1.33718633344232, 0.0, 0.0, 2.2206628198837794, 0.0],
        ),
    ],
)
@unit_test
@functional
def test_get_surrounding_obstacles_lane_circ(
    filename, time_step_gt, ego_vehicle_state_gt, p_rel_gt, v_rel_gt
):
    """
    Tests the `src/gym_commonroad/feature_extraction/surroundings_circle.py::get_surrounding_obstables()` method at the initial step
    """
    # Load meta scenario and problem dict
    meta_scenario_reset_dict_path = os.path.join(
        resource_root("test_surroundings/meta_scenario"),
        "meta_scenario_reset_dict.pickle",
    )
    with open(meta_scenario_reset_dict_path, "rb") as f:
        meta_scenario_reset_dict = pickle.load(f)
    problem_meta_scenario_dict_path = os.path.join(
        resource_root("test_surroundings/meta_scenario"),
        "problem_meta_scenario_dict.pickle",
    )
    with open(problem_meta_scenario_dict_path, "rb") as f:
        problem_meta_scenario_dict = pickle.load(f)

    # Load scenarios and problems
    fn = os.path.join(resource_root("test_surroundings/problem"), filename)
    with open(fn, "rb") as f:
        problem_dict = pickle.load(f)
    benchmark_id = filename.split(".")[0]

    # Set scenario and problem
    meta_scenario = problem_meta_scenario_dict[benchmark_id]
    meta_scenario_id = meta_scenario.benchmark_id
    obstacle_list = problem_dict["obstacle"]
    scenario = restore_scenario(meta_scenario, obstacle_list)
    planning_problem: PlanningProblem = random.choice(
        list(problem_dict["planning_problem_set"].planning_problem_dict.values())
    )

    reset_config = meta_scenario_reset_dict[meta_scenario_id]
    obstacle_lanelet_id_dict = problem_dict["obstacle_lanelet_id_dict"]
    connected_lanelet_dict = reset_config["connected_lanelet_dict"]

    # Create ground truth ego vehicle state
    ego_vehicle = Vehicle.create_vehicle({"vehicle_type": 2, "vehicle_model": 2})
    ego_vehicle.reset(planning_problem.initial_state, scenario.dt)
    ego_vehicle_state = ego_vehicle.state
    ego_vehicle_state.position = ego_vehicle_state_gt["position"]
    ego_vehicle_state.velocity = ego_vehicle_state_gt["velocity"]
    ego_vehicle_state.orientation = ego_vehicle_state_gt["orientation"]

    lanelet_polygons = [
        (l.lanelet_id, l.convert_to_polygon())
        for l in scenario.lanelet_network.lanelets
    ]
    lanelet_polygons_sg = pycrcc.ShapeGroup()
    for l_id, poly in lanelet_polygons:
        lanelet_polygons_sg.add_shape(create_collision_object(poly))

    ego_vehicle_lanelet_id = sorted_lanelets_by_state_realtime(
        scenario, ego_vehicle_state, lanelet_polygons, lanelet_polygons_sg
    )[0]
    ego_vehicle_lanelet = scenario.lanelet_network.find_lanelet_by_id(
        ego_vehicle_lanelet_id
    )

    max_lane_merge_range = 1000.0
    curvi_cosy, _ = get_local_curvi_cosy(
        scenario, ego_vehicle_lanelet_id, None, max_lane_merge_range
    )

    surrounding_dict = dict()
    lanelet_dict, all_lanelets_set = get_nearby_lanelet_id(
        connected_lanelet_dict, ego_vehicle_lanelet
    )
    surrounding_dict["lanelet_dict"] = lanelet_dict
    surrounding_dict["all_lanelets_set"] = all_lanelets_set

    # Test the method
    initial_time_step = 0
    dummy_rel_vol = 0.0
    dummy_rel_pos = 60.0
    lane_rect_sensor_range_radius = 50.0
    (
        v_rel,
        p_rel,
        detected_obstacles,
        surrounding_area,
    ) = get_surrounding_obstacles_lane_circ(
        scenario.dynamic_obstacles,
        scenario.static_obstacles,
        obstacle_lanelet_id_dict,
        surrounding_dict["all_lanelets_set"],
        curvi_cosy,
        surrounding_dict["lanelet_dict"],
        ego_vehicle_state,
        time_step_gt,
        dummy_rel_vol,
        dummy_rel_pos,
        lane_rect_sensor_range_radius,
    )

    # Check against ground truth
    assert np.allclose(np.array(v_rel), np.array(v_rel_gt))
    assert np.allclose(np.array(p_rel), np.array(p_rel_gt))


@pytest.mark.parametrize(
    (
        "filename",
        "time_step_gt",
        "ego_vehicle_state_gt",
        "prev_rel_pos_gt",
        "p_rel_gt",
        "p_rel_rate_gt",
    ),
    [
        (
            "DEU_AAH-4_1000_T-1.pickle",
            0,
            {
                "position": [130.95148999999998, -38.046040000000005],
                "orientation": 4.573031246397025,
                "velocity": 8.9459,
            },
            [
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                60.0,
                60.0,
                8.134482069004424,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),  # manually set to 0 at timestep 0
        (
            "DEU_AAH-4_1000_T-1.pickle",
            1,
            {
                "position": [130.91046689527747, -38.33872191154571],
                "orientation": 4.5736457771412145,
                "velocity": 8.733597598338127,
            },
            [
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                60.0,
                60.0,
                7.9603019648779325,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                0.0,
                0.0,
                52.039698035122065,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
        (
            "DEU_AAH-4_1000_T-1.pickle",
            2,
            {
                "position": [130.8706926599194, -38.625027045977326],
                "orientation": 4.575858469977114,
                "velocity": 8.568383732799687,
            },
            [
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                60.0,
                60.0,
                7.79517301,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                0.0,
                0.0,
                52.20482699,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
    ],
)
@unit_test
@functional
def test_get_surrounding_obstacles_lidar_elli(
    filename,
    time_step_gt,
    ego_vehicle_state_gt,
    prev_rel_pos_gt,
    p_rel_gt,
    p_rel_rate_gt,
):
    """
    Tests the `src/gym_commonroad/feature_extraction/surroundings_circle.py::get_surrounding_obstables()` method at the initial step
    """
    # Load meta scenario and problem dict
    meta_scenario_reset_dict_path = os.path.join(
        resource_root("test_surroundings/meta_scenario"),
        "meta_scenario_reset_dict.pickle",
    )
    with open(meta_scenario_reset_dict_path, "rb") as f:
        meta_scenario_reset_dict = pickle.load(f)
    problem_meta_scenario_dict_path = os.path.join(
        resource_root("test_surroundings/meta_scenario"),
        "problem_meta_scenario_dict.pickle",
    )
    with open(problem_meta_scenario_dict_path, "rb") as f:
        problem_meta_scenario_dict = pickle.load(f)

    # Load scenarios and problems
    fn = os.path.join(resource_root("test_surroundings/problem"), filename)
    with open(fn, "rb") as f:
        problem_dict = pickle.load(f)
    benchmark_id = filename.split(".")[0]

    # Set scenario and problem
    meta_scenario = problem_meta_scenario_dict[benchmark_id]
    meta_scenario_id = meta_scenario.benchmark_id
    obstacle_list = problem_dict["obstacle"]
    scenario = restore_scenario(meta_scenario, obstacle_list)
    planning_problem: PlanningProblem = random.choice(
        list(problem_dict["planning_problem_set"].planning_problem_dict.values())
    )

    reset_config = meta_scenario_reset_dict[meta_scenario_id]
    obstacle_lanelet_id_dict = problem_dict["obstacle_lanelet_id_dict"]
    connected_lanelet_dict = reset_config["connected_lanelet_dict"]

    # Create ground truth ego vehicle state
    ego_vehicle = Vehicle.create_vehicle({"vehicle_type": 2, "vehicle_model": 2})
    ego_vehicle.reset(planning_problem.initial_state, scenario.dt)
    ego_vehicle_state = ego_vehicle.state
    ego_vehicle_state.position = ego_vehicle_state_gt["position"]
    ego_vehicle_state.velocity = ego_vehicle_state_gt["velocity"]
    ego_vehicle_state.orientation = ego_vehicle_state_gt["orientation"]

    lanelet_polygons = [
        (l.lanelet_id, l.convert_to_polygon())
        for l in scenario.lanelet_network.lanelets
    ]
    lanelet_polygons_sg = pycrcc.ShapeGroup()
    for l_id, poly in lanelet_polygons:
        lanelet_polygons_sg.add_shape(create_collision_object(poly))

    ego_vehicle_lanelet_id = sorted_lanelets_by_state_realtime(
        scenario, ego_vehicle_state, lanelet_polygons, lanelet_polygons_sg
    )[0]
    ego_vehicle_lanelet = scenario.lanelet_network.find_lanelet_by_id(
        ego_vehicle_lanelet_id
    )

    max_lane_merge_range = 1000.0
    curvi_cosy, _ = get_local_curvi_cosy(
        scenario, ego_vehicle_lanelet_id, None, max_lane_merge_range
    )

    surrounding_dict = dict()
    lanelet_dict, all_lanelets_set = get_nearby_lanelet_id(
        connected_lanelet_dict, ego_vehicle_lanelet
    )
    surrounding_dict["lanelet_dict"] = lanelet_dict
    surrounding_dict["all_lanelets_set"] = all_lanelets_set

    # Test the method
    dummy_rel_vol = 0.0
    dummy_rel_pos = 60.0
    num_beams = 20
    lane_elli_sensor_range_semi_major_axis = 50.0
    lane_elli_sensor_range_semi_minor_axis = 30.0
    (
        p_rel,
        p_rel_rate,
        detected_obstacles,
        surrounding_beams,
    ) = get_surrounding_obstacles_lidar_elli(
        scenario.dynamic_obstacles,
        scenario.static_obstacles,
        ego_vehicle_state,
        time_step_gt,
        dummy_rel_vol,
        dummy_rel_pos,
        prev_rel_pos_gt,
        num_beams,
        lane_elli_sensor_range_semi_major_axis,
        lane_elli_sensor_range_semi_minor_axis,
    )

    # Check against ground truth
    assert np.allclose(p_rel, np.array(p_rel_gt))
    assert np.allclose(p_rel_rate, np.array(p_rel_rate_gt))
