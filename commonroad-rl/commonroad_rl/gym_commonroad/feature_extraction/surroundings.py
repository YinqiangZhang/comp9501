__author__ = "Brian Liao, Niels Muendler, Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

"""
Module for surrounding related feature extraction of the CommonRoad Gym envrionment
"""
import numpy as np
from typing import List, Tuple, Union, Dict, Set
from shapely.geometry import Polygon, Point, asLineString, LineString

from pycrccosy import CurvilinearCoordinateSystem

# from commonroad_rl.gym_commonroad.utils.ccosy_patch import CurvilinearCoordinateSystemPatch as CurvilinearCoordinateSystem
from commonroad.scenario.trajectory import State
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle
import commonroad_dc.pycrcc as pycrcc

from commonroad_rl.gym_commonroad.utils.scenario import approx_orientation_vector


def get_surrounding_obstacles_lane_rect(
    dyn_obstacles: List[DynamicObstacle],
    static_obstacles: List[StaticObstacle],
    obstacle_lanelet_id_dict: Dict[int, int],
    all_lanelets_set: Set[int],
    curvi_cosy: CurvilinearCoordinateSystem,
    lanelet_dict: Dict[str, List[int]],
    ego_vehicle_state: State,
    time_step: int,
    dummy_rel_vel: float,
    dummy_rel_pos: float,
    sensor_range_length: float,
    sensor_range_width: float,
) -> Tuple[List[float], List[float], List[DynamicObstacle or None], pycrcc.RectOBB]:
    """
    Get the lane-based relative velocities and positions of surrounding vehicles.
    This function finds all obstacles that are in a given rectangular area.
    The found surrounding vehicles will be mapped to the left, same and right adjacent lanelets.
    The nearest leading and nearest following vehicles will be considered as nearest surrounding in all these lanelets
    and the relative velocities, relative longitudinal positions over the local curvilinear coordinate system
    will be returned alongside with these obstacles and the considered surrounding area.
    :param dyn_obstacles: List of the dynamic obstacles in the scenario
    :param static_obstacles: List of the static obstacles in the scenario
    :param obstacle_lanelet_id_dict: Dictionary stores the lanelet ids of obstacles given their obstacle id
    :param all_lanelets_set: The set of all lanelet ids in the scenario
    :param curvi_cosy: The local ccosy to extract the longitudinal distances
    :param lanelet_dict: The lanelet dictionary
        stores the list of lanelet ids by given keywords as (ego_all, ego_right....)
    :param ego_vehicle_state: The current state of the ego vehicle
    :param time_step: The current time step
    :param dummy_rel_vel: Value to be returned as relative velocity if no obstacle can be found
    :param dummy_rel_pos: Value to be returned as relative position if no obstacle can be found
    :param sensor_range_length: The length of the surrounding area
    :param sensor_range_width: The width of the surrounding are
    :return: The relative velocities, relative longitudinal positions over the local curvilinear coordinate system
    will be returned alongside with these obstacles and the considered surrounding area
    """
    surrounding_area_ego_vehicle = pycrcc.RectOBB(
        sensor_range_length / 2,
        sensor_range_width / 2,
        ego_vehicle_state.orientation,
        ego_vehicle_state.position[0],
        ego_vehicle_state.position[1],
    )

    return get_surroundings_lane_based(
        dyn_obstacles,
        static_obstacles,
        obstacle_lanelet_id_dict,
        all_lanelets_set,
        curvi_cosy,
        lanelet_dict,
        ego_vehicle_state,
        time_step,
        dummy_rel_vel,
        dummy_rel_pos,
        surrounding_area_ego_vehicle,
    )


def get_surrounding_obstacles_lane_circ(
    dyn_obstacles: List[DynamicObstacle],
    static_obstacles: List[StaticObstacle],
    obstacle_lanelet_id_dict: dict,
    all_lanelets_set: set,
    curvi_cosy: CurvilinearCoordinateSystem,
    lanelet_dict: dict,
    ego_vehicle_state: State,
    time_step: int,
    dummy_rel_vel: float,
    dummy_rel_pos: float,
    sensor_range_radius: float,
) -> Tuple[List[float], List[float], List[DynamicObstacle or None], pycrcc.RectOBB]:
    """
    Get the lane-based relative velocities and positions of surrounding vehicles.
    This function finds all obstacles that are in a given circular area.
    The found surrounding vehicles will be mapped to the left, same and right adjacent lanelets.
    The nearest leading and nearest following vehicles will be considered as nearest surrounding in all these lanelets
    and the relative velocities, relative longitudinal positions over the local curvilinear coordinate system
    will be returned alongside with these obstacles and the considered surrounding area.
    :param dyn_obstacles: List of the dynamic obstacles in the scenario
    :param static_obstacles: List of the static obstacles in the scenario
    :param obstacle_lanelet_id_dict: Dictionary stores the lanelet ids of obstacles given their obstacle id
    :param all_lanelets_set: The set of all lanelet ids in the scenario
    :param curvi_cosy: The local ccosy to extract the longitudinal distances
    :param lanelet_dict: The lanelet dictionary
        stores the list of lanelet ids by given keywords as (ego_all, ego_right....)
    :param ego_vehicle_state: The current state of the ego vehicle
    :param time_step: The current time step
    :param dummy_rel_vel: Value to be returned as relative velocity if no obstacle can be found
    :param dummy_rel_pos: Value to be returned as relative position if no obstacle can be found
    :param sensor_range_radius: The radius of the surrounding area
    :return: The relative velocities relative longitudinal positions over the local curvilinear coordinate system
    will be returned alongside with these obstacles and the considered surrounding area
    """
    surrounding_area_ego_vehicle = pycrcc.Circle(
        sensor_range_radius,
        ego_vehicle_state.position[0],
        ego_vehicle_state.position[1],
    )

    return get_surroundings_lane_based(
        dyn_obstacles,
        static_obstacles,
        obstacle_lanelet_id_dict,
        all_lanelets_set,
        curvi_cosy,
        lanelet_dict,
        ego_vehicle_state,
        time_step,
        dummy_rel_vel,
        dummy_rel_pos,
        surrounding_area_ego_vehicle,
    )


def get_surroundings_lane_based(
    dyn_obstacles: List[DynamicObstacle],
    static_obstacles: List[StaticObstacle],
    obstacle_lanelet_id_dict: Dict[int, int],
    all_lanelets_set: Set[int],
    curvi_cosy: CurvilinearCoordinateSystem,
    lanelet_dict: Dict[str, List[int]],
    ego_vehicle_state: State,
    time_step: int,
    dummy_rel_vel: float,
    dummy_rel_pos: float,
    surrounding_area_ego_vehicle,
) -> Tuple[List[float], List[float], List[DynamicObstacle or None], pycrcc.RectOBB]:
    """
    Get the lane-based relative velocities and positions of surrounding vehicles.
    This function finds all obstacles that are in a given surrounding area.
    The found surrounding vehicles will be mapped to the left, same and right adjacent lanelets.
    The nearest leading and nearest following vehicles will be considered as nearest surrounding in all these lanelets
    and the relative velocities, relative longitudinal positions over the local curvilinear coordinate system
    will be returned alongside with these obstacles and the considered surrounding area.
    :param dyn_obstacles: List of the dynamic obstacles in the scenario
    :param static_obstacles: List of the static obstacles in the scenario
    :param obstacle_lanelet_id_dict: Dictionary stores the lanelet ids of obstacles given their obstacle id
    :param all_lanelets_set: The set of all lanelet ids in the scenario
    :param curvi_cosy: The local ccosy to extract the longitudinal distances
    :param lanelet_dict: The lanelet dictionary
        stores the list of lanelet ids by given keywords as (ego_all, ego_right....)
    :param ego_vehicle_state: The current state of the ego vehicle
    :param time_step: The current time step
    :param dummy_rel_vel: Value to be returned as relative velocity if no obstacle can be found
    :param dummy_rel_pos: Value to be returned as relative position if no obstacle can be found
    :param surrounding_area_ego_vehicle: The surrounding area
    :return: The relative velocities, relative longitudinal positions over the local curvilinear coordinate system
    will be returned alongside with these obstacles and the considered surrounding area
    """
    lanelet_ids, obstacle_states = get_obstacles_in_surrounding_area(
        dyn_obstacles,
        static_obstacles,
        obstacle_lanelet_id_dict,
        time_step,
        surrounding_area_ego_vehicle,
    )
    obstacle_lanelet, adj_obstacle_states = filter_obstacles_in_adj_lanelet(
        lanelet_ids,
        obstacle_states,
        all_lanelets_set,
    )
    v_rel, p_rel, detected_obstacle = get_rel_v_p_lane_based(
        obstacle_lanelet,
        adj_obstacle_states,
        curvi_cosy,
        lanelet_dict,
        ego_vehicle_state,
        dummy_rel_vel,
        dummy_rel_pos,
    )
    return v_rel, p_rel, detected_obstacle, surrounding_area_ego_vehicle


def get_surrounding_obstacles_lidar_elli(
    dyn_obstacles: List[DynamicObstacle],
    static_obstacles: List[StaticObstacle],
    ego_vehicle_state: State,
    time_step: int,
    dummy_distance_rate: float,
    dummy_distance: float,
    prev_distances: Union[np.ndarray, List[float]],
    num_beams: int,
    sensor_range_semi_major_axis: float,
    sensor_range_semi_minor_axis: float,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[float, float, float]]]:
    """
    Get the LiDAR-based relative distances and distance rates of surrounding vehicles.
    This function finds all obstacles that are in an circular surrounding area.
    The found surrounding vehicles will be checked whether the LiDAR beams with elliptical sensing area
    collide with them or not and the nearest distance will be determined in the direction of all beams.
    Using the previously measured distances, the distance rate will be returned as well alongside with the
    detection points and the surrounding area.
    :param dyn_obstacles: List of the dynamic obstacles in the scenario
    :param static_obstacles: List of the static obstacles in the scenario
    :param ego_vehicle_state: The current state of the ego vehicle
    :param time_step: The current time step
    :param dummy_distance_rate: Value to be returned as distance rate if no obstacle can be found
    :param dummy_distance: Value to be returned as distance if no obstacle can be found
    :param prev_distances: The list or array of previously measured distances
    :param num_beams: The number of beams to be used
    :param sensor_range_semi_major_axis: The major axis of the elliptical sensing area
    :param sensor_range_semi_minor_axis: The minor axis of the elliptical sensing area
    :return The distances, distance rates  alongside with the detection points and the surrounding area
    """
    surrounding_area_ego_vehicle = pycrcc.Circle(
        sensor_range_semi_major_axis,
        ego_vehicle_state.position[0],
        ego_vehicle_state.position[1],
    )
    # Create sensing area
    obstacle_shapes = get_obstacle_shapes_in_surrounding_area(
        dyn_obstacles,
        static_obstacles,
        time_step,
        surrounding_area_ego_vehicle,
    )

    # Create beam shapes (shapely line strings) around the ego vehicle, forming an ellipse sensing area as a whole
    surrounding_beams_ego_vehicle = []
    for i in range(num_beams):
        theta = i * (2 * np.pi / num_beams)
        x_delta = sensor_range_semi_major_axis * np.cos(theta)
        y_delta = sensor_range_semi_minor_axis * np.sin(theta)
        beam_start = ego_vehicle_state.position
        beam_length = np.sqrt(x_delta ** 2 + y_delta ** 2)
        beam_angle = ego_vehicle_state.orientation + theta
        surrounding_beams_ego_vehicle.append((beam_start, beam_length, beam_angle))

    obstacle_distances = get_obstacles_with_surrounding_beams(
        obstacle_shapes,
        ego_vehicle_state,
        surrounding_beams_ego_vehicle,
        dummy_distance,
    )

    distances, distance_rates, detection_points = get_distances_lidar_based(
        surrounding_beams_ego_vehicle,
        obstacle_distances,
        time_step,
        dummy_distance_rate,
        prev_distances,
    )
    return distances, distance_rates, detection_points, surrounding_beams_ego_vehicle


def get_obstacles_with_surrounding_beams(
    obstacle_shapes: List[Polygon],
    ego_vehicle_state: State,
    surrounding_beams: List[Tuple[float, float, float]],
    dummy_dist: float,
) -> np.ndarray:
    """
    Get the ditance to the nearest obstacles colliding with LIDAR beams
    :param obstacle_shapes: Obstacle shapes that detected with given sensing area
    :param ego_vehicle_state: State of ego vehicle
    :param surrounding_beams: List of beams as start, length and angle
    :param dummy_dist: Distance to be returned if no collision happened
    :return: List of obstacle states
    """
    # TODO: not iterate over all obstacles in scenario
    obstacle_distances = np.zeros(len(surrounding_beams))
    # For each beam, record all collisions with obstacles first, and report the one being closest to the ego vehicle
    ego_vehicle_center_shape = Point(ego_vehicle_state.position)
    for (i, (beam_start, beam_length, beam_angle)) in enumerate(surrounding_beams):
        beam_vec = approx_orientation_vector(beam_angle) * beam_length
        # asLineString recycles C-array as explained
        # beam = asLineString(np.array([beam_start, beam_start + beam_vec]))
        beam = LineString([beam_start, beam_start + beam_vec])
        obstacle_candidate = dummy_dist
        for obstacle_shape in obstacle_shapes:
            # TODO also support Shapegroups without shapely_object attribute
            if beam.intersects(obstacle_shape):
                dist = ego_vehicle_center_shape.distance(
                    beam.intersection(obstacle_shape)
                )
                if dist < obstacle_candidate:
                    obstacle_candidate = dist
        obstacle_distances[i] = obstacle_candidate
    return obstacle_distances


def get_obstacles_in_surrounding_area(
    dyn_obstacles: List[DynamicObstacle],
    static_obstacles: List[StaticObstacle],
    obstacle_lanelet_id_dict: dict,
    time_step: int,
    surrounding_area: pycrcc.Shape,
) -> Tuple[List[int], List[State]]:
    """
    Get the states and lanelet ids of all obstacles within the range of surrounding area of ego vehicle.
    :param dyn_obstacles: List of the dynamic obstacles in the scenario
    :param static_obstacles: List of the static obstacles in the scenario
    :param obstacle_lanelet_id_dict: Dictionary stores the lanelet ids of obstacles given their obstacle id
    :param time_step: Time step
    :param surrounding_area: Shapes of pycrcc classes
    :return: List of lanelet ids of obstacles, list of states obstacles
    """
    lanelet_ids = []
    obstacle_states = []

    for o in dyn_obstacles:
        if (
            o.initial_state.time_step
            <= time_step
            <= o.prediction.trajectory.final_state.time_step
        ):
            obstacle_state = o.state_at_time(time_step)
            obstacle_point = pycrcc.Point(
                obstacle_state.position[0], obstacle_state.position[1]
            )
            if surrounding_area.collide(obstacle_point):
                lanelet_ids.append(
                    obstacle_lanelet_id_dict[o.obstacle_id][
                        time_step - o.initial_state.time_step
                    ]
                )
                obstacle_states.append(obstacle_state)
    for o in static_obstacles:
        obstacle_state = o.initial_state
        obstacle_point = pycrcc.Point(
            obstacle_state.position[0], obstacle_state.position[1]
        )
        if surrounding_area.collide(obstacle_point):
            lanelet_ids.append(obstacle_lanelet_id_dict[o.obstacle_id][0])
            obstacle_states.append(obstacle_state)

    return lanelet_ids, obstacle_states


def get_obstacle_shapes_in_surrounding_area(
    dyn_obstacles: List[DynamicObstacle],
    static_obstacles: List[StaticObstacle],
    time_step: int,
    surrounding_area: pycrcc.Shape,
) -> List[Polygon]:
    """
    Get the occupancy shape and states and lanelet ids of all obstacles
    within the range of surrounding area of ego vehicle.
    :param dyn_obstacles: List of the dynamic obstacles in the scenario
    :param static_obstacles: List of the static obstacles in the scenario
    :param time_step: Time step
    :param surrounding_area: Shapes of pycrcc classes
    :return: List of obstacle shapely shapes
    """
    obstacle_shapes = []

    for o in dyn_obstacles:
        if (
            o.initial_state.time_step
            <= time_step
            <= o.prediction.trajectory.final_state.time_step
        ):
            obstacle_state = o.state_at_time(time_step)
            obstacle_point = pycrcc.Point(
                obstacle_state.position[0], obstacle_state.position[1]
            )
            if surrounding_area.collide(obstacle_point):
                obstacle_shapes.append(
                    o.occupancy_at_time(time_step).shape.shapely_object
                )
    for o in static_obstacles:
        obstacle_state = o.initial_state
        obstacle_point = pycrcc.Point(
            obstacle_state.position[0], obstacle_state.position[1]
        )
        if surrounding_area.collide(obstacle_point):
            obstacle_shapes.append(o.occupancy_at_time(time_step).shape.shapely_object)

    return obstacle_shapes


def filter_obstacles_in_adj_lanelet(
    lanelet_ids: List[int], states: List[State], all_lanelets_set: Set[int]
) -> Tuple[List[int], List[State]]:
    """
    Get all obstacles in the proximity of ego vehicle and in the adj lanelet.
    :param lanelet_ids: List of lanelet ids of obstacles
    :param states: List of states of obstacles
    :param all_lanelets_set: The set of all lanelet ids in the scenario
    :return: The list of lanelets of obstacles, the list of states
    """
    obstacle_state = []
    obstacle_lanelet = []
    for lanelet_id, state in zip(lanelet_ids, states):
        if lanelet_id in all_lanelets_set:  # Check if the obstacle is in adj lanelets
            obstacle_lanelet.append(lanelet_id)
            obstacle_state.append(state)
    return obstacle_lanelet, obstacle_state


def get_distances_lidar_based(
    beams: List[Tuple[float, float, float]],
    obstacle_distances: Union[List[float], np.ndarray],
    time_step: int,
    dummy_distances_rate: float,
    prev_distances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Get the distances and positions and their change rates of obstacles in the lidar-based fashion.
    :param beams: List of beams as start, length and angle
    :param obstacle_distances: List or array of obstacle distances
    :param time_step: Time step
    :param dummy_distances_rate: Distance rate to be returned if no collision happened
    :param prev_distances: List of distances in previous time step
    :return: Lists of distances and their change rates and the points of collision
    """
    # Relative positions
    dists = np.array(obstacle_distances)

    # Change rates
    if time_step == 0:
        dist_rates = np.full(len(beams), dummy_distances_rate)
    else:
        dist_rates = np.array([prev_distances[i] - dists[i] for i in range(len(beams))])

    # detection points
    detection_points = [
        beam_start + approx_orientation_vector(beam_angle) * dists[i]
        for i, ((beam_start, _, beam_angle), closest_collision) in enumerate(
            zip(beams, obstacle_distances)
        )
    ]

    return dists, dist_rates, detection_points


def get_rel_v_p_lane_based(
    obstacles_lanelet_ids: List[int],
    obstacle_states: List[State],
    curvi_cosy: CurvilinearCoordinateSystem,
    lanelet_dict: Dict[str, List[int]],
    ego_vehicle_state: State,
    dummy_rel_vel: float,
    dummy_rel_pos: float,
) -> Tuple[list, list, list]:
    """
    Get the relative velocity and position of obstacles in adj left, adj right and ego lanelet.
    In each lanelet, compute only the nearest leading and following obstacles.
    :param obstacles_lanelet_ids: The list of lanelets of obstacles
    :param obstacle_states: The list of states of obstacles
    :param curvi_cosy: The local ccosy to extract the longitudinal distances
    :param lanelet_dict: The lanelet dictionary
        stores the list of lanelet ids by given keywords as (ego_all, ego_right....)
    :param ego_vehicle_state: The state of the ego vehicle
    :param dummy_rel_vel: Value to be returned as relative velocity if no obstacle can be found
    :param dummy_rel_pos: Value to be returned as relative position if no obstacle can be found
    :return: Relative velocities, relative positions, and detected obstacle states
    """
    # Initialize dummy values, in case no obstacles are present
    v_rel_left_follow, v_rel_same_follow, v_rel_right_follow = (
        dummy_rel_vel,
        dummy_rel_vel,
        dummy_rel_vel,
    )
    v_rel_left_lead, v_rel_same_lead, v_rel_right_lead = (
        dummy_rel_vel,
        dummy_rel_vel,
        dummy_rel_vel,
    )

    p_rel_left_follow, p_rel_same_follow, p_rel_right_follow = (
        dummy_rel_pos,
        dummy_rel_pos,
        dummy_rel_pos,
    )
    p_rel_left_lead, p_rel_same_lead, p_rel_right_lead = (
        dummy_rel_pos,
        dummy_rel_pos,
        dummy_rel_pos,
    )

    try:
        ego_vehicle_long_position, _ = curvi_cosy.convert_to_curvilinear_coords(
            ego_vehicle_state.position[0], ego_vehicle_state.position[1]
        )

        (
            o_left_follow,
            o_left_lead,
            o_right_follow,
            o_right_lead,
            o_same_follow,
            o_same_lead,
        ) = (None, None, None, None, None, None)

        for o_state, o_lanelet_id in zip(obstacle_states, obstacles_lanelet_ids):
            try:
                o_curvi_long_position, _ = curvi_cosy.convert_to_curvilinear_coords(
                    o_state.position[0], o_state.position[1]
                )
            except ValueError:  # the position is out of project area of curvilinear coordinate system
                o_curvi_long_position = ego_vehicle_long_position + dummy_rel_pos

            distance_abs = np.abs(ego_vehicle_long_position - o_curvi_long_position)
            distance_sign = np.sign(
                ego_vehicle_long_position - o_curvi_long_position
            )  # positive if following

            if o_lanelet_id in lanelet_dict["ego_all"]:  # ego lanelet
                if (
                    distance_sign == 1 and distance_abs < p_rel_same_follow
                ):  # following vehicle, distance is smaller
                    v_rel_same_follow = (
                        ego_vehicle_state.velocity
                        - o_state.velocity
                        * np.cos(o_state.orientation - ego_vehicle_state.orientation)
                    )
                    p_rel_same_follow = distance_abs
                    o_same_follow = o_state
                elif (
                    distance_sign != 1 and distance_abs < p_rel_same_lead
                ):  # leading vehicle, distance is smaller
                    v_rel_same_lead = (
                        o_state.velocity
                        * np.cos(o_state.orientation - ego_vehicle_state.orientation)
                        - ego_vehicle_state.velocity
                    )
                    p_rel_same_lead = distance_abs
                    o_same_lead = o_state

            if o_lanelet_id in lanelet_dict["right_all"]:  # right lanelet
                if distance_sign == 1 and distance_abs < p_rel_right_follow:
                    v_rel_right_follow = (
                        ego_vehicle_state.velocity
                        - o_state.velocity
                        * np.cos(o_state.orientation - ego_vehicle_state.orientation)
                    )
                    p_rel_right_follow = distance_abs
                    o_right_follow = o_state

                elif distance_sign != 1 and distance_abs < p_rel_right_lead:
                    v_rel_right_lead = (
                        o_state.velocity
                        * np.cos(o_state.orientation - ego_vehicle_state.orientation)
                        - ego_vehicle_state.velocity
                    )
                    p_rel_right_lead = distance_abs
                    o_right_lead = o_state

            if o_lanelet_id in lanelet_dict["left_all"]:  # left lanelet
                if distance_sign == 1 and distance_abs < p_rel_left_follow:
                    v_rel_left_follow = (
                        ego_vehicle_state.velocity
                        - o_state.velocity
                        * np.cos(o_state.orientation - ego_vehicle_state.orientation)
                    )
                    p_rel_left_follow = distance_abs
                    o_left_follow = o_state
                elif distance_sign != 1 and distance_abs < p_rel_left_lead:
                    v_rel_left_lead = (
                        o_state.velocity
                        * np.cos(o_state.orientation - ego_vehicle_state.orientation)
                        - ego_vehicle_state.velocity
                    )
                    p_rel_left_lead = distance_abs
                    o_left_lead = o_state

        detected_obstacle = [
            o_left_follow,
            o_same_follow,
            o_right_follow,
            o_left_lead,
            o_same_lead,
            o_right_lead,
        ]

    except ValueError:
        detected_obstacle = [None] * 6

    v_rel = [
        v_rel_left_follow,
        v_rel_same_follow,
        v_rel_right_follow,
        v_rel_left_lead,
        v_rel_same_lead,
        v_rel_right_lead,
    ]
    p_rel = [
        p_rel_left_follow,
        p_rel_same_follow,
        p_rel_right_follow,
        p_rel_left_lead,
        p_rel_same_lead,
        p_rel_right_lead,
    ]
    return v_rel, p_rel, detected_obstacle
