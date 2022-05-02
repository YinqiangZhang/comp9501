__author__ = "Brian Liao, Niels Muendler, Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

"""
Module for scenario related helper methods for the CommonRoad Gym environment
"""
import numpy as np
from scipy import spatial
from typing import Tuple, List, Union, Dict
from shapely.geometry import LineString, Point, Polygon

import commonroad_dc.pycrcc as pycrcc

from pycrccosy import CurvilinearCoordinateSystem

from commonroad.common.solution import VehicleModel, VehicleType
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_ccosy.geometry.util import resample_polyline

from commonroad_dc.feasibility.feasibility_checker import trajectory_feasibility
from commonroad_dc.feasibility.vehicle_dynamics import (
    VehicleDynamics,
    VehicleParameters,
)
from commonroad_route_planner.route_planner import sorted_lanelet_ids
from commonroad_rl.gym_commonroad.feature_extraction.goal import GoalObservation


def extrapolate_resample_polyline(
    polyline: np.ndarray, step: float = 2.0
) -> np.ndarray:
    """
    Current ccosy (https://gitlab.lrz.de/cps/commonroad-curvilinear-coordinate-system/-/tree/development) creates
    wrong projection domain if polyline has large distance between waypoints --> resampling;
    initial and final points are not within projection domain -> extrapolation
    :param polyline: polyline to be used to create ccosy
    :param step: step distance for resampling
    :return: extrapolated and resampled polyline
    """
    p = np.poly1d(np.polyfit(polyline[:2, 0], polyline[:2, 1], 1))

    x = 2 * polyline[0, 0] - polyline[1, 0]
    a = np.array([[x, p(x)]])
    polyline = np.concatenate((a, polyline), axis=0)

    # extrapolate final point
    p = np.poly1d(np.polyfit(polyline[-2:, 0], polyline[-2:, 1], 1))

    # x = 2 * polyline[-1, 0] - polyline[-2, 0]
    # TODO: extend more for learning
    x = 100 * polyline[-1, 0] - 99 * polyline[-2, 0]
    a = np.array([[x, p(x)]])
    polyline = np.concatenate((polyline, a), axis=0)

    return resample_polyline(polyline, step=step)


def create_coordinate_system_from_polyline(
    polyline: np.ndarray,
) -> CurvilinearCoordinateSystem:
    """
    Create curvilinear coordinate system from polyline
    :param polyline: The polyline to be converted
    :return: The converted curvilinar coordinate system
    """
    new_polyline = extrapolate_resample_polyline(polyline, step=1.0)

    ccosy = CurvilinearCoordinateSystem(new_polyline)

    projection_error = True
    while projection_error:
        try:
            ccosy.convert_to_curvilinear_coords(polyline[0][0], polyline[0][1])
            ccosy.convert_to_curvilinear_coords(polyline[-1][0], polyline[-1][1])
            projection_error = False
        except ValueError:
            new_polyline = extrapolate_resample_polyline(new_polyline, step=1.0)
            ccosy = CurvilinearCoordinateSystem(new_polyline)

    return ccosy


def get_road_edge(scenario) -> Tuple[dict, dict, dict, dict]:
    """
    Get the road edge or solid white line of a lanelet.
    :return: Dictionary of left and right lanelet ids and road edge
    """
    left_road_edge_lanelet_id = {}
    right_road_edge_lanelet_id = {}
    left_road_edge = {}
    right_road_edge = {}
    for lanelet in scenario.lanelet_network.lanelets:
        if lanelet.lanelet_id not in right_road_edge_lanelet_id.keys():
            # Search the right most lanelet and use right vertices as right bound
            start = lanelet
            temp = []
            while start.adj_right_same_direction:
                temp.append(start.lanelet_id)
                start = scenario.lanelet_network.find_lanelet_by_id(start.adj_right)
            temp.append(start.lanelet_id)
            right_bound = LineString(start.right_vertices)
            right_road_edge_lanelet_id.update({k: start.lanelet_id for k in temp})
            right_road_edge[start.lanelet_id] = right_bound
        if lanelet.lanelet_id not in left_road_edge_lanelet_id.keys():
            start = lanelet
            temp = []
            # Search the left most lanelet and use right vertices as left bound
            while start.adj_left_same_direction:
                temp.append(start.lanelet_id)
                start = scenario.lanelet_network.find_lanelet_by_id(start.adj_left)
            temp.append(start.lanelet_id)
            left_bound = LineString(start.left_vertices)
            left_road_edge_lanelet_id.update({k: start.lanelet_id for k in temp})
            left_road_edge[start.lanelet_id] = left_bound
    return (
        left_road_edge_lanelet_id,
        left_road_edge,
        right_road_edge_lanelet_id,
        right_road_edge,
    )


def get_ego_lanelet_orientation(
    ego_vehicle_state: State, ego_vehicle_lanelet: Lanelet
) -> float:
    """
    Get the approximate orientation of the lanelet.
    :param ego_vehicle_state: The state of the ego vehicle
    :param ego_vehicle_lanelet: Lanelet of the ego vehicle
    :return: orientation (rad) of the lanelet
    """
    # TODO: This method could make use of the general relative orientation function or could be fully replaced by
    #  the functions moved to the route planner

    pt = np.asarray(ego_vehicle_state.position)
    center_vertices = ego_vehicle_lanelet.center_vertices
    idx = spatial.KDTree(center_vertices).query(pt)[1]
    if idx < pt.shape[0] - 1:
        orientation = np.arccos(
            (center_vertices[idx + 1, 0] - center_vertices[idx, 0])
            / np.linalg.norm(center_vertices[idx + 1] - center_vertices[idx])
        )
        sign = np.sign(center_vertices[idx + 1, 1] - center_vertices[idx, 1])
    else:
        orientation = np.arccos(
            (center_vertices[idx, 0] - center_vertices[idx - 1, 0])
            / np.linalg.norm(center_vertices[idx] - center_vertices[idx - 1])
        )
        sign = np.sign(center_vertices[idx, 1] - center_vertices[idx - 1, 1])
    if sign >= 0:
        orientation = np.abs(orientation)
    else:
        orientation = -np.abs(orientation)
    # orientation = shift_orientation(orientation)
    return orientation


def get_lane_marker(ego_vehicle_lanelet: Lanelet) -> Tuple[LineString, LineString]:
    """
    Get the lane marker for the desired lanelet.
    :param ego_vehicle_lanelet: lanelet of ego vehicle
    :return: left and right lane marker
    """
    left_marker_line = LineString(ego_vehicle_lanelet.left_vertices)
    right_marker_line = LineString(ego_vehicle_lanelet.right_vertices)
    return left_marker_line, right_marker_line


def shift_orientation(orientation: float) -> float:
    """
    Shift the orientation to the range of [-pi, pi[
    """
    x = np.cos(orientation)
    y = np.sin(orientation)
    new_orientation = np.arctan2(y, x)
    return new_orientation


def get_marker_distance(
    ego_vehicle_state: State,
    left_marker_line: LineString,
    right_marker_line: LineString,
) -> Tuple[float, float]:
    """
    Compute distance to lane marker.
    :param ego_vehicle_state: state of the ego vehicle
    :param left_marker_line: left marker line
    :param right_marker_line: right marker line
    :return: distance to left and right marker
    """
    p = Point(ego_vehicle_state.position[0], ego_vehicle_state.position[1])
    nearest_point_left = left_marker_line.interpolate(left_marker_line.project(p))
    distance_left_marker = nearest_point_left.distance(p)

    nearest_point_right = right_marker_line.interpolate(right_marker_line.project(p))
    distance_right_marker = nearest_point_right.distance(p)
    return distance_left_marker, distance_right_marker


def get_lane_relative_heading(
    ego_vehicle_state: State, ego_vehicle_lanelet: Lanelet
) -> float:
    """
    Get the heading angle in the Frenet frame.
    :param ego_vehicle_state: state of ego vehicle
    :param ego_vehicle_lanelet: lanelet of ego vehicle
    :return: heading angle in frenet coordinate system relative to lanelet center vertices between -pi and pi
    """
    lanelet_angle = get_ego_lanelet_orientation(ego_vehicle_state, ego_vehicle_lanelet)
    angle_vec = approx_orientation_vector(lanelet_angle)
    return angle_diff(
        angle_vec, approx_orientation_vector(ego_vehicle_state.orientation)
    )


def related_lanelets_by_state_realtime(
    state: State,
    lanelet_polygons: List[Tuple[int, Polygon]],
    lanelet_polygons_sg: pycrcc.ShapeGroup,
) -> List[int]:
    """
    Get the lanelet of a state.
    :param state: The state to which the related lanelets should be found
    :param lanelet_polygons: The polygons of the lanelets
    :param lanelet_polygons_sg: The pycrcc polygons of the lanelets
    :return: The list of lanelet ids
    """
    position = state.position

    # output list
    res = list()

    # look at each lanelet
    point_list = [position]

    point_sg = pycrcc.ShapeGroup()
    for el in point_list:
        point_sg.add_shape(pycrcc.Point(el[0], el[1]))

    lanelet_polygon_ids = point_sg.overlap_map(lanelet_polygons_sg)

    for lanelet_id_list in lanelet_polygon_ids.values():
        for lanelet_id in lanelet_id_list:
            res.append(lanelet_polygons[lanelet_id][0])

    return res


def sorted_lanelets_by_state_realtime(
    scenario: Scenario,
    state: State,
    lanelet_polygons: list,
    lanelet_polygons_sg: pycrcc.ShapeGroup,
) -> List[int]:
    """
    Returns the sorted list of lanelet ids which correspond to a given state
    :param scenario: The scenario to be used
    :param state: The state which lanelets ids are searched
    :param lanelet_polygons: The polygons of the lanelets
    :param lanelet_polygons_sg: Thy pycrcc polygons of the lanelets
    :return: The list of lanelet ids sorted by relative orientations, the nearest lanelet is the first elements
    """
    return sorted_lanelet_ids(
        related_lanelets_by_state_realtime(
            state, lanelet_polygons, lanelet_polygons_sg
        ),
        state.orientation,
        state.position,
        scenario,
    )


def get_local_curvi_cosy(
    scenario: Scenario,
    ego_vehicle_lanelet_id: int,
    ref_path_dict: Dict[str, Tuple[np.ndarray or None, Lanelet or None]],
    max_lane_merge_range: float,
) -> Tuple[CurvilinearCoordinateSystem, Lanelet]:
    """
    At every time step, update the local curvilinear coordinate system from the dict.
    :param scenario: The scenario to be used
    :param ego_vehicle_lanelet_id: The lanelet id where the ego vehicle is on
    :param ref_path_dict: The dictionary of the reference path, contains the paths by the starting lanelet ids
    :param max_lane_merge_range: Maximum range of lanes to be merged
    :return: Curvilinear coordinate system of the merged lanelets
    """
    if ref_path_dict is None:
        ref_path_dict = dict()

    lanelet_network = scenario.lanelet_network
    ref_path, ref_merged_lanelet = ref_path_dict.get(
        ego_vehicle_lanelet_id, (None, None)
    )

    if ref_path is None:

        for (
            lanelet
        ) in lanelet_network.lanelets:  # iterate in all lanelet in this scenario
            if lanelet.lanelet_id == ego_vehicle_lanelet_id and (
                not lanelet.predecessor and not lanelet.successor
            ):  # the lanelet is a lane itself
                ref_path = lanelet.center_vertices
            elif (
                not lanelet.predecessor
            ):  # the lanelet is the start of a lane, the lane can be created from here
                # TODO: cache merged lanelets in pickle or dict
                (
                    merged_lanelet_list,
                    sub_lanelet_ids_list,
                ) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                    lanelet, scenario.lanelet_network, max_lane_merge_range
                )
                for merged_lanelet, sub_lanelet_ids in zip(
                    merged_lanelet_list, sub_lanelet_ids_list
                ):
                    # print(f"sub_lanelet_ids={sub_lanelet_ids}")
                    if ego_vehicle_lanelet_id in sub_lanelet_ids:
                        ref_path = merged_lanelet.center_vertices
                        ref_merged_lanelet = merged_lanelet
                        break
        # TODO: Idea, the reference path dict could be updated on all successor of the current lanelet for optimization
        ref_path_dict[ego_vehicle_lanelet_id] = (ref_path, ref_merged_lanelet)

    curvi_cosy = create_coordinate_system_from_polyline(ref_path)
    return curvi_cosy, ref_merged_lanelet


def get_nearby_lanelet_id(
    connected_lanelet_dict: dict, ego_vehicle_lanelet: Lanelet
) -> Tuple[dict, set]:
    """
    Get all nearby lanelets ids. Nearby means adjacent left lanelet, its predecessors and successors, so for
    adjacent right lanelet and ego lanelet.
    :param connected_lanelet_dict: A dict with its keys as lanelet id and values as connected lanelet ids
    :param ego_vehicle_lanelet: The list lanelets of the ego vehicle
    :return: A dict of nearby lanelets ids and the set of all nearby lanelets ids.
    """
    keys = {
        "ego",
        "left",
        "right",
        "ego_other",
        "left_other",
        "right_other",
        "ego_all",
        "left_all",
        "right_all",
    }
    lanelet_dict = {key: set() for key in keys}
    ego_vehicle_lanelet_id = ego_vehicle_lanelet.lanelet_id
    lanelet_dict["ego"].add(ego_vehicle_lanelet_id)  # Add ego lanelet

    for predecessor_lanelet_id in ego_vehicle_lanelet.predecessor:
        lanelet_dict["ego_other"].update(connected_lanelet_dict[predecessor_lanelet_id])
    for successor_lanelet_id in ego_vehicle_lanelet.successor:
        lanelet_dict["ego_other"].update(connected_lanelet_dict[successor_lanelet_id])

    if (
        ego_vehicle_lanelet.adj_right_same_direction is True
    ):  # Get adj right lanelet with same direction
        lanelet_dict["right"].add(ego_vehicle_lanelet.adj_right)
    if (
        ego_vehicle_lanelet.adj_left_same_direction is True
    ):  # Get adj left lanelet with same direction
        lanelet_dict["left"].add(ego_vehicle_lanelet.adj_left)

    for ego_lanelet_id in lanelet_dict["ego"]:
        lanelet_dict["ego_other"].update(connected_lanelet_dict[ego_lanelet_id])
    for left_lanelet_id in lanelet_dict["left"]:
        lanelet_dict["left_other"].update(connected_lanelet_dict[left_lanelet_id])
    for r in lanelet_dict["right"]:
        lanelet_dict["right_other"].update(connected_lanelet_dict[r])

    lanelet_dict["ego_all"] = set().union(
        set(lanelet_dict["ego"]), set(lanelet_dict["ego_other"])
    )

    lanelet_dict["left_all"] = set().union(
        set(lanelet_dict["left"]), set(lanelet_dict["left_other"])
    )
    lanelet_dict["right_all"] = set().union(
        set(lanelet_dict["right"]), set(lanelet_dict["right_other"])
    )

    all_lanelets_set = set().union(
        lanelet_dict["ego_all"], lanelet_dict["left_all"], lanelet_dict["right_all"]
    )
    return lanelet_dict, all_lanelets_set


def get_relative_future_goal_offsets(
    goal: GoalObservation,
    state: State,
    step_parameter: List[float],
    static: bool = False,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Get the relative offset of current and future positions from center vertices. Positive if left.
    For a given static extrapolation, the future position at "static" m/s after step_parameter seconds is given.
    For static = True this means the future position in exactly step_parameter meters.
    Otherwise for static = False, the future position at the current velocity after step_parameter seconds is given.
    :param goal: The goal observation
    :param state: State of ego vehicle
    :param step_parameter: Parameter of evaluating the future position, see description above
    :param static: Curvilinear coordinate system
    :return: Offset of step_parameter future positions as well as the positions themselves
    """
    v = approx_orientation_vector(state.orientation) * (
        state.velocity if static is False else 1.0
    )

    # quadratic steps may make sense based on the fact that braking distance is
    # proportional to the velocity squared (https://en.wikipedia.org/wiki/Braking_distance)
    positions = [state.position + (v * i) for i in step_parameter]
    lat_offsets = [goal.get_long_lat_distance_to_goal(p)[1] for p in positions]

    return lat_offsets, positions


def get_relative_offset(
    curvi_cosy: CurvilinearCoordinateSystem, position: np.ndarray
) -> float:
    """
    Get the relative offset of ego vehicle from center vertices. Positive if left.
    :param curvi_cosy: curvilinear coordinate system
    :param position: The position of the ego vehicle
    :return: offset
    """
    try:
        _, ego_vehicle_lat_position = curvi_cosy.convert_to_curvilinear_coords(
            position[0], position[1]
        )
    except ValueError:
        ego_vehicle_lat_position = np.nan

    return ego_vehicle_lat_position


def interpolate_steering_angles(
    state_list: List[State], parameters: VehicleParameters, dt: float
) -> List[State]:
    """
    Interpolates the not defined steering angles based on KS Model
    :param state_list: The list of the states
    :param parameters: The parameters of the vehicle
    :param dt: dt of the scenario
    :return: The state list with interpolated steering angles
    """
    if len(state_list) == 0:
        return state_list

    l_wb = parameters.a + parameters.b

    [orientations, velocities] = np.array(
        [[state.orientation, state.velocity] for state in state_list]
    ).T

    orientation_vectors = approx_orientation_vector(orientations)
    psi_dots = (
        angle_diff(orientation_vectors[:, :-1].T, orientation_vectors[:, 1:].T) / dt
    )
    avg_velocities = np.mean(np.array([velocities[:-1], velocities[1:]]), axis=0)
    avg_velocities[avg_velocities == 0.0] += np.finfo(float).eps

    steering_angles = np.arctan(psi_dots * l_wb / avg_velocities)
    if len(steering_angles) > 0:
        steering_angles = np.hstack((steering_angles, steering_angles[-1]))
    else:
        default_steering_angle = 0.0
        steering_angles = np.array([default_steering_angle])

    steering_angles = np.clip(
        steering_angles, parameters.steering.min, parameters.steering.max
    )

    def get_state_with_steering_angle(state: State, steering_angle: float):
        if hasattr(state, "steering_angle"):
            if state.steering_angle is None:
                state.steering_angle = steering_angle
        else:
            setattr(state, "steering_angle", steering_angle)
        return state

    return list(
        map(
            lambda state, steering_angle: get_state_with_steering_angle(
                state, steering_angle
            ),
            state_list,
            steering_angles,
        )
    )


def interpolate_steering_angles_of_obstacle(
    obstacle: DynamicObstacle, parameters: VehicleParameters, dt: float
):
    """
    Interpolates the not defined steering angles of obstacle based on KS Model
    :param obstacle:
    :param parameters: The parameters of the vehicle
    :param dt: dt of the scenario
    """
    trajectory = obstacle.prediction.trajectory
    trajectory.state_list = interpolate_steering_angles(
        trajectory.state_list, parameters, dt
    )
    obstacle.initial_state.steering_angle = trajectory.state_list[0].steering_angle


def check_trajectory(
    obstacle: DynamicObstacle,
    vehicle_model: VehicleModel,
    vehicle_type: VehicleType,
    dt: float,
) -> bool:
    """
    Checks whether the trajectory of a given obstacle is feasible with a given vehicle model
    Note: Currently it is implemented for the KS model. As soon as the workaround is not needed,
    it can be rebased to fully use the Feasibility Checker

    :param obstacle: The obstacle which trajectory should be checked
    :param vehicle_model: The used vehicle model
    :param vehicle_type: THe type of the vehicle
    :param dt: Delta time of the simulation
    :return: True if the trajectory is feasible
    """
    trajectory = obstacle.prediction.trajectory
    vehicle_dynamics = VehicleDynamics.from_model(vehicle_model, vehicle_type)

    position_tolerance = 0.1
    orientation_tolerance = 2e-2

    e = np.array([position_tolerance, position_tolerance, orientation_tolerance])

    feasible, _ = trajectory_feasibility(trajectory, vehicle_dynamics, dt, e=e)
    return feasible


def get_distance_point_to_linestring(p: Point, line: LineString) -> float:
    """
    Get the distance of a point to the given line
    :param p: The point
    :param line: The line
    :return: The distance between the point and the line
    """
    nearest_point = line.interpolate(line.project(p))
    return nearest_point.distance(p)


def get_distance_to_marker_and_road_edge(
    ego_vehicle_state: State,
    left_marker_line: LineString,
    right_marker_line: LineString,
    left_road_edge: LineString,
    right_road_edge: LineString,
) -> Tuple[float, float, float, float]:
    """
    Get the distane to lane markers and the road edge
    :param ego_vehicle_state: The state of the ego vehicle
    :param left_marker_line: The left marker line
    :param right_marker_line: The right marker line
    :param left_road_edge: The left road edge
    :param right_road_edge: The right road edge
    :return: Tuple of the distances to the left marker, right marker, left road edge and right road edge
    """
    ego_vehicle_point = Point(
        ego_vehicle_state.position[0], ego_vehicle_state.position[1]
    )
    distance_left_marker = get_distance_point_to_linestring(
        ego_vehicle_point, left_marker_line
    )
    distance_right_marker = get_distance_point_to_linestring(
        ego_vehicle_point, right_marker_line
    )
    distance_left_road_edge = get_distance_point_to_linestring(
        ego_vehicle_point, left_road_edge
    )
    distance_right_road_edge = get_distance_point_to_linestring(
        ego_vehicle_point, right_road_edge
    )
    return (
        distance_left_marker,
        distance_right_marker,
        distance_left_road_edge,
        distance_right_road_edge,
    )


def get_safe_distance(
    ego_vehicle_vel: float,
    rel_vel: float,
    acc_max: float,
    react_time: float,
    is_ego_leading: bool,
) -> float:
    """
    Heuristics for determining safe distance to keep
    :param ego_vehicle_vel: The velocity of teh ego vehicle
    :param rel_vel: The relative velocity between the ego and obstacle
    :param acc_max: The maximal allowed acceleration
    :param react_time: The reaction time
    :param is_ego_leading: Indicates whether the ego is the leading vehicle or not
    :return: The safe distance which should be kept
    """
    # TODO: This function is not used anywhere in the project, could be removed, but left here for probable later usage
    #  Remove it if unnecessary or could be moved to a separate obsolete module
    if is_ego_leading:
        # ego vehicle is leading
        safe_distance = ((ego_vehicle_vel - rel_vel) ** 2 - ego_vehicle_vel ** 2) / (
            2 * acc_max
        ) + react_time * (ego_vehicle_vel - rel_vel)
    else:
        # ego vehicle is following
        safe_distance = (ego_vehicle_vel ** 2 - (ego_vehicle_vel + rel_vel) ** 2) / (
            2 * acc_max
        ) + react_time * ego_vehicle_vel
    return safe_distance


def approx_orientation_vector(orientation: Union[float, np.ndarray]) -> np.ndarray:
    """
    Approximate normed vector in a given orientation
    :param orientation: The orientation
    :return Normalized vector points to the defined orientation
    """
    return np.array([np.cos(orientation), np.sin(orientation)])


def angle_diff(vector_from: np.ndarray, vector_to: np.ndarray):
    """
    Returns angle between the two provided vectors, from v1 to v2
    :param vector_from: Vector from the angle should be measured
    :param vector_to: Vector to the angle should be measured
    :return: Signed relative angle between the two vectors
    """
    assert vector_from.ndim <= 2 and vector_to.ndim <= 2
    if vector_from.ndim == 1:
        vector_from = vector_from[None]
    if vector_to.ndim == 1:
        vector_to = vector_to[None]
    dot_product = np.einsum("ij,ij->i", vector_from, vector_to)
    determinant = np.einsum(
        "ij,ij->i", vector_from, (np.flip(vector_to, axis=-1) * (np.array([1, -1])))
    )
    return np.arctan2(determinant, dot_product)


def abs_angle_diff(v1: np.ndarray, v2: np.ndarray):
    """
    Returns absolute angle between the two provided vectors
    :param v1: Vector one
    :param v2: Vecotr two
    :return: The absolute angle between the two vectors
    """
    assert v1.ndim <= 2 and v2.ndim <= 2
    if v1.ndim == 1:
        v1 = v1[None]
    if v2.ndim == 1:
        v2 = v2[None]
    dot_product = np.einsum("ij,ij->i", v1, v2)
    return np.arccos(dot_product)

# new added for integrating discrete actions
def create_forks_for_lanelet_network(scenario: Scenario, from_left: bool=True) -> Dict:
    """
    Searches nearest forward fork roads for each lanelet
    :param scenario: the scenario of lanelet network
    :param from_left: results are sorted from left or right
    :return: The nearest fork dictionary of each lanelet
    """
    direction_list = dict()
    for l in scenario.lanelet_network.lanelets:
        direction_list[l.lanelet_id] = get_nearest_lanelet_fork(scenario, l.lanelet_id, from_left)
    
    return direction_list


def get_lanelet_orientation(scenario: Scenario, lanelet_id: int) -> float:
    """
    Gets the mean orientation of the lanelet
    :param scenario: the scenario of lanelet
    :param lanelet_id: id of the target lanelet
    :return: orientation (between beginning and ending center vertices) in the range of [-pi,pi]
    """
    center_vertices = scenario.lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices
    x0, y0 = center_vertices[0]
    xn, yn = center_vertices[-1]
    # calculate the arccos
    orientation = np.arccos((xn - x0) / np.linalg.norm(center_vertices[-1] - center_vertices[0]))
    # delta_y decides the sign of theta
    # when delta_y = 0, orientation = 0
    return np.sign(yn - y0) * np.abs(orientation)


def get_nearest_lanelet_fork(scenario: Scenario, lanelet_id: int, from_left: bool=True) -> List:
    """
    Finds the nearest lanelet fork (if there exists) from current lanelet. 
    If no fork, return the id of the last lanelet along this lane
    :param scenario: the scenario of lanelet
    :param lanelet_id: id of the target lanelet
    :param from_left: 
    :return: the nearest lanelet fork of the current lanelet id
    """
    current_lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
    successor = current_lanelet.successor
    if len(successor) > 1:
        lanelet_fork = successor
    elif len(successor) == 0:
        lanelet_fork = [lanelet_id]
    else:
        while len(successor) == 1:
            next_lanelet_id = scenario.lanelet_network.find_lanelet_by_id(successor[0]).successor
            if len(next_lanelet_id) == 0:
                lanelet_fork = successor
                break
            elif len(next_lanelet_id) > 1:
                lanelet_fork = next_lanelet_id
                break
            else:
                successor = next_lanelet_id
    # sort results
    directions = [[branch, get_lanelet_orientation(scenario, branch)] for branch in lanelet_fork]
    raw_orientation = list(fork_data[1] for fork_data in directions)
    if (max(raw_orientation) - min(raw_orientation)) > np.pi:
        for direction in directions:
            if direction[1] < 0:
                direction[1] = direction[1] + 2*np.pi
        
    # large direction value means a left from the perspective of the ego vehicle
    sorted_directions = sorted(directions, key=lambda x: x[1], reverse=from_left)
    # return the branchs
    lanelet_fork = [item[0] for item in sorted_directions]

    # fix bug for DEU_AAH-1_80029_T-1 or similar ones
    identifer = '_'.join(scenario.benchmark_id.split('_')[0:2])
    if identifer == 'DEU_AAH-1':
        if 153 in lanelet_fork:
            lanelet_fork.remove(153)
        
    return lanelet_fork


def create_lanes_for_lanelet_network(scenario: Scenario) -> Dict:
    """
    Searches nearest forward fork roads for each lanelet
    :param scenario: the scenario of lanelet network
    :return: The dictionary of lane list (sub lanelet ids and merged lanelet)
    """
    lane_list = []
    for l in scenario.lanelet_network.lanelets: 
        if l.predecessor:
            for p in l.predecessor: 
                # find the lanelet without predecessor (iteration ?)
                if not scenario.lanelet_network.find_lanelet_by_id(p).predecessor:
                    
                    (lane_lanelets, sub_lanelets_ids) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                        scenario.lanelet_network.find_lanelet_by_id(p),
                        scenario.lanelet_network, 1000)
                    # in InD dataset, there may be multiple lanelets to go
                    for lane_lanelet, sub_lanelet_ids in zip(lane_lanelets, sub_lanelets_ids):
                        # create curvilinear system
                        curvilinear_cosy_lane = create_coordinate_system_from_polyline(lane_lanelet.center_vertices)
                        lane_list.append([sub_lanelet_ids, lane_lanelet, curvilinear_cosy_lane])

        elif not l.predecessor and not l.successor:
            curvilinear_cosy_lane = create_coordinate_system_from_polyline(l.center_vertices)
            lane_list.append([[l.lanelet_id], l, curvilinear_cosy_lane])

    return lane_list


def interpolate_reference_path(xp: np.array, yp: np.array):

    xp_new = np.array([])
    yp_new = np.array([])

    for ind in range(len(xp)-1):

        distance = np.sqrt(np.square(xp[ind+1]-xp[ind]) + np.square(yp[ind+1]-yp[ind]))

        xvals = np.linspace(xp[ind], xp[ind+1], int(distance / 6) + 1, endpoint=False)
        # keep increasing in interpolation
        if xp[ind] <= xp[ind+1]:
            x = np.array([xp[ind], xp[ind+1]])
            y = np.array([yp[ind], yp[ind+1]])
        else:
            x = np.array([xp[ind+1], xp[ind]])
            y = np.array([yp[ind+1], yp[ind]])

        yinterp = np.interp(xvals, x, y)
        interpolated_data = np.stack((xvals, yinterp))

        xp_new = np.append(xp_new, interpolated_data[0])
        yp_new = np.append(yp_new, interpolated_data[1])
    
    xp_new = np.append(xp_new, xp[-1])
    yp_new = np.append(yp_new, yp[-1])
    
    return xp_new, yp_new
