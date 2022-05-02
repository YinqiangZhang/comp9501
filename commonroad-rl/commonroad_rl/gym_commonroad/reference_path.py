import yaml
import warnings
import numpy as np
from typing import List, Dict, Union
from abc import ABC, abstractmethod
from shapely.geometry import LineString, Point
from collections import defaultdict, OrderedDict

from scipy import spatial
from scipy.interpolate import splprep, splev

# commonroad-rp
from commonroad_rp.utils import CoordinateSystem

# commonroad-io
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.planning.planning_problem import PlanningProblem

# commonroad_rl 
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.gym_commonroad.utils.scenario import (
    create_coordinate_system_from_polyline,
    get_nearest_lanelet_fork,
    extrapolate_resample_polyline,
    get_lane_relative_heading,
    get_distance_point_to_linestring,
)


class BaseReferencePathManager(ABC):
    """
    Description:
    Abstract base class for the generator of reference path
    """
    # save all reference path for later use __init__
    # incorporate possible lane (sub_ids, maybe merged lanelet), reorganized list
    # fork directions (next lanelets)
    def __init__(self, scenario: Union[Scenario, None] = None):
        
        self.reference_accel_cache = defaultdict(lambda: None)
        self._reference_path = None
        self.config_file = PATH_PARAMS["project_configs"]
        with open(self.config_file, "r") as root_path:
            self.config = yaml.safe_load(root_path)
        # parameters for smooth
        self.smooth_factor = self.config['reference_manager']['smooth_factor']
        self.weight_coefficient = self.config['reference_manager']['weight_coefficient']
        
        if scenario is not None:
            self.reset(scenario)
        
    def reset(self, scenario: Scenario):
        self.scenario = scenario
        identifer = '_'.join(scenario.benchmark_id.split('_')[0:2])
        if self.reference_accel_cache[identifer] is not None:
            self.lane_list = self.reference_accel_cache[identifer]['lane_list']
            self.raw_lane_list = self.reference_accel_cache[identifer]['raw_lane_list']
            self.direction_list = self.reference_accel_cache[identifer]['direction_list']
            self.reference_path_list = self.reference_accel_cache[identifer]['reference_path_list']
        else:
            self.lane_list, self.raw_lane_list = self._get_reorgnized_lane_list()
            self.direction_list = self._create_forks_for_lanelet_network(from_left=True)
            self.reference_path_list = self._create_reference_path_list()
            # save in cache 
            temp_dict = dict()
            temp_dict['lane_list'] = self.lane_list
            temp_dict['raw_lane_list'] = self.raw_lane_list
            temp_dict['direction_list'] = self.direction_list
            temp_dict['reference_path_list'] = self.reference_path_list
            self.reference_accel_cache[identifer] = temp_dict

    @property
    def reference_path(self) -> np.ndarray:
        # give back the smoothed reference path
        return self._smoothing_reference_path(self._reference_path)

    def _create_lanes_for_lanelet_network(self) -> List:
        """
        Searches drivable lanes in the lanelet netwirk of the scenario
        :return: drivable lane list (sub lanelet ids and merged lanelet)
        """
        lanes = []
        for l in self.scenario.lanelet_network.lanelets: 
            if l.predecessor:
                for p in l.predecessor: 
                    # find the lanelet without predecessor (iteration ?)
                    if not self.scenario.lanelet_network.find_lanelet_by_id(p).predecessor:
                        
                        (lane_lanelets, sub_lanelets_ids) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                            self.scenario.lanelet_network.find_lanelet_by_id(p),
                            self.scenario.lanelet_network, 1000)
                        # in InD dataset, there may be multiple lanelets to go
                        for merged_lane_lanelet, sub_lanelet_ids in zip(lane_lanelets, sub_lanelets_ids):
                            identifer = '_'.join(self.scenario.benchmark_id.split('_')[0:2])
                            # fix imperfect design in the commonroad model
                            # planning problem: DEU_AAH-1_80029_T-1
                            if identifer == 'DEU_AAH-1':
                                self.weight_coefficient = 12
                                if 153 not in sub_lanelet_ids:
                                    if sub_lanelet_ids[0] == 112 and sub_lanelet_ids[-1] == 102:
                                        smooth_factor = 1.7
                                    else:
                                        smooth_factor = 1.5
                                    reference_path = extrapolate_resample_polyline(merged_lane_lanelet.center_vertices)
                                    reference_path = self._smoothing_reference_path(reference_path, smooth_factor=smooth_factor)
                                    ref_cossy = CoordinateSystem(reference_path)
                                    lanes.append([sub_lanelet_ids, merged_lane_lanelet, reference_path, ref_cossy])
                            elif identifer == 'DEU_AAH-2':
                                self.weight_coefficient = 8
                                reference_path = extrapolate_resample_polyline(merged_lane_lanelet.center_vertices)
                                reference_path = self._smoothing_reference_path(reference_path, smooth_factor=1.0)
                                ref_cossy = CoordinateSystem(reference_path)
                                lanes.append([sub_lanelet_ids, merged_lane_lanelet, reference_path, ref_cossy])
                            elif identifer == 'DEU_AAH-3':
                                self.weight_coefficient = 8
                                # remove unuseful road
                                if sub_lanelet_ids[0] != 149:
                                    reference_path = extrapolate_resample_polyline(merged_lane_lanelet.center_vertices)
                                    if sub_lanelet_ids[0] == 114 and sub_lanelet_ids[-1] in [154, 146]:
                                        reference_path = self._smoothing_reference_path(reference_path, smooth_factor=0.0)
                                    else:
                                        reference_path = self._smoothing_reference_path(reference_path, smooth_factor=0.7)
                                    ref_cossy = CoordinateSystem(reference_path)
                                    lanes.append([sub_lanelet_ids, merged_lane_lanelet, reference_path, ref_cossy])
                            else:
                                reference_path = extrapolate_resample_polyline(merged_lane_lanelet.center_vertices)
                                reference_path = self._smoothing_reference_path(reference_path, smooth_factor=1.0)
                                ref_cossy = CoordinateSystem(reference_path)
                                lanes.append([sub_lanelet_ids, merged_lane_lanelet, reference_path, ref_cossy])

            elif not l.predecessor and not l.successor:
                reference_path = extrapolate_resample_polyline(l.center_vertices)
                reference_path = self._smoothing_reference_path(reference_path, smooth_factor=1.2)
                ref_cossy = CoordinateSystem(reference_path)
                lanes.append([[l.lanelet_id], l, reference_path, ref_cossy])

        return lanes
    
    def _get_reorgnized_lane_list(self) -> Dict:
        """
        reorganized the lane list in terms of each lanelet id
        """
        raw_lanes_list = self._create_lanes_for_lanelet_network()
        reorgnized_lane_list = defaultdict(list)
        # generate all reference paths
        for lane in raw_lanes_list:
            for lanelet_id in lane[0]:
                reorgnized_lane_list[lanelet_id].append(lane)
        
        return reorgnized_lane_list, raw_lanes_list
    
    def _create_forks_for_lanelet_network(self, from_left: bool=True) -> Dict:
        """
        Searches nearest forward fork roads for each lanelet
        :param scenario: the scenario of lanelet network
        :param from_left: results are sorted from left or right
        :return: The nearest fork dictionary of each lanelet
        """
        direction_list = dict()
        for l in self.scenario.lanelet_network.lanelets:
            direction_list[l.lanelet_id] = get_nearest_lanelet_fork(self.scenario, l.lanelet_id, from_left)
        
        return direction_list

    def _create_reference_path_list(self) -> Dict:
        """
        generates reference path list for efficient query
        """
        reference_path_list = dict()
        for lanelet_id in self.lane_list.keys():
            lane_candidates = self.lane_list[lanelet_id]
            branch_reference = OrderedDict()
            for branch in self.direction_list[lanelet_id]:                
                for candidate in lane_candidates:
                    if branch in candidate[0]:
                        try: 
                            branch_reference[branch].append(candidate)
                        except KeyError:
                            branch_reference[branch] = [candidate]
                # select the reference path with smallest curvature
                largest_curvature_values = list()
                for candidate_lane in branch_reference[branch]:
                    largest_curvature_values.append(max(abs(candidate_lane[3].ref_curv())))
                branch_reference[branch] = branch_reference[branch][np.argmin(largest_curvature_values)]
            reference_path_list[lanelet_id] = branch_reference
        
        return reference_path_list

    def _smoothing_reference_path(self, reference_path: np.ndarray, smooth_factor = None) -> np.ndarray:
        """
        Smooths a given reference polyline for lower curvature rates. The smoothing is done using splines from scipy.
        :param reference_path: The reference_path to smooth [array]
        :return: The smoothed reference
        """
        if smooth_factor is None:
            smooth_factor = self.smooth_factor
        # generate a smooth reference path
        transposed_reference_path = reference_path.T
        # how to generate index okay
        okay = np.where(
            np.abs(np.diff(transposed_reference_path[0]))
            + np.abs(np.diff(transposed_reference_path[1]))
            > 0
        )
        xp = np.r_[transposed_reference_path[0][okay], transposed_reference_path[0][-1]]
        yp = np.r_[transposed_reference_path[1][okay], transposed_reference_path[1][-1]]
        
        curvature = compute_curvature_from_polyline(np.array([xp, yp]).T)
        
        # set weights for interpolation:
        # see details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html 
        weights = np.exp(-self.weight_coefficient * (abs(curvature)- np.min(abs(curvature))))
        # B spline interpolation
        tck, u = splprep([xp, yp], s=smooth_factor, w=weights)
        u_new = np.linspace(u.min(), u.max(), 2000)
        x_new, y_new = splev(u_new, tck, der=0)

        return np.array([x_new, y_new]).transpose()

    @abstractmethod
    def update_reference_path(self, action: np.ndarray, state: State) -> None:
        """
        updates a feasible reference path for planner
        """
        pass 


class ReactivePlannerReferencePathManager(BaseReferencePathManager):
    """
    Description:
    Reference path manager for reactive planner
    """

    def __init__(self, scenario: Union[Scenario, None] = None) -> None:
        super(ReactivePlannerReferencePathManager, self).__init__(scenario)
        # reference lane structure [[sub-lanelets ids], merged lanelet]
        self.source_reference_lane = None
        self.destination_reference_lane = None

    def reset(self, scenario: Union[Scenario, None]):
        super(ReactivePlannerReferencePathManager, self).reset(scenario)
        # reference lane structure [[sub-lanelets ids], merged lanelet]
        self.source_reference_lane = None
        self.destination_reference_lane = None

    def update_reference_path(self, action: np.ndarray, state: State, horizon: float) -> None:
        """
        get the reference path for specific action under the given vehicle state
        : param action: the discrete actions (direction and lane changing actions) to be executed
        : param state: the state of the vehicle
        : param horizon: time horizon for the lane changing
        : return: reference path for the reactive planner 
        """
        # unpack the action
        direction_action, lane_action = action

        # 1. determine current lanelet
        if self.source_reference_lane is None:
            [feasible_lanelet_id_list] = self.scenario.lanelet_network.find_lanelet_by_position([state.position])
            if len(feasible_lanelet_id_list) < 1:
                ValueError("the vehicle does not locate in the lanelet network")
            elif len(feasible_lanelet_id_list) == 1:
                lanelet_id = feasible_lanelet_id_list[0]
            else:
                feasible_lanelet_id = feasible_lanelet_id_list[0]
                feasible_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(feasible_lanelet_id)
                # modify the current lanelet id to the predecessor
                lanelet_id = feasible_lanelet.predecessor[0]
            self.source_reference_lane = self.lane_list[lanelet_id][0]
            current_projected_id = lanelet_id
            current_reference_path = extrapolate_resample_polyline(self.source_reference_lane[1].center_vertices, step=2)
        else:
            # update the previous target to the current source
            self.source_reference_lane = self.destination_reference_lane
            # unpack the lane
            current_reference_path = extrapolate_resample_polyline(self.source_reference_lane[1].center_vertices, step=2)
            current_path_sub_ids = self.source_reference_lane[0]
            # find the projected lanelet id
            current_projected_id = self.get_projected_lanelet_id(
                    state, current_reference_path, current_path_sub_ids
                )
                
        assert current_projected_id is not None, "cannot find the projected lanelet id"

        current_id = current_projected_id
        current_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(current_projected_id)
       
        # 2. determine target lane:
        target_lanelet = None
        if lane_action == 2:
            if current_lanelet.adj_left_same_direction not in (False, None):
                target_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(
                    current_lanelet.adj_left
                )
        elif lane_action == 0:
            if current_lanelet.adj_right_same_direction not in (False, None):
                target_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(
                    current_lanelet.adj_right
                )
        elif lane_action == 1:
            target_lanelet = current_lanelet
            
        if target_lanelet is None:
            raise ValueError("set_reference_lane: No adjacent lane in direction {lane_action}, stay in current lane.")
        
        # 3. get target reference-lane object (sub_ids, merged_lanelet) 
        road_fork = get_nearest_lanelet_fork(self.scenario, target_lanelet.lanelet_id)
        self.destination_reference_lane = self.reference_path_list[target_lanelet.lanelet_id][road_fork[direction_action]]
        # unpack the reference lane object
        target_reference_path = extrapolate_resample_polyline(self.destination_reference_lane[1].center_vertices, step=2)
        target_path_sub_ids = self.destination_reference_lane[0]

        # 4. connect the current and target reference lane
        if lane_action != 1:
            # create target curvilinear system
            # target_cossy = create_coordinate_system_from_polyline(target_reference_path)
            # lane_change_hypothenuse = horizon * state.velocity
            # [
            #     position_long,
            #     position_lat,
            # ] = target_cossy.convert_to_curvilinear_coords(
            #     state.position[0], state.position[1]
            # )
            # # the maximal length needed to change the lane
            # # TODO: remove scenarios that with low velocity
            # end_lane_change_long = (
            #     np.sqrt(lane_change_hypothenuse ** 2 - position_lat ** 2)
            #     + position_long
            # )
            # # convert the value from curvi to cartesian
            # end_lane_change = target_cossy.convert_to_cartesian_coords(
            #     end_lane_change_long, 0.0
            # )

            # to find the nearest the point (change)
            # distance, end_index = spatial.KDTree(current_reference_path).query(
            #     state.position
            # )
            # distance, start_index = spatial.KDTree(target_reference_path).query(
            #     end_lane_change
            # )
            
            # xvals = np.linspace(
            #     state.position[0],
            #     end_lane_change[0],
            #     int(lane_change_hypothenuse / 2),
            # )

            # keep increasing in interpolation
            # if state.position[0] < end_lane_change[0]:
            #     x = np.array([state.position[0], end_lane_change[0]])
            #     y = np.array([state.position[1], end_lane_change[1]])
            # else:
            #     x = np.array([end_lane_change[0], state.position[0]])
            #     y = np.array([end_lane_change[1], state.position[1]])

            # yinterp = np.interp(xvals, x, y)
            # interpolated_data = np.stack((xvals, yinterp))
            # if end_index != 0:
            #     end_index -= 1

            # if len(target_reference_path) >= start_index + 2:
            #     start_index += 1

            # # the interpolation is inserted into the original reference path
            # temp1 = np.concatenate(
            #     (current_reference_path[0:end_index], interpolated_data.T[1:]),
            #     axis=0,
            # )
            # self._reference_path = np.concatenate(
            #     (temp1, target_reference_path[start_index:]), axis=0
            # )
            self._reference_path = target_reference_path
        else:
            self.source_reference_lane = self.destination_reference_lane
            # directly use the combined reference path
            self._reference_path = target_reference_path

    def _get_projected_position(
            self, state: State, ref_path: np.ndarray
        ) -> np.ndarray:
        
        """
        finds nearest projected postion on the reference path
        """

        position = state.position
        current_reference_path = LineString(ref_path)
        current_posotion = Point(position)

        long_dist = current_reference_path.project(current_posotion)
        project_position = current_reference_path.interpolate(long_dist)

        return np.array([project_position.x, project_position.y])

    def get_projected_lanelet_id(
            self, state: State, ref_path: np.ndarray, path_sub_ids: list
        ) -> int:
            """
            finds the projected lanelet id on reference path
            """
            current_projected_id = None
            projected_position = self._get_projected_position(state, ref_path)
            project_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position(
                [projected_position]
            )[0]
            # select ids by reference path
            for sub_id in path_sub_ids:
                if sub_id in project_lanelet_ids:
                    current_projected_id = sub_id
                    break
            return current_projected_id

    def projected_target_lanelet_id(self, state: State):
        reference_path = extrapolate_resample_polyline(self.destination_reference_lane[1].center_vertices, step=2)
        path_sub_ids = self.destination_reference_lane[0]
        projected_id = self.get_projected_lanelet_id(
                state, reference_path, path_sub_ids
            )
        return projected_id
    
    def projected_source_lanelet_id(self, state: State):
        reference_path = extrapolate_resample_polyline(self.source_reference_lane[1].center_vertices, step=2)
        path_sub_ids = self.source_reference_lane[0]
        projected_id = self.get_projected_lanelet_id(
                state, reference_path, path_sub_ids
            )
        return projected_id
    
    def check_lane_change_finished(self, state: State, 
                                distance_threshold: float, orientation_threshold: float):
        """
        checks if the vehicle has finished the lane changing maneuver
        : param state: the state of the ego vehicle 
        : param distance_threshold: distance threshold to the center line
        : param orientation_threshold: orientation difference to the lanelet orientation
        """
        assert self.destination_reference_lane is not None, "please set the destination lane firstly"
        ref_path = LineString(self.destination_reference_lane[1].center_vertices)
        p = Point(state.position)
        # check the distance to the destination lane 
        if get_distance_point_to_linestring(p, ref_path) < distance_threshold:
            ego_lanelet_id = self.projected_target_lanelet_id(state)
            if ego_lanelet_id is not None:
                ego_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
                diff_angle = get_lane_relative_heading(state, ego_lanelet)
                if abs(diff_angle) < orientation_threshold:
                    return True
            else:
                # drive outside the road network
                return True
        return False
        
    def check_lane_change_fork_changed(self, next_state, next_lanelet_id):
        """
        check if the road fork is changed during the lane changing
        """
        # get predecessor
        if next_lanelet_id is None:
            return False
        else:
            next_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(next_lanelet_id)
            predecessor = next_lanelet.predecessor[0]
            lanelet_fork = get_nearest_lanelet_fork(self.scenario, predecessor)
            [lanelet_ids] = self.scenario.lanelet_network.find_lanelet_by_position([next_state.position])
            for lanelet_id in lanelet_ids:
                if lanelet_id == next_lanelet_id and next_lanelet_id in lanelet_fork:
                    return True
            return False


def compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the curvature of a given polyline
    :param polyline: The polyline for the curvature computation
    :return: The curvature of the polyline
    """
    assert (
        isinstance(polyline, np.ndarray)
        and polyline.ndim == 2
        and len(polyline[:, 0]) > 2
    ), "Polyline malformed for curvature computation p={}".format(polyline)
    x_d = np.gradient(polyline[:, 0])
    x_dd = np.gradient(x_d)
    y_d = np.gradient(polyline[:, 1])
    y_dd = np.gradient(y_d)

    return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3.0 / 2.0))
