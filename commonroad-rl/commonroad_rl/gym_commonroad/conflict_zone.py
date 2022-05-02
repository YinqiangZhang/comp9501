import yaml
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from abc import ABC, abstractmethod
from collections import defaultdict

from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as sha_Polygon
from shapely.ops import unary_union

import commonroad_dc.pycrcc as pycrcc
from pycrccosy import CurvilinearCoordinateSystem
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.trajectory import State
from commonroad.scenario.scenario import Scenario
from commonroad.geometry.shape import Polygon as Polygon

from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_rl.gym_commonroad.utils.scenario import get_lane_relative_heading
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS


__author__ = "Yinqiang Zhang"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "0.1"
__email__ = "yinqiang.zhang@tum.de"
__status__ = "Development"


class ConflictZone(ABC):
    """
    Description:
        Class for conflict zones constructed by intersection areas

    important concepts and variables:
    1. intersection-based conflict zones:
    2. lane-based conflict zones:
    3. unsafe lane pair pattern (groups):
    """

    def __init__(self, reaching_time_prediction=True):
        """
        initializes basic variables of conflict zones
        
        :param reaching_time_prediction: chooses which method is used to predict reaching time of other vehicles.
        """
        self.scenario = None
        self.lane_list = None # lane list generate by reference manager
        self.intersection_conflict_zone = None # intersection coflict region: C_I

        # lane-based conflict zone
        self.lane_conflict_information = dict()
        # unsafe lane pair
        self.unsafe_lane_pair_pattern = dict()
        # collision model for intersection conflict zone 
        self.region_collision_model = None
        # variable cache for computation acceleration
        self.cache_conflict_zone = defaultdict(lambda: None)
        
        # safe distance definition
        self.delta_conflict_region_width = 1.5
        self.reaching_time_based_prediction = reaching_time_prediction
        self.config_file = PATH_PARAMS["project_configs"]
        with open(self.config_file, "r") as root_path:
            self.config = yaml.safe_load(root_path)
        self.maximal_velocity = self.config['occupany_predictor']['maximal_velocity']
    
    def reset(self, scenario: Scenario, lane_list):
        """
        resets the class with new scenario
        
        :param scenario: base scenario used in planning
        :param lane_list: lane list extracted in corresponding scenario
        """
        self.scenario = scenario
        identifer = '_'.join(scenario.benchmark_id.split('_')[0:2])
        
        if self.cache_conflict_zone[identifer] is not None:
            self.lane_list = self.cache_conflict_zone[identifer]['lane_list']
            self.intersection_conflict_zone = self.cache_conflict_zone[identifer]['intersection_conflict_zone']
            self.lane_conflict_information = self.cache_conflict_zone[identifer]['conflict_information']
            self.unsafe_lane_pair_pattern = self.cache_conflict_zone[identifer]['unsafe_pattern']
            self.lane_velocity_upper_bound = self.cache_conflict_zone[identifer]['velocity_bound']
            self.lane_min_time_interval = self.cache_conflict_zone[identifer]['time_interval']
        else:
            self.lane_list = lane_list
            self.intersection_conflict_zone = self._conlict_region_generation(threshold_area=30)
            self.lane_conflict_information = self._generate_conflict_information()
            self.unsafe_lane_pair_pattern = self._generate_unsafe_pattern()
            self.lane_velocity_upper_bound = self.get_curvature_velocity_constraints(a_lat_max=4.0, v_max=20.0)
            self.lane_min_time_interval = self.get_minimal_time_to_drive()
            # save in cache 
            temp_dict = dict()
            temp_dict['lane_list'] = self.lane_list
            temp_dict['intersection_conflict_zone'] = self.intersection_conflict_zone
            temp_dict['conflict_information'] = self.lane_conflict_information
            temp_dict['unsafe_pattern'] = self.unsafe_lane_pair_pattern
            temp_dict['velocity_bound'] = self.lane_velocity_upper_bound
            temp_dict['time_interval'] = self.lane_min_time_interval
            self.cache_conflict_zone[identifer] = temp_dict
        
        # generate collsion model for intersection conflict zone
        self.conflict_collision_model = list()
        for region in self.intersection_conflict_zone:
            self.conflict_collision_model.append(self._generate_commonroad_cllision_model(region))

    def _conlict_region_generation(self, threshold_area: float = 10):
        """
        generates intersection conflict region according to the current scenario
        :param threshold_area: the minimal area considered when generating the conflict zone
        :return:
            list of valid intersection-based conflict zones 
        """
        show_list = list()
        conflict_regions = list()
        # only calculate once 
        for ind, lane in enumerate(self.lane_list):
            target_lane = lane
            target_ids = lane[0]

            # compare with other lane
            for k in range(ind+1, len(self.lane_list)):
                compared_lane = self.lane_list[k]
                compared_ids = compared_lane[0]

                # remove all the same lanelets
                trimmed_ids = list()
                for target_id in target_ids:
                    if target_id in compared_ids:
                        trimmed_ids.append(target_id)
                target_ids = list(set(target_ids).difference(set(trimmed_ids)))
                compared_ids = list(set(compared_ids).difference(set(trimmed_ids)))
                # no adjacent, but collision, not with same predecessor (3 conditions)
                # then calculate intersection area
                for target_id in target_ids:
                    target_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(target_id)
                    target_polygon = target_lanelet.convert_to_polygon()
                    polygon1 = sha_Polygon(target_polygon.vertices)
                    for compared_id in compared_ids:
                        compared_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(compared_id)
                        compared_polygon = compared_lanelet.convert_to_polygon()
                        polygon2 = sha_Polygon(compared_polygon.vertices)
                        if polygon1.overlaps(polygon2):
                            # recursively find the predecessors
                            from_same_lane = False
                            target_temp = target_lanelet
                            compared_temp = compared_lanelet
                            if target_temp.predecessor == compared_temp.predecessor:
                                from_same_lane = True
                            else:
                                while target_temp.predecessor != compared_temp.predecessor:
                                    # assume that if has different predecessors, only have one
                                    target_temp = self.scenario.lanelet_network.find_lanelet_by_id(target_temp.predecessor[0])
                                    compared_temp = self.scenario.lanelet_network.find_lanelet_by_id(compared_temp.predecessor[0])
                                    if len(target_temp.predecessor) == 0 or len(compared_temp.predecessor) == 0:
                                        break
                                    elif target_temp.predecessor == compared_temp.predecessor:
                                        from_same_lane = True
                                        break

                            if (not from_same_lane 
                                and ((target_id, compared_id) not in show_list 
                                and (compared_id, target_id) not in show_list)):
                                show_list.append((target_id, compared_id))
                                inter = polygon1.intersection(polygon2)
                                conflict_regions.append(inter)

        # calculate union
        conflict_union = unary_union(conflict_regions)
        try:
            valid_conflict_regions = list(region for region in conflict_union.geoms if region.area > threshold_area)
        except AttributeError:
            valid_conflict_regions = [conflict_union]
        
        # open and close operations to smooth the region
        for k in range(len(valid_conflict_regions)):
            valid_conflict_regions[k] = valid_conflict_regions[k].buffer(1.0)
            valid_conflict_regions[k] = valid_conflict_regions[k].buffer(-2.0)
            valid_conflict_regions[k] = valid_conflict_regions[k].buffer(1.0)
        
        return valid_conflict_regions
    
    def _generate_conflict_information(self):
        """
        generates conflict information for each lane
        
        :return: 
            information tuple (lane inforamtion, conflict zone, collision model, critical points(curvi))
        """
        assert self.intersection_conflict_zone is not None, "please generate the conflict region first"
        
        conflict_information_dict = dict()
        for lane in self.lane_list:
            # values for different conflict region
            lane_conflict_regions = list()
            conflict_collision_models = list()
            lane_critical_points = list()
            lane_id = lane[1].lanelet_id
            shapely_lane_polygon = sha_Polygon(lane[1].convert_to_polygon().vertices).buffer(-0.1)
            for conflict_zone in self.intersection_conflict_zone:
                lane_conflict_region = conflict_zone.intersection(shapely_lane_polygon)
                
                lane_conflict_region = lane_conflict_region.buffer(-0.2)
                lane_conflict_region = lane_conflict_region.buffer(0.2)
                
                if lane_conflict_region.area != 0:
                    # calculate critical points
                    # reference_line_string = LineString(lane[1].center_vertices)
                    reference_line_string = LineString(lane[3].reference())
                    extended_conflict_region = lane_conflict_region.buffer(2)
                    line_segments = reference_line_string.intersection(extended_conflict_region)
                    before_critical_point = line_segments.coords[0]
                    after_critical_point = line_segments.coords[-1]
                    try:
                        before_s, _ = lane[3].convert_to_curvilinear_coords(before_critical_point[0], before_critical_point[1])
                    except ValueError:
                        # project the results to the center line
                        idx = np.argmin(np.linalg.norm(lane[3].reference() - before_critical_point, axis=1))
                        pos = lane[3].reference()[idx]
                        before_s, _ = lane[3].convert_to_curvilinear_coords(pos[0], pos[1])
                    
                    try:
                        end_s, _ = lane[3].convert_to_curvilinear_coords(after_critical_point[0], after_critical_point[1])
                    except ValueError:
                        # project the results to the center line
                        idx = np.argmin(np.linalg.norm(lane[3].reference() - after_critical_point, axis=1))
                        pos = lane[3].reference()[idx]
                        end_s, _ = lane[3].convert_to_curvilinear_coords(pos[0], pos[1])
                    
                    critical_points_pair = (before_s, end_s)
                    # generate collision model (expended)
                    conflict_collision_model = self._generate_commonroad_cllision_model(lane_conflict_region)
                    lane_conflict_regions.append(lane_conflict_region)
                    conflict_collision_models.append(conflict_collision_model)
                    lane_critical_points.append(critical_points_pair)
            
            conflict_information_dict[lane_id] = (lane, lane_conflict_regions, conflict_collision_models, lane_critical_points)

        return conflict_information_dict
    
    def _generate_unsafe_pattern(self):
        """
        generate all unsafe lane pairs

        :return:
            unsafe pair dictionary
        """
        unsafe_pattern_dict = dict()
        for lane in self.lane_list:
            lane_id = lane[1].lanelet_id
            unsafe_pattern_dict[lane_id] = self._get_unsafe_lane_pair(lane)

        return unsafe_pattern_dict

    def _get_unsafe_lane_pair(self, ref_lane):
        """
        generates all unsafe pairs for the target lane
        
        :param reference lane: reference lane of ego vehicle        
        :return:
            unsafe lane pairs
        """
        unsafe_pairs = defaultdict(list)

        ref_lane_id = ref_lane[1].lanelet_id
        ref_lane_conflict_information = self.lane_conflict_information[ref_lane_id]
        ref_lane_conlict_zone = ref_lane_conflict_information[1]
        ref_lane_sublanelet_ids = ref_lane[0]

        for lane_id in self.lane_conflict_information:
            lane_conflict_information = self.lane_conflict_information[lane_id]
            lane = lane_conflict_information[0]
            lane_sublanelet_ids = lane_conflict_information[0][0]
            lane_conflict_zone = lane_conflict_information[1]
            # not from the same direction
            if ref_lane_sublanelet_ids[0] != lane_sublanelet_ids[0]:
                for conflict_zone in lane_conflict_zone:
                    for ref_conlict_zone in ref_lane_conlict_zone:
                        cross_zone = ref_conlict_zone.intersection(conflict_zone)
                        if cross_zone.area != 0:
                            cross_point = cross_zone.centroid.coords[0]
                            # generate curvi position for conflict lanes
                            ref_lane_s, _ = ref_lane[3].convert_to_curvilinear_coords(cross_point[0], cross_point[1])
                            lane_s, _ = lane[3].convert_to_curvilinear_coords(cross_point[0], cross_point[1])
                            cross_commonroad_zone = self._generate_commonroad_cllision_model(cross_zone)
                            unsafe_pairs[lane_id].append((ref_lane_s, lane_s, cross_commonroad_zone))
            
        return unsafe_pairs

    def get_unsafe_obstacle_association(self, conflict_obstacles_information):
        """
        generates the associations between dynamic obstacles and unsafe lane pairs

        :param conflict_obstacles_information: states of surrounding vehicles near intersection areas
        :return:
            unsafe obstacle-lane association
            intersection-related observations (states)
        """
        unsafe_association = list()
        observations = list()
        for state in conflict_obstacles_information:
            position = state.position
            orientation = state.orientation
            [lanlet_id_candidates] = self.scenario.lanelet_network.find_lanelet_by_position([position])
            
            delta_s = 50
            if len(lanlet_id_candidates) == 1:
                lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanlet_id_candidates[0])
                # not in the opposite direction of the lane
                if abs(get_lane_relative_heading(state, lanelet)) < 0.5 * np.pi: # 90 degree
                    unsafe_information = self._get_lane_conflict_information(lanlet_id_candidates[0], state)
                    if len(unsafe_information) != 0:
                        temp_delta_s = list(unsafe_information.values())[0][1]
                        delta_s = temp_delta_s if temp_delta_s < delta_s else delta_s
                    unsafe_association.append((state, unsafe_information))
            else:
                for lanelet_id in lanlet_id_candidates:
                    # a threshold for relative heading
                    lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                    if abs(get_lane_relative_heading(state, lanelet)) < 0.5:
                        unsafe_information = self._get_lane_conflict_information(lanelet_id, state)
                        if len(unsafe_information) != 0:
                            temp_delta_s = list(unsafe_information.values())[0][1]
                            delta_s = temp_delta_s if temp_delta_s < delta_s else delta_s
                        unsafe_association.append((state, unsafe_information))
            
            if delta_s != 50.0:
                observations.append((delta_s, state.velocity))
        
        # process observations
        observations = sorted(observations, key=lambda x: x[0])

        return unsafe_association, observations

    def check_intersection_safety(self, ego_state, ref_lane, unsafe_association, obs_a_stop, time_step, plan_horizon):
        """
        checks how long time the state of ego vehicle can satisfy the safety of intersection
        :param ego_state: the current state of ego vehicle
        :param ref_lane: current reference lane of ego vehicle
        :param unsafe_association: the vehicle-lane pairs that have a potential danger with ego vehicle
        :param obs_a_stop: largest acceleration used by obstacle before the intersection
        :param time_step: current time step
        :param time_step: planning horizon of low-level planner

        :return:
            time-variant collision model of conflict zones
        """
        # get information of ego reference lane and ego vehicle
        ref_lane_id = ref_lane[1].lanelet_id # reference lane id 
        unsafe_pattern = self.unsafe_lane_pair_pattern[ref_lane_id] # all unsafe lane pairs
        conflict_information = self.lane_conflict_information[ref_lane_id] # lane-based conflict information 
        
        if len(conflict_information[3]) == 0:
            # lane without passing intersection area
            return pycrcc.ShapeGroup()
        else:
            # critical points before and after intersection
            ego_critical_near, ego_criical_far = conflict_information[3][0]
            # current longitudinal position of ego vehicle
            try:
                ego_s, _ = ref_lane[3].convert_to_curvilinear_coords(ego_state.position[0], ego_state.position[1])
            except ValueError:
                # project the results to the center line
                idx = np.argmin(np.linalg.norm(ref_lane[3].reference() - ego_state.position, axis=1))
                pos = ref_lane[3].reference()[idx]
                ego_s, _ = ref_lane[3].convert_to_curvilinear_coords(pos[0], pos[1])
            # safety verification converted into collision avoidance
            collision_object_list = pycrcc.ShapeGroup()
            if ego_s < ego_criical_far: # ego vehicle not pass the intersection
                predicted_reach_time = 100 # 100 is a large enough number as a place holder
                unsafe_intersection = False
                for lane_id in unsafe_pattern: # seacrch in all unsafe lane pairs
                    for state, lane_dict in unsafe_association:
                        if lane_dict.get(lane_id, None) is not None:
                            # obs_s: location of obstacle in its lane
                            # delta_obs_s: distance to the conflict region
                            obs_s, delta_obs_s = lane_dict.get(lane_id, None)
                            # the position of crossing point in lanes of one unsafe lane pair
                            ego_critcal_s, obs_critical_s, cross_collision_object = unsafe_pattern.get(lane_id)[0]
                            # distance between obstacle and its corresponding crossing point
                            safe_delta_distance = obs_critical_s - obs_s
                            # reaching_time_based_prediction

                            if self.reaching_time_based_prediction:
                                # method 1: constant-velocity prediction
                                if safe_delta_distance > self.delta_conflict_region_width:
                                    reach_time = self.get_minimum_time_to_conflict_region(lane_id, state, obs_s, obs_critical_s)
                                    predicted_reach_time = reach_time if reach_time < predicted_reach_time else predicted_reach_time
                                elif abs(safe_delta_distance) <= self.delta_conflict_region_width:
                                    unsafe_intersection = True
                            else:
                                # method 2: safe braking prediction
                                if safe_delta_distance > self.delta_conflict_region_width:
                                    v_max = np.sqrt(2 * (obs_critical_s - obs_s) * obs_a_stop)
                                    if v_max <= state.velocity:
                                        v_0 = state.velocity
                                        reach_time = (v_0 - np.sqrt(v_0**2-v_max**2))/obs_a_stop
                                        predicted_reach_time = reach_time if reach_time < predicted_reach_time else predicted_reach_time
                                elif abs(safe_delta_distance) <= self.delta_conflict_region_width:
                                    unsafe_intersection = True
                                    
                # print(unsafe_intersection, predicted_reach_time)
                return self._generate_conflict_collision_model(conflict_information[2][0], 
                    unsafe_intersection, time_step, predicted_reach_time, plan_horizon)
            else:
                # ego vehicle has passed the intersection 
                return pycrcc.ShapeGroup()
                            
    def _get_lane_conflict_information(self, sub_lanelet_id, state):
        """
        gets the corresponding usafe conflict information at intersection area
        param: sub_lanelet_id: the lanelet id of the obstacle
        param: state: the state of the obstacle

        :return:
            features of unsafe lane pair group 
        """

        # find possible lanes
        lane_id_list = list()
        for lane in self.lane_list:
            if sub_lanelet_id in lane[0]:
                lane_id_list.append(lane[1].lanelet_id)
        
        # associate the obstacle with the corresponding lane
        unsafe_information = dict()
        for lane_id in lane_id_list:
            conflict_information = self.lane_conflict_information[lane_id]
            if len(conflict_information[3]) == 0:
                continue
            lane_information = conflict_information[0]
            try:
                obs_s, _ = lane_information[3].convert_to_curvilinear_coords(state.position[0], state.position[1])
            except ValueError:
                # project the results to the center line
                idx = np.argmin(np.linalg.norm(lane_information[3].reference() - state.position, axis=1))
                pos = lane_information[3].reference()[idx]
                obs_s, _ = lane_information[3].convert_to_curvilinear_coords(pos[0], pos[1])

            # get maixmal velocity for turning
            # contraints for predicted trajectory
            maximal_lateral_acceleration = 4 # a_lateral

            s_idx = np.argmin(np.abs(lane_information[3].ref_pos() - obs_s))
            if lane_information[3].ref_pos()[s_idx] > obs_s:
                s_idx -= 1
            s_lambda = (obs_s - lane_information[3].ref_pos()[s_idx]) / (
                        lane_information[3].ref_pos()[s_idx + 1] - lane_information[3].ref_pos()[s_idx])
            ego_curvature = (s_lambda) * lane_information[3].ref_curv()[s_idx + 1] + (1-s_lambda) * lane_information[3].ref_curv()[s_idx]
            v_max_limit = min(self.maximal_velocity, np.sqrt(maximal_lateral_acceleration / np.abs(ego_curvature)))

            critical_point_near, critical_point_far = conflict_information[3][0]
            # obstacle before or in ther intersection and its velocity fits for the corresponding lane
            if obs_s < critical_point_far and v_max_limit > state.velocity:
                # longitudinal position of obstacle on current lane
                # distance to the conflict region
                unsafe_information[lane_id] = (obs_s, critical_point_near-obs_s)

        return unsafe_information

    def _generate_commonroad_cllision_model(self, shapely_polygon):
        """
        convert the polygon in shaeply into the commonroad
        
        :return:
            collision model (commonroad)
        """
        points = shapely_polygon.exterior
        vertices = np.stack((np.array(points.xy[0].tolist()), np.array(points.xy[1].tolist())), axis=0).T
        # generate collision object
        return create_collision_object(Polygon(vertices))

    def _generate_conflict_collision_model(self, polygon_collision_object, unsafe_intersection, time_step, ttr, plan_horizon):
        
        """
        generates time-variant collision model for conflict zones
        :param polygon_collision_object: collision object (shape) of lane-based conflict zones
        :param unsafe_intersection: whether lane-based conflict zone is safe or not 
        :param time_step: current time step
        :param ttr: time to reach the conflict zones
        :param plan_horizon: time horizon of reactive planner

        :return:
            time-variant collision object
        """
        if unsafe_intersection:
            collision_object = polygon_collision_object
        else:
            if ttr < 4:
                ttr_time_steps = int(ttr / self.scenario.dt)
                horizon_time_steps = int(plan_horizon / self.scenario.dt)

                collision_object = pycrcc.TimeVariantCollisionObject(time_step + ttr_time_steps)
                i = ttr_time_steps
                while i <= horizon_time_steps:
                    collision_object.append_obstacle(polygon_collision_object)
                    i += 1
            else:
                collision_object = pycrcc.ShapeGroup()

        return collision_object

    def _generate_collision_object(self):
        """
        generates objects for later collision checker 
        """
        # TODO: consider multiple conflict regions
        points = self.intersection_conflict_zone.exterior
        vertices = np.stack((np.array(points.xy[0].tolist()), np.array(points.xy[1].tolist())), axis=0).T
        # generate collision object
        self.region_collision_model = create_collision_object(Polygon(vertices))
        self.region_velocity_bound = [0, 0]

    def draw_conflict_region(self, conflict_region = None, linewidth=1, color='chocolate'):
        """
        draws the generated conflict region in the render function
        """
        if conflict_region is None:
            assert self.intersection_conflict_zone is not None, "conflict region has not been generated yet"
            conflict_region = self.intersection_conflict_zone
        try:
            for item in conflict_region:
                plt.plot(*item.exterior.xy, zorder=21, linewidth=linewidth, color=color)
        except:
            plt.plot(*conflict_region.exterior.xy, linewidth=linewidth, zorder=21, color=color)
    
    def check_near_intersection_area(self, state: State, ref_lane, distane_threshold):
        """
        checks if the ego vehicle is near enough to the intersection areas

        :param state: current state of ego vehicle
        :param ref_lane: reference lane of ego vehicle
        :param distane_threshold: distance to the intersection area

        :return:
            if lane change is feasible
            distance to the nearest boundary of conflict zone
            distance to the farest boundary of conflict zone
        """
        ref_lane_id = ref_lane[1].lanelet_id
        ref_lane_conflict_information = self.lane_conflict_information[ref_lane_id]
        if len(ref_lane_conflict_information[3]) == 0:
            return True, -10, -0.1 # set feasible to change the lane
        critical_point_near, critical_point_far = ref_lane_conflict_information[3][0]

        # calculate current position in curvilinear system
        position = state.position
        ego_s, _ = ref_lane[3].convert_to_curvilinear_coords(position[0], position[1])
        
        delta_s_near, delta_s_far = critical_point_near - ego_s, critical_point_far - ego_s
        
        lane_changing_feasible = True
        if delta_s_near < distane_threshold and delta_s_far > 0:
            lane_changing_feasible = False
        
        return lane_changing_feasible, delta_s_near, delta_s_far

    def get_curvature_velocity_constraints(self, a_lat_max, v_max = None):
        """
        gets the velocity constraints of lane caused by curvature
        :param lane: the target lane
        :param a_lat_max: maximal lateral acceleration
        :param v_max: speed limit of the current lane

        :return:
            upper bound of velocity
        """
        lane_velocity_upper_bound = defaultdict(list)
        for lane in self.lane_list:
            lane_id = lane[1].lanelet_id
            ref_curvature = lane[3].ref_curv()
            max_velocity = self.maximal_velocity if v_max is None else v_max
            for curv in ref_curvature:
                v_max = min(max_velocity, np.sqrt(a_lat_max / max(curv, 0.005)))
                lane_velocity_upper_bound[lane_id].append(v_max)
        return lane_velocity_upper_bound

    def get_minimal_time_to_drive(self):
        """
        gets minimal time to traverse 

        return: 
            minimal reaching time
        """
        min_time_interval = defaultdict(list)
        for lane in self.lane_list:
            lane_id = lane[1].lanelet_id
            ref_position = lane[3].ref_pos()
            velocity_bound = self.lane_velocity_upper_bound[lane_id]
            for k in range(len(velocity_bound)-1):
                delta_s = ref_position[k+1] - ref_position[k]
                # 1. the most conservative
                # v_max = max(velocity_bound[k+1], velocity_bound[k])
                # 2. less conservative 
                v_max = (velocity_bound[k+1] + velocity_bound[k])/2
                delta_t_min = delta_s / v_max
                min_time_interval[lane_id].append(delta_t_min)

        return min_time_interval

    def get_minimum_time_to_conflict_region(self, lane_id, state, obs_s, obs_critical_s):
        
        """
        gets the reaching time for constant-velocity-based approach

        :param lane_id: ID of current lanelet 
        :param state: current state of ego vehicle
        :param obs_s: longitudinal position of obstacle
        :param obs_critical_s: longitudinal position of crossing point

        :return:
            predicted reaching time
        """

        # lane where obstacle is driving
        conflict_information = self.lane_conflict_information[lane_id]
        lane_information = conflict_information[0]
        
        ref_position = lane_information[3].ref_pos()
        delta_t_list = self.lane_min_time_interval[lane_id]
        velocity_bound = self.lane_velocity_upper_bound[lane_id]
        
        change_s_idx = np.where(velocity_bound <= state.velocity)[0]
        if len(change_s_idx) == 0:
            # constant velocity assumption
            time_to_conflict = (obs_critical_s-obs_s) / max(0.001, state.velocity)
        else:
            begin_s_idx = change_s_idx[0]
            t_constant = (ref_position[begin_s_idx]- obs_s) / max(0.001, state.velocity)
            s_idx_conflict = np.argmin(np.abs(ref_position - obs_critical_s))
            time_to_conflict = t_constant + np.sum(delta_t_list[begin_s_idx:s_idx_conflict])

        return time_to_conflict

    def check_in_conflict_region(self, ego_vehicle):
        """
        checks if the ego vehicle is in the intersection area now

        :return:
            whether the ego vehicle is in the intersection area
        """
        in_intersection = False
        for cc in self.conflict_collision_model:
            if cc.collide(ego_vehicle.collision_object):
                in_intersection = True
                break
        return in_intersection
