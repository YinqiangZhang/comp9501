import os
import yaml
import spot
import numpy as np
from copy import deepcopy
from typing import Dict, List
from collections import defaultdict, OrderedDict

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import Occupancy, SetBasedPrediction
from commonroad.geometry.shape import ShapeGroup, Polygon

import commonroad_dc.pycrcc as pycrcc

# commonroad_rl
from commonroad_rl.gym_commonroad.iss_occupancy import ISSDynamicOccupancy
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

# commonroad_rp
from commonroad_rp.utils import CoordinateSystem
from commonroad_rp.polynomial_trajectory import QuarticTrajectory


class BehaviorPredictior(object):
    """
    Description:
    predicts the behaviors of surrounding vehicles and situations in the intersection for decision making
    """ 
    def __init__(self):
        """
        initializes the class for occupancy prediction (SPOT)
        """
        # initinate the scenario without obstacles
        self.scenario = None
        self.planning_problem = None
        self.config_file = PATH_PARAMS["project_configs"]
        with open(self.config_file, "r") as root_path:
            self.config = yaml.safe_load(root_path)

        # velocity limits
        self.limit_v_max = self.config["occupany_predictor"]['maximal_velocity']
        self.limit_v_min = 0
        # can be set to a fit value
        self.maximal_acceleration = self.config["occupany_predictor"]['maximal_acceleration']
        self.maximal_velocity = self.config["occupany_predictor"]['maximal_velocity']
        self.delta_brake = self.config["occupany_predictor"]['reacting_time']
        self.compute_occ_m1 = self.config["occupany_predictor"]['compute_occ_m1']
        self.compute_occ_m2 = self.config["occupany_predictor"]['compute_occ_m2']
        self.compute_occ_m3 = self.config["occupany_predictor"]['compute_occ_m3']
        # ISS properties: define the maximal braking acceleration and the response time for bracking 
        self.iss_obstacle_params = {'a_max': self.maximal_acceleration, 'delta_brake': self.delta_brake}
        # SPOT properties
        self.spot_update_dict = {
            "obstacle": {
                0: {  # 0 means that all obstacles will be changed
                    "a_max": self.maximal_acceleration,
                    "v_max": self.maximal_velocity,
                    "compute_occ_m1": self.compute_occ_m1,
                    "compute_occ_m2": self.compute_occ_m2,
                    "compute_occ_m3": self.compute_occ_m3,
                    "onlyInLane": True,
                }
            },
            "egoVehicle":{
                0:{
                    "a_max": self.maximal_acceleration,
                    "compute_occ_m1": self.compute_occ_m1,
                    "compute_occ_m2": self.compute_occ_m2,
                    "compute_occ_m3": self.compute_occ_m3,
                    "onlyInLane": True
                }
            }
        }
        # variables for coarse trajectory sampling
        self.lane_list = None
        self.reference_lane_list = None
        self.reference_path_cossy = dict()
        self.cache_reference_path_cossy = defaultdict(lambda: None)
        self.ref_cossy = None
        self.max_acceleartion_lat = self.config["occupany_predictor"]['maximal_lateral_accleration']

    def reset(self, scenario: Scenario, planning_problem: PlanningProblem, reference_path_list: Dict):
        """
        resets the prediction model and set an initial information

        :param scenario: target secenario
        :param planning_problem: target planning problem
        :param reference_path_list: list of reference paths in reference manager
        """

        self.scenario = add_speed_limit(scenario, self.maximal_velocity)
        self.planning_problem = planning_problem
        self.dt = scenario.dt
        self.reference_path_list = reference_path_list

        # short term prediction
        self.short_term_prediction_time_number = self.config["occupany_predictor"]['prediction_time_horizon']

        # search and initialize the reference cossy
        # P.S. this reference path is resampled and extrapolated, but is not smoothed
        identifer = '_'.join(scenario.benchmark_id.split('_')[0:2])
        if self.cache_reference_path_cossy[identifer] is not None:
            self.reference_path_cossy = self.cache_reference_path_cossy[identifer]
        else:
            # generate cossy for different lane
            reference_path_cossy = dict()
            for key, value in self.reference_path_list.items():
                temp = OrderedDict()
                for sub_key, sub_value in value.items():
                    temp[sub_key] = sub_value[3]
                reference_path_cossy[key] = temp
            self.reference_path_cossy = reference_path_cossy
            self.cache_reference_path_cossy[identifer] = self.reference_path_cossy
        
        # history information for lidar observation
        self.lidar_based_history_observation = list()
        self.detected_dynamic_obstacles = None
    
    def update(self, obs_dict: Dict, obs_history_dict: Dict, state: State):
        """
        updates the oberservations periodically for SPOT prediction
        
        :param obs_dict: dictionary of current observations
        :param obs_history_dict: dictionary of historical observations
        :param state: current state of ego vehicle
        """  
        obs_tuples, dynamic_obstacles = self._extract_dynamic_obstacles(obs_dict, obs_history_dict)
        # current observation
        self.observations = obs_tuples
        # observed and tracked obstacles
        self.detected_dynamic_obstacles = dynamic_obstacles
        # generate SPOT toolbox
        self.get_short_term_prediction(state)

    def detect_conflict_obstacles(self, obs_dict, obs_history_dict, time_step):
        """
        :param obs_dict: dictionary of current observations
        :param obs_history_dict: dictionary of historical observations
        :param time_step: current time step

        :return 
            states of vehicles near intersection
        """
        # get detection area
        detection_area = obs_history_dict['lane_circ_surrounding_area'][-1]
        
        # obs_tuples, dynamic_obstacles = self._extract_dynamic_obstacles(obs_dict, obs_history_dict)
        # remove all lane-based obstacles
        scenario = deepcopy(self.scenario)
        if self.detected_dynamic_obstacles is not None:
            scenario.remove_obstacle(self.detected_dynamic_obstacles)

        conflict_obstacles_information = list()
        # lidar_based_obstacles = list()
        for obstacle in scenario.dynamic_obstacles:
            if (
                obstacle.initial_state.time_step
                <= time_step
                <= obstacle.prediction.trajectory.final_state.time_step
            ):
                obstacle_state = obstacle.state_at_time(time_step)
                obstacle_point = pycrcc.Point(
                    obstacle_state.position[0], obstacle_state.position[1]
                )
                if detection_area.collide(obstacle_point):
                    conflict_obstacles_information.append(obstacle_state)
                    # lidar_based_obstacles.append((obstacle, obstacle_state))

        # lidar_based_state = self.lidar_state_generation(ego_state, lidar_based_obstacles)
        
        return conflict_obstacles_information


    ###############################################################################################
    #                                   occupancy prediction                                      #
    ###############################################################################################

    def get_short_term_prediction(self, ego_state: State, predicted_time_step = None):
        """
        generates the occupancy list (Cartesian) and velocity interval for each detected obstacle
        
        :param ego_state: state of ego vehicle
        :param predicted_time_step: the predicted time horizon with the SPOT toolbox
        (the occunpancy and velocity list are save in the class attributes)
        """

        if predicted_time_step is None:
            predicted_time_step = self.short_term_prediction_time_number

        occupancy_prediction_dict = dict()
        velocity_prediction_dict = dict()
        predicted_obstacles = list()
        # generate reachable set 
        cpp_obstacles = self._spot_prediction(self.detected_dynamic_obstacles, ego_state, 
                                            start_time=0, predicted_step=predicted_time_step)
        
        for k, obstacle in enumerate(deepcopy(self.detected_dynamic_obstacles)):
            time_step = obstacle.initial_state.time_step
            occupancy_list, velocities_list = generate_occupancy_set(cpp_obstacles[k], time_step)
            if len(occupancy_list) != 0:
                obstacle.prediction = SetBasedPrediction(time_step+1, occupancy_list[0:])
                # obstacle_prediction_dict[obstacle.obstacle_id] = obstacle
                occupancy_prediction_dict[obstacle.obstacle_id] = occupancy_list
                velocity_prediction_dict[obstacle.obstacle_id] = velocities_list
                predicted_obstacles.append(obstacle)
            else:
                continue
        
        self.predicted_obstacles = predicted_obstacles
        self.short_occupancy_prediction = occupancy_prediction_dict
        self.short_velocity_prediction = velocity_prediction_dict

    def generate_iss_prediction(self, lanelet_id, fork_id, time_step = None):
        """
        generates ISS prediction object for planner action selection
        all included objects are: 1. dynamic obstacles, 2. coflict region (collision obstacle)
        :param collision object of the conflict region
        :param reference lane: the reference lane of the vehicle
        :param predicted time step

        return: ISS obstacles generated by selected predicted occupancies
        """
        if time_step is None:
            time_step = self.short_term_prediction_time_number

        reference_lane = self.reference_path_list[lanelet_id][fork_id]
        iss_obstacle_list = list()
        # firstly check some obstacles

        for obstacle in self.detected_dynamic_obstacles:
            obstacle_id = obstacle.obstacle_id
            position = obstacle.initial_state.position[np.newaxis,:]
            if reference_lane[1].contains_points(np.concatenate((position, [[0, 0]]), axis=0))[0]:
                # occupancy bound       
                cartesian_occupancy = self.short_occupancy_prediction.get(obstacle_id)[time_step-1]
                # velocity bound 
                velocity_bound = self.short_velocity_prediction.get(obstacle_id)[time_step-1]
                curvi_occupancy_bound = self._get_occupancy_in_cossy(cartesian_occupancy, reference_lane)
                if curvi_occupancy_bound is None:
                    continue
                curvi_position = self._get_curvilinear_state(obstacle.initial_state, reference_lane)
                # generate ISS dynamic obstacle
                iss_occ = ISSDynamicOccupancy(self.iss_obstacle_params, 
                                curvi_occupancy_bound[0], curvi_occupancy_bound[1], 
                                velocity_bound[1][0], velocity_bound[1][1])

                iss_obstacle_list.append((obstacle_id, curvi_position[0], iss_occ))

        # re-order the obstcles with there distance
        iss_obstacle_list.sort(key=lambda x:x[1])

        return iss_obstacle_list if len(iss_obstacle_list) != 0 else None

    def check_position_velocity_pair_validity(self, iss_obstacle_list: List, s_v_pair: list):
        """
        checks if the position-velocity pair is valid
        """
        idx = len(list(obs[1] for obs in iss_obstacle_list if obs[1]< s_v_pair[0]))
        # get following and leading vehicles
        occ_follow_tuple = iss_obstacle_list[idx-1] if idx > 0 else None
        occ_lead_tuple = iss_obstacle_list[idx] if idx < len(iss_obstacle_list) else None
        # following vehicle
        if occ_follow_tuple is not None:
            delta_s = s_v_pair[0] - occ_follow_tuple[2].s_furthest
            if delta_s < 0:
                return False
            v_min = occ_follow_tuple[2].safe_leading_velocity_brake(delta_s, 
                                        self.maximal_acceleration, self.delta_brake)
            v_min = max(self.limit_v_min, v_min)
        else:
            v_min = self.limit_v_min
        # leading vehicle
        if occ_lead_tuple is not None:
            delta_s = occ_lead_tuple[2].s_closest - s_v_pair[0]
            if delta_s < 0:
                return False
            v_max = occ_lead_tuple[2].safe_following_velocity_brake(delta_s, 
                                        self.maximal_acceleration, self.delta_brake)
            v_max = min(self.limit_v_max, v_max)                            
        else:
            v_max = self.limit_v_max
        # check validity
        print(f"v_max: {v_max}, v_current: {s_v_pair[1]}, v_min: {v_min}")
        if v_min <= s_v_pair[1] <= v_max:
            return True
        else:
            return False

    def _get_curvilinear_state(self, state: State, reference_lane):
        """
        gets the corresponding position in curvilinear corrdinate system 

        :param state: current state of vehicle
        :param reference_lane: current reference lane
        
        :return:
            positions in curvilinear coordinate system
        """
        ref_cossy = reference_lane[3]
        try:
            pos_vertice_curvilin = ref_cossy.convert_to_curvilinear_coords(
                                    state.position[0], state.position[1])
        except ValueError:
            # project the results to the center line
            idx = np.argmin(np.linalg.norm(reference_lane[3].reference() - state.position, axis=1))
            pos = reference_lane[3].reference()[idx]
            pos_vertice_curvilin = reference_lane[3].convert_to_curvilinear_coords(pos[0], pos[1])

        return pos_vertice_curvilin

    def _get_occupancy_in_cossy(self, cartesian_occupancy, reference_lane):
        """
        gets verices of occupancy polygon in curvilinear system
        :params reference lane: the reference lane of curvilinear system
        :params occupancy: the source occupancy

        :return:
            over-approximation of predicted occupancy
        """
        # reference of the target lane
        on_reference_lane = False
        ref_cossy = reference_lane[3]

        s_bound = dict()
        list_s_vertices_curvilin = []
        for plg in cartesian_occupancy.shape.shapes:
            center_plg = plg.center
            # check if the prediction is in current reference lane
            [lanelet_ids_plg] = self.scenario.lanelet_network.find_lanelet_by_position([center_plg])
            for lanelet_id in lanelet_ids_plg:
                if lanelet_id in reference_lane[0]:
                    # get vertices of the polygon under CCS
                    for pos_vertice_cart in plg.vertices:
                        # convert the occupancy to cossy
                        # remove out-of-domain points
                        try:
                            pos_vertice_curvilin = ref_cossy.convert_to_curvilinear_coords(
                                pos_vertice_cart[0], pos_vertice_cart[1])
                        except ValueError:
                            continue
                        list_s_vertices_curvilin.append(pos_vertice_curvilin[0])
                    break
        
        if len(list_s_vertices_curvilin) != 0:
            s_closest = min(list_s_vertices_curvilin)
            s_furthest = max(list_s_vertices_curvilin)
        else:
            return None

        return [s_closest, s_furthest] # curvilinear occupancy bound

    def _extract_dynamic_obstacles(self, obs_dict: Dict, obs_history_dict: Dict):
        """
        extracts dynamic obstacles from observation dictionary

        :param obs_dict: dictionary of current observations
        :param obs_history_dict: dictionary of historical observations

        :return:
            information tuple for dynamic obstacles 
            (position_number, obstacle object, relative velocity, relative position)
            dynamic obstacle objects
        """
        # extract obstacles
        obs_tuple = list()
        dynamic_obstacles = list()
        try:
            # TODO: here we only consider the circle detection area
            for k in range(len(obs_history_dict["lane_circ_obs"][-1])):
                state = obs_history_dict["lane_circ_obs"][-1][k]
                if state is not None:
                    for obstacle in deepcopy(self.scenario.dynamic_obstacles):
                        if (obstacle.state_at_time(state.time_step) is not None and 
                        (obstacle.state_at_time(state.time_step).position == state.position).all()
                        and obstacle.obstacle_type is not ObstacleType.PEDESTRIAN):
                        # TODO: not include the static obstacle
                            obstacle.initial_state = state
                            rel_vel = obs_dict["lane_circ_v_rel"][k]
                            rel_pos = obs_dict["lane_circ_p_rel"][k]
                            obs_tuple.append((k, obstacle, rel_vel, rel_pos))
                            dynamic_obstacles.append(obstacle)
        except IndexError:
            print("no obstacle is detected")
            pass
        # return a long_term collision object, and an iss collision object
        return obs_tuple, dynamic_obstacles

    def _spot_prediction(self, obstacles, ego_state: State, start_time: float = 0.0, predicted_step: int = 25):
        """
        does the short term prediction with SPOT toolbox
        :param obstacles: the list of obstacles to be predicted
        :param start_time: time to start the prediction
        :param predicted_step: time inverval of the predicted reachable set

        :return:
            predicted occupancies (velocity bound)
        """
        # planning_problem = deepcopy(self.planning_problem)
        # planning_problem.initial_state = ego_state

        # generate state and planning problem object
        slip_angle = 0.0
        # TODO: reconstruct the planning problem
        new_ego_state = State(**{'position': ego_state.position,
                                'orientation': ego_state.orientation,
                                'time_step': ego_state.time_step, 'velocity': ego_state.velocity,
                                'yaw_rate':ego_state.yaw_rate,
                                'slip_angle': slip_angle,
                                'acceleration': ego_state.acceleration})

        planning_problem = PlanningProblem(self.planning_problem.planning_problem_id, 
                                        new_ego_state, self.planning_problem.goal)
        
        spot.setLoggingMode(0)
        spot.registerScenario(1, self.scenario.lanelet_network.lanelets, obstacles, [planning_problem])
        spot.updateProperties(1, self.spot_update_dict)
        cpp_obstacles = spot.doOccupancyPrediction(1, float(start_time), 
                                        float(self.dt), float(predicted_step*self.dt + start_time), 8)
        spot.removeScenario(1)
        return cpp_obstacles
    
        
def generate_occupancy_set(cpp_obstacles, time_step):
    """
    generate a set-based prediction object for each obstacle
    """
    occupancy_list = list()
    velocities_list = list()
    i = time_step
    # each time step
    for vertices_at_time_step in cpp_obstacles[1]:
        occ = Occupancy(i + 1, ShapeGroup([]))
        velocity_interval = [i + 1, []]
        j = 1  # iterator over vertices_at_time_step
        b = 0  # index to select vertices_at_time_step that are the start of a new polygon
        while j < len(vertices_at_time_step[1]):
            compare_vertice = vertices_at_time_step[1][b]  # first vertice of next polygon
            if compare_vertice[0] == vertices_at_time_step[1][j][0] and compare_vertice[1] == \
                    vertices_at_time_step[1][j][1]:
                if (j + 1) - b < 3:  # polygon has less than 3 vertices
                    print('Warning: one duplicated vertice skipped when copying predicted occupancies to CommonRoad')
                    b += 1  # try next vertice as first vertice (in case of equal vertices directly after each other)
                else:
                    # this operation will extract all polygons in each occupancies
                    # maybe there are more than one polygon in each occupancy
                    shape_obj = Polygon(np.array(vertices_at_time_step[1][b:j + 1]))
                    # a share group has more than one shape, so its properties has a "shapes"
                    occ.shape.shapes.append(shape_obj)
                    j += 1
                    b = j
            j += 1
        # why we need to check the last polygon of the generated occupancy
        assert b == j - 1, ('Last polygon not closed (at time_step = ', i, ', b = ', b)
        occupancy_list.append(occ)

        v_min = vertices_at_time_step[0][0][1]
        v_max = vertices_at_time_step[0][0][2]

        velocity_interval[1].append(v_min)
        velocity_interval[1].append(v_max)
        velocities_list.append(velocity_interval)
        i += 1

    return occupancy_list, velocities_list


# add the speed limit of the scenario for the use of spot toolbox
def add_speed_limit(scenario, limited_speed=20.0):
    """
    sets speed limit for each scenario

    :return:
        modified scenario
    """
    for lanelet in scenario.lanelet_network.lanelets:
        lanelet.speed_limit = limited_speed
    return scenario

