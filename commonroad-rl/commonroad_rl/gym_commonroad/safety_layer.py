import time
import yaml
import pickle
import warnings
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict

from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

import commonroad_dc.pycrcc as pycrcc

from commonroad_rl.gym_commonroad.utils.scenario import get_nearest_lanelet_fork
from commonroad_rl.gym_commonroad.utils.scenario import get_lane_relative_heading
from commonroad_rl.gym_commonroad.conflict_zone import ConflictZone
from commonroad_rl.gym_commonroad.occupancy_predictor import BehaviorPredictior
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

# used for trajectory prediction
from commonroad_rp.reactive_planner import ReactivePlanner
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker


__author__ = "Yinqiang Zhang"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "0.1"
__email__ = "yinqiang.zhang@tum.de"
__status__ = "Development"


class BaseActionMaskGenerator(ABC):
    """
    Description:
        Abstract Base Class for the action mask
    """
    def __init__(self) -> None:
        # current action list
        self._action_mask = OrderedDict()
        # scenario information
        self.scenario = None
        # execuated action list
        self.executed_action_list = list()

    @abstractmethod
    def update(self, info_dict: Dict) -> None:
        """ 
        updates the internal state for action mask generation
        :param info_dict: other update information as dictionary
        """
        pass

    def get_action_mask(self, flatten: bool = True) -> np.array:
        """
        get current action mask for further action selection
        :param flatten: whether get the flattened action mask
        :return: the processed action mask
        """
        action_mask = list()
        for mask in self._action_mask.values():
            if flatten:
                action_mask.append(mask.T.flatten().tolist())
            else:
                raise ValueError("finally we want to use the flatten action representation")
        
        return action_mask


class SafetyLayer(BaseActionMaskGenerator):
    """
    Description:
        safety layer for generation safe action masks
        ---------- maneuver action mask -----------------
        1. lane change action [0:right, 1:middle, 2:left]
        2. direction action [0:left, 1:middle, 2:right]
        ---------- velocity action mask -----------------
        3. planner action in terms of different 
        average acceleration [-4.0, -2.0, -1.0, 0, 1.0, 2.0, 4.0]

        example: action mask without flatten
        (1), (2) maneuver action
                        |right lane |middle lane| left lane | 
        -------------------------------------------------------
        left direction  |     0     |     1     |     1     |
        middle direction|     0     |     1     |     0     |
        right direction |     0     |     0     |     0     |
        
        (3) velicty action of planner
        acceleration    | -4.0  | -2.0  | -1.0  | +0.0  | +1.0  | +2.0  | +4.0  |
        -------------------------------------------------------------------------
        action mask     |   0   |   1   |   1   |   1   |   0   |   0   |   0   |
    """

    def __init__(self, acceleration_mode=1, is_lane_safe=True, 
                is_intersection_safe=True, enable_intersection_states=True,
                reaching_time_prediction=True):
        """
        initializes settings for designed safety layer.
        
        :param acceleration_mode: sets whether the agent uses accelearation action or not. 
        :param is_lane_safe: whether safety layer uses invariably safe braking sets
        :param is_intersection_safe: whether safety layer uses conflict zones
        :param enable_intersection_states: whether safety layer uses intersection-related states
        :param reaching_time_prediction: decides which method is used to predict reaching time
        """
        super(SafetyLayer, self).__init__()
        
        # safe mode
        self.is_lane_safe = is_lane_safe
        self.is_intersection_safe = is_intersection_safe
        self.config_file = PATH_PARAMS["project_configs"]
        with open(self.config_file, "r") as root_path:
            self.config = yaml.safe_load(root_path)
        
        # acceleration mode
        if acceleration_mode == 1:
            # acceleration are decided by policy 
            self.enable_acceleration_action = True
            self.acceleration_action_num = self.config['agent_environment_interface']['acceleration_action_num']
        elif acceleration_mode == 0:
            # acceleration are decided by planner
            self.enable_acceleration_action = False
            self.acceleration_action_num = 1
            
        self.direction_action_num = self.config['agent_environment_interface']['direction_action_num']
        self.lane_action_num = self.config['agent_environment_interface']['lane_action_num']
        
        # set whether use additional intersection-based states
        self.enable_intersection_state = enable_intersection_states
        # safety mode
        self.reach_time_prediction = reaching_time_prediction
        # number of maneuver actions
        self.maneuver_action_num = self.direction_action_num * self.lane_action_num
        # lane fork in the last time step
        self.previous_nearest_fork: List or None = None
        # lane change avaliable distance
        self.lane_change_avaliable_distance = self.config['low_level_planner']['lane_change_avaliable_distance']
        # conflict zone
        self.conflict_region = ConflictZone(reaching_time_prediction=reaching_time_prediction)
        # occupancy prediction module
        self.behavior_prediction = BehaviorPredictior()
        # acceleration_set
        self.acc_action_set = self.config['low_level_planner']['accleration_action_set']
        # vehicle parameters 
        self.veh_param = parameters_vehicle2()
        # maximal acceleration value
        self.acceleration_to_stop = self.config['occupany_predictor']['maximal_acceleration']
                    
    def reset(self, scenario: Scenario, planning_problem: PlanningProblem, road_edge_collision_checker, 
                ego_vehicle, reference_path_list: Dict, raw_lane_list,
                observation_dict: Dict, observation_history_dict: Dict
            ):
        """
        resets the safety layer according to scenarios and planning problems

        :param scenario: target scenario of motion planning problem
        :param planning_problem: motion planning problem
        :param road_edge_collision_checker: collision model of road boundaries
        :param ego_vehicle: ego vehicle class
        :param reference_path_list: list of reference paths generated by reference manager
        :param raw_lane_list: original lane list generated by reference manager
        :param observation_dict: dictionary of current observations
        :param observation_history_dict: dictionary of history observations in this planning problem
        """
        self.current_step = 0
        self.reach_road_end = False
        self.conflict_obstacles_information = None
        self.unsafe_association = None
        self.enmergency_stop_feasible = False
        self.is_emergency_safe = False
        self.lane_change_abort = False
        self.back_risky = False

        self.scenario = scenario
        # reset conflict zones
        self.conflict_region.reset(scenario, lane_list=raw_lane_list)
        
        # reactive planner for trajectory prediction
        self.dt = scenario.dt
        self.t_h = self.config['low_level_planner']['planning_time_horizon']
        self.t_min = self.config['low_level_planner']['planning_time_horizon']
        lateral_offset = self.config['low_level_planner']['lateral_distance_offset']

        # planner used for predict the feasible trajectories
        self.predictive_planner = ReactivePlanner(self.dt, self.t_h, int(self.t_h / self.dt))
        self.predictive_planner.set_t_sampling_parameters(self.t_min , self.dt, self.t_h)
        self.predictive_planner.set_d_sampling_parameters(-lateral_offset, lateral_offset)
        self.predictive_planner._width = self.veh_param.w
        self.predictive_planner._length = self.veh_param.l

        # collision checker for predictive planner
        static_scenario = deepcopy(self.scenario)
        static_scenario.remove_obstacle(static_scenario.dynamic_obstacles)
        # add static obstacle in the checker
        self.static_cc_template = create_collision_checker(static_scenario)
        self.static_cc_template.add_collision_object(road_edge_collision_checker)
        
        # behavior prediction module
        self.behavior_prediction.reset(scenario, planning_problem, reference_path_list)
        
        # safe measurements
        if self.is_lane_safe and self.is_intersection_safe:
            self.behavior_prediction.update(observation_dict, observation_history_dict, ego_vehicle.state)
        
        # predict for the first time
        self.conflict_obstacles_information = self.behavior_prediction.detect_conflict_obstacles(observation_dict, observation_history_dict, self.current_step)
        self.unsafe_association, self.intersection_observations = self.conflict_region.get_unsafe_obstacle_association(self.conflict_obstacles_information)

        # generate initial action mask
        initial_position = ego_vehicle.state.position
        [initial_lanelet_id] = self.scenario.lanelet_network.find_lanelet_by_position([initial_position])
        # consider begin the planning problem on multiple lanelets
        if len(initial_lanelet_id) != 1:
            predecessor_list = list()
            relative_heading = list()
            for lanelet_id in initial_lanelet_id:
                ego_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                predecessor = ego_lanelet.predecessor[0]
                relative_heading.append(abs(get_lane_relative_heading(ego_vehicle.state, ego_lanelet)[0]))
                predecessor_list.append(predecessor)
            if len(set(predecessor_list)) == 1:
                initial_lanelet_id = predecessor_list[0]
            else:
                # planning problem: DEU_AAH-1_800204_T-1
                initial_lanelet_id = initial_lanelet_id[relative_heading.index(np.min(relative_heading))]
        else:
            initial_lanelet_id = initial_lanelet_id[0]

        maneuver_action_mask = self.get_default_action_mask(initial_lanelet_id)
        self.previous_nearest_fork = get_nearest_lanelet_fork(scenario, initial_lanelet_id)

        # not change the lane near the intersection
        ref_lane = self.behavior_prediction.reference_path_list[initial_lanelet_id][self.previous_nearest_fork[0]]
        lane_changing_feasible, delta_s_near, delta_s_far = self.conflict_region.check_near_intersection_area(ego_vehicle.state, ref_lane, self.lane_change_avaliable_distance)
        if not lane_changing_feasible:
            maneuver_action_mask[[0,2]] = np.zeros((self.direction_action_num,))
        
        # ego position raltive to the conflict zone 
        self.ego_conflict_data = np.array([delta_s_near, delta_s_far])

        # fix error: planning problem: DEU_AAH-1_900229_T-1 or similar
        # the velocity of ego vehicle is negative in intial state
        initial_state = ego_vehicle.state
        if initial_state.velocity < 0.0:
            initial_state.velocity = 0.0
            initial_state.acceleration = 0.0

        # safety verification
        self._action_mask['safe'] = self.check_trajectory_bundle(initial_state, initial_lanelet_id, maneuver_action_mask)
        
        # observe road fork
        #self.observed_road_fork = self.get_further_road_fork(1, False, initial_lanelet_id)
    
    def update(self, action: np.ndarray, observation_dict: Dict, observation_history_dict: Dict,
            next_lanelet_id: int, lane_changing_finished: bool, next_state: State, 
            next_new_replan: bool, reference_lane: List
            ):
        """
        updates the safety layer with features (information) from environment.

        param: observation_dict: dictionary of current observations
        param: observation_history_dict: dictionary of all previous observations in a planning problem
        param: next_lanelet_id: ID of lanelet in the next time step
        param: lane_changing_finished: whether the ego vehicle finishes changing the lane
        param: next_state: state in the next time step
        param: next_new_replan: whether ego vehicle replan its trajectory in the current time step
        param: reference_lane: information of the current followed lane
        """
        self.current_step += 1

        direction_action, lane_action, planner_action = action
        
        if next_lanelet_id is None:
            # the vehicle reach the road end
            self.reach_road_end = True
            self.executed_action_list.append(action)
            self._action_mask['safe'] = np.zeros((self.acceleration_action_num, self.maneuver_action_num))
            return
        
        # create joint mask
        lanelet_fork = get_nearest_lanelet_fork(self.scenario, next_lanelet_id)
        lane_changing_feasible, delta_s_near, delta_s_far = self.conflict_region.check_near_intersection_area(next_state, reference_lane, self.lane_change_avaliable_distance) 
        self.ego_conflict_data = np.array([delta_s_near, delta_s_far])

        maneuver_action_mask = np.zeros((self.lane_action_num, self.direction_action_num))
        if lane_action != 1 and not lane_changing_finished:
            maneuver_action_mask[lane_action, range(len(lanelet_fork))] = 1
        elif lane_changing_finished:
            maneuver_action_mask = self.get_default_action_mask(next_lanelet_id)
            if np.sum(maneuver_action_mask[[0, 2]]) != 0:
                if not lane_changing_feasible:
                    maneuver_action_mask[[0,2]] = np.zeros((self.direction_action_num,))
        elif lane_action == 1:
            # replan periodically or meet a new road fork
            if next_new_replan or self.previous_nearest_fork not in [lanelet_fork, None]:
                maneuver_action_mask = self.get_default_action_mask(next_lanelet_id)
                # check the distance to the intersection, near fork
                if np.sum(maneuver_action_mask[[0, 2]]) != 0:
                    if not lane_changing_feasible:
                        maneuver_action_mask[[0, 2]] = np.zeros((self.direction_action_num,))
            else:
                maneuver_action_mask[lane_action, direction_action] = 1

        # predicted surrounding states
        if self.enable_intersection_state:
            self.conflict_obstacles_information = self.behavior_prediction.detect_conflict_obstacles(observation_dict, observation_history_dict, self.current_step) 
            self.unsafe_association, self.intersection_observations = self.conflict_region.get_unsafe_obstacle_association(self.conflict_obstacles_information)

        if lane_action == 1:
            # replan periodically or meet a new road fork
            if next_new_replan or self.previous_nearest_fork not in [lanelet_fork, None]:
                if self.is_lane_safe and self.is_intersection_safe:
                    self.behavior_prediction.update(observation_dict, observation_history_dict, next_state)
                    if not self.enable_intersection_state:
                        self.conflict_obstacles_information = self.behavior_prediction.detect_conflict_obstacles(observation_dict, observation_history_dict, self.current_step)
                        self.unsafe_association, _ = self.conflict_region.get_unsafe_obstacle_association(self.conflict_obstacles_information)
                self._action_mask['safe'] = self.check_trajectory_bundle(next_state, next_lanelet_id, maneuver_action_mask)
            else:
                self._action_mask['safe'] = np.zeros((self.acceleration_action_num, self.maneuver_action_num))
                temp = np.zeros((self.acceleration_action_num,))
                temp[planner_action] = 1
                self._action_mask['safe'][:, self.direction_action_num * direction_action + lane_action] = temp.T
        else:
            # perform lane changing maneuver
            if lane_changing_finished:
                if self.is_lane_safe and self.is_intersection_safe:
                    self.behavior_prediction.update(observation_dict, observation_history_dict, next_state)
                    if not self.enable_intersection_state:
                        self.conflict_obstacles_information = self.behavior_prediction.detect_conflict_obstacles(observation_dict, observation_history_dict, self.current_step) 
                        self.unsafe_association, _ = self.conflict_region.get_unsafe_obstacle_association(self.conflict_obstacles_information)
                self._action_mask['safe'] = self.check_trajectory_bundle(next_state, next_lanelet_id, maneuver_action_mask)
            elif next_new_replan:
                if self.is_lane_safe and self.is_intersection_safe:
                    self.behavior_prediction.update(observation_dict, observation_history_dict, next_state)
                    if not self.enable_intersection_state:
                        self.conflict_obstacles_information = self.behavior_prediction.detect_conflict_obstacles(observation_dict, observation_history_dict, self.current_step)
                        self.unsafe_association, _ = self.conflict_region.get_unsafe_obstacle_association(self.conflict_obstacles_information)
                # replan for each road fork  
                feasible_direction_actions = range(len(lanelet_fork))
                self._action_mask['safe'] = self.check_lane_changing_trajectory_bundle(next_state, lane_action, feasible_direction_actions, next_lanelet_id)
                
                # if cannot perform lane change
                if (self._action_mask['safe'] == 0).all():
                    next_position = next_state.position
                    [lanelet_id_candidates] = self.scenario.lanelet_network.find_lanelet_by_position([next_position])
                    next_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(next_lanelet_id)
                    if lane_action == 0:
                        re_target_lanelet_id = next_lanelet.adj_left
                    elif lane_action == 2:
                        re_target_lanelet_id = next_lanelet.adj_right
                    self.lane_change_abort = False
                    if re_target_lanelet_id in lanelet_id_candidates:
                        self.lane_change_abort = True
                        new_action_mask = self.get_default_action_mask(re_target_lanelet_id)
                        new_action_mask[[0, 2]] = np.zeros((self.direction_action_num,))
                        self._action_mask['safe'] = self.check_trajectory_bundle(next_state, re_target_lanelet_id, new_action_mask)
            else:
                temp = np.zeros((self.maneuver_action_num, self.acceleration_action_num))
                temp[self.direction_action_num * direction_action + lane_action, planner_action] = 1
                self._action_mask['safe'] = temp.T

        # update previous lanelet fork
        self.previous_nearest_fork = lanelet_fork

        # update all variables
        self.executed_action_list.append(action)

        # generate surrounding road fork for obervation
        # self.observed_road_fork = self.get_further_road_fork(lane_action, lane_changing_finished, next_lanelet_id)

    def get_default_action_mask(self, lanelet_id: int) -> np.ndarray:
        
        """
        generates the initial maneuver mask according to road structures

        :param lanelet_id: ID of the lanelet
        return: 
            initial maneuver mask
        """
        ego_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        maneuver_action_mask = np.zeros([3,3])
        # current lanelet
        middle_lanelet_fork = get_nearest_lanelet_fork(self.scenario, lanelet_id)
        maneuver_action_mask[1, range(len(middle_lanelet_fork))] = 1
        # adjacent lanelets
        if ego_lanelet.adj_left_same_direction:
            left_lanelet_fork = get_nearest_lanelet_fork(self.scenario, ego_lanelet.adj_left)
            maneuver_action_mask[2, range(len(left_lanelet_fork))] = 1
        if ego_lanelet.adj_right_same_direction:
            right_lanelet_fork = get_nearest_lanelet_fork(self.scenario, ego_lanelet.adj_right)
            maneuver_action_mask[0, range(len(right_lanelet_fork))] = 1

        return maneuver_action_mask

    def check_trajectory_bundle(self, next_state: State, next_lanelet_id: int, 
                                maneuver_action_mask: np.ndarray):
        """
        safety verification of trajectories when driving in a lane
        
        :param next_state: state in the next time step
        :param next_lanelet_id: target lanelet id
        :param maneuver_action_mask: improved maneuver mask

        return:
            safe action mask
        """

        emergency_results = dict()
        maneuver_results = dict()
        maneuver_ref_cossys = dict()
        acceleration_mask_dict = dict()
        safety_level_results = dict()
        
        self.enmergency_stop_feasible = False
        next_feasible_actions = [idx for idx, action in enumerate(maneuver_action_mask.T.flatten().tolist()) if action == 1]

        for next_action in next_feasible_actions:

            next_direction_action = next_action // self.direction_action_num
            next_lane_action = next_action % self.direction_action_num

            if next_lane_action == 1:
                ego_lanelet_id = next_lanelet_id
            else:
                # only consider go straight action
                ego_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(next_lanelet_id)
                ego_lanelet_id = ego_lanelet.adj_right if next_lane_action == 0 else ego_lanelet.adj_left
            
            ref_cossys = self.behavior_prediction.reference_path_cossy[ego_lanelet_id]
            ref_cossy = list(cossy for cossy in ref_cossys.values())[next_direction_action]
            fork_lanelet_id = list(cossy for cossy in ref_cossys.keys())[next_direction_action]
            # save reference cossy 
            maneuver_ref_cossys[next_action] = (ref_cossy, ego_lanelet_id, fork_lanelet_id)
            
            # iss check
            if self.is_lane_safe:
                iss_obstacle_list = self.behavior_prediction.generate_iss_prediction(ego_lanelet_id, fork_lanelet_id)
            else:
                iss_obstacle_list = None

            # intersection safety
            if self.is_intersection_safe:
                ref_lane = self.behavior_prediction.reference_path_list[ego_lanelet_id][fork_lanelet_id]
                conflict_collsion_obejct = self.conflict_region.check_intersection_safety(next_state, ref_lane, 
                                            self.unsafe_association, self.acceleration_to_stop, self.current_step, self.t_h)

            combined_cc_template = deepcopy(self.static_cc_template)
            if self.is_intersection_safe:
                combined_cc_template.add_collision_object(conflict_collsion_obejct)

            self.predictive_planner._co = ref_cossy
            safety_level_back = np.zeros((self.acceleration_action_num,))
            safety_level_front= np.zeros((self.acceleration_action_num,))
            
            if self.enable_acceleration_action:
                results_list = list()
                acceleration_mask = np.zeros((self.acceleration_action_num,))
                for idx, acceleration in enumerate(self.acc_action_set):
                    
                    if next_state.velocity == 0.0 and acceleration < 0.0:
                        optimal_results = None
                    else:
                        # give the planner two sets of iss obstacles
                        # 1. the current lane (15)
                        # 2. the target lane (30)
                        optimal_results = self.predictive_planner.plan(next_state, combined_cc_template, 
                                                acceleration=[acceleration], iss_obstacle_list=iss_obstacle_list)

                        safety_level_back[idx] = self.predictive_planner.safety_level[0]
                        safety_level_front[idx] = self.predictive_planner.safety_level[1]

                        # re-generate trajectory for deceleration because of kinematically infeasible
                        num_of_collision = self.predictive_planner.no_of_infeasible_trajectories_collision()
                        if (num_of_collision == 0 
                            and optimal_results is None
                            and self.predictive_planner.iss_feasible
                            and next_state.velocity < 5.0 # a velocity threshold
                            and acceleration < 0.0
                            and next_lane_action == 1
                            ):
                            optimal_results = self.compute_brake_trajectory(next_state, acceleration, 
                                                                    ref_cossy, combined_cc_template)

                    if optimal_results is not None:
                        acceleration_mask[idx] = 1

                    results_list.append(optimal_results)
                
                # adapt the acceleration: back acceleration
                safe_back_acceleration_mask = np.ones((self.acceleration_action_num,))
                for idx, safety_level in enumerate(safety_level_back):
                    if safety_level != 0:
                        safe_back_acceleration_mask[idx] = 0

                combined_acceleration_mask = list(mask1 * mask2 for mask1, mask2 in zip(safe_back_acceleration_mask, acceleration_mask))
                
                self.back_risky = False
                if (np.array(combined_acceleration_mask) == 0).all() and (acceleration_mask != 0).any():
                    self.back_risky = True
                    valid_acceleration_actions = list(idx for idx, element in enumerate(acceleration_mask) if element == 1)
                    combined_acceleration_mask[valid_acceleration_actions[-1]] = 1
                
                acceleration_mask = np.array(combined_acceleration_mask)

                # emergency braking
                if (acceleration_mask == 0).all():
                    # print("potential emergency situation")
                    emergency_result = self.compute_brake_trajectory(next_state, -8, ref_cossy, combined_cc_template)
                    emergency_results[next_action] = emergency_result
                    if emergency_result is not None:
                        self.enmergency_stop_feasible = True

                maneuver_results[next_action] = results_list
                acceleration_mask_dict[next_action] = acceleration_mask
                safety_level_results[next_action] = (safety_level_back, safety_level_front)
            else:
                
                optimal_results = self.predictive_planner.plan(next_state, combined_cc_template, 
                            acceleration=self.acc_action_set, iss_obstacle_list=iss_obstacle_list)
                # front and back safety
                self.back_risky = False
                safety_level_back[0] = self.predictive_planner.safety_level[0]
                safety_level_front[0] = self.predictive_planner.safety_level[1]
                if safety_level_back[0] != 0:
                    self.back_risky = True

                maneuver_results[next_action] = optimal_results
                if optimal_results is None:
                    maneuver_action_mask[next_lane_action, next_direction_action] = 0

        # orgnize the action mask
        if self.enable_acceleration_action:
            final_action_mask = np.empty((1,0))
            for maneuver_action in range(maneuver_action_mask.size):
                if maneuver_action in acceleration_mask_dict.keys():
                    planner_action_mask =  np.array([acceleration_mask_dict[maneuver_action]])
                else:
                    planner_action_mask = np.zeros((1, self.acceleration_action_num))

                if final_action_mask.size == 0:
                    final_action_mask = planner_action_mask
                else:
                    final_action_mask = np.concatenate((final_action_mask, planner_action_mask), axis=0)
        else:
            final_action_mask = maneuver_action_mask.T
        
        # TODO:emergency maneuver
        if (final_action_mask == 0).all():
            if self.enmergency_stop_feasible:
                self.is_emergency_safe = True

        # save optimal trajectory for planning
        self.maneuver_results = maneuver_results
        self.maneuver_ref_cossys = maneuver_ref_cossys

        return final_action_mask.T

    def check_lane_changing_trajectory_bundle(self, next_state: State, lane_action:int,
                            direction_action_list: List, next_lanelet_id: int):
        """
        safety verification of trajectories when driving in a lane
        
        :param next_state: state in the next time step
        :param lane_action: selected lane action
        :param direction_action_list: feasible direction actions during replanning and lane-changing
        :param next_lanelet_id: target lanelet id

        return:
            safe action mask
        """

        final_action_mask = np.zeros((self.maneuver_action_num, self.acceleration_action_num))
        
        emergency_results = dict()
        maneuver_results = dict()
        maneuver_ref_cossys = dict()
        acceleration_mask_dict = dict()

        ref_cossys = self.behavior_prediction.reference_path_cossy[next_lanelet_id]

        for direction_action in direction_action_list:
            ref_cossy = list(cossy for cossy in ref_cossys.values())[direction_action]
            fork_lanelet_id = list(cossy for cossy in ref_cossys.keys())[direction_action]
            
            # iss check 
            if self.is_lane_safe:
                iss_obstacle_list = self.behavior_prediction.generate_iss_prediction(next_lanelet_id, fork_lanelet_id)
            else:
                iss_obstacle_list = None

            # intersection safety
            if self.is_intersection_safe:
                ref_lane = self.behavior_prediction.reference_path_list[next_lanelet_id][fork_lanelet_id]
                conflict_collsion_obejct = self.conflict_region.check_intersection_safety(next_state, ref_lane, 
                                            self.unsafe_association, self.acceleration_to_stop, self.current_step, self.t_h)
            
            # TODO: generate new collision checker
            combined_cc_template = deepcopy(self.static_cc_template)
            if self.is_intersection_safe:
                combined_cc_template.add_collision_object(conflict_collsion_obejct)

            self.predictive_planner._co = ref_cossy
            next_action = self.direction_action_num * direction_action + lane_action
            maneuver_ref_cossys[next_action] = (ref_cossy, next_lanelet_id, fork_lanelet_id)

            results_list = list()
            if self.enable_acceleration_action:
                acceleration_mask = np.zeros((self.acceleration_action_num,))
                safety_level_back = np.zeros((self.acceleration_action_num,))
                safety_level_front = np.zeros((self.acceleration_action_num,))

                for idx, acceleration in enumerate(self.acc_action_set):
                    optimal_results = self.predictive_planner.plan(next_state, combined_cc_template, 
                                    acceleration=[acceleration],  iss_obstacle_list=iss_obstacle_list)
                    
                    # back and front danger is not permitted
                    safety_level_back[idx] = self.predictive_planner.safety_level[0]
                    safety_level_front[idx] = self.predictive_planner.safety_level[1]

                    if np.sum(safety_level_back) != 0:
                        optimal_results = None
                        
                    if optimal_results is not None:
                        acceleration_mask[idx] = 1
                    results_list.append(optimal_results)
                
                # adapt the acceleration: back acceleration
                # safe_back_acceleration_mask = np.ones((self.acceleration_action_num,))
                # for idx, safety_level in enumerate(safety_level_back):
                #     if safety_level != 0:
                #         safe_back_acceleration_mask[idx] = 0

                # combined_acceleration_mask = list(mask1 * mask2 for mask1, mask2 in zip(safe_back_acceleration_mask, acceleration_mask))
                
                # if (np.array(combined_acceleration_mask) == 0).all() and (acceleration_mask != 0).any():
                #     valid_acceleration_actions = list(idx for idx, element in enumerate(acceleration_mask) if element == 1)
                #     combined_acceleration_mask[valid_acceleration_actions[-1]] = 1
                
                # acceleration_mask = np.array(combined_acceleration_mask)

                acceleration_mask_dict[next_action] = acceleration_mask
                maneuver_results[next_action] = results_list
            else:
                optimal_results = self.predictive_planner.plan(next_state, combined_cc_template, 
                                acceleration=self.acc_action_set, iss_obstacle_list=iss_obstacle_list)
                maneuver_results[next_action] = optimal_results
                results_list.append(optimal_results)            
        
        self.maneuver_results = maneuver_results
        self.maneuver_ref_cossys = maneuver_ref_cossys
        
        # orgnize the action mask
        if self.enable_acceleration_action:
            final_action_mask = np.zeros((self.maneuver_action_num, self.acceleration_action_num))
            for next_action in acceleration_mask_dict:
                final_action_mask[next_action, :] = acceleration_mask_dict[next_action]
        else:
            final_action_mask = np.zeros((self.lane_action_num, self.direction_action_num))
            for next_action in maneuver_results:
                if maneuver_results[next_action] is not None:
                    final_action_mask[next_action//self.direction_action_num, next_action%self.direction_action_num] = 1
        
        return final_action_mask.T

    def get_further_road_fork(self, lane_action, lane_changing_finished, next_lanelet_id):
        """
        extract road fork for observation
        """
        observed_road_fork = np.zeros([3,3])
        if lane_action == 1 or lane_changing_finished:
            ego_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(next_lanelet_id)
            middle_lanelet_fork = get_nearest_lanelet_fork(self.scenario, next_lanelet_id)
            observed_road_fork[1, range(len(middle_lanelet_fork))] = middle_lanelet_fork
            # adjacent lanelets
            if ego_lanelet.adj_left_same_direction:
                left_lanelet_fork = get_nearest_lanelet_fork(self.scenario, ego_lanelet.adj_left)
                observed_road_fork[2, range(len(left_lanelet_fork))] = left_lanelet_fork
            if ego_lanelet.adj_right_same_direction:
                right_lanelet_fork = get_nearest_lanelet_fork(self.scenario, ego_lanelet.adj_right)
                observed_road_fork[0, range(len(right_lanelet_fork))] = right_lanelet_fork
        elif lane_action != 1 and not lane_changing_finished:
            # lanelet_fork = get_nearest_lanelet_fork(self.scenario, next_lanelet_id)
            # observed_road_fork[lane_action, range(len(lanelet_fork))] = lanelet_fork
            observed_road_fork = self.observed_road_fork
        return observed_road_fork

    def compute_brake_trajectory(self, state, acceleration, ref_cossy, collision_checker):
        
        """
        computes trajectory for emergency braking

        :param state: current state of ego vehicle
        :param acceleration: desided acceleration
        :param ref_cossy: curvilinear corrdinate system of reference path 
        :param collision_checker: group of collision models for safe verifications

        return:
            braking trakectory
        """
        position = state.position
        v_0 = state.velocity
        s_0, d_0 = ref_cossy.convert_to_curvilinear_coords(position[0], position[1])
        # compute braking maneuver
        t_brake = int(abs(v_0/acceleration) / self.dt) * self.dt
        t = np.arange(0, t_brake + self.dt, self.dt)
        s_brake = s_0 + v_0 * (t) + 0.5 * acceleration * np.square(t)
        v_brake = v_0 + acceleration * t

        if t_brake < self.t_h:
            extended_length = int(self.t_h / self.dt) - len(t)
            s_final = s_0 + (v_0 * v_0) / abs(2 * acceleration)
            s_extend = s_final * np.ones((extended_length,))
            v_extend = np.zeros((extended_length,))
            s_brake = np.append(s_brake, s_extend)
            v_brake = np.append(v_brake, v_extend)
        
        x_brake = list()
        y_brake = list()
        shapes = list()
        # interpolate the theta
        ref_theta = ref_cossy.ref_theta()
        theta_diff = np.diff(ref_theta)
        diff_idx = np.argmax(abs(theta_diff))
        if max(abs(theta_diff)) >= np.pi:
            ref_theta[ref_theta > np.pi] -= 2 * np.pi

        # TODO: interpolation the theta between current orientation and the theta reference
        theta_brake = np.interp(s_brake, ref_cossy.ref_pos(), ref_theta)
        # if abs(theta_brake[0] % (2*np.pi) - state.orientation % (2*np.pi)) >= 0.2:
        #     print(abs(theta_brake[0] % (2*np.pi) - state.orientation % (2*np.pi)))
        #     return None
        
        # transform the theta
        for s in s_brake:
            try:
                x, y = ref_cossy.convert_to_cartesian_coords(s, d_0)
            except:
                # print(s, d_0)
                # with open(f'./emergency_{self.current_step}.pkl', 'wb') as f:
                #     pickle.dump((s, d_0), f)
                #     pickle.dump(ref_cossy.reference(), f)
                return None
                
            x_brake.append(x)
            y_brake.append(y)
        
        
        # compute list of rectangles
        ego_cc = pycrcc.TimeVariantCollisionObject(state.time_step)
        for i in range(len(x_brake)):
            shapes.append(State(time_step = state.time_step+i, 
                                position = np.array([x_brake[i], y_brake[i]]), 
                                orientation = theta_brake[i],
                                velocity = v_brake[i],
                                yaw_rate= 0.0, 
                                acceleration = acceleration))
            ego_cc.append_obstacle(
                pycrcc.RectOBB(0.5 * self.veh_param.l, 
                                0.5 * self.veh_param.w, 
                                theta_brake[i], x_brake[i], y_brake[i]))

        if collision_checker.collide(ego_cc):
            return None
        else:
            trajectory = Trajectory(initial_time_step=state.time_step, state_list=shapes)
            trajectory = self.predictive_planner.shift_orientation(trajectory, interval_start=0, interval_end= 2 * np.pi)
            return (trajectory, None, None, None)

