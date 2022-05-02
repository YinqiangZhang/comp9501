"""
Module containing the action base class
"""

__author__ = "Hanna Krasowski"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Development"

import time
import yaml
import cProfile
import numpy as np
from copy import deepcopy
from scipy import spatial
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, Dict, List, Tuple
from shapely.geometry import LineString, Point

from commonroad.scenario.trajectory import State
from commonroad.scenario.scenario import Scenario
from commonroad.common.util import make_valid_orientation
from commonroad.planning.planning_problem import PlanningProblem

import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
)
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics

# commonroad_rl
from commonroad_rl.gym_commonroad.vehicle import Vehicle
from commonroad_rl.gym_commonroad.reference_path import ReactivePlannerReferencePathManager
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

# commonroad_rp
from commonroad_rp.reactive_planner import ReactivePlanner
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

N_INTEGRATION_STEPS = 4


class Action(ABC):
    """
    Description:
        Abstract base class of all action spaces
    """
    def __init__(self):
        """ Initialize empty object """
        super().__init__()

    @abstractmethod
    def step(self, action: Union[np.ndarray, int], ego_vehicle: Union[Vehicle]) -> Union[Vehicle]:
        """
        Function which acts on the current state and generates the new state
        :param action: current action
        :param ego_vehicle: current ego vehicle
        :return: ego vehicle which includes the new collision object and state
        """
        pass


class HighLevelAction(Action):
    """
    Description:
        Abstract base class of all discrete action spaces. Each high-level discrete
        action is converted to a low-level trajectory by a specified planner.
    """
    def __init__(self):
        """ Initialize empty object """
        super().__init__()

    def step(self, action: Union[np.ndarray, int], ego_vehicle: Union[Vehicle]) -> Union[Vehicle]:
        """
        Function which acts on the current state and generates the new state
        :param action: current action
        :param ego_vehicle: current ego vehicle
        :return: New state of ego vehicle
        """
        state = self._get_new_state(action, ego_vehicle)
        ego_vehicle.set_current_state(state)
        ego_vehicle.update_collision_object()
        return ego_vehicle

    @abstractmethod
    def _get_new_state(self, action: Union[np.ndarray, int], ego_vehicle: Union[Vehicle]) -> State:
        """function which return new states given the action and current state"""
        pass


class ReactivePlannerAction(HighLevelAction):
    """
    Description:
        Discrete / High-level action class with reactive planner
    """
    def __init__(self):
        """ 
        Initializes high-level action object with reactive planner
        : params meta_scenario_reset_dict: target meta scenario
        : params planning horizon: time horizon of planning  
        """
        super().__init__()
        
        self._DEBUG = False
        self.config_file = PATH_PARAMS["project_configs"]
        with open(self.config_file, "r") as root_path:
            self.config = yaml.safe_load(root_path)

        # default parameters
        self.replan_horizon = self.config['low_level_planner']['replan_horizon']
        self.max_orientation_diff = self.config['low_level_planner']['max_orientation_diff']
        self.reference_lane_threshold = self.config['low_level_planner']['reference_lane_threshold']
        self.acc_action_set = self.config['low_level_planner']['accleration_action_set']
        # acceleration mode
        self.enable_acceleration_action = True
        # reference path manager 
        self.reference_manager = ReactivePlannerReferencePathManager()

    def reset(self, scenario: Scenario, ego_vehicle: Union[Vehicle]):
        """
        reset the reference manager with new scenario
        """
        self.scenario = scenario
        self.reference_manager.reset(scenario)
        self.collision_checker = pycrcc.ShapeGroup()

        # parameters for lane changing
        self.dt = scenario.dt
        self.t_h = self.config['low_level_planner']['planning_time_horizon']
        self.t_min = self.config['low_level_planner']['planning_time_horizon']
        # self.delta_t = 1
        self.i = 0 
        self.predicted_state_steps = self.config['low_level_planner']['predicted_state_steps']
        self.lateral_offsets = self.config['low_level_planner']['lateral_distance_offset']

        # behavior state
        self.optimal_results = None
        self.lane_changing_state = {'left': False, 'right': False, 'middle': False}
        self.lane_changing_finished = False
        self.failed_trajectory_sampling = False
        self.history_action = list()
        self.new_reference = False
        self.previous_action = None
        
        # vehicle state 
        self.x_0 = None
        self.next_state_cl = None
        self.next_state = None  
        self.projected_source_lanelet_id = None
        self.projected_target_lanelet_id = None

        # reactive planner for running mode
        self.planner = ReactivePlanner(self.dt, self.t_h, int(self.t_h / self.dt))
        self.planner.set_t_sampling_parameters(self.t_min , self.dt, self.t_h)
        self.planner.set_d_sampling_parameters(-self.lateral_offsets, self.lateral_offsets)
        self.veh_param = parameters_vehicle2()
        self.planner._width = self.veh_param.w
        self.planner._length = self.veh_param.l

        # maneuver results
        # used for action prediction
        self.enable_safe_prediction = True
        self.safe_maneuver_results = None
        self.maneuver_ref_cossys = None

    def step(self, action: Union[np.ndarray, int], ego_vehicle: Union[Vehicle]) -> Union[Vehicle]:
        """
        Function which acts on the current state and generates the new state
        
        :param action: current action
        :param ego_vehicle: current ego vehicle

        :return: 
            ego vehicle updated with new state
        """
        new_state = self._get_new_state(action, ego_vehicle)
        new_state = new_state if new_state is not None else ego_vehicle.state 
        
        ego_vehicle.current_time_step += 1
        ego_vehicle._update_state(new_state)    
        ego_vehicle._update_collision_object()
        
        return ego_vehicle

    def _get_new_state(self, action: Union[np.ndarray, int], ego_vehicle: Union[Vehicle]) -> State:
        """
        gets next state for planning
        """
        new_replan = False
        new_reference = False
        failed_trajectory_sampling = False
        # set current state
        self.x_0 = ego_vehicle.state
        self.history_action.append(action)
        # unpack action
        
        direction_action, lane_action, planner_action = action

        # check current lane changing state
        new_lane_changing_state = \
            self._update_lane_changing_state(lane_action, self.lane_changing_finished)
        differ = set(new_lane_changing_state.items()) ^ set(self.lane_changing_state.items())
        self.lane_changing_state = new_lane_changing_state
        
        # new replan for state changing
        if len(differ) != 0:
            new_replan |= True
        
        # replan periodically
        if self.i > self.replan_horizon:
            new_replan |= True
        
        # new replan for direction action
        if self.previous_action is None or self.previous_action[0] != direction_action:
            # print("new direction action")
            new_replan |= True
        
        # new replan for planner action
        if self.previous_action is None or self.previous_action[2] != planner_action:
            # print("new planner action")
            new_replan |= True
        
        # replan 
        if new_replan:

            # set reference path 
            # if self.previous_action is None or self.previous_action[:2] != action[:2]:
                # self.reference_manager.update_reference_path(action[:2], self.x_0, self.delta_t)
                # self.planner.set_reference_path(self.reference_manager.reference_path)
                # self.next_state_cl = None
            
            # update reference manager and conflict region when action is changed
            if self.enable_safe_prediction:
                maneuver_action = self.config['agent_environment_interface']['lane_action_num'] * direction_action + lane_action
                ref_cossy, target_id, fork_id = self.maneuver_ref_cossys[maneuver_action]
                ref_lane = self.reference_manager.reference_path_list[target_id][fork_id]
                self.reference_manager.destination_reference_lane = ref_lane
                self.planner._co = ref_cossy
                
                if self.enable_acceleration_action:
                    self.optimal_results = self.safe_maneuver_results[maneuver_action][planner_action]
                else:
                    self.optimal_results = self.safe_maneuver_results[maneuver_action]
            else:
                # normal planning (2 modes)
                if self.enable_acceleration_action:
                    self.optimal_results = self.planner.plan(self.x_0, self.collision_checker, 
                                self.next_state_cl, [self.acc_action_set[planner_action]])
                else:
                    self.optimal_results = self.planner.plan(self.x_0, self.collision_checker, 
                                self.next_state_cl, self.acc_action_set)

            if self.optimal_results is None:
                self.failed_trajectory_sampling = True
            else:
                self.optimal_trajectory = self.planner.shift_orientation(self.optimal_results[0])
                self.i = 1
        else:
            self.i += 1
        
        self.previous_action = action
        # obtain next state
        if not self.failed_trajectory_sampling:
            next_state = self.optimal_trajectory.state_list[self.i]
            self.predicted_positions = [self.optimal_trajectory.state_list[t+self.i].position for t in self.predicted_state_steps]
        else:
            next_state = None
            self.predicted_positions = [self.optimal_trajectory.state_list[t+self.i].position for t in [0,0,0,0,0]]

        if not self.failed_trajectory_sampling:
            # next state in curvilinear sysyem
            # self.next_state_cl = (self.optimal_results[2][self.i], self.optimal_results[3][self.i])
            self.next_state = next_state
            # record for action mask
            # self.projected_source_lanelet_id = self.reference_manager.projected_source_lanelet_id(next_state)
            self.projected_target_lanelet_id = self.reference_manager.projected_target_lanelet_id(next_state)
            
            self.new_reference = new_reference
            self.lane_changing_finished = False
            if (lane_action != 1 and 
                self.reference_manager.check_lane_change_finished(next_state, 
                                                                self.reference_lane_threshold,
                                                                self.max_orientation_diff,
                                                                )
                ):
                self.lane_changing_finished = True
            # check the road_fork of the next state
            if (lane_action != 1 and 
                self.reference_manager.check_lane_change_fork_changed(next_state, self.projected_target_lanelet_id)):
                self.lane_changing_finished = True

            if self._DEBUG:
                print('########################################################################')
                print('Chosen Action:')
                print(f'Direction: {direction_action}, Lane action {lane_action}, Planner action {planner_action}')
                print('------------------------------------------------------------------------')
                print('Vehicle Behavior State:')
                print(f"Left lane changing: \t{self.lane_changing_state['left']}")
                print(f"Middle lane running: \t{self.lane_changing_state['middle']}")
                print(f"Right lane changing: \t{self.lane_changing_state['right']}")
                print('------------------------------------------------------------------------')
                print("Vehicle Time State")
                print(f'time step in current trajectory \t{self.i}')
                print(f'Finished: \t{self.lane_changing_finished}')
                print(f'Failed trajectory sampling \t{self.failed_trajectory_sampling}')
                print('########################################################################')

        return next_state

    def get_update_info(self):
        """
        gets information for action mask update 
        """
        info_dict = dict()
        info_dict["next_lanelet_id"] = self.projected_target_lanelet_id
        info_dict["next_state"] = self.next_state
        info_dict["lane_changing_finished"] = self.lane_changing_finished
        info_dict["next_new_replan"] = self.i > self.replan_horizon
        info_dict["reference_lane"] = self.reference_manager.destination_reference_lane

        return info_dict

    def _update_lane_changing_state(self, lane_action: int, lane_change_finished: bool):
        lane_changing_state = {'left': False, 'right': False, 'middle': False}
        if lane_action == 0:
            lane_changing_state['right'] = True
        elif lane_action == 1:
            lane_changing_state['middle'] = True
        elif lane_action == 2:
            lane_changing_state['left'] = True
        if self._check_lane_changing_transition(lane_changing_state, lane_change_finished):
            return lane_changing_state
        else:
            raise ValueError("wrong lane change state")
    
    def _check_lane_changing_transition(self, lane_changing_state: Dict, lane_change_finished: bool):
        # check transition
        if self.lane_changing_state['middle']:
            pass
        elif self.lane_changing_state['left']:
            if not lane_change_finished:
                assert lane_changing_state['left'] is True and lane_changing_state['right'] is False \
                    and lane_changing_state['middle'] is False, "wrong action during left lane changing"
            else:
                pass
        elif self.lane_changing_state['right']:
            if not lane_change_finished:
                assert lane_changing_state['right'] is True and lane_changing_state['left'] is False \
                    and lane_changing_state['middle'] is False, "wrong action during right lane changing"
            else:
                pass
        # check state without no more than one True 
        assert len([s for s in lane_changing_state.values() if s is True]) == 1, "wrong state structure"
        return True


class LowLevelAction(Action):
    """
    Description:
        Continuous / Low-level action class
    """
    def __init__(self, vehicle_dynamics: VehicleDynamics):
        """ Initialize object """
        super().__init__()
        self._rescale_factor = np.array(
            [
                vehicle_dynamics.input_bounds.ub[1],
                (vehicle_dynamics.input_bounds.ub[0]- vehicle_dynamics.input_bounds.lb[0]) / 2.0,
            ]
        )
        self._rescale_bias = np.array(
            [0.0, (vehicle_dynamics.input_bounds.ub[0] + vehicle_dynamics.input_bounds.lb[0]) / 2.0]
        )

    def step(self, action: Union[np.ndarray, int], ego_vehicle: Union[Vehicle]) -> Union[Vehicle]:
        """
        Function which acts on the current state and generates the new state
        :param action: current action
        :param ego_vehicle: current ego vehicle
        :return: New state of ego vehicle
        """
        rescaled_action = self.rescale_action(action)
        new_state = self._get_new_state(rescaled_action, ego_vehicle)
        ego_vehicle.set_current_state(new_state)
        ego_vehicle.update_collision_object()
        return ego_vehicle

    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range
        :param action: action from the CommonroadEnv.
        :return: rescaled action
        """
        return self._rescale_factor * action + self._rescale_bias

    @staticmethod
    def _get_new_state(action: np.ndarray, vehicle) -> State:
        # generate the next state for the given action
        new_state = vehicle.get_new_state(action)
        return new_state
