__author__ = "Peter Kocsis, Niels Muendler"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

"""
Module for goal related feature extraction of the CommonRoad Gym envrionment
"""
import warnings
import numpy as np
from typing import List, Tuple
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_route_planner.route_planner import RoutePlanner


class GoalObservation:
    """
    Class responsible for extracting observations related to the goal
    """

    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem):
        """
        Initialize instance, which involves route planning to the goal
        :param scenario: The used scenario
        :param planning_problem: The planning problem which contains the initial position and the goal
        """
        self.scenario = scenario
        self.lanelet_network = scenario.lanelet_network
        self.planning_problem = planning_problem

        self.route_planner = RoutePlanner(
            scenario,
            planning_problem,
            backend=RoutePlanner.Backend.NETWORKX_REVERSED,
            log_to_console=False,
        )

        self.route_candidates = self.route_planner.get_route_candidates()
        self.route = self.route_candidates.get_most_likely_route_by_orientation()
        self.navigator = self.route.get_navigator()

    def get_long_lat_distance_to_goal(
        self, position: np.ndarray
    ) -> Tuple[float, float]:
        """
        Get longitudinal and lateral distances to the goal over the planned route
        :param position: The position which distance is queried
        :return: The tuple of the longitudinal and the lateral distances
        """
        try:
            return self.navigator.get_long_lat_distance_to_goal(position)
        except ValueError:
            return np.nan, np.nan

    def get_long_distance_until_lane_change(
        self, state: State, lanelet_ids: List[int] = None
    ) -> float:
        """
        Get the longitudinal distance until the lane change must be finished. It means that the ego vehicle is
        allowed to continue its was in the current lanelet and in its adjacent, but after this returned value,
        it must change the lane to the one which successor will lead to the goal
        :param state: The current state
        :param lanelet_ids: The lanelet ids of the current state. Only needed, if already available. (defualt: None)
        :return: The longitudinal distance until the lane change must be finished
        """
        try:
            return self.navigator.get_lane_change_distance(state, lanelet_ids)
        except ValueError:
            return np.nan

    def get_reference_lanelet_route(self) -> List[Lanelet]:
        """
        Get the reference route planned to the goal over the lanelets
        :return: List of merged lanelets of reference route
        """
        # Merge reference route
        merged_lanelets = []

        # Append predecessor of the initial to ensure that the goal state is not out of the projection domain
        initial_lanelet = self.lanelet_network.find_lanelet_by_id(self.route.route[0])
        predecessors_lanelet = initial_lanelet.predecessor
        if predecessors_lanelet is not None and len(predecessors_lanelet) != 0:
            predecessor_lanelet = self.lanelet_network.find_lanelet_by_id(
                predecessors_lanelet[0]
            )
            current_merged_lanelet = predecessor_lanelet
        else:
            current_merged_lanelet = None

        for current_lanelet_id, next_lanelet_id in zip(
            self.route.route[:-1], self.route.route[1:]
        ):
            lanelet = self.lanelet_network.find_lanelet_by_id(current_lanelet_id)
            # If the lanelet is the end of a section, then change section
            if next_lanelet_id not in lanelet.successor:
                if current_merged_lanelet is not None:
                    merged_lanelets.append(current_merged_lanelet)
                    current_merged_lanelet = None
            else:
                if current_merged_lanelet is None:
                    current_merged_lanelet = lanelet
                else:
                    current_merged_lanelet = Lanelet.merge_lanelets(
                        current_merged_lanelet, lanelet
                    )

        goal_lanelet = self.lanelet_network.find_lanelet_by_id(self.route.route[-1])
        if current_merged_lanelet is not None:
            current_merged_lanelet = Lanelet.merge_lanelets(
                current_merged_lanelet, goal_lanelet
            )
        else:
            current_merged_lanelet = goal_lanelet

        # Append successor of the goal to ensure that the goal state is not out of the projection domain
        goal_lanelet = self.lanelet_network.find_lanelet_by_id(self.route.route[-1])
        successors_of_goal = goal_lanelet.successor
        if successors_of_goal is not None and len(successors_of_goal) != 0:
            successor_lanelet = self.lanelet_network.find_lanelet_by_id(
                successors_of_goal[0]
            )
            current_merged_lanelet = Lanelet.merge_lanelets(
                current_merged_lanelet, successor_lanelet
            )

        merged_lanelets.append(current_merged_lanelet)
        return merged_lanelets

    def get_goal_orientation_distance(self, orientation: float) -> float:
        """
        calculate the distance of the current vehicle orientation to the goal

        returns orientation - goal_orientation_interval_start    if orientation < goal_orientation_interval_start
                orientation - goal_orientation_interval_end       if orientation > goal_orientation_interval_end
                0                                                 else

        :param orientation: orientation of current state
        :return difference to the nearest goal orientation boundary
        """
        if not hasattr(self.planning_problem.goal.state_list[0], "orientation"):
            warnings.warn("Trying to calculate relative goal orientation but goal state does not have orientation, "
                          "please set observe_distance_goal_orientation = False")
            return 0.
        else:
            orientation_start_list = np.array([s.orientation.start for s in self.planning_problem.goal.state_list])
            orientation_end_list = np.array([s.orientation.end for s in self.planning_problem.goal.state_list])
            goal_orientation_interval_start: float = np.mean(orientation_start_list)
            goal_orientation_interval_end: float = np.mean(orientation_end_list)
            if orientation < goal_orientation_interval_start:
                return orientation - goal_orientation_interval_start
            elif orientation > goal_orientation_interval_end:
                return orientation - goal_orientation_interval_end
            else:
                return 0.

    def get_goal_time_distance(self, time_step: int) -> int:
        """
        calculates the remaining time till the start of the goal time interval

        :param time_step: current time step
        :return difference to the nearest goal time boundary
        """
        # time_step is mandatory for GoalRegion, doesn't need to check attribute
        time_start_list = np.array([s.time_step.start for s in self.planning_problem.goal.state_list])
        time_end_list = np.array([s.time_step.end for s in self.planning_problem.goal.state_list])
        goal_time_interval_start: float = np.mean(time_start_list)
        goal_time_interval_end: float = np.mean(time_end_list)
        if time_step < goal_time_interval_start:
            return time_step - goal_time_interval_start
        elif time_step > goal_time_interval_end:
            return time_step - goal_time_interval_end
        else:
            return 0

    def get_goal_velocity_distance(self, velocity: float) -> float:
        """
        calculates the difference to the goal velocity

        :param velocity: velocity of current state
        :return difference to the nearest goal velocity boundary
        """
        if not hasattr(self.planning_problem.goal.state_list[0], "velocity"):
            warnings.warn("Trying to calculate relative goal velocity but goal state does not have velocity, "
                          "please set observe_distance_goal_velocity = False")
            return 0.
        else:
            velocity_start_list = np.array([s.velocity.start for s in self.planning_problem.goal.state_list])
            velocity_end_list = np.array([s.velocity.end for s in self.planning_problem.goal.state_list])
            goal_velocity_interval_start: float = np.mean(velocity_start_list)
            goal_velocity_interval_end: float = np.mean(velocity_end_list)
            if velocity < goal_velocity_interval_start:
                return velocity - goal_velocity_interval_start
            elif velocity > goal_velocity_interval_end:
                return velocity - goal_velocity_interval_end
            else:
                return 0.

