#! /usr/bin/env python

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

__desc__ = """
Includes all functionality to turn a dynamic obstacle into a planning problem
"""

import logging

from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.common.util import Interval, AngleInterval
from commonroad.common.solution import VehicleModel, VehicleType
from commonroad.geometry.shape import Shape
from commonroad.scenario.obstacle import State, DynamicObstacle
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario

from shapely.geometry import Point

from .common import make_valid_orientation_interval_pruned

from commonroad_route_planner.route_planner import RoutePlanner

LOGGER = logging.getLogger(__name__)

try:
    from commonroad_rl.gym_commonroad.utils.scenario import check_trajectory
except ImportError:
    # skip checking for non commonraod_rl oriented trajectories
    LOGGER.info(
        "commonroad_rl not installed, trajectory checking will not be conducted"
    )

    def check_trajectory(
        obstacle: DynamicObstacle,
        vehicle_model: VehicleModel,
        vehicle_type: VehicleType,
    ):
        return True


def planning_problem_from_dynamic_obstacle(
    ego_vehicle: DynamicObstacle,
    scenario: Scenario,
    vehicle_model=VehicleModel.KS,
    vehicle_type=VehicleType.FORD_ESCORT,
):
    """
    Converts an ego vehicle to a planning problem
    orientation, velocity and time_step will be relaxed with regard to the original track
    the goal region will be approximately the lanelet where the vehicle ends up
    :param ego_vehicle:
    :param scenario:
    :param orientation_relax_half_range:
    :param velocity_relax_half_range:
    :param time_step_relax_half_range:
    :return:
    """
    assert scenario.lanelet_network.find_lanelet_by_position(
        [ego_vehicle.initial_state.position]
    )[0], "Starting position off lanelet network"

    #    assert check_trajectory(
    #        ego_vehicle, vehicle_model, vehicle_type, scenario.dt
    #    ), f"Trajectory is infeasible with {vehicle_type}:{vehicle_model}"

    goal_shape, goal_lanelets = goal_shape_from_ego_vehicle(ego_vehicle, scenario)

    pp = PlanningProblem(
        scenario.generate_object_id(),
        initial_state_from_ego_vehicle(ego_vehicle),
        GoalRegion(
            [goal_state_from_ego_vehicle(ego_vehicle, goal_shape)], {0: goal_lanelets}
        ),
    )
    return pp


def initial_state_from_ego_vehicle(ego_vehicle: DynamicObstacle):
    """
    Constructs a valid initial state for a planning problem from the initial state of a dynamic obstacle
    :param ego_vehicle: dynamical obstacle, created from an inD dataset track
    """
    initial_state = ego_vehicle.initial_state
    initial_state = State(
        time_step=0,
        position=initial_state.position,
        orientation=initial_state.orientation,
        velocity=initial_state.velocity,
        # as required by the final state definition
        # but unknown due to lacking information in the dataset
        yaw_rate=0,
        slip_angle=0,
    )
    return initial_state


def goal_state_from_ego_vehicle(
    ego_vehicle: DynamicObstacle,
    goal_shape: Shape,
    orientation_relax_half_range=0.2,
    velocity_relax_half_range=10.0,
    time_step_relax_half_range=50,
):
    """
    Generate a suitable goal state from the last frame of  a dynamic obstacle
    :param ego_vehicle: dynamical obstacle, created from an inD dataset track
    :param goal_shape: Suitable goal shape for the planning problem. Note that it will not be changed inside this method
    :param orientation_relax_half_range: Relax the orientation constraint by x in each direction
    :param velocity_relax_half_range: Relax the velocity constraint by x in each direction
    :param time_step_relax_half_range: Relax the time step constraint by x in each direction
    """
    goal_state = ego_vehicle.prediction.trajectory.final_state
    goal_time_step = Interval(
        goal_state.time_step - time_step_relax_half_range,
        goal_state.time_step + time_step_relax_half_range,
    )
    goal_velocity = Interval(
        goal_state.velocity - velocity_relax_half_range,
        goal_state.velocity + velocity_relax_half_range,
    )
    goal_orientation = AngleInterval(
        *make_valid_orientation_interval_pruned(
            goal_state.orientation - orientation_relax_half_range,
            goal_state.orientation + orientation_relax_half_range,
        )
    )
    return State(
        position=goal_shape,
        time_step=goal_time_step,
        velocity=goal_velocity,
        orientation=goal_orientation,
    )


def goal_shape_from_ego_vehicle(ego_vehicle: DynamicObstacle, scenario: Scenario):
    """
    Create a sufficiently large goal shape that is oriented at the lanelet where the ego vehicle ends up
    Tries to make sure that the goal lanelet points in the same direction as the ego vehicle at the end
    :param ego_vehicle: ego vehicle
    :param scenario: scenario of the ego vehicle
    :return: a goal shape of area larger than the ego vehicle
    """
    goal_vehicle_state = ego_vehicle.prediction.trajectory.final_state
    goal_vehicle_shape = ego_vehicle.prediction.occupancy_set[-1].shape
    goal_lanelet_candidate_ids = scenario.lanelet_network.find_lanelet_by_shape(
        goal_vehicle_shape
    )

    # filter candidates:
    # filter goal lanelets by which of them are contained in any path from initial position to the goal
    route_planner = RoutePlanner(
        scenario,
        PlanningProblem(
            1,
            initial_state_from_ego_vehicle(ego_vehicle),
            GoalRegion(
                [goal_state_from_ego_vehicle(ego_vehicle, goal_vehicle_shape)],
                {0: goal_lanelet_candidate_ids},
            ),
        ),
        backend=RoutePlanner.Backend.NETWORKX_REVERSED,
        log_to_console=False,
    )

    route_candidates = route_planner.get_route_candidates().route_candidates
    goal_lanelet_candidate_ids = filter(
        lambda x: any((x in route_candidate) for route_candidate in route_candidates),
        goal_lanelet_candidate_ids,
    )

    # arbitrary multiple to ensure task is possible
    minimum_area = goal_vehicle_shape.shapely_object.area * 2
    goal_lanelet = None
    goal_lanelet_ids = []
    for g_id in goal_lanelet_candidate_ids:
        # choose any good lanelet as goal area
        # and extend if necessary and possible
        g = scenario.lanelet_network.find_lanelet_by_id(g_id)
        g_ids = [g_id]
        while (
            g.successor or g.predecessor
        ) and g.convert_to_polygon().shapely_object.area < minimum_area:
            for ex in [g.successor, g.predecessor]:
                if ex:
                    # DO NOT MODIFY ex SINCE THIS MODIFIES THE LANELET NETWORK
                    # instead, note that g.successor and g.predecessor is modified after each
                    # merge operation
                    new_lanelet = ex[0]
                    g = Lanelet.merge_lanelets(
                        g, scenario.lanelet_network.find_lanelet_by_id(new_lanelet)
                    )
                    g_ids.append(new_lanelet)
                    # breaks inner for loop if an extension has been found
                    # s.t. we prefer merging successors over merging predecessors
                    # and we stop extending as soon as the area is sufficiently big
                    break
        if g.convert_to_polygon().shapely_object.area >= minimum_area:
            # if the resulting area looks fine, terminate search
            goal_lanelet = g
            goal_lanelet_ids = g_ids
            break

    # discard planning problems where the car ends up off some lanelet
    assert goal_lanelet is not None, "Goal area not sufficiently placed on any lanelet"
    assert (
        goal_lanelet.convert_to_polygon().shapely_object.area > minimum_area
    ), "Goal area lanelets not sufficiently big"
    goal_shape = goal_lanelet.convert_to_polygon()

    # filter cases where the vehicle hardly moved throughout the whole scenario
    assert not goal_shape.shapely_object.contains(
        Point(ego_vehicle.initial_state.position)
    ), "Ego vehicle starts in goal area"

    return goal_shape, goal_lanelet_ids
