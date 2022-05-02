import pickle
import random
from enum import Enum
from typing import Tuple, List, Dict
from collections import defaultdict

from commonroad.common.solution import VehicleModel, VehicleType
from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

from commonroad_rl.gym_commonroad.utils.scenario import (
    check_trajectory,
    interpolate_steering_angles_of_obstacle,
)


def get_planning_problem(
    scenario,
    valid_ob_list,
    orientation_half_range=0.2,
    velocity_half_range=10.0,
    time_step_half_range=50,
):
    # random choose obstacle as ego vehicle
    dynamic_obstacle = random.choice(valid_ob_list)

    assert dynamic_obstacle.initial_state.time_step == 0

    dynamic_obstacle_shape = dynamic_obstacle.obstacle_shape
    dynamic_obstacle_final_state = dynamic_obstacle.prediction.trajectory.final_state
    scenario.remove_obstacle(dynamic_obstacle)

    final_state = dynamic_obstacle.prediction.trajectory.final_state

    # define position, orientation, velocity and time step intervals as goal region
    position_rect = Rectangle(
        length=dynamic_obstacle_shape.length + 2.0,
        width=max(dynamic_obstacle_shape.width + 1.0, 3.5),
        center=final_state.position,
        orientation=final_state.orientation,
    )
    orientation_interval = AngleInterval(
        dynamic_obstacle_final_state.orientation - orientation_half_range,
        dynamic_obstacle_final_state.orientation + orientation_half_range,
    )
    velocity_interval = Interval(
        dynamic_obstacle_final_state.velocity - velocity_half_range,
        dynamic_obstacle_final_state.velocity + velocity_half_range,
    )
    time_step_interval = Interval(
        max(0, dynamic_obstacle_final_state.time_step - time_step_half_range),
        dynamic_obstacle_final_state.time_step + time_step_half_range,
    )
    goal_region = GoalRegion(
        [
            State(
                position=position_rect,
                orientation=orientation_interval,
                velocity=velocity_interval,
                time_step=time_step_interval,
            )
        ]
    )

    dynamic_obstacle.initial_state.yaw_rate = 0.0
    dynamic_obstacle.initial_state.slip_angle = 0.0

    return PlanningProblem(
        dynamic_obstacle.obstacle_id, dynamic_obstacle.initial_state, goal_region
    )


class INVALID_OBSTACLE(Enum):
    TIME = "time"
    POSITION = "position"
    TRAJECTORY = "trajectory"


def get_valid_ob_list(
    scenario: Scenario,
    is_up_lanelet=False,
    vehicle_model: VehicleModel = VehicleModel.KS,
    vehicle_type: VehicleType = VehicleType.FORD_ESCORT,
) -> Tuple[List[DynamicObstacle], Dict[INVALID_OBSTACLE, List[DynamicObstacle]]]:

    vehicle_params = VehicleParameterMapping[vehicle_type.name].value
    distance = 100
    road_start_coord, road_end_coord = get_road_start_end_coord(scenario, is_up_lanelet)
    valid_ob_list = []
    invalid_ob_dict = defaultdict(lambda: [])
    for ob in scenario.dynamic_obstacles:
        initial_timestep = ob.initial_state.time_step
        initial_id = scenario.lanelet_network.find_lanelet_by_position(
            [ob.initial_state.position]
        )
        # Only pick obstacle which time step is 0 and not off road
        if not (initial_timestep == 0 and initial_id != -1):
            invalid_ob_dict[INVALID_OBSTACLE.TIME].append(ob)
            continue

        # Only pick if it is not at the very beginning of the lane
        if not (
            (
                is_up_lanelet
                and road_start_coord - ob.initial_state.position[0] > distance
            )
            or (
                not is_up_lanelet
                and ob.initial_state.position[0] - road_start_coord > distance
            )
        ):
            # if road_start_coord - ob.initial_state.position[0] > distance and \
            #         ob.initial_state.position[0] - road_end_coord > distance:
            invalid_ob_dict[INVALID_OBSTACLE.POSITION].append(ob)
            continue
        interpolate_steering_angles_of_obstacle(ob, vehicle_params, scenario.dt)
        if not check_trajectory(ob, vehicle_model, vehicle_type, scenario.dt):
            invalid_ob_dict[INVALID_OBSTACLE.TRAJECTORY].append(ob)
            # pickle.dump(scenario, open(f"{scenario.benchmark_id}.p", "wb"))
            continue
        valid_ob_list.append(ob)
    print(
        f"Valid obstacles: {len(valid_ob_list)}, "
        f"invalid obstacles: {({key: len(value) for key, value in invalid_ob_dict.items()})}"
    )
    return valid_ob_list, invalid_ob_dict


def get_road_start_end_coord(
    scenario: Scenario, is_up_lanelet=False
) -> Tuple[float, float]:
    start_coord = None
    end_coord = None
    if is_up_lanelet:
        for l in scenario.lanelet_network.lanelets:
            if start_coord is None:
                start_coord = l.center_vertices[0, 0]
            else:
                if l.center_vertices[0, 0] > start_coord:
                    start_coord = l.center_vertices[0, 0]
            if end_coord is None:
                end_coord = l.center_vertices[-1, 0]
            else:
                if l.center_vertices[-1, 0] < end_coord:
                    end_coord = l.center_vertices[0, 0]
    else:
        for l in scenario.lanelet_network.lanelets:
            if start_coord is None:
                start_coord = l.center_vertices[0, 0]
            else:
                if l.center_vertices[0, 0] < start_coord:
                    start_coord = l.center_vertices[0, 0]
            if end_coord is None:
                end_coord = l.center_vertices[-1, 0]
            else:
                if l.center_vertices[-1, 0] > end_coord:
                    end_coord = l.center_vertices[-1, 0]
    return start_coord, end_coord
