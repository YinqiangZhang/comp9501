"""
Module for checking the trajectory of obstacles
"""
import os
import pickle
from typing import Tuple, List, Union

import matplotlib.pyplot as plt

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution_writer import VehicleModel, VehicleType
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State

from commonroad_rl.gym_commonroad.utils.scenario import check_trajectory
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.utils_highd.planning_utils import get_valid_ob_list, INVALID_OBSTACLE
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
import matplotlib

matplotlib.use("TkAgg")

__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"


def plot_attributes(
    state_list: Union[List[float], List[State]],
    dt: float,
    label: str,
    attribute: str = None,
    attribute_index: int = None,
    color: str = None,
    units: str = "",
) -> None:
    """
    Plot attributes of trajectory
    :param state_list: The state list of the trajectory
    :param dt: The dt of the trajectory
    :param label: The label of the attribute
    :param attribute: The attribute to be plotted, if any
    :param attribute_index: The index of the attribute to be plotted, if any
    :param color: The color to be used during the plot
    :param units: The units of the y axis
    """

    if attribute is None:
        list_of_attributes = state_list
    else:
        list_of_attributes = [
            getattr(state, attribute)
            if attribute_index is None
            else getattr(state, attribute)[attribute_index]
            for state in state_list
        ]
    timesteps = np.arange(0.0, float(len(list_of_attributes))) * dt
    plt.plot(timesteps, list_of_attributes, color=color)
    plt.legend()
    plt.ylabel(
        f'{label}{f"[{attribute_index}]" if attribute_index is not None else ""} {units}'
    )
    plt.xlabel("Step (s)")


def plot_obstacles_attributes(
    obstacles: List[DynamicObstacle], dt, attribute, attribute_index=None
):
    """
    Plot the attributes of more obstacles into one figure
    :param obstacles
    """
    for obstacle in obstacles:
        plot_attributes(
            obstacle.prediction.trajectory.state_list,
            dt,
            str(obstacle.obstacle_id),
            attribute=attribute,
            attribute_index=attribute_index,
        )


def plot_obstacle_attributes(
    obstacles: List[DynamicObstacle], dt: float, title: str = ""
):
    """
    Plot more attributes of more obstacle
    :param obstacles: List of obstacles
    :param dt: Dt of the scenario
    :param title: The title of the figure
    """
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
    fig.suptitle(title)
    plt.subplot(2, 2, 1)
    plot_obstacles_attributes(obstacles, dt, "orientation")
    plt.subplot(2, 2, 2)
    plot_obstacles_attributes(obstacles, dt, "velocity")
    plt.subplot(2, 2, 3)
    plot_obstacles_attributes(obstacles, dt, "position", attribute_index=0)
    plt.subplot(2, 2, 4)
    plot_obstacles_attributes(obstacles, dt, "position", attribute_index=1)
    plt.show()


def get_obstacle_from_scenario(
    scenario_file: str, obstacle_idxs: List[int] = None
) -> Tuple[List[DynamicObstacle], float]:
    """
    Get obstacles from scenario
    :param scenario_file: Path to the scenario
    :param obstacle_idxs: The indexes of the obstacle
    :return: Tuple of obstacles and the dt of scenario
    """
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
    if obstacle_idxs is None:
        return scenario.dynamic_obstacles, scenario.dt
    else:
        return scenario.dynamic_obstacles[obstacle_idxs], scenario.dt


def get_obstacle_from_pickle(
    pickle_file: str, obstacle_idxs: List[int] = None
) -> Tuple[List[DynamicObstacle], float]:
    """
    Get obstacle from pickle
    :param pickle_file: Path to the pickled scenario file
    :param obstacle_idxs: The indexes of the obstacle
    :return: Tuple of obstacles and the dt of the scenario
    """
    with open(pickle_file, "rb") as f:
        problem_dict = pickle.load(f)
    obstacles = []
    if obstacle_idxs is None:
        obstacles = problem_dict["obstacle"]
    else:
        obstacles.append(problem_dict["obstacle"][obstacle_idxs])
    return obstacles, 0.1


def filter_obstacles_by_huge_orientation_change(
    obstacles: List[DynamicObstacle],
) -> List[DynamicObstacle]:
    """
    Find obstacles with very huge jumps in orientation
    :param obstacles: The obstacles to be checked
    :return: The obstacles, which has high jumps in the orientation
    """
    critical_obstacles = []

    dt = 0.1
    max_steering_rate = 0.4

    for obstacle in obstacles:
        orientations = np.array(
            [state.orientation for state in obstacle.prediction.trajectory.state_list]
        )
        raw_orientation_change = orientations[1:] - orientations[:-1]
        raw_orientation_change[raw_orientation_change > np.pi] -= 2 * np.pi
        raw_orientation_change[raw_orientation_change < -np.pi] += 2 * np.pi
        orientation_change = abs(raw_orientation_change)
        if any(orientation_change > max_steering_rate * dt):
            critical_obstacles.append(obstacle)
    return critical_obstacles


def filter_obstacles_by_infeasible_trajectory(
    obstacles: List[DynamicObstacle],
    vehicle_model: VehicleModel,
    vehicle_type: VehicleType,
    dt: float,
) -> List[DynamicObstacle]:
    """
    Finds obstacles with infeasible trajectory
    :param obstacles: List of obstacles to be checked
    :param vehicle_model: The vehicle model to be used for fesibility checking
    :param vehicle_type: The vehicle type to be used for fesibility checking
    :param dt: Dt of the scenario
    :return: List of obstacles
    """
    critical_obstacles = []

    for obstacle in obstacles:
        if not check_trajectory(obstacle, vehicle_model, vehicle_type, dt):
            critical_obstacles.append(obstacle)
    return critical_obstacles


def get_scenarios(scenario_pickle_path: str):
    """
    Generator which yields the content of the pickled scnearios
    :param scenario_pickle_path: Path to the folder which contains
    the problem and meta_scenario directories with the pickled scnearios
    """
    problem_pickle_path = os.path.join(scenario_pickle_path, "problem")
    print(f"Checking trajectories of folder {problem_pickle_path}")
    unique_locations = filter_unique_locations(problem_pickle_path)
    print(f"Unique locations found: {len(unique_locations)}, {unique_locations}")

    meta_scenario_dict_path = os.path.join(
        scenario_pickle_path, "meta_scenario", "problem_meta_scenario_dict.pickle"
    )
    with open(meta_scenario_dict_path, "rb") as f:
        problem_meta_scenario_dict = pickle.load(f)

    def read_scenario(file_path) -> Scenario:
        benchmark_id = os.path.splitext(os.path.basename(file_path))[0]
        scenario_raw = pickle.load(open(filepath, "rb"))
        scenario = restore_scenario(
            problem_meta_scenario_dict[benchmark_id], scenario_raw["obstacle"]
        )
        scenario.benchmark_id = benchmark_id
        return scenario

    for filepath in unique_locations:
        yield read_scenario(filepath)


def filter_unique_locations(root_path: str) -> List[str]:
    """
    Filters the inD scenarios by location to get unique scenarios
    :param root_path: Path to the root folder of the scenarios
    :return: List of pathes pointing to scenarios with unique locations
    """
    unique_locations = []
    current_location = "<start>"
    for subdir, dirs, files in os.walk(root_path):
        files = sorted(files)
        for filename in files:
            if not filename.startswith(f"DEU_Location{current_location}"):
                current_location = filename[12:16]
                filepath = subdir + os.sep + filename
                unique_locations.append(filepath)
    return unique_locations


def search_vibrating_obstacles_in_scenarios(scenario_pickle_path: str):
    """
    Search for obstacles with huge orientation change in all scenarios
    :param scenario_pickle_path: Path to the folder which contains
    the problem and meta_scenario directories with the pickled scnearios
    """
    count = 0

    scenarios = get_scenarios(scenario_pickle_path)

    for scenario in scenarios:
        print(f"Checking {scenario.benchmark_id}")
        obstacles, dt = scenario.dynamic_obstacles, scenario.dt
        obstacles = filter_obstacles_by_huge_orientation_change(obstacles)
        if len(obstacles) > 0:
            count += 1
            print(
                f"Critical obstacles found: {len(obstacles)} in {scenario.benchmark_id}"
            )
    print(f"Critical obstacles found in {count} scnearios")


def analyze_obstacles_in_scenarios(
    scenario_pickle_path: str, invalidity_reason_to_plot: List[str] or None = None
):
    """
    Function helps to analyze the trajectory of obstacles,
    runs obstacle validation and plots the most important attributes
    :param scenario_pickle_path: Path to the folder which contains
    the problem and meta_scenario directories with the pickled scnearios
    :param invalidity_reason_to_plot: Indicates which invalidated obstacles
    should be plotted by the reason of invalidity (default None means all invalid obstacles will be plotted)
    """
    count = 0

    scenarios = get_scenarios(scenario_pickle_path)

    if invalidity_reason_to_plot is None:
        invalidity_reason_to_plot = list(INVALID_OBSTACLE)

    for scenario in scenarios:
        # if scenario.benchmark_id != 'DEU_AAH-2_210025_T-1':
        #     continue
        print(f"Checking {scenario.benchmark_id}")
        valid_ob_list, invalid_ob_dict = get_valid_ob_list(scenario, is_up_lanelet=True)
        if len(invalid_ob_dict) > 0:
            count += 1
            for key in invalidity_reason_to_plot:
                invalid_obs = invalid_ob_dict.get(key)
                if invalid_obs is not None:
                    plot_obstacle_attributes(
                        invalid_obs,
                        scenario.dt,
                        title=f"{str(key)} in {scenario.benchmark_id}",
                    )

    print(f"Critical obstacles found in {count} scenarios")


def collect_experience_random_environment(
    num_of_steps: int, pickle_path: str, seed: int = 5
):
    """
    Collect experiments by applying uniformly sampled random normalized actions.
    All actions are held for 4 steps to see transient effects
    The collected experiments will be pickled
    :param num_of_steps: The number of steps
    :param pickle_path: Path to the resulting pickle
    :param seed: Seed for the random generator
    """
    env = CommonroadEnv(play=True)
    env.reset()
    action_change_period = 4
    np.random.seed(seed)
    actions = np.random.uniform(
        env.action_space.low,
        env.action_space.high,
        size=(int(num_of_steps / action_change_period), 2),
    )
    actions = np.repeat(actions, action_change_period, axis=0)

    for idx in range(num_of_steps):
        env.step(actions[idx])

    pickle.dump((actions, env.ego_vehicle.state_list), open(pickle_path, "wb"))


def plot_actions_and_states(pickle_path: str, title: str, action_labels: List[str]):
    """
    Plot the actions and states of the collected experiences
    :param pickle_path: Path to the pickled experiences
    :param title: Title of the plot
    :param action_labels: List of the labels of the actions
    """
    (actions, state_list) = pickle.load(open(pickle_path, "rb"))
    dt = 0.01

    fig = plt.figure(figsize=(10, 40), dpi=100)
    plt.subplots_adjust(hspace=0.3)

    fig.tight_layout(pad=50.0)
    fig.suptitle(title)
    plt.subplot(4, 2, 1)
    plot_attributes(state_list, dt, "orientation", "orientation", units="[rad]")
    plt.subplot(4, 2, 2)
    plot_attributes(state_list, dt, "velocity", "velocity", units="[m/s]")
    plt.subplot(4, 2, 3)
    plot_attributes(
        state_list, dt, "position", "position", attribute_index=0, units="[m]"
    )
    plt.subplot(4, 2, 4)
    plot_attributes(
        state_list, dt, "position", "position", attribute_index=1, units="[m]"
    )
    plt.subplot(4, 2, 5)
    plot_attributes(state_list, dt, "yaw_rate", "yaw_rate", units="[rad/s]")
    plt.subplot(4, 2, 6)
    plot_attributes(state_list, dt, "steering_angle", "steering_angle", units="[rad]")

    plt.subplot(4, 2, 7)
    plot_attributes(actions[:, 0], dt, action_labels[0], color="r", units="[m/s^2]")
    plt.subplot(4, 2, 8)
    plot_attributes(actions[:, 1], dt, action_labels[1], color="r", units="[rad/s]")


def experiment_vehicle_model(
    vehicle_params: dict,
    model_name: str,
    action_names: List[str],
    num_of_steps: int = 100,
):
    """
    Gather experiences using a given vehicle model and plot the actions and states
    :param vehicle_params: The vehicle parameters
    :param model_name: The name of the model to be used
    :param action_names: The name of the actions
    :param num_of_steps: The number of steps (default is 100)
    """
    global VEHICLE_PARAMS
    VEHICLE_PARAMS.clear()
    VEHICLE_PARAMS.update(vehicle_params)
    model_path = f"/home/peterkocsis/work/2020_CommonRoad/Praktikum/experiments/random_actions/{model_name}.p"
    collect_experience_random_environment(num_of_steps, model_path)
    plot_actions_and_states(model_path, model_name, action_names)
    plt.show()


if __name__ == "__main__":
    # # Helper functions to analyze the trajectory of obstacles
    # pickle_path = "/home/peterkocsis/work/2020_CommonRoad/Praktikum/dataset/inD-dataset/pickle_2020"
    #
    # # search_vibrating_obstacles_in_scenarios(pickle_path)
    # analyze_obstacles_in_scenarios(pickle_path, invalidity_reason_to_plot=[INVALID_OBSTACLE.TRAJECTORY])

    params_KS = {
        "vehicle_type": VehicleType.FORD_ESCORT,
        "vehicle_model": VehicleModel.KS,
    }
    model_name_1 = "KS_model"
    action_names_1 = ["action_acceleration", "action_steering_rate"]
    experiment_vehicle_model(params_KS, model_name_1, action_names_1)

    params_OLD = {
        "length": 4.508,
        "width": 1.610,
        "acc_max": 8.0,
        "vel_max": 41.7,
        "vel_min": -13.6,
        "turn_rate_max": 2.0,
        "react_time": 0.2,
        "l_wb": 2.35,
        "vehicle_model": None,
        "vehicle_type": None,
    }
    model_name_2 = "OLD_model"
    action_names_2 = ["action_acceleration", "action_yaw_rate"]
    experiment_vehicle_model(params_OLD, model_name_2, action_names_2)
