"""
Module for solving CommonRoad scenarios using trained models
"""
import argparse
import logging
import os

os.environ["KMP_WARNINGS"] = "off"

from typing import Union, List
import gym
import yaml
from commonroad.common.solution import (
    PlanningProblemSolution,
    Solution,
    CostFunction,
    CommonRoadSolutionWriter,
)
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.trajectory import Trajectory, State
from commonroad_dc.feasibility.solution_checker import valid_solution
from gym import Env
from stable_baselines.common.vec_env import VecNormalize

from commonroad_rl.utils_run.vec_env import CommonRoadVecEnv
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.utils_run.utils import ALGOS

__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluates PPO2 trained model with specified test scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--algo", type=str, default="ppo2")
    parser.add_argument(
        "--test_path",
        "-i",
        type=str,
        help="Path to pickled test scenarios",
        default=PATH_PARAMS["test_reset_config"],
    )
    parser.add_argument(
        "--model_path", "-model", type=str, help="Path to trained model", required=True
    )
    parser.add_argument("--multiprocessing", "-mpi", action="store_true")
    parser.add_argument(
        "--solution_path",
        "-sol",
        type=str,
        help="Path to the desired directory of the generated solution files",
        default=PATH_PARAMS["commonroad_solution"],
    )
    parser.add_argument("--cost_function", "-cost", type=str, default="SA1")
    parser.add_argument(
        "--hyperparam_filename", "-f", type=str, default="model_hyperparameters.yml"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (overrides verbose mode)"
    )
    parser.add_argument(
        "--config_filename",
        "-config_f",
        type=str,
        default="environment_configurations.yml",
    )

    return parser


def create_solution(
    commonroad_env: CommonroadEnv,
    output_directory: str,
    cost_function: str,
    computation_time: Union[float, None] = None,
    processor_name: Union[str, None] = "auto",
) -> bool:
    """
    Creates a CommonRoad solution file from the terminated environment
    :param commonroad_env: The terminated environment
    :param output_directory: The directory where the solution file will be written
    :param cost_function: The cost function to be used during the solution generation
    :param computation_time: The elapsed time during solving the scenario
    :param processor_name: The name of the used processor
    :return: True if the solution is valid and the solution file is written
    """
    os.makedirs(output_directory, exist_ok=True)
    list_state = list()
    for state in commonroad_env.ego_vehicle.state_list:
        kwarg = {
            "position": state.position,
            "velocity": state.velocity,
            "steering_angle": state.steering_angle,
            "orientation": state.orientation,
            "time_step": state.time_step,
        }
        list_state.append(State(**kwarg))

    trajectory = Trajectory(
        initial_time_step=list_state[0].time_step, state_list=list_state
    )

    planning_problem_solution = PlanningProblemSolution(
        planning_problem_id=commonroad_env.planning_problem.planning_problem_id,
        vehicle_model=commonroad_env.ego_vehicle.model_type,
        vehicle_type=commonroad_env.ego_vehicle.vehicle_type,
        cost_function=CostFunction[cost_function],
        trajectory=trajectory,
    )

    solution = Solution(
        scenario_id=commonroad_env.benchmark_id,
        # TODO: wrong usage, fix in commonroad environment to use scenario_id
        commonroad_version="2018b",
        planning_problem_solutions=[planning_problem_solution],
        computation_time=computation_time,
        processor_name=processor_name,
    )

    # Check the solution
    LOGGER.debug(f"Check solution of {commonroad_env.benchmark_id}")
    solution_valid, results = valid_solution(
        commonroad_env.scenario,
        PlanningProblemSet([commonroad_env.planning_problem]),
        solution,
    )
    if solution_valid is True:
        # write solution to a xml file
        csw = CommonRoadSolutionWriter(solution=solution)
        csw.write_to_file(output_path=output_directory, overwrite=True)
        LOGGER.info(
            f"Solution feasible, {commonroad_env.benchmark_id} printed to {output_directory}"
        )
    else:
        LOGGER.info(f"Unable to create solution, invalid trajectory!")
    return solution_valid


def load_model(model_path: str, algo: str):
    """
    Load trained model
    :param model_path: Path to the trained model
    :param algo: The used RL algorithm
    """
    # Load the trained agent
    model_path = os.path.join(model_path, "best_model.zip")
    model = ALGOS[algo].load(model_path)

    return model


def solve_scenarios(
    test_path: str,
    model_path: str,
    algo: str,
    solution_path: str,
    cost_function: str,
    multiprocessing: bool = False,
    hyperparam_filename: str = "model_hyperparameters.yml",
    config_filename: str = "environment_configurations.yml",
) -> List[bool]:
    """
    Solve a batch of scenarios using a trained model
    :param test_path: Path to the test files
    :param model_path: Path to the trained model
    :param algo: the used RL algorithm
    :param solution_path: Path to the folder where the solution files will be written
    :param cost_function: The cost function to be used during the solution generation
    :param multiprocessing: Indicates whether using multiprocessing or not (default is False)
    :param hyperparam_filename: The filename of the hyperparameters (default is model_hyperparameters.yml)
    :param config_filename: The environment configuration file name (default is environment_configurations.yml)
    :return: List of boolean values which indicates whether a scenario has been successfully solved or not
    """

    # mpi for parallel processing
    if multiprocessing:
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None
        if MPI is not None:
            rank = MPI.COMM_WORLD.Get_rank()
            test_path = os.path.join(test_path, str(rank))

    # Get environment keyword arguments including observation and reward configurations
    config_fn = os.path.join(model_path, config_filename)
    with open(config_fn, "r") as f:
        env_kwargs = yaml.load(f, Loader=yaml.Loader)

    # env_kwargs.update(
    #     {"meta_scenario_path": test_path + '/meta_scenario',
    #      "test_reset_config_path": test_path + '/problem_test'}
    # )

    def env_fn():
        return gym.make("commonroad-v0", play=True, **env_kwargs)

    env = CommonRoadVecEnv([env_fn])
    results = []

    def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
        if env.observation_history_dict["is_goal_reached"][-1]:
            LOGGER.info("Goal reached")
            os.makedirs(solution_path, exist_ok=True)
            solution_valid = create_solution(
                env, solution_path, cost_function, computation_time=elapsed_time
            )
        else:
            LOGGER.info("Goal not reached")
            solution_valid = False
        results.append(solution_valid)

    env.set_on_reset(on_reset_callback)

    # Load model hyperparameters:
    hyperparam_fn = os.path.join(model_path, hyperparam_filename)
    with open(hyperparam_fn, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.Loader)

    normalize = hyperparams["normalize"]
    if normalize:
        LOGGER.debug("Loading saved running average")
        vec_normalize_path = os.path.join(model_path, "vecnormalize.pkl")
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
        else:
            raise FileNotFoundError  # vecnormalize.pkl not found

    model = load_model(model_path, algo)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        try:
            obs, reward, done, info = env.step(action)
            LOGGER.debug(
                f"Step: {env.venv.envs[0].current_step}, \tReward: {reward}, \tDone: {done}"
            )
        except IndexError as e:
            # If the environment is done, it will be reset.
            # However the reset throws an exception if there are no more scenarios to be solved.
            LOGGER.info(f"Cannot choose more scenarios to be solved, msg: {e}")
            break
    return results


def load_and_check_solution(scenario_file_path: str, solution_file_path: str) -> bool:
    """
    Function to check whether a solution file is valid for a scenario or not
    :param scenario_file_path: Path to the scenario
    :param solution_file_path: Path to the solution
    :return: True if the solution is valid
    """
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad.common.solution import CommonRoadSolutionReader

    scenario, pp = CommonRoadFileReader(scenario_file_path).open()
    solution = CommonRoadSolutionReader.open(solution_file_path)
    solution_valid, results = valid_solution(scenario, pp, solution)
    if solution_valid is True:
        LOGGER.info(f"Solution feasible")
    else:
        LOGGER.info(f"Unable to create solution, invalid trajectory!")
    return solution_valid


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.INFO)
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    solve_scenarios(
        args.test_path,
        args.model_path,
        args.algo,
        args.solution_path,
        args.cost_function,
        multiprocessing=args.multiprocessing,
        hyperparam_filename=args.hyperparam_filename,
        config_filename=args.config_filename,
    )
