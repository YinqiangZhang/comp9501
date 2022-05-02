"""
Module for plaxing trained model using Stable baselines
"""
import argparse
import os

from stable_baselines.common import BaseRLModel

os.environ["KMP_WARNINGS"] = "off"
import logging
from typing import Union

from stable_baselines.common.vec_env import VecNormalize
from gym import Env
import gym
import yaml

from commonroad_rl.utils_run.utils import ALGOS
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.utils_run.vec_env import CommonRoadVecEnv
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

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

    parser.add_argument(
        "--env_id", type=str, default="commonroad-v0", help="environment ID"
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
        "--model_path",
        "-model",
        type=str,
        help="Path to trained model",
        default=PATH_PARAMS["log"] + "/ppo2/commonroad-v0_3",
    )
    parser.add_argument(
        "--viz_path", "-viz", type=str, default=PATH_PARAMS["visualization"]
    )
    parser.add_argument(
        "--num_scenarios",
        "-n",
        default=1,
        type=int,
        help="Maximum number of scenarios to draw",
    )
    parser.add_argument("--multiprocessing", "-mpi", action="store_true")
    parser.add_argument(
        "--combine_frames",
        "-1",
        action="store_true",
        help="Combine rendered environments into one picture",
    )
    parser.add_argument(
        "--skip_timesteps",
        "-st",
        type=int,
        default=1,
        help="Only render every nth frame (including first and last)",
    )
    parser.add_argument(
        "--no_render", "-nr", action="store_true", help="Whether store render images"
    )
    parser.add_argument(
        "--hyperparam_filename",
        "-hyperparam_f",
        type=str,
        default="model_hyperparameters.yml",
    )
    parser.add_argument(
        "--config_filename",
        "-config_f",
        type=str,
        default="environment_configurations.yml",
    )

    return parser


def create_environments(
    env_id: str,
    test_path: str,
    model_path: str,
    viz_path: str,
    hyperparam_filename: str,
    env_kwargs=None,
) -> CommonRoadVecEnv:
    """
    Create CommonRoad vectorized environment environment
    :param env_id: Environment gym id
    :param test_path: Path to the test files
    :param model_path: Path to the trained model
    :param viz_path: Output path for rendered images
    :param hyperparam_filename: The filename of the hyperparameters
    :param env_kwargs: Keyword arguments to be passed to the environment
    """

    env_kwargs.update(
        {
            "meta_scenario_path": test_path + "/meta_scenario",
            "test_reset_config_path": test_path + "/problem_test",
            "visualization_path": viz_path,
        }
    )
    print(test_path)
    # Create environment
    env_fn = lambda: gym.make(env_id, play=True, **env_kwargs)
    env = CommonRoadVecEnv([env_fn])

    def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
        if env.observation_history_dict["is_goal_reached"][-1]:
            LOGGER.info("Goal reached")
        else:
            LOGGER.info("Goal not reached")
        env.render()

    env.set_on_reset(on_reset_callback)

    # Load model hyperparameters:
    hyperparam_fn = os.path.join(model_path, hyperparam_filename)
    with open(hyperparam_fn, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.Loader)

    normalize = hyperparams["normalize"]

    if normalize:
        LOGGER.info("Loading saved running average")
        vec_normalize_path = os.path.join(model_path, "vecnormalize.pkl")
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
        else:
            raise FileNotFoundError  # vecnormalize.pkl not found

    return env


def load_model(model_path: str, algo: str) -> BaseRLModel:
    """
    Load trained model
    :param model_path: Path to the trained model
    :param algo: The used RL algorithm
    """
    # Load the trained agent
    model_path = os.path.join(model_path, "best_model.zip")
    model = ALGOS[algo].load(model_path)

    return model


def main():

    args = get_parser().parse_args()

    # mpi for parallel processing
    if args.multiprocessing:
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None
        if MPI is None:
            test_path = args.test_path
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            test_path = os.path.join(args.test_path, str(rank))
    else:
        test_path = args.test_path

    # Get environment keyword arguments including observation and reward configurations
    config_fn = os.path.join(args.model_path, args.config_filename)
    with open(config_fn, "r") as f:
        env_kwargs = yaml.load(f, Loader=yaml.Loader)
    env_kwargs["render_skip_timesteps"] = args.skip_timesteps
    env_kwargs["render_combine_frames"] = args.combine_frames

    env = create_environments(
        args.env_id,
        test_path,
        args.model_path,
        args.viz_path,
        args.hyperparam_filename,
        env_kwargs,
    )

    LOGGER.info(f"Testing a maximum of {args.num_scenarios} scenarios")

    model = load_model(args.model_path, args.algo)

    obs = env.reset()
    env.render()
    try:
        for _ in range(args.num_scenarios):
            for __ in range(300):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                env.render()
                LOGGER.debug(
                    f"Step: {env.venv.envs[0].current_step}, \tReward: {reward}, \tDone: {done}"
                )
    except IndexError as e:
        # If the environment is done, it will be reset. However the reset throws an exception if there are no more
        # scenarios to be solved. This is a workaround to resolve this issue,
        # proper solution needed inside the environment
        LOGGER.warning(f"Cannot choose more scenarios to be rendered, msg: {e}")


if __name__ == "__main__":
    main()
