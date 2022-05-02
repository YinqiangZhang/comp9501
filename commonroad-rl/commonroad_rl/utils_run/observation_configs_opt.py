"""
A utility function to be called when optimizing observation configurations
"""
import logging
import numpy as np
import pandas as pd
import optuna
import os
import yaml
from pprint import pformat
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.ddpg import (
    AdaptiveParamNoiseSpec,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines.her import HERGoalEnvWrapper
from commonroad_rl.utils_run.callbacks import ObservationConfigsTrialEvalCallback

LOGGER = logging.getLogger(__name__)

__author__ = "Brian Liao"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"


def optimize_observation_configs(
    algo,
    env,
    model_fn,
    env_fn,
    sampling_setting,
    n_timesteps=5000,
    eval_freq=1000,
    n_eval_episodes=5,
    n_trials=10,
    hyperparams=None,
    configs=None,
    n_jobs=1,
    sampler_method="random",
    pruner_method="halving",
    seed=13,
    verbose=1,
    log_path=None,
    best_model_save_path=None,
):
    """
    :param algo: (str)
    :param env: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict) model hyperparameters
    :param configs: (dict) environment configurations to be optimized
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param seed: (int)
    :param sampling_setting: (dict) sampling intervals for correspinding items
    :param verbose: (int)
    :param log_path: (str) folder for saving evaluation results during optimization
    :param best_model_save_path (str) folder for saving the best model
    :return: (dict) detailed result of the optimization
    """
    # Set logger if not yet exists
    if verbose == 2:
        LOGGER.setLevel(logging.DEBUG)
    elif verbose == 1:
        LOGGER.setLevel(logging.INFO)
    else:
        LOGGER.setLevel(logging.ERROR)

    if not len(LOGGER.handlers):
        formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        stream_handler = logging.StreamHandler()
        if verbose == 2:
            stream_handler.setLevel(logging.DEBUG)
        elif verbose == 1:
            stream_handler.setLevel(logging.INFO)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)

        if log_path is not None:
            file_handler = logging.FileHandler(
                filename=os.path.join(log_path, "console_copy.txt")
            )
            if verbose == 2:
                file_handler.setLevel(logging.DEBUG)
            elif verbose == 1:
                file_handler.setLevel(logging.INFO)
            else:
                file_handler.setLevel(logging.ERROR)
            file_handler.setFormatter(formatter)
            LOGGER.addHandler(file_handler)

    LOGGER.info("Optimizing observation configurations for {}".format(env))

    # Enviornment configurations including observation space settings and reward weights
    if configs is None:
        configs = {}

    n_startup_trials = 10
    n_evaluations = n_timesteps // eval_freq

    LOGGER.info(
        "Optimizing with {} trials using {} parallel jobs, each with {} maximal time steps and {} evaluations".format(
            n_trials, n_jobs, n_timesteps, n_evaluations
        )
    )

    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == "random":
        sampler = RandomSampler(seed=seed)
    elif sampler_method == "tpe":
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == "skopt":
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(
            skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"}
        )
    else:
        raise ValueError(
            "[observation_configs_opt.py] Unknown sampler: {}".format(sampler_method)
        )

    if pruner_method == "halving":
        pruner = SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=4, min_early_stopping_rate=0
        )
    elif pruner_method == "median":
        pruner = MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3
        )
    elif pruner_method == "none":
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError(
            "[observation_configs_opt.py] Unknown pruner: {}".format(pruner_method)
        )

    LOGGER.debug("Sampler: {} - Pruner: {}".format(sampler_method, pruner_method))

    # Create study object on environment configurations from Optuna
    observation_configs_study = optuna.create_study(sampler=sampler, pruner=pruner)

    # Prepare the sampler
    observation_configs_sampler = OBSERVATION_CONFIGS_SAMPLER[env]

    def objective_observation_configs(trial):
        """
        Optimization objective for environment configurations
        """
        trial.model_class = None

        if algo == "her":
            trial.model_class = hyperparams["model_class"]

        # Hack to use DDPG/TD3 noise sampler
        if algo in ["ddpg", "td3"] or trial.model_class in ["ddpg", "td3"]:
            trial.n_actions = env_fn(n_envs=1).action_space.shape[0]

        # Get keyword arguments for model hyperparameters and environment configurations
        kwargs_hyperparams = hyperparams.copy()
        kwargs_configs = configs.copy()

        # Sample environment configurations and keep model hyperparameters
        sampled_observation_configs = observation_configs_sampler(
            trial, sampling_setting
        )
        kwargs_configs.update(sampled_observation_configs)

        # Save data for later inspection
        tmp_path = os.path.join(log_path, "trial_" + str(trial.number))
        os.makedirs(tmp_path, exist_ok=True)
        with open(
            os.path.join(tmp_path, "sampled_observation_configurations.yml"), "w"
        ) as f:
            yaml.dump(kwargs_configs, f)
            LOGGER.info("Saving sampled observation configurations into " + tmp_path)
        LOGGER.debug("Sampled observation configurations:")
        LOGGER.debug(pformat(kwargs_configs))

        # Create model and environments for optimization
        model = model_fn(kwargs_hyperparams, kwargs_configs)
        eval_env = env_fn(n_envs=1, eval_env=True, **kwargs_configs)

        # Account for parallel envs
        eval_freq_ = eval_freq
        if isinstance(model.get_env(), VecEnv):
            eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)

        LOGGER.info(
            "Evaluating with {} episodes after every {} time steps".format(
                n_eval_episodes, eval_freq_
            )
        )

        observation_configs_eval_callback = ObservationConfigsTrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq_,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=True,
            verbose=verbose,
        )

        if algo == "her":
            # Wrap the env if need to flatten the dict obs
            if isinstance(eval_env, VecEnv):
                eval_env = _UnvecWrapper(eval_env)
            eval_env = HERGoalEnvWrapper(eval_env)

        try:
            model.learn(n_timesteps, callback=observation_configs_eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()

        is_pruned = observation_configs_eval_callback.is_pruned
        cost = observation_configs_eval_callback.cost
        del model.env, eval_env
        del model
        if is_pruned:
            raise optuna.exceptions.TrialPruned()
        return cost

    try:
        LOGGER.info(
            "Trying to optimize observation configurations with {} trials and {} jobs".format(
                n_trials, n_jobs
            )
        )
        observation_configs_study.optimize(
            objective_observation_configs, n_trials=n_trials, n_jobs=n_jobs
        )
    except KeyboardInterrupt:
        pass

    LOGGER.info(
        "Number of finished trials: {}".format(len(observation_configs_study.trials)),
    )
    LOGGER.info(
        "Best value: {}".format(observation_configs_study.best_trial.value),
    )
    LOGGER.info("Best observation configurations: ")
    LOGGER.info(pformat(observation_configs_study.best_trial.params))

    return observation_configs_study.best_trial.params


def sample_commonroad_observation_configs(trial, sampling_setting):
    observation_configs = {}
    for key, values in sampling_setting.items():
        method, interval = next(iter(values.items()))
        # for key, (method, interval) in sampling_setting.items():

        if method == "categorical":
            observation_configs[key] = trial.suggest_categorical(key, interval)
        elif method == "uniform":
            observation_configs[key] = trial.suggest_uniform(
                key, interval[0], interval[1]
            )
        elif method == "loguniform":
            observation_configs[key] = trial.suggest_loguniform(
                key, interval[0], interval[1]
            )
        else:
            print(
                "[observation_configs_opt.py] Sampling method "
                + method
                + " not supported"
            )
    return observation_configs


OBSERVATION_CONFIGS_SAMPLER = {
    "commonroad-v0": sample_commonroad_observation_configs,
}
