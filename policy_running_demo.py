"""
This demo shows how to test the previously learned policy and visualize the results.
The demo policy is learned with safety layer, use the constant-velocity appraoch to predict reaching time.
Besides, it enables the acceleration action.
"""

import os
os.environ["KMP_WARNINGS"] = "off"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_AFFINITY"] = "none"

import gym
import yaml
import copy
import pickle

from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.safety_wrapper import SafeWrapper
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

gym.logger.set_level(40)
config_file = PATH_PARAMS["project_configs"]
with open(config_file, "r") as root_path:
    configs = yaml.safe_load(root_path)

os.chdir(configs['general_configurations']['ROOT_PATH'])
log_path = os.path.join(os.getcwd(), f"training_results/demo_policy")

# load environment configurations
env_configs = {}
with open(os.path.join(log_path, "environment_configurations.yml"), "r") as config_file:
    env_configs = yaml.safe_load(config_file)

# only get original figure
env_configs['render_dynamic_extrapolated_positions'] = True
env_configs['render_lane_circ_surrounding_area'] = False
env_configs['render_lane_circ_surrounding_obstacles'] = False

# Read in model hyperparameters
hyperparams = {}
with open(os.path.join(log_path, "model_hyperparameters.yml"), "r") as hyperparam_file:
    hyperparams = yaml.safe_load(hyperparam_file)

# Deduce `policy` from the pretrained model
if "policy" in hyperparams:
    del hyperparams["policy"]

# Remove `normalize` as it will be handled explicitly later
if "normalize" in hyperparams:
    del hyperparams["normalize"]

# this part is similar to the vanilla training procedures
#######################################################################
# Create a Gym-based RL environment

meta_scenario_path = configs['general_configurations']['meta_scenario_path']
training_data_path = configs['general_configurations']['training_data_path']
testing_data_path = configs['general_configurations']['testing_data_path']

training_env = SafeWrapper(gym.make("commonroad-v0", 
                        meta_scenario_path=meta_scenario_path,
                        train_reset_config_path= training_data_path,
                        test_reset_config_path = testing_data_path,
                        visualization_path= "./img",
                        play=True,
                        **env_configs), 
                        is_safe=True,
                        acceleration_mode=1, 
                        result_analysis=False,
                        enable_intersection_related_states=True,
                        reaching_time_prediction=True)

training_env = DummyVecEnv([lambda: training_env])
training_env = VecNormalize.load(os.path.join(log_path, "check", "510000_vecnormalize.pkl"), training_env)
model_continual = PPO2.load(os.path.join(log_path, "check", "rl_model_510000_steps"), env=training_env, **hyperparams)

done = False
observation = training_env.reset()
training_env.render()

while True:
    try:
        acion_mask = training_env.get_attr('action_mask', 0)
        # select action
        action, _states = model_continual.predict(observation, action_mask=acion_mask, deterministic=True)
        observation, reward, done, info = training_env.step(action)
        training_env.render()
    except:
        # all scenarios are evaluated in the target folder
        break
