"""
This is a single demo for randomly test the environment during interaction.
In this demo, different configurations can be used, five different functions can be enabled and disabled:

1. enable_safety_layer: with or without the safety layer 
2. acceleration_mode: 0: set acceleration by planner, 1: set acceleration by policy
3. enable_intersection_related_states: with or without the intersection-related states 
4. result_analysis: analysis for the success rate of different types of scenarios 
5. prediction_method: 0: constant-velocity-based, 1: safe-braking based
"""

import os
import gym
import yaml
import pickle
import random
import numpy as np
from commonroad_rl.gym_commonroad.safety_wrapper import SafeWrapper
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

gym.logger.set_level(40)
config_file = PATH_PARAMS["project_configs"]
with open(config_file, "r") as root_path:
    configs = yaml.safe_load(root_path)

meta_scenario_path = configs['general_configurations']['meta_scenario_path']
training_data_path = configs['general_configurations']['training_data_path']
testing_data_path = configs['general_configurations']['testing_data_path']

enable_visualization = False
enable_safety_layer = configs['general_configurations']['enable_safety_layer']
acceleration_mode = configs['general_configurations']['acceleration_mode']
enable_intersection_related_states = configs['general_configurations']['observe_intersection_related_states']
result_analysis = configs['general_configurations']['result_analysis']
prediction_method = configs['general_configurations']['prediction_method']

################################ Environment Test Demo ######################################
#############################################################################################

# initialize the environment 
env = SafeWrapper(gym.make("commonroad-v0",meta_scenario_path=meta_scenario_path,
                        train_reset_config_path= training_data_path,
                        test_reset_config_path= testing_data_path, 
                        visualization_path = "./img",
                        play=True), 
                        acceleration_mode=acceleration_mode, 
                        is_safe=enable_safety_layer, 
                        enable_intersection_related_states=enable_intersection_related_states, 
                        result_analysis=result_analysis, 
                        reaching_time_prediction=prediction_method)

observation = env.reset()
done = False

t = 1
# while True:done
while not done:
    # get safe action mask
    feasible_action = env.history_feasible_actions[-1]
    # generate action randomly
    action = random.choice(feasible_action)
    observation, reward, done, info = env.step(action)
    if enable_visualization:
        env.render()
        
    print('####################################')
    print(f'{env.unwrapped.file_name_format % env.current_step}')
    print(f'feasible action {feasible_action}')
    print(f'selected action: {action}')
    print(f'velocity: {env.unwrapped.ego_vehicle.state.velocity}, acceleration: {env.unwrapped.ego_vehicle.state.acceleration}')
    print(f'reward: {reward}')
    print(f'next safe actions {env.history_feasible_actions[-1]}')
    print('####################################')

    t += 1
