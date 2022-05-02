"""
This demo shows how to train a policy to overfit several scenarios
"""

import os
os.environ["KMP_WARNINGS"] = "off"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_AFFINITY"] = "none"

import yaml
import copy
import gym
from commonroad_rl.gym_commonroad.safety_wrapper import SafeWrapper
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from stable_baselines.bench import Monitor
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
from stable_baselines.common.callbacks import (
    BaseCallback, 
    EvalCallback, 
    CallbackList, 
    EveryNTimesteps,
    EventCallback,
)
from stable_baselines.logger import configure
from stable_baselines import PPO2

config_file = PATH_PARAMS["project_configs"]
with open(config_file, "r") as root_path:
    configs = yaml.safe_load(root_path)

# environment configurations
os.chdir(configs['general_configurations']['ROOT_PATH'])

meta_scenario_path = configs['general_configurations']['meta_scenario_path']
training_data_path = configs['general_configurations']['training_data_path']
testing_data_path = configs['general_configurations']['testing_data_path']

log_root_name = 'AAH_1_demo'
results_folder = 'training_results'
# safe or action mode version
is_safe = True
action_mode = 1
intersection_obs = True
reaching_time_prediction = True

gym.logger.set_level(40)

env_configs = {}
with open(os.path.join(os.getcwd(),"commonroad-rl/commonroad_rl/gym_commonroad/configs.yaml"), 'r') as config_file:
    env_configs = yaml.safe_load(config_file)["env_configs"]

# type of reward
env_configs["reward_type"] = "sparse_reward"

# save settings for later use
log_path = os.path.join(os.getcwd(), os.path.join(results_folder, log_root_name))
os.makedirs(log_path, exist_ok=True)

# write to save 
with open(os.path.join(log_path, "environment_configurations.yml"), 'w') as config_file:
    yaml.dump(env_configs, config_file)

# read in model hyperparameters
hyperparams = {}
with open(os.path.join(os.getcwd(),"commonroad-rl/commonroad_rl/hyperparams/ppo2.yml"), 'r') as hyperparam_file:
    hyperparams = yaml.safe_load(hyperparam_file)["commonroad-v0"]

# write hyperparams for later use
with open(os.path.join(log_path, "model_hyperparameters.yml"), 'w') as hyperparam_file:
    yaml.dump(hyperparams, hyperparam_file)

# remove normalize 
if "normalize" in hyperparams:
    del hyperparams["normalize"]
######################################################################################

training_env = SafeWrapper(gym.make("commonroad-v0", 
                        meta_scenario_path=meta_scenario_path,
                        train_reset_config_path= training_data_path,
                        **env_configs), 
                        acceleration_mode=action_mode, 
                        is_safe=is_safe, 
                        enable_intersection_related_states=intersection_obs, 
                        reaching_time_prediction=reaching_time_prediction)

# Wrap the environment with a monitor to keep an record of the learning process
info_keywords=tuple(["is_collision", \
                     "is_time_out", \
                     "is_off_road", \
                     "is_back_collision", \
                     "is_goal_reached", \
                     "scenario_name"])


training_env = Monitor(training_env, log_path + "/infos", info_keywords=info_keywords)
training_env = DummyVecEnv([lambda: training_env])
training_env = VecCheckNan(training_env, raise_exception=True, check_inf=True)
training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True)
# training_env = make_vec_env(CommonroadEnv, n_envs= 4, env_kwargs=env_configs, monitor_dir=log_path + "/infos")

# Append the additional key: change test mode for later use
env_configs_test = copy.deepcopy(env_configs)
env_configs_test["test_env"] = True

testing_env = SafeWrapper(gym.make("commonroad-v0", 
                        meta_scenario_path=meta_scenario_path,
                        test_reset_config_path= testing_data_path,
                        **env_configs_test), 
                        acceleration_mode=action_mode, 
                        is_safe=is_safe,
                        enable_intersection_related_states=intersection_obs, 
                        reaching_time_prediction=reaching_time_prediction)

# Wrap the environment with a monitor to keep an record of the testing episodes
log_path_test = os.path.join(os.getcwd(), f"{results_folder}/{log_root_name}/test")
os.makedirs(log_path_test, exist_ok=True)

testing_env = Monitor(testing_env, log_path_test + "/infos", info_keywords=info_keywords)
# Vectorize the environment with a callable argument
testing_env = DummyVecEnv([lambda: testing_env])
testing_env = VecCheckNan(testing_env, raise_exception=True, check_inf=True)
testing_env = VecNormalize(testing_env, norm_obs=True, norm_reward=False)

# Define a customized callback function to save the vectorized and normalized environment wrapper
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1, n_steps: int=10000):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path
        self.step = 0
        self.n_steps = n_steps
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self) -> bool:
        self.step += self.n_steps
        save_path_name = os.path.join(self.save_path, f"{self.step}_vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(save_path_name)
        print("Saved vectorized and normalized environment to {}".format(save_path_name))

eval_callback = EvalCallback(testing_env, 
                             log_path=log_path, 
                             eval_freq=50000, 
                             n_eval_episodes=1,
                             deterministic=True,
                             )

log_path_check = os.path.join(os.getcwd(), f"{results_folder}/{log_root_name}/check")
os.makedirs(log_path_check, exist_ok=True)

checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=log_path_check,name_prefix='rl_model')
save_vec_normalize_callback = SaveVecNormalizeCallback(save_path=log_path_check, n_steps=20000)
save_vec_resutls_callback = EveryNTimesteps(n_steps=20000, callback=save_vec_normalize_callback)

callback = CallbackList([checkpoint_callback, eval_callback, save_vec_resutls_callback])
########################################################################################

configure(folder=f"{results_folder}/{log_root_name}/ppo2_tensorboard/", 
        format_strs=['stdout', 'log', 'csv', 'tensorboard'])

# two layers with 128 nodes
layers = [128, 128]
policy_prameters = dict(net_arch=[dict(vf=layers, pi=layers)])

# create a model with hyperparameters
model = PPO2(env=training_env, **hyperparams, verbose=1, policy_kwargs=policy_prameters,
        tensorboard_log=f"{results_folder}/{log_root_name}/ppo2_tensorboard/", full_tensorboard_log=False)

# Start the learning process with the evaluation callback
minibatch_size = 2048
n_timesteps = 250 * 2048
model.learn(n_timesteps, callback)

model.save(f"{results_folder}/{log_root_name}/final_model")