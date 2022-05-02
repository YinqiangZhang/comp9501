# commonroad-v0

This is a gym environment for the CommonRoad scenarios.  

## Usage

Configurations can be defined in `commonroad_rl/gym_commonroad/configs.yaml` or command line arguments to `commonroad_rl/run_stable_baselines.py`. 
This accomodates all general settings related to the gym environment such as `flatten_observation`, and also the following main components.  
Then, for observation, corresponding initialization and update methods are to be implemented in *`self._build_observation_space()`* and *`self._get_observation()`* respectively.  
Similarly, for reward, calculations are implemented in *`self._get_reward()`*; for rendering, plotting functions are located in *`self.render()`*.  

## Main components of commonroad-v0

### Action space - Continuous

Type: `gym.spaces.Box(2)`  
Note: Both actions are normalized to [-1, 1] and in function `_get_new_state` in vehicle.py, the normalized action values are multiplied with the scaling factor (defined in the used vehicle type, as `longitudinal.a_max` and `steering.v_max`) before the actions are executed.   
Overview:

| No   | variable type | description                      |
| ---- | ------------- | -------------------------------- |
| 0    | float         | ego vehicle acceleration         |
| 1    | float         | ego vehicle steering angle rate  |

### Observation space - Continuous

Type: `gym.spaces.Box(n)` or `gym.spaces.Dict(m)` depending on *`self.DEFAULT["flatten_observation"]`*.  
Note: The following tables list our all available items. Default configuration is given in *`commonroad_rl/gym_commonroad/configs.yaml`*.  Each of the items is initialized by `gym.spaces.Box(i)` with the corresponding size `i`, and allows updates with `numpy.array()`. See *`self._build_observation_space()`* and *`self._get_observation()`* for examples.  
Overview:  

#### Ego-related

| variable name            | variable type | variable description                                         |
| ------------------------ | ------------- | ------------------------------------------------------------ |
| v_ego                    | float         | absolute velocity of ego vehicle                             |
| a_ego                    | float         | absolute acceleration of ego vehicle                         |
| steering_angle           | float         | steering angle of ego vehicle                                |
| heading                  | float         | ego vehicle orientation                                      |
| global_turn_rate         | float         | global turn rate of ego vehicle                              |
| left_marker_distance     | float         | lateral distance from ego vehicle center to left marker of ego lanelet |
| right_marker_distance    | float         | lateral distance from ego vehicle center to right marker of ego lanelet |
| left_road_edge_distance  | float         | lateral distance from ego vehicle center to left road network bound |
| right_road_edge_distance | float         | lateral distance from ego vehicle center to right road network bound |
| lat_offset               | float         | lateral distance from ego vehicle center to ego lanelet center line |

#### Goal-related

| variable name                 | variable type | variable description                                         |
| ----------------------------- | ------------- | ------------------------------------------------------------ |
| distance_goal_long            | float         | relative longitudinal distance (euclidean) from ego vehicle to goal |
| distance_goal_long_advance    | float         | relative longitudinal distance (euclidean) from ego vehicle to goal |
| distance_goal_lat             | float         | relative lateral distance (euclidean) from ego vehicle to goal |
| distance_goal_lat_advance     | float         | relative lateral distance (euclidean) from ego vehicle to goal |
| distance_goal_long_lane       | float         | relative longitudinal distance (euclidean) from ego vehicle to end of current lane |
| distance_goal_lat_extrapolated_static      | list[float]   | relative lateral distances (euclidean) from statically extrapolated ego vehicle positions to goal |
| distance_goal_lat_extrapolated_dynamic     | list[float]   | relative lateral distances (euclidean) from dynamically extrapolated ego vehicle positions to goal |


#### Surrounding-related

##### Lane-based with a rectangular sensing area  

| variable name     | variable type | variable description                                         |
| ----------------- | ------------- | ------------------------------------------------------------ |
| lane_rect_p_rel   | list[float]   | relative positions longitudinally (local ccosy) of closest vehicles, with the sequence of left-lane following, same-lane following, right-lane following, left-lane leading, same-lane leading, right-lane leading. *`self.DEFAULT["dummy_rel_pos"]`* if none.   |
| lane_rect_v_rel   | list[float]   | relative velocities longitudinally (local ccosy) of closest vehicles, with the sequence of left-lane following, same-lane following, right-lane following, left-lane leading, same-lane leading, right-lane leading. *`self.DEFAULT["dummy_rel_vol"]`* if none.    |

##### Lane-based with a circular sensing area  

| variable name     | variable type | variable description                                         |
| ----------------- | ------------- | ------------------------------------------------------------ |
| lane_circ_p_rel   | list[float]   | relative positions longitudinally (local ccosy) of closest vehicles, with the sequence of left-lane following, same-lane following, right-lane following, left-lane leading, same-lane leading, right-lane leading. *`self.DEFAULT["dummy_rel_pos"]`* if none.   |
| lane_circ_v_rel   | list[float]   | relative velocities longitudinally (local ccosy) of closest vehicles, with the sequence of left-lane following, same-lane following, right-lane following, left-lane leading, same-lane leading, right-lane leading. *`self.DEFAULT["dummy_rel_vol"]`* if none.    |

##### Lidar-based with a elliptical sensing area  

| variable name         | variable type | variable description                                         |
| --------------------- | ------------- | ------------------------------------------------------------ |
| lidar_elli_dist       | list[float]   | euclidean distances of closest vehicles, with *`self.DEFAULT["num_beams"]`* number of elements. *`self.DEFAULT["dummy_dist"]`* if none.    |
| lidar_elli_dist_rate  | list[float]   | change rate of euclidean distances of closest vehicles, with *`self.DEFAULT["num_beams"]`* number of elements. *`self.DEFAULT["dummy_dist_rate"]`* if none. |

#### Termination-related

| variable name            | variable type | variable description                                         |
| ------------------------ | ------------- | ------------------------------------------------------------ |
| remaining_steps          | int           | number of time steps left for current episode                |
| is_goal_reached          | boolean       | identifier to determine if ego vehicle reaches goal region   |
| is_off_road              | boolean       | identifier to determine if ego vehicle is off road           |
| is_collision             | boolean       | identifier to determine if ego vehicle collides with other vehicles |
| is_time_out              | boolean       | identifier to determine if maximum episode length is met     |
| is_friction_violation    | boolean       | identifier to determine if ego vehicle violates the friction constraints |


### Reward functions

Type: Three types of reward functions are provided in the following, selected by *`commonroad_rl/gym_commonroad/configs.yaml ["reward_type"]`*.
- Sparse reward: reward_goal_reached + reward_collision + reward_off_road
- Hybrid reward: reward_goal_reached + reward_collision + reward_off_road + reward_time_out
		    + reward_friction_violation + reward_get_close_coefficient * distance_advancement
- Dense reward: reward_obs_distance_coefficient * normalized_distances_to_obstabcles
		    + reward_goal_distance_coefficient * normalized_distance_to_goal 

Note that rewards for reaching the goal and getting close to the goal are usually positive whereas other rewards may be chosen to be negative.
Of course this is up to the user.

Note: Default configuration is given in [*`commonroad_rl/gym_commonroad/configs.yaml`*].  
Overview: 

| variable name             | variable type | variable description                                          |
| ------------------------- | ------------- | ------------------------------------------------------------- |
| reward_goal_reached       | float         | reward for reaching the goal                                  | 
| reward_collision          | float         | reward for collision with another vehicle                     | 
| reward_off_road           | float         | reward for going odd road/ colliding with the road boundary   |
| reward_time_out           | float         | reward for exceeding the maximum goal reaching timestep       | 
| reward_friction_violation | float         | reward for friction violation                                 |
| reward_get_close_coefficient      | float         | reward coefficient for the decrease of goal distance  | 
| reward_obs_distance_coefficient   | float         | reward coefficient for normalized distances against obstacles  | 
| reward_goal_distance_coefficient  | float         | reward coefficient for normalized distances towards goal  |


### Episode termination

Note: Default configuration is given in [`commonroad_rl/gym_commonroad/configs.yaml`].  
Overview:

| variable name             | variable type | variable description                                          |
| ------------------------- | ------------- | ------------------------------------------------------------- |
| terminate_on_goal_reached | bool          | terminate when goal is reached if true                        |
| terminate_on_off_road     | bool          | terminate when ego vehicle is off-road if true                |
| terminate_on_collision    | bool          | terminate when ego vehicle collides with obstacles if true    |
| terminate_on_time_out     | bool          | terminate when maximum episode length is met if true  	    |
| terminate_on_friction_violation | bool    | terminate when friction limitation is violated if true        |
