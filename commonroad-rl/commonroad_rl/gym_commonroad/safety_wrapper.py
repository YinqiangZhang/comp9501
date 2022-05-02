import os
import yaml
import pickle
import numpy as np
import gym
from gym import Wrapper
from commonroad_rl.gym_commonroad.safety_layer import SafetyLayer

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.trajectory import State, Trajectory
import commonroad_dc.pycrcc as pycrcc

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from commonroad.geometry.shape import Polygon as commonroad_Polygon
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_rl.gym_commonroad.utils.scenario import approx_orientation_vector
from commonroad_rl.utils_ind.utils.trajectory_classification import classify_trajectory, TrajectoryType
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS


__author__ = "Yinqiang Zhang"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "0.1"
__email__ = "yinqiang.zhang@tum.de"
__status__ = "Development"


class SafeWrapper(Wrapper):
    """
    Description:
        Safety wrapper for safe RL implementation in urban scenarios, all high-level code for implementation 
        are included in this class. 
    """

    def __init__(self, env, acceleration_mode: int=1, is_safe: bool=True, 
        enable_intersection_related_states=True, result_analysis=False, reaching_time_prediction=True):
        """
        initializes safety wrapper for RL training
        :param env: training environment 
        :param acceleration_mode: whether use acceleration action (0: planner, 1: policy)
        :param is_safe: with or without safety layer
        :param enable_intersection_related_states: use intersection_related_states or not 
        :param result_analysis: mode for analyzing scenario types
        """
        super(SafeWrapper, self).__init__(env)
        # action setting for discrete action
        self.acceleration_mode = acceleration_mode
        self.enable_intersection_related_states = enable_intersection_related_states
        self.is_safe = is_safe
        self.reaching_time_prediction = reaching_time_prediction
        
        self.config_file = PATH_PARAMS["project_configs"]
        with open(self.config_file, "r") as root_path:
            self.config = yaml.safe_load(root_path)

        if is_safe:
            is_iss_safe = True
            is_intersection_safe = True
        else:
            is_iss_safe = False
            is_intersection_safe = False

        self.mask_generator = SafetyLayer(self.acceleration_mode, is_iss_safe, is_intersection_safe, 
                self.enable_intersection_related_states, reaching_time_prediction=self.reaching_time_prediction)
        
        if self.acceleration_mode == 1:
            self.env.ego_action.enable_acceleration_action = True
            self.acceleration_action_num = self.config['agent_environment_interface']['acceleration_action_num']
        elif self.acceleration_mode == 0:
            self.env.ego_action.enable_acceleration_action = False
            self.acceleration_action_num = 1
        
        self.lane_change_action_num = self.config['agent_environment_interface']['lane_action_num']
        self.direction_action_num = self.config['agent_environment_interface']['direction_action_num']

        # action space of internal environment
        self.env.action_space = gym.spaces.MultiDiscrete([self.direction_action_num, 
                                                        self.direction_action_num, 
                                                        self.acceleration_action_num])
        
        # action space for the wrapped environment         
        self.total_action_num = self.acceleration_action_num \
                                * self.lane_change_action_num \
                                * self.direction_action_num

        self.action_space = gym.spaces.Discrete(self.total_action_num)
        assert self.total_action_num == np.prod(self.env.action_space.nvec), \
                "dimension of action space are not the same"
        
        if self.enable_intersection_related_states:
            state_space_dimension = self.config['agent_environment_interface']['state_space_dimension']
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (state_space_dimension,), dtype=np.float32)
        
        # save road checker
        self.road_edge_checker_cache = dict()

        # monitor information
        self.result_analysis = result_analysis
        self.episode_number = 0
        self.collsion_num = 0
        self.back_collision_num = 0
        self.failed_sampling_num = 0
        self.action_infeasible_num = 0
        self.emergency_stop_num = 0

    def reset(self):
        """
        resets the safety wrapper and corresponding environments
        :return: 
            observations from environemnt
        """
        try: 
            observation = self.env.reset()
            
            self.no_feasible_action = False
            self.current_step = 0
            
            # remove all pedestrian obstacles
            # print(f'{self.env.file_name_format % self.env.current_step}')
            
            # modify the road boundary for CommonRoadEnv
            self._remove_meaningless_road_boundary()
            
            # gnerate road boundary for reactive planner
            lanelet_polygons = self.env.cache_lanelet_polygons_accel_struct.get(self.env.meta_scenario_id, None)
            identifer = '_'.join(self.env.scenario.benchmark_id.split('_')[0:2])
            if self.road_edge_checker_cache.get(identifer, None) is not None:
                road_edge_collision_checker = self.road_edge_checker_cache[identifer]
            else:
                road_edge_collision_checker = generate_collision_model(lanelet_polygons)
                self.road_edge_checker_cache[identifer] = road_edge_collision_checker

            # collect reset information 
            self.scenario = self.env.scenario
            self.planning_problem = self.env.planning_problem 
            self.goal = self.env.goal
            ego_vehicle = self.env.ego_vehicle  
            reference_path_list = self.env.ego_action.reference_manager.reference_path_list
            observation_dict = self.env.observation_dict
            observation_history_dict = self.env.observation_history_dict
            raw_lane_list = self.env.ego_action.reference_manager.raw_lane_list

            # total distance 
            self.total_goal_long_distance = observation_history_dict['distance_goal_long'][0]

            # mask generator reset
            self.mask_generator.reset(self.scenario, self.planning_problem, road_edge_collision_checker,
                ego_vehicle, reference_path_list, raw_lane_list, observation_dict, observation_history_dict)

            self.action_mask = self.mask_generator.get_action_mask()[0]
            self.safe_maneuver_results = self.mask_generator.maneuver_results
            self.maneuver_ref_cossys = self.mask_generator.maneuver_ref_cossys
            # set the results into the reactive planner
            self.env.ego_action.safe_maneuver_results = self.safe_maneuver_results
            self.env.ego_action.maneuver_ref_cossys = self.maneuver_ref_cossys

            # record information for monitoring and debug
            self.history_action = list()
            self.history_feasible_actions = list()

            self.history_action.append(None)
            self.history_feasible_actions.append(list(k for k, element in enumerate(self.action_mask) if element == 1))

            observation = self._get_extra_observation(observation)
        except:
            # with open('./bug_action.pkl', 'wb') as f:
            #     pickle.dump(self.env.file_name_format, f)
            #     pickle.dump(self.history_action, f)
            raise
        return observation

    def step(self, action):
        """
        interacts with environment with safe guarantee
        :param action: discrete action by high level difinition
        :return: observation, reward, status and other information

        Multi discrete action definition:
        0: direction action [0, 1, 2]
        1: lane action [0, 1, 2]
        2: low-level planner action [0, 1, 2, 3, 4, 5, 6]

        Transformation formula
        action = 7 * (3 * direction_action + lane_action) + planner_action
        """
        # convert the discrete action to multi-discrete
        reshaped_action = self._reshape_action(action)
        self.history_action.append(reshaped_action)

        try: 
            # environment step
            observation, reward, done, info = self.env.step(reshaped_action)

            self.current_step += 1
            # collect update infomation
            observation_dict = self.env.observation_dict
            observation_history_dict = self.env.observation_history_dict
            self.update_info_dict = self.env.ego_action.get_update_info()

            # update action mask
            self.mask_generator.update(reshaped_action, observation_dict, 
                                    observation_history_dict, **self.update_info_dict)
            self.action_mask = self.mask_generator.get_action_mask()[0]
            self.safe_maneuver_results = self.mask_generator.maneuver_results
            self.maneuver_ref_cossys = self.mask_generator.maneuver_ref_cossys
            # set the results into the reactive planner
            self.env.ego_action.safe_maneuver_results = self.safe_maneuver_results
            self.env.ego_action.maneuver_ref_cossys = self.maneuver_ref_cossys

            if self.mask_generator.lane_change_abort:
                self.env.ego_action.lane_changing_finished = True
            
            if done:
                self.episode_number += 1

            info['is_back_collision'] = 0
            if info['is_collision'] == 1:
                # self.env.render() # record collision object
                if self._check_back_collision():
                    self.back_collision_num += 1
                    info['is_collision'] = 0
                    info['is_back_collision'] = 1
                else:
                    self.collsion_num += 1
                    info['is_collision'] = 1
                    info['is_back_collision'] = 0


            # check if failed sampling
            if self.env.ego_action.failed_trajectory_sampling:
                print("failed trajectory sampling")
                info['is_failed_sampling'] = 1
                self.failed_sampling_num += 1
                done = True
            else:
                info['is_failed_sampling'] = 0

            # check feasible action
            if np.sum(self.action_mask) == 0:
                if not self.mask_generator.reach_road_end:
                    self.no_feasible_action = True
                if self.mask_generator.is_emergency_safe:
                    print('emergency_stop')
                    info['emergency_stop_feasible'] = 1
                    self.emergency_stop_num += 1
                else:
                    print("no feasible action")
                    self.action_infeasible_num += 1
                    info['emergency_stop'] = 0
                done = True
            else:
                info['emergency_stop'] = 0

            # update information
            info['action_mask'] = self.action_mask

            # record information for monitoring and debug
            self.history_feasible_actions.append(list(k for k, element in enumerate(self.action_mask) if element == 1))
        
            observation = self._get_extra_observation(observation)
            reward = self._get_extra_reward(reward)

            if done and self.result_analysis:
                state_list = self.ego_vehicle.state_list
                delattr(state_list[0], "velocity_y")
                delattr(state_list[0], "steering_angle")
                ego_trajectory = Trajectory(initial_time_step=0, state_list=state_list)
                try:
                    info['trajectory_type'] = classify_trajectory(ego_trajectory).value
                except IndexError:
                    info['trajectory_type'] = 3
                info['planning_type'] = self._goal_trajectory_analysis(angle_threshold=10.0).value
            else:
                info['trajectory_type'] = 0
                info['planning_type'] = 0
        except:
            # the try except is used to locate the bug situation
            # with open('./bug_action.pkl', 'wb') as f:
            #     pickle.dump(self.env.file_name_format, f)
            #     pickle.dump(self.history_action, f)
            raise

        return observation, reward, done, info
    
    def _reshape_action(self, action):
        
        """
        reshape the discrete action into a multi discrete quasi-indepnedent action
        :param action: selected action by policy
        """
        if self.acceleration_mode == 1:
            planner_action = action % self.acceleration_action_num
            maneuver_action = action // self.acceleration_action_num
            direction_action = maneuver_action // self.lane_change_action_num
            lane_action = maneuver_action % self.lane_change_action_num
        else:
            planner_action = action % self.acceleration_action_num
            direction_action = action // self.lane_change_action_num
            lane_action = action % self.lane_change_action_num
        return [direction_action, lane_action, planner_action]

    def _get_extra_reward(self, reward):
        """
        defines extra reward for discrete reinforcement learning 
        : param reward: reward got from the environment
        """
        extra_reward = 0

        # reward for along goal lane:
        observation_dict = self.env.observation_dict
        # get longitudinal advance
        if observation_dict["is_goal_reached"][0]:
            remained_distance_long = observation_dict["distance_goal_long"][0]
            long_advance = remained_distance_long
        else:
            long_advance = observation_dict["distance_goal_long_advance"][0]
        # get lateral advance
        lat_advance = observation_dict["distance_goal_lat_advance"][0]
        lat_distance = observation_dict["distance_goal_lat"][0]
        # distance reward with upper bound [-40, 20]
        extra_reward += self.config['agent_environment_interface']['lat_offset_rewards'] \
						* np.sign(lat_distance) * (lat_advance/self.total_goal_long_distance) \
                        + self.config['agent_environment_interface']['lon_offset_rewards'] * (long_advance / self.total_goal_long_distance)
        # reward for acceleration: how to design this reward
        selected_action = self.history_action[-1]
        direction_action, lane_action, planner_action = selected_action

        # reward for failed sampling
        if self.env.ego_action.failed_trajectory_sampling:
            extra_reward += self.config['agent_environment_interface']['fail_sampling_rewards']
        
        # reach the road end
        # if self.mask_generator.reach_road_end:
        #     extra_reward += -0

        # not safe action
        if self.no_feasible_action:
            if not self.mask_generator.is_emergency_safe:
                extra_reward += self.config['agent_environment_interface']['emergency_braking']
            else:
                extra_reward += self.config['agent_environment_interface']['emergency_braking']
        
        # panalty for stop
        if observation_dict['v_ego'] <= 0.01:
            extra_reward += self.config['agent_environment_interface']['stop_rewards']
        # panalty for intersection region
        elif self.mask_generator.conflict_region.check_in_conflict_region(self.env.ego_vehicle):
            extra_reward += self.config['agent_environment_interface']['intersection_rewards']
        # penalty for back unsafty
        if self.mask_generator.back_risky:
            extra_reward += self.config['agent_environment_interface']['rear_collision']
        
        return reward + extra_reward
    
    def _get_extra_observation(self, observation):
        """
        defines extra observation for discrete reinforcement learning 
        """

        if self.enable_intersection_related_states:
            # remove original label object
            indicator_num = 4
            observation = observation[:-indicator_num]
            remaing_time_step = observation[-1]
            observation = observation[:-1]
        
        # add obervations for intersection
            self.intersection_observations = self.mask_generator.intersection_observations        
            rel_p = 50 * np.ones((6,))
            rel_v = np.zeros((6,))
            for k in range(len(self.intersection_observations)):
                if k < 6:
                    rel_p[k] = min(50, self.intersection_observations[k][0])
                    rel_v[k] = self.intersection_observations[k][1]

            extra_observation = np.append(observation, self.mask_generator.ego_conflict_data)
            extra_observation = np.append(extra_observation, rel_v)
            extra_observation = np.append(extra_observation, rel_p)

            # add remaining time steps
            extra_observation = np.append(extra_observation, remaing_time_step)
        else:
            extra_observation = observation

        return extra_observation

    def _remove_meaningless_road_boundary(self):
        """
        remove meaningless part of the road boundary (inside the road network in AAH_1)
        """
        modified_road_boundary = pycrcc.ShapeGroup()
        if self.env.meta_scenario_id == 'DEU_AAH-1_0_T-1':
            boundary_shapes = list(shape for shape in self.env.road_edge["boundary_collision_object"].unpack() if not (59 < shape.center()[0]< 61 and -37 < shape.center()[1]< -35))
            for shape in boundary_shapes:
                modified_road_boundary.add_shape(shape)
        self.env.road_edge["boundary_collision_object"] = modified_road_boundary

    def _check_back_collision(self):
        """
        checks the situation of collisions
        return:
            is back collision or not
        """
        collision_ego_vehicle = self.env.ego_vehicle.collision_object
        collision_objects = self.env.collision_checker.find_all_colliding_objects(collision_ego_vehicle)
        
        ego_state = self.env.ego_vehicle.state
        ego_position = ego_state.position
        ego_vector = approx_orientation_vector(ego_state.orientation)
        
        back_collision = True
        for obstacle in collision_objects:
            shape = obstacle.obstacle_at_time(self.current_step)
            if shape is not None:
                distance_vector = shape.center()-ego_position
                distance = np.linalg.norm(distance_vector)
                distance_vector /= distance
                delta_angle = np.arccos(np.vdot(ego_vector, distance_vector))
                if delta_angle < 5/6 * np.pi: # 150 degree
                    back_collision = False
                    break

        return back_collision

    def _goal_trajectory_analysis(self, angle_threshold=10.0):

        """
        analyzes the trajecotry between goal and the initial states
            
        :param angle_threshold: threshold for sector angle
        return:
            trajectory type (right, left, straight)
        """
        goal_route = self.env.goal.route.route
        end_lanelet_id = goal_route[-1]
        end_lanelet = self.env.scenario.lanelet_network.find_lanelet_by_id(end_lanelet_id)
        goal_center = end_lanelet.convert_to_polygon().center

        ego_state = self.env.ego_vehicle.state_list[0]
        ego_position = ego_state.position
        ego_vector = approx_orientation_vector(ego_state.orientation)

        goal_vector = goal_center-ego_position
        goal_vector /= np.linalg.norm(goal_vector)
        delta_angle = np.arccos(np.vdot(ego_vector, goal_vector))

        # ensure type of planning problem
        if delta_angle < np.pi*(angle_threshold/180.0):
            return TrajectoryType.STRAIGHT
        elif np.cross(ego_vector, goal_vector) > 0:
            return TrajectoryType.LEFT
        else:
            return TrajectoryType.RIGHT


def generate_collision_model(lanelet_polygons):
    """
    generates collision model of road boundaries
    :param lanelet_polygons: polygons of lanelets from scenarios
    return:
        collision model of road boundaries
    """
    shapely_polygons = list()
    for lanelet_polygon in lanelet_polygons:
        shapely_polygons.append(Polygon(lanelet_polygon[1].vertices))

    total_polygon = unary_union(shapely_polygons)
    outside_polygon = Polygon(total_polygon.exterior.coords)

    inside_polygons = outside_polygon.difference(total_polygon)
    inside_polygons = inside_polygons.buffer(-0.5)
    # inside_polygons = inside_polygons.buffer(0.2)

    convex_hull = outside_polygon.convex_hull
    difference_polygons = convex_hull.difference(outside_polygon)
    difference_polygons = difference_polygons.buffer(-0.5)
    # difference_polygons = difference_polygons.buffer(0.1)

    # generate collision model
    road_edge_collision_checker = pycrcc.ShapeGroup()
    try: 
        for inside_polygon in inside_polygons.geoms:
            points = inside_polygon.exterior
            vertices = np.stack((np.array(points.xy[0].tolist()), np.array(points.xy[1].tolist())), axis=0).T
            road_edge_collision_checker.add_shape(create_collision_object(commonroad_Polygon(vertices)))
    except AttributeError:
        points = inside_polygons.exterior.coords
        if len(points) != 0:
            vertices = np.stack((np.array(points.xy[0].tolist()), np.array(points.xy[1].tolist())), axis=0).T
            road_edge_collision_checker.add_shape(create_collision_object(commonroad_Polygon(vertices)))

    try: 
        for difference_polygon in difference_polygons.geoms:
            points = difference_polygon.exterior
            vertices = np.stack((np.array(points.xy[0].tolist()), np.array(points.xy[1].tolist())), axis=0).T
            road_edge_collision_checker.add_shape(create_collision_object(commonroad_Polygon(vertices)))
    except AttributeError:
        points = difference_polygons.exterior.coords
        if len(points) != 0:
            vertices = np.stack((np.array(points.xy[0].tolist()), np.array(points.xy[1].tolist())), axis=0).T
            road_edge_collision_checker.add_shape(create_collision_object(commonroad_Polygon(vertices)))

    return road_edge_collision_checker
