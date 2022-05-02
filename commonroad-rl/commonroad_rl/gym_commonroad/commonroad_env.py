__author__ = "Xiao Wang, Brian Liao, Niels Muendler, Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

"""
Module for the CommonRoad Gym environment
"""
import os
import gym
import glob
import yaml
import time
import pickle
import random
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, Dict, Union
from shapely.geometry import LineString
from collections import OrderedDict, defaultdict

# import from commonroad-drivability-checker
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)
from commonroad_dc.collision.visualization import draw_dispatch as crdc_draw_dispatch

# import from commonroad-io
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import State
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.common.util import make_valid_orientation
from commonroad.geometry.shape import Rectangle

# import from commonroad-rl
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.gym_commonroad.feature_extraction.goal import GoalObservation
from commonroad_rl.gym_commonroad.feature_extraction.surroundings import (
    get_surrounding_obstacles_lane_rect,
    get_surrounding_obstacles_lane_circ,
    get_surrounding_obstacles_lidar_elli,
)
from commonroad_rl.gym_commonroad.utils.scenario import (
    get_lane_marker,
    get_nearby_lanelet_id,
    sorted_lanelets_by_state_realtime,
    get_relative_offset,
    get_lane_relative_heading,
    get_local_curvi_cosy,
    get_distance_to_marker_and_road_edge,
    get_relative_future_goal_offsets,
    approx_orientation_vector,
)

from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.gym_commonroad.vehicle import Vehicle

# new adding for discrete action implementation
from commonroad_rl.gym_commonroad.action import ReactivePlannerAction
from commonroad.scenario.obstacle import ObstacleType


matplotlib.use("AGG")
LOGGER = logging.getLogger(__name__)

verbose_to_logging = {1: logging.INFO, 2: logging.DEBUG, 0: logging.ERROR}


class CommonroadEnv(gym.Env):
    """
    Description:
        This environment simulates the ego vehicle in a traffic scenario using commonroad environment. The task of
        the ego vehicle is to reach the predefined goal without going off-road, collision with other vehicles, and
        finish the task in specific time frame. Please consult `commonroad_rl/gym_commonroad/README.md` for details.
    """

    metadata = {"render.modes": ["human"]}

    # For the current configuration check the ./configs.yaml file
    # TODO: Implementation the configuration in object-oriented fashion ->
    #  clean code, easier to find the available attributes, does not violate PEP8

    def __init__(
            self,
            meta_scenario_path=PATH_PARAMS["meta_scenario"],
            train_reset_config_path=PATH_PARAMS["train_reset_config"],
            test_reset_config_path=PATH_PARAMS["test_reset_config"],
            visualization_path=PATH_PARAMS["visualization"],
            logging_path=None,
            test_env=False,
            play=False,
            config_file=PATH_PARAMS["configs"],
            verbose=1,
            **kwargs,
    ) -> None:
        """
        Initialize environment, set scenario and planning problem.
        """
        # Set logger if not yet exists
        LOGGER.setLevel(verbose_to_logging[verbose])

        if not len(LOGGER.handlers):
            formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(verbose_to_logging[verbose])
            stream_handler.setFormatter(formatter)
            LOGGER.addHandler(stream_handler)

            if logging_path is not None:
                file_handler = logging.FileHandler(filename=os.path.join(logging_path, "console_copy.txt"))
                file_handler.setLevel(verbose_to_logging[verbose])
                file_handler.setFormatter(formatter)
                LOGGER.addHandler(file_handler)

        LOGGER.debug("Initialization started")

        # Default configuration
        with open(config_file, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Assume default environment configurations from self.DEFAULT
        self.configs = config["env_configs"]  # deepcopy(self.DEFAULT)

        # Overwrite environment configurations if specified
        if kwargs is not None:
            for k, v in kwargs.items():
                assert k in self.configs, f"Configuration item not supported: {k}"
                self.configs.update({k: v})

        # Make environment configurations as attributes
        for key, value in self.configs.items():
            setattr(self, key, value)

        # Flag for popping out scenarios
        self.play = play

        # Load scenarios and problems
        self.meta_scenario_path = meta_scenario_path
        self.all_problem_dict = dict()
        self.planning_problem_set_list = []

        # Accelerator structures
        self.cache_scenario_ref_path_dict = dict()
        self.cache_collision_checker_templates = dict()
        self.cache_goal_obs = dict()
        self.cache_lanelet_polygons_accel_struct = dict()
        self.cache_lanelet_polygons_sg_accel_struct = dict()

        meta_scenario_reset_dict_path = os.path.join(
            self.meta_scenario_path, "meta_scenario_reset_dict.pickle"
        )
        with open(meta_scenario_reset_dict_path, "rb") as f:
            self.meta_scenario_reset_dict = pickle.load(f)

        problem_meta_scenario_dict_path = os.path.join(
            self.meta_scenario_path, "problem_meta_scenario_dict.pickle"
        )
        with open(problem_meta_scenario_dict_path, "rb") as f:
            self.problem_meta_scenario_dict = pickle.load(f)

        if not test_env and not play:
            fns = glob.glob(os.path.join(train_reset_config_path, "*.pickle"))
            for fn in fns:
                with open(fn, "rb") as f:
                    self.all_problem_dict[os.path.basename(fn).split(".")[0]] = pickle.load(f)
            self.is_test_env = False
            LOGGER.info(f"Training on {train_reset_config_path} "
                        f"with {len(self.all_problem_dict.keys())} scenarios")
        else:
            fns = glob.glob(os.path.join(test_reset_config_path, "*.pickle"))
            for fn in fns:
                with open(fn, "rb") as f:
                    self.all_problem_dict[os.path.basename(fn).split(".")[0]] = pickle.load(f)
            LOGGER.info(f"Testing on {test_reset_config_path} "
                        f"with {len(self.all_problem_dict.keys())} scenarios")

        self.visualization_path = visualization_path
        self.current_step = 0
        self.terminated = False
        self.ego_vehicle: Union[Vehicle, None] = None
        self.goal: Union[GoalObservation, None] = None

        # action space
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 7])

        # high level action object
        self.ego_action = ReactivePlannerAction()

        # Observation space
        self._build_observation_space()

        LOGGER.debug(f"Meta scenario path: {meta_scenario_path}")
        LOGGER.debug(f"Training data path: {train_reset_config_path}")
        LOGGER.debug(f"Testing data path: {test_reset_config_path}")
        LOGGER.debug("Initialization done")

    def seed(self, seed=Union[None, int]):
        self.action_space.seed(seed)

    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        :return: observation
        """

        self._set_scenario_problem()
        self._set_ego_vehicle()
        self._set_goal()

        self.current_step = 0
        self.num_friction_violation = 0
        self.terminated = False

        observation = self._get_observation()

        self.ego_action.reset(self.scenario, self.ego_vehicle)

        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Propagate to next time step, compute next observations, reward and status.
        :param action: vehicle acceleration, vehicle steering velocity
        :return: observation, reward, status and other information
        """
        self.current_step += 1

        self.ego_vehicle = self.ego_action.step(action, self.ego_vehicle)

        observation = self._get_observation()
        
        done = False
        is_time_out = 0
        is_collision = 0
        is_off_road = 0
        is_goal_reached = 0
        is_friction_violation = 0
        
        if self.observation_history_dict["is_goal_reached"][-1]:  # Goal region is reached
            is_goal_reached = 1
            if self.terminate_on_goal_reached:
                done = True
        elif self.observation_history_dict["is_off_road"][-1]:  # Ego vehicle is off-road
            is_off_road = 1
            if self.terminate_on_off_road:
                done = True
        elif self.observation_history_dict["is_collision"][-1]:  # Collision with others
            is_collision = 1
            if self.terminate_on_collision:
                done = True
        elif self.observation_history_dict["is_time_out"][-1]:  # Max simulation time step is reached
            is_time_out = 1
            if self.terminate_on_time_out:
                done = True
        elif self.observation_history_dict["is_friction_violation"][-1]:  # Friction limitation is violated
            is_friction_violation = 1
            self.num_friction_violation += is_friction_violation
            if self.terminate_on_friction_violation:
                done = True

        reward = self._get_reward()

        info = {
            "scenario_name": self.benchmark_id,
            "chosen_action": action,
            "current_episode_time_step": self.current_step,
            "max_episode_time_steps": self.ep_length,
            "is_goal_reached": is_goal_reached,
            "is_collision": is_collision,
            "is_off_road": is_off_road,
            "is_time_out": is_time_out,
            "is_friction_violation": self.num_friction_violation,
        }

        return observation, reward, done, info

    def render(self, mode: str = "human") -> None:
        """
        Generate images for visualization.
        :param mode: default as human for visualization
        :return: None
        """
        # Render only every xth timestep, the first and the last
        if not (self.current_step % self.render_skip_timesteps == 0 or self.terminated):
            return

        # Draw scenario, goal, sensing range and detected obstacles
        draw_object(
            self.scenario,
            draw_params={"time_begin": self.current_step,
                "scenario": {"lanelet_network": {"lanelet": {"show_label": False, "fill_lanelet": True}}}},
        )

        # Draw certain objects only once
        if not self.render_combine_frames or self.current_step == 0:
            draw_object(self.planning_problem)

            # Draw road boundaries
            if self.render_road_boundaries:
                crdc_draw_dispatch.draw_object(self.road_edge["boundary_collision_object"],
                                               draw_params={"collision": {"rectobb": {"facecolor": "yellow"}}})

            if self.render_global_ccosy:
                # TODO: This functionality has been taken from commonroad-route-planner
                # As soon as the route-planner supports drawing only the ccosy, this part should be replaced
                for route_merged_lanelet in self.goal.navigator.merged_route_lanelets:
                    draw_object(route_merged_lanelet,
                                draw_params={
                                    "lanelet": {
                                    # 'left_bound_color': '#0de309',
                                    # 'right_bound_color': '#0de309',
                                    "center_bound_color": "#128c01",
                                    "unique_colors": False,
                                    # colorizes center_vertices and labels of each lanelet differently
                                    "draw_stop_line": True,
                                    "stop_line_color": "#ffffff",
                                    "draw_line_markings": False,
                                    "draw_left_bound": False,
                                    "draw_right_bound": False,
                                    "draw_center_bound": True,
                                    "draw_border_vertices": False,
                                    "draw_start_and_direction": False,
                                    "show_label": False,
                                    "draw_linewidth": 1,
                                    "fill_lanelet": False,
                                    "facecolor": "#128c01",
                                    }
                                },
                                )

        # Draw ego vehicle
        crdc_draw_dispatch.draw_object(self.ego_vehicle.collision_object,
                                       draw_params={"collision": {"facecolor": "green", "zorder": 30}})

        # Plot ego lanelet center vertices
        if self.render_ego_lanelet_center_vertices:
            ego_lanelet_center_vertices = (self.scenario.lanelet_network.find_lanelet_by_id(
                self.observation_history_dict["ego_vehicle_lanelet_id"][-1]).center_vertices)
            plt.plot(
                ego_lanelet_center_vertices[:, 0],
                ego_lanelet_center_vertices[:, 1],
                color="pink",
                zorder=5,
            )

        # Plot ccosys
        if self.render_local_ccosy:
            draw_object(
                self.local_merged_lanelet,
                draw_params={
                    "lanelet": {
                        # 'left_bound_color': '#0de309',
                        # 'right_bound_color': '#0de309',
                        # 'center_bound_color': '#0de309',
                        "unique_colors": False,  # colorizes center_vertices and labels of each lanelet differently
                        "draw_stop_line": False,
                        "stop_line_color": "#ffffff",
                        "draw_line_markings": False,
                        "draw_left_bound": False,
                        "draw_right_bound": False,
                        "draw_center_bound": False,
                        "draw_border_vertices": False,
                        "draw_start_and_direction": False,
                        "show_label": False,
                        "draw_linewidth": 0.5,
                        "fill_lanelet": True,
                        "facecolor": "wheat",
                    }
                },
            )

        # Mark surrounding obstacles (Only if corresponding oberservations are available)
        # Lane-based rectangular surrounding rendering
        if (self.observe_lane_rect_surrounding and self.render_lane_rect_surrounding_area and not self.terminated):
            crdc_draw_dispatch.draw_object(self.observation_history_dict["lane_rect_surrounding_area"][-1],
                                           draw_params={"collision": {"facecolor": "None"}})
        if (self.observe_lane_rect_surrounding and self.render_lane_rect_surrounding_obstacles and not self.terminated):
            colors = ["r", "r", "y", "y", "k", "k"]
            for obs, color in zip(self.observation_history_dict["lane_rect_obs"][self.current_step], colors):
                if obs is not None:
                    plt.plot( obs.position[0], obs.position[1], color=color, marker="*", zorder=20)

        # Lane-based circular surrounding rendering
        if (self.observe_lane_circ_surrounding and self.render_lane_circ_surrounding_area and not self.terminated):
            crdc_draw_dispatch.draw_object(self.observation_history_dict["lane_circ_surrounding_area"][-1],
                                           draw_params={"collision": {"facecolor": "None"}})
        if (self.observe_lane_circ_surrounding and self.render_lane_circ_surrounding_obstacles and not self.terminated):
            colors = ["r", "r", "y", "y", "k", "k"]
            for obs, color in zip(self.observation_history_dict["lane_circ_obs"][self.current_step], colors):
                if obs is not None:
                    plt.plot(obs.position[0], obs.position[1], color=color, marker="*", zorder=20)

        # Lidar-based elliptical surrounding rendering
        if (self.observe_lidar_elli_surrounding and self.render_lidar_elli_surrounding_beams and not self.terminated):
            for (beam_start, beam_length, beam_angle) in self.observation_history_dict["lidar_elli_surrounding_beams"][-1]:
                center = beam_start + 0.5 * beam_length * approx_orientation_vector(beam_angle)
                beam_angle = make_valid_orientation(beam_angle)
                beam_draw_object = Rectangle(length=beam_length, width=0.1, center=center, orientation=beam_angle)
                draw_object(beam_draw_object)
        if (self.observe_lidar_elli_surrounding and self.render_lidar_elli_surrounding_obstacles
                and not self.terminated):
            for idx, detection_point in enumerate(self.observation_history_dict["lidar_elli_obs"][-1]):
                plt.plot(
                    detection_point[0],
                    detection_point[1],
                    color="b",
                    marker="1",
                    zorder=20,
                )

        # Extrapolated future positions
        if (self.observe_static_extrapolated_positions and self.render_static_extrapolated_positions):
            for future_pos in self.observation_history_dict["extrapolation_static_pos"][-1]:
                plt.plot(future_pos[0], future_pos[1], color="r", marker="x", zorder=21)
        if (self.observe_dynamic_extrapolated_positions and self.render_dynamic_extrapolated_positions):
            for future_pos in self.observation_history_dict["extrapolation_dynamic_pos"][-1]:
                plt.plot(future_pos[0], future_pos[1], color="b", marker="x", zorder=21)

        # plot all sampled trajectories
        # self.ego_action.planner.draw_trajectory_set(
        #     self.ego_action.planner.bundle.trajectories
        # )
        # plt.plot(
        #     self.ego_action.reference_manager.source_reference_lane[1].center_vertices[:, 0],
        #     self.ego_action.reference_manager.source_reference_lane[1].center_vertices[:, 1],
        #     color="g",
        #     zorder=21,
        #     linewidth=1,
        # )
        # plt.plot(
        #     self.ego_action.reference_manager.destination_reference_lane[1].center_vertices[:, 0],
        #     self.ego_action.reference_manager.destination_reference_lane[1].center_vertices[:, 1],
        #     color="b",
        #     zorder=21,
        #     linewidth=1,
        # )
        # plt.plot(
        #     self.ego_action.reference_manager.reference_path[:, 0],
        #     self.ego_action.reference_manager.reference_path[:, 1],
        #     color="r",
        #     zorder=21,
        #     linewidth=1,
        # )
        
        plt.gca().set_aspect("equal")
        plt.autoscale()

        # Save figure, only if frames should not be combined or simulation is over
        os.makedirs(
            os.path.join(self.visualization_path, self.scenario.benchmark_id),
            exist_ok=True,
        )
        if not self.render_combine_frames or self.terminated:
            plt.savefig(os.path.join(self.visualization_path, self.scenario.benchmark_id,
                                     self.file_name_format % self.current_step) + ".png",
                        format="png",
                        dpi=300,
                        bbox_inches="tight")
            plt.close()

    # =================================================================================================================
    #
    #                                  __init__ functions
    #
    # =================================================================================================================

    def _build_observation_space(self) -> None:
        """
        Contruct observation space. Run once during __init__
        :return: None
        """
        observation_space_dict = OrderedDict()

        # Ego-related
        if self.observe_v_ego:
            observation_space_dict["v_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_a_ego:
            observation_space_dict["a_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_steering_angle:
            observation_space_dict["steering_angle"] = gym.spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        if self.observe_heading:
            observation_space_dict["heading"] = gym.spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        if self.observe_global_turn_rate:
            observation_space_dict["global_turn_rate"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32
                                                                        )
        if self.observe_left_marker_distance:
            observation_space_dict["left_marker_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_right_marker_distance:
            observation_space_dict["right_marker_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_left_road_edge_distance:
            observation_space_dict["left_road_edge_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_right_road_edge_distance:
            observation_space_dict["right_road_edge_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_lat_offset:
            observation_space_dict["lat_offset"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        # Goal-related
        if self.observe_distance_goal_long:
            observation_space_dict["distance_goal_long"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            observation_space_dict["distance_goal_long_advance"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                  dtype=np.float32)
        if self.observe_distance_goal_lat:
            observation_space_dict["distance_goal_lat"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            observation_space_dict["distance_goal_lat_advance"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                 dtype=np.float32)
        if self.observe_distance_goal_long_lane:
            observation_space_dict["distance_goal_long_lane"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_static_extrapolated_positions:
            sampling_points = self.static_extrapolation_samples
            observation_space_dict["extrapolation_static_off"] = gym.spaces.Box(-np.inf, np.inf,
                                                                                (len(sampling_points),),
                                                                                dtype=np.float32)
        if self.observe_distance_goal_time:
            observation_space_dict["distance_goal_time"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_distance_goal_orientation:
            observation_space_dict["distance_goal_orientation"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                 dtype=np.float32)
        if self.observe_distance_goal_velocity:
            observation_space_dict["distance_goal_velocity"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        if self.observe_dynamic_extrapolated_positions:
            sampling_points = self.dynamic_extrapolation_samples
            observation_space_dict["extrapolation_dynamic_off"] = gym.spaces.Box(-np.inf, np.inf,
                                                                                 (len(sampling_points),),
                                                                                 dtype=np.float32)

        # Surrounding-related
        # # Lane-based rectangular surrounding observation
        if self.observe_lane_rect_surrounding:
            observation_space_dict["lane_rect_v_rel"] = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
            observation_space_dict["lane_rect_p_rel"] = gym.spaces.Box(-self.dummy_dist, self.dummy_dist, (6,),
                                                                       dtype=np.float32)
        # # Lane-based circular surrounding observation
        if self.observe_lane_circ_surrounding:
            observation_space_dict["lane_circ_v_rel"] = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
            observation_space_dict["lane_circ_p_rel"] = gym.spaces.Box(-self.dummy_dist, self.dummy_dist, (6,),
                                                                       dtype=np.float32)
        # # Lidar-based elliptical surrounding observation
        if self.observe_lidar_elli_surrounding:
            num_beams = self.lidar_elli_num_beams
            observation_space_dict["lidar_elli_dist_rate"] = gym.spaces.Box(-np.inf, np.inf, (num_beams,),
                                                                            dtype=np.float32)
            observation_space_dict["lidar_elli_dist"] = gym.spaces.Box(-self.dummy_dist, self.dummy_dist, (num_beams,),
                                                                       dtype=np.float32)

        # Termination-related
        if self.observe_remaining_steps:
            observation_space_dict["remaining_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_is_time_out:
            observation_space_dict["is_time_out"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        if self.observe_is_collision:
            observation_space_dict["is_collision"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        if self.observe_is_off_road:
            observation_space_dict["is_off_road"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        if self.observe_is_friction_violation:
            observation_space_dict["is_friction_violation"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        if self.observe_is_goal_reached:
            observation_space_dict["is_goal_reached"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)

        # TODO: discrete action for space
        # observation_space_dict["lane_change_status"] = gym.spaces.Box(0, 1, (2,), dtype=np.int8)

        # Flatten observation if required
        self.observation_space_dict = observation_space_dict
        if self.flatten_observation:
            self.observation_space_size = sum([np.prod(i.shape) for i in self.observation_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.observation_space_size,), dtype=np.float32)
            LOGGER.debug(f"Size of flattened observation space: {self.observation_space_size}")
        else:
            self.observation_space = gym.spaces.Dict(self.observation_space_dict)
            LOGGER.debug(f"Length of dictionary observation space: {len(self.observation_space_dict)}")

    # =================================================================================================================
    #
    #                                    reset functions
    #
    # =================================================================================================================
    def _set_scenario_problem(self) -> None:
        """
        Set scenario and planning problem, create collision checker.
        :return: None
        """

        def clone_collision_checker(cc: pycrcc.CollisionChecker, ) -> pycrcc.CollisionChecker:
            return cc.clone()

        if self.play:
            # pop instead of reusing
            LOGGER.info(f"Number of scenarios left {len(list(self.all_problem_dict.keys()))}")
            self.benchmark_id = random.choice(list(self.all_problem_dict.keys()))
            problem_dict = self.all_problem_dict.pop(self.benchmark_id)
        else:
            self.benchmark_id, problem_dict = random.choice(list(self.all_problem_dict.items()))

        meta_scenario = self.problem_meta_scenario_dict[self.benchmark_id]
        obstacle_list = problem_dict["obstacle"]
        self.scenario = restore_scenario(meta_scenario, obstacle_list)
        
        ### remove pedestrian and bycicles 
        self.scenario = remove_useless_participants(self.scenario)
        
        self.meta_scenario_id = meta_scenario.benchmark_id

        if self.cache_scenario_ref_path_dict.get(self.meta_scenario_id, None) is None:
            self.cache_scenario_ref_path_dict[self.meta_scenario_id] = dict()

        self.planning_problem: PlanningProblem = random.choice(
            list(problem_dict["planning_problem_set"].planning_problem_dict.values())
        )

        # Set episode length
        self.ep_length = max(s.time_step.end for s in self.planning_problem.goal.state_list)

        # Set name format for visualization
        self.file_name_format = self.benchmark_id + "_ts_%03d"

        # Save state histories
        self.observation_history_dict = defaultdict(list)
        self.surrounding_dict = defaultdict(None)

        reset_config = self.meta_scenario_reset_dict[self.meta_scenario_id]
        self.obstacle_lanelet_id_dict = problem_dict["obstacle_lanelet_id_dict"]
        self.road_edge = {
            "left_road_edge_lanelet_id_dict": reset_config["left_road_edge_lanelet_id_dict"],
            "left_road_edge_dict": reset_config["left_road_edge_dict"],
            "right_road_edge_lanelet_id_dict": reset_config["right_road_edge_lanelet_id_dict"],
            "right_road_edge_dict": reset_config["right_road_edge_dict"],
            "boundary_collision_object": reset_config["boundary_collision_object"],
        }
        self.connected_lanelet_dict = reset_config["connected_lanelet_dict"]

        full_reset = False
        cc_template = self.cache_collision_checker_templates.get(self.benchmark_id, None)
        if cc_template is None:
            cc_template = create_collision_checker(self.scenario)
            if not full_reset:
                self.cache_collision_checker_templates[self.benchmark_id] = cc_template

        self.collision_checker = clone_collision_checker(cc_template)

    def _set_ego_vehicle(self) -> None:
        """
        Create ego vehicle and initialize its status with selected initial planning state and time step size.
        :return: None
        """
        if not self.ego_vehicle:
            self.ego_vehicle = Vehicle.create_vehicle(self.vehicle_params)
        self.ego_vehicle.reset(self.planning_problem.initial_state, self.scenario.dt)

    def _set_goal(self) -> None:
        """
        Set ego vehicle and initialize its status.
        :return: None
        """
        self.goal = self.cache_goal_obs.get(self.benchmark_id, None)
        if self.goal is None:
            self.goal = GoalObservation(self.scenario, self.planning_problem)
            self.cache_goal_obs[self.benchmark_id] = self.goal

            # Compute initial distance to goal for normalization if required
            if self.reward_type == "dense_reward" or "hybrid_reward":
                (
                    distance_goal_long,
                    distance_goal_lat,
                ) = self.goal.get_long_lat_distance_to_goal(
                    self.ego_vehicle.state.position
                )
                self.initial_goal_dist = np.sqrt(
                    distance_goal_long ** 2 + distance_goal_lat ** 2
                )
                # Prevent cases where the ego vehicle starts in the goal region
                if self.initial_goal_dist < 1.0:
                    self.initial_goal_dist = 1.0

    # =================================================================================================================
    #
    #                                    step functions
    #
    # =================================================================================================================

    def _get_observation(self) -> Union[np.ndarray, Dict]:
        """
        Get new observation for each time step.
        observation_history_dict contains hidden states over all time steps, which help monitor scenarios or optimize configurations
        observation_dict contains observations at current time step, which serve as inputs to training and reward function
        :return: new observation vector or dict, depending on flattened or not
        """
        # Update ego vehicle state and lanelet information
        ego_vehicle_state = self.ego_vehicle.state
        lanelet_polygons = self.cache_lanelet_polygons_accel_struct.get(self.meta_scenario_id, None)
        lanelet_polygons_sg = self.cache_lanelet_polygons_sg_accel_struct.get(self.meta_scenario_id, None)
        if lanelet_polygons is None:
            lanelet_polygons = [(lanelet.lanelet_id, lanelet.convert_to_polygon())
                                for lanelet in self.scenario.lanelet_network.lanelets]
            lanelet_polygons_sg = pycrcc.ShapeGroup()
            for l_id, poly in lanelet_polygons:
                lanelet_polygons_sg.add_shape(create_collision_object(poly))
            self.cache_lanelet_polygons_sg_accel_struct[self.meta_scenario_id] = lanelet_polygons_sg
            self.cache_lanelet_polygons_accel_struct[self.meta_scenario_id] = lanelet_polygons

        ego_vehicle_lanelet_ids = sorted_lanelets_by_state_realtime(
            self.scenario, ego_vehicle_state, lanelet_polygons, lanelet_polygons_sg
        )
        if len(ego_vehicle_lanelet_ids) == 0:
            ego_vehicle_lanelet_id = self.observation_history_dict["ego_vehicle_lanelet_id"][-1]
        else:
            ego_vehicle_lanelet_id = ego_vehicle_lanelet_ids[0]
        self.observation_history_dict["ego_vehicle_lanelet_id"].append(ego_vehicle_lanelet_id)
        ego_vehicle_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(ego_vehicle_lanelet_id)

        # Check terminating conditions such as goal reaching, collision, off-road, ...
        if self.current_step == 0:
            is_goal_reached = False
            is_off_road = False
            is_collision = False
            is_time_out = False
            is_friction_violation = False
        else:
            is_goal_reached = self._check_goal_reached(ego_vehicle_state)
            is_off_road = self.check_off_road(self.ego_vehicle)
            is_collision = (self.observation_history_dict["is_collision"][self.current_step - 1] or self._check_collision())
            is_time_out = self.current_step >= self.ep_length and not is_goal_reached
            is_friction_violation = self.ego_vehicle._check_friction_violation(self.ego_vehicle.params,
                                                                               self.ego_vehicle.previous_state)

        self.observation_history_dict["is_goal_reached"].append(is_goal_reached)
        self.observation_history_dict["is_off_road"].append(is_off_road)
        self.observation_history_dict["is_collision"].append(is_collision)
        self.observation_history_dict["is_time_out"].append(is_time_out)
        self.observation_history_dict["is_friction_violation"].append(is_friction_violation)

        # Get observations in either case
        observation_dict = OrderedDict()

        # Check if terminate episode
        if ((is_goal_reached and self.terminate_on_goal_reached)
                or (is_off_road and self.terminate_on_off_road)
                or (is_collision and self.terminate_on_collision)
                or (is_time_out and self.terminate_on_time_out)
                or (is_friction_violation and self.terminate_on_friction_violation)):
            self.terminated = True

            # Aggregate observations
            for key in self.observation_space_dict.keys():
                if key is "is_goal_reached":
                    observation_dict[key] = np.array([is_goal_reached])
                elif key is "is_off_road":
                    observation_dict[key] = np.array([is_off_road])
                elif key is "is_collision":
                    observation_dict[key] = np.array([is_collision])
                elif key is "is_time_out":
                    observation_dict[key] = np.array([is_time_out])
                elif key is "is_friction_violation":
                    observation_dict[key] = np.array([is_friction_violation])
                else:
                    # Borrow values from previous time step
                    observation_dict[key] = self.observation_dict[key]
        else:
            # If ego lanelet is changed, update coordinate system, nearby lanelet, lane marker and road edge
            is_ego_lanelet_changed = len(self.observation_history_dict["ego_vehicle_lanelet_id"]) == 1 or (
                    self.observation_history_dict["ego_vehicle_lanelet_id"][-1] != self.observation_history_dict["ego_vehicle_lanelet_id"][-2]
                    and self.observation_history_dict["ego_vehicle_lanelet_id"][-1] != -1)
            if is_ego_lanelet_changed:
                self.local_curvi_cosy, self.local_merged_lanelet = get_local_curvi_cosy(
                    self.scenario,
                    self.observation_history_dict["ego_vehicle_lanelet_id"][-1],
                    self.cache_scenario_ref_path_dict[self.meta_scenario_id],
                    self.max_lane_merge_range,
                )

                left_marker_line, right_marker_line = get_lane_marker(ego_vehicle_lanelet)
                self.surrounding_dict["left_marker_line"] = left_marker_line
                self.surrounding_dict["right_marker_line"] = right_marker_line

                current_left_road_edge, current_right_road_edge = self._get_road_edge(ego_vehicle_lanelet_id)
                self.surrounding_dict["current_left_road_edge"] = current_left_road_edge
                self.surrounding_dict["current_right_road_edge"] = current_right_road_edge

                lanelet_dict, all_lanelets_set = get_nearby_lanelet_id(self.connected_lanelet_dict, ego_vehicle_lanelet)
                self.surrounding_dict["lanelet_dict"] = lanelet_dict
                self.surrounding_dict["all_lanelets_set"] = all_lanelets_set

            # Aggregate observations
            # Ego-related
            if self.observe_v_ego:
                observation_dict["v_ego"] = np.array([ego_vehicle_state.velocity])
            if self.observe_a_ego:
                observation_dict["a_ego"] = np.array([ego_vehicle_state.acceleration])
            if self.observe_steering_angle:
                observation_dict["steering_angle"] = np.array([ego_vehicle_state.steering_angle])
            if self.observe_heading:
                heading = get_lane_relative_heading(ego_vehicle_state, ego_vehicle_lanelet)
                self.observation_history_dict["heading"].append(heading)
                observation_dict["heading"] = np.array([heading])

            if self.observe_global_turn_rate:
                observation_dict["global_turn_rate"] = np.array([ego_vehicle_state.yaw_rate])
            if self.observe_lat_offset:
                lat_offset = get_relative_offset(self.local_curvi_cosy, ego_vehicle_state.position)
                if np.isnan(lat_offset):
                    # note that we (reasonably) assume that the ego vehicle starts inside the ccosy
                    assert (len(self.observation_history_dict["lat_offset"]) > 0), \
                        "Ego vehicle started outside the local coordinate system"
                    lat_offset = self.observation_history_dict["lat_offset"][-1]
                self.observation_history_dict["lat_offset"].append(lat_offset)
                observation_dict["lat_offset"] = np.array([lat_offset])

            if (self.observe_left_marker_distance
                    or self.observe_right_marker_distance
                    or self.observe_left_road_edge_distance
                    or self.observe_right_road_edge_distance):
                (left_marker_distance, right_marker_distance, left_road_edge_distance, right_road_edge_distance, ) = \
                    get_distance_to_marker_and_road_edge(ego_vehicle_state, self.surrounding_dict["left_marker_line"],
                                                         self.surrounding_dict["right_marker_line"],
                                                         self.surrounding_dict["current_left_road_edge"],
                                                         self.surrounding_dict["current_right_road_edge"])
                if self.observe_left_marker_distance:
                    observation_dict["left_marker_distance"] = np.array([left_marker_distance])
                if self.observe_right_marker_distance:
                    observation_dict["right_marker_distance"] = np.array([right_marker_distance])
                if self.observe_left_road_edge_distance:
                    observation_dict["left_road_edge_distance"] = np.array([left_road_edge_distance])
                if self.observe_right_road_edge_distance:
                    observation_dict["right_road_edge_distance"] = np.array([right_road_edge_distance])

            # Goal-related
            if self.observe_distance_goal_long or self.observe_distance_goal_lat:
                (distance_goal_long, distance_goal_lat, ) = \
                    self.goal.get_long_lat_distance_to_goal(ego_vehicle_state.position)
                if np.isnan(distance_goal_long):
                    assert (len(self.observation_history_dict["distance_goal_long"]) > 0), \
                        "Ego vehicle started outside the global coordinate system"
                    distance_goal_long = self.observation_history_dict["distance_goal_long"][-1]
                    distance_goal_lat = self.observation_history_dict["distance_goal_lat"][-1]
                if self.observe_distance_goal_long:
                    self.observation_history_dict["distance_goal_long"].append(distance_goal_long)
                    observation_dict["distance_goal_long"] = np.array([distance_goal_long])
                    if self.current_step == 0:
                        distance_goal_long_advance = 0.0
                    else:
                        distance_goal_long_advance = (self.observation_history_dict["distance_goal_long"][-2]
                                                      - self.observation_history_dict["distance_goal_long"][-1])
                    self.observation_history_dict["distance_goal_long_advance"].append(distance_goal_long_advance)
                    observation_dict["distance_goal_long_advance"] = np.array([distance_goal_long_advance])
                if self.observe_distance_goal_lat:
                    self.observation_history_dict["distance_goal_lat"].append(distance_goal_lat)
                    observation_dict["distance_goal_lat"] = np.array([distance_goal_lat])
                    if self.current_step == 0:
                        distance_goal_lat_advance = 0.0
                    else:
                        distance_goal_lat_advance = (self.observation_history_dict["distance_goal_lat"][-2]
                                                     - self.observation_history_dict["distance_goal_lat"][-1])
                    self.observation_history_dict["distance_goal_lat_advance"].append(distance_goal_lat_advance)
                    observation_dict["distance_goal_lat_advance"] = np.array([distance_goal_lat_advance])

            if self.observe_distance_goal_time:
                distance_goal_time = self.goal.get_goal_time_distance(ego_vehicle_state.time_step)
                observation_dict["distance_goal_time"] = np.array([distance_goal_time])
            if self.observe_distance_goal_orientation:
                distance_goal_orientation = self.goal.get_goal_orientation_distance(
                    ego_vehicle_state.orientation)
                observation_dict["distance_goal_orientation"] = np.array([distance_goal_orientation])
            if self.observe_distance_goal_velocity:
                distance_goal_velocity = self.goal.get_goal_velocity_distance(ego_vehicle_state.velocity)
                observation_dict["distance_goal_velocity"] = np.array([distance_goal_velocity])
            if self.observe_distance_goal_long_lane:
                distance_goal_long_lane = self.goal.get_long_distance_until_lane_change(ego_vehicle_state,
                                                                                        ego_vehicle_lanelet_ids)
                if np.isnan(distance_goal_long_lane):
                    assert (len(self.observation_history_dict["distance_goal_long_lane"]) > 0), \
                        "Ego vehicle started outside the global coordinate system"
                    distance_goal_long_lane = self.observation_history_dict["distance_goal_long_lane"][-1]
                self.observation_history_dict["distance_goal_long_lane"].append(distance_goal_long_lane)
                observation_dict["distance_goal_long_lane"] = np.array([distance_goal_long_lane])

            if self.observe_static_extrapolated_positions:
                sampling_points = self.static_extrapolation_samples
                static_lat_offset, static_pos = get_relative_future_goal_offsets(self.goal, ego_vehicle_state,
                                                                                 sampling_points, static=True)
                # Fix nans for positions outside the ccosy
                for i in range(len(static_lat_offset)):
                    if np.isnan(static_lat_offset[i]):
                        if len(self.observation_history_dict["extrapolation_static_off"]) > 0:
                            # We can assume that the ego vehicle starts inside the ccosy,
                            # however not that any extrapolated position is inside the ccosy
                            static_lat_offset[i] = self.observation_history_dict["extrapolation_static_off"][-1][i]
                        else:
                            static_lat_offset[i] = 0
                self.observation_history_dict["extrapolation_static_off"].append(np.array(static_lat_offset))
                self.observation_history_dict["extrapolation_static_pos"].append(np.array(static_pos))
                observation_dict["extrapolation_static_off"] = np.array(static_lat_offset)

            if self.observe_dynamic_extrapolated_positions:
                if self.current_step == 0:
                    sampling_points = self.dynamic_extrapolation_samples
                    dynamic_lat_offset, dynamic_pos = get_relative_future_goal_offsets(self.goal, ego_vehicle_state,
                                                                                    sampling_points, static=False)
                else:
                    # add for reactive planner
                    dynamic_pos = self.ego_action.predicted_positions
                    dynamic_lat_offset = [self.goal.get_long_lat_distance_to_goal(p)[1] for p in dynamic_pos]
                                                                            
                # Fix nans for positions outside the ccosy
                for i in range(len(dynamic_lat_offset)):
                    if np.isnan(dynamic_lat_offset[i]):
                        if len(self.observation_history_dict["extrapolation_dynamic_off"]) > 0:
                            # We can assume that the ego vehicle starts inside the ccosy,
                            # however not that any extrapolated position is inside the ccosy
                            dynamic_lat_offset[i] = self.observation_history_dict["extrapolation_dynamic_off"][-1][i]
                        else:
                            dynamic_lat_offset[i] = 0
                self.observation_history_dict["extrapolation_dynamic_off"].append(np.array(dynamic_lat_offset))
                self.observation_history_dict["extrapolation_dynamic_pos"].append(np.array(dynamic_pos))
                observation_dict["extrapolation_dynamic_off"] = np.array(dynamic_lat_offset)

            # Surrounding-related
            # # Lane-based rectangular surrounding observation
            if self.observe_lane_rect_surrounding:
                (
                    rel_vel,
                    rel_pos,
                    detected_obstacle,
                    surrounding_area,
                ) = get_surrounding_obstacles_lane_rect(
                    self.scenario.dynamic_obstacles,
                    self.scenario.static_obstacles,
                    self.obstacle_lanelet_id_dict,
                    self.surrounding_dict["all_lanelets_set"],
                    self.local_curvi_cosy,
                    self.surrounding_dict["lanelet_dict"],
                    ego_vehicle_state,
                    self.current_step,
                    self.dummy_rel_vol,
                    self.dummy_dist,
                    self.lane_rect_sensor_range_length,
                    self.lane_rect_sensor_range_width,
                )
                observation_dict["lane_rect_v_rel"] = np.array(rel_vel)
                observation_dict["lane_rect_p_rel"] = np.array(rel_pos)
                self.observation_history_dict["lane_rect_obs"].append(detected_obstacle)
                self.observation_history_dict["lane_rect_surrounding_area"].append(surrounding_area)

            # # Lane-based circular surrounding observation
            if self.observe_lane_circ_surrounding:
                (
                    rel_vel,
                    rel_pos,
                    detected_obstacle,
                    surrounding_area,
                ) = get_surrounding_obstacles_lane_circ(
                    self.scenario.dynamic_obstacles,
                    self.scenario.static_obstacles,
                    self.obstacle_lanelet_id_dict,
                    self.surrounding_dict["all_lanelets_set"],
                    self.local_curvi_cosy,
                    self.surrounding_dict["lanelet_dict"],
                    ego_vehicle_state,
                    self.current_step,
                    self.dummy_rel_vol,
                    self.dummy_dist,
                    self.lane_circ_sensor_range_radius,
                )
                observation_dict["lane_circ_v_rel"] = np.array(rel_vel)
                observation_dict["lane_circ_p_rel"] = np.array(rel_pos)
                self.observation_history_dict["lane_circ_obs"].append(detected_obstacle)
                self.observation_history_dict["lane_circ_surrounding_area"].append(surrounding_area)

            # # Lidar-based elliptical surrounding observation
            if self.observe_lidar_elli_surrounding:
                if self.current_step > 0:
                    prev_dist = self.observation_dict["lidar_elli_dist"]
                else:
                    prev_dist = np.full(self.lidar_elli_num_beams, self.dummy_dist)
                (
                    dist,
                    dist_rate,
                    detection_points,
                    surrounding_beams,
                ) = get_surrounding_obstacles_lidar_elli(
                    self.scenario.dynamic_obstacles,
                    self.scenario.static_obstacles,
                    ego_vehicle_state,
                    self.current_step,
                    self.dummy_dist_rate,
                    self.dummy_dist,
                    prev_dist,
                    self.lidar_elli_num_beams,
                    self.lidar_elli_sensor_range_semi_major_axis,
                    self.lidar_elli_sensor_range_semi_minor_axis,
                )
                observation_dict["lidar_elli_dist_rate"] = np.array(dist_rate)
                observation_dict["lidar_elli_dist"] = np.array(dist)
                self.observation_history_dict["lidar_elli_obs"].append(detection_points)
                self.observation_history_dict["lidar_elli_obs_detection"].append(
                    [1 if x != self.dummy_dist else 0 for x in dist]
                )
                self.observation_history_dict["lidar_elli_surrounding_beams"].append(
                    surrounding_beams
                )

            # Extrapolated position observations
            if self.observe_static_extrapolated_positions:
                sampling_points = self.static_extrapolation_samples
                static_lat_offset, static_pos = get_relative_future_goal_offsets(
                    self.goal, ego_vehicle_state, sampling_points, static=True
                )
                # Fix nans for positions outside the ccosy
                for i in range(len(static_lat_offset)):
                    if np.isnan(static_lat_offset[i]):
                        if len(self.observation_history_dict["extrapolation_static_off"]) > 0:
                            # We can assume that the ego vehicle starts inside the ccosy,
                            # however not that any extrapolated position is inside the ccosy
                            static_lat_offset[i] = self.observation_history_dict["extrapolation_static_off"][-1][i]
                        else:
                            static_lat_offset[i] = 0
                self.observation_history_dict["extrapolation_static_off"].append(np.array(static_lat_offset))
                self.observation_history_dict["extrapolation_static_pos"].append(np.array(static_pos))
                observation_dict["extrapolation_static_off"] = np.array(static_lat_offset)

            # Termination-related
            if self.observe_remaining_steps:
                observation_dict["remaining_steps"] = np.array([self.ep_length - self.current_step])
            if self.observe_is_goal_reached:
                observation_dict["is_goal_reached"] = np.array([is_goal_reached])

            if self.observe_is_off_road:
                observation_dict["is_off_road"] = np.array([is_off_road])

            if self.observe_is_collision:
                observation_dict["is_collision"] = np.array([is_collision])

            if self.observe_is_time_out:
                observation_dict["is_time_out"] = np.array([is_time_out])

            if self.observe_is_friction_violation:
                observation_dict["is_friction_violation"] = np.array([is_friction_violation])

        # Flatten observation if required
        self.observation_dict = observation_dict
        if self.flatten_observation:
            observation_vector = np.zeros(self.observation_space_size)
            index = 0
            for k in self.observation_dict.keys():
                size = np.prod(self.observation_dict[k].shape)
                observation_vector[index: index + size] = self.observation_dict[k].flat
                index += size
            return observation_vector
        else:
            return self.observation_dict

    def check_off_road(self, ego_vehicle: Vehicle) -> bool:
        """
        Check if the ego vehicle is off road.
        :param ego_vehicle: The ego vehicle
        :return: if ego vehicle is off-road
        """
        if self.strict_off_road_check == True:
            collision_ego_vehicle = ego_vehicle.collision_object
            if collision_ego_vehicle.collide(
                self.road_edge["boundary_collision_object"]
            ):
                return True
            return False
        else:
            # Check if a circle area around ego vehicle center collide with road boundary
            collision_ego_vehicle_center = pycrcc.Circle(
                self.non_strict_check_circle_radius, ego_vehicle.state.position[0], ego_vehicle.state.position[1]
            )
            if collision_ego_vehicle_center.collide(
                self.road_edge["boundary_collision_object"]
            ):
                return True
            return False

    def _check_collision(self) -> bool:
        """
        Check if ego vehicle collide with other obstacles.
        :return: True if collision happens, else False
        """
        collision_ego_vehicle = self.ego_vehicle.collision_object
        return self.collision_checker.collide(collision_ego_vehicle)

    def _check_goal_reached(self, ego_vehicle_state: State) -> bool:
        """
        Check if goal is reached by ego vehicle.
        :param ego_vehicle_state: state of ego vehicle.
        :return: True if goal is reached
        """
        if self.relax_is_goal_reached:
            for state in self.planning_problem.goal.state_list:
                if state.position.contains_point(ego_vehicle_state.position):
                    return True
            return False
        else:
            return self.planning_problem.goal.is_reached(ego_vehicle_state)

    def _get_road_edge(self, ego_vehicle_lanelet_id: int) -> Tuple[LineString, LineString]:
        """
        Get the left and right road edge of ego vehicle lanelet.
        :param ego_vehicle_lanelet_id: id of ego vehicle lanelet
        :return: left and right road edge
        """
        left_most_lanelet_id = self.road_edge["left_road_edge_lanelet_id_dict"][ego_vehicle_lanelet_id]
        right_most_lanelet_id = self.road_edge["right_road_edge_lanelet_id_dict"][ego_vehicle_lanelet_id]
        left_road_edge = self.road_edge["left_road_edge_dict"][left_most_lanelet_id]
        right_road_edge = self.road_edge["right_road_edge_dict"][right_most_lanelet_id]
        return left_road_edge, right_road_edge

    def _get_reward(self) -> float:
        """
        Calculate reward based on observation.
        Note that each self.observation_dict[key] is a numpy array.
        :return: reward
        """
        reward = 0.0

        # Reach goal
        if self.observe_is_goal_reached and self.observation_dict["is_goal_reached"][0]:
            reward += self.reward_goal_reached
            # LOGGER.debug("GOAL REACHED!")
        # Collision
        if self.observe_is_collision and self.observation_dict["is_collision"][0]:
            reward += self.reward_collision
        # Off-road
        if self.observe_is_off_road and self.observation_dict["is_off_road"][0]:
            reward += self.reward_off_road
        # Friction violation
        if (
            self.observe_is_friction_violation
            and self.observation_dict["is_friction_violation"][0]
        ):
            reward += self.reward_friction_violation
        # Exceed maximum episode length
        if self.observe_is_time_out and self.observation_dict["is_time_out"][0]:
            reward += self.reward_time_out
        # Penalize reverse driving
        reward += self.reward_reverse_driving * (self.ego_vehicle.state_list[-1].velocity < 0.0)

        if self.reward_type == "sparse_reward":
            return reward

        if self.reward_type == "hybrid_reward":
            # Distance advancement
            if self.observe_distance_goal_long and self.observe_distance_goal_lat:
                long_advance = self.observation_dict["distance_goal_long_advance"][0]
                lat_advance = self.observation_dict["distance_goal_lat_advance"][0]
                reward += self.reward_get_close_coefficient * (
                    lat_advance * 5 + long_advance
                )
            # Deviation from lane center
            if self.observe_lat_offset:
                reward += (
                    -self.reward_stay_in_road_center
                    * self.observation_dict["lat_offset"][0] ** 2
                )
            # Degree of violation of friction constraint
            if (
                self.observe_a_ego
                and self.observe_v_ego
                and self.observe_steering_angle
            ):
                a_ego = self.observation_dict["a_ego"][0]
                v_ego = self.observation_dict["v_ego"][0]
                steering_ego = self.observation_dict["steering_angle"][0]
                l_wb = self.ego_vehicle.params.a + self.ego_vehicle.params.b
                a_max = self.ego_vehicle.params.longitudinal.a_max
                reward += self.reward_friction * (
                    a_max
                    - (a_ego ** 2 + (v_ego ** 2 * np.tan(steering_ego) / l_wb) ** 2)
                    ** 0.5
                )

            return reward

        if self.reward_type == "dense_reward":
            # Calculate normalized distance to obstables as a positive reward
            # Lane-based
            rel_pos = []
            if self.observe_lane_rect_surrounding:
                rel_pos += self.observation_dict["lane_rect_p_rel"].tolist()
            if self.observe_lane_circ_surrounding:
                rel_pos += self.observation_dict["lane_circ_p_rel"].tolist()

            if len(rel_pos) == 0:
                r_obs_lane = 0.0
            else:
                # Minus 5 meters from each of the lane-based relative positions,
                # to get approximiately minimal distances between vehicles,
                # intead of exact distances between centers of vehicles
                r_obs_lane = (self.reward_obs_distance_coefficient * (np.sum(rel_pos) - 5.0 * len(rel_pos)) / (
                            self.dummy_dist * len(rel_pos)))

            # Lidar-based
            dist = []
            if self.observe_lidar_elli_surrounding:
                dist += self.observation_dict["lidar_elli_dist"].tolist()

            if len(dist) == 0:
                r_obs_lidar = 0.0
            else:
                r_obs_lidar = (self.reward_obs_distance_coefficient * np.sum(dist) / (self.dummy_dist * len(dist)))

            # Calculate normalized distance to goal as a negative reward
            dist_goal = np.sqrt(self.observation_dict["distance_goal_long"][0] ** 2
                                + self.observation_dict["distance_goal_lat"][0] ** 2)
            r_goal = (
                    -self.reward_goal_distance_coefficient
                    * dist_goal
                    / self.initial_goal_dist
            )

            return r_obs_lane + r_obs_lidar + r_goal

        raise NotImplementedError(f"Reward type {self.reward_type} not implemented.")


def remove_useless_participants(scenario):
    """
    removes useless objects (bicycle and pedestrain)
    """
    for obstacle in scenario.dynamic_obstacles:
        if obstacle.obstacle_type in [ObstacleType.PEDESTRIAN, ObstacleType.BICYCLE]:
            scenario.remove_obstacle(obstacle)
    
    return scenario

if __name__ == "__main__":

    env = gym.make("commonroad-v0")
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observations, rewards, done, info = env.step(action)