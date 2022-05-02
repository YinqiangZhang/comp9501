#! /usr/bin/env python

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

__desc__ = """
Extracts planning problems from a big recording by removing a dynamic vehicle and replacing its first and last state
with ego vehicle start and goal position
"""

from math import floor, radians
from time import time
import logging
import numpy as np
from numpy import average as avg
from typing import List
import random

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import (
    ObstacleType,
    DynamicObstacle,
    StaticObstacle,
    ObstacleRole,
)
from commonroad.geometry.shape import Rectangle, Circle
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.common.file_reader import CommonRoadFileReader

from .dyn_obs_to_planning_problem import planning_problem_from_dynamic_obstacle
from .common import make_valid_orientation_pruned

class_to_obstacleType = {
    "car": ObstacleType.CAR,
    "truck_bus": ObstacleType.TRUCK,
    "pedestrian": ObstacleType.PEDESTRIAN,
    "bicycle": ObstacleType.BICYCLE,
}

LOGGER = logging.getLogger(__name__)

locationId_to_name = {
    1: "Bendplatz",
    2: "frankenberg",
    3: "heckstrasse",
    4: "aseag",
}

AACHEN_THREE_LETTER_CITY_CODE = "AAH"
locationId_to_scene = {
    1: "{}-1".format(AACHEN_THREE_LETTER_CITY_CODE),
    2: "{}-2".format(AACHEN_THREE_LETTER_CITY_CODE),
    3: "{}-3".format(AACHEN_THREE_LETTER_CITY_CODE),
    4: "{}-4".format(AACHEN_THREE_LETTER_CITY_CODE),
}

# has to be loaded before usage
# by calling "load_lanelet_networks"
locationId_to_lanelet_network = {}


def load_lanelet_networks(lanelet_d):
    """
    Load all lanelet networks from the given path into the static variable of the file
    :param lanelet_d: Path to lanelet network
    :return:
    """
    # load all lanelet networks in cache
    for i, location_name in locationId_to_name.items():
        LOGGER.info(
            "Loading lanelet network {} from {}".format(location_name, lanelet_d)
        )
        locationId_to_lanelet_network[i] = CommonRoadFileReader(
            lanelet_d.joinpath("{}.xml".format(location_name))
        ).open_lanelet_network()
    # also return the *global* dictionary in case s.o. wants to further manipulate it
    return locationId_to_lanelet_network


def meta_scenarios(
    lanelet_network_dict=locationId_to_lanelet_network,
) -> List[Scenario]:
    """
    Generates meta scenarios for all loaded lanelet networks
    """
    m_s = []
    for i, loc_network in locationId_to_lanelet_network.items():
        m_s.append(meta_scenario_from_recording(i, -1, 30))
    return m_s


def meta_scenario_from_recording(location_id: int, recording_id: int, frame_rate=30):
    """
    Generate a meta scenario from the recording meta information
    :param location_id: ID of the location in inD dataset
    :param recording_id: ID of the recording in inD dataset
    :param frame_rate: of the recording
    :return: Meta scenario, suiting the recording
    """
    # compute time step from frame rate
    # frame-rate [hz] = frames/second -> time-step = seconds/frame = 1/frame-rate
    scenario_dt = 1 / frame_rate
    location_id = location_id

    # id should not be 0 indexed, increase by one to prevent recording id = 0
    scenario = Scenario(
        scenario_dt,
        "DEU_{}_{:d}_T-1".format(locationId_to_scene[location_id], recording_id + 1),
    )

    lanelet_network = locationId_to_lanelet_network[location_id]
    scenario.add_objects(lanelet_network)
    return scenario


def interval_overlap(a_start, a_end, b_start, b_end):
    """
    Return amount of overlap of [a_start, a_end] and [b_start, b_end]
    """
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def state_from_track_tuple(
    frame: int,
    center: np.ndarray,
    heading: float,
    lat_velocity: float,
    lon_velocity: float,
    lat_acceleration: float,
    lon_acceleration: float,
):
    """
    Convert a tuple of informations (mostly raw from the inD dataset) to a State object
    Description of parameters mostly copied from https://www.ind-dataset.com/format
    The name of the parameter corresponds to the name of the column in the corresponding csv
    :param frame: The frame for which the information are given. [-]
    :param center: The [x,y] position of the object's centroid in the local coordinate system. [m]
    :param heading: The heading in the local coordinate system. [deg]
    :param lat_velocity: The lateral velocity. [m/s]
    :param lon_velocity: The longitudinal velocity. [m/s]
    :param lat_acceleration: The lateral acceleration. [m/s²]
    :param lon_acceleration: The longitudinal acceleration. [m/s²]
    :return:
    """
    return State(
        time_step=int(frame),
        position=center,
        orientation=make_valid_orientation_pruned(radians(heading)),
        velocity=lon_velocity,
        velocity_y=lat_velocity,
        acceleration=lon_acceleration,
    )


def offset_obstacle_from_track(
    track: dict,
    track_meta: dict,
    obstacle_id: int,
    offset: int,
    limit: int,
    detect_static_vehicles=False,
):
    """
    Converts a single track from a inD dataset recording to a CommonRoad obstacle
    Assumes that the cutting will leave at least 2 frames remaining
    and takes into account whether the traffic participant is parking and which type it has
    :param limit: last frame of the recording to be included
    :param offset: first frame of the recording to be included
    :param track: track values of an recorded obstacle
    :param track_meta: track meta values of an recorded obstacle
    :param obstacle_id: ID of obstacle
    :return: A new Obstacle with unique obstacle ID, Static or Dynamic corresponding to movement in the scenario
    """

    # cut the track to appropriate length
    _first_frame = max(offset - track_meta["initialFrame"], 0)
    _last_frame = min(limit - track_meta["initialFrame"] + 1, len(track["frame"]))

    assert (
        _last_frame - _first_frame >= 2
    ), "Overlap is only {}-{}={}, not 2 as required ({})".format(
        _last_frame, _first_frame, _last_frame - _first_frame, track_meta
    )

    track = {
        key: value[_first_frame:_last_frame]
        for key, value in track.items()
        if type(value) is np.ndarray
    }

    obs_id = obstacle_id  # track_meta["trackId"] DOES NOT work (ids conflict with lanelet network)
    obs_type = class_to_obstacleType[track_meta["class"].lower()]
    # if its VRU (pedestrian or cyclist) the rectangle size is 0 (likely undesireable)
    if obs_type == ObstacleType.PEDESTRIAN:
        # as surveyed by the author (approximation, harmonized with https://commonroad.in.tum.de/static/scenario_xml/2018b/ZAM_Intersect-1_2_S-1.xml)
        obs_shape = Circle(0.35)
    elif obs_type == ObstacleType.BICYCLE:
        # as surveyed by the author (handle_width x bicycle length), harmonized with https://commonroad.in.tum.de/static/scenario_xml/2018b/DEU_Muc-30_1_S-1.xml
        obs_shape = Rectangle(width=0.6, length=1.8)
    else:
        obs_shape = Rectangle(width=track_meta["width"], length=track_meta["length"])

    # determine if vehicle is parked
    min_x = track["xCenter"].min()
    max_x = track["xCenter"].max()
    min_y = track["yCenter"].min()
    max_y = track["yCenter"].max()
    # arbitrary 1 meter threshold: if moved at least one meter, is not a parked vehicle
    # also note that if it disappears before the recording ends or appears after it begins, it is not parked
    if (
        detect_static_vehicles
        and obs_type != ObstacleType.PEDESTRIAN
        # vehicle moved less than one meter total during recording
        and pow(max_x - min_x, 2) + pow(max_y - min_y, 2) < 1
        # vehicle is in recording at least full recording duration
        and track_meta["initialFrame"] <= offset
        and track_meta["finalFrame"] >= limit
    ):
        obs_type = ObstacleType.PARKED_VEHICLE
        obs_initial_state = state_from_track_tuple(
            0,
            np.array([avg(track["xCenter"]), avg(track["yCenter"])]),
            avg(track["heading"]),
            0,
            0,
            0,
            0,
        )
        return StaticObstacle(obs_id, obs_type, obs_shape, obs_initial_state)

    track_tuples = zip(
        track["frame"] - offset,
        track["center"],
        track["heading"],
        track["latVelocity"],
        track["lonVelocity"],
        track["latAcceleration"],
        track["lonAcceleration"],
    )
    obs_initial_state = state_from_track_tuple(*next(track_tuples))
    obs_traj_states = [
        state_from_track_tuple(*track_tuple) for track_tuple in track_tuples
    ]
    obs_trajectory = Trajectory(obs_traj_states[0].time_step, obs_traj_states[0:])
    obs_trajectory_prediction = TrajectoryPrediction(obs_trajectory, obs_shape)

    return DynamicObstacle(
        obs_id, obs_type, obs_shape, obs_initial_state, obs_trajectory_prediction
    )


def planning_problems_from_recording(
    tracks,
    tracksMeta,
    recordingMeta,
    amount: int,
    min_length=0,
    max_length=-1,
    detect_static_vehicles=False,
    include_obstacles=True,
):
    """
    Takes a recording and cuts it into many smaller scenarios.
    Each smaller scenario is created by choosing one car C from the bigger scenario,
    setting C as the ego vehicle
    and extracting the remaining vehicles such that only the frames remain in which they
    coexist with C
    :param tracks:
    :param tracksMeta:
    :param recordingMeta:
    :param amount: number of smaller scenarios to extract
    :param min_length: Minimum length of the track to convert
    :param max_length: Maximum length of the track to convert. Longer tracks will be pruned to max_length
    :return: Generates new scenarios contiuosly as a generator:
         yields resulting scenario and problem set
    """
    # Generate a meta scenario
    meta_scenario = meta_scenario_from_recording(
        recordingMeta["locationId"],
        recordingMeta["recordingId"],
        recordingMeta["frameRate"],
    )

    country_code, city_code, recording_id, pred = meta_scenario.benchmark_id.split("_")
    success_count, track_id = 0, 0
    num_tracks = len(tracks)
    while success_count < amount and track_id < num_tracks:
        # choose a vehicle from all obstacles
        LOGGER.info(
            "Starting conversion from csv to CR for ego vehicle {} ({}/{})".format(
                track_id, success_count, amount
            )
        )
        _begin = time()

        # Quick fix: to prevent conflicts (recording id 1 scenario 11, recording id 11 scenario 1)
        # insert a 0 between recording id and scenario (max recording id << 100)
        benchmark_id = "{}_{}_{}00{}_T-1".format(
            country_code, city_code, recording_id, track_id
        )
        try:
            new_scenario, new_problem_set = planning_problem_from_track(
                track_id,
                tracks,
                tracksMeta,
                meta_scenario,
                benchmark_id,
                min_length=min_length,
                max_length=max_length,
                detect_static_vehicles=detect_static_vehicles,
                include_obstacles=include_obstacles,
            )
            LOGGER.info("Conversion took {} s".format(time() - _begin))
            yield new_scenario, new_problem_set
            success_count += 1
        except AssertionError as a:
            # Note that a number of things may go wrong at conversion time, so simply an exception is thrown
            # at any level and intercepted here
            LOGGER.info("Conversion stopped: {}".format(a))
        finally:
            track_id += 1


def planning_problem_from_track(
    track_id: int,
    tracks,
    tracksMeta,
    meta_scenario: Scenario,
    benchmark_id: str,
    min_length=0,
    max_length=-1,
    detect_static_vehicles=False,
    include_obstacles=True,
):
    """
    Extract a singleton planning problem from the bigger recording.
    The given obstacle will be the ego vehicle of the resulting planning problem
    :param track_id: id of the ego vehicle of the new planning problem
    :param tracks:
    :param tracksMeta:
    :param meta_scenario:
    :param benchmark_id: benchmark id string for the resulting scenario
    :param min_length: Minimum length of the track to convert
    :param max_length: Maximum length of the track to convert. Longer tracks will be pruned to max_length
    :return: scenario and problem set of new planning problem
    """
    ego_track, ego_track_meta = tracks[track_id], tracksMeta[track_id]
    init_time_step, final_time_step = (
        ego_track_meta["initialFrame"],
        ego_track_meta["finalFrame"],
    )
    track_length = final_time_step - init_time_step

    assert (
        track_length >= min_length
    ), "Provided track is too short: {} < {} frames".format(track_length, min_length)

    if max_length >= 0 and track_length > max_length:
        # the track is too long, choose a random window over the existance of the vehicle
        # the window will have maximum length
        init_time_step = random.randrange(init_time_step, final_time_step - max_length)
        final_time_step = init_time_step + max_length
        track_length = max_length
        LOGGER.info(
            "Track [{}, {}] too long (> {}), pruning to [{}, {}]".format(
                ego_track_meta["initialFrame"],
                ego_track_meta["finalFrame"],
                max_length,
                init_time_step,
                final_time_step,
            )
        )

    final_time_step += 50  # TODO quick fix to account for relaxed time_step interval

    # extract all vehicles that live in the same time line
    ego_vehicle = offset_obstacle_from_track(
        ego_track,
        ego_track_meta,
        0,
        init_time_step,
        final_time_step,
        detect_static_vehicles,
    )

    # assert that we are given a track of a dynamic car
    assert (
        ego_vehicle.obstacle_type == ObstacleType.CAR
    ), "Vehicle was of type {} != CAR".format(ego_vehicle.obstacle_type)

    new_scenario = Scenario(dt=meta_scenario.dt, benchmark_id=benchmark_id)
    new_scenario.add_objects(meta_scenario.lanelet_network)

    # determine planning problem first to omit unpromising scenarios quicker
    pp = planning_problem_from_dynamic_obstacle(ego_vehicle, new_scenario)
    new_problem_set = PlanningProblemSet([pp])
    # note that the planning problem is never explicitely added to the scenario
    # so we have to manually make sure its id is exclusive

    if include_obstacles:
        res_dynamic_obstacles = []
        # make sure the planning problem id is never used
        # (the value output by new_scenario.generate_object_id() is exactly the id of the planning problem)
        object_id_base = new_scenario.generate_object_id() + 1
        for i, (t, t_m) in enumerate(zip(tracks, tracksMeta)):
            try:
                assert i != track_id, "Vehicle is ego vehicle"
                # copy vehicle over (including subtracting the time line beginning)
                o = offset_obstacle_from_track(
                    t,
                    t_m,
                    object_id_base + i,
                    init_time_step,
                    final_time_step,
                    detect_static_vehicles,
                )
                # TODO off lanelet vehicles are not handled by crcurv
                # Check if obstacle is always on a lanelet
                on_road = new_scenario.lanelet_network.find_lanelet_by_position(
                    [o.initial_state.position]
                )[0]
                if o.obstacle_role == ObstacleRole.DYNAMIC:
                    for s in o.prediction.trajectory.state_list:
                        if new_scenario.lanelet_network.find_lanelet_by_position(
                            [s.position]
                        )[0]:
                            on_road = True
                            break
                assert on_road, f"Obstacle is off road at all times"

                res_dynamic_obstacles.append(o)
            except AssertionError as a:
                # silent fail for frequent overlap fail
                if not "overlap" in str(a).lower():
                    LOGGER.info(f"Omitting vehicle {i} because of {a}")

        new_scenario.add_objects(res_dynamic_obstacles)

    return new_scenario, new_problem_set
