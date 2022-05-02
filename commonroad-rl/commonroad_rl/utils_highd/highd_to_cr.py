import argparse
import copy
import glob
import os
import time

import numpy as np
import pandas as pd
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.common.solution import VehicleModel, VehicleType
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Tag

from commonroad_rl.utils_highd.map_utils import get_meta_scenario
from commonroad_rl.utils_highd.obstacle_utils import generate_dynamic_obstacle
from commonroad_rl.utils_highd.planning_utils import (
    get_valid_ob_list,
    get_planning_problem,
)

AUTHOR = "Xiao Wang"
AFFILIATION = "Technical University of Munich, Germany"
SOURCE = "The HighWay Drone Dataset (highD)"
TAGS = {Tag.HIGHWAY, Tag.MULTI_LANE, Tag.PARALLEL_LANES, Tag.NO_ONCOMING_TRAFFIC}

# dicts to map location id to location names and obstacle types
location_dict = {
    1: "LocationA",
    2: "LocationB",
    3: "LocationC",
    4: "LocationD",
    5: "LocationE",
    6: "LocationF",
}


def get_file_lists(path):
    """
    :param path: path to a folder with the same file format, e.g. folder_to_file/*_tracks.csv
    :return: list of sorted file names
    """
    listing = glob.glob(path)
    listing.sort()
    return listing


def get_args():
    parser = argparse.ArgumentParser(
        description="Generates CommonRoad scenarios from highD dataset"
    )
    parser.add_argument(
        "--highd_dir", "-i", type=str, default="/data/highD-dataset-v1.0/"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="/data/highD-dataset-v1.0/cr_scenarios"
    )
    parser.add_argument("--num_timesteps", "-nt", type=int, default=1000)
    parser.add_argument("--num_planning_problems", "-np", type=int, default=1)
    parser.add_argument("--min_time_steps", "-min_t", type=int, default=10)
    parser.add_argument("--multiprocessing", "-mpi", action="store_true")
    parser.add_argument(
        "--vehicle_model",
        "-vm",
        type=VehicleModel,
        choices=list(VehicleModel),
        default=VehicleModel.KS,
    )
    parser.add_argument(
        "--vehicle_type",
        "-vt",
        type=VehicleType,
        choices=list(VehicleType),
        default=VehicleType.FORD_ESCORT,
    )

    return parser.parse_args()


def generate_cr_scenarios(
    recording_meta_fn,
    tracks_meta_fn,
    tracks_fn,
    min_time_steps,
    num_timesteps,
    num_planning_problems,
    output_dir,
    vehicle_model,
    vehicle_type,
):
    """
    Generate CommonRoad xml files with given paths to highD recording, tracks_meta, tracks files
    :param recording_meta_fn: path to *_recordingMeta.csv
    :param tracks_meta_fn: path to *_tracksMeta.csv
    :param tracks_fn: path to *_tracks.csv
    :param min_time_steps: vehicles have to appear more than min_time_steps per .xml to be converted
    :param num_timesteps: maximal number of timesteps per .xml file
    :param num_planning_problems: number of planning problems per .xml file
    :param output_dir: path to store generated .xml files
    :return: None
    """

    def enough_time_steps(vehicle_id, tracks_meta_df, min_time_steps):
        vehicle_meta = tracks_meta_df[tracks_meta_df.id == vehicle_id]
        if (
            frame_end - int(vehicle_meta.initialFrame) < min_time_steps
            or int(vehicle_meta.finalFrame) - frame_start < min_time_steps
        ):
            return False
        return True

    # read data frames from three files
    recording_meta_df = pd.read_csv(recording_meta_fn, header=0)
    tracks_meta_df = pd.read_csv(tracks_meta_fn, header=0)
    tracks_df = pd.read_csv(tracks_fn, header=0)

    # generate meta scenario with lanelet network
    meta_scenario_up, _, _ = get_meta_scenario(recording_meta_df, is_up=True)
    meta_scenario_down, _, _ = get_meta_scenario(recording_meta_df, is_up=False)

    # number of scenarios generated from this group of files
    num_scenarios = max(tracks_meta_df.finalFrame) // num_timesteps

    for i in range(num_scenarios):
        # copy meta_scenario with lanelet networks
        scenario_up = copy.deepcopy(meta_scenario_up)
        scenario_down = copy.deepcopy(meta_scenario_down)

        # benchmark id format: COUNTRY_SCENE_CONFIG_PRED
        benchmark_id_up = "DEU_{0}-{1}_{2}_T-1".format(
            location_dict[recording_meta_df.locationId.values[0]],
            int(recording_meta_df.id),
            2 * (i + 1) - 1,
        )
        benchmark_id_down = "DEU_{0}-{1}_{2}_T-1".format(
            location_dict[recording_meta_df.locationId.values[0]],
            int(recording_meta_df.id),
            2 * (i + 1),
        )
        scenario_up.benchmark_id = benchmark_id_up
        scenario_down.benchmark_id = benchmark_id_down

        # convert obstacles appearing between [frame_start, frame_end]
        frame_start = i * num_timesteps + 1
        frame_end = (i + 1) * num_timesteps

        # read tracks appear between [frame_start, frame_end]
        scenario_tracks_df = tracks_df[
            (tracks_df.frame >= frame_start) & (tracks_df.frame <= frame_end)
        ]

        # generate CR obstacles
        for vehicle_id in scenario_tracks_df.id.unique():
            # if appearing time steps < min_time_steps, skip vehicle
            if not enough_time_steps(vehicle_id, tracks_meta_df, min_time_steps):
                continue
            print(
                f"Generating scenario {i + 1}/{num_scenarios}, vehicle id {vehicle_id}",
                end="\r",
            )
            # pickle obstacle with different direction into different scenario
            if (
                tracks_meta_df.loc[tracks_meta_df["id"] == vehicle_id][
                    "drivingDirection"
                ].values[0]
                == 1
            ):
                do_up = generate_dynamic_obstacle(
                    scenario_up,
                    vehicle_id,
                    tracks_meta_df,
                    scenario_tracks_df,
                    frame_start,
                )
                scenario_up.add_objects(do_up)
            elif (
                tracks_meta_df.loc[tracks_meta_df["id"] == vehicle_id][
                    "drivingDirection"
                ].values[0]
                == 2
            ):
                do_down = generate_dynamic_obstacle(
                    scenario_down,
                    vehicle_id,
                    tracks_meta_df,
                    scenario_tracks_df,
                    frame_start,
                )
                scenario_down.add_objects(do_down)

        # generate planning problems and write scenario and planning problems to disk
        planning_problem_set_up = PlanningProblemSet()
        # get valid obstacle which initial_state time_step is 0 and is not off road at the first beginning
        # obstacles should be on road at the first beginning and initial_state time_step is not 0,
        # there will be two bug in commonroad-v0
        valid_ob_list_up, _ = get_valid_ob_list(
            scenario_up,
            is_up_lanelet=True,
            vehicle_model=vehicle_model,
            vehicle_type=vehicle_type,
        )
        # if there is no valid obstacles, let's give up this scenario and planning problem
        if len(valid_ob_list_up) != 0:
            for i in range(num_planning_problems):
                planning_problem_up = get_planning_problem(
                    scenario_up, valid_ob_list_up
                )
                planning_problem_set_up.add_planning_problem(planning_problem_up)

            # set parameter to rotate scenario_up and planning_problem_set_up
            translation = np.array([0.0, 0.0])
            angle = np.pi

            # rotate scenario_up and planning_problem_set_up to be consistent with scenario_down
            scenario_up.translate_rotate(translation, angle)
            planning_problem_set_up.translate_rotate(translation, angle)
            print(scenario_up)
            fw = CommonRoadFileWriter(
                scenario_up, planning_problem_set_up, AUTHOR, AFFILIATION, SOURCE, TAGS
            )
            filename = os.path.join(
                output_dir, "{}.xml".format(scenario_up.benchmark_id)
            )
            fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
            print("Scenario file stored in {}".format(filename))

        planning_problem_set_down = PlanningProblemSet()
        valid_ob_list_down, _ = get_valid_ob_list(
            scenario_down,
            is_up_lanelet=False,
            vehicle_model=vehicle_model,
            vehicle_type=vehicle_type,
        )
        if len(valid_ob_list_down) != 0:
            for i in range(num_planning_problems):
                planning_problem_down = get_planning_problem(
                    scenario_down, valid_ob_list_down
                )
                planning_problem_set_down.add_planning_problem(planning_problem_down)

            print(scenario_down)
            fw = CommonRoadFileWriter(
                scenario_down,
                planning_problem_set_down,
                AUTHOR,
                AFFILIATION,
                SOURCE,
                TAGS,
            )
            filename = os.path.join(
                output_dir, "{}.xml".format(scenario_down.benchmark_id)
            )
            fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
            print("Scenario file stored in {}".format(filename))


def main():
    start_time = time.time()

    # get arguments
    args = get_args()

    # mpi for parallel processing
    if args.multiprocessing:
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None
        if MPI is None:
            highd_path = args.highd_dir
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            highd_path = os.path.join(args.highd_dir, str(rank))
    else:
        highd_path = args.highd_dir

    # make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # generate path to highd data files
    path_tracks = os.path.join(highd_path, "data/*_tracks.csv")
    path_metas = os.path.join(highd_path, "data/*_tracksMeta.csv")
    path_recording = os.path.join(highd_path, "data/*_recordingMeta.csv")

    # get all file names
    listing_tracks = get_file_lists(path_tracks)
    listing_metas = get_file_lists(path_metas)
    listing_recording = get_file_lists(path_recording)

    for index, (recording_meta_fn, tracks_meta_fn, tracks_fn) in enumerate(
        zip(listing_recording, listing_metas, listing_tracks)
    ):
        print("=" * 80)
        print("Processing file {}...".format(tracks_fn), end="\n")
        print("=" * 80)
        generate_cr_scenarios(
            recording_meta_fn,
            tracks_meta_fn,
            tracks_fn,
            args.min_time_steps,
            args.num_timesteps,
            args.num_planning_problems,
            args.output_dir,
            args.vehicle_model,
            args.vehicle_type,
        )

    print("Elapsed time: {} s".format(time.time() - start_time), end="\r")


if __name__ == "__main__":
    main()
