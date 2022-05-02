#! /usr/bin/env python

__author__ = "Niels Mündler"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

__desc__ = """
Converts inD files to Commonroad Format, creating smaller Planning Problems if required
"""

import os
import sys
import random
import logging
import pickle as pkl

from pathlib import Path
from multiprocessing import Pool
from argparse import ArgumentParser

from commonroad.scenario.scenario import Scenario
from commonroad.common.file_writer import (
    CommonRoadFileWriter,
    OverwriteExistingFile,
    Tag,
)

from .utils.tracks_import import read_from_csv
from .utils.csv_to_planning_problem import (
    planning_problems_from_recording,
    load_lanelet_networks,
    meta_scenarios,
)
from .utils.common import get_end_time

# Pickling requires the commonroad_rl package to be installed
try:
    from commonroad_rl.tools.pickle_scenario.preprocessing import (
        generate_obstacle_lanelet_id,
        generate_reset_config,
    )

    PICKLE_SUPPORT = True
except ImportError:

    def generate_obstacle_lanelet_id(scenario: Scenario) -> dict:
        pass

    def generate_reset_config(scenario: Scenario) -> dict:
        pass

    PICKLE_SUPPORT = False

LOGGER = logging.getLogger(__name__)
META_SCENARIO = "meta_scenario"
PROBLEM = "problem"
META_SCENARIOS = {}

# Tagging information for the scenarios to be created
author = "Niels Mündler, Xiao Wang"
affiliation = "Technical University of Munich, Germany"
source = "https://www.ind-dataset.com/"
tags = {Tag.INTERSECTION, Tag.MULTI_LANE, Tag.SPEED_LIMIT, Tag.URBAN}


def get_parser():
    parser = ArgumentParser(description=__desc__)
    parser.add_argument(
        "-i",
        default="../../data",
        help="Path to input directory (converts all files)",
        dest="input",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        default=[0],
        type=int,
        help="Track numbers to be read (-1 = all)",
        dest="numbers",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        default="../../data_cr",
        help="Path to output directory (if not existing, will be created)",
        dest="output",
    )
    parser.add_argument(
        "-v", "--verbose", help="Print verbose", dest="verbose", action="store_true"
    )
    parser.add_argument(
        "-l",
        "--lanelets",
        default=os.path.join(
            os.path.dirname(__file__), "inD_LaneletMaps/convert_tinkered_translated"
        ),
        help="Path to directory containing commonroad formatted lanelets",
        dest="lanelets",
    )
    parser.add_argument(
        "-n",
        "--numproblems",
        default=1,
        type=int,
        help="Number of smaller planning problems to be generated (-1 = all)",
        dest="planning_problems",
    )
    parser.add_argument(
        "--min-length",
        default=125,
        type=int,
        help="Minimum number of timeframes per resulting scenario (default 125 ~ 5 seconds at 25fps)",
        dest="min_length",
    )
    parser.add_argument(
        "--max-length",
        default=250,
        type=int,
        help="Maximum number of timeframes per resulting scenario (default 250 ~ 10 seconds at 25 fps)",
        dest="max_length",
    )
    parser.add_argument(
        "--static-vehicles",
        action="store_true",
        help="Enable detection of static vehicles",
        dest="static_vehicles",
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="Disable all obstacles other than roads",
        dest="no_obstacles",
    )
    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Dump a pickle of meta scenarios and scenarios instead of xml-files",
    )
    parser.add_argument(
        "-s", "--seed", help="Set the seed for randomization", default="0"
    )
    parser.add_argument(
        "--multiprocessing",
        type=int,
        help="Multiprocessing on the given number of cores",
        default=1,
    )
    return parser


def main(argv):
    args = get_parser().parse_args(argv)

    if args.pickle and not PICKLE_SUPPORT:
        raise ImportError(
            "Pickling without package commonroad_rl is not supported. Try installing commonroad_rl first."
        )

    if args.verbose:
        LOGGER.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

    # Check if the given paths exist
    input_d = Path(args.input)
    if not input_d.exists():
        print("Error: path {} does not exist".format(input_d))
        exit(-1)

    lanelet_d = Path(args.lanelets)
    if not lanelet_d.exists():
        print("Error: path {} does not exist".format(lanelet_d))
        exit(-1)
    if not lanelet_d.is_dir():
        print("Error: path {} is not a directory".format(lanelet_d))
        exit(-1)

    # Create output dir if necessary
    output_d = Path(args.output)
    output_d.mkdir(parents=True, exist_ok=True)
    if args.pickle:
        output_d.joinpath(META_SCENARIO).mkdir(exist_ok=True)
        output_d.joinpath(PROBLEM).mkdir(exist_ok=True)

    load_lanelet_networks(lanelet_d)
    # If pickling is enabled, dump these networks already
    # as meta scenarios as they will not change
    if args.pickle:
        store_meta_scenarios(output_d)

    # set the seed for random slices
    random.seed(args.seed)

    # Decide whether to load all tracks or only given ones
    if -1 in args.numbers:
        numbers = list(range(33))
    else:
        numbers = args.numbers
    total = len(numbers)
    # start the conversion process
    # if multiprocessing is enabled, parallelity is on recording file level
    with Pool(processes=args.multiprocessing) as p:
        p.starmap(
            convert_recording_to_scenario,
            [
                (
                    number,
                    input_d,
                    i,
                    total,
                    args.planning_problems,
                    output_d,
                    args.min_length,
                    args.max_length,
                    True,
                    args.pickle,
                    args.static_vehicles,
                    not args.no_obstacles,
                )
                for i, number in enumerate(numbers, start=1)
            ],
        )

    # merge problem_meta_scenario dicts
    if args.pickle:
        merge_problem_meta_scenario_dicts(total, output_d)


def store_meta_scenarios(output_d):
    """
    Stores pickles of the 4 location meta scenarios as meta scenario reset dict pickles
    Also initializes the META_SCENARIOS dictionary
    """
    meta_scenario_reset_dict = {}
    for loc_scenario in meta_scenarios():
        meta_scenario_reset_dict[loc_scenario.benchmark_id] = generate_reset_config(
            loc_scenario
        )
        META_SCENARIOS[loc_scenario.benchmark_id] = loc_scenario
    with output_d.joinpath(META_SCENARIO, "meta_scenario_reset_dict.pickle").open(
        "wb"
    ) as file:
        pkl.dump(meta_scenario_reset_dict, file)


def merge_problem_meta_scenario_dicts(total, output_d):
    """
    Merge the resulting problem meta scenario mappings
    and remove the original subfolders
    """
    LOGGER.info(f"Merging {total} problem meta scenario dictionaries in {output_d}")
    merged_problem_meta_scenario_dict = {}
    for i in range(1, total + 1):
        with output_d.joinpath(
            META_SCENARIO, str(i), "problem_meta_scenario_dict.pickle"
        ).open("rb") as file:
            problem_meta_scenario_dict = pkl.load(file)
        merged_problem_meta_scenario_dict.update(problem_meta_scenario_dict)
    # dump merged dict
    with output_d.joinpath(META_SCENARIO, "problem_meta_scenario_dict.pickle").open(
        "wb"
    ) as file:
        pkl.dump(merged_problem_meta_scenario_dict, file)
    # remove temporary dicts
    for i in range(1, total + 1):
        output_d.joinpath(
            META_SCENARIO, str(i), "problem_meta_scenario_dict.pickle"
        ).unlink()
        output_d.joinpath(META_SCENARIO, str(i)).rmdir()


def convert_recording_to_scenario(
    number: int,
    input_d: Path,
    i: int,
    total: int,
    num_planning_problems: int,
    output_d: Path,
    min_length: int,
    max_length: int,
    overwrite=False,
    pickle=True,
    detect_static_vehicles=False,
    include_obstacles=True,
):
    """
    Convert a given recording to scenarios and write them to disk
    :param number: Number of the recording to load
    :param input_d: Directory of all recordings
    :param num_planning_problems: number of planning problems to extract
    :param output_d: directory to output planning problems to
    :param i: number of current recording in global scope
    :param total: total number of loaded recordings in global scope
    :param pickle: if set, also dumps a pickle of the scenario to be loaded by the commonroad gym env
    :return: Nothing
    """
    # Load a recording
    LOGGER.info("Loading track {} from {} ({}/{})".format(number, input_d, i, total))
    file_names = [
        input_d.joinpath("{:02d}_{}.csv".format(number, a))
        for a in ["tracks", "tracksMeta", "recordingMeta"]
    ]
    tracks, tracks_meta, recording_meta = read_from_csv(*file_names)

    # keep track of which scenario belongs to which meta-scenario for pickling
    problem_meta_scenario_dict = {}

    num_dynamic_obstacles = recording_meta["numVehicles"]
    if num_planning_problems > num_dynamic_obstacles:
        LOGGER.info(
            "Extracting more planning problems than tracked vehicles, reducing to maxmimum possible"
        )
    # Extract min(dynamic_obstacles, n) planning problems or all if the argument was -1
    num_planning_problems = (
        min(num_dynamic_obstacles, num_planning_problems)
        if num_planning_problems != -1
        else num_dynamic_obstacles
    )

    overwrite_policy = (
        OverwriteExistingFile.ALWAYS
        if overwrite
        else OverwriteExistingFile.ASK_USER_INPUT
    )

    j = 0
    # Extracts smaller planning problems from the one recording directly
    # This is done since writing the huge scenario is too inefficient
    for res_scenario, res_problem_set in planning_problems_from_recording(
        tracks,
        tracks_meta,
        recording_meta,
        num_planning_problems,
        min_length=min_length,
        max_length=max_length,
        detect_static_vehicles=detect_static_vehicles,
        include_obstacles=include_obstacles,
    ):
        LOGGER.info(
            "Writing planning problem {} of {} to {} ({}/{})".format(
                j + 1, num_planning_problems, output_d, i, total
            )
        )
        if pickle:
            # Dump a pickle of a generated problem dict
            # the id is generated by "meta_scenario_number _ recording_number 00 planning_problem_numer"
            # so the corresponding meta scenario number (meta_scenario_number _ 0)
            # can be found by only recycling the original id up to the meta scenario number
            # note: this will only work for up to 9 meta scenarios
            meta_scenario_id = res_scenario.benchmark_id[:9] + "_0_T-1"
            problem_meta_scenario_dict[res_scenario.benchmark_id] = META_SCENARIOS[
                meta_scenario_id
            ]

            # then generate the pickle dict to be stored
            # extract dynamic AND static obstacles
            obstacle_list = res_scenario.obstacles
            t_end = get_end_time(res_scenario)
            problem_dict = {
                "obstacle": obstacle_list,
                "end_time": t_end,
                "planning_problem_set": res_problem_set,
                "obstacle_lanelet_id_dict": generate_obstacle_lanelet_id(res_scenario),
            }

            with output_d.joinpath(PROBLEM, f"{res_scenario.benchmark_id}.pickle").open(
                "wb"
            ) as file:
                pkl.dump(problem_dict, file)
        else:
            # Write out the xml of the scenario
            fw = CommonRoadFileWriter(
                res_scenario, res_problem_set, author, affiliation, source, tags
            )
            fn = output_d.joinpath("{}.xml".format(res_scenario.benchmark_id))
            fw.write_to_file(str(fn), overwrite_policy)

        j += 1

    if pickle:
        # finally dump the meta scenario mapping dict
        output_d.joinpath(META_SCENARIO, str(i)).mkdir(exist_ok=True)
        with output_d.joinpath(
            META_SCENARIO, str(i), "problem_meta_scenario_dict.pickle"
        ).open("wb") as file:
            pkl.dump(problem_meta_scenario_dict, file)


if __name__ == "__main__":
    main(sys.argv[1:])
