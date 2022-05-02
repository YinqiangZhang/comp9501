import os
import glob
import copy
import shutil
import pickle
import argparse
import numpy as np

from time import time
from typing import List

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario

from commonroad_rl.tools.pickle_scenario.preprocessing import (
    generate_reset_config,
    generate_obstacle_lanelet_id,
)


def get_end_time(scenario: Scenario):
    return max(
        len(o.prediction.trajectory.state_list) for o in scenario.dynamic_obstacles
    )


def get_meta_scenario_obstacle(scenario: Scenario):
    t_end = get_end_time(scenario)
    meta_scenario = copy.deepcopy(scenario)
    meta_scenario.benchmark_id = "meta_" + scenario.benchmark_id
    meta_scenario.remove_obstacle(meta_scenario.obstacles)
    obstacles = scenario.dynamic_obstacles
    return meta_scenario, obstacles, t_end


def compare_lanelet_network(
    lanelet_network1: LaneletNetwork, lanelet_network2: LaneletNetwork
):
    for l1, l2 in zip(lanelet_network1.lanelets, lanelet_network2.lanelets):
        if (
            not np.array_equal(l1.center_vertices, l2.center_vertices)
            or not np.array_equal(l1.left_vertices, l2.left_vertices)
            or not np.array_equal(l1.right_vertices, l2.right_vertices)
            or l1.predecessor != l2.predecessor
            or l1.successor != l2.successor
        ):
            return False
    return True


def find_meta_scenario_in_list(
    meta_scenario: Scenario, meta_scenario_list: List[Scenario]
):
    for m in meta_scenario_list:
        if compare_lanelet_network(meta_scenario.lanelet_network, m.lanelet_network):
            return True, m
    return False, None


def get_args():
    parser = argparse.ArgumentParser(
        description="Converts CommonRoad xml files to pickle files"
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, default="/data/highD-dataset-v1.0/cr_scenarios"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="/data/highD-dataset-v1.0/pickles"
    )
    parser.add_argument("--multiprocessing", "-mpi", action="store_true")
    parser.add_argument(
        "--shared_dir",
        action="store_true",
        help="Whether processes access one shared directory or have a rank-named directory on their own",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="Remove all obstacles from the scenario",
        dest="no_obstacles",
    )
    parser.add_argument(
        "--lanelet-assignment",
        action="store_true",
        help="Assign lanelets when opening xml file",
        dest="lanelet_assignment",
    )

    parser.add_argument(
        "--duplicate",
        "-d",
        action="store_true",
        help="Duplicate one scenario file to problem_train and problem_test, for overfitting",
    )

    return parser.parse_args()


def pickle_xml_scenarios(
    input_dir: str,
    output_dir: str,
    shared_dir: bool,
    multiprocessing: bool = False,
    no_obstacles: bool = False,
    verbose: bool = False,
    lanelet_assignment: bool = False,
    duplicate: bool = False,
):
    # makedir
    os.makedirs(output_dir, exist_ok=True)

    # mpi for parallel processing
    MPI = None
    if multiprocessing:
        try:
            from mpi4py import MPI
        except ImportError:
            pass

    if not multiprocessing or MPI is None:
        rank = 0
        world_size = 1
        meta_scenario_path = "meta_scenario"
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
        meta_scenario_path = f"meta_scenario_{rank}"
        if not shared_dir:
            input_dir = os.path.join(input_dir, str(rank))

    # Directory initialization
    if shared_dir:
        # slice the files such that no overlapping scenarios are processed
        fn_i = world_size
        fn_start = rank
    else:
        # process the whole directory
        fn_i = 1
        fn_start = 0
    fns = glob.glob(os.path.join(input_dir, "*.xml"))
    if shared_dir:
        fns = fns[rank::fn_i]
    if verbose:
        print("=" * 80)
        print(f"Processing every {fn_i}th entry of {input_dir} starting at {fn_start}")
        print("=" * 80)

    problem_meta_scenario_dict = dict()
    meta_scenario_reset_dict = dict()

    processed_location_list = []

    os.makedirs(os.path.join(output_dir, meta_scenario_path), exist_ok=True)
    if not duplicate:
        os.makedirs(os.path.join(output_dir, "problem"), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_dir, "problem_train"), exist_ok=True)

    _start_time = time()
    for i, fn in enumerate(fns):

        _intermediate_time = time()
        if verbose:
            print(f"{i + 1}/{len(fns)} {fn}")
            print(f"lanelet_assignment={lanelet_assignment}")
        else:
            print(f"{i + 1}/{len(fns)}", end="\r")
        scenario, planning_problem_set = CommonRoadFileReader(fn).open(
            lanelet_assignment=lanelet_assignment
        )
        if no_obstacles:
            scenario.remove_obstacle(scenario.obstacles)
        meta_scenario, obstacles_list, t_end = get_meta_scenario_obstacle(scenario)
        # TODO: remove obstacle_lanelet_id_dict
        obstacle_lanelet_id_dict = generate_obstacle_lanelet_id(scenario)
        problem_dict = {
            "obstacle": obstacles_list,
            "end_time": t_end,
            "planning_problem_set": planning_problem_set,
            "obstacle_lanelet_id_dict": obstacle_lanelet_id_dict,
        }
        is_processed, old_meta_scenario = find_meta_scenario_in_list(
            meta_scenario, processed_location_list
        )
        if not is_processed:
            env_reset = generate_reset_config(scenario)
            meta_scenario_reset_dict[meta_scenario.benchmark_id] = env_reset
            processed_location_list.append(meta_scenario)
            problem_meta_scenario_dict[scenario.benchmark_id] = meta_scenario
        else:
            problem_meta_scenario_dict[scenario.benchmark_id] = old_meta_scenario
        if duplicate:
            with open(
                os.path.join(
                    output_dir, "problem_train", f"{scenario.benchmark_id}.pickle"
                ),
                "wb",
            ) as f:
                pickle.dump(problem_dict, f)
            shutil.copytree(
                os.path.join(output_dir, "problem_train"),
                os.path.join(output_dir, "problem_test"),
            )
        else:
            with open(
                os.path.join(output_dir, "problem", f"{scenario.benchmark_id}.pickle"),
                "wb",
            ) as f:
                pickle.dump(problem_dict, f)
        if verbose:
            print("({}s)".format(time() - _intermediate_time), end=" ")
    _end_time = time()

    print("Meta scenarios: {}".format(len(meta_scenario_reset_dict)))
    print("Scenarios: {}".format(len(problem_meta_scenario_dict)))
    print("Took {}s".format(_end_time - _start_time))

    with open(
        os.path.join(output_dir, meta_scenario_path, "meta_scenario_reset_dict.pickle"),
        "wb",
    ) as f:
        pickle.dump(meta_scenario_reset_dict, f)
    with open(
        os.path.join(
            output_dir, meta_scenario_path, "problem_meta_scenario_dict.pickle"
        ),
        "wb",
    ) as f:
        pickle.dump(problem_meta_scenario_dict, f)


if __name__ == "__main__":
    # get arguments
    args = get_args()

    pickle_xml_scenarios(
        args.input_dir,
        args.output_dir,
        args.shared_dir,
        multiprocessing=args.multiprocessing,
        no_obstacles=args.no_obstacles,
        verbose=args.verbose,
        lanelet_assignment=args.lanelet_assignment,
        duplicate=args.duplicate,
    )
