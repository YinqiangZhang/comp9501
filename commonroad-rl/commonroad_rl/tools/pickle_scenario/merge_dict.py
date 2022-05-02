import os
import pickle
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Converts CommonRoad xml files to pickle files"
    )
    parser.add_argument("--num_cpus", "-n", type=int, default=1)
    parser.add_argument(
        "--output_dir", "-o", type=str, default="/data/highD-dataset-v1.0/pickles"
    )

    return parser.parse_args()


def main():

    args = get_args()
    # print(args.num_cpu)
    os.makedirs(args.output_dir, exist_ok=True)

    #    meta_scenario_reset_dict = {}
    problem_meta_scenario_dict = {}

    for i in range(1, args.num_cpus + 1):
        #        with open(
        #            os.path.join(f"{args.output_dir}_{i}", "meta_scenario_reset_dict.pickle"),
        #            "rb",
        #        ) as f:
        #            meta_scenario_reset_dict.update(pickle.load(f))
        with open(
            os.path.join(f"{args.output_dir}/{i}", "problem_meta_scenario_dict.pickle"),
            "rb",
        ) as f:
            problem_meta_scenario_dict.update(pickle.load(f))

    #    with open(
    #        os.path.join(args.output_dir, "meta_scenario_reset_dict.pickle"), "wb"
    #    ) as f:
    #        pickle.dump(meta_scenario_reset_dict, f)
    with open(
        os.path.join(args.output_dir, "problem_meta_scenario_dict.pickle"), "wb"
    ) as f:
        pickle.dump(problem_meta_scenario_dict, f)

    #    print(len(meta_scenario_reset_dict.keys()))
    print(len(problem_meta_scenario_dict.keys()))


if __name__ == "__main__":
    main()
