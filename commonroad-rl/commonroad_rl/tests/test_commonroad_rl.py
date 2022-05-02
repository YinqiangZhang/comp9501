__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Integration tests of the CommonRoad-RL repository
"""
import os
import logging

from commonroad_rl.run_stable_baselines import (
    run_stable_baselines,
    run_stable_baselines_argsparser,
)
from commonroad_rl.solve_stable_baselines import solve_scenarios
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import resource_root, output_root
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios

logging.root.setLevel(logging.DEBUG)

env_id = "commonroad-v0"

resource_path = resource_root("test_commonroad_rl")
output_path = output_root("test_commonroad_rl")


def run_overfit(test_batch, goal_relaxation, num_of_steps):

    xml_scenarios_path = os.path.join(resource_path, test_batch)
    output_base_path = os.path.join(output_path, test_batch)
    pickle_path = os.path.join(output_base_path, "pickles")
    log_path = os.path.join(output_base_path, "logs")
    solution_path = os.path.join(output_base_path, "solutions")

    # Pickle CommonRoad scenarios
    pickle_xml_scenarios(
        input_dir=xml_scenarios_path,
        output_dir=pickle_path,
        shared_dir=False,
        multiprocessing=False,
        verbose=True,
    )

    # Overfit model
    meta_scenario_path = os.path.join(pickle_path, "meta_scenario")
    train_reset_config_path = os.path.join(pickle_path, "problem")
    test_reset_config_path = os.path.join(pickle_path, "problem")
    visualization_path = os.path.join(output_path, "images")

    algo = "ppo2"

    args_str = (
        f"--algo {algo} --seed 6 --eval-freq 1000 --log-folder {log_path} --n-timesteps {num_of_steps}"
        f" --env-kwargs"
        f' reward_type:"hybrid_reward"'
        f' meta_scenario_path:"{meta_scenario_path}"'
        f' train_reset_config_path:"{train_reset_config_path}"'
        f' test_reset_config_path:"{test_reset_config_path}"'
        f' visualization_path:"{visualization_path}"'
        f" relax_is_goal_reached:{goal_relaxation}"
    )

    args = run_stable_baselines_argsparser().parse_args(args_str.split(sep=" "))
    run_stable_baselines(args)

    # Solve scenarios
    model_path = os.path.join(log_path, algo, "commonroad-v0_1")
    cost_function = "SA1"

    results = solve_scenarios(
        test_path=test_reset_config_path,
        model_path=model_path,
        algo=algo,
        solution_path=solution_path,
        cost_function=cost_function,
    )

    assert all(results)


@pytest.mark.parametrize(
    ("test_batch", "goal_relaxation", "num_of_steps"), [("DEU_A9-2_1_T-1", False, 3000)]
)
@functional
@integration_test
def test_overfit_model(test_batch, goal_relaxation, num_of_steps):
    run_overfit(test_batch, goal_relaxation, num_of_steps)


# TODO: add more difficult batch
@pytest.mark.parametrize(
    ("test_batch", "goal_relaxation", "num_of_steps"),
    [("DEU_A99-1_1_T-1", False, 30000)],
)
@slow
@functional
@integration_test
def test_overfit_model_slow(test_batch, goal_relaxation, num_of_steps):
    run_overfit(test_batch, goal_relaxation, num_of_steps)
