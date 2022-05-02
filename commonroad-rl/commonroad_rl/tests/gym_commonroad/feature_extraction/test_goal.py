__author__ = "Xiao Wang"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "under development"

"""
Unit test script for goal-related observations
"""
from commonroad_rl.gym_commonroad.utils.scenario import *

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle
from commonroad.common.util import Interval, AngleInterval

from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.evaluation import *
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_rl.gym_commonroad.feature_extraction.goal import GoalObservation


dummy_time_step = Interval(0.0, 0.0)


@pytest.mark.parametrize(
    ("ego_position", "expected_output"),
    [
        (np.array([0.0, 0.0]), (4.0, 0.0)),
        (np.array([4.0, 0.0]), (0.0, 0.0)),
        (np.array([5.0, 0.0]), (0.0, 0.0)),
        (np.array([6.0, 0.0]), (0.0, 0.0)),
        (np.array([10.0, 0.0]), (-4.0, 0.0)),
        (np.array([0.0, 1.0]), (4.0, 0.0)),
        (np.array([4.0, 1.0]), (0.0, 0.0)),
        (np.array([5.0, 1.0]), (0.0, 0.0)),
        (np.array([6.0, 1.0]), (0.0, 0.0)),
        (np.array([10.0, 1.0]), (-4.0, 0.0)),
        (np.array([0.0, 2.0]), (4.0, -1.0)),
        (np.array([4.0, 2.0]), (0.0, -1.0)),
        (np.array([5.0, 2.0]), (0.0, -1.0)),
        (np.array([6.0, 2.0]), (0.0, -1.0)),
        (np.array([10.0, 2.0]), (-4.0, -1.0)),
        (np.array([0.0, -1.0]), (4.0, 0.0)),
        (np.array([4.0, -1.0]), (0.0, 0.0)),
        (np.array([5.0, -1.0]), (0.0, 0.0)),
        (np.array([6.0, -1.0]), (0.0, 0.0)),
        (np.array([10.0, -1.0]), (-4.0, 0.0)),
        (np.array([0.0, -2.0]), (4.0, 1.0)),
        (np.array([4.0, -2.0]), (0.0, 1.0)),
        (np.array([5.0, -2.0]), (0.0, 1.0)),
        (np.array([6.0, -2.0]), (0.0, 1.0)),
        (np.array([10.0, -2.0]), (-4.0, 1.0)),
    ],
)
@unit_test
@functional
def test_get_long_lat_distance_to_goal(ego_position, expected_output):
    """Tests the test_get_long_lat_distance_to_goal function"""

    dummy_state = {
        "velocity": 0.0,
        "orientation": 0.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0.0,
    }
    ego_state = State(**dummy_state, position=ego_position)
    goal_state = State(time_step=dummy_time_step,
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])))
    planning_problem = PlanningProblem(planning_problem_id=0, initial_state=ego_state,
                                       goal_region=GoalRegion([goal_state]))

    lanelet = Lanelet(lanelet_id=0,
                      left_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
                      center_vertices=np.array([[0.0, 0.0], [10.0, 0.0]]),
                      right_vertices=np.array([[0.0, -3.0], [10.0, -3.0]]))

    scenario = Scenario(dt=0.1, benchmark_id="test")
    scenario.lanelet_network.add_lanelet(lanelet)
    goal_obs = GoalObservation(scenario, planning_problem)

    min_distance_long, min_distance_lat = goal_obs.get_long_lat_distance_to_goal(ego_state.position)
    assert np.isclose(min_distance_long, expected_output[0]) \
           and np.isclose(min_distance_lat, expected_output[1])


@pytest.mark.parametrize(
    ("ego_orientation", "expected_output"),
    [
        (-np.pi, -np.pi / 2.),
        (-1., 0.),
        (0., 0.),
        (1., 0.),
        (np.pi, np.pi / 2.),
    ],
)
# Matthias Hamacher's tests
@module_test
@functional
def test_get_goal_orientation_distance(ego_orientation, expected_output):
    """Tests GoalObservation.get_goal_orientation_distance"""

    dummy_state = {
        "velocity": 0.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0.0,
        "position": np.array([0.0, 0.0])
    }
    ego_state = State(**dummy_state, orientation=ego_orientation)
    goal_state = State(time_step=dummy_time_step,
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])),
                       orientation=AngleInterval(-np.pi / 2, np.pi / 2))
    planning_problem = PlanningProblem(planning_problem_id=0, initial_state=ego_state,
                                       goal_region=GoalRegion([goal_state]))
    lanelet = Lanelet(lanelet_id=0,
                      left_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
                      center_vertices=np.array([[0.0, 0.0], [10.0, 0.0]]),
                      right_vertices=np.array([[0.0, -3.0], [10.0, -3.0]]))

    scenario = Scenario(dt=0.1, benchmark_id="test")
    scenario.lanelet_network.add_lanelet(lanelet)
    goal_obs = GoalObservation(scenario, planning_problem)

    min_goal_orientation_distance = goal_obs.get_goal_orientation_distance(ego_state.orientation)

    assert np.isclose(min_goal_orientation_distance, expected_output)
# [10, 20]
@pytest.mark.parametrize(
    ("ego_time_step", "expected_output"),
    [
        (5, -5),
        (10, 0),
        (15, 0),
        (20, 0),
        (30, 10),
    ],
)

@module_test
@functional
def test_get_goal_time_distance(ego_time_step, expected_output):
    """Tests GoalObservation.get_goal_time_distance"""

    dummy_state = {
        "velocity": 0.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "orientation": 0.0,
        "position": np.array([0.0, 0.0])
    }
    ego_state = State(**dummy_state, time_step=ego_time_step)
    goal_state = State(time_step=Interval(10, 20),
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])),
                       orientation=AngleInterval(-np.pi / 2, np.pi / 2))
    planning_problem = PlanningProblem(planning_problem_id=0, initial_state=ego_state,
                                       goal_region=GoalRegion([goal_state]))
    lanelet = Lanelet(lanelet_id=0,
                      left_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
                      center_vertices=np.array([[0.0, 0.0], [10.0, 0.0]]),
                      right_vertices=np.array([[0.0, -3.0], [10.0, -3.0]]))

    scenario = Scenario(dt=0.1, benchmark_id="test")
    scenario.lanelet_network.add_lanelet(lanelet)
    goal_obs = GoalObservation(scenario, planning_problem)

    min_goal_time_distance = goal_obs.get_goal_time_distance(ego_state.time_step)

    assert np.isclose(min_goal_time_distance, expected_output)

@pytest.mark.parametrize(
    ("ego_velocity", "expected_output"),
    [
        (5., -5.),
        (10., 0.),
        (15., 0.),
        (20., 0.),
        (30., 10.),
    ],
)

@module_test
@functional
def test_get_goal_velocity_distance(ego_velocity, expected_output):
    """Tests GoalObservation.get_goal_velocity_distance"""
    dummy_state = {
        "time_step": dummy_time_step,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "orientation": 0.0,
        "position": np.array([0.0, 0.0])
    }
    ego_state = State(**dummy_state, velocity=ego_velocity)
    goal_state = State(time_step=Interval(10, 20),
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])),
                       orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                       velocity=Interval(10., 20.))
    planning_problem = PlanningProblem(planning_problem_id=0, initial_state=ego_state,
                                       goal_region=GoalRegion([goal_state]))
    lanelet = Lanelet(lanelet_id=0,
                      left_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
                      center_vertices=np.array([[0.0, 0.0], [10.0, 0.0]]),
                      right_vertices=np.array([[0.0, -3.0], [10.0, -3.0]]))

    scenario = Scenario(dt=0.1, benchmark_id="test")
    scenario.lanelet_network.add_lanelet(lanelet)
    goal_obs = GoalObservation(scenario, planning_problem)

    min_goal_velocity_distance = goal_obs.get_goal_velocity_distance(ego_state.velocity)

    assert np.isclose(min_goal_velocity_distance, expected_output)