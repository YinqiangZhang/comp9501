from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario, State
from commonroad.common.util import (
    make_valid_orientation,
    make_valid_orientation_interval,
)

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"


def get_end_time(scenario: Scenario):
    """
    Return the last timestep that any dynamic vehicle is visible in the scenario
    """
    if scenario.dynamic_obstacles:
        return max(
            len(o.prediction.trajectory.state_list) for o in scenario.dynamic_obstacles
        )
    return -1


def make_valid_orientation_pruned(orientation: float):
    """
    Make orientation valid and prune to correct representation for XML with 6 significant digits
    """
    orientation = make_valid_orientation(orientation)
    return max(min(orientation, 6.283185), -6.283185)


def make_valid_orientation_interval_pruned(o1: float, o2: float):
    """
    Make orientation valid and prune to correct representation for XML with 6 significant digits
    """
    o1, o2 = make_valid_orientation_interval(o1, o2)
    return make_valid_orientation_pruned(o1), make_valid_orientation_pruned(o2)
