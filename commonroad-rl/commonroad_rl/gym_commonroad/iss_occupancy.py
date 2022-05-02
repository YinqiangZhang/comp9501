import numpy as np
import math
from abc import ABC, abstractmethod 

"""
modified version of ISS occupancy for simplification
remove the whole obstacle as the input parameter, 
only select the max acceleration and the delta t of brake operation
"""


class ISSOccupancy(ABC):
    """
    Abstract base class for the occupancy of an obstacle.

    Parameters
    ----------
    obstacle : FrenetObstacle
        The FrenetObstacle this occupancy belongs to.
    s_closest : float
        The closest possible longitudinal position [m] of the obstacle at
        time t.
    s_furthest : float
        The furthest possible longitudinal position [m] of the obstacle at
        time t.
    """

    def __init__(self, obstacle_params, s_closest, s_furthest):
        self.obstacle_params = obstacle_params
        self.s_closest = s_closest
        self.s_furthest = s_furthest

    @abstractmethod
    def safe_distance_back(self, v_ego, a_max_ego, delta_brake_ego):
        """
        The minimal distance the ego vehicle needs to keep behind this
        occupancy in order to be able to come to come to a stop behind the
        occupancy.

        Parameters
        ----------
        v_ego : float
            The velocity [m/s] of the ego vehicle
        a_max_ego : float
            The maximal acceleration [m/s] of the ego vehicle
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        pass

    @abstractmethod
    def safe_distance_front(self, v_ego, a_max_ego, delta_brake_ego):
        """
        The minimal distance the ego vehicle needs to keep in front of this
        occupancy in order for the occupancy's vehicle to be able to come to
        a stop behind the ego vehicle.

        Parameters
        ----------
        v_ego : float
            The velocity [m/s] of the ego vehicle
        a_max_ego : float
            The maximal acceleration [m/s] of the ego vehicle
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        pass

    @abstractmethod
    def safe_leading_velocity_brake(self, delta_s, a_max_ego, delta_brake_ego):
        """
        Compute the minimal velocity the ego vehicle would have to have in
        order to be safe when driving delta_s m in front of this occupancy.

        Parameters
        ----------
        delta_s : float
            The distance [m] between the furthest point of this occupancy and
            the closest point of the ego vehicle.
        a_max_ego : float
            The maximal acceleration [m/s²] of the ego vehicle.
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        pass

    @abstractmethod
    def safe_following_velocity_brake(self, delta_s, a_max_ego, delta_brake_ego):
        """
        Compute the maximal velocity the ego vehicle would have to have in
        order to be safe when driving delta_s m behind this occupancy.

        Parameters
        ----------
        delta_s : float
            The distance [m] between the closest point of this occupancy and
            the furthest point of the ego vehicle.
        a_max_ego : float
            The maximal acceleration [m/s] of the ego vehicle
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        pass

    @abstractmethod
    def safe_leading_velocity_evade(self, distance_ego_to_obstacle, t_eva):
        pass

    @abstractmethod
    def safe_following_velocity_evade(self, distance_ego_to_obstacle, a_max_ego):
        pass

    @abstractmethod
    def evasive_distance_back(self, t_eva, v_ego):
        """
        Distance the ego vehicle needs to have when it wants to perform an
        evasive maneuver.
        """
        pass

    @abstractmethod
    def evasive_distance_front(self, t_eva, v_ego):
        """
        Distance the ego vehicle needs to have (optionally) when it wants to perform an
        evasive maneuver.
        """
        pass


class ISSStaticOccupancy(ISSOccupancy):
    """
    Occupancy of a static obstacle.

    Parameters
    ----------
    obstacle : StaticFrenetObstacle
        The obstacle this occupancy belongs to.
    s_closest : float
        The closest possible longitudinal position [m] of the obstacle at
        time t.
    s_furthest : float
        The furthest possible longitudinal position [m] of the obstacle at
        time t.
    """

    def __init__(self, obstacle_params, s_closest, s_furthest):
        super().__init__(obstacle_params, s_closest, s_furthest)
        # For plotting
        self.color = 'r'

    def safe_distance_back(self, v_ego, a_max_ego, delta_brake_ego):
        """
        The minimal distance the ego vehicle needs to keep behind this
        occupancy in order to be able to come to come to a stop behind the
        occupancy. As this is a static occupancy, this is simply the distance
        travelled when the ego vehicle brakes with a_max_ego.

        Parameters
        ----------
        v_ego : float
            The velocity [m/s] of the ego vehicle
        a_max_ego : float
            The maximal acceleration [m/s] of the ego vehicle
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        return v_ego ** 2 / (2 * a_max_ego) + delta_brake_ego * v_ego

    def safe_distance_front(self, v_ego, a_max_ego, delta_brake_ego):
        """
        The minimal distance the ego vehicle needs to keep in front of this
        occupancy in order for the occupancy's vehicle to be able to come to
        a stop behind the ego vehicle. As this is a static occupancy, any
        non-negative distance will be fine.

        Parameters
        ----------
        v_ego : float
            The velocity [m/s] of the ego vehicle
        a_max_ego : float
            The maximal acceleration [m/s] of the ego vehicle
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        return 0

    def safe_leading_velocity_brake(self, delta_s, a_max_ego, delta_brake_ego):
        """
        Compute the minimal velocity the ego vehicle would have to have in
        order to be safe when driving delta_s m in front of this occupancy.
        As this obstacle is static, any non-negative velocity will be safe.

        Parameters
        ----------
        delta_s : float
            The distance [m] between the furthest point of this occupancy and
            the closest point of the ego vehicle.
        a_max_ego : float
            The maximal acceleration [m/s²] of the ego vehicle.
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        return 0

    def safe_following_velocity_brake(self, delta_s, a_max_ego, delta_brake_ego):
        """
        Compute the maximal velocity the ego vehicle would have to have in
        order to be safe when driving delta_s m behind this occupancy. As this
        obstacle is safe, this is the highest velocity that will allow
        the vehicle to come to a stop before s_closest.

        Parameters
        ----------
        delta_s : float
            The distance [m] between the closest point of this occupancy and
            the furthest point of the ego vehicle.
        a_max_ego : float
            The maximal acceleration [m/s²] of the ego vehicle.
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        # Solve quadratic equation for non-negative solution
        a = 1 / (-2 * a_max_ego)
        b = np.array(-delta_brake_ego)
        c = delta_s
        v_f = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        v_f = max(v_f, (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a))

        return v_f

    def safe_following_velocity_evade(self, delta_s, a_max_ego):
        # todo: implement this
        return 0

    def safe_leading_velocity_evade(self, distance_ego_to_obstacle, t_eva):
        return 0

    def __str__(self):
        return 's: {} --- {}'.format(self.s_closest, self.s_furthest)

    def evasive_distance_back(self, t_eva, v_ego):
        """
        Distance the ego vehicle needs to have to the leading obstacle when it wants to perform an
        evasive maneuver.
        """
        t_eva = t_eva / 10
        return v_ego * t_eva

    def evasive_distance_front(self, t_eva, v_ego):
        """
        Distance the ego vehicle needs to have (optionally) to the following obstacle when it wants to perform an
        evasive maneuver.
        """
        return 0


class ISSDynamicOccupancy(ISSOccupancy):
    """
    Occupancy of a dynamic obstacle.

    Parameters
    ----------
    obstacle :
        The obstacle this occupancy belongs to.
    s_closest : float
        The closest possible longitudinal position [m] of the obstacle at time t.
    s_furthest : float
        The furthest possible longitudinal position [m] of the obstacle at time t.
    v_closest : float
        The velocity [m/s] the obstacle would have at s_closest at time t.
    v_furthest : float
        The velocity [m/s] the obstacle would have at s_furthest at time t.
    """

    def __init__(self, obstacle_params, s_closest, s_furthest, v_closest, v_furthest):
        super().__init__(obstacle_params, s_closest, s_furthest)
        self.v_closest = v_closest
        self.v_furthest = v_furthest
        # For plotting
        self.color = 'b'

    def safe_distance_back(self, v_ego, a_max_ego, delta_brake_ego):
        """
        The minimal distance the ego vehicle needs to keep behind this
        occupancy in order to be able to come to come to a stop behind the
        occupancy. As this is a dynamic occupancy, this is given by the safe
        distance equation from "Verifying the Safety of Lane Change Maneuvers
        of Self-driving Vehicles Based on Formalized Traffic Rules", Pek et
        al. (2017).

        Parameters
        ----------
        v_ego : float
            The velocity [m/s] of the ego vehicle
        a_max_ego : float
            The maximal acceleration [m/s] of the ego vehicle
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        return self.v_closest ** 2 / (-2 * self.obstacle_params['a_max']) - \
               v_ego ** 2 / (-2 * a_max_ego) + delta_brake_ego * v_ego

    def safe_distance_front(self, v_ego, a_max_ego, delta_brake_ego):
        """
        The minimal distance the ego vehicle needs to keep in front of this
        occupancy in order for the occupancy's vehicle to be able to come to
        a stop behind the ego vehicle. As this is a dynamic occupancy, this
        is given by the safe distance equation from "Verifying the Safety of
        Lane Change Maneuvers of Self-driving Vehicles Based on Formalized
        Traffic Rules", Pek et al. (2017).

        Parameters
        ----------
        v_ego : float
            The velocity [m/s] of the ego vehicle
        a_max_ego : float
            The maximal acceleration [m/s] of the ego vehicle
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        return v_ego ** 2 / (-2 * a_max_ego) - self.v_furthest ** 2 / \
               (-2 * self.obstacle_params['a_max']) + self.obstacle_params['delta_brake'] * \
               self.v_furthest

    def safe_leading_velocity_brake(self, delta_s, a_max_ego, delta_brake_ego):
        """
        Compute the minimal velocity the ego vehicle would have to have in
        order to be safe when driving delta_s m in front of this occupancy.
        As this obstacle is dynamic, solve safe distance equation.

        Parameters
        ----------
        delta_s : float
            The distance [m] between the furthest point of this occupancy and
            the closest point of the ego vehicle.
        a_max_ego : float
            The maximal acceleration [m/s²] of the ego vehicle.
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        first_term = delta_s + self.v_furthest ** 2 / (-2 * self.obstacle_params['a_max']) - \
                     self.v_furthest * self.obstacle_params['delta_brake']
        # above term > 0 implies negative velocity would be possible, but driving
        # backwards is not allowed.
        if first_term > 0:
            return 0

        return math.sqrt(first_term * (-2 * a_max_ego))

    def safe_following_velocity_brake(self, delta_s, a_max_ego, delta_brake_ego):
        """
        Compute the maximal velocity the ego vehicle would have to have in
        order to be safe when driving delta_s m behind this occupancy. As this
        obstacle is dynamic, solve safe distance equation.

        Parameters
        ----------
        delta_s : float
            The distance [m] between the closest point of this occupancy and
            the furthest point of the ego vehicle.
        a_max_ego : float
            The maximal acceleration [m/s²] of the ego vehicle.
        delta_brake_ego : float
            The time [s] it takes for the ego vehicle to react to an emergency
            brake maneuver of this occupancy's vehicle.
        """
        a = 1 / (-2 * a_max_ego)
        b = -delta_brake_ego
        c = delta_s - self.v_closest ** 2 / (-2 * self.obstacle_params['a_max'])
        root_term = math.sqrt(b ** 2 - 4 * a * c)
        v_f = (-b + root_term) / (2 * a)
        if v_f < 0:
            v_f = (-b - root_term) / (2 * a)

        return v_f

    def safe_following_velocity_evade(self, dist_between_ego_occ_lead, t_evade):
        t_evade = t_evade / 10
        t_brake_occ_lead = min(self.v_closest / self.obstacle_params['a_max'], t_evade)

        # travel distance of leading vehicle within time t
        delta_s_lead = self.v_closest * t_brake_occ_lead - 0.5 * self.obstacle_params['a_max'] * t_brake_occ_lead ** 2
        delta_s_safe = delta_s_lead + dist_between_ego_occ_lead
        v_safe_max = delta_s_safe / t_evade
        return v_safe_max

    def safe_leading_velocity_evade(self, distance_ego_to_obstacle, t_evade):
        """
        Distance the ego vehicle needs to have to the following obstacle when it wants to perform an
        evasive maneuver.
        """
        t_evade = t_evade / 10

        # todo: ideally, we should consider the case that the obstacle has reached its maximum velocity
        # t_acc = v_max - v_furthest / obstacle.a_max
        # delta_s_follow = part with acceleration for duration of t_acc + part with constant max vel
        delta_s_follow = self.v_furthest * t_evade + 0.5 * self.obstacle_params['a_max'] * t_evade ** 2

        delta_s_ego_safe = delta_s_follow - distance_ego_to_obstacle
        v_safe_min = delta_s_ego_safe / t_evade

        return v_safe_min

    def __str__(self):
        return 's: {} --- {}\n' \
               'v: {} --- {}'.format(self.s_closest, self.s_furthest,
                                     self.v_closest, self.v_furthest)

    def evasive_distance_back(self, t_evade, v_ego):
        """
        Distance the ego vehicle needs to have when it wants to perform an
        evasive maneuver.
        """
        t_evade = t_evade / 10
        t = min(self.v_closest / self.obstacle_params['a_max'], t_evade)

        # travel distance of ego vehicle during t_eva
        delta_s_ego = v_ego * t_evade
        # travel distance of leading vehicle within time t
        delta_s_lead = self.v_closest * t - 0.5 * self.obstacle_params['a_max'] * t ** 2
        # distance = following vehicle - leading vehicle
        return delta_s_ego - delta_s_lead

    def evasive_distance_front(self, t_evade, v_ego):
        """
        Distance the ego vehicle needs to have to the following obstacle when it wants to perform an
        evasive maneuver.
        """
        t_evade = t_evade / 10

        # travel distance of ego vehicle during t_eva
        delta_s_ego = v_ego * t_evade
        # travel distance of following vehicle with max acceleration within t_eva
        delta_s_follow = self.v_furthest * t_evade + 0.5 * self.obstacle_params['a_max'] * t_evade ** 2
        # distance = following vehicle - leading vehicle
        return delta_s_follow - delta_s_ego

