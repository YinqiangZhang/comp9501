__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

"""
Module for managing the vehicle in the CommonRoad Gym environment
"""
import numpy as np
import commonroad_dc.pycrcc as pycrcc

from typing import List
from abc import ABC, abstractmethod
from aenum import extend_enum

from commonroad.scenario.trajectory import State
from commonroad.common.solution_writer import VehicleModel, VehicleType
from commonroad.common.util import make_valid_orientation

from vehiclemodels.vehicle_parameters import VehicleParameters
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks

N_INTEGRATION_STEPS = 4


extend_enum(VehicleModel, 'YawRate', len(VehicleModel))


class Vehicle(ABC):
    """
    Description:
        Abstract base class of the vehicle
    """

    @staticmethod
    def create_vehicle(params_dict: dict) -> "Vehicle":
        """
        Create concrete vehicle from vehicle_type and vehicle_model
        :param params_dict: The params of the vehicle
        :return: The created vehicle
        """
        vehicle_type = VehicleType(params_dict["vehicle_type"])
        vehicle_model = VehicleModel(params_dict["vehicle_model"])
        if vehicle_model == VehicleModel.PM:
            vehicle = _VehiclePM(vehicle_type)
        elif vehicle_model == VehicleModel.KS:
            vehicle = _VehicleKS(vehicle_type)
        elif vehicle_model == VehicleModel.ST:
            raise NotImplementedError(f"This vehicle model is not supported yet: {vehicle_model}")
        elif vehicle_model == VehicleModel.MB:
            raise NotImplementedError(f"This vehicle model is not supported yet: {vehicle_model}")
        elif vehicle_model == VehicleModel.YawRate:
            vehicle = _VehicleOld(vehicle_type)
            print("Using OLD vehicle model!")
        else:
            raise ValueError(f"Unknown vehicle model: {vehicle_model}")

        return vehicle

    def __init__(self, vehicle_type: VehicleType or None) -> None:
        """ Initialize empty object """
        self.vehicle_type = vehicle_type
        self.name, self.params = self._get_vehicle_info(self.vehicle_type)
        self.current_time_step = None
        self.dt = None
        self._friction_violation = None
        self._collision_object = None
        self._rescale_factor = np.array(
            [
                self.params.longitudinal.a_max,
                (self.params.steering.v_max - self.params.steering.v_min) / 2.0,
            ]
        )
        self._rescale_bias = np.array(
            [0.0, (self.params.steering.v_max + self.params.steering.v_min) / 2.0]
        )

    @staticmethod
    def _get_vehicle_info(vehicle_type: VehicleType) -> (str, VehicleParameters):
        if vehicle_type == VehicleType.FORD_ESCORT:
            veh_param = parameters_vehicle1()
            veh_name = "FORD_ESCORT"
        elif vehicle_type == VehicleType.BMW_320i:
            veh_param = parameters_vehicle2()
            veh_name = "BMW320i"
        elif vehicle_type == VehicleType.VW_VANAGON:
            veh_param = parameters_vehicle3()
            veh_name = "VW_VANAGON"
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")
        return veh_name, veh_param

    def step(self, normalized_action: np.ndarray) -> None:
        """
        Propagate to the next time step.
        :param normalized_action: Normalized action from the CommonroadEnv.
        :return: None
        """
        self.current_time_step += 1
        rescaled_action = self.rescale_action(normalized_action)
        new_state = self._get_new_state(rescaled_action)
        self._update_state(new_state)

    @staticmethod
    def _check_friction_violation(vehicle_parameters: VehicleParameters, ego_vehicle_state: State):
        """
        Check if the ego vehicle violates the friction circle constraint with the given input.
        :param ego_vehicle_state: The state of the ego vehicle
        :param normalized_action: The action which the agent will apply
        :return: If the ego vehicle violates the friction circle constraint with the given input
        """
        pass

    @abstractmethod
    def _get_new_state(self, action: np.ndarray) -> State:
        """
        Get the new state using the kinematic single track model.
        :param action: action from the CommonroadEnv.
        :return: new state
        """
        pass

    def rescale_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range
        :param normalized_action: action from the CommonroadEnv.
        :return: rescaled action
        """
        return self._rescale_factor * normalized_action + self._rescale_bias

    @property
    def state(self) -> State:
        """
        Get the current state of the vehicle
        :return: The current state of the vehicle
        """
        return self.state_list[-1]

    @property
    def previous_state(self) -> State:
        """
        Get the previous state of the vehicle
        :return: The previous state of the vehicle
        """
        return self.state_list[-2]

    @state.setter
    def state(self, state: State):
        """ Set the current state of the vehicle is not supported """
        raise ValueError("To set the state of the vehicle directly is prohibited!")

    def _reset_car_state(self, init_state: State) -> None:
        """
        Update the car parameters according to scenario and planning problem.
        :param init_state: initial state according to problem
        :return: None
        """
        assert self.current_time_step == 0
        self.initial_state = init_state
        self.state_list: List[State] = [init_state]
        self._update_collision_object()

    @property
    def collision_object(self) -> pycrcc.RectOBB:
        """
        Get the collision object of the vehicle
        :return: The collision object of the vehicle
        """
        return self._collision_object

    @collision_object.setter
    def collision_object(self, collision_object: pycrcc.RectOBB):
        """ Set the collision_object of the vehicle is not supported """
        raise ValueError(
            "To set the collision_object of the vehicle directly is prohibited!"
        )

    def _update_collision_object(self):
        """ Updates the collision_object of the vehicle """
        self._collision_object = pycrcc.TimeVariantCollisionObject(
            self.current_time_step
        )
        current_state = self.state
        self._collision_object.append_obstacle(
            pycrcc.RectOBB(
                self.params.l / 2,
                self.params.w / 2,
                current_state.orientation
                if hasattr(current_state, "orientation")
                else 0.0,
                current_state.position[0],
                current_state.position[1],
            )
        )

    def _update_state(self, new_state: State) -> None:
        """
        Update vehicle trajectory prediction
        :param new_state: new state
        :return: None
        """
        current_state = self.state
        current_state.acceleration = new_state.acceleration
        current_state.yaw_rate = (
            new_state.yaw_rate if hasattr(new_state, "yaw_rate") else 0.0
        )
        self.state_list.append(new_state)
        self._update_collision_object()

    def reset(self, initial_state: State, dt: float) -> None:
        """
        Reset vehicle parameters.
        :param initial_state: The initial state of the vehicle
        :param dt: Simulation dt of the scenario
        :return: None
        """
        self.current_time_step = 0
        self.dt = dt
        self._friction_violation = False if hasattr(initial_state, "yaw_rate") else None

        problem_init_state = initial_state
        init_state = State(
            **{
                "position": problem_init_state.position,
                "steering_angle": problem_init_state.steering_angle
                if hasattr(problem_init_state, "steering_angle")
                else 0.0,
                "orientation": problem_init_state.orientation
                if hasattr(problem_init_state, "orientation")
                else 0.0,
                "yaw_rate": problem_init_state.yaw_rate
                if hasattr(problem_init_state, "yaw_rate")
                else 0.0,
                "time_step": problem_init_state.time_step,
                "velocity": problem_init_state.velocity,
                "velocity_y": problem_init_state.velocity_y
                if hasattr(problem_init_state, "velocity_y")
                else 0.0,
                "acceleration": 0.0,
            }
        )
        self._reset_car_state(init_state)


class _VehicleKS(Vehicle):
    """
    Description:
        Concrete vehicle using KS model
    """

    def __init__(self, vehicle_type: VehicleType):
        super().__init__(vehicle_type)
        self.model_type = VehicleModel.KS
        self.vehicle_type = vehicle_type

    def _get_new_state(self, action: np.ndarray) -> State:
        """
        Get the new state using the kinematic single track model.
        :param action: rescaled action from the CommonroadEnv.
        :return: new state

        Actions:
            Type: Box(2)
            Num Action                                  Min                                 Max
            0   ego vehicle acceleration                -vehicle.params.longitudinal.a_max  vehicle.params.longitudinal.a_max
            1   ego vehicle steering angular velocity   -vehicle.params.steering.v_max      vehicle.params.steering.v_max

        Forward simulation of states
            ref: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/blob/master/vehicleModels_commonRoad.pdf

            x1 = x-position in a global coordinate system
            x2 = y-position in a global coordinate system
            x3 = steering angle of front wheels
            x4 = velocity in x-direction
            x5 = yaw angle

            u1 = steering angle velocity of front wheels
            u2 = longitudinal acceleration
        """
        current_state = self.state
        x_current = np.array(
            [
                current_state.position[0],
                current_state.position[1],
                current_state.steering_angle,
                current_state.velocity,
                current_state.orientation,
            ]
        )
        x_dot = None
        u_input = np.array([action[1], action[0]])

        for _ in range(N_INTEGRATION_STEPS):
            # simulate state transition
            x_dot = np.array(vehicle_dynamics_ks(x_current, u_input, self.params))

            # update state
            x_current = x_current + x_dot * (self.dt / N_INTEGRATION_STEPS)

        # feed in required slots
        kwarg = {
            "position": np.array([x_current[0], x_current[1]]),
            "velocity": x_current[3],
            "steering_angle": x_current[2],
            "orientation": make_valid_orientation(x_current[4]),
            "acceleration": u_input[1],
            "yaw_rate": x_dot[4],
            "time_step": self.current_time_step,
        }

        # append state
        new_state = State(**kwarg)

        return new_state

    @staticmethod
    def _check_friction_violation(vehicle_parameters: VehicleParameters, ego_vehicle_state: State):
        """
        Check if the ego vehicle violates the friction circle constraint with the given input.
        :param ego_vehicle_state: The state of the ego vehicle
        :param normalized_action: The action which the agent will apply
        :return: If the ego vehicle violates the friction circle constraint with the given input
        """
        return (ego_vehicle_state.acceleration ** 2 + (
                ego_vehicle_state.velocity * ego_vehicle_state.yaw_rate) ** 2) ** 0.5 > vehicle_parameters.longitudinal.a_max


class YawParameters():
    def __init__(self):
        #constraints regarding yaw
        self.v_min = []  #minimum yaw velocity [rad/s]
        self.v_max = []  #maximum yaw velocity [rad/s]


def extend_vehicle_params(p: VehicleParameters) -> VehicleParameters:
    p.yaw = YawParameters()
    p.yaw.v_min = -2.  # minimum yaw velocity [rad/s]
    p.yaw.v_max = 2.  # maximum yaw velocity [rad/s]

    return p


class _VehicleOld(Vehicle):
    """
    Description:
        Concrete vehicle using KS model
    """

    def __init__(self, vehicle_type: None) -> None:
        super().__init__(vehicle_type)
        self.params = extend_vehicle_params(self.params)
        self.current_time_step = 0
        self._rescale_factor = np.array([self.params.longitudinal.a_max,
                                         (self.params.yaw.v_max - self.params.yaw.v_min) / 2.0])
        self._rescale_bias = np.array([0.0, (self.params.yaw.v_max + self.params.yaw.v_min) / 2.0])

    def _get_new_state(self, action: np.ndarray) -> State:
        """
        Get the new state using the kinematic single track model.
        :param action: rescaled action from the CommonroadEnv.
        :return: new state
        """
        if self.current_time_step == 1:
            old_state = self.initial_state
        else:
            old_state = self.state_list[-1]
        theta = old_state.orientation
        v = old_state.velocity
        x = old_state.position[0]
        y = old_state.position[1]

        theta_dot = action[1]
        a = action[0]

        delta_t = self.dt / N_INTEGRATION_STEPS

        for _ in range(N_INTEGRATION_STEPS):
            x += v * np.cos(theta) * delta_t
            y += v * np.sin(theta) * delta_t
            theta += theta_dot * delta_t
            v += a * delta_t

        theta = shift_orientation(theta)
        assert np.pi >= theta >= -np.pi

        new_state = State(**{"position": np.array([x, y]),
                             "orientation": theta,
                             "time_step": self.current_time_step,
                             "yaw_rate": theta_dot,
                             "steering_angle": np.arctan(theta_dot * (self.params.a + self.params.b) / v),
                             "velocity": v,
                             "acceleration": a})

        return new_state

    def rescale_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range
        :param normalized_action: action from the CommonroadEnv.
        :return: rescaled action
        """
        return self._rescale_factor * normalized_action + self._rescale_bias

    def _update_collision_object(self):
        """ Updates the collision_object of the vehicle """
        self._collision_object = pycrcc.TimeVariantCollisionObject(
            self.current_time_step
        )
        current_state = self.state
        self._collision_object.append_obstacle(
            pycrcc.RectOBB(
                self.params.l / 2,
                self.params.w / 2,
                current_state.orientation,
                current_state.position[0],
                current_state.position[1],
            )
        )

    @staticmethod
    def _check_friction_violation(vehicle_parameters: VehicleParameters, ego_vehicle_state: State):
        """
        Check if the ego vehicle violates the friction circle constraint with the given input.
        :param ego_vehicle_state: The state of the ego vehicle
        :param normalized_action: The action which the agent will apply
        :return: If the ego vehicle violates the friction circle constraint with the given input
        """
        return (ego_vehicle_state.acceleration ** 2 + (
                ego_vehicle_state.velocity * ego_vehicle_state.yaw_rate) ** 2) ** 0.5 > vehicle_parameters.longitudinal.a_max


class _VehiclePM(Vehicle):
    """
    Description:
        Concrete vehicle using PM model
    """

    def __init__(self, vehicle_type: VehicleType):
        super().__init__(vehicle_type)
        self.model_type = VehicleModel.PM
        self.vehicle_type = vehicle_type

    def _get_new_state(self, action: np.ndarray) -> State:
        """
        Get the new state using the point mass model.
        :param action: rescaled action from the CommonroadEnv.
        :return: new state

        Forward simulation of states
            ref: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/blob/master/vehicleModels_commonRoad.pdf

            x1 = x-position in a global coordinate system
            x2 = y-position in a global coordinate system
            x3 = velocity in x-direction
            x4 = velocity in y-direction

            u1 = longitudinal acceleration
            u2 = lateral acceleration
        """

        current_state = self.state

        x_current = np.array(
            [
                current_state.position[0],
                current_state.position[1],
                current_state.velocity,
                current_state.velocity_y,
            ]
        )

        # if maximum absolute acceleration is exceeded, rescale the acceleration
        absolute_acc = np.sqrt(action[0] ** 2 + action[1] ** 2)
        if absolute_acc > self.params.longitudinal.a_max:
            rescale_factor = self.params.longitudinal.a_max / absolute_acc
            # rescale the acceleration
            action[0] *= rescale_factor
            action[1] *= rescale_factor

        delta_t = self.dt / N_INTEGRATION_STEPS

        for _ in range(N_INTEGRATION_STEPS):
            # update state
            x_current[0] += x_current[2] * delta_t
            x_current[1] += x_current[3] * delta_t
            x_current[2] += action[0] * delta_t
            x_current[3] += action[1] * delta_t

        # feed in required slots
        kwarg = {
            "position": np.array([x_current[0], x_current[1]]),
            "velocity": x_current[2],
            "velocity_y": x_current[3],
            "acceleration": action,
            "orientation": 0.0,
            "steering_angle": 0.0,
            "yaw_rate": 0.0,
            "time_step": self.current_time_step,
        }
        # append state
        new_state = State(**kwarg)

        return new_state

    @staticmethod
    def _check_friction_violation(
        vehicle_parameters: VehicleParameters, ego_vehicle_state: State
    ):
        """
        Check if the ego vehicle violates the friction circle constraint with the given input.
        :param ego_vehicle_state: The state of the ego vehicle
        :param normalized_action: The action which the agent will apply
        :return: If the ego vehicle violates the friction circle constraint with the given input
        """
        return False


    def rescale_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range
        :param normalized_action: action from the CommonroadEnv.
        :return: rescaled action
        """
        a_max = self.params.longitudinal.a_max
        factor = np.array([a_max, a_max])
        return factor * normalized_action


def shift_orientation(orientation):
    x = np.cos(orientation)
    y = np.sin(orientation)
    new_orientation = np.arctan2(y, x)
    # temp = np.arccos(np.cos(orientation))
    # if np.sin(orientation) < 0:
    #     new_orientation = -np.abs(temp)
    # else:
    #     new_orientation = np.abs(temp)
    return new_orientation
