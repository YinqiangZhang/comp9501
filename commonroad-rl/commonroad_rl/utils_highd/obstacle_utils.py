import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from commonroad.scenario.trajectory import Trajectory

obstacle_class_dict = {"Truck": ObstacleType.TRUCK, "Car": ObstacleType.CAR}


def get_velocity(df):
    return np.sqrt(df.xVelocity ** 2 + df.yVelocity ** 2)


def get_orientation(df):
    return np.arctan2(-df.yVelocity, df.xVelocity)


def get_acceleration(orientation, df):
    a = np.cos(orientation) * df.xAcceleration + np.sin(orientation) * (
        -df.yAcceleration
    )
    return round(a, 18)


def get_yawRate(df):
    yawRate = df.yVelocity / df.xVelocity
    return yawRate


def generate_dynamic_obstacle(
    scenario, vehicle_id, tracks_meta_df, tracks_df, frame_start
):
    vehicle_meta = tracks_meta_df[tracks_meta_df.id == vehicle_id]
    vehicle_tracks = tracks_df[tracks_df.id == vehicle_id]

    length = vehicle_meta.width.values[0]
    width = vehicle_meta.height.values[0]

    initial_time_step = int(vehicle_tracks.frame.values[0]) - frame_start
    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = obstacle_class_dict[vehicle_meta["class"].values[0]]
    dynamic_obstacle_shape = Rectangle(width=width, length=length)

    xs = np.array(vehicle_tracks.x)  # checked x signals, no need to filter
    ys = np.array(-vehicle_tracks.y)
    velocities = get_velocity(vehicle_tracks)
    orientations = get_orientation(vehicle_tracks)
    accelerations = get_acceleration(orientations, vehicle_tracks)
    yawRate = get_yawRate(vehicle_tracks)
    state_list = []
    for i, (x, y, v, theta, a, yaw) in enumerate(
        zip(xs, ys, velocities, orientations, accelerations, yawRate)
    ):
        state_list.append(
            State(
                position=np.array([x, y]),
                velocity=v,
                orientation=theta,
                acceleration=a,
                yaw_rate=yaw,
                time_step=initial_time_step + i,
            )
        )
    dynamic_obstacle_initial_state = state_list[0]
    dynamic_obstacle_trajectory = Trajectory(initial_time_step + 1, state_list[1:])
    dynamic_obstacle_prediction = TrajectoryPrediction(
        dynamic_obstacle_trajectory, dynamic_obstacle_shape
    )
    return DynamicObstacle(
        dynamic_obstacle_id,
        dynamic_obstacle_type,
        dynamic_obstacle_shape,
        dynamic_obstacle_initial_state,
        dynamic_obstacle_prediction,
    )
