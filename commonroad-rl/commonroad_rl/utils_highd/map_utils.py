import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import (
    TrafficSign,
    TrafficSignElement,
    TrafficSignIDGermany,
)


def resample_polyline(polyline, step=2.0):
    new_polyline = [polyline[0]]
    current_position = 0 + step
    current_length = np.linalg.norm(polyline[0] - polyline[1])
    current_idx = 0
    while current_idx < len(polyline) - 1:
        if current_position >= current_length:
            current_position = current_position - current_length
            current_idx += 1
            if current_idx > len(polyline) - 2:
                break
            current_length = np.linalg.norm(
                polyline[current_idx + 1] - polyline[current_idx]
            )
        else:
            rel = current_position / current_length
            new_polyline.append(
                (1 - rel) * polyline[current_idx] + rel * polyline[current_idx + 1]
            )
            current_position += step
    return np.array(new_polyline)


def get_lane_markings(df):
    #  example:upper [-8, -12, -16]
    #  example:lower [-22, -26, -30]
    upper_lane_markings_initial = [
        -float(x) for x in df.upperLaneMarkings.values[0].split(";")
    ]
    lower_lane_markings_initial = [
        -float(x) for x in df.lowerLaneMarkings.values[0].split(";")
    ]
    len_upper = len(upper_lane_markings_initial)
    len_lower = len(lower_lane_markings_initial)
    # -8 + 1 = -7
    upper_lane_markings_initial[0] = upper_lane_markings_initial[0] + 1
    # -16 -1 = -17
    upper_lane_markings_initial[len_upper - 1] = (
        upper_lane_markings_initial[len_upper - 1] - 1
    )
    # -22 + 1 = -21
    lower_lane_markings_initial[0] = lower_lane_markings_initial[0] + 1
    # -30 -1 = -31
    lower_lane_markings_initial[len_lower - 1] = (
        lower_lane_markings_initial[len_lower - 1] - 1
    )

    return upper_lane_markings_initial, lower_lane_markings_initial


def get_dt(df):
    return 1.0 / df.frameRate.values[0]


def get_location_id(df):
    return df.locationId.values[0]


def get_file_id(df):
    return df.id.values[0]


def get_speed_limit(df):
    speed_limit = df.speedLimit.values[0]
    if speed_limit < 0:
        return np.inf
    else:
        return speed_limit


def get_meta_scenario(df, is_up=False):
    benchmark_id = "meta_map"
    meta_scenario = Scenario(get_dt(df), benchmark_id)
    upper_lane_markings, lower_lane_markings = get_lane_markings(df)

    speed_limit = get_speed_limit(df)

    # add traffic sign to scenario
    speed_limit_sign_id = 100
    meta_scenario.add_objects(
        TrafficSign(
            traffic_sign_id=speed_limit_sign_id,
            traffic_sign_elements=[
                TrafficSignElement(TrafficSignIDGermany.MAX_SPEED, [str(speed_limit)])
            ],
            first_occurrence=set([]),
            position=np.array([0.0, 0.0]),
            virtual=True,
        ),
        lanelet_ids=set([]),
    )

    lanelet_type = LaneletType.HIGHWAY
    if is_up:
        for i in range(len(upper_lane_markings) - 1):
            # [0,-4,-8] instead of [-8, -12, -16]
            # get two lines of current lane
            next_lane_y = upper_lane_markings[i + 1]
            lane_y = upper_lane_markings[i]

            # get vertices of three lines
            # setting length 450 and -50 can cover all vehicle in this range
            right_vertices = resample_polyline(
                np.array([[450.0, lane_y], [-50.0, lane_y]])
            )
            left_vertices = resample_polyline(
                np.array([[450.0, next_lane_y], [-50.0, next_lane_y]])
            )
            center_vertices = (left_vertices + right_vertices) / 2.0

            # assign lane ids and adjacent ids
            lanelet_id = i + 1
            adjacent_left = lanelet_id + 1
            adjacent_right = lanelet_id - 1
            adjacent_left_same_direction = True
            adjacent_right_same_direction = True
            if i == 0:
                adjacent_right = None
            elif i == len(upper_lane_markings) - 2:
                adjacent_left = None

            # add lanelet to scenario
            meta_scenario.add_objects(
                Lanelet(
                    lanelet_id=lanelet_id,
                    left_vertices=left_vertices,
                    right_vertices=right_vertices,
                    center_vertices=center_vertices,
                    adjacent_left=adjacent_left,
                    adjacent_left_same_direction=adjacent_left_same_direction,
                    adjacent_right=adjacent_right,
                    adjacent_right_same_direction=adjacent_right_same_direction,
                    lanelet_type={lanelet_type},
                    traffic_signs={speed_limit_sign_id},
                )
            )
    else:

        for i in range(len(lower_lane_markings) - 1):

            # get two lines of current lane
            next_lane_y = lower_lane_markings[i + 1]
            lane_y = lower_lane_markings[i]

            # get vertices of three lines
            # setting length 450 and -50 can cover all vehicle in this range
            left_vertices = resample_polyline(
                np.array([[-50.0, lane_y], [450.0, lane_y]])
            )
            right_vertices = resample_polyline(
                np.array([[-50.0, next_lane_y], [450.0, next_lane_y]])
            )
            center_vertices = (left_vertices + right_vertices) / 2.0

            # assign lane ids and adjacent ids
            lanelet_id = i + 1 + len(upper_lane_markings) - 1
            adjacent_left = lanelet_id - 1
            adjacent_right = lanelet_id + 1
            adjacent_left_same_direction = True
            adjacent_right_same_direction = True
            if i == 0:
                adjacent_left = None
            elif i == len(lower_lane_markings) - 2:
                adjacent_right = None

            # add lanelet to scenario
            meta_scenario.add_objects(
                Lanelet(
                    lanelet_id=lanelet_id,
                    left_vertices=left_vertices,
                    right_vertices=right_vertices,
                    center_vertices=center_vertices,
                    adjacent_left=adjacent_left,
                    adjacent_left_same_direction=adjacent_left_same_direction,
                    adjacent_right=adjacent_right,
                    adjacent_right_same_direction=adjacent_right_same_direction,
                    lanelet_type={lanelet_type},
                    traffic_signs={speed_limit_sign_id},
                )
            )

    return meta_scenario, upper_lane_markings, lower_lane_markings
