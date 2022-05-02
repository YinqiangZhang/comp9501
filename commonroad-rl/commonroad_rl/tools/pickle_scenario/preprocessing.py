from collections import defaultdict

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle

from commonroad_rl.gym_commonroad.utils.scenario import get_road_edge

from commonroad_route_planner.route_planner import sorted_lanelet_ids


def get_all_connected_lanelets(scenario: Scenario) -> dict:
    """
    Create all possible lanes by merging predecessors and successors, then create a dict with its keys as lanelet id
    and values as connected lanelet ids.
    :return: dict
    """
    merged_lanelet_dict = defaultdict(set)
    for (
        l
    ) in scenario.lanelet_network.lanelets:  # iterate in all lanelet in this scenario
        if not l.predecessor and not l.successor:  # the lanelet is a lane itself
            merged_lanelet_dict[l.lanelet_id].add(l.lanelet_id)
        elif not l.predecessor:
            max_lane_merge_range = 1000.0
            (
                _,
                sub_lanelet_ids,
            ) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                l, scenario.lanelet_network, max_lane_merge_range
            )
            for s in sub_lanelet_ids:
                for i in s:
                    merged_lanelet_dict[i].update(s)
    return merged_lanelet_dict


# def add_initial_state_to_trajectory(scenario_path: str, scenario_name: str) -> None:
#     """
#     Add the initial state to trajectory prediction of a dynamic obstacle and write it as an XML file.
#     :param scenario_path: path of the scenario
#     :param scenario_name: name of the scenario
#     :return: None
#     """
#     scenario, planning_problem_set = read_scenario(scenario_path, scenario_name)
#     all_obstacles = scenario.dynamic_obstacles
#     for o in all_obstacles:
#         init_state = o.initial_state
#         try:
#             state_list = o.prediction.trajectory.state_list
#         except AttributeError:
#             state_list = []
#         state_list.insert(0, init_state)
#         veh_trajectory = Trajectory(initial_time_step=0, state_list=state_list)
#         o.prediction = TrajectoryPrediction(trajectory=veh_trajectory, shape=o.obstacle_shape)
#     fw = CommonRoadFileWriter(scenario, planning_problem_set, "", "", "", "")
#     fw.write_to_file(scenario_name, OverwriteExistingFile.ALWAYS)


def generate_obstacle_lanelet_id(scenario: Scenario) -> dict:
    """
    Generate a dict with its key as id of dynamic obstacles and value as a list of lanelet ids of the obstacles.
    :param scenario: commonroad scenario
    :return: dict of obstacle lanelet ids
    """
    lanelet_id_dict = dict()
    for o in scenario.dynamic_obstacles:
        # print(o)
        ids = []
        count = 0
        for i in range(
            o.initial_state.time_step, o.prediction.trajectory.final_state.time_step + 1
        ):
            obst_state = o.state_at_time(i)
            ids_candidate = scenario.lanelet_network.find_lanelet_by_position(
                [obst_state.position]
            )[0]
            if len(ids_candidate) > 1:
                lanelet_id = sorted_lanelet_ids(
                    ids_candidate, obst_state.orientation, obst_state.position, scenario
                )[0]
                ids.append(lanelet_id)
            elif len(ids_candidate) == 1:
                ids.append(ids_candidate[0])
            else:
                if len(ids) != 0:
                    ids.append(ids[-1])
                else:
                    count = count + 1
        # if some vehicle off road initially,we add the lanelet index for them later
        if count != 0:
            # if some vehicle off road always, remove it!
            if len(ids) == 0:
                print(
                    f"Obstacle {o.obstacle_id} in {scenario.benchmark_id} always off road"
                )
                # scenario.remove_obstacle(o)
                # fw = CommonRoadFileWriter(scenario, planning_problem_set, AUTHOR, AFFILIATION, SOURCE, TAGS)
                # filename = os.path.join(PATH_PARAMS["scenario_backup"], "{}.xml".format(scenario.benchmark_id))
                # fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
                continue
            ids2 = []
            num = ids[0]
            for i in range(count):
                ids2.append(num)
            ids = ids2 + ids
        assert (
            len(ids)
            == o.prediction.trajectory.final_state.time_step
            - o.initial_state.time_step
            + 1
        )
        lanelet_id_dict[o.obstacle_id] = ids
    for o in scenario.static_obstacles:
        ids_candidate = scenario.lanelet_network.find_lanelet_by_position(
            [o.initial_state.position]
        )[0]
        if len(ids_candidate) == 0:
            # if some vehicle off road always, remove it!
            print(
                f"Obstacle {o.obstacle_id} in {scenario.benchmark_id} always off road"
            )
            continue
        elif len(ids_candidate) == 1:
            lanelet_id_dict[o.obstacle_id] = ids_candidate.copy()
        else:
            lanelet_id = sorted_lanelet_ids(
                ids_candidate,
                o.initial_state.orientation,
                o.initial_state.position,
                scenario,
            )[0]
            lanelet_id_dict[o.obstacle_id] = [lanelet_id]
    return lanelet_id_dict


def generate_reset_config(scenario: Scenario) -> dict:
    """
    Generate a dict of reset configurations which contains obstacle lanelet ids, road edge, collision checker,
    lanelet boundary and lanelet connection dict.
    :param scenario: commonroad scenario
    :return:
    """
    (
        left_road_edge_lanelet_id,
        left_road_edge,
        right_road_edge_lanelet_id,
        right_road_edge,
    ) = get_road_edge(scenario)
    _, lanelet_boundary = create_road_boundary_obstacle(
        scenario, method="obb_rectangles"
    )
    connected_lanelet_dict = get_all_connected_lanelets(scenario)

    reset_config = {
        "left_road_edge_lanelet_id_dict": left_road_edge_lanelet_id,
        "left_road_edge_dict": left_road_edge,
        "right_road_edge_lanelet_id_dict": right_road_edge_lanelet_id,
        "right_road_edge_dict": right_road_edge,
        "boundary_collision_object": lanelet_boundary,
        "connected_lanelet_dict": connected_lanelet_dict,
    }
    return reset_config
