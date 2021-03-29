from envs.cityflow.road import IncomingDirection, Lane, Road, RoadLink
from envs.cityflow.intersections import Intersection
from typing import Dict, List
import math
import json


def id_shortcut_parased(id_):
    if id_ == "1x3":
        id_ = "syn_1x3_gaussian_500_1h"
    elif id_ == "1x1":
        id_ = "hangzhou_1x1_bc-tyc_18041607_1h"
    elif id_ == "2x2":
        id_ = "syn_2x2_gaussian_500_1h"
    elif id_ == "1x4":
        id_ = "LA_1x4"
    elif id_ == "1x5":
        id_ = "atlanta_1x5"
    elif id_ == "3x4":
        id_ = "ji_nan_3_4"
    # 都不匹配则直接返回
    return id_


def parase_roadnet(
    roadnet_file_path,
    flow_info,
) -> Dict[str, Intersection]:
    with open(roadnet_file_path, "r") as f:
        roadnet_json = json.load(f)
    total_roads_info = parase_roads_json(
        roadnet_json["roads"],
        flow_info["vehicle"]["proportion"],
    )

    inters = {}
    for inter_json in roadnet_json["intersections"]:
        if inter_json['virtual']:
            continue
        phase_plan = parase_phase_plan(
            inter_json["trafficLight"]["lightphases"])
        if len(phase_plan) == 1:
            continue
        inter_roads_id = inter_json["roads"]
        roads_info = {}
        for id in inter_roads_id:
            roads_info[id] = total_roads_info[id]
        roadlinks, roads = parase_roadlink(
            inter_json["roadLinks"],
            roads_info,
        )

        inter_id = inter_json["id"]

        inters[inter_id] = Intersection(
            id=inter_id,
            phase_plan=phase_plan,
            roadlinks=roadlinks,
            roads=roads,
        )
    return inters


def parase_flow(flow_file_path):
    with open(flow_file_path, "r") as f:
        flow_json = json.load(f)
    flow = {}
    vehicle_json = flow_json[0]["vehicle"]
    vehicle_proportion = float(vehicle_json["length"]) + float(
        vehicle_json["minGap"])
    max_time = flow_json[-1]["endTime"]
    flow["vehicle"] = {"proportion": vehicle_proportion}
    flow["max_time"] = max_time + 10

    return flow


def parase_roads_json(roads_json, vehicle_proportion):
    roads_info = {}
    for r_json in roads_json:
        id_ = r_json["id"]
        points = r_json["points"]
        x1 = float(points[0]["x"])
        y1 = float(points[0]["y"])
        x2 = float(points[1]["x"])
        y2 = float(points[1]["y"])
        length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        lanes_id = []
        max_speed = r_json["lanes"][0]["maxSpeed"]
        for i in range(len(r_json["lanes"])):
            lanes_id.append(f"{id_}_{i}")

        roads_info[id_] = {
            "length": length,
            "start": r_json["startIntersection"],
            "end": r_json["endIntersection"],
            "lanes_id": lanes_id,
            "lane_capacity": max(int(length / vehicle_proportion), 1),
            "max_speed": max_speed,
        }

    return roads_info


def parase_roadlink(roadlinks_json, roads_info):

    roadlinks_info = []
    lanes_incoming_type = {}
    for roadlink_json in roadlinks_json:

        start_road_id = roadlink_json["startRoad"]
        end_road_id = roadlink_json["endRoad"]

        incoming_type = IncomingDirection(roadlink_json["type"])

        lane_links_json = roadlink_json["laneLinks"]
        # store a pair (start index, end index)
        lane_links = []
        for lane_json in lane_links_json:
            start_lane_id = "{}_{}".format(start_road_id,
                                           lane_json["startLaneIndex"])
            end_lane_id = "{}_{}".format(end_road_id,
                                         lane_json["endLaneIndex"])
            lane_links.append([start_lane_id, end_lane_id])
            if start_lane_id not in lanes_incoming_type.keys():
                lanes_incoming_type[start_lane_id] = incoming_type

        roadlinks_info.append({
            "start": start_road_id,
            "end": end_road_id,
            "lane_links": lane_links,
            "link_type": incoming_type,
        })

    roads: Dict[str, Road] = {}
    for road_id, info in roads_info.items():
        lanes: Dict[str, Lane] = {}
        for lane_id in info["lanes_id"]:
            incoming_type = lanes_incoming_type.get(lane_id, None)
            lanes[lane_id] = Lane(
                id=lane_id,
                belonged_road=road_id,
                capacity=info["lane_capacity"],
                incoming_type=incoming_type,
                max_speed=info["max_speed"],
            )
        road = Road(
            id=road_id,
            start=info["start"],
            end=info["end"],
            lanes=lanes,
        )
        roads[road_id] = road

    roadlinks: List[RoadLink] = []
    for rlink_info in roadlinks_info:

        start_road = roads[rlink_info["start"]]
        end_road = roads[rlink_info["end"]]
        lane_links: List[List[str]] = rlink_info["lane_links"]
        incoming_type = rlink_info["link_type"]
        roadlink = RoadLink(
            start_road=start_road,
            end_road=end_road,
            lane_links=lane_links,
            movement=incoming_type,
        )
        roadlinks.append(roadlink)

    return roadlinks, roads


def parase_phase_plan(phase_plan_json):
    phase_plan = []
    for phase_json in phase_plan_json:
        phase_plan.append(phase_json["availableRoadLinks"])
    return phase_plan
