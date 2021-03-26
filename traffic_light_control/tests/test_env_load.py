from envs.enum import Movement
import json
import math
import unittest
from envs.factory import _parase_flow, _parase_roads_json, _parase_phase_plan, _parase_roadlink, _parase_roadnet


class TestEnvLoad(unittest.TestCase):
    def setUp(self) -> None:
        cityflow_config_dir = "cityflow_config/hangzhou_1x1_bc-tyc_18041607_1h/"
        self.roadnet_file = cityflow_config_dir + "roadnet.json"
        self.flow_file = cityflow_config_dir + "flow.json"

    def test_parase_flow(self):
        flow_info = _parase_flow(self.flow_file)
        vehicle_proportion = flow_info["vehicle"]["proportion"]
        max_time = flow_info['max_time']
        self.assertEqual(vehicle_proportion, 7.5)
        self.assertNotEqual(vehicle_proportion, 1.0)
        self.assertEqual(max_time, 3602)

    def test_parase_roads_info(self):
        with open(self.roadnet_file, "r") as f:
            roadnet_json = json.load(f)["roads"]
        vehicle_proprotion = 7.5
        roads_info = _parase_roads_json(roadnet_json, vehicle_proprotion)
        road_1_id = "road_0_1_0"
        road_1_info = roads_info[road_1_id]
        road_1_length = math.sqrt(
            math.pow((-300 - 0), 2) + math.pow((0 - 0), 2))
        road_1_lanes = ["road_0_1_0_0", "road_0_1_0_1"]
        self.assertEqual(road_1_info["start"], "intersection_0_1")
        self.assertEqual(road_1_info["end"], "intersection_1_1")
        self.assertAlmostEqual(road_1_info["length"], road_1_length)
        self.assertAlmostEqual(road_1_info["lane_capacity"],
                               road_1_length // vehicle_proprotion)
        self.assertListEqual(road_1_info["lanes_id"], road_1_lanes)

    def test_parase_phase_plan(self):
        with open(self.roadnet_file, "r") as f:
            roadnet_json = json.load(f)
        intersection_json = roadnet_json["intersections"][2]
        phase_plan_json = intersection_json["trafficLight"]["lightphases"]
        phase_plan = _parase_phase_plan(phase_plan_json)
        phase_0 = []
        phase_1 = [0, 4]
        phase_2 = [2, 7]
        self.assertListEqual(phase_plan[0], phase_0)
        self.assertListEqual(phase_plan[1], phase_1)
        self.assertListEqual(phase_plan[2], phase_2)

    def test_parase_roadlink(self):
        with open(self.roadnet_file, "r") as f:
            roadnet_json = json.load(f)
        vehicle_proprotion = 7.5
        roads_info = _parase_roads_json(roadnet_json["roads"],
                                        vehicle_proprotion)
        intersection_json = roadnet_json["intersections"][2]
        roadlink_json = intersection_json["roadLinks"]
        roadlinks, roads = _parase_roadlink(
            roadlinks_json=roadlink_json,
            roads_info=roads_info,
        )
        road_1_id = "road_0_1_0"
        road_1 = roads[road_1_id]
        road_1_lanes = ["road_0_1_0_0", "road_0_1_0_1"]
        self.assertEqual(road_1.start, "intersection_0_1")
        self.assertEqual(road_1.end, "intersection_1_1")
        self.assertIsNotNone(road_1.get_lane(road_1_lanes[0]))
        self.assertIsNotNone(road_1.get_lane(road_1_lanes[1]))
        roadlink_1 = roadlinks[0]
        self.assertEqual(roadlink_1.start_road.id, "road_0_1_0")
        self.assertEqual(roadlink_1.end_road.id, "road_1_1_0")
        self.assertEqual(roadlink_1.in_direction, Movement.STRAIGHT)
        self.assertListEqual(roadlink_1.lane_links[0],
                             ["road_0_1_0_1", "road_1_1_0_0"])
        self.assertListEqual(roadlink_1.lane_links[1],
                             ["road_0_1_0_1", "road_1_1_0_1"])

    def test_parase_roadnet(self):
        flow_info = _parase_flow(self.flow_file)
        intersections = _parase_roadnet(self.roadnet_file, flow_info)
        intersection_1_1 = intersections["intersection_1_1"]
        road_id_1 = "road_0_1_0"
        self.assertTrue(road_id_1 in intersection_1_1.get_roads())

        phase_plan = intersection_1_1.phase_plan
        rlinks = intersection_1_1.roadlinks
        phase_1 = phase_plan[0]
        self.assertFalse(phase_1)

        phase_2 = phase_plan[1]
        rlink_1 = rlinks[phase_2[0]]
        rlink_2 = rlinks[phase_2[1]]
        self.assertEqual(rlink_1.start_road.id, "road_0_1_0")
        self.assertEqual(rlink_1.end_road.id, "road_1_1_0")
        self.assertListEqual(rlink_1.lane_links[0],
                             ["road_0_1_0_1", "road_1_1_0_0"])
        self.assertListEqual(rlink_1.lane_links[1],
                             ["road_0_1_0_1", "road_1_1_0_1"])
        self.assertEqual(rlink_1.in_direction, Movement.STRAIGHT)
        self.assertEqual(rlink_2.start_road.id, "road_2_1_2")
        self.assertEqual(rlink_2.end_road.id, "road_1_1_2")
        self.assertEqual(rlink_2.in_direction, Movement.STRAIGHT)
        self.assertListEqual(rlink_2.lane_links[0],
                             ["road_2_1_2_1", "road_1_1_2_0"])
        self.assertListEqual(rlink_2.lane_links[1],
                             ["road_2_1_2_1", "road_1_1_2_1"])


if __name__ == "__main__":
    unittest.main()