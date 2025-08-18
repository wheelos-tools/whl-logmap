#!/usr/bin/env python3

###############################################################################
# Copyright 2017 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from typing import Tuple
import numpy as np

from modules.map.proto import map_pb2, map_lane_pb2, map_road_pb2, map_geometry_pb2
from shapely.geometry import LineString, Point
from google.protobuf import text_format

LANE_WIDTH = 3.75
DEFAULT_SPEED_LIMIT = 20.0
DEFAULT_LANE_TYPE = map_lane_pb2.Lane.CITY_DRIVING
DEFAULT_LANE_DIRECTION = map_lane_pb2.Lane.FORWARD
DEFAULT_LANE_TURN = map_lane_pb2.Lane.NO_TURN
BOUNDARY_TYPE = map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW
ROAD_ID = "1"
SECTION_ID = "2"
LANE_SEGMENT_LENGTH = 100  # Estimated length of each lane segment


def calculate_offset_points(p1: Point, p2: Point, distance: float) -> Tuple[list[float], list[float]]:
    vector = np.array([p2.x - p1.x, p2.y - p1.y])
    norm = np.linalg.norm(vector)
    if norm == 0:
        # Handle the case where p1 and p2 are the same
        return [p1.x, p1.y], [p1.x, p1.y]

    unit_vector = vector / norm
    normal_vector = np.array([-unit_vector[1], unit_vector[0]])  # Left normal

    left_point = [p1.x + normal_vector[0] * distance,
                  p1.y + normal_vector[1] * distance]
    right_point = [p1.x - normal_vector[0] * distance,
                   p1.y - normal_vector[1] * distance]  # Right normal is the opposite

    return left_point, right_point


def create_new_lane(map_object: map_pb2.Map, lane_id: int) -> tuple[map_lane_pb2.Lane,
                                                                    map_geometry_pb2.CurveSegment,
                                                                    map_geometry_pb2.CurveSegment,
                                                                    map_geometry_pb2.CurveSegment]:
    """Creates a new lane object and initializes basic attributes.

    Args:
        map_object: The map object.
        lane_id: The lane ID.

    Returns:
        A tuple containing the lane object, central curve segment, left boundary curve segment,
        and right boundary curve segment.
    """
    lane = map_object.lane.add()
    lane.id.id = str(lane_id)
    lane.type = DEFAULT_LANE_TYPE
    lane.direction = DEFAULT_LANE_DIRECTION
    lane.length = 0.0  # Initialize length to 0, will be updated later
    lane.speed_limit = DEFAULT_SPEED_LIMIT
    lane.turn = DEFAULT_LANE_TURN

    left_boundary = lane.left_boundary.curve.segment.add()
    right_boundary = lane.right_boundary.curve.segment.add()
    central = lane.central_curve.segment.add()
    central.length = 0.0  # Initialize length to 0, will be updated later

    # Set boundary types
    left_boundary_type = lane.left_boundary.boundary_type.add()
    left_boundary_type.s = 0
    left_boundary_type.types.append(BOUNDARY_TYPE)
    lane.left_boundary.length = 0.0  # Initialize length to 0, will be updated later

    right_boundary_type = lane.right_boundary.boundary_type.add()
    right_boundary_type.s = 0
    right_boundary_type.types.append(BOUNDARY_TYPE)
    lane.right_boundary.length = 0.0  # Initialize length to 0, will be updated later

    return lane, central, left_boundary, right_boundary


def create_road_section(map_object: map_pb2.Map) -> tuple[map_road_pb2.RoadSection,
                                                          map_road_pb2.BoundaryEdge,
                                                          map_geometry_pb2.CurveSegment,
                                                          map_road_pb2.BoundaryEdge,
                                                          map_geometry_pb2.CurveSegment]:
    """Initializes road and road section information.

    Args:
        map_object: The map object.

    Returns:
        A tuple containing the road section object, left boundary edge object, left boundary curve segment object,
        right boundary edge object, and right boundary curve segment object.
    """
    road = map_object.road.add()
    road.id.id = ROAD_ID
    section = road.section.add()
    section.id.id = SECTION_ID

    left_edge = section.boundary.outer_polygon.edge.add()
    left_edge.type = map_road_pb2.BoundaryEdge.LEFT_BOUNDARY
    left_edge_segment = left_edge.curve.segment.add()

    right_edge = section.boundary.outer_polygon.edge.add()
    right_edge.type = map_road_pb2.BoundaryEdge.RIGHT_BOUNDARY
    right_edge_segment = right_edge.curve.segment.add()

    return section, left_edge, left_edge_segment, right_edge, right_edge_segment


def _add_points_at_index(path: LineString, index: int, central: map_geometry_pb2.CurveSegment,
                         left_boundary: map_geometry_pb2.CurveSegment, right_boundary: map_geometry_pb2.CurveSegment,
                         left_edge_segment: map_geometry_pb2.CurveSegment,
                         right_edge_segment: map_geometry_pb2.CurveSegment, extra_roi_extension: float):
    if index >= path.length:
        print(
            f"Warning: Index {index} exceeds path length {path.length}. Skipping point addition.")
        return

    p1 = path.interpolate(index)
    p2 = path.interpolate(index + 0.5)
    distance = LANE_WIDTH / 2.0
    left_point, right_point = calculate_offset_points(p1, p2, distance)

    # 添加中心点
    central_point = central.line_segment.point.add()
    central_point.x = p1.x
    central_point.y = p1.y

    # 添加左右边界点
    left_bound_point = left_boundary.line_segment.point.add()
    left_bound_point.x = left_point[0]
    left_bound_point.y = left_point[1]

    right_bound_point = right_boundary.line_segment.point.add()
    right_bound_point.x = right_point[0]
    right_bound_point.y = right_point[1]

    # 添加左右道路边界点
    edge_distance = distance + extra_roi_extension
    left_edge_point_offset, right_edge_point_offset = calculate_offset_points(
        p1, p2, edge_distance)

    left_edge_point = left_edge_segment.line_segment.point.add()
    left_edge_point.x = left_edge_point_offset[0]
    left_edge_point.y = left_edge_point_offset[1]

    right_edge_point = right_edge_segment.line_segment.point.add()
    right_edge_point.x = right_edge_point_offset[0]
    right_edge_point.y = right_edge_point_offset[1]


def _add_lane_samples(lane: map_lane_pb2.Lane, index: int):
    left_sample = lane.left_sample.add()
    left_sample.s = float(index % int(LANE_SEGMENT_LENGTH))
    left_sample.width = LANE_WIDTH / 2.0

    right_sample = lane.right_sample.add()
    right_sample.s = float(index % int(LANE_SEGMENT_LENGTH))
    right_sample.width = LANE_WIDTH / 2.0


def check_loopback(first_lane: map_lane_pb2.Lane, current_lane: map_lane_pb2.Lane, detection_threshold=1.0) -> bool:
    if first_lane is None or current_lane is None:
        return False
    if int(current_lane.id.id) < 2:
        return False

    # Use first_lane point[2], because we need the paths partially overlap
    start_point = first_lane.central_curve.segment[0].point[2]
    end_point = current_lane.central_curve.segment[0].point[-1]
    dist = np.sqrt((start_point.x - end_point.x)**2 +
                   (start_point.y - end_point.y)**2)
    if dist < detection_threshold:
        return True
    return False


def process_path(map_object: map_pb2.Map, path: LineString, extra_roi_extension: float, enable_loopback: bool = True):
    """Processes the path, creating lane and road structures.

    Args:
        map_object: The map object.
        path: The centerline path of the road.
        extra_roi_extension: The extra ROI extension distance.
    """
    if not isinstance(path.length, (int, float)) or path.length <= 0:
        print(
            f"Error: Invalid path length: {path.length}. Path length must be a positive numeric value.")
        return
    length = int(path.length)
    lane = None
    section, left_edge, left_edge_segment, right_edge, right_edge_segment = create_road_section(
        map_object)
    lane_id_counter = 0
    first_lane = None

    for i in range(length):
        if i % int(LANE_SEGMENT_LENGTH) == 0:
            lane_id_counter += 1
            if lane is not None:
                lane.successor_id.add().id = str(lane_id_counter)
                lane.length = central_points_count * 1.0  # Update previous lane's length

            lane, central, left_boundary, right_boundary = create_new_lane(
                map_object, lane_id_counter)
            if first_lane is None:
                first_lane = lane

            section.lane_id.add().id = str(lane_id_counter)
            central_points_count = 0  # Reset counter for the new lane

            if i > 0:
                lane.predecessor_id.add().id = str(lane_id_counter - 1)

                # Add the first point of the new lane
                prev_p = path.interpolate(i - 1)
                prev_p2 = path.interpolate(i - 1 + 0.5)
                distance = LANE_WIDTH / 2.0
                lp, rp = calculate_offset_points(prev_p, prev_p2, distance)

                left_bound_point = left_boundary.line_segment.point.add()
                left_bound_point.y = lp[1]
                left_bound_point.x = lp[0]
                right_bound_point = right_boundary.line_segment.point.add()
                right_bound_point.y = rp[1]
                right_bound_point.x = rp[0]
                central_point = central.line_segment.point.add()
                central_point.x = prev_p.x
                central_point.y = prev_p.y
                central_points_count += 1

                edge_distance = distance + extra_roi_extension
                lp_edge, rp_edge = calculate_offset_points(
                    prev_p, prev_p2, edge_distance)
                left_edge_point = left_edge_segment.line_segment.point.add()
                left_edge_point.y = lp_edge[1]
                left_edge_point.x = lp_edge[0]
                right_edge_point = right_edge_segment.line_segment.point.add()
                right_edge_point.y = rp_edge[1]
                right_edge_point.x = rp_edge[0]

                # For the first point of the new lane
                _add_lane_samples(lane, 0)

        _add_points_at_index(path, i, central, left_boundary, right_boundary,
                             left_edge_segment, right_edge_segment, extra_roi_extension)
        _add_lane_samples(lane, i)
        central_points_count += 1

        if lane is not None:
            lane.length = central_points_count * 1.0  # Update the length of the last lane
            lane.left_boundary.length = lane.length
            lane.right_boundary.length = lane.length
            # TODO(zero): Use segment[0], Need to ensure that there is only one segment
            lane.central_curve.segment[0].length = lane.length

        if enable_loopback and check_loopback(first_lane, lane):
            lane.successor_id.add().id = first_lane.id.id
            first_lane.predecessor_id.add().id = lane.id.id
            break
