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
import math
import os
import sys

from modules.map.proto import map_pb2, map_lane_pb2, map_road_pb2, map_geometry_pb2
from shapely.geometry import LineString, Point
from google.protobuf import text_format

LANE_WIDTH = 3.3
DEFAULT_SPEED_LIMIT = 20.0
DEFAULT_LANE_TYPE = map_lane_pb2.Lane.CITY_DRIVING
DEFAULT_LANE_DIRECTION = map_lane_pb2.Lane.FORWARD
DEFAULT_LANE_TURN = map_lane_pb2.Lane.NO_TURN
BOUNDARY_TYPE = map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW
ROAD_ID = "1"
SECTION_ID = "2"
LANE_SEGMENT_LENGTH = 100  # Estimated length of each lane segment


def calculate_offset_points(p1: Point, p2: Point, distance: float) -> Tuple[list[float], list[float]]:
    """Calculates the left and right offset points relative to a line segment.

    Args:
        p1: The starting point of the line segment.
        p2: The ending point of the line segment.
        distance: The offset distance.

    Returns:
        A tuple containing the coordinates of the left and right offset points.
    """
    delta_y = p2.y - p1.y
    delta_x = p2.x - p1.x
    angle = math.atan2(delta_y, delta_x)
    left_angle = angle + math.pi / 2.0
    right_angle = angle - math.pi / 2.0

    left_point = [p1.x + (math.cos(left_angle) * distance),
                  p1.y + (math.sin(left_angle) * distance)]
    right_point = [p1.x + (math.cos(right_angle) * distance),
                   p1.y + (math.sin(right_angle) * distance)]
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


def add_lane_points(path: LineString, lane: map_lane_pb2.Lane, central: map_geometry_pb2.CurveSegment,
                    left_boundary: map_geometry_pb2.CurveSegment, right_boundary: map_geometry_pb2.CurveSegment,
                    left_edge_segment: map_geometry_pb2.CurveSegment,
                    right_edge_segment: map_geometry_pb2.CurveSegment, index: int, extra_roi_extension: float):
    """Adds points to the lane, central line, and boundary lines.

    Args:
        path: The centerline path of the road.
        lane: The lane object.
        central: The central curve segment object.
        left_boundary: The left boundary curve segment object.
        right_boundary: The right boundary curve segment object.
        left_edge_segment: The left road boundary curve segment object.
        right_edge_segment: The right road boundary curve segment object.
        index: The current path index being processed.
    if index >= path.length:
        print(
            f"Warning: Index {index} exceeds path length {path.length}. Skipping point addition.")
        return

    p1 = path.interpolate(index)
    """
    p1 = path.interpolate(index)
    # Use an intermediate point to calculate the offset direction
    p2 = path.interpolate(index + 0.5)
    distance = LANE_WIDTH / 2.0
    left_point, right_point = calculate_offset_points(p1, p2, distance)

    # Add central point
    central_point = central.line_segment.point.add()
    central_point.x = p1.x
    central_point.y = p1.y

    # Add left and right boundary points
    left_bound_point = left_boundary.line_segment.point.add()
    left_bound_point.x = left_point[0]
    left_bound_point.y = left_point[1]

    right_bound_point = right_boundary.line_segment.point.add()
    right_bound_point.x = right_point[0]
    right_bound_point.y = right_point[1]

    # Add left and right road boundary points with extra extension
    edge_distance = distance + extra_roi_extension
    left_edge_point_offset, right_edge_point_offset = calculate_offset_points(
        p1, p2, edge_distance)

    left_edge_point = left_edge_segment.line_segment.point.add()
    left_edge_point.x = left_edge_point_offset[0]
    left_edge_point.y = left_edge_point_offset[1]

    right_edge_point = right_edge_segment.line_segment.point.add()
    right_edge_point.x = right_edge_point_offset[0]
    right_edge_point.y = right_edge_point_offset[1]

    # Add left and right samples
    left_sample = lane.left_sample.add()
    # Distance along the current segment
    left_sample.s = float(index % int(LANE_SEGMENT_LENGTH))
    left_sample.width = LANE_WIDTH / 2.0

    right_sample = lane.right_sample.add()
    # Distance along the current segment
    right_sample.s = float(index % int(LANE_SEGMENT_LENGTH))
    right_sample.width = LANE_WIDTH / 2.0


def initialize_road_section(map_object: map_pb2.Map) -> tuple[map_road_pb2.RoadSection,
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


def process_path(map_object: map_pb2.Map, path: LineString, extra_roi_extension: float):
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
    section, left_edge, left_edge_segment, right_edge, right_edge_segment = initialize_road_section(
        map_object)
    lane_id_counter = 0

    for i in range(length):
        if i % int(LANE_SEGMENT_LENGTH) == 0:
            lane_id_counter += 1
            if lane is not None:
                lane.successor_id.add().id = str(lane_id_counter)
                lane.length = central_points_count * 1.0  # Update previous lane's length

            lane, central, left_boundary, right_boundary = create_new_lane(
                map_object, lane_id_counter)
            section.lane_id.add().id = str(lane_id_counter)
            central_points_count = 0  # Reset counter for the new lane

            if i > 0:
                lane.predecessor_id.add().id = str(lane_id_counter - 1)

                # Add the first point, aligned with the last point of the previous lane
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

                left_sample = lane.left_sample.add()
                left_sample.s = 0.0
                left_sample.width = LANE_WIDTH / 2.0

                right_sample = lane.right_sample.add()
                right_sample.s = 0.0
                right_sample.width = LANE_WIDTH / 2.0

        add_lane_points(path, lane, central, left_boundary, right_boundary,
                        left_edge_segment, right_edge_segment, i, extra_roi_extension)
        central_points_count += 1

    if lane is not None:
        lane.length = central_points_count * 1.0  # Update the length of the last lane
        lane.left_boundary.length = lane.length
        lane.right_boundary.length = lane.length
        # TODO(zero): Use segment[0], Need to ensure that there is only one segment
        lane.central_curve.segment[0].length = lane.length


def read_points_from_file(filepath: str) -> list[tuple[float, float]]:
    """Reads path points from a file.

    Args:
        filepath: The path to the file.

    Returns:
        A list of tuples containing the coordinates of the path points.
    """
    points = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = line.split(',')
                if len(data) == 2:
                    try:
                        x = float(data[0])
                        y = float(data[1])
                        points.append((x, y))
                    except ValueError:
                        print(
                            f"Warning: Skipping line {f.tell()} with invalid coordinate format: {line}")
                elif line:  # Skip empty lines
                    print(
                        f"Warning: Skipping line with incorrect number of values: {line}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {filepath}")
        sys.exit(1)
    return points


def save_map_to_file(map_object: map_pb2.Map, output_filepath: str):
    """Saves the map data to text and binary files.

    Args:
        map_object: The map object.
        output_filepath: The output file path.
    """
    # Save as text format
    txt_filepath = os.path.join(output_filepath, "base_map.txt")
    try:
        with open(txt_filepath, 'w', encoding='utf-8') as f_txt:
            f_txt.write(text_format.MessageToString(map_object))
        print(f"Map text format saved to: {txt_filepath}")
    except Exception as e:
        print(f"Error: Could not convert map object to text format: {e}")

    # Save as binary format
    bin_filepath = os.path.join(output_filepath, "base_map.bin")
    try:
        # Ensure the directory exists
        with open(bin_filepath, 'wb') as f_bin:
            f_bin.write(map_object.SerializeToString())
        print(f"Map binary format saved to: {bin_filepath}")
    except IOError as e:
        print(f"Error: Could not save map binary file to {bin_filepath}: {e}")
