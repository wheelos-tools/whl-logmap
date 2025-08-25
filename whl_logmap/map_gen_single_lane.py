#!/usr/bin/env python3
"""
本模块提供单车道生成功能，
1. generate_lane_from_points：根据中心线点数据生成车道几何（中心线、左右边界），供 GUI 可视化使用；
2. generate_and_save_map：根据中心线点数据利用 protobuf 构造完整地图数据，并分别保存为 .txt（文本格式）和 .bin（二进制格式）文件；
3. load_map_from_file 与 extract_lane_geometry：支持从保存的地图文件中加载并提取车道几何数据。
"""

import math
from typing import List, Tuple
from shapely.geometry import LineString, Point

# 以下 protobuf 模块请确保环境中存在
from modules.map.proto import map_pb2, map_lane_pb2, map_road_pb2
from google.protobuf import text_format

LANE_WIDTH = 3.3

def convert(p: Point, p2: Point, distance: float) -> Tuple[List[float], List[float]]:
    """
    根据点 p 与 p2 定义的方向，计算 p 左右两侧偏移 distance 后的坐标。
    """
    delta_y = p2.y - p.y
    delta_x = p2.x - p.x
    left_angle = math.atan2(delta_y, delta_x) + math.pi / 2.0
    right_angle = math.atan2(delta_y, delta_x) - math.pi / 2.0
    lp = []
    lp.append(p.x + math.cos(left_angle) * distance)
    lp.append(p.y + math.sin(left_angle) * distance)

    rp = []
    rp.append(p.x + math.cos(right_angle) * distance)
    rp.append(p.y + math.sin(right_angle) * distance)
    return lp, rp

def shift(p: Point, p2: Point, distance: float, isleft: bool = True) -> Tuple[Point, Point]:
    """
    根据给定方向，计算 p 和 p2 沿左右方向平移 distance 后的点。
    此函数在原代码中定义但未使用，保留以确保功能完整性。
    """
    delta_y = p2.y - p.y
    delta_x = p2.x - p.x
    angle = math.atan2(delta_y, delta_x) + (math.pi / 2.0 if isleft else -math.pi / 2.0)
    p1n = [p.x + math.cos(angle) * distance, p.y + math.sin(angle) * distance]
    p2n = [p2.x + math.cos(angle) * distance, p2.y + math.sin(angle) * distance]
    return Point(p1n), Point(p2n)

def create_lane(map_obj, id: int):
    """
    根据 lane id 创建一个 lane，并返回 lane 及其中央、左边界和右边界的曲线段对象。
    """
    lane = map_obj.lane.add()
    lane.id.id = str(id)
    lane.type = map_lane_pb2.Lane.CITY_DRIVING
    lane.direction = map_lane_pb2.Lane.FORWARD
    lane.length = 100.0
    lane.speed_limit = 20.0
    lane.turn = map_lane_pb2.Lane.NO_TURN
    left_boundary = lane.left_boundary.curve.segment.add()
    right_boundary = lane.right_boundary.curve.segment.add()
    central = lane.central_curve.segment.add()
    central.length = 100.0

    t = lane.left_boundary.boundary_type.add()
    t.s = 0
    t.types.append(map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW)
    lane.right_boundary.length = 100.0

    t = lane.right_boundary.boundary_type.add()
    t.s = 0
    t.types.append(map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW)
    lane.left_boundary.length = 100.0

    return lane, central, left_boundary, right_boundary

def generate_lane_from_points(points: List[Tuple[float, float]],
                                lane_width: float = LANE_WIDTH,
                                step: float = 1.0,
                                sample_interval: float = 0.5,
                                extra_extension: float = 0.0
                                ) -> Tuple[List[Tuple[float, float]],
                                           List[Tuple[float, float]],
                                           List[Tuple[float, float]]]:
    """
    根据中心线点数据生成车道几何信息：中心线、左边界和右边界。
    
    :param points: 中心线坐标列表，至少包含两个 (x,y) 点
    :param lane_width: 车道总宽度，默认为 3.3
    :param step: 沿中心线采样的步长，默认为 1.0
    :param sample_interval: 用于计算方向的间隔，默认为 0.5
    :param extra_extension: 附加偏移（目前未参与计算，可扩展）
    :return: 三元组 (center_line, left_boundary, right_boundary)
    """
    if len(points) < 2:
        raise ValueError("至少需要两个点才能生成车道几何信息。")

    path = LineString(points)
    center_line = []
    left_boundary = []
    right_boundary = []
    
    d = 0.0
    while d <= path.length:
        p = path.interpolate(d)
        d2 = d + sample_interval
        if d2 > path.length:
            d2 = path.length
        p2 = path.interpolate(d2)
        center_line.append((p.x, p.y))
        lp, rp = convert(p, p2, lane_width / 2.0)
        left_boundary.append((lp[0], lp[1]))
        right_boundary.append((rp[0], rp[1]))
        d += step
        
    return center_line, left_boundary, right_boundary

def generate_and_save_map(points: List[Tuple[float, float]],
                          output_txt: str,
                          output_bin: str,
                          extra_extension: float = 0.0) -> None:
    """
    根据中心线点数据生成完整地图数据（使用 protobuf 构造），
    并分别保存为文本格式 (.txt) 和二进制格式 (.bin) 文件。
    
    :param points: 中心线坐标列表，至少包含两个 (x,y) 点
    :param output_txt: 保存文本格式地图数据的文件路径
    :param output_bin: 保存二进制格式地图数据的文件路径
    :param extra_extension: 附加偏移量，默认为 0.0
    """
    if len(points) < 2:
        raise ValueError("至少需要两个点才能生成地图。")
    
    path = LineString(points)
    length = int(path.length)
    
    map_obj = map_pb2.Map()
    road = map_obj.road.add()
    road.id.id = "1"
    section = road.section.add()
    section.id.id = "2"
    lane_obj = None
    lane_id = 0

    # 为 outer polygon 边界提前创建容器
    left_edge_segment = None
    right_edge_segment = None

    for i in range(length - 1):
        if i % 100 == 0:
            lane_id += 1
            if lane_obj is not None:
                lane_obj.successor_id.add().id = str(lane_id)
            lane_obj, central, left_boundary, right_boundary = create_lane(map_obj, lane_id)
            section.lane_id.add().id = str(lane_id)
            
            # 创建边界边
            left_edge = section.boundary.outer_polygon.edge.add()
            left_edge.type = map_road_pb2.BoundaryEdge.LEFT_BOUNDARY
            left_edge_segment = left_edge.curve.segment.add()
            
            right_edge = section.boundary.outer_polygon.edge.add()
            right_edge.type = map_road_pb2.BoundaryEdge.RIGHT_BOUNDARY
            right_edge_segment = right_edge.curve.segment.add()
            
            if i > 0:
                lane_obj.predecessor_id.add().id = str(lane_id - 1)
                
                left_bound_point = left_boundary.line_segment.point.add()
                right_bound_point = right_boundary.line_segment.point.add()
                central_point = central.line_segment.point.add()
                
                right_edge_point = right_edge_segment.line_segment.point.add()
                left_edge_point = left_edge_segment.line_segment.point.add()
                
                p = path.interpolate(i - 1)
                p2 = path.interpolate(i - 1 + 0.5)
                distance = LANE_WIDTH / 2.0
                
                lp, rp = convert(p, p2, distance)
                left_bound_point.x = lp[0]
                left_bound_point.y = lp[1]
                right_bound_point.x = rp[0]
                right_bound_point.y = rp[1]
                
                lp, rp = convert(p, p2, distance + extra_extension)
                left_edge_point.x = lp[0]
                left_edge_point.y = lp[1]
                right_edge_point.x = rp[0]
                right_edge_point.y = rp[1]
                
                central_point.x = p.x
                central_point.y = p.y
                
                left_sample = lane_obj.left_sample.add()
                left_sample.s = 0
                left_sample.width = LANE_WIDTH / 2.0
                
                right_sample = lane_obj.right_sample.add()
                right_sample.s = 0
                right_sample.width = LANE_WIDTH / 2.0

        # 对每个采样点添加曲线点
        left_bound_point = left_boundary.line_segment.point.add()
        right_bound_point = right_boundary.line_segment.point.add()
        central_point = central.line_segment.point.add()
        
        right_edge_point = right_edge_segment.line_segment.point.add()
        left_edge_point = left_edge_segment.line_segment.point.add()
        
        p = path.interpolate(i)
        p2 = path.interpolate(i + 0.5)
        distance = LANE_WIDTH / 2.0
        lp, rp = convert(p, p2, distance)
        
        central_point.x = p.x
        central_point.y = p.y
        left_bound_point.x = lp[0]
        left_bound_point.y = lp[1]
        right_bound_point.x = rp[0]
        right_bound_point.y = rp[1]
        
        left_edge_point.x = lp[0]
        left_edge_point.y = lp[1]
        right_edge_point.x = rp[0]
        right_edge_point.y = rp[1]
        
        left_sample = lane_obj.left_sample.add()
        left_sample.s = (i % 100) + 1
        left_sample.width = LANE_WIDTH / 2.0
        
        right_sample = lane_obj.right_sample.add()
        right_sample.s = (i % 100) + 1
        right_sample.width = LANE_WIDTH / 2.0

    # 保存为文本格式文件
    with open(output_txt, 'w') as f_txt:
        f_txt.write(text_format.MessageToString(map_obj))
    # 保存为二进制格式文件
    with open(output_bin, 'wb') as f_bin:
        f_bin.write(map_obj.SerializeToString())

def load_map_from_file(file_path: str):
    """
    根据文件扩展名，从指定路径加载地图（支持文本或二进制格式）。
    
    :param file_path: 地图文件路径（.txt 或 .bin）
    :return: 解析后的 map_pb2.Map 对象
    """
    m = map_pb2.Map()
    if file_path.endswith('.bin'):
        with open(file_path, 'rb') as f:
            bin_data = f.read()
        m.ParseFromString(bin_data)
    else:
        with open(file_path, 'r') as f:
            text_data = f.read()
        text_format.Merge(text_data, m)
    return m

def extract_lane_geometry(m) -> Tuple[List[Tuple[float, float]],
                                        List[Tuple[float, float]],
                                        List[Tuple[float, float]]]:
    """
    从加载的地图 proto 对象中提取车道几何数据（中心线、左边界和右边界）。
    这里假设地图中所有 lane 的曲线点均为连续显示。
    
    :param m: map_pb2.Map 对象
    :return: 三元组 (center_line, left_boundary, right_boundary)
    """
    center_line = []
    left_boundary = []
    right_boundary = []
    for lane in m.lane:
        # 提取中心线
        for seg in lane.central_curve.segment:
            for pt in seg.line_segment.point:
                center_line.append((pt.x, pt.y))
        # 提取左边界
        for seg in lane.left_boundary.curve.segment:
            for pt in seg.line_segment.point:
                left_boundary.append((pt.x, pt.y))
        # 提取右边界
        for seg in lane.right_boundary.curve.segment:
            for pt in seg.line_segment.point:
                right_boundary.append((pt.x, pt.y))
    return center_line, left_boundary, right_boundary

# 当直接以命令行方式调用时，支持传入参数生成并保存地图
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: {} <input_points_file> <output_txt_file> <output_bin_file> [extra_extension]".format(sys.argv[0]))
        sys.exit(1)
    input_file = sys.argv[1]
    output_txt = sys.argv[2]
    output_bin = sys.argv[3]
    extra_extension = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.0

    pts = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    pts.append((x, y))
                except Exception:
                    continue
    try:
        center_line, left_boundary, right_boundary = generate_lane_from_points(pts, extra_extension=extra_extension)
        print("生成车道几何：中心线点数 = {}, 左边界点数 = {}, 右边界点数 = {}".format(
            len(center_line), len(left_boundary), len(right_boundary)))
        generate_and_save_map(pts, output_txt, output_bin, extra_extension=extra_extension)
        print("地图已保存为 {} (txt) 和 {} (bin)".format(output_txt, output_bin))
    except Exception as e:
        print("Error:", e)
