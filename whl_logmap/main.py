#!/usr/bin/env python

# Copyright 2025 daohu527 <daohu527@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import argparse
import logging
import matplotlib.pyplot as plt 

from modules.map.proto import map_pb2
from shapely.geometry import LineString

from extract_path import SortMode, extract_path, get_sorted_records
from map_gen import read_points_from_file, save_map_to_file, process_path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def plot_points(points: list[tuple[float, float]], output_dir: str):  # 新增函数
    """Plots a list of (x, y) points using matplotlib and saves to output_dir."""
    if not points:
        logging.warning("No points to plot.")
        return

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Path Points")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    output_path = os.path.join(output_dir, 'output.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 防止内存泄漏
    logging.info(f"轨迹图已保存至 {output_path}")

def main(args=None):
    """Main function to process the path and generate map data."""
    parser = argparse.ArgumentParser(
        description="Generates map data from a path file.")
    parser.add_argument(
        "--input_path", type=str, help="Path to the input file containing path points (CSV format: x,y).")
    parser.add_argument(
        "--output_path", type=str, default="map", help="Path to the output map file (will create .txt and .bin files).")
    parser.add_argument("--extra_roi_extension", type=float, default=0.3,
                        help="Extra ROI extension distance.")

    parsed_args = parser.parse_args(args)

    input_path = parsed_args.input_path
    output_path = parsed_args.output_path
    extra_roi_extension = parsed_args.extra_roi_extension

    try:
        logging.info(f"Processing input path: {input_path}")
        record_files = get_sorted_records(input_path, sort_mode=SortMode.TIME)

        if not os.path.exists(output_path):
            logging.info(f"Output path '{output_path}' not exists. Createing it.")
            os.makedirs(output_path, exist_ok=True)
        logging.info(f"Output path created: {output_path}")
        output_path_file = os.path.join(output_path, "path.txt")
        logging.info(f"Extracting path to: {output_path_file}")
        extract_path(record_files, output_path_file)

        logging.info(f"Reading path points from: {output_path_file}")
        path_points = read_points_from_file(output_path_file)

        if not path_points:
            logging.error(
                "Could not read any valid path points from the input file.")
            sys.exit(1)

        path = LineString(path_points)
        logging.info(f"Path created with {len(path_points)} points.")

        map_data = map_pb2.Map()
        logging.info("Processing path to generate map data.")
        process_path(map_data, path, extra_roi_extension)

        logging.info(f"Saving map data to: {output_path}")
        save_map_to_file(map_data, output_path)

        logging.info("Generating path visualization...")
        plot_points(path_points, output_path)
        
        print("Map data generation complete.")

    except FileNotFoundError:
        logging.error(f"Input file not found at '{input_path}'.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
