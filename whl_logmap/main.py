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

"""Main module for generating map data from path files.

This module provides the main entry point for processing trajectory data
and generating map files with optional curvature constraints.
"""

import argparse
import logging
import os
import sys
from typing import List
import numpy as np

from modules.map.proto import map_pb2
from shapely.geometry import LineString

from whl_logmap import map_gen, plots, preprocess, utils
from whl_logmap.extract_path import SortMode, extract_path, get_sorted_records

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def path_params_type(value: str) -> List[float]:
    """
    Custom argparse type to parse and validate comma-separated path parameters.
    Format: "x,y,heading[,forward,backward,spacing]"
    """
    try:
        parts = [float(p.strip()) for p in value.split(",")]
        if not (3 <= len(parts) <= 6):
            raise ValueError()
        return parts
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "Invalid format for path_params. Expected 'x,y,heading[,forward,backward,spacing]' "
            "with 3 to 6 numeric values."
        )


def main(args=None):
    """Main function to process the path and generate map data."""
    parser = argparse.ArgumentParser(
        description="Generates map data from a dynamically created straight line path.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="Path to the input file containing path points (CSV format: x,y).",
    )
    group.add_argument(
        "--path_params",
        type=path_params_type,
        help=(
            "Comma-separated parameters for dynamic path generation.\n"
            "Format: 'x,y,heading[,forward,backward,spacing]'\n"
            "Example: '100.0,200.0,45.0,100,10,0.25'"
        ),
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="map",
        help="Path to the output map file (will create .txt and .bin files).",
    )
    parser.add_argument(
        "--extra_roi_extension",
        type=float,
        default=0.3,
        help="Extra ROI extension distance.",
    )
    parser.add_argument(
        "--enable_loopback", type=bool, default=False, help="Enable loopback detection."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of path.txt even if it already exists.",
    )
    parser.add_argument(
        "--max_curvature",
        type=float,
        default=None,
        help="Maximum allowed curvature (default: None, no curvature constraint). "
        "Recommended values: 95% of vehicle limit.",
    )
    parser.add_argument(
        "--curvature_method",
        type=str,
        default="strict",
        choices=["adaptive", "iterative", "strict"],
        help="Curvature constraint method (default: strict, only used when "
        "--max_curvature is specified). Available methods: "
        "adaptive (Savitzky-Golay based), iterative (adaptive smoothing), strict (aggressive smoothing)",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum number of iterations for curvature constraint algorithms "
        "(default: 100, only used when --max_curvature is specified). "
        "If curvature constraints are not satisfied, try increasing this value.",
    )

    parsed_args = parser.parse_args(args)

    input_path = parsed_args.input_path
    output_path = parsed_args.output_path
    extra_roi_extension = parsed_args.extra_roi_extension
    force = parsed_args.force
    enable_loopback = parsed_args.enable_loopback
    max_curvature = parsed_args.max_curvature
    curvature_method = parsed_args.curvature_method
    max_iterations = parsed_args.max_iterations

    try:
        if not os.path.exists(output_path):
            logging.info(f"Output path '{output_path}' not exists. Creating it.")
            os.makedirs(output_path, exist_ok=True)

        if parsed_args.input_path:
            logging.info(f"Processing input path: {input_path}")
            record_files = get_sorted_records(input_path, sort_mode=SortMode.NAME)

            output_file = os.path.join(output_path, "path.txt")
            if not force and os.path.exists(output_file):
                logging.info(
                    f"path.txt already exists at '{output_file}', skipping extraction."
                )
            else:
                logging.info(f"Extracting path to: {output_file}")
                extract_path(record_files, output_file)

            trajectory = utils.read_points_from_file(output_file)
            if trajectory.size == 0:
                logging.error(
                    "Could not read any valid path points from the input file."
                )
                sys.exit(1)

            # Sampling and smoothing curves with optional curvature constraint
            filtered_trajectory = preprocess.optimize_trajectory(
                trajectory,
                kernel_size=3,
                threshold_factor=3.0,
                rdp_epsilon=0.01,
                resample_spacing=0.5,
                smooth_window=10,
                smooth_polyorder=3,
                max_curvature=max_curvature,
                curvature_constraint_method=curvature_method,
                max_iterations=max_iterations,
            )
            logging.info("Filtered trajectory points without curvature constraint.")
        elif parsed_args.path_params:
            params = parsed_args.path_params
            defaults = [50.0, 0.0, 0.5]  # Defaults for forward, backward, spacing
            params.extend(defaults[len(params) - 3 :])
            start_x, start_y, heading_deg, forward_dist, backward_dist, spacing = params

            logging.info(f"Generating dynamic path from parameters...")
            trajectory = utils.generate_line_points(
                x=start_x,
                y=start_y,
                heading=heading_deg,
                forward=forward_dist,
                backward=backward_dist,
                spacing=spacing,
                heading_in_degrees=False,
            )
            if not trajectory:
                logging.error(
                    "Failed to generate any path points. Check your parameters."
                )
                sys.exit(1)

            # This becomes the final trajectory, no optimization needed
            filtered_trajectory = np.array(trajectory)
            logging.info(
                f"Successfully generated {len(filtered_trajectory)} trajectory points."
            )

        plot_output_file = os.path.join(output_path, "output.png")
        logging.info(f"Plotting path points to {plot_output_file}")
        plots.plot_points(filtered_trajectory, plot_output_file)

        map_data = map_pb2.Map()
        path = LineString(filtered_trajectory)
        logging.info("Processing path to generate map data.")
        map_gen.process_path(map_data, path, extra_roi_extension, enable_loopback)

        logging.info(f"Saving map data to: {output_path}")
        utils.save_map_to_file(map_data, output_path)
        print("Map data generation complete.")

        # ================================================
        # Plot and analyze the trajectory
        # ================================================

        # Plot original trajectory
        plot_output_file = os.path.join(output_path, "output.png")
        logging.info(f"Plotting path points to {plot_output_file}")
        plots.plot_points(filtered_trajectory, plot_output_file)

        # Print summary
        print("\n=== Processing Summary ===")
        print(f"Original points: {len(trajectory)}")
        print(f"Processed points: {len(filtered_trajectory)}")
        print(f"Reduction ratio: {len(filtered_trajectory)/len(trajectory)*100:.2f}%")

        # Analyze and plot curvature statistics
        if max_curvature is not None:
            curvature_stats = plots.plot_curvature_analysis(
                filtered_trajectory, output_path, "Processed", max_curvature
            )

            print(
                f"Max curvature after processing: {curvature_stats['max_curvature']:.6f}"
            )
            print(
                f"Curvature constraint: {max_curvature:.3f} (method: {curvature_method})"
            )
            if curvature_stats["max_curvature"] <= max_curvature:
                print(f"✓ Curvature constraint satisfied (≤ {max_curvature:.3f})")
            else:
                print(f"⚠ Curvature constraint violated (> {max_curvature:.3f})")
                violations = curvature_stats["violations_threshold_4"]
                violation_percentage = (
                    violations / curvature_stats["total_points"] * 100
                )
                print(
                    f"  Violations: {violations} points ({violation_percentage:.2f}%)"
                )
                print("\n💡 Suggested solutions:")
                print(
                    f"   1. Increase iterations: --max_iterations {max_iterations * 2}"
                )
                print(
                    f"   2. Try different method: --curvature_method {'adaptive' if curvature_method == 'strict' else 'strict'}"
                )
                print(
                    f"   3. Relax curvature limit: --max_curvature {max_curvature * 1.2:.3f}"
                )
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
