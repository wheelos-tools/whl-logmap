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

"""
Plot points from a file using matplotlib.
"""

import logging
import sys

import matplotlib.pyplot as plt

from whl_logmap.map_gen import read_points_from_file


def plot_points(points: list[tuple[float, float]], output_file: str = 'map/output.png'):
    """Plots a list of (x, y) points using matplotlib.

    Args:
        points: A list of (x, y) coordinate tuples.
        output_file: The path to save the plot. Defaults to 'map/output.png'.
    """
    if not points:
        logging.warning("No points to plot.")
        return

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.scatter(x_coords, y_coords)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Path Points")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error(
            "Usage: python script.py <input_file> [output_image_file]")
        sys.exit(1)

    output_path_file = sys.argv[1]
    path_points = read_points_from_file(output_path_file)
    plot_points(path_points)
