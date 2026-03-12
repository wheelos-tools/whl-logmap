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
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(__file__)
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def get_rect_point(s, width, height):
    """
    Return the theoretical coordinate on the rectangle and the outward normal
    vector for a given traveled distance `s` along the perimeter. The
    rectangle origin is at (0,0) (bottom-left). Edge traversal order is:
    bottom (-> right) -> right (-> up) -> top (-> left) -> left (-> down).
    """
    perimeter = 2 * (width + height)
    s = s % perimeter  # allow wrapping around the perimeter

    if s < width:
        # bottom edge, moving right
        return np.array([s, 0.0]), np.array([0.0, 1.0])  # normal points up
    elif s < width + height:
        # right edge, moving up
        return np.array([width, s - width]), np.array([-1.0, 0.0])
    elif s < 2 * width + height:
        # top edge, moving left
        return np.array([width - (s - width - height), height]), np.array([0.0, -1.0])
    else:
        # left edge, moving down
        return np.array([0.0, height - (s - 2 * width - height)]), np.array([1.0, 0.0])


def generate_trajectory(
    width, height, start_offset, total_length, spacing=0.5, noise_std=0.1
):
    """
    Generate a noisy rectangular trajectory.

    Args:
        width: rectangle width
        height: rectangle height
        start_offset: offset along the perimeter for the start point
            (used to avoid starting at a corner)
        total_length: total length of trajectory to generate
        spacing: sample spacing along the path
        noise_std: standard deviation of Gaussian noise added to points

    Returns:
        points: (N,2) array of coordinates
        normals: (N,2) array of local outward normals
        distances: array of sampled distances
    """
    distances = np.arange(0, total_length + spacing / 2, spacing)
    points = []
    normals = []

    for d in distances:
        s = start_offset + d
        pt, normal = get_rect_point(s, width, height)
        points.append(pt)
        normals.append(normal)

    points = np.array(points)
    normals = np.array(normals)

    # Add independent Gaussian noise on X and Y to simulate GPS jitter
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, points.shape)
        points += noise

    return points, normals, distances


def plot_scenario(points, title, filename):
    """
    Visualize the trajectory, mark start/end, and draw a 2-meter threshold
    circle around the start point (useful for loopback demonstrations).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # draw the path
    ax.plot(
        points[:, 0],
        points[:, 1],
        "-.",
        color="gray",
        label="Path",
        markersize=2,
        alpha=0.6,
    )
    ax.scatter(points[:, 0], points[:, 1], c="blue", s=5, alpha=0.5)

    # mark start and end points
    start_pt = points[0]
    end_pt = points[-1]
    ax.scatter(
        [start_pt[0]],
        [start_pt[1]],
        c="green",
        s=100,
        label="Start",
        zorder=5,
        marker="*",
    )
    ax.scatter([end_pt[0]], [end_pt[1]], c="red", s=80, label="End", zorder=5)

    # draw a 2m radius circle centered at the start (visual aid)
    circle = plt.Circle(
        (start_pt[0], start_pt[1]),
        2.0,
        color="orange",
        fill=False,
        linestyle="--",
        linewidth=2,
        label="2m threshold",
    )
    ax.add_patch(circle)

    # 计算起点到终点的实际直线距离
    gap = np.linalg.norm(end_pt - start_pt)

    ax.set_title(f"{title}\nGap from Start to End: {gap:.2f}m")
    ax.set_aspect("equal", "box")
    ax.legend(loc="center")
    ax.grid(True, linestyle=":", alpha=0.6)

    # set view range to include the rectangle
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 35)

    fig.savefig(filename, dpi=150)
    plt.close(fig)


def make_scenarios():
    width = 50.0
    height = 30.0
    spacing = 0.5
    noise_std = 0.15  # 0.15 m Gaussian noise
    start_offset = 15.0  # start 15m along the bottom edge (not at a corner)

    perimeter = 2 * (width + height)  # perimeter length

    scenarios = []

    # 1. Closed trajectory with 8m overlap (total length = perimeter + 8)
    pts_1, _, _ = generate_trajectory(
        width, height, start_offset, perimeter + 8.0, spacing, noise_std
    )
    scenarios.append(("1_Overlap_8m", pts_1))

    # 2. Not closed, but gap under 2m (e.g. gap = 1.5m)
    pts_2, _, _ = generate_trajectory(
        width, height, start_offset, perimeter - 1.5, spacing, noise_std
    )
    scenarios.append(("2_Gap_Under_2m", pts_2))

    # 3. Not closed, gap over 2m (e.g. gap = 4.0m)
    pts_3, _, _ = generate_trajectory(
        width, height, start_offset, perimeter - 4.0, spacing, noise_std
    )
    scenarios.append(("3_Gap_Over_2m", pts_3))

    # 4. Not closed with lateral deviation (not just a longitudinal gap).
    #    Apply a gradual lateral drift over the final segment.
    drift_distance = 10.0  # drift applied over the last 10 meters
    total_len = perimeter - 0.5  # small longitudinal gap
    pts_4, normals, dists = generate_trajectory(
        width, height, start_offset, total_len, spacing, noise_std
    )

    # apply lateral drift progressively
    for i, d in enumerate(dists):
        remaining_d = total_len - d
        if remaining_d < drift_distance:
            # larger lateral shift the closer we are to the end, up to 1.5m
            drift_ratio = (drift_distance - remaining_d) / drift_distance
            lateral_shift = 1.5 * drift_ratio
            pts_4[i] += normals[i] * lateral_shift

    scenarios.append(("4_Lateral_Deviation", pts_4))

    return scenarios


def run():
    scenarios = make_scenarios()
    for name, pts in scenarios:
        # save trajectory text file
        raw_file = os.path.join(OUT_DIR, f"{name}.txt")
        with open(raw_file, "w") as f:
            for x, y in pts:
                f.write(f"{x:.3f},{y:.3f}\n")

        # save visualization image
        plot_file = os.path.join(PLOT_DIR, f"{name}.png")
        plot_scenario(pts, name.replace("_", " "), plot_file)

        print(f"Generated {raw_file} and {plot_file}")


if __name__ == "__main__":
    run()
