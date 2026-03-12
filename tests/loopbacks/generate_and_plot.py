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
    根据行驶距离 s 获取矩形上的理论坐标和法向量（用于后续添加横向偏差）。
    矩形左下角为(0,0)，顺时针或逆时针均可，这里设定为：
    底部(->右) -> 右侧(->上) -> 顶部(->左) -> 左侧(->下)
    """
    perimeter = 2 * (width + height)
    s = s % perimeter  # 允许套圈

    if s < width:
        # 底部边，向右
        return np.array([s, 0.0]), np.array([0.0, 1.0])  # 法向量朝上
    elif s < width + height:
        # 右侧边，向上
        return np.array([width, s - width]), np.array([-1.0, 0.0])
    elif s < 2 * width + height:
        # 顶部边，向左
        return np.array([width - (s - width - height), height]), np.array([0.0, -1.0])
    else:
        # 左侧边，向下
        return np.array([0.0, height - (s - 2 * width - height)]), np.array([1.0, 0.0])


def generate_trajectory(
    width, height, start_offset, total_length, spacing=0.5, noise_std=0.1
):
    """
    生成带有噪音的轨迹
    start_offset: 起点距离(0,0)的距离（用来保证起点不在角点）
    total_length: 轨迹总长度
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

    # 加入高斯噪音模拟真实轨迹 (X和Y独立)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, points.shape)
        points += noise

    return points, normals, distances


def plot_scenario(points, title, filename):
    """
    可视化轨迹，画出起点、终点，并绘制一个2米的阈值判定圆
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 画轨迹
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

    # 标出起点和终点
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

    # 画出以起点为圆心，2米为半径的圆（辅助判定）
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

    # 动态调整视图范围，包含整个矩形
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 35)

    fig.savefig(filename, dpi=150)
    plt.close(fig)


def make_scenarios():
    width = 50.0
    height = 30.0
    spacing = 0.5
    noise_std = 0.15  # 15厘米的随机抖动
    start_offset = 15.0  # 起点在底部边15米处，不在任何角点

    perimeter = 2 * (width + height)  # 160米

    scenarios = []

    # 1. 闭合且重叠 8 米 (总长度 = 周长 + 8)
    pts_1, _, _ = generate_trajectory(
        width, height, start_offset, perimeter + 8.0, spacing, noise_std
    )
    scenarios.append(("1_Overlap_8m", pts_1))

    # 2. 不闭合，但相差2米以内 (例如缺口1.5米：总长度 = 周长 - 1.5)
    pts_2, _, _ = generate_trajectory(
        width, height, start_offset, perimeter - 1.5, spacing, noise_std
    )
    scenarios.append(("2_Gap_Under_2m", pts_2))

    # 3. 不闭合，相差2米外 (例如缺口4.0米：总长度 = 周长 - 4.0)
    pts_3, _, _ = generate_trajectory(
        width, height, start_offset, perimeter - 4.0, spacing, noise_std
    )
    scenarios.append(("3_Gap_Over_2m", pts_3))

    # 4. 不闭合，有横向偏差 (不仅仅是纵向缺口)
    # 我们生成到接近起点的距离，但在最后10米的轨迹中人为加上渐进的横向偏移（例如漂移了1.8米）
    drift_distance = 10.0  # 在最后10米发生漂移
    total_len = perimeter - 0.5  # 纵向仅差0.5米
    pts_4, normals, dists = generate_trajectory(
        width, height, start_offset, total_len, spacing, noise_std
    )

    # 叠加横向漂移
    for i, d in enumerate(dists):
        remaining_d = total_len - d
        if remaining_d < drift_distance:
            # 越接近终点，横向漂移越大，最大漂移 1.5 米
            drift_ratio = (drift_distance - remaining_d) / drift_distance
            lateral_shift = 1.5 * drift_ratio
            pts_4[i] += normals[i] * lateral_shift

    scenarios.append(("4_Lateral_Deviation", pts_4))

    return scenarios


def run():
    scenarios = make_scenarios()
    for name, pts in scenarios:
        # 保存轨迹文本文件
        raw_file = os.path.join(OUT_DIR, f"{name}.txt")
        with open(raw_file, "w") as f:
            for x, y in pts:
                f.write(f"{x:.3f},{y:.3f}\n")

        # 保存可视化图像
        plot_file = os.path.join(PLOT_DIR, f"{name}.png")
        plot_scenario(pts, name.replace("_", " "), plot_file)

        print(f"Generated {raw_file} and {plot_file}")


if __name__ == "__main__":
    run()
