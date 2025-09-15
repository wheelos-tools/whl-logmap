#!/usr/bin/env python
"""Plotting and visualization module.

This module provides functionality for plotting trajectory data,
curvature analysis, and generating visualization outputs.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from whl_logmap.funcs import calculate_curvature

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_curvature_analysis(points, output_path, title_prefix="Processed", max_curvature=None):
    """
    Plot curvature analysis and statistics for trajectory points.

    Args:
        points: numpy array of shape (n, 2) containing (x, y) coordinates
        output_path: directory to save the plots
        title_prefix: prefix for plot titles
        max_curvature: maximum curvature threshold for dynamic threshold calculation
    """
    if len(points) < 3:
        logging.warning("Not enough points for curvature analysis")
        return

    # Calculate curvature
    curvature = calculate_curvature(points)

    # Calculate statistics
    actual_max_curvature = np.max(curvature)
    min_curvature = np.min(curvature)
    mean_curvature = np.mean(curvature)
    std_curvature = np.std(curvature)
    median_curvature = np.median(curvature)

    # Dynamic threshold calculation based on max_curvature parameter
    if max_curvature is not None:
        # Use the provided max_curvature as the highest threshold
        # Calculate other thresholds as percentages of max_curvature
        threshold_1 = max_curvature * 0.5   # 50% of max
        threshold_2 = max_curvature * 0.75  # 75% of max
        threshold_3 = max_curvature * 0.95  # 95% of max
        threshold_4 = max_curvature         # 100% of max (actual constraint)
    else:
        # Fallback to default thresholds if max_curvature not provided
        threshold_1 = 0.1
        threshold_2 = 0.15
        threshold_3 = 0.19
        threshold_4 = 0.2

    # Count violations of different thresholds
    violations_1 = np.sum(curvature > threshold_1)
    violations_2 = np.sum(curvature > threshold_2)
    violations_3 = np.sum(curvature > threshold_3)
    violations_4 = np.sum(curvature > threshold_4)

    total_points = len(curvature)

    # Print statistics
    print(f"\n=== {title_prefix} Trajectory Curvature Statistics ===")
    print(f"Total points: {total_points}")
    print(f"Max curvature: {actual_max_curvature:.6f}")
    print(f"Min curvature: {min_curvature:.6f}")
    print(f"Mean curvature: {mean_curvature:.6f}")
    print(f"Std curvature: {std_curvature:.6f}")
    print(f"Median curvature: {median_curvature:.6f}")
    print("\nCurvature violations:")
    print(
        f"  > {threshold_1:.3f}: {violations_1:4d} points ({violations_1/total_points*100:.2f}%)"
    )
    print(
        f"  > {threshold_2:.3f}: {violations_2:4d} points ({violations_2/total_points*100:.2f}%)"
    )
    print(
        f"  > {threshold_3:.3f}: {violations_3:4d} points ({violations_3/total_points*100:.2f}%)"
    )
    print(
        f"  > {threshold_4:.3f}: {violations_4:4d} points ({violations_4/total_points*100:.2f}%)"
    )

    # Create plots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Trajectory
    ax1.plot(points[:, 0], points[:, 1], "b-", linewidth=1, alpha=0.7)
    ax1.plot(points[:, 0], points[:, 1], "ro", markersize=1, alpha=0.5)
    ax1.set_title(f"{title_prefix} Trajectory")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.grid(True)
    ax1.axis("equal")

    # Plot 2: Curvature vs Point Index
    point_indices = np.arange(len(curvature))
    ax2.plot(point_indices, curvature, "r-", linewidth=1)
    ax2.axhline(y=threshold_1, color="orange", linestyle="--", alpha=0.7, label=f"{threshold_1:.3f} threshold")
    ax2.axhline(
        y=threshold_2, color="yellow", linestyle="--", alpha=0.7, label=f"{threshold_2:.3f} threshold"
    )
    ax2.axhline(
        y=threshold_3, color="green", linestyle="--", alpha=0.7, label=f"{threshold_3:.3f} threshold"
    )
    ax2.axhline(y=threshold_4, color="red", linestyle="--", alpha=0.7, label=f"{threshold_4:.3f} threshold")
    ax2.set_title(f"{title_prefix} Curvature vs Point Index")
    ax2.set_xlabel("Point Index")
    ax2.set_ylabel("Curvature")
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Curvature Histogram
    ax3.hist(curvature, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax3.axvline(x=threshold_1, color="orange", linestyle="--", alpha=0.7, label=f"{threshold_1:.3f} threshold")
    ax3.axvline(
        x=threshold_2, color="yellow", linestyle="--", alpha=0.7, label=f"{threshold_2:.3f} threshold"
    )
    ax3.axvline(
        x=threshold_3, color="green", linestyle="--", alpha=0.7, label=f"{threshold_3:.3f} threshold"
    )
    ax3.axvline(x=threshold_4, color="red", linestyle="--", alpha=0.7, label=f"{threshold_4:.3f} threshold")
    ax3.set_title(f"{title_prefix} Curvature Distribution")
    ax3.set_xlabel("Curvature")
    ax3.set_ylabel("Frequency")
    ax3.grid(True)
    ax3.legend()

    # Plot 4: Cumulative Distribution
    sorted_curvature = np.sort(curvature)
    cumulative_prob = np.arange(1, len(sorted_curvature) + 1) / len(sorted_curvature)
    ax4.plot(sorted_curvature, cumulative_prob, "b-", linewidth=2)
    ax4.axvline(x=threshold_1, color="orange", linestyle="--", alpha=0.7, label=f"{threshold_1:.3f} threshold")
    ax4.axvline(
        x=threshold_2, color="yellow", linestyle="--", alpha=0.7, label=f"{threshold_2:.3f} threshold"
    )
    ax4.axvline(
        x=threshold_3, color="green", linestyle="--", alpha=0.7, label=f"{threshold_3:.3f} threshold"
    )
    ax4.axvline(x=threshold_4, color="red", linestyle="--", alpha=0.7, label=f"{threshold_4:.3f} threshold")
    ax4.set_title(f"{title_prefix} Curvature Cumulative Distribution")
    ax4.set_xlabel("Curvature")
    ax4.set_ylabel("Cumulative Probability")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()

    # Save the plot
    curvature_plot_file = os.path.join(
        output_path, f"{title_prefix.lower()}_curvature_analysis.png"
    )
    plt.savefig(curvature_plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Curvature analysis plot saved to: {curvature_plot_file}")

    return {
        "max_curvature": actual_max_curvature,
        "min_curvature": min_curvature,
        "mean_curvature": mean_curvature,
        "std_curvature": std_curvature,
        "median_curvature": median_curvature,
        "violations_threshold_1": violations_1,
        "violations_threshold_2": violations_2,
        "violations_threshold_3": violations_3,
        "violations_threshold_4": violations_4,
        "threshold_1": threshold_1,
        "threshold_2": threshold_2,
        "threshold_3": threshold_3,
        "threshold_4": threshold_4,
        "total_points": total_points,
    }


def plot_points(points: np.ndarray, output_file: str = "map/output.png"):
    """Plots a list of (x, y) points using matplotlib.

    Args:
        points: A list of (x, y) coordinate tuples.
        output_file: The path to save the plot. Defaults to 'map/output.png'.
    """
    if points.size == 0:
        logging.warning("No points to plot.")
        return

    # Create a square figure (same width and height)
    plt.figure(figsize=(8, 8))

    # Plot the points as a scatter plot
    plt.plot(points[:, 0], points[:, 1], "-o", markersize=3)

    # Set the labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Path Points")

    # Ensure equal scaling on both axes
    plt.axis("equal")

    # Add grid
    plt.grid(True)

    # Save the plot as a square image
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
