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


import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, medfilt

# ==============================
# 1. Outlier Filtering: Median Filter
# ==============================


def filter_outliers(points, kernel_size=3, threshold_factor=3.0):
    x = points[:, 0]
    y = points[:, 1]

    # Smooth the data using median filter
    x_filtered = medfilt(x, kernel_size)
    y_filtered = medfilt(y, kernel_size)

    # Calculate the absolute difference between original and smoothed data
    x_diff = np.abs(x - x_filtered)
    y_diff = np.abs(y - y_filtered)

    # Calculate threshold based on median absolute deviation
    x_threshold = threshold_factor * np.median(x_diff)
    y_threshold = threshold_factor * np.median(y_diff)

    # Determine outliers (either x or y deviation exceeds threshold)
    is_outlier = (x_diff > x_threshold) | (y_diff > y_threshold)

    # Return non-outlier points using boolean indexing
    return points[~is_outlier]

# =========================================
# 2. Ramer-Douglas-Peucker (RDP) Algorithm
# =========================================


def rdp(points, epsilon):
    """
    Simplify the curve using the RDP algorithm.

    Args:
      points: (N,2) array of 2D trajectory points
      epsilon: distance threshold; points with smaller deviation will be discarded

    Returns:
      Simplified trajectory points
    """
    if points.shape[0] < 3:
        return points

    # Compute the line vector from the first to the last point
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    # Avoid division by zero
    if line_len == 0:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        # Compute perpendicular distance from each point to the line
        norm_line_vec = line_vec / line_len
        vec_from_start = points - start
        proj_lengths = np.dot(vec_from_start, norm_line_vec)
        proj_points = np.outer(proj_lengths, norm_line_vec) + start
        dists = np.linalg.norm(points - proj_points, axis=1)

    # Find the point with the maximum distance to the line
    idx = np.argmax(dists)
    max_dist = dists[idx]

    if max_dist > epsilon:
        # Recursively simplify the two segments
        first_half = rdp(points[:idx+1], epsilon)
        second_half = rdp(points[idx:], epsilon)
        # Combine the results (avoid duplicating the middle point)
        return np.vstack((first_half[:-1], second_half))
    else:
        # All points are close enough to the line; keep only endpoints
        return np.array([start, end])

# =================================================
# 3. Uniform Resampling: Fill Sparse Regions via Interpolation
# =================================================


def uniform_resample(points, spacing=1.0, kind='cubic'):
    """
    Uniformly resample trajectory points by:
      1. Calculating cumulative arc length
      2. Interpolating x and y using interp1d

    Args:
      points: (N,2) array of original trajectory points (should be simplified or evenly spaced)
      spacing: desired distance between new sample points
      kind: interpolation method ('linear', 'cubic', etc.)

    Returns:
      Uniformly resampled trajectory points
    """
    # Compute distances and cumulative distance
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumdist = np.concatenate(([0], np.cumsum(distances)))

    # Generate new sample positions
    new_distances = np.arange(0, cumdist[-1], spacing)

    # Interpolate x and y separately
    interp_func_x = interp1d(cumdist, points[:, 0], kind=kind)
    interp_func_y = interp1d(cumdist, points[:, 1], kind=kind)

    new_x = interp_func_x(new_distances)
    new_y = interp_func_y(new_distances)
    new_points = np.column_stack((new_x, new_y))

    return new_points

# ================================================
# 4. Smoothing: Savitzky-Golay Filter
# ================================================


def smooth_track(points, window_length=5, polyorder=3):
    """
    Smooth trajectory points using Savitzky-Golay filter.

    Args:
      points: (N,2) array of uniformly resampled points
      window_length: smoothing window size (must be odd)
      polyorder: order of the polynomial used in filtering

    Returns:
      Smoothed trajectory points
    """
    # Apply the filter to x and y separately
    x_smooth = savgol_filter(points[:, 0], window_length, polyorder)
    y_smooth = savgol_filter(points[:, 1], window_length, polyorder)
    return np.column_stack((x_smooth, y_smooth))

# =====================================================
# Pipeline: A Full Trajectory Optimization Example
# =====================================================


def optimize_trajectory(raw_points,
                        kernel_size=3,
                        threshold_factor=3.0,
                        rdp_epsilon=0.1,
                        resample_spacing=1.0,
                        smooth_window=5,
                        smooth_polyorder=3):
    """
    Given raw trajectory points, perform the following:
      1. Outlier filtering
      2. RDP simplification (reduce dense segments)
      3. Uniform resampling (fill sparse segments)
      4. Savitzky-Golay smoothing

    Args:
      raw_points: (N,2) original trajectory points
      rdp_epsilon: RDP simplification threshold
      resample_spacing: spacing for uniform resampling
      smooth_window, smooth_polyorder: smoothing parameters

    Returns:
      Optimized trajectory points
    """
    # 1. Filter outliers
    filtered_points = filter_outliers(
        raw_points, kernel_size, threshold_factor)

    # 2. Simplify trajectory using RDP
    simplified_points = rdp(filtered_points, epsilon=rdp_epsilon)

    # 3. Resample for uniform spacing
    resampled_points = uniform_resample(
        simplified_points, spacing=resample_spacing, kind='cubic')

    # 4. Smooth the trajectory
    smoothed_points = smooth_track(
        resampled_points, window_length=smooth_window, polyorder=smooth_polyorder)

    return smoothed_points
