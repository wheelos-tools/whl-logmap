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
Mathematical functions for path analysis.
"""

import numpy as np


def calculate_curvature(points):
    """
    Calculate curvature for a series of points.

    Args:
        points: numpy array of shape (n, 2) containing (x, y) coordinates

    Returns:
        numpy array of curvature values
    """

    def _calculate_derivatives(points):
        """Calculate first and second derivatives of points."""
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        return dx, dy, ddx, ddy

    def _compute_curvature_formula(dx, dy, ddx, ddy):
        """Compute curvature using the formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)."""
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5)
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
        )

    if len(points) < 3:
        return np.zeros(len(points))

    # Calculate derivatives and curvature directly on original points
    dx, dy, ddx, ddy = _calculate_derivatives(points)
    curvature = _compute_curvature_formula(dx, dy, ddx, ddy)

    return curvature
