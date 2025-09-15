#!/usr/bin/env python

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
