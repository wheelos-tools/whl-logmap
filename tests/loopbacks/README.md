# Loopback Testing Suite

This directory contains testing scripts to verify the functionality of the trajectory loopback processing logic within `whl-logmap`.

## Overview

The `generate_and_plot.py` script automatically generates synthetic test data mimicking real-world vehicle trajectories, intentionally injecting various types of loop closure errors. It then runs these synthetic paths through the `optimize_trajectory` pipeline to visually and programmatically confirm that the `loopback` closure and smoothing algorithms work properly.

## Scenarios Tested

The script generates a rectangular "lap" but manipulates the end segment to simulate common GPS or odometry drift scenarios:

1. **Perfect Loop**: Ends exactly where it started.
2. **Horizontal Gap (X-offset)**: The vehicle finishes slightly parallel to the start line.
3. **Vertical Gap (Y-offset)**: The vehicle stops slightly short of the finish line.
4. **Combined Offset**: Standard drift on both X and Y axes.
5. **Overlap Tail**: The trajectory passes the start line and overlaps by a few meters.
6. **No Loop (Open Path)**: A U-shape or incomplete track where loopback *should not* trigger.

## Usage

Ensure you are in an active virtual environment with `whl_logmap`'s dependencies installed.

```bash
cd tests/loopbacks
python generate_and_plot.py
```

## Outputs

The script outputs `scene_*.png` plots in this directory. 
- **Blue line**: Shows the original generated "noisy" base trajectory.
- **Red dashed line/points**: Shows the final optimized trajectory after filtering, RDP, resampling, loopback welding, and curvature smoothing. 
- **Start/End points**: Marked for visual confirmation that the connection was closed perfectly without abrupt orientation changes.

You can verify that regardless of the initial gap (within the `loopback_threshold`), the red trajectory perfectly connects and blends back into the main loop.
