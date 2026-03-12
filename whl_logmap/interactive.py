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
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from .funcs import calculate_curvature


class InteractivePipelineEditor:
    """
    An interactive matplotlib-based GUI to visualize and control the full preprocessing pipeline.
    """

    def __init__(
        self,
        raw_points,
        enable_loopback,
        loopback_threshold,
        init_thresh=5.0,
        init_rdp=0.01,
        init_space=0.5,
        init_window=7,
        polyorder=3,
        init_max_curvature=0.2,
    ):
        self.raw_points = raw_points
        self.enable_loopback = enable_loopback
        self.loopback_threshold = loopback_threshold

        # Current pipeline parameters
        self.thresh = init_thresh
        self.rdp = init_rdp
        self.space = init_space
        self.window = init_window
        self.poly = polyorder
        self.max_curv = init_max_curvature if init_max_curvature is not None else 0.2

        # State
        self.resampled_points = None
        self.is_loop = False
        self.selected_idx = None

        # --- Figure & Axes Setup ---
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(bottom=0.45, right=0.95)

        # Plot markers
        (self.line_raw,) = self.ax.plot(
            raw_points[:, 0],
            raw_points[:, 1],
            ".",
            color="lightgray",
            alpha=0.6,
            label="Raw Input Points",
        )
        (self.line_resampled,) = self.ax.plot(
            [],
            [],
            "ro--",
            alpha=0.6,
            picker=True,
            pickradius=5,
            label="Sampled Points (Draggable)",
        )
        (self.line_smooth,) = self.ax.plot(
            [], [], "b-", linewidth=2, label="Smoothed Prev."
        )
        (self.line_high_curv,) = self.ax.plot(
            [], [], "yo", markersize=4, label="High Curvature (> Max)", zorder=5
        )

        # Determine strict bounds based on raw points to prevent moving view on updates
        x_min, x_max = np.min(raw_points[:, 0]), np.max(raw_points[:, 0])
        y_min, y_max = np.min(raw_points[:, 1]), np.max(raw_points[:, 1])
        x_ptp, y_ptp = x_max - x_min, y_max - y_min

        self.ax.set_xlim(x_min - x_ptp * 0.1, x_max + x_ptp * 0.1)
        self.ax.set_ylim(y_min - y_ptp * 0.1, y_max + y_ptp * 0.1)
        self.ax.axis("equal")  # crucial for maps so curves aren't distorted

        self.search_thresh = max(x_ptp, y_ptp) * 0.02

        self.ax.legend()
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.set_title(
            "Full Pipeline Tuner: Adjust params -> Drag RED points to fine-tune"
        )

        # --- UI Sliders Setup ---
        # Stage 1: Structure (Affects resampling logic)
        axcolor_struct = "lightcyan"
        ax_thresh = plt.axes([0.15, 0.28, 0.3, 0.03], facecolor=axcolor_struct)
        ax_rdp = plt.axes([0.15, 0.21, 0.3, 0.03], facecolor=axcolor_struct)
        ax_space = plt.axes([0.15, 0.14, 0.3, 0.03], facecolor=axcolor_struct)

        self.sl_thresh = Slider(
            ax_thresh, "Outlier Thresh", 1.0, 15.0, valinit=self.thresh
        )
        self.sl_rdp = Slider(
            ax_rdp, "RDP Epsilon", 0.001, 1.0, valinit=self.rdp, valfmt="%0.3f"
        )
        self.sl_space = Slider(ax_space, "Resample Space", 0.1, 5.0, valinit=self.space)

        ax_loopback = plt.axes([0.15, 0.07, 0.15, 0.05], facecolor="lightgray")
        self.cb_loopback = CheckButtons(
            ax_loopback, ["Enable Loopback"], [self.enable_loopback]
        )

        # Stage 2: Smoothing & Presentation (Does not affect red dot structure)
        axcolor_smooth = "honeydew"
        ax_window = plt.axes([0.6, 0.28, 0.3, 0.03], facecolor=axcolor_smooth)
        ax_curv = plt.axes([0.6, 0.21, 0.3, 0.03], facecolor=axcolor_smooth)

        self.sl_window = Slider(
            ax_window, "Smooth Window", 3, 101, valinit=self.window, valstep=2
        )
        self.sl_curv = Slider(
            ax_curv, "Max Curvature", 0.01, 1.0, valinit=self.max_curv, valfmt="%0.3f"
        )

        # Titles and Explanations for UI blocks
        # Stage 1 Texts
        self.fig.text(
            0.3,
            0.35,
            "STAGE 1: Auto Resampling & Shape",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="darkblue",
        )
        self.fig.text(
            0.3,
            0.32,
            "(Changes here rebuild the red trajectory points)",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
        )

        # Stage 2 Texts
        self.fig.text(
            0.75,
            0.35,
            "STAGE 2: Manual Edit & Smoothing",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="darkgreen",
        )
        self.fig.text(
            0.75,
            0.32,
            "(Drag Red points on plot. Adjust smooth window.)",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
        )

        # Handlers
        self.sl_thresh.on_changed(self.on_param_change)
        self.sl_rdp.on_changed(self.on_param_change)
        self.sl_space.on_changed(self.on_param_change)
        self.cb_loopback.on_clicked(self.on_loopback_change)
        self.sl_window.on_changed(self.on_smooth_window_change)
        self.sl_curv.on_changed(self.on_smooth_window_change)

        # --- UI Buttons ---
        ax_btn = plt.axes([0.8, 0.04, 0.15, 0.06])
        self.btn_save = Button(ax_btn, "Save & Proceed", color="lightgreen")
        self.btn_save.on_clicked(self.save_and_close)

        # --- Canvas Events (Draggable Points) ---
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        # Initial execution
        self.run_pipeline()
        plt.show(block=True)

    def run_pipeline(self):
        """Executes the pipeline up to Step 3.5 (Loopback welding)."""
        from .preprocess import filter_outliers, rdp, uniform_resample, process_loopback

        # 1. Filter
        p = filter_outliers(
            self.raw_points,
            kernel_size=3,
            threshold_factor=self.thresh,
            is_loop=self.enable_loopback,
        )
        # 2. RDP
        p = rdp(p, epsilon=self.rdp)
        # 3. Resample
        p = uniform_resample(p, spacing=self.space, kind="linear")
        # 3.5 Loopback
        if self.enable_loopback:
            p, self.is_loop = process_loopback(
                p,
                distance_threshold=self.loopback_threshold,
                min_loop_length=20.0,
                blend_distance=10.0,
            )
        else:
            self.is_loop = False

        self.resampled_points = p.copy()
        self.update_plots()

    def on_loopback_change(self, label):
        """Toggle loopback setting and run pipeline."""
        self.enable_loopback = not self.enable_loopback
        self.run_pipeline()

    def on_param_change(self, val):
        """When an architectural parameter changes, we must rebuild the resampled points."""
        self.thresh = self.sl_thresh.val
        self.rdp = self.sl_rdp.val
        self.space = self.sl_space.val
        # Notice: Any custom edits (drags) are lost when structural params change!
        self.run_pipeline()

    def on_smooth_window_change(self, val):
        """When only the smoothing window changes, just update the plot without rebuilding points."""
        self.window = int(self.sl_window.val)
        if self.window % 2 == 0:
            self.window += 1
        self.max_curv = float(self.sl_curv.val)
        self.update_plots()

    def update_plots(self):
        """Apply smoothing and update Line2D data."""
        from .preprocess import smooth_track

        self.line_resampled.set_data(
            self.resampled_points[:, 0], self.resampled_points[:, 1]
        )

        smoothed = smooth_track(
            self.resampled_points,
            window_length=self.window,
            polyorder=self.poly,
            is_loop=self.is_loop,
        )
        self.line_smooth.set_data(smoothed[:, 0], smoothed[:, 1])

        # High curvature visualization
        curvatures = calculate_curvature(smoothed)
        high_curv_mask = curvatures > self.max_curv
        high_curv_pts = smoothed[high_curv_mask]

        if len(high_curv_pts) > 0:
            self.line_high_curv.set_data(high_curv_pts[:, 0], high_curv_pts[:, 1])
        else:
            self.line_high_curv.set_data([], [])

        self.fig.canvas.draw_idle()

    # --- Mouse Event Handlers ---
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        # Find nearest point index
        dists = np.sqrt(
            (self.resampled_points[:, 0] - event.xdata) ** 2
            + (self.resampled_points[:, 1] - event.ydata) ** 2
        )
        idx = np.argmin(dists)
        if dists[idx] < self.search_thresh:
            self.selected_idx = idx

    def on_release(self, event):
        if self.selected_idx is not None:
            # Force exact closure wrap upon release
            if self.is_loop and (
                self.selected_idx == 0
                or self.selected_idx == len(self.resampled_points) - 1
            ):
                self.resampled_points[-1] = self.resampled_points[0]
            self.selected_idx = None
            self.update_plots()

    def on_motion(self, event):
        if self.selected_idx is None or event.inaxes != self.ax:
            return

        # Move the selected point
        self.resampled_points[self.selected_idx, 0] = event.xdata
        self.resampled_points[self.selected_idx, 1] = event.ydata

        # Enforce loop visually while moving
        if self.is_loop:
            if self.selected_idx == 0:
                self.resampled_points[-1] = self.resampled_points[0]
            elif self.selected_idx == len(self.resampled_points) - 1:
                self.resampled_points[0] = self.resampled_points[-1]

        # Trigger fast preview update
        self.update_plots()

    def save_and_close(self, event):
        plt.close(self.fig)


def launch_interactive_editor(
    raw_points,
    enable_loopback=False,
    loopback_threshold=3.0,
    init_thresh=5.0,
    init_rdp=0.01,
    init_space=0.5,
    init_window=7,
    polyorder=3,
    init_max_curvature=0.2,
):
    """
    Launch interactive tuning window. Blocks execution until window closes.
    Returns: (resampled_points, is_loop, thresh, rdp, space, window, max_curvature)
    """
    print("Launching Full Pipeline Interactive Editor...")
    print("- Use Sliders to tune structural preprocessing.")
    print("- Drag red dots to fix local map anomalies manually.")

    editor = InteractivePipelineEditor(
        raw_points,
        enable_loopback,
        loopback_threshold,
        init_thresh,
        init_rdp,
        init_space,
        init_window,
        polyorder,
        init_max_curvature,
    )
    return (
        editor.resampled_points,
        editor.enable_loopback,  # Returning the possibly toggled loopback status
        editor.thresh,
        editor.rdp,
        editor.space,
        editor.window,
        editor.max_curv,
    )
