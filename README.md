# whl-logmap

The **whl-logmap** tool enables the automated and interactive creation of Apollo-format HD maps from raw recorded trajectories. A single pass along a path is enough to generate a fully optimized, spatially smooth map. The tool features an industry-leading **Interactive Pipeline Editor**, empowering developers to visually tune smoothing parameters, resolve high-curvature anomalies, and perfectly close loopbacks.

## Key Features

- **Trajectory Extraction & Map Generation:** Extract your trajectory (position and heading) from standard Cyber records and compile it directly into Apollo `map_pb` `.txt` and `.bin` formats.
- **Parametric Generation:** Generate perfectly straight lanes parametrically utilizing standard start positions (x, y), heading, and lengths.
- **Interactive UI Editor (Enabled by default):** Review your map dynamically. You can use sliders to tune resampling logic via the Ramer-Douglas-Peucker (RDP) algorithm, smoothing windows, and max curvatures.
- **High-Curvature Highlighting & Point Editing:** If point curvatures exceed a maximum tolerable limit (indicating erratic vehicle tracking or sharp GPS jumps), they are **flagged on the visualization with small yellow dots**. You can directly drag-and-drop red map nodes to smooth the anomalies seamlessly.
- **Automatic Loop closure (Loopback):** Identify and smoothly weld cyclic trajectories that begin and end near the same spot.

## Install

```bash
pip3 install whl_logmap
```

*(Note: We recommend running this inside an activated virtual environment or your Apollo workspace).*

## Quick Start

### 1. Generate from a Recorded Trajectory

If you have recorded your vehicle's trajectory using tools like `cyber_recorder record -c /apollo/localization/pose`, you can feed the directory directly:

```bash
# Will launch interactive editor by default
whl-logmap --input_path=your_record_dir/
# OR
whl-logmap -i your_record_file.record.00000
```

### 2. Generate from raw x,y,heading params

Useful for creating baseline testing scenarios without vehicle recordings.

```bash
# x,y,heading(radians)[,forward,backward,spacing]
whl-logmap --path_params="221247.67,2517672.17,1.0,10,50,0.5"
```

## Interactive Editor Guide

By default, executing the tool processes your points and pops up an interactive **Matplotlib Viewer**.
- **Stage 1 (Structure):** Tune *Outlier Threshold*, *RDP Epsilon* (simplifies raw points before interpolation), and *Resample Space* (uniform spacing for the nodes). Changing these will rebuild your initial scaffolding.
- **Stage 2 (Smoothing & Edit):**
  - Enable **Loopback** to close loop tracks.
  - Tune the **Smooth Window** for overall trajectory smoothness.
  - Set a **Max Curvature**. Any points on the smoothed path that violate your vehicle's turning limits will light up as yellow elements.
  - Using your mouse, simply **drag the red resampled layout points** near those yellow alerts outward/inward to relax the curvature securely.

## Options & Flags

- `-i, --input_path`: Path to the records (directory or single file) (CSV format internally managed: x,y).
- `--path_params`: Comma-separated parameters for dynamic straight path generation.
- `-o, --output_path`: Path to the output maps directory (default: `map`).
- `--max_curvature`: Maximum allowed baseline curvature (default `None`). Suggested `0.2` or approx 95% of your vehicle's physical limit.
- `--loopback_threshold`: Distance threshold to close the loop in meters (default `3.0`).
- `--interactive`: Enable the Matplotlib tuning UI (Enabled by **default**).
- `--no_interactive`: Disables the GUI, allowing pure headless execution in CI loops based on default or passed params.
- `--force`: Force regeneration of `path.txt` even if it already exists.

