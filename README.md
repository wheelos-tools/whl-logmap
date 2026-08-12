# whl-logmap

## Overview

`whl-logmap` is a Python command-line tool for generating Apollo map protobuf
files from 2D vehicle trajectories. It accepts either record files containing
Apollo localization poses or a parametrically generated straight path.

For record input, the tool extracts `x` and `y` coordinates from the
`/apollo/localization/pose` topic, optionally preprocesses the trajectory, and
converts it into a map containing a road, lane centerline, lane boundaries,
and road boundary geometry. The output includes a text protobuf and a binary
protobuf, plus a path plot. The preprocessing pipeline provides outlier
filtering, Ramer-Douglas-Peucker simplification, uniform resampling, smoothing,
optional loopback processing, and an interactive Matplotlib editor.

## Role in WheelOS

`whl-logmap` is a **Tools** component in WheelOS. It supports offline map
generation from recorded localization data and is not a runtime vehicle
control or perception component.

```text
WheelOS
 |
 +--- Tools
      |
      +--- whl-logmap
```

## Architecture

```text
Record files
    |
    v
Record reader
    |
    +--> /apollo/localization/pose --> x,y trajectory
    |
    v
Preprocessing
(outlier filter -> RDP -> resample -> loopback -> smoothing)
    |
    +--> optional Matplotlib interactive editor
    |
    v
Shapely LineString
    |
    v
Apollo map_pb2.Map generation
    |
    +--> base_map.txt
    +--> base_map.bin
    +--> output.png
```

The `--path_params` input bypasses record extraction and preprocessing by
creating a straight trajectory directly.

## Installation

Install the package and its declared dependencies from the repository:

```bash
python -m pip install .
```

The package requires Python 3.6 or newer. The command-line entry point is
`whl-logmap`.

## Examples

Generate a straight path from `x`, `y`, heading in radians, forward distance,
backward distance, and point spacing:

```bash
whl-logmap --path_params="221247.67,2517672.17,1.0,10,50,0.5"
```

Generate a map from one record file or a directory containing record files:

```bash
whl-logmap --input_path=your_record_dir/ --output_path=map
whl-logmap --input_path=your_record_file.record.00000 --output_path=map
```

For a headless record-processing run, disable the interactive editor:

```bash
whl-logmap --input_path=your_record_dir/ --no_interactive
```

The repository also contains a synthetic loopback example:

```bash
cd tests/loopbacks
python generate_and_plot.py
```

It writes generated trajectory text files under `tests/loopbacks/` and plots
under `tests/loopbacks/plots/`.

## Documentation

- [Loopback testing README](tests/loopbacks/README.md)
- [Project metadata and build configuration](pyproject.toml)
- [Apache License 2.0](LICENSE)
- [Project homepage](https://github.com/wheelos-tools/whl-logmap)
- [Issue tracker](https://github.com/wheelos-tools/whl-logmap/issues)
