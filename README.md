# whl-logmap

A tool to generate map data from path files.

## Install

```bash
pip3 install whl_logmap
```

## Quick Start

Generate map data from a path file (CSV format with x,y values):

```bash
whl_logmap --input_path=your_record_dir
```

## Options

- **--input_path** (required): Path to the input file containing path points.
- **--output_path**: Path to the output directory (default: `"map"`). The tool will create corresponding `.txt` and `.bin` files.
- **--extra_roi_extension**: Extra ROI extension distance (default: `0.3`).
