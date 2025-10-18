# whl-logmap

The **whl-logmap** tool enables the automatic creation of Apollo-format HD maps from recorded trajectory. Just one pass along the road is enough to generate a map. It’s a simple, efficient, and powerful solution for fast map production.

## Install

```bash
pip3 install whl_logmap
```

## Quick Start

### Generate from record

1. Start the GPS/transform/localization module and record the vehicle's trajectory

```bash
cyber_recorder record -c /apollo/localization/pose
```

2. Generate base_map from records

```bash
whl-logmap --i=your_record_dir

// or

whl-logmap --i=your_record_file
```

### Generate form x,y,heading

1. Generate base_map from x,y,heading

```bash
# x,y,heading(radians)[,forward,backward,spacing]
whl-logmap --path_params="221247.67,2517672.17,1.0,10,50,0.5"
```

## Options

- **--input_path** (required): Path to the records.
- **--output_path**: Path to the output directory (default: `"map"`). The tool will create corresponding `.txt` and `.bin` files.
- **--extra_roi_extension**: Extra ROI extension distance (default: `0.3`).
