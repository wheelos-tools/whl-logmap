#!/bin/bash

# Define the root path for Apollo map data
APOLLO_MAP_PATH="/apollo/modules/map/data"

# Define your map directory name
YOUR_MAP_DIR="your_map_dir"

# Construct the full path to your map directory
YOUR_MAP_PATH="${APOLLO_MAP_PATH}/${YOUR_MAP_DIR}"

# Ensure the map directory exists (optional, but recommended)
if [ ! -d "${YOUR_MAP_PATH}" ]; then
  echo "Error: Map directory '${YOUR_MAP_PATH}' does not exist."
  exit 1
fi

# Define the executable file paths for sim_map_generator and topo_creator
SIM_MAP_GENERATOR="./bazel-bin/modules/map/tools/sim_map_generator"
TOPO_CREATOR="./bazel-bin/modules/routing/topo_creator"

# Run the map generator
echo "Generating map files to: ${YOUR_MAP_PATH}"
"${SIM_MAP_GENERATOR}" -map_dir="${YOUR_MAP_PATH}" -output_dir="${YOUR_MAP_PATH}"

# Check if the map generator ran successfully (optional, but recommended)
if [ $? -ne 0 ]; then
  echo "Error: Map generator failed to run."
  exit 1
fi

# Run the topology creator
echo "Creating topology information based on the map: ${YOUR_MAP_PATH}"
"${TOPO_CREATOR}" -map_dir="${YOUR_MAP_PATH}"

# Check if the topology creator ran successfully (optional, but recommended)
if [ $? -ne 0 ]; then
  echo "Error: Topology creator failed to run."
  exit 1
fi

echo "Map generation and topology creation complete!"
