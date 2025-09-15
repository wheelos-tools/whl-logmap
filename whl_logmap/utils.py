#!/usr/bin/env python
"""Utility functions module.

This module provides utility functions for file I/O operations,
map data handling, and other common operations.
"""

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


import os

import numpy as np

from modules.map.proto import map_pb2
from google.protobuf import text_format


def read_points_from_file(filepath: str) -> np.ndarray:
    """Reads path points from a file.

    Args:
        filepath: The path to the file.

    Returns:
        A list of tuples containing the coordinates of the path points.
    """
    return np.loadtxt(filepath, delimiter=',')


def save_map_to_file(map_object: map_pb2.Map, output_filepath: str):
    """Saves the map data to text and binary files.

    Args:
        map_object: The map object.
        output_filepath: The output file path.
    """
    # Save as text format
    txt_filepath = os.path.join(output_filepath, "base_map.txt")
    with open(txt_filepath, 'w', encoding='utf-8') as f_txt:
        f_txt.write(text_format.MessageToString(map_object))
        print(f"Map text format saved to: {txt_filepath}")

    # Save as binary format
    bin_filepath = os.path.join(output_filepath, "base_map.bin")
    with open(bin_filepath, 'wb') as f_bin:
        f_bin.write(map_object.SerializeToString())
        print(f"Map binary format saved to: {bin_filepath}")