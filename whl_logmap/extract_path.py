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
Extract x and y coordinates from localization messages in data record files,
and save them into a specified file in CSV format.

Usage:
    extract_path.py save_fileName  bag1 bag2

See the gflags for more optional args.
"""

import os
from typing import List
from enum import Enum

from cyber_record.record import Record


class SortMode(Enum):
    """_summary_

    Args:
        Enum (_type_): Enumeration for sorting modes.
    """
    TIME = 0  # Sort by file modification time
    NAME = 1  # Sort by file name


def get_sorted_records(input_path: str, sort_mode: SortMode) -> List[str]:
    """
    Get all record files from the input_path and sort them based on the sort_mode.

    Args:
        input_path: Directory path containing record files.
        sort_mode: Sorting mode, either SortMode.TIME or SortMode.NAME.

    Returns:
        A list of sorted record file paths.
    """
    # Get all files in the directory that contain "record" in their name
    record_files = [os.path.join(input_path, f) for f in os.listdir(
        input_path) if "record" in f and os.path.isfile(os.path.join(input_path, f))]

    if sort_mode == SortMode.TIME:
        # Sort files by modification time
        record_files.sort(key=lambda file_path: os.path.getmtime(file_path))
    elif sort_mode == SortMode.NAME:
        # Sort files by name
        record_files.sort()
    return record_files


def extract_path(record_files: List[str], output_file: str) -> bool:
    """
    Extract path information from the given record files and write it to the output file.

    Args:
        record_files: List of record file paths.
        output_file: Path to the output file.

    Returns:
        True if the file is successfully written, False otherwise.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for record_file in record_files:
            # Read each record file
            record = Record(record_file)
            for topic, message, _ in record.read_messages_fallback():
                # Check if the topic matches the desired one
                if topic == "/apollo/localization/pose":
                    # Extract x and y coordinates from the message
                    x = message.pose.position.x
                    y = message.pose.position.y
                    # Write the coordinates to the output file
                    f.write(f"{x},{y}\n")
    print(f"File written to: {output_file}")
    return True
