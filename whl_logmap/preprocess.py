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
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter1d
from scipy.signal import savgol_filter, medfilt


def filter_trajectory(trajectory: np.ndarray, filter_type: str = "gaussian_filter", **kwargs) -> np.ndarray:
    """
    Filters a vehicle trajectory (an Nx2 numpy array, each row is [x, y]) using one of several filtering methods.

    Parameters:
        trajectory: np.ndarray
            Trajectory data with shape (N, 2)
        filter_type: str
            The type of filter to apply. Options include:
              - 'median_filter' : using scipy.ndimage.median_filter
              - 'uniform_filter': using scipy.ndimage.uniform_filter
              - 'gaussian_filter': using scipy.ndimage.gaussian_filter1d
              - 'savgol_filter'  : using scipy.signal.savgol_filter
              - 'medfilt'        : using scipy.signal.medfilt
        **kwargs:
            Additional filter-specific parameters:
              For median_filter and uniform_filter:
                  size (int or tuple): The window size (default is 3)
              For gaussian_filter:
                  sigma (float): The standard deviation for Gaussian kernel (default is 1.0)
              For savgol_filter:
                  window_length (int): Must be odd (default is 5)
                  polyorder (int): The order of the polynomial (default is 2)
              For medfilt:
                  kernel_size (int or tuple): The size of the kernel (default is 5)

    Returns:
        np.ndarray: The filtered trajectory with the same shape (N, 2)
    """
    if not isinstance(trajectory, np.ndarray):
        raise ValueError("The input trajectory must be of type np.ndarray.")
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("The shape of the input trajectory must be (N, 2).")

    # Separate the x and y signals
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    if filter_type == 'median_filter':
        size = kwargs.get('size', 3)
        x_filtered = median_filter(x, size=size)
        y_filtered = median_filter(y, size=size)

    elif filter_type == 'uniform_filter':
        size = kwargs.get('size', 3)
        x_filtered = uniform_filter(x, size=size)
        y_filtered = uniform_filter(y, size=size)

    elif filter_type == 'gaussian_filter':
        # Use gaussian_filter1d to filter along one axis
        sigma = kwargs.get('sigma', 1.0)
        x_filtered = gaussian_filter1d(x, sigma=sigma)
        y_filtered = gaussian_filter1d(y, sigma=sigma)

    elif filter_type == 'savgol_filter':
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 2)
        # savgol_filter requires window_length to be odd and less than the signal length
        if window_length % 2 == 0:
            raise ValueError("savgol_filter: window_length must be odd")
        x_filtered = savgol_filter(
            x, window_length=window_length, polyorder=polyorder)
        y_filtered = savgol_filter(
            y, window_length=window_length, polyorder=polyorder)

    elif filter_type == 'medfilt':
        kernel_size = kwargs.get('kernel_size', 5)
        x_filtered = medfilt(x, kernel_size=kernel_size)
        y_filtered = medfilt(y, kernel_size=kernel_size)

    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # Recombine the filtered x and y signals back into an (N,2) array
    return np.stack((x_filtered, y_filtered), axis=1)
