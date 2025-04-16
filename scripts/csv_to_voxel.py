import pandas as pd
import numpy as np
import math
import torch
import os

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    try:
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                tis[valid_indices] * width * height, vals_left[valid_indices])
    except:
        breakpoint()

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

vid_name = "driving_sample"

df = pd.read_csv(f"{vid_name}.csv", header=None)

# swap columns to match expected format
df[[0, 1, 2, 3]] = df[[3, 0, 1, 2]]

# match required dimensions of network
width = 346
height = 260

x_offset = 400
y_offset = 200

df.iloc[:, 1] = df.iloc[:, 1] - x_offset
df.iloc[:, 2] = df.iloc[:, 2] - y_offset

df = df[(df.iloc[:, 1] < width) & (df.iloc[:, 1] >= 0) & (df.iloc[:, 2] < height) & (df.iloc[:, 2] >= 0)]

# split into time segments
dt = 50000.0
min_time = df.iloc[:, 0].min()
max_time = df.iloc[:, 0].max()

num_voxel_grids = math.ceil((max_time - min_time) / dt)

event_arrays = []
timestamps = np.zeros(num_voxel_grids)

for i in range(num_voxel_grids):
    filtered_df = df[np.floor((df.iloc[:, 0] - min_time) / dt) == i]
    filtered_df = filtered_df.sort_values(by=0)
    event_arrays.append(filtered_df.to_numpy())
    timestamps[i] = (min_time + (i+1) * dt)

# check if empty time range
[print(f"Warning: empty time range at {i}") for i, arr in enumerate(event_arrays) if arr.size == 0]

non_empty_indices = [i for i, arr in enumerate(event_arrays) if arr.size > 0]
event_arrays = [event_arrays[i] for i in non_empty_indices]
timestamps = timestamps[non_empty_indices]

num_bins = 5

num_voxel_grids = len(event_arrays)
voxel_grids = []

for i in range(num_voxel_grids):
    voxel_grids.append(events_to_voxel_grid(event_arrays[i], num_bins, width, height))

directory = f"{vid_name}_fixed_voxel_grids"

if not os.path.exists(directory):
    os.makedirs(directory)

filename_number_length = 10

for i in range(num_voxel_grids):
    filename = f"event_tensor_{str(i).zfill(filename_number_length)}.npy"
    path = os.path.join(directory, filename)
    np.save(path, voxel_grids[i])

with open(os.path.join(directory, 'timestamps.txt'), 'w') as file:
    for index, num in enumerate(timestamps):
        file.write(f"{index} {num}\n")

# boundary_timestamps.txt is a required file but is not actually used, so just create an empty file to avoid error
with open(os.path.join(directory, 'boundary_timestamps.txt'), "w") as file:
    pass 