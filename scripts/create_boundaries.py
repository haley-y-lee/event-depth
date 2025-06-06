import os
import numpy as np

# Each sequence in original dataset contains 1000 depth maps + voxel grid pairs.
# For each voxel grid, we compute the corresponding sequence of upsampled frames
# using the timestamps of the upsampled frames. Specifically, we compute the index
# of the starting and ending frame for each voxel grid and save this data in a .txt file.

dt = 0.0333
frame_folder = "/home/abc256/mono/DENSE/test_upsampled_3/test_sequence_00_town10/frames"

frame_stamps = np.loadtxt(
    os.path.join(frame_folder, 'timestamps.txt'))

index = 0


with open(os.path.join(frame_folder, 'boundaries.txt'), 'w') as f:
    for i in range(1,1001):
        start = index
        while (index < frame_stamps.shape[0]-1 and frame_stamps[index] < i * dt):
            index += 1
        
        f.write(f"{start} {index}\n")
        index += 1

