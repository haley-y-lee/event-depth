import os
import numpy as np

dt = 0.0333
frame_folder = "/home/abc256/mono/DENSE/small_upsampled/seq/frames"

frame_stamps = np.loadtxt(
    os.path.join(frame_folder, 'timestamps.txt'))

index = 0

with open('/home/abc256/mono/DENSE/output.txt', 'w') as f:
    for i in range(1,1001):
        start = index
        while (index < frame_stamps.shape[0]-1 and frame_stamps[index] < i * dt):
            index += 1
        
        f.write(f"{start} {index}\n")
        index += 1

