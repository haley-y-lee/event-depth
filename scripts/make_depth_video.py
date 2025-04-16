import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

vid_name = "test_100"

# Path to the folder containing .npy files that store depth estimates
input_folder = f'../{vid_name}_output/reconstruction/data'
output_video = f"../{vid_name}_metric_depth.mp4"

npy_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.npy')])

if not npy_files:
    raise ValueError("No .npy files found in the specified folder.")

arrays = [np.load(os.path.join(input_folder, file)) for file in npy_files]
reg_factor=3.70378
d_max = 80
arrays = np.array(arrays)
arrays = d_max * np.exp(-reg_factor * (np.ones(arrays.shape) - arrays))
global_min = min(arr.min() for arr in arrays)
global_max = max(arr.max() for arr in arrays)


# Process each frame
for i, (array, file) in enumerate(zip(arrays, npy_files)):
    normalized_array = (array - global_min) / (global_max - global_min)

    fig, ax = plt.subplots()
    
    img = ax.imshow(normalized_array, cmap='magma', vmin=0, vmax=1)
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Predicted Depth (m)")
    cbar.set_ticks([0, 1]) 
    cbar.set_ticklabels([f"{global_min:.2f}", f"{global_max:.2f}"])

    ax.set_xticks([])
    ax.set_yticks([])

    temp_frame_path = "temp_frame.png"
    plt.savefig(temp_frame_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

    frame = cv2.imread(temp_frame_path)

    if i == 0:
        # Video settings
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = frame.shape
        frame_size = (width, height)
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
        
    frame = cv2.resize(frame, frame_size) 
    video_writer.write(frame)
    print(f"Processed frame {i+1}/{len(npy_files)}")

video_writer.release()
print(f"Video saved as {output_video}")
