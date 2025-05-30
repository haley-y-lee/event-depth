import numpy as np
import torch
import glob
import cv2

def event_frames_to_voxel_grid(event_frames, timestamps, num_bins):
    """
    Computes the corresponding voxel grid for a list of event frames.

    Parameters:
        event_frames: N x height x width, where N is the number of event frames contributing to this voxel grid.
        timestamps: N-length list of timestamps for each event frame.
        num_bins: Number of bins for this voxel grid.
    """
    _, height, width = event_frames.shape

    voxel_grid = (torch.zeros((num_bins, height, width), dtype=torch.float32)).to("cuda:0")

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = timestamps[-1]
    first_stamp = timestamps[0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    ts = (num_bins - 1) * (timestamps - first_stamp) / deltaT   # normalized timestamps

    # Each event frame falls between two bins of the voxel grid, and contributes to both of these bins.

    # tis and tis_plus_1 represent the bins to the left and right of each event frame, respectively.
    tis = torch.floor(ts).to(torch.int)     # rounded-down timestamps
    tis_plus_1 = torch.clamp(tis + 1, max=num_bins - 1)

    dts = ts - tis      # each will be between 0 and 1. Represents weight to be placed on accumulation to left vs right bin

    vals_left = ((1 - dts)[:, None, None] * event_frames)
    vals_right = (dts[:, None, None] * event_frames)

    # Accumulate left side
    voxel_grid.index_add_(0, tis, vals_left)

    # Accumulate right side
    voxel_grid.index_add_(0, tis_plus_1, vals_right)

    return voxel_grid


def compute_event_frame(diff):
    """
    Implements equation (3) from "Differentiable Event Stream Simulator for Non-Rigid 3D Tracking.
    The parameter values for eps and w are chosen by my best guess. C is chosen to be what was used
    in the Vid2Events codebase for their event simulation method.
    """

    eps = 1e-4
    C = 0.2
    w = 100
    return ((diff + eps) / (torch.abs(diff) + eps)) * (1 / (1 + torch.exp(-w*torch.abs(diff)+w*C)))


# The "small" directory contains a small number of upsampled frames (10) and corresponding timestamps.
image_files = sorted(glob.glob("../../DENSE/small/frames/*.png"))
images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])

timestamps_s = np.genfromtxt("../../DENSE/small/frames/timestamps.txt")
timestamps_ns = (timestamps_s * 1e9).astype("int64")

log_images = np.log(images.astype("float32") / 255 + 1e-4)

print("Loading data to GPU")
device = "cuda:0"
log_images = torch.from_numpy(log_images).to(device)
timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

log_images.requires_grad_()

num_images = log_images.size()[0]
diffs = log_images[1:] - log_images[:num_images-1]  # compute difference images

event_frames = torch.stack([compute_event_frame(d) for d in diffs])

timestamps_ns = timestamps_ns[1:]   # If we say each difference image is computed as (cur-prev), we are letting the timestamp of cur be the timestamp for the event frame.
dt = float(5e7)     # dt (timespan for each voxel grid) is set to 50ms, following "Learning Monocular Dense Depth from Events"
start_time = timestamps_ns[0]
end_time = timestamps_ns[-1]

grid_indices = (torch.floor((timestamps_ns - start_time) / dt)).to(torch.int)   # computing which voxel grid each event frame corresponds to.
counts = torch.bincount(grid_indices)   # counts how many event frames are in each voxel grid, used for indexing
num_grids = counts.shape[0]

cur_index = 0
num_bins = 5

_ , height, width = event_frames.shape

voxel_grids = (torch.zeros((num_grids, num_bins, height, width), dtype=torch.float32)).to("cuda:0")

# compute voxel grids
for i in range(num_grids):
    voxel_grids[i,:] = event_frames_to_voxel_grid(event_frames[cur_index:cur_index+counts[i]], timestamps_ns[cur_index:cur_index+counts[i]], num_bins)

try:
    voxel_grids.backward(torch.ones_like(voxel_grids))
    breakpoint()
    print("Gradient computation ran successfully")
except:
    print("Something went wrong with gradient computation")
