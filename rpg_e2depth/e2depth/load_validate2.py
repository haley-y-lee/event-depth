# Auto-generated from: loadmodel copy 4.ipynb
# This script executes all notebook cells sequentially.
# Notes:
# - IPython magics and shell escapes were commented out for script safety.
# - The last cell was modified to print its final expression (or last assignment).
# - You can run: python "loadmodel_copy_4.py"


from tqdm import tqdm
################################################################################
# Cell 1
################################################################################
import torch
from utils.loading_utils import load_model, get_device
from data_loader.dataset import VoxelGridDataset
from matplotlib import pyplot as plt
from os.path import join, basename
import numpy as np
import json
import argparse
from utils.timers import cuda_timers
import time
import shutil
import os
from psf_depth_reconstructor import PSFDepthReconstructor
from options.inference_options import set_depth_inference_options
from data_loader.dataset import UpsampledFramesDataset
from types import SimpleNamespace
from model.unet import *
from types import SimpleNamespace
import torch
from model.model import E2VIDRecurrentPSF  
from model.model import E2VIDRecurrent
from model import unet as unet_mod          
from torch.utils.data import DataLoader
from data_loader.dataset import SequenceUpsampledFramesDataset
import matplotlib.image as mpimg
from model.unet import compute_event_frame, event_frames_to_voxel_grid
import torch.nn.functional as F


################################################################################
# Cell 2
################################################################################
## Defind Plot Grid Funcion
def plotgrid(stack):
    num_psf = stack.shape[0]  

    ncols = 5  # 원하는대로
    nrows = (num_psf + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))

    for i in range(num_psf):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        ax.imshow(stack[i].detach().numpy(), cmap='gray')
        ax.set_title(f"Depth {i+2}")   # depth index는 2부터일 수도 있으니 맞게 조절
        ax.axis('off')

    for i in range(num_psf, nrows*ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


################################################################################
# Cell 3
################################################################################
# %load_ext autoreload
# %autoreload 2

PATH_TO_MODEL = '/home/yl3836/mono_event/rpg_e2depth/e2depth/saved/1003_off_delta_0_5/example_psf/model_best.pth.tar'
tag = " [delta_0.5]"
BASE_FOLDER = '/home/yl3836/DENSE/valid_upsampled/valid_sequence_00_town06/'
#BASE_FOLDER = '/home/yl3836/DENSE/train_upsampled/train_sequence_03_town04/'
USE_GPU = True

args = SimpleNamespace(
    path_to_model=PATH_TO_MODEL,
    base_folder=BASE_FOLDER,
    use_gpu=USE_GPU,
)


################################################################################
# Cell 4
################################################################################
#### Load model
PATH = args.path_to_model
ckpt = torch.load(PATH, map_location='cpu')

#### Same config as used in training
cfg = dict(num_bins=5, skip_type='sum', recurrent_block_type='convlstm',
           num_encoders=3, base_num_channels=32,
           num_residual_blocks=2, use_upsample_conv=True, norm='none', psf_init='random')



#### Top-level model class
model = E2VIDRecurrentPSF(cfg)


######## Loading the weights
model.load_state_dict(ckpt['state_dict'])  

######## Evaluation mode : disable dropout
model.eval()


#### Load optimized psf
net = model.unetrecurrentpsf


################################################################################
# Cell 5
################################################################################
#### CPU safe collate copied from train.py 

def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x

def collate_keep_sequence(batch, device):
    # batch: [N] where each item is a sequence [L] of dicts
    return _to_device(batch, device)


################################################################################
# Cell 6
################################################################################
device = 'cpu' 

unet_mod.gpu = device   # used by UNetRecurrentPSF to move tensors
model.to(device).eval() 

#### load dataset
ds = SequenceUpsampledFramesDataset(
    base_folder=BASE_FOLDER,
    event_folder='',
    depth_folder='imgs',
    frame_folder='frames',
    sequence_length=20, step_size=1, normalize=True
)

loader = DataLoader(
    ds, batch_size=1, shuffle=False,
    collate_fn=lambda b: collate_keep_sequence(b, device),
    num_workers=0, pin_memory=False
)

batch = next(iter(loader))                

##### Forming sequence, copied from lstm.py
sequence = list(map(list, zip(*batch)))   
cur_input   = sequence[0]                 
prev_states = None

#### This is the forward model of top-level model
with torch.no_grad():
    voxel, frame, pred, states, timing = model.unetrecurrentpsf(cur_input, prev_states, downsample=False, measure_time = True)


################################################################################
# Cell 7
################################################################################
import os
import torch
import torch.nn.functional as F
import math
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
device = next(model.parameters()).device
use_amp = device.type == 'cuda'

def ensure_batched(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() >= 3 else x.unsqueeze(0)
mse_sum = torch.zeros((), device=device)
mse_sq_sum = torch.zeros((), device=device)
count_items = 0
model.eval()
autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16) if use_amp else torch.no_grad()
with torch.inference_mode():
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16) if use_amp else torch.no_grad():
        for batch in tqdm(loader, desc="Processing last cell"):
            sequence = list(map(list, zip(*batch)))
            cur_input = sequence[0]
            prev_states = None
            voxel, frame, pred, states = model.unetrecurrentpsf(cur_input, prev_states, downsample=False, measure_time=False)
            y_pred = pred.float().to(device, non_blocking=True).squeeze()
            y_true = frame[1].float().to(device, non_blocking=True).squeeze()
            y_pred = ensure_batched(y_pred)
            y_true = ensure_batched(y_true)
            diff = (y_pred - y_true).pow(2)
            mse_per_item = diff.flatten(1).mean(dim=1)
            mse_sum += mse_per_item.sum()
            mse_sq_sum += (mse_per_item ** 2).sum()
            count_items += mse_per_item.numel()
count = max(count_items, 1)
mean_mse = (mse_sum / count).item()
var_pop = mse_sq_sum / count - mean_mse ** 2
std_pop = float(torch.clamp(var_pop, min=0).sqrt().item())
if count > 1:
    var_sample = (mse_sq_sum - count * mean_mse ** 2) / (count - 1)
    std_sample = float(torch.clamp(var_sample, min=0).sqrt().item())
else:
    std_sample = float('nan')
print(f'Dataset: {BASE_FOLDER}')
print(f'Items evaluated: {count}')
print(f'Mean MSE: {mean_mse:.6f}')
print(f'Std MSE (population): {std_pop:.6f}')
print(print(f'Std MSE (sample):     {std_sample:.6f}'))

if __name__ == "__main__":
    # Everything is already at top-level; importing would not auto-run.
    # Running the script directly will execute all cells.
    pass