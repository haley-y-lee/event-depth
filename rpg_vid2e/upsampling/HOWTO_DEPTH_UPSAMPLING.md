# Full Upsampling Workflow (Frames + Depth)

This document describes the complete, end-to-end workflow for upsampling an event camera dataset, starting with the RGB frames and following up with the corresponding depth maps to ensure perfectly synchronized timestamps.

## Prerequisites & Setup

Before starting, ensure you have:
1. A sequence of original RGB frames.
2. A sequence of original depth maps corresponding to the same scene.
3. The pre-trained upsampling model downloaded. Follow the instructions at the [rpg_vid2e repository](https://github.com/uzh-rpg/rpg_vid2e) to download the required SuperSloMo model checkpoint.

### Requesting a GPU Node (SLURM)

If you are running this on a SLURM cluster, request an interactive node with a GPU before executing the upsampling scripts. You can use the following command:

```bash
srun --partition=monakhova --gpus=1 --mem=32G --cpus-per-task=4 --time=02:00:00 --pty bash
```

## Step 1: Upsample the RGB Frames

First, you must upsample the RGB frame sequence. This step will not only generate the high-framerate images but also produce a critical `timestamps.txt` file that dictates the exact time steps for the depth interpolation.

```bash
# Specify a GPU device for faster processing (or set to 'cpu')
device=0
# device=cpu

CUDA_VISIBLE_DEVICES=$device python upsample.py \
    --input_dir /share/monakhova/depthpsf_simul/square_dataset/test_sparse/test_seed_210 \
    --output_dir /share/monakhova/depthpsf_simul/square_dataset/test_sparse_upsampled/test_seed_210
```


```bash
cd /home/sp2577/computational_imaging/event-depth/rpg_vid2e/upsampling && \
CUDA_VISIBLE_DEVICES=0 python upsample.py \
    --input_dir /share/monakhova/depthpsf_simul/complex_dataset/complex_1 \
    --output_dir /share/monakhova/depthpsf_simul/complex_dataset/complex_1_upsampled
```
```bash
CUDA_VISIBLE_DEVICES=0 python upsample_depth.py \
    --input_dir /share/monakhova/depthpsf_simul/complex_dataset/complex_1 \
    --output_dir /share/monakhova/depthpsf_simul/complex_dataset/complex_1_upsampled \
    --timestamps_file /share/monakhova/depthpsf_simul/complex_dataset/complex_1_upsampled/timestamps.txt
```

## Step 2: Upsample the Depth Maps

Once the RGB frames have been upsampled, you use the newly generated `timestamps.txt` file to upsample the depth maps. This ensures the depth sequence matches the RGB sequence perfectly on a frame-by-frame basis.

```bash
CUDA_VISIBLE_DEVICES=$device python upsample_depth.py \
    --input_dir /share/monakhova/depthpsf_simul/square_dataset/test_sparse/test_seed_210 \
    --output_dir /share/monakhova/depthpsf_simul/square_dataset/test_sparse_upsampled/test_seed_210 \
    --timestamps_file /share/monakhova/depthpsf_simul/square_dataset/test_sparse_upsampled/test_seed_210/timestamps.txt
```
*(Note: Ensure the `--timestamps_file` points to the `timestamps.txt` created in Step 1 inside the upsampled output directory).*

## Directory Structures

**Expected Input Directory Structure:**
```text
/share/monakhova/depthpsf_simul/square_dataset/test_sparse/
├── test_seed_210
│   ├── frames
│   │   ├── 00000001.png
│   │   ├── 00000002.png
│   │   └── ...
│   └── imgs
│       ├── 00000001.png
│       ├── 00000002.png
│       └── ...
```

**Resulting Output Directory Structure:**
```text
/share/monakhova/depthpsf_simul/square_dataset/test_sparse_upsampled/
├── test_seed_210
│   ├── frames
│   │   ├── 00000001.png
│   │   ├── 00000002.png
│   │   └── ...
│   └── timestamps.txt
└── imgs
    ├── 00000001.png
    ├── 00000002.png
    └── ...
```

The upsampled `frames/` and `imgs/` directories will contain a larger, identical number of files, corresponding exactly one-to-one with the number of timestamps in the generated `timestamps.txt` file.
