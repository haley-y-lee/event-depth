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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', type=str,
                        help='path to the model weights')
    parser.add_argument('-i', '--base_folder', default=None, type=str,
                        help="name of the folder containing the frames and depths directories")
    parser.add_argument('--start_time', default=0.0, type=float)
    parser.add_argument('--stop_time', default=0.0, type=float)

    set_depth_inference_options(parser)

    args = parser.parse_args()

    print_every_n = 50

    # Load model to device
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model = model.to(device)
    model.eval()

    base_folder = args.base_folder
    event_folder = ""
    depth_folder = 'depths'
    frame_folder = 'frames'

    # hack to get the image size: create a dummy dataset,
    # grab the first data item and read the required info
    dummy_dataset = UpsampledFramesDataset(base_folder, event_folder, depth_folder, frame_folder, transform=False)
    # breakpoint()

    data = dummy_dataset[0]
    _, _, height, width = data['frames'].shape

    depth_reconstructor = PSFDepthReconstructor(model, height, width, model.num_bins, args)

    dataset = UpsampledFramesDataset(base_folder, event_folder, depth_folder, frame_folder, transform=False)

    output_dir = args.output_folder
    dataset_name = args.dataset_name
    print('Processing {}'.format(dataset_name))
    N = len(dataset)
    #print(f"[DEBUG] len(dataset): {N}") Passed
    # breakpoint()

    # are either of these needed?
    if output_dir is not None:
        shutil.copyfile(join(join(base_folder, depth_folder), 'timestamps.txt'),
                        join(output_dir, dataset_name, 'timestamps.txt'))
        # check if this is needed or not (seems like not)
        # shutil.copyfile(join(args.input_folder, 'boundary_timestamps.txt'),
        #                 join(output_dir, dataset_name, 'boundary_timestamps.txt'))

    idx = 0
    while idx < N:
        if idx % print_every_n == 0:
            print('{} / {}'.format(idx, N))

        data = dataset[idx]
        print(f"[DEBUG] data: {data}")
        #print(f"[DEBUG] size of data: {data.size()}")

        depth_reconstructor.update_reconstruction(data, idx)
        idx += 1
