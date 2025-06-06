import torch
import cv2
import numpy as np
from model.model import *
from utils.inference_utils import CropParameters, IntensityRescaler, ImageDepthWriter
from utils.event_tensor_utils import EventPreprocessor
from utils.image_display_utils import DepthDisplay
from utils.util import robust_min, robust_max
from utils.timers import CudaTimer, cuda_timers
from os.path import join
from collections import deque
import torch.nn.functional as F


class PSFDepthReconstructor:
    def __init__(self, model, height, width, num_bins, options):

        self.model = model
        self.use_gpu = options.use_gpu
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.options = options

        self.initialize(self.height, self.width, self.options)

    def initialize(self, height, width, options):
        print('== Image reconstruction == ')
        print('Image size: {}x{}'.format(self.height, self.width))

        self.last_stamp = None

        self.no_recurrent = options.no_recurrent
        if self.no_recurrent:
            print('!!Recurrent connection disabled!!')

        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)

        self.last_states = None

        self.event_preprocessor = EventPreprocessor(options)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_writer = ImageDepthWriter(options)
        self.image_display = DepthDisplay(options)

    def update_reconstruction(self, data, index, stamp=None):
        # breakpoint()

        # max duration without events before we reinitialize
        self.max_duration_before_reinit_s = 5.0

        # we reinitialize if stamp < last_stamp, or if stamp > last_stamp + max_duration_before_reinit_s
        if stamp is not None and self.last_stamp is not None:
            if stamp < self.last_stamp or stamp > self.last_stamp + self.max_duration_before_reinit_s:
                print('Reinitialization detected!')
                self.initialize(self.height, self.width, self.options)

        self.last_stamp = stamp

        with torch.no_grad():
            # breakpoint()

            with CudaTimer('Reconstruction'):
                data['frames'] = data['frames'].permute(1, 0, 2, 3)
                data['frames'] = self.crop.pad(data['frames'])
                data['metric_depth'] = self.crop.pad(data['metric_depth'])
                data['frame'] = self.crop.pad(data['frame'])

                data = [data]
                
                with CudaTimer('Inference'):
                    voxel_grids, frame, new_predicted_frame, states = self.model(data, self.last_states, downsample=False)

                if self.no_recurrent:
                    self.last_states = None
                else:
                    self.last_states = states

                crop = self.crop

                with CudaTimer('Tensor (GPU) -> NumPy (CPU)'):
                    out = new_predicted_frame[0, 0, crop.iy0:crop.iy1, crop.ix0:crop.ix1].cpu().numpy()

            self.image_writer(out, index, stamp, events=voxel_grids)
            self.image_display(out, voxel_grids)
