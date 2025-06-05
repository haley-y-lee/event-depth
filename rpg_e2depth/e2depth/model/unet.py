import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU
from scipy.ndimage import gaussian_filter, rotate
import numpy as np

gpu = "cuda:0"

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


class BaseUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)


class UNet(BaseUNet):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNet, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                   num_encoders, base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size, output_size, kernel_size=5,
                                           stride=2, padding=2, norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNetRecurrent, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type=recurrent_block_type,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i])
            blocks.append(x)
            states.append(state)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img, states


# Code that I added for our model, could potentially be refactored into a separate file.

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

def event_frames_to_voxel_grid(event_frames, timestamps, num_bins=5):
    """
    Computes the corresponding voxel grid for a list of event frames.

    Parameters:
        event_frames: N x height x width, where N is the number of event frames contributing to this voxel grid.
        timestamps: N-length list of timestamps for each event frame.
        num_bins: Number of bins for this voxel grid.
    """
    _, height, width = event_frames.shape

    voxel_grid = (torch.zeros((num_bins, height, width), dtype=torch.float32)).to(gpu)

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

class DepthDependentPSFLayer(nn.Module):
    """Module containing a collection of learnable depth-dependent psfs"""
    def __init__(self, min_depth, max_depth, psf_init, psf_size=5):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.num_depths = self.max_depth - self.min_depth + 1
        self.psf_size = psf_size
        self.psf_init = psf_init

        # in config file, set psf initialization by setting the psf_init variable to 'random', 'rotated', or 'delta'

        if self.psf_init == 'random':
            psfs = torch.randn((self.num_depths, 1, psf_size, psf_size), device=gpu) * 1
            self.psfs = nn.Parameter(psfs)

        elif self.psf_init == 'rotated':
            angles = torch.linspace(0, 90, steps=self.num_depths)  

            for theta in angles:
                # create a 2D Gaussian 
                base = torch.zeros((psf_size, psf_size), dtype=torch.float32)
                center = psf_size // 2
                base[center, center] = 1.0
                gauss = gaussian_filter(base.numpy(), sigma=[1.0, 2.0]) 
                rotated = rotate(gauss, angle=float(theta), reshape=False, order=1, mode='nearest')

                # normalize to sum to 1
                rotated /= rotated.sum()

                # In forward, we apply softplus function to make psf nonnegative. Here we apply an approximate inverse
                # of the softplus function softplus^{-1}(x) ≈ log(exp(x) - 1) so that after application of softplus, 
                # we (approximately) are applying rotated gaussian psfs at initialization.
                eps = 1e-6
                inv_softplus = np.log(np.exp(rotated + eps) - 1.0)

                psfs.append(torch.tensor(inv_softplus, dtype=torch.float32))

            psfs = torch.stack(psfs, dim=0).unsqueeze(1).to(gpu)  # [num_depths, 1, psf_size, psf_size]

            self.psfs = nn.Parameter(psfs)

        elif self.psf_init == 'delta':
            # after applying the softplus function and normalization, these initial values of 5 at the center and -5
            # elsewhere will result in a psf that is approximately the delta function (1 at center and 0 elsewhere)
            psfs = torch.full((self.num_depths, 1, psf_size, psf_size), -5.0, device=gpu)  
            center = psf_size // 2
            psfs[:, 0, center, center] = 5.0
            self.psfs = nn.Parameter(psfs)


    def forward(self, image, depth_bins):
        """
        Applies depth-dependent psfs to sequence of frames.

        Parameters:
            image: sequence_length x 1 x height x width tensor containing sequence of frames
            depth_bins: 1 x height x width tensor containing depth map rounded to nearest integer
        """
        # breakpoint()
        output = torch.zeros_like(image)

        # iterate through depths
        for d in range(self.min_depth, self.max_depth+1):
            mask = (depth_bins == d).float() 

            if mask.sum() == 0:
                continue  # no pixels at this depth, skip

            raw_psf = self.psfs[d-self.min_depth:d-self.min_depth+1]               
            nonneg_psf = f.softplus(raw_psf)    # apply the softplus function to make psf nonnegative          
            psf = nonneg_psf / nonneg_psf.sum(dim=(-2, -1), keepdim=True)   # normalize psf to sum to 1
            psf = torch.flip(psf, dims=[-2, -1])    # flip to perform convolution instead of cross-correlation
            # breakpoint()
            
            filtered = f.conv2d(image, psf, padding=self.psf_size // 2)
            output += filtered * mask   # only counting contributions from pixels at depth d
        # breakpoint()
        return output


class UNetRecurrentPSF(nn.Module):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True, psf_init='random'):
        super().__init__()

        self.unet_recurrent = UNetRecurrent(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            skip_type=skip_type,
            recurrent_block_type=recurrent_block_type,
            activation=activation,
            num_encoders=num_encoders,
            base_num_channels=base_num_channels,
            num_residual_blocks=num_residual_blocks,
            norm=norm,
            use_upsample_conv=use_upsample_conv
        )

        self.psf_layer = DepthDependentPSFLayer(min_depth=2, max_depth=80, psf_init=psf_init, psf_size=9)

    def forward(self, sequence, prev_states):
        """
        :param cur_seq: N-length list
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        N = len(sequence)

        voxel_grid_list = []
        frame_list = []
        # breakpoint()

        for i in range(N):
            # move everything to gpu
            sequence[i]['frames'] = sequence[i]['frames'].to(gpu)
            sequence[i]['stamps'] = sequence[i]['stamps'].to(gpu)
            sequence[i]['frame'] = sequence[i]['frame'].to(gpu)
            sequence[i]['metric_depth'] = sequence[i]['metric_depth'].to(gpu)
            # breakpoint()

            # apply depth-dependent psfs
            depth = torch.clamp(sequence[i]['metric_depth'], min=2, max=80)
            depth_bins = torch.round(depth)
            convolved_frames = self.psf_layer(sequence[i]['frames'], depth_bins)

            # event simulation
            epsilon = 1e-6
            log_frames = torch.log(convolved_frames + epsilon)
            num_images = (sequence[i]['frames']).shape[0]

            diffs = log_frames[1:] - log_frames[:num_images-1]
            event_frames = torch.stack([compute_event_frame(d) for d in diffs])
            
            # voxel grid computation
            stamps = sequence[i]['stamps']
            stamps = stamps[1:]     # Each event frame is computed using diff of some frame_0 and frame_1. We use timestamp of frame_1 in voxel grid computation.
            stamps = stamps.float()

            voxel_grid = event_frames_to_voxel_grid(torch.squeeze(event_frames), stamps)
            voxel_grid = (voxel_grid - voxel_grid.mean()) / (voxel_grid.std() + epsilon)

            # downsampling (done to input events + depths by original model immediately after loading, we do it here since we need to apply psfs first)
            scale_factor = 0.5
            downsampled_voxel_grid = f.interpolate(voxel_grid.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=True)
            downsampled_frame = f.interpolate(sequence[i]['frame'].unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=True)

            voxel_grid_list.append(downsampled_voxel_grid)  # shape [1, C, H, W]
            frame_list.append(downsampled_frame)

        _, num_bins, height, width = voxel_grid_list[0].shape
    
        voxel_grids = torch.stack(voxel_grid_list).view(N, num_bins, height, width)
        frame = torch.stack(frame_list).view(N, 1, height, width)

        new_predicted_frame, states = self.unet_recurrent(voxel_grids, prev_states)

        return voxel_grids, frame, new_predicted_frame, states



