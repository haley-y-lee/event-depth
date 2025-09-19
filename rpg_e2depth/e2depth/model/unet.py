import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU
from scipy.ndimage import gaussian_filter, rotate
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import torch.nn.functional as F
# matplotlib.use('Qt5Agg')
gpu = "cuda:0"
#gpu = 'cpu'

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

        self.decoders = nn. ModuleList()
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

        for i, decoder in enumerate(self.decoders):
            skip_connection = blocks[self.num_encoders - i - 1]

            # 공간 크기를 정확히 맞추기 위해 추가된 코드 (필수)
            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = f.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=True)

            x = decoder(self.apply_skip_connection(x, skip_connection))




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
        for i, decoder in enumerate(self.decoders):
            skip_connection = blocks[self.num_encoders - i - 1]

            # 공간 크기를 정확히 맞추기 위한 interpolation (필수)
            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = f.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=True)

            x = decoder(self.apply_skip_connection(x, skip_connection))


        if x.shape[-2:] != head.shape[-2:]:
            x = f.interpolate(x, size=head.shape[-2:], mode='bilinear', align_corners=True)


        # tail
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img, states



def compute_event_frame(diff):
    """
    Implements equation (3) from "Differentiable Event Stream Simulator for Non-Rigid 3D Tracking.
    The parameter values for eps and w are chosen by my best guess. C is chosen to be what was used
    in the Vid2Events codebase for their event simulation method.
    """

    eps = 1e-4
    C = 0.06
    w = 100
    return ((diff + eps) / (torch.abs(diff) + eps)) * (1 / (1 + torch.exp(-w*torch.abs(diff)+w*C)))


#  
def event_frames_to_voxel_grid(event_frames, timestamps, num_bins=5):
    """
    Computes the corresponding voxel grid for a list of event frames.
    """
    N, height, width = event_frames.shape

    # [NEW] 디바이스/타입 맞추기
    dev = event_frames.device
    timestamps = timestamps.to(dev).float()

    # [NEW] T 또는 T-1 모두 허용 (Δ프레임은 N=T-1)
    if timestamps.shape[0] == N + 1:
        timestamps = timestamps[1:]          # 오른쪽 경계에 정렬 (권장)
    elif timestamps.shape[0] != N:
        raise ValueError(f"mismatch: frames={N}, stamps={timestamps.shape[0]}")

    # [CHG] event_frames와 동일 디바이스로 생성
    voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float32, device=dev)


    first_stamp = timestamps[0]
    last_stamp = timestamps[-1]
    # [CHG] 0 나눗셈 방지
    deltaT = (last_stamp - first_stamp).clamp_min(1e-6)

    ts = (num_bins - 1) * (timestamps - first_stamp) / deltaT   # normalized timestamps

    ts = ts.clamp(0, num_bins - 1 - 1e-6)

    tis = torch.floor(ts).long()     
    tis_plus_1 = torch.clamp(tis + 1, max=num_bins - 1)

    dts = ts - tis      # weight for right bin

    # print(f"[DEBUG] dts: {dts.shape}")
    # print(f"[DEBUG] event_frames: {event_frames.shape}")

    vals_left = ((1 - dts)[:, None, None] * event_frames)
    vals_right = (dts[:, None, None] * event_frames)

    # Accumulate left side
    voxel_grid.index_add_(0, tis, vals_left)

    # Accumulate right side
    voxel_grid.index_add_(0, tis_plus_1, vals_right)

    return voxel_grid



def voxel_grid_batched(event_frames, timestamps, num_bins=5):
    """
    event_frames : (B, N, H, W)
    timestamps   : (B, N)
    returns      : (B, num_bins, H, W)
    """
    # print(f"[DEBUG] event_frames shape: {event_frames.shape}")
    B, N, H, W = event_frames.shape
    device = event_frames.device

    # 1) timestamp 정규화 (각 배치별 0~num_bins-1)
    t0 = timestamps[:, :1]                          # (B,1)
    dt = (timestamps[:, -1:] - t0).clamp(min=1e-6)  # (B,1)
    ts = (num_bins - 1) * (timestamps - t0) / dt    # (B,N)

    #### CODE ADDED 0807 
    ts = ts.clamp_min(0.).clamp_max(num_bins - 1 - 1e-4)  # ★추가: 0‥num_bins-1 로 한정
    tis = ts.floor().long()  

    tis        = ts.floor().long()                  # (B,N)
    tis_plus_1 = (tis + 1).clamp(max=num_bins-1)
    dts        = ts - tis                           # (B,N)

    # 2) 좌/우 bin 으로 분배할 값
    vals_left  = (1 - dts)[..., None, None] * event_frames   # (B,N,H,W)
    vals_right =       dts[..., None, None] * event_frames   # (B,N,H,W)

    # 3) (B,N) → (B*N) 로 평탄화하면서 배치별 bin 오프셋 추가
    offset      = (torch.arange(B, device=device) * num_bins).view(B, 1)
    idx_left    =  (tis        + offset).reshape(-1)         # (B*N,)
    idx_right   =  (tis_plus_1 + offset).reshape(-1)

    vals_left   = vals_left .reshape(-1, H, W)               # (B*N,H,W)
    vals_right  = vals_right.reshape(-1, H, W)

    # 4) 한 번에 누적
    voxel = torch.zeros((B * num_bins, H, W), device=device)
    voxel.index_add_(0, idx_left , vals_left )
    voxel.index_add_(0, idx_right, vals_right)

    return voxel.view(B, num_bins, H, W)


class DepthDependentPSFLayer(nn.Module): 
    """Module containing a collection of learnable depth-dependent psfs"""
    def __init__(self, min_depth, max_depth, psf_init, psf_size=9,lower=None, higher=None):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.num_depths = self.max_depth - self.min_depth + 1
        self.psf_size = psf_size
        self.psf_init = psf_init

        # in config file, set psf initialization by setting the psf_init variable to 'random', 'rotated', or 'delta'

        if self.psf_init == 'random':
            psfs = torch.randn((self.num_depths, 1, psf_size, psf_size), device=gpu) * 10000
            self.psfs = nn.Parameter(psfs)

        if self.psf_init == 'rotated':
            angles = torch.linspace(0, 90, steps=self.num_depths)  
            psf_list = [] 

            for theta in angles: 
                base = torch.zeros((psf_size, psf_size), dtype=torch.float32)
                center = psf_size // 2
                base[center, center] = 1

                gauss = gaussian_filter(base.numpy(), sigma=[0.5, 0.5]) 
                rotated = rotate(gauss, angle=float(theta), reshape=False, order=1, mode='nearest')
                rotated = (rotated-rotated.min())/(rotated.max() - rotated.min())
                rotated = rotated + 1e-6
                rotated = np.clip(rotated, 1e-6, 1.0)
                psf_list.append(torch.tensor(rotated, dtype=torch.float32))
                # eps = 1e-6
                # inv_softplus = np.log(np.exp(rotated + eps) - 1.0)



            psfs = torch.stack(psf_list, dim=0).unsqueeze(1).to(gpu)  # [num_depths, 1, psf_size, psf_size]

            self.psfs = nn.Parameter(psfs, requires_grad=False)

        if self.psf_init == 'gaussian':                 

            fixed_sigma = 1.3887
            sigmas = torch.full((self.num_depths,), fixed_sigma)  
            psf_list = []
            eps   = 1e-6
            floor = 0.002            # tail 최소값


            for σ in sigmas:
                # 1) Gaussian
                ax  = torch.arange(psf_size, dtype=torch.float32) - psf_size // 2
                xx, yy = torch.meshgrid(ax, ax, indexing='ij')
                gauss = torch.exp(-(xx**2 + yy**2) / (2 * σ**2))

                # 2) 합을 1로 정규화 ─ 모양만 유지
                gauss /= gauss.sum()

                # 3) σ² 만큼 스케일 ↑  → 깊이 커질수록 더 강한 블러
                gauss *= σ**2

                # 4) 바닥값 주기 (tail 살리기)
                gauss = gauss * (1 - floor) + floor

                # 5) softplus 역변환 (수치 안정용 eps)
                inv_softplus = torch.log(torch.expm1(gauss.clamp_min(eps)))

                psf_list.append(inv_softplus)

            psfs = torch.stack(psf_list, dim=0).unsqueeze(1).to(gpu)

            self.psfs = nn.Parameter(psfs)



        if self.psf_init == 'delta':

            # psfs = torch.full((self.num_depths, 1, psf_size, psf_size), -5.0, device=gpu)  
            # center = psf_size // 2
            # psfs[:, 0, center, center] = 5.0
            
            # #self.psfs = nn.Parameter(psfs)
            # self.psfs = nn.Parameter(psfs, requires_grad=False)


            psfs = torch.full((self.num_depths, 1, psf_size, psf_size), -5.0, device=gpu)  
            center = psf_size // 2
            psfs[:, 0, center, center] = 5.0
            psfs = f.softplus(psfs)
            #self.psfs = nn.Parameter(psfs)
            psfs = psfs/psfs.sum(dim=(-2,-1),keepdim = True)

            #psfs = (psfs-psfs.min())/(psfs.max() - psfs.min())

               # psf_list.append(torch.tensor(rotated, dtype=torch.float32))
            self.psfs = nn.Parameter(psfs)
            #self.psfs = nn.Parameter(psfs, requires_grad=False)

        ##### Commented on 0829 #####
        # if lower is None or higher is None:
        #     # 균일한 1단위 구간: 2,3,…,10  (예시)
        #     lower = torch.arange(min_depth, max_depth + 1, device=gpu, dtype=torch.float32)
        #     higher = lower + 1.
        # else:
        #     lower = torch.as_tensor(lower, dtype=torch.float32, device=gpu)
        #     higher = torch.as_tensor(higher, dtype=torch.float32, device=gpu)

        # assert len(lower) == self.num_depths and len(higher) == self.num_depths, \
        #     "lower/higher 길이는 num_depths 와 같아야 합니다."

        # # 학습 대상은 아니므로 buffer 로 등록
        # self.register_buffer("lower", lower)
        # self.register_buffer("higher", higher)


    def save_psf_stack(self, step):
        with torch.no_grad():
            # if self.batch_step % 50 == 0:
                # psf_stack = f.softplus(self.psfs)
                # psf_stack = psf_stack / psf_stack.sum(dim=(-2, -1), keepdim=True)
                # psf_stack = torch.flip(psf_stack, dims=[-2, -1])
                psf_stack = self.psfs
                os.makedirs("/home/yl3836/saved_psfs_test", exist_ok=True)
                torch.save(psf_stack.cpu(), f"/home/yl3836/saved_psfs_test/epoch_{step:05d}.pt")


    def forward(self, image):
        """
        Applies depth-dependent psfs to sequence of frames.

        Parameters:
            image: sequence_length x 1 x height x width tensor containing sequence of frames
            depth_bins: 1 x height x width tensor containing depth map rounded to nearest integer
        """

        image = image.squeeze(0)
        T, _, H, W = image.shape
        D = self.num_depths
        k = self.psf_size

        ######################## 09032025 : Softplus removed for delta function ###############################
        
        
        # scaled = psfs
        # psfs  = scaled / scaled.sum(dim=(-2, -1), keepdim=True) * (self.psf_size ** 2)
        # psfs = torch.flip(psfs, (-2,-1))
        psfs = self.psfs
    
        psfs = psfs/psfs.sum(dim=(-2,-1),keepdim = True)
        #######################################################


        ###### CODE COMMENTED 0829 ########
        # filtered = f.conv2d(image, psfs, padding=self.psf_size // 2)

        # #### Setting up depth indexing
        # boundaries = self.higher[:-1]  
        # depth_idx = torch.bucketize(depth_bins.squeeze(0), boundaries)  # [H,W], 0..D-1
        # depth_idx_plus = torch.clamp(depth_idx + 1, max=D-1) 

      
        # w = (depth_bins.squeeze(0) - self.lower[depth_idx]) / \
        #     (self.higher[depth_idx] - self.lower[depth_idx]+ 1e-8) 
        # w = w.expand(T, 1, H, W)
  
        # idx_e      = depth_idx     .expand(T, H, W).unsqueeze(1)               # [T,1,H,W]
        # idx_e_plus = depth_idx_plus.expand(T, H, W).unsqueeze(1)               # [T,1,H,W]
        # out_d        = torch.gather(filtered, 1, idx_e)                  # PSF_d
        # out_d_plus   = torch.gather(filtered, 1, idx_e_plus)              # PSF_{d+1}

                               
        # output = (1 - w) * out_d + w * out_d_plus


        return psfs

class UNetRecurrentPSF(nn.Module):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    Additionally contains a DepthDependentPSF layer which is applied prior to the UNet.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True, psf_init='delta', scale_factor=1, max_depth = 31):
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

        ##################################################################################################
        self.psf_layer = DepthDependentPSFLayer(min_depth=2, max_depth=31, psf_init='rotated', psf_size=21)
        ###### Change the depth bin Depth as well!! ########

        #################################################################################################
        self.scale_factor = scale_factor
        self.max_depth = max_depth

    ##### ORIGINIAL CODE ###########

    def forward(self, cur_input, prev_states, downsample=False, measure_time=True):
        """
        :param cur_input: batch of input data (formatted by UpsampledFramesDataset class)
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        min_depth = 2.0
        max_depth = 31.0

        N = len(cur_input)

        voxel_grid_list = []
        frame_list = []
        # breakpoint()

        psf_time = 0
        sim_time = 0
        voxel_time = 0
        pred_time = 0


        for i in range(N):     
            # move everything to gpu
            cur_input[i]['frames'] = cur_input[i]['frames'].to(gpu)
            cur_input[i]['stamps'] = cur_input[i]['stamps'].to(gpu)
            cur_input[i]['frame'] = cur_input[i]['frame'].to(gpu)
            cur_input[i]['metric_depth'] = cur_input[i]['metric_depth'].to(gpu)

        if measure_time: t0 = time.time()



        try:
            depth = cur_input[i]['metric_depth']
            # depth = depth / 255.0
            # depth = depth * (max_depth - min_depth) + 2
            #depth = depth * (max_depth - 2) + 2
            depth = depth * (max_depth - min_depth) + min_depth

        except Exception as e:
            print(f"[ERROR in depth calculation] cur_input[{i}]['metric_depth']: {cur_input[i].get('metric_depth', 'N/A')}")
            raise e

        ##### CODE ADDED 0829 #####
        depth_bins = depth
        lower = torch.arange(min_depth, max_depth+1, device = gpu)
        higher = lower + 1
        lower = lower.unsqueeze(-1).unsqueeze(-1) # Shape [1,1,15] for broadcast
        higher = higher.unsqueeze(-1).unsqueeze(-1) # Shape [1,1,15] for broadcast

        mask = ((depth_bins >= lower) & (depth_bins < higher)).float()  
        mask = mask.to(gpu)

        # print(f"[DEBUG] shape of mask : {mask.shape}")
        # print(f"[DEBUG] shape of cur_input[i]['frames'] : {cur_input[i]['frames'].shape}")

        masked_frames = mask * cur_input[i]['frames']
        # print(f"shape of masked_frames : {masked_frames.shape}")
        #convolved_frames = self.psf_layer(masked_frames)

        initialized_psfs = self.psf_layer(masked_frames)

        ###### DEBUG PRINT PSF WEIGHTS #########
        # print(f"[DEBUG initialized psfs] : {initialized_psfs}")
        # print(f"[DEBUG psf weights] : {self.psf_layer.psfs}")

        C = masked_frames.shape[1]
        convolved_frames = F.conv2d(masked_frames, initialized_psfs, padding="same", groups=C)
        convolved_frames = convolved_frames.sum(dim=1)

        if measure_time: psf_time += time.time() - t0

        ##### print frames #######
        #img_test = cur_input[i]['frames']

        #conv_np_test = (convolved_frames.detach().cpu().numpy()*255).astype(np.uint8)

        # print(f"shape of convolved_frames : {convolved_frames.shape}")
        if measure_time: t0 = time.time()
        epsilon = 1e-6
        log_frames = torch.log(convolved_frames + epsilon)

        diffs = log_frames[1:] - log_frames[:-1] 

        event_frames = compute_event_frame(diffs)  # [T-1, H, W]
        # print(f"shape of event_frames : {event_frames.shape}")

        #num_images = cur_input[i]['frames'].shape[0]
        
        #event_np_test = (event_frames.detach().cpu().numpy()*255).astype(np.uint8)
    

        if measure_time: t0 = time.time()

        voxel_grid = event_frames_to_voxel_grid(event_frames.squeeze(1), cur_input[i]['stamps'], num_bins=5)
        
        if measure_time: voxel_time += time.time() - t0
        if downsample:
            pass
        else:
            downsampled_voxel_grid = voxel_grid 
            frame = cur_input[i]['frame']  
            downsampled_frame = frame.unsqueeze(0)


        voxel_grid_list.append(downsampled_voxel_grid)  # shape [1, C, H, W]
        frame_list.append(downsampled_frame)
        

        voxel_grids = torch.stack(voxel_grid_list, dim=0)  # [N, C, H, W]
        frame = torch.stack(frame_list, dim=0)             # [N, 1, H, W]
        frame = frame.squeeze(1)  # [N, H, W] - remove channel dimension
        frame = frame.squeeze(0)
        
        # breakpoint()
        if measure_time: t0 = time.time()
        new_predicted_frame, states = self.unet_recurrent(voxel_grids, prev_states)
        if measure_time: pred_time = time.time() - t0

        # print(f"[DEBUG] shape of frame : {frame.shape}")
        # print(f"[DEBUG] shape of new_predicted_frame : {new_predicted_frame.shape}")
        if measure_time:
            return voxel_grids, frame, new_predicted_frame, states, {
                'psf_time': psf_time,
                'sim_time': sim_time,
                'voxel_time': voxel_time,
                'pred_time': pred_time
            }
        else:
            return voxel_grids, frame, new_predicted_frame, states
