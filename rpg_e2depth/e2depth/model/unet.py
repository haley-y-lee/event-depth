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
# matplotlib.use('Qt5Agg')

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
        # for i, decoder in enumerate(self.decoders):
        #     x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))
        # decoder
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

        # decoder
        # for i, decoder in enumerate(self.decoders):
        #     x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))
        # decoder
        for i, decoder in enumerate(self.decoders):
            skip_connection = blocks[self.num_encoders - i - 1]

            # 공간 크기를 정확히 맞추기 위한 interpolation (필수)
            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = f.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=True)

            x = decoder(self.apply_skip_connection(x, skip_connection))
        # event_tensor shape: [num_bins, H, W] or [1, num_bins, H, W]

        if x.shape[-2:] != head.shape[-2:]:
            x = f.interpolate(x, size=head.shape[-2:], mode='bilinear', align_corners=True)


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
    C = 0.25
    w = 100
    return ((diff + eps) / (torch.abs(diff) + eps)) * (1 / (1 + torch.exp(-w*torch.abs(diff)+w*C)))

# def event_frames_to_voxel_grid(event_frames, timestamps, num_bins=5):
#     """
#     Computes the corresponding voxel grid for a list of event frames.

#     Parameters:
#         event_frames: N x height x width, where N is the number of event frames contributing to this voxel grid.
#         timestamps: N-length list of timestamps for each event frame.
#         num_bins: Number of bins for this voxel grid.
#     """
    
#     _, height, width = event_frames.shape

#     voxel_grid = (torch.zeros((num_bins, height, width), dtype=torch.float32)).to(gpu)

#     # normalize the event timestamps so that they lie between 0 and num_bins
#     last_stamp = timestamps[-1]
#     first_stamp = timestamps[0]
#     deltaT = last_stamp - first_stamp

#     if deltaT == 0:
#         deltaT = 1.0

#     ts = (num_bins - 1) * (timestamps - first_stamp) / deltaT   # normalized timestamps

#     # Each event frame falls between two bins of the voxel grid, and contributes to both of these bins.

#     # tis and tis_plus_1 represent the bins to the left and right of each event frame, respectively.
#     tis = torch.floor(ts).to(torch.int)     # rounded-down timestamps
#     tis_plus_1 = torch.clamp(tis + 1, max=num_bins - 1)

#     dts = ts - tis      # each will be between 0 and 1. Represents weight to be placed on accumulation to left vs right bin

#     print(f"[DEBUG] dts: {dts.shape}")
#     print(f"[DEBUG] event_frames: {event_frames.shape}")

#     vals_left = ((1 - dts)[:, None, None] * event_frames)
#     vals_right = (dts[:, None, None] * event_frames)

#     # Accumulate left side
#     voxel_grid.index_add_(0, tis, vals_left)

#     # Accumulate right side
#     voxel_grid.index_add_(0, tis_plus_1, vals_right)

#     return voxel_grid
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

    # normalize the event timestamps so that they lie between 0 and num_bins
    first_stamp = timestamps[0]
    last_stamp = timestamps[-1]
    # [CHG] 0 나눗셈 방지
    deltaT = (last_stamp - first_stamp).clamp_min(1e-6)

    ts = (num_bins - 1) * (timestamps - first_stamp) / deltaT   # normalized timestamps
    # [OPT] 경계 살짝 보호 (희귀 케이스)
    ts = ts.clamp(0, num_bins - 1 - 1e-6)

    # tis and tis_plus_1 represent the bins to the left and right of each event frame, respectively.
    # [CHG] 인덱스는 long이 안전
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



# voxel_grid = voxel_grid_batched(torch.squeeze(event_frames), stamps )
# print(f"squeezed event_frames shape: {torch.squeeze(event_frames).shape}")


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

                psf_list.append(torch.tensor(inv_softplus, dtype=torch.float32))

            psfs = torch.stack(psf_list, dim=0).unsqueeze(1).to(gpu)  # [num_depths, 1, psf_size, psf_size]

            self.psfs = nn.Parameter(psfs)

        if self.psf_init == 'gaussian':                   # <-- 새 옵션
            # sigmas   = torch.linspace(0.5, 5.0, steps=self.num_depths)   # 깊이에 따라 σ 변화
            # #sigmas = torch.logspace(math.log10(0.4), math.log10(8.0), steps=self.num_depths)
            # psf_list = []
            # for σ in sigmas:
            #     # ① 2‑D isotropic Gaussian kernel
            #     ax  = torch.arange(psf_size, dtype=torch.float32) - psf_size // 2
            #     xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            #     gauss   = torch.exp(-(xx**2 + yy**2) / (2*σ**2))
            #     gauss  /= gauss.sum()                    # 합이 1이 되도록 정규화

            #     # ② softplus^-1 로 역변환 (forward 에서 다시 softplus 적용되므로)
            #     eps = 1e-6
            #     inv_softplus = torch.log(torch.exp(gauss + eps) - 1.0)
            #     psf_list.append(inv_softplus)

            # psfs = torch.stack(psf_list, dim=0).unsqueeze(1).to(gpu)
            # self.psfs = nn.Parameter(psfs)

            
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
            # after applying the softplus function and normalization, these initial values of 5 at the center and -5
            # elsewhere will result in a psf that is approximately the delta function (1 at center and 0 elsewhere)
            psfs = torch.full((self.num_depths, 1, psf_size, psf_size), -2.0, device=gpu)  
            center = psf_size // 2
            psfs[:, 0, center, center] = 10.0
            self.psfs = nn.Parameter(psfs)

        if lower is None or higher is None:
            # 균일한 1단위 구간: 2,3,…,10  (예시)
            lower = torch.arange(min_depth, max_depth + 1, device=gpu, dtype=torch.float32)
            higher = lower + 1.
        else:
            lower = torch.as_tensor(lower, dtype=torch.float32, device=gpu)
            higher = torch.as_tensor(higher, dtype=torch.float32, device=gpu)

        assert len(lower) == self.num_depths and len(higher) == self.num_depths, \
            "lower/higher 길이는 num_depths 와 같아야 합니다."

        # 학습 대상은 아니므로 buffer 로 등록
        self.register_buffer("lower", lower)
        self.register_buffer("higher", higher)


    def save_psf_stack(self, step):
        with torch.no_grad():
            # if self.batch_step % 50 == 0:
                psf_stack = f.softplus(self.psfs)
                #psf_stack = self.psfs
                psf_stack = psf_stack / psf_stack.sum(dim=(-2, -1), keepdim=True)
                psf_stack = torch.flip(psf_stack, dims=[-2, -1])
                os.makedirs("/home/yl3836/saved_psfs_test", exist_ok=True)
                torch.save(psf_stack.cpu(), f"/home/yl3836/saved_psfs_test/epoch_{step:05d}.pt")


    def forward(self, image, depth_bins):
        """
        Applies depth-dependent psfs to sequence of frames.

        Parameters:
            image: sequence_length x 1 x height x width tensor containing sequence of frames
            depth_bins: 1 x height x width tensor containing depth map rounded to nearest integer
        """

        # # forward() 제일 앞에 추가
        # with torch.no_grad():
        #     print("unique depth values ->", torch.unique(depth_bins))
        #     depth_idx_tmp = torch.bucketize(depth_bins.squeeze(0), self.higher[:-1])
        #     print("unique depth_idx     ->", torch.unique(depth_idx_tmp))
        # print(f"[DEBUG] image shape: {image.shape}")
        image = image.squeeze(0)
        T, _, H, W = image.shape
        D = self.num_depths
        k = self.psf_size

        psfs = f.softplus(self.psfs)
        
        #psfs = self.psfs
        # log_k = self.psfs             # unconstrained
        # psfs  = torch.softmax(log_k, (-2, -1)) 
        
        # #psfs = self.psfs
        # psf_min  = psfs.amin(dim=(-2, -1), keepdim=True)
        # psf_max  = psfs.amax(dim=(-2, -1), keepdim=True)

        # scaled  = (psfs - psf_min) / (psf_max - psf_min + 1e-8)  # 0‥1
        scaled = psfs
        psfs  = scaled / scaled.sum(dim=(-2, -1), keepdim=True) * (self.psf_size ** 2)
        #psfs  = scaled / scaled.sum(dim=(-2, -1), keepdim=True) 
        psfs = torch.flip(psfs, (-2,-1))

        filtered = f.conv2d(image, psfs, padding=self.psf_size // 2)
        # Image Shape = [C,1,H,W] / Psfs Shape = [D,1,H,W] / Filtered Shape = [C,D,H,W]

        # print(f"lower : {self.lower}")
        # print(f"higher: {self.higher}")

        boundaries = self.higher[:-1]  
        depth_idx = torch.bucketize(depth_bins.squeeze(0), boundaries)  # [H,W], 0..D-1
        depth_idx_plus = torch.clamp(depth_idx + 1, max=D-1) 

        ######## NEW CODE : SOFT ASSIGN ##########
        # (2) 같은 픽셀 안에서 0‥1 위치 가중치 w 계산
        w = (depth_bins.squeeze(0) - self.lower[depth_idx]) / \
            (self.higher[depth_idx] - self.lower[depth_idx]+ 1e-8) 
        w = w.expand(T, 1, H, W)
                # (3) 두 채널 모두 모아서
        idx_e      = depth_idx     .expand(T, H, W).unsqueeze(1)               # [T,1,H,W]
        idx_e_plus = depth_idx_plus.expand(T, H, W).unsqueeze(1)               # [T,1,H,W]
        out_d        = torch.gather(filtered, 1, idx_e)                  # PSF_d
        out_d_plus   = torch.gather(filtered, 1, idx_e_plus)              # PSF_{d+1}

        # (4) 가중합
        #w_e = w.expand(T, 1, H, W)                                             # [T,1,H,W]
        output = (1 - w) * out_d + w * out_d_plus
        ##########################################


        # ###### HARD ASSIGN #######
        # # ---- (4)  gather 로 픽셀별 채널 선택 ---------------------------
        # depth_idx = depth_idx.expand(T, H, W).unsqueeze(1)            # [T,1,H,W]
        # # print(f"depth idx : {depth_idx}")
        # output = torch.gather(filtered, 1, depth_idx).contiguous()    # [T,1,H,W]
        # #########################

        return output



        # ##### ORIGINIAL CODE ###########
        # # breakpoint()
        # output = torch.zeros_like(image)

        # # iterate through depths
        # for d in range(self.min_depth, self.max_depth+1):
        #     mask = (depth_bins == d).float() 

        #     if mask.sum() == 0:
        #         continue  # no pixels at this depth, skip
        #     # try:
        #     #     print(f"self.psf size: {self.psfs.shape}") [79,1,9,9]
        #     # except:
        #     #     pass


        #     ########## PRINT PSF ############
        #     #print("PSFs: ", self.psfs[0])
        #     #################################



        #     raw_psf = self.psfs[d-self.min_depth:d-self.min_depth+1]  
        #     # try:
        #     #     print(f"self.psf size: {self.psfs.shape}")
        #     # except:
        #     #     pass     
        #     nonneg_psf = f.softplus(raw_psf)    # apply the softplus function to make psf nonnegative      
        #     #nonneg_psf = torch.relu(raw_psf)    
        #     psf = nonneg_psf / nonneg_psf.sum(dim=(-2, -1), keepdim=True)   # normalize psf to sum to 1
        #     psf = torch.flip(psf, dims=[-2, -1])    # flip to perform convolution instead of cross-correlation
        #     # breakpoint()
        #     # try:
        #     #     print(f"image size : {image.shape}")
        #     #     print(f"psf size : {psf.shape}")
        #     # except:
        #     #     pass

        # #### SAVE PSF #####
        # # with torch.no_grad():
        # #     if self.batch_step % 50 == 0:
        # #         psf_stack = f.softplus(self.psfs)
        # #         psf_stack = psf_stack / psf_stack.sum(dim=(-2, -1), keepdim=True)
        # #         psf_stack = torch.flip(psf_stack, dims=[-2, -1])
        # #         os.makedirs("/home/yl3836/DENSE/saved_psfs_test", exist_ok=True)
        # #         torch.save(psf_stack.cpu(), f"/home/yl3836/DENSE/saved_psfs/psf_step_{self.batch_step:05d}.pt")




        #     filtered = f.conv2d(image, psf, padding=self.psf_size // 2)
        #     output += filtered * mask   # only counting contributions from pixels at depth d

        # # breakpoint()
        # return output
    


        ################################

class UNetRecurrentPSF(nn.Module):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    Additionally contains a DepthDependentPSF layer which is applied prior to the UNet.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True, psf_init='gaussian', scale_factor=1):
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

        self.psf_layer = DepthDependentPSFLayer(min_depth=2, max_depth=80, psf_init='gaussian', psf_size=5)
        self.scale_factor = scale_factor

    ##### ORIGINIAL CODE ###########

    def forward(self, cur_input, prev_states, downsample=False, measure_time=True):
        """
        :param cur_input: batch of input data (formatted by UpsampledFramesDataset class)
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        N = len(cur_input)
        # try:
        #     print(f"[DEBUG] len(cur_input)= {N}", flush = True)
        # except:
        #     pass

        voxel_grid_list = []
        frame_list = []
        # breakpoint()

        psf_time = 0
        sim_time = 0
        voxel_time = 0
        pred_time = 0

        #### 072925 COOE ADDED ####
        # device = next(self.parameters()).device 
        # frames = cur_input['frames'].to(device, non_blocking=True)
        # stamps = cur_input['stamps'].to(device, non_blocking=True)
        # metric_depth  = cur_input['metric_depth'].to(device, non_blocking=True)
        # frame= cur_input['frame'].to(device, non_blocking=True)
        # N, T, _, H, W = frames.shape

        ##### 080725 Code Added ######################

        # frames = cur_input['frames']
        # stamps = cur_input['stamps']
        # frame = cur_input['frame']
        # metric_depth = cur_input['metric_depth']
        # frame= cur_input['frame']
        # N, T, _, H, W = frames.shape

        # depth_min = 0
        # depth_max = 1

        for i in range(N):     
            # move everything to gpu
            # breakpoint()
            cur_input[i]['frames'] = cur_input[i]['frames'].to(gpu)
            #print ("frames size: ", cur_input[i]['frames'].size())
            cur_input[i]['stamps'] = cur_input[i]['stamps'].to(gpu)
            cur_input[i]['frame'] = cur_input[i]['frame'].to(gpu)
            cur_input[i]['metric_depth'] = cur_input[i]['metric_depth'].to(gpu)

            ###########
            # cur_input[i]['events'] =  cur_input[i]['events'].to(gpu)
            
            # # breakpoint()
            # event_tensor = cur_input[i]['events']
            # event_np = event_tensor[0, 0].detach().cpu().numpy()  # (H, W)
            # event_np_img = (event_np * 255 / np.max(np.abs(event_np))).astype(np.uint8)

            # from PIL import Image
            # Image.fromarray(event_np_img).save("/home/yl3836/non_psf_event_test.jpeg")

            ############



            # apply depth-dependent psfs
        if measure_time: t0 = time.time()
        #depth = torch.clamp(cur_input[i]['metric_depth'], min=2, max=80)

        ### CHANGE DEPTH METHOD ###
        try:
            # depth = (cur_input[i]['metric_depth'] - depth_min) / (depth_max - depth_min)
            depth = cur_input[i]['metric_depth']
            #print(f"cur_input[i]['metric_depth'] shape :, {cur_input[i]['metric_depth'].shape}")
            #print(f"[RAW] depth max {depth.max()}, depth min {depth.min()}")
            depth = depth / 255.0
            #print(f"[DIVIDE] depth max {depth.max()}, depth min {depth.min()}")
            depth = depth * (80 - 2) + 2
            #print(f"[CALC] depth max {depth.max()}, depth min {depth.min()}")
        except Exception as e:
            print(f"[ERROR in depth calculation] cur_input[{i}]['metric_depth']: {cur_input[i].get('metric_depth', 'N/A')}")
            raise e
        # print(f"Before round depth bins : {depth}")
        # #depth_bins = torch.round(depth) # [1, 206, 346]
        # print(f"After round depth bins : {depth_bins}")

        depth_bins = depth

        #print ("depth_bins size: ", depth_bins.size())

        # convolved_frames = self.psf_layer(frames, depth_bins)
        convolved_frames = self.psf_layer(cur_input[i]['frames'], depth_bins)
        # filtered = self.psf_layer(frames.view(-1,1,H,W),
        #                   depth_bins.view(-1,H,W))           # (B*T,1,H,W)
        # convolved_frames = filtered.view(B, T, 1, H, W)


        # print ("convolved_frames size: ", convolved_frames.size())
        if measure_time: psf_time += time.time() - t0

        ##### print frames #######
        img_test = cur_input[i]['frames']
        # try:
        #     print(f"img_test shape: {img_test.shape}")
            
        # except:
        #     pass
        # img_test = (img_test.detach().cpu().numpy()*255).astype(np.uint8)

        # im = Image.fromarray(img_test[0,0,:,:])
        # im.save("/home/yl3836/frame_test.jpeg")

    


        # im = Image.fromarray(conv_np_test[0,0,:,:])
        # im.save("/home/yl3836/conv_test.jpeg")
        # exit(0)


        # try:
        #     #print(f"img_test shape: {img_test.shape}")
        #     print(f"max_img : {max(img_test[0,0,:,:])}")
        #     print(f"convolved shape: {convolved_frames.shape}")
        #     print(f"{max(convolved_frames[0,0,:,:])}")
        # except:
        #     pass

        conv_np_test = (convolved_frames.detach().cpu().numpy()*255).astype(np.uint8)
        
        # im = Image.fromarray(conv_np_test[0,0,:,:])
        # im.save("/home/yl3836/conv_test.jpeg")
        
        #exit(0)

        # event simulation
        if measure_time: t0 = time.time()
        epsilon = 1e-6
        log_frames = torch.log(convolved_frames + epsilon)
        #num_images = (cur_input[i]['frames']).shape[0]

        # try:
        #     print(f"num_images = {num_images}", flush=True)
        #     diffs = log_frames[1:] - log_frames[:num_images-1]
        #     event_frames = torch.stack([compute_event_frame(d) for d in diffs])
        # except Exception as e:
        #     print(f"[DEBUG] num_images = {num_images}", flush=True)
        #     print(f"[DEBUG] diffs shape = {diffs.shape if 'diffs' in locals() else 'not defined'}", flush=True)
        #     raise e

        diffs = log_frames[1:] - log_frames[:-1] 
        #event_frames = torch.stack([compute_event_frame(d) for d in diffs])

        event_frames = compute_event_frame(diffs)  # [T-1, H, W]
        # print ("event_frames size: ", event_frames.size())
        num_images = cur_input[i]['frames'].shape[0]
        
        event_np_test = (event_frames.detach().cpu().numpy()*255).astype(np.uint8)
    
        # plt.plot(event_np_test[0,0,:,:])
        # plt.savefig('/home/yl3836/event_test.png')

        # print(f"event_np_test shape: {event_np_test.shape}") torch.Size([4, 7, 1, 260, 346])
        # im = Image.fromarray(event_np_test[0,0,0])
        #im = Image.fromarray(event_np_test[0].cpu().numpy().squeeze())  
        # im.save("/home/yl3836/event_test.jpeg")
        #exit(0)


        # if measure_time: sim_time += time.time() - t0
        
        
        # voxel grid computation
        if measure_time: t0 = time.time()
        # stamps = stamps
        # stamps = stamps[:,1:]     # Each event frame is computed using diff of some frame_0 and frame_1. We use timestamp of frame_1 in voxel grid computation.
        # stamps = cur_input[i]['stamps'].float()
        
        #print(f"event_frames shape: {event_frames.shape}")
        # voxel_grid = voxel_grid_batched(event_frames.squeeze(2), stamps )
        # print(f"squeezed event_frames shape: {torch.squeeze(event_frames).shape}")
        
       # print(f"[DEBUG] event_frames shape: {event_frames.squeeze(1).shape}")
        voxel_grid = event_frames_to_voxel_grid(event_frames.squeeze(1), cur_input[i]['stamps'], num_bins=5)
        
        # voxel_grid = voxel_grid_batched(event_frames, cur_input[i]['stamps'] )
        # voxel_grid = (voxel_grid - voxel_grid.mean()) / (voxel_grid.std() + epsilon)
        
        if measure_time: voxel_time += time.time() - t0
        #print(f"voxel_grid shape: {voxel_grid.shape}")  
        if downsample:
            # downsampling (done to input events + depths by original model immediately after loading, we do it here since we need to apply psfs first)
            # scale_factor = 0.5      # in the future, maybe try not to hard-code the scale factor
            # downsampled_voxel_grid = f.interpolate(voxel_grid.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=True)
            # downsampled_frame = f.interpolate(cur_input[i]['frame'].unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=True)
            pass
        else:
            downsampled_voxel_grid = voxel_grid 
            frame = cur_input[i]['frame']  
            downsampled_frame = frame.unsqueeze(0)
        #print(f"downsampled_voxel_grid shape: {downsampled_voxel_grid.shape}")

        voxel_grid_list.append(downsampled_voxel_grid)  # shape [1, C, H, W]
        frame_list.append(downsampled_frame)

        #print(f"[DEBUG test] voxel_grid_list size: {voxel_grid_list[0].shape}")
        num_bins, height, width = voxel_grid_list[0].shape
    
        voxel_grids = torch.stack(voxel_grid_list).view(N, num_bins, height, width)
        frame = torch.stack(frame_list).view(N, 1, height, width)

        # breakpoint()
        if measure_time: t0 = time.time()
        new_predicted_frame, states = self.unet_recurrent(voxel_grids, prev_states)
        if measure_time: pred_time = time.time() - t0

        if measure_time:
            return voxel_grids, frame, new_predicted_frame, states, {
                'psf_time': psf_time,
                'sim_time': sim_time,
                'voxel_time': voxel_time,
                'pred_time': pred_time
            }
        else:
            return voxel_grids, frame, new_predicted_frame, states
################################
        ############## Change 0710 ##################
    # def forward(self, cur_input_list, prev_states, downsample=True, measure_time=True):
    #     """
    #     Vectorized forward pass for UNetRecurrentPSF

    #     :param cur_input_list: list of dicts (length = batch size), each with keys:
    #         'frames': [T x 1 x H x W]
    #         'frame': [1 x H x W]
    #         'metric_depth': [1 x H x W]
    #         'stamps': [T]
    #     :param prev_states: previous ConvLSTM states (or None)
    #     :return: voxel_grids, ground_truth, prediction, new_states, timing (optional)
    #     """
    #     import time
    #     N = len(cur_input_list)
    #     T, _, H, W = cur_input_list[0]['frames'].shape  # assume fixed shape across batch

    #     psf_time = sim_time = voxel_time = pred_time = 0

    #     # ------------------ Stack inputs ------------------ #
    #     t0 = time.time() if measure_time else None

    #     frames = torch.stack([x['frames'] for x in cur_input_list])             # [N, T, 1, H, W]
    #     print ("frames size: ", frames.size())
    #     metric_depth = torch.stack([x['metric_depth'] for x in cur_input_list]) # [N, 1, H, W]
    #     print ("metric depth size: ", metric_depth.size())
    #     frame = torch.stack([x['frame'] for x in cur_input_list])               # [N, 1, H, W]
    #     print ("frame size: ", frame.size())
    #     stamps = torch.stack([x['stamps'] for x in cur_input_list])             # [N, T]

    #     frames = frames.to(gpu)
    #     metric_depth = metric_depth.to(gpu)
    #     frame = frame.to(gpu)
    #     stamps = stamps.to(gpu)

    #     # Normalize depth to [2, 80], then round to bins
    #     depth_min, depth_max = 2, 80
    #     depth = (metric_depth - depth_min) / (depth_max - depth_min) * (depth_max - depth_min) + depth_min
    #     depth_bins = torch.round(depth).long().clamp(depth_min, depth_max)

    #     # PSF application (vectorized)
    #     convolved = self.psf_layer(frames, depth_bins)  # [N, T, 1, H, W]

    #     psf_time = time.time() - t0 if measure_time else 0

    #     # ------------------ Simulate Events ------------------ #
    #     t0 = time.time() if measure_time else None

    #     log_frames = torch.log(convolved + 1e-6)          # [N, T, 1, H, W]
    #     # diffs = log_frames[:, 1:] - log_frames[:, :-1]    # [N, T-1, 1, H, W]
    #     # print( "diffs size: ", diffs.size())
    #     # event_frames = torch.stack([compute_event_frame(d) for d in diffs.view(-1, 1, H, W)])
    #     num_images = (cur_input[i]['frames']).shape[0]
    #     diffs = log_frames[1:] - log_frames[:num_images-1]
    #     event_frames = torch.stack([compute_event_frame(d) for d in diffs])

  
    #     event_frames = event_frames.view(N, T-1, 1, H, W) # [N, T-1, 1, H, W]
    #     print ("event frames size: ", event_frames.size())
    #     exit(0)

    #     sim_time = time.time() - t0 if measure_time else 0

    #     # ------------------ Voxel Grid Computation ------------------ #
    #     t0 = time.time() if measure_time else None

    #     voxel_grids = []
    #     for i in range(N):
    #         ts = stamps[i, 1:].float()        # use second-to-last timestamps
    #         ev = event_frames[i].squeeze(1)   # [T-1, H, W]
    #         vg = event_frames_to_voxel_grid(ev, ts)  # [C, H, W]
    #         vg = (vg - vg.mean()) / (vg.std() + 1e-6)
    #         voxel_grids.append(vg)
    #     voxel_grids = torch.stack(voxel_grids)  # [N, C, H, W]

    #     voxel_time = time.time() - t0 if measure_time else 0

    #     # ------------------ Downsample if needed ------------------ #
    #     if downsample:
    #         scale_factor = self.scale_factor if hasattr(self, 'scale_factor') else 0.5
    #         voxel_grids = f.interpolate(voxel_grids, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    #         frame = f.interpolate(frame, scale_factor=scale_factor, mode='bilinear', align_corners=True)

    #     # ------------------ UNet Prediction ------------------ #
    #     t0 = time.time() if measure_time else None

    #     predicted_frame, new_states = self.unet_recurrent(voxel_grids, prev_states)

    #     pred_time = time.time() - t0 if measure_time else 0

    #     if measure_time:
    #         timing = {
    #             'psf_time': psf_time,
    #             'sim_time': sim_time,
    #             'voxel_time': voxel_time,
    #             'pred_time': pred_time
    #         }
    #         return voxel_grids, frame, predicted_frame, new_states, timing
    #     else:
    #         return voxel_grids, frame, predicted_frame, new_states




