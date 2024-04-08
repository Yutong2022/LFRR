import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import einops
import math
from .disparity_utils import warp_back_projection_no_range, warp


class RB(nn.Module):
    '''
    Residual Block
    '''
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class SELayer(nn.Module):
    '''
    Channel Attention
    '''
    def __init__(self, out_ch,g=12):
        super(SELayer, self).__init__()
        self.att_c = nn.Sequential(
                nn.Conv2d(out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )

    def forward(self,fm):
        ##channel
        fm_pool = F.adaptive_avg_pool2d(fm, (1, 1))
        att = self.att_c(fm_pool)
        fm = fm * att
        return fm

class FBM(nn.Module):
    '''
    Feature Blending 
    '''
    def __init__(self, channel):
        super(FBM, self).__init__()
        self.FERB_1 = RB(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = RB(channel)
        self.FERB_4 = RB(channel)
        self.att1 = SELayer(channel)
        self.att2 = SELayer(channel)
        self.att3 = SELayer(channel)
        self.att4 = SELayer(channel)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer_init = x.contiguous().view(b*n, -1, h, w)
        buffer_1 = self.att1(self.FERB_1(buffer_init))
        buffer_2 = self.att2(self.FERB_2(buffer_1))
        buffer_3 = self.att3(self.FERB_3(buffer_2))
        buffer_4 = self.att4(self.FERB_4(buffer_3))
        buffer = buffer_4.contiguous().view(b, n, -1, h, w)
        return buffer

class MCB(nn.Module):
    '''
    Multi-view Contex Block
    '''
    def __init__(self, channels, angRes):
        super(MCB, self).__init__()
        self.prelu1 = nn.LeakyReLU(0.02, inplace=True)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.ASPP = D3ResASPP(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

    def forward(self, x_init):
        b, c, n, h, w = x_init.shape
        x_init = einops.rearrange(x_init, 'b c n h w -> b n c h w')
        x = self.conv1(x_init)
        buffer = self.prelu1(x)
        buffer = self.ASPP(buffer)
        x = self.conv2(buffer)+x_init
        x = einops.rearrange(x, 'b n c h w -> b c n h w')
        #x = self.prelu2(x)
        return x#.permute(0,2,1,3,4)

def ChannelSplit(input):
    _, C, _, _ = input.shape
    c = C//4
    output_1 = input[:, :c, :, :]
    output_2 = input[:, c:, :, :]
    return output_1, output_2


class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel,channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel*3, channel, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


class D3ResASPP(nn.Module):
    def __init__(self, channel):
        super(D3ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1,1,1), bias=False),
                                              nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1), dilation=(2,1,1), bias=False),
                                              nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(4, 1, 1), dilation=(4,1,1), bias=False),
                                              nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv3d(channel*3, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1,1,1))

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(intra_fea[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.L1_Loss = torch.nn.L1Loss()

    def forward(self, dereflection, transmission):

        loss = self.L1_Loss(dereflection, transmission)

        return loss

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def feature_reshape(x, window_size, patition_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (B, window_size, window_size, num_windows*C)
    """
    B, C, H, W = x.shape

    # patch patition
    x1 = x.view(B, C, H // patition_size, patition_size, W // patition_size, patition_size)
    x1 = einops.rearrange(x1, 'b c p1 ps1 p2 ps2 -> (b p1 p2) c ps1 ps2')

    # feature shuffle
    b, c, h, w = x1.shape
    x1 = F.unfold(x1, kernel_size=(window_size, window_size), padding=0, stride=window_size)
    x1 = F.fold(x1, output_size=(h//window_size, w//window_size), kernel_size=(1, 1), padding=0, stride=1)
    # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
    windows = einops.rearrange(x1, 'B C H W -> B H W C')

    return windows


def feature_reverse(windows, window_size, patition_size, H, W):
    """
    Args:
        windows: (B, (window_size, window_size,) num_windows*C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    windows = einops.rearrange(windows, 'B (w1 w2) C -> B C w1 w2', w1=patition_size//window_size, w2=patition_size//window_size)
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
    x = F.unfold(windows, kernel_size=(1, 1), padding=0, stride=1)
    x = F.fold(x, output_size=(patition_size, patition_size), kernel_size=(window_size, window_size), padding=0, stride=window_size)
    # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
    x = einops.rearrange(x, '(b p1 p2) c ps1 ps2 -> b c (p1 ps1) (p2 ps2)', p1=H//patition_size, p2=W//patition_size)

    return x

def feature_reshape_to_MacPI(x, window_size, patition_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (B, window_size, window_size, num_windows*C)
    """
    B, C, H, W = x.shape
    x1 = F.unfold(x, kernel_size=(window_size, window_size), padding=0, stride=window_size)
    x1 = F.fold(x1, output_size=(H//window_size, W//window_size), kernel_size=(1, 1), padding=0, stride=1)

    x1 = x1.permute(0,2,3,1).contiguous()
    b, h, w, c = x1.shape
    x1 = x1.view(b, h // patition_size, patition_size, w // patition_size, patition_size, c)
    windows = x1.permute(0, 1, 3, 2, 4, 5)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = einops.rearrange(windows, '(B a1 a2) s1 s2 w1 w2 c -> (B s1 s2) (a1 w1) (a2 w2) c', a1 = 5, a2=5)

    return windows


def feature_reverse_to_MacPI(windows, window_size, patition_size, H, W):
    """
    Args:
        windows: (B, (window_size, window_size,) num_windows*C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    windows = einops.rearrange(windows, '(B s1 s2) (a1 w1 a2 w2) c -> (B a1 a2) c (s1 w1) (s2 w2)', a1 = 5, w1=patition_size, a2=5, s1=H//(window_size*patition_size), s2=W//(window_size*patition_size))

    x = F.unfold(windows, kernel_size=(1, 1), padding=0, stride=1)
    x = F.fold(x, output_size=(H, W), kernel_size=(window_size, window_size), padding=0, stride=window_size)
    # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
    return x

def feature_warp_to_ref_view_parallel(input_lf, disparity, refPos = [2,2], padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Unlike warp_to_central_view_lf, we do not use for circle here, we use parallel computation.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, C, UV, H, W = input_lf.shape
    U = int(math.sqrt(float(UV)))
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate
    ## generate angular grid
    # x here denotes the horizontal line
    # so uu here also denotes the horizontal line (column number)
    arange_spatial = torch.arange(0, H)
    uu = torch.arange(0, U).view(1, -1).repeat(U, 1) # u direction, X
    vv = torch.arange(0, U).view(-1, 1).repeat(1, U) # v direction, Y

    uu = uu.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_u
    vv = vv.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_v
    # uu = arange_uu - ref_u
    # vv = arange_vv - ref_v
    deta_uv = torch.cat([uu, vv], dim=2) # [B, U*V, 2, 1, 1]
    if input_lf.is_cuda:
        deta_uv = deta_uv.cuda()
    deta_uv = deta_uv.float()
    ## generate the full disparity maps
    full_disp = disparity.unsqueeze(1) # [B, 1, 2, H, W]
    full_disp = full_disp.repeat(1, UV, 1, 1, 1) # [B, U*V, 2, H, W]
    full_disp = full_disp * deta_uv # [B, U*V, 2, H, W]
    ## warp
    input_lf = einops.rearrange(input_lf, 'B C UV H W -> (B UV) C H W')
    # input_lf = input_lf.view(-1, 1, H, W) # [B*U*V, 1, H, W]
    full_disp = full_disp.view(-1, 2, H, W) # [B*U*V, 2, H, W]
    warped_lf = warp(input_lf, full_disp, arange_spatial, padding_mode=padding_mode) # [B*U*V, C, H, W]
    warped_lf = einops.rearrange(warped_lf, '(B UV) C H W -> B C UV H W', UV = UV)
    return warped_lf

def back_projection_from_HR_ref_view(sr_ref, refPos, disparity, angular_resolution, scale, padding_mode="zeros"):
    # sr_ref: [B, 1, H, W]
    # refPos: [u, v]
    # disparity: [B, 2, h, w]
    # angular_resolution: U
    UV = angular_resolution * angular_resolution
    B = sr_ref.shape[0]
    C = sr_ref.shape[1]
    ref_u = refPos[1]  # horizontal angular coordinate
    ref_v = refPos[0]  # vertical angular coordinate
    ## generate angular grid
    # x here denotes the horizontal line
    # so uu here also denotes the horizontal line (column number)
    uu = torch.arange(0, angular_resolution).view(1, -1).repeat(angular_resolution, 1) # u direction, X
    vv = torch.arange(0, angular_resolution).view(-1, 1).repeat(1, angular_resolution) # v direction, Y
    uu = uu.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_u
    vv = vv.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_v
    # uu = arange_uu - ref_u
    # vv = arange_vv - ref_v
    deta_uv = torch.cat([uu, vv], dim=2)  # [B, U*V, 2, 1, 1]
    if sr_ref.is_cuda:
        deta_uv = deta_uv.cuda()
    deta_uv = deta_uv.float()
    ## generate the full disparity maps
    full_disp = disparity.unsqueeze(1)  # [B, 1, 2, h, w]
    full_disp = full_disp.repeat(1, UV, 1, 1, 1)  # [B, U*V, 2, h, w]
    full_disp = full_disp * deta_uv  # [B, U*V, 2, h, w]

    # repeat sr_ref
    sr_ref = sr_ref.repeat(1, UV, 1, 1, 1) # [B, U*V, 1, H, W]

    # view
    full_disp = full_disp.view(-1, 2, full_disp.shape[3], full_disp.shape[4])
    sr_ref = sr_ref.view(-1, C, sr_ref.shape[3], sr_ref.shape[4])

    # output the back-projected light fields
    bp_lr_lf = warp_back_projection_no_range(sr_ref, full_disp, scale, padding_mode=padding_mode) # [BUV, C, h, w]
    bp_lr_lf = bp_lr_lf.view(-1, UV, C, bp_lr_lf.shape[2], bp_lr_lf.shape[3])
    return bp_lr_lf

def coordinate_transform(x, scale):
    # x can be tensors with any dimensions
    # scale is the scaling factors, when it's less than 1, HR2LR, when it's larger than 1, LR2HR
    y = x / scale - 0.5 * (1 - 1.0 / scale) # for python coordinate system
    return y