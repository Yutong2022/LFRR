# @Time = 2020.02.03
# @Author = Zhen

"""
This script is used to define the networks we will use for LFVDSR external learning and internal learning.
"""


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class LFVDSR(nn.Module):
    def __init__(self, view_num=9, scale=2, layer_num=18, resize_mode="resize"):
        super(LFVDSR, self).__init__()
        self.ch_num = view_num * view_num
        self.scale = scale
        self.layerNum = layer_num
        # modules
        self.relu = nn.ReLU(inplace=True)
        if resize_mode == "interpolate":
            self.imresize = bicubic_interpolation()
        else:
            self.imresize = bicubic_imresize()

        # LFVDSR is VDSR network with multiple inputs and outputs
        self.residual_layer = self.make_layer(Conv_ReLU_Block, self.layerNum)
        self.input_layer = nn.Conv2d(in_channels=self.ch_num, out_channels=64, kernel_size=3,
                                     stride=1, padding=1, bias=True)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=self.ch_num, kernel_size=3,
                                      stride=1, padding=1, bias=True)
        # kaiming initilization
        for m in self._modules:
            weights_init_kaiming(m)

    def forward(self, input_lf):
        # input_lf: [N, C, H, W], C = self.ch_num
        # H and W are the dimensions of the low-resolution SAIs, we need to upsample them first

        ## bicubic upsampling
        H, W = input_lf.shape[2], input_lf.shape[3]
        input_lf = input_lf.view(-1, 1, H, W)

        bicubic_up_lf = self.imresize(input_lf, self.scale)
        bicubic_up_lf = bicubic_up_lf.view(-1, self.ch_num, bicubic_up_lf.shape[2], bicubic_up_lf.shape[3])

        ## reconstruction
        out = self.relu(self.input_layer(bicubic_up_lf))
        out = self.residual_layer(out)
        residual = self.output_layer(out)
        res = bicubic_up_lf + residual
        return res, bicubic_up_lf

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             # if isinstance(param, nn.Parameter):
    #             param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 raise RuntimeError('Cannot copying the parameter named {}'
    #                                    .format(name))


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)

        return x + res
        # here residual block can also use the residual factor


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class bicubic_imresize(nn.Module):

    def __init__(self):
        super(bicubic_imresize, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale, cuda_flag):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)
        if cuda_flag:
            x0 = x0.cuda()
            x1 = x1.cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        if cuda_flag:
            indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
            indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        else:
            indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
            indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        if cuda_flag:
            indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0),
                                torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
            indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1),
                                torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)
        else:
            indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0),
                                torch.FloatTensor([in_size[0]])).unsqueeze(0)
            indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1),
                                torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1 / 4):
        [b, c, h, w] = input.shape
        output_size = [b, c, int(h * scale), int(w * scale)]
        cuda_flag = input.is_cuda

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale, cuda_flag)

        # weight0 = np.asarray(weight0[0], dtype=np.float32)
        # # weight0 = torch.from_numpy(weight0).cuda()
        # weight0 = torch.from_numpy(weight0)
        weight0 = weight0.squeeze(0)

        # indice0 = np.asarray(indice0[0], dtype=np.float32)
        # # indice0 = torch.from_numpy(indice0).cuda().long()
        # indice0 = torch.from_numpy(indice0).long()
        indice0 = indice0.squeeze(0).long()
        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        # weight1 = np.asarray(weight1[0], dtype=np.float32)
        # # weight1 = torch.from_numpy(weight1).cuda()
        # weight1 = torch.from_numpy(weight1)
        weight1 = weight1.squeeze(0)

        # indice1 = np.asarray(indice1[0], dtype=np.float32)
        # # indice1 = torch.from_numpy(indice1).cuda().long()
        # indice1 = torch.from_numpy(indice1).long()
        indice1 = indice1.squeeze(0).long()
        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        # out = torch.round(255 * torch.sum(out, dim=3).permute(0, 1, 3, 2)) / 255
        out = torch.sum(out, dim=3).permute(0, 1, 3, 2)
        return out

class LFVDSRfromBIC(nn.Module):
    def __init__(self, view_num=9):
        super(LFVDSRfromBIC, self).__init__()
        self.ch_num = view_num * view_num
        self.relu = nn.ReLU(inplace=True)

        # LFVDSR is VDSR network with multiple inputs and outputs
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input_layer = nn.Conv2d(in_channels=self.ch_num, out_channels=64, kernel_size=3,
                                     stride=1, padding=1, bias=True)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=self.ch_num, kernel_size=3,
                                      stride=1, padding=1, bias=True)
        # kaiming initilization
        for m in self._modules:
            weights_init_kaiming(m)

    def forward(self, input_lf):
        # input_lf: [N, C, H, W], C = self.ch_num
        # H and W are the dimensions of the low-resolution SAIs after bicubic interpolation

        ## reconstruction
        out = self.relu(self.input_layer(input_lf))
        out = self.residual_layer(out)
        residual = self.output_layer(out)
        res = input_lf + residual
        return res

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             # if isinstance(param, nn.Parameter):
    #             param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 raise RuntimeError('Cannot copying the parameter named {}'
    #                                    .format(name))

class LFVDSRInternal(nn.Module):
    # this module is defined for internal learning
    # after LFVDSR, the result HR light field will be downsampled again for internal training
    def __init__(self, view_num=9, scale=2, layerNum=18, resize_mode="resize"):
        super(LFVDSRInternal, self).__init__()
        self.ch_num = view_num * view_num
        self.scale = scale
        self.layerNum = layerNum
        self.relu = nn.ReLU(inplace=True)
        if resize_mode == "interpolate":
            self.imresize = bicubic_interpolation()
        else:
            self.imresize = bicubic_imresize()
        self.upsample = bicubic_imresize()

        # LFVDSR is VDSR network with multiple inputs and outputs
        self.residual_layer = self.make_layer(Conv_ReLU_Block, self.layerNum)
        self.input_layer = nn.Conv2d(in_channels=self.ch_num, out_channels=64, kernel_size=3,
                                     stride=1, padding=1, bias=True)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=self.ch_num, kernel_size=3,
                                      stride=1, padding=1, bias=True)
        # kaiming initilization
        for m in self._modules:
            weights_init_kaiming(m)

    def forward(self, input_lf):
        # input_lf: [N, C, H, W], C = self.ch_num
        # H and W are the dimensions of the low-resolution SAIs, we need to upsample them first

        ## bicubic upsampling
        H, W = input_lf.shape[2], input_lf.shape[3]
        input_lf = input_lf.view(-1, 1, H, W)

        bicubic_up_lf = self.upsample(input_lf, self.scale)
        bicubic_up_lf = bicubic_up_lf.view(-1, self.ch_num, bicubic_up_lf.shape[2], bicubic_up_lf.shape[3])

        ## reconstruction
        out = self.relu(self.input_layer(bicubic_up_lf))
        out = self.residual_layer(out)
        residual = self.output_layer(out)
        res = bicubic_up_lf + residual

        ## bicbic downsampling
        res_hr = res.view(-1, 1, H*self.scale, W*self.scale)
        downsampled_res = self.imresize(res_hr, 1.0 / self.scale)
        downsampled_res = downsampled_res.view(-1, self.ch_num, downsampled_res.shape[2], downsampled_res.shape[3])
        return downsampled_res, res

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             # if isinstance(param, nn.Parameter):
    #             param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 raise RuntimeError('Cannot copying the parameter named {}'
    #                                    .format(name))

class bicubic_interpolation(nn.Module):
    def __init__(self):
        super(bicubic_interpolation, self).__init__()
    def forward(self, x, scale):
        resized = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)
        # resized = torch.round(resized * 255.0) / 255.0 # transfer to [0,1]
        return resized
# input_lf = torch.rand(1, 81, 48, 48)
#
# net = LFVDSR(view_num=9, scale=2)
#
# input_lf = input_lf.cuda()
# net = net.cuda()
#
# output_lf = net(input_lf)
# writer = SummaryWriter(log_dir='./logs/LFVDSR_external', comment="LFVDSR")
#
# with writer:
#     writer.add_graph(net, (input_lf,))