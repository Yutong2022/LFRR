'''
DMINet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from .swin import *
import einops
from .warpnet import Net
from .disparity_utils import warp_to_ref_view_parallel, warp
from .LFRRN_utils import *
from .vgg import Vgg19
import pytorch_ssim

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        n_blocks, channel, dispchannel = 3, args.channel, 32
        self.factor = args.scale_factor
        self.angRes = args.angRes_in
        self.DispFeaExtract = FeaExtract(dispchannel)
        self.IntraFeaExtract = FeaExtract(channel)
        self.InterFeaExtract = Extract_inter_fea(channel, self.angRes)

        self.CGIGroup = CGIGroup(channel, self.angRes, args.patch_size, args.layers)
        self.CGIGroup_tail = CGIGroup(channel, self.angRes, args.patch_size, args.layers-1, last=True)

        # hilo
        self.pre_conv = FBM(channel*2)
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0, bias=False))

        # warp
        self.disp_net = Net(self.angRes)

    def forward(self, x, info=None):
        
        x_multi = LFsplit(x, self.angRes) #([1, 25, 3, 96, 96])
        
        disp_fea_initial = self.DispFeaExtract(x_multi)
        global_fea_initial = self.IntraFeaExtract(x_multi)
        centra_fea_initial = self.InterFeaExtract(x_multi)

        # calculate disp
        disp_fea_initial = einops.rearrange(disp_fea_initial, 'b (a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        Disparity = self.disp_net(disp_fea_initial)
        Disparity = Disparity.expand(-1, 2, -1, -1)

        # centra-global interaction
        global_fea_1, centra_fea_1 = self.CGIGroup(global_fea_initial, centra_fea_initial, Disparity)
        global_fea_2, _ = self.CGIGroup_tail(global_fea_1, centra_fea_1, Disparity)

        # fusion
        global_out = torch.cat((global_fea_1, global_fea_2), 2)
        global_out = self.pre_conv(global_out) # B an2 c h w
        global_out = einops.rearrange(global_out, 'B an2 c h w -> (B an2) c h w')
        out = self.decoder(global_out)
        out = einops.rearrange(out, '(B a1 a2) c h w -> B c (a1 h) (a2 w)', a1 = 5, a2 = 5)

        return Disparity, out


class FeaExtract(nn.Module):
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = ResASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x_mv):
        b, n, r, h, w = x_mv.shape
        x_mv = x_mv.contiguous().view(b*n, -1, h, w)
        intra_fea_0 = self.FEconv(x_mv)
        intra_fea = self.FERB_1(intra_fea_0)
        intra_fea = self.FERB_2(intra_fea)
        intra_fea = self.FERB_3(intra_fea)
        intra_fea = self.FERB_4(intra_fea)
        _, c, h, w = intra_fea.shape
        intra_fea = intra_fea.unsqueeze(1).contiguous().view(b, -1, c, h, w)#.permute(0,2,1,3,4)  # intra_fea:  B, N, C, H, W

        return intra_fea


class Extract_inter_fea(nn.Module):
    def __init__(self, channel, angRes):
        super(Extract_inter_fea, self).__init__()
        # self.FEconv = nn.Conv2d(angRes*angRes*3, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FEconv = nn.Sequential(
            nn.Conv2d(angRes*angRes*3, channel*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = ResASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x_mv):
        b, n, r, h, w = x_mv.shape
        x_mv = x_mv.contiguous().view(b,-1, h, w)
        inter_fea_0 = self.FEconv(x_mv)
        inter_fea = self.FERB_1(inter_fea_0)
        inter_fea = self.FERB_2(inter_fea)
        inter_fea = self.FERB_3(inter_fea)
        inter_fea = self.FERB_4(inter_fea)
        return inter_fea


class CGIGroup(nn.Module):
    def __init__(self, channel, angRes, patch_size, n_block, last=False):
        super(CGIGroup, self).__init__()
        self.n_block = n_block
        self.angRes = angRes
        self.last = last
        Blocks = []
        for i in range(n_block-1):
            Blocks.append(cg_interaction(channel, angRes, patch_size, iteration_idx=i))
        Blocks.append(cg_interaction(channel, angRes, patch_size, last=last, iteration_idx=n_block-1))
        self.Block = nn.Sequential(*Blocks)
        self.conv_G = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_C = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, global_fea, centra_fea, Disparity):
        for i in range(self.n_block):
            global_fea, centra_fea = self.Block[i](global_fea, centra_fea, Disparity)

        global_fea = einops.rearrange(global_fea, 'b an2 c h w -> (b an2) c h w')
        global_fea = self.conv_G(global_fea)
        global_fea = einops.rearrange(global_fea, '(b an2) c h w -> b an2 c h w', an2=self.angRes*self.angRes)

        if not self.last:            
            centra_fea = self.conv_C(centra_fea)
        else:
            centra_fea = centra_fea      

        return global_fea, centra_fea
    

class cg_interaction(nn.Module):
    def __init__(self, channel, angRes, patch_size, last=False, iteration_idx=0):
        super(cg_interaction, self).__init__()
        self.GTC_blocks = GTC_block(channel, angRes, patch_size, last=last, iteration_idx=iteration_idx)
        self.CTG_blocks = CTG_block(channel, angRes, patch_size, last=last, iteration_idx=iteration_idx)

    def forward(self, global_fea, centra_fea, Disparity):

        update_centra = self.GTC_blocks(global_fea, centra_fea, Disparity)
        update_global = self.CTG_blocks(global_fea, update_centra, Disparity)

        return update_global, update_centra


class CTG_block(nn.Module):
    '''
    Inter-assist-intra feature updating module & intra-assist-inter feature updating module 
    '''
    def __init__(self, channel, angRes, patch_size, last=False, 
                 window_size=[5,5,8,8], input_resolution=[5,5,96,96], 
                 iteration_idx=0, num_heads=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super(CTG_block, self).__init__()
        self.conv_fusing = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_regular = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.angRes = angRes

        dim = channel*2
        self.s_fomer =  Spatial_SwinTransformerBlock(dim=dim, input_resolution=input_resolution[2:4],
                            num_heads=num_heads, window_size=window_size[2],
                            shift_size=0 if (iteration_idx % 2 == 0) else window_size[2] // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[iteration_idx] if isinstance(drop_path, list) else drop_path,
                            norm_layer=norm_layer)

    def forward(self, global_fea, centra_fea, Disparity):
        #intra_fea = intra_fea.permute(0,2,1,3,4)
        _, n, _, h, w = global_fea.shape
        # b, c, h, w = inter_fea.shape
        
        ##update inter-view feature
        warped_centra_fea = back_projection_from_HR_ref_view(centra_fea, refPos=[2,2], disparity=Disparity, angular_resolution=self.angRes, scale=1) #torch.Size([1, 25, 64, 96, 96])
        # warped_global_fea = feature_warp_to_ref_view_parallel(global_fea, Disparity) #torch.Size([1, 25, 64, 96, 96])

        global_fea = torch.cat([global_fea, warped_centra_fea], dim=2)
        global_fea = einops.rearrange(global_fea, 'b an2 c h w -> (b an2) (h w) c')
        global_fea = self.s_fomer(global_fea,[h,w]) 
        global_fea = einops.rearrange(global_fea, '(b an2) (h w) c -> (b an2) c h w', an2=n, h=h) #torch.Size([25, 96, 96, 96])

        global_fea = self.lrelu(self.conv_fusing(global_fea))
        global_fea = einops.rearrange(global_fea, '(b an2) c h w -> b an2 c h w', an2=n)

        return global_fea


class GTC_block(nn.Module):
    '''
    Inter-assist-intra feature updating module & intra-assist-inter feature updating module 
    '''
    def __init__(self, channel, angRes, patch_size, last=False, dim=48, 
                 window_size=[5,5,8,8], input_resolution=[5,5,96,96], 
                 iteration_idx=0, num_heads=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super(GTC_block, self).__init__()
        # self.conv_fusing = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.conv_regular = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.angRes = angRes
        self.last = last

        if not last:
            self.conv_f1 = nn.Conv2d(angRes*angRes*channel, channel, kernel_size=1, stride=1, padding=0)
            self.conv_f2 = nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0)        

        dim = channel
        self.a_fomer =  Angular_SwinTransformerBlock(dim=dim, input_resolution=input_resolution[0:2],
                            num_heads=num_heads, window_size=window_size[0],
                            shift_size=0,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[iteration_idx] if isinstance(drop_path, list) else drop_path,
                            norm_layer=norm_layer)

    def forward(self, global_fea, centra_fea, Disparity):
        #intra_fea = intra_fea.permute(0,2,1,3,4)
        _, n, _, h, w = global_fea.shape
        # b, c, h, w = inter_fea.shape

        # warped_centra_fea = back_projection_from_HR_ref_view(centra_fea, refPos=[2,2], disparity=Disparity, angular_resolution=self.angRes, scale=1) #torch.Size([1, 25, 64, 96, 96])
        global_fea = einops.rearrange(global_fea, 'b an2 c h w -> b c an2 h w')
        warped_global_fea = feature_warp_to_ref_view_parallel(global_fea, Disparity) #torch.Size([1, 48, 25, 96, 96])
        warped_global_fea = einops.rearrange(warped_global_fea, 'b c an2 h w -> b an2 c h w')

        # global_fea = torch.cat([global_fea, warped_centra_fea], dim=2)
        warped_global_fea = einops.rearrange(warped_global_fea, 'b an2 c h w -> (b h w) an2 c')
        warped_global_fea = self.a_fomer(warped_global_fea, [self.angRes, self.angRes]) 
        warped_global_fea = einops.rearrange(warped_global_fea, '(b h w) an2 c -> b (an2 c) h w', w=w, h=h) #torch.Size([25, 96, 96, 96])

        if not self.last:            
            fea_c = self.conv_f1(warped_global_fea)
            out_c = self.conv_f2(torch.cat((fea_c, centra_fea), 1))
        else:
            out_c = centra_fea       

        return out_c

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class disp_loss(nn.Module):
    def __init__(self):
        super(disp_loss, self).__init__()
        self.L1_Loss = torch.nn.L1Loss()
        self.L2_Loss = torch.nn.MSELoss()
        self.TVLoss = TVLoss()

    def loss_disp_smoothness(self, disp, img):
        img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
        weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

        loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
                ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
            (weight_x.sum() + weight_y.sum())
        return loss

    def forward(self, disparity, gt, angular_res):
        refPos = [angular_res // 2, angular_res // 2]
        gt = einops.rearrange(gt, 'b c (u h) (v w) -> b c (u v) h w', u = angular_res, v = angular_res)

        PSV = feature_warp_to_ref_view_parallel(gt, disparity, refPos)
        cnter_view = gt[:, :, refPos[0]*angular_res + refPos[1]]
        loss = 0.
        for view in range(angular_res*angular_res):
            loss += self.L1_Loss(PSV[:, :, view], cnter_view) + self.L2_Loss(PSV[:, :, view], cnter_view) * 0.1

        loss1 = self.TVLoss(disparity) * 0.005
        loss2 = self.loss_disp_smoothness(disparity, cnter_view) * 0.1

        return loss + loss1 + loss2

class grad_loss(nn.Module):
    def __init__(self, args):
        super(grad_loss, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        self.kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        self.kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.angRes = args.angRes_in
        self.criterion_Loss = torch.nn.L1Loss()
        # self.criterion_Loss = torch.nn.MSELoss()

    def forward(self, SRt, HRt):

        self.weight_h = nn.Parameter(data=self.kernel_h, requires_grad=False).to(SRt.device)
        self.weight_v = nn.Parameter(data=self.kernel_v, requires_grad=False).to(SRt.device)

        # yv
        l0 = 0.
        for i in range(3):
            SR = SRt[:, i:i+1]
            HR = HRt[:, i:i+1]
            SR = einops.rearrange(SR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
            HR = einops.rearrange(HR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
            SR_v = F.conv2d(SR, self.weight_v, padding=2)
            HR_v = F.conv2d(HR, self.weight_v, padding=2)
            l1 = self.criterion_Loss(SR_v, HR_v)
            SR_h = F.conv2d(SR, self.weight_h, padding=2)
            HR_h = F.conv2d(HR, self.weight_h, padding=2)
            l2 = self.criterion_Loss(SR_h, HR_h)
            l0 = l0 + l1 + l2 

        return l0

class VGGLoss(nn.Module):
    def __init__(self, device='cuda', vgg=None, weights=None, indices=None, normalize=False):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        self.device = device
        if normalize:
            self.normalize = None
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss

class SSIMLoss(pytorch_ssim.SSIM):
    def forward(self, SR, HR):
        SR = einops.rearrange(SR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=5, a2=5)
        HR = einops.rearrange(HR, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w', a1=5, a2=5)
        return 1. - super().forward(SR, HR)

class saptial_loss(nn.Module):
    def __init__(self, args):
        super(saptial_loss, self).__init__()
        # self.L1_Loss = torch.nn.L1Loss()
        self.L2_Loss = torch.nn.MSELoss()
        self.grad_loss = grad_loss(args)
        self.vgg_loss = VGGLoss()
        self.SSIMLoss = SSIMLoss()

    def forward(self, result, gt, info=None):
        
        loss1 = self.L2_Loss(result, gt)
        loss2 = self.SSIMLoss(result, gt)
        loss3 = self.vgg_loss(result, gt)
        loss4 = self.grad_loss(result, gt)

        loss = loss1 + loss2 + 0.1*loss3 + 0.5*loss4
        return loss

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.saptial_loss = saptial_loss(args)
        self.disp_loss = disp_loss()
        self.angRes = args.angRes_in

    def forward(self, disparity, result, gt, info=None):
        loss1 = self.saptial_loss(result, gt)
        loss2 = self.disp_loss(disparity, gt, self.angRes) * 0.1
        loss = loss1+loss2
        return loss

def weights_init(m):
    pass
