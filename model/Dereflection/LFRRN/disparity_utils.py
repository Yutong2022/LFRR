# @Time = 2019.12.10
# @Author = Zhen

"""
    utils.py is used to define useful functions
"""

import math
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
from tools.LFVDSR_config import bicubic_imresize
import os
import h5py
# import torch.nn.functional as F

def get_time_gpu():
    torch.cuda.synchronize()
    return time.time()

def get_time_gpu_str():
    torch.cuda.synchronize()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# def LF_downscale(input_lf, scale, cuda):
#     # input the light field with values in [0,255]
#     # we transfer it to [0,1] and resize
#     # then transfer it back to [0,255]
#     # input_lf: [U,V,X,Y]
#     # scale should be large than 1
#     U, H, W = input_lf.shape[1], input_lf.shape[2], input_lf.shape[3]
#     H = H - H % scale
#     W = W - W % scale
#     h = H // scale
#     w = W // scale
#     resize = bicubic_imresize()
#     input_lf = input_lf[:, :, :H, :W]
#     input_lf = torch.Tensor(input_lf.astype(np.float32) / 255.0).contiguous().view(1, -1, H, W)
#     if cuda:
#         input_lf = input_lf.cuda()
#     resize_lf = resize(input_lf, 1.0/scale)  # [1, U*V, H/s, W/s]
#     resize_lf = resize_lf.view(-1, U, h, w)
#     resize_lf = torch.round(resize_lf * 255.0)
#     if cuda:
#         resize_lf = resize_lf.cpu().data.numpy().astype(np.uint8)
#     else:
#         resize_lf = resize_lf.numpy().astype(np.uint8)
#     return resize_lf
#
def LF_downscale(input_lf, scale):
    # input the light field with values in [0,255]
    # we transfer it to [0,1] and resize
    # then transfer it back to [0,255]
    # input_lf: [U,V,X,Y]
    # scale should be large than 1
    U, H, W = input_lf.shape[1], input_lf.shape[2], input_lf.shape[3]
    H = H - H % scale
    W = W - W % scale
    h = H // scale
    w = W // scale
    resize = bicubic_imresize()
    input_lf = input_lf[:, :, :H, :W]
    input_lf = torch.Tensor(input_lf.astype(np.float32) / 255.0).contiguous().view(1, -1, H, W)
    resize_lf = resize(input_lf, 1.0/scale)  # [1, U*V, H/s, W/s]
    resize_lf = resize_lf.view(-1, U, h, w)
    resize_lf = torch.round(resize_lf * 255.0)
    resize_lf = torch.clamp(resize_lf, 0.0, 255.0)
    resize_lf = resize_lf.numpy().astype(np.uint8)
    return resize_lf

def LF_downscale_RGB(input_lf, scale):
    # input the light field with values in [0,255]
    # we transfer it to [0,1] and resize
    # then transfer it back to [0,255]
    # input_lf: [C,U,V,X,Y]
    # scale should be large than 1
    C, U, H, W = input_lf.shape[0], input_lf.shape[1], input_lf.shape[3], input_lf.shape[4]
    H = H - H % scale
    W = W - W % scale
    h = H // scale
    w = W // scale
    resize = bicubic_imresize()
    # resize = bicubic_interpolation()
    input_lf = input_lf[:, :, :, :H, :W]
    input_lf = torch.Tensor(input_lf.astype(np.float32) / 255.0).contiguous().view(1, -1, H, W)
    resize_lf = resize(input_lf, 1.0/scale)  # [1, C*U*V, H/s, W/s]
    resize_lf = resize_lf.view(C, U, -1, h, w)
    resize_lf = torch.round(resize_lf * 255.0)
    resize_lf = torch.clamp(resize_lf, 0.0, 255.0)
    resize_lf = resize_lf.numpy().astype(np.uint8)
    return resize_lf

def back_projection_refinement(lr_img, sr_res, scale):
    # back projection refinement for further improvement of SR results.
    # For once iteration
    # lr_img and sr_res should be Tensors within [0,1]
    # lr_img: [1, 1, h, w]
    # sr_res: [1, 1, H, W]
    resizer = bicubic_imresize()
    refined_res = sr_res + resizer(lr_img - resizer(sr_res, 1.0/scale), scale)
    return refined_res

def PSNR(pred, gt, shave_border=0):
# define PSNR function, the input images should be inside the interval of [0,255]
    pred = pred.astype(float)
    gt = gt.astype(float)
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def transfer_img_to_uint8(img):
    # the input image is within (0,1) interval
    img = img * 255.0
    img = np.clip(img, 0.0, 255.0)
    img = np.uint8(np.around(img))
    return img

def colorize(y, ycbcr):
# colorize a grayscale image
# ycbcr means the upscaled YCbCr using Bicubic interpolation
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def modcrop(imgs, scale):
# modcrop the input image, the input image is a matrix with 1 or 3 channels
    if len(imgs.shape) == 2:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row, :cropped_col]
    elif len(imgs.shape) == 3:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row, :cropped_col, :]
    else:
        raise IOError('Img Channel > 3.')

    return cropped_img

def lf_modcrop(lf, scale):
    # modcrop the input light field, the light field shoud be as format as [U,V,X,Y]
    [U, V, X, Y] = lf.shape
    x = X - (X % scale)
    y = Y - (Y % scale)
    output = np.zeros([U, V, x, y])
    for u in range(0, U):
        for v in range(0, V):
            sub_img = lf[u,v]
            output[u,v] = modcrop(sub_img, scale)
    return output

def single_image_downscale(hr_img, scale):
    # input the HR image with values in [0, 255]
    # we transfer it to [0,1] and resize it
    # then transfer it back to [0, 255]
    # hr_img: [H, W]
    # scale: scale should be large than 1
    hr_img = modcrop(hr_img, scale)
    resize = bicubic_imresize()
    hr_img = torch.Tensor(hr_img.astype(np.float32) / 255.0).contiguous().unsqueeze(0).unsqueeze(0)
    lr_img = resize(hr_img, 1.0/scale)
    lr_img = torch.clamp(torch.round(lr_img * 255.0), 0.0, 255.0)
    lr_img = lr_img.squeeze().numpy().astype(np.uint8) # [h, w]
    return lr_img

def single_image_upscale(lr_img, scale):
    # input the LR image with values in [0, 255]
    # we transfer it to [0,1] and resize it
    # then transfer it back to [0, 255]
    # lr_img: [h, w]
    # scale: scale should be large than 1
    resize = bicubic_imresize()
    lr_img = torch.Tensor(lr_img.astype(np.float32) / 255.0).contiguous().unsqueeze(0).unsqueeze(0)
    hr_img = resize(lr_img, scale)
    hr_img = torch.clamp(torch.round(hr_img * 255.0), 0.0, 255.0)
    hr_img = hr_img.squeeze().numpy().astype(np.uint8) # [h, w]
    return hr_img



def img_rgb2ycbcr(img):
    # the input image data format should be uint8
    if not len(img.shape) == 3:
        raise IOError('Img channle is not 3')
    if not img.dtype == 'uint8':
        raise IOError('Img should be uint8')
    img = img/255.0
    img_ycbcr = np.zeros(img.shape, 'double')
    img_ycbcr[:, :, 0] = 65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2] + 16
    img_ycbcr[:, :, 1] = -37.797 * img[:, :, 0] - 74.203 * img[:, :, 1] + 112 * img[:, :, 2] + 128
    img_ycbcr[:, :, 2] = 112 * img[:, :, 0] - 93.786 * img[:, :, 1] - 18.214 * img[:, :, 2] + 128
    img_ycbcr = np.round(img_ycbcr)
    img_ycbcr = np.clip(img_ycbcr,0,255)
    img_ycbcr = np.uint8(img_ycbcr)
    return img_ycbcr

def img_ycbcr2rgb(im):
    # the input image data format should be uint8
    if not len(im.shape) == 3:
        raise IOError('Img channle is not 3')
    if not im.dtype == 'uint8':
        raise IOError('Img should be uint8')
    im_YCrCb = np.zeros(im.shape, 'double')
    im_YCrCb = im * 1.0
    tmp = np.zeros(im.shape, 'double')
    tmp[:, :, 0] = im_YCrCb[:, :, 0] - 16.0
    tmp[:, :, 1] = im_YCrCb[:, :, 1] - 128.0
    tmp[:, :, 2] = im_YCrCb[:, :, 2] - 128.0
    im_my = np.zeros(im.shape, 'double')
    im_my[:, :, 0] = 0.00456621 * tmp[:, :, 0] + 0.00625893 * tmp[:, :, 2]
    im_my[:, :, 1] = 0.00456621 * tmp[:, :, 0] - 0.00153632 * tmp[:, :, 1] - 0.00318811 * tmp[:, :, 2]
    im_my[:, :, 2] = 0.00456621 * tmp[:, :, 0] + 0.00791071 * tmp[:, :, 1]
    im_my = im_my * 255
    im_my = np.round(im_my)
    im_my = np.clip(im_my, 0, 255)
    im_my = np.uint8(im_my)
    return im_my

def warp(x, flo, arange_spatial, padding_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    x and flo should be inside the same device (CPU or GPU)

    """
    B, C, H, W = x.size()
    # mesh grid
    # print("Before mesh grid {}".format(get_time_gpu()))
    # xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    # yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

    # for speed
    xx = arange_spatial.view(1, -1).repeat(H, 1)
    yy = arange_spatial.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    # print("Before grid + flo {}".format(get_time_gpu()))
    vgrid = grid + flo

    # scale grid to [-1,1]
    # print("Before grid rescaling {}".format(get_time_gpu()))
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)
    # print("Before grid permutation {}".format(get_time_gpu()))
    vgrid = vgrid.permute(0, 2, 3, 1).contiguous()
    # print("Before grid sample {}".format(get_time_gpu()))
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)
    # print("After grid sample {}".format(get_time_gpu()))

    return output

def warp_to_ref_view_parallel(input_lf, disparity, refPos,
                              arange_angular, arange_spatial, padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Unlike warp_to_central_view_lf, we do not use for circle here, we use parallel computation.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(float(UV)))
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate
    ## generate angular grid
    # x here denotes the horizontal line
    # so uu here also denotes the horizontal line (column number)
    # uu = torch.arange(0, U).view(1, -1).repeat(U, 1) # u direction, X
    # vv = torch.arange(0, U).view(-1, 1).repeat(1, U) # v direction, Y

    # for speed
    uu = arange_angular.view(1, -1).repeat(U, 1) # u direction, X
    vv = arange_angular.view(-1, 1).repeat(1, U) # v direction, Y

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
    input_lf = input_lf.view(-1, 1, H, W) # [B*U*V, 1, H, W]
    full_disp = full_disp.view(-1, 2, H, W) # [B*U*V, 2, H, W]
    warped_lf = warp(input_lf, full_disp, arange_spatial, padding_mode=padding_mode) # [B*U*V, 1, H, W]
    warped_lf = warped_lf.view(B, -1, H, W) # [B, U*V, H, W]
    return warped_lf

def warp_no_range(x, flo, padding_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    x and flo should be inside the same device (CPU or GPU)

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)

    return output

def warp_to_ref_view_parallel_no_range(input_lf, disparity, refPos, padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Unlike warp_to_central_view_lf, we do not use for circle here, we use parallel computation.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(float(UV)))
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate
    ## generate angular grid
    uu = torch.arange(0, U).view(1, -1).repeat(U, 1) # u direction, X
    vv = torch.arange(0, U).view(-1, 1).repeat(1, U) # v direction, Y
    uu = uu.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_u
    vv = vv.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1) - ref_v
    deta_uv = torch.cat([uu, vv], dim=2) # [B, U*V, 2, 1, 1]
    if input_lf.is_cuda:
        deta_uv = deta_uv.cuda()
    deta_uv = deta_uv.float()
    ## generate the full disparity maps
    full_disp = disparity.unsqueeze(1) # [B, 1, 2, H, W]
    full_disp = full_disp.repeat(1, UV, 1, 1, 1) # [B, U*V, 2, H, W]
    full_disp = full_disp * deta_uv # [B, U*V, 2, H, W]
    ## warp
    input_lf = input_lf.view(-1, 1, H, W) # [B*U*V, 1, H, W]
    full_disp = full_disp.view(-1, 2, H, W) # [B*U*V, 2, H, W]
    warped_lf = warp_no_range(input_lf, full_disp, padding_mode=padding_mode) # [B*U*V, 1, H, W]
    warped_lf = warped_lf.view(B, -1, H, W) # [B, U*V, H, W]
    return warped_lf

def warp_double_range(x, flo, arange_spatial_x, arange_spatial_y, padding_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    x and flo should be inside the same device (CPU or GPU)

    """
    B, C, H, W = x.size()
    # mesh grid
    # print("Before mesh grid {}".format(get_time_gpu()))
    # xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    # yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

    # for speed
    xx = arange_spatial_x.view(1, -1).repeat(H, 1)
    yy = arange_spatial_y.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    # print("Before grid + flo {}".format(get_time_gpu()))
    vgrid = grid + flo

    # scale grid to [-1,1]
    # print("Before grid rescaling {}".format(get_time_gpu()))
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)
    # print("Before grid permutation {}".format(get_time_gpu()))
    vgrid = vgrid.permute(0, 2, 3, 1).contiguous()
    # print("Before grid sample {}".format(get_time_gpu()))
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)
    # print("After grid sample {}".format(get_time_gpu()))

    return output

def warp_to_ref_view_parallel_double_range(input_lf, disparity, refPos,
                                           arange_angular, arange_spatial_x,
                                           arange_spatial_y, padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Unlike warp_to_central_view_lf, we do not use for circle here, we use parallel computation.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(float(UV)))
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate
    ## generate angular grid
    # x here denotes the horizontal line
    # so uu here also denotes the horizontal line (column number)
    # uu = torch.arange(0, U).view(1, -1).repeat(U, 1) # u direction, X
    # vv = torch.arange(0, U).view(-1, 1).repeat(1, U) # v direction, Y

    # for speed
    uu = arange_angular.view(1, -1).repeat(U, 1) # u direction, X
    vv = arange_angular.view(-1, 1).repeat(1, U) # v direction, Y

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
    input_lf = input_lf.view(-1, 1, H, W) # [B*U*V, 1, H, W]
    full_disp = full_disp.view(-1, 2, H, W) # [B*U*V, 2, H, W]
    warped_lf = warp_double_range(input_lf, full_disp, arange_spatial_x,
                                  arange_spatial_y, padding_mode=padding_mode) # [B*U*V, 1, H, W]
    warped_lf = warped_lf.view(B, -1, H, W) # [B, U*V, H, W]
    return warped_lf

def warp_to_ref_view_serial_no_range(input_lf, disparity, refPos, padding_mode="zeros"):
    """
        This is the function used for warping a light field to the reference view.
        :param input_lf: [B, U*V, H, W]
        :param disparity: [B, 2, H, W]
        :param refPos: u and v coordinates of the reference view point
        :param padding_mode: mode for padding
        :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(UV))
    V = U
    ref_u = refPos[0]
    ref_v = refPos[1]

    input_lf = input_lf.view(-1, U, V, H, W)
    # using clone(), gradients propagating to the cloned tensor will propagate to the original tensor.
    warped_ref_view = []

    for u in range(U):
        for v in range(V):
            # disparity_copy = disparity.clone()
            disparity_x = disparity[:, 0, :, :].clone().unsqueeze(1)
            disparity_y = disparity[:, 1, :, :].clone().unsqueeze(1)
            deta_u = v - ref_v
            deta_v = u - ref_u

            # disparity_copy[:, 0, :, :] = deta_u * disparity_copy[:, 0, :, :]
            disparity_x = deta_u * disparity_x
            # disparity_copy[:, 1, :, :] = deta_v * disparity_copy[:, 1, :, :]
            disparity_y = deta_v * disparity_y
            disparity_copy = torch.cat([disparity_x, disparity_y], dim=1)

            sub_img = input_lf[:, u, v, :, :].clone().unsqueeze(1)

            warped_img = warp_no_range(sub_img, disparity_copy, padding_mode=padding_mode)
            warped_ref_view.append(warped_img)
    warped_ref_view = torch.cat(warped_ref_view, dim=1)
    return warped_ref_view

## Here we define some metrics for disparity estimation
def rmse_error(img_gt, img_reg):
    return np.sqrt(np.mean((img_gt - img_reg) * (img_gt - img_reg)))

def mae_error(img_gt, img_reg):
    return np.mean(np.absolute(img_gt - img_reg))

# def store2hdf5_lf_pairs_uint8(filename, data, labels, data_chunksz, label_chunksz, chunksz, create, startloc):
def store2hdf5_lf_pairs_uint8(filename, data, labels, data_chunksz, label_chunksz, chunksz):
    """
    store light field pairs with uint8 format into hdf5 files.
    :param filename: the filename of hdf5 file.
    :param data: [N U V h w], light field images should be within the interval of [0,255].
    :param labels: [N U V H W], the format and value range are the same as data.
    :param data_chunksz: patch chunksz.
    :param label_chunksz: patch chunksz.
    :param chunksz: batch size, only used in create mode.
    :param create: if create=True(create mode), startloc.data=[1,1,1,1,1], startloc.label=[1,1,1,1,1];
                   if create=False(append mode), startloc.data=[K+1,1,1,1,1], startloc.label=[K+1,1,1,1,1], K is the
                   current number of samples store in the hdf5 file.
    :param startloc: used for append mode.
    create and startloc are not involved here.
    :return: curr_dat_sz, curr_lab_sz, current sizes of data and label.
    """
    # verify the format
    data_dims = data.shape
    label_dims = labels.shape
    num_samples = data_dims[0]
    if not (num_samples==label_dims[0]):
        raise Exception("Number of samples should be matched between data and label")

    if os.path.exists(filename):
        print("Warning: Replacing the existing file {}".format(filename))
        os.remove(filename)
    f = h5py.File(filename, 'w')
    f.create_dataset(name="data", dtype=np.uint8, data=data,
                     maxshape=(None, data_dims[1], data_dims[2], data_dims[3], data_dims[4]),
                     chunks=(chunksz, data_chunksz[0], data_chunksz[1], data_chunksz[2], data_chunksz[3]),
                     fillvalue=0)
    f.create_dataset(name="label", dtype=np.uint8, data=labels,
                     maxshape=(None, label_dims[1], label_dims[2], label_dims[3], label_dims[4]),
                     chunks=(chunksz, label_chunksz[0], label_chunksz[1], label_chunksz[2], label_chunksz[3]),
                     fillvalue=0)
    # if create:
    #     # create mode
    #     if os.path.exists(filename):
    #         print("Warning: Replacing the existing file {}".format(filename))
    #         os.remove(filename)
    #     f = h5py.File(filename, 'w')
    #     f.create_dataset(name="data", dtype=np.uint8, data=data,
    #                      maxshape=data_dims[:-1] + [None],
    #                      chunks=data_chunksz + [chunksz])
    #     f.create_dataset(name="label", dtype=np.uint8, data=labels,
    #                      maxshape=label_dims[:-1] + [None],
    #                      chunks=label_chunksz + [chunksz])
    # else:
    #     # we do not find the add or append function now
    #     pass

    curr_dat_sz = f["data"].shape
    curr_lab_sz = f["label"].shape
    f.close()
    return curr_dat_sz, curr_lab_sz

def store2hdf5_lf_pairs_float32(filename, data, labels, data_chunksz, label_chunksz, chunksz):
    """
    store light field pairs with float32 format into hdf5 files.
    :param filename: the filename of hdf5 file.
    :param data: [N U V h w], light field patches should be within the interval of [0, 1].
    :param labels: [N U V H W], the format and value range are the same as data.
    :param data_chunksz: patch chunksz.
    :param label_chunksz: patch chunksz.
    :param chunksz: batch size, only used in create mode.
    :param create: if create=True(create mode), startloc.data=[1,1,1,1,1], startloc.label=[1,1,1,1,1];
                   if create=False(append mode), startloc.data=[K+1,1,1,1,1], startloc.label=[K+1,1,1,1,1], K is the
                   current number of samples store in the hdf5 file.
    :param startloc: used for append mode.
    create and startloc are not involved here.
    :return: curr_dat_sz, curr_lab_sz, current sizes of data and label.
    """
    # verify the format
    data_dims = data.shape
    label_dims = labels.shape
    num_samples = data_dims[0]
    if not (num_samples==label_dims[0]):
        raise Exception("Number of samples should be matched between data and label")

    if os.path.exists(filename):
        print("Warning: Replacing the existing file {}".format(filename))
        os.remove(filename)
    f = h5py.File(filename, 'w')
    f.create_dataset(name="data", dtype=np.float32, data=data,
                     maxshape=(None, data_dims[1], data_dims[2], data_dims[3], data_dims[4]),
                     chunks=(chunksz, data_chunksz[0], data_chunksz[1], data_chunksz[2], data_chunksz[3]),
                     fillvalue=0)
    f.create_dataset(name="label", dtype=np.float32, data=labels,
                     maxshape=(None, label_dims[1], label_dims[2], label_dims[3], label_dims[4]),
                     chunks=(chunksz, label_chunksz[0], label_chunksz[1], label_chunksz[2], label_chunksz[3]),
                     fillvalue=0)
    # if create:
    #     # create mode
    #     if os.path.exists(filename):
    #         print("Warning: Replacing the existing file {}".format(filename))
    #         os.remove(filename)
    #     f = h5py.File(filename, 'w')
    #     f.create_dataset(name="data", dtype=np.uint8, data=data,
    #                      maxshape=data_dims[:-1] + [None],
    #                      chunks=data_chunksz + [chunksz])
    #     f.create_dataset(name="label", dtype=np.uint8, data=labels,
    #                      maxshape=label_dims[:-1] + [None],
    #                      chunks=label_chunksz + [chunksz])
    # else:
    #     # we do not find the add or append function now
    #     pass

    curr_dat_sz = f["data"].shape
    curr_lab_sz = f["label"].shape
    f.close()
    return curr_dat_sz, curr_lab_sz


# def store2hdf5_lf_pairs_uint8(filename, data, labels, data_chunksz, label_chunksz, chunksz, create, startloc):
def store2hdf5(filename, data, labels, chunksz):
    """
    store light field pairs with uint8 format into hdf5 files.
    :param filename: the filename of hdf5 file.
    :param data: [N C h w], data with interval of [0,1], np.float32.
    :param labels: [N C H W], the format and value range are the same as data.
    :param chunksz: batch size, only used in create mode.
    :return: curr_dat_sz, curr_lab_sz, current sizes of data and label.
    """
    # verify the format
    data_dims = data.shape
    label_dims = labels.shape
    num_samples = data_dims[0]
    if not (num_samples==label_dims[0]):
        raise Exception("Number of samples should be matched between data and label")

    if os.path.exists(filename):
        print("Warning: Replacing the existing file {}".format(filename))
        os.remove(filename)
    f = h5py.File(filename, 'w')
    f.create_dataset(name="data", dtype=np.float32, data=data,
                     maxshape=(None, data_dims[1], data_dims[2], data_dims[3]),
                     chunks=(chunksz, data_dims[1], data_dims[2], data_dims[3]),
                     fillvalue=0)
    f.create_dataset(name="label", dtype=np.float32, data=labels,
                     maxshape=(None, label_dims[1], label_dims[2], label_dims[3]),
                     chunks=(chunksz, label_dims[1], label_dims[2], label_dims[3]),
                     fillvalue=0)
    # if create:
    #     # create mode
    #     if os.path.exists(filename):
    #         print("Warning: Replacing the existing file {}".format(filename))
    #         os.remove(filename)
    #     f = h5py.File(filename, 'w')
    #     f.create_dataset(name="data", dtype=np.uint8, data=data,
    #                      maxshape=data_dims[:-1] + [None],
    #                      chunks=data_chunksz + [chunksz])
    #     f.create_dataset(name="label", dtype=np.uint8, data=labels,
    #                      maxshape=label_dims[:-1] + [None],
    #                      chunks=label_chunksz + [chunksz])
    # else:
    #     # we do not find the add or append function now
    #     pass

    curr_dat_sz = f["data"].shape
    curr_lab_sz = f["label"].shape
    f.close()
    return curr_dat_sz, curr_lab_sz


def warp_to_central_view_lf(input_lf, disparity, padding_mode="zeros"):
    """
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W], disparity map of central view
    :param padding_mode: mode for padding, "zeros", "reflection" or "border"
    :return: return the warped central view images
    """
    B, UV, H, W = input_lf.size()
    U = int(math.sqrt(float(UV)))
    V = U
    mid_u = U // 2
    mid_v = V // 2

    input_lf = input_lf.view(-1, U, V, H, W)
    # using clone(), gradients propagating to the cloned tensor will propagate to the original tensor.
    warped_central_view = []

    for u in range(U):
        for v in range(V):
            # disparity_copy = disparity.clone()
            disparity_x = disparity[:, 0, :, :].clone().unsqueeze(1)
            disparity_y = disparity[:, 1, :, :].clone().unsqueeze(1)
            deta_u = v - mid_v
            deta_v = u - mid_u

            # disparity_copy[:, 0, :, :] = deta_u * disparity_copy[:, 0, :, :]
            disparity_x = deta_u * disparity_x
            # disparity_copy[:, 1, :, :] = deta_v * disparity_copy[:, 1, :, :]
            disparity_y = deta_v * disparity_y
            disparity_copy = torch.cat([disparity_x, disparity_y], dim=1)

            sub_img = input_lf[:, u, v, :, :].clone().unsqueeze(1)

            warped_img = warp_no_range(sub_img, disparity_copy, padding_mode=padding_mode)
            warped_central_view.append(warped_img)
    warped_central_view = torch.cat(warped_central_view, dim=1)
    return warped_central_view



#########################
## These functions are used for back-projection
def back_projection_from_HR_ref_view(sr_ref, refPos, disparity, angular_resolution, scale, padding_mode="zeros"):
    # sr_ref: [B, 1, H, W]
    # refPos: [u, v]
    # disparity: [B, 2, h, w]
    # angular_resolution: U
    UV = angular_resolution * angular_resolution
    B = sr_ref.shape[0]
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
    sr_ref = sr_ref.view(-1, 1, sr_ref.shape[3], sr_ref.shape[4])

    # output the back-projected light fields
    bp_lr_lf = warp_back_projection_no_range(sr_ref, full_disp, scale, padding_mode=padding_mode) # [BUV, 1, h, w]
    bp_lr_lf = bp_lr_lf.view(-1, UV, bp_lr_lf.shape[2], bp_lr_lf.shape[3])
    return bp_lr_lf


def warp_back_projection_no_range(x, flo, scale, padding_mode="zeros"):
    """
    sample the points from HR images with LR flow for back-projection.

    x: [B, C, H, W] HR image
    flo: [B, 2, h, w] LR_flow
    x and flo should be inside the same device (CPU or GPU)

    """
    # B, C, H, W = x.shape
    B, _, H, W = flo.shape
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid - flo

    # make coordinate transformation from LR to HR
    vgrid = coordinate_transform(vgrid, 1.0/scale)

    # scale grid to [-1,1]
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W * scale - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H * scale - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)

    return output


def warp_back_projection_double_range(x, flo, scale, range_x, range_y, padding_mode="zeros"):
    """
        sample the points from HR images with LR flow for back-projection.

        x: [B, C, H, W] HR image
        flo: [B, 2, h, w] LR_flow
        scale: H / h
        range_x: torch.range(w)
        range_y: torch.range(h)
        x and flo should be inside the same device (CPU or GPU)

        """
    # B, C, H, W = x.shape
    B, _, H, W = flo.shape
    # mesh grid
    # xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    # yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = range_x.view(1, -1).repeat(H, 1)
    yy = range_y.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid - flo

    # make coordinate transformation from LR to HR
    vgrid = coordinate_transform(vgrid, 1.0 / scale)

    # scale grid to [-1,1]
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W * scale - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H * scale - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)

    return output


def back_projection_from_HR_ref_view_double_range(sr_ref, refPos, disparity, angular_resolution,
                                                  scale, range_angular, range_x, range_y, padding_mode="zeros"):
    # sr_ref: [B, 1, H, W]
    # refPos: [u, v]
    # disparity: [B, 2, h, w]
    # angular_resolution: U
    UV = angular_resolution * angular_resolution
    B = sr_ref.shape[0]
    ref_u = refPos[1]  # horizontal angular coordinate
    ref_v = refPos[0]  # vertical angular coordinate
    ## generate angular grid
    # x here denotes the horizontal line
    # so uu here also denotes the horizontal line (column number)
    # uu = torch.arange(0, angular_resolution).view(1, -1).repeat(angular_resolution, 1)  # u direction, X
    # vv = torch.arange(0, angular_resolution).view(-1, 1).repeat(1, angular_resolution)  # v direction, Y
    uu = range_angular.view(1, -1).repeat(angular_resolution, 1)  # u direction, X
    vv = range_angular.view(-1, 1).repeat(1, angular_resolution)  # v direction, Y
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
    sr_ref = sr_ref.repeat(1, UV, 1, 1, 1)  # [B, U*V, 1, H, W]

    # view
    full_disp = full_disp.view(-1, 2, full_disp.shape[3], full_disp.shape[4])
    sr_ref = sr_ref.view(-1, 1, sr_ref.shape[3], sr_ref.shape[4])

    # output the back-projected light fields
    bp_lr_lf = warp_back_projection_double_range(sr_ref, full_disp, scale,
                                                 range_x, range_y,
                                                 padding_mode=padding_mode)
    bp_lr_lf = bp_lr_lf.view(-1, UV, bp_lr_lf.shape[2], bp_lr_lf.shape[3])
    return bp_lr_lf



def coordinate_transform(x, scale):
    # x can be tensors with any dimensions
    # scale is the scaling factors, when it's less than 1, HR2LR, when it's larger than 1, LR2HR
    y = x / scale - 0.5 * (1 - 1.0 / scale) # for python coordinate system
    return y
