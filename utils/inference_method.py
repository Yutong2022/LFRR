import importlib
import torch
import torch.backends.cudnn as cudnn
from .utils import *
import torch.nn as nn
import imageio
import cv2
import einops
from scipy.io import savemat
import time
from ..model.Dereflection.LFRRN.LFRRN_utils import *

def test_m1(test_loader, device, net, save_dir=None, logger=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    # for idx_iter, (Data, label, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
    for idx_iter, (Data, label, data_info, LF_name) in enumerate(test_loader):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()
        patch_size = 160

        ''' Without Crop LFs into Patches '''
        Data = Data.to(device) #624,432

        lf_input = einops.rearrange(Data, 'b c (u h) (v w) -> (b c) (u v) h w', u=5, v=5)
        _, _, H, W = lf_input.shape # 3 25 432 624
        lf_output = torch.zeros_like(lf_input).to(device)
        disparity = torch.zeros([1, 2, H, W]).to(device)

        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch = lf_input[:, :, i:i+patch_size, j:j+patch_size]
                pad_h = patch_size - patch.size(2)
                pad_w = patch_size - patch.size(3)
                if pad_h > 0 or pad_w > 0:
                    patch = nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')

                subinput = einops.rearrange(patch, '(b c) (u v) h w -> b c (u h) (v w)', c=3, u=5, v=5)
                # breakpoint()
                with torch.no_grad():
                    net.eval()
                    torch.cuda.empty_cache()
                    subdisp, subout = net(subinput, data_info)
                
                subout = einops.rearrange(subout, 'b c (u h) (v w) -> (b c) (u v) h w', u=5, v=5)
                subout = subout[:, :, :patch_size-pad_h, :patch_size-pad_w]
                lf_output[:, :, i:i+patch_size-pad_h, j:j+patch_size-pad_w] = subout
        
                subdisp = subdisp[:, :, :patch_size-pad_h, :patch_size-pad_w]
                disparity[:, :, i:i+patch_size-pad_h, j:j+patch_size-pad_w] = subdisp

        Data_record = einops.rearrange(lf_output, '(b c) (u v) h w -> (b c) (u h) (v w)', c=3, u=5, v=5)

        ''' Restore the Patches to LFs '''
        label = einops.rearrange(label, 'b c (u h) (v w) -> (b c) (u h) (v w)', u=5, v=5)
        Removal_img = Data_record
        
        PSNR, SSIM = cal_metrics_all(Removal_img.data.cpu(), label.data.cpu(), 5)

        if logger:
            logger.log_string(' Restore the LF {} to PSNR {}/SSIM {}'.format(LF_name, PSNR, SSIM))
        else:
            print(' Restore the LF {} to PSNR {}'.format(LF_name, PSNR))
            
        psnr_iter_test.append(PSNR)
        ssim_iter_test.append(SSIM)

    return psnr_iter_test, ssim_iter_test, LF_name

def test_m2(test_loader, device, net, save_dir=None, logger=None):
    # if the GPU memory is large enough, you can select the entire image input and achieve a better performance
    psnr_iter_test = None
    ssim_iter_test = None
    LF_name = None

    return psnr_iter_test, ssim_iter_test, LF_name

def test_m3(test_loader, device, net, save_dir=None, logger=None):
    # the test method from basiclfsr
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Data, label, data_info, LF_name) in enumerate(test_loader):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Data = Data.squeeze().to(device)

        ''' Crop LFs into Patches '''
        subLFin = []
        for i in range(3):
            subData = Data[i]
            subLFin_i = LFdivide(subData, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
            numU, numV, H, W = subLFin_i.size()
            subLFin_i = rearrange(subLFin_i, 'n1 n2 a1h a2w -> (n1 n2) a1h a2w')
            subLFin.append(subLFin_i)
        subLFin = torch.stack(subLFin, dim=1)
        subLFout = torch.zeros(numU * numV, 3, args.patch_size_for_test*args.angRes_in,
                               args.patch_size_for_test*args.angRes_in)
        subdispout = torch.zeros(numU * numV, 2, args.patch_size_for_test, args.patch_size_for_test)

        ''' SR the Patches '''
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            with torch.no_grad():
                net.eval()
                torch.cuda.empty_cache()
                disparity, out = net(tmp.to(device), data_info)
                subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
                subdispout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = disparity
        subLFout = rearrange(subLFout, '(n1 n2) c a1h a2w -> n1 n2 c a1h a2w', n1=numU, n2=numV)
        subdispout = rearrange(subdispout, '(n1 n2) c a1h a2w -> n1 n2 c a1h a2w', n1=numU, n2=numV)

        ''' Restore the Patches to LFs '''
        label = einops.rearrange(label, 'b c (u h) (v w) -> (b c) (u h) (v w)', u=5, v=5)
        disp_img = LFintegrate_RGB(subdispout, 1, args.patch_size_for_test, args.stride_for_test, label.size(-2),
                                      label.size(-1))
        Removal_img = LFintegrate_RGB_all(subLFout, 5, args.patch_size_for_test, args.stride_for_test)

        PSNR, SSIM = cal_metrics_all(Removal_img.data.cpu(), label.data.cpu(), 5)

        if logger:
            logger.log_string(' Restore the LF {} to PSNR {}/SSIM {}'.format(LF_name, PSNR, SSIM))
        else:
            print(' Restore the LF {} to PSNR {}'.format(LF_name, PSNR))
            
        psnr_iter_test.append(PSNR)
        ssim_iter_test.append(SSIM)

    return psnr_iter_test, ssim_iter_test, LF_name

def cal_metrics_all(img1, img2, angRes):
    if len(img1.size())==3:
        [_ ,H, W] = img1.size()
        img1 = img1.view(3, angRes, H // angRes, angRes, W // angRes).permute(1,3,2,4,0)
    if len(img2.size())==3:
        [_ ,H, W] = img2.size()
        img2 = img2.view(3, angRes, H // angRes, angRes, W // angRes).permute(1,3,2,4,0)

    [U, V, h, w, _] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')

    for u in range(U):
        for v in range(V):
            PSNR[u, v] = cal_psnr_all(img1[u, v, :, :, :], img2[u, v, :, :, :])
            SSIM[u, v] = cal_ssim_all(img1[u, v, :, :, :], img2[u, v, :, :, :])
            pass
        pass

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean

def cal_psnr_all(img1, img2):
    img1_np = img1.clip(0,1).data.cpu().numpy()
    img2_np = img2.clip(0,1).data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)

def cal_ssim_all(img1, img2):
    img1_np = img1.clip(0,1).data.cpu().numpy()
    img2_np = img2.clip(0,1).data.cpu().numpy()

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, multichannel=True)

def LFintegrate_RGB(subLF, angRes, pz, stride, h, w):
    ''' if centra view only: angres=1'''
    if subLF.dim() == 5:
        subLF = rearrange(subLF, 'n1 n2 c (a1 h) (a2 w) -> n1 n2 c a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 c a1 a2 h w -> a1 a2 c (n1 h) (n2 w)')
    outLF = outLF[:, :, :, 0:h, 0:w]
    outLF = outLF.squeeze()

    return outLF

def LFintegrate_RGB_centra(subLF, angRes, pz, stride, h, w):
    ''' if centra view only: angres=1'''
    if subLF.dim() == 5:
        subLF = rearrange(subLF, 'n1 n2 c (a1 h) (a2 w) -> n1 n2 c a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 c a1 a2 h w -> a1 a2 c (n1 h) (n2 w)')
    outLF = outLF[:, :, :, 0:h, 0:w]
    outLF = outLF.squeeze()

    return outLF

def LFintegrate_RGB_all(subLF, angRes, pz, stride):
    if subLF.dim() == 5:
        subLF = rearrange(subLF, 'n1 n2 c (a1 h) (a2 w) -> n1 n2 c a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 c a1 a2 h w -> c (a1 n1 h) (a2 n2 w)')
    return outLF

if __name__ == '__main__':
    from option import args
