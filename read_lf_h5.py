import h5py
import numpy as np
import cv2
import einops
import torch
from torchvision.transforms import Resize
import os

def show_LF(LF_T, an=5):
    if len(LF_T.shape) == 5:
        LF = einops.rearrange(LF_T, 'U V H W C -> (U H) (V W) C')
    if len(LF_T.shape) == 4:
        LF = einops.rearrange(LF_T, '1 C W H -> H W C')
    LF = LF.numpy()
    cv2.imshow('a', LF)
    cv2.waitKey(0)

h = 432
w = 624
an = 5

path0 = 'LFRR_DATA/LFRR_testing/synthetic'
src_datasets = os.listdir(path0)
src_datasets.sort()

for index_scenes in range(len(src_datasets)):
    name_scenes = src_datasets[index_scenes]
    reflection_root = path0 + '/' + name_scenes
    data = h5py.File(reflection_root)
    trans_LF = np.array(data[('trans_LF')])
    blended_LF = np.array(data[('blended_LF')])
    file_name = ['./LFRR_DATA_centraview/train_syn/' + name_scenes + '.png']
    os.makedirs('./LFRR_DATA_centraview/train_syn/', exist_ok=True)

    blended_LF = np.transpose(blended_LF, (2, 1, 0))
    LF_B_centra = blended_LF[h * (an // 2):h * (an // 2 + 1), w * (an // 2):w * (an // 2 + 1), :]
    cv2.imwrite(file_name[0], LF_B_centra[:, :, ::-1] * 255)
