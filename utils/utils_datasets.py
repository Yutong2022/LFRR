import os

import einops
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from utils import *
import einops
from torchvision.transforms import Resize

# def re_shape(img, h_re=576, w_re=384):
#     img = einops.rearrange(img, 'c (u h) (v w) -> c u h v w', u=5, v=5)
#     img_reshape = img[:,:,:h_re,:,:w_re]
#     img_reshape = einops.rearrange(img_reshape, 'c u h v w -> c (u h) (v w)')
#     return img_reshape

class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_train + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/'
            pass
        elif args.task == 'Dereflection':
            self.dataset_dir = args.path_for_train
            pass

        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

        self.patch_size = args.patch_size

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            trans_LF = np.array(hf.get('trans_LF'))  # trans_LF
            blended_LF = np.array(hf.get('blended_LF')) # Lr_SAI_y
            # syn_reflection_LF = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y


            trans_LF = torch.from_numpy(trans_LF)
            blended_LF = torch.from_numpy(blended_LF)
            # window_size = random.randrange(self.patch_size, self.patch_size*2, 8)
            window_size = self.patch_size

            '''crop'''
            trans_LF = einops.rearrange(trans_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
            blended_LF = einops.rearrange(blended_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
            _, _, H, W = trans_LF.size()
            x = random.randrange(0, H - window_size, 8)
            y = random.randrange(0, W - window_size, 8)

            trans_LF = trans_LF[:, :, x:x + window_size, y:y + window_size]  # [ah,aw,ph,pw]
            blended_LF = blended_LF[:, :, x:x + window_size, y:y + window_size]  # [ah,aw,ph,pw]

            # torch_resize = Resize([self.patch_size,self.patch_size]) # 定义Resize类对象
            # trans_LF = torch_resize(trans_LF)
            # blended_LF = torch_resize(blended_LF)

            trans_LF = einops.rearrange(trans_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)
            blended_LF = einops.rearrange(blended_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in,
                                          v=self.angRes_in)

            '''augmentation'''
            trans_LF, blended_LF = augmentation(trans_LF, blended_LF)
            # reflection_LF = blended_LF - trans_LF
            # trans_LF = ToTensor()(trans_LF.copy())
            # blended_LF = ToTensor()(blended_LF.copy())
            # reflection_LF = ToTensor()(syn_reflection_LF.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return  blended_LF, trans_LF, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num

# class TrainSetDataLoader(Dataset):
#     def __init__(self, args):
#         super(TrainSetDataLoader, self).__init__()
#         self.angRes_in = args.angRes_in
#         self.angRes_out = args.angRes_out
#         if args.task == 'SR':
#             self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
#                                str(args.scale_factor) + 'x/'
#         elif args.task == 'RE':
#             self.dataset_dir = args.path_for_train + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
#                                str(args.angRes_out) + 'x' + str(args.angRes_out) + '/'
#             pass
#         elif args.task == 'Dereflection':
#             self.dataset_dir = args.path_for_train
#             pass

#         if args.data_name == 'ALL':
#             self.data_list = os.listdir(self.dataset_dir)
#         else:
#             self.data_list = [args.data_name]

#         self.file_list = []
#         for data_name in self.data_list:
#             tmp_list = os.listdir(self.dataset_dir + data_name)
#             for index, _ in enumerate(tmp_list):
#                 tmp_list[index] = data_name + '/' + tmp_list[index]

#             self.file_list.extend(tmp_list)

#         self.item_num = len(self.file_list)

#         self.patch_size = args.patch_size

#     def __getitem__(self, index):
#         file_name = [self.dataset_dir + self.file_list[index]]
#         with h5py.File(file_name[0], 'r') as hf:
#             trans_LF = np.array(hf.get('trans_LF'))  # trans_LF
#             blended_LF = np.array(hf.get('blended_LF')) # Lr_SAI_y
#             # syn_reflection_LF = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y


#             trans_LF = torch.from_numpy(trans_LF)
#             blended_LF = torch.from_numpy(blended_LF)
#             window_size = random.randrange(self.patch_size, self.patch_size*2, 8)

#             '''crop'''
#             trans_LF = einops.rearrange(trans_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
#             blended_LF = einops.rearrange(blended_LF, 'c (u H) (v W) -> c (u v) H W', u=self.angRes_in, v=self.angRes_in)
#             _, _, H, W = trans_LF.size()
#             x = random.randrange(0, H - self.patch_size, 8)
#             y = random.randrange(0, W - self.patch_size, 8)

#             trans_LF = trans_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]  # [ah,aw,ph,pw]
#             blended_LF = blended_LF[:, :, x:x + self.patch_size, y:y + self.patch_size]  # [ah,aw,ph,pw]

#             trans_LF = einops.rearrange(trans_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in, v=self.angRes_in)
#             blended_LF = einops.rearrange(blended_LF, 'c (u v) H W -> c (u H) (v W)', u=self.angRes_in,
#                                           v=self.angRes_in)

#             '''augmentation'''
#             # Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
#             # reflection_LF = blended_LF - trans_LF
#             # trans_LF = ToTensor()(trans_LF.copy())
#             # blended_LF = ToTensor()(blended_LF.copy())
#             # reflection_LF = ToTensor()(syn_reflection_LF.copy())

#         Lr_angRes_in = self.angRes_in
#         Lr_angRes_out = self.angRes_out

#         return  blended_LF, trans_LF, [Lr_angRes_in, Lr_angRes_out]

#     def __len__(self):
#         return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None

    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
        elif args.task == 'Dereflection':
            dataset_dir = args.path_for_test
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]
        elif args.task == 'Dereflection':
            self.dataset_dir = args.path_for_test
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            trans_LF = np.array(hf.get('trans_LF'))  # trans_LF
            blended_LF = np.array(hf.get('blended_LF')) # Lr_SAI_y
            # syn_reflection_LF = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y
            trans_LF = torch.from_numpy(trans_LF)
            blended_LF = torch.from_numpy(blended_LF)

            # trans_LF = re_shape(trans_LF)
            # blended_LF = re_shape(blended_LF)

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return  blended_LF, trans_LF, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[2])
        label = torch.flip(label, dims=[2])
        # data = data[:, :, ::-1]
        # label = label[:, :, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[1])
        label = torch.flip(label, dims=[1])
        # data = data[:, ::-1, :]
        # label = label[:, ::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.permute(0, 2, 1)
        label = label.permute(0, 2, 1)
    return data, label

