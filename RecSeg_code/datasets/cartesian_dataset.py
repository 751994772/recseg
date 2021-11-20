import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from datasets.utilizes import *
from models.utils import fft2, ifft2, to_tensor
# import h5py

class MRIDataset_Cartesian(data.Dataset):
    def __init__(self, opts, mode):
        self.mode = mode
        if self.mode == 'TRAIN':
            self.data_dir_image = os.path.join(opts.data_root, 'train')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = None

        if self.mode == 'VALI':
            self.data_dir_image = os.path.join(opts.data_root, 'vali')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 1234

        if self.mode == 'TEST':
            self.data_dir_image = os.path.join(opts.data_root, 'test')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 5678

        self.data_dir_flair = os.path.join(self.data_dir_flair)    # ref kspace directory (T1)
        self.mask_path = opts.mask_dir
    def __getitem__(self, idx):

        mask = sio.loadmat(self.mask_path)['mask']
        mask = mask[np.newaxis,:,:]
        mask = np.concatenate([mask, mask], axis=0)
        mask400 = torch.from_numpy(mask.astype(np.float32))

        slice_name = self.sample_list[idx].strip('\n')
        image_data = sio.loadmat(os.path.join(self.data_dir_image, slice_name + '.mat'))

        image_data_real = image_data['ZF'].real#ZF
        image_data_real = image_data_real[np.newaxis,:, :]
        image_data_imag = image_data['ZF'].imag
        image_data_imag = image_data_imag[np.newaxis,:, :]
        zf_img = np.concatenate([image_data_real, image_data_imag], axis=0)#2,192,192
        zf_img = to_tensor(zf_img).float()#.permute(2, 0, 1) #  flair zf

        image_data_real = image_data['GT'].real#gt
        image_data_real = image_data_real[np.newaxis,:, :]
        image_data_imag = image_data['GT'].imag
        image_data_imag = image_data_imag[np.newaxis,:, :]
        gt_img = np.concatenate([image_data_real, image_data_imag], axis=0)#2,192,192
        gt_img = to_tensor(gt_img).float()#.permute(2, 0, 1) #  flair gt

        label_img = image_data['LABEL']#LABEL
        label_img[np.where(label_img == 0)] = 0
        label_img[np.where(label_img >0)] = 1

        label = torch.from_numpy(label_img.astype(np.float32))
        label = label.long()

# ---------------------over------
        return {
                'tag_image_full': gt_img,
                'tag_image_sub': zf_img,
                'label': label,
                'tag_kspace_mask2d': mask400,
                'case_name':self.sample_list[idx].strip('\n'),}

    def __len__(self):
        return len(self.sample_list)

if __name__ == '__main__':
    a = 1
