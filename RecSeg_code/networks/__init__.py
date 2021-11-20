import torch.nn as nn
import numpy as np
from utils import arange
from networks.unet import UNet
import pdb

def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts):
    if name == 'RECON':
        network = UNet(n_channels=2, n_classes=2, bilinear=True)
    elif name =='SEG':
        network = UNet(n_channels=2, n_classes=2, bilinear=True)
    else:
        raise NotImplementedError

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)