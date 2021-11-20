import os
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from skimage.measure import compare_ssim as ssim
# from torch.nn.modules.loss import CrossEntropyLoss

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, DataConsistencyInKspace_I, fft2,ifft2, complex_abs_eval,DiceLoss, dice_coeff

class RecurrentModel(nn.Module):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = opts.n_recurrent

        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False
        self.net_G_I = get_generator('RECON', opts)
        self.net_G_Seg = get_generator('SEG', opts)

        self.networks.append(self.net_G_I)
        self.networks.append(self.net_G_Seg)


        if self.is_train:
            self.loss_names += ['loss_G_L1']
            param = list(self.net_G_I.parameters()) +list(self.net_G_Seg.parameters())
            self.optimizer_G = torch.optim.Adam(param,lr=opts.lr,betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()
        self.criterion_dice = DiceLoss(2)
        self.opts = opts

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I())
        self.dcs_I = dcs_I

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.tag_image_full = data['tag_image_full'].to(self.device)
        self.tag_image_sub = data['tag_image_sub'].to(self.device)
        self.label = data['label'].to(self.device)
        self.tag_kspace_mask2d = data['tag_kspace_mask2d'].to(self.device)
        self.case_name = data['case_name'][0]

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        I = self.tag_image_sub
        I.requires_grad_(True)
        net = {}
        for i in range(1, self.n_recurrent + 1):
            '''Image Space'''
            net['r%d_img_pred' % i] = self.net_G_I(I)  # output recon image
            I = self.dcs_I[i - 1](net['r%d_img_pred' % i], self.tag_image_full, self.tag_kspace_mask2d)
            '''segmentation'''
            seg_img = self.net_G_Seg(I)
        self.net = net
        self.recon = I
        self.segment = seg_img

    def update_G(self):
        self.optimizer_G.zero_grad()
        # Image domain
        loss_img_dc = 0
        loss_mse = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion_mse(self.recon, self.tag_image_full)
            loss_mse = loss_mse + self.criterion_l1(self.recon, self.tag_image_full)
        # # Kspace domain
        # seg domain
        loss_ce = 0
        loss_dice = 0
        for j in range(1, self.n_recurrent + 1):
            loss_ce = loss_ce + self.criterion_ce(self.segment, self.label.long())#torch.squeeze
            loss_dice = loss_dice + self.criterion_dice(self.segment, self.label, softmax=True)
        loss_seg =  loss_ce + loss_dice
        loss_G_L1 = loss_img_dc + loss_mse + loss_seg

        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img = loss_img_dc.item()
        self.loss_mse = loss_mse.item()
        self.loss_seg = loss_seg.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_L1 = 0
        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'loss_G_L1: {:.4f} loss_img: {:.4f} loss_mse: {:.4f} loss_seg: {:.4f}'.format(self.loss_G_L1, self.loss_img, self.loss_mse,self.loss_seg)
        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):
        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.module.state_dict()
            state['net_G_Seg'] = self.net_G_Seg.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()
        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G_I.module.load_state_dict(checkpoint['net_G_I'])
            self.net_G_Seg.module.load_state_dict(checkpoint['net_G_Seg'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_dice = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []
        segment_images = []
        label_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()
            if self.opts.wr_L1 > 0:
                psnr_recon = psnr(complex_abs_eval(self.recon), complex_abs_eval(self.tag_image_full))
                avg_psnr.update(psnr_recon)

                ssim_recon = ssim(complex_abs_eval(self.recon)[0,0,:,:].cpu().numpy(), complex_abs_eval(self.tag_image_full)[0,0,:,:].cpu().numpy())
                avg_ssim.update(ssim_recon)

                dice_seg = dice_coeff(torch.argmax(torch.softmax(self.segment, dim=1), dim=1).cpu().numpy(), self.label.cpu().numpy())
                avg_dice.update(dice_seg)
                recon_images.append(self.recon[0].cpu())
                gt_images.append(self.tag_image_full[0].cpu())
                input_images.append(self.tag_image_sub[0].cpu())
                segment_images.append(torch.argmax(torch.softmax(self.segment, dim=1), dim=1).cpu())
                label_images.append(self.label.cpu())

            message = 'PSNR: {:4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim.avg)
            message += 'DICE: {:4f} '.format(avg_dice.avg)
            val_bar.set_description(desc=message)

        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg
        self.dice_seg = avg_dice.avg

        self.results = {}
        self.results['rec'] = torch.stack(recon_images).squeeze().numpy()
        self.results['gt'] = torch.stack(gt_images).squeeze().numpy()
        self.results['zf'] = torch.stack(input_images).squeeze().numpy()
        self.results['seg'] = torch.stack(segment_images).squeeze().numpy()
        self.results['label'] = torch.stack(label_images).squeeze().numpy()