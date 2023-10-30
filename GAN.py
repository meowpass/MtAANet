import os
import torch
import torch.nn as nn
from Dis import define_D
from GANLoss import GANLoss
from Imagepool import ImagePool
from summation_version.Ablation_experiments.old_model.Unet_SegNet_CFF_DS import shared_encoder, seg_decoder, dose_decoder


class PGAN(nn.Module):
    def __init__(self, gpu_ids=[], is_Train=True, continue_train=True):
        super(PGAN, self).__init__()
        self.isTrain = is_Train
        self.gpu_ids = gpu_ids
        self.continue_train = continue_train
        self.fake_AB_pool = ImagePool(50)
        self.netGEnc = shared_encoder()
        self.netG4seg = seg_decoder()
        self.netG4dose = dose_decoder()
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netGEnc.cuda(gpu_ids[0])
            self.netG4seg.cuda(gpu_ids[0])
            self.netG4dose.cuda(gpu_ids[0])
        self.netD = define_D(gpu_ids=gpu_ids)
        if self.isTrain:
            self.lr_cla = 0.0004
            self.lr_reg = 1e-5
            self.lr = 0.0004
            self.epoch = 0
            self.criterionGAN = GANLoss()
            self.criterionL1 = nn.L1Loss()
            self.criterion4seg = nn.CrossEntropyLoss()

            self.optimizer_encoder = torch.optim.Adam(self.netGEnc.parameters(), lr=self.lr_reg, betas=(0.9, 0.999))
            self.optimizer_dose_decoder = torch.optim.Adam(self.netG4dose.parameters(), lr=self.lr_reg,
                                                           betas=(0.9, 0.999))
            self.optimizer_seg_decoder = torch.optim.Adam(self.netG4seg.parameters(), lr=self.lr_cla,
                                                          betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.9, 0.999))

        if not self.isTrain or self.continue_train:
            print('----------load model-------------')
            save_dir = r'D:\code\MtAA_NET\multitask4segdose\CrossValidation1'
            netD_name = 'latest_netD.pth'
            netGenc_name = 'latest_netGEnc.pth'
            netG4seg_name = 'latest_netG4seg.pth'
            netG4dose_name = 'latest_netG4dose.pth'

            net_D = os.path.join(save_dir, netD_name)
            netGEnc = os.path.join(save_dir, netGenc_name)
            netG4seg = os.path.join(save_dir, netG4seg_name)
            netG4dose = os.path.join(save_dir, netG4dose_name)

            self.netD.load_state_dict(torch.load(net_D))
            self.netGEnc.load_state_dict(torch.load(netGEnc))
            self.netG4seg.load_state_dict(torch.load(netG4seg))
            self.netG4dose.load_state_dict(torch.load(netG4dose))

    def forward(self, origin, target, seg_GT):
        self.origin = origin
        self.seg_GT = seg_GT
        self.dose_GT = target
        x5, x4, x3, x2, x1 = self.netGEnc(origin)
        self.seg, d4, d3, d2, d1 = self.netG4seg(x5, x4, x3, x2, x1)
        self.dose = self.netG4dose(x5, x4, x3, x2, x1, d4, d3, d2, d1)

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.origin, self.dose), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.origin, self.dose_GT), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    def optimizer_parameters(self, origin, target, seg_GT):
        self.forward(origin, target, seg_GT)
        self.optimizer_D.zero_grad()
        self.loss_segmentation = self.criterion4seg(self.seg, torch.squeeze(self.seg_GT).long())
        fake_AB = torch.cat((self.origin, self.dose), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.dose, self.dose_GT)
        self.loss_prediction = self.loss_G_GAN + 100 * self.loss_G_L1
        self.loss_G = self.loss_segmentation + self.loss_prediction
        self.backward_D()
        self.optimizer_encoder.zero_grad()
        self.optimizer_seg_decoder.zero_grad()
        self.optimizer_dose_decoder.zero_grad()
        self.loss_G.backward()
        self.optimizer_D.step()
        self.optimizer_encoder.step()
        self.optimizer_seg_decoder.step()
        self.optimizer_dose_decoder.step()

    def save_model(self, epoch):
        save_dir = r'D:\code\MtAA_NET\multitask4segdose\CrossValidation1'
        netD_name = '%s_netD.pth' % epoch
        netGEnc_name = '%s_netGEnc.pth' % epoch
        netG4seg_name = '%s_netG4seg.pth' % epoch
        netG4dose_name = '%s_netG4dose.pth' % epoch

        netD = os.path.join(save_dir, netD_name)
        netGEnc = os.path.join(save_dir, netGEnc_name)
        netG4seg = os.path.join(save_dir, netG4seg_name)
        netG4dose = os.path.join(save_dir, netG4dose_name)

        torch.save(self.netD.cpu().state_dict(), netD)
        self.netD.cuda(self.gpu_ids[0])
        torch.save(self.netGEnc.cpu().state_dict(), netGEnc)
        self.netGEnc.cuda(self.gpu_ids[0])
        torch.save(self.netG4seg.cpu().state_dict(), netG4seg)
        self.netG4seg.cuda(self.gpu_ids[0])
        torch.save(self.netG4dose.cpu().state_dict(), netG4dose)
        self.netG4dose.cuda(self.gpu_ids[0])
