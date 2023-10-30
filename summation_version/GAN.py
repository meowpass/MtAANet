import torch
import torch.nn as nn
from Dis import define_D
from GANLoss import GANLoss
import os
from Imagepool import ImagePool
from summation_version.Ablation_experiments.U_net import U_Net

class PGAN(nn.Module):
    def __init__(self, gpu_ids=[], is_Train=True, continue_train=True):
        super(PGAN, self).__init__()
        self.isTrain = is_Train
        self.gpu_ids = gpu_ids
        self.fake_AB_pool = ImagePool(50)
        self.netG = U_Net()
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netG.cuda(gpu_ids[0])
        self.netD = define_D(gpu_ids=self.gpu_ids)
        numG = sum(p.numel() for p in self.netG.parameters())
        numD = sum(p.numel() for p in self.netD.parameters())
        total = (numG + numD) / (2 ** 20)
        print("Network parameters:%.3f Mb" % total)
        self.continue_train = continue_train
        if self.isTrain:
            self.lr4reg = 5e-6
            self.lr = 0.0004
            self.epoch = 0
            self.criterionGAN = GANLoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=self.lr4reg, betas=(0.9, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=self.lr, betas=(0.9, 0.999))

        if not self.isTrain or self.continue_train:
            print('----------load_model----------')
            save_dir = r'D:\code\MtAA_NET\multitask4segdose\U_Net'
            # save_dir = '.\\checkpoints_Unetzoo\\Unet'
            netD_name = 'latest_netD.pth'
            netG_name = 'latest_netG.pth'
            net_D = os.path.join(save_dir, netD_name)
            net_G = os.path.join(save_dir, netG_name)

            self.netD.load_state_dict(torch.load(net_D))
            self.netG.load_state_dict(torch.load(net_G))

    def forward(self, inputs, target):  # 2 6 512 512
        self.inputs = inputs
        self.fake = self.netG(inputs)
        self.target = target

    def backward_G(self):
        fake_AB = torch.cat((self.inputs, self.fake), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake, self.target)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * 10

        self.loss_G.backward()

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.inputs, self.fake), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.inputs, self.target), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_real + self.loss_D_fake)

        self.loss_D.backward()

    def optimizer_parameters(self, inputs, target):
        self.forward(inputs, target)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()  # 查看了最原始gan 此处也没有设置netD的requestGrad为false
        self.optimizer_G.step()

    def save_model(self, epoch):

        save_dir = r'D:\code\MtAA_NET\multitask4segdose\U_Net'
        netD_name = '%s_netD.pth' % epoch
        netG_name = '%s_netG.pth' % epoch
        net_D = os.path.join(save_dir, netD_name)
        net_G = os.path.join(save_dir, netG_name)
        torch.save(self.netD.cpu().state_dict(), net_D)
        self.netD.cuda(self.gpu_ids[0])  # 要有 因为网络还要继续用
        torch.save(self.netG.cpu().state_dict(), net_G)
        self.netG.cuda(self.gpu_ids[0])
