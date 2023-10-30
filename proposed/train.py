import time
import torch
import numpy as np
from GAN import PGAN
from multitask4segdose.Visualize import Visualizer
from multitask4segdose.proposed.dataset import make_datasetS
from collections import OrderedDict
from torch.autograd import Variable

if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [0]
    vis = Visualizer('Multi_task4seg_dose')
    # fp_lossG = open('.\\result\\netG_losses.txt','w')
    # fp_lossD = open('.\\result\\netD_losses.txt', 'w')

    trainData, _ = make_datasetS()
    model = PGAN(gpu_ids=param['gpu_ids'], is_Train=True, continue_train=False)
    batch = 2  # dataloader的bs是1 但是由于数据是3D的 取出来时是1 1 185 6 512 512（其中第二个1是dataloder是人为误操作） 我们人工设置batch=2得到2 6 512 512（其中2是依次取到185）来达到2D输入网络
    for epoch in range(200):
        epoch_start_time = time.time()
        for ii, batch_sample in enumerate(trainData):
            inputs, target, channel = batch_sample['inputs'], batch_sample['rd'], batch_sample[
                'channel']  # inputs shape:[1, 154, 7, 512, 512]
            inputs = inputs.squeeze(0)  # inputs shape:[154, 7, 512, 512]
            target = target.squeeze(0)
            # print(c)
            for i in range(channel // batch):
                if batch * i + batch <= channel - 1:  # 这样会少最后一个batch 所以改为<=c
                    # print(batch*i + batch)
                    main_inputs = inputs[batch * i:batch * (i + 1), :, :, :]  # inputs shape:[2, 7, 512, 512]
                    main_target = target[batch * i:batch * (i + 1), :, :, :]
                else:
                    break
                origin = main_inputs[:, 0, :, :]
                # print(origin.shape)
                origin = origin.unsqueeze(1)
                seg_GT = main_inputs[:, 1, :, :] + main_inputs[:, 2, :, :] + main_inputs[:, 3, :, :] + \
                         main_inputs[:, 4, :, :] + main_inputs[:, 5, :, :] + main_inputs[:, 6, :, :]
                seg_GT[seg_GT > 1] = 1
                seg_GT = seg_GT.unsqueeze(1)
                origin, targets, seg_GT = Variable(origin).cuda(), Variable(main_target).cuda(), Variable(
                    seg_GT).cuda()  # Variable默认不求梯度
                # print(torch.max(targets))
                model.optimizer_parameters(origin, targets, seg_GT)
                dose = model.dose  # 2 1 512 512
                seg = model.seg
                seg_GT = model.seg_GT
                target1 = targets
                seg = torch.max(seg, 1)[1]
                dose_show = dose[
                    0].detach().cpu().numpy()  # 如果去掉detach 报错：loss.cpu().numpy())#报错 RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                target_show = target1[0].detach().cpu().numpy()  # 就是说要numpy不能存有grad的东西 所以detach阻断反向传播（即去掉grad属性）
                seg_show = (seg[0].detach().cpu().numpy() * 255).astype(np.uint8)
                seg_GT_show = (seg_GT[0].detach().cpu().numpy() * 255).astype(np.uint8)
                vis.img("dose", img_=dose_show)
                vis.img("target", img_=target_show)
                # vis.img("original", img_=(((orig_to_show))))
                vis.img("seg", img_=seg_show)
                vis.img("seg_GT", img_=seg_GT_show)
                vis.plot("loss_G", model.loss_G.item())
                vis.plot("loss_D", model.loss_D.item())
        epoch_end_time = time.time()
        print('time consuming:', epoch_end_time - epoch_start_time)
        print("local time", time.localtime())
        # if epoch == 0 or epoch >= 30:
        print('save_model....')
        print('epoch: %d' % epoch)
        model.save_model(epoch)
        model.save_model('latest')
