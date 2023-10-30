import time
from summation_version.GAN import PGAN
from Visualize import Visualizer
from collections import OrderedDict
from torch.autograd import Variable
from summation_version.MT_GAN import MT_PGAN
from summation_version.dataset import make_datasetS

if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [0]
    vis = Visualizer('Dose_prediction_UNet')
    # fp_lossG = open('.\\result\\netG_losses.txt','w')
    # fp_lossD = open('.\\result\\netD_losses.txt', 'w')
    x_time = []
    y_loss_G = []
    y_loss_D = []
    trainData, _ = make_datasetS()
    model = PGAN(gpu_ids=param['gpu_ids'], is_Train=True, continue_train=False)
    batch = 4  # dataloader的bs是1 但是由于数据是3D的 取出来时是1 1 185 6 512 512（其中第二个1是dataloder是人为误操作） 我们人工设置batch=2得到2 6 512 512（其中2是依次取到185）来达到2D输入网络
    for epoch in range(200):
        epoch_start_time = time.time()
        for ii, batch_sample in enumerate(trainData):
            inputs, target, channel = batch_sample['inputs'], batch_sample['rd'], \
                                      batch_sample['channel']
            # torch.Size([1, 140, 512, 512])
            inputs = inputs.squeeze(0)  # inputs shape:[154,512, 512]
            inputs = inputs.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
            target = target.squeeze(0)
            target = target.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
            # print(c)
            for i in range(channel // batch):
                if batch * i + batch <= channel - 1:  # 这样会少最后一个batch 所以改为<=c
                    # print(batch*i + batch)
                    main_inputs = inputs[batch * i:batch * (i + 1), :, :, :]  # inputs shape:[2, 1, 512, 512]
                    main_target = target[batch * i:batch * (i + 1), :, :, :]
                else:
                    break
                origin = main_inputs  # torch.Size([2, 1, 512, 512])
                # print('origin:', origin.shape)
                # print('seg_GT:',origin_seg_GT.shape)
                origin, targets = Variable(origin).cuda(), Variable(main_target).cuda()
                # print(torch.max(targets))
                model.optimizer_parameters(origin, targets)
                dose = model.fake  # 2 1 512 512
                target1 = targets
                dose_show = dose[
                    0].detach().cpu().numpy()  # 如果去掉detach 报错：loss.cpu().numpy())#报错 RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                target_show = target1[0].detach().cpu().numpy()  # 就是说要numpy不能存有grad的东西 所以detach阻断反向传播（即去掉grad属性）
                vis.img("dose", img_=dose_show)
                vis.img("target", img_=target_show)
                # vis.img("original", img_=(((orig_to_show))))
                vis.plot("loss_G", model.loss_G.item())
                vis.plot("loss_D", model.loss_D.item())
                y_loss_G.append(model.loss_G.item())
                y_loss_D.append(model.loss_D.item())
        print('lr=', model.lr4reg, ',loss_G=', model.loss_G.item(), ',loss_D=', model.loss_D.item())
        epoch_end_time = time.time()
        print('time consuming:', (epoch_end_time - epoch_start_time) / 60)
        print("local time", time.strftime('%c'))
        # if epoch == 0 or epoch >= 30:
        print('save_model....')
        print('epoch: %d' % epoch)
        model.save_model(epoch)
        model.save_model('latest')
