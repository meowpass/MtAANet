import torch
import numpy as np
from summation_version.MT_GAN import MT_PGAN
import SimpleITK as sit
from Visualize import Visualizer
from summation_version.dataset import make_datasetS
from collections import OrderedDict
from torch.autograd import Variable

if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [0]
    vis = Visualizer('Multi_task4seg_dose')
    save_path = r'E:\dataset\cervical_cancer\Ablation\U_NET+GAN+SegNet+CtFF+DS180epoch\results'
    _, testData = make_datasetS()
    model = MT_PGAN(gpu_ids=param['gpu_ids'], is_Train=False, continue_train=True)
    batch = 1
    for ii, batch_sample in enumerate(testData):
        inputs, seg_GT, target, channel, name = batch_sample['inputs'], batch_sample['rs'], batch_sample['rd'], \
                                                batch_sample['channel'], batch_sample[
                                                    'name']  # inputs shape:[1, 154, 7, 512, 512]
        print(name)
        print(inputs.shape, target.shape)
        seg_GT = seg_GT.squeeze(0)
        seg_GT = seg_GT.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
        inputs = inputs.squeeze(0)  # inputs shape:[154,512, 512]
        inputs = inputs.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
        target = target.squeeze(0)
        target = target.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
        dose = np.zeros(shape=(channel, 512, 512))
        seg = np.zeros(shape=(channel, 512, 512))
        dose_real = np.zeros(shape=(channel, 512, 512))
        seg_real = np.zeros(shape=(channel, 512, 512))
        for i in range(channel // batch):
            if batch * i + batch <= channel - 1:  # 这样会少最后一个batch 所以改为<=c
                # print(batch*i + batch)
                main_inputs = inputs[batch * i:batch * (i + 1), :, :, :]  # inputs shape:[2, 1, 512, 512]
                main_target = target[batch * i:batch * (i + 1), :, :, :]
                main_seg_GT = seg_GT[batch * i:batch * (i + 1), :, :, :]
            else:
                break
            origin = main_inputs  # torch.Size([2, 1, 512, 512])
            # print('origin:', origin.shape)
            origin_seg_GT = main_seg_GT

            origin_seg_GT[origin_seg_GT > 1] = 1
            # print('seg_GT:',origin_seg_GT.shape)
            print(type(origin_seg_GT))
            # origin, targets, seg_GT4test = Variable(origin).cuda(), Variable(main_target).cuda(), Variable(
            #     origin_seg_GT).cuda()
            # print(torch.max(targets))
            model.forward(origin.cuda(), main_target.cuda(), origin_seg_GT.cuda())
            # model.forward(origin, targets, seg_GT4test)
            dose_pre = model.dose.squeeze(0)
            seg_pre = model.seg
            seg_pre = torch.max(seg_pre, 1)[1]
            dose_pre = dose_pre.detach().cpu().numpy()
            seg_pre = (seg_pre[0].detach().cpu().numpy() * 255)
            # targets = targets.detach().cpu().numpy()
            targets = main_target.detach().cpu().numpy()
            # seg_GT = seg_GT4test.detach().cpu().numpy()
            rs = main_seg_GT.detach().cpu().numpy()

            dose[i, :, :] = dose_pre
            seg[i, :, :] = seg_pre
            print(targets.shape)
            dose_real[i, :, :] = targets
            print('seg_GT:', seg_GT.shape)
            seg_real[i, :, :] = rs

        dose_pre_name = str(name[0]) + '_dosePre.mha'
        dose_real_name = str(name[0]) + '_doseReal.mha'
        seg_pre_name = str(name[0]) + '_segPre.mha'
        seg_real_name = str(name[0]) + '_segReal.mha'

        dose_fake: sit.Image = sit.GetImageFromArray(dose)
        dose_fake.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '\\' + dose_pre_name)
        s.Execute(dose_fake)
        sit.WriteImage(dose_fake, save_path + '\\' + dose_pre_name)

        dose_true: sit.Image = sit.GetImageFromArray(dose_real)
        dose_true.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '\\' + dose_real_name)
        s.Execute(dose_true)
        sit.WriteImage(dose_true, save_path + '\\' + dose_real_name)

        seg_fake: sit.Image = sit.GetImageFromArray(seg)
        seg_fake.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '\\' + seg_pre_name)
        s.Execute(seg_fake)
        sit.WriteImage(seg_fake, save_path + '\\' + seg_pre_name)

        seg_true: sit.Image = sit.GetImageFromArray(seg_real)
        seg_true.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '\\' + seg_real_name)
        s.Execute(seg_true)
        sit.WriteImage(seg_true, save_path + '\\' + seg_real_name)
