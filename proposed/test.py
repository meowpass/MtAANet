import torch
import numpy as np
from GAN import PGAN
import SimpleITK as sit
from multitask4segdose.Visualize import Visualizer
from multitask4segdose.proposed.dataset import make_datasetS
from collections import OrderedDict
from torch.autograd import Variable

if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [0]
    vis = Visualizer('Multi_task4seg_dose')
    save_path = r'F:\result'
    _, testData = make_datasetS()
    model = PGAN(gpu_ids=param['gpu_ids'], is_Train=False, continue_train=False)
    batch = 1
    for ii, batch_sample in enumerate(testData):
        inputs, target, channel, name = batch_sample['inputs'], batch_sample['rd'], batch_sample[
            'channel'], batch_sample['name']  # inputs shape:[1, 154, 7, 512, 512]
        print(name)
        inputs = inputs.squeeze(0)  # inputs shape:[154, 7, 512, 512]
        target = target.squeeze(0)
        dose = np.zeros(shape=(channel, 512, 512))
        seg = np.zeros(shape=(channel, 512, 512))
        dose_real = np.zeros(shape=(channel, 512, 512))
        seg_real = np.zeros(shape=(channel, 512, 512))
        x4_ = np.zeros(shape=(channel, 64, 64))
        up4_ = np.zeros(shape=(channel, 64, 64))
        for i in range(channel):
            if batch * i + batch <= channel - 1:  # 这样会少最后一个batch 所以改为<=c
                # print(batch*i + batch)
                main_inputs = inputs[batch * i:batch * (i + 1), :, :, :]  # inputs shape:[2, 7, 512, 512]
                main_target = target[batch * i:batch * (i + 1), :, :, :]
            else:
                break
            seg_GT = main_inputs[:, 1, :, :] + main_inputs[:, 2, :, :] + main_inputs[:, 3, :, :] + \
                     main_inputs[:, 4, :, :] + main_inputs[:, 5, :, :] + main_inputs[:, 6, :, :]
            seg_GT[seg_GT > 1] = 1
            seg_GT = seg_GT.unsqueeze(1)
            origin = main_inputs[:, 0, :, :]
            origin = origin.unsqueeze(1)
            origin, targets, seg_GT = Variable(origin).cuda(), Variable(main_target).cuda(), Variable(
                seg_GT).cuda()

            model.forward(origin, targets, seg_GT)
            dose_pre = model.dose.squeeze(0)
            seg_pre = model.seg
            seg_pre = torch.max(seg_pre, 1)[1]
            dose_pre = dose_pre.detach().cpu().numpy()
            seg_pre = (seg_pre[0].detach().cpu().numpy() * 255)
            targets = targets.detach().cpu().numpy()
            seg_GT = seg_GT.detach().cpu().numpy()

            x4 = model.x4
            x4 = (x4[0].detach().cpu().numpy()*255)

            up4 = model.up4
            up4 = (up4[0].detach().cpu().numpy()*255)

            dose[i, :, :] = dose_pre
            seg[i, :, :] = seg_pre
            dose_real[i, :, :] = targets
            seg_real[i, :, :] = seg_GT
            x4_[i,:,:] = x4
            up4_[i,:,:] = up4_

        dose_pre_name = str(name[0]) + '_dosePre.mha'
        dose_real_name = str(name[0]) + '_doseReal.mha'
        seg_pre_name = str(name[0]) + '_segPre.mha'
        seg_real_name = str(name[0]) + '_segReal.mha'

        x4_name = str(name[0]) + '_x4.mha'
        up4_name = str(name[0]) + '_up4.mha'

        x4_pre: sit.Image = sit.GetImageFromArray(x4_)
        x4_pre.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '\\' + x4_name)
        s.Execute(x4_pre)
        sit.WriteImage(x4_pre, save_path + '\\' + x4_name)

        up4_pre: sit.Image = sit.GetImageFromArray(up4_)
        up4_pre.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '\\' + up4_name)
        s.Execute(up4_pre)
        sit.WriteImage(up4_pre, save_path + '\\' + up4_name)

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
