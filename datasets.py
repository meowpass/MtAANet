import os.path
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import SimpleITK as sit

from PIL import Image


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, phase):
    paths = os.path.join(dir, phase)
    # assert os.path.isdir(paths), '%s is not a valid directory' % dir
    rd = ""
    origin = ""
    rs = ""
    for root, files, _ in sorted(os.walk(paths)):
        for file in files:
            if "origin" in file:
                origin = os.path.join(root, file)
            elif "rd" in file:
                rd = os.path.join(root, file)
            # elif "rs" in file:
            else:
                rs = os.path.join(root, file)
    return origin, rd, rs


def make_files(origin, rs, rd):
    names = []
    images = {}
    allRs = []
    for root, _, fnames in sorted(os.walk(rs)):
        for fname in fnames:
            pathrs = os.path.join(root, fname)
            allRs.append(pathrs)
    for root, _, fnames in sorted(os.walk(origin)):
        for fname in fnames:
            name = fname.split('_')[0]
            opath = os.path.join(root, fname)
            # names = os.path.join(root,name)
            names.append(name)
            # if name not in images.keys():
            images[name] = []
            images[name].append(opath)
            for item in allRs:
                # print(item)
                t = item.find(name)
                if t > 0:
                    images[name].append(item)
            rdname = str(name) + '_rd.mha'
            pathrd = os.path.join(rd, rdname)
            images[name].append(pathrd)
    names_ = []  # 为什么不用name而增加一个name_用于返回呢 我认为只是为了增加下面 “ == 7:” 这一行方便，其实if里面continue并且最后return name也行
    images_ = {}
    for i in range(len(names)):
        if len(images[names[i]]) == 8:  # 由于rd和origin一样 但是有病人的危机器官信息可能少 所以去掉问题数据
            names_.append(names[i])
            images_[names[i]] = images[names[i]]

    return names_, images_


class TrainDataset():
    def __init__(self, dir, phase):
        super(TrainDataset, self).__init__()
        self.dir = dir
        self.phase = phase
        self.orgin, self.rd, self.rs = sorted(make_dataset(self.dir, self.phase))
        self.names, self.images = make_files(self.orgin, self.rs, self.rd)

        self.transform = torch.from_numpy

    def __getitem__(self, index):
        image_path = self.names[index]
        print(image_path)
        images = self.images[image_path]

        origin = images[0]
        rd = images[-1]
        Bladder = images[1]
        FemoralHeadL = images[2]
        FemoralHeadR = images[3]
        PCTV = images[4]
        Smallintestine = images[5]
        Rectum = images[6]

        origin_ = sit.ReadImage(origin)
        rd_ = sit.ReadImage(rd)
        Bladder_ = sit.ReadImage(Bladder)
        FemoralHeadL_ = sit.ReadImage(FemoralHeadL)
        FemoralHeadR_ = sit.ReadImage(FemoralHeadR)
        PCTV_ = sit.ReadImage(PCTV)
        Smallintestine_ = sit.ReadImage(Smallintestine)
        Rectum_ = sit.ReadImage(Rectum)

        origin_np = sit.GetArrayFromImage(origin_)
        rd_np = sit.GetArrayFromImage(rd_)
        Bladder_np = sit.GetArrayFromImage(Bladder_)
        FemoralHeadL_np = sit.GetArrayFromImage(FemoralHeadL_)
        FemoralHeadR_np = sit.GetArrayFromImage(FemoralHeadR_)
        PCTV_np = sit.GetArrayFromImage(PCTV_)
        Smallintestine_np = sit.GetArrayFromImage(Smallintestine_)
        Rectum_np = sit.GetArrayFromImage(Rectum_)

        if origin_np.max() != 0:
            origin_np = (origin_np - origin_np.min()) / (origin_np.max() - origin_np.min())
            rd_np = (rd_np - rd_np.min()) / (rd_np.max() - rd_np.min())
        else:
            origin_np = (origin_np - origin_np.min()) / (origin_np.max() + 1 - origin_np.min())
            rd_np = (rd_np - rd_np.min()) / (rd_np.max() + 1 - rd_np.min())

        c = min(origin_np.shape[0], rd_np.shape[0], Bladder_np.shape[0], FemoralHeadL_np.shape[0],
                FemoralHeadR_np.shape[0], PCTV_np.shape[0], Smallintestine_np.shape[0], Rectum_np.shape[0])

        inputs = np.zeros(shape=(c, 7, 512, 512))
        for i in range(c):
            channel = origin_np[np.newaxis, i, :, :]
            channel = np.append(channel, Bladder_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, FemoralHeadL_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, FemoralHeadR_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, PCTV_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, Smallintestine_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, Rectum_np[np.newaxis, i, :, :], axis=0)

            inputs[i, :, :, :] = channel[:, :, :]

        rd = rd_np[:c, np.newaxis, :, :]
        inputs = self.transform(inputs).type(torch.FloatTensor)
        rd = self.transform(rd).type(torch.FloatTensor)

        return {'inputs': inputs, 'rd': rd, 'channel': c, 'name': image_path}

    def __len__(self):
        return len(self.names)


class TestDataset():
    def __init__(self, dir, phase):
        super(TestDataset, self).__init__()
        self.dir = dir
        self.phase = phase
        self.origin, self.rd, self.rs = sorted(make_dataset(self.dir, self.phase))
        self.names, self.images = make_files(self.origin, self.rs, self.rd)

        self.transform = torch.from_numpy

    def __getitem__(self, index):
        image_path = self.names[index]
        images = self.images[image_path]

        origin = images[0]
        rd = images[-1]
        Bladder = images[1]
        FemoralHeadL = images[2]
        FemoralHeadR = images[3]
        PCTV = images[4]
        Smallintestine = images[5]
        Rectum = images[6]

        origin_ = sit.ReadImage(origin)
        rd_ = sit.ReadImage(rd)
        Bladder_ = sit.ReadImage(Bladder)
        FemoralHeadL_ = sit.ReadImage(FemoralHeadL)
        FemoralHeadR_ = sit.ReadImage(FemoralHeadR)
        PCTV_ = sit.ReadImage(PCTV)
        Smallintestine_ = sit.ReadImage(Smallintestine)
        Rectum_ = sit.ReadImage(Rectum)

        origin_np = sit.GetArrayFromImage(origin_)
        rd_np = sit.GetArrayFromImage(rd_)
        Bladder_np = sit.GetArrayFromImage(Bladder_)
        FemoralHeadL_np = sit.GetArrayFromImage(FemoralHeadL_)
        FemoralHeadR_np = sit.GetArrayFromImage(FemoralHeadR_)
        PCTV_np = sit.GetArrayFromImage(PCTV_)
        Smallintestine_np = sit.GetArrayFromImage(Smallintestine_)
        Rectum_np = sit.GetArrayFromImage(Rectum_)

        if origin_np.max() != 0:
            origin_np = (origin_np - origin_np.min()) / (origin_np.max() - origin_np.min())
            rd_np = (rd_np - rd_np.min()) / (rd_np.max() - rd_np.min())
        else:
            origin_np = (origin_np - origin_np.min()) / (origin_np.max() + 1 - origin_np.min())
            rd_np = (rd_np - rd_np.min()) / (rd_np.max() + 1 - rd_np.min())

        c = min(origin_np.shape[0], rd_np.shape[0], Bladder_np.shape[0], FemoralHeadL_np.shape[0],
                FemoralHeadR_np.shape[0], PCTV_np.shape[0], Smallintestine_np.shape[0], Rectum_np.shape[0])

        inputs = np.zeros(shape=(c, 7, 512, 512))
        for i in range(c):
            channel = origin_np[np.newaxis, i, :, :]
            channel = np.append(channel, Bladder_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, FemoralHeadL_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, FemoralHeadR_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, PCTV_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, Smallintestine_np[np.newaxis, i, :, :], axis=0)
            channel = np.append(channel, Rectum_np[np.newaxis, i, :, :], axis=0)

            inputs[i, :, :, :] = channel[:, :, :]

        rd = rd_np[:c, np.newaxis, :, :]
        inputs = self.transform(inputs).type(torch.FloatTensor)
        rd = self.transform(rd).type(torch.FloatTensor)

        return {'inputs': inputs, 'rd': rd, 'channel': c, 'name': image_path}

    def __len__(self):
        return len(self.names)


def make_datasetS():
    dir = r'E:\dataset\cervical_cancer\CrossValidation\5'
    batch_size = 1
    Syn_train = TrainDataset(dir, "train")
    Syn_test = TestDataset(dir, "test")
    SynData_train = DataLoader(dataset=Syn_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    SynData_test = DataLoader(dataset=Syn_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return SynData_train, SynData_test


if __name__ == '__main__':
    tra, _ = make_datasetS()
    for ii, batch_sample in enumerate(tra):
        inputs, target, c, name = batch_sample['inputs'], batch_sample['rd'], batch_sample['channel'], batch_sample[
            'name']
