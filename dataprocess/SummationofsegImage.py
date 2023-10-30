import numpy as np
import SimpleITK as sit
import glob
import os
import time

source_path = r'E:\dataset\cervical_cancer\train_rs_zancun'
save_path = r'E:\dataset\cervical_cancer\rs_summation4train'

names = []
PTV = []
bladder = []
femoralHeadL = []
femoralHeadR = []
smallIntestine = []
rectum = []

for root, _, files in os.walk(source_path):
    for file in files:
        name = file.split('_')[0]
        if name not in names:
            names.append(name)
        if file.split('_')[1] == 'PTV':
            PTV.append(os.path.join(source_path, file))
        elif file.split('_')[1] == 'Bladder':
            bladder.append(os.path.join(source_path, file))
        elif file.split('_')[1] == 'FemoralHeadL':
            femoralHeadL.append(os.path.join(source_path, file))
        elif file.split('_')[1] == 'FemoralHeadR':
            femoralHeadR.append(os.path.join(source_path, file))
        elif file.split('_')[1] == 'Smallintestine':
            smallIntestine.append(os.path.join(source_path, file))
        else:
            rectum.append(os.path.join(source_path, file))

for i in range(len(names)):
    start_time = time.time()
    name = names[i]
    ptv = sit.ReadImage(PTV[i])
    ptv = sit.GetArrayFromImage(ptv)
    Bladder = sit.ReadImage(bladder[i])
    Bladder = sit.GetArrayFromImage(Bladder)
    FemoralHeadL = sit.ReadImage(femoralHeadL[i])
    FemoralHeadL = sit.GetArrayFromImage(FemoralHeadL)
    FemoralHeadR = sit.ReadImage(femoralHeadR[i])
    FemoralHeadR = sit.GetArrayFromImage(FemoralHeadR)
    SmallIntestine = sit.ReadImage(smallIntestine[i])
    SmallIntestine = sit.GetArrayFromImage(SmallIntestine)
    Rectum = sit.ReadImage(rectum[i])
    Rectum = sit.GetArrayFromImage(Rectum)

    for j in range(ptv.shape[0]):
        for k in range(ptv.shape[1]):
            for l in range(ptv.shape[2]):
                ptv[j][k][l] = ptv[j][k][l] * 1000 + Bladder[j][k][l] + FemoralHeadL[j][k][l] + FemoralHeadR[j][k][l] + \
                               SmallIntestine[j][k][l] + Rectum[j][k][l]
                if ptv[j][k][l] >= 1000:
                    ptv[j][k][l] = 255
                elif 0 < ptv[j][k][l] < 1000:
                    ptv[j][k][l] = 128
                else:
                    ptv[j][k][l] = 0

    summation = sit.GetImageFromArray(ptv)
    sit.WriteImage(summation, save_path + '\\' + name + '_rs.mha')
    end_time = time.time()
    print('time consuming:', end_time - start_time)
