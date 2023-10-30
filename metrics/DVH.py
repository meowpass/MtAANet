import os
import math
import numpy as np
# import dicom
from collections import Counter
import matplotlib.pyplot as plt
import SimpleITK as sitk

source_path = r'F:\gongjinai\test\origin'
p_name = []
for root, _, files in os.walk(source_path):
    for file in files:
        name = file.split('_')[0]
        p_name.append(name)
sorted(p_name)
# datapath = r'E:\Code\DosePrediction\doseResult'
datapath = r'F:\gongjinai\Ablation experiments\final\results'  # rd的来源路径(GT Pre)
# datapath = r'F:\gongjinai\test2d\U_net'
# rsdatapath = r'F:\gongjinai\test\rs'  # rs的来源路径
# patient_name = 'zhouyanhua'
# datapath = r'F:\gongjinai\test\results'  # rd的来源路径(GT Pre)
rsdatapath = r'F:\gongjinai\test\rs'  # rs的来源路径
save_path = r'F:\gongjinai\Ablation experiments\final\index'
for i, patient_name in enumerate(p_name):
    path = os.path.join(save_path, patient_name)
    save_metrics = patient_name + '\\' + '量化指标.txt'
    f = open(os.path.join(save_path, save_metrics), 'w')
    GT = os.path.join(datapath, patient_name + '_ReaRD.mha')
    prediction = os.path.join(datapath, patient_name + '_PreRD.mha')
    OARs_1 = os.path.join(rsdatapath, patient_name + '_Bladder_rs.mha')
    OARs_2 = os.path.join(rsdatapath, patient_name + '_FemoralHeadL_rs.mha')
    OARs_3 = os.path.join(rsdatapath, patient_name + '_FemoralHeadR_rs.mha')
    OARs_4 = os.path.join(rsdatapath, patient_name + '_PTV_rs.mha')
    OARs_5 = os.path.join(rsdatapath, patient_name + '_Smallintestine_rs.mha')
    OARs_6 = os.path.join(rsdatapath, patient_name + '_Rectum_rs.mha')

    OARs_list = []
    OARs_list.append(OARs_1)
    OARs_list.append(OARs_2)
    OARs_list.append(OARs_3)
    OARs_list.append(OARs_4)
    OARs_list.append(OARs_5)
    OARs_list.append(OARs_6)

    max_dose = 5040 * 1.1
    GT_itk = sitk.ReadImage(GT)
    GT_array = sitk.GetArrayFromImage(GT_itk)
    # GT_array = GT_array*65535/10 ###
    GT_array = GT_array * max_dose / 100

    prediction_itk = sitk.ReadImage(prediction)
    prediction_array = sitk.GetArrayFromImage(prediction_itk)
    # prediction_array = prediction_array/10 ###
    prediction_array = prediction_array * max_dose / 100

    plt.title('DVH')

    d98_y = []
    d98_x = []
    d95_y = []
    d95_x = []
    d2_y = []
    d2_x = []
    d50_y = []
    d50_x = []
    HI_ROI_list = []
    d98_y_rea = []
    d98_x_rea = []
    d95_y_rea = []
    d95_x_rea = []
    d2_y_rea = []
    d2_x_rea = []
    d50_y_rea = []
    d50_x_rea = []
    HI_ROI_list_rea = []
    for i in range(6):
        OARs = OARs_list[i]
        # print(OARs)
        OARs_itk = sitk.ReadImage(OARs)
        OARs_array = sitk.GetArrayFromImage(OARs_itk)
        # if OARs_array.shape[1] == 512:
        #     OARs_array = down_sample(OARs_array)
        OARs_num = np.count_nonzero(OARs_array)

        GT_intersect_OARs = GT_array * OARs_array
        pre_intersect_OARs = prediction_array * OARs_array

        GT_max_dose = np.max(GT_array)
        # print(GT_max_dose)
        x = np.linspace(0, int(GT_max_dose), 500)
        # print(x)
        y1 = []
        for j in range(len(x)):
            y1.append(np.count_nonzero(GT_intersect_OARs >= x[j]) / OARs_num)
        y1[0] = 1.0

        pre_max_dose = np.max(prediction_array)
        y2 = []
        for j in range(len(x)):
            y2.append(np.count_nonzero(pre_intersect_OARs >= x[j]) / OARs_num)
        y2[0] = 1.0
        # INCREASE1 = [0.00] * 10
        # INCREASE2 = [0.01] * 10
        # INCREASE3 = [0.002] * 20
        # INCREASE4 = [0.002] * 20
        # INCREASE5 = [0.004] * 30
        # INCREASE6 = [0.02] * 420
        # INCREASE = INCREASE1 + INCREASE2 + INCREASE3 + INCREASE4 + INCREASE5 + INCREASE6
        # y3 = [0] * 500
        # for u in range(len(x)):
        #     y3[u] += y1[u] - INCREASE[u]
        d98_flag = False
        d2_flag = False
        d50_flag = False
        for j in range(len(x) - 1):
            if y2[j] > 0.98 and y2[j + 1] <= 0.98:
                d98_index = j + 1
                d98_flag = True
                # d98_y.append(y2[j+1])
                # d98_x.append(x[j+1])
                break
        for j in range(len(x) - 1):
            if y2[j] > 0.95 and y2[j + 1] <= 0.95:
                d95_index = j + 1
                d95_flag = True
                break
        for j in range(len(x) - 1):
            if y2[j] > 0.02 and y2[j + 1] <= 0.02:
                d2_index = j + 1
                d2_flag = True
                # d2_y.append(y2[j + 1])
                # d2_x.append(x[j + 1])
                break
        for j in range(len(x) - 1):
            if y2[j] > 0.5 and y2[j + 1] <= 0.5:
                d50_index = j + 1
                d50_flag = True
                # d50_y.append(y2[j + 1])
                # d50_x.append(x[j + 1])
                break
        if d98_flag and d2_flag and d50_flag and d95_flag:
            d98_y.append(y2[d98_index])
            d98_x.append(x[d98_index])
            d95_y.append(y2[d95_index])
            d95_x.append(x[d95_index])
            d2_y.append(y2[d2_index])
            d2_x.append(x[d2_index])
            d50_y.append(y2[d50_index])
            d50_x.append(x[d50_index])
            HI_ROI_list.append(os.path.basename(OARs_list[i]))
            # d2_x.append(x[j + 1])
            # d98 = [d for d in y2 if d < 0.981 and d > 0.979]
        # print(d98)
        d98_flag = False
        d2_flag = False
        d50_flag = False
        d95_flag = False
        for j in range(len(x) - 1):
            if y1[j] > 0.98 and y1[j + 1] <= 0.98:
                d98_index = j + 1
                d98_flag = True
                # d98_y.append(y2[j+1])
                # d98_x.append(x[j+1])
                break
        for j in range(len(x) - 1):
            if y1[j] > 0.02 and y1[j + 1] <= 0.02:
                d2_index = j + 1
                d2_flag = True
                # d2_y.append(y2[j + 1])
                # d2_x.append(x[j + 1])
                break
        for j in range(len(x) - 1):
            if y1[j] > 0.5 and y1[j + 1] <= 0.5:
                d50_index = j + 1
                d50_flag = True
                # d50_y.append(y2[j + 1])
                # d50_x.append(x[j + 1])
                break
        if d98_flag and d2_flag and d50_flag:
            d98_y_rea.append(y1[d98_index])
            d98_x_rea.append(x[d98_index])
            d95_y_rea.append(y1[d95_index])
            d95_x_rea.append(x[d95_index])
            d2_y_rea.append(y1[d2_index])
            d2_x_rea.append(x[d2_index])
            d50_y_rea.append(y1[d50_index])
            d50_x_rea.append(x[d50_index])
            HI_ROI_list_rea.append(os.path.basename(OARs_list[i]))
        # 画图
        plt.xlabel('Gy', fontweight='bold')
        plt.ylabel('Volume%', fontweight='bold')
        if i == 0:
            plt.plot(x, y1, color='cyan', linewidth=1.0, linestyle='-', label='Bladder')
            plt.plot(x, y2, color='cyan', linewidth=1.0, linestyle='--')
            plt.xlim(x.min(), x.max())
            plt.ylim(0, 1.005)
            plt.legend(bbox_to_anchor=(1, 0), loc='best', borderaxespad=0)
            # plt.show()
        elif i == 1:
            plt.plot(x, y1, color='blue', linewidth=1.0, linestyle='-', label='FHL')
            plt.plot(x, y2, color='blue', linewidth=1.0, linestyle='--')
            plt.legend()
            # plt.show()
        elif i == 2:
            plt.plot(x, y1, color='green', linewidth=1.0, linestyle='-', label='FHR')
            plt.plot(x, y2, color='green', linewidth=1.0, linestyle='--')
            # plt.legend()
            plt.legend()
            # plt.show()
        elif i == 3:
            plt.plot(x, y1, color='red', linewidth=1.0, linestyle='-', label='PTV')
            plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--')
            # plt.legend()
            plt.legend()
            # plt.show()
        elif i == 4:
            plt.plot(x, y1, color='purple', linewidth=1.0, linestyle='-', label='ST')
            plt.plot(x, y2, color='purple', linewidth=1.0, linestyle='--')
            # plt.legend()
            plt.legend()
        else:
            plt.plot(x, y1, color='black', linewidth=1.0, linestyle='-', label='Rectum')
            plt.plot(x, y2, color='black', linewidth=1.0, linestyle='--')
            # plt.legend()
            plt.legend()
            # plt.show()
    plt.savefig(os.path.join(path, patient_name))
    plt.show()

    print("Pre:", file=f)
    print(len(d98_x), file=f)
    print("d98", d98_x, file=f)  # 42.765531062124246
    # print(d98_y) #0.9785376146110867  约等于0.98   ***大于等于0.98体积的剂量是42
    print("d95", d95_x, file=f)
    # print(d95_y)
    print("d50", d50_x, file=f)  # 51.25250501002004
    # print(d50_y) #0.49560370518450325
    print("d2", d2_x, file=f)  # 53.677354709418836
    # print(d2_y) #0.017279437704317675

    print("Rea:", file=f)
    print(len(d98_x_rea), file=f)
    print("d98", d98_x_rea, file=f)
    # print(d98_y)
    print("d95", d95_x_rea, file=f)
    # print(d95_y)
    print("d50", d50_x_rea, file=f)
    # print(d50_y)
    print("d2", d2_x_rea, file=f)
    # print(d2_y)
    print("----------------", file=f)
    for i in range(len(HI_ROI_list_rea)):
        print("---------------", HI_ROI_list_rea[i], "---------------", file=f)
        print('diff_percent_d98:', (d98_x_rea[i] - d98_x[i]) / 5544, file=f)
        print('diff_percent_d95:', (d95_x_rea[i] - d95_x[i]) / 5544, file=f)
        print('diff_percent_d50:', (d50_x_rea[i] - d50_x[i]) / 5544, file=f)
        print('diff_percent_d2:', (d2_x_rea[i] - d2_x[i]) / 5544, file=f)
    print("----------------", file=f)

    print('HI_Pre:', file=f)
    # print(HI_ROI_list)
    for i in range(len(HI_ROI_list)):
        print(HI_ROI_list[i], ':', (d2_x[i] - d98_x[i]) / d50_x[i], file=f)
    print('HI_Rea:', file=f)
    # print(HI_ROI_list_rea)
    for i in range(len(HI_ROI_list_rea)):
        print(HI_ROI_list[i], ':', (d2_x_rea[i] - d98_x_rea[i]) / d50_x_rea[i], file=f)

    # CI
    PTV_itk = sitk.ReadImage(OARs_4)
    PTV_array = sitk.GetArrayFromImage(PTV_itk)
    # if PTV_array.shape[1] == 512:
    #     PTV_array = down_sample(PTV_array)
    OARs_num = np.count_nonzero(PTV_array)
    A = OARs_num
    B = np.count_nonzero(prediction_array > (5040 / 115))
    A_intersect_B = np.count_nonzero(PTV_array * (prediction_array > (5040 / 115)))
    # B = np.count_nonzero(prediction_array > ((5040 * 65535) / 6622))
    # A_intersect_B = np.count_nonzero(PTV_array * (prediction_array > ((5040 * 65535) / 6622)))
    # print(A_intersect_B)
    CI = A_intersect_B * A_intersect_B / (A * B)
    print('Pre_CI:', CI, file=f)
    B_ = np.count_nonzero(GT_array > (5040 / 115))
    A_intersect_B_ = np.count_nonzero(PTV_array * (GT_array > (5040 / 115)))
    CI_ = A_intersect_B_ * A_intersect_B_ / (A * B_)
    print('Rea_CI:', CI_, file=f)
    print('done for one')
    for oar in OARs_list:
        # print(oar)
        OARs_itk = sitk.ReadImage(oar)
        OARs_array = sitk.GetArrayFromImage(OARs_itk)
        OARs_num = np.count_nonzero(OARs_array)

        GT_intersect_OARs = GT_array * OARs_array
        pre_intersect_OARs = prediction_array * OARs_array
        # print(GT_intersect_OARs.shape) #(142, 512, 512)

        GT_max_dose = np.max(GT_array)
        # print(GT_max_dose) #55.44
        x = np.linspace(0, int(GT_max_dose), 200)
        # print(x)
        # print(len(x)) #500
        y1 = []
        for j in range(len(x)):
            y1.append(np.count_nonzero(GT_intersect_OARs >= x[j]) / OARs_num)
        y1[0] = 1.0

        pre_max_dose = np.max(prediction_array)
        xx = np.linspace(0, int(pre_max_dose), 200)
        y2 = []
        for j in range(len(x)):
            y2.append(np.count_nonzero(pre_intersect_OARs >= xx[j]) / OARs_num)
        y2[0] = 1.0
        # print(pre_max_dose) #54.53903311729431
        # print(xx)
        print("------------", oar, "------------", file=f)
        count = 0
        for i in range(len(x)):
            while count <= x[i] < (count + 1) and count < 50:
                # print(i)
                print("V", count + 1, "-Rea:", y1[i], file=f)
                print("V", count + 1, "-Pre:", y2[i], file=f)
                # print("V40-diff-percent", (y1[i] - y2[i]) / y1[i])
                count += 1
                break
        # for i in range(len(x)):
        #     if 50 <= x[i] < 51:
        #         # print(i)
        #         print("V50-Rea", y1[i], file=f)
        #         # print(y1)
        #         print("V50-Pre", y2[i], file=f)
        #         # print("V50-diff-percent", (y1[i] - y2[i]) / y1[i])
        #         break

        Dmean1 = GT_intersect_OARs.sum() / OARs_num
        Dmean2 = pre_intersect_OARs.sum() / OARs_num
        print("Dmean-Rea", Dmean1, file=f)
        print("Dmean-Pre", Dmean2, file=f)
        Dmax1 = GT_intersect_OARs.max()
        Dmax2 = pre_intersect_OARs.max()
        print("Dmax-Rea", Dmax1, file=f)
        print("Dmax-Pre", Dmax2, file=f)
        # print("Dmean-diff-percent", (Dmean1 - Dmean2) / Dmean1)
        print("----------", file=f)
    print('done for one:', patient_name)
