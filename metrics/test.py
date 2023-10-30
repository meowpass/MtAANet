import os

source_path = r'F:\gongjinai\test\origin'
save_path = r'F:\gongjinai\Ablation experiments\final\index'
p_name = []
for root, _, files in os.walk(source_path):
    for file in files:
        name = file.split('_')[0]
        p_name.append(name)
        if not os.path.exists(os.path.join(save_path, name)):
            os.makedirs(os.path.join(save_path, name))
sorted(p_name)
for i, patient_name in enumerate(p_name):
    save_metrics = patient_name + '\\' + '量化指标.txt'
    f = open(os.path.join(save_path, save_metrics), 'w')
