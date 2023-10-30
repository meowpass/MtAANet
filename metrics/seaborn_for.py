import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

img1 = r'F:\gongjinai\Ablation experiments\liujinhua_rd.png'
img2 = r'F:\gongjinai\Ablation experiments\Deeplabv3_plus\index\liujinhua\liujinhua_.png'
# img3 = r'F:\gongjinai\test\liyuying_0.87_P3\liyuying_differmap.png'
sava_path = r'F:\gongjinai\Ablation experiments\Deeplabv3_plus\index\liujinhua\jet_liujinhua_.png'
img1 = np.array(Image.open(img1).convert('L'))
img1 = np.squeeze(img1).astype(float)

img2 = np.array(Image.open(img2).convert('L'))
img2 = np.squeeze(img2).astype(float)
img = img1 - img2
hotmap1 = np.abs(img)

# img3 = np.array(Image.open(img3).convert('L'))
# img3 = (img3 - img3.min()) / (img3.max() - img3.min()) * 255.0
# img3 = np.squeeze(img3).astype(float)
# hotmap2 = img3

# pal = sns.dark_palette('')
print(hotmap1.shape)
# plt.figure(figsize=(512, 512))
# heatmap = sns.heatmap(hotmap1,cmap='Blues',xticklabels=False, yticklabels=False,vmin= 0,vmax=90)
heatmap = sns.heatmap(hotmap1, cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=100)
plt.savefig(sava_path)
plt.show()
