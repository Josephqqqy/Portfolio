import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# 定义图像目录和样本数量
gen_dir = "./generated_images_cosine"
real_dir = "./processed_real_images"
n = 30  # 样本数量

# 根据命名规则构造文件列表
gen_files = [os.path.join(gen_dir, f"gen_{i}.jpg") for i in range(n)]
real_files = [os.path.join(real_dir, f"real_{i}.png") for i in range(n)]

# 定义转换，将 PIL.Image 转为 tensor（数值范围 [0,1]）
to_tensor = transforms.ToTensor()

mse_total = 0.0

for gen_path, real_path in zip(gen_files, real_files):
    # 打开图片并转换为 RGB 格式
    gen_img = Image.open(gen_path).convert("RGB")
    real_img = Image.open(real_path).convert("RGB")
    
    # 转换为 tensor
    gen_tensor = to_tensor(gen_img)
    real_tensor = to_tensor(real_img)
    
    # 计算均方误差 (MSE)，取平均值
    mse = F.mse_loss(gen_tensor, real_tensor, reduction="mean")
    mse_total += mse.item()

average_mse = mse_total / n
print("Average MSE:", average_mse)
