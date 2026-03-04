import os
from PIL import Image
import torch
import lpips
import torchvision.transforms as transforms

# 定义图像目录和样本数量
gen_dir = "./generated_images_cosine"
real_dir = "./processed_real_images"
n = 30  # 样本数量

# 根据命名规则构造文件列表
gen_files = [os.path.join(gen_dir, f"gen_{i}.jpg") for i in range(n)]
real_files = [os.path.join(real_dir, f"real_{i}.png") for i in range(n)]
# 初始化 LPIPS 模型（选择 'alex' 作为特征提取网络）
loss_fn = lpips.LPIPS(net='alex')

# 定义预处理：统一调整尺寸并转换为 tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 此处统一尺寸可以根据需要调整
    transforms.ToTensor(),
])

lpips_total = 0.0
n = len(gen_files)

for gen_path, real_path in zip(gen_files, real_files):
    # 打开图片并转换为 RGB 格式
    img_gen = Image.open(gen_path).convert("RGB")
    img_real = Image.open(real_path).convert("RGB")
    
    # 预处理图片
    img_gen_tensor = transform(img_gen).unsqueeze(0)  # [1, C, H, W]
    img_real_tensor = transform(img_real).unsqueeze(0)
    
    # LPIPS 要求输入范围为 [-1, 1]，因此需要归一化
    img_gen_tensor = img_gen_tensor * 2 - 1
    img_real_tensor = img_real_tensor * 2 - 1
    
    # 计算两张图片之间的 LPIPS 距离
    lpips_value = loss_fn(img_gen_tensor, img_real_tensor)
    lpips_total += lpips_value.item()

average_lpips = lpips_total / n
print("Average LPIPS:", average_lpips)