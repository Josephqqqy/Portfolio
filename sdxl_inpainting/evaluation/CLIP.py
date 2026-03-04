import torch
from PIL import Image
import clip  # 请先安装 clip： pip install git+https://github.com/openai/CLIP.git

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型和预处理工具
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义文本提示（可以替换成你希望评估的文本）
text = clip.tokenize(["a kitchen with a large island and a large island"]).to(device)

# 加载并预处理图像（确保图像是 RGB 格式）
image = preprocess(Image.open("repaired_image_cosine.png").convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    # 编码图像和文本，获得特征向量
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 对特征向量归一化（使模长为1）
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度（CLIP Score）
    clip_score = (image_features @ text_features.T).item()

print("CLIP Score:", clip_score)