import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from clip import clip
import os
from stylegan2_pytorch import StyleGAN2Generator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 加载 StyleGAN2 模型
stylegan = StyleGAN2Generator(size=1024, style_dim=512)

# 设置设备
model.to(device)
stylegan.to(device)

# 定义文本提示
text = "a landscape with mountains and a lake"

# 指定支持的ImageNet类别名称
class_name = "mountain"

# 预处理文本提示
text_encoded = model.encode_text(clip.tokenize(text).to(device)).float()

# 将类别名称转换为对应的随机噪声种子向量
np.random.seed(42)
latent = np.random.randn(1, 512).astype(np.float32)

# 从随机噪声种子向量生成图像
with torch.no_grad():
    latent = torch.from_numpy(latent).to(device)
    output = stylegan(latent, truncation=0.7)

# 后处理生成的图像
output = ((output + 1.0) / 2.0)
output = output.clamp(0, 1)
output = output[0].cpu()

# 保存生成的图像
save_image(output, "generated_image.png")
