import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
from clip import clip
import nltk
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.cuda.device(1)

nltk.data.path.append("/home/worker/nlp/hao/nltk_data")

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 加载 BigGAN 模型
biggan = BigGAN.from_pretrained('biggan-deep-256')
# 设置设备
model.to(device)
biggan.to(device)

# 定义文本提示
text = "a landscape with mountains and a lake"

# 指定支持的ImageNet类别名称
class_name = "mountain"

# 将类别名称转换为对应的One-Hot编码
class_vector = one_hot_from_names([class_name], batch_size=1)

# 预处理文本提示
text_encoded = model.encode_text(clip.tokenize(text).to(device)).float()

# 从文本提示生成图像
with torch.no_grad():
    noise_vector = truncated_noise_sample(truncation=0.6, batch_size=1, seed=42)
    print(class_vector, '----')
    truncation = 0.6
    noise_vector = torch.from_numpy(noise_vector).to(device)
    class_vector = torch.from_numpy(class_vector).to(device)
    output = biggan(noise_vector, class_vector, truncation)

# 后处理生成的图像
output = ((output + 1.0) / 2.0)
output = output.clamp(0, 1)
output = output[0].cpu()

# 保存生成的图像
save_image(output, "generated_image.png")
