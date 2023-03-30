import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model, preprocess = load_from_name("RN50", device=device, download_root="./")

model.eval()

# image preprocess

image_data = Image.open("examples/22.png")
infer_data = preprocess(image_data).unsqueeze(0).to(device)

# text data
text_data = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(infer_data)
    text_features = model.encode_text(text_data)

    # # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_per_image, logits_per_text = model.get_similarity(infer_data, text_data)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
