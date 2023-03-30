import torch
from clip import clip
import diffusion

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 加载 Diffusion 模型
diffusion_model = diffusion.Denoise(diffusion.Unet(num_channels=256,
                                                   num_res_blocks=2,
                                                   channels_multiplier=2),
                                    diffusion.LaplaceLogProb(),
                                    diffusion.TimestepSchedule(1000, 0.5, 50000),
                                    diffusion.NoiseSchedule("linear", 0.01, 0.05, 2000),
                                    diffusion.Optimizer(0.01),
                                    "cuda")


# 定义生成图片的函数
def generate_image(text):
    # 对文本进行编码
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).float()

    # 定义 Diffusion 生成图片的参数
    clip_guidance_scale = 1000
    clip_guidance_denoise_scale = 0.01
    tv_scale = 150
    init_scale = 0.01
    num_timesteps = 1000

    # 生成图片
    z = torch.randn(1, 256, 256, device=device)
    image = diffusion_model.generate_with_ema(z,
                                              text_features,
                                              clip_guidance_scale=clip_guidance_scale,
                                              clip_guidance_denoise_scale=clip_guidance_denoise_scale,
                                              tv_scale=tv_scale,
                                              init_scale=init_scale,
                                              num_timesteps=num_timesteps,
                                              progress=True)
    # 将生成的图片进行后处理
    image = image.clamp(0, 1)
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = preprocess.deprocess(image)
    return image


# 测试生成图片的函数
text = "a red apple with a green leaf"
image = generate_image(text)
# 显示生成的图片
import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()
