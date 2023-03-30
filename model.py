import torch
from PIL import Image
import os
import json
from tqdm import tqdm
import cn_clip.clip as clip
from cn_clip.clip import available_models

print("Available models:", available_models())


# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']


class Chinese_CLIP:
    def __init__(self) -> None:
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.splits = ["valid"]
        self.model, self.preprocess = clip.load_from_name("RN50", device=self.device, download_root="./")
        self.model.eval()

    def extract_image_features(self, img_name):  # for upload
        image_data = Image.open(img_name).convert("RGB")
        infer_data = self.preprocess(image_data)
        infer_data = infer_data.unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(infer_data)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]  # [1, 1024]

    def extract_text_features(self, text):  # for search

        text_data = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_data)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]  # [1, 1024]

    def build_Flickr30kCN_image_db(self, data_path):  # for load
        image_ids = list()
        image_feats = list()
        for split in self.splits:
            image_path = os.path.join(data_path, split + "_imgs.img_feat.jsonl")
            if not os.path.isfile(image_path):
                print("error, file {} is not exist.".format(image_path))
                continue
            with open(image_path, "r") as fin:
                for line in tqdm(fin, desc="build image {} part: ".format(split)):
                    obj = json.loads(line.strip())
                    image_ids.append(obj['image_id'])
                    image_feats.append(obj['feature'])
        return image_ids, image_feats

    def load_images(self, data_path):  # for mysql
        image_dicts = dict()
        for split in self.splits:
            file_path = os.path.join(data_path, split + "_imgs.tsv")
            if not os.path.isfile(file_path):
                print("error, file {} is not exist.".format(file_path))
            with open(file_path, "r") as fin_imgs:
                for line in tqdm(fin_imgs):
                    line = line.strip()
                    image_id, b64 = line.split("\t")
                    image_dicts[image_id] = b64.encode()
        return image_dicts
