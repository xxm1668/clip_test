from clip_benchmark.datasets.builder import build_dataset
import pandas as pd
import os

root_path = "/home/worker/nlp/hao/clip_test/coco/data"  # set this to smth meaningful
ds = build_dataset("mscoco_captions", root=root_path,
                   split="train")  # this downloads the dataset if it is not there already
coco = ds.coco
imgs = coco.loadImgs(coco.getImgIds())
future_df = {"filepath": [], "title": []}
for img in imgs:
    caps = coco.imgToAnns[img["id"]]
    for cap in caps:
        future_df["filepath"].append(img["file_name"])
        future_df["title"].append(cap["caption"])
pd.DataFrame.from_dict(future_df).to_csv(
    os.path.join(root_path, "train2014.csv"), index=False, sep="\t"
)
