import sys
import logging
from milvus import Milvus
from model import Chinese_CLIP
import json

DEFAULT_TABLE = 'xxm'


# Get the vector of images
def extract_features(data_path, model):
    try:
        image_ids, image_feats = model.build_Flickr30kCN_image_db(data_path)
        print(f"Extracting feature from {len(image_ids)} images in total")

        return image_ids, image_feats
    except Exception as e:
        logging.error(f"Error with extracting feature from image {e}")
        sys.exit(1)


# Combine the id of the vector and the name of the image into a list
def format_data(ids, image_ids, image_dicts):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), str(image_ids[i]), image_dicts[str(image_ids[i])])
        data.append(value)
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(table_name, image_dir, model, milvus_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    # 利用模型提取图像特征得到特征向量
    image_ids, vectors = extract_features(image_dir, model)

    ids = milvus_client.insert(table_name, vectors)
    milvus_client.create_index(table_name)

    image_dicts = model.load_images(image_dir)

    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, image_ids, image_dicts))
    return len(ids)


def do_search(table_name, text_content, top_k, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        features = model.extract_text_features(text_content)
        vectors = milvus_client.search_vectors(table_name, [features], top_k)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return paths, distances
    except Exception as e:
        logging.error(f"Error with search : {e}")
        sys.exit(1)


def do_search_by_text(table_name, text_content, top_k, model, milvus_client):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        features = model.extract_text_features(text_content).tolist()
        status, vectors = milvus_client.search(table_name, top_k, [features])
        vids = [str(x.id) for x in vectors[0]]
        distances = [x.distance for x in vectors[0]]
        print(vids)
        print(distances)
        return vids
    except Exception as e:
        logging.error(f"Error with search : {e}")
        sys.exit(1)


if __name__ == '__main__':
    filename = r'/home/worker/nlp/hao/clip_test/cache.txt'
    with open(filename, 'r', encoding='utf-8') as data:
        lines = data.readlines()
        for line in lines:
            line = line.strip()
            json_data = json.loads(line)
    clip2 = Chinese_CLIP()
    milvus = Milvus(host='172.16.19.81', port='19530')
    text = '一只猫'
    results = do_search_by_text('xxm', text, 3, clip2, milvus)
    print('和', text, '相关的图片有：')
    for result in results:
        img_name = json_data[result]
        print(img_name)
