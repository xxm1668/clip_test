import sys
from model import Chinese_CLIP
from milvus import Milvus, IndexType, MetricType
import logging
import os
import json

VECTOR_DIMENSION = 1024


class mycollection:
    def __init__(self):
        self.milvus = Milvus(host='172.16.19.81', port='19530')

    def create_collection(self, collection_name):  # 创建collection对象
        # Create milvus collection if not exists
        try:
            if self.milvus.has_collection(collection_name):
                param = {'collection_name': collection_name, 'dimension': VECTOR_DIMENSION, 'index_file_size': 1024,
                         'metric_type': MetricType.L2}
                self.milvus.create_collection(param)
                logging.debug('创建表成功！！！')
            else:
                self.milvus.drop_collection(collection_name)
                param = {'collection_name': collection_name, 'dimension': VECTOR_DIMENSION, 'index_file_size': 1024,
                         'metric_type': MetricType.L2}
                self.milvus.create_collection(param)
                logging.debug('删除并创建表成功！！！')
            return "OK"
        except Exception as e:
            logging.error(f"Failed to create collection to Milvus: {e}")
            sys.exit(1)


if __name__ == '__main__':
    img_num = {}
    target_filename = r'/home/worker/nlp/hao/clip_test/cache.txt'
    collection1 = mycollection()
    collection1.create_collection('xxm')
    img_name = '/home/worker/nlp/hao/clip_test/examples/22.png'
    clip2 = Chinese_CLIP()
    count = 1
    img_vector = clip2.extract_image_features(img_name).tolist()
    result, id = collection1.milvus.insert('xxm', [img_vector], [count])
    path = r'/home/worker/nlp/hao/clip_test/images'
    count += 1
    for filename in os.listdir(path):
        filename2 = filename
        filename = path + '/' + filename
        try:
            img_vector = clip2.extract_image_features(filename).tolist()
        except:
            continue
        result, id = collection1.milvus.insert('xxm', [img_vector], [count])
        print(result, id)
        img_num[id[0]] = filename2
        count += 1
    target_w = open(target_filename, 'a+', encoding='utf-8')
    target_w.write(json.dumps(img_num, ensure_ascii=False) + '\n')
