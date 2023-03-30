import json
import requests


def download_img(img_url, img_name):
    print(img_url)
    r = requests.get(img_url, stream=True)
    print(r.status_code)  # 返回状态码
    if r.status_code == 200:
        open('/Users/haojingkun/PycharmProjects/clip_test/images/' + img_name, 'wb').write(r.content)  # 将内容写入图片
        print("done")
    del r


filename = r'/Users/haojingkun/PycharmProjects/clip_test/captions_val2017.json'

with open(filename, 'r', encoding='utf-8') as data:
    lines = data.readlines()
    for line in lines:
        line = line.strip()
        images = json.loads(line)['images']
        for image in images:
            name = image['file_name']
            url = image['coco_url']
            download_img(url, name)
            print('----')
