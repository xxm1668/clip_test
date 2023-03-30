import finetuner
from docarray import DocumentArray, Document

filename = r'/home/worker/nlp/hao/clip_test/captions.txt'
imgage_path = r'/home/worker/nlp/hao/Flicker8k_Dataset'
doc_array = DocumentArray()
with open(filename, 'r', encoding='utf-8') as data:
    lines = data.readlines()
    count = 0
    for line in lines:
        line = line.strip()
        if count == 0:
            count += 1
            continue
        count += 1
        texts = line.split(',')
        image = imgage_path + '/' + texts[0]
        doc = Document(url=image, text=texts[1])
        doc_array.append(doc)
        print('-----')
        #
flickr8k_training_data = doc_array  # create training dataset

finetuner.login()

run = finetuner.fit(
    model='ViT-B-32#openai',
    train_data=flickr8k_training_data,
    run_name='my-clip-run',
    loss='CLIPLoss',
    epochs=5,
    learning_rate=1e-6,
)
