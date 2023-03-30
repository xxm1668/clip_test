from sklearn.model_selection import train_test_split

texts = []
filename = r'/Users/haojingkun/PycharmProjects/clip_test/captions2.txt'
with open(filename, 'r', encoding='utf-8') as data:
    lines = data.readlines()
    for line in lines:
        line = line.strip()
        if line == 'image,caption':
            continue
        texts.append(line)

train, dev = train_test_split(texts, test_size=0.15, random_state=42)
print('-----')
train_filename = r'/Users/haojingkun/PycharmProjects/clip_test/train.txt'
dev_filename = r'/Users/haojingkun/PycharmProjects/clip_test/dev.txt'

train_w = open(train_filename, 'a+', encoding='utf-8')
dev_w = open(dev_filename, 'a+', encoding='utf-8')

for t in train:
    train_w.write(t + '\n')
for d in dev:
    dev_w.write(d + '\n')
