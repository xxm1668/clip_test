import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
from clip import clip
import logging

'''
targets = torch.zeros(batch_size, dtype=torch.long).to(device)
这表示，图片和文本均不匹配

'''

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('ViT-B/32', device=device)

# 加载图片前缀
nums = set()
filename = r'captions2.txt'
with open(filename, 'r', encoding='utf-8') as data:
    lines = data.readlines()
    for line in lines:
        line = line.strip()
        if line == 'image,caption':
            continue
        img, text = line.split('.jpg,')
        num = img.split('_')[0]
        nums.add(num)
        print('----')
nums = list(nums)


# 定义微调数据集
class MyDataset(Dataset):
    def __init__(self, data_path, text_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # 读取中文文本
        with open(text_path, 'r', encoding='utf-8') as f:
            self.texts = []
            self.imgs = []
            self.labels = []
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == 'image,caption':
                    continue
                img, text = line.split('.jpg,')
                self.imgs.append(img + '.jpg')
                self.texts.append(text)
                label = img.split('_')[0]
                self.labels.append(nums.index(label))

    def __getitem__(self, index):
        img_path = r'/home/worker/nlp/hao/Flicker8k_Dataset'
        image_path = img_path + '/' + self.imgs[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.texts)


# 定义微调模型
class MyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.clip = clip_model
        self.linear = torch.nn.Linear(1024, num_classes)

    def forward(self, images, texts):
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            text_features = self.clip.encode_text(texts)
        features = torch.cat([image_features, text_features], dim=1)
        logits = self.linear(features.float())
        return logits


def train():
    # 定义训练参数
    lr = 0.001
    epochs = 10
    batch_size = 8

    # 定义数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 定义数据集和数据加载器
    train_dataset = MyDataset('/', 'train.txt', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义模型、损失函数和优化器
    model = MyModel(num_classes=len(nums)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 循环训练多个 epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0.0
        train_acc = 0.0
        model.train()  # 将模型设置为训练模式
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            text_input = clip.tokenize(train_dataset.texts[i * batch_size:(i + 1) * batch_size]).to(device)
            targets = labels.to(device)

            # 前向传播计算损失值和准确率
            outputs = model(images, text_input)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, dim=1)
            acc = torch.sum(preds == targets) / batch_size

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失值和准确率
            train_loss += loss.item() * batch_size
            train_acc += acc * batch_size
            print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
            eval_acc = evaluate(model, '/', 'dev.txt', 4)

        # 输出本 epoch 的平均损失值和准确率
        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        # 保存微调模型
        torch.save(model.state_dict(), 'my_model_{}.pth'.format(epoch))


def evaluate(model, data_path, text_path, batch_size):
    # 定义数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    criterion = torch.nn.CrossEntropyLoss()

    # 定义数据集和数据加载器
    eval_dataset = MyDataset(data_path, text_path, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model.eval()  # 将模型设置为评估模式
    eval_loss = 0.0
    eval_acc = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(eval_loader):
            images = images.to(device)
            text_input = clip.tokenize(eval_dataset.texts[i * batch_size:(i + 1) * batch_size]).to(device)
            targets = labels.to(device)

            # 前向传播计算损失值和准确率
            outputs = model(images, text_input)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, dim=1)
            acc = torch.sum(preds == targets) / batch_size

            # 累加损失值和准确率
            eval_loss += loss.item() * batch_size
            eval_acc += acc * batch_size

    # 输出平均损失值和准确率
    eval_loss /= len(eval_dataset)
    eval_acc /= len(eval_dataset)
    print(f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")
    return eval_acc


def predict(image_path, text):
    # 加载图片并进行预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # 对文本进行编码
    with torch.no_grad():
        text_input = clip.tokenize([text]).to(device)
        text_features = clip_model.encode_text(text_input)

    # 使用微调模型进行预测
    model = MyModel(num_classes=len(nums)).to(device)
    model.load_state_dict(torch.load('my_model.pth'))
    model.eval()
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        features = torch.cat([image_features, text_features], dim=1)
        logits = model.linear(features.float())
        preds = torch.argmax(logits, dim=1)

    return preds.item()


train()
