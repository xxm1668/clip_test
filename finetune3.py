import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
from clip import clip
import PIL

'''
targets = torch.zeros(batch_size, dtype=torch.long).to(device)
这表示，图片和文本均不匹配

'''

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('ViT-B/32', device=device)


# 定义微调数据集
class MyDataset(Dataset):
    def __init__(self, data_path, text_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # 读取中文文本
        with open(text_path, 'r', encoding='utf-8') as f:
            self.texts = []
            self.imgs = []
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == 'image,caption':
                    continue
                img, text = line.split('.jpg,')
                self.imgs.append(img + '.jpg')
                self.texts.append(text)
            print(len(self.imgs), len(self.texts))

    def __getitem__(self, index):
        img_path = r'/home/worker/nlp/hao/Flicker8k_Dataset'
        image_path = img_path + '/' + self.imgs[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        text = self.texts[index]
        return image, text

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
    batch_size = 32

    # 定义数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 定义数据集和数据加载器
    train_dataset = MyDataset('/', 'captions.txt', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义模型、损失函数和优化器
    model = MyModel(num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 循环训练多个 epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0.0
        train_acc = 0.0
        model.train()  # 将模型设置为训练模式
        for i, (images, texts) in enumerate(train_loader):
            images = images.to(device)
            text_input = clip.tokenize(texts).to(device)
            targets = torch.zeros(batch_size, dtype=torch.long).to(device)  # 都不匹配

            # 前向传播计算损失值和准确率
            outputs = model(images, text_input)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, dim=1)
            acc = torch.sum(preds == targets) / batch_size

            # 反向传播更新模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_acc += acc.item() * batch_size
            # 打印当前batch的精确度
            print(f"Batch {i + 1}/{len(train_loader)}, Train acc: {acc.item():.4f}")

        # 计算平均损失值和准确率
        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)

        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
    # 保存微调模型
    torch.save(model.state_dict(), 'my_model.pth')


train()
