from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

# 配置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "D:/111bertmodel/bertmodel"
batch_size = 8
epochs = 20
maxlen = 256


# ----------------- 数据加载与预处理 -----------------
def load_data(control_path, ad_path):
    # 读取数据并添加标签
    control_df = pd.read_csv(control_path)
    ad_df = pd.read_csv(ad_path)

    # 合并数据集
    control_df['label'] = 1  # 正常人标签
    ad_df['label'] = 0  # AD患者标签
    full_df = pd.concat([control_df, ad_df], axis=0)

    # 清洗数据
    full_df = full_df[['data', 'label']].dropna()
    return full_df


# 加载数据（请替换实际路径）
full_data = load_data(
    control_path="D:/AD_detect/control_labeled.csv",
    ad_path="D:/AD_detect/ad_labeled.csv"
)

# 分层分割数据集
train_df, test_df = train_test_split(
    full_data,
    test_size=0.3,
    stratify=full_data['label'],
    random_state=42
)


# ----------------- 数据集类 -----------------
class ADDataset(Data.Dataset):
    def __init__(self, texts, labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=maxlen,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ----------------- 模型定义 -----------------
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        #print(pooled.shape)
        return self.classifier(pooled)


# ----------------- 训练与评估 -----------------
def train_model():
    # 初始化
    model = BertClassifier().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 创建数据集
    train_dataset = ADDataset(
        train_df['data'].tolist(),
        train_df['label'].tolist()
    )
    test_dataset = ADDataset(
        test_df['data'].tolist(),
        test_df['label'].tolist()
    )

    # 数据加载器
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                outputs = model(**inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Test Acc: {100 * correct / total:.2f}%")
        print("------------------------")


if __name__ == "__main__":
    train_model()