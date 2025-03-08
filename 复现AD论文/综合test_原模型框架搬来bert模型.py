import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

# 配置部分
torch.backends.cuda.enable_flash_sdp(False)
save_dir = "path"
os.makedirs(save_dir, exist_ok=True)
train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
batch_size = 16
epoches = 5
model_path = "D:/111bertmodel/bertmodel"
hidden_size = 768
#n_class = 20  #决定输出的维度
maxlen = 260 #320可行，260不行了？
# 加载数据
ad_path = "ad_labeled.csv"
control_path = "control_labeled.csv"

df_ad = pd.read_csv(ad_path)
df_control = pd.read_csv(control_path)

# 显式定义标签并合并数据集（关键改动）
df_control['label'] = 1  # 正常人标签
df_ad['label'] = 0    # AD患者标签
df = pd.concat([df_control, df_ad], axis=0)

# 数据清洗（新增）
df = df[['data', 'label']].dropna()  # 删除缺失值行

# 分层抽样分割数据集（关键改动）
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['label'],   # 保持类别分布一致性
    random_state=42
)
# 数据集类
class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)  # 修正为实例变量

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.sentences[index],
            padding='max_length',
            truncation=True,
            max_length=maxlen,
            return_tensors='pt'
        )

        if self.with_labels:
            return (
                encoded['input_ids'].squeeze(0),  # 修正为实例变量
                encoded['attention_mask'].squeeze(0),  # 修正为实例变量
                encoded['token_type_ids'].squeeze(0),  # 修正为实例变量
                self.labels[index]  # 修正为实例变量
            )
        else:
            return (
                encoded['input_ids'].squeeze(0),
                encoded['attention_mask'].squeeze(0),
                encoded['token_type_ids'].squeeze(0)
            )

# 构造数据集（改动）
dataset_train = MyDataset(train_df['data'].tolist(), train_df['label'].tolist())
dataset_test = MyDataset(test_df['data'].tolist(), test_df['label'].tolist())

# 数据加载器（保持batch_size一致）
train_loader = Data.DataLoader(dataset_train, batch_size=8, shuffle=True)
test_loader = Data.DataLoader(dataset_test, batch_size=1, shuffle=False)

# 整体模型
class BertBlendCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask,token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token只取第一个位置CLS位置的向量
        # print(pooled.shape)
        return self.classifier(pooled)
    #

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBlendCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for input_ids, attn_mask, token_types, labels in train_loader:
            input_ids, attn_mask, token_types, labels = (
                input_ids.to(device), attn_mask.to(device), token_types.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            outputs = model(input_ids, attn_mask, token_types)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attn_mask, token_types, labels in test_loader:
            input_ids, attn_mask, token_types, labels = (
                input_ids.to(device), attn_mask.to(device), token_types.to(device), labels.to(device)
            )
            outputs = model(input_ids, attn_mask, token_types)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.4f}%')






if __name__ == "__main__":
    mp.freeze_support()
    main()
    # pd.DataFrame(train_curve).plot()
    # plt.show()


