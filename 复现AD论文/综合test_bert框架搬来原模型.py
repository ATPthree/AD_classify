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
class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        filter_sizes = [5, 10, 15, 20 ]
        num_filters = 1
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, 768))  #沿着时间序列走
            for size in filter_sizes
        ])
        #self.output_dim=
        self.linear = nn.Linear(325,100)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len, hidden]
        pooled = []
        i=1
        for conv in self.convs:      #总体融合
            h = F.relu(conv(x)).squeeze(3)  # [batch, num_filters, seq_len]  卷积操作
            h = F.max_pool1d(h, kernel_size=3).squeeze(2)  #池化操作减少特征数量
            i=i+1
            pooled.append(h)
        h_pool = torch.cat(pooled, 2)#这个是全连接层的输入
        #print("h_pool{}",h_pool.size()) #但是原论文中好像没有linear
        h_flap=h_pool.squeeze(1) #去掉中间的维度
        #print("h_flap{}",h_flap.shape)
        return self.linear(h_flap)
class Lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):#可以对
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,  # BERT输出的隐藏维度（768）
            hidden_size=hidden_dim,  # LSTM内部隐藏层维度（需论文指定）  后续可以自己改一下，还记得以前写的那个期货预测的LSTM吗用的两个隐藏层（或者是两个lstm拼接）加一个softmax还是linear直接输出了
            num_layers=num_layers,  # LSTM层数（需论文指定）
            batch_first=True  # 输入格式为 (batch, seq, feature)
        )

    def forward(self, x):
        # x形状：(batch_size, sequence_length, input_dim)
        _, (h_n, _) = self.lstm(x)
        #print("lstm输出的维度{}",h_n[-1].shape)
        return h_n[-1]  # 返回最后一层的最终状态（形状：(batch_size, hidden_dim)）


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.cnn = TextCNN()
        self.lstm = Lstm(input_dim=768, hidden_dim=768, num_layers=1)
        self.linear1 = nn.Linear(868, 120)  # 自己加的控制维度操作
        self.rule = nn.ReLU()
        self.linear2 = nn.Linear(120, 2)
        self.softmax = nn.Softmax(dim=1)  # 输出的是概率值他的形状是[batch_size, n_class]，n_class是类别数

    def forward(self, input_ids, attention_mask):
        # print(input_ids, attention_mask, token_type_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # print(outputs.last_hidden_state.size())
        part1 = self.cnn(outputs.last_hidden_state)
        part2 = self.lstm(outputs.last_hidden_state)
        # print("cnn:",part1.shape)
        # print("lstm",part2.shape)
        # print("cnn:"+part1.shape+"lstm:"+part2.shape+"合并后的维度：",torch.cat((part1,part2),dim=1).shape)
        final_part = torch.cat((part1, part2), 1)  # 这里是直接相加，如果要拼接的话就是torch.cat((part1,part2),1)
        # print(final_part.shape)
        final_part = self.rule(self.linear1(final_part))
        # return self.softmax(self.linear2(final_part))
        return self.linear2(final_part)


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