import pandas as pd
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer, AutoModel
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 加载数据
ad_path = "ad_labeled.csv"
control_path = "control_labeled.csv"

df_ad = pd.read_csv(ad_path)
df_control = pd.read_csv(control_path)

df = pd.concat([df_ad, df_control], ignore_index=True)

# 打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 计算训练集和测试集的分割索引
train_size = int(0.7 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

# 设定BERT模型路径
model_path = "D:/111bertmodel/bertmodel"
tokenizer = AutoTokenizer.from_pretrained(model_path)


# 数据集类
class MyDataset(Data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            padding='max_length',
            truncation=True,
            max_length=320,
            return_tensors='pt'
        )
        return (
            encoded['input_ids'].squeeze(0),
            encoded['attention_mask'].squeeze(0),
            encoded['token_type_ids'].squeeze(0),
            torch.tensor(self.labels[index], dtype=torch.long)
        )


# 构造数据集
dataset_train = MyDataset(train_data['data'].tolist(), train_data['label'].tolist())
dataset_test = MyDataset(test_data['data'].tolist(), test_data['label'].tolist())

# 构造数据加载器
train_loader = Data.DataLoader(dataset_train, batch_size=1, shuffle=True)
test_loader = Data.DataLoader(dataset_test, batch_size=1, shuffle=False)


# LSTM 和 TextCNN 模型
class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        filter_sizes = [5, 10, 15, 20]
        num_filters = 1
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, 768)) for size in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), 20)

    def forward(self, x):
        x = x.unsqueeze(1)
        pooled = [F.max_pool1d(F.relu(conv(x)).squeeze(3), conv(x).size(2)).squeeze(2) for conv in self.convs]
        h_pool = torch.cat(pooled, 1)
        return self.fc(h_pool)


class Lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


# 主模型
class BertBlendCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.cnn = TextCNN()
        self.lstm = Lstm(input_dim=768, hidden_dim=768, num_layers=1)
        self.linear = nn.Linear(788, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        part1 = self.cnn(outputs.last_hidden_state)
        part2 = self.lstm(outputs.last_hidden_state)
        final_part = torch.cat((part1, part2), 1)
        return self.softmax(self.linear(final_part))


# 训练和测试函数
def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBlendCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
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
    train_and_evaluate()