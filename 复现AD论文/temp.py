import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# 配置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "D:/111bertmodel/bertmodel"
batch_size = 8
epochs = 5
max_len = 256


# 自定义数据集类
class DementiaDataset(Dataset):
    def __init__(self, texts, labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# TextCNN 模块
class TextCNN(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        filter_sizes = [3, 4, 5]
        num_filters = 100
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, hidden_size))
            for size in filter_sizes
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x形状: [batch_size, seq_len, hidden_size]
        x = x.unsqueeze(1)  # 添加通道维度 [batch, 1, seq, hidden]

        pooled_outputs = []
        for conv in self.convs:
            conv_out = conv(x)  # [batch, num_filters, seq - filter_size + 1, 1]
            conv_out = conv_out.squeeze(3)  # [batch, num_filters, seq - filter_size + 1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch, num_filters, 1]
            pooled_outputs.append(pooled.squeeze(2))

        cnn_output = torch.cat(pooled_outputs, 1)  # [batch, num_filters * len(filter_sizes)]
        return self.dropout(cnn_output)


# LSTM 模块
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x形状: [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, 2*hidden_dim]
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步 [batch, 2*hidden_dim]
        return self.dropout(lstm_out)


# 完整模型
class BertBlendCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.cnn = TextCNN()
        self.lstm = LSTMClassifier()
        self.classifier = nn.Sequential(
            nn.Linear(300 + 512, 256),  # CNN输出300维 + LSTM输出512维
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        # BERT处理
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, 768]

        # 多模态特征提取
        cnn_features = self.cnn(sequence_output)  # [batch, 300]
        lstm_features = self.lstm(sequence_output)  # [batch, 512]

        # 特征融合
        combined = torch.cat([cnn_features, lstm_features], dim=1)  # [batch, 812]

        # 分类
        logits = self.classifier(combined)
        return logits


# 训练流程
def train_model():
    # 示例数据（需替换为真实数据）
    texts = ["Sample text 1", "Sample text 2"] * 100
    labels = [1, 0] * 100

    # 数据加载
    dataset = DementiaDataset(texts, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = BertBlendCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

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

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")


# 运行训练
if __name__ == "__main__":
    train_model()