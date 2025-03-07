import os
import numpy as np
import pandas as pd
import torch
import  random
import torch.nn as nn
import torch.utils.data as Data
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

from 复现AD论文.testcnn_other import n_class


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.deterministic = True  # 固定CUDA操作顺序
        torch.backends.cuda.flash_sdp = False  # 根据需求设置（您已禁用了flash_sdp）

set_seed(42)

# 配置部分
torch.backends.cuda.enable_flash_sdp(False)
save_dir = "path"
os.makedirs(save_dir, exist_ok=True)
train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
batch_size = 8
epoches = 5
model_path = "D:/111bertmodel/bertmodel"
hidden_size = 768
maxlen = 320
n_class=20
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

# 构造数据集
dataset_train = MyDataset(train_data['data'].tolist(), train_data['label'].tolist())
dataset_test = MyDataset(test_data['data'].tolist(), test_data['label'].tolist())

# 构造数据加载器
train_loader = Data.DataLoader(dataset_train, batch_size=8, shuffle=True)
test_loader = Data.DataLoader(dataset_test, batch_size=1, shuffle=False)
# TextCNN 模型
class TextCNN(nn.Module): #修改，通过池化操作降低特征维度
    def __init__(self):
        super().__init__()
        filter_sizes = [5, 10, 15, 20 ]
        num_filters = 1 #不同的卷积核的个数先设置为1

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, hidden_size))  #沿着时间序列走
            #q.这里的num_filters是什么意思？
            #a.是指每种卷积核的个数
            for size in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), n_class)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len, hidden]
        pooled = []
        for conv in self.convs:      #总体融合
            h = F.relu(conv(x)).squeeze(3)  # [batch, num_filters, seq_len]  卷积操作
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  #池化操作
            pooled.append(h)
        h_pool = torch.cat(pooled, 1)#这个是全连接层的输入

        return self.fc(h_pool)
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

# 整体模型
class BertBlendCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.cnn = TextCNN()
        self.lstm = Lstm(input_dim=768,hidden_dim=768,num_layers=1)
        self.linear=nn.Linear(788,2)#自己加的控制维度操作
        self.softmax = nn.Softmax(dim=1) #输出的是概率值他的形状是[batch_size, n_class]，n_class是类别数
        self.dropout = nn.Dropout(0.2)
    def forward(self, input_ids, attention_mask, token_type_ids):
       # print(input_ids, attention_mask, token_type_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        #print(outputs.last_hidden_state.size())
        part1= self.cnn(outputs.last_hidden_state)
        part2 = self.lstm(outputs.last_hidden_state)
        #print("合并后的维度{}",torch.cat((part1,part2),dim=1).shape)
        final_part=torch.cat((part1,part2),1) #这里是直接相加，如果要拼接的话就是torch.cat((part1,part2),1)
        final_part = self.dropout(final_part)
        return self.softmax(self.linear(final_part))
    #

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBlendCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
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


