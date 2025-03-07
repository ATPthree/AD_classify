from transformers import AutoTokenizer, AutoModel
import torch
import os
model_path=os.path.abspath("D:/111bertmodel/bertmodel")
# 1. 加载预训练的分词器和BERT模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
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
batch_size = 1
epoches = 1
model_path = "D:/111bertmodel/bertmodel"
hidden_size = 768
n_class = 20  #决定输出的维度
maxlen = 320

# 数据
sentences = [
    "I like playing basketball This camera is very beautiful I had a great time today I like you too a good thing I don't like you It's terrible It's really sa hat's terrible I don't like playing basketball "
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


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
                #q.所以这里的input_ids，attention_mask，token_type_ids是什么意思？
                #a.是指输入的三个参数
                #q.有什么用？具体含义是什么？
                #a.是指输入的三个参数
                self.labels[index]  # 修正为实例变量
            )
        else:
            return (
                encoded['input_ids'].squeeze(0),
                encoded['attention_mask'].squeeze(0),
                encoded['token_type_ids'].squeeze(0)
            )


# TextCNN 模型
class TextCNN(nn.Module): #修改，通过池化操作降低特征维度
    def __init__(self):
        super().__init__()
        filter_sizes = [5, 10, 15, 20 ]
        num_filters = 130

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
        #print(h_pool.size()) #但是原论文中好像没有linear
        print("testcnn的输出{}",h_pool.size())
        print("testcnn经过fc的输出{}",self.fc(h_pool).shape)
        return self.fc(h_pool)
class Lstm(nn.Module):
    # def __init__(self):
        # super(Lstm, self).__init__()
        # self.hidden_size = 768
        # self.input_size = 768
        # self.num_layers = 1
        # self.output_size = 1
        # self.dropout = 0
        # self.batch_first = True
        # self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        # self.linear = nn.Linear(self.hidden_size, self.output_size)
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
        print("lstm输出的维度{}",h_n[-1].shape)
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

    def forward(self, input_ids=768, attention_mask=768, token_type_ids=1):
       # print(input_ids, attention_mask, token_type_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        #print(outputs.last_hidden_state.size())
        part1= self.cnn(outputs.last_hidden_state)
        part2 = self.lstm(outputs.last_hidden_state)
        print("合并后的维度{}",torch.cat((part1,part2),dim=1).shape)
        final_part=torch.cat((part1,part2),1) #这里是直接相加，如果要拼接的话就是torch.cat((part1,part2),1)

        return self.softmax(self.linear(final_part))
    #

# 主函数
def main():
    # 初始化
    model = BertBlendCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    labels = [1]
    # 数据加载
    train_loader = Data.DataLoader(
        MyDataset(sentences, labels),
        batch_size=batch_size,
        shuffle=True
    )

    # 训练循环
    for epoch in range(epoches):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # 正确解包
            input_ids, attn_mask, token_types, labels = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_types = token_types.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(input_ids, attn_mask, token_types)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #if (epoch + 1) % 10 == 0:
            #print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")
        train_curve.append(total_loss)

    # 测试
    model.eval()
    with torch.no_grad():
        # 测试文本（必须为列表）
        test_texts = ["I hate playing basketball and you don't ask me why beacause I am a student you just want to show off to me that't not fair at all"]

        # 创建测试数据集
        test_dataset = MyDataset(test_texts, with_labels=False)

        sample = test_dataset[0]
        # 获取第一个样本（索引0）
        input_ids, attn_mask, token_types = sample  # 关键修改

        # 添加batch维度并转移设备
        input_ids = input_ids.unsqueeze(0).to(device)  # [1, seq_len]
        attn_mask = attn_mask.unsqueeze(0).to(device)
        token_types = token_types.unsqueeze(0).to(device)

        # 预测
        logits = model(input_ids, attn_mask, token_types)
        print("最后的输出形状{}",logits.shape)
        print(logits)






if __name__ == "__main__":
    mp.freeze_support()
    main()


