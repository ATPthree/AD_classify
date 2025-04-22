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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置部分
torch.backends.cuda.enable_flash_sdp(False)
save_dir = "./results"  # 改为相对路径，更直观
os.makedirs(save_dir, exist_ok=True)
train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
batch_size = 16
epoches = 20    #5个epoch的效果就区域收敛
model_path = "D:/111bertmodel/bertmodel"
hidden_size = 768
maxlen = 260 
# 数据暂时按照260来后期看看到底如何才能涵盖所有的字长，但是那个掩码遮蔽什么的还没有考虑（问问ai）
train_data= "../data1.csv"
test_data = "../data2.csv"

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)


# 数据集类
class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.sentences[index],
            padding='max_length',
            truncation=True,
            max_length=maxlen,
            return_tensors='pt'
        )

        if self.with_labels:
            # 将标签转换为浮点数，用于回归任务
            label = torch.tensor(float(self.labels[index]), dtype=torch.float)
            return (
                encoded['input_ids'].squeeze(0),
                encoded['attention_mask'].squeeze(0),
                encoded['token_type_ids'].squeeze(0),
                label
            )
        else:
            return (
                encoded['input_ids'].squeeze(0),
                encoded['attention_mask'].squeeze(0),
                encoded['token_type_ids'].squeeze(0)
            )

# 构造数据集
dataset_train = MyDataset(df_train['X'].tolist(), df_train['y'].tolist())
dataset_test = MyDataset(df_test['X'].tolist(), df_test['y'].tolist())

# 数据加载器
train_loader = Data.DataLoader(dataset_train, batch_size=8, shuffle=True)
test_loader = Data.DataLoader(dataset_test, batch_size=1, shuffle=False)

# 整体模型 - 修改为回归模型
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
        self.linear = nn.Linear(330,100)

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

class AttentionLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.linear = nn.Linear(hidden_dim, 100)
        
    def forward(self, x, mask=None):
        # x: [batch, seq, hidden], mask: [batch, seq]
        outputs, (h_n, _) = self.lstm(x)  # outputs: [batch, seq, hidden]
        
        # 应用注意力机制，但考虑mask
        attn_scores = self.attention(outputs)  # [batch, seq, 1]
        
        # 如果提供了mask，将填充位置的注意力分数设为一个非常小的值
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch, seq, 1]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        
        # 应用softmax得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq, 1]
        
        # 加权汇总输出
        context = torch.bmm(attn_weights.transpose(1, 2), outputs)  # [batch, 1, hidden]
        return self.linear(context.squeeze(1))  # [batch, hidden]

class BertRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.cnn = TextCNN()
        self.lstm = AttentionLstm(input_dim=768, hidden_dim=768, num_layers=1)
        self.linear1 = nn.Linear(200, 60)  # 自己加的控制维度操作
        self.rule = nn.ReLU()
        self.linear2 = nn.Linear(60, 1)
        self.linear3 = nn.Linear(60,1)


    def forward(self, input_ids, attention_mask):
        # print(input_ids, attention_mask, token_type_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # 获取完整的隐藏状态用于CNN
        hidden_states = outputs.last_hidden_state
        part1 = self.cnn(hidden_states)
        
        # 将完整序列传给LSTM，但同时传递attention_mask以忽略填充token
        seq_tokens = hidden_states  # 使用完整序列
        part2 = self.lstm(seq_tokens, attention_mask)  # 传递mask到LSTM
        
        #print("part1",part1.shape) #8,100
        #print("part2",part2.shape) #8,100
        # 拼接CNN和LSTM的输出
        final_part = torch.cat((part1, part2), 1)
        final_part = self.rule(self.linear1(final_part))
        # return self.softmax(self.linear2(final_part))
        #final_part = self.rule(self.linear2(final_part))
        
        # 确保输出是一个单一值（而不是带有额外维度）
        output = self.linear3(final_part)
        #print("output",output.shape)
        
        return output.squeeze(-1)  # 移除最后一个维度，从[batch, 1]变为[batch]

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertRegression().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    # 使用MSE损失函数用于回归任务
    criterion = nn.MSELoss()

    # 训练模型
    print(f"开始训练模型，设备: {device}")
    for epoch in range(epoches):  # 使用全局定义的epoches变量
        model.train()
        total_loss = 0
        for input_ids, attn_mask, token_types, labels in train_loader:
            input_ids, attn_mask, token_types, labels = (
                input_ids.to(device), attn_mask.to(device), token_types.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            outputs = model(input_ids, attn_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epoches} Loss: {total_loss:.4f}")
        train_curve.append(total_loss)  # 记录训练损失用于可视化

    # 评估阶段 - 修改为计算回归指标和生成预测表格
    print("开始评估模型...")
    model.eval()
    predictions = []
    actual_values = []
    diff_values = []
    
    with torch.no_grad():
        for input_ids, attn_mask, token_types, labels in test_loader:
            input_ids, attn_mask, token_types, labels = (
                input_ids.to(device), attn_mask.to(device), token_types.to(device), labels.to(device)
            )
            outputs = model(input_ids, attn_mask)
            
            # 记录预测值和真实值
            pred_value = outputs.item()
            true_value = labels.item()
            diff = abs(pred_value - true_value)
            
            predictions.append(pred_value)
            actual_values.append(true_value)
            diff_values.append(diff)
    
    # 计算平均误差
    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    mean_diff = np.mean(diff_values)
    
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'均方根误差 (RMSE): {rmse:.4f}')
    print(f'平均差值: {mean_diff:.4f}')
    
    # 生成预测结果表格
    results_df = pd.DataFrame({
        '预测值': predictions,
        '实际值': actual_values,
        '差值': diff_values
    })
    
    # 保存结果到CSV
    results_path = os.path.join(save_dir, 'prediction_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')  # 使用UTF-8编码并添加BOM标记，解决中文乱码
    print(f'预测结果已保存到: {results_path}')
    
    # 可视化训练曲线
    if train_curve:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_curve) + 1), train_curve, marker='o')
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        curve_path = os.path.join(save_dir, 'training_curve.png')
        plt.savefig(curve_path)
        print(f'训练曲线已保存到: {curve_path}')

if __name__ == "__main__":
    mp.freeze_support()
    main()



