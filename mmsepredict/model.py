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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 配置部分
torch.backends.cuda.enable_flash_sdp(False)
save_dir = "./results"  # 改为相对路径，更直观
os.makedirs(save_dir, exist_ok=True)
train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
batch_size = 16
epoches = 20    #20个epoch的效果已经可以了
model_path = "D:/111bertmodel/bertmodel"
hidden_size = 768
maxlen = 260 
# 加载数据
train_data= "./data1.csv"
test_data = "./data2.csv"

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
class BertRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        # 修改输出层为单个值
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个连续值
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        # 回归任务，直接输出预测值，不需要softmax
        return self.regressor(pooled).squeeze(-1)  # 将形状从[batch_size, 1]变为[batch_size]

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
            outputs = model(input_ids, attn_mask, token_types)
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
            outputs = model(input_ids, attn_mask, token_types)
            
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



