import torch
from transformers import AutoTokenizer, AutoModel
import os
from torch import nn
# 加载预训练模型路径
model_path = os.path.abspath("D:/111bertmodel/bertmodel")

# 1. 加载分词器和BERT模型（最后一层输出为768维）
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 2. 输入文本处理
text = "I am a student"
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,  # 自动填充至最长序列长度
    truncation=True  # 截断超过最大长度的序列
)

# 3. BERT前向传播（不计算梯度）
with torch.no_grad():
    outputs = model(**inputs)

# 4. 提取BERT最后一层的隐藏状态序列（形状：[batch, seq_len, 768]）
bert_outputs = outputs.last_hidden_state  # (1, 5, 768)
#print(bert_outputs.shape)

# 5. 定义LSTM模型（按论文要求调整参数）
class CustomLSTM(nn.Module):
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
        return h_n[-1]  # 返回最后一层的最终状态（形状：(batch_size, hidden_dim)）


# 根据论文要求设置参数（示例假设论文使用单层LSTM且隐藏层维度=768）
lstm_input_dim = 768  # BERT输出的维度（不可修改）
lstm_hidden_dim = 768  # 论文要求的LSTM隐藏层维度（需根据论文调整）
lstm_num_layers = 1  # 论文要求的LSTM层数（需根据论文调整）

lstm_model = CustomLSTM(lstm_input_dim, lstm_hidden_dim, lstm_num_layers)

# 6. 运行LSTM并获取输出
lstm_output = lstm_model(bert_outputs)  # 形状：(1, 768)

print("LSTM输出形状:", lstm_output.shape)