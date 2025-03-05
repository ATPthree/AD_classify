from transformers import AutoTokenizer, AutoModel
import torch
import os
model_path=os.path.abspath("D:/111/bertmodel")
# 1. 加载预训练的分词器和BERT模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 2. 分词器处理文本（生成输入ID和注意力掩码）
#text = "Natural language processing is fascinating."
text ="I am a student"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 3. BERT前向传播（不计算梯度，仅用于特征提取）
with torch.no_grad():
    outputs = model(**inputs)

# 4. 提取token的上下文向量矩阵（形状为 [1, seq_len, 768]）
token_vectors = outputs.last_hidden_state

print("输入文本的token列表:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("Token向量矩阵形状:", token_vectors.shape)

