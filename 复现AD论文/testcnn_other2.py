import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import os
from testcnn_bert_lasthidden import BertBlendCNN, MyDataset
from 复现AD论文.testcnn_bert_lasthidden import save_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir="path"


def main():
    bert_blend_cnn = BertBlendCNN().to(device)
    state_dict = torch.load(os.path.join(save_dir, 'model.pt'),
                            map_location=device,
                            weights_only=True)  # 修复安全警告

    # 键名适配（若训练时保存的是textcnn子模块）
    # state_dict = {'textcnn.' + k: v for k, v in state_dict.items()}

    bert_blend_cnn.load_state_dict(state_dict, strict=False)  # 非严格模式忽略多余键
    # test
    bert_blend_cnn.eval()
    with torch.no_grad():
      test_text = ['I very hate you']
      test_data=MyDataset(test_text, with_labels=False)

      sample = test_data[0]
      # 获取第一个样本（索引0）
      input_ids, attn_mask, token_types = sample # 关键修改
      print(input_ids.size)
      print(attn_mask.size)
      print(token_types.size)
      # 添加batch维度并转移设备
      input_ids = input_ids.unsqueeze(0).to(device)  # [1, seq_len]
      attn_mask = attn_mask.unsqueeze(0).to(device)
      token_types = token_types.unsqueeze(0).to(device)

      # 预测
      logits = bert_blend_cnn(input_ids, attn_mask, token_types)
      pred_label = logits.argmax().item()
      print("Prediction:", "积极" if pred_label == 1 else "消极")



if __name__ == '__main__':
    main()
