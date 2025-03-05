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
from testcnn_other import Bert_Blend_CNN, MyDataset
from 复现AD论文.testcnn_other import save_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir="path"


def main():
    bert_blend_cnn = Bert_Blend_CNN().to(device)
    state_dict = torch.load(os.path.join(save_dir, 'final_textcnn_weights.pth'),
                            map_location=device,
                            weights_only=True)  # 修复安全警告

    # 键名适配（若训练时保存的是textcnn子模块）
    # state_dict = {'textcnn.' + k: v for k, v in state_dict.items()}

    bert_blend_cnn.load_state_dict(state_dict, strict=False)  # 非严格模式忽略多余键
    # test
    bert_blend_cnn.eval()
    with torch.no_grad():
      test_text = ['a beautiful music']
      test = MyDataset(test_text, labels=None, with_labels=False)
      x = test.__getitem__(0)
      x = tuple(p.unsqueeze(0).to(device) for p in x)
      pred = bert_blend_cnn([x[0], x[1], x[2]])
      pred = pred.data.max(dim=1, keepdim=True)[1]
      if pred[0][0] == 0:
        print('消极')
      else:
        print('积极')


if __name__ == '__main__':
    main()
