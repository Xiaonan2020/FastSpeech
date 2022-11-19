import torch
import torch.nn as nn
import numpy as np

# 本文件中是实现了一个点乘模块，用于计算注意力

# 自定义的点乘模块，用于计算注意力
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # 对q、k进行矩阵乘法
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature # 除以温度参数

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
