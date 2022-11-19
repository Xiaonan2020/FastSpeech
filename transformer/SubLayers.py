import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer.Modules import ScaledDotProductAttention
import hparams as hp
# 本文件中定义了FFT块中的多头注意力层和基于位置的前馈神经网络(FFN)

# 自定义的多头注意力模块
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 计算注意力之前，对q、k、v进行映射的线性层
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # 初始化上述线性层
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        # 备份，用于最后与经过计算后的输出相加再输出
        residual = q
        # 将q、k、v转换为四维，即切分为多头
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # 将q、k、v中的batch_size和n_head两个维度合并，降维三维
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk [8 124 128]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk [8 124 128]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv [8 124 128]
        # 为每个头复制一个相同的mask掩码
        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x .. [8 124 124]
        output, attn = self.attention(q, k, v, mask=mask) # 计算注意力输出和attn矩阵 output [8 124 128] attn [8 124 124]

        output = output.view(n_head, sz_b, len_q, d_v) # [2 4 124 128]
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)  # 先还原为四维，然后再将特征维度还原，降为三维 [4 124 256]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual) # residual [4 124 256]

        return output, attn

# 自定义FFT中基于位置的前馈神经网络/FFN，内部均使用一位卷积，也计算了残差
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=hp.fft_conv1d_kernel[0], padding=hp.fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=hp.fft_conv1d_kernel[1], padding=hp.fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
