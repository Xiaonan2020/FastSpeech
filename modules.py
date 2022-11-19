import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from numba import jit
import numpy as np
import copy
import math

import hparams as hp
import utils

# 相比于transformer路径下文件中的模块，本文件中的各个模块更加上层，包括Length Regulator等

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table 正弦信号位置编码表'''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) # 深层clone，即参数不共享


# @jit(nopython=True)
def create_alignment(base_mat, duration_predictor_output):
    """
    基于对齐信息（音素序列对应的音素持续时间）调整对齐矩阵
    @param base_mat:元素值全为0的初始对齐矩阵，[batch_size, max_mel_length, max_sequence_len]
    @param duration_predictor_output:音素持续时间矩阵，[batch_size, max_sequence_len]
    @return:经过调整后的对齐矩阵
    """

    N, L = duration_predictor_output.shape # 音素持续时间矩阵的尺寸
    for i in range(N): # batch中的第i个音素序列（一句话转换成音素序列）
        count = 0
        for j in range(L):  # 第i个音素序列中第j个音素
            for k in range(duration_predictor_output[i][j]): # duration_predictor_output[i][j]值表示[i,j]位置对应音素的长度
                base_mat[i][count+k][j] = 1 # duration_predictor_output[i][j]长度范围内连续位置均设置为1，出现几个1表示重复几次
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor()

    # 对输入的音素序列x进行长度调整
    def LR(self, x, duration_predictor_output, mel_max_length=None):
        """
        基于音素持续时间将音素序列长度与音素谱图序列长度对齐一致
        @param x:经过FFT块转换后的音素序列，[batch_size, max_sequence_len, encoder_dim]
        @param duration_predictor_output:音素持续时间矩阵，[batch_size, max_sequence_len]
        @param mel_max_length:音素谱图序列中最大长度
        @return:长度经过调整后的音素序列，[batch_size, expand_max_len, encoder_dim]
        """
        # 获取所有音素谱图序列中长度最大值
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        # 以0初始化对齐矩阵，[batch_size, expand_max_len, max_sequence_len]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        # 以音素持续时间调整对齐矩阵，[batch_size, expand_max_len, max_sequence_len]
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(device)
        # 以0初始化对齐矩阵，[batch_size, expand_max_len, max_sequence_len]
        output = alignment @ x # [batch_size, expand_max_len, encoder_dim] [4 764 256]=[4 764 114]@[4 114 256]
        if mel_max_length: # 如果设置了音素谱图序列最大值，还需要使用0进行pad
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None): # target是预先提取的音素持续时间，其有无决定模块输出
        duration_predictor_output = self.duration_predictor(x) # duration predictor计算输出的音素持续时间 [4 114]

        if target is not None: # 如果target存在，表示训练过程，目的是作为监督信息训练duration predictor
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output # [4 764 256] [4 114]
        else: # target不存在，为推理过程，直接使用duration predictor输出的音素持续时间
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack(
                [torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(device)

            return output, mel_pos

# 音素持续时间预测模块，内部是两层一维卷积再加上一个全连接层
class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.input_size = hp.encoder_dim # 256
        self.filter_size = hp.duration_predictor_filter_size # 256
        self.kernel = hp.duration_predictor_kernel_size # 3
        self.conv_output_size = hp.duration_predictor_filter_size # 256
        self.dropout = hp.dropout # 0.1

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = self.relu(out) # [4 114 1]
        out = out.squeeze() # [4 114]
        if not self.training:
            out = out.unsqueeze(0)
        return out

# 一维卷积后批量正则
class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None, w_init_gain='linear'):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

        torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)

# 自定义一维卷积层，卷积计算前会将数据的后两个维度进行转置
class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

# 自定义线性全连接层
class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)

# 前处理网络
class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', Linear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', Linear(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5)),
        ]))

    def forward(self, x):
        out = self.layer(x)
        return out

# 此模块将mel谱图转换为线性幅度谱图，以供griff-lim或其它vocoder使用幅度谱图重建音频
class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]): #projections=[256, 80], K=8
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1] # [640, 256]
        activations = [self.relu] * (len(projections) - 1) + [None] # [ReLU(), None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False) # projections [256, 80]
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T]
                       for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks) # 80 * 8
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2) # [4 764 80]

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


if __name__ == "__main__":
    # TEST # TEST，可以为音素序列长度调整的过程
    a = torch.Tensor([[2, 3, 4], [1, 2, 3]]) # 音素序列1 [2, 3] T D
    b = torch.Tensor([[5, 6, 7], [7, 8, 9]]) # 音素序列2 [2, 3]
    c = torch.stack([a, b]) # 相当于一个batch的音素序列 [2 2 3] B T D

    d = torch.Tensor([[1, 4], [6, 3]]).int() # 相当于duration predictor的输出，即音素持续时间 [2 2]
    expand_max_len = torch.max(torch.sum(d, -1), -1)[0]  # 获取同一batch中音素谱图序列的长度最大值
    base = torch.zeros(c.size(0), expand_max_len, c.size(1)) # 以0初始化对齐矩阵 [b expand_max_len T] [ 2 9 2]
    # 基于音素持续时间调整对齐矩阵
    alignment = create_alignment(base.numpy(), d.numpy())
    print(alignment) # [2 9 2]
    # 将对齐矩阵与batch数据执行矩阵乘法，使得音素序列长度得到调整，与音素谱图序列长度对齐
    # 结果如d[0][0]的值为1，那么c中[0][0]位置的音素就出现1次，结果如d[0][1]的值为4，那么c中[0][1]位置的音素就出现4次
    print(torch.from_numpy(alignment) @ c) # [2 9 2] * [2 2 3] = [2 9 3]
