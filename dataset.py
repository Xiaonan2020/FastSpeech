import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import time
import os

import hparams
import audio

from utils import process_text, pad_1D, pad_2D
from utils import pad_1D_tensor, pad_2D_tensor
from text import text_to_sequence
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 该文件是主要用于数据加载和数据转换，将文本、持续时间和mel谱图序列加载封装至定义的BufferDataset对象中，然后定义回调函数collate_fn_tensor将对数据进行pad等操作，转换为模型训练所需的格式


def get_data_to_buffer():
    """
    该函数的作用是： 加载训练数据，组装数据集
    输出参数：
    buffer:
    """
    buffer = list() # 创建列表
    # 将全部的音频文本读取到一个列表对象中，text是一个列表，每一个元素是一个字符串，即一个音频对应的文本
    text = process_text(os.path.join("data", "train.txt"))

    start = time.perf_counter() # 记录开始时间
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1)) # 获取 mel 谱图文件路径，并加载至内存 './mels\\ljspeech-mel-00001.npy'
        mel_gt_target = np.load(mel_gt_name) # 加载文本对应的音频文件的mel谱图
        # 加载音素持续时间文件
        duration = np.load(os.path.join(
            hparams.alignment_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1] # 删除最后的换行符
        # 将文本与音素进行对应，转换为音素编号序列
        character = np.array(
            text_to_sequence(character, hparams.text_cleaners)) # 将英文文本转换为数值序列，相当于分词

        # character和duration的长度一致，即duration中的i的值，表示character中i位置的数值出现的次数
        character = torch.from_numpy(character)
        # dutation中所有数值之和与mel的长度相等，即character经过duration调整后，文本长度将于mel谱图长度对齐
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        # 将一个音频文件的文本、持续时间和mel谱图数据组合成一个字典对象存在在列表中
        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target})
    # 记录结束时间，打印加载日志
    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer # 返回组装的训练数据


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer # 加载所有数据
        self.length_dataset = len(self.buffer) # 数据集总数量

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    """
    以传入的batch数据和对应的序列索引，给文本序列、mel谱图序列建立位置信息，同时将其封装在一起输出
    该函数的作用是： 处理数据，并将该批数据进行补齐，使序列长度保持一致
    输入参数：
    batch	:	从数据集中每批加载的数据
    cut_list	:	部分批数据列表 一个real batch大小的索引列表，其对应的文本长度从大到小降序排列
    输出参数：
    out		:	处理后的批输出
    """
    # 获取部分批数据对应的音素序列
    texts = [batch[ind]["text"] for ind in cut_list] # batch中的文本
    # 获取部分批数据对应的 mel 谱图
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list] # batch中的gt梅尔谱图
    # 获取部分批数据对应的持续时间序列
    durations = [batch[ind]["duration"] for ind in cut_list] # batch中的duration时间

    # 建立数组存储音素序列长度
    length_text = np.array([]) # 存储所有文本序列的长度大小
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    # 以每批最长的序列为基准在末端补齐序列，并转换为 Tensor 数据
    src_pos = list()
    max_len = int(max(length_text)) # 最大文本长度
    for length_src_row in length_text:
        # 给每个文本生成src_pos，从1到文本的长度，如果长度小于max_len，对应部分用0填充
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    #  建立数组存储 mel 谱图长度
    length_mel = np.array(list()) # 存储所有mel谱图序列的长度大小
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    # 以每批最长的谱图长度为基准在末端补齐，并转换为 Tensor 数据
    mel_pos = list()
    max_mel_len = int(max(length_mel))  # 最大mel谱图序列长度
    for length_mel_row in length_mel:
        # 给每个mel谱图序列生成mel_pos，从1到序列的长度，如果长度小于max_mel_len，对应部分用0填充
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    # 依次补齐音素序列、持续时间与梅尔谱图，并转换为 Tensor 数据
    # 其中对数据进行pad是使用了utils.py文件中的两个函数
    texts = pad_1D_tensor(texts) # 将所有的文本都pad到文本的最大长度 [real_batchsize pad_text_maxlen]
    durations = pad_1D_tensor(durations) # 将所有的duration持续时间pad到最大长度 [real_batchsize pad_text_maxlen]
    mel_targets = pad_2D_tensor(mel_targets) # 将所有mel谱图序列pad到最大长度 [real_batchsize pad_mel_maxlen(max T) D(80)]

    # 将得到的数据组装为字典，作为输出结果返回
    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out

# 构建Loader时数据转换的回调函数
def collate_fn_tensor(batch):
    """
    该函数的作用是：处理从数据集中加载的数据，作为 DataLoader 的最终输出
    输入参数：
		batch:从数据集中每批加载的数据
    输出参数：
		output:最终DataLoader每批的输出
    """
    # 使用数组为每段语音记录音素序列长度
    len_arr = np.array([d["text"].size(0) for d in batch]) # 一个batch中文本序列的长度列表
    # 按从大到小的顺序记录音素序列长度排序
    index_arr = np.argsort(-len_arr) # 对len_arr进行降序排序后，从大到小返回值在原列表中的索引
    # 记录一整批数据的长度
    batchsize = len(batch)
    # 计算真正每批数据的长度
    real_batchsize = batchsize // hparams.batch_expand_size

    # 创建分割后的批数据列表
    cut_list = list()
    # 按真正批大小切分整批数据，添加至批数据列表
    for i in range(hparams.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize]) # 将index_arr分成hparams.batch_expand_size段

    # 创建输出列表
    output = list()
    # 为每一小批数据执行预处理，并添加至输出列表
    for i in range(hparams.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    # 将输出列表返回
    return output # output中一个元素就是一个real batch的数据


if __name__ == "__main__":
    # TEST
    # get_data_to_buffer()

    a = get_data_to_buffer()
    print(len(a))
    print(a[0]['text'].shape)
    print(a[0]['duration'].sum())
    print(a[0]['mel_target'].shape)
