import torch
import numpy as np
import shutil
import os

from data import ljspeech
import hparams as hp


def preprocess_ljspeech(filename): # 'data\\LJSpeech-1.1'
    """
    该函数的作用是： 执行数据预处理
    输入参数：
        filename:LJSpeech 数据集路径
    """
    in_dir = filename
    out_dir = hp.mel_ground_truth # mel谱图的保存路径 './mels'
    # mel 谱图输出路径为 ./mels ，若路径不存在则创建路径
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # 执行语音波形-mel谱图转换，并保存mel谱图，得到LJSpeech数据集语音文本列表
    metadata = ljspeech.build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir) # 保存文本数据
    # 将生成的train.txt文件移动到data路径下
    shutil.move(os.path.join(hp.mel_ground_truth, "train.txt"),
                os.path.join("data", "train.txt"))

# 将每个音频文件对应的文本数据进行保存
def write_metadata(metadata, out_dir):
    """
    该函数的作用是： 按指定路径写入文件
    输入参数：
		metadata:LJSpeech 数据集语音文本列表
		out_dir	:输出路径
    """
    # 打开待写入文件
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata: # 将语音文本列表按行写入
            f.write(m + '\n')


def main():
    path = os.path.join("data", "LJSpeech-1.1") # 'data\\LJSpeech-1.1'
    preprocess_ljspeech(path)


if __name__ == "__main__":
    main()
