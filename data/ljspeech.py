import numpy as np
import os
import audio

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# 本文件则是主要基于audio目录中文件从LJSpeech中提取数据


def build_from_path(in_dir, out_dir):
    """
    该函数的作用是： 为 LJSpeech 数据集制作 mel 谱图，并返回语音文本列表。
    输入参数：
		in_dir	:	LJSpeech数据集路径
		out_dir	:	mel谱图输出路径
	输出参数：
		texts	:	LJSpeech数据集语音文本列表
    """
    index = 1 # 计数器
    # executor = ProcessPoolExecutor(max_workers=4)
    # futures = []
    texts = [] # 语音文本列表

    # 根据输入的 LJSpeech 数据集所在路径打开 metadata.csv 文件
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines(): # 按行读取语音文本内容 'LJ001-0001|Printing, in the...|...'
            if index % 100 == 0: # 每处理 100 个文件打印进度
                print("{:d} Done".format(index))
            parts = line.strip().split('|') # 删除前后空格后使用|将文本分割
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0]) # 第一段为语音路径 'data\\LJSpeech-1.1\\wavs\\LJ001-0001.wav'
            text = parts[2] # 第三段为语音文本 'Printing, in the onl...'
            # futures.append(executor.submit(
            #     partial(_process_utterance, out_dir, index, wav_path, text)))
            texts.append(_process_utterance(out_dir, index, wav_path, text)) # 处理音频文件，保存转换的 mel 谱图，添加语音文本至列表

            index = index + 1 # 计数器更新

    # return [future.result() for future in tqdm(futures)]
    return texts # 返回的是所有音频文件对应的文本数据

# 调用tools.py中的get_mel函数获取音频文件的mel谱图
def _process_utterance(out_dir, index, wav_path, text):
    """
    输入参数：
		out_dir		:	mel谱图输出路径
		index		:	计数器
		wav_path	:	音频文件路径
		text		:	文本内容
	输出参数：
		texts	:	LJSpeech数据集语音文本列表
    """
    # Compute a mel-scale spectrogram from the wav:
    # 将语音波形转换为 mel 频谱图:
    mel_spectrogram = audio.tools.get_mel(wav_path).numpy().astype(np.float32) # [D T] 80 832

    # Write the spectrograms to disk:
    # 将 mel 频谱图保存至文件
    mel_filename = 'ljspeech-mel-%05d.npy' % index # 'ljspeech-mel-00001.npy'
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    return text # 返回一个音频文件对应的文本内容 'Printing, in the on...'
