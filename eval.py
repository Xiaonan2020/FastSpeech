import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
import audio
import utils
import dataset
import text
import model as M
import waveglow

# 评估文件中就是使用训练好的FastSpeech模型基于文本预测mel谱图，
# 然后使用grif-lim算法和Waveglow模型生成音频



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    """
    该函数的作用是：根据训练次数加载已训练的模型
    输入参数：
		num	:需要恢复的模型训练步数
    输出参数：
		model:恢复后的模型
    """
    # 组装要恢复的模型路径
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    # 将模型转换为并行训练模型并移入计算设备中
    model = nn.DataParallel(M.FastSpeech()).to(device)
    # 更新模型参数
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path))['model'])
    # 将模型置于生成状态并返回
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    """
    该函数的作用是： 根据文本使用已训练模型合成 mel 谱图
    输入参数：
		model:已训练模型
		text:语音文本
		alpha:合成语音速度参数
    """
    # 转换语音文本格式
    text = np.array(phn)
    text = np.stack([text])
    # 创建同长度索引数组
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    # 将文本与索引移入计算设备
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()

    # 在不计算梯度的条件下执行语音合成，得到 mel 谱图
    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha) # mel谱图预测
    # 将得到的 mel 谱图移入不同的计算设备并返回
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    """
    该函数的作用是： 获取待合成的语音文本数据
    输出参数：
		data_list	:	语音文本列表
    """
    # 设置一系列待合成的语音文本
    test1 = "I am very happy to see you again!"
    test2 = "Durian model is a very good speech synthesis!"
    test3 = "When I was twenty, I fell in love with a girl."
    test4 = "I remove attention module in decoder and use average pooling to implement predicting r frames at once"
    test5 = "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted."
    test6 = "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
    # 创建数据列表
    data_list = list()
    # 将语音文本转换为字符序列添加进数据列表
    data_list.append(text.text_to_sequence(test1, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test2, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test3, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test4, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test5, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test6, hp.text_cleaners))
    # 返回数据列表
    return data_list


if __name__ == "__main__":
    # 该函数的作用是： 执行语音转换

    # Test

    # 获取预训练的 WaveGlow 模型
    # WaveGlow = utils.get_WaveGlow() # 加载Waveglow作为声码器

    # 设置 FastSpeech 模型恢复步数与说话速度参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    print("use griffin-lim and waveglow")
    # 恢复已训练的 FastSpeech 模型
    model = get_DNN(args.step) # 加载FastSpeech模型
    # 获取语音文本列表
    data_list = get_data() # 加载文本数据
    for i, phn in enumerate(data_list):
        # 依次合成 mel 谱图
        mel, mel_cuda = synthesis(model, phn, args.alpha)
        if not os.path.exists("results"):
            os.mkdir("results")
        # 使用 griffin-lim 还原语音波形
        audio.tools.inv_mel_spec(
            mel, "results/"+str(args.step)+"_"+str(i)+".wav")

        # 使用 waveglow 还原语音波形
        # waveglow.inference.inference(
        #     mel_cuda, WaveGlow,
        #     "results/"+str(args.step)+"_"+str(i)+"_waveglow.wav")

        print("Done", i + 1)

    # 记录开始时间
    s_t = time.perf_counter()
    for i in range(100):
        for _, phn in enumerate(data_list):
            # 执行语音合成
            _, _, = synthesis(model, phn, args.alpha)
        print(i)
    # 记录结束时间
    e_t = time.perf_counter()
    # 记录合成 mel 谱图平均用时
    print((e_t - s_t) / 100.)
