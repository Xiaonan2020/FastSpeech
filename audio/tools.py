""" from https://github.com/NVIDIA/tacotron2 """
# 本文件主要基于上述的三个文件实现mel数据抽取、保存的具体功能
import torch
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write

import audio.stft as stft
import audio.hparams_audio as hparams
from audio.audio_processing import griffin_lim

# 初始化stft.py中构建的短时傅里叶变化/STFT的模块
_stft = stft.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)

# 根据音频文件路径加载数据
def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path) # 调用scipy中的音频读取接口
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

# 从音频文件中获取对应的mel谱图
def get_mel(filename):
    audio, sampling_rate = load_wav_to_torch(filename) # 加载一条音频文件
    if sampling_rate != _stft.sampling_rate: # 音频文件的采样频率要与设置的频率一致
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0) # [1, L]
    # 将音频数据文件设置为不需要求梯度
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = _stft.mel_spectrogram(audio_norm) # 获取mel谱图 [1 80 832]
    melspec = torch.squeeze(melspec, 0) # [D T] [80 832]
    # melspec = torch.from_numpy(_normalize(melspec.numpy()))

    return melspec

# 直接从音频文件中获取对应的mel谱图
def get_mel_from_wav(audio):
    sampling_rate = hparams.sampling_rate
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    return melspec

# 从mel谱图中重建音频文件
def inv_mel_spec(mel, out_filename, griffin_iters=60):
    mel = torch.stack([mel])
    # mel = torch.stack([torch.from_numpy(_denormalize(mel.numpy()))])

    # 将mel谱图解压，抽取对应的线性谱图
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    # 基于线性谱图使用griffin_lim算法重建音频文件
    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), _stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, hparams.sampling_rate, audio)# 保存音频文件
