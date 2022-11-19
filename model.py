import torch
import torch.nn as nn
import hparams as hp
import utils

from transformer.Models import Encoder, Decoder
from transformer.Layers import Linear, PostNet
from modules import LengthRegulator, CBHG


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 本文件基于前面所有的模块实现完成FastSpeech模型搭建


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()  # Length Regulator之前网络为编码器
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder() # Length Regulator之后网络为解码器

        self.mel_linear = Linear(hp.decoder_dim, hp.num_mels)
        self.postnet = CBHG(hp.num_mels, K=8,
                            projections=[256, hp.num_mels]) # 使用CBHG作为后处理网络
        self.last_linear = Linear(hp.num_mels * 2, hp.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0] # real_mel_len
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length) # 填充部分置为true [4 764]
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1)) # [ 4 764 80]
        return mel_output.masked_fill(mask, 0.) # 编码器输出，[b, max_sequence_len, encoder_dim] [ 4 764 80]

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        encoder_output, _ = self.encoder(src_seq, src_pos) # [4 114 256]

        if self.training: # 训练
            # length_regulator_output是长度调整后的音素序列，与音素谱图序列长度对齐，[b, max_mel_len, encoder_dim]
            # duration_predictor_output是训练过程中duration predictor计算数据的音素持续时间，[b, max_sequence_len]
            # 训练过程中，使用预先提取好的音素持续时间target作为监督信息训练duration predictor
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output,
                                                                                       target=length_target,
                                                                                       alpha=alpha,
                                                                                       mel_max_length=mel_max_length)
            # 解码器输出，mel谱图尺寸的数据，[b, max_mel_len, num_mels]
            decoder_output = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output) # [b, max_mel_len, num_mels]
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length) # 将pad部分的数据置0 # [b, max_mel_len, num_mels]
            residual = self.postnet(mel_output)   # [b, max_mel_len, num_mels * 2] 经过双向GRU 80 * 2
            residual = self.last_linear(residual) # [b, max_mel_len, num_mels]
            mel_postnet_output = mel_output + residual # [b, max_mel_len, num_mels]
            mel_postnet_output = self.mask_tensor(mel_postnet_output,
                                                  mel_pos,
                                                  mel_max_length) # [b, max_mel_len, num_mels]

            return mel_output, mel_postnet_output, duration_predictor_output # 用于训练duration predictor [4 764 80] [4 764 80] [ 4 114]
        else: # 推理
            # 推理时直接使用训练后duration predictor输出的音素持续时间进行音素序列长度调整
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output,
                                                                         alpha=alpha)

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output


if __name__ == "__main__":
    # Test
    model = FastSpeech()
    print(sum(param.numel() for param in model.parameters()))
