import torch
import torch.nn as nn

# FastSpeech在训练时会对duration predictor同时训练，结合之前自回归模型均会对最后mel经过postnet处理的前后计算损失，故训练过程中会计算三个损失。loss.py文件中就定义了损失类

# 自定义的损失，由两种损失组成，一种分为三块，mel谱图损失分由两圈，即和之前的模型一样，postnet前后都计算损失
class DNNLoss(nn.Module):
    def __init__(self):
        super(DNNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target):
        mel_target.requires_grad = False # 目标信息不需要计算梯度，此处计算mel损失
        mel_loss = self.mse_loss(mel, mel_target) # postnet之前的损失
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)  # postnet之后的损失

        duration_predictor_target.requires_grad = False  # 目标信息不需要计算梯度，此处计算音素持续时间损失
        # 训练duration predictor的损失
        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        return mel_loss, mel_postnet_loss, duration_predictor_loss
