import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math

from model import FastSpeech
from loss import DNNLoss
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
from optimizer import ScheduledOptim
import hparams as hp
import utils

# 该文件是FastSpeech模型训练过程实现代码，整体流程与普通模型训练一样，
# 需要注意的一点就是数据划分过程中，是分成了一个大batch，其中包含数个real batch，
# 故训练过程是三个for循环的嵌套，与正常的两个for循环嵌套不同，
# 该现象也可以在dataset.py文件中观察到



def main(args):
    """
    该函数的作用是：为 FastSpeech 模型进行训练
    输入参数：
		args:可控训练参数
    """

    # Get device 设置计算使用的设备，可按实际情况决定
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model 定义模型
    print("Use FastSpeech")
    # 将模型并行训练并移入计算设备中
    model = nn.DataParallel(FastSpeech()).to(device)
    print("Model Has Been Defined")
    # 计算模型参数量
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)
    # Get buffer # 获取训练元数据
    print("Load data to buffer")
    buffer = get_data_to_buffer() # [{'text':[1 2 ... 22], 'duration':[23 4 ...], 'mel_target':[...]}, { ... }, { ... }, ...]

    # Optimizer and loss
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 betas=(0.9, 0.98),
                                 eps=1e-9)
    scheduled_optim = ScheduledOptim(optimizer,
                                     hp.decoder_dim, # 256
                                     hp.n_warm_up_step, # 4000
                                     args.restore_step)
    # 设置损失函数
    fastspeech_loss = DNNLoss().to(device)
    print("Defined Optimizer and Loss Function.")

    # Load checkpoint if exists #设置断点训练恢复
    try:
        # 加载训练模型
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        # 更新模型与优化器的参数
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger # 检查日志地址
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Get dataset # 将训练元数据转换为训练数据集
    dataset = BufferDataset(buffer)

    # Get Training Loader # 为训练数据集创建数据加载器
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)
    # 计算总训练步数
    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training # 将模型置于训练状态
    model = model.train()

    # 开始训练
    for epoch in range(hp.epochs):
        for i, batchs in enumerate(training_loader): # 此处是一个大batch
            # real batch start here # 进一步划分批数据
            for j, db in enumerate(batchs): # db才是一个real batch
                start_time = time.perf_counter()

                # 计算当前训练次数
                current_step = i * hp.batch_expand_size + j + args.restore_step + \
                    epoch * len(training_loader) * hp.batch_expand_size + 1

                # Init # 初始优化器
                scheduled_optim.zero_grad()

                # Get Data # 获取数据，移入计算设备
                character = db["text"].long().to(device)
                mel_target = db["mel_target"].float().to(device)
                duration = db["duration"].int().to(device)
                mel_pos = db["mel_pos"].long().to(device)
                src_pos = db["src_pos"].long().to(device)
                max_mel_len = db["mel_max_len"]

                # Forward # 输入训练数据，前向传播
                mel_output, mel_postnet_output, duration_predictor_output = model(character,
                                                                                  src_pos,
                                                                                  mel_pos=mel_pos,
                                                                                  mel_max_length=max_mel_len,
                                                                                  length_target=duration)

                # Cal Loss # 计算预测的 mel 谱图与持续时间的损失
                mel_loss, mel_postnet_loss, duration_loss = fastspeech_loss(mel_output,
                                                                            mel_postnet_output,
                                                                            duration_predictor_output,
                                                                            mel_target,
                                                                            duration)
                total_loss = mel_loss + mel_postnet_loss + duration_loss

                # Logger # 从张量中获取数据值
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = duration_loss.item()

                # 将计算出的损失在日志文件中记录
                with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")

                with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")

                with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")

                with open(os.path.join("logger", "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")

                # Backward # 反向传播
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion # 梯度剪枝
                nn.utils.clip_grad_norm_(
                    model.parameters(), hp.grad_clip_thresh)

                # Update weights # 更新模型权重参数 可是否设置学习率冻结
                if args.frozen_learning_rate:
                    scheduled_optim.step_and_update_lr_frozen(
                        args.learning_rate_frozen)
                else:
                    scheduled_optim.step_and_update_lr()

                # Print # 打印训练结果至终端与日志
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f};".format(
                        m_l, m_p_l, d_l)
                    str3 = "Current Learning Rate is {:.6f}.".format(
                        scheduled_optim.get_learning_rate())
                    str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    print(str4)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write(str4 + "\n")
                        f_logger.write("\n")

                # 保存当前模型参数
                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                # 记录结束时间，计算平均用时
                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--restore_step', type=int, default=3000)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
