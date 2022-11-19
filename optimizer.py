import numpy as np

# 该文件中封装了一个学习率优化类，其可以实现学习率动态变化和冻结两种更新方式

# 为学习率方案封装的一个包装器类
class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_steps):
        self._optimizer = optimizer # 优化器
        self.n_warmup_steps = n_warmup_steps # warmup的步数
        self.n_current_steps = current_steps # 训练时的当前步数
        self.init_lr = np.power(d_model, -0.5) # 学习率

    # 将学习率冻结之后，再进行参数更新
    def step_and_update_lr_frozen(self, learning_rate_frozen):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = learning_rate_frozen
        self._optimizer.step()

    # 使用设置的学习率方案进行参数更新
    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    # 返回当前的学习率
    def get_learning_rate(self):
        learning_rate = 0.0
        for param_group in self._optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    # 清除梯度
    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    # 学习率变化规则
    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    # 该学习方案中每步的学习率
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale() # 计算当前step的学习率
        # 给所有参数设置学习率
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
