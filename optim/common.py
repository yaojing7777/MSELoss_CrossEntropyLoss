import numpy as np

class Optimizer:
    """
    优化器父类：负责持有模型参数和当前 batch 的梯度，
    具体的参数更新规则由子类实现 step()。
    """

    def __init__(self, parameters, learning_rate=0.05, decay=0.95, epsilon=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.grad = np.zeros_like(parameters)

    def zero_grad(self):
        """清空上一个 batch 保存的梯度。"""
        self.grad.fill(0.0)

    def compute_delta(self, gradients):
        """保存当前 batch 计算出的平均梯度。"""
        self.grad[:] = gradients

    def step(self):
        """使用当前 batch 的梯度更新模型参数，由子类实现。"""
        raise NotImplementedError("子类必须实现 step()")

