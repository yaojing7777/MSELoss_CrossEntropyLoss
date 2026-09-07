import numpy as np

class LossFunction:
    """
    损失函数基类：规范前向/反向接口。

    - loss(y_pred, y_true)：前向传播的最后一步，接收模型输出与真实标签，返回 batch 平均损失；
    - backward(x, y_pred, y_true)：反向传播，返回损失对模型参数的梯度，形状与 model.parameters 一致。

    子类持有 model 引用，是因为求梯度需要通过 model.design_matrix(x)
    生成带偏置的设计矩阵（用哪些特征由模型自己决定）。
    """

    def __init__(self, model):
        self.model = model

    def loss(self, y_pred, y_true):
        """根据模型输出与真实标签计算平均损失。"""
        raise NotImplementedError("子类必须实现 loss()")

    def backward(self, x, y_pred, y_true):
        """计算损失对模型参数的梯度。"""
        raise NotImplementedError("子类必须实现 backward()")

