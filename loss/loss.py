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


class MSELoss(LossFunction):
    """
    均方误差 MSE = mean((y_pred - y_true)²)，配合 LinearRegression 使用。

    前向：线性回归直接输出连续值 y_pred = w^T x + b；
    反向：dLoss/dparam = 2/n * X^T (y_pred - y_true)。
    """

    def loss(self, y_pred, y_true):
        errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        return float(np.mean(errors ** 2))

    def backward(self, x, y_pred, y_true):
        errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        x_with_bias = self.model.design_matrix(x)
        return 2.0 * x_with_bias.T @ errors / len(x_with_bias)


class CrossEntropyLoss(LossFunction):
    """
    交叉熵损失 CE = -mean(y·log(p) + (1-y)·log(1-p))，配合 LogisticRegression 使用。

    前向：逻辑回归输出概率 y_pred = sigmoid(w^T x + b)；
    反向：sigmoid 与交叉熵组合后的梯度很简洁，dLoss/dparam = 1/n * X^T (y_pred - y_true)。
    """

    EPSILON = 1e-15

    def loss(self, y_pred, y_true):
        # 防止 log(0)，把概率裁剪到 [EPSILON, 1 - EPSILON]。
        probabilities = np.clip(
            np.asarray(y_pred, dtype=np.float64), self.EPSILON, 1.0 - self.EPSILON
        )
        sample_losses = -(
            y_true * np.log(probabilities)
            + (1.0 - y_true) * np.log(1.0 - probabilities)
        )
        return float(np.mean(sample_losses))

    def backward(self, x, y_pred, y_true):
        x_with_bias = self.model.design_matrix(x)
        return x_with_bias.T @ (y_pred - y_true) / len(x_with_bias)
