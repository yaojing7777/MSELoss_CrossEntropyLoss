import numpy as np
from loss.common import LossFunction

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
