import numpy as np
from loss.common import LossFunction


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

