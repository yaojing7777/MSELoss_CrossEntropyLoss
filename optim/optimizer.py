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


class SGD(Optimizer):
    """随机梯度下降，可选动量。"""

    def __init__(self, parameters, learning_rate=0.05, momentum=0.0):
        super().__init__(parameters, learning_rate=learning_rate)
        self.momentum = momentum
        self.velocity = np.zeros_like(parameters)

    def step(self):
        update = self.learning_rate * self.grad
        if self.momentum:
            self.velocity = self.momentum * self.velocity + update
            update = self.velocity
        # 参数减去更新量。
        self.parameters -= update


class Momentum(Optimizer):
    """动量梯度下降：累积历史梯度方向，压制震荡并加速收敛。"""

    def __init__(self, parameters, learning_rate=0.05, momentum=0.9):
        super().__init__(parameters, learning_rate=learning_rate)
        self.momentum = momentum
        self.velocity = np.zeros_like(parameters)

    def step(self):
        self.velocity = self.momentum * self.velocity + self.learning_rate * self.grad
        self.parameters -= self.velocity


class NAG(Optimizer):
    """
    Nesterov Accelerated Gradient：动量法的改进。
    参数更新按 cs231n 的等价形式展开：先按动量前瞻，再回到当前位置应用梯度修正。
    """

    def __init__(self, parameters, learning_rate=0.05, momentum=0.9):
        super().__init__(parameters, learning_rate=learning_rate)
        self.momentum = momentum
        self.velocity = np.zeros_like(parameters)

    def step(self):
        velocity_prev = self.velocity
        self.velocity = self.momentum * self.velocity - self.learning_rate * self.grad
        self.parameters += (
            -self.momentum * velocity_prev + (1 + self.momentum) * self.velocity
        )


class AdaGrad(Optimizer):
    """Adaptive Gradient：按历史梯度平方和缩放学习率。"""

    def __init__(self, parameters, learning_rate=0.05, epsilon=1e-8):
        super().__init__(parameters, learning_rate=learning_rate, epsilon=epsilon)
        self.gradient_square = np.zeros_like(parameters)

    def step(self):
        self.gradient_square += self.grad ** 2
        update = self.learning_rate * self.grad / np.sqrt(
            self.gradient_square + self.epsilon
        )
        self.parameters -= update


class RMSProp(Optimizer):
    """RMSProp：用指数滑动平均代替 AdaGrad 的梯度平方累加。"""

    def __init__(self, parameters, learning_rate=0.05, decay=0.95, epsilon=1e-8):
        super().__init__(parameters, learning_rate=learning_rate, decay=decay, epsilon=epsilon)
        self.gradient_square = np.zeros_like(parameters)

    def step(self):
        self.gradient_square = (
            self.decay * self.gradient_square
            + (1 - self.decay) * self.grad ** 2
        )
        update = self.learning_rate * self.grad / np.sqrt(
            self.gradient_square + self.epsilon
        )
        self.parameters -= update


class AdaDelta(Optimizer):
    """AdaDelta：无需全局学习率，用更新量的滑动平均自适应步长。"""

    def __init__(self, parameters, decay=0.95, epsilon=1e-8):
        super().__init__(parameters, decay=decay, epsilon=epsilon)
        self.gradient_square = np.zeros_like(parameters)
        self.update_square = np.zeros_like(parameters)

    def step(self):
        self.gradient_square = (
            self.decay * self.gradient_square
            + (1 - self.decay) * self.grad ** 2
        )
        update = (
            np.sqrt(self.update_square + self.epsilon)
            / np.sqrt(self.gradient_square + self.epsilon)
            * self.grad
        )
        self.update_square = (
            self.decay * self.update_square
            + (1 - self.decay) * update ** 2
        )
        self.parameters -= update


class Adam(Optimizer):
    """Adam：结合动量与 RMSProp，用一阶/二阶矩估计加偏差修正。"""

    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, learning_rate=learning_rate, epsilon=epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros_like(parameters)
        self.v = np.zeros_like(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * self.grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.parameters -= update
