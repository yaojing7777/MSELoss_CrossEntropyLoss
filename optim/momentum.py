from optim import Optimizer
import numpy as np

class Momentum(Optimizer):
    """动量梯度下降：累积历史梯度方向，压制震荡并加速收敛。"""

    def __init__(self, parameters, learning_rate=0.05, momentum=0.9):
        super().__init__(parameters, learning_rate=learning_rate)
        self.momentum = momentum
        self.velocity = np.zeros_like(parameters)

    def step(self):
        self.velocity = self.momentum * self.velocity + self.learning_rate * self.grad
        self.parameters -= self.velocity

