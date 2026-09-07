from optim import Optimizer
import numpy as np

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
