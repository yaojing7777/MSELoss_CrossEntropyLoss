from optim import Optimizer
import numpy as np

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
