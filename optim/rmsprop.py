from optim import Optimizer
import numpy as np

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