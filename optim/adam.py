from optim import Optimizer
import numpy as np

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
