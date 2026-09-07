import numpy as np
from optim.common import Optimizer

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

