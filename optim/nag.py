from optim import Optimizer
import numpy as np

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
