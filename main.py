from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import LocalDataset
from loss import CrossEntropyLoss, MSELoss
from model import LinearRegression, LogisticRegression
from optim import SGD, AdaDelta, AdaGrad, Adam, Momentum, NAG, RMSProp

# ==== 超参数与路径配置 ====
CSV_PATH = "Social_Network_Ads.csv"
OUTPUT_DIR = "out_dir"
SEED = 42
BATCH_SIZE = 32
MODEL_TYPE = "logistic"  # 可选 "logistic"（逻辑回归，分类任务）或 "linear"（线性回归，回归任务）
USE_POLYNOMIAL = True  # True: 曲线边界/曲面拟合；False: 直线边界/平面拟合
OPTIMIZER_NAME = "AdaDelta"  # 可选 "SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "AdaDelta", "Adam"
MODELS = {
    "logistic": LogisticRegression,
    "linear": LinearRegression,
}
LOSSES = {
    "logistic": CrossEntropyLoss,
    "linear": MSELoss,
}
OPTIMIZERS = {
    "SGD": SGD,
    "Momentum": Momentum,
    "NAG": NAG,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
    "AdaDelta": AdaDelta,
    "Adam": Adam,
}
EPOCHS = 10000
PRINT_EVERY = 1000
PLOT_PATH = "shannon_result.png"
WEIGHTS_PATH = Path(__file__).parent / "runs" / "weights.pkl"
MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"


class Trainer:
    """
    小批量梯度下降训练模型：
    每个 batch 依次执行前向传播（模型输出、损失计算）与反向传播（参数梯度），
    逻辑回归用交叉熵，线性回归用均方误差。
    一个 epoch 内所有 batch 处理完后，记录整个 epoch 的平均损失。
    """

    def __init__(self, dataset, model, loss_fn, optimizer):
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.losses = []
        self.validation_losses = []

    @staticmethod
    def accuracy(model, x, y):
        """predict() 已在模型内部完成阈值判断，直接对比标签即可。"""
        predictions = model.predict(x)
        return float(np.mean(predictions == y))

    def train(self, epochs=10000, print_every=1000):
        for epoch in range(epochs):
            # 循环调用 LocalDataset.__iter__()
            # 用新的随机种子打乱 train_x / train_y，再切成 batch 逐个 yield，
            epoch_loss = 0.0
            sample_count = 0
            for batch_x, batch_y in self.dataset:
                # 前向传播：模型输出预测值（逻辑回归为概率，线性回归为连续值）。
                y_pred = self.model(batch_x)
                batch_loss = self.loss_fn.loss(y_pred, batch_y)
                epoch_loss += batch_loss * len(batch_x)
                sample_count += len(batch_x)

                # 反向传播：计算损失对模型参数的梯度，交给优化器更新。
                gradients = self.loss_fn.backward(batch_x, y_pred, batch_y)
                self.optimizer.zero_grad()
                self.optimizer.compute_delta(gradients)
                self.optimizer.step()

            train_loss = epoch_loss / sample_count
            # 验证集只做前向传播与损失计算，不更新参数。
            valid_pred = self.model(self.dataset.valid_x)
            validation_loss = self.loss_fn.loss(valid_pred, self.dataset.valid_y)
            self.losses.append(train_loss)
            self.validation_losses.append(validation_loss)

            if (epoch + 1) % print_every == 0:
                accuracy = self.accuracy(
                    self.model, self.dataset.valid_x, self.dataset.valid_y
                )
                print(
                    f"epoch={epoch + 1:5d}, "
                    f"loss={train_loss:.6f}, "
                    f"validation_accuracy={accuracy:.4f}"
                )
        return np.asarray(self.losses)


def plot_results(dataset, model, optimizer_name, losses, validation_losses, save_path):
    """画出决策边界和 loss 曲线，与训练逻辑解耦。"""
    if not len(losses):
        raise RuntimeError("请先调用 train() 再绘图")

    raw_train_x = dataset.train_x * dataset.std + dataset.mean
    raw_valid_x = dataset.valid_x * dataset.std + dataset.mean
    all_x = np.vstack((raw_train_x, raw_valid_x))
    age = np.linspace(all_x[:, 0].min() - 1, all_x[:, 0].max() + 1, 250)
    salary = np.linspace(all_x[:, 1].min() - 5000, all_x[:, 1].max() + 5000, 250)
    grid_age, grid_salary = np.meshgrid(age, salary)
    grid_raw = np.column_stack((grid_age.ravel(), grid_salary.ravel()))
    grid_output = model(dataset.normalize(grid_raw)).reshape(grid_age.shape)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].contourf(grid_age, grid_salary, grid_output, levels=30, cmap="RdBu", alpha=0.35)
    axes[0].contour(grid_age, grid_salary, grid_output, levels=[0.5], colors="black", linestyles="--")
    for label, marker, color in [(0, "x", "crimson"), (1, "o", "royalblue")]:
        selected = dataset.train_y == label
        points = raw_train_x[selected]
        axes[0].scatter(
            points[:, 0], points[:, 1], marker=marker, c=color,
            label="Did not Purchase" if label == 0 else "Purchased",
        )
    axes[0].set_title("Training Data")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Estimated Salary")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    epochs = np.arange(1, len(losses) + 1)
    axes[1].plot(epochs, losses, label="Training loss", color="tab:blue")
    axes[1].plot(epochs, validation_losses, label="Validation loss", color="tab:orange")
    axes[1].set_title(f"Cost History - {optimizer_name}")
    axes[1].set_xlabel("# iterations")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.show()


def main():
    if MODEL_TYPE not in MODELS:
        raise ValueError(f"MODEL_TYPE 必须是 {sorted(MODELS)} 之一")
    if OPTIMIZER_NAME not in OPTIMIZERS:
        raise ValueError(f"OPTIMIZER_NAME 必须是 {sorted(OPTIMIZERS)} 之一")

    dataset = LocalDataset(
        file_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
    )
    model = MODELS[MODEL_TYPE](use_polynomial=USE_POLYNOMIAL)
    loss_fn = LOSSES[MODEL_TYPE](model)
    optimizer = OPTIMIZERS[OPTIMIZER_NAME](model.parameters)
    trainer = Trainer(dataset, model, loss_fn, optimizer)

    print(f"模型类型：{MODEL_TYPE}（use_polynomial={USE_POLYNOMIAL}，loss={type(loss_fn).__name__}）")
    trainer.train(epochs=EPOCHS, print_every=PRINT_EVERY)
    plot_results(
        dataset, model, OPTIMIZER_NAME, trainer.losses, trainer.validation_losses, PLOT_PATH
    )

    test_accuracy = trainer.accuracy(model, dataset.test_x, dataset.test_y)
    print(f"训练集样本：{len(dataset.train_y)}，验证集样本：{len(dataset.valid_y)}，测试集样本：{len(dataset.test_y)}")
    print(f"测试集准确率：{test_accuracy:.4f}")
    print(f"模型参数 [w0, w1, ...]：{model.parameters}")

    # 保存参数与完整模型（含标准化参数），供 one_hot_test.py 等测试脚本加载。
    model.save(WEIGHTS_PATH, mean=dataset.mean, std=dataset.std)
    model.save(MODEL_PATH, mean=dataset.mean, std=dataset.std)
    print(f"参数已保存到：{WEIGHTS_PATH}")
    print(f"模型已保存到：{MODEL_PATH}")


if __name__ == "__main__":
    main()
