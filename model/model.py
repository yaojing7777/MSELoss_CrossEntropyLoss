import pickle
from pathlib import Path

import numpy as np

from model.features import polynomial_features


class BaseModel:
    """
    线性模型基类：统一管理参数、设计矩阵、logits 计算与保存/加载，
    子类只需实现 __call__()（模型输出）和 predict()。

    两者的共同点都是线性部分 logits = w^T x + b，区别只在输出层：
    - LinearRegression 直接输出 logits 的连续值（回归任务）；
    - LogisticRegression 把 logits 过 sigmoid 变成概率（分类任务）。
    """

    # 子类用 model_type 标识自己，保存时写入 pkl，加载时据此还原具体类。
    model_type = "base"

    def __init__(self, feature_count, use_polynomial=True):
        self.use_polynomial = use_polynomial
        # parameters[0] 是偏置，其余参数对应各个特征。
        self.parameters = np.zeros(feature_count + 1, dtype=np.float64)
        # 保存/加载模型时写入的标准化参数，推理时对原始输入直接 normalize。
        self.mean = None
        self.std = None

    def design_matrix(self, x):
        """生成带偏置项的设计矩阵 X。"""
        if self.use_polynomial:
            expanded_x = polynomial_features(x)
        else:
            expanded_x = np.asarray(x, dtype=np.float64)
        return np.column_stack((np.ones(len(expanded_x)), expanded_x))

    def logits(self, x):
        """计算线性部分 w^T x + b。"""
        return self.design_matrix(x) @ self.parameters

    def __call__(self, x):
        raise NotImplementedError("子类必须实现 __call__()")

    def predict(self, x, threshold=0.5):
        raise NotImplementedError("子类必须实现 predict()")

    def save(self, path, mean=None, std=None):
        """把参数、模型类型和标准化参数一起序列化到 path（如 model/model.pkl）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_type": self.model_type,
            "parameters": self.parameters,
            "use_polynomial": self.use_polynomial,
            "mean": mean,
            "std": std,
        }
        with path.open("wb") as file:
            pickle.dump(state, file)

    @classmethod
    def load(cls, path):
        """按文件里保存的 model_type 还原成对应子类，返回类型以文件内容为准。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到模型文件：{path}")
        with path.open("rb") as file:
            state = pickle.load(file)

        # 旧版本保存的 pkl 没有 model_type 字段，视为逻辑回归，保持向后兼容。
        model_type = state.get("model_type", "logistic")
        target_class = MODEL_CLASSES.get(model_type)
        if target_class is None:
            raise ValueError(f"未知的模型类型：{model_type}")

        model = target_class(use_polynomial=state["use_polynomial"])
        model.parameters = np.asarray(state["parameters"], dtype=np.float64)
        model.mean = state["mean"]
        model.std = state["std"]
        return model


class LinearRegression(BaseModel):
    """
    线性回归模型：直接输出 w^T x + b 的连续值，不做 sigmoid 压缩。
    训练时配合 MSELoss（均方误差），用于回归任务。
    use_polynomial=True  : 特征为 [Age, Salary, Age², Salary², Age*Salary]，共 5 个特征 + 1个偏置。
    use_polynomial=False : 特征只有 [Age, Salary]，共 2 个特征 + 1 个偏置。
    """

    model_type = "linear"

    def __init__(self, use_polynomial=True):
        super().__init__(
            feature_count=5 if use_polynomial else 2, use_polynomial=use_polynomial
        )

    def __call__(self, x):
        """返回连续的回归预测值。"""
        return self.logits(x)

    def predict(self, x, threshold=0.5):
        """线性回归本身输出连续值；按阈值切分类别仅供把最小二乘当分类器演示时使用。"""
        return (self(x) >= threshold).astype(np.int64)


class LogisticRegression(BaseModel):
    """
    逻辑回归模型：sigmoid(w^T x + b) 输出 0~1 的概率。
    训练时配合 CrossEntropyLoss（交叉熵），用于二分类任务。
    use_polynomial=True  : 特征为 [Age, Salary, Age², Salary², Age*Salary]，共 5 个特征 + 1个偏置，0.5 决策边界是曲线。
    use_polynomial=False : 特征只有 [Age, Salary]，共 2 个特征 + 1 个偏置，0.5 决策边界是直线。
    """

    model_type = "logistic"

    def __init__(self, use_polynomial=True):
        super().__init__(
            feature_count=5 if use_polynomial else 2, use_polynomial=use_polynomial
        )

    @staticmethod
    def _sigmoid(z):
        """数值稳定的 sigmoid：先把输入裁剪到 [-700, 700] 再取倒数，避免 exp 溢出。"""
        z = np.clip(z, -700, 700)
        return 1.0 / (1.0 + np.exp(-z))

    def __call__(self, x):
        """返回购买概率。"""
        return self._sigmoid(self.logits(x))

    def predict(self, x, threshold=0.5):
        """概率 >= threshold 判为 1（购买），否则判为 0。"""
        return (self(x) >= threshold).astype(np.int64)


# model_type -> 具体类的映射，定义在两个子类之后，供 BaseModel.load 还原实例。
MODEL_CLASSES = {
    "linear": LinearRegression,
    "logistic": LogisticRegression,
}


def load_model(path):
    """从 pkl 文件加载模型，自动还原成 LinearRegression 或 LogisticRegression。"""
    return BaseModel.load(path)
