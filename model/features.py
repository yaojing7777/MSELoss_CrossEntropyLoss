"""模型侧的特征映射：把标准化后的两个原始特征扩展成设计矩阵所需的列。"""

import numpy as np


def polynomial_features(features):
    """
    把两个特征扩展成二次特征。输出为：
    [Age, Salary, Age², Salary², Age*Salary]，共 5 列。
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features 必须是形状为 (样本数, 2) 的数组")

    age = features[:, 0]
    salary = features[:, 1]
    return np.column_stack(
        (age, salary, age ** 2, salary ** 2, age * salary)
    )
