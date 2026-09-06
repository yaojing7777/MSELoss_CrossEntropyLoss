import numpy as np

"""
把两个标准化特征扩展成二次特征。输出为：
[Age, Salary, Age², Salary², Age*Salary]。
"""


def polynomial_features(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features 必须是形状为 (样本数, 2) 的数组")

    age = features[:, 0]
    salary = features[:, 1]
    return np.column_stack(
        (age, salary, age ** 2, salary ** 2, age * salary)
    )
