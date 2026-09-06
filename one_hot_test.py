"""加载训练好的模型做推理/评估。

用法：
    python main.py           # 先训练并保存 model.pkl
    python one_hot_test.py   # 再加载模型做预测
"""
from pathlib import Path

import numpy as np

from data import LocalDataset
from model import LogisticRegression, load_model

MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"
CSV_PATH = "Social_Network_Ads.csv"  # 数据文件不在时自动跳过测试集评估


def compute_accuracy(model, x, y):
    """predict() 已在模型内部完成阈值判断（概率/预测值 >= 0.5 判为 1）。"""
    predictions = model.predict(x)
    return float(np.mean(predictions == y))


def demo_predictions(model):
    """对几条手工输入的 [Age, Salary] 做预测。"""
    samples = np.array(
        [[32, 127000], [42, 43000], [55, 150000], [25, 20000]],
        dtype=np.float64,
    )
    if model.mean is not None and model.std is not None:
        normalized = (samples - model.mean) / model.std
    else:
        normalized = samples

    outputs = model(normalized)
    labels = model.predict(normalized)
    print("\n单条预测（Age, Salary -> 模型输出 -> 结论）：")
    for (age, salary), output, label in zip(samples, outputs, labels):
        if isinstance(model, LogisticRegression):
            outcome = "购买" if label else "不购买"
            print(f"  Age={age:6.0f}  Salary={salary:8.0f} -> p={output:.4f} -> {outcome}")
        else:
            outcome = "偏购买" if output >= 0.5 else "偏不购买"
            print(f"  Age={age:6.0f}  Salary={salary:8.0f} -> y_hat={output:.4f} -> {outcome}")


def evaluate_test_set(model):
    """CSV 存在时，重新切分数据并在测试集上评估准确率。"""
    if not Path(CSV_PATH).exists():
        print(f"未找到 {CSV_PATH}，跳过测试集评估。")
        return

    dataset = LocalDataset(
        file_path=CSV_PATH, output_dir="cest_seesion", seed=42, batch_size=40
    )
    accuracy = compute_accuracy(model, dataset.test_x, dataset.test_y)
    print(f"\n测试集样本：{len(dataset.test_y)}")
    print(f"加载模型的测试集准确率：{accuracy:.4f}")


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"找不到 {MODEL_PATH}，请先运行 python main.py"
        )

    model = load_model(MODEL_PATH)
    print(f"已加载模型：{MODEL_PATH}")
    print(f"模型类型：{type(model).__name__}，use_polynomial={model.use_polynomial}")
    print(f"模型参数 [w0, w1, ...]：{model.parameters}")

    demo_predictions(model)
    evaluate_test_set(model)


if __name__ == "__main__":
    main()
