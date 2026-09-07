# work_one —— 从零实现的线性回归与逻辑回归

不依赖任何机器学习框架，仅用 NumPy 手写实现线性回归（LinearRegression）与逻辑回归（LogisticRegression），
包括模型定义、损失函数（前向/反向传播）、7 种梯度下降优化器、数据集切分与标准化、训练可视化与模型保存/加载。

数据集根据 `main.py` 中的 `CSV_PATH` 自行配置。

## 项目结构

```
work_one/
├── main.py                  # 训练入口：配置超参数、前向/反向传播训练、绘图、保存模型
├── one_hot_test.py          # 推理脚本：加载已训练的 model.pkl 做预测与评估
├── Social_Network_Ads.csv   # 原始数据（Age, EstimatedSalary, Purchased）
├── requirements.txt         # conda 环境依赖清单
├── model/                   # 模型层：模型定义与特征映射
│   ├── __init__.py          # 包导出：BaseModel / LinearRegression / LogisticRegression / load_model / polynomial_features
│   ├── model.py             # BaseModel 基类 + LinearRegression + LogisticRegression + MODEL_CLASSES + load_model
│   ├── features.py          # polynomial_features：模型侧的二次特征映射
│   └── model.pkl            # 训练后保存的完整模型（含标准化参数）
├── loss/                    # 损失函数层：规范的前向 loss() / 反向 backward() 接口
│   ├── __init__.py          # 包导出
│   ├── common.py            # LossFunction 基类
│   ├── mseloss.py           # MSELoss（配合线性回归）
│   └── crossentropyloss.py  # CrossEntropyLoss（配合逻辑回归）
├── optim/                   # 优化器层：一个优化器一个文件
│   ├── __init__.py          # 包导出全部优化器
│   ├── common.py            # Optimizer 基类（zero_grad / compute_delta / step）
│   ├── sgd.py               # SGD（可选动量）
│   ├── momentum.py          # Momentum
│   ├── nag.py               # NAG（Nesterov 加速梯度）
│   ├── adagrad.py           # AdaGrad
│   ├── rmsprop.py           # RMSProp
│   ├── adadelta.py          # AdaDelta
│   └── adam.py              # Adam
├── data/                    # 数据层：只负责数据 IO 与预处理
│   ├── __init__.py          # 包导出 LocalDataset
│   └── dataset.py           # LocalDataset：读 CSV、7:2:1 切分、标准化、按 batch 迭代
├── runs/
│   └── weights.pkl          # 训练后保存的参数副本
├── out_dir/                 # 训练时自动生成：切分出的 train / validation / test CSV
└── shannon_result.png       # 训练后生成：决策边界 + 损失曲线图
```

分层依赖方向（单向，无循环导入）：

```
main.py ──> model / loss / optim / data
model.model ──> model.features
loss.* ──> loss.common        optim.* ──> optim.common
```

## 环境要求

- Python 3.12（conda 环境）
- NumPy、Matplotlib

```bash
# 方式一：按 requirements.txt 重建 conda 环境
conda create --name work_one --file requirements.txt

# 方式二：已有环境直接装两个依赖
pip install numpy matplotlib
```

## 快速开始

### 1. 训练

```bash
python main.py
```

按 `main.py` 顶部的配置区块选择模型与超参数，训练结束后：

- 弹出并保存 `shannon_result.png`（左图：决策边界与训练样本；右图：训练/验证损失曲线）；
- 打印验证集准确率、测试集准确率与最终参数；
- 模型保存到 `runs/weights.pkl` 和 `model/model.pkl`。

### 2. 推理

```bash
python one_hot_test.py
```

加载 `model/model.pkl`，对几条手工输入的 `[Age, Salary]` 打印预测结果，
并在测试集上评估加载模型的准确率。

## 配置说明（main.py 顶部）

| 配置项 | 可选值 | 说明 |
|---|---|---|
| `MODEL_TYPE` | `"logistic"` / `"linear"` | 逻辑回归（分类）或线性回归（回归演示） |
| `USE_POLYNOMIAL` | `True` / `False` | `True`：5 个二次特征，0.5 决策边界是曲线；`False`：2 个原始特征，边界是直线 |
| `OPTIMIZER_NAME` | `"SGD"` `"Momentum"` `"NAG"` `"AdaGrad"` `"RMSProp"` `"AdaDelta"` `"Adam"` | 参数更新算法 |
| `BATCH_SIZE` | 正整数 | 小批量梯度下降的 batch 大小 |
| `EPOCHS` | 正整数 | 训练轮数（默认 10000） |
| `PRINT_EVERY` | 正整数 | 每多少轮打印一次日志 |
| `SEED` | 整数 | 随机种子，保证数据切分与打乱可复现 |

> 模型与损失自动配对：`logistic → CrossEntropyLoss`，`linear → MSELoss`，无需手动指定。

## 核心设计

### 前向 / 反向传播

损失函数统一遵循 `LossFunction` 基类的前向/反向接口，`Trainer` 中每个 batch 的训练循环：

```python
y_pred = model(batch_x)                             # 前向传播：模型输出
loss = loss_fn.loss(y_pred, batch_y)                # 前向：计算损失
grads = loss_fn.backward(batch_x, y_pred, batch_y)  # 反向：计算梯度
optimizer.zero_grad()
optimizer.compute_delta(grads)
optimizer.step()                                    # 更新参数
```

验证集只做前向（`loss_fn.loss`），不计算梯度、不更新参数。

### 模型：BaseModel 与两个子类

两个模型共用基类 `BaseModel`（参数、设计矩阵、logits、保存/加载），
核心都是线性部分 `logits = w^T x + b`，区别只在输出层：

| | `LinearRegression` | `LogisticRegression` |
|---|---|---|
| 输出 | `w^T x + b` 连续值，可越界 | `sigmoid(w^T x + b)`，(0, 1) 之间，可解释为概率 |
| 任务 | 回归 | 二分类 |
| 损失 | MSE（均方误差） | 交叉熵 |
| `predict()` | 预测值 `>= 0.5` 判为 1（仅供演示） | 概率 `>= threshold`（默认 0.5）判为 1 |

### 在代码中使用

```python
import numpy as np
from model import LinearRegression, LogisticRegression, load_model, polynomial_features

# 新建模型（use_polynomial 控制是否使用二次特征）
model = LogisticRegression(use_polynomial=True)

# 前向输出：逻辑回归返回概率，线性回归返回连续值
probs = model(normalized_x)

# 二值预测
labels = model.predict(normalized_x, threshold=0.5)

# 保存 / 加载（mean、std 为训练集标准化参数，推理时需要）
model.save("model/model.pkl", mean=mean, std=std)
model = load_model("model/model.pkl")  # 按文件里的 model_type 自动还原成对应的类
```

## 优化器一览（optim/）

每个优化器一个文件，均继承 `optim/common.py` 的 `Optimizer` 基类，
实现统一的 `zero_grad()` / `compute_delta()` / `step()` 接口：

| 优化器 | 文件 | 思想 | 默认超参数 |
|---|---|---|---|
| `SGD` | `sgd.py` | 随机梯度下降，可选动量 | `learning_rate=0.05` |
| `Momentum` | `momentum.py` | 累积历史梯度方向，压制震荡 | `learning_rate=0.05, momentum=0.9` |
| `NAG` | `nag.py` | Nesterov 加速梯度，先前瞻再修正 | `learning_rate=0.05, momentum=0.9` |
| `AdaGrad` | `adagrad.py` | 按历史梯度平方和缩放学习率 | `learning_rate=0.05` |
| `RMSProp` | `rmsprop.py` | 指数滑动平均代替 AdaGrad 累加 | `learning_rate=0.05, decay=0.95` |
| `AdaDelta` | `adadelta.py` | 无需全局学习率，自适应步长 | `decay=0.95` |
| `Adam` | `adam.py` | 动量 + RMSProp，一/二阶矩加偏差修正 | `learning_rate=0.001, beta1=0.9, beta2=0.999` |

## 数据说明

`Social_Network_Ads.csv` 需包含三列：`Age`、`EstimatedSalary`、`Purchased`（0/1）。
`LocalDataset`（`data/dataset.py`）在初始化时自动完成：

1. 读取并校验 CSV 列名；
2. 按固定种子随机打乱，切分为 70% 训练 / 20% 验证 / 10% 测试，并把未标准化的切分结果写入 `out_dir/`；
3. 用**训练集**统计均值与标准差做标准化（避免验证/测试集信息泄露）；
4. `__iter__` 按 `batch_size` 产出小批量，每个 epoch 重新打乱。

多项式特征展开（`model/features.py` 的 `polynomial_features`）发生在模型侧：
标准化后的 `[Age, Salary]` 进入模型前被扩展为 `[Age, Salary, Age², Salary², Age*Salary]`。
