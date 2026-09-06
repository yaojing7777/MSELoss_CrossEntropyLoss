import csv
from pathlib import Path

import numpy as np

from data.features import polynomial_features  # re-export 供外部统一从 data 导入


class LocalDataset:
    # 7:2:1随机
    def __init__(
        self,
        file_path="Social_Network_Ads.csv",
        output_dir="outdir_data",
        seed=42,
        batch_size=20,
    ):
        self.csv_path = Path(file_path)
        self.output_dir = Path(output_dir)
        self.seed = seed
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.batch_size = batch_size
        # 每次 for 循环（即每个 epoch）递增一次，让打乱结果随轮次变化。
        self.epoch_counter = 0
        self.feature_names = ["Age", "EstimatedSalary"]
        self.target_name = "Purchased"

        self.x, self.y = self._read_csv()
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = (
            self._split_data()
        )
        self._save_splits()

        # 标准化参数只能由训练集计算，避免验证集和测试集的信息泄露。
        self.mean = self.train_x.mean(axis=0)
        self.std = self.train_x.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.train_x = self.normalize(self.train_x)
        self.valid_x = self.normalize(self.valid_x)
        self.test_x = self.normalize(self.test_x)

    def _read_csv(self):
        """读取CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"找不到数据文件：{self.csv_path}")

        features = []
        labels = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = set(self.feature_names + [self.target_name])
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("CSV 必须包含 Age、EstimatedSalary、Purchased 三列")
            for row in reader:
                features.append([float(row[name]) for name in self.feature_names])
                labels.append(int(row[self.target_name]))

        return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)

    def _split_data(self):
        """先随机打乱，再按 70%、20%、10% 切分，保证实验可复现。"""
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(self.x))
        train_end = int(len(indices) * 0.7)
        valid_end = train_end + int(len(indices) * 0.2)

        train_indices = indices[:train_end]
        valid_indices = indices[train_end:valid_end]
        test_indices = indices[valid_end:]
        return (
            self.x[train_indices], self.y[train_indices],
            self.x[valid_indices], self.y[valid_indices],
            self.x[test_indices], self.y[test_indices],
        )

    def _save_splits(self):
        """把未标准化的原始特征保存下来，便于检查和复用。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        datasets = {
            "train.csv": (self.train_x, self.train_y),
            "validation.csv": (self.valid_x, self.valid_y),
            "test.csv": (self.test_x, self.test_y),
        }
        for filename, (features, labels) in datasets.items():
            with (self.output_dir / filename).open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.feature_names + [self.target_name])
                writer.writerows(
                    [f"{age:g}" for age in row] + [int(label)]
                    for row, label in zip(features, labels)
                )

    def normalize(self, features):
        """使用训练集的均值和标准差进行标准化。"""
        return (features - self.mean) / self.std

    def __len__(self):
        return len(self.train_x)

    def __iter__(self):
        """
        训练时按 batch 返回数据；最后一个 batch 可以不足 batch_size。
        每次迭代重新打乱训练集，
        """
        rng = np.random.default_rng(self.seed + self.epoch_counter)
        self.epoch_counter += 1
        shuffled_indices = rng.permutation(len(self.train_x))
        shuffled_x = self.train_x[shuffled_indices]
        shuffled_y = self.train_y[shuffled_indices]
        for start in range(0, len(shuffled_x), self.batch_size):
            end = start + self.batch_size
            yield shuffled_x[start:end], shuffled_y[start:end]
