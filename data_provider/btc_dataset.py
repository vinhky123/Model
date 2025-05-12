import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class BTCDataset(Dataset):
    def __init__(
        self,
        data_path,
        seq_len=96,
        pred_len=5,
        features="M",
        target="Close",
        scale=True,
        freq="d",
    ):
        """
        Args:
            data_path: đường dẫn đến file CSV chứa dữ liệu BTC
            seq_len: độ dài chuỗi input
            pred_len: độ dài chuỗi dự đoán (5 ngày)
            features: 'M' cho multivariate
            target: cột target để dự đoán
            scale: có normalize dữ liệu không
            freq: tần suất của dữ liệu ('d' cho daily)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale
        self.freq = freq

        # Đọc và xử lý dữ liệu
        df = pd.read_csv(data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # Tính toán target (1 nếu giá tăng, 0 nếu giảm)
        df["target"] = (df[target].shift(-pred_len) > df[target]).astype(int)

        # Chọn features
        if features == "M":
            self.data = df[["Open", "High", "Low", "Close", "Volume"]].values
        else:
            self.data = df[[target]].values

        self.targets = df["target"].values

        # Tạo time features
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["hour"] = df["Date"].dt.hour
        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        df["dayofyear"] = df["Date"].dt.dayofyear
        df["weekofyear"] = df["Date"].dt.isocalendar().week
        df["quarter"] = df["Date"].dt.quarter

        self.time_features = df[
            ["dayofweek", "hour", "day", "month", "dayofyear", "weekofyear", "quarter"]
        ].values

        # Scale dữ liệu nếu cần
        if scale:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)

        # Tạo các cặp (X, y) cho training
        self.samples = []
        for i in range(len(df) - seq_len - pred_len + 1):
            x = self.data[i : i + seq_len]
            time_x = self.time_features[i : i + seq_len]
            y = self.targets[i + seq_len - 1]
            self.samples.append((x, time_x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, time_x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(time_x), torch.LongTensor([y])


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean
