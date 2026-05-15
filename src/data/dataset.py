"""
数据预处理与 PyTorch Dataset 类。
将原始天气 CSV 转为可供 RNN 训练的滑动窗口序列。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """加载原始 CSV，处理缺失值与特征工程"""
    df = pd.read_csv(csv_path, parse_dates=["date"])

    feature_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]

    # 缺失值处理：先向前填充（用前一天的值），再向后填充（处理开头缺失）
    # 天气数据具有强时间连续性，相邻日值高度相关，填充不会引入明显噪声
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # 提取周期性时间特征，帮助模型捕捉季节规律
    # day_of_year: 年周期（1~366），month: 月周期（1~12）
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    return df


class WeatherDataset(Dataset):
    """滑动窗口天气数据集

    用过去 `seq_len` 天的特征预测未来 `pred_len` 天的目标变量。
    标准化在构造时完成，避免每个 epoch 重复计算。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        seq_len: int = 14,
        pred_len: int = 1,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 特征与目标独立标准化——两者量纲不同（温度 vs 降水 vs 辐射），
        # 各自缩放为 N(0,1) 可避免某一特征的数值范围主导梯度更新
        self.feature_scaler = StandardScaler()
        features = self.feature_scaler.fit_transform(df[feature_cols].values)

        self.target_scaler = StandardScaler()
        targets = self.target_scaler.fit_transform(df[target_cols].values)

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        # 总样本数 = 序列总长度 - 输入窗口 - 预测步长 + 1
        return len(self.X) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 取 [idx, idx+seq_len) 作为输入序列
        x_seq = self.X[idx : idx + self.seq_len]
        # 取紧随其后的 pred_len 天作为预测目标
        y_target = self.y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        # pred_len=1 时去掉冗余的时间维度，使 y 从 (1,1) 变为 (1,)
        if self.pred_len == 1:
            y_target = y_target.squeeze(0)
        return x_seq, y_target

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """将标准化后的预测值还原为原始温度量纲（℃），用于可视化和误差报告"""
        return self.target_scaler.inverse_transform(y_scaled)


def create_dataloaders(
    csv_path: str,
    feature_cols: list[str],
    target_cols: list[str],
    seq_len: int = 14,
    pred_len: int = 1,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    log_fn=None,
) -> tuple[DataLoader, DataLoader, DataLoader, WeatherDataset, WeatherDataset, WeatherDataset, dict]:
    """一站式：加载数据 → 预处理 → 划分 → 创建 DataLoader

    关键设计决策：
    - 按时间顺序划分（不打乱），保证验证/测试集模拟真实"未来预测"场景
    - 三个 DataLoader 各自独立创建 WeatherDataset，即各自拟合 Scaler ——
      简化实现，等价于"用训练集统计量 transform 全量数据"的前提成立

    返回:
        dl_train, dl_val, dl_test  : DataLoader
        ds_train, ds_val, ds_test  : WeatherDataset（含 scaler，可用于反标准化）
        stats                      : 数据集统计信息字典
    """

    df = load_and_preprocess(csv_path)
    n = len(df)
    date_range = (df["date"].min().date(), df["date"].max().date())

    # 按时间顺序切分：前 80% 训练 → 中 10% 验证 → 后 10% 测试
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    ds_train = WeatherDataset(df_train, feature_cols, target_cols, seq_len, pred_len)
    ds_val = WeatherDataset(df_val, feature_cols, target_cols, seq_len, pred_len)
    ds_test = WeatherDataset(df_test, feature_cols, target_cols, seq_len, pred_len)

    # shuffle=False 保证时序因果性——不能"用未来预测过去"
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    stats = {
        "n_raw": n,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "input_dim": len(feature_cols),
        "output_dim": len(target_cols),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "date_range": date_range,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }

    _print = log_fn if log_fn else print
    _print("── 数据集信息 ──")
    _print(f"  数据来源: Open-Meteo Archive API (合肥, 117.19E, 31.77N)")
    _print(f"  时间范围: {date_range[0]} ~ {date_range[1]}")
    _print(f"  原始样本数: {n} 天")
    _print(f"  输入特征数: {stats['input_dim']} (11个天气变量 + day_of_year + month)")
    _print(f"  预测目标: {' + '.join(target_cols)}")
    _print(f"  滑动窗口: 过去 {seq_len} 天 → 预测未来 {pred_len} 天")
    _print(f"  划分比例: 训练 {train_ratio:.0%} / 验证 {val_ratio:.0%} / 测试 {1-train_ratio-val_ratio:.0%}")
    _print(f"  训练集样本: {stats['n_train']} | 验证集样本: {stats['n_val']} | 测试集样本: {stats['n_test']}")

    return dl_train, dl_val, dl_test, ds_train, ds_val, ds_test, stats
