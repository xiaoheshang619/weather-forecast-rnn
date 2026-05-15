"""评估指标——MAE、MSE、RMSE 及 DataLoader 评估函数

MAE:  平均绝对误差，与温度同量纲（℃），直观反映预测偏差
RMSE: 均方根误差，对大误差惩罚更重，适合检测模型是否产生严重离群预测
"""

import numpy as np
import torch


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差 Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差 Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差 Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion,
    device: str = "cpu",
) -> dict[str, float]:
    """在给定 DataLoader 上计算平均损失和各项指标

    使用 torch.no_grad() 禁用梯度计算，减少显存占用和计算开销
    """
    model.eval()
    total_loss, total_mae, total_rmse = 0.0, 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            total_mae += mae(y.cpu().numpy(), pred.cpu().numpy())
            total_rmse += rmse(y.cpu().numpy(), pred.cpu().numpy())
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mae": total_mae / n_batches,
        "rmse": total_rmse / n_batches,
    }
