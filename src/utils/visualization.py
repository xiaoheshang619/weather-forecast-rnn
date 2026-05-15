"""可视化工具——Loss 曲线、预测对比图、多组超参数 Loss 对比图"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 GUI 阻塞训练流程
import numpy as np
from pathlib import Path


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str = "outputs/figures/loss_curve.png",
):
    """绘制单组模型的训练与验证 Loss 曲线"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction vs Ground Truth",
    save_path: str = "outputs/figures/predictions.png",
    max_points: int = 200,
):
    """绘制预测值与真实值的对比折线图

    max_points: 最多显示的点数——测试集可能很长，截取末尾避免图像过于密集
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if len(y_true) > max_points:
        y_true = y_true[-max_points:]
        y_pred = y_pred[-max_points:]

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Ground Truth", linewidth=1.5)
    plt.plot(y_pred, label="Prediction", linewidth=1.5, alpha=0.8)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_hparam_comparison(
    results: list[dict],
    save_path: str = "outputs/figures/hparam_comparison.png",
):
    """将多组超参数的训练/验证 Loss 曲线叠放在一张图上对比

    左右两栏分别展示训练 Loss 和验证 Loss，不同颜色对应不同配置
    results: [{"label": "rnn_base", "train_losses": [...], "val_losses": [...]}, ...]
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        epochs = range(1, len(r["train_losses"]) + 1)
        axes[0].plot(epochs, r["train_losses"], color=c, linewidth=1.5, label=r["label"])
        axes[1].plot(epochs, r["val_losses"], color=c, linewidth=1.5, label=r["label"])

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Hyperparameter Comparison — Loss Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
