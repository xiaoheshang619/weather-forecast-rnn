"""
训练入口：加载配置 → 准备数据 → 训练 RNN/LSTM/GRU → 评估 → 保存结果
可作为模块被 hparam_search.py 调用，也可独立运行。
"""

import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# 全局随机种子，保证权重初始化、dropout、数据加载等完全可复现
SEED = 4044

from data.dataset import create_dataloaders
from models import SimpleRNN, LSTM, GRU
from utils.early_stopping import EarlyStopping
from utils.metrics import evaluate
from utils.visualization import plot_loss_curve, plot_predictions
from utils.logger import setup_logger


def build_model(model_type: str, input_dim: int, output_dim: int, cfg: dict) -> nn.Module:
    """根据配置构造对应的 RNN 变体模型"""
    if model_type == "rnn":
        return SimpleRNN(input_dim, cfg["hidden_dim"], cfg["num_layers"], output_dim, cfg["dropout"])
    elif model_type == "lstm":
        return LSTM(input_dim, cfg["hidden_dim"], cfg["num_layers"], output_dim, cfg["dropout"])
    elif model_type == "gru":
        return GRU(input_dim, cfg["hidden_dim"], cfg["num_layers"], output_dim, cfg["dropout"])
    raise ValueError(f"未知模型类型: {model_type}")


def train_epoch(model, loader, criterion, optimizer, device):
    """单轮训练：遍历 DataLoader，反向传播更新权重，返回平均损失"""
    model.train()
    total_loss, batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / batches


def train_with_config(
    config_path: str,
    log_dir: str = "outputs/logs",
    label: str = "",
    quiet: bool = False,
) -> dict:
    """用指定配置文件完成一次完整训练，返回训练指标和 loss 列表。

    参数:
        config_path: YAML 配置文件路径
        log_dir: 日志输出目录
        label: 用于命名模型文件和日志的唯一标识，未指定时使用 model_type
    返回:
        test_metrics:  {"loss", "mae", "rmse"}
        train_losses:  [float, ...]  每个 epoch 的训练损失
        val_losses:    [float, ...]  每个 epoch 的验证损失
        best_val_loss: float         早停时的最佳验证损失
        model_type:    str
        label:         str
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 优先使用 GPU，不可用时自动降级为 CPU
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    # 固定随机种子——排除初始化波动对实验结论的干扰，保证结果可复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model_type = cfg["model"]["type"]
    run_label = label if label else model_type
    log_path = f"{log_dir}/train_{run_label}.log"
    log = setup_logger(f"train_{run_label}", log_path)

    log.info(f"设备: {device}")
    log.info(f"随机种子: {SEED}")
    log.info(f"配置文件: {config_path}")

    # ── 数据 ──
    # create_dataloaders 内部完成：加载CSV → 缺失值填充 → 时间特征提取 →
    #   Z-Score标准化 → 滑动窗口构造 → 时序划分(不打乱) → 封装DataLoader
    dl_train, dl_val, dl_test, ds_train, ds_val, ds_test, stats = create_dataloaders(
        csv_path=cfg["data"]["csv_path"],
        feature_cols=cfg["data"]["feature_cols"],
        target_cols=cfg["data"]["target_cols"],
        seq_len=cfg["data"]["seq_len"],
        pred_len=cfg["data"]["pred_len"],
        batch_size=cfg["train"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        log_fn=log.info,
    )

    input_dim = stats["input_dim"]
    output_dim = stats["output_dim"]

    # ── 模型 ──
    model = build_model(model_type, input_dim, output_dim, cfg["model"]).to(device)
    log.info(f"\n── 模型结构 ──\n{model}\n")

    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"  可训练参数量: {num_params:,}")

    log.info(f"\n── 超参数 ──")
    log.info(f"  模型类型: {model_type}")
    log.info(f"  hidden_dim: {cfg['model']['hidden_dim']}")
    log.info(f"  num_layers: {cfg['model']['num_layers']}")
    log.info(f"  dropout: {cfg['model']['dropout']}")
    log.info(f"  lr: {cfg['train']['lr']}")
    log.info(f"  batch_size: {cfg['train']['batch_size']}")

    # MSE 对大误差惩罚更重，适合温度预测这类对异常值敏感的任务
    criterion = nn.MSELoss()
    # Adam 结合动量和自适应学习率，在小批量时序数据上收敛稳定
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    model_save_path = f"{cfg['output']['model_dir']}/{run_label}_best.pt"
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    # 早停机制：验证 Loss 连续 patience 轮不下降则终止训练，避免无意义计算
    early_stop = EarlyStopping(patience=cfg["train"]["patience"])

    train_losses, val_losses = [], []

    # ── 训练循环 ──
    # 日志只打印第1轮、每10整轮、以及验证Loss创新低时的epoch，
    # 既保留完整过程证据，又避免日志过于冗长
    log.info(f"\n── 开始训练 (max {cfg['train']['epochs']} epochs) ──")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_epoch(model, dl_train, criterion, optimizer, device)
        val_metrics = evaluate(model, dl_val, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        improved = early_stop(val_metrics["loss"], model, model_save_path)

        if epoch % 10 == 0 or epoch == 1 or improved:
            log.info(
                f"Epoch {epoch:3d}/{cfg['train']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"MAE: {val_metrics['mae']:.4f} | "
                f"RMSE: {val_metrics['rmse']:.4f}"
                f"{' *' if improved else ''}"
            )

        if early_stop.should_stop:
            log.info(f"\n早停触发于 epoch {epoch}")
            break

    # ── 测试评估 ──
    # 加载早停保存的最佳模型（而非最后一轮），避免过拟合模型影响测试结果
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_metrics = evaluate(model, dl_test, criterion, device)
    log.info(f"\n── 测试集结果 ──")
    log.info(f"  Loss: {test_metrics['loss']:.4f}")
    log.info(f"  MAE:  {test_metrics['mae']:.4f}")
    log.info(f"  RMSE: {test_metrics['rmse']:.4f}")

    # ── 可视化 ──
    plot_loss_curve(train_losses, val_losses, f"{cfg['output']['figure_dir']}/loss_curve_{run_label}.png")

    # 在测试集上收集所有预测值，反标准化后绘制预测 vs 真实对比图
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())
    # 将标准化后的预测值还原为原始温度量纲（℃）
    y_pred = ds_test.target_scaler.inverse_transform(np.concatenate(all_preds))
    y_true = ds_test.target_scaler.inverse_transform(np.concatenate(all_targets))

    plot_predictions(
        y_true, y_pred,
        title=f"{model_type.upper()} — Prediction vs Ground Truth",
        save_path=f"{cfg['output']['figure_dir']}/predictions_{run_label}.png",
    )

    log.info(f"图表已保存至 {cfg['output']['figure_dir']}/")
    log.info(f"模型已保存至 {model_save_path}")
    log.info(f"日志已保存至 {log_path}")

    return {
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": early_stop.best_loss,
        "model_type": model_type,
        "label": run_label,
    }


def main():
    """独立运行入口：使用默认配置训练单次"""
    result = train_with_config("configs/config.yaml")
    print(f"\n训练完成。测试集 MAE: {result['test_metrics']['mae']:.4f}")


if __name__ == "__main__":
    main()
