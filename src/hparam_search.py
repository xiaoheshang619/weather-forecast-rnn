"""
超参数搜索：依次训练多组配置，自动生成对比表格（CSV）+ 对比图（PNG）。

设计目的: 满足课程要求(3)——至少 3 组不同超参数，以表格和 Loss 对比图呈现
每组配置训练完成后自动保存独立日志、模型、图表，避免结果互相覆盖
"""

import csv
from pathlib import Path
from datetime import datetime

from train import train_with_config
from utils.visualization import plot_hparam_comparison
from utils.logger import setup_logger

# ── 搜索配置 ——
# 前 3 组为同参数模型对照（仅循环单元类型不同），用于考察门控机制的独立贡献
# 后 2 组为过拟合 / 欠拟合案例，用于验证模型规模与数据量匹配的重要性
SEARCH_CONFIGS = [
    {
        "label": "rnn_base",
        "config_path": "configs/rnn_base.yaml",
        "desc": "基准RNN: RNN hidden=64 layers=2 lr=0.001 batch=32 dropout=0.2",
    },
    {
        "label": "lstm_base",
        "config_path": "configs/lstm_base.yaml",
        "desc": "基准LSTM: LSTM hidden=64 layers=2 lr=0.001 batch=32 dropout=0.2",
    },
    {
        "label": "gru_base",
        "config_path": "configs/gru_base.yaml",
        "desc": "基准GRU: GRU hidden=64 layers=2 lr=0.001 batch=32 dropout=0.2",
    },
    {
        "label": "lstm_large",
        "config_path": "configs/lstm_large.yaml",
        "desc": "大模型LSTM: LSTM hidden=128 layers=3 lr=0.0005 batch=64 dropout=0.3",
    },
    {
        "label": "gru_small",
        "config_path": "configs/gru_small.yaml",
        "desc": "小模型GRU: GRU hidden=32 layers=1 lr=0.01 batch=16 dropout=0.1",
    },
]

LOG_DIR = "outputs/logs"
CSV_PATH = "outputs/logs/hparam_results.csv"
FIGURE_PATH = "outputs/figures/hparam_comparison.png"


def main():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    log = setup_logger("hparam_search", f"{LOG_DIR}/hparam_search.log")

    log.info("=" * 60)
    log.info(f"超参数搜索开始 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"共 {len(SEARCH_CONFIGS)} 组配置")
    log.info("=" * 60)

    all_results = []

    # 依次训练每组配置，label 确保模型/日志/图片独立命名，避免覆盖
    for i, entry in enumerate(SEARCH_CONFIGS, 1):
        label = entry["label"]
        cfg_path = entry["config_path"]
        desc = entry["desc"]

        log.info(f"\n{'='*60}")
        log.info(f"[{i}/{len(SEARCH_CONFIGS)}] {label}")
        log.info(f"    描述: {desc}")
        log.info(f"    配置: {cfg_path}")
        log.info(f"{'='*60}")

        result = train_with_config(cfg_path, log_dir=LOG_DIR, label=label, quiet=True)
        result["label"] = label
        result["desc"] = desc
        all_results.append(result)

    # ── 打印对比表格（控制台 + 日志文件）──
    log.info(f"\n{'='*70}")
    log.info("超参数对比结果")
    log.info(f"{'='*70}")
    header = f"{'组名':<14} {'模型':<6} {'最佳Val Loss':<14} {'测试Loss':<12} {'测试MAE':<10} {'测试RMSE':<10}"
    log.info(header)
    log.info("-" * 70)

    for r in all_results:
        tm = r["test_metrics"]
        log.info(
            f"{r['label']:<14} {r['model_type']:<6} "
            f"{r['best_val_loss']:<14.4f} {tm['loss']:<12.4f} "
            f"{tm['mae']:<10.4f} {tm['rmse']:<10.4f}"
        )
    log.info("-" * 70)

    # ── 保存 CSV——方便在 Excel 中打开做进一步分析 ──
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "model_type", "best_val_loss", "test_loss", "test_mae", "test_rmse", "desc"])
        for r in all_results:
            tm = r["test_metrics"]
            writer.writerow([r["label"], r["model_type"], f"{r['best_val_loss']:.6f}",
                             f"{tm['loss']:.6f}", f"{tm['mae']:.6f}", f"{tm['rmse']:.6f}", r["desc"]])
    log.info(f"\n对比表格已保存至 {CSV_PATH}")

    # ── 绘制 Loss 对比图——满足课程要求"至少两次不同超参数下的 loss 曲线对比" ──
    plot_data = [
        {
            "label": r["label"],
            "train_losses": r["train_losses"],
            "val_losses": r["val_losses"],
        }
        for r in all_results
    ]
    plot_hparam_comparison(plot_data, FIGURE_PATH)
    log.info(f"对比图已保存至 {FIGURE_PATH}")

    log.info(f"\n超参数搜索完成 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
