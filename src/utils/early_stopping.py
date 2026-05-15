"""早停模块——验证 Loss 连续不下降时终止训练，保留最佳模型"""

import torch


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        """
        patience:  容忍轮数——验证 Loss 连续不下降超过此值则停止
        min_delta: 最小改善阈值——Loss 下降小于此值视为"无改善"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0          # 连续无改善计数器
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str):
        """每 epoch 调用一次，验证 Loss 创新低时保存模型，连续无改善时标记停止"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)  # 只保存最佳模型的权重
            return True  # 本轮是新的最佳
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
