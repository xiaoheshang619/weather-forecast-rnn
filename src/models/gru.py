"""GRU 模型——引入重置门和更新门的门控循环单元

结构: 输入(B,T,F) → GRU → 取最后时间步 → Linear → 输出(B,1)

门控机制:
- 重置门: 决定遗忘多少过去信息
- 更新门: 决定保留多少过去信息（同时控制遗忘和输入）
双门控相比 LSTM 三门控参数量更少（约 3/4），在小数据和中等数据上
经常取得比 LSTM 更好的均衡性能
"""

import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
