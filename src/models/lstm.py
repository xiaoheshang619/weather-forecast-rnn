"""LSTM 模型——引入遗忘门、输入门、输出门的长短时记忆网络

结构: 输入(B,T,F) → LSTM → 取最后时间步 → Linear → 输出(B,1)

门控机制:
- 遗忘门: 决定丢弃哪些旧信息
- 输入门: 决定存储哪些新信息
- 输出门: 决定输出哪些信息
三门控使 LSTM 能选择性记忆长期依赖，但参数量约为同配置 RNN 的 4 倍，
需要更多训练样本才能充分发挥作用
"""

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
