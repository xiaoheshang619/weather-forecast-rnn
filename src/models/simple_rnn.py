"""基础 RNN 模型——无门控机制的原始循环神经网络

结构: 输入(B,T,F) → RNN → 取最后时间步 → Linear → 输出(B,1)

优点: 参数少、收敛快，小数据上不易过拟合
局限: 无门控机制，梯度消失导致长期依赖建模能力弱
"""

import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,       # 输入特征维度
        hidden_dim: int = 64, # 隐层向量维度
        num_layers: int = 2,  # RNN 堆叠层数（>1 时层间加 dropout）
        output_dim: int = 1,  # 预测目标维度
        dropout: float = 0.2, # 层间 dropout 比例（仅 num_layers > 1 时生效）
    ):
        super().__init__()
        # batch_first=True: 输入形状为 (batch, seq_len, features)
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        # 取 RNN 最后一个时间步的隐状态映射为预测值
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        out, _ = self.rnn(x)
        # out[:, -1, :] 取每个样本最后时间步的输出 → (B, hidden_dim)
        return self.fc(out[:, -1, :])
