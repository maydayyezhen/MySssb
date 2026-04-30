"""
CE-CSL BiLSTM-CTC 模型定义

作用：
1. 定义连续手语 gloss 识别 baseline 模型。
2. 输入为 B × T × 166 的时序特征。
3. 输出为 B × T × vocab_size 的 log_probs。
4. 后续训练时配合 PyTorch CTCLoss 使用。

模型结构：
B × T × 166
↓
Linear(166 → 256)
↓
LayerNorm
↓
ReLU
↓
Dropout
↓
2 层 BiLSTM
↓
Dropout
↓
Linear(512 → vocab_size)
↓
log_softmax
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLstmCtcModel(nn.Module):
    """
    CE-CSL gloss 序列识别 BiLSTM-CTC baseline 模型。
    """

    def __init__(
        self,
        input_dim: int = 166,
        projection_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        vocab_size: int = 3840,
        input_dropout: float = 0.2,
        lstm_dropout: float = 0.3,
        output_dropout: float = 0.3,
    ) -> None:
        """
        初始化模型。

        Args:
            input_dim: 单帧输入特征维度，当前为 166。
            projection_dim: 输入投影后的维度。
            hidden_size: 单向 LSTM 隐藏层维度。
            num_layers: LSTM 层数。
            vocab_size: gloss 词表大小，包含 <blank>。
            input_dropout: 输入投影后的 dropout。
            lstm_dropout: LSTM 层间 dropout。
            output_dropout: LSTM 输出后的 dropout。
        """
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # 输入投影层：把每帧 166 维特征投影成 256 维中间表示
        self.input_projection = nn.Linear(input_dim, projection_dim)

        # 对每一帧的 256 维特征做 LayerNorm
        self.input_norm = nn.LayerNorm(projection_dim)

        # ReLU 激活
        self.activation = nn.ReLU()

        # 输入 dropout
        self.input_dropout_layer = nn.Dropout(input_dropout)

        # 双向 LSTM
        # batch_first=True 表示输入输出都是 B × T × C
        self.bilstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        # LSTM 输出 dropout
        self.output_dropout_layer = nn.Dropout(output_dropout)

        # 输出分类层：双向 LSTM 输出维度为 hidden_size × 2
        self.output_layer = nn.Linear(hidden_size * 2, vocab_size)

        # log softmax，用于 CTC Loss
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            features: 输入特征，形状为 B × T × 166。

        Returns:
            log_probs，形状为 B × T × vocab_size。
        """
        if features.ndim != 3:
            raise ValueError(f"features 应为三维张量 B×T×C，实际 shape={tuple(features.shape)}")

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入特征维度错误，期望 {self.input_dim}，实际 {features.shape[-1]}"
            )

        # B × T × 166 -> B × T × 256
        x = self.input_projection(features)

        # B × T × 256
        x = self.input_norm(x)

        # B × T × 256
        x = self.activation(x)

        # B × T × 256
        x = self.input_dropout_layer(x)

        # B × T × 256 -> B × T × 512
        x, _ = self.bilstm(x)

        # B × T × 512
        x = self.output_dropout_layer(x)

        # B × T × 512 -> B × T × vocab_size
        logits = self.output_layer(x)

        # B × T × vocab_size
        log_probs = self.log_softmax(logits)

        return log_probs


class TemporalConvBlock(nn.Module):
    """
    保持时间长度不变的轻量时序卷积残差块。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数，才能保持时间长度不变")

        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: B × T × C

        Returns:
            B × T × C
        """
        residual = x

        y = x.transpose(1, 2)
        y = self.depthwise(y)
        y = self.pointwise(y)
        y = y.transpose(1, 2)
        y = self.activation(y)
        y = self.dropout(y)

        return self.norm(residual + y)


class TemporalConvBiLstmCtcModel(nn.Module):
    """
    BiLSTM-CTC with a local temporal convolution frontend.

    目的：先用短窗口卷积提取局部动作模式，再交给 BiLSTM 做长程时序建模。
    """

    def __init__(
        self,
        input_dim: int = 166,
        projection_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        vocab_size: int = 3840,
        input_dropout: float = 0.2,
        temporal_kernel_size: int = 5,
        temporal_dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        temporal_dropout: float = 0.1,
        lstm_dropout: float = 0.3,
        output_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.input_projection = nn.Linear(input_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        self.activation = nn.ReLU()
        self.input_dropout_layer = nn.Dropout(input_dropout)

        self.temporal_frontend = nn.Sequential(
            *[
                TemporalConvBlock(
                    channels=projection_dim,
                    kernel_size=temporal_kernel_size,
                    dilation=int(dilation),
                    dropout=temporal_dropout,
                )
                for dilation in temporal_dilations
            ]
        )

        self.bilstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        self.output_dropout_layer = nn.Dropout(output_dropout)
        self.output_layer = nn.Linear(hidden_size * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"features 应为三维张量 B×T×C，实际 shape={tuple(features.shape)}")

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入特征维度错误，期望 {self.input_dim}，实际 {features.shape[-1]}"
            )

        x = self.input_projection(features)
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.input_dropout_layer(x)

        x = self.temporal_frontend(x)
        x, _ = self.bilstm(x)
        x = self.output_dropout_layer(x)

        logits = self.output_layer(x)
        log_probs = self.log_softmax(logits)

        return log_probs


class PackedBiLstmCtcModel(nn.Module):
    """
    使用 pack_padded_sequence 的 BiLSTM-CTC。

    与 BiLstmCtcModel 的结构保持一致，但 LSTM 不再看到 padding 帧。
    """

    def __init__(
        self,
        input_dim: int = 166,
        projection_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        vocab_size: int = 3840,
        input_dropout: float = 0.2,
        lstm_dropout: float = 0.3,
        output_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.input_projection = nn.Linear(input_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        self.activation = nn.ReLU()
        self.input_dropout_layer = nn.Dropout(input_dropout)

        self.bilstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        self.output_dropout_layer = nn.Dropout(output_dropout)
        self.output_layer = nn.Linear(hidden_size * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: B × T × C padded batch.
            input_lengths: B, 每条样本真实帧数。

        Returns:
            B × T × vocab_size log_probs.
        """
        if features.ndim != 3:
            raise ValueError(f"features 应为三维张量 B×T×C，实际 shape={tuple(features.shape)}")

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入特征维度错误，期望 {self.input_dim}，实际 {features.shape[-1]}"
            )

        x = self.input_projection(features)
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.input_dropout_layer(x)

        packed = pack_padded_sequence(
            x,
            lengths=input_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)
        x, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=int(features.shape[1]),
        )

        x = self.output_dropout_layer(x)
        logits = self.output_layer(x)
        log_probs = self.log_softmax(logits)

        return log_probs


class SinusoidalPositionalEncoding(nn.Module):
    """
    标准 sinusoidal position encoding，输入输出均为 B × T × C。
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"序列长度 {x.shape[1]} 超过 positional encoding 上限 {self.pe.shape[1]}"
            )

        return x + self.pe[:, : x.shape[1], :]


class TransformerCtcModel(nn.Module):
    """
    Transformer Encoder + CTC 模型。
    """

    def __init__(
        self,
        input_dim: int = 166,
        projection_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        vocab_size: int = 3840,
        input_dropout: float = 0.2,
        transformer_dropout: float = 0.2,
        output_dropout: float = 0.2,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.vocab_size = vocab_size

        self.input_projection = nn.Linear(input_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        self.activation = nn.GELU()
        self.input_dropout_layer = nn.Dropout(input_dropout)
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=projection_dim,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_dropout_layer = nn.Dropout(output_dropout)
        self.output_layer = nn.Linear(projection_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    @staticmethod
    def build_padding_mask(features: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_time, _ = features.shape
        steps = torch.arange(max_time, device=features.device).unsqueeze(0)
        lengths = input_lengths.to(features.device).unsqueeze(1)
        return steps.expand(batch_size, max_time) >= lengths

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"features 应为三维张量 B×T×C，实际 shape={tuple(features.shape)}")

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入特征维度错误，期望 {self.input_dim}，实际 {features.shape[-1]}"
            )

        padding_mask = self.build_padding_mask(features, input_lengths)

        x = self.input_projection(features)
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.input_dropout_layer(x)
        x = self.position_encoding(x)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.output_dropout_layer(x)
        logits = self.output_layer(x)
        log_probs = self.log_softmax(logits)

        return log_probs
