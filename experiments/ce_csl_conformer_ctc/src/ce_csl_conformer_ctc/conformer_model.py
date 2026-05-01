"""
CE-CSL Conformer-CTC 模型定义。

本文件属于独立实验：
experiments/ce_csl_conformer_ctc

作用：
1. 定义 Conformer Encoder + CTC 连续手语识别模型。
2. 输入为 B × T × 332 的 raw_delta 特征。
3. 输出为 B × T × vocab_size 的 log_probs。
4. 后续训练时配合 PyTorch CTCLoss 使用。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦位置编码。

    输入输出均为 B × T × C。
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        """
        初始化位置编码。

        Args:
            d_model: 模型隐藏维度。
            max_len: 支持的最大序列长度。
        """
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
        """
        添加位置编码。

        Args:
            x: B × T × C。

        Returns:
            加入位置编码后的 B × T × C。
        """
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"序列长度 {x.shape[1]} 超过位置编码上限 {self.pe.shape[1]}"
            )

        return x + self.pe[:, : x.shape[1], :]


class ConformerFeedForwardModule(nn.Module):
    """
    Conformer 前馈模块。
    """

    def __init__(
        self,
        d_model: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        """
        初始化前馈模块。

        Args:
            d_model: 主干隐藏维度。
            feedforward_dim: 前馈网络中间维度。
            dropout: dropout 概率。
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: B × T × C。

        Returns:
            B × T × C。
        """
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Conformer 卷积模块。

    作用：
    1. 使用 pointwise conv + GLU 进行通道门控。
    2. 使用 depthwise conv 建模局部连续动作变化。
    3. 使用 GroupNorm 降低小 batch 下 BatchNorm 不稳定的问题。
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 15,
        dropout: float = 0.2,
    ) -> None:
        """
        初始化卷积模块。

        Args:
            d_model: 主干隐藏维度。
            kernel_size: 时序卷积核大小，必须为奇数。
            dropout: dropout 概率。
        """
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数")

        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(d_model)

        self.pointwise_conv_1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
        )

        self.glu = nn.GLU(dim=1)

        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )

        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=d_model,
        )

        self.activation = nn.SiLU()

        self.pointwise_conv_2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: B × T × C。

        Returns:
            B × T × C。
        """
        x = self.layer_norm(x)

        # B × T × C -> B × C × T
        x = x.transpose(1, 2)

        x = self.pointwise_conv_1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv_2(x)
        x = self.dropout(x)

        # B × C × T -> B × T × C
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """
    简化版 Conformer Block。

    结构：
    1. 0.5 × FFN
    2. Multi-Head Self-Attention
    3. Convolution Module
    4. 0.5 × FFN
    5. Final LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        """
        初始化 Conformer Block。

        Args:
            d_model: 主干隐藏维度。
            num_heads: 注意力头数。
            feedforward_dim: 前馈网络中间维度。
            conv_kernel_size: 卷积核大小。
            dropout: dropout 概率。
        """
        super().__init__()

        self.ffn_1 = ConformerFeedForwardModule(
            d_model=d_model,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        self.self_attn_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = ConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.ffn_2 = ConformerFeedForwardModule(
            d_model=d_model,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def mask_padding_positions(
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        将 padding 位置置零，避免卷积模块污染 padding 区域。

        Args:
            x: B × T × C。
            padding_mask: B × T，True 表示 padding。

        Returns:
            B × T × C。
        """
        if padding_mask is None:
            return x

        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: B × T × C。
            padding_mask: B × T，True 表示 padding。

        Returns:
            B × T × C。
        """
        x = x + 0.5 * self.ffn_1(x)
        x = self.mask_padding_positions(x, padding_mask)

        residual = x
        attn_input = self.self_attn_norm(x)

        attn_output, _ = self.self_attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        x = residual + self.self_attn_dropout(attn_output)
        x = self.mask_padding_positions(x, padding_mask)

        x = x + self.conv_module(x)
        x = self.mask_padding_positions(x, padding_mask)

        x = x + 0.5 * self.ffn_2(x)
        x = self.mask_padding_positions(x, padding_mask)

        return self.final_norm(x)


class ConformerCtcModel(nn.Module):
    """
    Conformer Encoder + CTC 连续手语识别模型。

    输入：
        B × T × input_dim

    输出：
        B × T × vocab_size 的 log_probs
    """

    def __init__(
        self,
        input_dim: int = 332,
        projection_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        conv_kernel_size: int = 15,
        vocab_size: int = 3840,
        input_dropout: float = 0.2,
        conformer_dropout: float = 0.2,
        output_dropout: float = 0.2,
        max_len: int = 512,
    ) -> None:
        """
        初始化模型。

        Args:
            input_dim: 输入特征维度。raw_delta 模式下为 332。
            projection_dim: 投影后的主干维度。
            num_layers: Conformer Block 层数。
            num_heads: 多头注意力头数。
            feedforward_dim: 前馈网络中间维度。
            conv_kernel_size: 卷积核大小。
            vocab_size: gloss 词表大小，包含 blank。
            input_dropout: 输入投影后的 dropout。
            conformer_dropout: Conformer Block 内部 dropout。
            output_dropout: 输出层前 dropout。
            max_len: 位置编码最大长度。
        """
        super().__init__()

        if projection_dim % num_heads != 0:
            raise ValueError(
                f"projection_dim 必须能被 num_heads 整除，"
                f"当前 projection_dim={projection_dim}, num_heads={num_heads}"
            )

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.conv_kernel_size = conv_kernel_size
        self.vocab_size = vocab_size

        self.input_projection = nn.Linear(input_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        self.activation = nn.GELU()
        self.input_dropout_layer = nn.Dropout(input_dropout)

        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=projection_dim,
            max_len=max_len,
        )

        self.encoder_layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=projection_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=conformer_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_dropout_layer = nn.Dropout(output_dropout)
        self.output_layer = nn.Linear(projection_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    @staticmethod
    def build_padding_mask(
        features: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造 padding mask。

        Args:
            features: B × T × C。
            input_lengths: B，每条样本真实长度。

        Returns:
            B × T，True 表示 padding。
        """
        batch_size, max_time, _ = features.shape

        steps = torch.arange(
            max_time,
            device=features.device,
        ).unsqueeze(0)

        lengths = input_lengths.to(features.device).unsqueeze(1)

        return steps.expand(batch_size, max_time) >= lengths

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            features: B × T × input_dim。
            input_lengths: B，每条样本真实帧数。

        Returns:
            B × T × vocab_size 的 log_probs。
        """
        if features.ndim != 3:
            raise ValueError(
                f"features 应为三维张量 B×T×C，实际 shape={tuple(features.shape)}"
            )

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入特征维度错误，期望 {self.input_dim}，实际 {features.shape[-1]}"
            )

        padding_mask = self.build_padding_mask(
            features=features,
            input_lengths=input_lengths,
        )

        x = self.input_projection(features)
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.input_dropout_layer(x)
        x = self.position_encoding(x)

        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(
                x=x,
                padding_mask=padding_mask,
            )

        x = self.output_dropout_layer(x)

        logits = self.output_layer(x)

        log_probs = self.log_softmax(logits)

        return log_probs