"""
CTC 解码与序列评估工具

作用：
1. 对模型输出的帧级 log_probs 做 CTC greedy decode。
2. 将预测 id 序列转换成 gloss 序列。
3. 计算 token 级编辑距离和 WER。
"""

from __future__ import annotations

from typing import Dict, List

import torch


def ctc_greedy_decode(
    log_probs_btc: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """
    CTC greedy decode。

    Args:
        log_probs_btc: 模型输出，形状为 B × T × vocab_size。
        input_lengths: 每条样本真实输入长度，形状为 B。
        blank_id: CTC blank id。

    Returns:
        每条样本解码后的 token id 序列。
    """
    if log_probs_btc.ndim != 3:
        raise ValueError(f"log_probs_btc 应为 B×T×V，实际 shape={tuple(log_probs_btc.shape)}")

    # B × T
    pred_ids = torch.argmax(log_probs_btc, dim=-1)

    decoded_results: List[List[int]] = []

    batch_size = pred_ids.shape[0]

    for batch_index in range(batch_size):
        valid_length = int(input_lengths[batch_index].item())

        raw_ids = pred_ids[batch_index, :valid_length].tolist()

        decoded_ids: List[int] = []
        previous_id = None

        for token_id in raw_ids:
            # CTC 规则 1：合并连续重复
            if token_id == previous_id:
                continue

            previous_id = token_id

            # CTC 规则 2：删除 blank
            if token_id == blank_id:
                continue

            decoded_ids.append(int(token_id))

        decoded_results.append(decoded_ids)

    return decoded_results


def ids_to_gloss(token_ids: List[int], id_to_gloss: Dict[int, str]) -> List[str]:
    """
    将 token id 序列转换成 gloss 序列。

    Args:
        token_ids: token id 列表。
        id_to_gloss: id 到 gloss 的映射。

    Returns:
        gloss 字符串列表。
    """
    return [id_to_gloss.get(token_id, f"<id:{token_id}>") for token_id in token_ids]


def edit_distance(source: List[str], target: List[str]) -> int:
    """
    计算两个 token 序列之间的编辑距离。

    Args:
        source: 预测序列。
        target: 真实序列。

    Returns:
        编辑距离。
    """
    source_len = len(source)
    target_len = len(target)

    dp = [[0] * (target_len + 1) for _ in range(source_len + 1)]

    for i in range(source_len + 1):
        dp[i][0] = i

    for j in range(target_len + 1):
        dp[0][j] = j

    for i in range(1, source_len + 1):
        for j in range(1, target_len + 1):
            if source[i - 1] == target[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[source_len][target_len]


def token_error_rate(prediction: List[str], reference: List[str]) -> float:
    """
    计算 token 级错误率。

    Args:
        prediction: 预测 gloss 序列。
        reference: 真实 gloss 序列。

    Returns:
        编辑距离 / 真实序列长度。
    """
    if not reference:
        return 0.0 if not prediction else 1.0

    distance = edit_distance(prediction, reference)

    return distance / len(reference)