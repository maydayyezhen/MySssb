"""
CTC beam search 解码工具

作用：
1. 对单条样本的 T × vocab_size log_probs 做 CTC prefix beam search。
2. 支持 beam_size、top_k_per_frame、token_insert_bonus。
3. 用于比较 greedy decode 和 beam search decode 的效果。

说明：
- 这里不使用语言模型。
- token_insert_bonus 是一个可选的长度奖励，用于缓解 CTC 输出过短的问题。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


NEG_INF = -1e30


def log_add(*values: float) -> float:
    """
    稳定的 log-sum-exp。

    Args:
        values: 若干 log 概率。

    Returns:
        log(exp(v1) + exp(v2) + ...)
    """
    valid_values = [value for value in values if value > NEG_INF / 2]

    if not valid_values:
        return NEG_INF

    max_value = max(valid_values)

    total = sum(math.exp(value - max_value) for value in valid_values)

    return max_value + math.log(total)


def get_beam_score(
    prefix: Tuple[int, ...],
    blank_score: float,
    non_blank_score: float,
    token_insert_bonus: float,
) -> float:
    """
    计算 beam 排序分数。

    Args:
        prefix: 当前候选 token 序列。
        blank_score: 以 blank 结尾的路径 log 概率。
        non_blank_score: 以非 blank 结尾的路径 log 概率。
        token_insert_bonus: token 长度奖励。

    Returns:
        beam 排序分数。
    """
    base_score = log_add(blank_score, non_blank_score)

    return base_score + len(prefix) * token_insert_bonus


def ctc_prefix_beam_search_one(
    log_probs_tv: torch.Tensor,
    input_length: int,
    blank_id: int = 0,
    beam_size: int = 5,
    top_k_per_frame: int = 50,
    token_insert_bonus: float = 0.0,
) -> List[int]:
    """
    对单条样本做 CTC prefix beam search。

    Args:
        log_probs_tv: 单条样本的 log_probs，形状为 T × vocab_size。
        input_length: 真实输入长度，不包含 padding。
        blank_id: CTC blank token id。
        beam_size: 每个时间步保留多少条候选前缀。
        top_k_per_frame: 每帧只扩展概率最高的 top-k token，避免全词表暴力展开。
        token_insert_bonus: token 长度奖励，适当增大可缓解预测偏短。

    Returns:
        解码后的 token id 序列。
    """
    if log_probs_tv.ndim != 2:
        raise ValueError(f"log_probs_tv 应为 T×V，实际 shape={tuple(log_probs_tv.shape)}")

    valid_log_probs = log_probs_tv[:input_length]

    vocab_size = int(valid_log_probs.shape[1])
    top_k = min(top_k_per_frame, vocab_size)

    # beams:
    # prefix -> (p_blank, p_non_blank)
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {
        tuple(): (0.0, NEG_INF)
    }

    for time_index in range(input_length):
        frame_log_probs = valid_log_probs[time_index]

        top_values, top_indices = torch.topk(
            frame_log_probs,
            k=top_k,
            dim=-1,
        )

        top_token_scores = [
            (int(token_id), float(log_prob))
            for token_id, log_prob in zip(top_indices.tolist(), top_values.tolist())
        ]

        next_beams = defaultdict(lambda: (NEG_INF, NEG_INF))

        for prefix, (prefix_blank_score, prefix_non_blank_score) in beams.items():
            for token_id, token_log_prob in top_token_scores:
                if token_id == blank_id:
                    # 追加 blank：前缀不变，blank 结尾概率增加
                    old_blank_score, old_non_blank_score = next_beams[prefix]

                    new_blank_score = log_add(
                        old_blank_score,
                        prefix_blank_score + token_log_prob,
                        prefix_non_blank_score + token_log_prob,
                    )

                    next_beams[prefix] = (
                        new_blank_score,
                        old_non_blank_score,
                    )

                    continue

                last_token = prefix[-1] if prefix else None

                if token_id == last_token:
                    # 情况 1：重复 token 且没有 blank 分隔，CTC collapse 后前缀不变
                    old_blank_score, old_non_blank_score = next_beams[prefix]

                    new_non_blank_score = log_add(
                        old_non_blank_score,
                        prefix_non_blank_score + token_log_prob,
                    )

                    next_beams[prefix] = (
                        old_blank_score,
                        new_non_blank_score,
                    )

                    # 情况 2：重复 token 但从 blank 状态转移，可以形成真正的重复 token
                    # 例如 A blank A
                    extended_prefix = prefix + (token_id,)

                    old_blank_score, old_non_blank_score = next_beams[extended_prefix]

                    new_non_blank_score = log_add(
                        old_non_blank_score,
                        prefix_blank_score + token_log_prob,
                    )

                    next_beams[extended_prefix] = (
                        old_blank_score,
                        new_non_blank_score,
                    )

                else:
                    # 普通扩展：prefix + token_id
                    extended_prefix = prefix + (token_id,)

                    old_blank_score, old_non_blank_score = next_beams[extended_prefix]

                    new_non_blank_score = log_add(
                        old_non_blank_score,
                        prefix_blank_score + token_log_prob,
                        prefix_non_blank_score + token_log_prob,
                    )

                    next_beams[extended_prefix] = (
                        old_blank_score,
                        new_non_blank_score,
                    )

        # 按 beam 分数剪枝
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: get_beam_score(
                    prefix=item[0],
                    blank_score=item[1][0],
                    non_blank_score=item[1][1],
                    token_insert_bonus=token_insert_bonus,
                ),
                reverse=True,
            )[:beam_size]
        )

    best_prefix = max(
        beams.items(),
        key=lambda item: get_beam_score(
            prefix=item[0],
            blank_score=item[1][0],
            non_blank_score=item[1][1],
            token_insert_bonus=token_insert_bonus,
        ),
    )[0]

    return list(best_prefix)


def ctc_prefix_beam_search_batch(
    log_probs_btv: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int = 0,
    beam_size: int = 5,
    top_k_per_frame: int = 50,
    token_insert_bonus: float = 0.0,
) -> List[List[int]]:
    """
    对一个 batch 做 CTC prefix beam search。

    Args:
        log_probs_btv: B × T × vocab_size。
        input_lengths: B。
        blank_id: CTC blank token id。
        beam_size: beam 大小。
        top_k_per_frame: 每帧扩展 top-k token。
        token_insert_bonus: token 长度奖励。

    Returns:
        每条样本的预测 token id 序列。
    """
    if log_probs_btv.ndim != 3:
        raise ValueError(f"log_probs_btv 应为 B×T×V，实际 shape={tuple(log_probs_btv.shape)}")

    # beam search 用 Python 循环，放 CPU 更稳
    log_probs_cpu = log_probs_btv.detach().cpu()
    input_lengths_cpu = input_lengths.detach().cpu()

    results: List[List[int]] = []

    batch_size = int(log_probs_cpu.shape[0])

    for batch_index in range(batch_size):
        decoded_ids = ctc_prefix_beam_search_one(
            log_probs_tv=log_probs_cpu[batch_index],
            input_length=int(input_lengths_cpu[batch_index].item()),
            blank_id=blank_id,
            beam_size=beam_size,
            top_k_per_frame=top_k_per_frame,
            token_insert_bonus=token_insert_bonus,
        )

        results.append(decoded_ids)

    return results