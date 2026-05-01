# -*- coding: utf-8 -*-
"""WLASL gloss to Chinese display mapping."""

LABEL_ZH_MAP = {
    "bad": "不好",
    "deaf": "听障",
    "family": "家人",
    "friend": "朋友",
    "go": "去",
    "good": "好",
    "help": "帮助",
    "home": "家",
    "language": "语言",
    "learn": "学习",
    "meet": "见面",
    "no": "不",
    "please": "请",
    "school": "学校",
    "sorry": "抱歉",
    "teacher": "老师",
    "today": "今天",
    "tomorrow": "明天",
    "want": "想要",
    "what": "什么",
    "who": "谁",
    "why": "为什么",
    "work": "工作",
    "yes": "是",
    "you": "你",
}


def to_zh(label: str) -> str:
    """Map an English gloss to Chinese display text."""
    return LABEL_ZH_MAP.get(label, label)


def sequence_to_zh_text(sequence: list[str]) -> str:
    """Convert a gloss sequence to Chinese display text."""
    return " ".join(to_zh(item) for item in sequence)
