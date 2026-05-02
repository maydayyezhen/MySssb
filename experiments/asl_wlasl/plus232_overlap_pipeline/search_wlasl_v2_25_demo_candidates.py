# -*- coding: utf-8 -*-
"""Run the v2-25 full demo search using the existing v2-20 search pipeline.

The v2-20 search script already knows how to compose sentence videos, run
sliding-window inference, and summarize exact/deletion-only candidates. This
wrapper keeps that machinery but swaps in the active 25-word dataset, the
current semantic sentence set, and a validator that allows 2-4 word v2-25
sentences.
"""

from __future__ import annotations

import sys
from typing import List

import search_wlasl_v2_20_demo_candidates as search


V2_25_LABELS = {
    "bad",
    "deaf",
    "family",
    "friend",
    "go",
    "good",
    "help",
    "home",
    "language",
    "learn",
    "meet",
    "no",
    "please",
    "school",
    "sorry",
    "teacher",
    "today",
    "tomorrow",
    "want",
    "what",
    "who",
    "why",
    "work",
    "yes",
    "you",
}

V2_25_SENTENCES = [
    "friend,meet,today",
    "please,help,friend",
    "teacher,help,you",
    "you,want,help",
    "you,want,work,tomorrow",
    "you,go,school,today",
    "what,you,want",
    "who,help,teacher",
    "why,you,sorry",
    "deaf,friend,learn,language",
    "family,go,home",
    "no,work,today",
    "please,meet,teacher",
    "yes,you,go,school",
]

DEFAULT_ARGS = {
    "--samples_csv": "D:/datasets/WLASL-mini-v2-25/samples.csv",
    "--feature_dir": "D:/datasets/WLASL-mini-v2-25/features_20f_plus",
    "--model_dir": "D:/datasets/WLASL-mini-v2-25/models_20f_plus",
    "--video_output_dir": "D:/datasets/WLASL-mini-v2-25/demo_videos_trimmed/search_25",
    "--output_dir": "D:/datasets/WLASL-mini-v2-25/demo_search_25",
}


def validate_v2_25_sentence(sentence: str) -> None:
    labels = search.parse_sentence(sentence)
    if not 2 <= len(labels) <= 4:
        raise ValueError(f"v2-25 demo sentence must contain 2-4 words: {sentence}")

    invalid = [label for label in labels if label not in V2_25_LABELS]
    if invalid:
        raise ValueError(
            f"v2-25 demo sentence contains unknown labels: "
            f"sentence={sentence}, invalid={invalid}"
        )


def with_default_args(argv: List[str]) -> List[str]:
    patched = list(argv)
    existing_options = set(item for item in patched[1:] if item.startswith("--"))

    for option, value in DEFAULT_ARGS.items():
        if option not in existing_options:
            patched.extend([option, value])

    return patched


def main() -> None:
    search.ALLOWED_LABELS = set(V2_25_LABELS)
    search.REMOVED_LABELS = set()
    search.DEFAULT_SENTENCES = list(V2_25_SENTENCES)
    search.PREFERRED_MAINLINE_SENTENCES = set(V2_25_SENTENCES)
    search.validate_sentence = validate_v2_25_sentence

    sys.argv = with_default_args(sys.argv)
    search.main()


if __name__ == "__main__":
    main()
