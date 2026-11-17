"""Token estimate for prompt budgets."""

import math


def est_tokens(text: str) -> int:
    # shortcut: chars/4 is ~10-20% off for Vietnamese, use a real tokenizer if budgets get tight
    return math.ceil(len(text) / 4)
