"""Standard SQuAD-style answer scoring. Part of the locked harness — do not edit."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match_score(prediction: str, gold_answers: list[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(g) for g in gold_answers))


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
