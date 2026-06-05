"""Evaluation metrics: CER/WER, error breakdown, domain-term R/P, hallucination.

Domain-term lists must be built from TRAIN text only (never test). See guide §15.
"""
from __future__ import annotations

import re
from collections import Counter

import jiwer


def normalize_text(s: str) -> str:
    """Collapse whitespace and strip. (Light normalization for fair comparison.)"""
    return re.sub(r"\s+", " ", s).strip()


def cer(reference: str, hypothesis: str) -> float:
    return float(jiwer.cer(reference, hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    return float(jiwer.wer(reference, hypothesis))


def error_counts(references, hypotheses) -> dict:
    """Word-level substitution/deletion/insertion/hits aggregated over a corpus."""
    out = jiwer.process_words(list(references), list(hypotheses))
    return {
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
        "wer": float(out.wer),
    }


def build_domain_terms(texts, min_count: int = 2, min_len: int = 2) -> set:
    """Domain vocabulary from train texts: words seen >= min_count, length >= min_len."""
    counter = Counter()
    for t in texts:
        for w in normalize_text(t).split():
            if len(w) >= min_len:
                counter[w] += 1
    return {w for w, c in counter.items() if c >= min_count}


def domain_term_recall(reference: str, hypothesis: str, terms: set) -> float:
    ref_words = set(normalize_text(reference).split())
    hyp_words = set(normalize_text(hypothesis).split())
    ref_terms = ref_words & terms
    if not ref_terms:
        return float("nan")
    return len(ref_terms & hyp_words) / len(ref_terms)


def domain_term_precision(reference: str, hypothesis: str, terms: set) -> float:
    ref_words = set(normalize_text(reference).split())
    hyp_words = set(normalize_text(hypothesis).split())
    hyp_terms = hyp_words & terms
    if not hyp_terms:
        return float("nan")
    return len(hyp_terms & ref_words) / len(hyp_terms)


def length_ratio(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    if len(ref) == 0:
        return float("inf") if normalize_text(hypothesis) else 0.0
    return len(normalize_text(hypothesis)) / len(ref)


def max_repeated_ngram(text: str, n: int = 5) -> int:
    """Highest occurrence count of any word-level n-gram (repetition / hallucination)."""
    words = normalize_text(text).split()
    if len(words) < n:
        return 0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(grams.values())
