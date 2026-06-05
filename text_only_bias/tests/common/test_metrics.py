import math

from text_only_bias.common.metrics import (
    normalize_text,
    cer,
    wer,
    error_counts,
    build_domain_terms,
    domain_term_recall,
    domain_term_precision,
    length_ratio,
    max_repeated_ngram,
)


def test_normalize_collapses_whitespace():
    assert normalize_text("  안녕   하세요 ") == "안녕 하세요"


def test_cer_simple():
    # ref 4 chars, 1 deletion -> 0.25
    assert math.isclose(cer("abcd", "abc"), 0.25, rel_tol=1e-6)


def test_wer_simple():
    # 4 words, 1 deletion -> 0.25
    assert math.isclose(wer("a b c d", "a b c"), 0.25, rel_tol=1e-6)


def test_error_counts_breakdown():
    # ref: a b c d ; hyp: a x c   -> 1 sub (b->x), 1 del (d), 0 ins, 2 hits
    out = error_counts(["a b c d"], ["a x c"])
    assert out["substitutions"] == 1
    assert out["deletions"] == 1
    assert out["insertions"] == 0
    assert out["hits"] == 2


def test_build_domain_terms_by_frequency():
    texts = ["보험 상담 보험", "보험 안내"]
    terms = build_domain_terms(texts, min_count=2, min_len=2)
    assert "보험" in terms
    assert "상담" not in terms  # appears once
    assert "안내" not in terms


def test_domain_term_recall_and_precision():
    terms = {"보험", "간병"}
    rec = domain_term_recall("보험 간병 안내", "보험 안내", terms)
    pre = domain_term_precision("보험 간병 안내", "보험 안내", terms)
    assert math.isclose(rec, 0.5, rel_tol=1e-6)   # 보험 hit, 간병 missed
    assert math.isclose(pre, 1.0, rel_tol=1e-6)   # 보험 predicted and correct


def test_length_ratio():
    assert math.isclose(length_ratio("abcd", "abcdef"), 1.5, rel_tol=1e-6)


def test_max_repeated_ngram():
    # "a b" repeats 3 times
    assert max_repeated_ngram("a b a b a b", n=2) == 3
    assert max_repeated_ngram("a b c d", n=2) == 1
