"""Exp03 Phase 5: Bregman / domain-weighted loss (paper eq.8, λ_BD term).

Up-weight cross-entropy on target tokens that belong to the domain vocabulary D
(built from TRAIN text only). Directly targets domain-term recall — our core
stuck metric. KL term is omitted (paper ambiguous; design §4c).

  weight_i = 1 + λ_BD · I(target_i token ∈ D)
  L = Σ_i weight_i · CE_i  /  Σ_i weight_i      (over non-ignored positions)

See docs/experiments/exp03_paper_faithful/design.md §4c.
"""
import pytest
import torch
import torch.nn.functional as F

from text_only_bias.exp03_paper_faithful.loss import (
    bregman_weighted_loss,
    build_domain_token_ids,
)


def test_lambda_zero_equals_plain_ce():
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    target = torch.randint(0, 50, (6,))
    got = bregman_weighted_loss(logits, target, domain_token_ids=set(), lambda_bd=0.0)
    expected = F.cross_entropy(logits, target)
    assert torch.allclose(got, expected, atol=1e-6)


def test_matches_manual_weighted_formula():
    torch.manual_seed(1)
    logits = torch.randn(5, 30)
    target = torch.randint(0, 30, (5,))
    domain_ids = {target[0].item(), target[2].item()}
    lam = 2.0

    ce = F.cross_entropy(logits, target, reduction="none")
    w = torch.tensor([1.0 + lam if t.item() in domain_ids else 1.0 for t in target])
    expected = (w * ce).sum() / w.sum()

    got = bregman_weighted_loss(logits, target, domain_token_ids=domain_ids, lambda_bd=lam)
    assert torch.allclose(got, expected, atol=1e-6)


def test_ignore_index_excluded():
    torch.manual_seed(2)
    logits = torch.randn(4, 20)
    target = torch.randint(0, 20, (4,))
    target[1] = -100  # masked position must not contribute
    domain_ids = {target[0].item()}
    lam = 1.5

    keep = target != -100
    ce = F.cross_entropy(logits[keep], target[keep], reduction="none")
    w = torch.tensor([1.0 + lam if t.item() in domain_ids else 1.0 for t in target[keep]])
    expected = (w * ce).sum() / w.sum()

    got = bregman_weighted_loss(logits, target, domain_token_ids=domain_ids, lambda_bd=lam)
    assert torch.allclose(got, expected, atol=1e-6)


def test_domain_error_penalized_more():
    """Same CE increase on a domain token raises loss more than on a non-domain token."""
    V = 10
    base = torch.full((2, V), 0.0)
    # both positions confidently correct
    base[0, 1] = 10.0
    base[1, 2] = 10.0
    target = torch.tensor([1, 2])
    domain_ids = {1}  # token 1 is a domain token, token 2 is not
    lam = 3.0

    # introduce equal error at the domain position vs the non-domain position
    err_domain = base.clone(); err_domain[0, 1] = 0.0      # ruin domain token pred
    err_nondom = base.clone(); err_nondom[1, 2] = 0.0      # ruin non-domain token pred

    l_dom = bregman_weighted_loss(err_domain, target, domain_ids, lambda_bd=lam)
    l_non = bregman_weighted_loss(err_nondom, target, domain_ids, lambda_bd=lam)
    assert l_dom > l_non


def test_build_domain_token_ids(whisper):
    tok = whisper["processor"].tokenizer
    terms = {"보험료", "급여상해수술비"}
    ids = build_domain_token_ids(terms, tok)
    assert isinstance(ids, set)
    assert len(ids) > 0
    # every id appears when tokenizing at least one of the terms
    all_term_ids = set()
    for t in terms:
        all_term_ids.update(tok(t, add_special_tokens=False).input_ids)
    assert ids == all_term_ids
