"""Exp03 Phase 5: Bregman / domain-weighted loss (paper eq.8, λ_BD term).

Standard next-token CE, but errors on target tokens belonging to the domain
vocabulary D are up-weighted, steering the model toward domain-term recall.

  weight_i = 1 + λ_BD · I(target_i ∈ D_token_ids)
  L = Σ_i weight_i · CE_i / Σ_i weight_i   (over non-ignored positions)

D is built from TRAIN text only (build_domain_token_ids over metrics.build_domain_terms),
keeping the leakage guard. KL term omitted (paper ambiguous; design §4c).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_domain_token_ids(domain_terms, tokenizer) -> set:
    """Map domain *words* (from train text) to the set of subword token ids they use.

    Token-level membership is an approximation of phrase-level domain terms, but it
    is what the per-token CE weighting needs. Special tokens are excluded.
    """
    ids = set()
    for term in domain_terms:
        ids.update(tokenizer(term, add_special_tokens=False).input_ids)
    return ids


def bregman_weighted_loss(logits, target_labels, domain_token_ids, lambda_bd: float,
                          ignore_index: int = -100):
    """Domain-weighted cross-entropy.

    logits: [..., V] ; target_labels: [...] (matching leading dims). λ_BD=0 and an
    empty domain set both reduce exactly to plain mean cross-entropy.
    """
    V = logits.size(-1)
    flat_logits = logits.reshape(-1, V)
    flat_target = target_labels.reshape(-1)

    ce = F.cross_entropy(flat_logits, flat_target, ignore_index=ignore_index, reduction="none")
    keep = flat_target != ignore_index

    if lambda_bd and domain_token_ids:
        dom = torch.zeros(V + 1, dtype=torch.bool, device=flat_logits.device)
        valid = [i for i in domain_token_ids if 0 <= i < V]
        if valid:
            dom[torch.tensor(valid, device=flat_logits.device)] = True
        # safe-index: ignored targets (-100) map to the sentinel row V (always False)
        safe = torch.where(keep, flat_target, torch.full_like(flat_target, V))
        is_dom = dom[safe]
        weight = 1.0 + lambda_bd * is_dom.to(ce.dtype)
    else:
        weight = torch.ones_like(ce)

    weight = weight * keep.to(ce.dtype)
    denom = weight.sum().clamp_min(1e-8)
    return (weight * ce).sum() / denom
