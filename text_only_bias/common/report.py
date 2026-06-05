"""Build the final comparison report: baseline vs adapted alpha grid (method A).

Reports the full alpha curve, not a single hand-picked best alpha
(docs/design_and_implementation.md §8).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import jiwer

from . import metrics as M


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _nanmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def compute_condition(refs, hyps, terms, cfg):
    norm = M.normalize_text if cfg["metrics"]["normalize_text"] else (lambda s: s)
    refs_n = [norm(r) for r in refs]
    hyps_n = [norm(h) for h in hyps]

    ec = M.error_counts(refs_n, hyps_n)
    rep_n = cfg["metrics"]["repeat_ngram"]
    lr_th = cfg["metrics"]["length_ratio_threshold"]
    lrs = [M.length_ratio(r, h) for r, h in zip(refs_n, hyps_n)]

    return {
        "cer": float(jiwer.cer(refs_n, hyps_n)),
        "wer": float(jiwer.wer(refs_n, hyps_n)),
        "sub": ec["substitutions"],
        "del": ec["deletions"],
        "ins": ec["insertions"],
        "dt_recall": _nanmean([M.domain_term_recall(r, h, terms) for r, h in zip(refs, hyps)]),
        "dt_precision": _nanmean([M.domain_term_precision(r, h, terms) for r, h in zip(refs, hyps)]),
        "len_ratio": _nanmean(lrs),
        "n_len_gt": sum(1 for x in lrs if x > lr_th),
        "n_repeat": sum(1 for h in hyps_n if M.max_repeated_ngram(h, rep_n) >= 2),
    }


def build_report(cfg, baseline_path, adapted_path, out_name="final_test_comparison.md", exp="common"):
    from .data import load_splits
    from .paths import out_dir

    baseline = _read_jsonl(baseline_path)
    adapted = _read_jsonl(adapted_path)

    # Fair comparison: restrict baseline to the same chunk_ids present in adapted
    # (adapted may cover only a subset of the test set, e.g. --limit-test).
    keep = {r["chunk_id"] for r in adapted}
    baseline = [r for r in baseline if r["chunk_id"] in keep]

    train_ds, _ = load_splits(
        cfg["dataset"]["path"], cfg["dataset"]["train_split"], cfg["dataset"]["test_split"]
    )
    terms = M.build_domain_terms(train_ds[cfg["dataset"]["text_column"]])

    conditions = {}
    conditions["baseline"] = compute_condition(
        [r["ref"] for r in baseline], [r["hyp"] for r in baseline], terms, cfg
    )
    by_alpha = defaultdict(list)
    for r in adapted:
        by_alpha[r["alpha"]].append(r)
    for alpha in sorted(by_alpha, reverse=True):
        rows = by_alpha[alpha]
        conditions[f"alpha={alpha}"] = compute_condition(
            [r["ref"] for r in rows], [r["hyp"] for r in rows], terms, cfg
        )

    header = (
        "| condition | CER | WER | sub | del | ins | dt_recall | dt_prec | len_ratio | len>th | repeat |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for name, m in conditions.items():
        lines.append(
            f"| {name} | {m['cer']:.4f} | {m['wer']:.4f} | {m['sub']} | {m['del']} | {m['ins']} | "
            f"{m['dt_recall']:.3f} | {m['dt_precision']:.3f} | {m['len_ratio']:.3f} | "
            f"{m['n_len_gt']} | {m['n_repeat']} |"
        )

    out_path = os.path.join(out_dir(exp, "eval_reports"), out_name)
    with open(out_path, "w") as f:
        f.write("# Final test comparison (method A: full alpha grid)\n\n")
        f.write(f"- domain terms (from train text): {len(terms)}\n")
        f.write(f"- baseline samples: {len(baseline)} / adapted rows: {len(adapted)}\n\n")
        f.write(header)
        f.write("\n".join(lines) + "\n\n")
        f.write(
            "> 해석: `alpha=1.0`은 baseline sanity check(거의 동일해야 함). "
            "dt_recall↑ + CER/WER↓ + ins/len>th/repeat 증가 없음이면 성공. "
            "특정 alpha에만 민감하면 보류.\n"
        )
    print(f"saved {out_path}", flush=True)
    return out_path


def _main():
    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--adapted", required=True)
    ap.add_argument("--out-name", default="final_test_comparison.md")
    ap.add_argument("--exp", default="common")
    args = ap.parse_args()
    build_report(load_config(args.config), args.baseline, args.adapted, args.out_name, exp=args.exp)


if __name__ == "__main__":
    _main()
