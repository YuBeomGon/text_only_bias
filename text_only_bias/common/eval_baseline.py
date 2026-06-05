"""Baseline evaluation: plain Whisper on the test split.

Decoding conditions (gen_kwargs in model_utils) are shared by every adapted eval
(guide §14) so the only difference between baseline and adapted is the bias.
Baseline output goes to outputs/common/eval_reports (reused by all experiments).
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .data import load_splits, row_to_input_features
from .model_utils import build_frozen_model, gen_kwargs
from .paths import out_dir


@torch.no_grad()
def run_baseline(cfg, limit_test=None, device=None, out_name="baseline_test.jsonl", exp="common"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build_frozen_model(cfg, device)
    _, test_ds = load_splits(
        cfg["dataset"]["path"], cfg["dataset"]["train_split"], cfg["dataset"]["test_split"]
    )
    n = limit_test or test_ds.num_rows
    kwargs = gen_kwargs(cfg)

    out_path = os.path.join(out_dir(exp, "eval_reports"), out_name)

    rows = []
    for i in range(n):
        row = test_ds[i]
        feats = row_to_input_features(row, processor).unsqueeze(0).to(device)
        ids = model.generate(feats, **kwargs)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        rows.append({"chunk_id": row["chunk_id"], "ref": row["text"], "hyp": hyp})
        if (i + 1) % 20 == 0:
            print(f"baseline {i+1}/{n}", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {out_path} ({len(rows)} rows)", flush=True)
    return out_path


def _main():
    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--out-name", default="baseline_test.jsonl")
    ap.add_argument("--exp", default="common")
    args = ap.parse_args()
    run_baseline(load_config(args.config), args.limit_test, out_name=args.out_name, exp=args.exp)


if __name__ == "__main__":
    _main()
