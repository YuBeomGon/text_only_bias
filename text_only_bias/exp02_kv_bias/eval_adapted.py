"""Exp02: adapted inference with direct K/V bias (Option A or B).

Attaches KVBias hooks, loads a trained checkpoint, and runs the strength grid
(A: alpha grid; B: gate g grid). Decoding conditions are shared with
eval_baseline.gen_kwargs so only the bias differs. The grid value is stored
under the "alpha" key so report.py works unchanged for both modes.
See docs/experiments/exp02_kv_bias/design.md §6.
"""
from __future__ import annotations

import argparse
import json
import os

import torch


@torch.no_grad()
def run_kv_adapted(cfg, ckpt, mode, limit_test=None, device=None, out_name=None,
                   exp="exp02", decoder_ckpt=None):
    from .kv_bias_model import KVBias
    from ..common.data import load_splits, row_to_input_features
    from ..common.model_utils import build_frozen_model, gen_kwargs
    from ..common.paths import out_dir

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kcfg = cfg.get("kv_bias", {})
    out_name = out_name or f"kv_adapted_{mode}_grid.jsonl"

    model, processor = build_frozen_model(cfg, device)
    bias = KVBias(
        num_layers=model.config.decoder_layers,
        embed_dim=model.config.d_model,
        mode=mode,
        length=kcfg.get("length", 1500),
        M=kcfg.get("M", 32),
    ).to(device)
    bias.load(ckpt)
    bias = bias.to(device)
    bias.attach(model)

    # exp03 phase 3: load fine-tuned decoder weights if provided
    if decoder_ckpt:
        model.model.decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
        print(f"loaded fine-tuned decoder from {decoder_ckpt}", flush=True)

    _, test_ds = load_splits(
        cfg["dataset"]["path"], cfg["dataset"]["train_split"], cfg["dataset"]["test_split"]
    )
    n = limit_test or test_ds.num_rows
    grid = kcfg.get("alpha_grid") if mode == "A" else kcfg.get("g_grid")
    kwargs = gen_kwargs(cfg)

    out_path = os.path.join(out_dir(exp, "eval_reports"), out_name)

    rows = []
    try:
        for i in range(n):
            row = test_ds[i]
            feats = row_to_input_features(row, processor).unsqueeze(0).to(device)
            for s in grid:
                if mode == "A":
                    bias.set_mode("A", alpha=s)
                else:
                    bias.set_mode("B", g=s)
                ids = model.generate(feats, **kwargs)
                hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
                rows.append({"chunk_id": row["chunk_id"], "alpha": s, "ref": row["text"], "hyp": hyp})
            if (i + 1) % 20 == 0:
                print(f"kv-adapted[{mode}] {i+1}/{n}", flush=True)
    finally:
        bias.detach()

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {out_path} ({len(rows)} rows, mode={mode}, grid={grid})", flush=True)
    return out_path


def _main():
    from ..common.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--exp", default="exp02")
    ap.add_argument("--decoder-ckpt", default=None)
    args = ap.parse_args()
    run_kv_adapted(load_config(args.config), args.ckpt, args.mode, args.limit_test,
                   out_name=args.out_name, exp=args.exp, decoder_ckpt=args.decoder_ckpt)


if __name__ == "__main__":
    _main()
