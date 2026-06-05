"""Adapted inference: mix B_H into encoder hidden states and decode.

H_mix = alpha * H_audio + (1 - alpha) * B_H is injected into `generate` via
`encoder_outputs`, so the encoder runs once and the decoder cross-attention
sees the mixed state. alpha=1.0 reproduces the baseline exactly.
See docs/design_and_implementation.md §6.5.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers.modeling_outputs import BaseModelOutput


@torch.no_grad()
def encode_audio(model, input_features) -> torch.Tensor:
    """Run the Whisper encoder once -> H_audio [batch, 1500, d_model]."""
    return model.model.encoder(input_features).last_hidden_state


@torch.no_grad()
def adapted_generate(model, bias, input_features, alpha: float, **gen_kwargs):
    """Generate with H_mix = alpha*H_audio + (1-alpha)*B_H as encoder_outputs."""
    H_audio = encode_audio(model, input_features)
    H_mix = bias.mix(H_audio, alpha)
    encoder_outputs = BaseModelOutput(last_hidden_state=H_mix)
    return model.generate(encoder_outputs=encoder_outputs, **gen_kwargs)


@torch.no_grad()
def run_adapted(cfg, ckpt, limit_test=None, device=None, out_name="adapted_test_alpha_grid.jsonl"):
    """Run the full alpha grid on test and dump one row per (alpha, sample)."""
    from .bias_model import TrainableEncoderBias
    from ..common.data import load_splits, row_to_input_features
    from ..common.model_utils import build_frozen_model, gen_kwargs
    from ..common.paths import out_dir

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build_frozen_model(cfg, device)
    bias = TrainableEncoderBias(
        cfg["bias"]["encoder_seq_len"], model.config.d_model, init="zero"
    ).to(device)
    bias.load(ckpt)
    bias.B_H.data = bias.B_H.data.to(device)

    _, test_ds = load_splits(
        cfg["dataset"]["path"], cfg["dataset"]["train_split"], cfg["dataset"]["test_split"]
    )
    n = limit_test or test_ds.num_rows
    alphas = cfg["inference"]["alphas"]
    kwargs = gen_kwargs(cfg)

    out_path = os.path.join(out_dir("exp01", "eval_reports"), out_name)

    rows = []
    for i in range(n):
        row = test_ds[i]
        feats = row_to_input_features(row, processor).unsqueeze(0).to(device)
        for alpha in alphas:
            ids = adapted_generate(model, bias, feats, alpha=alpha, **kwargs)
            hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
            rows.append(
                {"chunk_id": row["chunk_id"], "alpha": alpha, "ref": row["text"], "hyp": hyp}
            )
        if (i + 1) % 20 == 0:
            print(f"adapted {i+1}/{n}", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {out_path} ({len(rows)} rows, alphas={alphas})", flush=True)
    return out_path


def _main():
    from ..common.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--out-name", default="adapted_test_alpha_grid.jsonl")
    args = ap.parse_args()
    run_adapted(load_config(args.config), args.ckpt, args.limit_test, out_name=args.out_name)


if __name__ == "__main__":
    _main()
