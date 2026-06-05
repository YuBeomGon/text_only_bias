"""Exp03: paper-faithful K/V bias training.

Builds on exp02's KVBias (mode A) but adds the paper's missing pieces, staged:
  Phase 2 (--decoder none): E_pretrained init (eq.4) + B-only training.
  Phase 3 (--decoder full): also fine-tune the decoder (encoder frozen).

Audio is used ONLY to seed B via init_from_encoder; the training loss uses
train TEXT only (leakage-safe). See docs/experiments/exp03_paper_faithful/design.md.
"""
from __future__ import annotations

import argparse

import torch

from ..exp02_kv_bias.kv_bias_model import KVBias
from ..exp02_kv_bias.train import kv_text_only_loss


def _init_features(cfg, model, processor, k, device):
    """Stack K train-audio clips -> input_features for E_pretrained init (seed only)."""
    from ..common.data import load_splits, row_to_input_features

    train_ds, _ = load_splits(
        cfg["dataset"]["path"], cfg["dataset"]["train_split"], cfg["dataset"]["test_split"]
    )
    feats = [row_to_input_features(train_ds[i], processor) for i in range(min(k, train_ds.num_rows))]
    return torch.stack(feats).to(device)


def train(cfg, init_samples=16, decoder="none", limit_train=None, max_steps=None,
          lr=None, out_name=None, device=None):
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from ..common.data import load_splits, TextOnlyCollator
    from ..common.model_utils import build_frozen_model
    from ..common.paths import out_dir

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    kcfg = cfg.get("kv_bias", {})
    lr = lr if lr is not None else kcfg.get("learning_rate", 3e-4)
    out_name = out_name or f"exp03_A_{decoder}.pt"
    torch.manual_seed(tcfg.get("seed", 42))

    model, processor = build_frozen_model(cfg, device)

    bias = KVBias(
        num_layers=model.config.decoder_layers,
        embed_dim=model.config.d_model,
        mode="A",
        length=kcfg.get("length", 1500),
        init_std=kcfg.get("init_std", 0.02),
    ).to(device)

    # #1 E_pretrained init (paper eq.4) — audio used as seed only
    print(f"[exp03] E_pretrained init from {init_samples} train-audio clips (seed only)", flush=True)
    bias.init_from_encoder(model, _init_features(cfg, model, processor, init_samples, device))
    bias.attach(model)

    # #3 decoder fine-tune (optional)
    params = list(bias.parameters())
    if decoder == "full":
        dec = model.model.decoder
        for p in dec.parameters():
            p.requires_grad_(True)
        dec.train()
        params += [p for p in dec.parameters()]
        print("[exp03] decoder FULL fine-tune enabled", flush=True)
    elif decoder != "none":
        raise ValueError(f"--decoder {decoder} not supported yet (none|full)")

    # train TEXT only
    train_ds, _ = load_splits(
        cfg["dataset"]["path"], cfg["dataset"]["train_split"], cfg["dataset"]["test_split"]
    )
    texts = list(train_ds[cfg["dataset"]["text_column"]])
    if limit_train:
        texts = texts[:limit_train]
    collator = TextOnlyCollator(
        processor, cfg["model"]["language"], cfg["model"]["task"], tcfg["max_label_length"]
    )
    loader = DataLoader(texts, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collator)

    opt = AdamW(params, lr=lr, weight_decay=tcfg["weight_decay"])
    epochs = kcfg.get("epochs", tcfg["epochs"])
    total_steps = max_steps or (len(loader) * epochs)
    sched = get_linear_schedule_with_warmup(opt, tcfg["warmup_steps"], total_steps)

    ckpt_dir = out_dir("exp03", "checkpoints")
    step = 0
    bias.train()
    for epoch in range(epochs):
        for batch in loader:
            din = batch["decoder_input_ids"].to(device)
            lab = batch["labels"].to(device)
            loss, _ = kv_text_only_loss(model, bias, din, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % tcfg["log_steps"] == 0:
                print(f"[exp03/{decoder}] epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    bias.detach()
    import os
    path = os.path.join(ckpt_dir, out_name)
    bias.save(path)
    if decoder == "full":
        torch.save(model.model.decoder.state_dict(), path.replace(".pt", "_decoder.pt"))
    print(f"saved {path} (decoder={decoder}, steps={step}, lr={lr})", flush=True)
    return path


def _main():
    from ..common.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--init-samples", type=int, default=16)
    ap.add_argument("--decoder", choices=["none", "full"], default="none")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    train(load_config(args.config), args.init_samples, args.decoder, args.limit_train,
          args.max_steps, args.lr, args.out_name)


if __name__ == "__main__":
    _main()
