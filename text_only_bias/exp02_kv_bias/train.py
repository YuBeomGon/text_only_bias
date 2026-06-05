"""Exp02: text-only training of direct K/V bias (B_K, B_V per layer).

Whisper is fully frozen; only B_K/B_V train. Audio/encoder are never used: the
KVBias hook (in 'train' mode) replaces each cross-attention K/V with B directly.
See docs/experiments/exp02_kv_bias/design.md §5.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F


def kv_text_only_loss(model, bias, decoder_input_ids, target_labels, ignore_index: int = -100):
    """Decoder forward with K/V = B (audio ignored) -> proj_out -> CE loss.

    `bias` must already be attached to `model`. Sets the hook to 'train' mode.
    A dummy encoder_hidden_states is passed only to trigger cross-attention;
    the hook replaces the projected K/V with B regardless of its content.
    """
    bias.set_mode("train")
    bsz = decoder_input_ids.shape[0]
    dtype = next(model.parameters()).dtype
    dummy = torch.zeros(bsz, 1, model.config.d_model, device=decoder_input_ids.device, dtype=dtype)
    decoder_outputs = model.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=dummy,
        use_cache=False,
    )
    logits = model.proj_out(decoder_outputs.last_hidden_state)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return loss, logits


def train(cfg, mode, limit_train=None, max_steps=None, lr=None, out_name=None, device=None,
          include_test_text=False):
    """Train K/V bias for a given mode ('A' or 'B').

    include_test_text=True is a DELIBERATE-LEAKAGE oracle/upper-bound probe: it
    adds the test split's transcripts to the training text so the domain signal
    is guaranteed present. NOT a valid generalization result.
    """
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from .kv_bias_model import KVBias
    from ..common.data import load_splits, TextOnlyCollator
    from ..common.model_utils import build_frozen_model
    from ..common.paths import out_dir

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    kcfg = cfg.get("kv_bias", {})
    lr = lr if lr is not None else kcfg.get("learning_rate", 3e-4)
    out_name = out_name or f"kv_bias_{mode}_final.pt"
    torch.manual_seed(tcfg.get("seed", 42))

    model, processor = build_frozen_model(cfg, device)

    train_ds, test_ds = load_splits(
        cfg["dataset"]["path"], cfg["dataset"]["train_split"], cfg["dataset"]["test_split"]
    )
    text_col = cfg["dataset"]["text_column"]
    texts = list(train_ds[text_col])
    if include_test_text:
        texts += list(test_ds[text_col])
        print(f"[ORACLE] include_test_text=True : +{test_ds.num_rows} test transcripts (DELIBERATE LEAKAGE)", flush=True)
    if limit_train:
        texts = texts[:limit_train]
    print(f"[{mode}] training texts: {len(texts)}", flush=True)

    collator = TextOnlyCollator(
        processor, cfg["model"]["language"], cfg["model"]["task"], tcfg["max_label_length"]
    )
    loader = DataLoader(texts, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collator)

    bias = KVBias(
        num_layers=model.config.decoder_layers,
        embed_dim=model.config.d_model,
        mode=mode,
        length=kcfg.get("length", 1500),
        M=kcfg.get("M", 32),
        init_std=kcfg.get("init_std", 0.02),
    ).to(device)
    bias.attach(model)

    opt = AdamW(bias.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    epochs = kcfg.get("epochs", tcfg["epochs"])
    total_steps = max_steps or (len(loader) * epochs)
    sched = get_linear_schedule_with_warmup(opt, tcfg["warmup_steps"], total_steps)

    ckpt_dir = out_dir("exp02", "checkpoints")

    step = 0
    bias.train()
    for epoch in range(epochs):
        for batch in loader:
            din = batch["decoder_input_ids"].to(device)
            lab = batch["labels"].to(device)
            loss, _ = kv_text_only_loss(model, bias, din, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bias.parameters(), tcfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % tcfg["log_steps"] == 0:
                print(f"[{mode}] epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    bias.detach()
    final_path = os.path.join(ckpt_dir, out_name)
    bias.save(final_path)
    print(f"saved {final_path} (mode={mode}, steps={step}, lr={lr})", flush=True)
    return final_path


def _main():
    from ..common.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--include-test-text", action="store_true",
                    help="ORACLE probe: add test transcripts to training (deliberate leakage)")
    args = ap.parse_args()
    train(load_config(args.config), args.mode, args.limit_train, args.max_steps, args.lr,
          args.out_name, include_test_text=args.include_test_text)


if __name__ == "__main__":
    _main()
