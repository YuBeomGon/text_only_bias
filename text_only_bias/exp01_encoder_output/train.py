"""Text-only training of the encoder-output bias B_H.

The whole Whisper model is frozen; only B_H is trained. Audio / encoder are
never used during training: the decoder's cross-attention reads B_H directly
via the `encoder_hidden_states` slot. See docs/design_and_implementation.md §6.3.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F


def text_only_loss(model, encoder_hidden_states, decoder_input_ids, target_labels, ignore_index: int = -100):
    """Run decoder + proj_out with B_H as encoder_hidden_states and compute CE loss.

    Args:
        model: WhisperForConditionalGeneration (frozen).
        encoder_hidden_states: [batch, encoder_seq_len, d_model]  (B_H expanded).
        decoder_input_ids: [batch, t]  (labels shifted right).
        target_labels: [batch, t]  (labels shifted left, pad -> ignore_index).

    Returns:
        (loss, logits)
    """
    decoder_outputs = model.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=False,
    )
    logits = model.proj_out(decoder_outputs.last_hidden_state)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return loss, logits


def train(cfg, limit_train=None, max_steps=None, lr=None, out_name="bias_final.pt", device=None):
    """Train B_H on train TEXT only (audio/encoder never used)."""
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from .bias_model import TrainableEncoderBias
    from ..common.data import load_splits, TextOnlyCollator
    from ..common.model_utils import build_frozen_model
    from ..common.paths import out_dir

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tcfg, bcfg = cfg["training"], cfg["bias"]
    lr = lr if lr is not None else tcfg["learning_rate"]
    torch.manual_seed(tcfg.get("seed", 42))

    model, processor = build_frozen_model(cfg, device)

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

    bias = TrainableEncoderBias(
        bcfg["encoder_seq_len"], model.config.d_model, bcfg["init"], bcfg["init_std"]
    ).to(device)

    opt = AdamW(bias.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    total_steps = (max_steps or (len(loader) * tcfg["epochs"]))
    sched = get_linear_schedule_with_warmup(opt, tcfg["warmup_steps"], total_steps)

    ckpt_dir = out_dir("exp01", "checkpoints")

    step = 0
    bias.train()
    for epoch in range(tcfg["epochs"]):
        for batch in loader:
            din = batch["decoder_input_ids"].to(device)
            lab = batch["labels"].to(device)
            enc = bias.forward(din.size(0))
            loss, _ = text_only_loss(model, enc, din, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bias.parameters(), tcfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % tcfg["log_steps"] == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
            if step % tcfg["save_steps"] == 0:
                bias.save(os.path.join(ckpt_dir, f"bias_step_{step}.pt"))
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    final_path = os.path.join(ckpt_dir, out_name)
    bias.save(final_path)
    print(f"saved {final_path} (steps={step}, lr={lr})", flush=True)
    return final_path


def _main():
    from ..common.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-name", default="bias_final.pt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.limit_train, args.max_steps, args.lr, args.out_name)


if __name__ == "__main__":
    _main()
