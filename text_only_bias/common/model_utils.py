"""Shared model helpers used by every experiment."""
from __future__ import annotations


def build_frozen_model(cfg, device):
    """Load Whisper, move to device, eval + freeze every parameter."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    name = cfg["model"]["name_or_path"]
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def gen_kwargs(cfg):
    """Decoding conditions shared by baseline and all adapted evals (guide §14)."""
    icfg = cfg["inference"]
    return dict(
        language=cfg["model"]["language"],
        task=cfg["model"]["task"],
        num_beams=icfg["beam_size"],
        do_sample=False,
    )
