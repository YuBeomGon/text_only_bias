"""Data pipeline: HF dataset load, audio float conversion, tokenize + collator.

The dataset stores `audio` as a raw float list (NOT a datasets.Audio feature),
with a separate `sampling_rate` column (16000). See docs/dataset_description.md §3.

Training uses train TEXT only (no audio). Evaluation uses test audio + text.
"""
from __future__ import annotations

import numpy as np
import torch


def load_splits(path: str, train_split: str = "train", test_split: str = "validation"):
    """Return (train_ds, test_ds). Existing 'validation' is used as test (no valid)."""
    from datasets import load_from_disk

    ds = load_from_disk(path)
    return ds[train_split], ds[test_split]


def audio_to_array(row) -> np.ndarray:
    """Raw float list -> float32 waveform array."""
    return np.asarray(row["audio"], dtype=np.float32)


def row_to_input_features(row, processor) -> torch.Tensor:
    """Row -> Whisper log-mel input features [n_mels, 3000] (30s padded)."""
    wav = audio_to_array(row)
    feats = processor.feature_extractor(
        wav, sampling_rate=int(row["sampling_rate"]), return_tensors="pt"
    ).input_features
    return feats[0]


class TextOnlyCollator:
    """Turn a list of transcripts into shifted decoder inputs + masked targets.

    decoder_input_ids = labels[:, :-1]
    labels (target)   = labels[:, 1:], pad -> -100
    Labels carry the Whisper special prefix (<|sot|><|ko|><|transcribe|><|notimestamps|>).
    """

    def __init__(self, processor, language: str = "ko", task: str = "transcribe", max_label_length: int = 448):
        self.tok = processor.tokenizer
        self.tok.set_prefix_tokens(language=language, task=task, predict_timestamps=False)
        self.max_label_length = max_label_length

    def __call__(self, texts):
        if isinstance(texts, dict):  # datasets batched dict
            texts = texts["text"]
        enc = self.tok(
            text_target=list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt",
        )
        labels = enc.input_ids
        decoder_input_ids = labels[:, :-1].contiguous()
        target = labels[:, 1:].clone()
        target[target == self.tok.pad_token_id] = -100
        return {"decoder_input_ids": decoder_input_ids, "labels": target}
