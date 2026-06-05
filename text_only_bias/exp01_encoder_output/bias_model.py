"""Trainable encoder-output bias (B_H) for Whisper text-only domain adaptation.

B_H is NOT an encoder output. It is a trainable domain-prior tensor that is
fed into the decoder's cross-attention in the `encoder_hidden_states` slot
(same shape as the encoder's last_hidden_state). See docs/design_and_implementation.md §1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrainableEncoderBias(nn.Module):
    def __init__(
        self,
        encoder_seq_len: int,
        d_model: int,
        init: str = "normal",
        init_std: float = 0.02,
    ):
        super().__init__()
        if init == "zero":
            tensor = torch.zeros(encoder_seq_len, d_model)
        elif init == "normal":
            tensor = torch.randn(encoder_seq_len, d_model) * init_std
        else:
            raise ValueError(f"unknown init: {init!r} (expected 'normal' or 'zero')")
        self.B_H = nn.Parameter(tensor)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Expand B_H to [batch, encoder_seq_len, d_model]."""
        return self.B_H.unsqueeze(0).expand(batch_size, -1, -1)

    def mix(self, encoder_hidden_states: torch.Tensor, alpha: float) -> torch.Tensor:
        """H_mix = alpha * H_audio + (1 - alpha) * B_H (broadcast over batch).

        alpha == 1.0 reproduces the audio path exactly (baseline sanity check).
        """
        return alpha * encoder_hidden_states + (1.0 - alpha) * self.B_H.unsqueeze(0)

    def save(self, path) -> None:
        torch.save({"B_H": self.B_H.detach().cpu()}, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location="cpu")
        with torch.no_grad():
            self.B_H.copy_(state["B_H"])
