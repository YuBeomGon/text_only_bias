"""Exp02: direct K/V bias injected into Whisper decoder cross-attention.

We register forward hooks on each decoder layer's `encoder_attn.k_proj` / `v_proj`
and modify their output (the projected K / V, shape [batch, seq, d]) before the
attention head reshape. The reshape uses `view(bsz, -1, heads, head_dim)`, so
changing the seq length (Option B concat) is handled automatically.

Two inference formulations (training is shared: K/V = B, audio ignored):
  - Option A (blend):  K' = alpha*K + (1-alpha)*B_K          B_K: [length, d]
  - Option B (concat): K' = [K ; B_K], V' = [V ; g*B_V]      B_K: [M, d]

Baseline sanity: A at alpha=1.0 and B at g=0 reproduce plain Whisper exactly.
See docs/experiments/exp02_kv_bias/design.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class KVBias(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mode: str = "A",
        length: int = 1500,
        M: int = 32,
        init_std: float = 0.02,
        use_gate: bool = False,
    ):
        super().__init__()
        if mode not in ("A", "B"):
            raise ValueError(f"mode must be 'A' or 'B', got {mode!r}")
        self.mode = mode
        self.embed_dim = embed_dim
        self.L = length if mode == "A" else M
        self.B_K = nn.ParameterList(
            [nn.Parameter(torch.randn(self.L, embed_dim) * init_std) for _ in range(num_layers)]
        )
        self.B_V = nn.ParameterList(
            [nn.Parameter(torch.randn(self.L, embed_dim) * init_std) for _ in range(num_layers)]
        )
        # ---- gating (paper eq.3): B_gated = tanh(W_g.B) ⊙ B, per-layer W_g [d,d]
        # W_g zero-initialized => gate starts CLOSED (tanh(0)=0 -> B_gated=0), so
        # training begins from baseline behavior and learns to open the gate onto
        # the scale-matched init underneath. One W_g per layer, shared by K and V.
        self.use_gate = use_gate
        if use_gate:
            self.W_g = nn.ModuleList(
                [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
            )
            for w in self.W_g:
                nn.init.zeros_(w.weight)
        self._handles = []
        # runtime state read by hooks: "train" | "A" | "B"
        self._runtime = {"mode": "train", "alpha": 1.0, "g": 1.0}

    def gated(self, B, layer_idx: int = 0):
        """Apply tanh gate (eq.3): B_gated = tanh(W_g_l · B) ⊙ B. No-op if disabled."""
        if not self.use_gate:
            return B
        return torch.tanh(self.W_g[layer_idx](B)) * B

    # ---- hooks ----------------------------------------------------------
    def _make_hook(self, layer_idx: int, which: str):
        params = self.B_K if which == "K" else self.B_V

        def hook(module, inputs, output):
            B = self.gated(params[layer_idx], layer_idx)
            bsz = output.shape[0]
            Bb = B.unsqueeze(0).expand(bsz, -1, -1).to(output.dtype).to(output.device)
            rt = self._runtime
            mode = rt["mode"]
            if mode == "train":
                return Bb  # audio ignored: K/V = B
            if mode == "A":
                a = rt["alpha"]
                return a * output + (1.0 - a) * Bb
            if mode == "B":
                g = rt["g"]
                if g == 0.0:
                    return output  # exact baseline: no memory slots
                scaled = Bb if which == "K" else g * Bb
                return torch.cat([output, scaled], dim=1)
            return output

        return hook

    def attach(self, model) -> None:
        layers = model.model.decoder.layers
        for i, layer in enumerate(layers):
            self._handles.append(layer.encoder_attn.k_proj.register_forward_hook(self._make_hook(i, "K")))
            self._handles.append(layer.encoder_attn.v_proj.register_forward_hook(self._make_hook(i, "V")))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def set_mode(self, mode: str, alpha: float = 1.0, g: float = 1.0) -> None:
        self._runtime = {"mode": mode, "alpha": alpha, "g": g}

    @torch.no_grad()
    def init_from_encoder(self, model, input_features) -> None:
        """E_pretrained init (paper eq.4): seed B_K/B_V with the REAL post-projection
        K/V of the frozen encoder, so scale/distribution match actual cross-attention
        (fixes exp01/02 OOD/scale failure). Mode A only (length must = encoder len).

        Audio is used ONLY to seed the initial value, never as a training signal.
        """
        E = model.model.encoder(input_features).last_hidden_state  # [B, L_enc, d]
        E_bar = E.mean(dim=0)  # [L_enc, d]
        if E_bar.shape[0] != self.L:
            raise ValueError(
                f"init_from_encoder needs L({self.L}) == encoder length({E_bar.shape[0]}); use mode='A'."
            )
        layers = model.model.decoder.layers
        for l, layer in enumerate(layers):
            self.B_K[l].data = layer.encoder_attn.k_proj(E_bar).to(self.B_K[l].dtype)
            self.B_V[l].data = layer.encoder_attn.v_proj(E_bar).to(self.B_V[l].dtype)

    # ---- persistence ----------------------------------------------------
    def save(self, path) -> None:
        state = {
            "mode": self.mode,
            "L": self.L,
            "embed_dim": self.embed_dim,
            "use_gate": self.use_gate,
            "B_K": [p.detach().cpu() for p in self.B_K],
            "B_V": [p.detach().cpu() for p in self.B_V],
        }
        if self.use_gate:
            state["W_g"] = [w.weight.detach().cpu() for w in self.W_g]
        torch.save(state, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location="cpu")
        with torch.no_grad():
            for p, q in zip(self.B_K, state["B_K"]):
                p.copy_(q)
            for p, q in zip(self.B_V, state["B_V"]):
                p.copy_(q)
            if self.use_gate and state.get("W_g") is not None:
                for w, q in zip(self.W_g, state["W_g"]):
                    w.weight.copy_(q)
