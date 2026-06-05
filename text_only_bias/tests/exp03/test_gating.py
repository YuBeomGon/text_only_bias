"""Exp03 Phase 4: tanh gating (paper eq.3).

G = tanh(W_g · B), B_gated = G ⊙ B, per-layer learnable W_g [d, d].
The gate is applied just before injection (both train and inference modes).

Design choices verified here:
  - use_gate=False (default) keeps exp02/Phase2-3 behavior byte-identical.
  - use_gate=True injects B_gated = tanh(W_g(B)) * B (element-wise).
  - W_g initialized to zeros => gate starts CLOSED (B_gated = 0) so training
    starts from baseline behavior and learns to open (stable; preserves the
    scale-matched init underneath as the gate opens).
  - W_g receives gradient during training.
See docs/experiments/exp03_paper_faithful/design.md §4b.
"""
import pytest

from text_only_bias.exp02_kv_bias.kv_bias_model import KVBias
from text_only_bias.exp02_kv_bias.train import kv_text_only_loss


def test_gate_disabled_by_default_no_params():
    bias = KVBias(num_layers=2, embed_dim=8, mode="A", length=4)
    assert bias.use_gate is False
    assert not hasattr(bias, "W_g") or len(list(bias.W_g)) == 0


def test_gate_zero_init_closes_gate():
    """With use_gate=True and zero-init W_g, tanh(0)=0 => B_gated == 0."""
    import torch

    bias = KVBias(num_layers=2, embed_dim=8, mode="A", length=4, use_gate=True)
    g = bias.gated(bias.B_K[0])
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-7)


def test_gate_applies_tanh_elementwise():
    """B_gated == tanh(W_g(B)) * B for arbitrary (nonzero) W_g."""
    import torch

    bias = KVBias(num_layers=1, embed_dim=8, mode="A", length=4, use_gate=True)
    with torch.no_grad():
        bias.W_g[0].weight.copy_(torch.randn(8, 8) * 0.5)
    B = bias.B_K[0]
    expected = torch.tanh(bias.W_g[0](B)) * B
    got = bias.gated(bias.B_K[0], layer_idx=0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_gate_disabled_returns_input_unchanged():
    import torch

    bias = KVBias(num_layers=1, embed_dim=8, mode="A", length=4, use_gate=False)
    B = bias.B_K[0]
    assert torch.allclose(bias.gated(B, layer_idx=0), B)


@pytest.fixture
def batch(whisper):
    processor = whisper["processor"]
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="ko", task="transcribe", predict_timestamps=False)
    labels = tok(text_target=["안녕하세요 고객님", "보험료 안내드리겠습니다"],
                 padding=True, return_tensors="pt").input_ids
    decoder_input_ids = labels[:, :-1]
    target = labels[:, 1:].clone()
    target[target == tok.pad_token_id] = -100
    return {"decoder_input_ids": decoder_input_ids, "target": target}


def test_gate_receives_gradient_in_training(whisper, batch):
    """W_g must get gradient so the gate can learn to open."""
    torch = whisper["torch"]
    model = whisper["model"]
    bias = KVBias(num_layers=model.config.decoder_layers, embed_dim=model.config.d_model,
                  mode="A", length=1500, use_gate=True)
    # open the gate a little so gradient is nonzero (zero-init gives zero grad start)
    with torch.no_grad():
        for w in bias.W_g:
            w.weight.normal_(0.0, 0.02)
    bias.attach(model)
    try:
        loss, _ = kv_text_only_loss(model, bias, batch["decoder_input_ids"], batch["target"])
        loss.backward()
        assert bias.W_g[0].weight.grad is not None
        assert bias.W_g[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.parameters())
    finally:
        bias.detach()
