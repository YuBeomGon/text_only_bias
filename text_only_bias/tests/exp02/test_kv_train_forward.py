"""Exp02 risk #1: text-only training forward with K/V = B (audio ignored).

Verify CE loss is produced and gradient flows to B_K/B_V ONLY (Whisper frozen).
"""
import pytest

from text_only_bias.exp02_kv_bias.kv_bias_model import KVBias
from text_only_bias.exp02_kv_bias.train import kv_text_only_loss


@pytest.fixture
def batch(whisper):
    processor = whisper["processor"]
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="ko", task="transcribe", predict_timestamps=False)
    labels = tok(text_target=["안녕하세요 고객님", "보험료 안내드리겠습니다"], padding=True, return_tensors="pt").input_ids
    decoder_input_ids = labels[:, :-1]
    target = labels[:, 1:].clone()
    target[target == tok.pad_token_id] = -100
    return {"decoder_input_ids": decoder_input_ids, "target": target}


@pytest.mark.parametrize("mode,kw", [("A", {"length": 1500}), ("B", {"M": 32})])
def test_loss_finite_and_grad_to_bias_only(whisper, batch, mode, kw):
    torch = whisper["torch"]
    model = whisper["model"]
    bias = KVBias(num_layers=model.config.decoder_layers, embed_dim=model.config.d_model, mode=mode, **kw)
    bias.attach(model)
    try:
        loss, logits = kv_text_only_loss(model, bias, batch["decoder_input_ids"], batch["target"])
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        # bias got gradient
        assert bias.B_K[0].grad is not None and bias.B_K[0].grad.abs().sum() > 0
        assert bias.B_V[0].grad is not None and bias.B_V[0].grad.abs().sum() > 0
        # frozen model params got none
        assert all(p.grad is None for p in model.parameters())
    finally:
        bias.detach()
