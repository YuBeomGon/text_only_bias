"""Exp03 risk #1: E_pretrained initialization (paper eq.4).

B_K/B_V must be initialized to the REAL post-projection K/V of the frozen
encoder, so the scale matches actual cross-attention K/V (~L2 44), fixing the
exp01/02 OOD/scale failure (random init was ~L2 0.5).
"""
import pytest

from text_only_bias.exp02_kv_bias.kv_bias_model import KVBias


@pytest.fixture
def input_features(whisper):
    torch = whisper["torch"]
    processor = whisper["processor"]
    torch.manual_seed(0)
    wav = torch.randn(16000 * 30).numpy()
    return processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features


def test_init_from_encoder_matches_real_kv_scale(whisper, input_features):
    torch = whisper["torch"]
    m = whisper["model"]
    bias = KVBias(num_layers=m.config.decoder_layers, embed_dim=m.config.d_model, mode="A", length=1500)

    rand_l2 = bias.B_K[0].detach().norm(dim=-1).mean().item()  # random init (~small)
    bias.init_from_encoder(m, input_features)
    init_l2 = bias.B_K[0].detach().norm(dim=-1).mean().item()

    # real post-projection K of layer 0
    with torch.no_grad():
        E = m.model.encoder(input_features).last_hidden_state
        realK = m.model.decoder.layers[0].encoder_attn.k_proj(E[0])
    real_l2 = realK.norm(dim=-1).mean().item()

    assert init_l2 > 10 * rand_l2                      # far larger than random
    assert abs(init_l2 - real_l2) / real_l2 < 0.01     # matches real K scale
    # B_V too, and still trainable
    assert bias.B_K[0].requires_grad and bias.B_V[0].requires_grad
    assert bias.B_V[0].detach().norm(dim=-1).mean().item() > 1.0
