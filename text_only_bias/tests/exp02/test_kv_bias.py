"""Exp02 K/V bias — injection mechanism tests.

Risk areas:
 [위험2] baseline sanity: A alpha=1.0 / B g=0 must reproduce plain Whisper token-for-token.
 [위험3] concat (Option B) must run without shape errors at g>0.
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


@pytest.fixture
def gen_kwargs():
    return dict(language="ko", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=16)


def test_param_shapes_mode_A(whisper):
    m = whisper["model"]
    bias = KVBias(num_layers=m.config.decoder_layers, embed_dim=m.config.d_model, mode="A", length=1500)
    assert len(bias.B_K) == m.config.decoder_layers
    assert bias.B_K[0].shape == (1500, m.config.d_model)
    assert bias.B_V[0].shape == (1500, m.config.d_model)
    assert bias.B_K[0].requires_grad is True


def test_param_shapes_mode_B(whisper):
    m = whisper["model"]
    bias = KVBias(num_layers=m.config.decoder_layers, embed_dim=m.config.d_model, mode="B", M=32)
    assert bias.B_K[0].shape == (32, m.config.d_model)


def test_attach_detach_registers_hooks(whisper):
    m = whisper["model"]
    bias = KVBias(num_layers=m.config.decoder_layers, embed_dim=m.config.d_model, mode="A")
    bias.attach(m)
    assert len(bias._handles) == 2 * m.config.decoder_layers  # k_proj + v_proj per layer
    bias.detach()
    assert len(bias._handles) == 0


def test_baseline_sanity_mode_A_alpha_one(whisper, input_features, gen_kwargs):
    torch = whisper["torch"]
    m = whisper["model"]
    plain = m.generate(input_features, **gen_kwargs)

    bias = KVBias(num_layers=m.config.decoder_layers, embed_dim=m.config.d_model, mode="A", length=1500)
    bias.attach(m)
    try:
        bias.set_mode("A", alpha=1.0)
        adapted = m.generate(input_features, **gen_kwargs)
    finally:
        bias.detach()
    assert torch.equal(plain, adapted)


def test_baseline_sanity_mode_B_g_zero(whisper, input_features, gen_kwargs):
    torch = whisper["torch"]
    m = whisper["model"]
    plain = m.generate(input_features, **gen_kwargs)

    bias = KVBias(num_layers=m.config.decoder_layers, embed_dim=m.config.d_model, mode="B", M=32)
    bias.attach(m)
    try:
        bias.set_mode("B", g=0.0)
        adapted = m.generate(input_features, **gen_kwargs)
    finally:
        bias.detach()
    assert torch.equal(plain, adapted)


def test_concat_mode_B_runs(whisper, input_features, gen_kwargs):
    torch = whisper["torch"]
    m = whisper["model"]
    bias = KVBias(num_layers=m.config.decoder_layers, embed_dim=m.config.d_model, mode="B", M=32, init_std=0.5)
    bias.attach(m)
    try:
        bias.set_mode("B", g=1.0)
        out = m.generate(input_features, **gen_kwargs)
    finally:
        bias.detach()
    assert out.shape[0] == 1
    assert out.shape[1] >= 1
