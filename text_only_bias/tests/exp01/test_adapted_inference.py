"""Risk area #2: adapted inference via encoder_outputs injection.

Verify that mixing B_H into the encoder hidden states and feeding it to
`generate` works in transformers 4.57, and that alpha=1.0 reproduces the
baseline token-for-token (the critical sanity check from guide §21.5).
"""
import pytest

from text_only_bias.exp01_encoder_output.bias_model import TrainableEncoderBias
from text_only_bias.exp01_encoder_output.eval_adapted import encode_audio, adapted_generate


@pytest.fixture
def input_features(whisper):
    torch = whisper["torch"]
    processor = whisper["processor"]
    # 30s of (deterministic) pseudo-audio at 16k -> log-mel features
    torch.manual_seed(0)
    wav = torch.randn(16000 * 30).numpy()
    feats = processor.feature_extractor(
        wav, sampling_rate=16000, return_tensors="pt"
    ).input_features
    return feats


@pytest.fixture
def gen_kwargs(whisper):
    return dict(language="ko", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=16)


def test_encode_audio_shape(whisper, input_features):
    model = whisper["model"]
    H = encode_audio(model, input_features)
    assert H.shape == (1, 1500, whisper["d_model"])


def test_alpha_one_matches_baseline(whisper, input_features, gen_kwargs):
    torch = whisper["torch"]
    model = whisper["model"]
    bias = TrainableEncoderBias(1500, whisper["d_model"], init="normal")

    baseline_ids = model.generate(input_features, **gen_kwargs)
    adapted_ids = adapted_generate(model, bias, input_features, alpha=1.0, **gen_kwargs)

    assert torch.equal(baseline_ids, adapted_ids)
