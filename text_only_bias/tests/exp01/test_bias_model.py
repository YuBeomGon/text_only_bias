import torch

from text_only_bias.exp01_encoder_output.bias_model import TrainableEncoderBias


def test_bias_param_shape_and_trainable():
    bias = TrainableEncoderBias(encoder_seq_len=1500, d_model=768)
    assert bias.B_H.shape == (1500, 768)
    assert bias.B_H.requires_grad is True


def test_zero_init_is_all_zeros():
    bias = TrainableEncoderBias(encoder_seq_len=4, d_model=3, init="zero")
    assert torch.equal(bias.B_H.detach(), torch.zeros(4, 3))


def test_normal_init_is_not_all_zeros():
    torch.manual_seed(0)
    bias = TrainableEncoderBias(encoder_seq_len=8, d_model=8, init="normal", init_std=0.02)
    assert not torch.allclose(bias.B_H.detach(), torch.zeros(8, 8))
    # std roughly in the right ballpark
    assert bias.B_H.detach().std().item() < 0.1


def test_forward_expands_to_batch():
    bias = TrainableEncoderBias(encoder_seq_len=5, d_model=4, init="normal")
    out = bias.forward(batch_size=3)
    assert out.shape == (3, 5, 4)
    # every batch row is the same B_H
    assert torch.equal(out[0], out[1])
    assert torch.equal(out[1], out[2])


def test_mix_alpha_one_returns_audio_unchanged():
    # critical sanity: alpha=1.0 must reproduce the audio path exactly
    bias = TrainableEncoderBias(encoder_seq_len=5, d_model=4, init="normal")
    H = torch.randn(2, 5, 4)
    mixed = bias.mix(H, alpha=1.0)
    assert torch.allclose(mixed, H)


def test_mix_alpha_zero_returns_bias_only():
    bias = TrainableEncoderBias(encoder_seq_len=5, d_model=4, init="normal")
    H = torch.randn(2, 5, 4)
    mixed = bias.mix(H, alpha=0.0)
    expected = bias.B_H.unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(mixed, expected)


def test_mix_math():
    bias = TrainableEncoderBias(encoder_seq_len=5, d_model=4, init="normal")
    H = torch.randn(2, 5, 4)
    alpha = 0.7
    mixed = bias.mix(H, alpha=alpha)
    expected = alpha * H + (1 - alpha) * bias.B_H.unsqueeze(0)
    assert torch.allclose(mixed, expected)


def test_save_and_load_roundtrip(tmp_path):
    bias = TrainableEncoderBias(encoder_seq_len=6, d_model=4, init="normal")
    p = tmp_path / "bias.pt"
    bias.save(p)

    loaded = TrainableEncoderBias(encoder_seq_len=6, d_model=4, init="zero")
    loaded.load(p)
    assert torch.equal(loaded.B_H.detach(), bias.B_H.detach())
