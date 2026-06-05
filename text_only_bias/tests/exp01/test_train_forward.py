"""Risk area #1: text-only training forward.

Verify that with encoder_hidden_states = B_H (no audio, no encoder call):
  - a valid CE loss is produced via decoder + proj_out (transformers 4.57 API)
  - gradient flows to B_H ONLY; the frozen Whisper params stay grad-free.
"""
import pytest

from text_only_bias.exp01_encoder_output.bias_model import TrainableEncoderBias
from text_only_bias.exp01_encoder_output.train import text_only_loss


@pytest.fixture
def batch(whisper):
    torch = whisper["torch"]
    processor = whisper["processor"]
    # two short Korean transcripts -> label ids with special tokens
    texts = ["안녕하세요 고객님", "보험료 안내드리겠습니다"]
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="ko", task="transcribe", predict_timestamps=False)
    labels = tok(text_target=texts, padding=True, return_tensors="pt").input_ids
    # shift: decoder input = labels[:, :-1], target = labels[:, 1:]
    decoder_input_ids = labels[:, :-1]
    target = labels[:, 1:].clone()
    target[target == tok.pad_token_id] = -100
    return {"decoder_input_ids": decoder_input_ids, "target": target}


def test_loss_is_finite_scalar(whisper, batch):
    torch = whisper["torch"]
    model = whisper["model"]
    bias = TrainableEncoderBias(1500, whisper["d_model"], init="normal")
    B = batch["decoder_input_ids"].shape[0]
    enc = bias.forward(B)
    loss, logits = text_only_loss(model, enc, batch["decoder_input_ids"], batch["target"])
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert logits.shape[0] == B


def test_gradient_flows_to_bias_only(whisper, batch):
    torch = whisper["torch"]
    model = whisper["model"]
    bias = TrainableEncoderBias(1500, whisper["d_model"], init="normal")
    B = batch["decoder_input_ids"].shape[0]
    enc = bias.forward(B)
    loss, _ = text_only_loss(model, enc, batch["decoder_input_ids"], batch["target"])
    loss.backward()

    # B_H received gradient
    assert bias.B_H.grad is not None
    assert bias.B_H.grad.abs().sum().item() > 0

    # frozen model params received none
    assert all(p.grad is None for p in model.parameters())
