import numpy as np
import torch

from text_only_bias.common.data import (
    audio_to_array,
    TextOnlyCollator,
    load_splits,
    row_to_input_features,
)

DATASET_PATH = "/data/MyProject/stt/data-gen/aig-audio-3/data/processed/hf_dataset"


def test_audio_to_array_is_float32():
    row = {"audio": [0.0, 0.5, -0.5, 1.0], "sampling_rate": 16000}
    arr = audio_to_array(row)
    assert arr.dtype == np.float32
    assert arr.shape == (4,)


def test_collator_shift_and_mask(whisper):
    processor = whisper["processor"]
    coll = TextOnlyCollator(processor, language="ko", task="transcribe")
    batch = coll(["안녕하세요", "보험 상담 도와드릴게요 고객님"])
    din = batch["decoder_input_ids"]
    lab = batch["labels"]
    # decoder input and target are the same length (labels[:, :-1] vs labels[:, 1:])
    assert din.shape == lab.shape
    assert din.shape[0] == 2
    # padded target positions are masked with -100
    assert (lab == -100).any()
    # decoder input never contains -100 (it is real token ids)
    assert (din != -100).all()


def test_collator_includes_special_prefix(whisper):
    processor = whisper["processor"]
    tok = processor.tokenizer
    coll = TextOnlyCollator(processor, language="ko", task="transcribe")
    batch = coll(["테스트"])
    # first decoder input token is the start-of-transcript token
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    assert batch["decoder_input_ids"][0, 0].item() == sot


def test_load_splits_train_test(tmp_path):
    train, test = load_splits(DATASET_PATH, train_split="train", test_split="validation")
    assert train.num_rows == 2556
    assert test.num_rows == 424
    assert "text" in train.column_names


def test_row_to_input_features_shape(whisper):
    processor = whisper["processor"]
    row = {"audio": list(np.zeros(16000 * 5, dtype=np.float32)), "sampling_rate": 16000}
    feats = row_to_input_features(row, processor)
    # whisper pads/truncates to 30s -> 3000 mel frames, 80 mel bins
    assert feats.shape[-1] == 3000
