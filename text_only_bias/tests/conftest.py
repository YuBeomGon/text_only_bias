import pytest


@pytest.fixture(scope="session")
def whisper():
    """Load whisper-small once on CPU for the heavy integration tests."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {
        "processor": processor,
        "model": model,
        "d_model": model.config.d_model,
        "torch": torch,
    }
