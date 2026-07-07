import json
import random

import numpy as np
import pytest
import sentencepiece as spm

from distillembed.tokenizer import train_tokenizer

WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "sensor",
    "reading", "temperature", "humidity", "device", "status", "error",
    "battery", "level", "signal", "network", "connect", "update", "firmware",
    "restart", "module", "value", "threshold", "alarm", "system", "power",
    "mode", "sleep", "wake", "data", "packet", "queue", "memory", "flash",
    "clock", "timer", "input",
]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A complete model directory with a real (tiny) tokenizer and a random table."""
    root = tmp_path_factory.mktemp("tinymodel")
    rng = random.Random(0)
    lines = [" ".join(rng.choices(WORDS, k=rng.randint(3, 10))) for _ in range(400)]
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(lines) + "\n")
    train_tokenizer(corpus, root / "spm", vocab_size=320, character_coverage=1.0)

    sp = spm.SentencePieceProcessor(model_file=str(root / "spm.model"))
    table = np.random.default_rng(0).standard_normal((sp.vocab_size(), 32)).astype(np.float32)
    np.save(root / "embeddings.npy", table)
    (root / "config.json").write_text(json.dumps({"teacher": "random", "dim": 32}))
    return root


@pytest.fixture(scope="session")
def tiny_sp(tiny_model_dir):
    return spm.SentencePieceProcessor(model_file=str(tiny_model_dir / "spm.model"))


@pytest.fixture(scope="session")
def tiny_table(tiny_model_dir):
    return np.load(tiny_model_dir / "embeddings.npy")
