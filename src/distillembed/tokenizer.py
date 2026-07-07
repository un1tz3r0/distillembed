"""SentencePiece tokenizer training, constrained for C++ parity.

The C++ engine re-implements unigram Viterbi inference, so every flag that
affects segmentation is pinned here (see DESIGN.md "Tokenizer parity contract").
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


def train_tokenizer(
    corpus: str | Path,
    model_prefix: str | Path,
    vocab_size: int = 8192,
    character_coverage: float = 0.9995,
    max_piece_length: int = 16,
    **kwargs,
) -> Path:
    """Train a unigram tokenizer on `corpus` (one sentence per line).

    Returns the path to the trained .model file.
    """
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        max_sentencepiece_length=max_piece_length,
        # --- parity contract with the C++ engine: do not change ---
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        byte_fallback=True,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        # ----------------------------------------------------------
        **kwargs,
    )
    return Path(f"{model_prefix}.model")


def load_tokenizer(model_path: str | Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(model_path))
