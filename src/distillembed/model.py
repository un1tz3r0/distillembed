"""Numpy reference encoder for a distilled model directory.

This is the ground truth the C++ engine is tested against. A model directory
contains: spm.model, embeddings.npy (weight-folded f32 table), config.json,
and optionally pca_mean.npy / pca_components.npy (used by refine/eval).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


class StaticModel:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.dir = d
        self.sp = spm.SentencePieceProcessor(model_file=str(d / "spm.model"))
        self.table = np.load(d / "embeddings.npy")
        config_path = d / "config.json"
        self.config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.dim = int(self.table.shape[1])

    def tokenize(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def encode(self, texts: str | list[str], normalize: bool = True) -> np.ndarray:
        """Embed text(s): sum of (weight-folded) token rows, L2-normalized."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for k, text in enumerate(texts):
            ids = self.sp.encode(text)
            if ids:
                out[k] = self.table[ids].sum(axis=0)
        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            np.divide(out, norms, out=out, where=norms > 0)
        return out[0] if single else out
