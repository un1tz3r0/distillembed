"""Distill a sentence-transformer teacher into a static token-embedding table.

Model2Vec-style: embed every vocabulary piece individually with the teacher,
PCA down to the target dimension, then fold each piece's SIF weight
(w = a / (a + p), p from the unigram score) into its row. Requires the
`distill` extra (sentence-transformers + torch).
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import sentencepiece as spm

from .export import TYPE_BYTE, TYPE_NORMAL, piece_type


def _parse_byte_piece(piece: str) -> int | None:
    """'<0xAB>' -> 0xAB, else None."""
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        try:
            return int(piece[3:5], 16)
        except ValueError:
            return None
    return None


def _piece_texts(sp) -> tuple[list[str], np.ndarray]:
    """Surface text to feed the teacher for each piece, plus a zero-row mask."""
    vocab = sp.vocab_size()
    texts: list[str] = []
    zero_mask = np.zeros(vocab, dtype=bool)
    for i in range(vocab):
        ptype = piece_type(sp, i)
        piece = sp.id_to_piece(i)
        text = ""
        if ptype == TYPE_NORMAL:
            text = piece.replace("▁", " ").strip()
        elif ptype == TYPE_BYTE:
            b = _parse_byte_piece(piece)
            if b is not None and 33 <= b <= 126:  # printable ASCII, sans space
                text = chr(b)
        if not text:  # unk/control/whitespace-only/non-printable-byte
            zero_mask[i] = True
            text = "."  # placeholder; row is zeroed after encoding
        texts.append(text)
    return texts, zero_mask


def distill(
    spm_model: str | Path,
    teacher: str,
    dim: int,
    out_dir: str | Path,
    sif_a: float = 1e-3,
    batch_size: int = 256,
    device: str | None = None,
) -> Path:
    from sentence_transformers import SentenceTransformer

    sp = spm.SentencePieceProcessor(model_file=str(spm_model))
    vocab = sp.vocab_size()
    texts, zero_mask = _piece_texts(sp)

    model = SentenceTransformer(teacher, device=device)
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    # PCA to the target dimension (mean/components saved for refine/eval,
    # which must project teacher sentence embeddings into student space).
    mean = emb.mean(axis=0)
    if dim < emb.shape[1]:
        centered = emb - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:dim]
        emb = centered @ components.T
    else:
        dim = emb.shape[1]
        components = np.eye(dim, dtype=np.float32)

    # SIF weighting from unigram log-probs; folding into rows is exact because
    # the final embedding is L2-normalized (see DESIGN.md).
    scores = np.array([sp.get_score(i) for i in range(vocab)], dtype=np.float64)
    probs = np.exp(scores)
    weights = sif_a / (sif_a + probs)
    weights[zero_mask] = 0.0

    table = (emb * weights[:, None]).astype(np.float32)
    table[zero_mask] = 0.0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", table)
    np.save(out / "pca_mean.npy", mean.astype(np.float32))
    np.save(out / "pca_components.npy", components.astype(np.float32))
    spm_src = Path(spm_model)
    if spm_src.resolve() != (out / "spm.model").resolve():
        shutil.copy(spm_src, out / "spm.model")
    (out / "config.json").write_text(
        json.dumps(
            {
                "teacher": teacher,
                "dim": int(dim),
                "vocab_size": int(vocab),
                "sif_a": sif_a,
                "refined": False,
                "created": date.today().isoformat(),
            },
            indent=2,
        )
    )
    return out
