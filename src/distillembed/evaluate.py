"""Similarity-fidelity evaluation: how well does the student's cosine-similarity
ranking track the teacher's? Reports Pearson/Spearman over sentence pairs drawn
from a corpus. Requires the `distill` extra (for the teacher).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import StaticModel


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        r = np.empty(len(x), dtype=np.float64)
        r[np.argsort(x)] = np.arange(len(x))
        return r

    return _pearson(rank(a), rank(b))


def evaluate(
    model_dir: str | Path,
    corpus_path: str | Path,
    teacher: str | None = None,
    n_pairs: int = 500,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    from sentence_transformers import SentenceTransformer

    student = StaticModel(model_dir)
    teacher = teacher or student.config.get("teacher")
    if not teacher:
        raise ValueError("no teacher in config.json; pass teacher explicitly")

    with open(corpus_path, encoding="utf-8") as f:
        lines = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    if len(lines) < 3:
        raise ValueError("corpus too small for evaluation")

    # Half adjacent pairs (often related text), half random pairs.
    rng = np.random.default_rng(seed)
    pairs = [(i, i + 1) for i in range(min(n_pairs // 2, len(lines) - 1))]
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, len(lines), size=2)
        if i != j:
            pairs.append((int(i), int(j)))

    a_texts = [lines[i] for i, _ in pairs]
    b_texts = [lines[j] for _, j in pairs]

    teacher_model = SentenceTransformer(teacher, device=device)
    ta = teacher_model.encode(a_texts, normalize_embeddings=True, convert_to_numpy=True)
    tb = teacher_model.encode(b_texts, normalize_embeddings=True, convert_to_numpy=True)
    teacher_sims = (ta * tb).sum(axis=1)

    sa = student.encode(a_texts)
    sb = student.encode(b_texts)
    student_sims = (sa * sb).sum(axis=1)

    # Retrieval agreement: of the teacher's top-k neighbors for each query,
    # what fraction does the student also rank in its top-k? This tracks
    # search quality much more directly than pair-similarity correlation.
    n_queries, n_candidates, k = 100, min(2000, len(lines) - 1), 10
    rng2 = np.random.default_rng(seed + 1)
    pool = rng2.permutation(len(lines))
    queries = [lines[i] for i in pool[:n_queries]]
    candidates = [lines[i] for i in pool[n_queries : n_queries + n_candidates]]

    tq = teacher_model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    tc = teacher_model.encode(candidates, normalize_embeddings=True, convert_to_numpy=True)
    sq = student.encode(queries)
    sc = student.encode(candidates)
    teacher_topk = np.argsort(-(tq @ tc.T), axis=1)[:, :k]
    student_topk = np.argsort(-(sq @ sc.T), axis=1)[:, :k]
    overlap = np.mean(
        [len(set(t) & set(s)) / k for t, s in zip(teacher_topk, student_topk)]
    )

    return {
        "teacher": teacher,
        "n_pairs": len(pairs),
        "pearson": _pearson(teacher_sims, student_sims),
        "spearman": _spearman(teacher_sims, student_sims),
        "retrieval_overlap@10": float(overlap),
    }
