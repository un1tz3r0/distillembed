"""Tokenlearn-style refinement: train the static table so that pooled student
embeddings match (PCA-projected) teacher sentence embeddings on an unlabeled
corpus. Typically recovers several points of retrieval quality over plain
per-piece distillation. Requires the `distill` extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def refine(
    model_dir: str | Path,
    corpus_path: str | Path,
    teacher: str | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 2e-3,
    max_lines: int = 50_000,
    device: str | None = None,
) -> Path:
    import torch
    from sentence_transformers import SentenceTransformer

    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text())
    teacher = teacher or config["teacher"]
    sp = spm.SentencePieceProcessor(model_file=str(d / "spm.model"))
    table = np.load(d / "embeddings.npy")

    with open(corpus_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()][:max_lines]
    if not lines:
        raise ValueError(f"no usable lines in {corpus_path}")

    teacher_model = SentenceTransformer(teacher, device=device)
    t_emb = teacher_model.encode(
        lines, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    # Project teacher sentence embeddings into student PCA space, then normalize.
    mean = np.load(d / "pca_mean.npy")
    components = np.load(d / "pca_components.npy")
    targets = (t_emb - mean) @ components.T
    norms = np.linalg.norm(targets, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    targets /= norms

    ids_per_line = sp.encode(lines)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.nn.Parameter(torch.tensor(table, dtype=torch.float32, device=dev))
    opt = torch.optim.Adam([weight], lr=lr)
    targets_t = torch.tensor(targets, dtype=torch.float32, device=dev)

    order = np.arange(len(lines))
    for epoch in range(epochs):
        np.random.default_rng(epoch).shuffle(order)
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            flat, offsets = [], []
            for j in batch:
                offsets.append(len(flat))
                flat.extend(ids_per_line[j] or [0])  # unk row is zero
            ids = torch.tensor(flat, dtype=torch.long, device=dev)
            offs = torch.tensor(offsets, dtype=torch.long, device=dev)
            pooled = torch.nn.functional.embedding_bag(ids, weight, offs, mode="sum")
            pooled = torch.nn.functional.normalize(pooled, dim=-1, eps=1e-9)
            loss = (1.0 - (pooled * targets_t[batch]).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        print(f"epoch {epoch + 1}/{epochs}: cosine loss {total / max(batches, 1):.4f}")

    np.save(d / "embeddings.npy", weight.detach().cpu().numpy())
    config["refined"] = True
    (d / "config.json").write_text(json.dumps(config, indent=2))
    return d
