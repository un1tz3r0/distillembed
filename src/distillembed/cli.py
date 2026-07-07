"""Command-line interface. Heavy deps (torch, sentence-transformers) are
imported lazily inside the commands that need them, so tokenizer training,
export, and the reference encoder work with the core install alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="distillembed",
        description="Distill sentence embedding models into tiny static lookup tables.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tokenizer", help="train a SentencePiece unigram tokenizer")
    s.add_argument("--corpus", required=True, help="text file, one sentence per line")
    s.add_argument("--out", required=True, help="model prefix, e.g. models/base/spm")
    s.add_argument("--vocab-size", type=int, default=8192)
    s.add_argument("--character-coverage", type=float, default=0.9995)

    s = sub.add_parser("distill", help="distill a teacher into a static table")
    s.add_argument("--spm", required=True, help="path to trained spm.model")
    s.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    s.add_argument("--dim", type=int, default=128)
    s.add_argument("--out", required=True, help="output model directory")
    s.add_argument("--sif-a", type=float, default=1e-3)
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--device", default=None)

    s = sub.add_parser("refine", help="fine-tune the table vs teacher sentence embeddings")
    s.add_argument("--model", required=True, help="model directory")
    s.add_argument("--corpus", required=True)
    s.add_argument("--teacher", default=None, help="defaults to the teacher in config.json")
    s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--max-lines", type=int, default=50_000)
    s.add_argument("--device", default=None)

    s = sub.add_parser("export", help="export a model directory to a .dem binary")
    s.add_argument("--model", required=True)
    s.add_argument("--out", default=None, help="defaults to <model>/model.dem")
    s.add_argument("--quant", choices=["f32", "int8", "int4"], default="int8")

    s = sub.add_parser("corpus", help="embed a docs file (one chunk per line) into a .demc")
    s.add_argument("--model", required=True)
    s.add_argument("--docs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--quant", choices=["f32", "int8", "binary", "rescore"], default="int8")

    s = sub.add_parser("carray", help="render a .dem/.demc blob as a C++ header for flash linking")
    s.add_argument("--in", dest="src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--symbol", default="g_distillembed_blob")

    s = sub.add_parser("embed", help="embed text with the numpy reference encoder")
    s.add_argument("--model", required=True)
    s.add_argument("text")
    s.add_argument("--full", action="store_true", help="print all dimensions")

    s = sub.add_parser("eval", help="similarity-fidelity eval: student vs teacher")
    s.add_argument("--model", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--teacher", default=None)
    s.add_argument("--pairs", type=int, default=500)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "tokenizer":
        from .tokenizer import train_tokenizer

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        path = train_tokenizer(args.corpus, args.out, args.vocab_size, args.character_coverage)
        print(f"wrote {path}")

    elif args.cmd == "distill":
        from .distill import distill

        out = distill(
            args.spm, args.teacher, args.dim, args.out,
            sif_a=args.sif_a, batch_size=args.batch_size, device=args.device,
        )
        print(f"wrote model dir {out}")

    elif args.cmd == "refine":
        from .refine import refine

        refine(
            args.model, args.corpus, teacher=args.teacher, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, max_lines=args.max_lines,
            device=args.device,
        )
        print(f"refined table in {args.model}")

    elif args.cmd == "export":
        import numpy as np
        import sentencepiece as spm

        from .export import write_dem

        model_dir = Path(args.model)
        out = Path(args.out) if args.out else model_dir / "model.dem"
        sp = spm.SentencePieceProcessor(model_file=str(model_dir / "spm.model"))
        table = np.load(model_dir / "embeddings.npy")
        write_dem(out, sp, table, args.quant)
        print(f"wrote {out} ({out.stat().st_size:,} bytes, {args.quant})")

    elif args.cmd == "corpus":
        from .export import write_corpus
        from .model import StaticModel

        student = StaticModel(args.model)
        with open(args.docs, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        vectors = student.encode(texts)
        out = write_corpus(args.out, vectors, texts, args.quant)
        print(f"wrote {out} ({len(texts)} chunks, dim {student.dim}, {args.quant})")

    elif args.cmd == "carray":
        from .export import write_c_array

        out = write_c_array(args.src, args.out, args.symbol)
        print(f"wrote {out} (symbol {args.symbol})")

    elif args.cmd == "embed":
        from .model import StaticModel

        student = StaticModel(args.model)
        ids = student.tokenize(args.text)
        vec = student.encode(args.text)
        # Same output shape as cpp/src/demo_embed.cpp, for eyeball parity checks.
        print(f"tokens: {len(ids)}")
        print("ids:", *ids)
        print(f"dim: {student.dim}")
        shown = vec if args.full else vec[:8]
        print("vec:", " ".join(f"{v:.6f}" for v in shown))

    elif args.cmd == "eval":
        from .evaluate import evaluate

        result = evaluate(args.model, args.corpus, teacher=args.teacher, n_pairs=args.pairs)
        print(json.dumps(result, indent=2))
