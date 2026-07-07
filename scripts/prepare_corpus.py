"""Build training/eval text from a tree of scraped-markdown documentation.

Produces two files:
  - corpus.txt: one cleaned prose sentence/line per row (tokenizer + distill/refine)
  - docs.txt:   one paragraph-level chunk per row (search-demo corpus)

The source markdown is web-scraped and full of nav junk (link lists, banners,
tables of contents), so lines are aggressively filtered to keep actual prose.

Usage:
  uv run python scripts/prepare_corpus.py <collections_dir> data/ \
      --max-lines 300000 --max-files-per-collection 400
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

RE_CODE_FENCE = re.compile(r"^(```|~~~)")
RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
RE_HTML = re.compile(r"<[^>]+>")
RE_INLINE_CODE = re.compile(r"`([^`]*)`")
RE_HEADING = re.compile(r"^#{1,6}\s+")
RE_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
RE_WS = re.compile(r"\s+")
RE_WORD = re.compile(r"[A-Za-z]{2,}")


def clean_line(line: str) -> str:
    line = RE_HEADING.sub("", line)
    line = RE_BULLET.sub("", line)
    line = RE_IMAGE.sub(" ", line)
    line = RE_LINK.sub(r"\1", line)
    line = RE_INLINE_CODE.sub(r"\1", line)
    line = RE_HTML.sub(" ", line)
    return RE_WS.sub(" ", line).strip()


def is_prose(line: str, min_len: int = 40, max_len: int = 400) -> bool:
    if not (min_len <= len(line) <= max_len):
        return False
    words = RE_WORD.findall(line)
    if len(words) < 5:
        return False
    alpha = sum(c.isalpha() or c in " ,.;:()'-" for c in line)
    if alpha / len(line) < 0.7:
        return False
    # Nav/TOC leftovers: Title Case runs with no sentence structure.
    if len(words) >= 5 and sum(w[0].isupper() for w in words) / len(words) > 0.7:
        return False
    return True


def iter_paragraphs(path: Path):
    """Yield cleaned paragraphs, skipping fenced code blocks."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    in_code = False
    para: list[str] = []
    for raw in text.splitlines():
        if RE_CODE_FENCE.match(raw.strip()):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = clean_line(raw)
        if line:
            para.append(line)
        elif para:
            yield " ".join(para)
            para = []
    if para:
        yield " ".join(para)


RE_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def iter_chunks(paragraph: str, target: int = 350, max_sentence: int = 500):
    """Pack sentences into ~target-char chunks with one-sentence overlap."""
    sentences = [s.strip() for s in RE_SENTENCE.split(paragraph) if s.strip()]
    current: list[str] = []
    length = 0
    yielded = False
    for sentence in sentences:
        if len(sentence) > max_sentence:
            continue
        current.append(sentence)
        length += len(sentence) + 1
        if length >= target:
            yield " ".join(current)
            yielded = True
            current = [current[-1]]  # overlap so boundary context isn't lost
            length = len(current[0]) + 1
    # Tail: new material beyond the overlap sentence, or an unyielded short paragraph.
    if (len(current) > 1 or not yielded) and current:
        chunk = " ".join(current)
        if len(chunk) >= 80:
            yield chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("collections_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--max-lines", type=int, default=300_000)
    ap.add_argument("--max-chunks", type=int, default=20_000)
    ap.add_argument("--max-files-per-collection", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    collections = sorted(d for d in args.collections_dir.iterdir() if d.is_dir())
    print(f"{len(collections)} collections")

    sentences: set[str] = set()
    chunks: set[str] = set()
    for coll in collections:
        files = sorted(coll.rglob("*.md"))
        rng.shuffle(files)
        n_before = len(sentences)
        for path in files[: args.max_files_per_collection]:
            for para in iter_paragraphs(path):
                if len(chunks) < args.max_chunks * 5:
                    for chunk in iter_chunks(para):
                        if is_prose(chunk, min_len=100, max_len=700):
                            chunks.add(chunk)
                for sent in RE_SENTENCE.split(para):
                    sent = sent.strip()
                    if is_prose(sent):
                        sentences.add(sent)
        print(f"  {coll.name}: +{len(sentences) - n_before} sentences")

    sentence_list = sorted(sentences)
    rng.shuffle(sentence_list)
    sentence_list = sentence_list[: args.max_lines]
    chunk_list = sorted(chunks)
    rng.shuffle(chunk_list)
    chunk_list = chunk_list[: args.max_chunks]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "corpus.txt").write_text("\n".join(sentence_list) + "\n")
    (args.out_dir / "docs.txt").write_text("\n".join(chunk_list) + "\n")
    print(f"wrote {len(sentence_list)} sentences -> {args.out_dir / 'corpus.txt'}")
    print(f"wrote {len(chunk_list)} chunks -> {args.out_dir / 'docs.txt'}")


if __name__ == "__main__":
    main()
