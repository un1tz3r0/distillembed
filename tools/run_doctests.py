#!/usr/bin/env python3
"""Extract and run the code blocks embedded in project markdown docs, against
a clean throwaway clone of the repo -- so HOWTO.md / QUICKSTART.md can't drift
from what the CLI actually does.

A fenced code block only participates when the line immediately above the
fence is a directive comment:

    <!-- doctest: run -->
    ```sh
    uv run distillembed tokenizer --corpus examples/device-docs/corpus.txt ...
    ```

Recognized directive tokens (comma/space separated):
  run        execute this block as shell (default action for a tagged block)
  skip       ignore this block entirely (used for sample-output snippets)
  slow       only run with --slow (needs network + the `distill` extra +
             real training time -- distill/refine/eval and anything chained
             after them in the same recipe)
  file=PATH  don't execute; materialize the block's contents verbatim at
             PATH (relative to the clone root) so a later `run` block can
             compile/invoke it (e.g. a .cpp or .py listing)

Untagged code blocks (no directive comment above the fence) are left alone --
they're prose or illustrative sample output, not something to run.

There is no dependency solver: document order *is* the dependency order.
Every participating block, across all given files, is stitched in document
order into one bash script (`set -euo pipefail`) and run in a single
`git clone --local` of the repo, so cwd/venv/generated files persist across
blocks exactly like a reader following the doc top to bottom in one shell.

Usage:
  tools/run_doctests.py HOWTO.md QUICKSTART.md
  tools/run_doctests.py --slow QUICKSTART.md   # full pipeline incl. teacher
  tools/run_doctests.py --keep --slow HOWTO.md # inspect the clone after
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DIRECTIVE_RE = re.compile(r"<!--\s*doctest:\s*(.*?)\s*-->\s*\n")
FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)\n```", re.DOTALL)


def parse_blocks(md_path: Path) -> list[dict]:
    text = md_path.read_text()
    blocks = []
    pos = 0
    while True:
        m = DIRECTIVE_RE.search(text, pos)
        if not m:
            break
        fence = FENCE_RE.match(text, m.end())
        if not fence:
            pos = m.end()
            continue
        lineno = text.count("\n", 0, m.start()) + 1
        tokens = [t for t in re.split(r"[,\s]+", m.group(1)) if t]
        blocks.append(
            {
                "doc": md_path.name,
                "line": lineno,
                "tokens": tokens,
                "lang": fence.group(1),
                "content": fence.group(2),
            }
        )
        pos = fence.end()
    return blocks


def build_script(blocks: list[dict], slow: bool) -> tuple[str, int]:
    lines = ["set -euo pipefail"]
    n_skipped_slow = 0
    for i, b in enumerate(blocks):
        opts, flags = {}, set()
        for tok in b["tokens"]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                opts[k] = v
            else:
                flags.add(tok)

        marker = f"{b['doc']}:{b['line']}"
        if "skip" in flags:
            continue
        if "slow" in flags and not slow:
            n_skipped_slow += 1
            continue

        if "file" in opts:
            path = opts["file"]
            delim = f"DOCTEST_EOF_{i}"
            lines.append(f"echo '### doctest file {marker} -> {path} ###'")
            lines.append(f"mkdir -p \"$(dirname '{path}')\"")
            lines.append(f"cat > '{path}' <<'{delim}'")
            lines.append(b["content"])
            lines.append(delim)
            continue

        lines.append(f"echo '### doctest run {marker} ###'")
        lines.append(b["content"])
    return "\n".join(lines) + "\n", n_skipped_slow


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("docs", nargs="+", type=Path, help="markdown files to extract from")
    ap.add_argument(
        "--slow", action="store_true", help="also run blocks tagged 'slow' (network + torch)"
    )
    ap.add_argument("--keep", action="store_true", help="keep the temporary clone on exit")
    ap.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parent.parent,
        help="repo to clone (default: the repo containing this script)",
    )
    args = ap.parse_args()

    blocks = [b for doc in args.docs for b in parse_blocks(doc)]
    if not blocks:
        print("no doctest blocks found (missing <!-- doctest: ... --> tags?)", file=sys.stderr)
        sys.exit(1)

    script, n_skipped_slow = build_script(blocks, args.slow)

    workdir = Path(tempfile.mkdtemp(prefix="distillembed-doctest-"))
    clone = workdir / "repo"
    print(f"cloning {args.repo} -> {clone}")
    subprocess.run(
        ["git", "clone", "--local", "--quiet", str(args.repo), str(clone)], check=True
    )

    script_path = clone / "_doctest.sh"
    script_path.write_text(script)

    print(
        f"running {len(blocks) - n_skipped_slow} block(s)"
        + (f" ({n_skipped_slow} slow block(s) skipped -- pass --slow to include)"
           if n_skipped_slow else "")
    )
    result = subprocess.run(["bash", "_doctest.sh"], cwd=clone)

    if result.returncode != 0:
        print(f"\nFAILED (exit {result.returncode}). Script preserved at {script_path}",
              file=sys.stderr)
        sys.exit(result.returncode)

    print("\nOK -- all doctest blocks passed")
    if args.keep:
        print(f"clone kept at {clone}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
