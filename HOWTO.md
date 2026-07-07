# HOWTO

Task-oriented recipes for distillembed. For a single guided walkthrough from
zero to a running on-device search program, see [QUICKSTART.md](QUICKSTART.md)
instead — this file assumes you've read that once and want a specific thing.

Every runnable snippet below is tested for real (see
[Testing these docs](#testing-these-docs)). Recipes that need a distilled
model share one setup step:

<!-- doctest: slow -->
```sh
uv sync --extra distill
uv run distillembed tokenizer --corpus examples/device-docs/corpus.txt \
    --out models/howto/spm --vocab-size 512
uv run distillembed distill --spm models/howto/spm.model \
    --teacher sentence-transformers/all-MiniLM-L6-v2 --dim 64 --out models/howto
uv run distillembed refine --model models/howto \
    --corpus examples/device-docs/corpus.txt --epochs 3
```

## Build a corpus from a scraped-markdown documentation tree

<!-- doctest: skip -->
```sh
uv run python scripts/prepare_corpus.py path/to/collections_dir data/ \
    --max-lines 300000 --max-chunks 20000 --max-files-per-collection 400
```

`collections_dir` should contain one subdirectory per documentation
collection, each a tree of `.md` files (this is the shape web-scraped doc
sites usually end up in). The script:

- strips headings/bullets/links/inline-code/HTML, skips fenced code blocks
- keeps only lines that look like prose (word count, alpha ratio, rejects
  Title-Case nav/TOC runs)
- writes `data/corpus.txt` — deduplicated sentences, for tokenizer + refine
- writes `data/docs.txt` — ~350-char sentence-packed chunks with one-sentence
  overlap, for the search-demo corpus

Tune `--max-lines`/`--max-chunks` to your corpus size and `is_prose`'s
`min_len`/`max_len` bounds in the script itself if your source docs run
unusually short or long.

## Pick a tokenizer vocab size for a small corpus

SentencePiece cannot train a vocabulary bigger than the number of distinct
substrings your corpus supports, and it tells you the ceiling rather than
silently truncating:

<!-- doctest: run -->
```sh
uv run distillembed tokenizer --corpus examples/device-docs/corpus.txt \
    --out /tmp/vocab_probe/spm --vocab-size 4096 2>&1 | tail -1 || true
```

```text
RuntimeError: ... Vocabulary size too high (4096). Please set it to a value <= 533.
```

Rule of thumb: start at the README default (8192) for a real corpus of tens
of thousands of lines or more; for a small/synthetic corpus, drop to a few
hundred and let the error message above tell you the exact ceiling rather
than guessing. `--character-coverage` (default `0.9995`) trades rare-character
coverage for vocabulary efficiency — lower it (e.g. `0.98`) for corpora that
are overwhelmingly ASCII prose.

## Distill with a different teacher or output dimension

Any `sentence-transformers` model works as the teacher; `--dim` controls the
PCA target (must be ≤ the teacher's native dimension):

<!-- doctest: slow -->
```sh
uv run distillembed distill --spm models/howto/spm.model \
    --teacher sentence-transformers/all-MiniLM-L6-v2 --dim 32 \
    --out models/howto-dim32
```

Smaller `--dim` shrinks every downstream artifact (table, `.dem`, `.demc`)
roughly linearly, at a fidelity cost — always re-run `eval` (below) after
changing it before committing to a dimension.

## Continue refining an already-refined model

`refine` re-reads whatever table is currently in `--model` and overwrites it
in place — so pointing it at more epochs, a bigger corpus, or newly collected
domain text just keeps improving the same model directory. `config.json`'s
`"refined"` flag flips to `true` after the first call and stays there:

<!-- doctest: slow -->
```sh
uv run distillembed refine --model models/howto \
    --corpus examples/device-docs/corpus.txt --epochs 2 --lr 1e-3
```

## Check fidelity against the teacher

`eval` reports Pearson/Spearman correlation of cosine similarities on
sampled pairs, plus retrieval overlap@10 (of the teacher's top-10 neighbors
for a query, how many the student also ranks top-10) — the metric that
tracks real search quality most directly:

<!-- doctest: slow -->
```sh
uv run distillembed eval --model models/howto \
    --corpus examples/device-docs/corpus.txt --pairs 200
```

If `retrieval_overlap@10` is low, first check whether you've refined at all
(`plain distill` is much weaker than `+ refine` — see the README table),
then try more epochs or a larger/more representative refine corpus before
reaching for a bigger `--dim`.

## Choose a model quantization

`export --quant` trades size for precision; `int4` roughly halves `int8`:

<!-- doctest: slow -->
```sh
uv run distillembed export --model models/howto --quant int8 --out models/howto/model_int8.dem
uv run distillembed export --model models/howto --quant int4 --out models/howto/model_int4.dem
ls -la models/howto/model_int8.dem models/howto/model_int4.dem
```

```text
-rw-r--r-- 1 user user 43500 ... models/howto/model_int8.dem
-rw-r--r-- 1 user user 27116 ... models/howto/model_int4.dem
```

Both quantizations are exercised by `demo_embed`/`demo_search` identically —
nothing else in the pipeline needs to know which one you picked. Use `f32`
only for debugging numerical parity, never for deployment.

## Choose a corpus (search index) quantization

`corpus --quant` picks how chunk vectors are stored and scored:

| quant | bytes/vector @ dim 64 | scoring |
|---|---|---|
| `f32` | 256 | exact cosine |
| `int8` | 68 (4 scale + 64 codes) | cosine via int8 dot |
| `binary` | 8 | Hamming distance via popcount |
| `rescore` | 76 (int8 + binary sections) | Hamming prefilter, then exact int8 on survivors |

<!-- doctest: slow -->
```sh
uv run distillembed corpus --model models/howto --docs examples/device-docs/docs.txt \
    --out models/howto/corpus_int8.demc --quant int8
uv run distillembed corpus --model models/howto --docs examples/device-docs/docs.txt \
    --out models/howto/corpus_binary.demc --quant binary
uv run distillembed corpus --model models/howto --docs examples/device-docs/docs.txt \
    --out models/howto/corpus_rescore.demc --quant rescore
```

`binary` is smallest and fastest to scan but its Hamming-distance score is
only a proxy for cosine; `rescore` gets binary's fast prefilter *and* int8's
exact top-k scores by scanning all rows cheaply, then re-scoring only the
survivors — use it when the corpus is too large to brute-force in `int8` but
you still want exact scores on the results you show.

## Search from the command line, and cross-check quant choices

<!-- doctest: slow -->
```sh
make -C cpp
cpp/build/demo_search models/howto/model_int8.dem models/howto/corpus_int8.demc \
    "battery level low" 2
cpp/build/demo_search models/howto/model_int8.dem models/howto/corpus_rescore.demc \
    "battery level low" 2
```

`rescore`'s top-1 score should match `int8`'s exactly (it's the same int8 dot
product on the same winning row, just found via a cheaper prefilter first) —
`tests/test_search.py::test_rescore_matches_int8_scores` pins this down.

## Hybrid (dense + lexical) search

`--hybrid` fuses the dense ranking with a token-overlap ranking via
reciprocal rank fusion, which helps for exact-term queries (identifiers,
error codes) a dense embedding alone can miss:

<!-- doctest: slow -->
```sh
cpp/build/demo_search models/howto/model_int8.dem models/howto/corpus_int8.demc \
    "wifi sleep power saving" 3 --hybrid
```

## Cross-check the Python and C++ encoders agree

`distillembed embed` and `demo_embed` print the same token ids and (modulo
int8 quantization noise) the same vector — useful whenever you're debugging a
custom tokenizer/model and want to know whether a mismatch is in Python or
C++:

<!-- doctest: slow -->
```sh
uv run distillembed embed --model models/howto "battery level low"
cpp/build/demo_embed models/howto/model_int8.dem "battery level low"
```

For an exact (not just eyeball) parity check across the whole vocabulary,
run the test suite (next section) — `tests/test_parity.py` compares token
ids one-for-one between SentencePiece and the C++ Viterbi tokenizer, and
`tests/test_noheap.py` checks the zero-heap `EncodeBuffers` path against the
heap path.

## Run the test suite

Core format/tokenizer/search tests need no teacher and run in well under a
second; they're the fast regression check to run after any change to the
binary format, tokenizer, or C++ engine:

<!-- doctest: run -->
```sh
uv run pytest -q
```

```text
18 passed, 2 warnings in 0.24s
```

`test_search.py`'s C++-backed tests are automatically skipped if
`cpp/build/demo_search` doesn't exist yet — run `make -C cpp` first if you
want them included.

## Generate a minimal on-target embedding-only program

If you only need embeddings on-device (e.g. you're doing search server-side
and just want a compact/fast query encoder on the edge), you don't need the
`search.hpp` half at all:

<!-- doctest: slow file=examples/device-docs/embed_only.cpp -->
```cpp
#include <cstdio>
#include <vector>

#include "distillembed/model.hpp"
#include "model_blob.hpp"

int main(int argc, char** argv) {
  const auto model = demb::Model::load(g_model, g_model_len);
  if (!model) return 1;

  const std::string_view text = argc > 1 ? argv[1] : "battery level low";
  std::vector<float> vec(model->dim());
  const size_t n_tokens = model->embed(text, vec);

  std::printf("%zu tokens -> [%f, %f, %f, ...]\n", n_tokens, vec[0], vec[1], vec[2]);
  return 0;
}
```

<!-- doctest: slow -->
```sh
uv run distillembed carray --in models/howto/model_int8.dem \
    --out examples/device-docs/model_blob.hpp --symbol g_model
g++ -std=c++23 -O2 -Iexamples/device-docs -Icpp/include \
    examples/device-docs/embed_only.cpp -o /tmp/embed_only
/tmp/embed_only "battery level low"
```

For the zero-heap variant (fixed-size static scratch, no `std::vector`), see
`EncodeBuffers` in `cpp/include/distillembed/tokenizer.hpp` and how
`demb_selftest` (`cpp/src/demb_selftest.cpp`) exercises it against the heap
path for exact agreement.

## Testing these docs

Every block above tagged `<!-- doctest: ... -->` is extracted and actually
run, in document order, inside a fresh `git clone --local` of the repo:

```sh
uv run python tools/run_doctests.py HOWTO.md            # fast blocks only
uv run python tools/run_doctests.py --slow HOWTO.md      # incl. teacher-dependent recipes
uv run python tools/run_doctests.py --slow --keep HOWTO.md QUICKSTART.md  # both, keep the clone
```

There's no dependency solver — a directive comment tags each block as `run`
(default), `skip`, `slow` (needs network/torch/time; opt-in via `--slow`), or
`file=PATH` (materialize verbatim instead of executing, for a later block to
compile/invoke). All tagged blocks across the given files are stitched, in
document order, into one `set -euo pipefail` bash script and run once — the
same way a reader would, top to bottom, in one shell. See
`tools/run_doctests.py` for the full scheme.
