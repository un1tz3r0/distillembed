# Quickstart

One straight walkthrough: install, get a corpus, distill a tiny student model
from an off-the-shelf sentence-transformer teacher, fine-tune it on your own
text, then generate C++ and run an on-device search index against it.

Every command below is real and tested — see [Testing these docs](#testing-these-docs)
at the bottom. Commands are shown as run from the repo root.

The running example is the project's own domain: searching device/firmware
documentation (battery, Wi-Fi sleep, OTA updates, ...) from a microcontroller
with no filesystem and no heap-resident model. We use the small synthetic
corpus checked in at `examples/device-docs/` so the whole walkthrough runs in
about a minute, teacher included — swap in your own text at the "fine-tune on
your corpus" step and nothing else changes.

## 1. Install

```sh
uv sync
```

This installs the core runtime (`numpy`, `sentencepiece`) — enough for
tokenizer training, export, and the reference encoder. Distillation and
refinement additionally need a real teacher model, which pulls in torch:

<!-- doctest: slow -->
```sh
uv sync --extra distill
```

Build the C++ demo binaries (`demo_embed`, `demo_search`, and the self-test):

<!-- doctest: run -->
```sh
make -C cpp
```

## 2. Get a corpus

You need two text files, one sentence/line each:

- a **training corpus** (`corpus.txt`) — plain prose, used to train the
  tokenizer and to fine-tune the student against the teacher
- a **docs corpus** (`docs.txt`) — the paragraph-sized chunks you actually
  want to search over at runtime

For this walkthrough we use the bundled example:

<!-- doctest: run -->
```sh
wc -l examples/device-docs/corpus.txt examples/device-docs/docs.txt
```

```text
900 examples/device-docs/corpus.txt
40 examples/device-docs/docs.txt
```

To build these two files from your own scraped-markdown documentation tree
instead, see `scripts/prepare_corpus.py` (covered in HOWTO.md) — it filters
out nav junk/code fences and packs prose into sentence- and chunk-level
corpora automatically.

## 3. Train the tokenizer

A SentencePiece unigram vocabulary sized to your corpus. Vocab size must be
well under your corpus's unique-substring budget — SentencePiece will tell
you the ceiling if you overshoot; 512 fits this 900-line example (a
real-world corpus of tens/hundreds of thousands of lines comfortably supports
the README's default of 8192):

<!-- doctest: run -->
```sh
uv run distillembed tokenizer --corpus examples/device-docs/corpus.txt \
    --out models/quickstart/spm --vocab-size 512
```

## 4. Distill the teacher into a static table

One vector per vocabulary piece, embedded by the teacher, PCA'd down to your
target dimension (128 is the README default; we use 64 here to keep the demo
snappy):

<!-- doctest: slow -->
```sh
uv run distillembed distill --spm models/quickstart/spm.model \
    --teacher sentence-transformers/all-MiniLM-L6-v2 --dim 64 \
    --out models/quickstart
```

This writes `models/quickstart/{embeddings.npy,pca_mean.npy,pca_components.npy,config.json}`
plus a copy of the tokenizer. At this point the student already works, just
not very well yet — refinement is where most of the fidelity comes from.

## 5. Fine-tune on your corpus

Trains the table so pooled student embeddings match the teacher's *sentence*
embeddings (not just per-piece embeddings) on your unlabeled corpus. This is
the step to point at custom/domain text — swap `examples/device-docs/corpus.txt`
for your own file and everything else in this walkthrough is unchanged:

<!-- doctest: slow -->
```sh
uv run distillembed refine --model models/quickstart \
    --corpus examples/device-docs/corpus.txt --epochs 5
```

```text
epoch 1/5: cosine loss 0.1788
epoch 2/5: cosine loss 0.0528
epoch 3/5: cosine loss 0.0292
epoch 4/5: cosine loss 0.0196
epoch 5/5: cosine loss 0.0160
```

(This toy corpus is templated and highly repetitive, so loss drops further
and faster than it will on natural text — see the real 300k-sentence run in
the top-level README for realistic numbers: spearman 0.59, retrieval
overlap@10 0.37.)

## 6. Measure what you kept

Compares student cosine similarities and retrieval rankings against the
teacher, on held-out pairs from the same corpus:

<!-- doctest: slow -->
```sh
uv run distillembed eval --model models/quickstart \
    --corpus examples/device-docs/corpus.txt --pairs 200
```

```json
{
  "teacher": "sentence-transformers/all-MiniLM-L6-v2",
  "n_pairs": 200,
  "pearson": 0.74,
  "spearman": 0.72,
  "retrieval_overlap@10": 0.55
}
```

## 7. Export the deployable blob

Renders the tokenizer + table as one binary (`.dem`), quantized:

<!-- doctest: slow -->
```sh
uv run distillembed export --model models/quickstart --quant int8
```

```text
wrote models/quickstart/model.dem (43,500 bytes, int8)
```

## 8. Build a searchable corpus

Embeds `docs.txt` (one chunk per line) with the *student* you just built and
writes the vectors + source texts as a `.demc`:

<!-- doctest: slow -->
```sh
uv run distillembed corpus --model models/quickstart \
    --docs examples/device-docs/docs.txt \
    --out models/quickstart/corpus.demc --quant int8
```

```text
wrote models/quickstart/corpus.demc (40 chunks, dim 64, int8)
```

Sanity-check it from the command line before writing any C++:

<!-- doctest: slow -->
```sh
cpp/build/demo_search models/quickstart/model.dem models/quickstart/corpus.demc \
    "device is running low on battery" 2
```

```text
0.7554  [29] Refer to the section on power consumption tuning if ...
0.7469  [7] Refer to the section on battery charging thresholds if ...
```

## 9. Generate C++ headers for on-device linking

`carray` renders any `.dem`/`.demc` blob as a `#pragma once` C++ header
holding a `constexpr` byte array — meant to be linked straight into flash, no
filesystem required on the target:

<!-- doctest: slow -->
```sh
uv run distillembed carray --in models/quickstart/model.dem \
    --out models/quickstart/model_blob.hpp --symbol g_model
uv run distillembed carray --in models/quickstart/corpus.demc \
    --out models/quickstart/corpus_blob.hpp --symbol g_corpus
```

## 10. The real-world target: an on-device search program

This is the actual deployment shape — model and corpus are compile-time
constants, `demb::Model::load`/`demb::Corpus::load` read them zero-copy
straight out of flash, and search is a brute-force scan with no heap, no
filesystem, no libc string/stream dependency beyond what you bring:

<!-- doctest: slow file=examples/device-docs/onboard_search.cpp -->
```cpp
// Flash-resident search: model + corpus are compiled-in C++ arrays (as
// produced by `distillembed carray`), zero-copy loaded, no filesystem.
#include <cstdio>

#include "distillembed/model.hpp"
#include "distillembed/search.hpp"
#include "model_blob.hpp"
#include "corpus_blob.hpp"

int main(int argc, char** argv) {
  const auto model = demb::Model::load(g_model, g_model_len);
  if (!model) {
    std::fprintf(stderr, "model load failed: %s\n", demb::to_string(model.error()));
    return 1;
  }
  const auto corpus = demb::Corpus::load(
      std::span{reinterpret_cast<const std::byte*>(g_corpus), g_corpus_len});
  if (!corpus) {
    std::fprintf(stderr, "corpus load failed: %s\n", demb::to_string(corpus.error()));
    return 1;
  }

  const std::string_view query = argc > 1 ? argv[1] : "battery level low";
  std::vector<float> qvec(model->dim());
  model->embed(query, qvec);

  for (const auto& [index, score] : corpus->search(qvec, 3)) {
    std::printf("%.4f  [%u] %.*s\n", score, index,
                static_cast<int>(corpus->text(index).size()), corpus->text(index).data());
  }
  return 0;
}
```

Build it against the headers from step 9 and the engine headers under
`cpp/include/`, exactly as you would for a real target (swap `g++`/native
flags for your cross-compiler; the engine itself is freestanding-friendly —
no exceptions, no RTTI, little-endian `static_assert`ed):

<!-- doctest: slow -->
```sh
g++ -std=c++23 -O2 -Wall -Wextra \
    -Imodels/quickstart -Icpp/include \
    examples/device-docs/onboard_search.cpp -o models/quickstart/onboard_search
models/quickstart/onboard_search "firmware update fails"
```

```text
0.8127  [35] The power management unit logs a warning to the console when a firmware update is pending.
0.7785  [20] Refer to the section on flash partition layout if ...
0.7740  [25] Refer to the section on over-the-air firmware updates if ...
```

That's the whole loop: teacher → tiny static table → fine-tuned on your text
→ one flash blob + one C++ header → a search program with zero runtime
dependencies. For quantization tradeoffs, hybrid lexical+dense search,
incremental refinement, and other variations, see [HOWTO.md](HOWTO.md).

## Testing these docs

Every shell/C++ block above tagged with a `<!-- doctest: ... -->` comment is
extracted and actually run — in document order, inside a fresh
`git clone --local` of the repo — by:

```sh
uv run python tools/run_doctests.py QUICKSTART.md          # fast blocks only
uv run python tools/run_doctests.py --slow QUICKSTART.md   # full pipeline (network + torch)
```

See `tools/run_doctests.py` for how the tagging scheme works.
