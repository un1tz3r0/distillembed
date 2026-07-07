# distillembed

Distill sentence-transformer embedding models into **tiny static lookup-table
models** that run semantic search on microcontrollers — then run them with a
**zero-dependency, header-only C++23 engine** that reads its model straight
out of flash.

```
teacher (MiniLM, 90MB, transformer)          student (1.2MB blob, table lookups)
  "configure the wifi connection"      →      sim = 0.76      ←  "set up wireless networking"
                                              sim = 0.16      ←  "allocate a buffer for the audio codec"
```

## How it works

The student is not a small transformer — it is a **static embedding table**
(Model2Vec-style): one vector per tokenizer piece, distilled from the teacher.
Inference is *tokenize → sum rows → L2-normalize*. No attention, no GEMM, no
floating-point model runtime — which is what makes microcontrollers viable.

1. **Tokenize** a training corpus into a SentencePiece unigram vocabulary.
2. **Distill**: embed every vocabulary piece with the teacher, PCA to the
   target dimension, fold per-piece SIF weights into the rows.
3. **Refine** (optional but worth ~2× fidelity): train the table so pooled
   student embeddings match the teacher's sentence embeddings on unlabeled text.
4. **Quantize + export** one binary blob (`.dem`) containing tokenizer + table.
5. **Search**: precompute chunk vectors off-device into a `.demc` corpus;
   on-device, embed the query and brute-force scan (int8 dot or binary Hamming).

## Results (first real model)

Teacher `all-MiniLM-L6-v2` (22.7M params, 384 dims) distilled on 300k sentences
of technical documentation; student 8192 pieces × 128 dims:

| artifact | size | notes |
|---|---|---|
| student `.dem`, int8 | 1.27 MB | tokenizer (+search index) + table, one blob |
| student `.dem`, int4 | 745 KB | identical top-3 results, scores within ±0.002 |
| corpus vectors, int8 | 132 B/chunk | cosine via int8 dot |
| corpus vectors, binary | 16 B/chunk | Hamming via popcount, 8× smaller |
| corpus vectors, rescore | 148 B/chunk | Hamming prefilter → exact int8 rescore |

RAM cost of a loaded model: **~1KB** (byte-fallback table) — the tokenizer
binary-searches its index directly in the flash blob. Inference can run with
**zero heap allocation** via the `EncodeBuffers` overloads (~14KB of static
scratch for 256-byte queries at dim 128).

| metric (vs teacher) | plain distill | + refine |
|---|---|---|
| Spearman (pair sims) | 0.25 | **0.59** |
| retrieval overlap@10 (2000 candidates) | — | **0.37** |

Example query against 20k documentation chunks, running in the C++ engine:
*"reduce power consumption in sleep mode"* → ESP32 Light/Deep-sleep guide,
Wi-Fi station sleep mode, XTAL power configuration.

## Quickstart

```sh
uv sync                  # core: numpy + sentencepiece (export/inference only)
uv sync --extra distill  # + CPU torch + sentence-transformers (distill/refine/eval)
make -C cpp              # demo_embed, demo_search, demb_selftest
uv run pytest            # 18 tests incl. exact Python↔C++ parity & no-heap agreement
```

For a full guided walkthrough (install → corpus → distill → fine-tune → export
→ on-device search program), see [QUICKSTART.md](QUICKSTART.md). For
task-oriented recipes (quantization tradeoffs, hybrid search, incremental
refine, ...), see [HOWTO.md](HOWTO.md). Every command in both is tested for
real by `tools/run_doctests.py`.

### Pipeline

```sh
# 1. Tokenizer (one sentence per line; see scripts/prepare_corpus.py for markdown trees)
uv run distillembed tokenizer --corpus data/corpus.txt --out models/base/spm --vocab-size 8192

# 2. Distill the teacher into a static table
uv run distillembed distill --spm models/base/spm.model \
    --teacher sentence-transformers/all-MiniLM-L6-v2 --dim 128 --out models/base

# 3. Refine against teacher sentence embeddings (the big fidelity win)
uv run distillembed refine --model models/base --corpus data/corpus.txt --epochs 5

# 4. Measure what you kept
uv run distillembed eval --model models/base --corpus data/corpus.txt

# 5. Export the deployable blob
uv run distillembed export --model models/base --quant int8   # or: int4

# 6. Build a searchable corpus (one chunk per line)
uv run distillembed corpus --model models/base --docs data/docs.txt \
    --out models/base/corpus.demc --quant int8     # or: binary / rescore

# 7. Flash-resident deployment: render blobs as C++ headers
uv run distillembed carray --in models/base/model.dem --out model_blob.hpp --symbol g_model
```

### Search from the command line

```sh
cpp/build/demo_embed  models/base/model.dem "a query phrase"
cpp/build/demo_search models/base/model.dem models/base/corpus.demc "a query phrase" 5
cpp/build/demo_search ... "a query phrase" 5 --hybrid   # RRF-fused dense + lexical
```

### On-target usage (no filesystem, no heap-resident model)

```cpp
#include "distillembed/model.hpp"
#include "model_blob.hpp"  // from `distillembed carray`: g_model, g_model_len

const auto model = demb::Model::load(g_model, g_model_len);  // zero-copy from flash
if (!model) { /* demb::to_string(model.error()) */ }

float vec[128];
model->embed("open the pod bay doors", vec);  // tokenizes, pools, L2-normalizes
```

The engine is exception-free (`std::expected` errors), RTTI-free, and the
loaders are zero-copy: tables are read in place from the blob. Little-endian
targets (Cortex-M, ESP32, RISC-V) only — enforced by `static_assert`.

## Repository layout

```
src/distillembed/     Python pipeline (tokenizer/distill/refine/quantize/export/eval + CLI)
scripts/              corpus preparation from scraped-markdown documentation trees
cpp/include/          header-only engine: io / quant / tokenizer / model / search
cpp/src/              demo_embed, demo_search
tests/                pytest suite incl. Python↔C++ tokenizer & embedding parity
DESIGN.md             architecture, binary format spec, parity contract, MCU notes
TODO.md               roadmap and status
```

## Design invariants worth knowing

- **Tokenizer parity contract**: the C++ unigram Viterbi exactly reproduces
  SentencePiece for models trained through `distillembed tokenizer`
  (identity normalization, byte fallback, pinned flags). Enforced by tests
  that compare token ids one-for-one. Details in DESIGN.md.
- **Weight folding**: SIF weights are folded into table rows, valid because
  the output is L2-normalized — so device-side pooling is a plain sum.
- **Quantization**: per-row symmetric int8 for tables; corpora additionally
  support 1-bit sign quantization scored by Hamming distance.

## License / status

Experimental scaffold, evolving fast. See CHANGELOG.md.
