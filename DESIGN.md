# distillembed — Design

Distill a sentence-transformer teacher into a tiny static-embedding student that
runs on microcontrollers, plus a dependency-free C++ inference engine.

## Goals

- **Part 1 (Python, `uv` project):** distillation pipeline. Train a SentencePiece
  tokenizer, distill a teacher model into a per-token static embedding table
  (Model2Vec-style), optionally refine against teacher sentence embeddings,
  quantize, and export a single binary blob.
- **Part 2 (C++17, zero external deps):** inference engine that loads the blob
  (from file or directly from a flash-resident `const uint8_t*`), tokenizes,
  embeds via table lookup + sum + L2-normalize, and does brute-force cosine
  search over a precomputed corpus.

## Architecture

```
corpus.txt ──► [tokenizer]  spm.model (SentencePiece unigram)
spm.model  ──► [distill]    embeddings.npy   (teacher → per-piece vectors → PCA → SIF-weight fold)
           ──► [refine]     embeddings.npy   (optional: train table vs teacher sentence embeddings)
model dir  ──► [export]     model.dem        (tokenizer + quantized table, one blob)
docs.txt   ──► [corpus]     corpus.demc      (quantized chunk vectors + texts)

model.dem + corpus.demc ──► C++ engine ──► query embedding, top-k cosine search
```

## Key decisions

### Tokenizer parity contract
The C++ side re-implements SentencePiece **unigram** inference (Viterbi over
piece scores). To make that exact, tokenizers are always trained with:

- `model_type=unigram`, `normalization_rule_name=identity` (no NFKC — the C++
  side does not implement Unicode normalization; normalize text upstream if needed)
- `add_dummy_prefix=true`, `remove_extra_whitespaces=true` — mirrored in C++ as:
  collapse `[ \t\n\r]+` runs to a single `▁` (U+2581), prepend one `▁`, drop trailing whitespace
- `byte_fallback=true` — an unknown *character* is a single lattice node scored
  `min_score − 10.0` (SentencePiece's unk penalty); byte-piece scores are never
  used as edge weights (they are stored as 0.0). The unk character's surface is
  expanded into `<0xNN>` byte pieces during backtrack, matching SentencePiece
- `unk_id=0`, no bos/eos/pad (dense table, no wasted rows)

### Student = weighted static table
Each vocabulary piece is embedded **individually** by the teacher, PCA-projects
to the target dim, then each row is scaled by its SIF weight
`w = a / (a + p)` where `p = exp(unigram_score)` is the piece's unigram
probability. Because the final embedding is L2-normalized, folding weights into
rows makes inference a plain **sum of rows + normalize** — no separate weight
table on device. Rows for unk/control pieces (and non-printable byte pieces)
are zeroed so they contribute nothing.

`refine` (tokenlearn-style) trains the folded table directly against
PCA-projected teacher sentence embeddings (cosine loss) on an unlabeled corpus;
`pca_mean.npy` / `pca_components.npy` are saved by `distill` for this purpose.

### Quantization
Model tables: per-row symmetric **int8** (`scale_r = max|row|/127`) or **int4**
(qtype=2: codes in [-7,7], two per byte — even dim in the low nibble — plus the
per-row f32 scale). int4 halves the table again (8192×128 ≈ 560KB) at a small
fidelity cost; sign-extension is `(nibble ^ 8) - 8`, branch-free. f32 export
supported for debugging/parity tests. Corpus vectors are L2-normalized before
quantization, so dot product ≈ cosine.

Corpus files additionally support **binary** quantization (qtype=3): 1 sign bit
per dimension, little bit-order (numpy `packbits(bitorder="little")`), scored
by Hamming distance via `std::popcount` and mapped to a cosine proxy
`1 − 2·H/dim`. A 128-dim vector is 16 bytes — 8× smaller than int8 — at a
modest recall cost; ideal when the chunk texts live off-device.

**Two-stage rescore corpora** (qtype=4): the int8 section followed by the
binary section (int8 row bytes are whole bytes, so the binary section stays
byte-aligned). Search scans *all* rows with the cheap Hamming prefilter, keeps
the top `k × prefilter_factor` (default 8), rescored with exact int8 dot
products. Full int8 accuracy at prefilter speed, for +12.5% storage over int8.

### C++ engine architecture (C++23)
Header-only, exception-free (`std::expected` errors), zero external deps:
- `io.hpp` — `ByteReader` (fold-expression `read_all`), `consteval` FOURCC
  magics, `LoadError`/`Result`, little-endian `static_assert`.
- `quant.hpp` — `quant_traits<QType>` + `Quantized`/`DenseQuantized` concepts;
  `RowMatrix<Q>` non-owning view whose kernels are enabled per-scheme with
  `requires` clauses; `make_table<Allowed...>` folds a runtime qtype into the
  matching `std::variant` alternative (adding a scheme = appending to a pack).
- `tokenizer.hpp` — unigram Viterbi; transparent heterogeneous map lookup so
  the hot loop does zero allocations.
- `model.hpp` / `search.hpp` — loaders return `Result<Model>`/`Result<Corpus>`;
  scoring dispatches through `std::visit` with a templated lambda +
  `if constexpr` on `RowMatrix::qtype`.

### Hybrid search (demo_search --hybrid)
Reciprocal rank fusion (`Σ 1/(60+rank)`) of the dense ranking with a lexical
ranking (cosine over unique-token-id sets). Catches exact-term queries
(identifiers, error codes) that dense embeddings miss.

## Binary formats (little-endian)

### `.dem` model file (v2) — header 32 bytes

| field | type | notes |
|---|---|---|
| magic | `char[4]` | `"DEMB"` |
| version | u16 | 2 |
| qtype | u8 | 0=f32, 1=int8, 2=int4 |
| reserved | u8 | 0 |
| dim | u32 | embedding dimension |
| vocab_size | u32 | |
| unk_id | u32 | |
| max_piece_len | u32 | max surface bytes over normal pieces |
| min_score | f32 | min unigram score over normal pieces (unk penalty base) |
| tok_section_size | u32 | bytes incl. padding |

Tokenizer section (v2, all offsets little-endian):
1. `u32 n_search` — number of *normal* pieces (the searchable surface set)
2. `{u32 rec_offset, u32 piece_id}[n_search]` — **sorted by surface bytes**
   (unsigned byte order); this is the flash-resident binary-search index
3. `vocab_size` packed records **in id order**:
   `f32 score, u16 len, u8 type, u8 pad, u8 bytes[len]`
   (type: 0=normal, 1=unk, 2=control/unused, 3=byte)
4. zero-padding to a 4-byte boundary

The index means the C++ loader builds *nothing*: lookup is a binary search
over the blob itself (unk/control/byte pieces are excluded from the index so
they can never match surface text). The only RAM cost is the 1KB
byte-fallback table, filled by one sequential record scan at init.

Embedding section: qtype 0 → `f32 table[vocab*dim]`;
qtype 1 → `f32 scales[vocab]` then `i8 codes[vocab*dim]`;
qtype 2 → `f32 scales[vocab]` then packed nibbles `u8[vocab*ceil(dim/2)]`.

### `.demc` corpus file (v1) — header 20 bytes

`magic "DEMC", u16 version=1, u8 qtype, u8 reserved, u32 dim, u32 count,
u32 vec_section_size`, then the vector section (padded to 4), then `count`
texts as `u32 len, u8 bytes[len]`. qtype: 0=f32, 1=int8, 3=binary (packed sign
bits), 4=rescore (int8 section followed by binary section).

## MCU port notes

- `Model::load(const uint8_t*, size_t)` is zero-copy end to end: tables *and*
  the tokenizer's search index are read in place from a flash-resident blob
  (`distillembed carray` renders any .dem/.demc as an `alignas(4)` C++
  header). Loading a model costs ~1KB of RAM (byte-fallback table).
- **No-heap inference**: `Tokenizer::encode_into` / the `Model::embed`
  overload taking `EncodeBuffers` + an id scratch span never allocate. Buffer
  sizing: `Tokenizer::normalized_capacity(max_text_len)` bytes of norm buffer,
  that +1 floats/int32s/size_ts for the lattice, same count of u32 ids, plus
  `dim` floats for the output. For 256-byte queries at dim 128 that is ~14KB
  of statically-allocatable scratch (see cpp/src/demb_selftest.cpp).
- No exceptions or RTTI required; errors are `std::expected`.
- The demo binaries use stdio/heap; the core headers avoid I/O. Remaining
  MCU-hardening work (static allocation, trie instead of `unordered_map`,
  int-only accumulation) is tracked in TODO.md.
- Assumes little-endian target (true for Cortex-M, ESP32, RISC-V).

## Sizing rule of thumb

int8 table = `vocab × dim` bytes + `4 × vocab` scales.
Default 8192 × 128 ≈ 1.03 MB flash. Halve via `--vocab-size 4096` or `--dim 64`.
