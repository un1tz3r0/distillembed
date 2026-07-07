# Changelog

## 2026-07-03 (night)

### Added
- int4 model tables (qtype=2): per-row scale, two codes/byte, branch-free
  nibble sign-extension. Real model: 745KB vs 1.27MB int8, identical top-3
  search results.
- Two-stage rescore corpora (qtype=4): binary Hamming prefilter over all rows,
  exact int8 rescore of the top k×prefilter_factor survivors.
- No-heap inference: `Tokenizer::encode_into` + `Model::embed(text, out,
  EncodeBuffers, ids)` — caller-provided scratch, zero allocation; verified
  against the heap path by the new `demb_selftest` binary (static buffers).
- Comprehensive README rewrite (results tables, on-target usage, invariants).

### Changed
- `.dem` format v2: header gains `min_score` (f32); tokenizer section gains a
  surface-sorted `(rec_offset, piece_id)` search index so the C++ tokenizer
  binary-searches the flash blob in place — replaces the RAM hash map
  entirely (~1KB RAM for a loaded model, down from the whole vocab in RAM).
  Corpus format unchanged (v1). Test suite now 18 tests.

### Removed
- Vocab pruning dropped from the roadmap by decision: the model stays usable
  on general non-technical text.

## 2026-07-03 (evening)

### Added
- Binary corpus quantization (qtype=3): 1 sign bit/dim, `std::popcount` Hamming
  scoring mapped to a cosine proxy. 8× smaller vectors than int8.
- `distillembed carray`: render .dem/.demc blobs as `alignas(4)` C++ headers;
  flash-resident `Model::load(ptr, len)` verified by compiling and running a
  no-filesystem demo against the real 1.2MB model.
- `demo_search --hybrid`: reciprocal rank fusion of dense + lexical
  (token-overlap) rankings.
- Sentence-packing chunker (~350 chars, one-sentence overlap) in
  prepare_corpus.py; end-to-end search tests (f32/int8/binary/hybrid), 11 total.

### Changed
- C++ engine rewritten in C++23 (GCC 15): `std::expected` error handling
  (exception-free), `quant_traits` + concepts with `requires`-constrained
  `RowMatrix<QType>` kernels, fold-expression `make_table` dispatch into
  `std::variant`, `consteval` FOURCC magics, transparent-hash heterogeneous
  map lookup (zero-allocation Viterbi hot loop), ranges algorithms.
- `models/docs` refined 5 epochs over the full 300k-line corpus:
  spearman 0.55 → 0.59, retrieval overlap@10 0.32 → 0.37.

## 2026-07-03 (later)

### Added
- `scripts/prepare_corpus.py`: builds cleaned sentence corpus + paragraph chunks
  from scraped-markdown documentation trees (strips code fences, links, nav junk).
- Retrieval-agreement metric (`retrieval_overlap@10` vs teacher) in `eval`.
- First real model, `models/docs` (untracked): MiniLM-L6-v2 distilled to
  8192×128 int8 (1.2MB) on 300k sentences from ../markdown-manuals collections.
  Refine (2 epochs, 30k lines) lifted spearman 0.25→0.55; C++ demo_search
  returns correct chunks for natural-language queries over 20k doc chunks.

### Fixed
- C++ unigram Viterbi mis-scored byte-fallback: byte pieces carry score 0.0 and
  must not be lattice edges. Now matches SentencePiece: unknown character = one
  node at `min_score − 10`, expanded to byte pieces at backtrack. Exact token-id
  parity with Python verified by tests.

## 2026-07-03

### Added
- Initial scaffold, part 1 (Python, uv): `distillembed` package — SentencePiece
  unigram tokenizer training, Model2Vec-style static-table distillation
  (per-piece teacher embeddings → PCA → SIF weight folding), optional
  tokenlearn-style refinement, per-row int8 quantization, `.dem`/`.demc`
  binary export + reader, numpy reference encoder, similarity-fidelity eval,
  CLI with subcommands.
- Initial scaffold, part 2 (C++17, no external deps): header-only engine —
  SentencePiece-compatible unigram Viterbi tokenizer with byte fallback,
  zero-copy `.dem` loader (flash-friendly), f32/int8 embedding, brute-force
  cosine search; `demo_embed` and `demo_search` CLIs with Makefile.
- Tests: binary format roundtrip, Python↔C++ tokenizer and embedding parity.
- Docs: DESIGN.md (format spec, parity contract, MCU notes), TODO.md, README.md.
