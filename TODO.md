# TODO

## Phase 1 — scaffold (this session)
- [x] uv project layout, pyproject with `distill` extra (torch kept optional)
- [x] SentencePiece tokenizer training wrapper (parity-constrained flags)
- [x] Distillation: per-piece teacher embedding → PCA → SIF weight folding
- [x] Refinement (tokenlearn-style) training loop vs projected teacher embeddings
- [x] int8 per-row quantization + `.dem` / `.demc` binary export + Python reader
- [x] Numpy reference encoder (`StaticModel`) + CLI (tokenizer/distill/refine/export/corpus/embed/eval)
- [x] C++17 header-only engine: unigram Viterbi tokenizer w/ byte fallback, model loader, embed, brute-force search
- [x] `demo_embed` / `demo_search` binaries + Makefile
- [x] Tests: format roundtrip, Python↔C++ tokenizer & embedding parity

## Phase 2 — quality
- [x] Run a real distillation and eval fidelity — `models/docs`: MiniLM-L6-v2 →
      8192×128 int8 (1.2MB), corpus = markdown-manuals collections (300k sentences).
      Pre-refine spearman 0.25 → post-refine 0.55, retrieval overlap@10 = 0.32.
      Search demo answers real queries correctly. (2026-07-03)
- [x] Retrieval-agreement metric (overlap@10 vs teacher) added to eval
- [x] Longer refine: 5 epochs × 300k lines, loss 0.131→0.123. Final fidelity:
      spearman 0.59, retrieval overlap@10 = 0.37 (was 0.25 / — pre-refine). (2026-07-03)
- [x] Sentence-packing chunker with one-sentence overlap in prepare_corpus.py
- [x] Hybrid scoring in demo_search (`--hybrid`: RRF of dense + token-overlap rankings)
- [~] Vocab pruning — **skipped by decision** (2026-07-03): the model should
      stay flexible on general non-technical text, so vocabulary outside the
      training corpus is deliberately kept.

## Phase 3 — compression (done 2026-07-03)
- [x] Binary-quantized corpus vectors (qtype=3, 1 bit/dim, popcount Hamming) —
      16 bytes/vec at dim 128, 8× smaller than int8; top-1 preserved in tests
- [x] Export blobs as C++ headers (`distillembed carray`); flash-resident
      `Model::load(ptr, len)` compile+run verified
- [x] int4 packed model tables (qtype=2): real model 745KB vs 1.27MB int8,
      identical top-3 search results, scores within ±0.002
- [x] Two-stage rescore search (corpus qtype=4): Hamming prefilter over all
      rows → exact int8 dot on top k×8; matches int8 scores in tests

## Phase 3.5 — engine modernization (done 2026-07-03)
- [x] C++23 rewrite: std::expected loaders, concepts-constrained RowMatrix<QType>
      kernels, pack/fold make_table dispatch into std::variant, consteval FOURCC,
      transparent-hash zero-alloc Viterbi, ranges. See DESIGN.md.

## Phase 4 — MCU hardening (core done 2026-07-03)
- [x] Zero-RAM tokenizer lookup: .dem v2 carries a surface-sorted search index;
      C++ binary-searches the flash blob in place (supersedes the trie plan —
      no load-time build, only the 1KB byte-fallback table in RAM)
- [x] No-heap encode/embed path (EncodeBuffers + id scratch; demb_selftest
      verifies exact agreement with the heap path using static buffers)
- [ ] Int-only accumulation option (shared exponent) for FPU-less cores
- [ ] Prefix-narrowing in the Viterbi lookup (reuse the sorted range across
      lengths instead of a fresh binary search per length) — perf, not correctness
- [ ] Cross-compile smoke on arm-none-eabi / ESP-IDF toolchains

## Blockers / open questions
- None currently. Non-ASCII text needs upstream Unicode normalization (identity
  normalizer — see DESIGN.md).
