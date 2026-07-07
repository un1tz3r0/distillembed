// Semantic search: embed a query with a .dem model and rank a .demc corpus.
//
// --hybrid fuses the dense ranking with a lexical (token-overlap) ranking via
// reciprocal rank fusion: score(d) = Σ 1/(60 + rank). Covers exact-term
// queries (identifiers, error codes) that dense embeddings can miss.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

#include "distillembed/model.hpp"
#include "distillembed/search.hpp"

namespace {

[[nodiscard]] std::vector<uint32_t> unique_ids(const demb::Tokenizer& tok,
                                               std::string_view text) {
  auto ids = tok.encode(text);
  std::ranges::sort(ids);
  const auto dup = std::ranges::unique(ids);
  ids.erase(dup.begin(), dup.end());
  return ids;
}

// Cosine similarity between two sorted unique id sets.
[[nodiscard]] float lexical_score(std::span<const uint32_t> a, std::span<const uint32_t> b) {
  if (a.empty() || b.empty()) return 0.0f;
  size_t common = 0, i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) { ++common; ++i; ++j; }
    else if (a[i] < b[j]) { ++i; }
    else { ++j; }
  }
  return static_cast<float>(common) /
         std::sqrt(static_cast<float>(a.size()) * static_cast<float>(b.size()));
}

// Reciprocal rank fusion of pre-ranked hit lists (all the same length).
template <class... Ranked>
[[nodiscard]] std::vector<demb::Hit> rrf_fuse(size_t count, size_t k, const Ranked&... rankings) {
  std::vector<float> fused(count, 0.0f);
  const auto add = [&](const std::vector<demb::Hit>& ranking) {
    for (size_t rank = 0; rank < ranking.size(); ++rank) {
      if (ranking[rank].score > 0.0f) fused[ranking[rank].index] += 1.0f / (60.0f + rank);
    }
  };
  (add(rankings), ...);
  std::vector<demb::Hit> hits;
  hits.reserve(count);
  for (uint32_t i = 0; i < count; ++i) hits.push_back({.index = i, .score = fused[i]});
  demb::Corpus::top_k(hits, k);
  return hits;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model.dem> <corpus.demc> <query> [k] [--hybrid]\n", argv[0]);
    return 2;
  }
  const auto model = demb::Model::load_file(argv[1]);
  if (!model) {
    std::fprintf(stderr, "error: %s: %s\n", argv[1], demb::to_string(model.error()));
    return 1;
  }
  const auto corpus = demb::Corpus::load_file(argv[2]);
  if (!corpus) {
    std::fprintf(stderr, "error: %s: %s\n", argv[2], demb::to_string(corpus.error()));
    return 1;
  }
  if (corpus->dim() != model->dim()) {
    std::fprintf(stderr, "error: corpus dim %u != model dim %u\n", corpus->dim(), model->dim());
    return 1;
  }
  const std::string_view query = argv[3];
  const size_t k = argc > 4 && argv[4][0] != '-' ? static_cast<size_t>(std::atoi(argv[4])) : 5;
  const bool hybrid = std::strcmp(argv[argc - 1], "--hybrid") == 0;

  std::vector<float> qvec(model->dim());
  if (model->embed(query, qvec) == 0) {
    std::fprintf(stderr, "error: query produced no tokens\n");
    return 1;
  }

  std::vector<demb::Hit> hits;
  if (hybrid) {
    // Full dense ranking + full lexical ranking, fused with RRF.
    auto dense = corpus->search(qvec, corpus->count());
    const auto qids = unique_ids(model->tokenizer(), query);
    std::vector<demb::Hit> lexical;
    lexical.reserve(corpus->count());
    for (uint32_t i = 0; i < corpus->count(); ++i) {
      const auto doc_ids = unique_ids(model->tokenizer(), corpus->text(i));
      lexical.push_back({.index = i, .score = lexical_score(qids, doc_ids)});
    }
    demb::Corpus::top_k(lexical, lexical.size());
    hits = rrf_fuse(corpus->count(), k, dense, lexical);
  } else {
    hits = corpus->search(qvec, k);
  }

  for (const auto& [index, score] : hits) {
    const auto text = corpus->text(index);
    std::printf("%.4f  [%u] %.*s\n", score, index, static_cast<int>(text.size()), text.data());
  }
  return 0;
}
