// .demc corpus loader + brute-force top-k. Binary format: DESIGN.md.
//
// Corpus vectors are L2-normalized before quantization, so for the dense
// schemes the dot product is the cosine similarity. The binary scheme stores
// sign bits (16 bytes per 128-dim vector) and scores by Hamming distance via
// std::popcount, mapped to [-1, 1]. Zero-copy like Model.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "io.hpp"
#include "model.hpp"
#include "quant.hpp"

namespace demb {

struct Hit {
  uint32_t index;
  float score;
};

// Sign-quantize a float vector into packed bits (little bit-order, matching
// numpy packbits(bitorder="little") on the Python side).
[[nodiscard]] inline std::vector<std::byte> binarize(std::span<const float> v) {
  std::vector<std::byte> bits((v.size() + 7) / 8);
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i] > 0.0f) bits[i / 8] |= std::byte{1} << (i % 8);
  }
  return bits;
}

class Corpus {
 public:
  using Table =
      std::variant<RowMatrix<QType::f32>, RowMatrix<QType::int8>, RowMatrix<QType::binary>>;

  [[nodiscard]] static Result<Corpus> load(std::span<const std::byte> blob) noexcept {
    ByteReader r{blob};
    uint32_t magic = 0, dim = 0, count = 0, vec_size = 0;
    uint16_t version = 0;
    uint8_t qtype = 0, reserved = 0;
    if (!r.read_all(magic, version, qtype, reserved, dim, count, vec_size)) {
      return std::unexpected(LoadError::truncated);
    }
    if (magic != kCorpusMagic) return std::unexpected(LoadError::bad_magic);
    if (version != kCorpusVersion) return std::unexpected(LoadError::bad_version);
    if (dim == 0) return std::unexpected(LoadError::bad_layout);

    Corpus c;
    c.dim_ = dim;
    c.count_ = count;
    const auto vec_blob = r.take(vec_size);
    if (!vec_blob) return std::unexpected(vec_blob.error());

    // rescore = int8 main table + binary prefilter section appended.
    auto q = static_cast<QType>(qtype);
    if (q == QType::rescore) {
      const size_t main_bytes = RowMatrix<QType::int8>::bytes_needed(count, dim);
      if (vec_blob->size() < main_bytes) return std::unexpected(LoadError::truncated);
      auto prefilter =
          RowMatrix<QType::binary>::from_blob(vec_blob->subspan(main_bytes), count, dim);
      if (!prefilter) return std::unexpected(prefilter.error());
      c.prefilter_ = *prefilter;
      q = QType::int8;  // from_blob only reads its own prefix of the span
    }
    auto table =
        make_table<QType::f32, QType::int8, QType::binary>(q, *vec_blob, count, dim);
    if (!table) return std::unexpected(table.error());
    c.table_ = *std::move(table);

    c.texts_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t len = 0;
      if (!r.read(len)) return std::unexpected(LoadError::truncated);
      const auto text = r.take(len);
      if (!text) return std::unexpected(text.error());
      c.texts_.emplace_back(reinterpret_cast<const char*>(text->data()), text->size());
    }
    return c;
  }

  [[nodiscard]] static Result<Corpus> load_file(const char* path) {
    return read_file(path).and_then([](std::vector<std::byte> data) -> Result<Corpus> {
      auto c = load(data);
      if (c) c->owned_ = std::move(data);  // views/texts point into `data`
      return c;
    });
  }

  // Brute-force scan; returns the top k hits, best first. Scores are cosine
  // (dense) or a Hamming-derived cosine proxy in [-1, 1] (binary).
  //
  // With a rescore corpus, a two-stage search runs instead: the binary
  // prefilter scans all rows cheaply (popcount over dim/8 bytes), then only
  // the top k*prefilter_factor survivors get exact int8 dot products.
  [[nodiscard]] std::vector<Hit> search(std::span<const float> query, size_t k,
                                        size_t prefilter_factor = 8) const {
    if (prefilter_) {
      const auto qbits = binarize(query);
      const float inv_dim = 1.0f / static_cast<float>(dim_);
      std::vector<Hit> hits;
      hits.reserve(count_);
      for (uint32_t i = 0; i < count_; ++i) {
        const auto h = static_cast<float>(prefilter_->hamming(i, qbits));
        hits.push_back({.index = i, .score = 1.0f - 2.0f * h * inv_dim});
      }
      top_k(hits, std::min<size_t>(count_, std::max(k * prefilter_factor, k)));
      for (auto& hit : hits) hit.score = dense_score(hit.index, query);
      top_k(hits, k);
      return hits;
    }
    auto hits = std::visit(
        [&]<class M>(const M& table) {
          std::vector<Hit> out;
          out.reserve(count_);
          if constexpr (M::qtype == QType::binary) {
            const auto qbits = binarize(query);
            const float inv_dim = 1.0f / static_cast<float>(dim_);
            for (uint32_t i = 0; i < count_; ++i) {
              const auto h = static_cast<float>(table.hamming(i, qbits));
              out.push_back({.index = i, .score = 1.0f - 2.0f * h * inv_dim});
            }
          } else {
            for (uint32_t i = 0; i < count_; ++i) {
              out.push_back({.index = i, .score = table.dot(i, query)});
            }
          }
          return out;
        },
        table_);
    top_k(hits, k);
    return hits;
  }

  static void top_k(std::vector<Hit>& hits, size_t k) {
    k = std::min(k, hits.size());
    std::ranges::partial_sort(hits, hits.begin() + static_cast<ptrdiff_t>(k),
                              std::ranges::greater{}, &Hit::score);
    hits.resize(k);
  }

  [[nodiscard]] constexpr uint32_t dim() const noexcept { return dim_; }
  [[nodiscard]] constexpr uint32_t count() const noexcept { return count_; }
  [[nodiscard]] std::string_view text(uint32_t i) const noexcept { return texts_[i]; }

  Corpus(Corpus&&) = default;
  Corpus& operator=(Corpus&&) = default;
  Corpus(const Corpus&) = delete;
  Corpus& operator=(const Corpus&) = delete;

 private:
  Corpus() = default;

  [[nodiscard]] float dense_score(uint32_t index, std::span<const float> query) const {
    return std::visit(
        [&]<class M>(const M& table) -> float {
          if constexpr (M::qtype != QType::binary) {
            return table.dot(index, query);
          } else {
            return 0.0f;  // unreachable: prefilter_ implies a dense main table
          }
        },
        table_);
  }

  Table table_{RowMatrix<QType::f32>{}};
  std::optional<RowMatrix<QType::binary>> prefilter_;
  std::vector<std::string_view> texts_;
  std::vector<std::byte> owned_;
  uint32_t dim_ = 0;
  uint32_t count_ = 0;
};

}  // namespace demb
