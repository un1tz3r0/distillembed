// .dem model loader + static-embedding encoder. Binary format: DESIGN.md.
//
// Zero-copy: the table views alias the input blob, which must outlive the
// Model. On MCU targets point load() at a flash-resident const array — only
// the tokenizer's lookup map lives in RAM. load_file() keeps an owned copy.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "io.hpp"
#include "quant.hpp"
#include "tokenizer.hpp"

namespace demb {

inline void l2_normalize(std::span<float> v) noexcept {
  double sq = 0.0;
  for (const float x : v) sq += static_cast<double>(x) * x;
  if (sq <= 0.0) return;
  const float inv = 1.0f / static_cast<float>(std::sqrt(sq));
  for (float& x : v) x *= inv;
}

[[nodiscard]] inline Result<std::vector<std::byte>> read_file(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return std::unexpected(LoadError::io);
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size <= 0) {
    std::fclose(f);
    return std::unexpected(LoadError::io);
  }
  std::vector<std::byte> data(static_cast<size_t>(size));
  const bool ok = std::fread(data.data(), 1, data.size(), f) == data.size();
  std::fclose(f);
  if (!ok) return std::unexpected(LoadError::io);
  return data;
}

class Model {
 public:
  using Table =
      std::variant<RowMatrix<QType::f32>, RowMatrix<QType::int8>, RowMatrix<QType::int4>>;

  [[nodiscard]] static Result<Model> load(std::span<const std::byte> blob) noexcept {
    ByteReader r{blob};
    uint32_t magic = 0, dim = 0, vocab = 0, unk_id = 0, max_piece_len = 0, tok_size = 0;
    uint16_t version = 0;
    uint8_t qtype = 0, reserved = 0;
    float min_score = 0.0f;
    if (!r.read_all(magic, version, qtype, reserved, dim, vocab, unk_id, max_piece_len,
                    min_score, tok_size)) {
      return std::unexpected(LoadError::truncated);
    }
    if (magic != kModelMagic) return std::unexpected(LoadError::bad_magic);
    if (version != kModelVersion) return std::unexpected(LoadError::bad_version);
    if (dim == 0 || vocab == 0) return std::unexpected(LoadError::bad_layout);

    Model m;
    m.dim_ = dim;
    m.vocab_ = vocab;

    // Tokenizer section: u32 n_search, SearchEntry[n_search], records, pad.
    const auto tok_blob = r.take(tok_size);
    if (!tok_blob) return std::unexpected(tok_blob.error());
    ByteReader tr{*tok_blob};
    uint32_t n_search = 0;
    if (!tr.read(n_search)) return std::unexpected(LoadError::truncated);
    const auto entry_bytes = tr.take(sizeof(SearchEntry) * static_cast<size_t>(n_search));
    if (!entry_bytes) return std::unexpected(entry_bytes.error());
    // Aligned: blob base is 4-byte aligned, header is 32 bytes, n_search is 4.
    const std::span entries{reinterpret_cast<const SearchEntry*>(entry_bytes->data()),
                            n_search};
    if (!m.tok_.init(tr.rest(), entries, vocab, unk_id, min_score, max_piece_len)) {
      return std::unexpected(LoadError::bad_layout);
    }

    auto table = make_table<QType::f32, QType::int8, QType::int4>(static_cast<QType>(qtype),
                                                                  r.rest(), vocab, dim);
    if (!table) return std::unexpected(table.error());
    m.table_ = *std::move(table);
    return m;
  }

  [[nodiscard]] static Result<Model> load(const uint8_t* data, size_t size) noexcept {
    return load(std::span{reinterpret_cast<const std::byte*>(data), size});
  }

  [[nodiscard]] static Result<Model> load_file(const char* path) {
    return read_file(path).and_then([](std::vector<std::byte> data) -> Result<Model> {
      auto m = load(data);
      // Table views point into `data`; moving the vector keeps them valid.
      if (m) m->owned_ = std::move(data);
      return m;
    });
  }

  // Embed text into out (size dim()): sum of weight-folded token rows,
  // L2-normalized. Returns the number of tokens consumed.
  size_t embed(std::string_view text, std::span<float> out) const {
    return pool(tok_.encode(text), out);
  }

  // No-heap variant: caller provides all scratch (see Tokenizer::encode_into
  // for the capacity requirements; ids_scratch ≥ normalized text length).
  size_t embed(std::string_view text, std::span<float> out, EncodeBuffers buf,
               std::span<uint32_t> ids_scratch) const noexcept {
    const auto count = tok_.encode_into(text, buf, ids_scratch);
    return pool(ids_scratch.first(count.value_or(0)), out);
  }

  [[nodiscard]] constexpr uint32_t dim() const noexcept { return dim_; }
  [[nodiscard]] constexpr uint32_t vocab_size() const noexcept { return vocab_; }
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return tok_; }

  Model(Model&&) = default;
  Model& operator=(Model&&) = default;
  Model(const Model&) = delete;  // views would alias the moved-from owned_
  Model& operator=(const Model&) = delete;

 private:
  Model() = default;

  size_t pool(std::span<const uint32_t> ids, std::span<float> out) const noexcept {
    for (float& x : out) x = 0.0f;
    std::visit(
        [&](const auto& table) {
          for (const uint32_t id : ids) {
            if (id < vocab_) table.accumulate(id, out);
          }
        },
        table_);
    l2_normalize(out);
    return ids.size();
  }

  Tokenizer tok_;
  Table table_{RowMatrix<QType::f32>{}};
  uint32_t dim_ = 0;
  uint32_t vocab_ = 0;
  std::vector<std::byte> owned_;
};

}  // namespace demb
