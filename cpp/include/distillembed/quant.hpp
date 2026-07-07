// Quantization schemes as compile-time traits. Each scheme gets a
// RowMatrix<Q> specialization-by-constraint: one class template whose
// operations are enabled per-scheme with `requires` clauses, so an
// instantiation only carries the kernels that make sense for it.
// Tables are non-owning views — flash-resident blobs on MCU targets.
#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <variant>

#include "io.hpp"

namespace demb {

enum class QType : uint8_t {
  f32 = 0,
  int8 = 1,     // per-row symmetric scale
  int4 = 2,     // per-row scale, two codes per byte (even dim = low nibble)
  binary = 3,   // 1 bit/dim sign quantization, Hamming scoring (corpus only)
  rescore = 4,  // corpus composite: int8 table + binary prefilter section
};

template <QType Q>
struct quant_traits;  // primary template intentionally undefined

template <>
struct quant_traits<QType::f32> {
  static constexpr bool has_scales = false;
  static constexpr size_t row_bytes(uint32_t dim) noexcept { return sizeof(float) * dim; }
};

template <>
struct quant_traits<QType::int8> {
  static constexpr bool has_scales = true;
  static constexpr size_t row_bytes(uint32_t dim) noexcept { return dim; }
};

template <>
struct quant_traits<QType::int4> {
  static constexpr bool has_scales = true;
  static constexpr size_t row_bytes(uint32_t dim) noexcept { return (dim + 1u) / 2u; }
};

template <>
struct quant_traits<QType::binary> {
  static constexpr bool has_scales = false;
  static constexpr size_t row_bytes(uint32_t dim) noexcept { return (dim + 7u) / 8u; }
};

// Sign-extend a 4-bit two's-complement nibble.
[[nodiscard]] constexpr int unpack_nibble(uint8_t byte, size_t d) noexcept {
  const int nib = (byte >> ((d & 1u) * 4u)) & 0xF;
  return (nib ^ 8) - 8;
}

template <QType Q>
concept Quantized = requires(uint32_t dim) {
  { quant_traits<Q>::row_bytes(dim) } -> std::convertible_to<size_t>;
  { quant_traits<Q>::has_scales } -> std::convertible_to<bool>;
};

template <QType Q>
concept DenseQuantized = Quantized<Q> && (Q != QType::binary);

// Non-owning view of `rows` quantized vectors of `dim` dimensions.
template <QType Q>
  requires Quantized<Q>
class RowMatrix {
 public:
  static constexpr QType qtype = Q;

  [[nodiscard]] static constexpr size_t bytes_needed(uint32_t rows, uint32_t dim) noexcept {
    return scale_bytes(rows) + static_cast<size_t>(rows) * quant_traits<Q>::row_bytes(dim);
  }

  // Blob layout: [f32 scales[rows] if has_scales][codes rows*row_bytes].
  // The writer 4-byte-aligns the blob, so the scale array may be aliased.
  [[nodiscard]] static Result<RowMatrix> from_blob(std::span<const std::byte> blob,
                                                   uint32_t rows, uint32_t dim) noexcept {
    if (blob.size() < bytes_needed(rows, dim)) return std::unexpected(LoadError::truncated);
    RowMatrix m;
    m.rows_ = rows;
    m.dim_ = dim;
    if constexpr (quant_traits<Q>::has_scales) {
      m.scales_ = reinterpret_cast<const float*>(blob.data());
    }
    m.codes_ = blob.data() + scale_bytes(rows);
    return m;
  }

  // out += row  (embedding pooling)
  void accumulate(uint32_t row, std::span<float> out) const noexcept
    requires DenseQuantized<Q>
  {
    if constexpr (Q == QType::f32) {
      const float* r = row_as<float>(row);
      for (size_t d = 0; d < out.size(); ++d) out[d] += r[d];
    } else if constexpr (Q == QType::int8) {
      const int8_t* r = row_as<int8_t>(row);
      const float scale = scales_[row];
      for (size_t d = 0; d < out.size(); ++d) out[d] += scale * static_cast<float>(r[d]);
    } else {  // int4
      const uint8_t* r = row_as<uint8_t>(row);
      const float scale = scales_[row];
      for (size_t d = 0; d < out.size(); ++d) {
        out[d] += scale * static_cast<float>(unpack_nibble(r[d >> 1], d));
      }
    }
  }

  // <row, query>  (≈ cosine when both sides are L2-normalized pre-quantization)
  [[nodiscard]] float dot(uint32_t row, std::span<const float> query) const noexcept
    requires DenseQuantized<Q>
  {
    double acc = 0.0;
    if constexpr (Q == QType::f32) {
      const float* r = row_as<float>(row);
      for (size_t d = 0; d < query.size(); ++d) acc += static_cast<double>(query[d]) * r[d];
    } else if constexpr (Q == QType::int8) {
      const int8_t* r = row_as<int8_t>(row);
      for (size_t d = 0; d < query.size(); ++d) acc += static_cast<double>(query[d]) * r[d];
      acc *= scales_[row];
    } else {  // int4
      const uint8_t* r = row_as<uint8_t>(row);
      for (size_t d = 0; d < query.size(); ++d) {
        acc += static_cast<double>(query[d]) * unpack_nibble(r[d >> 1], d);
      }
      acc *= scales_[row];
    }
    return static_cast<float>(acc);
  }

  [[nodiscard]] uint32_t hamming(uint32_t row, std::span<const std::byte> query_bits) const noexcept
    requires (Q == QType::binary)
  {
    const std::byte* r = codes_ + static_cast<size_t>(row) * quant_traits<Q>::row_bytes(dim_);
    uint32_t h = 0;
    for (size_t j = 0; j < query_bits.size(); ++j) {
      h += static_cast<uint32_t>(std::popcount(std::to_integer<unsigned>(r[j] ^ query_bits[j])));
    }
    return h;
  }

  [[nodiscard]] constexpr uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] constexpr uint32_t dim() const noexcept { return dim_; }

 private:
  static constexpr size_t scale_bytes(uint32_t rows) noexcept {
    return quant_traits<Q>::has_scales ? sizeof(float) * static_cast<size_t>(rows) : 0u;
  }

  template <class T>
  [[nodiscard]] const T* row_as(uint32_t row) const noexcept {
    return reinterpret_cast<const T*>(codes_ +
                                      static_cast<size_t>(row) * quant_traits<Q>::row_bytes(dim_));
  }

  const std::byte* codes_ = nullptr;
  const float* scales_ = nullptr;
  uint32_t rows_ = 0;
  uint32_t dim_ = 0;
};

// Runtime qtype -> the matching std::variant alternative, driven by a fold
// over the Allowed pack. Adding a scheme to a table kind is: append it to the
// pack at the call site — no switch to maintain.
template <QType... Allowed>
[[nodiscard]] inline Result<std::variant<RowMatrix<Allowed>...>> make_table(
    QType q, std::span<const std::byte> blob, uint32_t rows, uint32_t dim) noexcept {
  using Table = std::variant<RowMatrix<Allowed>...>;
  Result<Table> out = std::unexpected(LoadError::bad_qtype);
  (void)((q == Allowed
              ? (out = RowMatrix<Allowed>::from_blob(blob, rows, dim)
                           .transform([](auto m) -> Table { return m; }),
                 true)
              : false) ||
         ...);
  return out;
}

}  // namespace demb
