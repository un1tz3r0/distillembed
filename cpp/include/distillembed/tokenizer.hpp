// SentencePiece-compatible unigram tokenizer: Viterbi decode with byte fallback.
// Only matches models trained via distillembed.tokenizer.train_tokenizer, i.e.
// model_type=unigram, normalization_rule_name=identity, add_dummy_prefix=true,
// remove_extra_whitespaces=true, byte_fallback=true (see DESIGN.md).
//
// Zero-RAM design (.dem v2): piece lookup binary-searches a surface-sorted
// index precomputed by the exporter, read in place from the (flash-resident)
// blob. The only RAM this class owns is the 1KB byte-fallback table.
// encode_into() is the no-heap path: the caller provides all scratch.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace demb {

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's whitespace marker.
inline constexpr std::string_view kWhitespacePiece = "\xE2\x96\x81";

enum class PieceType : uint8_t { normal = 0, unk = 1, control = 2, byte = 3 };

// One entry of the surface-sorted search index (wire format, .dem v2).
struct SearchEntry {
  uint32_t rec_offset;
  uint32_t piece_id;
};
static_assert(sizeof(SearchEntry) == 8 && std::is_trivially_copyable_v<SearchEntry>);

// Caller-provided scratch for the no-heap encode path; see encode_into().
struct EncodeBuffers {
  std::span<char> norm;
  std::span<float> best;
  std::span<int32_t> back_id;
  std::span<size_t> back_from;
};

class Tokenizer {
 public:
  // Worst case: 3-byte dummy prefix plus 3 bytes per input char.
  [[nodiscard]] static constexpr size_t normalized_capacity(size_t text_len) noexcept {
    return 3 * text_len + 3;
  }

  // Zero-copy init over the .dem tokenizer section (records in id order,
  // entries surface-sorted). One sequential scan fills the byte-piece table
  // and validates every record and index entry.
  [[nodiscard]] bool init(std::span<const std::byte> records,
                          std::span<const SearchEntry> entries, uint32_t vocab,
                          uint32_t unk_id, float min_score, uint32_t max_piece_len) noexcept {
    records_ = records;
    entries_ = entries;
    unk_id_ = unk_id;
    min_score_ = min_score;
    max_surface_ = max_piece_len;
    byte_ids_.fill(-1);
    size_t off = 0;
    for (uint32_t id = 0; id < vocab; ++id) {
      if (off + kRecordHeader > records_.size()) return false;
      const PieceRef rec = piece_at(static_cast<uint32_t>(off));
      if (off + kRecordHeader + rec.surface.size() > records_.size()) return false;
      if (rec.type == PieceType::byte) {
        if (const int b = parse_byte_piece(rec.surface); b >= 0) {
          byte_ids_[static_cast<size_t>(b)] = static_cast<int32_t>(id);
        }
      }
      off += kRecordHeader + rec.surface.size();
    }
    const auto valid_offset = [&](const SearchEntry& e) { return e.rec_offset < off; };
    return std::ranges::all_of(entries_, valid_offset);
  }

  struct Match {
    uint32_t id;
    float score;
  };

  // Binary search over the flash-resident surface-sorted index.
  [[nodiscard]] std::optional<Match> find(std::string_view surface) const noexcept {
    const auto surface_of = [&](const SearchEntry& e) { return piece_at(e.rec_offset).surface; };
    const auto it = std::ranges::lower_bound(entries_, surface, std::ranges::less{}, surface_of);
    if (it == entries_.end()) return std::nullopt;
    const PieceRef rec = piece_at(it->rec_offset);
    if (rec.surface != surface) return std::nullopt;
    return Match{.id = it->piece_id, .score = rec.score};
  }

  // No-heap encode. Capacity requirements:
  //   buf.norm                          ≥ normalized_capacity(text.size())
  //   buf.best / back_id / back_from    ≥ normalized length + 1
  //   out_ids                           ≥ normalized length
  // Returns the token count, or nullopt if any buffer is too small.
  [[nodiscard]] std::optional<size_t> encode_into(std::string_view text, EncodeBuffers buf,
                                                  std::span<uint32_t> out_ids) const noexcept {
    const auto norm_len = normalize_into(text, buf.norm);
    if (!norm_len) return std::nullopt;
    const size_t n = *norm_len;
    if (n == 0) return 0;
    if (buf.best.size() < n + 1 || buf.back_id.size() < n + 1 ||
        buf.back_from.size() < n + 1 || out_ids.size() < n) {
      return std::nullopt;
    }
    const std::string_view s{buf.norm.data(), n};

    constexpr float kNegInf = -1e30f;
    std::ranges::fill(buf.best.first(n + 1), kNegInf);
    std::ranges::fill(buf.back_id.first(n + 1), -1);
    buf.best[0] = 0.0f;

    // SentencePiece semantics: an unknown *character* is one lattice node with
    // score min_score - kUnkPenalty(10); byte-piece scores are never edge
    // weights. The unk surface is expanded to byte pieces during backtrack.
    const float unk_score = min_score_ - 10.0f;

    const auto relax = [&](size_t from, size_t to, int32_t id, float score) {
      if (const float cand = buf.best[from] + score; cand > buf.best[to]) {
        buf.best[to] = cand;
        buf.back_id[to] = id;
        buf.back_from[to] = from;
      }
    };

    for (size_t i = 0; i < n; ++i) {
      if (buf.best[i] <= kNegInf * 0.5f) continue;
      const size_t max_len = std::min<size_t>(max_surface_, n - i);
      for (size_t len = 1; len <= max_len; ++len) {
        if (const auto match = find(s.substr(i, len))) {
          relax(i, i + len, static_cast<int32_t>(match->id), match->score);
        }
      }
      const size_t char_len = std::min(utf8_char_len(static_cast<uint8_t>(s[i])), n - i);
      relax(i, i + char_len, kFallbackId, unk_score);
    }

    size_t count = 0;
    for (size_t pos = n; pos > 0; pos = buf.back_from[pos]) {
      if (buf.back_id[pos] == kFallbackId) {
        // Reversed here; the final reverse restores byte order.
        for (size_t b = pos; b-- > buf.back_from[pos];) {
          const int32_t byte_id = byte_ids_[static_cast<uint8_t>(s[b])];
          out_ids[count++] = byte_id >= 0 ? static_cast<uint32_t>(byte_id) : unk_id_;
        }
      } else if (buf.back_id[pos] < 0) {
        return std::nullopt;  // unreachable: the fallback edge spans every gap
      } else {
        out_ids[count++] = static_cast<uint32_t>(buf.back_id[pos]);
      }
    }
    std::ranges::reverse(out_ids.first(count));
    return count;
  }

  // Convenience heap path (desktop tools, tests).
  [[nodiscard]] std::vector<uint32_t> encode(std::string_view text) const {
    const size_t cap = normalized_capacity(text.size());
    std::vector<char> norm(cap);
    std::vector<float> best(cap + 1);
    std::vector<int32_t> back_id(cap + 1);
    std::vector<size_t> back_from(cap + 1);
    std::vector<uint32_t> ids(cap);
    const auto count = encode_into(
        text, {.norm = norm, .best = best, .back_id = back_id, .back_from = back_from}, ids);
    ids.resize(count.value_or(0));
    return ids;
  }

  // Mirrors the trained SentencePiece normalizer: collapse whitespace runs to
  // one U+2581, prepend a dummy prefix, drop trailing whitespace. Returns the
  // normalized length, or nullopt if `out` is too small.
  [[nodiscard]] static std::optional<size_t> normalize_into(std::string_view in,
                                                            std::span<char> out) noexcept {
    size_t len = 0;
    bool pending_ws = true;  // leading state produces the dummy prefix
    for (const char c : in) {
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        pending_ws = true;
        continue;
      }
      if (pending_ws) {
        if (len + kWhitespacePiece.size() > out.size()) return std::nullopt;
        std::ranges::copy(kWhitespacePiece, out.data() + len);
        len += kWhitespacePiece.size();
        pending_ws = false;
      }
      if (len + 1 > out.size()) return std::nullopt;
      out[len++] = c;
    }
    return len;
  }

 private:
  static constexpr size_t kRecordHeader = 8;   // f32 score, u16 len, u8 type, u8 pad
  static constexpr int32_t kFallbackId = -2;   // unknown-character lattice node

  struct PieceRef {
    std::string_view surface;
    float score;
    PieceType type;
  };

  // Unchecked record accessor — every offset is validated by init().
  [[nodiscard]] PieceRef piece_at(uint32_t off) const noexcept {
    float score = 0.0f;
    uint16_t len = 0;
    uint8_t type = 0;
    std::memcpy(&score, records_.data() + off, sizeof(score));
    std::memcpy(&len, records_.data() + off + 4, sizeof(len));
    std::memcpy(&type, records_.data() + off + 6, sizeof(type));
    return {.surface = {reinterpret_cast<const char*>(records_.data()) + off + kRecordHeader,
                        len},
            .score = score,
            .type = static_cast<PieceType>(type)};
  }

  [[nodiscard]] static constexpr size_t utf8_char_len(uint8_t lead) noexcept {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;  // continuation/invalid byte: treat as a single unknown
  }

  [[nodiscard]] static constexpr int parse_byte_piece(std::string_view s) noexcept {
    // "<0xAB>" -> 0xAB
    constexpr auto hex = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      return -1;
    };
    if (s.size() != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return -1;
    const int hi = hex(s[3]);
    const int lo = hex(s[4]);
    return (hi < 0 || lo < 0) ? -1 : ((hi << 4) | lo);
  }

  std::span<const std::byte> records_;
  std::span<const SearchEntry> entries_;
  std::array<int32_t, 256> byte_ids_{};
  float min_score_ = 0.0f;
  size_t max_surface_ = 0;
  uint32_t unk_id_ = 0;
};

}  // namespace demb
