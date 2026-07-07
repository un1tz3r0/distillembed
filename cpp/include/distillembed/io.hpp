// Binary-blob reading primitives shared by the .dem / .demc loaders.
// No exceptions, no I/O here: errors travel as std::expected<_, LoadError>,
// which keeps the headers usable on -fno-exceptions embedded toolchains.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <type_traits>

namespace demb {

static_assert(std::endian::native == std::endian::little,
              "distillembed blobs are little-endian; add byte-swapping for BE targets");

enum class LoadError : uint8_t {
  io,          // file could not be read
  bad_magic,   // not a .dem / .demc blob
  bad_version, // format version mismatch
  bad_qtype,   // quantization type not supported by this table kind
  truncated,   // blob shorter than its header claims
  bad_layout,  // internally inconsistent header fields
};

[[nodiscard]] constexpr const char* to_string(LoadError e) noexcept {
  switch (e) {
    case LoadError::io: return "io error";
    case LoadError::bad_magic: return "bad magic";
    case LoadError::bad_version: return "unsupported format version";
    case LoadError::bad_qtype: return "unsupported quantization type";
    case LoadError::truncated: return "truncated blob";
    case LoadError::bad_layout: return "inconsistent header";
  }
  return "unknown";
}

template <class T>
using Result = std::expected<T, LoadError>;

// Compile-time FOURCC: the magic constants live in the type system, not in
// runtime memcmp against string literals.
consteval uint32_t fourcc(const char (&s)[5]) {
  return static_cast<uint32_t>(s[0]) | static_cast<uint32_t>(s[1]) << 8 |
         static_cast<uint32_t>(s[2]) << 16 | static_cast<uint32_t>(s[3]) << 24;
}

inline constexpr uint32_t kModelMagic = fourcc("DEMB");
inline constexpr uint32_t kCorpusMagic = fourcc("DEMC");
inline constexpr uint16_t kModelVersion = 2;   // v2: surface-sorted tokenizer index
inline constexpr uint16_t kCorpusVersion = 1;

// Bounds-checked cursor over an immutable byte span.
class ByteReader {
 public:
  explicit constexpr ByteReader(std::span<const std::byte> data) noexcept : data_(data) {}

  template <class T>
    requires std::is_trivially_copyable_v<T>
  [[nodiscard]] bool read(T& out) noexcept {
    if (sizeof(T) > remaining()) return false;
    std::memcpy(&out, data_.data() + off_, sizeof(T));
    off_ += sizeof(T);
    return true;
  }

  // Read a whole header in one call; short-circuits on the first failure.
  template <class... Ts>
  [[nodiscard]] bool read_all(Ts&... fields) noexcept {
    return (read(fields) && ...);
  }

  [[nodiscard]] Result<std::span<const std::byte>> take(size_t n) noexcept {
    if (n > remaining()) return std::unexpected(LoadError::truncated);
    const auto out = data_.subspan(off_, n);
    off_ += n;
    return out;
  }

  [[nodiscard]] bool seek(size_t absolute) noexcept {
    if (absolute > data_.size()) return false;
    off_ = absolute;
    return true;
  }

  [[nodiscard]] constexpr size_t offset() const noexcept { return off_; }
  [[nodiscard]] constexpr size_t remaining() const noexcept { return data_.size() - off_; }
  [[nodiscard]] constexpr std::span<const std::byte> rest() const noexcept {
    return data_.subspan(off_);
  }

 private:
  std::span<const std::byte> data_;
  size_t off_ = 0;
};

}  // namespace demb
