// Self-test: the no-heap encode/embed path must agree exactly with the heap
// path, using only static buffers — the way an MCU would provision them.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <vector>

#include "distillembed/model.hpp"

namespace {

constexpr std::string_view kTexts[] = {
    "the quick brown fox",
    "sensor reading temperature  humidity",
    "  battery level low ",
    "unknown-token 12345 !!",
    "unicode: h\xC3\xA9llo w\xC3\xB6rld \xE2\x98\x82",
    "a",
    "",
    "   ",
};

constexpr size_t kMaxText = 256;
constexpr size_t kCap = demb::Tokenizer::normalized_capacity(kMaxText);
constexpr uint32_t kMaxDim = 512;

// Static provisioning, zero heap in the encode/embed calls below.
char g_norm[kCap];
float g_best[kCap + 1];
int32_t g_back_id[kCap + 1];
size_t g_back_from[kCap + 1];
uint32_t g_ids[kCap];
float g_vec_noheap[kMaxDim];

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.dem>\n", argv[0]);
    return 2;
  }
  const auto model = demb::Model::load_file(argv[1]);
  if (!model) {
    std::fprintf(stderr, "FAIL load: %s\n", demb::to_string(model.error()));
    return 1;
  }
  if (model->dim() > kMaxDim) {
    std::fprintf(stderr, "FAIL dim %u exceeds selftest buffer\n", model->dim());
    return 1;
  }
  const demb::EncodeBuffers buffers{
      .norm = g_norm, .best = g_best, .back_id = g_back_id, .back_from = g_back_from};

  int checked = 0;
  for (const auto text : kTexts) {
    const auto heap_ids = model->tokenizer().encode(text);
    const auto count = model->tokenizer().encode_into(text, buffers, g_ids);
    if (!count) {
      std::fprintf(stderr, "FAIL encode_into rejected buffers for %zu-byte text\n", text.size());
      return 1;
    }
    if (*count != heap_ids.size() ||
        !std::equal(heap_ids.begin(), heap_ids.end(), g_ids)) {
      std::fprintf(stderr, "FAIL id mismatch on \"%.*s\" (heap %zu vs noheap %zu tokens)\n",
                   static_cast<int>(text.size()), text.data(), heap_ids.size(), *count);
      return 1;
    }

    std::vector<float> vec_heap(model->dim());
    model->embed(text, vec_heap);
    const std::span vec_noheap{g_vec_noheap, model->dim()};
    model->embed(text, vec_noheap, buffers, g_ids);
    for (uint32_t d = 0; d < model->dim(); ++d) {
      if (std::fabs(vec_heap[d] - vec_noheap[d]) > 1e-6f) {
        std::fprintf(stderr, "FAIL embed mismatch on \"%.*s\" dim %u\n",
                     static_cast<int>(text.size()), text.data(), d);
        return 1;
      }
    }
    ++checked;
  }
  std::printf("ok %d\n", checked);
  return 0;
}
