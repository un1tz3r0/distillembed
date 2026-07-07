// Embed a text with a .dem model and print tokens + vector.
// Output format matches `distillembed embed` for parity checks.
#include <cstdio>
#include <cstring>
#include <vector>

#include "distillembed/model.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.dem> <text> [--full]\n", argv[0]);
    return 2;
  }
  const auto model = demb::Model::load_file(argv[1]);
  if (!model) {
    std::fprintf(stderr, "error: %s: %s\n", argv[1], demb::to_string(model.error()));
    return 1;
  }
  const std::string_view text = argv[2];
  const bool full = argc > 3 && std::strcmp(argv[3], "--full") == 0;

  const auto ids = model->tokenizer().encode(text);
  std::vector<float> vec(model->dim());
  model->embed(text, vec);

  std::printf("tokens: %zu\nids:", ids.size());
  for (const auto id : ids) std::printf(" %u", id);
  std::printf("\ndim: %u\nvec:", model->dim());
  const size_t shown = full ? vec.size() : std::min<size_t>(8, vec.size());
  for (size_t d = 0; d < shown; ++d) std::printf(" %.6f", vec[d]);
  std::printf("\n");
  return 0;
}
