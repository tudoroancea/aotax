#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "generated.h"
#include "program_layout.h"

static generated_context* context;
static std::vector<void*> inputs, outputs;

static void* buffer(std::size_t bytes, const char* name) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, bytes ? bytes : 64)) std::abort();
  if (name) {
    std::string path = std::string(INPUTS_DIR) + "/" + name;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f || std::fread(p, 1, bytes, f) != bytes) {
      std::fprintf(stderr, "could not read %zu bytes from %s\n", bytes, path.c_str());
      std::exit(1);
    }
    std::fclose(f);
  }
  return p;
}

static void evaluate() {
  if (PROGRAM_CALL) std::abort();
}

template <typename T>
static bool check_leaf(int leaf, const void* reference) {
  const auto* got = static_cast<const T*>(outputs[leaf]);
  const auto* want = static_cast<const T*>(reference);
  for (std::size_t i = 0; i < OUTPUT_COUNTS[leaf]; ++i) {
    bool same;
    if constexpr (std::is_integral_v<T>) same = got[i] == want[i];
    else same = std::isfinite(got[i]) && std::isfinite(want[i]) &&
                std::fabs(got[i] - want[i]) <= ATOL[leaf] + RTOL[leaf] * std::fabs(want[i]);
    if (!same) {
      std::fprintf(stderr, "output leaf %d element %zu = %.17Lg, reference %.17Lg\n", leaf, i,
                   static_cast<long double>(got[i]), static_cast<long double>(want[i]));
      return false;
    }
  }
  return true;
}

static void BM_Program(benchmark::State& state) {
  for (auto _ : state) {
    evaluate();
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Program);

int main(int argc, char** argv) {
  for (int i = 0; i < NINPUTS; ++i) inputs.push_back(buffer(INPUT_BYTES[i], INPUT_NAMES[i]));
  for (int i = 0; i < NOUTPUTS; ++i) outputs.push_back(buffer(OUTPUT_BYTES[i], nullptr));
  if (generated_create(&context)) return 1;
  evaluate();
  for (int i = 0; i < NOUTPUTS; ++i) {
    void* reference = buffer(OUTPUT_BYTES[i], OUTPUT_NAMES[i]);
    bool ok = OUTPUT_TYPES[i] == 0 ? check_leaf<double>(i, reference) :
              OUTPUT_TYPES[i] == 1 ? check_leaf<int32_t>(i, reference) : check_leaf<int64_t>(i, reference);
    std::free(reference);
    if (!ok) return 2;
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  generated_destroy(context);
  for (void* p : inputs) std::free(p);
  for (void* p : outputs) std::free(p);
}
