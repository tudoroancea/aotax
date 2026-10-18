#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kernel.h"
#include "layout.h"

static std::vector<double> load(const char* name, std::size_t count) {
  std::string path = std::string(INPUTS_DIR) + "/" + name;
  std::vector<double> values(count);
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(values.data(), sizeof(double), count, f) != count) {
    std::fprintf(stderr, "could not read %zu doubles from %s\n", count, path.c_str());
    std::exit(1);
  }
  std::fclose(f);
  return values;
}

static std::vector<double> g_out(NNZ);

static bool check() {
  std::vector<double> reference = load("reference.bin", std::size_t(NROWS) * NCOLS), got(std::size_t(NROWS) * NCOLS, 0.0);
  for (int i = 0; i < NNZ; ++i) {
    if (ROWS[i] < 0 || ROWS[i] >= NROWS || COLS[i] < 0 || COLS[i] >= NCOLS || (LOWER_TRIANGLE && ROWS[i] < COLS[i])) {
      std::fprintf(stderr, "output %d at (%d, %d) has an invalid coordinate\n", i, ROWS[i], COLS[i]);
      return false;
    }
    double want = reference[std::size_t(ROWS[i]) * NCOLS + COLS[i]];
    if (!(g_out[i] == want || std::fabs(g_out[i] - want) <= 1e-9 + 1e-9 * std::fabs(want))) {
      std::fprintf(stderr, "output %d at (%d, %d) = %.17g, reference %.17g\n", i, ROWS[i], COLS[i], g_out[i], want);
      return false;
    }
    got[std::size_t(ROWS[i]) * NCOLS + COLS[i]] = g_out[i];
  }
  for (int row = 0; row < NROWS; ++row) {
    for (int col = 0; col < NCOLS && (!LOWER_TRIANGLE || col <= row); ++col) {
      double have = got[std::size_t(row) * NCOLS + col], want = reference[std::size_t(row) * NCOLS + col];
      if (!(have == want || std::fabs(have - want) <= 1e-9 + 1e-9 * std::fabs(want))) {
        std::fprintf(stderr, "matrix[%d, %d] = %.17g, reference %.17g\n", row, col, have, want);
        return false;
      }
    }
  }
  return true;
}

static void BM_Kernel(benchmark::State& state) {
  for (auto _ : state) {
    kernel_evaluate(g_out.data());
    benchmark::DoNotOptimize(g_out.data());
  }
}
BENCHMARK(BM_Kernel);

int main(int argc, char** argv) {
  std::vector<std::vector<double>> inputs;
  std::vector<const double*> pointers;
  for (int i = 0; i < NINPUTS; ++i) inputs.push_back(load(INPUT_NAMES[i], INPUT_SIZES[i]));
  for (const auto& input : inputs) pointers.push_back(input.data());
  kernel_setup(pointers.data());
  kernel_evaluate(g_out.data());
  if (!check()) return 2;
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
