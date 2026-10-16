// The harness every backend shares, mirroring Scaly's benchmark cells: read the inputs, check the
// lower triangle against the dense reference, then time one evaluation.
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

static std::vector<double> g_z, g_p, g_lam_g, g_out(NNZ);

// Every output coordinate must lie in the lower triangle, and every lower entry must match the
// reference to 1e-9 absolute plus 1e-9 relative; an entry the backend does not store must be zero.
static bool check() {
  std::vector<double> reference = load("hessian.bin", std::size_t(NZ) * NZ), got(std::size_t(NZ) * NZ, 0.0);
  for (int i = 0; i < NNZ; ++i) {
    if (ROWS[i] < COLS[i]) {
      std::fprintf(stderr, "output %d at (%d, %d) is above the diagonal\n", i, ROWS[i], COLS[i]);
      return false;
    }
    got[std::size_t(ROWS[i]) * NZ + COLS[i]] = g_out[i];
  }
  for (int row = 0; row < NZ; ++row) {
    for (int col = 0; col <= row; ++col) {
      double have = got[std::size_t(row) * NZ + col], want = reference[std::size_t(row) * NZ + col];
      if (!(have == want || std::fabs(have - want) <= 1e-9 + 1e-9 * std::fabs(want))) {
        std::fprintf(stderr, "H[%d, %d] = %.17g, reference %.17g\n", row, col, have, want);
        return false;
      }
    }
  }
  return true;
}

static void BM_Hessian(benchmark::State& state) {
  for (auto _ : state) {
    kernel_evaluate(g_out.data());
    benchmark::DoNotOptimize(g_out.data());
  }
}
BENCHMARK(BM_Hessian);

int main(int argc, char** argv) {
  g_z = load("z.bin", NZ);
  g_p = load("p.bin", NP);
  g_lam_g = load("lam_g.bin", NLAM);
  kernel_setup(g_z.data(), g_p.data(), g_lam_g.data());
  kernel_evaluate(g_out.data());
  if (!check()) return 2;
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
