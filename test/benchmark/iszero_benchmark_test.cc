// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <benchmark/benchmark.h>
#include "absl/base/attributes.h"
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_misc.h"

namespace mujoco {
namespace {

// Run benchmark for the isZero functions, for mjtNum directly or bytewise.
ABSL_ATTRIBUTE_NO_TAIL_CALL static void IzZeroBenchmark(benchmark::State& state,
                                                        int n, bool bytewise) {
  std::vector<mjtNum> data(n, 0);
  data[n - 1] = 1;

  if (bytewise) {
    for (auto s : state) {
      int iszero =
          mju_isZeroByte((const unsigned char*)data.data(), n * sizeof(mjtNum));
      benchmark::DoNotOptimize(iszero);
    }
  } else {
    for (auto s : state) {
      int iszero = mju_isZero(data.data(), n);
      benchmark::DoNotOptimize(iszero);
    }
  }

  state.SetItemsProcessed(state.iterations() * n);
}

// Define benchmarks.
void BM_isZeroByte_10(benchmark::State& state) {
  IzZeroBenchmark(state, 10, true);
}
BENCHMARK(BM_isZeroByte_10);

void BM_isZeroByte_1e2(benchmark::State& state) {
  IzZeroBenchmark(state, 100, true);
}
BENCHMARK(BM_isZeroByte_1e2);

void BM_isZeroByte_1e5(benchmark::State& state) {
  IzZeroBenchmark(state, 100000, true);
}
BENCHMARK(BM_isZeroByte_1e5);

void BM_isZero_10(benchmark::State& state) {
  IzZeroBenchmark(state, 10, false);
}
BENCHMARK(BM_isZero_10);

void BM_isZero_1e2(benchmark::State& state) {
  IzZeroBenchmark(state, 100, false);
}
BENCHMARK(BM_isZero_1e2);

void BM_isZero_1e5(benchmark::State& state) {
  IzZeroBenchmark(state, 100000, false);
}
BENCHMARK(BM_isZero_1e5);

}  // namespace
}  // namespace mujoco
