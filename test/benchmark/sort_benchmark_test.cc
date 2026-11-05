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

#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "absl/base/attributes.h"
#include <mujoco/mujoco.h>
#include "src/engine/engine_sort.h"

namespace mujoco {
namespace {

// A struct with data to be sorted.
struct Sortable {
  mjtNum value;
  int id1;
  int id2;
};

// Comparison function for Sortable struct.
int CompareSortable(const Sortable* a, const Sortable* b, void* context) {
  if (a->value < b->value) {
    return -1;
  } else if (a->value == b->value) {
    return 0;
  } else {
    return 1;
  }
}

// Instantiate the sort function.
mjSORT(SortSortable, Sortable, CompareSortable);

// Generate data for sorting benchmarks.
std::vector<Sortable> GenerateData(int n, double unsorted_fraction) {
  std::vector<Sortable> data(n);
  for (int i = 0; i < n; ++i) {
    data[i] = {static_cast<mjtNum>(i), i, -i};
  }

  if (unsorted_fraction > 0.0) {
    std::mt19937 g(12345);
    if (unsorted_fraction >= 1.0) {
      std::shuffle(data.begin(), data.end(), g);
    } else {
      int num_to_shuffle = n * unsorted_fraction;
      for (int i = 0; i < num_to_shuffle; ++i) {
        int j = std::uniform_int_distribution<int>(i, n - 1)(g);
        std::swap(data[i], data[j]);
      }
    }
  }
  return data;
}

// Run a sorting benchmark with the given size and unsorted fraction.
ABSL_ATTRIBUTE_NO_TAIL_CALL static void SortBenchmark(
    benchmark::State& state, int n, double unsorted_fraction) {
  auto data = GenerateData(n, unsorted_fraction);
  std::vector<Sortable> buf(n);
  std::vector<Sortable> copy = data;

  for (auto s : state) {
    state.PauseTiming();
    copy = data;
    state.ResumeTiming();
    SortSortable(copy.data(), buf.data(), n, nullptr);
  }

  state.SetItemsProcessed(state.iterations() * n);
}

// Define benchmarks.
void BM_Sort_1k_AlmostSorted(benchmark::State& state) {
  SortBenchmark(state, 1000, 0.05);
}
BENCHMARK(BM_Sort_1k_AlmostSorted);

void BM_Sort_1k_Random(benchmark::State& state) {
  SortBenchmark(state, 1000, 1.0);
}
BENCHMARK(BM_Sort_1k_Random);

void BM_Sort_100k_AlmostSorted(benchmark::State& state) {
  SortBenchmark(state, 100000, 0.05);
}
BENCHMARK(BM_Sort_100k_AlmostSorted);

void BM_Sort_100k_Random(benchmark::State& state) {
  SortBenchmark(state, 100000, 1.0);
}
BENCHMARK(BM_Sort_100k_Random);

// Run a sorting benchmark with std::stable_sort
ABSL_ATTRIBUTE_NO_TAIL_CALL static void StdSortBenchmark(
    benchmark::State& state, int n, double unsorted_fraction) {
  auto data = GenerateData(n, unsorted_fraction);
  std::vector<Sortable> copy = data;

  for (auto s : state) {
    state.PauseTiming();
    copy = data;
    state.ResumeTiming();
    std::stable_sort(
        copy.begin(), copy.end(),
        [](const Sortable& a, const Sortable& b) { return a.value < b.value; });
  }

  state.SetItemsProcessed(state.iterations() * n);
}

void BM_StdSort_1k_AlmostSorted(benchmark::State& state) {
  StdSortBenchmark(state, 1000, 0.05);
}
BENCHMARK(BM_StdSort_1k_AlmostSorted);

void BM_StdSort_1k_Random(benchmark::State& state) {
  StdSortBenchmark(state, 1000, 1.0);
}
BENCHMARK(BM_StdSort_1k_Random);

void BM_StdSort_100k_AlmostSorted(benchmark::State& state) {
  StdSortBenchmark(state, 100000, 0.05);
}
BENCHMARK(BM_StdSort_100k_AlmostSorted);

void BM_StdSort_100k_Random(benchmark::State& state) {
  StdSortBenchmark(state, 100000, 1.0);
}
BENCHMARK(BM_StdSort_100k_Random);

}  // namespace
}  // namespace mujoco
