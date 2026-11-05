// Copyright 2026 DeepMind Technologies Limited
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

// Benchmarks for sparse matrix operations.

#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "absl/base/attributes.h"
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_sparse.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

// ================================ Test Data ==================================
// Stores pre-computed sparse matrix inputs extracted from MuJoCo simulations.
// Each benchmark computes its own outputs (H, L, etc.) from these inputs.

struct SparseTestData {
  // Dimensions
  int nv;    // number of DoFs
  int nefc;  // number of constraint rows
  int nJ;    // nnz in J

  // J (Jacobian) - nefc x nv sparse
  std::vector<mjtNum> J;
  std::vector<int> J_rownnz, J_rowadr, J_colind, J_rowsuper;

  // J' (transpose)
  std::vector<mjtNum> JT;
  std::vector<int> JT_rownnz, JT_rowadr, JT_colind, JT_rowsuper;

  // D (diagonal weights for constraints)
  std::vector<mjtNum> D;

  // M structure (mass matrix, lower triangle)
  std::vector<int> M_rownnz, M_rowadr, M_colind;

  void Setup(const mjModel* m, mjData* d) {
    // initialize simulation state
    mj_resetDataKeyframe(m, d, 0);
    mj_step(m, d);
    mj_forward(m, d);
    nv = m->nv;
    nefc = d->nefc;
    nJ = d->nJ;

    // copy J
    J.assign(d->efc_J, d->efc_J + nJ);
    J_rownnz.assign(d->efc_J_rownnz, d->efc_J_rownnz + nefc);
    J_rowadr.assign(d->efc_J_rowadr, d->efc_J_rowadr + nefc);
    J_colind.assign(d->efc_J_colind, d->efc_J_colind + nJ);
    J_rowsuper.assign(d->efc_J_rowsuper, d->efc_J_rowsuper + nefc);

    // transpose J
    JT.assign(nJ, 0);
    JT_rownnz.assign(nv, 0);
    JT_rowadr.assign(nv, 0);
    JT_colind.assign(nJ, 0);
    JT_rowsuper.assign(nv, 0);
    mju_transposeSparse(JT.data(), J.data(), nefc, nv, JT_rownnz.data(),
                        JT_rowadr.data(), JT_colind.data(), JT_rowsuper.data(),
                        J_rownnz.data(), J_rowadr.data(), J_colind.data());

    // compute D corresponding to quadratic constraint states
    D.resize(nefc);
    for (int i = 0; i < nefc; i++) {
      if (d->efc_state[i] == mjCNSTRSTATE_QUADRATIC) {
        D[i] = d->efc_D[i];
      } else {
        D[i] = 0;
      }
    }

    // copy M structure
    M_rownnz.assign(m->M_rownnz, m->M_rownnz + nv);
    M_rowadr.assign(m->M_rowadr, m->M_rowadr + nv);
    int nM = M_rowadr[nv - 1] + M_rownnz[nv - 1];
    M_colind.assign(m->M_colind, m->M_colind + nM);
  }
};

// ================================ Model Sizes ================================

enum class Size { H2_100, H100 };

template <Size S>
const char* ModelPath() {
  if constexpr (S == Size::H2_100) {
    return "../test/benchmark/testdata/2humanoid100_chol.xml";
  } else {
    return "../test/benchmark/testdata/100_humanoids_chol.xml";
  }
}

template <Size S>
mjModel* GetModel() {
  static mjModel* m = LoadModelFromPath(ModelPath<S>());
  m->opt.jacobian = mjJAC_SPARSE;
  m->opt.solver = mjSOL_NEWTON;
  m->opt.disableflags |= mjDSBL_ISLAND;
  return m;
}

template <Size S>
SparseTestData& GetData() {
  static SparseTestData data;
  static bool initialized = false;
  if (!initialized) {
    mjModel* m = GetModel<S>();
    mjData* d = mj_makeData(m);
    data.Setup(m, d);
    mj_deleteData(d);
    initialized = true;
  }
  return data;
}

// ========================== Baseline Implementations =========================

// Baseline sqrMatTD (uncompressed layout, from old implementation)
void ABSL_ATTRIBUTE_NOINLINE mju_sqrMatTDSparse_baseline(
    mjtNum* res, const mjtNum* mat, const mjtNum* matT, const mjtNum* diag,
    int nr, int nc, int* res_rownnz, int* res_rowadr, int* res_colind,
    const int* rownnz, const int* rowadr, const int* colind,
    const int* rowsuper, const int* rownnzT, const int* rowadrT,
    const int* colindT, const int* rowsuperT, mjData* d) {
  mj_markStack(d);
  int* chain = mj_stackAllocInt(d, 2 * nc);
  mjtNum* buffer = mj_stackAllocNum(d, nc);

  for (int r = 0; r < nc; r++) {
    res_rowadr[r] = r * nc;
  }

  for (int r = 0; r < nc; r++) {
    if (rowsuperT && r > 0 && rowsuperT[r - 1] > 0) {
      res_rownnz[r] = res_rownnz[r - 1];
      memcpy(res_colind + res_rowadr[r], res_colind + res_rowadr[r - 1],
             res_rownnz[r] * sizeof(int));
      if (rownnzT[r]) {
        res_colind[res_rowadr[r] + res_rownnz[r]] = r;
        res_rownnz[r]++;
      }
    } else {
      int nchain = 0;
      int inew = 0, iold = nc;
      int lastadded = -1;
      for (int i = 0; i < rownnzT[r]; i++) {
        int c = colindT[rowadrT[r] + i];
        if (rowsuper && lastadded >= 0 &&
            (c - lastadded) <= rowsuper[lastadded]) {
          continue;
        } else {
          lastadded = c;
        }

        int adr = inew;
        inew = iold;
        iold = adr;

        int nnewchain = 0;
        adr = 0;
        int end = rowadr[c] + rownnz[c];
        for (int adr1 = rowadr[c]; adr1 < end; adr1++) {
          int col_mat = colind[adr1];
          while (adr < nchain && chain[iold + adr] < col_mat &&
                 chain[iold + adr] <= r) {
            chain[inew + nnewchain++] = chain[iold + adr++];
          }
          if (col_mat > r) {
            break;
          }
          if (adr < nchain && chain[iold + adr] == col_mat) {
            adr++;
          }
          chain[inew + nnewchain++] = col_mat;
        }

        while (adr < nchain && chain[iold + adr] <= r) {
          chain[inew + nnewchain++] = chain[iold + adr++];
        }
        nchain = nnewchain;
      }
      res_rownnz[r] = nchain;
      if (nchain) {
        memcpy(res_colind + res_rowadr[r], chain + inew, nchain * sizeof(int));
      }
    }
  }

  for (int r = 0; r < nc; r++) {
    int adr = res_rowadr[r];
    for (int i = 0; i < res_rownnz[r]; i++) {
      buffer[res_colind[adr + i]] = 0;
    }
    for (int i = 0; i < rownnzT[r]; i++) {
      int c = colindT[rowadrT[r] + i];
      mjtNum matTrc = matT[rowadrT[r] + i];
      if (diag) {
        matTrc *= diag[c];
      }

      int end = rowadr[c] + rownnz[c];
      for (int adr2 = rowadr[c]; adr2 < end; adr2++) {
        int adr1;
        if ((adr1 = colind[adr2]) > r) {
          break;
        }
        buffer[adr1] += matTrc * mat[adr2];
      }
    }
    adr = res_rowadr[r];
    for (int i = 0; i < res_rownnz[r]; i++) {
      res[adr + i] = buffer[res_colind[adr + i]];
    }
  }
  for (int r = 1; r < nc; r++) {
    int end = res_rowadr[r] + res_rownnz[r] - 1;
    for (int adr = res_rowadr[r]; adr < end; adr++) {
      int adr1 = res_rowadr[res_colind[adr]] + res_rownnz[res_colind[adr]]++;
      res[adr1] = res[adr];
      res_colind[adr1] = r;
    }
  }

  mj_freeStack(d);
}

// ========================== SqrMatTD Benchmarks ==============================

enum class SqrMatTDVariant {
  kBaseline,
  kRow,
  kCol,
  kSplitCol,
  kSymbolicOnly,
  kNumericOnly
};

template <Size S>
static void BM_sqrMatTD_impl(benchmark::State& state, SqrMatTDVariant variant) {
  SparseTestData& data = GetData<S>();
  mjModel* m = GetModel<S>();
  mjData* d = mj_makeData(m);

  int nv = data.nv;

  // nothing to benchmark if no constraints
  if (data.nefc == 0) {
    for (auto s : state) {
    }
    mj_deleteData(d);
    return;
  }

  // allocate H output (uncompressed for baseline, compressed for others)
  int max_nnz = (variant == SqrMatTDVariant::kBaseline) ? nv * nv : 0;
  std::vector<mjtNum> H;
  std::vector<int> H_rownnz(nv);
  std::vector<int> H_rowadr(nv);
  std::vector<int> H_colind;
  std::vector<int> diagind(nv);

  if (variant == SqrMatTDVariant::kBaseline) {
    H.resize(max_nnz);
    H_colind.resize(max_nnz);
  } else if (variant != SqrMatTDVariant::kRow) {
    // use symbolic to count nnz and fill pattern
    int nH = mju_sqrMatTDSparseSymbolic(
        H_rownnz.data(), H_rowadr.data(), nullptr, nullptr, data.nefc, nv,
        data.J_rownnz.data(), data.J_rowadr.data(), data.J_colind.data(),
        data.J_rowsuper.data(), data.JT_rownnz.data(), data.JT_rowadr.data(),
        data.JT_colind.data(), data.JT_rowsuper.data(), d);
    H.resize(nH);
    H_colind.resize(nH);
    if (variant == SqrMatTDVariant::kNumericOnly) {
      mju_sqrMatTDSparseSymbolic(
          H_rownnz.data(), H_rowadr.data(), H_colind.data(), nullptr, data.nefc,
          nv, data.J_rownnz.data(), data.J_rowadr.data(), data.J_colind.data(),
          data.J_rowsuper.data(), data.JT_rownnz.data(), data.JT_rowadr.data(),
          data.JT_colind.data(), data.JT_rowsuper.data(), d);
    }
  } else {
    // row: use Count (lower triangle only)
    mju_sqrMatTDSparseCount(
        H_rownnz.data(), H_rowadr.data(), nv, data.J_rownnz.data(),
        data.J_rowadr.data(), data.J_colind.data(), data.JT_rownnz.data(),
        data.JT_rowadr.data(), data.JT_colind.data(), nullptr, d, 0);
    int nH = H_rowadr[nv - 1] + H_rownnz[nv - 1];
    H.resize(nH);
    H_colind.resize(nH);
  }

  for (auto s : state) {
    switch (variant) {
      case SqrMatTDVariant::kBaseline:
        mju_superSparse(data.nefc, data.J_rowsuper.data(), data.J_rownnz.data(),
                        data.J_rowadr.data(), data.J_colind.data());
        mju_sqrMatTDSparse_baseline(
            H.data(), data.J.data(), data.JT.data(), data.D.data(), data.nefc,
            nv, H_rownnz.data(), H_rowadr.data(), H_colind.data(),
            data.J_rownnz.data(), data.J_rowadr.data(), data.J_colind.data(),
            data.J_rowsuper.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(),
            data.JT_rowsuper.data(), d);
        break;
      case SqrMatTDVariant::kRow:
        mju_sqrMatTDSparseCount(
            H_rownnz.data(), H_rowadr.data(), nv, data.J_rownnz.data(),
            data.J_rowadr.data(), data.J_colind.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(), nullptr, d, 0);
        mju_sqrMatTDSparse_row(
            H.data(), data.J.data(), data.JT.data(), data.D.data(), data.nefc,
            nv, H_rownnz.data(), H_rowadr.data(), H_colind.data(),
            data.J_rownnz.data(), data.J_rowadr.data(), data.J_colind.data(),
            nullptr, data.JT_rownnz.data(), data.JT_rowadr.data(),
            data.JT_colind.data(), data.JT_rowsuper.data(), d, nullptr);
        break;
      case SqrMatTDVariant::kCol:
        mju_sqrMatTDSparseCount(
            H_rownnz.data(), H_rowadr.data(), nv, data.J_rownnz.data(),
            data.J_rowadr.data(), data.J_colind.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(), nullptr, d, 0);
        mju_sqrMatTDSparse(
            H.data(), data.J.data(), data.JT.data(), data.D.data(), data.nefc,
            nv, H_rownnz.data(), H_rowadr.data(), H_colind.data(),
            data.J_rownnz.data(), data.J_rowadr.data(), data.J_colind.data(),
            nullptr, data.JT_rownnz.data(), data.JT_rowadr.data(),
            data.JT_colind.data(), data.JT_rowsuper.data(), d, nullptr);
        break;

      case SqrMatTDVariant::kSplitCol:
        mju_sqrMatTDSparseSymbolic(
            H_rownnz.data(), H_rowadr.data(), nullptr, nullptr, data.nefc, nv,
            data.J_rownnz.data(), data.J_rowadr.data(), data.J_colind.data(),
            data.J_rowsuper.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(),
            data.JT_rowsuper.data(), d);
        mju_sqrMatTDSparseSymbolic(
            H_rownnz.data(), H_rowadr.data(), H_colind.data(), nullptr,
            data.nefc, nv, data.J_rownnz.data(), data.J_rowadr.data(),
            data.J_colind.data(), data.J_rowsuper.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(),
            data.JT_rowsuper.data(), d);
        mju_sqrMatTDSparseNumeric(
            H.data(), nv, H_rownnz.data(), H_rowadr.data(), H_colind.data(),
            nullptr, data.J.data(), data.J_rownnz.data(), data.J_rowadr.data(),
            data.J_colind.data(), data.JT.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(),
            data.JT_rowsuper.data(), data.D.data(), d);
        break;

      case SqrMatTDVariant::kSymbolicOnly:
        mju_sqrMatTDSparseSymbolic(
            H_rownnz.data(), H_rowadr.data(), nullptr, nullptr, data.nefc, nv,
            data.J_rownnz.data(), data.J_rowadr.data(), data.J_colind.data(),
            data.J_rowsuper.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(),
            data.JT_rowsuper.data(), d);
        mju_sqrMatTDSparseSymbolic(
            H_rownnz.data(), H_rowadr.data(), H_colind.data(), nullptr,
            data.nefc, nv, data.J_rownnz.data(), data.J_rowadr.data(),
            data.J_colind.data(), data.J_rowsuper.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(),
            data.JT_rowsuper.data(), d);
        break;

      case SqrMatTDVariant::kNumericOnly:
        mju_sqrMatTDSparseNumeric(
            H.data(), nv, H_rownnz.data(), H_rowadr.data(), H_colind.data(),
            nullptr, data.J.data(), data.J_rownnz.data(), data.J_rowadr.data(),
            data.J_colind.data(), data.JT.data(), data.JT_rownnz.data(),
            data.JT_rowadr.data(), data.JT_colind.data(),
            data.JT_rowsuper.data(), data.D.data(), d);
        break;
    }
  }

  mj_deleteData(d);
  state.SetItemsProcessed(state.iterations());
}

void BM_sqrMatTD_2H100_baseline(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H2_100>(state, SqrMatTDVariant::kBaseline);
}
BENCHMARK(BM_sqrMatTD_2H100_baseline);

void BM_sqrMatTD_2H100_row(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H2_100>(state, SqrMatTDVariant::kRow);
}
BENCHMARK(BM_sqrMatTD_2H100_row);

void BM_sqrMatTD_2H100_col(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H2_100>(state, SqrMatTDVariant::kCol);
}
BENCHMARK(BM_sqrMatTD_2H100_col);

void BM_sqrMatTD_2H100_splitCol(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H2_100>(state, SqrMatTDVariant::kSplitCol);
}
BENCHMARK(BM_sqrMatTD_2H100_splitCol);

void BM_sqrMatTD_2H100_symbolic(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H2_100>(state, SqrMatTDVariant::kSymbolicOnly);
}
BENCHMARK(BM_sqrMatTD_2H100_symbolic);

void BM_sqrMatTD_2H100_numeric(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H2_100>(state, SqrMatTDVariant::kNumericOnly);
}
BENCHMARK(BM_sqrMatTD_2H100_numeric);

void BM_sqrMatTD_100H_baseline(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H100>(state, SqrMatTDVariant::kBaseline);
}
BENCHMARK(BM_sqrMatTD_100H_baseline);

void BM_sqrMatTD_100H_row(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H100>(state, SqrMatTDVariant::kRow);
}
BENCHMARK(BM_sqrMatTD_100H_row);

void BM_sqrMatTD_100H_col(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H100>(state, SqrMatTDVariant::kCol);
}
BENCHMARK(BM_sqrMatTD_100H_col);

void BM_sqrMatTD_100H_splitCol(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H100>(state, SqrMatTDVariant::kSplitCol);
}
BENCHMARK(BM_sqrMatTD_100H_splitCol);

void BM_sqrMatTD_100H_symbolic(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H100>(state, SqrMatTDVariant::kSymbolicOnly);
}
BENCHMARK(BM_sqrMatTD_100H_symbolic);

void BM_sqrMatTD_100H_numeric(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_sqrMatTD_impl<Size::H100>(state, SqrMatTDVariant::kNumericOnly);
}
BENCHMARK(BM_sqrMatTD_100H_numeric);

// ========================= Dense SqrMatTD Benchmarks =========================

struct DenseSqrMatDataset {
  int nr;
  int nc;
  std::vector<mjtNum> mat;
  std::vector<mjtNum> diag;

  DenseSqrMatDataset(int nr_, int nc_)
      : nr(nr_), nc(nc_), mat(nr_ * nc_), diag(nr_) {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0, 1);
    for (int i = 0; i < nr; i++) {
      diag[i] = std::abs(dist(rng)) + 0.1;
      for (int j = 0; j < nc; j++) {
        mat[i * nc + j] = dist(rng);
      }
    }
  }
};

static void DenseSqrMatTDBenchmark(benchmark::State& state, int nr, int nc,
                                   bool use_diag) {
  static DenseSqrMatDataset d_60_30(60, 30);
  static DenseSqrMatDataset d_200_100(200, 100);
  static DenseSqrMatDataset d_600_300(600, 300);

  const DenseSqrMatDataset* dp = &d_60_30;
  if (nc == 100) dp = &d_200_100;
  if (nc == 300) dp = &d_600_300;
  const DenseSqrMatDataset& d = *dp;

  std::vector<mjtNum> res(nc * nc);
  const mjtNum* diag_ptr = use_diag ? d.diag.data() : nullptr;

  for (auto s : state) {
    mju_sqrMatTD(res.data(), d.mat.data(), diag_ptr, nr, nc);
    benchmark::DoNotOptimize(res.data());
  }

  state.SetItemsProcessed(state.iterations());
}

void BM_sqrMatTD_Dense_60_30_NoDiag(benchmark::State& state) {
  DenseSqrMatTDBenchmark(state, 60, 30, false);
}
void BM_sqrMatTD_Dense_60_30_Diag(benchmark::State& state) {
  DenseSqrMatTDBenchmark(state, 60, 30, true);
}
void BM_sqrMatTD_Dense_200_100_NoDiag(benchmark::State& state) {
  DenseSqrMatTDBenchmark(state, 200, 100, false);
}
void BM_sqrMatTD_Dense_200_100_Diag(benchmark::State& state) {
  DenseSqrMatTDBenchmark(state, 200, 100, true);
}
void BM_sqrMatTD_Dense_600_300_NoDiag(benchmark::State& state) {
  DenseSqrMatTDBenchmark(state, 600, 300, false);
}
void BM_sqrMatTD_Dense_600_300_Diag(benchmark::State& state) {
  DenseSqrMatTDBenchmark(state, 600, 300, true);
}

BENCHMARK(BM_sqrMatTD_Dense_60_30_NoDiag);
BENCHMARK(BM_sqrMatTD_Dense_60_30_Diag);
BENCHMARK(BM_sqrMatTD_Dense_200_100_NoDiag);
BENCHMARK(BM_sqrMatTD_Dense_200_100_Diag);
BENCHMARK(BM_sqrMatTD_Dense_600_300_NoDiag);
BENCHMARK(BM_sqrMatTD_Dense_600_300_Diag);

}  // namespace
}  // namespace mujoco
