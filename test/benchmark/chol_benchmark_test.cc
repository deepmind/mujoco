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

// A benchmark for comparing old vs new Cholesky factorization implementations.

#include <algorithm>
#include <cstring>
#include <vector>

#include <benchmark/benchmark.h>
#include "absl/base/attributes.h"
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_support.h"
#include "src/engine/engine_util_solve.h"
#include "src/engine/engine_util_sparse.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

// Helper to compute H = M + J'*D*J using sparse matrices
struct HessianData {
  int nv;
  int nL;
  int nH;

  // H sparse structure
  std::vector<mjtNum> H;
  std::vector<int> H_rownnz;
  std::vector<int> H_rowadr;
  std::vector<int> H_colind;

  // H transpose for symbolics
  std::vector<int> HT_rownnz;
  std::vector<int> HT_rowadr;
  std::vector<int> HT_colind;

  // L and LT factor structure
  std::vector<int> L_rownnz;
  std::vector<int> L_rowadr;
  std::vector<int> LT_rownnz;
  std::vector<int> LT_rowadr;

  // L initial values for BM_chol_old (lower triangle of H, zero-filled for
  // fill-in)
  std::vector<mjtNum> L_init;
  std::vector<int> L_rownnz_init;
  std::vector<int> L_colind_init;

  // J transpose
  std::vector<mjtNum> JT;
  std::vector<int> JT_rownnz;
  std::vector<int> JT_rowadr;
  std::vector<int> JT_colind;
  std::vector<int> JT_rowsuper;

  // D diagonal
  std::vector<mjtNum> D;

  int nefc;

  void Setup(const mjModel* m, mjData* d) {
    // initialize simulation state
    mj_resetDataKeyframe(m, d, 0);
    mj_forward(m, d);

    nv = m->nv;
    nefc = d->nefc;

    // compute D corresponding to quad states
    D.resize(nefc);
    for (int i = 0; i < nefc; i++) {
      if (d->efc_state[i] == mjCNSTRSTATE_QUADRATIC) {
        D[i] = d->efc_D[i];
      } else {
        D[i] = 0;
      }
    }

    // transpose J
    JT.resize(d->nJ);
    JT_rownnz.resize(nv);
    JT_rowadr.resize(nv);
    JT_colind.resize(d->nJ);
    JT_rowsuper.resize(nv);
    mju_transposeSparse(JT.data(), d->efc_J, nefc, nv, JT_rownnz.data(),
                        JT_rowadr.data(), JT_colind.data(), JT_rowsuper.data(),
                        d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind);

    // count H sparsity: nH from J'*D*J
    H_rownnz.resize(nv);
    H_rowadr.resize(nv);
    mju_sqrMatTDSparseCount(H_rownnz.data(), H_rowadr.data(), nv,
                            d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                            JT_rownnz.data(), JT_rowadr.data(),
                            JT_colind.data(), nullptr, d, 1);

    // add M elements to the H row counts and addresses
    for (int r = 0; r < nv; r++) {
      H_rownnz[r] += m->M_rownnz[r];
    }
    H_rowadr[0] = 0;
    for (int r = 1; r < nv; r++) {
      H_rowadr[r] = H_rowadr[r - 1] + H_rownnz[r - 1];
    }
    nH = H_rowadr[nv - 1] + H_rownnz[nv - 1];

    // allocate H and colind with proper sparse size, zero-initialize H
    H.assign(nH, 0);
    H_colind.assign(nH, 0);

    // reset rownnz for filling (sqrMatTDSparse will fill it again)
    std::fill(H_rownnz.begin(), H_rownnz.end(), 0);

    // recount just J'*D*J (without M shift)
    mju_sqrMatTDSparseCount(H_rownnz.data(), H_rowadr.data(), nv,
                            d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                            JT_rownnz.data(), JT_rowadr.data(),
                            JT_colind.data(), nullptr, d, 1);

    // add shift for M to rowadr
    int shift = 0;
    for (int r = 0; r < nv - 1; r++) {
      shift += m->M_rownnz[r];
      H_rowadr[r + 1] += shift;
    }

    // compute H = J'*D*J
    mju_sqrMatTDSparse(H.data(), d->efc_J, JT.data(), D.data(), nefc, nv,
                       H_rownnz.data(), H_rowadr.data(), H_colind.data(),
                       d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                       nullptr, JT_rownnz.data(), JT_rowadr.data(),
                       JT_colind.data(), JT_rowsuper.data(), d, nullptr);

    // add M to H using mj_addM
    mj_addM(m, d, H.data(), H_rownnz.data(), H_rowadr.data(), H_colind.data());

    // transpose H for symbolic
    HT_rownnz.resize(nv);
    HT_rowadr.resize(nv);
    HT_colind.resize(nH);
    mju_transposeSparse(nullptr, nullptr, nv, nv, HT_rownnz.data(),
                        HT_rowadr.data(), HT_colind.data(), nullptr,
                        H_rownnz.data(), H_rowadr.data(), H_colind.data());

    // count L fill-in (also counts LT structure)
    L_rownnz.resize(nv);
    L_rowadr.resize(nv);
    LT_rownnz.resize(nv);
    LT_rowadr.resize(nv);
    nL = mju_cholFactorSymbolic(nullptr, L_rownnz.data(), L_rowadr.data(),
                                nullptr, LT_rownnz.data(), LT_rowadr.data(),
                                nullptr, HT_rownnz.data(), HT_rowadr.data(),
                                HT_colind.data(), nv, d);

    // precompute initial L state for BM_chol_old
    // extract lower triangle of H into L format, zero-fill for fill-in
    L_init.assign(nL, 0);
    L_colind_init.assign(nL, 0);
    L_rownnz_init.resize(nv);
    for (int r = 0; r < nv; r++) {
      int l_adr = L_rowadr[r];
      int h_adr = H_rowadr[r];
      int lower_nnz = 0;
      for (int i = 0; i < H_rownnz[r]; i++) {
        int col = H_colind[h_adr + i];
        if (col <= r) {
          L_init[l_adr + lower_nnz] = H[h_adr + i];
          L_colind_init[l_adr + lower_nnz] = col;
          lower_nnz++;
        }
      }
      L_rownnz_init[r] = lower_nnz;
    }
  }
};

// ----------------------------- benchmark ------------------------------------

enum class Size { L, XL };

template <Size S>
const char* ModelPath() {
  if constexpr (S == Size::L) {
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
  return m;
}

template <Size S>
HessianData& GetHessianData() {
  static HessianData data;
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

// old implementation benchmark
template <Size S>
static void BM_chol_old(benchmark::State& state) {
  mjModel* m = GetModel<S>();
  mjData* d = mj_makeData(m);

  HessianData& hd = GetHessianData<S>();

  std::vector<mjtNum> L_work(hd.nL);
  std::vector<int> L_colind_work(hd.nL);
  std::vector<int> L_rownnz_work(hd.nv);

  for (auto s : state) {
    // fast reset using memcpy from precomputed initial state
    std::memcpy(L_work.data(), hd.L_init.data(), hd.nL * sizeof(mjtNum));
    std::memcpy(L_colind_work.data(), hd.L_colind_init.data(),
                hd.nL * sizeof(int));
    std::memcpy(L_rownnz_work.data(), hd.L_rownnz_init.data(),
                hd.nv * sizeof(int));
    mju_cholFactorSparse(L_work.data(), hd.nv, mjMINVAL, L_rownnz_work.data(),
                         hd.L_rowadr.data(), L_colind_work.data(), d);
  }

  mj_deleteData(d);
  state.SetItemsProcessed(state.iterations());
}

// new symbolic implementation benchmark
template <Size S>
static void BM_chol_symbolic(benchmark::State& state) {
  mjModel* m = GetModel<S>();
  mjData* d = mj_makeData(m);

  HessianData& hd = GetHessianData<S>();

  std::vector<int> L_colind_work(hd.nL);
  std::vector<int> LT_colind_work(hd.nL);
  std::vector<int> LT_pos_work(hd.nL);

  for (auto s : state) {
    mju_cholFactorSymbolic(L_colind_work.data(), hd.L_rownnz.data(),
                           hd.L_rowadr.data(), LT_colind_work.data(),
                           hd.LT_rownnz.data(), hd.LT_rowadr.data(),
                           LT_pos_work.data(), hd.HT_rownnz.data(),
                           hd.HT_rowadr.data(), hd.HT_colind.data(), hd.nv, d);
  }

  mj_deleteData(d);
  state.SetItemsProcessed(state.iterations());
}

// new numeric implementation benchmark
template <Size S>
static void BM_chol_numeric(benchmark::State& state) {
  mjModel* m = GetModel<S>();
  mjData* d = mj_makeData(m);

  HessianData& hd = GetHessianData<S>();

  std::vector<mjtNum> L_work(hd.nL);
  std::vector<int> L_colind_work(hd.nL);
  std::vector<int> LT_colind_work(hd.nL);
  std::vector<int> LT_pos_work(hd.nL);
  std::vector<mjtNum> scratch(hd.nv);

  // symbolic setup (not benchmarked)
  mju_cholFactorSymbolic(L_colind_work.data(), hd.L_rownnz.data(),
                         hd.L_rowadr.data(), LT_colind_work.data(),
                         hd.LT_rownnz.data(), hd.LT_rowadr.data(),
                         LT_pos_work.data(), hd.HT_rownnz.data(),
                         hd.HT_rowadr.data(), hd.HT_colind.data(), hd.nv, d);

  for (auto s : state) {
    mju_cholFactorNumeric(L_work.data(), hd.nv, mjMINVAL, hd.L_rownnz.data(),
                          hd.L_rowadr.data(), L_colind_work.data(),
                          hd.LT_rownnz.data(), hd.LT_rowadr.data(),
                          LT_colind_work.data(), LT_pos_work.data(),
                          hd.H.data(), hd.H_rownnz.data(), hd.H_rowadr.data(),
                          hd.H_colind.data(), scratch.data());
  }

  mj_deleteData(d);
  state.SetItemsProcessed(state.iterations());
}

void BM_old_L(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_old<Size::L>(state);
}
BENCHMARK(BM_old_L);

void BM_symbolic_L(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_symbolic<Size::L>(state);
}
BENCHMARK(BM_symbolic_L);

void BM_numeric_L(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_numeric<Size::L>(state);
}
BENCHMARK(BM_numeric_L);

void BM_old_XL(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_old<Size::XL>(state);
}
BENCHMARK(BM_old_XL);

void BM_symbolic_XL(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_symbolic<Size::XL>(state);
}
BENCHMARK(BM_symbolic_XL);

void BM_numeric_XL(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_numeric<Size::XL>(state);
}
BENCHMARK(BM_numeric_XL);

// sparse Cholesky solve benchmark
template <Size S>
static void BM_chol_solve(benchmark::State& state) {
  mjModel* m = GetModel<S>();
  mjData* d = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  mj_forward(m, d);

  HessianData& hd = GetHessianData<S>();

  std::vector<mjtNum> L_work(hd.nL);
  std::vector<int> L_colind_work(hd.nL);
  std::vector<int> LT_colind_work(hd.nL);
  std::vector<int> LT_pos_work(hd.nL);
  std::vector<mjtNum> scratch(hd.nv);

  mju_cholFactorSymbolic(L_colind_work.data(), hd.L_rownnz.data(),
                         hd.L_rowadr.data(), LT_colind_work.data(),
                         hd.LT_rownnz.data(), hd.LT_rowadr.data(),
                         LT_pos_work.data(), hd.HT_rownnz.data(),
                         hd.HT_rowadr.data(), hd.HT_colind.data(), hd.nv, d);
  mju_cholFactorNumeric(L_work.data(), hd.nv, mjMINVAL, hd.L_rownnz.data(),
                        hd.L_rowadr.data(), L_colind_work.data(),
                        hd.LT_rownnz.data(), hd.LT_rowadr.data(),
                        LT_colind_work.data(), LT_pos_work.data(), hd.H.data(),
                        hd.H_rownnz.data(), hd.H_rowadr.data(),
                        hd.H_colind.data(), scratch.data());

  std::vector<mjtNum> rhs(hd.nv);
  std::vector<mjtNum> res(hd.nv);
  for (int i = 0; i < hd.nv; i++) {
    rhs[i] = d->qfrc_smooth[i];
  }

  for (auto s : state) {
    mju_cholSolveSparse(res.data(), L_work.data(), rhs.data(), hd.nv,
                        hd.L_rownnz.data(), hd.L_rowadr.data(),
                        L_colind_work.data());
    benchmark::DoNotOptimize(res.data());
  }

  mj_deleteData(d);
  state.SetItemsProcessed(state.iterations());
}

void BM_solve_L(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_solve<Size::L>(state);
}
BENCHMARK(BM_solve_L);

void BM_solve_XL(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_chol_solve<Size::XL>(state);
}
BENCHMARK(BM_solve_XL);

// -------------------- rank-1 update/downdate benchmarks ----------------------

constexpr int kNumUpdateVectors = 25;

// local copy of the engine's mju_combineSparseInc, which is not exported from
// the shared library; copying it here keeps that symbol internal and freezes
// the behavior the old implementation below is benchmarked against
static void combineSparseInc_local(mjtNum* dst, const mjtNum* src, int n,
                                   mjtNum a, mjtNum b, int dst_nnz, int src_nnz,
                                   const int* dst_ind, const int* src_ind) {
  // check for identical pattern
  if (dst_nnz == src_nnz) {
    if (mju_compare(dst_ind, src_ind, dst_nnz)) {
      // combine mjtNum data directly
      mju_addToSclScl(dst, src, a, b, dst_nnz);
      return;
    }
  }

  // scale dst by a
  if (a != 1) {
    mju_scl(dst, dst, a, dst_nnz);
  }

  // prepare to merge
  int di = 0, si = 0;
  int dadr = di < dst_nnz ? dst_ind[di] : n + 1;
  int sadr = si < src_nnz ? src_ind[si] : n + 1;

  // add src*b at common indices
  while (di < dst_nnz) {
    // both
    if (dadr == sadr) {
      dst[di++] += b * src[si++];

      dadr = di < dst_nnz ? dst_ind[di] : n + 1;
      sadr = si < src_nnz ? src_ind[si] : n + 1;
    } else if (dadr < sadr) {
      // dst only
      di++;
      dadr = di < dst_nnz ? dst_ind[di] : n + 1;
    } else {
      // src only
      si++;
      sadr = si < src_nnz ? src_ind[si] : n + 1;
    }
  }
}

// old implementation using sparse merge
int ABSL_ATTRIBUTE_NOINLINE mju_cholUpdateSparse_old(
    mjtNum* mat, mjtNum* x, int n, int flg_plus, const int* rownnz,
    const int* rowadr, const int* colind, int x_nnz, int* x_ind, mjData* d) {
  int rank = n, i = x_nnz - 1;
  while (i >= 0) {
    int nnz = rownnz[x_ind[i]], adr = rowadr[x_ind[i]];
    mjtNum tmp = mat[adr + nnz - 1] * mat[adr + nnz - 1] +
                 (flg_plus ? x[i] * x[i] : -x[i] * x[i]);
    if (tmp < mjMINVAL) {
      tmp = mjMINVAL;
      rank--;
    }
    mjtNum r = mju_sqrt(tmp);
    mjtNum c = r / mat[adr + nnz - 1];
    mjtNum s = x[i] / mat[adr + nnz - 1];
    mat[adr + nnz - 1] = r;
    combineSparseInc_local(mat + adr, x, n, 1 / c, (flg_plus ? s / c : -s / c),
                           nnz - 1, i, colind + adr, x_ind);
    int new_x_nnz =
        mju_combineSparse(x, mat + adr, c, -s, i, nnz - 1, x_ind, colind + adr);
    i = i - 1 + (new_x_nnz - i);
  }
  return rank;
}

// old update implementation benchmark
template <Size S>
static void BM_update_old(benchmark::State& state) {
  mjModel* m = GetModel<S>();
  mjData* d = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  mj_forward(m, d);

  HessianData& hd = GetHessianData<S>();

  int nv = hd.nv;

  // factorize L using old method (we'll just keep updating this)
  std::vector<mjtNum> L_work(hd.nL);
  std::vector<int> L_colind_work(hd.nL);
  std::vector<int> L_rownnz_work(hd.nv);
  std::memcpy(L_work.data(), hd.L_init.data(), hd.nL * sizeof(mjtNum));
  std::memcpy(L_colind_work.data(), hd.L_colind_init.data(),
              hd.nL * sizeof(int));
  std::memcpy(L_rownnz_work.data(), hd.L_rownnz_init.data(),
              hd.nv * sizeof(int));
  mju_cholFactorSparse(L_work.data(), nv, mjMINVAL, L_rownnz_work.data(),
                       hd.L_rowadr.data(), L_colind_work.data(), d);

  // prepare update vectors: pick kNumUpdateVectors rows from J (constraint
  // rows) Each J row has DoF indices which is correct for updating L (nv x nv)
  int nefc = d->nefc;
  std::vector<std::vector<mjtNum>> update_vecs(kNumUpdateVectors);
  std::vector<std::vector<int>> update_inds(kNumUpdateVectors);
  for (int k = 0; k < kNumUpdateVectors; k++) {
    int row = (k * 7) % nefc;
    int nnz = d->efc_J_rownnz[row];
    int adr = d->efc_J_rowadr[row];
    update_vecs[k].resize(nnz);
    update_inds[k].resize(nnz);
    for (int i = 0; i < nnz; i++) {
      update_vecs[k][i] = d->efc_J[adr + i] * 0.01;
      update_inds[k][i] = d->efc_J_colind[adr + i];
    }
  }

  // working copy of update vector (sized to nv since pattern can grow)
  std::vector<mjtNum> x_work(nv);
  std::vector<int> x_ind_work(nv);

  int vec_idx = 0;
  for (auto s : state) {
    int nnz = update_inds[vec_idx].size();
    std::memset(x_work.data(), 0, nv * sizeof(mjtNum));
    std::memcpy(x_work.data(), update_vecs[vec_idx].data(),
                nnz * sizeof(mjtNum));
    std::memcpy(x_ind_work.data(), update_inds[vec_idx].data(),
                nnz * sizeof(int));
    mju_cholUpdateSparse_old(L_work.data(), x_work.data(), nv, 1,
                             L_rownnz_work.data(), hd.L_rowadr.data(),
                             L_colind_work.data(), nnz, x_ind_work.data(), d);
    vec_idx = (vec_idx + 1) % kNumUpdateVectors;
  }

  mj_deleteData(d);
  state.SetItemsProcessed(state.iterations());
}

// new update implementation benchmark
template <Size S>
static void BM_update_new(benchmark::State& state) {
  mjModel* m = GetModel<S>();
  mjData* d = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  mj_forward(m, d);

  HessianData& hd = GetHessianData<S>();

  int nv = hd.nv;

  // factorize L using new method
  std::vector<mjtNum> L_work(hd.nL);
  std::vector<int> L_colind_work(hd.nL);
  std::vector<int> LT_colind_work(hd.nL);
  std::vector<int> LT_pos_work(hd.nL);
  std::vector<mjtNum> scratch(nv);

  mju_cholFactorSymbolic(L_colind_work.data(), hd.L_rownnz.data(),
                         hd.L_rowadr.data(), LT_colind_work.data(),
                         hd.LT_rownnz.data(), hd.LT_rowadr.data(),
                         LT_pos_work.data(), hd.HT_rownnz.data(),
                         hd.HT_rowadr.data(), hd.HT_colind.data(), nv, d);
  mju_cholFactorNumeric(L_work.data(), nv, mjMINVAL, hd.L_rownnz.data(),
                        hd.L_rowadr.data(), L_colind_work.data(),
                        hd.LT_rownnz.data(), hd.LT_rowadr.data(),
                        LT_colind_work.data(), LT_pos_work.data(), hd.H.data(),
                        hd.H_rownnz.data(), hd.H_rowadr.data(),
                        hd.H_colind.data(), scratch.data());

  // prepare update vectors: pick kNumUpdateVectors rows from J (constraint
  // rows) Each J row has DoF indices which is correct for updating L (nv x nv)
  int nefc = d->nefc;
  std::vector<std::vector<mjtNum>> update_vecs(kNumUpdateVectors);
  std::vector<std::vector<int>> update_inds(kNumUpdateVectors);
  for (int k = 0; k < kNumUpdateVectors; k++) {
    int row = (k * 7) % nefc;
    int nnz = d->efc_J_rownnz[row];
    int adr = d->efc_J_rowadr[row];
    update_vecs[k].resize(nnz);
    update_inds[k].resize(nnz);
    for (int i = 0; i < nnz; i++) {
      update_vecs[k][i] = d->efc_J[adr + i] * 0.01;
      update_inds[k][i] = d->efc_J_colind[adr + i];
    }
  }

  int vec_idx = 0;
  for (auto s : state) {
    int nnz = update_inds[vec_idx].size();
    mju_cholUpdateSparse(L_work.data(), update_vecs[vec_idx].data(), nv, 1,
                         hd.L_rownnz.data(), hd.L_rowadr.data(),
                         L_colind_work.data(), nnz, update_inds[vec_idx].data(),
                         scratch.data());
    vec_idx = (vec_idx + 1) % kNumUpdateVectors;
  }

  mj_deleteData(d);
  state.SetItemsProcessed(state.iterations());
}

void BM_update_old_L(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_update_old<Size::L>(state);
}
BENCHMARK(BM_update_old_L);

void BM_update_new_L(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_update_new<Size::L>(state);
}
BENCHMARK(BM_update_new_L);

void BM_update_old_XL(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_update_old<Size::XL>(state);
}
BENCHMARK(BM_update_old_XL);

void BM_update_new_XL(benchmark::State& state) {
  MujocoErrorTestGuard guard;
  BM_update_new<Size::XL>(state);
}
BENCHMARK(BM_update_new_XL);

}  // namespace
}  // namespace mujoco
