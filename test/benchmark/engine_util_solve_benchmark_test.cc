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

// A benchmark comparing generic (runtime-sized) and fixed-size dense LU
// factorization and solve at n = 6, with dense Cholesky as a reference.
// The 6x6 case is the free-body block in the implicitfast integrator.

#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include "absl/base/attributes.h"
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_blas.h"
#include "src/engine/engine_util_solve.h"

namespace mujoco {
namespace {

constexpr int kN = 6;
constexpr int kNumMat = 256;  // cycle through matrices to avoid cache pinning

// generate matrices with the structure of the free-body block M - h*D + h*G:
// symmetric positive definite plus a scaled antisymmetric part
struct LUData {
  std::vector<mjtNum> mats;  // kNumMat x 36
  std::vector<mjtNum> rhs;   // kNumMat x 6

  LUData() : mats(kNumMat * kN * kN), rhs(kNumMat * kN) {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0, 1);

    for (int m = 0; m < kNumMat; m++) {
      mjtNum* A = mats.data() + m * kN * kN;
      mjtNum sqrtH[kN * kN];
      for (int i = 0; i < kN * kN; i++) {
        sqrtH[i] = dist(rng);
      }

      // A = sqrtH' * sqrtH + n*I: SPD
      mju_mulMatTMat(A, sqrtH, sqrtH, kN, kN, kN);
      for (int i = 0; i < kN; i++) {
        A[i * kN + i] += kN;
      }

      // add antisymmetric part (gyroscopic-like), scaled to ~10% of diagonal
      for (int r = 0; r < kN; r++) {
        for (int c = r + 1; c < kN; c++) {
          mjtNum g = 0.6 * dist(rng);
          A[r * kN + c] += g;
          A[c * kN + r] -= g;
        }
      }

      for (int i = 0; i < kN; i++) {
        rhs[m * kN + i] = dist(rng);
      }
    }
  }
};

LUData& GetData() {
  static LUData* data = new LUData();
  return *data;
}

enum class Variant {
  kGeneric,  // mju_factorLU / mju_solveLU with n = 6
  kFixed,    // mju_factorLU6 / mju_solveLU6
  kCholesky  // mju_cholFactor / mju_cholSolve (symmetric part; reference)
};

// factor only (includes the copy, identical across variants)
ABSL_ATTRIBUTE_NO_TAIL_CALL static void FactorBenchmark(benchmark::State& state,
                                                        Variant variant) {
  LUData& data = GetData();
  mjtNum A[kN * kN];
  int pivot[kN];
  int m = 0;

  for (auto s : state) {
    mju_copy(A, data.mats.data() + m * kN * kN, kN * kN);
    m = (m + 1) % kNumMat;

    int rank;
    switch (variant) {
      case Variant::kGeneric:
        rank = mju_factorLU(A, kN, pivot);
        break;
      case Variant::kFixed:
        rank = mju_factorLU6(A, pivot);
        break;
      case Variant::kCholesky:
        rank = mju_cholFactor(A, kN, 0);
        break;
    }
    benchmark::DoNotOptimize(rank);
    benchmark::DoNotOptimize(A);
  }

  state.SetItemsProcessed(state.iterations());
}

// factor and solve
ABSL_ATTRIBUTE_NO_TAIL_CALL static void FactorSolveBenchmark(
    benchmark::State& state, Variant variant) {
  LUData& data = GetData();
  mjtNum A[kN * kN];
  mjtNum x[kN];
  int pivot[kN];
  int m = 0;

  for (auto s : state) {
    mju_copy(A, data.mats.data() + m * kN * kN, kN * kN);
    const mjtNum* b = data.rhs.data() + m * kN;
    m = (m + 1) % kNumMat;

    switch (variant) {
      case Variant::kGeneric:
        mju_factorLU(A, kN, pivot);
        mju_solveLU(x, A, b, pivot, kN);
        break;
      case Variant::kFixed:
        mju_factorLU6(A, pivot);
        mju_solveLU6(x, A, b, pivot);
        break;
      case Variant::kCholesky:
        mju_cholFactor(A, kN, 0);
        mju_cholSolve(x, A, b, kN);
        break;
    }
    benchmark::DoNotOptimize(x);
  }

  state.SetItemsProcessed(state.iterations());
}

void BM_factorLU_generic6(benchmark::State& state) {
  FactorBenchmark(state, Variant::kGeneric);
}
BENCHMARK(BM_factorLU_generic6);

void BM_factorLU_fixed6(benchmark::State& state) {
  FactorBenchmark(state, Variant::kFixed);
}
BENCHMARK(BM_factorLU_fixed6);

void BM_factorChol6(benchmark::State& state) {
  FactorBenchmark(state, Variant::kCholesky);
}
BENCHMARK(BM_factorChol6);

void BM_factorSolveLU_generic6(benchmark::State& state) {
  FactorSolveBenchmark(state, Variant::kGeneric);
}
BENCHMARK(BM_factorSolveLU_generic6);

void BM_factorSolveLU_fixed6(benchmark::State& state) {
  FactorSolveBenchmark(state, Variant::kFixed);
}
BENCHMARK(BM_factorSolveLU_fixed6);

void BM_factorSolveChol6(benchmark::State& state) {
  FactorSolveBenchmark(state, Variant::kCholesky);
}
BENCHMARK(BM_factorSolveChol6);

// ----------------------------- mju_boxQP -------------------------------------

struct BoxQPDataset {
  int n;
  static constexpr int kNumProb = 27;
  std::vector<mjtNum> H;
  std::vector<mjtNum> g;
  std::vector<mjtNum> lower;
  std::vector<mjtNum> upper;
  std::vector<mjtNum> warm;

  explicit BoxQPDataset(int n_in)
      : n(n_in),
        H(kNumProb * n * n),
        g(kNumProb * n),
        lower(kNumProb * n),
        upper(kNumProb * n),
        warm(kNumProb * n) {
    int seed = 1;
    int idx = 0;
    std::vector<mjtNum> sqrtH(n * n);
    std::vector<mjtNum> R(n * (n + 7));
    std::vector<int> index(n);
    for (mjtNum scaleH : {.01, 1.0, 100.0}) {
      for (mjtNum scaleg : {.01, 1.0, 100.0}) {
        for (mjtNum scalebounds : {.01, 1.0, 100.0}) {
          std::mt19937_64 rng(seed++);
          std::normal_distribution<double> dist(0, 1);
          mjtNum* Hp = H.data() + idx * n * n;
          mjtNum* gp = g.data() + idx * n;
          mjtNum* lp = lower.data() + idx * n;
          mjtNum* up = upper.data() + idx * n;
          mjtNum* wp = warm.data() + idx * n;
          for (int i = 0; i < n; i++) {
            gp[i] = dist(rng);
            lp[i] = 5 * dist(rng);
            up[i] = 5 * dist(rng);
            if (lp[i] > up[i]) std::swap(lp[i], up[i]);
            for (int j = 0; j < n; j++) {
              sqrtH[n * i + j] = dist(rng);
            }
          }
          mju_mulMatTMat(Hp, sqrtH.data(), sqrtH.data(), n, n, n);
          mju_scl(Hp, Hp, scaleH, n * n);
          mju_scl(gp, gp, scaleg, n);
          mju_scl(lp, lp, scalebounds, n);
          mju_scl(up, up, scalebounds, n);

          // compute warmstart from slightly perturbed g
          std::vector<mjtNum> g_pert(n);
          for (int i = 0; i < n; i++) {
            g_pert[i] = gp[i] + 0.1 * scaleg * dist(rng);
          }
          mju_zero(wp, n);
          mju_boxQP(wp, R.data(), index.data(), Hp, g_pert.data(), n, lp, up);
          idx++;
        }
      }
    }
  }
};

static mjtNum mulVecMatVecSym_old(const mjtNum* vec, const mjtNum* mat, int n) {
  mjtNum res = 0;
  for (int i = 0; i < n; i++) {
    res += vec[i] * mat[n * i + i] * vec[i];
    res += 2 * vec[i] * mju_dot(mat + n * i, vec, i);
  }
  return res;
}

static void mulSymVec_old(mjtNum* __restrict res, const mjtNum* mat,
                          const mjtNum* vec, int n) {
  for (int i = 0; i < n; i++) {
    res[i] = mat[n * i + i] * vec[i] + mju_dot(mat + n * i, vec, i);
    mju_addToScl(res, mat + n * i, vec[i], i);
  }
}

ABSL_ATTRIBUTE_NOINLINE static int mju_boxQPoption_old(
    mjtNum* res, mjtNum* R, int* index, const mjtNum* H, const mjtNum* g, int n,
    const mjtNum* lower, const mjtNum* upper, int maxiter, mjtNum mingrad,
    mjtNum backtrack, mjtNum minstep, mjtNum armijo, char* log, int logsz) {
  int status = 0;
  int factorize = 1;
  int nfree = n;
  int nfactor = 0;
  mjtNum sdotg, improvement = 0, value = 0, norm2 = 0;

  if (n <= 0) {
    mju_error("problem size n must be positive");
  }
  if (upper && lower) {
    for (int i = 0; i < n; i++) {
      if (lower[i] >= upper[i]) {
        mju_error("upper bounds must be stricly larger than lower bounds");
      }
    }
  }

  mjtNum* scratch = R + n * n;
  mjtNum* grad = scratch + 0 * n;
  mjtNum* search = scratch + 1 * n;
  mjtNum* candidate = scratch + 2 * n;
  mjtNum* temp = scratch + 3 * n;
  int* clamped = reinterpret_cast<int*>(scratch + 4 * n);
  int* oldclamped = reinterpret_cast<int*>(scratch + 5 * n);
  if (!index) {
    index = reinterpret_cast<int*>(scratch + 6 * n);
  }

  if (!lower && !upper) {
    mju_copy(R, H, n * n);
    int rank = mju_cholFactor(R, n, mjMINVAL);
    if (rank == n) {
      mju_cholSolve(res, R, g, n);
      mju_scl(res, res, -1, n);
      nfactor = 1;
      status = 4;
    } else {
      status = -1;
    }
    for (int i = 0; i < n; i++) {
      index[i] = i;
    }
  } else {
    for (int i = 0; i < n; i++) {
      if (lower) res[i] = mju_max(res[i], lower[i]);
      if (upper) res[i] = mju_min(res[i], upper[i]);
    }
  }

  int iter, logptr = 0;
  mjtNum oldvalue;
  for (iter = 0; iter < maxiter; iter++) {
    if (status != 0) break;
    value = 0.5 * mulVecMatVecSym_old(res, H, n) + mju_dot(res, g, n);
    oldvalue = value;
    mulSymVec_old(grad, H, res, n);
    mju_addTo(grad, g, n);

    for (int i = 0; i < n; i++) {
      clamped[i] = (lower && res[i] == lower[i] && grad[i] > 0) ||
                   (upper && res[i] == upper[i] && grad[i] < 0);
    }

    nfree = 0;
    for (int i = 0; i < n; i++) {
      if (!clamped[i]) index[nfree++] = i;
    }
    if (!nfree) {
      status = 5;
      break;
    }

    if (iter) {
      factorize = 0;
      for (int i = 0; i < n; i++) {
        if (clamped[i] != oldclamped[i]) {
          factorize = 1;
          break;
        }
      }
    }
    for (int i = 0; i < n; i++) oldclamped[i] = clamped[i];

    for (int i = 0; i < n; i++) temp[i] = clamped[i] ? res[i] : 0;
    mulSymVec_old(search, H, temp, n);
    mju_addTo(search, g, n);
    for (int i = 0; i < nfree; i++) search[i] = search[index[i]];

    if (factorize) {
      for (int i = 0; i < nfree; i++) {
        for (int j = 0; j < i + 1; j++) {
          R[i * nfree + j] = H[index[i] * n + index[j]];
        }
      }
    }
    int rank = factorize ? mju_cholFactor(R, nfree, mjMINVAL) : nfree;
    nfactor += factorize;
    if (rank != nfree) {
      status = -1;
      break;
    }

    mju_cholSolve(temp, R, search, nfree);
    mju_zero(search, n);
    for (int i = 0; i < nfree; i++) search[index[i]] = -temp[i] - res[index[i]];

    norm2 = 0;
    for (int i = 0; i < nfree; i++) {
      mjtNum grad_i = grad[index[i]];
      norm2 += grad_i * grad_i;
    }
    if (norm2 < mingrad) {
      status = nfree == n ? 4 : 3;
      break;
    }
    if ((sdotg = mju_dot(search, grad, n)) >= 0) break;

    mjtNum step = 1;
    int nstep = 0;
    do {
      mju_scl(candidate, search, step, n);
      mju_addTo(candidate, res, n);
      for (int i = 0; i < n; i++) {
        if (lower && candidate[i] < lower[i])
          candidate[i] = lower[i];
        else if (upper && candidate[i] > upper[i])
          candidate[i] = upper[i];
      }
      value =
          0.5 * mulVecMatVecSym_old(candidate, H, n) + mju_dot(candidate, g, n);
      nstep++;
      step = step * backtrack;
      if (step < minstep) {
        status = 2;
        break;
      }
      improvement = (value - oldvalue) / (step * sdotg);
    } while (improvement < armijo);

    if (log && logptr < logsz) {
      logptr += 1;
    }

    mju_copy(res, candidate, n);
  }
  if (iter == maxiter) {
    status = 1;
  }
  (void)nfactor;
  return (status == 0 || status == -1) ? -1 : nfree;
}

ABSL_ATTRIBUTE_NOINLINE static int mju_boxQP_old(mjtNum* res, mjtNum* R,
                                                 int* index, const mjtNum* H,
                                                 const mjtNum* g, int n,
                                                 const mjtNum* lower,
                                                 const mjtNum* upper) {
  return mju_boxQPoption_old(res, R, index, H, g, n, lower, upper, 100, 1E-16,
                             0.5, 1E-22, 0.1, nullptr, 0);
}

static void BoxQPBenchmark(benchmark::State& state, int n, bool warmstart,
                           bool use_new) {
  static BoxQPDataset data10(10);
  static BoxQPDataset data30(30);
  static BoxQPDataset data100(100);
  static BoxQPDataset data300(300);
  const BoxQPDataset& d =
      (n == 10) ? data10 : (n == 30 ? data30 : (n == 100 ? data100 : data300));

  std::vector<mjtNum> res(n);
  std::vector<mjtNum> R(n * (n + 7));
  std::vector<int> index(n);
  int p = 0;

  for (auto s : state) {
    if (warmstart) {
      mju_copy(res.data(), d.warm.data() + p * n, n);
    } else {
      mju_zero(res.data(), n);
    }
    int nfree;
    if (use_new) {
      nfree = mju_boxQP(res.data(), R.data(), index.data(),
                        d.H.data() + p * n * n, d.g.data() + p * n, n,
                        d.lower.data() + p * n, d.upper.data() + p * n);
    } else {
      nfree = mju_boxQP_old(res.data(), R.data(), index.data(),
                            d.H.data() + p * n * n, d.g.data() + p * n, n,
                            d.lower.data() + p * n, d.upper.data() + p * n);
    }
    benchmark::DoNotOptimize(nfree);
    benchmark::DoNotOptimize(res.data());
    p = (p + 1) % BoxQPDataset::kNumProb;
  }

  state.SetItemsProcessed(state.iterations());
}

void BM_BoxQP_Old_Cold_10(benchmark::State& s) {
  BoxQPBenchmark(s, 10, false, false);
}
void BM_BoxQP_New_Cold_10(benchmark::State& s) {
  BoxQPBenchmark(s, 10, false, true);
}
void BM_BoxQP_Old_Cold_30(benchmark::State& s) {
  BoxQPBenchmark(s, 30, false, false);
}
void BM_BoxQP_New_Cold_30(benchmark::State& s) {
  BoxQPBenchmark(s, 30, false, true);
}
void BM_BoxQP_Old_Cold_100(benchmark::State& s) {
  BoxQPBenchmark(s, 100, false, false);
}
void BM_BoxQP_New_Cold_100(benchmark::State& s) {
  BoxQPBenchmark(s, 100, false, true);
}
void BM_BoxQP_Old_Cold_300(benchmark::State& s) {
  BoxQPBenchmark(s, 300, false, false);
}
void BM_BoxQP_New_Cold_300(benchmark::State& s) {
  BoxQPBenchmark(s, 300, false, true);
}
void BM_BoxQP_Old_Warm_10(benchmark::State& s) {
  BoxQPBenchmark(s, 10, true, false);
}
void BM_BoxQP_New_Warm_10(benchmark::State& s) {
  BoxQPBenchmark(s, 10, true, true);
}
void BM_BoxQP_Old_Warm_30(benchmark::State& s) {
  BoxQPBenchmark(s, 30, true, false);
}
void BM_BoxQP_New_Warm_30(benchmark::State& s) {
  BoxQPBenchmark(s, 30, true, true);
}
void BM_BoxQP_Old_Warm_100(benchmark::State& s) {
  BoxQPBenchmark(s, 100, true, false);
}
void BM_BoxQP_New_Warm_100(benchmark::State& s) {
  BoxQPBenchmark(s, 100, true, true);
}
void BM_BoxQP_Old_Warm_300(benchmark::State& s) {
  BoxQPBenchmark(s, 300, true, false);
}
void BM_BoxQP_New_Warm_300(benchmark::State& s) {
  BoxQPBenchmark(s, 300, true, true);
}
BENCHMARK(BM_BoxQP_Old_Cold_10);
BENCHMARK(BM_BoxQP_New_Cold_10);
BENCHMARK(BM_BoxQP_Old_Cold_30);
BENCHMARK(BM_BoxQP_New_Cold_30);
BENCHMARK(BM_BoxQP_Old_Cold_100);
BENCHMARK(BM_BoxQP_New_Cold_100);
BENCHMARK(BM_BoxQP_Old_Cold_300);
BENCHMARK(BM_BoxQP_New_Cold_300);
BENCHMARK(BM_BoxQP_Old_Warm_10);
BENCHMARK(BM_BoxQP_New_Warm_10);
BENCHMARK(BM_BoxQP_Old_Warm_30);
BENCHMARK(BM_BoxQP_New_Warm_30);
BENCHMARK(BM_BoxQP_Old_Warm_100);
BENCHMARK(BM_BoxQP_New_Warm_100);
BENCHMARK(BM_BoxQP_Old_Warm_300);
BENCHMARK(BM_BoxQP_New_Warm_300);

// ----------------------------- Banded Cholesky -------------------------------

struct CholBandDataset {
  static constexpr int kNumMat = 32;
  int ntotal;
  int nband;
  int ndense;
  int nmat;
  std::vector<mjtNum> B;
  std::vector<mjtNum> L;
  std::vector<mjtNum> vec;

  CholBandDataset(int ntotal_, int nband_, int ndense_)
      : ntotal(ntotal_),
        nband(nband_),
        ndense(ndense_),
        nmat((ntotal_ - ndense_) * nband_ + ndense_ * ntotal_),
        B(kNumMat * nmat),
        L(kNumMat * nmat),
        vec(kNumMat * ntotal_) {
    int nsparse = ntotal - ndense;
    std::vector<mjtNum> L0(ntotal * ntotal);
    std::vector<mjtNum> L0T(ntotal * ntotal);
    std::vector<mjtNum> H(ntotal * ntotal);

    for (int m = 0; m < kNumMat; m++) {
      std::mt19937_64 rng(100 + m);
      std::normal_distribution<double> dist(0, 1);
      mjtNum* Bp = B.data() + m * nmat;
      mjtNum* Lp = L.data() + m * nmat;
      mjtNum* vp = vec.data() + m * ntotal;

      mju_zero(L0.data(), ntotal * ntotal);
      for (int i = 0; i < ntotal; i++) {
        vp[i] = dist(rng);
        int jstart = (i < nsparse) ? mjMAX(0, i - nband + 1) : 0;
        for (int j = jstart; j < i; j++) {
          L0[i * ntotal + j] = dist(rng);
        }
        L0[i * ntotal + i] = std::abs(dist(rng)) + 1.0;
      }
      mju_transpose(L0T.data(), L0.data(), ntotal, ntotal);
      mju_mulMatTMat(H.data(), L0T.data(), L0T.data(), ntotal, ntotal, ntotal);
      mju_dense2Band(Bp, H.data(), ntotal, nband, ndense);
      mju_copy(Lp, Bp, nmat);
      mju_cholFactorBand(Lp, ntotal, nband, ndense, 0, 0);
    }
  }
};

static const CholBandDataset& GetCholBandDataset(int ntotal, int nband,
                                                 int ndense) {
  static const auto* d_100_6_0 = new CholBandDataset(100, 6, 0);
  static const auto* d_100_12_6 = new CholBandDataset(100, 12, 6);
  static const auto* d_300_6_0 = new CholBandDataset(300, 6, 0);
  static const auto* d_300_24_6 = new CholBandDataset(300, 24, 6);
  if (ntotal == 100 && nband == 6 && ndense == 0) return *d_100_6_0;
  if (ntotal == 100 && nband == 12 && ndense == 6) return *d_100_12_6;
  if (ntotal == 300 && nband == 6 && ndense == 0) return *d_300_6_0;
  if (ntotal == 300 && nband == 24 && ndense == 6) return *d_300_24_6;
  mju_error("GetCholBandDataset: unsupported (%d, %d, %d)", ntotal, nband,
            ndense);
  return *d_100_6_0;
}

static void CholFactorBandBenchmark(benchmark::State& state, int ntotal,
                                    int nband, int ndense) {
  const CholBandDataset& d = GetCholBandDataset(ntotal, nband, ndense);
  std::vector<mjtNum> mat(d.nmat);
  int m = 0;

  for (auto s : state) {
    mju_copy(mat.data(), d.B.data() + m * d.nmat, d.nmat);
    mjtNum mindiag =
        mju_cholFactorBand(mat.data(), ntotal, nband, ndense, 0, 0);
    benchmark::DoNotOptimize(mindiag);
    benchmark::DoNotOptimize(mat.data());
    m = (m + 1) % CholBandDataset::kNumMat;
  }

  state.SetItemsProcessed(state.iterations());
}

static void CholSolveBandBenchmark(benchmark::State& state, int ntotal,
                                   int nband, int ndense) {
  const CholBandDataset& d = GetCholBandDataset(ntotal, nband, ndense);
  std::vector<mjtNum> res(ntotal);
  int m = 0;

  for (auto s : state) {
    const mjtNum* Lp = d.L.data() + m * d.nmat;
    const mjtNum* vp = d.vec.data() + m * ntotal;
    mju_cholSolveBand(res.data(), Lp, vp, ntotal, nband, ndense);
    benchmark::DoNotOptimize(res.data());
    m = (m + 1) % CholBandDataset::kNumMat;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BandMulMatVecBenchmark(benchmark::State& state, int ntotal,
                                   int nband, int ndense) {
  const CholBandDataset& d = GetCholBandDataset(ntotal, nband, ndense);
  std::vector<mjtNum> res(ntotal);
  int m = 0;

  for (auto s : state) {
    const mjtNum* Bp = d.B.data() + m * d.nmat;
    const mjtNum* vp = d.vec.data() + m * ntotal;
    mju_bandMulMatVec(res.data(), Bp, vp, ntotal, nband, ndense, 1, true);
    benchmark::DoNotOptimize(res.data());
    m = (m + 1) % CholBandDataset::kNumMat;
  }

  state.SetItemsProcessed(state.iterations());
}

void BM_CholBand_Factor_100_6_0(benchmark::State& s) {
  CholFactorBandBenchmark(s, 100, 6, 0);
}
void BM_CholBand_Factor_100_12_6(benchmark::State& s) {
  CholFactorBandBenchmark(s, 100, 12, 6);
}
void BM_CholBand_Factor_300_6_0(benchmark::State& s) {
  CholFactorBandBenchmark(s, 300, 6, 0);
}
void BM_CholBand_Factor_300_24_6(benchmark::State& s) {
  CholFactorBandBenchmark(s, 300, 24, 6);
}

void BM_CholBand_Solve_100_6_0(benchmark::State& s) {
  CholSolveBandBenchmark(s, 100, 6, 0);
}
void BM_CholBand_Solve_100_12_6(benchmark::State& s) {
  CholSolveBandBenchmark(s, 100, 12, 6);
}
void BM_CholBand_Solve_300_6_0(benchmark::State& s) {
  CholSolveBandBenchmark(s, 300, 6, 0);
}
void BM_CholBand_Solve_300_24_6(benchmark::State& s) {
  CholSolveBandBenchmark(s, 300, 24, 6);
}

void BM_CholBand_MulMatVec_100_6_0(benchmark::State& s) {
  BandMulMatVecBenchmark(s, 100, 6, 0);
}
void BM_CholBand_MulMatVec_100_12_6(benchmark::State& s) {
  BandMulMatVecBenchmark(s, 100, 12, 6);
}
void BM_CholBand_MulMatVec_300_6_0(benchmark::State& s) {
  BandMulMatVecBenchmark(s, 300, 6, 0);
}
void BM_CholBand_MulMatVec_300_24_6(benchmark::State& s) {
  BandMulMatVecBenchmark(s, 300, 24, 6);
}

BENCHMARK(BM_CholBand_Factor_100_6_0);
BENCHMARK(BM_CholBand_Factor_100_12_6);
BENCHMARK(BM_CholBand_Factor_300_6_0);
BENCHMARK(BM_CholBand_Factor_300_24_6);
BENCHMARK(BM_CholBand_Solve_100_6_0);
BENCHMARK(BM_CholBand_Solve_100_12_6);
BENCHMARK(BM_CholBand_Solve_300_6_0);
BENCHMARK(BM_CholBand_Solve_300_24_6);
BENCHMARK(BM_CholBand_MulMatVec_100_6_0);
BENCHMARK(BM_CholBand_MulMatVec_100_12_6);
BENCHMARK(BM_CholBand_MulMatVec_300_6_0);
BENCHMARK(BM_CholBand_MulMatVec_300_24_6);

// ----------------------------- Dense Cholesky --------------------------------

struct CholDataset {
  static constexpr int kNumMat = 32;
  int n;
  std::vector<mjtNum> H;
  std::vector<mjtNum> L;
  std::vector<mjtNum> vec;

  explicit CholDataset(int n_)
      : n(n_), H(kNumMat * n_ * n_), L(kNumMat * n_ * n_), vec(kNumMat * n_) {
    std::vector<mjtNum> sqrtH(n * n);
    for (int m = 0; m < kNumMat; m++) {
      std::mt19937_64 rng(200 + m);
      std::normal_distribution<double> dist(0, 1);
      mjtNum* Hp = H.data() + m * n * n;
      mjtNum* Lp = L.data() + m * n * n;
      mjtNum* vp = vec.data() + m * n;

      for (int i = 0; i < n; i++) {
        vp[i] = dist(rng);
        for (int j = 0; j < n; j++) {
          sqrtH[i * n + j] = dist(rng);
        }
      }
      mju_mulMatTMat(Hp, sqrtH.data(), sqrtH.data(), n, n, n);
      for (int i = 0; i < n; i++) {
        Hp[i * (n + 1)] += 1.0;
      }
      mju_copy(Lp, Hp, n * n);
      mju_cholFactor(Lp, n, mjMINVAL);
    }
  }
};

static const CholDataset& GetCholDataset(int n) {
  static const auto* d_10 = new CholDataset(10);
  static const auto* d_30 = new CholDataset(30);
  static const auto* d_100 = new CholDataset(100);
  static const auto* d_300 = new CholDataset(300);
  if (n == 10) return *d_10;
  if (n == 30) return *d_30;
  if (n == 100) return *d_100;
  if (n == 300) return *d_300;
  mju_error("GetCholDataset: unsupported n = %d", n);
  return *d_10;
}

static void CholFactorBenchmark(benchmark::State& state, int n) {
  const CholDataset& d = GetCholDataset(n);
  std::vector<mjtNum> mat(n * n);
  int m = 0;

  for (auto s : state) {
    mju_copy(mat.data(), d.H.data() + m * n * n, n * n);
    int rank = mju_cholFactor(mat.data(), n, mjMINVAL);
    benchmark::DoNotOptimize(rank);
    benchmark::DoNotOptimize(mat.data());
    m = (m + 1) % CholDataset::kNumMat;
  }

  state.SetItemsProcessed(state.iterations());
}

static void CholSolveBenchmark(benchmark::State& state, int n) {
  const CholDataset& d = GetCholDataset(n);
  std::vector<mjtNum> res(n);
  int m = 0;

  for (auto s : state) {
    const mjtNum* Lp = d.L.data() + m * n * n;
    const mjtNum* vp = d.vec.data() + m * n;
    mju_cholSolve(res.data(), Lp, vp, n);
    benchmark::DoNotOptimize(res.data());
    m = (m + 1) % CholDataset::kNumMat;
  }

  state.SetItemsProcessed(state.iterations());
}

void BM_Chol_Factor_10(benchmark::State& s) { CholFactorBenchmark(s, 10); }
void BM_Chol_Factor_30(benchmark::State& s) { CholFactorBenchmark(s, 30); }
void BM_Chol_Factor_100(benchmark::State& s) { CholFactorBenchmark(s, 100); }
void BM_Chol_Factor_300(benchmark::State& s) { CholFactorBenchmark(s, 300); }

void BM_Chol_Solve_10(benchmark::State& s) { CholSolveBenchmark(s, 10); }
void BM_Chol_Solve_30(benchmark::State& s) { CholSolveBenchmark(s, 30); }
void BM_Chol_Solve_100(benchmark::State& s) { CholSolveBenchmark(s, 100); }
void BM_Chol_Solve_300(benchmark::State& s) { CholSolveBenchmark(s, 300); }

BENCHMARK(BM_Chol_Factor_10);
BENCHMARK(BM_Chol_Factor_30);
BENCHMARK(BM_Chol_Factor_100);
BENCHMARK(BM_Chol_Factor_300);
BENCHMARK(BM_Chol_Solve_10);
BENCHMARK(BM_Chol_Solve_30);
BENCHMARK(BM_Chol_Solve_100);
BENCHMARK(BM_Chol_Solve_300);

}  // namespace
}  // namespace mujoco
