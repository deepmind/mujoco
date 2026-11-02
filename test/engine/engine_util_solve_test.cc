// Copyright 2021 DeepMind Technologies Limited
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

// Tests for engine/engine_util_solve.c.

#include "src/engine/engine_util_solve.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_blas.h"
#include "src/engine/engine_util_misc.h"
#include "src/engine/engine_util_spatial.h"
#include "src/engine/engine_util_sparse.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::Pointwise;

using ::std::setw;
using ::std::string;
using ::std::vector;
using ::testing::ElementsAre;
using QCQP2Test = MujocoTest;

TEST_F(QCQP2Test, DegenerateAMatrix) {
  // A 2x2 matrix with determinant zero.
  const mjtNum Ain[9]{6, -15, 2, -5};

  // Any values will do for these three inputs.
  const mjtNum bin[3]{-12, 49};
  const mjtNum d[3]{11, 31};
  const mjtNum r = 0.01;

  // Make output array explicitly nonzero to simulate uninitialized memory.
  mjtNum res[2]{999, 999};

  EXPECT_EQ(mju_QCQP2(res, Ain, bin, d, r), 0);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[1], 0);
}

using QCQP3Test = MujocoTest;

TEST_F(QCQP3Test, DegenerateAMatrix) {
  // A 3x3 matrix with determinant zero.
  const mjtNum Ain[9]{1, 4, -2, -3, -7, 5, 2, -9, 0};

  // Any values will do for these three inputs.
  const mjtNum bin[3]{-12, 49, 8};
  const mjtNum d[3]{11, 31, -23};
  const mjtNum r = 0.1;

  // Make output array explicitly nonzero to simulate uninitialized memory.
  mjtNum res[3]{999, 999, 999};

  EXPECT_EQ(mju_QCQP3(res, Ain, bin, d, r), 0);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[1], 0);
  EXPECT_EQ(res[2], 0);
}

// --------------------------- mju_boxQP ---------------------------------------

using BoxQPTest = MujocoTest;

// utility: test if res is the minimum of a given box-QP problem
bool isQPminimum(const mjtNum* res, const mjtNum* H, const mjtNum* g, int n,
                 const mjtNum* lower, const mjtNum* upper) {
  const mjtNum eps = MjEps(1e-4, 5e-2);      // epsilon used for nudging
  const mjtNum threshold = MjTol(0, -2e-3);  // comparison threshold
  for (int i = 0; i < n; i++) {
    mjtNum grad_i = mju_dot(H + i * n, res, n) + g[i];
    mjtNum Hii = H[i * n + i];

    // nudge down
    mjtNum lo = res[i] - eps;
    if (lower) {
      lo = mju_max(lower[i], lo);
    }
    mjtNum dlo = lo - res[i];
    if (dlo * grad_i + 0.5 * Hii * dlo * dlo < threshold) {
      return false;
    }

    // nudge up
    mjtNum hi = res[i] + eps;
    if (upper) {
      hi = mju_min(upper[i], hi);
    }
    mjtNum dhi = hi - res[i];
    if (dhi * grad_i + 0.5 * Hii * dhi * dhi < threshold) {
      return false;
    }
  }
  return true;
}

// utility: define QP with pseudorandom values
void randomBoxQP(int n, mjtNum* H, mjtNum* g, mjtNum* lower, mjtNum* upper,
                 int seed) {
  // make distribution using seed
  std::mt19937_64 rng;
  rng.seed(seed);
  std::normal_distribution<double> dist(0, 1);

  // square root of H
  mjtNum* sqrtH = (mjtNum*)mju_malloc(sizeof(mjtNum) * n * n);

  for (int i = 0; i < n; i++) {
    g[i] = dist(rng);
    lower[i] = 5 * dist(rng);
    upper[i] = 5 * dist(rng);

    // fix invalid bounds
    if (lower[i] > upper[i]) {
      mjtNum tmp = upper[i];
      upper[i] = lower[i];
      lower[i] = tmp;
    }

    // sample temp
    for (int j = 0; j < n; j++) {
      sqrtH[n * i + j] = dist(rng);
    }
  }

  // make SPD matrix H
  mju_mulMatTMat(H, sqrtH, sqrtH, n, n, n);

  mju_free(sqrtH);
}

// test mju_boxQP on a small unbounded QP
TEST_F(BoxQPTest, UnboundedQP) {
  // small arrays, allocate on stack
  static const int n = 2;
  mjtNum H[n * n] = {2, 0, 0, 2};
  mjtNum g[n] = {1, 3};
  mjtNum res[n] = {0, 0};
  mjtNum R[n * (n + 7)];

  int nfree = mju_boxQP(res, R, /*index=*/nullptr, H, g, n,
                        /*lower=*/nullptr, /*upper=*/nullptr);

  // no bounds, expect Newton point
  EXPECT_EQ(nfree, 2);
  EXPECT_MJTNUM_EQ(res[0], -g[0] / H[0]);
  EXPECT_MJTNUM_EQ(res[1], -g[1] / H[3]);

  // check that solution is actual minimum
  EXPECT_TRUE(isQPminimum(res, H, g, n, /*lower=*/nullptr, /*upper=*/nullptr));

  // perturb solution, expected it no longer be the minimum
  res[0] += MjTol(0.001, 0.1);
  EXPECT_FALSE(isQPminimum(res, H, g, n, /*lower=*/nullptr, /*upper=*/nullptr));

  // negative-definite Hessian, no solution
  H[0] = -1;
  nfree = mju_boxQP(res, R, /*index=*/nullptr, H, g, n,
                    /*lower=*/nullptr, /*upper=*/nullptr);

  EXPECT_EQ(nfree, -1);
}

TEST_F(BoxQPTest, LineSearchFailureRejectsCandidateAndStops) {
  static const int n = 2;
  mjtNum H[n * n] = {100, 90, 90, 100};
  mjtNum g[n] = {-190, 0};
  mjtNum lower[n] = {-100, 0};
  mjtNum upper[n] = {100, 100};
  mjtNum res[n] = {0, 0};
  int index[n];
  mjtNum R[n * (n + 7)];
  char log[1024] = {0};

  // Clipped step=1 increases objective (+3100 > 0); backtrack=0.5 < minstep=0.6
  // fails line search.
  mju_boxQPoption(res, R, index, H, g, n, lower, upper, /*maxiter=*/5,
                  /*mingrad=*/MjTol(1e-12, 1e-5), /*backtrack=*/0.5,
                  /*minstep=*/0.6, /*armijo=*/0.1, log, sizeof(log));

  EXPECT_MJTNUM_EQ(res[0], 0);
  EXPECT_MJTNUM_EQ(res[1], 0);
  EXPECT_THAT(std::string(log),
              ::testing::HasSubstr("Maximum line-search iterations exceeded"));
}

TEST_F(BoxQPTest, LineSearchEvaluatesArmijoBeforeBacktrack) {
  static constexpr int n = 2;
  mjtNum H[n * n] = {100.0, -90.0, -90.0, 100.0};
  mjtNum g[n] = {-10.0, -10.0};
  mjtNum lower[n] = {-10.0, -10.0};
  mjtNum upper[n] = {0.58, 10.0};
  mjtNum res[n] = {0.0, 0.0};
  mjtNum R[n * (n + 7)] = {0.0};
  int index[n] = {-1, -1};

  // At step=1.0, clipped candidate (0.58, 1.0) has Armijo ratio 0.059 < 0.1;
  // backtracks to 0.5.
  int nfree = mju_boxQPoption(res, R, index, H, g, n, lower, upper,
                              /*maxiter=*/1, /*mingrad=*/MjTol(1e-12, 1e-5),
                              /*backtrack=*/0.5, /*minstep=*/MjTol(1e-10, 1e-6),
                              /*armijo=*/0.1, nullptr, 0);
  EXPECT_EQ(nfree, 2);
  EXPECT_MJTNUM_EQ(res[0], 0.5);
  EXPECT_MJTNUM_EQ(res[1], 0.5);
}

TEST_F(BoxQPTest, UnboundedUpperTrianglePoisoned) {
  static constexpr int n = 2;
  mjtNum H[n * n] = {4.0, -999.0, 2.0, 5.0};
  mjtNum g[n] = {-4.0, -2.0};
  mjtNum res[n] = {0.0, 0.0};
  mjtNum R[n * (n + 7)] = {77.0, 77.0, 77.0, 77.0};
  int index[n] = {-1, -1};

  int nfree = mju_boxQP(res, R, index, H, g, n, nullptr, nullptr);
  EXPECT_EQ(nfree, n);
  EXPECT_MJTNUM_EQ(R[1], 77.0);
}

// small bounded QP with asymmetric Hessian (upper triangle ignored)
TEST_F(BoxQPTest, AsymmetricUpperIgnored) {
  // small arrays, allocate on stack
  static const int n = 2;
  mjtNum H[n * n] = {1, -400, 0, 1};
  mjtNum g[n] = {1, 3};
  mjtNum res[n] = {0, 0};
  mjtNum lower[n] = {-2, -2};
  mjtNum upper[n] = {0, 0};
  int index[n];
  mjtNum R[n * (n + 7)];

  // solve box-QP
  int nfree = mju_boxQP(res, R, index, H, g, n, lower, upper);

  EXPECT_EQ(nfree, 1);

  EXPECT_MJTNUM_EQ(res[0], -g[0] / H[0]);
  EXPECT_MJTNUM_EQ(res[1], lower[1]);
}

// verify mju_boxQP reads only the lower triangle of H by poisoning the upper
// triangle with NaN and comparing to a clean symmetric solve
TEST_F(BoxQPTest, UpperTrianglePoisoned) {
  int n = 30;
  const mjtNum nan = std::numeric_limits<mjtNum>::quiet_NaN();

  // allocate on heap
  mjtNum *H, *g, *lower, *upper;  // inputs
  mjtNum *res, *R;                // outputs
  int* index;                     // outputs
  mju_boxQPmalloc(&res, &R, &index, &H, &g, n, &lower, &upper);

  struct Cleanup {
    mjtNum *res, *R, *H, *g, *lower, *upper;
    int* index;
    ~Cleanup() {
      mju_free(res);
      mju_free(R);
      mju_free(index);
      mju_free(H);
      mju_free(g);
      mju_free(lower);
      mju_free(upper);
    }
  } cleanup{res, R, H, g, lower, upper, index};

  // generate a symmetric SPD Hessian and bounded QP problem
  randomBoxQP(n, H, g, lower, upper, /*seed=*/1);

  int maxiter = 100;
  mjtNum mingrad = MjTol(1E-16, 1E-5);
  mjtNum backtrack = 0.5;
  mjtNum minstep = MjTol(1E-22, 1E-10);
  mjtNum armijo = 0.1;

  // solve with symmetric H to get the reference result
  mju_zero(res, n);
  int nfree_ref =
      mju_boxQPoption(res, R, index, H, g, n, lower, upper, maxiter, mingrad,
                      backtrack, minstep, armijo, nullptr, 0);
  ASSERT_GT(nfree_ref, -1);

  // save reference
  std::vector<mjtNum> res_ref(res, res + n);
  std::vector<int> index_ref(index, index + n);
  std::vector<mjtNum> R_ref(R, R + nfree_ref * nfree_ref);

  // poison the strict upper triangle of H with NaN
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      H[n * i + j] = nan;
    }
  }

  // solve again; result must match because only lower triangle should be read
  mju_zero(res, n);
  int nfree_poisoned =
      mju_boxQPoption(res, R, index, H, g, n, lower, upper, maxiter, mingrad,
                      backtrack, minstep, armijo, nullptr, 0);

  EXPECT_EQ(nfree_poisoned, nfree_ref);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(res[i], res_ref[i]) << "mismatch at index " << i;
  }
  for (int i = 0; i < nfree_ref; i++) {
    EXPECT_EQ(index[i], index_ref[i]) << "index mismatch at " << i;
    for (int j = 0; j <= i; j++) {
      int k = i * nfree_ref + j;
      EXPECT_EQ(R[k], R_ref[k]) << "R mismatch at row " << i << ", col " << j;
    }
  }
}

// test mju_boxQP on a single random bounded QP
TEST_F(BoxQPTest, BoundedQP) {
  int n = 50;  // problem size

  // allocate on heap
  mjtNum *H, *g, *lower, *upper;  // inputs
  mjtNum *res, *R;                // outputs
  int* index;                     // outputs
  mju_boxQPmalloc(&res, &R, &index, &H, &g, n, &lower, &upper);

  randomBoxQP(n, H, g, lower, upper, /*seed=*/1);

  // initialize res
  mju_zero(res, n);

  // use default options
  int maxiter = 100;  // maximum number of iterations
  mjtNum mingrad =
      MjTol(1E-16, 1E-5);  // minimum squared norm of (unclamped) gradient
  mjtNum backtrack = 0.5;  // backtrack factor for decreasing stepsize
  mjtNum minstep = MjTol(1E-22, 1E-10);  // minimum stepsize for linesearch
  mjtNum armijo = 0.1;                   // Armijo parameter

  // logging
  static const int logsz = 10000;
  char log[logsz];

  int nfree = mju_boxQPoption(res, R, index, H, g, n, lower, upper, maxiter,
                              mingrad, backtrack, minstep, armijo, log, logsz);

  // ADD_FAILURE() << log;  // uncomment to print `log` to error log

  // check solution
  EXPECT_GT(nfree, -1);
  EXPECT_TRUE(isQPminimum(res, H, g, n, lower, upper));

  // verify clamping
  int j = nfree > 0 ? 0 : -1;
  for (int i = 0; i < n; i++) {
    if (j >= 0 && i == index[j]) {  // free dimension
      EXPECT_GT(res[i], lower[i]);
      EXPECT_LT(res[i], upper[i]);
      j++;
    } else {  // clamped dimension
      EXPECT_TRUE(res[i] == lower[i] || res[i] == upper[i]);
    }
  }
  mju_free(res);
  mju_free(R);
  mju_free(index);
  mju_free(H);
  mju_free(g);
  mju_free(lower);
  mju_free(upper);
}

// test mju_boxQP on a set of random bounded QPs
TEST_F(BoxQPTest, BoundedQPvariations) {
  int nmax = 100;

  // allocate maximum size on heap
  mjtNum *H, *g, *lower, *upper;  // inputs
  mjtNum *res, *R;                // outputs
  int* index;                     // outputs
  mju_boxQPmalloc(&res, &R, &index, &H, &g, nmax, &lower, &upper);

  // logging
  static const int logsz = 10000;
  char log[logsz];

  int seed = 1;
  for (int n : {3, 30, 100}) {
    int count = 0;
    int factorizations = 0;
    for (mjtNum scaleH : {.01, 1.0, 100.0}) {
      for (mjtNum scaleg : {.01, 1.0, 100.0}) {
        for (mjtNum scalebounds : {.01, 1.0, 100.0}) {
          // make random box-QP
          randomBoxQP(n, H, g, lower, upper, seed++);

          mju_scl(H, H, scaleH, n * n);
          mju_scl(g, g, scaleg, n);
          mju_scl(lower, lower, scalebounds, n);
          mju_scl(upper, upper, scalebounds, n);

          // initialize with zeros
          mju_zero(res, n);

          // default algorithm options
          int maxiter = 100;
          mjtNum mingrad = MjTol(1E-16, 1E-5);
          mjtNum backtrack = 0.5;
          mjtNum minstep = MjTol(1E-22, 1E-10);
          mjtNum armijo = 0.1;

          // solve box-QP with logging
          int nfree =
              mju_boxQPoption(res, R, index, H, g, n, lower, upper, maxiter,
                              mingrad, backtrack, minstep, armijo, log, logsz);

          // check solution
          EXPECT_GT(nfree, -1) << log;
          EXPECT_TRUE(isQPminimum(res, H, g, n, lower, upper))
              << "n " << n << '\n'
              << "scaleH " << scaleH << '\n'
              << "scaleg " << scaleg << '\n'
              << "scalebounds " << scalebounds << '\n'
              << log;

          // wrap log with string, count factorizations
          string slog(log);
          string factorstr = "factorizations=";
          std::size_t index = slog.find(factorstr) + factorstr.length();
          int num_factor = std::stoi(slog.substr(index, 3));

          // never more than 6 factorizations
          EXPECT_LE(num_factor, 6);

          factorizations += num_factor;
          count++;
        }
      }
    }
    double meanfactor = ((double)factorizations) / count;

    // average of 4.5 factorizations is expected
    EXPECT_LE(meanfactor, 5.0);

    std::cerr << "n=" << setw(3) << n << ": average of " << meanfactor
              << " factorizations\n";
  }
  mju_free(res);
  mju_free(R);
  mju_free(index);
  mju_free(H);
  mju_free(g);
  mju_free(lower);
  mju_free(upper);
}

// ------------------------- band matrices -------------------------------------

using BandMatrixTest = MujocoTest;

// utility: random "arrowhead", banded-then-dense SPD matrix
//   optional random vector and diagonal regularizer
void randomBanded(mjtNum* H, int nTotal, int nBand, int nDense, int seed,
                  mjtNum* vec = nullptr, mjtNum reg = 0) {
  // make distribution using seed
  std::mt19937_64 rng;
  rng.seed(seed);
  std::normal_distribution<double> dist(0, 1);

  // allocate square root
  mjtNum* sqrtH = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal * nTotal);

  // sample
  for (int i = 0; i < nTotal; i++) {
    if (vec) vec[i] = dist(rng);
    for (int j = 0; j < nTotal; j++) {
      sqrtH[nTotal * i + j] = dist(rng);
    }
  }

  // make SPD matrix H
  mju_mulMatTMat(H, sqrtH, sqrtH, nTotal, nTotal, nTotal);

  // set zeros
  int nSparse = nTotal - nDense;
  for (int i = 0; i < nSparse; i++) {
    int nzeros = mjMAX(0, i + 1 - nBand);
    for (int j = 0; j < nzeros; j++) {
      H[nTotal * i + j] = 0;
      H[nTotal * j + i] = 0;
    }
  }

  // add regularizer to diagonal
  for (int i = 0; i < nTotal; i++) {
    H[nTotal * i + i] += reg;
  }

  mju_free(sqrtH);
}

// test banded-vector diagonal values
TEST_F(BandMatrixTest, Diagonal) {
  int seed = 1;
  int nTotal = 8;
  for (int nBand : {1, 3}) {
    for (int nDense : {0, 2}) {
      // allocate
      int nB = (nTotal - nDense) * nBand + nDense * nTotal;
      mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * nB);
      int nH = nTotal * nTotal;
      mjtNum* H = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);

      // make random banded SPD matrix, dense representation
      randomBanded(H, nTotal, nBand, nDense, seed++);

      // convert to banded representation
      mju_dense2Band(B, H, nTotal, nBand, nDense);

      // expect diagonals to be equal
      for (int i = 0; i < nTotal; i++) {
        EXPECT_EQ(H[i * nTotal + i], B[mju_bandDiag(i, nTotal, nBand, nDense)]);
      }

      mju_free(H);
      mju_free(B);
    }
  }
}

// test conversion of banded <-> dense
TEST_F(BandMatrixTest, Conversion) {
  int seed = 1;
  int nTotal = 8;
  for (int nBand : {0, 1, 3}) {
    for (int nDense : {0, 2}) {
      // allocate
      int nB = (nTotal - nDense) * nBand + nDense * nTotal;
      mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * nB);
      int nH = nTotal * nTotal;
      mjtNum* H = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);
      mjtNum* H1 = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);

      // make random banded SPD matrix, dense
      randomBanded(H, nTotal, nBand, nDense, seed++);

      // convert to banded
      mju_dense2Band(B, H, nTotal, nBand, nDense);

      // convert back to dense
      mju_band2Dense(H1, B, nTotal, nBand, nDense, /*flg_sym=*/1);

      // expect exact equality
      EXPECT_EQ(AsVector(H, nH), AsVector(H1, nH));

      mju_free(H1);
      mju_free(H);
      mju_free(B);
    }
  }
}

// test banded-vector multiplication
TEST_F(BandMatrixTest, Multiplication) {
  int seed = 1;
  int nTotal = 8;
  for (int nBand : {0, 1, 3}) {
    for (int nDense : {0, 2}) {
      // allocate
      int nB = (nTotal - nDense) * nBand + nDense * nTotal;
      mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * nB);
      int nH = nTotal * nTotal;
      mjtNum* H = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);
      mjtNum* vec = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);
      mjtNum* res = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);
      mjtNum* res1 = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);

      // make random banded SPD matrix, dense
      randomBanded(H, nTotal, nBand, nDense, seed++, vec);

      // multiply dense
      mju_mulMatVec(res, H, vec, nTotal, nTotal);

      // convert to banded, multiply
      mju_dense2Band(B, H, nTotal, nBand, nDense);
      mju_bandMulMatVec(res1, B, vec, nTotal, nBand, nDense,
                        /*nVec=*/1, /*flg_sym=*/1);

      // expect numerical equality
      mjtNum eps = MjTol(1e-12, 5e-5);
      EXPECT_THAT(AsVector(res, nTotal),
                  Pointwise(MjNear(eps, eps), AsVector(res1, nTotal)));

      mju_free(res1);
      mju_free(res);
      mju_free(vec);
      mju_free(H);
      mju_free(B);
    }
  }
}

// test banded factorization and vector product with factor
TEST_F(BandMatrixTest, Factorization) {
  int seed = 1;
  int nTotal = 8;
  for (int nBand : {1, 3}) {
    for (int nDense : {0, 2}) {
      for (mjtNum diagadd : {0.0, 1.0}) {
        for (int diagmul : {0.0, 1.3}) {
          // allocate
          int nB = (nTotal - nDense) * nBand + nDense * nTotal;
          mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * nB);
          int nH = nTotal * nTotal;
          mjtNum* H = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);
          mjtNum* H1 = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);
          mjtNum* vec = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);
          mjtNum* res = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);
          mjtNum* res1 = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);

          // make random banded matrix, dense representation
          //  add regularizer to ensure PD
          randomBanded(H, nTotal, nBand, nDense, seed++, vec, /*reg=*/nTotal);

          // convert to banded
          mju_dense2Band(B, H, nTotal, nBand, nDense);

          // apply diagadd and diagmul
          for (int i = 0; i < nTotal; i++) {
            H[nTotal * i + i] += diagadd + diagmul * H[nTotal * i + i];
          }

          // in-place dense factorization
          int rank = mju_cholFactor(H, nTotal, /*mindiag=*/0);

          // expect factorization to have succeeded
          EXPECT_EQ(rank, nTotal);

          // banded factorization
          mjtNum minDiag =
              mju_cholFactorBand(B, nTotal, nBand, nDense, diagadd, diagmul);

          // expect factorization to have succeeded
          EXPECT_GT(minDiag, 0);

          // convert back to dense, lower triangle only
          mju_band2Dense(H1, B, nTotal, nBand, nDense, /*flg_sym=*/0);

          // zero upper triangle of H (unused)
          for (int i = 0; i < nTotal - 1; i++) {
            mju_zero(H + nTotal * i + i + 1, nTotal - i - 1);
          }

          // expect numerical equality
          mjtNum eps = MjTol(1e-12, 5e-5);
          EXPECT_THAT(AsVector(H, nH),
                      Pointwise(MjNear(eps, eps), AsVector(H1, nH)));

          // multiply dense
          mju_mulMatVec(res, H, vec, nTotal, nTotal);

          // multiply sparse, only lower triangle
          mju_bandMulMatVec(res1, B, vec, nTotal, nBand, nDense,
                            /*nVec=*/1, /*flg_sym=*/0);

          // expect numerical equality
          EXPECT_THAT(AsVector(res, nTotal),
                      Pointwise(MjNear(eps, eps), AsVector(res1, nTotal)));

          mju_free(res1);
          mju_free(res);
          mju_free(vec);
          mju_free(H1);
          mju_free(H);
          mju_free(B);
        }
      }
    }
  }
}

// test banded solve
TEST_F(BandMatrixTest, Solve) {
  int seed = 1;
  int nTotal = 8;
  for (int nBand : {1, 3}) {
    for (int nDense : {0, 2}) {
      // allocate
      int nB = (nTotal - nDense) * nBand + nDense * nTotal;
      mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * nB);
      int nH = nTotal * nTotal;
      mjtNum* H = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);
      mjtNum* H1 = (mjtNum*)mju_malloc(sizeof(mjtNum) * nH);
      mjtNum* vec = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);
      mjtNum* res = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);
      mjtNum* res1 = (mjtNum*)mju_malloc(sizeof(mjtNum) * nTotal);

      // make random banded matrix, dense representation
      //  add regularizer to ensure PD
      randomBanded(H, nTotal, nBand, nDense, seed++, vec, /*reg=*/nTotal);

      // convert to banded
      mju_dense2Band(B, H, nTotal, nBand, nDense);

      // in-place dense factorization
      int rank = mju_cholFactor(H, nTotal, /*mindiag=*/0);

      // expect factorization to have succeeded
      EXPECT_EQ(rank, nTotal);

      // banded factorization
      mjtNum minDiag = mju_cholFactorBand(B, nTotal, nBand, nDense,
                                          /*diagadd=*/0, /*diagmul=*/0);

      // expect factorization to have succeeded
      EXPECT_GT(minDiag, 0);

      // convert back to dense, lower triangle only
      mju_band2Dense(H1, B, nTotal, nBand, nDense, /*flg_sym=*/0);

      // solve with dense
      mju_cholSolve(res, H, vec, nTotal);

      // solve with banded
      mju_cholSolveBand(res1, B, vec, nTotal, nBand, nDense);

      // expect numerical equality
      mjtNum eps = MjTol(1e-12, 5e-5);
      EXPECT_THAT(AsVector(res, nTotal),
                  Pointwise(MjNear(eps, eps), AsVector(res1, nTotal)));

      mju_free(res1);
      mju_free(res);
      mju_free(vec);
      mju_free(H1);
      mju_free(H);
      mju_free(B);
    }
  }
}

TEST_F(BandMatrixTest, CholFactorBandReturnsMinFactorizedDiag) {
  mjtNum B[3] = {9.0, 0.0, 4.0};
  mjtNum minDiag =
      mju_cholFactorBand(B, /*ntotal=*/2, /*nband=*/1, /*ndense=*/1,
                         /*diagadd=*/0, /*diagmul=*/0);
  EXPECT_MJTNUM_EQ(B[mju_bandDiag(0, 2, 1, 1)], 3.0);
  EXPECT_MJTNUM_EQ(B[mju_bandDiag(1, 2, 1, 1)], 2.0);
  EXPECT_MJTNUM_EQ(minDiag, 2.0);
}

using EngineUtilSolveTest = MujocoTest;

TEST_F(EngineUtilSolveTest, MjuCholFactorSymbolic) {
  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr d = MakeData(model);

  // Test matrix (upper triangular, representing symmetric matrix):
  //   [10  1  2  3]
  //   [ 0 10  0  0]
  //   [ 0  0 10  1]
  //   [ 0  0  0 10]
  //
  // mju_cholFactorSymbolic reads entries where col >= row

  int n = 4;
  mjtNum H[16] = {10, 1, 2, 3, 0, 10, 0, 0, 0, 0, 10, 1, 0, 0, 0, 10};

  // convert to sparse
  mjtNum sparseH[16];
  int H_rownnz[4], H_rowadr[4], H_colind[16];
  mju_dense2sparse(sparseH, H, n, n, H_rownnz, H_rowadr, H_colind, 16);

  // phase 1: counting
  int L_rownnz[4], L_rowadr[4];
  int LT_rownnz[4], LT_rowadr[4];
  int nnz = mju_cholFactorSymbolic(nullptr, L_rownnz, L_rowadr, nullptr,
                                   LT_rownnz, LT_rowadr, nullptr, H_rownnz,
                                   H_rowadr, H_colind, n, d.get());

  // verify counting phase outputs
  EXPECT_EQ(nnz, 8);
  // L (CSR) structure for reverse Cholesky (rows filled in reverse order)
  EXPECT_THAT(AsVector(L_rownnz, 4), ElementsAre(1, 2, 2, 3));
  EXPECT_THAT(AsVector(L_rowadr, 4), ElementsAre(0, 1, 3, 5));
  // LT (CSC) structure: transpose of L
  EXPECT_THAT(AsVector(LT_rownnz, 4), ElementsAre(4, 1, 2, 1));
  EXPECT_THAT(AsVector(LT_rowadr, 4), ElementsAre(0, 4, 5, 7));

  // phase 2: filling
  int L_colind[8], LT_colind[8], LT_pos[8];
  mju_cholFactorSymbolic(L_colind, L_rownnz, L_rowadr, LT_colind, LT_rownnz,
                         LT_rowadr, LT_pos, H_rownnz, H_rowadr, H_colind, n,
                         d.get());

  // verify L_colind
  EXPECT_THAT(AsVector(L_colind, 8), ElementsAre(0, 0, 1, 0, 2, 0, 2, 3));
  // Explanation (reverse Cholesky builds from bottom to top):
  //   Row 0 (1 entry): diagonal 0
  //   Row 1 (2 entries): col 0, then diagonal 1
  //   Row 2 (2 entries): col 0, then diagonal 2
  //   Row 3 (3 entries): col 0, col 2, then diagonal 3

  // verify LT_colind: transpose of L
  EXPECT_THAT(AsVector(LT_colind, 8), ElementsAre(0, 1, 2, 3, 1, 2, 3, 3));
  // Explanation:
  //   Column 0 (4 entries): rows 0, 1, 2, 3 (all have L[row, 0] != 0)
  //   Column 1 (1 entry): row 1
  //   Column 2 (2 entries): rows 2, 3
  //   Column 3 (1 entry): row 3

  // verify LT_pos: for each entry in LT, should point to correct position in L
  for (int c = 0; c < n; c++) {
    int adr = LT_rowadr[c];
    for (int k = 0; k < LT_rownnz[c]; k++) {
      int L_idx = LT_pos[adr + k];
      EXPECT_EQ(L_colind[L_idx], c)
          << "LT_pos mismatch at column " << c << ", entry " << k;
    }
  }
}

// Test for mju_cholUpdate: rank-one Cholesky update L*L' +/- x*x'
// Verifies that after applying the update, reconstructing H = L*L' produces
// the expected result H_original +/- x*x'.
TEST_F(EngineUtilSolveTest, MjuCholUpdate) {
  std::mt19937_64 rng;
  rng.seed(42);
  std::normal_distribution<double> dist(0, 1);

  for (int n : {4, 6, 8}) {
    vector<mjtNum> H(n * n);
    vector<mjtNum> H_expected(n * n);
    vector<mjtNum> L_dense(n * n);
    vector<mjtNum> sqrtH(n * n);
    vector<mjtNum> x(n);
    vector<mjtNum> x_copy(n);
    vector<mjtNum> H_reconstructed(n * n);

    for (int flg_plus : {0, 1}) {
      // generate random lower-triangular matrix for sqrtH
      mju_zero(sqrtH.data(), n * n);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
          sqrtH[n * i + j] = dist(rng);
        }
        sqrtH[n * i + i] = mju_abs(sqrtH[n * i + i]) + n;
      }

      // create SPD matrix H = sqrtH * sqrtH' (forward-order Cholesky: L*L')
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          mjtNum sum = 0;
          for (int k = 0; k <= mju_min(i, j); k++) {
            sum += sqrtH[n * i + k] * sqrtH[n * j + k];
          }
          H[n * i + j] = sum;
        }
      }

      // generate random update vector x
      for (int i = 0; i < n; i++) {
        x[i] = dist(rng) * 0.5;
      }

      // compute expected result: H_expected = H +/- x*x'
      mju_copy(H_expected.data(), H.data(), n * n);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (flg_plus) {
            H_expected[i * n + j] += x[i] * x[j];
          } else {
            H_expected[i * n + j] -= x[i] * x[j];
          }
        }
      }

      // test using dense Cholesky (forward-order: L*L')
      mju_copy(L_dense.data(), H.data(), n * n);

      // dense factorization
      int rank_factor = mju_cholFactor(L_dense.data(), n, 0);
      EXPECT_EQ(rank_factor, n) << "Initial factorization failed";

      // make copy of x for update (it's modified in-place)
      mju_copy(x_copy.data(), x.data(), n);

      // apply dense rank-one update
      int rank_update =
          mju_cholUpdate(L_dense.data(), x_copy.data(), n, flg_plus);
      EXPECT_EQ(rank_update, n)
          << "Dense update rank loss for n=" << n << ", flg_plus=" << flg_plus;

      // zero out upper triangle (Cholesky only uses lower triangle)
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          L_dense[i * n + j] = 0;
        }
      }

      // reconstruct H from L*L' and compare to expected
      mju_mulMatMatT(H_reconstructed.data(), L_dense.data(), L_dense.data(), n,
                     n, n);

      // compare
      mjtNum eps = MjTol(1e-8, 5e-4);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          EXPECT_NEAR(H_reconstructed[i * n + j], H_expected[i * n + j], eps)
              << "Dense mismatch at (" << i << "," << j << ") for n=" << n
              << ", flg_plus=" << flg_plus;
        }
      }
    }
  }
}

// Test for mju_cholUpdateSparse: sparse rank-one Cholesky update L'*L +/- x*x'
// Uses sparse reverse Cholesky factorization and sparse rank-one update.
TEST_F(EngineUtilSolveTest, MjuCholUpdateSparse) {
  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr d = MakeData(model);

  std::mt19937_64 rng;
  rng.seed(123);
  std::normal_distribution<double> dist(0, 1);

  for (int n : {4, 6, 8}) {
    int max_nnz = n * (n + 1) / 2;

    vector<mjtNum> H(n * n);
    vector<mjtNum> H_lower(n * n);
    vector<mjtNum> H_expected(n * n);
    vector<mjtNum> sqrtH(n * n);
    vector<mjtNum> L_sparse(max_nnz);
    vector<int> rownnz(n);
    vector<int> rowadr(n);
    vector<int> colind(max_nnz);
    vector<mjtNum> x(n);
    vector<mjtNum> x_sparse(n);
    vector<int> x_ind(n);
    vector<mjtNum> L_sparse_dense(n * n);
    vector<mjtNum> scratch(n);
    vector<mjtNum> H_sparse_reconstructed(n * n);

    for (int flg_plus : {0, 1}) {
      // generate random lower-triangular matrix for sqrtH
      mju_zero(sqrtH.data(), n * n);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
          sqrtH[n * i + j] = dist(rng);
        }
        sqrtH[n * i + i] = mju_abs(sqrtH[n * i + i]) + n;
      }

      // create SPD matrix H = sqrtH * sqrtH' (forward-order: L*L')
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          mjtNum sum = 0;
          for (int k = 0; k <= mju_min(i, j); k++) {
            sum += sqrtH[n * i + k] * sqrtH[n * j + k];
          }
          H[n * i + j] = sum;
        }
      }

      // generate random update vector x
      for (int i = 0; i < n; i++) {
        x[i] = dist(rng) * 0.5;
      }

      // compute expected result: H_expected = H +/- x*x'
      mju_copy(H_expected.data(), H.data(), n * n);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (flg_plus) {
            H_expected[i * n + j] += x[i] * x[j];
          } else {
            H_expected[i * n + j] -= x[i] * x[j];
          }
        }
      }

      // copy H to H_lower and zero upper triangle (sparse expects lower only)
      mju_copy(H_lower.data(), H.data(), n * n);
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          H_lower[i * n + j] = 0;
        }
      }

      // convert lower-triangular H to sparse format
      mju_dense2sparse(L_sparse.data(), H_lower.data(), n, n, rownnz.data(),
                       rowadr.data(), colind.data(), max_nnz);

      // prepare sparse update vector (all elements, fully dense)
      int x_nnz = n;
      mju_copy(x_sparse.data(), x.data(), n);
      for (int i = 0; i < n; i++) {
        x_ind[i] = i;
      }

      // sparse Cholesky factorization (reverse-order: L'*L)
      mju_cholFactorSparse(L_sparse.data(), n, 0, rownnz.data(), rowadr.data(),
                           colind.data(), d.get());

      // apply sparse rank-one update
      int rank_sparse = mju_cholUpdateSparse(
          L_sparse.data(), x_sparse.data(), n, flg_plus, rownnz.data(),
          rowadr.data(), colind.data(), x_nnz, x_ind.data(), scratch.data());
      EXPECT_EQ(rank_sparse, n)
          << "Sparse update rank loss for n=" << n << ", flg_plus=" << flg_plus;

      // reconstruct H from L'*L and compare to expected
      mju_sparse2dense(L_sparse_dense.data(), L_sparse.data(), n, n,
                       rownnz.data(), rowadr.data(), colind.data());
      mju_mulMatTMat(H_sparse_reconstructed.data(), L_sparse_dense.data(),
                     L_sparse_dense.data(), n, n, n);

      // compare
      mjtNum eps = MjTol(1e-8, 5e-4);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          EXPECT_NEAR(H_sparse_reconstructed[i * n + j], H_expected[i * n + j],
                      eps)
              << "Sparse mismatch at (" << i << "," << j << ") for n=" << n
              << ", flg_plus=" << flg_plus;
        }
      }
    }
  }
}

// Test that mju_cholFactorSymbolic + mju_cholFactorNumeric produces identical
// results to the reference implementation mju_cholFactorSparse
TEST_F(EngineUtilSolveTest, CholFactorSymbolicNumeric) {
  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr d = MakeData(model);

  // test matrix with fill-in: upper triangle structure
  int n = 4;
  mjtNum H[16] = {10, 1, 2, 3, 1, 10, 0, 0, 2, 0, 10, 1, 3, 0, 1, 10};

  // convert to sparse (lower triangle only)
  mjtNum sparseH[16];
  int H_rownnz[4], H_rowadr[4], H_colind[16];
  mju_dense2sparse(sparseH, H, n, n, H_rownnz, H_rowadr, H_colind, 16);

  // transpose for upper triangle (needed by cholFactorSymbolic)
  int HT_rownnz[4], HT_rowadr[4], HT_colind[16];
  mju_transposeSparse(nullptr, nullptr, n, n, HT_rownnz, HT_rowadr, HT_colind,
                      nullptr, H_rownnz, H_rowadr, H_colind);

  // count fill-in (also computes LT structure)
  int L_rownnz[4], L_rowadr[4];
  int LT_rownnz[4], LT_rowadr[4];
  int nnz = mju_cholFactorSymbolic(nullptr, L_rownnz, L_rowadr, nullptr,
                                   LT_rownnz, LT_rowadr, nullptr, HT_rownnz,
                                   HT_rowadr, HT_colind, n, d.get());

  // filling phase: compute L_colind, LT_colind, and LT_pos
  int L_colind[16], LT_colind[16], LT_pos[16];
  mju_cholFactorSymbolic(L_colind, L_rownnz, L_rowadr, LT_colind, LT_rownnz,
                         LT_rowadr, LT_pos, HT_rownnz, HT_rowadr, HT_colind, n,
                         d.get());

  // verify LT structure matches what we'd get from a separate transpose
  int LT_rownnz_ref[4], LT_rowadr_ref[4], LT_colind_ref[16];
  mju_transposeSparse(nullptr, nullptr, n, n, LT_rownnz_ref, LT_rowadr_ref,
                      LT_colind_ref, nullptr, L_rownnz, L_rowadr, L_colind);

  // LT structure should match
  EXPECT_THAT(AsVector(LT_rownnz, 4),
              ElementsAre(LT_rownnz_ref[0], LT_rownnz_ref[1], LT_rownnz_ref[2],
                          LT_rownnz_ref[3]));
  EXPECT_THAT(AsVector(LT_rowadr, 4),
              ElementsAre(LT_rowadr_ref[0], LT_rowadr_ref[1], LT_rowadr_ref[2],
                          LT_rowadr_ref[3]));

  // verify LT_colind and LT_pos match
  for (int r = 0; r < n; r++) {
    int adr = LT_rowadr[r];
    for (int k = 0; k < LT_rownnz[r]; k++) {
      int L_idx = LT_pos[adr + k];  // index in L array
      // verify L_colind at this position is indeed r
      EXPECT_EQ(L_colind[L_idx], r)
          << "LT_pos mismatch at LT[" << r << ", " << k << "]";
    }
  }

  // numeric factorization using new function
  mjtNum L_new[16], numeric_scratch[4];
  int rank_new =
      mju_cholFactorNumeric(L_new, n, 1e-10, L_rownnz, L_rowadr, L_colind,
                            LT_rownnz, LT_rowadr, LT_colind, LT_pos, sparseH,
                            H_rownnz, H_rowadr, H_colind, numeric_scratch);

  // reference implementation: copy sparse H into L_ref, then factor in-place
  mjtNum L_ref[16];
  int L_ref_rownnz[4], L_ref_colind[16];
  for (int r = 0; r < n; r++) {
    int nnz_r = H_rownnz[r];
    // count lower triangle elements for this row
    int lower_nnz = 0;
    for (int i = 0; i < nnz_r; i++) {
      if (H_colind[H_rowadr[r] + i] <= r) {
        L_ref[L_rowadr[r] + lower_nnz] = sparseH[H_rowadr[r] + i];
        L_ref_colind[L_rowadr[r] + lower_nnz] = H_colind[H_rowadr[r] + i];
        lower_nnz++;
      }
    }
    L_ref_rownnz[r] = lower_nnz;
  }
  int rank_ref = mju_cholFactorSparse(L_ref, n, 1e-10, L_ref_rownnz, L_rowadr,
                                      L_ref_colind, d.get());

  // compare results
  EXPECT_EQ(rank_new, rank_ref);
  EXPECT_EQ(rank_new, n);

  // compare L values
  mjtNum eps = MjTol(1e-10, 5e-5);
  for (int i = 0; i < nnz; i++) {
    EXPECT_NEAR(L_new[i], L_ref[i], eps) << "mismatch at index " << i;
  }
}

// A pivot lost to rounding leaves its row's coupling in place. The clamp must
// decouple the row, as mju_cholFactor does: dividing the coupling by
// sqrt(mindiag) would drive every later pivot of the factorization negative.
TEST_F(EngineUtilSolveTest, CholFactorNumericClampDecouples) {
  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr d = MakeData(model);

  // reverse elimination: row 2 has pivot 1, leaving row 1 with pivot 0 and
  // coupling 0.5 to row 0, whose pivot is 1 once row 1 is decoupled
  constexpr int n = 3;
  mjtNum H[n * n] = {1, .5, 0, .5, 1, 1, 0, 1, 1};
  mjtNum sparseH[n * n];
  int H_rownnz[n], H_rowadr[n], H_colind[n * n];
  mju_dense2sparse(sparseH, H, n, n, H_rownnz, H_rowadr, H_colind, n * n);

  // symbolic factorization
  int L_rownnz[n], L_rowadr[n], LT_rownnz[n], LT_rowadr[n];
  int nnz = mju_cholFactorSymbolic(nullptr, L_rownnz, L_rowadr, nullptr,
                                   LT_rownnz, LT_rowadr, nullptr, H_rownnz,
                                   H_rowadr, H_colind, n, d.get());
  int L_colind[n * n], LT_colind[n * n], LT_map[n * n];
  mju_cholFactorSymbolic(L_colind, L_rownnz, L_rowadr, LT_colind, LT_rownnz,
                         LT_rowadr, LT_map, H_rownnz, H_rowadr, H_colind, n,
                         d.get());

  // numeric factorization: only row 1 is clamped
  mjtNum L[n * n], scratch[n];
  int rank = mju_cholFactorNumeric(
      L, n, mjMINVAL, L_rownnz, L_rowadr, L_colind, LT_rownnz, LT_rowadr,
      LT_colind, LT_map, sparseH, H_rownnz, H_rowadr, H_colind, scratch);
  EXPECT_EQ(rank, n - 1);

  // row 1 is decoupled: clamped diagonal, zero off-diagonals
  int adr1 = L_rowadr[1], nnz1 = L_rownnz[1];
  EXPECT_EQ(L[adr1 + nnz1 - 1], mju_sqrt(mjMINVAL));
  for (int i = 0; i < nnz1 - 1; i++) {
    EXPECT_EQ(L[adr1 + i], 0);
  }

  // row 0 keeps its pivot, every entry is bounded
  EXPECT_EQ(L[L_rowadr[0] + L_rownnz[0] - 1], 1);
  for (int i = 0; i < nnz; i++) {
    EXPECT_LE(mju_abs(L[i]), 1) << "entry " << i;
  }
}

// NaN pivots must trigger rank-deficiency clamping and row decoupling rather
// than bypassing `diag < mindiag` and poisoning the factor with NaN while
// reporting full rank.
TEST_F(EngineUtilSolveTest, CholFactorNaNPivotClamps) {
  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr d = MakeData(model);
  const mjtNum nan = std::numeric_limits<mjtNum>::quiet_NaN();

  // rows {0, 1, 2} form a 3-row supernode; row 3 is an isolated leaf
  constexpr int n = 4;
  mjtNum H[n * n] = {
      4, 1, 1, 0, 1, 4, 1, 0, 1, 1, 4, 0, 0, 0, 0, 4,
  };
  mjtNum sparseH[n * n];
  int H_rownnz[n], H_rowadr[n], H_colind[n * n];
  mju_dense2sparse(sparseH, H, n, n, H_rownnz, H_rowadr, H_colind, n * n);

  int L_rownnz[n], L_rowadr[n], LT_rownnz[n], LT_rowadr[n];
  int nnz = mju_cholFactorSymbolic(nullptr, L_rownnz, L_rowadr, nullptr,
                                   LT_rownnz, LT_rowadr, nullptr, H_rownnz,
                                   H_rowadr, H_colind, n, d.get());
  int L_colind[n * n], LT_colind[n * n], LT_map[n * n];
  mju_cholFactorSymbolic(L_colind, L_rownnz, L_rowadr, LT_colind, LT_rownnz,
                         LT_rowadr, LT_map, H_rownnz, H_rowadr, H_colind, n,
                         d.get());

  // poison supernode pivot (1, 1) and leaf pivot (3, 3) with NaN
  H[1 * n + 1] = nan;
  H[3 * n + 3] = nan;
  for (int r : {1, 3}) {
    for (int k = H_rowadr[r]; k < H_rowadr[r] + H_rownnz[r]; k++) {
      if (H_colind[k] == r) {
        sparseH[k] = nan;
      }
    }
  }

  // dense mju_cholFactor
  mjtNum L_dense[n * n];
  mju_copy(L_dense, H, n * n);
  EXPECT_EQ(mju_cholFactor(L_dense, n, mjMINVAL), n - 2);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      EXPECT_FALSE(mju_isBad(L_dense[i * n + j]))
          << "dense (" << i << ", " << j << ")";
    }
  }
  EXPECT_EQ(L_dense[1 * n + 1], mju_sqrt(mjMINVAL));
  EXPECT_EQ(L_dense[3 * n + 3], mju_sqrt(mjMINVAL));

  // sparse mju_cholFactorNumeric (hits both leaf fast path and supernode path)
  mjtNum L_num[n * n], scratch[n];
  EXPECT_EQ(
      mju_cholFactorNumeric(L_num, n, mjMINVAL, L_rownnz, L_rowadr, L_colind,
                            LT_rownnz, LT_rowadr, LT_colind, LT_map, sparseH,
                            H_rownnz, H_rowadr, H_colind, scratch),
      n - 2);
  for (int i = 0; i < nnz; i++) {
    EXPECT_FALSE(mju_isBad(L_num[i])) << "numeric entry " << i;
  }
  EXPECT_EQ(L_num[L_rowadr[1] + L_rownnz[1] - 1], mju_sqrt(mjMINVAL));
  EXPECT_EQ(L_num[L_rowadr[1]], 0);
  EXPECT_EQ(L_num[L_rowadr[3] + L_rownnz[3] - 1], mju_sqrt(mjMINVAL));

  // sparse reference mju_cholFactorSparse
  mjtNum L_ref[n * n];
  int L_ref_rownnz[n], L_ref_colind[n * n];
  for (int r = 0; r < n; r++) {
    int lower_nnz = 0;
    for (int k = 0; k < H_rownnz[r]; k++) {
      int c = H_colind[H_rowadr[r] + k];
      if (c <= r) {
        L_ref[L_rowadr[r] + lower_nnz] = sparseH[H_rowadr[r] + k];
        L_ref_colind[L_rowadr[r] + lower_nnz] = c;
        lower_nnz++;
      }
    }
    L_ref_rownnz[r] = lower_nnz;
  }
  EXPECT_EQ(mju_cholFactorSparse(L_ref, n, mjMINVAL, L_ref_rownnz, L_rowadr,
                                 L_ref_colind, d.get()),
            n - 2);
  for (int r = 0; r < n; r++) {
    for (int k = 0; k < L_ref_rownnz[r]; k++) {
      EXPECT_FALSE(mju_isBad(L_ref[L_rowadr[r] + k]))
          << "sparse (" << r << ", " << k << ")";
    }
  }

  // banded mju_cholFactorBand returns 0 on NaN pivot
  mjtNum B[n] = {2, nan, 2, 2};
  EXPECT_EQ(mju_cholFactorBand(B, n, /*nband=*/1, /*ndense=*/0, 0, 0), 0);
}

// ----------------------------- dense LU --------------------------------------

// mju_cholFactorSymbolicBlocked on a block pattern computes the same L arrays,
// entry for entry, as mju_cholFactorSymbolic on the expanded dof pattern, and
// the block transpose of its LT arrays
TEST_F(EngineUtilSolveTest, CholFactorSymbolicBlocked) {
  std::mt19937 rng(7);
  for (int bs = 1; bs <= 3; bs++) {
    for (int trial = 0; trial < 20; trial++) {
      int nb = 1 + trial;
      std::uniform_real_distribution<double> unif(0, 1);
      double density = trial % 2 ? 0.15 : 0.4;

      // a random symmetric block pattern, columns ascending, diagonal listed
      // on odd trials only
      vector<vector<int>> rows(nb);
      vector<vector<bool>> adj(nb, vector<bool>(nb, false));
      for (int p = 0; p < nb; p++) {
        for (int q = p + 1; q < nb; q++) {
          if (unif(rng) < density) adj[p][q] = adj[q][p] = true;
        }
      }
      vector<int> rownnz(nb), rowadr(nb), colind;
      for (int p = 0; p < nb; p++) {
        rowadr[p] = colind.size();
        for (int q = 0; q < nb; q++) {
          if (adj[p][q] || (q == p && trial % 2)) colind.push_back(q);
        }
        rownnz[p] = colind.size() - rowadr[p];
      }

      // the expanded dof pattern: dense blocks, dense diagonal blocks
      int n = bs * nb;
      vector<int> drownnz(n), drowadr(n), dcolind;
      for (int r = 0; r < n; r++) {
        drowadr[r] = dcolind.size();
        for (int c = 0; c < n; c++) {
          if (r / bs == c / bs || adj[r / bs][c / bs]) dcolind.push_back(c);
        }
        drownnz[r] = dcolind.size() - drowadr[r];
      }

      // counting
      vector<int> L_rownnz(n), L_rowadr(n), LT_rownnz(n), LT_rowadr(n);
      vector<int> bL_rownnz(n), bL_rowadr(n), BT_adr(nb + 1);
      vector<int> scratch(3 * nb);
      int nnz = mju_cholFactorSymbolic(
          nullptr, L_rownnz.data(), L_rowadr.data(), nullptr, LT_rownnz.data(),
          LT_rowadr.data(), nullptr, drownnz.data(), drowadr.data(),
          dcolind.data(), n, nullptr);
      int bnnz = mju_cholFactorSymbolicBlocked(
          nullptr, bL_rownnz.data(), bL_rowadr.data(), BT_adr.data(), nullptr,
          nullptr, rownnz.data(), rowadr.data(), colind.data(), nb, bs,
          scratch.data());
      ASSERT_EQ(bnnz, nnz) << "bs " << bs << " trial " << trial;
      EXPECT_EQ(bL_rownnz, L_rownnz);
      EXPECT_EQ(bL_rowadr, L_rowadr);

      // filling
      vector<int> L_colind(nnz), LT_colind(nnz), LT_map(nnz);
      vector<int> bL_colind(nnz), BT_row(BT_adr[nb]), BT_off(BT_adr[nb]);
      mju_cholFactorSymbolic(L_colind.data(), L_rownnz.data(), L_rowadr.data(),
                             LT_colind.data(), LT_rownnz.data(),
                             LT_rowadr.data(), LT_map.data(), drownnz.data(),
                             drowadr.data(), dcolind.data(), n, nullptr);
      mju_cholFactorSymbolicBlocked(
          bL_colind.data(), L_rownnz.data(), L_rowadr.data(), BT_adr.data(),
          BT_row.data(), BT_off.data(), rownnz.data(), rowadr.data(),
          colind.data(), nb, bs, scratch.data());
      EXPECT_EQ(bL_colind, L_colind) << "bs " << bs << " trial " << trial;

      // the block transpose: the first dof of each block row below the
      // diagonal block in the transpose row of the first dof of each column
      vector<int> eBT_adr(nb + 1, 0), eBT_row, eBT_off;
      for (int q = 0; q < nb; q++) {
        eBT_adr[q] = eBT_row.size();
        for (int e = LT_rowadr[bs * q];
             e < LT_rowadr[bs * q] + LT_rownnz[bs * q]; e++) {
          int r = LT_colind[e];
          if (r >= bs * (q + 1) && r % bs == 0) {
            eBT_row.push_back(r / bs);
            eBT_off.push_back(LT_map[e] - L_rowadr[r]);
          }
        }
      }
      eBT_adr[nb] = eBT_row.size();
      EXPECT_EQ(BT_adr, eBT_adr) << "bs " << bs << " trial " << trial;
      EXPECT_EQ(BT_row, eBT_row) << "bs " << bs << " trial " << trial;
      EXPECT_EQ(BT_off, eBT_off) << "bs " << bs << " trial " << trial;
    }
  }
}

using DenseLUTest = MujocoTest;

// factor identity, solve recovers b exactly
TEST_F(DenseLUTest, Identity) {
  constexpr int n = 4;
  mjtNum A[n * n] = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  };
  int pivot[n];
  mjtNum b[n] = {1, 2, 3, 4};
  mjtNum x[n];

  EXPECT_EQ(mju_factorLU(A, n, pivot), 1);
  mju_solveLU(x, A, b, pivot, n);

  for (int i = 0; i < n; i++) {
    EXPECT_MJTNUM_EQ(x[i], b[i]);
  }
}

// 3x3 with known solution
TEST_F(DenseLUTest, SmallKnown) {
  constexpr int n = 3;
  // A = [2 1 1; 4 3 3; 8 7 9], b = [1; 1; 1]
  // solution: x = [1; -1; 0]  (verified: A*x = [2-1; 4-3; 8-7] = [1;1;1])
  mjtNum A[n * n] = {
      2, 1, 1, 4, 3, 3, 8, 7, 9,
  };
  int pivot[n];
  mjtNum b[n] = {1, 1, 1};
  mjtNum x[n];

  EXPECT_EQ(mju_factorLU(A, n, pivot), 1);
  mju_solveLU(x, A, b, pivot, n);

  mjtNum eps = MjTol(1e-14, 1e-6);
  EXPECT_NEAR(x[0], 1, eps);
  EXPECT_NEAR(x[1], -1, eps);
  EXPECT_NEAR(x[2], 0, eps);
}

// random SPD matrices: compare LU solve against Cholesky solve
TEST_F(DenseLUTest, RandomSPD) {
  std::mt19937_64 rng;
  rng.seed(7);
  std::normal_distribution<double> dist(0, 1);

  for (int n : {4, 8, 16}) {
    vector<mjtNum> sqrtH(n * n);
    vector<mjtNum> A(n * n);
    vector<mjtNum> A_chol(n * n);
    vector<mjtNum> b(n);
    vector<mjtNum> x_lu(n);
    vector<mjtNum> x_chol(n);
    vector<int> pivot(n);

    // generate random SPD matrix
    for (int i = 0; i < n * n; i++) sqrtH[i] = dist(rng);
    mju_mulMatTMat(A.data(), sqrtH.data(), sqrtH.data(), n, n, n);

    // add diagonal regularizer
    for (int i = 0; i < n; i++) A[i * n + i] += n;

    // generate random rhs
    for (int i = 0; i < n; i++) b[i] = dist(rng);

    // solve with Cholesky
    mju_copy(A_chol.data(), A.data(), n * n);
    int rank = mju_cholFactor(A_chol.data(), n, 0);
    EXPECT_EQ(rank, n);
    mju_cholSolve(x_chol.data(), A_chol.data(), b.data(), n);

    // solve with LU
    int ok = mju_factorLU(A.data(), n, pivot.data());
    EXPECT_EQ(ok, 1);
    mju_solveLU(x_lu.data(), A.data(), b.data(), pivot.data(), n);

    // compare
    mjtNum eps = MjTol(1e-15, 1e-7);
    EXPECT_THAT(AsVector(x_lu.data(), n),
                Pointwise(MjNear(eps, eps), AsVector(x_chol.data(), n)));
  }
}

// random non-symmetric matrices: verify A*x == b
TEST_F(DenseLUTest, RandomGeneral) {
  std::mt19937_64 rng;
  rng.seed(42);
  std::normal_distribution<double> dist(0, 1);

  for (int n : {3, 5, 10, 20}) {
    vector<mjtNum> A(n * n);
    vector<mjtNum> A_orig(n * n);
    vector<mjtNum> b(n);
    vector<mjtNum> x(n);
    vector<mjtNum> Ax(n);
    vector<int> pivot(n);

    // random non-symmetric matrix with diagonal dominance
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        A[i * n + j] = dist(rng);
      }
      A[i * n + i] += 2 * n;
    }
    mju_copy(A_orig.data(), A.data(), n * n);

    // random rhs
    for (int i = 0; i < n; i++) b[i] = dist(rng);

    // factor and solve
    int ok = mju_factorLU(A.data(), n, pivot.data());
    EXPECT_EQ(ok, 1);
    mju_solveLU(x.data(), A.data(), b.data(), pivot.data(), n);

    // verify: A_orig * x == b
    mju_mulMatVec(Ax.data(), A_orig.data(), x.data(), n, n);

    mjtNum eps = MjTol(1e-14, 1e-5);
    EXPECT_THAT(AsVector(Ax.data(), n),
                Pointwise(MjNear(eps, eps), AsVector(b.data(), n)));
  }
}

// near-singular matrix returns 0
TEST_F(DenseLUTest, Singular) {
  constexpr int n = 3;
  // all zeros: maximally singular
  mjtNum A[n * n] = {0};
  int pivot[n];

  EXPECT_EQ(mju_factorLU(A, n, pivot), 0);
}

// 6x6 specialization: identical results to the generic version
TEST_F(DenseLUTest, Fixed6MatchesGeneric) {
  constexpr int n = 6;
  std::mt19937_64 rng;
  rng.seed(3);
  std::normal_distribution<double> dist(0, 1);

  for (int trial = 0; trial < 100; trial++) {
    mjtNum A[n * n], A6[n * n], b[n], x[n], x6[n];
    int pivot[n], pivot6[n];

    // random asymmetric matrix; first trial zeroes the diagonal to force
    // pivoting
    for (int i = 0; i < n * n; i++) {
      A[i] = dist(rng);
    }
    if (trial == 0) {
      for (int i = 0; i < n; i++) {
        A[i * n + i] = 0;
      }
    }
    for (int i = 0; i < n; i++) {
      b[i] = dist(rng);
    }
    mju_copy(A6, A, n * n);

    // same algorithm; results agree to rounding (not bitwise: the compiler may
    // contract the unrolled fixed-size version differently)
    int rank = mju_factorLU(A, n, pivot);
    int rank6 = mju_factorLU6(A6, pivot6);
    ASSERT_EQ(rank, rank6);
    if (!rank) {
      continue;
    }
    for (int i = 0; i < n; i++) {
      EXPECT_EQ(pivot[i], pivot6[i]);
    }
    for (int i = 0; i < n * n; i++) {
      EXPECT_NEAR(A[i], A6[i], MjTol(1e-13, 1e-4));
    }

    mju_solveLU(x, A, b, pivot, n);
    mju_solveLU6(x6, A6, b, pivot6);
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(x[i], x6[i], MjTol(1e-12, 1e-3));
    }
  }
}

// nonsingular matrix with all-zero diagonal: requires pivoting
TEST_F(DenseLUTest, ZeroDiagonal) {
  constexpr int n = 3;
  // A = [0 2 1; 1 0 3; 4 1 0], det(A) = 25
  mjtNum A[n * n] = {
      0, 2, 1, 1, 0, 3, 4, 1, 0,
  };
  mjtNum A_orig[n * n];
  mju_copy(A_orig, A, n * n);
  int pivot[n];
  mjtNum b[n] = {1, 2, 3};
  mjtNum x[n], Ax[n];

  EXPECT_EQ(mju_factorLU(A, n, pivot), 1);
  mju_solveLU(x, A, b, pivot, n);

  // verify A_orig * x == b
  mju_mulMatVec(Ax, A_orig, x, n, n);
  mjtNum eps = MjTol(1e-14, 1e-6);
  EXPECT_THAT(AsVector(Ax, n), Pointwise(MjNear(eps, eps), AsVector(b, n)));
}

// --------------------------------- mju_eig3 ----------------------------------

using Eig3Test = MujocoTest;

// residuals of the decomposition mat = eigvec * diag(eigval) * eigvec':
// reconstruction error relative to the largest element of mat, and deviation of
// eigvec from orthonormality
static void Eig3Residual(mjtNum* recon, mjtNum* orth, const mjtNum mat[9],
                         const mjtNum eigval[3], const mjtNum eigvec[9]) {
  mjtNum scale = 0;
  for (int i = 0; i < 9; i++) scale = mju_max(scale, mju_abs(mat[i]));

  *recon = *orth = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      mjtNum A = 0, I = 0;
      for (int k = 0; k < 3; k++) {
        A += eigvec[3 * i + k] * eigval[k] * eigvec[3 * j + k];
        I += eigvec[3 * k + i] * eigvec[3 * k + j];
      }
      *recon = mju_max(*recon, mju_abs(A - mat[3 * i + j]) / scale);
      *orth = mju_max(*orth, mju_abs(I - (i == j)));
    }
  }
}

// the decomposition converges to roundoff, whatever the scale of the matrix
TEST_F(Eig3Test, ConvergesAtAnyScale) {
  for (mjtNum scale : {1e-15, 1.0, 1e15}) {
    mjtNum mat[9] = {2, .1, .2, .1, 3, .3, .2, .3, 4};
    mju_scl(mat, mat, scale, 9);

    mjtNum eigval[3], eigvec[9], quat[4];
    EXPECT_LT(mju_eig3(eigval, eigvec, quat, mat), 20);

    // eigenvalues are decreasing
    EXPECT_GT(eigval[0], eigval[1]);
    EXPECT_GT(eigval[1], eigval[2]);

    mjtNum recon, orth;
    Eig3Residual(&recon, &orth, mat, eigval, eigvec);
    EXPECT_LE(recon, MjTol(1e-13, 3e-5)) << "scale " << scale;
    EXPECT_LE(orth, MjTol(5e-14, 5e-6)) << "scale " << scale;
  }
}

// repeated and nearly repeated eigenvalues, where the Jacobi rotation is
// ill-defined, converge in a few iterations
TEST_F(Eig3Test, RepeatedEigenvalues) {
  std::mt19937_64 rng;
  rng.seed(3);
  std::normal_distribution<double> dist(0, 1);

  const mjtNum eps = std::numeric_limits<mjtNum>::epsilon();
  const mjtNum spectrum[4][3] = {
      {3, 1, 1}, {3, 3, 1}, {1, 1, 1}, {1 + 100 * eps, 1, 1 - 100 * eps}};

  for (const mjtNum* w : spectrum) {
    int max_iter = 0;
    mjtNum max_recon = 0;
    for (int n = 0; n < 1000; n++) {
      // random rotation
      mjtNum R[9], q[4] = {(mjtNum)dist(rng), (mjtNum)dist(rng),
                           (mjtNum)dist(rng), (mjtNum)dist(rng)};
      mju_normalize4(q);
      mju_quat2Mat(R, q);

      // mat = R * diag(w) * R', exactly symmetric
      mjtNum mat[9];
      for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
          mat[3 * i + j] = mat[3 * j + i] = R[3 * i] * w[0] * R[3 * j] +
                                            R[3 * i + 1] * w[1] * R[3 * j + 1] +
                                            R[3 * i + 2] * w[2] * R[3 * j + 2];
        }
      }

      mjtNum eigval[3], eigvec[9], quat[4], recon, orth;
      max_iter = mjMAX(max_iter, mju_eig3(eigval, eigvec, quat, mat));
      Eig3Residual(&recon, &orth, mat, eigval, eigvec);
      max_recon = mju_max(max_recon, recon);
    }
    EXPECT_LT(max_iter, 20)
        << "spectrum " << w[0] << " " << w[1] << " " << w[2];
    EXPECT_LE(max_recon, MjTol(1e-13, 3e-5))
        << "spectrum " << w[0] << " " << w[1] << " " << w[2];
  }
}

// a loose tolerance stops early: off-diagonal elements are below the tolerance
// and eigvec is orthonormal, as the broadphase requires of its frame
TEST_F(Eig3Test, Tolerance) {
  const mjtNum mat[9] = {2, .1, .2, .1, 3, .3, .2, .3, 4};
  const mjtNum reltol = 1e-3;

  mjtNum eigval[3], eigvec[9], quat[4];
  int tight = mju_eig3(eigval, eigvec, quat, mat);
  int loose = mju_eig3Tol(eigval, eigvec, quat, mat, reltol);
  EXPECT_LT(loose, tight);

  // D = eigvec' * mat * eigvec
  mjtNum tmp[9], D[9];
  mju_mulMatTMat(tmp, eigvec, mat, 3, 3, 3);
  mju_mulMatMat(D, tmp, eigvec, 3, 3, 3);
  EXPECT_LE(mju_abs(D[1]), reltol * 4);
  EXPECT_LE(mju_abs(D[2]), reltol * 4);
  EXPECT_LE(mju_abs(D[5]), reltol * 4);

  mjtNum recon, orth;
  Eig3Residual(&recon, &orth, mat, eigval, eigvec);
  EXPECT_LE(orth, MjTol(5e-14, 5e-6));
}

}  // namespace
}  // namespace mujoco
