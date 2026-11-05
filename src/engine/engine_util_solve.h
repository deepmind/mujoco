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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_UTIL_SOLVE_H_
#define MUJOCO_SRC_ENGINE_ENGINE_UTIL_SOLVE_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

// Cholesky decomposition: mat = L*L'; return rank
MJAPI int mju_cholFactor(mjtNum* mat, int n, mjtNum mindiag);

// Cholesky solve
MJAPI void mju_cholSolve(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n);

// Cholesky rank-one update: L*L' +/- x*x'; return rank
MJAPI int mju_cholUpdate(mjtNum* mat, mjtNum* x, int n, int flg_plus);

// sparse reverse-order Cholesky decomposition: mat = L'*L; return 'rank'
//  mat must be lower-triangular, have preallocated space for fill-in
MJAPI int mju_cholFactorSparse(mjtNum* mat, int n, mjtNum mindiag,
                               int* rownnz, const int* rowadr, int* colind, mjData* d);

// symbolic reverse-Cholesky: compute both L (CSR) and LT (CSC) structures
//   if L_colind is NULL, perform counting logic (fill rownnz/rowadr arrays and return total nnz)
//   if L_colind is not NULL, assume rownnz/rowadr are precomputed and fill colind/map arrays
//   reads pattern from upper triangle
//   d may be NULL: scratch is then heap-allocated
//   based on ldl_symbolic from 'Algorithm 8xx: a concise sparse Cholesky factorization package'
MJAPI int mju_cholFactorSymbolic(int* L_colind, int* L_rownnz, int* L_rowadr,
                                 int* LT_colind, int* LT_rownnz, int* LT_rowadr, int* LT_map,
                                 const int* rownnz, const int* rowadr, const int* colind,
                                 int n, mjData* d);

// symbolic reverse-Cholesky of a matrix of dense bs x bs blocks, from its block pattern: the same
// L arrays, entry for entry, as mju_cholFactorSymbolic computes from the dof-level pattern that
// expands each block, for the n = bs*nb dofs, in time and scratch linear in the block entries
// rather than in the dof entries; the transpose is kept by block: the block rows reaching block
// column q are BT_row[BT_adr[q]] .. BT_row[BT_adr[q+1]-1], in the order of the dof transpose, with
// BT_off the offset of column bs*q in their rows
//   the block pattern has nb rows, columns ascending in each, upper triangle read; the diagonal
//   blocks are dense, whether listed or not
//   if L_colind is NULL, perform counting logic (fill L_rownnz/L_rowadr/BT_adr (nb+1) and return
//   total nnz); else assume them precomputed and fill L_colind/BT_row/BT_off
//   scratch: caller-provided workspace of 3*nb ints, contents ignored
MJAPI int mju_cholFactorSymbolicBlocked(int* L_colind, int* L_rownnz, int* L_rowadr,
                                        int* BT_adr, int* BT_row, int* BT_off,
                                        const int* rownnz, const int* rowadr, const int* colind,
                                        int nb, int bs, int* scratch);

// numeric reverse-Cholesky: compute L values given fixed sparsity pattern, returns rank
//  L_colind must already contain the correct sparsity pattern (from mju_cholFactorSymbolic)
//  LT_map[k] gives index in L for LT_colind[k]
//  scratch: caller-provided workspace of size n, contents ignored
MJAPI int mju_cholFactorNumeric(mjtNum* L, int n, mjtNum mindiag,
                                const int* L_rownnz, const int* L_rowadr, const int* L_colind,
                                const int* LT_rownnz, const int* LT_rowadr, const int* LT_colind,
                                const int* LT_map, const mjtNum* H,
                                const int* H_rownnz, const int* H_rowadr, const int* H_colind,
                                mjtNum* scratch);

// sparse reverse-order Cholesky solve
MJAPI void mju_cholSolveSparse(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n,
                               const int* rownnz, const int* rowadr, const int* colind);

// sparse reverse-order Cholesky rank-one update: L'*L +/i x*x'; return rank
//  x is sparse, change in sparsity pattern of mat is not allowed
//  scratch: caller-provided workspace of size n, contents ignored
MJAPI int mju_cholUpdateSparse(mjtNum* mat, const mjtNum* x, int n, int flg_plus,
                               const int* rownnz, const int* rowadr, const int* colind,
                               int x_nnz, const int* x_ind, mjtNum* scratch);

// band-dense Cholesky decomposition
//  returns minimum value in the factorized diagonal, or 0 if rank-deficient
//  mat has (ntotal-ndense) x nband + ndense x ntotal elements
//  the first (ntotal-ndense) x nband store the band part, left of diagonal, inclusive
//  the second ndense x ntotal store the band part as entire dense rows
//  add diagadd+diagmul*mat_ii to diagonal before factorization
MJAPI mjtNum mju_cholFactorBand(mjtNum* mat, int ntotal, int nband, int ndense,
                                mjtNum diagadd, mjtNum diagmul);

// solve (mat*mat')*res = vec with band-Cholesky decomposition
MJAPI void mju_cholSolveBand(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int ntotal, int nband, int ndense);

// convert banded matrix to dense matrix, fill upper triangle if flg_sym>0
MJAPI void mju_band2Dense(mjtNum* res, const mjtNum* mat, int ntotal, int nband, int ndense,
                          mjtBool flg_sym);

// convert dense matrix to banded matrix
MJAPI void mju_dense2Band(mjtNum* res, const mjtNum* mat, int ntotal, int nband, int ndense);

// multiply band-diagonal matrix with vector, include upper triangle if flg_sym>0
MJAPI void mju_bandMulMatVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int ntotal, int nband, int ndense, int nvec, mjtBool flg_sym);

// address of diagonal element i in band-dense matrix representation
MJAPI int mju_bandDiag(int i, int ntotal, int nband, int ndense);

// dense LU factorization with partial pivoting
//   factorizes n x n row-major matrix A in-place into L and U
//   L has unit diagonal (not stored), U has explicit diagonal
//   pivot stores row permutation: row i of original = row pivot[i] of result
//   return 1 if successful, 0 if singular (diagonal element < mjMINVAL)
MJAPI int mju_factorLU(mjtNum* A, int n, int* pivot);

// solve A*x = b given LU factorization of A, LU and pivot are output of mju_factorLU
MJAPI void mju_solveLU(mjtNum* x, const mjtNum* LU, const mjtNum* b, const int* pivot, int n);

// 6x6 specialization of mju_factorLU (identical results, allows full unrolling)
MJAPI int mju_factorLU6(mjtNum A[36], int pivot[6]);

// solve A*x = b given 6x6 LU factorization from mju_factorLU6
MJAPI void mju_solveLU6(mjtNum x[6], const mjtNum LU[36], const mjtNum b[6], const int pivot[6]);

// sparse reverse-order LU factorization, assume tree topology (only dofs in index, if given)
//  LU = L + U; original = (U+I) * L; scratch is size n
//  clamp pivots with magnitude below mjMINVAL, return first clamped dof index or -1 if none
int mju_factorLUSparse(mjtNum *LU, int n, int* scratch,
                       const int *rownnz, const int *rowadr, const int *colind, const int *index);

// solve mat*res=vec given LU factorization of mat (only dofs in index, if given)
void mju_solveLUSparse(mjtNum *res, const mjtNum *LU, const mjtNum* vec, int n,
                       const int *rownnz, const int *rowadr, const int* diag, const int *colind,
                       const int *index);

// solve 3x3 linear system A*x = b using Gaussian elimination
void mju_solve3(mjtNum x[3], const mjtNum A[9], const mjtNum b[3]);

// solve QCQP in 2 dimensions:
//  min  0.5*x'*A*x + x'*b  s.t.  sum (xi/di)^2 <= r^2
// return 0 if unconstrained, 1 if constrained
MJAPI int mju_QCQP2(mjtNum* res, const mjtNum* Ain, const mjtNum* bin, const mjtNum* d, mjtNum r);

// solve QCQP in 3 dimensions:
//  min  0.5*x'*A*x + x'*b  s.t.  sum (xi/di)^2 <= r^2
// return 0 if unconstrained, 1 if constrained
MJAPI int mju_QCQP3(mjtNum* res, const mjtNum* Ain, const mjtNum* bin, const mjtNum* d, mjtNum r);

// solve QCQP in n<=5 dimensions:
//  min  0.5*x'*A*x + x'*b  s.t.  sum (xi/di)^2 <= r^2
// return 0 if unconstrained, 1 if constrained
int mju_QCQP(mjtNum* res, const mjtNum* Ain, const mjtNum* bin, const mjtNum* d, mjtNum r, int n);

// solve box-constrained Quadratic Program
//  min 0.5*x'*H*x + x'*g  s.t. lower <= x <=upper
// return rank of unconstrained subspace or -1 on failure
MJAPI int mju_boxQP(mjtNum* res, mjtNum* R, int* index,
                    const mjtNum* H, const mjtNum* g, int n,
                    const mjtNum* lower, const mjtNum* upper);

// allocate memory for box-constrained Quadratic Program
MJAPI void mju_boxQPmalloc(mjtNum** res, mjtNum** R, int** index,
                           mjtNum** H, mjtNum** g, int n,
                           mjtNum** lower, mjtNum** upper);

// minimize 0.5*x'*H*x + x'*g  s.t. lower <= x <=upper, explicit options (see implementation)
MJAPI int mju_boxQPoption(mjtNum* res, mjtNum* R, int* index,
                          const mjtNum* H, const mjtNum* g, int n,
                          const mjtNum* lower, const mjtNum* upper,
                          int maxiter, mjtNum mingrad, mjtNum backtrack,
                          mjtNum minstep, mjtNum armijo,
                          char* log, int logsz);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_UTIL_SOLVE_H_
