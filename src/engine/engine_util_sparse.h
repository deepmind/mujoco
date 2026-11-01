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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPARSE_H_
#define MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPARSE_H_

#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjtype.h>
#include "engine/engine_util_sparse_avx.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------ sparse operations -------------------------------------------------

// dot-product, both vectors are sparse
MJAPI mjtNum mju_dotSparse2(const mjtNum* vec1, const int* ind1, int nnz1,
                            const mjtNum* vec2, const int* ind2, int nnz2);

// dot-productX3, first vector is sparse; supernode of size 3
void mju_dotSparseX3(mjtNum* res0, mjtNum* res1, mjtNum* res2,
                     const mjtNum* vec10, const mjtNum* vec11, const mjtNum* vec12,
                     const mjtNum* vec2, int nnz1, const int* ind1);

// convert matrix from dense to sparse
//  nnz is size of res and colind, return 1 if too small, 0 otherwise
MJAPI int mju_dense2sparse(mjtNum* res, const mjtNum* mat, int nr, int nc,
                           int* rownnz, int* rowadr, int* colind, int nnz);

// convert matrix from sparse to dense
MJAPI void mju_sparse2dense(mjtNum* res, const mjtNum* mat, int nr, int nc, const int* rownnz,
                            const int* rowadr, const int* colind);

// convert lower-triangular symmetric CSR matrix to full dense matrix
MJAPI void mju_sym2dense(mjtNum* res, const mjtNum* mat, int n,
                         const int* rownnz, const int* rowadr, const int* colind);

// res[row, :] = mat[row, :]
void mju_copySparse(mjtNum* res, const mjtNum* mat, const int* rownnz, const int* rowadr,
                    const int* row, int nrow);

// res[row, :] = 0
void mju_zeroSparse(mjtNum* res, const int* rownnz, const int* rowadr, const int* row, int nrow);

// multiply sparse matrix and dense vector:  res = mat * vec
MJAPI void mju_mulMatVecSparse(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                               int nr, const int* rownnz, const int* rowadr,
                               const int* colind, const int* rowsuper);

// multiply transposed sparse matrix and dense vector:  res = mat' * vec
MJAPI void mju_mulMatTVecSparse(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc,
                                const int* rownnz, const int* rowadr, const int* colind);

// add sparse matrix M to sparse destination matrix
MJAPI void mju_addToMatSparse(mjtNum* dst, int* rownnz, int* rowadr, int* colind, int nr,
                              const mjtNum* M, const int* M_rownnz, const int* M_rowadr,
                              const int* M_colind);

// add symmetric matrix (only lower triangle represented) to dense matrix
MJAPI void mju_addToSymSparse(mjtNum* res, const mjtNum* mat, int n,
                              const int* rownnz, const int* rowadr, const int* colind,
                              int flg_upper);

// multiply symmetric matrix (only lower triangle represented) by vector:
//  res = (mat + strict_upper(mat')) * vec
MJAPI void mju_mulSymVecSparse(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n,
                               const int* rownnz, const int* rowadr, const int* colind);

// compress sparse matrix, remove elements with abs(value) <= minval, return total non-zeros
MJAPI int mju_compressSparse(mjtNum* mat, int nr, int nc,
                             int* rownnz, int* rowadr, int* colind, mjtNum minval);

// count the number of nonzeros in the sum of two sparse vectors
MJAPI int mju_combineSparseCount(int a_nnz, int b_nnz, const int* a_ind, const int* b_ind);

// incomplete combine sparse: dst = a*dst + b*src at common indices
void mju_combineSparseInc(mjtNum* dst, const mjtNum* src, int n, mjtNum a, mjtNum b,
                          int dst_nnz, int src_nnz, const int* dst_ind, const int* src_ind);

// dst += scl * src, only at common non-zero indices
void mju_addToSclSparseInc(mjtNum* dst, const mjtNum* src,
                           int nnzdst, const int* inddst,
                           int nnzsrc, const int* indsrc, mjtNum scl);

// add to sparse matrix: dst = dst + scl*src, return nnz of result
int mju_addToSparseMat(mjtNum* dst, const mjtNum* src, int n, int nrow, mjtNum scl,
                       int dst_nnz, int src_nnz, int* dst_ind, const int* src_ind,
                       mjtNum* buf, int* buf_ind);

// add(merge) two chains
int mju_addChains(int* res, int n, int NV1, int NV2,
                  const int* chain1, const int* chain2);

// transpose sparse matrix, optionally compute row supernodes
MJAPI void mju_transposeSparse(mjtNum* res, const mjtNum* mat, int nr, int nc,
                               int* res_rownnz, int* res_rowadr, int* res_colind, int* res_rowsuper,
                               const int* rownnz, const int* rowadr, const int* colind);

// construct row supernodes
MJAPI void mju_superSparse(int nr, int* rowsuper,
                           const int* rownnz, const int* rowadr, const int* colind);

// compute sparse M'*diag*M (diag=NULL: compute M'*M), res_rowadr must be precomputed
MJAPI void mju_sqrMatTDSparse(mjtNum* res, const mjtNum* mat, const mjtNum* matT,
                              const mjtNum* diag, int nr, int nc,
                              int* res_rownnz, const int* res_rowadr, int* res_colind,
                              const int* rownnz, const int* rowadr,
                              const int* colind, const int* rowsuper,
                              const int* rownnzT, const int* rowadrT,
                              const int* colindT, const int* rowsuperT,
                              mjData* d, int* diagind);

// LEGACY: row-based implementation
MJAPI void mju_sqrMatTDSparse_row(mjtNum* res, const mjtNum* mat, const mjtNum* matT,
                                  const mjtNum* diag, int nr, int nc,
                                  int* res_rownnz, const int* res_rowadr, int* res_colind,
                                  const int* rownnz, const int* rowadr,
                                  const int* colind, const int* rowsuper,
                                  const int* rownnzT, const int* rowadrT,
                                  const int* colindT, const int* rowsuperT,
                                  mjData* d, int* diagind);

// precount res_rownnz and precompute res_rowadr for mju_sqrMatTDSparse, return total non-zeros
MJAPI int mju_sqrMatTDSparseCount(int* res_rownnz, int* res_rowadr, int nr,
                                  const int* rownnz, const int* rowadr, const int* colind,
                                  const int* rownnzT, const int* rowadrT, const int* colindT,
                                  const int* rowsuperT, mjData* d, int flg_upper);

// symbolic phase for mju_sqrMatTDSparse: compute sparsity pattern of M'*M
//   if res_colind is NULL: count mode, fill res_rownnz/res_rowadr, return nnz
//   if res_colind is not NULL: fill mode, write sorted column indices
//   if res_diagind is not NULL: also fill upper triangle and output diagonal indices
//     (res_diagind[r] = -1 for empty rows)
MJAPI int mju_sqrMatTDSparseSymbolic(
    int* res_rownnz, int* res_rowadr, int* res_colind, int* res_diagind, int nr, int nc,
    const int* rownnz, const int* rowadr, const int* colind,
    const int* rownnzT, const int* rowadrT, const int* colindT, const int* rowsuperT, mjData* d);

// numeric phase for mju_sqrMatTDSparse: compute values given pre-computed sparsity
//   res_colind, res_rownnz, res_rowadr must be pre-computed by mju_sqrMatTDSparseSymbolic
MJAPI void mju_sqrMatTDSparseNumeric(
    mjtNum* res, int nc,
    const int* res_rownnz, const int* res_rowadr, const int* res_colind, const int* res_diagind,
    const mjtNum* mat, const int* rownnz, const int* rowadr, const int* colind,
    const mjtNum* matT, const int* rownnzT, const int* rowadrT, const int* colindT,
    const int* rowsuperT, const mjtNum* diag, mjData* d);

// precompute res_rowadr for mju_sqrMatTDSparse using uncompressed memory
MJAPI void mju_sqrMatTDUncompressedInit(int* res_rowadr, int nc);

// extract a single block of a dense matrix
void mju_block(mjtNum* res, const mjtNum* mat, int nc_mat, int nc_res, int nr,
               const int* perm_r, const int* perm_c);

// block-diagonalize a dense matrix
MJAPI void mju_blockDiag(mjtNum* res, const mjtNum* mat, int nc_mat, int nc_res, int nb,
                         const int* perm_r, const int* perm_c,
                         const int* block_nr, const int* block_nc,
                         const int* block_r, const int* block_c);

// extract a single block of a sparse matrix
void mju_blockSparse(
  mjtNum* res, int* res_rownnz, int* res_rowadr, int* res_colind,
  const mjtNum* mat, const int* rownnz, const int* rowadr, const int* colind,
  int nr,
  const int* perm_r, const int* perm_c,
  int col_offset, int res_offset,
  mjtNum* res2, const mjtNum* mat2);

// block-diagonalize a sparse matrix
MJAPI void mju_blockDiagSparse(
  mjtNum* res, int* res_rownnz, int* res_rowadr, int* res_colind,
  const mjtNum* mat, const int* rownnz, const int* rowadr, const int* colind,
  int nr, int nb,
  const int* perm_r, const int* perm_c,
  const int* block_r, const int* block_c,
  mjtNum* res2, const mjtNum* mat2);

// ------------------------------ inlined functions ------------------------------------------------

// dot-product, first vector is sparse
static inline
mjtNum mju_dotSparse(const mjtNum* vec1, const mjtNum* vec2, int nnz1, const int* ind1) {
#ifdef mjUSEAVX
  return mju_dotSparse_avx(vec1, vec2, nnz1, ind1);
#else
  int i = 0;
  mjtNum res = 0;
  int n_4 = nnz1 - 4;
  mjtNum res0 = 0;
  mjtNum res1 = 0;
  mjtNum res2 = 0;
  mjtNum res3 = 0;

  for (; i <= n_4; i+=4) {
    res0 += vec1[i+0] * vec2[ind1[i+0]];
    res1 += vec1[i+1] * vec2[ind1[i+1]];
    res2 += vec1[i+2] * vec2[ind1[i+2]];
    res3 += vec1[i+3] * vec2[ind1[i+3]];
  }

  res = (res0 + res2) + (res1 + res3);

  // scalar part
  for (; i < nnz1; i++) {
    res += vec1[i] * vec2[ind1[i]];
  }

  return res;
#endif  // mjUSEAVX
}



// return 1 if vec1==vec2, 0 otherwise
static inline
int mju_compare(const int* vec1, const int* vec2, int n) {
#ifdef mjUSEAVX
  return mju_compare_avx(vec1, vec2, n);
#else
  return !memcmp(vec1, vec2, n*sizeof(int));
#endif  // mjUSEAVX
}



// merge unique sorted integers, merge array must be large enough (not checked for)
static inline
int mj_mergeSorted(int* merge, const int* chain1, int n1, const int* chain2, int n2) {
  // special case: one or both empty
  if (n1 == 0) {
    if (n2 == 0) {
      return 0;
    }
    memcpy(merge, chain2, n2 * sizeof(int));
    return n2;
  } else if (n2 == 0) {
    memcpy(merge, chain1, n1 * sizeof(int));
    return n1;
  }

  // special case: identical pattern
  if (n1 == n2 && mju_compare(chain1, chain2, n1)) {
    memcpy(merge, chain1, n1 * sizeof(int));
    return n1;
  }

  // merge while both chains are non-empty
  int i = 0, j = 0, k = 0;
  while (i < n1 && j < n2) {
    int c1 = chain1[i];
    int c2 = chain2[j];

    if (c1 < c2) {
      merge[k++] = c1;
      i++;
    } else if (c1 > c2) {
      merge[k++] = c2;
      j++;
    } else {  // c1 == c2
      merge[k++] = c1;
      i++;
      j++;
    }
  }

  // copy remaining
  if (i < n1) {
    memcpy(merge + k, chain1 + i, (n1 - i)*sizeof(int));
    k += n1 - i;
  } else if (j < n2) {
    memcpy(merge + k, chain2 + j, (n2 - j)*sizeof(int));
    k += n2 - j;
  }

  return k;
}



// res = res*scl1 + vec*scl2
static inline
void mju_addToSclScl(mjtNum* res, const mjtNum* vec, mjtNum scl1, mjtNum scl2, int n) {
#ifdef mjUSEAVX
  mju_addToSclScl_avx(res, vec, scl1, scl2, n);
#else
  for (int i=0; i < n; i++) {
    res[i] = res[i]*scl1 + vec[i]*scl2;
  }
#endif  // mjUSEAVX
}



// combine two sparse vectors: dst = a*dst + b*src, return nnz of result
static inline
int mju_combineSparse(mjtNum* dst, const mjtNum* src, mjtNum a, mjtNum b,
                      int dst_nnz, int src_nnz, int* dst_ind, const int* src_ind) {
  // check for identical pattern
  if (dst_nnz == src_nnz) {
    if (mju_compare(dst_ind, src_ind, dst_nnz)) {
      // combine mjtNum data directly
      mju_addToSclScl(dst, src, a, b, dst_nnz);
      return dst_nnz;
    }
  }

  // compute total nnz of result
  int nnz = mju_combineSparseCount(dst_nnz, src_nnz, dst_ind, src_ind);

  // set up read/write pointers at end of arrays
  int bi = dst_nnz - 1, si = src_nnz - 1, w = nnz - 1;

  // merge backwards
  while (bi >= 0 && si >= 0) {
    int badr = dst_ind[bi];
    int sadr = src_ind[si];

    if (badr == sadr) {
      dst[w] = a*dst[bi] + b*src[si];
      dst_ind[w] = badr;
      bi--;
      si--;
    }

    // dst only
    else if (badr > sadr) {
      dst[w] = a*dst[bi];
      dst_ind[w] = badr;
      bi--;
    }

    // src only
    else {
      dst[w] = b*src[si];
      dst_ind[w] = sadr;
      si--;
    }
    w--;
  }

  // remaining src elements
  while (si >= 0) {
    dst[w] = b*src[si];
    dst_ind[w] = src_ind[si];
    si--;
    w--;
  }

  // remaining dst elements: already in place, scale by a
  if (a != 1) {
    while (bi >= 0) {
      dst[bi] *= a;
      bi--;
    }
  }

  return nnz;
}

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPARSE_H_
