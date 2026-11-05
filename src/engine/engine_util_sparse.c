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

#include "engine/engine_util_sparse.h"

#include <stddef.h>
#include <stdint.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjtype.h>
#include "engine/engine_crossplatform.h"
#include "engine/engine_memory.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_sparse_avx.h"  // IWYU pragma: keep

//------------------------------ sparse operations -------------------------------------------------


// dot-productX3, first vector is sparse; supernode of size 3
void mju_dotSparseX3(mjtNum* res0, mjtNum* res1, mjtNum* res2,
                     const mjtNum* vec10, const mjtNum* vec11, const mjtNum* vec12,
                     const mjtNum* vec2, int nnz1, const int* ind1) {
#ifdef mjUSEAVX
  mju_dotSparseX3_avx(res0, res1, res2, vec10, vec11, vec12, vec2, nnz1, ind1);
#else
  int i = 0;

  // clear result
  mjtNum RES0 = 0;
  mjtNum RES1 = 0;
  mjtNum RES2 = 0;

  for (; i < nnz1; i++) {
    mjtNum v2 = vec2[ind1[i]];

    RES0 += vec10[i] * v2;
    RES1 += vec11[i] * v2;
    RES2 += vec12[i] * v2;
  }

  // copy result
  *res0 = RES0;
  *res1 = RES1;
  *res2 = RES2;
#endif  // mjUSEAVX
}


// dot-product, both vectors are sparse
mjtNum mju_dotSparse2(const mjtNum* vec1, const int* ind1, int nnz1,
                      const mjtNum* vec2, const int* ind2, int nnz2) {
  int i1 = 0, i2 = 0;
  mjtNum res = 0;

  // check for empty array
  if (!nnz1 || !nnz2) {
    return 0;
  }

  while (i1 < nnz1 && i2 < nnz2) {
    // get current indices
    int adr1 = ind1[i1], adr2 = ind2[i2];

    // match: accumulate result, advance both
    if (adr1 == adr2) {
      res += vec1[i1++] * vec2[i2++];
    }

    // otherwise advance smaller
    else if (adr1 < adr2) {
      i1++;
    } else {
      i2++;
    }
  }

  return res;
}


// convert matrix from dense to sparse
//  nnz is size of res and colind, return 1 if too small, 0 otherwise
int mju_dense2sparse(mjtNum* res, const mjtNum* mat, int nr, int nc,
                     int* rownnz, int* rowadr, int* colind, int nnz) {
  if (nnz <= 0) {
    return 1;
  }

  int adr = 0;

  // find non-zeros and construct sparse
  for (int r=0; r < nr; r++) {
    // init row
    rownnz[r] = 0;
    rowadr[r] = adr;

    // find non-zeros
    for (int c=0; c < nc; c++) {
      if (mat[r*nc+c]) {
        // check for out of bounds
        if (adr >= nnz) {
          return 1;
        }

        // record index and count
        colind[adr] = c;
        rownnz[r]++;

        // copy element
        res[adr++] = mat[r*nc+c];
      }
    }
  }
  return 0;
}


// convert matrix from sparse to dense
void mju_sparse2dense(mjtNum* res, const mjtNum* mat, int nr, int nc,
                      const int* rownnz, const int* rowadr, const int* colind) {
  // clear
  mju_zero(res, nr*nc);

  // copy non-zeros
  for (int r=0; r < nr; r++) {
    for (int i=0; i < rownnz[r]; i++) {
      res[r*nc + colind[rowadr[r]+i]] = mat[rowadr[r]+i];
    }
  }
}


// convert lower-triangular symmetric CSR matrix to full dense matrix
void mju_sym2dense(mjtNum* res, const mjtNum* mat, int n,
                   const int* rownnz, const int* rowadr, const int* colind) {
  mju_zero(res, n*n);
  for (int i = 0; i < n; i++) {
    int adr = rowadr[i];
    for (int j = 0; j < rownnz[i]; j++) {
      int col = colind[adr+j];
      if (col <= i) {
        res[i*n+col] = mat[adr+j];
        res[col*n+i] = mat[adr+j];
      }
    }
  }
}


// res[row, :] = mat[row, :]
void mju_copySparse(mjtNum* res, const mjtNum* mat, const int* rownnz, const int* rowadr,
                    const int* row, int nrow) {
  for (int i=0; i < nrow; i++) {
    int r = row[i];
    mju_copy(res + rowadr[r], mat + rowadr[r], rownnz[r]);
  }
}


// res[row, :] = 0
void mju_zeroSparse(mjtNum* res, const int* rownnz, const int* rowadr, const int* row, int nrow) {
  for (int i=0; i < nrow; i++) {
    int r = row[i];
    mju_zero(res + rowadr[r], rownnz[r]);
  }
}


// multiply sparse matrix and dense vector:  res = mat * vec.
void mju_mulMatVecSparse(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                         int nr, const int* rownnz, const int* rowadr,
                         const int* colind, const int* rowsuper) {
#ifdef mjUSEAVX
  mju_mulMatVecSparse_avx(res, mat, vec, nr, rownnz, rowadr, colind, rowsuper);
#else
  // regular sparse dot-product
  for (int r=0; r < nr; r++) {
    res[r] = mju_dotSparse(mat+rowadr[r], vec, rownnz[r], colind+rowadr[r]);
  }
#endif  // mjUSEAVX
}


// multiply transposed sparse matrix and dense vector:  res = mat' * vec.
void mju_mulMatTVecSparse(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc,
                          const int* rownnz, const int* rowadr, const int* colind) {
  // clear res
  mju_zero(res, nc);

  for (int i=0; i < nr; i++) {
    mjtNum scl = vec[i];

    // skip if 0
    if (!scl) continue;

    // add row scaled by the corresponding vector element
    int nnz = rownnz[i];
    int adr = rowadr[i];
    const int* ind = colind + adr;
    const mjtNum* row = mat + adr;
    for (int j=0; j < nnz; j++) {
      res[ind[j]] += row[j] * scl;
    }
  }
}


// add sparse matrix M to sparse destination matrix
void mju_addToMatSparse(mjtNum* dst, int* rownnz, int* rowadr, int* colind, int nr,
                        const mjtNum* M, const int* M_rownnz, const int* M_rowadr,
                        const int* M_colind) {
  for (int i=0; i < nr; i++) {
    rownnz[i] = mju_combineSparse(dst + rowadr[i], M + M_rowadr[i], 1, 1,
                                  rownnz[i], M_rownnz[i], colind + rowadr[i],
                                  M_colind + M_rowadr[i]);
  }
}


// add symmetric matrix (lower triangle) to dense matrix, upper triangle optional
void mju_addToSymSparse(mjtNum* res, const mjtNum* mat, int n,
                        const int* rownnz, const int* rowadr, const int* colind, int flg_upper) {
  for (int i=0; i < n; i++) {
    int start = rowadr[i];
    int end = start + rownnz[i];
    for (int adr=start; adr < end; adr++) {
      mjtNum val = mat[adr];
      int j = colind[adr];

      // lower + diagonal
      res[i*n + j] += val;

      // strict upper
      if (flg_upper && j < i) {
        res[j*n + i] += val;
      }
    }
  }
}


// multiply symmetric matrix (only lower triangle represented) by vector:
//  res = (mat + strict_upper(mat')) * vec
void mju_mulSymVecSparse(mjtNum* restrict res, const mjtNum* restrict mat,
                         const mjtNum* restrict vec, int n,
                         const int* restrict rownnz, const int* restrict rowadr,
                         const int* restrict colind) {
  // clear res
  mju_zero(res, n);

  // multiply
  for (int i=0; i < n; i++) {
    int adr = rowadr[i];
    int diag = rownnz[i] - 1;
    const mjtNum* row = mat + adr;

    // diagonal
    res[i] = row[diag] * vec[i];

    // off-diagonals
    const int* ind = colind + adr;
    for (int k=diag-1; k >= 0; k--) {
      int j = ind[k];
      mjtNum val = row[k];
      res[i] += val * vec[j];  // strict lower
      res[j] += val * vec[i];  // strict upper
    }
  }
}


// count the number of non-zeros in the sum of two sparse vectors
int mju_combineSparseCount(int a_nnz, int b_nnz, const int* a_ind, const int* b_ind) {
  int a = 0, b = 0, c_nnz = 0;

  // count c_nnz: nonzero indices common to both a and b
  while (a < a_nnz && b < b_nnz) {
    // common index, increment everything
    if (a_ind[a] == b_ind[b]) {
      c_nnz++;
      a++;
      b++;
    }

    // update smallest index
    else if (a_ind[a] < b_ind[b]) {
      a++;
    } else {
      b++;
    }
  }

  // union minus the intersection
  return a_nnz + b_nnz - c_nnz;
}


// incomplete combine sparse: dst = a*dst + b*src at common indices
void mju_combineSparseInc(mjtNum* dst, const mjtNum* src, int n, mjtNum a, mjtNum b,
                          int dst_nnz, int src_nnz, const int* dst_ind, const int* src_ind) {
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
  int dadr = di < dst_nnz ? dst_ind[di] : n+1;
  int sadr = si < src_nnz ? src_ind[si] : n+1;

  // add src*b at common indices
  while (di < dst_nnz) {
    // both
    if (dadr == sadr) {
      dst[di++] += b*src[si++];

      dadr = di < dst_nnz ? dst_ind[di] : n+1;
      sadr = si < src_nnz ? src_ind[si] : n+1;
    }

    // dst only
    else if (dadr < sadr) {
      di++;
      dadr = di < dst_nnz ? dst_ind[di] : n+1;
    }

    // src only
    else {
      si++;
      sadr = si < src_nnz ? src_ind[si] : n+1;
    }
  }
}


// dst += scl*src, only at common non-zero indices
void mju_addToSclSparseInc(mjtNum* restrict dst, const mjtNum* restrict src,
                           int nnzdst, const int* restrict inddst,
                           int nnzsrc, const int* restrict indsrc, mjtNum scl) {
  if (!nnzdst || !nnzsrc) {
    return;
  }

  int adrs = 0, adrd = 0;
  while (adrs < nnzsrc && adrd < nnzdst) {
    int inds = indsrc[adrs];
    int indd = inddst[adrd];

    if (inds == indd) {
      dst[adrd] += scl * src[adrs];
      adrs++;
      adrd++;
    } else if (inds < indd) {
      adrs++;
    } else {
      adrd++;
    }
  }
}


// add to sparse matrix: dst = dst + scl*src, return nnz of result
int mju_addToSparseMat(mjtNum* dst, const mjtNum* src, int n, int nrow, mjtNum scl,
                       int dst_nnz, int src_nnz, int* dst_ind, const int* src_ind,
                       mjtNum* buf, int* buf_ind) {
  // check for identical pattern
  if (dst_nnz == src_nnz) {
    if (dst_nnz == 0) {
      return 0;
    }
    if (mju_compare(dst_ind, src_ind, dst_nnz)) {
      // combine mjtNum data directly
      mju_addToScl(dst, src, scl, nrow*dst_nnz);
      return dst_nnz;
    }
  }

  // prepare to merge scr and dst into buf^T
  int si = 0, di = 0, nnz = 0;
  int sadr = src_nnz ? src_ind[0] : n+1;
  int dadr = dst_nnz ? dst_ind[0] : n+1;

  // merge matrices
  while (si < src_nnz || di < dst_nnz) {
    // both
    if (sadr == dadr) {
      for (int k=0; k < nrow; k++) {
        buf[nrow*nnz + k] = dst[di + k*dst_nnz] + scl*src[si + k*src_nnz];
      }

      buf_ind[nnz++] = sadr;
      si++;
      di++;
      sadr = si < src_nnz ? src_ind[si] : n+1;
      dadr = di < dst_nnz ? dst_ind[di] : n+1;
    }

    // dst only
    else if (dadr < sadr) {
      for (int k=0; k < nrow; k++) {
        buf[nrow*nnz + k] = dst[di + k*dst_nnz];
      }

      buf_ind[nnz++] = dadr;
      di++;
      dadr = di < dst_nnz ? dst_ind[di] : n+1;
    }

    // src only
    else {
      for (int k=0; k < nrow; k++) {
        buf[nrow*nnz + k] = scl*src[si + k*src_nnz];
      }

      buf_ind[nnz++] = sadr;
      si++;
      sadr = si < src_nnz ? src_ind[si] : n+1;
    }
  }

  // copy transposed buf into dst
  mju_transpose(dst, buf, nnz, nrow);
  mju_copyInt(dst_ind, buf_ind, nnz);

  return nnz;
}


// add(merge) two chains
int mju_addChains(int* res, int n, int NV1, int NV2,
                  const int* chain1, const int* chain2) {
  // check for identical pattern
  if (NV1 == NV2) {
    if (NV1 == 0) {
      return 0;
    }
    if (mju_compare(chain1, chain2, NV1)) {
      mju_copyInt(res, chain1, NV1);
      return NV1;
    }
  }

  // prepare to merge
  int i1 = 0, i2 = 0, NV = 0;
  int adr1 = NV1 ? chain1[0] : n+1;
  int adr2 = NV2 ? chain2[0] : n+1;

  // merge chains
  while (i1 < NV1 || i2 < NV2) {
    // both
    if (adr1 == adr2) {
      res[NV++] = adr1;
      i1++;
      i2++;
      adr1 = i1 < NV1 ? chain1[i1] : n+1;
      adr2 = i2 < NV2 ? chain2[i2] : n+1;
    }

    // chain1 only
    else if (adr1 < adr2) {
      res[NV++] = adr1;
      i1++;
      adr1 = i1 < NV1 ? chain1[i1] : n+1;
    }

    // chain2 only
    else {
      res[NV++] = adr2;
      i2++;
      adr2 = i2 < NV2 ? chain2[i2] : n+1;
    }
  }

  return NV;
}


// compress sparse matrix, remove elements with abs(value) <= minval, return total non-zeros
int mju_compressSparse(mjtNum* mat, int nr, int nc, int* rownnz, int* rowadr, int* colind,
                       mjtNum minval) {
  int remove_small = minval >= 0;
  int adr = 0;
  for (int r=0; r < nr; r++) {
    // save old rowadr, record new
    int rowadr_old = rowadr[r];
    rowadr[r] = adr;

    // shift mat and colind
    int nnz = 0;
    int end = rowadr_old + rownnz[r];
    for (int adr_old=rowadr_old; adr_old < end; adr_old++) {
      if (remove_small && mju_abs(mat[adr_old]) <= minval) {
        continue;
      }
      if (adr != adr_old) {
        mat[adr] = mat[adr_old];
        colind[adr] = colind[adr_old];
      }
      adr++;
      if (remove_small) nnz++;
    }
    if (remove_small) rownnz[r] = nnz;
  }

  return rowadr[nr-1] + rownnz[nr-1];
}


// transpose sparse matrix, optionally compute row supernodes
void mju_transposeSparse(mjtNum* res, const mjtNum* mat, int nr, int nc,
                         int* res_rownnz, int* res_rowadr, int* res_colind, int* res_rowsuper,
                         const int* rownnz, const int* rowadr, const int* colind) {
  if (!nr || !nc) return;

  // clear number of non-zeros for each row of transposed
  mju_zeroInt(res_rownnz, nc);

  // handle the case where the first row of mat is nonzero (offset wrt the base pointers)
  int row_offset = rowadr[0];

  // count the number of non-zeros for each row of the transposed matrix
  for (int r = 0; r < nr; r++) {
    int start = rowadr[r] - row_offset;
    int end = start + rownnz[r];
    for (int j = start; j < end; j++) {
      res_rownnz[colind[j]]++;
    }
  }

  // init res_rowsuper
  if (res_rowsuper) {
    for (int i = 0; i < nc - 1; i++) {
      res_rowsuper[i] = (res_rownnz[i] == res_rownnz[i + 1]);
    }
    res_rowsuper[nc - 1] = 0;
  }

  // compute the row addresses for the transposed matrix
  res_rowadr[0] = 0;
  for (int i = 1; i < nc; i++) {
    res_rowadr[i] = res_rowadr[i-1] + res_rownnz[i-1];
  }

  // iterate through each row (column) of mat (res)
  for (int r = 0; r < nr; r++) {
    int c_prev = -1;
    int start = rowadr[r] - row_offset;
    int end = start + rownnz[r];
    for (int i = start; i < end; i++) {
      // swap rows with columns and increment res_rowadr
      int c = colind[i];
      int adr = res_rowadr[c]++;
      res_colind[adr] = r;
      if (res) {
        res[adr] = mat[i];
      }

      // mark non-supernodes
      if (res_rowsuper) {
        if (c > 0 && c != c_prev + 1 && res_rowsuper[c - 1]) {
          res_rowsuper[c - 1] = 0;
        }
        c_prev = c;
      }
    }
  }

  // shift back row addresses
  for (int i = nc-1; i > 0; i--) {
    res_rowadr[i] = res_rowadr[i-1];
  }
  res_rowadr[0] = 0;

  // accumulate supernodes
  if (res_rowsuper) {
    for (int i = nc - 2; i >= 0; i--) {
      if (res_rowsuper[i]) {
        res_rowsuper[i] += res_rowsuper[i + 1];
      }
    }
  }
}


// construct row supernodes
void mju_superSparse(int nr, int* rowsuper,
                     const int* rownnz, const int* rowadr, const int* colind) {
  // no rows: nothing to do
  if (!nr) {
    return;
  }

  // find match to child
  for (int r=0; r < nr-1; r++) {
    // different number of nonzeros: cannot be a match
    if (rownnz[r] != rownnz[r+1]) {
      rowsuper[r] = 0;
    }

    // same number of nonzeros: compare colind vectors
    else {
      rowsuper[r] = mju_compare(colind+rowadr[r], colind+rowadr[r+1], rownnz[r]);
    }
  }

  // clear last (by definition)
  rowsuper[nr-1] = 0;

  // accumulate in reverse
  for (int r=nr-2; r >= 0; r--) {
    if (rowsuper[r]) {
      rowsuper[r] += rowsuper[r+1];
    }
  }
}


// precount res_rownnz and precompute res_rowadr for mju_sqrMatTDSparse, return total non-zeros
int mju_sqrMatTDSparseCount(int* res_rownnz, int* res_rowadr, int nr,
                            const int* rownnz, const int* rowadr, const int* colind,
                            const int* rownnzT, const int* rowadrT, const int* colindT,
                            const int* rowsuperT, mjData* d, int flg_upper) {
  mj_markStack(d);
  int* chain = mjSTACKALLOC(d, 2*nr, int);
  int nchain = 0;
  int* res_colind = NULL;

  for (int r=0; r < nr; r++) {
    // supernode; copy everything to next row
    if (rowsuperT && r > 0 && rowsuperT[r-1] > 0) {
      res_rownnz[r] = res_rownnz[r - 1];

      // fill in upper triangle
      if (flg_upper) {
        for (int j=0; j < nchain; j++) {
          res_rownnz[res_colind[j]]++;
        }
      }

      // update chain with diagonal
      if (rownnzT[r]) {
        res_colind[nchain++] = r;
        res_rownnz[r]++;
      }
    } else {
      int inew = 0, iold = nr;
      nchain = 0;

      for (int i=0; i < rownnzT[r]; i++) {
        int c = colindT[rowadrT[r] + i];

        int adr = inew;
        inew = iold;
        iold = adr;

        int nnewchain = 0;
        adr = 0;
        int end = rowadr[c] + rownnz[c];
        for (int adr1=rowadr[c]; adr1 < end; adr1++) {
          int col_mat = colind[adr1];
          while (adr < nchain && chain[iold + adr] < col_mat && chain[iold + adr] <= r) {
            chain[inew + nnewchain++] = chain[iold + adr++];
          }

          // skip upper triangle
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

      // only computed for lower triangle
      res_rownnz[r] = nchain;
      res_colind = chain + inew;

      // update upper triangle
      if (flg_upper) {
        int nchain_end = nchain;

        // avoid double counting
        if (nchain > 0 && res_colind[nchain-1] == r) {
          nchain_end = nchain - 1;
        }

        for (int j=0; j < nchain_end; j++) {
          res_rownnz[res_colind[j]]++;
        }
      }
    }
  }

  res_rowadr[0] = 0;
  for (int r = 1; r < nr; r++) {
    res_rowadr[r] = res_rowadr[r-1] + res_rownnz[r-1];
  }

  mj_freeStack(d);

  return res_rowadr[nr-1] + res_rownnz[nr-1];
}


// precompute res_rowadr for mju_sqrMatTDSparse using uncompressed memory
void mju_sqrMatTDUncompressedInit(int* res_rowadr, int nc) {
  for (int r=0; r < nc; r++) {
    res_rowadr[r] = r*nc;
  }
}


// max number of supernodes handled by column-based matrix squaring functions
#define mjMAXSUPER 8

// column-based symbolic phase for sparse matrix squaring: compute sparsity pattern of M'*M
//   if res_colind is NULL: count mode, fill res_rownnz/res_rowadr, return nnz
//   if res_colind is not NULL: fill mode, write sorted column indices
//   if res_diagind is not NULL: also fill upper triangle and output diagonal indices
//     (res_diagind[r] = -1 for empty rows)
int mju_sqrMatTDSparseSymbolic(
    int* restrict res_rownnz, int* restrict res_rowadr,
    int* restrict res_colind, int* restrict res_diagind, int nr, int nc,
    const int* rownnz, const int* rowadr, const int* colind, const int* rowsuper,
    const int* rownnzT, const int* rowadrT, const int* colindT, const int* rowsuperT, mjData* d) {
  if (nc <= 0 || nr <= 0) {
    if (nc > 0) {
      mju_zeroInt(res_rownnz, nc);
      if (!res_colind) {
        mju_zeroInt(res_rowadr, nc);
      }
      if (res_diagind) {
        mju_fillInt(res_diagind, -1, nc);
      }
    }
    return 0;
  }

  mj_markStack(d);

  // reinterpret M^T as CSC
  const int* colnnz = rownnzT;
  const int* coladr = rowadrT;
  const int* rowind = colindT;
  const int* colsuper = rowsuperT;

  // reinterpret M as CSC
  const int* colnnzT = rownnz;
  const int* coladrT = rowadr;
  const int* rowindT = colind;

  // marker[j] = 1 if row j has been visited in current column batch
  int* marker = mjSTACKALLOC(d, nc, int);
  mju_zeroInt(marker, nc);

  // buffer_idx: list of row indices with nonzeros in current column batch
  int* buffer_idx = mjSTACKALLOC(d, nc, int);

  // rowstart[r]: first index in row r of M where column > current result column
  int* rowstart = mjSTACKALLOC(d, nr, int);
  mju_zeroInt(rowstart, nr);

  // clear res_rownnz (used for both counting and filling)
  mju_zeroInt(res_rownnz, nc);

  // process result columns c = 0, 1, ..., nc-1
  int ns;  // set in the loop
  for (int c = 0; c < nc; c += ns) {
    int buffer_nnz = 0;

    // column c of M^T
    int nnz_c = colnnz[c];
    int adr_c = coladr[c];
    const int* ind_c = rowind + adr_c;

    // supernode size: how many consecutive columns share the same sparsity pattern
    ns = 1;
    int cs;
    if (colsuper && (cs = colsuper[c])) {
      ns += mjMIN(cs, mjMAXSUPER - 1);
    }

    // if column batch is empty, nothing to do (must follow ns computation since loop steps by ns)
    if (!nnz_c) {
      continue;
    }

    // for each row r where M^T[r, c] != 0, look at row r of M past the supernode block
    for (int i = 0; i < nnz_c; i++) {
      int r = ind_c[i];
      int adrT = coladrT[r];
      int nnzT = colnnzT[r];
      const int* indT = rowindT + adrT;

      // columns c .. c+ns-1 are at rowstart[r] .. rowstart[r]+ns-1; scan j >= c+ns
      for (int k = (rowstart[r] += ns); k < nnzT; k++) {
        int j = indT[k];

        // new nonzero in row j of result
        if (!marker[j]) {
          marker[j] = 1;
          buffer_idx[buffer_nnz++] = j;
        }
      }
      if (rowsuper) {
        i += rowsuper[r];
      }
    }

    // scatter: update result rows j >= c + ns that have nonzeros in this column batch

    // fill mode: write column indices, clear markers (specialize ns == 1 to avoid inner s loop)
    if (res_colind) {
      if (ns == 1) {
        res_colind[res_rowadr[c] + res_rownnz[c]++] = c;
        for (int i = 0; i < buffer_nnz; i++) {
          int j = buffer_idx[i];
          marker[j] = 0;
          res_colind[res_rowadr[j] + res_rownnz[j]++] = c;
        }
      } else {
        // write supernode lower-triangle and diagonal entries (rows c .. c+ns-1)
        for (int s = 0; s < ns; s++) {
          int col = c + s;
          int adr_col = res_rowadr[col] + res_rownnz[col];
          for (int k = 0; k <= s; k++) {
            res_colind[adr_col + k] = c + k;
          }
          res_rownnz[col] += s + 1;
        }

        // write off-diagonal entries for rows j >= c + ns
        for (int i = 0; i < buffer_nnz; i++) {
          int j = buffer_idx[i];
          marker[j] = 0;
          int adr_j = res_rowadr[j] + res_rownnz[j];
          for (int s = 0; s < ns; s++) {
            res_colind[adr_j + s] = c + s;
          }
          res_rownnz[j] += ns;
        }
      }
    }

    // count mode: just count and clear markers
    else {
      for (int i = 0; i < buffer_nnz; i++) {
        int j = buffer_idx[i];
        marker[j] = 0;
        res_rownnz[j] += ns;
      }

      // count supernode lower-triangle, diagonal, and optional upper-triangle entries
      if (res_diagind) {
        for (int s = 0; s < ns; s++) {
          res_rownnz[c + s] += ns + buffer_nnz;
        }
      } else {
        for (int s = 0; s < ns; s++) {
          res_rownnz[c + s] += s + 1;
        }
      }
    }
  }

  // count mode: compute res_rowadr from res_rownnz
  if (!res_colind) {
    res_rowadr[0] = 0;
    for (int r = 1; r < nc; r++) {
      res_rowadr[r] = res_rowadr[r - 1] + res_rownnz[r - 1];
    }
  }

  // fill mode with upper triangle: record diagonal positions and mirror from lower
  if (res_colind && res_diagind) {
    // save current counts (lower + diagonal)
    int* lower_nnz = mjSTACKALLOC(d, nc, int);
    mju_copyInt(lower_nnz, res_rownnz, nc);

    // save diagonal indices
    for (int r = 0; r < nc; r++) {
      res_diagind[r] = lower_nnz[r] ? (res_rowadr[r] + lower_nnz[r] - 1) : -1;
    }

    // fill upper triangle: for each (r, c) with c < r, write to (c, r)
    for (int r = 0; r < nc; r++) {
      int adr = res_rowadr[r];
      int nnz = lower_nnz[r];
      for (int j = 0; j < nnz; j++) {
        int col = res_colind[adr + j];
        if (col < r) {
          res_colind[res_rowadr[col] + res_rownnz[col]++] = r;
        }
      }
    }
  }

  mj_freeStack(d);

  return res_rowadr[nc - 1] + res_rownnz[nc - 1];
}


// accumulate supernode block and off-diagonal row entries with fixed SIMD width (force inline)
static mjALWAYS_INLINE
void accumSuperRow(
    mjtNum block[mjMAXSUPER][mjMAXSUPER], mjtNum* restrict buffer,
    const mjtNum* restrict scale, const mjtNum* const matT_s[mjMAXSUPER],
    const int* restrict ind_k, const mjtNum* restrict val_k,
    int i, int r, int ns, int any_off, int width) {
  // accumulate supernode block
  for (int t = 0; t < ns; t++) {
    mjtNum v = matT_s[t][i];
    for (int s = 0; s < width; s++) {
      block[t][s] += scale[s] * v;
    }
  }

  // accumulate off-diagonal row entries (column r is in M[k, :] since k is in M'[r, :])
  if (any_off) {
    for (int j = 0; ; j++) {
      int c = ind_k[j];
      if (c >= r) break;
      mjtNum v = val_k[j];
      mjtNum* buf = buffer + c * mjMAXSUPER;
      for (int s = 0; s < width; s++) {
        buf[s] += scale[s] * v;
      }
    }
  }
}


// numeric phase for sparse matrix squaring: compute values given pre-computed sparsity
//   if res_diagind is not NULL, also fill upper triangle
void mju_sqrMatTDSparseNumeric(
    mjtNum* restrict res, int nc,
    const int* res_rownnz, const int* res_rowadr, const int* res_colind, const int* res_diagind,
    const mjtNum* mat, const int* rownnz, const int* rowadr, const int* colind,
    const mjtNum* matT, const int* rownnzT, const int* rowadrT, const int* colindT,
    const int* rowsuperT, const mjtNum* diag, mjData* d) {
  if (nc <= 0) {
    return;
  }

  mj_markStack(d);

  // dense accumulator for off-diagonal columns c < r, zeroed lazily up to flat index buf_zeroed
  mjtNum* restrict buffer = mjSTACKALLOC(d, nc * mjMAXSUPER, mjtNum);
  int buf_zeroed = 0;

  // process result rows
  int ns;  // set in the loop
  for (int r = 0; r < nc; r += ns) {
    // determine supernode size
    ns = 1;
    if (rowsuperT) {
      ns = rowsuperT[r] + 1;
      if (ns > mjMAXSUPER) ns = mjMAXSUPER;
    }

    // if row (or supernode) is empty, zero out result rows and continue
    int nnzT_r = rownnzT[r];
    if (!nnzT_r) {
      for (int s = 0; s < ns; s++) {
        int row = r + s;
        int adr = res_rowadr[row];
        int nnz = res_diagind ? res_diagind[row] - adr + 1 : res_rownnz[row];
        if (nnz > 0) {
          mju_zero(res + adr, nnz);
        }
      }
      continue;
    }
    int adr_r = rowadrT[r];

    // single row
    if (ns == 1) {
      int res_adr = res_rowadr[r];
      int res_nnz = res_diagind ? res_diagind[r] - res_adr + 1 : res_rownnz[r];
      int off_nnz = res_nnz - 1;
      mjtNum diag_r = 0;

      // lazily zero buffer up to column r
      if (off_nnz > 0 && r > buf_zeroed) {
        mju_zero(buffer + buf_zeroed, r - buf_zeroed);
        buf_zeroed = r;
      }

      // accumulate: res[r, :] = sum over k in M'[r, :] of diag[k] * M'[r, k] * M[k, :]
      for (int i = 0; i < nnzT_r; i++) {
        int k = colindT[adr_r + i];
        mjtNum valT = matT[adr_r + i];
        mjtNum scale = diag ? diag[k] * valT : valT;
        if (scale == 0) continue;
        diag_r += scale * valT;

        // column r is in M[k, :] since k is in M'[r, :], acting as a sentinel for c >= r
        if (off_nnz > 0) {
          int adr_k = rowadr[k];
          const int* ind_k = colind + adr_k;
          const mjtNum* val_k = mat + adr_k;
          for (int j = 0; ; j++) {
            int c = ind_k[j];
            if (c >= r) break;
            buffer[c] += scale * val_k[j];
          }
        }
      }

      // scatter off-diagonal columns from dense buffer and write diagonal
      const int* res_ind = res_colind + res_adr;
      mjtNum* res_val = res + res_adr;
      for (int j = 0; j < off_nnz; j++) {
        int c = res_ind[j];
        res_val[j] = buffer[c];
        buffer[c] = 0;
      }
      res_val[off_nnz] = diag_r;
    }

    // supernode: ns > 1 rows share the same sparsity pattern in M'
    else {
      // row addresses and counts of columns left of the supernode (c < r)
      int res_adr_s[mjMAXSUPER];
      int off_nnz_s[mjMAXSUPER];
      int any_off = 0;
      const mjtNum* matT_s[mjMAXSUPER];
      for (int s = 0; s < ns; s++) {
        int row = r + s;
        int adr = res_rowadr[row];
        int nnz = res_diagind ? res_diagind[row] - adr + 1 : res_rownnz[row];
        res_adr_s[s] = adr;
        off_nnz_s[s] = nnz - (s + 1);
        any_off |= off_nnz_s[s];
        matT_s[s] = matT + rowadrT[row];
      }

      // lazily zero buffer for off-diagonal columns (c < r)
      if (any_off && r * mjMAXSUPER > buf_zeroed) {
        mju_zero(buffer + buf_zeroed, r * mjMAXSUPER - buf_zeroed);
        buf_zeroed = r * mjMAXSUPER;
      }

      // supernode block: block[t][s] = res[r+s, r+t] (transposed for SIMD)
      mjtNum block[mjMAXSUPER][mjMAXSUPER] = {{0}};
      mjtNum scale[mjMAXSUPER] = {0};

      // accumulate: res[r:r+ns-1, :] = sum over k in M'[r:r+ns-1, :] of
      //   diag[k] * M'[r:r+ns-1, k] * M[k, :]
      for (int i = 0; i < nnzT_r; i++) {
        int k = colindT[adr_r + i];
        if (diag) {
          mjtNum dk = diag[k];
          if (dk == 0) continue;
          for (int s = 0; s < ns; s++) {
            scale[s] = dk * matT_s[s][i];
          }
        } else {
          for (int s = 0; s < ns; s++) {
            scale[s] = matT_s[s][i];
          }
        }

        // accumulate supernode block and off-diagonal row entries (specialize SIMD width=4)
        int adr_k = rowadr[k];
        if (ns <= 4) {
          accumSuperRow(block, buffer, scale, matT_s,
                        colind + adr_k, mat + adr_k, i, r, ns, any_off, 4);
        } else {
          accumSuperRow(block, buffer, scale, matT_s,
                        colind + adr_k, mat + adr_k, i, r, ns, any_off, mjMAXSUPER);
        }
      }

      // write off-diagonal columns c < r and supernode lower triangle
      for (int s = 0; s < ns; s++) {
        const int* res_ind = res_colind + res_adr_s[s];
        mjtNum* res_val = res + res_adr_s[s];
        int off_s = off_nnz_s[s];

        // copy off-diagonal entries from buffer to result
        for (int j = 0; j < off_s; j++) {
          int c = res_ind[j];
          res_val[j] = buffer[c * mjMAXSUPER + s];
          buffer[c * mjMAXSUPER + s] = 0;
        }

        // copy supernode lower triangle entries from block to result
        for (int t = 0; t <= s; t++) {
          res_val[off_s + t] = block[t][s];
        }
      }
    }
  }

  // fill upper triangle: mirror values from lower triangle
  if (res_diagind) {
    // initialize write positions after diagonal
    int* upper_pos = mjSTACKALLOC(d, nc, int);
    for (int r = 0; r < nc; r++) {
      upper_pos[r] = res_diagind[r] + 1;
    }

    // for each (r, c) with c < r, write r to row c
    for (int r = 0; r < nc; r++) {
      int adr = res_rowadr[r];
      int lower_nnz = res_diagind[r] - adr + 1;
      for (int j = 0; j < lower_nnz; j++) {
        int c = res_colind[adr + j];
        if (c < r) {
          res[upper_pos[c]++] = res[adr + j];
        }
      }
    }
  }

  mj_freeStack(d);
}


// compute sparse M'*diag*M (diag=NULL: compute M'*M), res_rowadr must be precomputed
void mju_sqrMatTDSparse(mjtNum* res, const mjtNum* mat, const mjtNum* matT,
                        const mjtNum* diag, int nr, int nc,
                        int* res_rownnz, const int* res_rowadr, int* res_colind,
                        const int* rownnz, const int* rowadr,
                        const int* colind, const int* rowsuper,
                        const int* rownnzT, const int* rowadrT,
                        const int* colindT, const int* rowsuperT,
                        mjData* d, int* diagind) {
  mj_markStack(d);

  // reinterpret transposed matrices as compressed sparse column
  const mjtNum* mat_csc  = matT;
  const int* colnnz      = rownnzT;
  const int* coladr      = rowadrT;
  const int* rowind      = colindT;
  const int* colsuper    = rowsuperT;
  const mjtNum* matT_csc = mat;
  const int* colnnzT     = rownnz;
  const int* coladrT     = rowadr;
  const int* rowindT     = colind;
  // rowsuper is unused

  // marker[i] = 1 if row i is set in current column
  int* marker = mjSTACKALLOC(d, nc, int);
  mju_zeroInt(marker, nc);

  // dense buffer (considered column-major) containing up to mjMAXSUPER columns
  mjtNum* buffer = mjSTACKALLOC(d, nc*mjMAXSUPER, mjtNum);

  // dense index vector of the current column (unsorted)
  int* buffer_idx = mjSTACKALLOC(d, nc, int);

  // rowstart[i]: address of first row in column mat'[:, i] with index > current column
  int* rowstart = mjSTACKALLOC(d, nr, int);
  mju_zeroInt(rowstart, nr);

  // clear res_rownnz
  mju_zeroInt(res_rownnz, nc);

  // construct res[lower+diagonal], by column
  for (int c=0; c < nc; c++) {
    int buffer_nnz = 0;

    // prepare column c of mat
    int nnz = colnnz[c];
    int adr = coladr[c];
    const int* ind = rowind + adr;

    // val: array of ns > 0 column pointers with identical pattern to c
    const mjtNum* val[mjMAXSUPER];

    // first column is c
    int ns = 1;
    val[0] = mat_csc + adr;

    // add c's supernodes, if any
    int cs;
    if (colsuper && (cs = colsuper[c])) {
      ns += mjMIN(cs, mjMAXSUPER - 1);
      for (int s=1; s < ns; s++) {
        val[s] = mat_csc + coladr[c + s];
      }
    }

    // diagonal special-case: dense dot product of column c, with/out diag
    mjtNum diag_c[mjMAXSUPER];
    if (diag) {
      for (int s=0; s < ns; s++) {
        mjtNum ds = 0;
        for (int k=0; k < nnz; k++) {
          ds += (val[s][k] * val[s][k]) * diag[ind[k]];
        }
        diag_c[s] = ds;
      }
    } else {
      for (int s=0; s < ns; s++) {
        diag_c[s] = mju_dot(val[s], val[s], nnz);
      }
    }

    // in the strict lower triangle, compute
    // res[:, c] = mat' * mat[:, c] = sum_r(diag[r] * mat'[:, r] * mat[:, c])
    for (int i=0; i < nnz; i++) {
      // prepare column r of mat'
      int r = ind[i];
      int adrT = coladrT[r];
      int nnzT = colnnzT[r];
      const int* indT = rowindT + adrT;
      const mjtNum* valT = matT_csc + adrT;

      // get v[s] = diag[r] * mat[r, c + s] for s in [0, ns)
      mjtNum v[mjMAXSUPER];
      if (diag) {
        mjtNum diag_r = diag[r];
        for (int s=0; s < ns; s++) {
          v[s] = diag_r * val[s][i];
        }
      } else {
        for (int s=0; s < ns; s++) {
          v[s] = val[s][i];
        }
      }

      // gather to dense buffer columns:  buffer[:, s] += mat'[:, r] * v[s]
      for (int k=rowstart[r]; k < nnzT; k++) {
        int j = indT[k];

        // if j is not in the strict lower triangle, increment rowstart and continue
        if (j <= c) {
          rowstart[r]++;
          continue;
        }

        // first nonzero in row j: mark and set value
        if (!marker[j]) {
          // mark j and save it
          marker[j] = 1;
          buffer_idx[buffer_nnz++] = j;

          // set value
          mjtNum vk = valT[k];
          for (int s=0; s < ns; s++) {
            buffer[s*nc + j] = vk * v[s];
          }
        }

        // otherwise existing nonzero in row j: add to value
        else {
          mjtNum vk = valT[k];
          for (int s=0; s < ns; s++) {
            buffer[s*nc + j] += vk * v[s];
          }
        }
      }
    }

    // scatter to res from dense buffer:  res[:, c + s] = buffer[:, s] for s in [0, ns)

    // write values under diagonal
    for (int i=0; i < buffer_nnz; i++) {
      int j = buffer_idx[i];
      marker[j] = 0;
      int adr_j = res_rowadr[j] + res_rownnz[j];

      // truncate row to strict lower triangle
      int lower = j - c;
      int nm = mjMIN(ns, lower);

      // increment nonzeros
      res_rownnz[j] += nm;

      // write value
      for (int s=0; s < nm; s++) {
        res[adr_j + s] = buffer[s*nc + j];
      }

      // write index
      for (int s=0; s < nm; s++) {
        res_colind[adr_j + s] = c + s;
      }
    }

    // write diagonal value
    for (int s=0; s < ns; s++) {
      int adr_s = res_rowadr[c + s] + res_rownnz[c + s]++;
      res_colind[adr_s] = c + s;
      res[adr_s] = diag_c[s];
    }

    // supernode: skip ahead if ns > 1
    c += ns - 1;
  }

  // upper triangle requested: save diagonal indices and fill
  if (diagind) {
    // save diagonal indices
    for (int i=0; i < nc; i++) {
      diagind[i] = res_rowadr[i] + res_rownnz[i] - 1;
    }

    // fill upper triangle
    for (int i=0; i < nc; i++) {
      int start = res_rowadr[i];
      int end = start + res_rownnz[i] - 1;
      for (int j=start; j < end; j++) {
        int adr = res_rowadr[res_colind[j]] + res_rownnz[res_colind[j]]++;
        res[adr] = res[j];
        res_colind[adr] = i;
      }
    }
  }

  mj_freeStack(d);
}

#undef mjMAXSUPER



// legacy row-based implementation (reference)
void mju_sqrMatTDSparse_row(mjtNum* res, const mjtNum* mat, const mjtNum* matT,
                            const mjtNum* diag, int nr, int nc,
                            int* res_rownnz, const int* res_rowadr, int* res_colind,
                            const int* rownnz, const int* rowadr,
                            const int* colind, const int* rowsuper,
                            const int* rownnzT, const int* rowadrT,
                            const int* colindT, const int* rowsuperT,
                            mjData* d, int* diagind) {
  // allocate space for accumulation buffer and matT
  mj_markStack(d);

  // a dense row buffer that stores the current row in the resulting matrix
  mjtNum* buffer = mjSTACKALLOC(d, nc, mjtNum);

  // these mark the currently set columns in the dense row buffer,
  // used for when creating the resulting sparse row
  int* markers = mjSTACKALLOC(d, nc, int);

  for (int i=0; i < nc; i++) {
    int rowadr_i = res_rowadr[i];
    int* cols = res_colind + rowadr_i;

    res_rownnz[i] = 0;
    buffer[i] = 0;
    markers[i] = 0;

    // if rowsuper, use the previous row sparsity structure
    if (rowsuperT && i > 0 && rowsuperT[i-1]) {
      res_rownnz[i] = res_rownnz[i-1];
      mju_copyInt(cols, res_colind+res_rowadr[i-1], res_rownnz[i]);
    }

    // iterate through each row of M'
    int adrT = rowadrT[i];
    int end_r = adrT + rownnzT[i];
    for (int r = adrT; r < end_r; r++) {
      int t = colindT[r];
      int adr = rowadr[t];
      int end_c = adr + rownnz[t];
      for (int c=adr; c < end_c; c++) {
        int cc = colind[c];

        // ignore upper triangle
        if (cc > i) {
          break;
        }

        // add value to buffer
        if (diag) {
          buffer[cc] += matT[r] * diag[t] * mat[c];
        } else {
          buffer[cc] += matT[r] * mat[c];
        }

        // only need to insert nnz if not marked
        if (!markers[cc]) {
          markers[cc] = 1;

          // since i is the rightmost column, it can be inserted at the end
          if (cc == i) {
            cols[res_rownnz[i]++] = cc;
            continue;
          }

          // insert col in order via binary search
          int l = 0, h = res_rownnz[i];
          while (l < h) {
            int m = (l + h) >> 1;
            if (cols[m] < cc) {
              l = m + 1;
            } else {
              h = m;
            }
          }

          // cc is the rightmost column so far, it can be inserted at the end
          if (l == res_rownnz[i]) {
            cols[l] = cc;
            res_rownnz[i]++;
            continue;
          }

          // move the cols to the right
          h = res_rownnz[i];
          while (l < h) {
            cols[h] = cols[h-1];
            h--;
          }

          // insert
          cols[l] = cc;
          res_rownnz[i]++;
        }
      }
    }

    end_r = res_rownnz[i];

    // rowsuperT: reuse sparsity, copy into res
    if (rowsuperT && rowsuperT[i]) {
      for (int r=0; r < end_r; r++) {
        int c = cols[r];
        res[rowadr_i + r] = buffer[c];
        buffer[c] = 0;
      }
    }

    // clear out buffers, sparsity cannot be reused
    else {
      for (int r=0; r < end_r; r++) {
        int c = cols[r];
        int adr = rowadr_i + r;
        res[adr] = buffer[c];
        res_colind[adr] = c;
        buffer[c] = 0;
        markers[c] = 0;
      }
    }
  }


  // diagonal indices requested: fill upper triangle
  if (diagind) {
    // save diagonal indices
    for (int i=0; i < nc; i++) {
      diagind[i] = res_rowadr[i] + res_rownnz[i] - 1;
    }

    // fill upper triangle
    for (int i=0; i < nc; i++) {
      int start = res_rowadr[i];
      int end = start + res_rownnz[i] - 1;
      for (int j=start; j < end; j++) {
        int adr = res_rowadr[res_colind[j]] + res_rownnz[res_colind[j]]++;
        res[adr] = res[j];
        res_colind[adr] = i;
      }
    }
  }

  mj_freeStack(d);
}


// extract a single block of a dense matrix
//   res         output matrix
//   mat         input matrix
//   nc_mat      number of columns in mat
//   nc_res      number of columns in res
//   nr          number of rows in res
//   perm_r      reverse permutation of rows (res -> mat)
//   perm_c      reverse permutation of columns (res -> mat)
void mju_block(mjtNum* restrict res, const mjtNum* restrict mat,
               int nc_mat, int nc_res, int nr,
               const int* restrict perm_r, const int* restrict perm_c) {
  for (int r = 0; r < nr; r++) {
    mjtNum* res_r = res + r * nc_res;
    const mjtNum* mat_r = mat + perm_r[r] * nc_mat;
    mju_gather(res_r, mat_r, perm_c, nc_res);
  }
}

// block-diagonalize a dense matrix
//   res         target matrix
//   mat         source full matrix
//   nc_mat      number of columns in source matrix
//   nc_res      number of columns in the block-diagonal target matrix
//   nb          number of blocks
//   perm_r      reverse permutation of rows (res -> mat)
//   perm_c      reverse permutation of columns (res -> mat)
//   block_nr    number of rows in each block
//   block_nc    number of columns in each block
//   block_r     first row of each block
//   block_c     first column of each block
void mju_blockDiag(mjtNum* restrict res, const mjtNum* restrict mat,
                   int nc_mat, int nc_res, int nb,
                   const int* restrict perm_r, const int* restrict perm_c,
                   const int* restrict block_nr, const int* restrict block_nc,
                   const int* restrict block_r, const int* restrict block_c) {
  for (int b = 0; b < nb; b++) {
    int adr = nc_res * block_r[b];
    mju_block(res + adr, mat, nc_mat, block_nc[b], block_nr[b],
              perm_r + block_r[b], perm_c + block_c[b]);
  }
}


// extract a single block of a sparse matrix
//   res, res2     target matrix values
//   res_rownnz    non-zeros in each row of target matrix
//   res_rowadr    row address of each initial row in target matrix
//   res_colind    column indices for each extracted value (relative)
//   mat, mat2     source matrix values
//   rownnz        non-zeros in each row of source matrix
//   rowadr        addresses within the source matrix values
//   colind        source matrix column indices
//   nr            number of rows to extract
//   perm_r        row permutation (maps local row to source row)
//   perm_c        column permutation (maps source col to local col)
//   col_offset    subtrahend to shift mapped absolute columns into relative block space
//   res_offset    rowadr starting offset for the extracted submatrix
void mju_blockSparse(mjtNum* restrict res, int* restrict res_rownnz,
                     int* restrict res_rowadr, int* restrict res_colind,
                     const mjtNum* restrict mat, const int* restrict rownnz,
                     const int* restrict rowadr, const int* restrict colind,
                     int nr,
                     const int* restrict perm_r, const int* restrict perm_c,
                     int col_offset, int res_offset,
                     mjtNum* restrict res2, const mjtNum* restrict mat2) {
  for (int r = 0; r < nr; r++) {
    int k = perm_r[r];
    int nnz = rownnz[k];
    res_rownnz[r] = nnz;

    int res_adr = (r == 0) ? res_offset : (res_rowadr[r-1] + res_rownnz[r-1]);
    res_rowadr[r] = res_adr;

    int* res_colind_r = res_colind + (res_adr - res_offset);
    int mat_adr = rowadr[k];
    const int* colind_k = colind + mat_adr;
    for (int j = 0; j < nnz; j++) {
      res_colind_r[j] = perm_c[colind_k[j]] - col_offset;
    }

    mju_copy(res + (res_adr - res_offset), mat + mat_adr, nnz);
    if (mat2 && res2) {
      mju_copy(res2 + (res_adr - res_offset), mat2 + mat_adr, nnz);
    }
  }
}


// block-diagonalize a sparse matrix
//   res         values of the target matrix res
//   res_rownnz  number of non-zeros in each row of res
//   res_rowadr  row address of each non-zero in res
//   res_colind  column index of each non-zero in res
//   mat         values of the source matrix mat
//   mat_rownnz  number of non-zeros in each row of mat
//   mat_rowadr  row address of each non-zero in mat
//   mat_colind  column index of each non-zero in mat
//   nr          number of rows in mat/res
//   nb          number of blocks
//   perm_r      reverse permutation of rows (res -> mat)
//   perm_c      forward permutation of columns (mat -> res)
//   block_r     first row of each block in res
//   block_c     first column of each block in res
//   mat2        optional additional source matrix (same structure as mat)
//   res2        optional additional target matrix (same structure as res)
void mju_blockDiagSparse(mjtNum* restrict res, int* restrict res_rownnz,
                         int* restrict res_rowadr, int* restrict res_colind,
                         const mjtNum* restrict mat, const int* restrict rownnz,
                         const int* restrict rowadr, const int* restrict colind,
                         int nr, int nb,
                         const int* restrict perm_r, const int* restrict perm_c,
                         const int* restrict block_r, const int* restrict block_c,
                         mjtNum* restrict res2, const mjtNum* restrict mat2) {
  for (int b = 0; b < nb; b++) {
    int nr_block = (b + 1 < nb ? block_r[b + 1] : nr) - block_r[b];
    int res_adr = (block_r[b] == 0) ? 0 : (res_rowadr[block_r[b] - 1] + res_rownnz[block_r[b] - 1]);

    mju_blockSparse(res + res_adr, res_rownnz + block_r[b],
                    res_rowadr + block_r[b], res_colind + res_adr,
                    mat, rownnz, rowadr, colind,
                    nr_block, perm_r + block_r[b], perm_c,
                    block_c[b], res_adr,
                    res2 ? res2 + res_adr : NULL, mat2);
  }
}


// sort sparse row (colind, val) of length nnz ascending by colind
void mju_sortSparseRow(int nnz, int* restrict colind, mjtNum* restrict val) {
  int gap0 = nnz > 24 ? (nnz >> 1) : 1;
  for (int gap=gap0; gap > 0; gap >>= 1) {
    for (int a=gap; a < nnz; a++) {
      int c = colind[a];
      mjtNum v = val[a];
      int b = a - gap;
      while (b >= 0 && colind[b] > c) {
        colind[b+gap] = colind[b];
        val[b+gap] = val[b];
        b -= gap;
      }
      colind[b+gap] = c;
      val[b+gap] = v;
    }
  }
}


// permute lower-triangular + diagonal CSR matrix into sorted lower-triangular + diagonal CSR
void mju_permuteLowerCSR(int nv, const int* restrict perm,
                         int* restrict dst_rownnz, int* restrict dst_rowadr,
                         int* restrict dst_colind, mjtNum* restrict dst_val,
                         const int* restrict src_rownnz, const int* restrict src_rowadr,
                         const int* restrict src_colind, const mjtNum* restrict src_val) {
  mju_zeroInt(dst_rownnz, nv);
  for (int r=0; r < nv; r++) {
    int pr = perm[r];
    int end = src_rowadr[r] + src_rownnz[r];
    for (int j=src_rowadr[r]; j < end; j++) {
      int pc = perm[src_colind[j]];
      dst_rownnz[pr >= pc ? pr : pc]++;
    }
  }
  int adr = 0;
  for (int r=0; r < nv; r++) {
    dst_rowadr[r] = adr;
    adr += dst_rownnz[r];
    dst_rownnz[r] = 0;
  }
  for (int r=0; r < nv; r++) {
    int pr = perm[r];
    int end = src_rowadr[r] + src_rownnz[r];
    for (int j=src_rowadr[r]; j < end; j++) {
      int pc = perm[src_colind[j]];
      int row = pr >= pc ? pr : pc;
      int col = pr >= pc ? pc : pr;
      int w = dst_rowadr[row] + dst_rownnz[row]++;
      dst_colind[w] = col;
      dst_val[w] = src_val[j];
    }
  }
  for (int r=0; r < nv; r++) {
    mju_sortSparseRow(dst_rownnz[r], dst_colind+dst_rowadr[r], dst_val+dst_rowadr[r]);
  }
}


// degree-bucketed greedy minimum-degree ordering on undirected bitset graph adj (nb x words,
// modified in-place); writes reverse-Cholesky block permutation bperm[v] in [0, nb).
// If bsz is non-NULL, accumulates block-supernode Cholesky nL and flops, and aborts early
// (returns 0) if 5 * (*flops_out) >= 4 * max_flops.
// Scratch: deg_scratch has length 5*nb + words, nz_bits has length words.
int mju_minDegree(int* restrict bperm, uint64_t* restrict adj, int nb,
                  const int* restrict bsz, size_t* nL_out, mjtSize* flops_out, mjtSize max_flops,
                  int* restrict deg_scratch, uint64_t* restrict nz_bits) {
  int words = (nb + 63) / 64;
  int* deg  = deg_scratch;
  int* head = deg + nb;
  int* next = head + nb;
  int* prev = next + nb;
  int* nbr  = prev + nb;
  int* nz_w = nbr + nb;

  for (int d=0; d < nb; d++) {
    head[d] = -1;
  }

  // initialize degrees and insert into degree buckets in descending k so bucket heads start ascending
  int mindeg = nb;
  for (int k=nb-1; k >= 0; k--) {
    uint64_t* ak = adj + (size_t)k*words;
    ak[k>>6] &= ~(1ULL << (k & 63));
    int dg = 0;
    for (int w=0; w < words; w++) {
      dg += mju_popcount64(ak[w]);
    }
    deg[k] = dg;
    int h = head[dg];
    prev[k] = -1;
    next[k] = h;
    if (h >= 0) {
      prev[h] = k;
    }
    head[dg] = k;
    if (dg < mindeg) {
      mindeg = dg;
    }
  }

  size_t nL = 0;
  mjtSize flops = 0;
  for (int step=0; step < nb; step++) {
    while (head[mindeg] < 0) {
      mindeg++;
    }
    int v = head[mindeg];
    int nxt = next[v];
    head[mindeg] = nxt;
    if (nxt >= 0) {
      prev[nxt] = -1;
    }
    deg[v] = -1;

    // extract alive neighbors of v and accumulate block-supernode Cholesky flops if requested
    const uint64_t* av = adj + (size_t)v*words;
    int nw = 0, nnbr = 0, w_off = 0;
    for (int w=0; w < words; w++) {
      uint64_t bits = av[w];
      if (bits) {
        nz_w[nw] = w;
        nz_bits[nw++] = bits;
        while (bits) {
          int u = (w << 6) + mju_ctz64(bits);
          bits &= bits - 1;
          nbr[nnbr++] = u;
          if (bsz) {
            w_off += bsz[u];
          }
        }
      }
    }
    if (bsz) {
      int sv = bsz[v];
      for (int a=1; a <= sv; a++) {
        mjtSize lnnz = w_off + a;
        nL += (size_t)lnnz;
        flops += lnnz * lnnz;
      }
      if (max_flops && 5 * flops >= 4 * max_flops) {
        if (nL_out) *nL_out = nL;
        if (flops_out) *flops_out = flops;
        return 0;
      }
    }

    bperm[v] = nb - 1 - step;

    // remove v from neighbors' adjacency rows and add fill clique among v's neighbors
    int vw = v >> 6;
    uint64_t vmask = ~(1ULL << (v & 63));
    for (int idx=0; idx < nnbr; idx++) {
      int u = nbr[idx];
      int old_dg = deg[u];
      int p = prev[u], n = next[u];
      if (p >= 0) {
        next[p] = n;
      } else {
        head[old_dg] = n;
      }
      if (n >= 0) {
        prev[n] = p;
      }

      uint64_t* au = adj + (size_t)u*words;
      au[vw] &= vmask;
      int dg = old_dg - 1;
      int uw = u >> 6;
      uint64_t ubit = 1ULL << (u & 63);
      for (int i=0; i < nw; i++) {
        int w = nz_w[i];
        uint64_t add = nz_bits[i] & ~au[w];
        if (w == uw) {
          add &= ~ubit;
        }
        if (add) {
          au[w] |= add;
          dg += mju_popcount64(add);
        }
      }

      deg[u] = dg;
      int h = head[dg];
      prev[u] = -1;
      next[u] = h;
      if (h >= 0) {
        prev[h] = u;
      }
      head[dg] = u;
      if (dg < mindeg) {
        mindeg = dg;
      }
    }
  }

  if (nL_out) *nL_out = nL;
  if (flops_out) *flops_out = flops;
  return 1;
}
