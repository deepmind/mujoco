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

#include "engine/engine_util_solve.h"

#include <stdio.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_crossplatform.h"
#include "engine/engine_inline.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_memory.h"
#include "engine/engine_util_spatial.h"

//---------------------------- dense Cholesky ------------------------------------------------------

// Cholesky decomposition: mat = L*L'; return 'rank'
int mju_cholFactor(mjtNum* restrict mat, int n, mjtNum mindiag) {
  int rank = n;
  mjtNum tmp;

  // in-place Cholesky factorization
  for (int j=0; j < n; j++) {
    // compute new diagonal
    tmp = mat[j*(n+1)];
    if (j) {
      tmp -= mju_dot(mat+j*n, mat+j*n, j);
    }

    // correct diagonal values below threshold (negated >= so NaN counts as deficient)
    int deficient = !(tmp >= mindiag);
    if (deficient) {
      tmp = mindiag;
      rank--;
    }

    // save diagonal
    mat[j*(n+1)] = mju_sqrt(tmp);

    // process off-diagonal entries
    if (deficient) {
      // clear off-diagonals if deficient
      for (int i=j+1; i < n; i++) {
        mat[i*n+j] = 0;
      }
    } else {
      tmp = 1/mat[j*(n+1)];
      for (int i=j+1; i < n; i++) {
        mat[i*n+j] = (mat[i*n+j] - mju_dot(mat+i*n, mat+j*n, j)) * tmp;
      }
    }
  }

  return rank;
}


// Cholesky solve
void mju_cholSolve(mjtNum* res, const mjtNum* restrict mat, const mjtNum* vec, int n) {
  // copy if source and destination are different
  if (res != vec) {
    mju_copy(res, vec, n);
  }

  // forward substitution: solve L*res = vec
  for (int i=0; i < n; i++) {
    // off-diagonal
    if (i) {
      res[i] -= mju_dot(mat+i*n, res, i);
    }

    // diagonal
    res[i] /= mat[i*(n+1)];
  }

  // backward substitution: solve L'*res = res
  for (int i=n-1; i >= 0; i--) {
    // diagonal
    res[i] /= mat[i*(n+1)];

    // update res[0..i-1] with row i of L
    if (i && res[i]) {
      mju_addToScl(res, mat+i*n, -res[i], i);
    }
  }
}


// Cholesky rank-one update: L*L' +/- x*x'; return rank
int mju_cholUpdate(mjtNum* restrict mat, mjtNum* restrict x, int n, int flg_plus) {
  int rank = n;
  mjtNum r, c, cinv, s, Lkk, tmp;

  for (int k=0; k < n; k++) {
    if (x[k]) {
      // prepare constants
      Lkk = mat[k*(n+1)];
      tmp = Lkk*Lkk + (flg_plus ? x[k]*x[k] : -x[k]*x[k]);
      if (!(tmp >= mjMINVAL)) {
        tmp = mjMINVAL;
        rank--;
      }
      r = mju_sqrt(tmp);
      c = r / Lkk;
      cinv = 1 / c;
      s = x[k] / Lkk;

      // update diagonal
      mat[k*(n+1)] = r;

      // update mat and x
      mjtNum s_signed = flg_plus ? s : -s;
      for (int i=k+1; i < n; i++) {
        mjtNum m = (mat[i*n+k] + s_signed*x[i])*cinv;
        mat[i*n+k] = m;
        x[i] = c*x[i] - s*m;
      }
    }
  }

  return rank;
}


//---------------------------- sparse Cholesky -----------------------------------------------------

// sparse reverse-order Cholesky decomposition: mat = L'*L; return 'rank'
//  mat must be lower-triangular, have preallocated space for fill-in
int mju_cholFactorSparse(mjtNum* mat, int n, mjtNum mindiag,
                         int* rownnz, const int* rowadr, int* colind,
                         mjData* d) {
  int rank = n;
  (void) d;

  // backpass over rows
  for (int r=n-1; r >= 0; r--) {
    // get rownnz and rowadr for row r
    int nnz = rownnz[r], adr = rowadr[r];

    // update row r diagonal
    mjtNum tmp = mat[adr+nnz-1];
    int deficient = !(tmp >= mindiag);
    if (deficient) {
      tmp = mindiag;
      rank--;
    }
    mat[adr+nnz-1] = mju_sqrt(tmp);

    // clear off-diagonals if deficient
    if (deficient) {
      mju_zero(mat+adr, nnz-1);
    } else {
      tmp = 1/mat[adr+nnz-1];

      // update row r before diagonal
      for (int i=0; i < nnz-1; i++) {
        mat[adr+i] *= tmp;
      }
    }

    // update row c<r where mat(r,c)!=0 (zero coefficients still merge the pattern)
    for (int i=0; i < nnz-1; i++) {
      // get column index
      int c = colind[adr+i];

      // mat(c,0:c) = mat(c,0:c) - mat(r,c) * mat(r,0:c)
      int nnz_c = mju_combineSparse(mat + rowadr[c], mat+rowadr[r], 1, -mat[adr+i],
                                    rownnz[c], i+1, colind+rowadr[c], colind+rowadr[r]);

      // assign new nnz to row c
      rownnz[c] = nnz_c;
    }
  }

  return rank;
}


// return index of first upper-triangular column (> r) in sorted row colind[start .. end-1]
static inline int upperStart(const int* colind, int start, int end, int r) {
  if (start < end && colind[start] <= r) {
    if (colind[start] == r) {
      return start + 1;
    }
    int lo = start + 1, hi = end;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (colind[mid] <= r) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }
  return start;
}


// symbolic reverse-Cholesky: compute both L (CSR) and LT (CSC) structures
//   if L_colind is NULL, perform counting logic (fill rownnz/rowadr arrays and return total nnz)
//   if L_colind is not NULL, assume rownnz/rowadr are precomputed and fill colind/map arrays
//   reads pattern from upper triangle
//   based on ldl_symbolic from 'Algorithm 8xx: a concise sparse Cholesky factorization package'
int mju_cholFactorSymbolic(int* restrict L_colind, int* restrict L_rownnz, int* restrict L_rowadr,
                           int* restrict LT_colind, int* restrict LT_rownnz,
                           int* restrict LT_rowadr, int* restrict LT_map,
                           const int* restrict rownnz, const int* restrict rowadr,
                           const int* restrict colind, int n, mjData* d) {
  // d supplies stack scratch; if NULL, scratch is heap-allocated
  if (d) {
    mj_markStack(d);
  }
  int* restrict parent = d ? mjSTACKALLOC(d, n, int) : (int*) mju_malloc(sizeof(int)*n);
  int* restrict flag = d ? mjSTACKALLOC(d, n, int) : (int*) mju_malloc(sizeof(int)*n);

  // counting phase: fill rownnz/rowadr and return total non-zeros
  if (!L_colind) {
    // loop over rows in reverse order
    for (int r=n-1; r >= 0; r--) {
      parent[r] = -1;
      flag[r] = r;

      // start with 1 for diagonal
      L_rownnz[r] = 1;
      int lt_nnz = 1;

      // loop over non-zero columns of upper triangle (> r)
      int end = rowadr[r] + rownnz[r];
      for (int c=upperStart(colind, rowadr[r], end, r); c < end; c++) {
        int i = colind[c];

        // traverse elimination tree from i toward root, stop when row is already flagged for r
        while (flag[i] != r) {
          // count non-zero in L[i, r] and LT[r, i], mark visited, attach unparented root to r
          L_rownnz[i]++;
          lt_nnz++;
          flag[i] = r;
          int p = parent[i];
          if (p == -1) {
            parent[i] = r;
            break;
          }
          i = p;
        }
      }
      LT_rownnz[r] = lt_nnz;
    }

    if (d) {
      mj_freeStack(d);
    } else {
      mju_free(parent);
      mju_free(flag);
    }

    // compute row addresses, add up total non-zeros
    int nnz = L_rownnz[0];
    L_rowadr[0] = 0;
    LT_rowadr[0] = 0;
    for (int r=1; r < n; r++) {
      L_rowadr[r] = L_rowadr[r-1] + L_rownnz[r-1];
      LT_rowadr[r] = LT_rowadr[r-1] + LT_rownnz[r-1];
      nnz += L_rownnz[r];
    }
    return nnz;
  }

  // filling phase: populate L_colind, LT_colind, and LT_map
  // cursor[i] tracks next write slot in row i of L (filled right-to-left as r decreases)
  int* restrict cursor = d ? mjSTACKALLOC(d, n, int) : (int*) mju_malloc(sizeof(int)*n);

  // loop over rows in reverse order
  for (int r=n-1; r >= 0; r--) {
    parent[r] = -1;
    flag[r] = r;

    // write L diagonal at end of row r and initialize write cursor just before diagonal
    int diag_idx = L_rowadr[r] + L_rownnz[r] - 1;
    cursor[r] = diag_idx - 1;
    L_colind[diag_idx] = r;

    // write LT diagonal at head of transpose row r
    int lt_write = LT_rowadr[r];
    LT_colind[lt_write] = r;
    LT_map[lt_write] = diag_idx;
    lt_write++;

    // loop over non-zero columns of upper triangle (> r)
    int end = rowadr[r] + rownnz[r];
    for (int c=upperStart(colind, rowadr[r], end, r); c < end; c++) {
      int i = colind[c];

      // traverse elimination tree from i toward root, stop when row is already flagged for r
      while (flag[i] != r) {
        // record column r in L[i] and row i (with its L index) in LT[r]
        int L_idx = cursor[i]--;
        L_colind[L_idx] = r;
        LT_colind[lt_write] = i;
        LT_map[lt_write] = L_idx;
        lt_write++;

        // mark row i visited for r, advance to parent (or attach unparented root to r)
        flag[i] = r;
        int p = parent[i];
        if (p == -1) {
          parent[i] = r;
          break;
        }
        i = p;
      }
    }
  }

  if (d) {
    mj_freeStack(d);
  } else {
    mju_free(parent);
    mju_free(flag);
    mju_free(cursor);
  }

  return 0;
}


// symbolic reverse-Cholesky of a matrix of dense bs x bs blocks, from its block pattern
//   mju_cholFactorSymbolic on the expanded pattern walks the dofs of each block together: the
//   rows of a block share their block columns, and the dof elimination tree is the block
//   elimination tree with each block a chain. So the walks run on the blocks, each visit filling
//   one block slot of the rows of L and one entry of the block transpose
int mju_cholFactorSymbolicBlocked(int* restrict L_colind, int* restrict L_rownnz,
                                  int* restrict L_rowadr, int* restrict BT_adr,
                                  int* restrict BT_row, int* restrict BT_off, const int* rownnz,
                                  const int* rowadr, const int* colind, int nb, int bs,
                                  int* scratch) {
  int* restrict parent = scratch;          // block elimination tree
  int* restrict flag = scratch + nb;       // the last block column whose walk reached the block
  int* restrict count = scratch + 2*nb;    // counting: block columns of the block's rows
                                           // filling: those still to fill, right to left

  // counting phase: the block walks, then the dof row sizes and addresses
  if (!L_colind) {
    for (int q = nb - 1; q >= 0; q--) {
      parent[q] = -1;
      flag[q] = q;
      count[q] = 0;
      int ncol = 0;
      int end = rowadr[q] + rownnz[q];
      for (int j = upperStart(colind, rowadr[q], end, q); j < end; j++) {
        int p = colind[j];
        while (flag[p] != q) {
          flag[p] = q;
          count[p]++;
          ncol++;
          int par = parent[p];
          if (par == -1) {
            parent[p] = q;
            break;
          }
          p = par;
        }
      }
      BT_adr[q] = ncol;   // block rows its column reaches, accumulated below
    }
    int nnz = 0, nnzT = 0;
    for (int p = 0; p < nb; p++) {
      int ncol = BT_adr[p];
      BT_adr[p] = nnzT;
      nnzT += ncol;
      for (int a = 0; a < bs; a++) {
        int r = bs*p + a;
        L_rownnz[r] = bs*count[p] + a + 1;
        L_rowadr[r] = nnz;
        nnz += L_rownnz[r];
      }
    }
    BT_adr[nb] = nnzT;
    return nnz;
  }

  // filling phase: the diagonal block and cursor of q, then the walk from its upper columns
  for (int q = nb - 1; q >= 0; q--) {
    int x = BT_adr[q];
    parent[q] = -1;
    flag[q] = q;
    count[q] = (L_rownnz[bs*q] - 1) / bs;
    int off = bs*count[q];
    for (int c = 0; c < bs; c++) {
      for (int b = 0; b <= c; b++) {
        L_colind[L_rowadr[bs*q + c] + off + b] = bs*q + b;
      }
    }
    int end = rowadr[q] + rownnz[q];
    for (int j = upperStart(colind, rowadr[q], end, q); j < end; j++) {
      int p = colind[j];
      while (flag[p] != q) {
        flag[p] = q;
        int ip = --count[p];
        for (int c = 0; c < bs; c++) {
          int* col = L_colind + L_rowadr[bs*p + c] + bs*ip;
          for (int b = 0; b < bs; b++) {
            col[b] = bs*q + b;
          }
        }
        BT_row[x] = p;
        BT_off[x++] = bs*ip;
        int par = parent[p];
        if (par == -1) {
          parent[p] = q;
          break;
        }
        p = par;
      }
    }
  }
  return 0;
}


// suffix update of ns target rows from batch descendant rows (ns, batch are compile-time constants)
static mjALWAYS_INLINE
void cholSuffixUpdate(int ns, int batch, int nlead,
                      mjtNum* restrict dptr0, mjtNum* restrict dptr1, mjtNum* restrict dptr2,
                      const mjtNum* restrict p0, const mjtNum* restrict p1,
                      const mjtNum* restrict p2, const mjtNum* restrict p3,
                      mjtNum s[3][4]) {
  // leading columns shared by all ns rows
  for (int i=0; i < nlead; i++) {
    mjtNum v[4] = {p0[i], (batch >= 2) ? p1[i] : 0,
                   (batch >= 3) ? p2[i] : 0, (batch >= 4) ? p3[i] : 0};
    for (int b=0; b < batch; b++) dptr0[i] -= s[0][b] * v[b];
    if (ns >= 2) {
      for (int b=0; b < batch; b++) dptr1[i] -= s[1][b] * v[b];
    }
    if (ns >= 3) {
      for (int b=0; b < batch; b++) dptr2[i] -= s[2][b] * v[b];
    }
  }

  // triangular tail: column nlead+u is present in rows t <= ns-2-u
  for (int u=0; u < ns-1; u++) {
    int idx = nlead + u;
    mjtNum v[4] = {p0[idx], (batch >= 2) ? p1[idx] : 0,
                   (batch >= 3) ? p2[idx] : 0, (batch >= 4) ? p3[idx] : 0};
    for (int b=0; b < batch; b++) dptr0[idx] -= s[0][b] * v[b];
    if (u == 0 && ns >= 3) {
      for (int b=0; b < batch; b++) dptr1[idx] -= s[1][b] * v[b];
    }
  }
}


// non-suffix scatter of batch descendant rows into dense (batch is a compile-time constant)
static mjALWAYS_INLINE
void cholScatterUpdate(int batch, int num_cols, mjtNum* restrict dense,
                       const int* restrict colptr,
                       const mjtNum* restrict p0, const mjtNum* restrict p1,
                       const mjtNum* restrict p2, const mjtNum* restrict p3,
                       const mjtNum s[4]) {
  for (int i=0; i < num_cols; i++) {
    int col = colptr[i];
    mjtNum acc = dense[col] - s[0] * p0[i];
    if (batch >= 2) acc -= s[1] * p1[i];
    if (batch >= 3) acc -= s[2] * p2[i];
    if (batch >= 4) acc -= s[3] * p3[i];
    dense[col] = acc;
  }
}


// factor supernode of size ns (1, 2, or 3) starting at row r; return rank decrement (force inline)
static mjALWAYS_INLINE
int cholFactorSupernode(
    int r, int ns, mjtNum mindiag, mjtNum* restrict L,
    const int* restrict L_rownnz, const int* restrict L_rowadr, const int* restrict L_colind,
    const int* restrict LT_rownnz, const int* restrict LT_rowadr,
    const int* restrict LT_colind, const int* restrict LT_map,
    const mjtNum* restrict H,
    const int* restrict H_rownnz, const int* restrict H_rowadr, const int* restrict H_colind,
    mjtNum* restrict dense) {
  int L_adr = L_rowadr[r];
  int L_nnz = L_rownnz[r];
  const int* restrict L_colptr = L_colind + L_adr;
  int LT_adr = LT_rowadr[r];
  int LT_nnz = LT_rownnz[r];

  mjtNum* restrict Lptr0 = L + L_adr;
  mjtNum* restrict Lptr1 = (ns >= 2) ? L + L_rowadr[r-1] : NULL;
  mjtNum* restrict Lptr2 = (ns >= 3) ? L + L_rowadr[r-2] : NULL;
  mjtNum* restrict Lptr_s[3] = {Lptr0, Lptr1, Lptr2};

  // initialize L rows for the supernode from H (copy if no fill-in, else scatter/gather)
  int used_dense = 0;
  for (int s=0; s < ns; s++) {
    int row = r - s;
    int nnz_s = L_nnz - s;
    int h_adr = H_rowadr[row];
    int h_nnz = H_rownnz[row];
    mjtNum* restrict row_s = Lptr_s[s];

    if (h_nnz >= nnz_s && H_colind[h_adr+nnz_s-1] == row) {
      mju_copy(row_s, H+h_adr, nnz_s);
    } else if (ns == 1) {
      mju_zero(row_s, nnz_s);
      for (int i=0; i < h_nnz; i++) {
        int c = H_colind[h_adr+i];
        if (c > row) break;
        dense[c] = H[h_adr+i];
      }
      used_dense = 1;
    } else {
      for (int i=0; i < h_nnz; i++) {
        int c = H_colind[h_adr+i];
        if (c > row) break;
        dense[c] = H[h_adr+i];
      }
      for (int i=0; i < nnz_s; i++) {
        int c = L_colptr[i];
        row_s[i] = dense[c];
        dense[c] = 0;
      }
    }
  }

  // accumulate updates from descendant rows c > r in LT[r] (skip k=0 diagonal entry);
  // cap batch at 3 when ns == 3 so 3*3 broadcast scalars + row/descendant vectors fit in 16 YMM regs
  const int max_batch = (ns == 3) ? 3 : 4;
  int k = 1;
  while (k < LT_nnz) {
    int c0 = LT_colind[LT_adr+k];
    int idx0 = LT_map[LT_adr+k];
    int c0_adr = L_rowadr[c0];
    int num_cols = idx0 - c0_adr + 1;

    // by Cholesky elimination, cols(L[c, 0:r]) is a subset of cols(L[r, 0:r]) ending at r;
    // it matches the suffix L_colptr[L_nnz-num_cols .. L_nnz-1] iff its first column matches
    const int* restrict colptr = L_colind + c0_adr;
    const mjtNum* p[4] = {L + c0_adr, NULL, NULL, NULL};
    int idx[4] = {idx0, 0, 0, 0};
    int col0 = colptr[0];
    int is_suffix = (L_colptr[L_nnz-num_cols] == col0);
    int batch = 1;

    // fuse up to max_batch descendant rows in LT[r] that share the identical pattern on [0, r]
    if (is_suffix || ns == 1) {
      int prev_c = c0, prev_adr = c0_adr;
      while (k + batch < LT_nnz && batch < max_batch) {
        int next_c = LT_colind[LT_adr+k+batch];
        int next_idx = LT_map[LT_adr+k+batch];
        int next_adr = L_rowadr[next_c];
        if (next_idx - next_adr + 1 != num_cols) {
          break;
        }
        if (is_suffix) {
          if (L_colind[next_adr] != col0) break;
        } else {
          int hi = next_c > prev_c ? next_c : prev_c;
          int lo = next_c > prev_c ? prev_c : next_c;
          int hi_adr = next_c > prev_c ? next_adr : prev_adr;
          int lo_nnz = L_rownnz[lo];
          if (lo_nnz >= L_rownnz[hi] || L_colind[hi_adr+lo_nnz-1] != lo) break;
          prev_c = next_c;
          prev_adr = next_adr;
        }
        p[batch] = L + next_adr;
        idx[batch] = next_idx;
        batch++;
      }
    }

    // suffix fast path: update target supernode rows contiguously in L
    if (is_suffix) {
      int off = L_nnz - num_cols;
      int nlead = num_cols - (ns - 1);
      mjtNum* restrict dptr0 = Lptr0 + off;
      mjtNum* restrict dptr1 = (ns >= 2) ? Lptr1 + off : NULL;
      mjtNum* restrict dptr2 = (ns >= 3) ? Lptr2 + off : NULL;
      mjtNum s[3][4];
      for (int t=0; t < ns; t++) {
        for (int b=0; b < batch; b++) {
          s[t][b] = L[idx[b] - t];
        }
      }
      switch (batch) {
        case 1:
          cholSuffixUpdate(ns, 1, nlead, dptr0, dptr1, dptr2, p[0], p[1], p[2], p[3], s);
          break;
        case 2:
          cholSuffixUpdate(ns, 2, nlead, dptr0, dptr1, dptr2, p[0], p[1], p[2], p[3], s);
          break;
        case 3:
          cholSuffixUpdate(ns, 3, nlead, dptr0, dptr1, dptr2, p[0], p[1], p[2], p[3], s);
          break;
        case 4:
          cholSuffixUpdate(ns, 4, nlead, dptr0, dptr1, dptr2, p[0], p[1], p[2], p[3], s);
          break;
      }
    }

    // non-suffix subset: scatter row r update into dense; rows r-1, r-2 are revisited below
    else {
      used_dense = 1;
      mjtNum s[4];
      for (int b=0; b < batch; b++) {
        s[b] = L[idx[b]];
      }
      switch (batch) {
        case 1: cholScatterUpdate(1, num_cols, dense, colptr, p[0], p[1], p[2], p[3], s); break;
        case 2: cholScatterUpdate(2, num_cols, dense, colptr, p[0], p[1], p[2], p[3], s); break;
        case 3: cholScatterUpdate(3, num_cols, dense, colptr, p[0], p[1], p[2], p[3], s); break;
        case 4: cholScatterUpdate(4, num_cols, dense, colptr, p[0], p[1], p[2], p[3], s); break;
      }
    }

    k += batch;
  }

  // flush non-suffix updates from dense for s = 0 .. ns-1
  if (used_dense) {
    for (int s=0; s < ns; s++) {
      if (s > 0) {
        for (int j=1; j < LT_nnz; j++) {
          int c0 = LT_colind[LT_adr+j];
          int L_cr0_idx = LT_map[LT_adr+j];
          int c0_adr = L_rowadr[c0];
          int num_cols = L_cr0_idx - c0_adr + 1;
          if (L_colptr[L_nnz-num_cols] != L_colind[c0_adr]) {
            mjtNum scale = L[L_cr0_idx-s];
            const mjtNum* restrict p0 = L + c0_adr;
            const int* restrict colptr = L_colind + c0_adr;
            for (int i=0; i < num_cols-s; i++) {
              dense[colptr[i]] -= scale * p0[i];
            }
          }
        }
      }
      mjtNum* restrict row_s = Lptr_s[s];
      int nnz_s = L_nnz - s;
      for (int i=0; i < nnz_s; i++) {
        int c = L_colptr[i];
        row_s[i] += dense[c];
        dense[c] = 0;
      }
    }
  }

  // factor supernode rows r, r-1, ... and fuse scaling with intra-supernode elimination
  int rank_dec = 0;
  int n_off = L_nnz - ns;
  for (int s=0; s < ns; s++) {
    mjtNum* restrict row_s = Lptr_s[s];
    int nnz_s = L_nnz - s;
    mjtNum diag = row_s[nnz_s-1];
    int deficient = !(diag >= mindiag);
    if (deficient) {
      diag = mindiag;
      rank_dec++;
    }
    mjtNum L_rr = mju_sqrt(diag);
    if (deficient) {
      mju_zero(row_s, nnz_s-1);
    } else {
      mjtNum L_rr_inv = 1.0 / L_rr;
      mjtNum s_st[3] = {0, 0, 0};
      for (int t=s+1; t < ns; t++) {
        s_st[t] = (row_s[L_nnz-1-t] *= L_rr_inv);
      }
      for (int i=0; i < n_off; i++) {
        mjtNum l_si = (row_s[i] *= L_rr_inv);
        if (s + 1 < ns) Lptr_s[s+1][i] -= s_st[s+1] * l_si;
        if (s + 2 < ns) Lptr_s[s+2][i] -= s_st[s+2] * l_si;
      }
      for (int t=s+1; t < ns; t++) {
        for (int u=t; u < ns; u++) {
          Lptr_s[t][L_nnz-1-u] -= s_st[t] * s_st[u];
        }
      }
    }
    row_s[nnz_s-1] = L_rr;
  }

  return rank_dec;
}


// numeric reverse-Cholesky: compute L values given fixed sparsity pattern, returns rank
//  L_colind must already contain the correct sparsity pattern (from mju_cholFactorSymbolic)
//  LT_map[k] gives index in L for LT_colind[k]
int mju_cholFactorNumeric(mjtNum* restrict L, int n, mjtNum mindiag,
                          const int* restrict L_rownnz, const int* restrict L_rowadr,
                          const int* restrict L_colind,
                          const int* restrict LT_rownnz, const int* restrict LT_rowadr,
                          const int* restrict LT_colind, const int* restrict LT_map,
                          const mjtNum* restrict H,
                          const int* restrict H_rownnz, const int* restrict H_rowadr,
                          const int* restrict H_colind, mjtNum* restrict scratch) {
  int rank = n;

  // zero dense accumulator workspace; each row cleans up its own entries below
  mjtNum* restrict dense = scratch;
  mju_zero(dense, n);

  // backpass over rows
  int ns;
  for (int r=n-1; r >= 0; r -= ns) {
    int L_adr = L_rowadr[r];
    int L_nnz = L_rownnz[r];
    const int* restrict L_colptr = L_colind + L_adr;
    mjtNum* restrict Lptr = L + L_adr;

    int LT_nnz = LT_rownnz[r];

    // lower triangle of H[r] has no fill-in iff its L_nnz-th column is the diagonal r
    int H_adr = H_rowadr[r];
    int H_nnz = H_rownnz[r];
    int H_nofill = (H_nnz >= L_nnz && H_colind[H_adr+L_nnz-1] == r);

    // leaf fast path: no updates from rows c > r and no fill-in on row r
    if (LT_nnz == 1 && H_nofill) {
      ns = 1;
      const mjtNum* restrict Hptr = H + H_adr;
      mjtNum diag = Hptr[L_nnz-1];
      int deficient = !(diag >= mindiag);
      if (deficient) {
        diag = mindiag;
        rank--;
      }

      // scale off-diagonals directly from H into L; clear if rank-deficient
      mjtNum L_rr = mju_sqrt(diag);
      if (deficient) {
        mju_zero(Lptr, L_nnz-1);
      } else {
        mjtNum L_rr_inv = 1.0 / L_rr;
        for (int i=0; i < L_nnz-1; i++) {
          Lptr[i] = Hptr[i] * L_rr_inv;
        }
      }
      Lptr[L_nnz-1] = L_rr;
      continue;
    }

    // detect target supernode size ns in {1, 2, 3}: consecutive rows (r, r-1, ...) that share
    // the exact same off-diagonal columns in L and the exact same descendant rows c > r in LT
    ns = 1;
    while (ns < 3 && r - ns >= 0 && L_nnz > ns &&
           L_colptr[L_nnz-1-ns] == r - ns &&
           L_rownnz[r-ns] == L_nnz - ns &&
           LT_rownnz[r-ns] == LT_nnz + ns) {
      ns++;
    }

    // pass ns as a compile-time literal so cholFactorSupernode specializes per supernode size
    if (ns == 3) {
      rank -= cholFactorSupernode(r, 3, mindiag, L, L_rownnz, L_rowadr, L_colind,
                                  LT_rownnz, LT_rowadr, LT_colind, LT_map,
                                  H, H_rownnz, H_rowadr, H_colind, dense);
    } else if (ns == 2) {
      rank -= cholFactorSupernode(r, 2, mindiag, L, L_rownnz, L_rowadr, L_colind,
                                  LT_rownnz, LT_rowadr, LT_colind, LT_map,
                                  H, H_rownnz, H_rowadr, H_colind, dense);
    } else {
      rank -= cholFactorSupernode(r, 1, mindiag, L, L_rownnz, L_rowadr, L_colind,
                                  LT_rownnz, LT_rowadr, LT_colind, LT_map,
                                  H, H_rownnz, H_rowadr, H_colind, dense);
    }
  }

  return rank;
}


// sparse reverse-order Cholesky solve
//  requires a complete factor: supernode detection assumes that L(r,c) != 0 (r > c) implies
//  struct(L(r,0:c)) is a subset of struct(L(c,0:c)), which incomplete factors may violate
void mju_cholSolveSparse(mjtNum* restrict res, const mjtNum* restrict mat,
                         const mjtNum* vec, int n,
                         const int* restrict rownnz, const int* restrict rowadr,
                         const int* restrict colind) {
  // copy into result if different
  if (res != vec) {
    mju_copy(res, vec, n);
  }

  // backward substitution: res <- L^-T res
  int i = n - 1;
  while (i >= 0) {
    const int adr = rowadr[i], nnz = rownnz[i];

    // fuse adjacent supernode rows (up to 3) when they share the exact same off-diagonal columns
    if (i > 0 && nnz >= 2 && rownnz[i-1] == nnz-1 && colind[adr+nnz-2] == i-1) {
      const mjtNum* restrict row0 = mat + adr;
      const mjtNum* restrict row1 = mat + rowadr[i-1];
      const int* restrict ci = colind + adr;

      // 3-row supernode (i, i-1, i-2)
      if (i > 1 && nnz >= 3 && rownnz[i-2] == nnz-2 && colind[adr+nnz-3] == i-2) {
        const int nob = nnz - 3;
        const mjtNum* restrict row2 = mat + rowadr[i-2];

        mjtNum x0 = (res[i] /= row0[nob+2]);
        mjtNum x1 = (res[i-1] = (res[i-1] - row0[nob+1]*x0) / row1[nob+1]);
        mjtNum x2 = (res[i-2] = (res[i-2] - row0[nob]*x0 - row1[nob]*x1) / row2[nob]);

        if (x0 || x1 || x2) {
          for (int j=0; j < nob; j++) {
            int c = ci[j];
            res[c] = res[c] - row0[j]*x0 - row1[j]*x1 - row2[j]*x2;
          }
        }
        i -= 3;
        continue;
      }

      // 2-row supernode (i, i-1)
      const int nob = nnz - 2;
      mjtNum x0 = (res[i] /= row0[nob+1]);
      mjtNum x1 = (res[i-1] = (res[i-1] - row0[nob]*x0) / row1[nob]);

      if (x0 || x1) {
        for (int j=0; j < nob; j++) {
          int c = ci[j];
          res[c] = res[c] - row0[j]*x0 - row1[j]*x1;
        }
      }
      i -= 2;
      continue;
    }

    if (res[i]) {
      mjtNum tmp = (res[i] /= mat[adr+nnz-1]);
      const mjtNum* restrict row = mat + adr;
      const int* restrict ci = colind + adr;
      for (int j=0; j < nnz-1; j++) {
        res[ci[j]] -= row[j] * tmp;
      }
    }
    i--;
  }

  // forward substitution: res <- L^-1 res
  i = 0;
  while (i < n) {
    const int adr = rowadr[i], nnz = rownnz[i];

    // fuse adjacent supernode rows (up to 3) when they share the exact same off-diagonal columns
    if (i + 1 < n && rownnz[i+1] == nnz+1 && colind[rowadr[i+1]+nnz-1] == i) {
      const int nob = nnz - 1;
      const mjtNum* restrict row0 = mat + adr;
      const mjtNum* restrict row1 = mat + rowadr[i+1];
      const int* restrict ci = colind + adr;

      // 3-row supernode (i, i+1, i+2)
      if (i + 2 < n && rownnz[i+2] == nnz+2 && colind[rowadr[i+2]+nnz-1] == i) {
        const mjtNum* restrict row2 = mat + rowadr[i+2];
        mjtNum s0 = 0, s1 = 0, s2 = 0;
        for (int j=0; j < nob; j++) {
          mjtNum y = res[ci[j]];
          s0 += row0[j] * y;
          s1 += row1[j] * y;
          s2 += row2[j] * y;
        }
        res[i] = (res[i] - s0) / row0[nob];
        res[i+1] = (res[i+1] - s1 - row1[nob]*res[i]) / row1[nob+1];
        res[i+2] = (res[i+2] - s2 - row2[nob]*res[i] - row2[nob+1]*res[i+1]) / row2[nob+2];
        i += 3;
        continue;
      }

      // 2-row supernode (i, i+1)
      mjtNum s0 = 0, s1 = 0;
      for (int j=0; j < nob; j++) {
        mjtNum y = res[ci[j]];
        s0 += row0[j] * y;
        s1 += row1[j] * y;
      }
      res[i] = (res[i] - s0) / row0[nob];
      res[i+1] = (res[i+1] - s1 - row1[nob]*res[i]) / row1[nob+1];
      i += 2;
      continue;
    }

    // x(i) -= sum_j L(i,j)*x(j), j=0:i-1
    if (nnz > 1) {
      res[i] -= mju_dotSparse(mat+adr, res, nnz-1, colind+adr);
      // modulo AVX, the above line does
      // for (int j=0; j<nnz-1; j++)
      //   res[i] -= mat[adr+j]*res[colind[adr+j]];
    }

    // x(i) /= L(i,i)
    res[i] /= mat[adr+nnz-1];
    i++;
  }
}


// sparse reverse-order Cholesky rank-one update: L'*L +/- x*x'; return rank
//  x is sparse (with x_ind sorted ascending), change in sparsity pattern of mat is not allowed
int mju_cholUpdateSparse(mjtNum* restrict mat, const mjtNum* restrict x, int n, int flg_plus,
                         const int* restrict rownnz, const int* restrict rowadr,
                         const int* restrict colind, int x_nnz, const int* restrict x_ind,
                         mjtNum* scratch) {
  // early return if x is empty
  if (x_nnz == 0) {
    return n;
  }

  // get starting row (last non-zero in x) and initial minimum active row
  int start = x_ind[x_nnz-1];
  int min_row = x_ind[0];

  // zero active range of dense accumulator, then scatter x into dense
  mjtNum* restrict dense = scratch;
  mju_zero(dense+min_row, start-min_row+1);
  mju_scatter(dense, x, x_ind, x_nnz);

  // backpass over rows from start down to min_row
  int rank = n;
  for (int row=start; row >= min_row; row--) {
    // skip if zero
    mjtNum x_row = dense[row];
    if (x_row == 0) continue;

    // get rownnz (excluding diagonal), rowadr
    int nnz = rownnz[row] - 1;
    int adr = rowadr[row];

    // update diagonal, handle rank-deficient case
    mjtNum diag = mat[adr+nnz];
    mjtNum tmp = diag*diag + (flg_plus ? x_row*x_row : -x_row*x_row);
    if (!(tmp >= mjMINVAL)) {
      tmp = mjMINVAL;
      rank--;
    }
    mjtNum r = mju_sqrt(tmp);
    mat[adr+nnz] = r;

    if (nnz == 0) {
      continue;
    }

    // compute Givens rotation parameters https://en.wikipedia.org/wiki/Givens_rotation
    mjtNum c = diag / r;
    mjtNum s = -x_row / r;
    mjtNum s_signed = flg_plus ? -s : s;

    // extend zeroed min_row range if this row reaches earlier columns
    const int* restrict ci = colind + adr;
    int col0 = ci[0];
    if (col0 < min_row) {
      mju_zero(dense+col0, min_row-col0);
      min_row = col0;
    }

    // apply Givens rotation to off-diagonals and dense workspace (contiguous fast path if dense)
    mjtNum* restrict mrow = mat + adr;
    if (nnz == ci[nnz-1] - col0 + 1) {
      mjtNum* restrict dptr = dense + col0;
      for (int i=0; i < nnz; i++) {
        mjtNum dense_j = dptr[i];
        mjtNum mat_val = mrow[i];
        mrow[i] = c*mat_val + s_signed*dense_j;
        dptr[i] = s*mat_val + c*dense_j;
      }
    } else {
      for (int i=0; i < nnz; i++) {
        int j = ci[i];
        mjtNum dense_j = dense[j];
        mjtNum mat_val = mrow[i];
        mrow[i]  = c*mat_val + s_signed*dense_j;
        dense[j] = s*mat_val + c*dense_j;
      }
    }
  }

  return rank;
}


//---------------------------- banded Cholesky -----------------------------------------------------

// integer min and max without double evaluation
static inline int minInt(int a, int b) {
  return a < b ? a : b;
}
static inline int maxInt(int a, int b) {
  return a > b ? a : b;
}


// address of diagonal element i in band-dense matrix representation
int mju_bandDiag(int i, int ntotal, int nband, int ndense) {
  int nsparse = ntotal - ndense;

  // sparse part
  if (i < nsparse) {
    return i*nband + nband - 1;
  }

  // dense part
  else {
    return nsparse*nband + (i-nsparse)*ntotal + i;
  }
}


// band-dense Cholesky decomposition
//  returns minimum value in the factorized diagonal, or 0 if rank-deficient
//  mat has (ntotal-ndense) x nband + ndense x ntotal elements
//  the first (ntotal-ndense) x nband store the band part, left of diagonal, inclusive
//  the second ndense x ntotal store the band part as entire dense rows
//  add diagadd+diagmul*mat_ii to diagonal before factorization
mjtNum mju_cholFactorBand(mjtNum* restrict mat, int ntotal, int nband, int ndense,
                          mjtNum diagadd, mjtNum diagmul) {
  int nsparse = ntotal - ndense;
  mjtNum mindiag = -1;

  // sparse part, including sparse-sparse and sparse-dense
  for (int j=0; j < nsparse; j++) {
    // number of non-zeros left of (j,j)
    int width_jj = minInt(j, nband-1);

    // number of non-zeros below (j,j), sparse part
    int height = minInt(nsparse-j-1, nband-1);

    // address of (j,j)
    int adr_jj = (j+1)*nband - 1;

    // compute L(j,j), before sqrt
    mjtNum left_jj = width_jj > 1 ? mju_dot(mat+adr_jj-width_jj, mat+adr_jj-width_jj, width_jj) :
                     (width_jj == 1 ? mat[adr_jj-1]*mat[adr_jj-1] : 0);
    mjtNum Ljj = diagadd + diagmul*mat[adr_jj] + mat[adr_jj] - left_jj;

    // stop if rank-deficient
    if (!(Ljj >= mjMINVAL)) {
      return 0;
    }

    // compute Ljj, scale = 1/Ljj
    Ljj = mju_sqrt(Ljj);
    mjtNum scale = 1 / Ljj;

    // update mindiag
    if (Ljj < mindiag || mindiag < 0) {
      mindiag = Ljj;
    }

    // compute L(i,j) for i > j, sparse part
    int adr_ij = adr_jj + nband - 1;
    for (int i=j+1; i <= j+height; i++, adr_ij += nband-1) {
      // number of non-zeros left of (i,j)
      int width_ij = minInt(j, nband-1-i+j);

      // in-place computation of L(i,j)
      mjtNum left_ij = width_ij > 1 ? mju_dot(mat+adr_jj-width_ij, mat+adr_ij-width_ij, width_ij) :
                       (width_ij == 1 ? mat[adr_jj-1]*mat[adr_ij-1] : 0);
      mat[adr_ij] = scale * (mat[adr_ij] - left_ij);
    }

    // compute L(i,j) for i >= nsparse, dense part
    adr_ij = nsparse*nband + j;
    for (int i=nsparse; i < ntotal; i++, adr_ij += ntotal) {
      // in-place computation of L(i,j)
      //  number of non-zeros left of (i,j) now equals width_jj
      mjtNum left_ij = width_jj > 1 ? mju_dot(mat+adr_jj-width_jj, mat+adr_ij-width_jj, width_jj) :
                       (width_jj == 1 ? mat[adr_jj-1]*mat[adr_ij-1] : 0);
      mat[adr_ij] = scale * (mat[adr_ij] - left_ij);
    }

    // save L(j,j)
    mat[adr_jj] = Ljj;
  }

  // dense part
  for (int j=nsparse; j < ntotal; j++) {
    // address of (j,j)
    int adr_jj = nsparse*nband + (j-nsparse)*ntotal + j;

    // compute Ljj
    mjtNum Ljj = diagadd + diagmul*mat[adr_jj] + mat[adr_jj] -
                 mju_dot(mat+adr_jj-j, mat+adr_jj-j, j);

    // stop if rank-deficient
    if (!(Ljj >= mjMINVAL)) {
      return 0;
    }

    // compute Ljj, scale = 1/Ljj
    Ljj = mju_sqrt(Ljj);
    mjtNum scale = 1 / Ljj;

    // update mindiag
    if (Ljj < mindiag || mindiag < 0) {
      mindiag = Ljj;
    }

    // compute L(i,j) for i > j
    int adr_ij = adr_jj + ntotal;
    for (int i=j+1; i < ntotal; i++, adr_ij += ntotal) {
      // in-place computation of L(i,j)
      mat[adr_ij] = scale * (mat[adr_ij] - mju_dot(mat+adr_jj-j, mat+adr_ij-j, j));
    }

    // save L(j,j)
    mat[adr_jj] = Ljj;
  }

  return mindiag;
}


// solve with band-Cholesky decomposition
void mju_cholSolveBand(mjtNum* res, const mjtNum* restrict mat, const mjtNum* vec,
                       int ntotal, int nband, int ndense) {
  int nsparse = ntotal - ndense;

  // copy into result if different
  if (res != vec) {
    mju_copy(res, vec, ntotal);
  }

  //------- forward substitution: solve L*res = vec

  // sparse part
  for (int i=0, adr_ii=nband-1; i < nsparse; i++, adr_ii += nband) {
    // number of non-zeros left of (i,i)
    int width = minInt(i, nband-1);

    // off-diagonal
    if (width > 1) {
      res[i] -= mju_dot(mat + adr_ii - width, res + i - width, width);
    } else if (width == 1) {
      res[i] -= mat[adr_ii - 1] * res[i - 1];
    }

    // diagonal
    res[i] /= mat[adr_ii];
  }

  // dense part
  for (int i=nsparse, row_i=nsparse*nband; i < ntotal; i++, row_i += ntotal) {
    // off-diagonal
    if (i) {
      res[i] -= mju_dot(mat + row_i, res, i);
    }

    // diagonal
    res[i] /= mat[row_i + i];
  }

  //------- backward substitution: solve L'*res = res

  // dense part
  for (int i=ntotal-1, row_i=nsparse*nband+(ndense-1)*ntotal; i >= nsparse; i--, row_i -= ntotal) {
    // diagonal
    res[i] /= mat[row_i + i];

    // update res[0..i-1] with row i of L
    if (i && res[i]) {
      mju_addToScl(res, mat + row_i, -res[i], i);
    }
  }

  // sparse part
  for (int i=nsparse-1, adr_ii=nsparse*nband-1; i >= 0; i--, adr_ii -= nband) {
    // number of non-zeros left of (i,i)
    int width = minInt(i, nband-1);

    // diagonal
    res[i] /= mat[adr_ii];

    // update res[i-width..i-1] with row i of L
    if (width > 1 && res[i]) {
      mju_addToScl(res + i - width, mat + adr_ii - width, -res[i], width);
    } else if (width == 1) {
      res[i - 1] -= mat[adr_ii - 1] * res[i];
    }
  }
}


// convert band matrix to dense matrix
void mju_band2Dense(mjtNum* restrict res, const mjtNum* restrict mat,
                    int ntotal, int nband, int ndense, mjtBool flg_sym) {
  int nsparse = ntotal - ndense;

  // clear all
  mju_zero(res, ntotal*ntotal);

  // sparse part
  for (int i=0; i < nsparse; i++) {
    // number of non-zeros left of (i,i)
    int width = minInt(i, nband-1);

    // copy data
    mju_copy(res + i*ntotal + i-width, mat + (i+1)*nband - (width+1), width+1);
  }

  // dense part
  for (int i=nsparse; i < ntotal; i++) {
    mju_copy(res + i*ntotal, mat + nsparse*nband + (i-nsparse)*ntotal, i+1);
  }

  // make symmetric
  if (flg_sym) {
    for (int i=0; i < ntotal; i++) {
      for (int j=i+1; j < ntotal; j++) {
        res[i*ntotal + j] = res[j*ntotal + i];
      }
    }
  }
}


// convert dense matrix to band matrix
void mju_dense2Band(mjtNum* restrict res, const mjtNum* restrict mat,
                    int ntotal, int nband, int ndense) {
  int nsparse = ntotal - ndense;

  // sparse part
  for (int i=0; i < nsparse; i++) {
    // number of non-zeros left of (i,i)
    int width = minInt(i, nband-1);

    // copy data
    mju_copy(res + (i+1)*nband - (width+1), mat + i*ntotal + i-width, width+1);
  }

  // dense part
  for (int i=nsparse; i < ntotal; i++) {
    mju_copy(res + nsparse*nband + (i-nsparse)*ntotal, mat + i*ntotal, i+1);
  }
}


// multiply band-diagonal matrix with vector, res must not alias vec
void mju_bandMulMatVec(mjtNum* restrict res, const mjtNum* restrict mat,
                       const mjtNum* restrict vec, int ntotal, int nband, int ndense,
                       int nvec, mjtBool flg_sym) {
  int nsparse = ntotal - ndense;

  // handle multiple vectors
  for (int j=0; j < nvec; j++) {
    // precompute pointer to corresponding vector in vec and res
    const mjtNum* restrict vec_j = vec + ntotal*j;
    mjtNum* restrict res_j = res + ntotal*j;

    // sparse part
    int row_end = nband;
    for (int i=0; i < nsparse; i++, row_end += nband) {
      int width = minInt(i+1, nband);
      int adr = row_end - width;
      int offset = i + 1 - width;
      res_j[i] = mju_dot(mat+adr, vec_j+offset, width);  // lower triangle
      if (flg_sym && width > 1) {
        // strict upper triangle
        mju_addToScl(res_j+offset, mat+adr, vec_j[i], width-1);
      }
    }

    // dense part
    int adr = nsparse*nband;
    for (int i=nsparse; i < ntotal; i++, adr += ntotal) {
      res_j[i] = mju_dot(mat+adr, vec_j, i+1);          // lower triangle
      if (flg_sym && i > 0) {
        // strict upper triangle
        mju_addToScl(res_j, mat+adr, vec_j[i], i);
      }
    }
  }
}


//------------------------------ dense LU factorization --------------------------------------------

// dense LU factorization with partial pivoting
//   factorizes n x n row-major matrix A in-place into L and U
//   L has unit diagonal (not stored), U has explicit diagonal
//   pivot stores row permutation: row i of original = row pivot[i] of result
//   return 1 if successful, 0 if singular (diagonal element < mjMINVAL)
int mju_factorLU(mjtNum* restrict A, int n, int* pivot) {
  for (int k=0; k < n; k++) {
    // initialize pivot
    pivot[k] = k;

    // find pivot: max absolute value in column k, rows k..n-1
    mjtNum maxval = mju_abs(A[k*n+k]);
    int maxrow = k;
    for (int i=k+1; i < n; i++) {
      mjtNum val = mju_abs(A[i*n+k]);
      if (val > maxval) {
        maxval = val;
        maxrow = i;
      }
    }

    // check singularity
    if (maxval < mjMINVAL) {
      return 0;
    }

    // swap rows k and maxrow
    if (maxrow != k) {
      pivot[k] = maxrow;
      for (int j=0; j < n; j++) {
        mjtNum tmp = A[k*n+j];
        A[k*n+j] = A[maxrow*n+j];
        A[maxrow*n+j] = tmp;
      }
    }

    // compute multipliers and update trailing submatrix
    mjtNum diaginv = 1.0 / A[k*n+k];
    for (int i=k+1; i < n; i++) {
      A[i*n+k] *= diaginv;
      mjtNum Aik = A[i*n+k];
      for (int j=k+1; j < n; j++) {
        A[i*n+j] -= Aik * A[k*n+j];
      }
    }
  }

  return 1;
}


// solve A*x = b given LU factorization of A, LU and pivot are output of mju_factorLU
void mju_solveLU(mjtNum* restrict x, const mjtNum* LU, const mjtNum* b, const int* pivot, int n) {
  // copy b into x
  mju_copy(x, b, n);

  // apply row permutation and forward substitution: solve L*y = P*b
  for (int i=0; i < n; i++) {
    // apply pivot swap
    if (pivot[i] != i) {
      mjtNum tmp = x[i];
      x[i] = x[pivot[i]];
      x[pivot[i]] = tmp;
    }

    // subtract known terms
    for (int j=0; j < i; j++) {
      x[i] -= LU[i*n+j] * x[j];
    }
  }

  // back substitution: solve U*x = y
  for (int i=n-1; i >= 0; i--) {
    for (int j=i+1; j < n; j++) {
      x[i] -= LU[i*n+j] * x[j];
    }
    x[i] /= LU[i*n+i];
  }
}


// 6x6 specialization of mju_factorLU: same algorithm with compile-time size,
// allowing full unrolling; produces identical results
int mju_factorLU6(mjtNum A[36], int pivot[6]) {
  for (int k=0; k < 6; k++) {
    // initialize pivot
    pivot[k] = k;

    // find pivot: max absolute value in column k, rows k..n-1
    mjtNum maxval = mju_abs(A[k*6+k]);
    int maxrow = k;
    for (int i=k+1; i < 6; i++) {
      mjtNum val = mju_abs(A[i*6+k]);
      if (val > maxval) {
        maxval = val;
        maxrow = i;
      }
    }

    // check singularity
    if (maxval < mjMINVAL) {
      return 0;
    }

    // swap rows k and maxrow
    if (maxrow != k) {
      pivot[k] = maxrow;
      for (int j=0; j < 6; j++) {
        mjtNum tmp = A[k*6+j];
        A[k*6+j] = A[maxrow*6+j];
        A[maxrow*6+j] = tmp;
      }
    }

    // compute multipliers and update trailing submatrix
    mjtNum diaginv = 1.0 / A[k*6+k];
    for (int i=k+1; i < 6; i++) {
      A[i*6+k] *= diaginv;
      mjtNum Aik = A[i*6+k];
      for (int j=k+1; j < 6; j++) {
        A[i*6+j] -= Aik * A[k*6+j];
      }
    }
  }

  return 1;
}


// solve A*x = b given 6x6 LU factorization from mju_factorLU6
void mju_solveLU6(mjtNum x[6], const mjtNum LU[36], const mjtNum b[6], const int pivot[6]) {
  for (int i=0; i < 6; i++) {
    x[i] = b[i];
  }

  // apply row permutation and forward substitution: solve L*y = P*b
  for (int i=0; i < 6; i++) {
    // apply pivot swap
    if (pivot[i] != i) {
      mjtNum tmp = x[i];
      x[i] = x[pivot[i]];
      x[pivot[i]] = tmp;
    }

    // subtract known terms
    for (int j=0; j < i; j++) {
      x[i] -= LU[i*6+j] * x[j];
    }
  }

  // back substitution: solve U*x = y
  for (int i=6-1; i >= 0; i--) {
    for (int j=i+1; j < 6; j++) {
      x[i] -= LU[i*6+j] * x[j];
    }
    x[i] /= LU[i*6+i];
  }
}


//------------------------------ sparse LU factorization -------------------------------------------

// sparse reverse-order LU factorization, no fill-in (assuming tree topology)
//   result: LU = L + U; original = (U+I) * L; scratch size is n
//   clamp pivots with magnitude below mjMINVAL, return first clamped dof index or -1 if none
int mju_factorLUSparse(mjtNum* LU, int n, int* scratch,
                       const int* rownnz, const int* rowadr, const int* colind,
                       const int* index) {
  int* remaining = scratch;
  int clamped = -1;

  // set remaining = rownnz
  if (index) {
    for (int i=0; i < n; i++) {
      remaining[i] = rownnz[index[i]];
    }
  } else {
    mju_copyInt(remaining, rownnz, n);
  }

  // diagonal elements (i,i)
  for (int r=n-1; r >= 0; r--) {
    int i = index ? index[r] : r;

    // get address of last remaining element of row i, adjust remaining counter
    int ii = rowadr[i] + remaining[r] - 1;
    remaining[r]--;

    // make sure ii is on diagonal
    if (colind[ii] != i) {
      mjERROR("missing diagonal element");
    }

    // near-singular pivot: clamp, preserving the sign
    if (mju_abs(LU[ii]) < mjMINVAL) {
      LU[ii] = LU[ii] < 0 ? -mjMINVAL : mjMINVAL;
      if (clamped < 0) {
        clamped = i;
      }
    }

    // rows j above i
    for (int c=r-1; c >= 0; c--) {
      int j = index ? index[c] : c;

      // get address of last remaining element of row j
      int ji = rowadr[j] + remaining[c] - 1;

      // process row j if (j,i) is non-zero
      if (colind[ji] == i) {
        // adjust remaining counter
        remaining[c]--;

        // (j,i) = (j,i) / (i,i)
        LU[ji] = LU[ji] / LU[ii];
        mjtNum LUji = LU[ji];

        // (j,k) = (j,k) - (i,k) * (j,i) for k<i; handle incompatible sparsity
        int icnt = rowadr[i], jcnt = rowadr[j];
        while (jcnt < rowadr[j]+remaining[c]) {
          // both non-zero
          if (colind[icnt] == colind[jcnt]) {
            // update LU, advance counters
            LU[jcnt++] -= LU[icnt++] * LUji;
          }

          // only (j,k) non-zero
          else if (colind[icnt] > colind[jcnt]) {
            // advance j counter
            jcnt++;
          }

          // only (i,k) non-zero
          else {
            mjERROR("requires fill-in");
          }
        }

        // make sure both rows fully processed
        if (icnt != rowadr[i]+remaining[r] || jcnt != rowadr[j]+remaining[c]) {
          mjERROR("row processing incomplete");
        }
      }
    }
  }

  // make sure remaining points to diagonal
  for (int r=0; r < n; r++) {
    int i = index ? index[r] : r;
    if (remaining[r] < 0 || colind[rowadr[i]+remaining[r]] != i) {
      mjERROR("unexpected sparse matrix structure");
    }
  }

  return clamped;
}


// solve mat*res=vec given LU factorization of mat
void mju_solveLUSparse(mjtNum* res, const mjtNum* LU, const mjtNum* vec, int n,
                       const int* rownnz, const int* rowadr, const int* diag, const int* colind,
                       const int* index) {
  // solve (U+I)*res = vec
  for (int k=n-1; k >= 0; k--) {
    int i = index ? index[k] : k;

    // init: diagonal of (U+I) is 1
    res[i] = vec[i];

    int d1 = diag[i]+1;
    int nnz = rownnz[i] - d1;
    if (nnz > 0) {
      int adr = rowadr[i] + d1;
      res[i] -= mju_dotSparse(LU+adr, res, nnz, colind+adr);
    }
  }

  //------------------ solve L*res(new) = res
  for (int k=0; k < n; k++) {
    int i = index ? index[k] : k;

    // res[i] -= sum_k<i res[k]*LU(i,k)
    int d = diag[i];
    int adr = rowadr[i];
    if (d > 0) {
      res[i] -= mju_dotSparse(LU+adr, res, d, colind+adr);
    }

    // divide by diagonal element of L
    res[i] /= LU[adr + d];
  }
}


//--------------------------- 3x3 linear solve -----------------------------------------------------

// solve 3x3 linear system A*x = b using Gaussian elimination
void mju_solve3(mjtNum x[3], const mjtNum A[9], const mjtNum b[3]) {
  mjtNum M[3][4] = {
    {A[0], A[1], A[2], b[0]},
    {A[3], A[4], A[5], b[1]},
    {A[6], A[7], A[8], b[2]}
  };

  for (int i=0; i<3; i++) {
    mjtNum pivot = M[i][i];
    for (int j=i; j<4; j++) {
      M[i][j] /= pivot;
    }

    for (int k=0; k<3; k++) {
      if (k != i) {
        mjtNum factor = M[k][i];
        for (int j=i; j<4; j++) {
          M[k][j] -= factor * M[i][j];
        }
      }
    }
  }
  x[0] = M[0][3];
  x[1] = M[1][3];
  x[2] = M[2][3];
}


//--------------------------- eigen decomposition --------------------------------------------------

// off-diagonal tolerance and eigenvalue swap threshold, relative to max element
#ifdef mjUSESINGLE
static const mjtNum eigTOL = 2e-6f;
static const mjtNum eigEPS = 1e-5f;
#else
static const mjtNum eigTOL = 4e-15;
static const mjtNum eigEPS = 1e-12;
#endif

// eigenvalue decomposition of symmetric 3x3 matrix, using eigTOL; returns number of iterations
int mju_eig3(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4], const mjtNum mat[9]) {
  return mju_eig3Tol(eigval, eigvec, quat, mat, eigTOL);
}


// same as mju_eig3, stop when off-diagonal elements are below reltol times the largest element
int mju_eig3Tol(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4], const mjtNum mat[9],
                mjtNum reltol) {
  mjtNum D[9], tmp[9];
  mjtNum tau, t;
  int iter, rk, ck, rotk;

  // off-diagonal tolerance: no smaller than the roundoff level of D, about 16 epsilons
  mjtNum scale = 0;
  for (int i=0; i < 9; i++) {
    scale = mju_max(scale, mju_abs(mat[i]));
  }
  mjtNum tol = scale * mju_max(reltol, eigTOL);

  // initialize with unit quaternion
  quat[0] = 1;
  quat[1] = quat[2] = quat[3] = 0;

  // Jacobi iteration
  for (iter=0; iter < 500; iter++) {
    // make quaternion matrix eigvec, compute D = eigvec'*mat*eigvec
    mju_quat2Mat(eigvec, quat);
    mji_mulMatTMat3(tmp, eigvec, mat);
    mji_mulMatMat3(D, tmp, eigvec);

    // assign eigenvalues
    eigval[0] = D[0];
    eigval[1] = D[4];
    eigval[2] = D[8];

    // find max off-diagonal element, set indices
    if (mju_abs(D[1]) > mju_abs(D[2]) && mju_abs(D[1]) > mju_abs(D[5])) {
      rk = 0;     // row
      ck = 1;     // column
      rotk = 2;   // rotation axis
    } else if (mju_abs(D[2]) > mju_abs(D[5])) {
      rk = 0;
      ck = 2;
      rotk = 1;
    } else {
      rk = 1;
      ck = 2;
      rotk = 0;
    }

    // terminate if max off-diagonal element too small
    if (mju_abs(D[3*rk+ck]) <= tol) {
      break;
    }

    // 2x2 symmetric Schur decomposition: t = tan(angle)
    tau = (D[4*ck]-D[4*rk])/(2*D[3*rk+ck]);
    if (tau >= 0) {
      t = 1.0/(tau + mju_sqrt(1 + tau*tau));
    } else {
      t = -1.0/(-tau + mju_sqrt(1 + tau*tau));
    }

    // express rotation as quaternion, using h = tan(angle/2): accurate for small angles
    mjtNum h = t/(1 + mju_sqrt(1 + t*t));
    tmp[0] = 1/mju_sqrt(1 + h*h);
    tmp[1] = tmp[2] = tmp[3] = 0;
    tmp[rotk+1] = (rotk == 1 ? h : -h) * tmp[0];

    // accumulate quaternion rotation
    mju_mulQuat(quat, quat, tmp);
    mju_normalize4(quat);
  }

  // sort eigenvalues in decreasing order (bubble sort: 0, 1, 0)
  mjtNum eps = scale * eigEPS;
  for (int j=0; j < 3; j++) {
    int j1 = j%2;       // lead index

    // only swap if the eigenvalues are different
    if (eigval[j1]+eps < eigval[j1+1]) {
      // swap eigenvalues
      t = eigval[j1];
      eigval[j1] = eigval[j1+1];
      eigval[j1+1] = t;

      // rotate quaternion
      tmp[0] = 0.707106781186548;  // = cos(pi/4) = sin(pi/4)
      tmp[1] = tmp[2] = tmp[3] = 0;
      tmp[(j1+2)%3+1] = tmp[0];
      mju_mulQuat(quat, quat, tmp);
      mju_normalize4(quat);
    }
  }

  // recompute eigvec
  mju_quat2Mat(eigvec, quat);

  return iter;
}


//---------------------------------- QCQP ----------------------------------------------------------

// solve QCQP in 2 dimensions:
//  min  0.5*x'*A*x + x'*b  s.t.  sum (xi/di)^2 <= r^2
// return 0 if unconstrained, 1 if constrained
int mju_QCQP2(mjtNum* res, const mjtNum* Ain, const mjtNum* bin,
              const mjtNum* d, mjtNum r) {
  mjtNum A11, A22, A12, b1, b2;
  mjtNum P11, P22, P12, det, detinv, v1, v2, la, val, deriv;

  // scale A,b so that constraint becomes x'*x <= r*r
  b1 = bin[0]*d[0];
  b2 = bin[1]*d[1];
  A11 = Ain[0]*d[0]*d[0];
  A22 = Ain[3]*d[1]*d[1];
  A12 = Ain[1]*d[0]*d[1];

  // Newton iteration
  la = 0;
  for (int iter=0; iter < 20; iter++) {
    // det(A+la)
    det = (A11+la)*(A22+la) - A12*A12;

    // check SPD, with 1e-10 threshold
    if (det < 1e-10) {
      res[0] = 0;
      res[1] = 0;
      return 0;
    }

    // P = inv(A+la)
    detinv = 1/det;
    P11 = (A22+la)*detinv;
    P22 = (A11+la)*detinv;
    P12 = -A12*detinv;

    // v = -P*b
    v1 = -P11*b1 - P12*b2;
    v2 = -P12*b1 - P22*b2;

    // val = v'*v - r*r
    val = v1*v1 + v2*v2 - r*r;

    // check for convergence, or initial solution inside constraint set
    if (val < 1e-10) {
      break;
    }

    // deriv = -2 * v' * P * v
    deriv = -2.0*(P11*v1*v1 + 2.0*P12*v1*v2 + P22*v2*v2);

    // compute update, exit if too small
    mjtNum delta = -val/deriv;
    if (delta < 1e-10) {
      break;
    }

    // update
    la += delta;
  }

  // undo scaling
  res[0] = v1*d[0];
  res[1] = v2*d[1];

  return (la != 0);
}


// solve QCQP in 3 dimensions:
//  min  0.5*x'*A*x + x'*b  s.t.  sum (xi/di)^2 <= r^2
// return 0 if unconstrained, 1 if constrained
int mju_QCQP3(mjtNum* res, const mjtNum* Ain, const mjtNum* bin,
              const mjtNum* d, mjtNum r) {
  mjtNum A11, A22, A33, A12, A13, A23, b1, b2, b3;
  mjtNum P11, P22, P33, P12, P13, P23, det, detinv, v1, v2, v3, la, val, deriv;

  // scale A,b so that constraint becomes x'*x <= r*r
  b1 = bin[0]*d[0];
  b2 = bin[1]*d[1];
  b3 = bin[2]*d[2];
  A11 = Ain[0]*d[0]*d[0];
  A22 = Ain[4]*d[1]*d[1];
  A33 = Ain[8]*d[2]*d[2];
  A12 = Ain[1]*d[0]*d[1];
  A13 = Ain[2]*d[0]*d[2];
  A23 = Ain[5]*d[1]*d[2];

  // Newton iteration
  la = 0;
  for (int iter=0; iter < 20; iter++) {
    // unscaled P
    P11 = (A22+la)*(A33+la) - A23*A23;
    P22 = (A11+la)*(A33+la) - A13*A13;
    P33 = (A11+la)*(A22+la) - A12*A12;
    P12 = A13*A23 - A12*(A33+la);
    P13 = A12*A23 - A13*(A22+la);
    P23 = A12*A13 - A23*(A11+la);

    // det(A+la)
    det = (A11+la)*P11 + A12*P12 + A13*P13;

    // check SPD, with 1e-10 threshold
    if (det < 1e-10) {
      res[0] = 0;
      res[1] = 0;
      res[2] = 0;
      return 0;
    }

    // detinv
    detinv = 1/det;

    // final P
    P11 *= detinv;
    P22 *= detinv;
    P33 *= detinv;
    P12 *= detinv;
    P13 *= detinv;
    P23 *= detinv;

    // v = -P*b
    v1 = -P11*b1 - P12*b2 - P13*b3;
    v2 = -P12*b1 - P22*b2 - P23*b3;
    v3 = -P13*b1 - P23*b2 - P33*b3;

    // val = v'*v - r*r
    val = v1*v1 + v2*v2 + v3*v3 - r*r;

    // check for convergence, or initial solution inside constraint set
    if (val < 1e-10) {
      break;
    }

    // deriv = -2 * v' * P * v
    deriv = -2.0*(P11*v1*v1 + P22*v2*v2 + P33*v3*v3)
            -4.0*(P12*v1*v2 + P13*v1*v3 + P23*v2*v3);

    // compute update, exit if too small
    mjtNum delta = -val/deriv;
    if (delta < 1e-10) {
      break;
    }

    // update
    la += delta;
  }

  // undo scaling
  res[0] = v1*d[0];
  res[1] = v2*d[1];
  res[2] = v3*d[2];

  return (la != 0);
}


// solve QCQP in n dimensions:
//  min  0.5*x'*A*x + x'*b  s.t.  sum (xi/di)^2 <= r^2
// return 0 if unconstrained, 1 if constrained
int mju_QCQP(mjtNum* res, const mjtNum* Ain, const mjtNum* bin,
             const mjtNum* d, mjtNum r, int n) {
  mjtNum A[25], Ala[25], b[5];
  mjtNum la, val, deriv, tmp[5];

  // check size
  if (n > 5) {
    mjERROR("n is only supported up to 5");
  }

  // scale A,b so that constraint becomes x'*x <= r*r
  for (int i=0; i < n; i++) {
    b[i] = bin[i] * d[i];

    for (int j=0; j < n; j++) {
      A[j+i*n] = Ain[j+i*n] * d[i] * d[j];
    }
  }

  // Newton iteration
  la = 0;
  for (int iter=0; iter < 20; iter++) {
    // make A+la
    mju_copy(Ala, A, n*n);
    for (int i=0; i < n; i++) {
      Ala[i*(n+1)] += la;
    }

    // factorize, check rank with 1e-10 threshold
    if (mju_cholFactor(Ala, n, 1e-10) < n) {
      mju_zero(res, n);
      return 0;
    }

    // set res = -Ala \ b
    mju_cholSolve(res, Ala, b, n);
    mju_scl(res, res, -1, n);

    // val = b' * Ala^-2 * b - r*r
    val = mju_dot(res, res, n) - r*r;

    // check for convergence, or initial solution inside constraint set
    if (val < 1e-10) {
      break;
    }

    // deriv = -2 * b' * Ala^-3 * b
    mju_cholSolve(tmp, Ala, res, n);
    deriv = -2.0 * mju_dot(res, tmp, n);

    // compute update, exit if too small
    mjtNum delta = -val/deriv;
    if (delta < 1e-10) {
      break;
    }

    // update
    la += delta;
  }

  // undo scaling
  for (int i=0; i < n; i++) {
    res[i] = res[i] * d[i];
  }

  return (la != 0);
}


//--------------------------- box-constrained quadratic program ------------------------------------

// minimize 0.5*x'*H*x + x'*g  s.t. lower <= x <= upper, return rank or -1 if failed
//   inputs:
//     n           - problem dimension
//     H           - SPD matrix                n*n
//     g           - bias vector               n
//     lower       - lower bounds              n
//     upper       - upper bounds              n
//     res         - solution warmstart        n
//   return value:
//     nfree <= n  - rank of unconstrained subspace, -1 if failure
//   outputs (required):
//     res         - solution                  n
//     R           - subspace Cholesky factor  nfree*nfree    allocated: n*(n+7)
//   outputs (optional):
//     index       - set of free dimensions    nfree          allocated: n
//   notes:
//     the initial value of res is used to warmstart the solver
//     R must have allocated size n*(n+7), but only nfree*nfree values are used in output
//     index (if given) must have allocated size n, but only nfree values are used in output
//     only lower triangles of H and R are read from and written to, respectively
int mju_boxQP(mjtNum* res, mjtNum* R, int* index,
              const mjtNum* H, const mjtNum* g, int n,
              const mjtNum* lower, const mjtNum* upper) {
  // algorithm options
  int    maxiter    = 100;    // maximum number of iterations
  mjtNum mingrad    = 1E-16;  // minimum squared norm of (unclamped) gradient
  mjtNum backtrack  = 0.5;    // backtrack factor for decreasing stepsize
  mjtNum minstep    = 1E-22;  // minimum stepsize for linesearch
  mjtNum armijo     = 0.1;    // Armijo parameter (fraction of expected linear improvement)

  // logging (disabled)
  char*  log        = NULL;   // buffer to write log messages into
  int    logsz      = 0;      // size of log buffer

  return mju_boxQPoption(res, R, index, H, g, n, lower, upper,
                         maxiter, mingrad, backtrack, minstep, armijo, log, logsz);
}


// allocate heap memory for box-constrained Quadratic Program
//   as in mju_boxQP, index, lower and upper are optional
//   free all pointers with mju_free()
void mju_boxQPmalloc(mjtNum** res, mjtNum** R, int** index,
                     mjtNum** H, mjtNum** g, int n,
                     mjtNum** lower, mjtNum** upper) {
  // required arrays
  *res = (mjtNum*) mju_malloc(sizeof(mjtNum)*n);
  *R   = (mjtNum*) mju_malloc(sizeof(mjtNum)*n*(n+7));
  *H   = (mjtNum*) mju_malloc(sizeof(mjtNum)*n*n);
  *g   = (mjtNum*) mju_malloc(sizeof(mjtNum)*n);

  // optional arrays
  if (lower) *lower = (mjtNum*) mju_malloc(sizeof(mjtNum)*n);
  if (upper) *upper = (mjtNum*) mju_malloc(sizeof(mjtNum)*n);
  if (index) *index = (int*) mju_malloc(sizeof(int)*n);
}


// local enum encoding mju_boxQP solver status (purely for readability)
enum mjtStatusBoxQP {
  mjBOXQP_NOT_SPD     = -1,        //  Hessian is not positive definite
  mjBOXQP_NO_DESCENT  = 0,         //  no descent direction found
  mjBOXQP_MAX_ITER    = 1,         //  maximum main iterations exceeded
  mjBOXQP_MAX_LS_ITER = 2,         //  maximum line-search iterations exceeded
  mjBOXQP_TOL_GRAD    = 3,         //  gradient norm smaller than tolerance
  mjBOXQP_UNBOUNDED   = 4,         //  no dimensions clamped, returning Newton point
  mjBOXQP_ALL_CLAMPED = 5,         //  all dimensions clamped

  mjNBOXQP            = 7          //  number of boxQP status values
};


// multiply symmetric matrix with vector: res = mat*vec
//   assumes symmetry of mat, ignores upper triangle, res must not alias vec
static void mulSymVec(mjtNum* restrict res, const mjtNum* mat, const mjtNum* vec, int n) {
  for (int i=0; i < n; i++) {
    // diagonal + strict lower triangle: res[i] = sum_{j<=i} mat[i,j] * vec[j]
    res[i] = mat[n*i+i]*vec[i] + mju_dot(mat+n*i, vec, i);

    // strict upper mirror contribution: res[k] += mat[i,k] * vec[i] for k < i
    if (vec[i]) {
      mju_addToScl(res, mat+n*i, vec[i], i);
    }
  }
}


// delete row and column p from m-by-m Cholesky factor R, return rank of (m-1)-by-(m-1) factor
static int cholDelete(mjtNum* restrict R, mjtNum* restrict temp, int m, int p) {
  // shift rows 1..p-1 to stride m-1
  for (int i=1; i < p; i++) {
    for (int j=0; j <= i; j++) {
      R[i*(m-1)+j] = R[i*m+j];
    }
  }

  // zero leading entries of update vector
  mju_zero(temp, p);

  // shift rows p+1..m-1 to rows p..m-2 (stride m-1) and extract column p into temp
  for (int i=p+1; i < m; i++) {
    for (int j=0; j < p; j++) {
      R[(i-1)*(m-1)+j] = R[i*m+j];
    }
    temp[i-1] = R[i*m+p];
    for (int j=p+1; j <= i; j++) {
      R[(i-1)*(m-1)+(j-1)] = R[i*m+j];
    }
  }

  // rank-1 update on trailing (m-1-p) block
  return mju_cholUpdate(R, temp, m-1, 1);
}


// insert variable idx at position p into m-by-m Cholesky factor R, return rank of (m+1)-by-(m+1)
static int cholInsert(mjtNum* restrict R, mjtNum* restrict temp, mjtNum* restrict rowp,
                      const mjtNum* H, int n, const int* index, int m, int p, int idx) {
  // solve L_11 * l_21 = h_21 for new row p (entries 0..p-1)
  for (int j=0; j < p; j++) {
    mjtNum s = H[idx*n+index[j]];
    if (j) {
      s -= mju_dot(R+j*m, rowp, j);
    }
    rowp[j] = s / R[j*(m+1)];
  }

  // diagonal entry l_22
  mjtNum diag = H[idx*(n+1)] - mju_dot(rowp, rowp, p);
  if (!(diag >= mjMINVAL)) {
    return -1;
  }
  mjtNum l22 = mju_sqrt(diag);
  rowp[p] = l22;

  // compute new column p below diagonal: l_32 = (h_32 - L_31 * l_21) / l_22
  mju_zero(temp, p+1);
  mjtNum inv_l22 = 1 / l22;
  for (int i=p; i < m; i++) {
    temp[i+1] = (H[index[i]*n+idx] - mju_dot(R+i*m, rowp, p)) * inv_l22;
  }

  // expand R from stride m to stride m+1 (in-place backward pass)
  for (int i=m-1; i >= p; i--) {
    for (int j=i; j >= p; j--) {
      R[(i+1)*(m+1)+(j+1)] = R[i*m+j];
    }
    R[(i+1)*(m+1)+p] = temp[i+1];
    for (int j=p-1; j >= 0; j--) {
      R[(i+1)*(m+1)+j] = R[i*m+j];
    }
  }

  // write new row p
  for (int j=0; j <= p; j++) {
    R[p*(m+1)+j] = rowp[j];
  }

  // shift rows p-1..1 to stride m+1
  for (int i=p-1; i >= 1; i--) {
    for (int j=i; j >= 0; j--) {
      R[i*(m+1)+j] = R[i*m+j];
    }
  }

  // rank-1 downdate on trailing (m-p) block
  return mju_cholUpdate(R, temp, m+1, 0);
}


// minimize 0.5*x'*H*x + x'*g  s.t. lower <= x <=upper, explicit options
//  additional arguments to mju_boxQP (see mju_boxQP documentation):
//   maxiter     maximum number of iterations
//   mingrad     minimum squared norm of (unclamped) gradient
//   backtrack   backtrack factor for decreasing stepsize
//   minstep     minimum stepsize for linesearch
//   armijo      Armijo parameter (fraction of expected linear improvement)
//   log         buffer to write log messages into
//   logsz       size of log buffer
int mju_boxQPoption(mjtNum* res, mjtNum* R, int* index,               // outputs
                    const mjtNum* H, const mjtNum* g, int n,          // QP definition
                    const mjtNum* lower, const mjtNum* upper,         // bounds
                    int maxiter, mjtNum mingrad, mjtNum backtrack,    // options
                    mjtNum minstep, mjtNum armijo,                    // options
                    char* log, int logsz)                             // logging
{
  int status = mjBOXQP_NO_DESCENT;  // initial status: no descent direction found
  int nfree = n;                    // initialize nfree with n
  int nfactor = 0;
  mjtNum value = 0, norm2 = 0;

  // basic checks
  if (n <= 0) {
    mjERROR("problem size n must be positive");
  }
  if (upper && lower) {
    for (int i=0; i < n; i++) {
      if (lower[i] >= upper[i]) {
        mjERROR("upper bounds must be strictly larger than lower bounds");
      }
    }
  }

  // local scratch vectors, allocate in R
  mjtNum* scratch   = R + n*n;
  mjtNum* grad      = scratch + 0*n;
  mjtNum* search    = scratch + 1*n;
  mjtNum* candidate = scratch + 2*n;
  mjtNum* temp      = scratch + 3*n;
  int* clamped      = (int*) (scratch + 4*n);
  int* oldclamped   = (int*) (scratch + 5*n);

  // if index vector not given, use scratch space
  if (!index) {
    index = (int*) (scratch + 6*n);
  }

  static const char status_string[mjNBOXQP][50]= {
    "Hessian is not positive definite",
    "No descent direction found",
    "Maximum main iterations exceeded",
    "Maximum line-search iterations exceeded",
    "Gradient norm smaller than tolerance",
    "No dimensions clamped, returning Newton point",
    "All dimensions clamped"
  };

  // no bounds: return Newton point
  if (!lower && !upper) {
    // try to factorize (only lower triangle of H is read and written to R)
    for (int i=0; i < n; i++) {
      for (int j=0; j <= i; j++) {
        R[i*n+j] = H[i*n+j];
      }
    }
    int rank = mju_cholFactor(R, n, mjMINVAL);
    if (rank == n) {
      mju_cholSolve(res, R, g, n);
      mju_scl(res, res, -1, n);
      value = 0.5 * mju_dot(res, g, n);
      nfactor = 1;
      status = mjBOXQP_UNBOUNDED;
    } else {
      status = mjBOXQP_NOT_SPD;
    }

    // full index set (no clamping)
    for (int i=0; i < n; i++) {
      index[i] = i;
    }
  }

  // have bounds: clamp res
  else {
    for (int i=0; i < n; i++) {
      if (lower) {
        res[i] = mju_max(res[i], lower[i]);
      }
      if (upper) {
        res[i] = mju_min(res[i], upper[i]);
      }
    }
  }

  // ------ main loop
  int iter, logptr = 0;
  for (iter=0; iter < maxiter; iter++) {
    // stop if already resolved (e.g. unbounded fast path or line-search failure)
    if (status != mjBOXQP_NO_DESCENT) {
      break;
    }

    // compute gradient and objective: value = 0.5*res'*H*res + res'*g
    mulSymVec(grad, H, res, n);
    if (iter == 0) {
      value = 0.5 * mju_dot(res, grad, n) + mju_dot(res, g, n);
    }
    mju_addTo(grad, g, n);

    // save last value
    mjtNum oldvalue = value;

    // find clamped dimensions, count free dimensions
    int oldnfree = nfree;
    nfree = 0;
    for (int i=0; i < n; i++) {
      clamped[i] = ( lower && res[i] == lower[i] && grad[i] > 0 ) ||
                   ( upper && res[i] == upper[i] && grad[i] < 0 );
      if (!clamped[i]) {
        nfree++;
      }
    }

    // all dimensions are clamped: minimum found
    if (!nfree) {
      status = mjBOXQP_ALL_CLAMPED;
      break;
    }

    // update or re-factorize if clamped dimensions have changed
    int rank = nfree;
    int factorize = (iter == 0);
    if (iter) {
      // count newly clamped and unclamped dimensions
      int nclamp = 0, nunclamp = 0;
      for (int i=0; i < n; i++) {
        if (clamped[i] && !oldclamped[i]) {
          nclamp++;
        } else if (!clamped[i] && oldclamped[i]) {
          nunclamp++;
        }
      }

      // maybe update Cholesky factor with rank-1 operations
      if (nclamp + nunclamp > 0) {
        // use rank-1 updates rather than full re-factorization when:
        //  - at least one previously free variable remains free (nclamp < oldnfree)
        //  - insertions are weighted 2x deletions because cholInsert adds a triangular solve
        //  - weighted count <= max(2, nfree/8): flop crossover is nfree/3, but cholFactor uses
        //    row-wise SIMD mju_dot while updates shift R and do column-strided Givens rotations
        if (nclamp < oldnfree && nclamp + 2*nunclamp <= maxInt(2, nfree/8)) {
          int m = oldnfree;
          int rank_ok = 1;

          // delete newly clamped dimensions in reverse order
          for (int p=m-1; p >= 0; p--) {
            if (clamped[index[p]]) {
              if (cholDelete(R, temp, m, p) != m-1) {
                rank_ok = 0;
                break;
              }
              for (int j=p; j < m-1; j++) {
                index[j] = index[j+1];
              }
              m--;
            }
          }

          // insert newly unclamped dimensions
          if (nunclamp && rank_ok) {
            int p = 0;
            for (int i=0; i < n; i++) {
              if (!clamped[i]) {
                if (oldclamped[i]) {
                  if (cholInsert(R, temp, search, H, n, index, m, p, i) != m+1) {
                    rank_ok = 0;
                    break;
                  }
                  for (int j=m; j > p; j--) {
                    index[j] = index[j-1];
                  }
                  index[p] = i;
                  m++;
                }
                p++;
              }
            }
          }

          if (!rank_ok) {
            factorize = 1;
          }
        } else {
          factorize = 1;
        }
      }
    }

    // full factorization if required
    if (factorize) {
      int j = 0;
      for (int i=0; i < n; i++) {
        if (!clamped[i]) {
          index[j++] = i;
        }
      }
      for (int i=0; i < nfree; i++) {
        for (int k=0; k < i+1; k++) {
          R[i*nfree+k] = H[index[i]*n+index[k]];
        }
      }
      rank = mju_cholFactor(R, nfree, mjMINVAL);
      nfactor++;
    }

    // save last clamped
    for (int i=0; i < n; i++) {
      oldclamped[i] = clamped[i];
    }

    // abort if factorization failed
    if (rank != nfree) {
      status = mjBOXQP_NOT_SPD;
      break;
    }

    // ------ check gradient

    // squared norm of free gradient, compress grad_free into temp
    norm2 = 0;
    for (int i=0; i < nfree; i++) {
      mjtNum grad_i = grad[index[i]];
      temp[i] = grad_i;
      norm2 += grad_i*grad_i;
    }

    // small gradient: minimum found
    if (norm2 < mingrad) {
      status = nfree == n ? mjBOXQP_UNBOUNDED : mjBOXQP_TOL_GRAD;
      break;
    }

    // get search direction: solve H_free,free * temp = grad_free
    mju_cholSolve(temp, R, temp, nfree);

    // search = -temp, compute directional derivative sdotg
    mjtNum sdotg = 0;
    for (int i=0; i < nfree; i++) {
      search[i] = -temp[i];
      sdotg -= temp[i] * grad[index[i]];
    }

    // sanity check: make sure we have a descent direction
    if (sdotg >= 0) {
      break;  // SHOULD NOT OCCUR
    }

    // ------ projected Armijo line search in free subspace
    mjtNum step = 1, improvement = 0, reduction = 0;
    int nstep = 0, clipped = 0;
    do {
      // candidate = clamp(res_free + step*search)
      clipped = 0;
      for (int i=0; i < nfree; i++) {
        int idx = index[i];
        mjtNum c = res[idx] + step*search[i];
        if (lower && c < lower[idx]) {
          c = lower[idx];
          clipped = 1;
        } else if (upper && c > upper[idx]) {
          c = upper[idx];
          clipped = 1;
        }
        candidate[i] = c;
      }

      // objective reduction: -grad_free'*delta - 0.5*||R'*delta||^2
      if (!clipped) {
        reduction = -step*(1 - 0.5*step)*sdotg;
      } else {
        mju_zero(temp, nfree);
        mjtNum ddotg = 0;
        for (int i=0; i < nfree; i++) {
          int idx = index[i];
          mjtNum di = candidate[i] - res[idx];
          if (di) {
            ddotg += di * grad[idx];
            mju_addToScl(temp, R+i*nfree, di, i+1);
          }
        }
        reduction = -ddotg - 0.5*mju_dot(temp, temp, nfree);
      }
      value = oldvalue - reduction;

      // repeat until relative improvement >= Armijo, or break if step is too small
      nstep++;
      improvement = -reduction / (step*sdotg);
      if (improvement < armijo) {
        step = step*backtrack;
        if (step < minstep) {
          status = mjBOXQP_MAX_LS_ITER;
          break;
        }
      }
    } while (improvement < armijo);


    // print iteration info
    if (log && logptr < logsz) {
      int written = snprintf(log+logptr, logsz-logptr,
                             "iter %-3d:  |grad|: %-8.2g  reduction: %-8.2g  improvement: %-8.4g  "
                             "linesearch: %g^%-2d  factorized: %d  nfree: %d\n",
                             iter+1, mju_sqrt(norm2), reduction, improvement,
                             backtrack, nstep-1, factorize, nfree);
      if (written > 0) {
        logptr = minInt(logptr + written, logsz);
      }
    }

    // stop if line search failed without accepting rejected candidate
    if (status == mjBOXQP_MAX_LS_ITER) {
      value = oldvalue;
      break;
    }

    // accept candidate
    for (int i=0; i < nfree; i++) {
      res[index[i]] = candidate[i];
    }

    // stop if unclipped step produced no floating-point decrease in objective
    if (!clipped && value >= oldvalue) {
      status = nfree == n ? mjBOXQP_UNBOUNDED : mjBOXQP_TOL_GRAD;
      break;
    }
  }

  // max iterations exceeded
  if (iter == maxiter) {
    status = mjBOXQP_MAX_ITER;
  }

  // print final info
  if (log && logptr < logsz) {
    snprintf(log+logptr, logsz-logptr, "BOXQP: %s.\n"
             "iterations= %d,  factorizations= %d,  |grad|= %-12.6g, final value= %-12.6g\n",
             status_string[status+1], iter, nfactor, mju_sqrt(norm2), value);
  }

  // return nf or -1 if failure
  return (status == mjBOXQP_NO_DESCENT || status == mjBOXQP_NOT_SPD) ? -1 : nfree;
}
