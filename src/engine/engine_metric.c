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

#include "engine/engine_metric.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjtype.h>
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_smooth.h"
#include "engine/engine_core_util.h"
#include "engine/engine_crossplatform.h"
#include "engine/engine_derivative.h"
#include "engine/engine_memory.h"
#include "engine/engine_sleep.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"



//------------------- implicit effective metric Mtilde = M + h*D + h^2*K --------------------------
// The flex part is K = (h^2 + h*damping) * (K_bend + K_stretch), the PSD implicit flex stiffness,
// whose h^2 and h*damping parts enter only when the spring and damper forces are enabled;
// the diagonal part holds joint damping and stiffness (h*D + h^2*K per dof, clamped PSD). Built
// once per step on the arena by mj_effBuild (under integrator=discrete), then consumed
// uniformly: the smooth acceleration, the constraint solver and inverse dynamics all see the
// same metric.

// arena allocation with hard failure (mirrors stack overflow semantics)
static void* effAlloc(mjData* d, size_t bytes, size_t align) {
  void* p = mj_arenaAllocByte(d, bytes, align);
  if (!p) {
    mjERROR("arena overflow in implicit effective metric");
  }
  return p;
}
#define EFMALLOC(type, n) (type*) effAlloc(d, sizeof(type)*(size_t)(n), _Alignof(type))

// yield the next live rank-1 term of the metric; see the declaration for the contract
int mj_effRank1Next(const mjModel* m, const mjData* d, mjEffRank1Iter* it,
                    mjEffRank1* e, int flg_contact) {
  // class 0: one entry per tendon with metric terms
  while (it->cls == 0) {
    if (it->i >= d->nefmT) {
      it->cls = 1;
      it->i = 0;
      break;
    }
    int t = it->i++;
    mjtNum s = d->efm_ts[t];
    int id = d->efm_tid[t];
    int nnz = m->ten_J_rownnz[id];
    if (!s || !nnz) {
      continue;
    }
    e->val = d->ten_J + m->ten_J_rowadr[id];
    e->colind = m->ten_J_colind + m->ten_J_rowadr[id];
    e->nnz = nnz;
    e->scale = s;
    return 1;
  }

  // class 1: one entry per output row of each actuator with metric terms
  while (it->cls == 1) {
    if (it->i >= d->nefmA) {
      it->cls = 2;
      it->i = 0;
      it->k = 0;  // class 2 walks the packed rows by offset
      break;
    }
    int id = d->efm_aid[it->i];
    if (it->k >= m->actuator_outnum[id]) {
      it->i++;
      it->k = 0;
      continue;
    }
    mjtNum s = d->efm_as[it->i];
    int r = m->actuator_outadr[id] + it->k++;
    int nnz = d->moment_rownnz[r];
    if (!s || !nnz) {
      continue;
    }
    e->val = d->actuator_moment + d->moment_rowadr[r];
    e->colind = d->moment_colind + d->moment_rowadr[r];
    e->nnz = nnz;
    e->scale = s;
    return 1;
  }

  // class 2: one entry per passive flex contact, published by effContactBuild. Skipped
  // wholesale when the caller accounts for contact separately
  while (it->cls == 2) {
    if (!flg_contact || it->k >= d->nefmcon) {
      it->cls = 3;  // not a producer class: the cursor's done state
      break;
    }
    int adr = it->k, nnz = d->efm_con_ind[adr];   // packed row: header then entries
    e->nnz = nnz;
    e->scale = d->efm_con_val[adr];
    e->colind = d->efm_con_ind + adr + 2;
    e->val = d->efm_con_val + adr + 2;
    it->k = adr + 2 + nnz;
    return 1;
  }
  return 0;
}


// res += B*vec, the stiffness part of the metric. The per-dof diagonal classes are applied
// from efm_diag (scales pre-folded); the stretch (and, when assemblable, bending and interp)
// part from the per-step CSR; terms not in the CSR fall back to the matrix-free operators.
void mj_effMulAdd(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec, int flg_contact) {
  mjtNum h = m->opt.timestep;
  if (d->efm_diag) {
    int nv = m->nv;
    for (int i=0; i < nv; i++) {
      res[i] += d->efm_diag[i] * vec[i];
    }
  }

  // fluid drag blocks: res += F*vec, F symmetric in M's lower-triangle pattern
  if (d->efm_fluid) {
    int nv = m->nv;
    for (int i=0; i < nv; i++) {
      int start = m->M_rowadr[i];
      int diag = start + m->M_rownnz[i] - 1;
      mjtNum acc = d->efm_fluid[diag] * vec[i];
      for (int a=start; a < diag; a++) {
        int c = m->M_colind[a];
        mjtNum F = d->efm_fluid[a];
        acc += F * vec[c];
        res[c] += F * vec[i];
      }
      res[i] += acc;
    }
  }

  // rank-1 terms (tendon, actuator): res += scale * row' * (row * vec)
  mjEffRank1Iter it = {0};
  mjEffRank1 e;
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/0)) {
    mjtNum dot = 0;
    for (int j=0; j < e.nnz; j++) {
      dot += e.val[j] * vec[e.colind[j]];
    }
    dot *= e.scale;
    for (int j=0; j < e.nnz; j++) {
      res[e.colind[j]] += dot * e.val[j];
    }
  }

  // contact class: the same rank-1 apply over the packed rows, walked inline; the iterator's
  // per-row call is measurable with thousands of published pairs
  if (flg_contact) {
    const int* ind = d->efm_con_ind;
    const mjtNum* val = d->efm_con_val;
    for (int adr=0; adr < d->nefmcon; ) {
      int nnz = ind[adr];
      const int* ci = ind + adr + 2;
      const mjtNum* rv = val + adr + 2;
      mjtNum dot = 0;
      for (int j=0; j < nnz; j++) {
        dot += rv[j] * vec[ci[j]];
      }
      dot *= val[adr];
      for (int j=0; j < nnz; j++) {
        res[ci[j]] += dot * rv[j];
      }
      adr += 2 + nnz;
    }
  }

  if (d->nefmK && d->nefmdof) {
    // the stiffness rows come in vertex triples with 3x3 blocks per neighbour (the assembly
    // writes the three dofs of each neighbour in turn), so one column index per block and the
    // three rows share the vector loads
    const int* rowadr = d->efm_K_rowadr;
    const int* rownnz = d->efm_K_rownnz;
    const int* colind = d->efm_K_colind;
    const mjtNum* val = d->efm_K_val;
    for (int k=0; k < d->nefmdof; k++) {
      int i = d->efm_dofid[k];
      const mjtNum* v0 = val + rowadr[i];
      const mjtNum* v1 = val + rowadr[i+1];
      const mjtNum* v2 = val + rowadr[i+2];
      const int* ci = colind + rowadr[i];
      int nn = rownnz[i] / 3;
      mjtNum r0 = 0, r1 = 0, r2 = 0;
      for (int j=0; j < nn; j++) {
        int c = ci[3*j];
        mjtNum x0 = vec[c], x1 = vec[c+1], x2 = vec[c+2];
        r0 += v0[3*j]*x0 + v0[3*j+1]*x1 + v0[3*j+2]*x2;
        r1 += v1[3*j]*x0 + v1[3*j+1]*x1 + v1[3*j+2]*x2;
        r2 += v2[3*j]*x0 + v2[3*j+1]*x1 + v2[3*j+2]*x2;
      }
      res[i] += r0;
      res[i+1] += r1;
      res[i+2] += r2;
    }
  } else if (d->nefmK) {
    int nv = m->nv;
    for (int i=0; i < nv; i++) {
      int nnz = d->efm_K_rownnz[i];
      if (!nnz) {
        continue;
      }
      res[i] += mju_dotSparse(d->efm_K_val + d->efm_K_rowadr[i], vec, nnz,
                              d->efm_K_colind + d->efm_K_rowadr[i]);
    }
  }

  // A standard flex is assembled only when all its attachments have fixed-frame XYZ
  // mappings. Dispatch general flexes individually: another flex's CSR must not suppress
  // their operators. Both the stiffness and damping parts follow the passive-force flags.
  mjtNum s1 = mjDISABLED(mjDSBL_SPRING) ? 0 : h*h;
  mjtNum s2 = mjDISABLED(mjDSBL_DAMPER) ? 0 : h;
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_interp[f] || m->flex_rigid[f]) continue;
    if (m->flex_dim[f] == 1 && m->flex_bendingadr[f] < 0) continue;
    if (!d->nefmK || !mj_flexSimple(m, f)) {
      mjd_flexBend_mulRange(m, d, res, vec, s1, s2, f, f+1);
      mjd_flexStretch_mulRange(m, d, res, vec, s1, s2, f, f+1);
    }
  }
  if (!d->nefmK || !mjd_flexInterpAssemblable(m)) {
    mjd_flexInterp_mul(m, d, res, vec, -s1, -s2, d->flexelem_krot);
  }
}


// the unfactored 3x3 diagonal block of M + K for the covered dof triple starting at i
static void effBlockRaw(const mjModel* m, const mjData* d, int i, mjtNum* Bk) {
  mju_zero(Bk, 9);
  for (int r = 0; r < 3; r++) {
    int row = i + r;
    for (int a = m->M_rowadr[row]; a < m->M_rowadr[row] + m->M_rownnz[row]; a++) {
      int c = m->M_colind[a];
      if (c >= i && c < i+3) Bk[3*r + (c-i)] += d->M[a];
    }
    for (int a = d->efm_K_rowadr[row]; a < d->efm_K_rowadr[row] + d->efm_K_rownnz[row]; a++) {
      int c = d->efm_K_colind[a];
      if (c >= i && c < i+3) Bk[3*r + (c-i)] += d->efm_K_val[a];
    }
  }
}


//------------------ sparse Cholesky metric preconditioner ----------------------------------------
// When every covered flex is a 1D rod or a 2D flex (effChol12D), the covered block of M + K is
// factored by a reverse-order sparse Cholesky, blocked by vertex, rather than by its 3x3 diagonal
// blocks only. 2D flexes take the factor only where the constraint solver folds into it (IPC, or
// monolithic CG: effCholApplies, mj_effFoldMonolithic); 3D flexes keep the blocks.
// The ordering is the model's: mj_setConst orders the vertices of the model's efm_K pattern by
// minimum degree (mj_effCholSetConst, into efmC_perm). Each step analyses the pattern it
// assembled under that ordering (effCholReserve), on the vertex graph
// (mju_cholFactorSymbolicBlocked), and packs the analysis into the metric's buffer efm_L, next to
// the factor it describes:
//   [9*nb 3x3 blocks][status][the analysis (effPackLayout)][nL factor values]
// so the blocks remain the fallback whenever the factor is unusable. When the constraint solver
// can fold its efc rows into the factor, the step's pattern also takes the vertex pairs those
// rows couple (effCholConPairs), reordered by minimum degree when some are not flex pairs. The
// numeric factorization runs lazily for the metric's copy (effCholEnsure) and per constraint
// solve, on the same analysis, for the solver's folded copy in mjEffFold.F
// (mj_effCholFoldFactor), with any pairs off the pattern complemented on the diagonal.

#define EFF_HDR 1
#define EFF_CHOL_PENDING 2
#define EFF_ISLAND_TAG -1
#define EFF_CHOL_MAXVERT 8192
#define EFF_CHOL_PIVOT 1e-12

typedef struct {
  int nefc;
  const mjtNum* D;
  int is_sparse;
  const mjtNum* J;
  const int *rownnz, *rowadr, *colind;
} mjEffConRows;

typedef struct {
  int nb, n, nL, ne;
  const int* dofid;
  int *perm, *L_rownnz, *L_rowadr, *L_colind, *BT_adr, *BT_row, *BT_off;
  int *eadr, *enbr;
} mjEffChol;

static size_t effIntBytes(size_t n) {
  return mj_stackBytes(sizeof(int)*(n > 0 ? n : 1), _Alignof(int));
}

static size_t effNumBytes(size_t n) {
  return mj_stackBytes(sizeof(mjtNum)*n, _Alignof(mjtNum));
}

static size_t effBitBytes(size_t n) {
  return mj_stackBytes(sizeof(uint64_t)*(n > 0 ? n : 1), _Alignof(uint64_t));
}

static int* effInts(mjData* d, size_t n) {
  return mjSTACKALLOC(d, n > 0 ? n : 1, int);
}

static uint64_t* effBits(mjData* d, size_t n) {
  return mjSTACKALLOC(d, n > 0 ? n : 1, uint64_t);
}

static int effPopcount(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(x);
#else
  x = x - ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  return (int)((x * 0x0101010101010101ULL) >> 56);
#endif
}

static int effCtz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(x);
#else
  int b = 0;
  while (!((x >> b) & 1ULL)) b++;
  return b;
#endif
}

// off-diagonal 3x3 blocks of a blocked factor with nL values: row r of vertex p holds 3 values per
// block of the vertex and r - 3*p + 1 in the diagonal block
static int effCholBlocks(int nb, int nL) {
  return (nL - 6*nb) / 9;
}

// the step's analysis, packed into efm_L before the factor values: nL, ne, then perm (nb),
// L_rownnz, L_rowadr (n), L_colind (nL), BT_adr (nb+1), BT_row, BT_off (effCholBlocks),
// eadr (nb+1), enbr (ne). Holds no pointer, so mj_copyData carries it with efm_L
static void effPackLayout(int* A, int nb, const int* dofid, mjEffChol* c) {
  int n = 3*nb;
  c->nb = nb;
  c->n = n;
  c->nL = A[0];
  c->ne = A[1];
  c->dofid = dofid;
  int* p = A + 2;
  c->perm = p;       p += nb;
  c->L_rownnz = p;   p += n;
  c->L_rowadr = p;   p += n;
  c->L_colind = p;   p += c->nL;
  c->BT_adr = p;     p += nb + 1;
  c->BT_row = p;     p += effCholBlocks(nb, c->nL);
  c->BT_off = p;     p += effCholBlocks(nb, c->nL);
  c->eadr = p;       p += nb + 1;
  c->enbr = p;
}

static size_t effPackNums(int nb, int nL, int ne) {
  size_t ints = 4 + 9*(size_t)nb + (size_t)nL + 2*(size_t)effCholBlocks(nb, nL) + (size_t)ne;
  return (sizeof(int)*ints + sizeof(mjtNum) - 1) / sizeof(mjtNum);
}

static void effCholStep(const mjData* d, mjEffChol* c) {
  int nb = d->nefmdof;
  effPackLayout((int*)(d->efm_L + 9*nb + EFF_HDR), nb, d->efm_dofid, c);
}

static int effCholReserved(const mjData* d) {
  int nb = d->nefmdof;
  return nb && d->nefmL > 9*nb && d->efm_L[9*nb] >= 0;
}

// per-island flex/triple lists, packed into efm_L after the 9*nefmdof 3x3 blocks when islands are
// active, IPC is disabled and no sparse factor is reserved (effCholReserve):
//   [9*nb 3x3 blocks][EFF_ISLAND_TAG][nisland, nb, nbd, nflex, cov_adr, cov_ind, bend_adr, ...]
typedef struct {
  int nisland, nb, nbd, nflex;
  int *cov_adr, *cov_ind;    // covered block indices k in [0, nefmdof) per island
  int *bend_adr, *bend_ind;  // bending dof indices b in [0, nefm0dof) per island
  int *flex_adr, *flex_ind;  // matrix-free flex indices f in [0, nflex) per island
} mjEffIsland;

static int effIslandTail(const mjModel* m, const mjData* d, int nb) {
  int nisland = d->nisland;
  int nbd = nb ? 0 : m->nefm0dof;
  int nflex = m->nflex;
  size_t ints = 4 + 3*(size_t)(nisland + 1) + (size_t)nb + (size_t)nbd + (size_t)nflex;
  size_t npack = (sizeof(int)*ints + sizeof(mjtNum) - 1) / sizeof(mjtNum);
  return EFF_HDR + (int)npack;
}

static void effIslandPackLayout(int* A, mjEffIsland* isl) {
  isl->nisland = A[0];
  isl->nb = A[1];
  isl->nbd = A[2];
  isl->nflex = A[3];
  int* p = A + 4;
  isl->cov_adr = p;   p += isl->nisland + 1;
  isl->cov_ind = p;   p += isl->nb;
  isl->bend_adr = p;  p += isl->nisland + 1;
  isl->bend_ind = p;  p += isl->nbd;
  isl->flex_adr = p;  p += isl->nisland + 1;
  isl->flex_ind = p;
}

static int effIslandStep(const mjData* d, mjEffIsland* isl) {
  int nb = d->nefmdof;
  if (d->nefmL <= 9*nb || d->efm_L[9*nb] != EFF_ISLAND_TAG) {
    return 0;
  }
  int* A = (int*)(d->efm_L + 9*nb + EFF_HDR);
  if (A[0] != d->nisland) {
    return 0;
  }
  effIslandPackLayout(A, isl);
  return 1;
}

static mjtNum* effCholStepValues(const mjData* d, const mjEffChol* c) {
  return d->efm_L + 9*c->nb + EFF_HDR + effPackNums(c->nb, c->nL, c->ne);
}

// the covered blocks of a K pattern with row sizes rownnz (nv rows): each nonzero row starts a
// vertex triple, its 3 consecutive dofs
static int effCoveredBlocks(int nv, const int* rownnz, int* dofid) {
  int nb = 0;
  for (int i=0; i < nv; ) {
    if (rownnz[i]) {
      if (dofid) dofid[nb] = i;
      nb++;
      i += 3;
    } else {
      i++;
    }
  }
  return nb;
}

static void effCovered(int nv, int nb, const int* dofid, int* cov) {
  for (int i=0; i < nv; i++) cov[i] = -1;
  for (int k=0; k < nb; k++) {
    for (int a=0; a < 3; a++) cov[dofid[k]+a] = k;
  }
}

// greedy minimum-degree elimination on the vertex graph (symmetric CSR adr/cnt/nbr), with dense
// bitset adjacency; the first eliminated vertex goes last (reverse-order Cholesky)
static void effCholMinDegree(mjData* d, int nb, const int* adr, const int* cnt, const int* nbr,
                             int* perm) {
  int words = (nb + 63) / 64;
  uint64_t* adj = effBits(d, (size_t)nb*words);
  uint64_t* alive = effBits(d, words);
  int* deg = effInts(d, nb);
  memset(adj, 0, sizeof(uint64_t)*(size_t)nb*words);
  memset(alive, 0, sizeof(uint64_t)*words);
  for (int k=0; k < nb; k++) {
    alive[k/64] |= 1ULL << (k%64);
    for (int j=0; j < cnt[k]; j++) {
      int q = nbr[adr[k]+j];
      adj[(size_t)k*words+q/64] |= 1ULL << (q%64);
      adj[(size_t)q*words+k/64] |= 1ULL << (k%64);
    }
  }
  for (int k=0; k < nb; k++) {
    adj[(size_t)k*words+k/64] &= ~(1ULL << (k%64));
    int dg = 0;
    for (int w=0; w < words; w++) dg += effPopcount(adj[(size_t)k*words+w]);
    deg[k] = dg;
  }
  for (int step=0; step < nb; step++) {
    int v = -1;
    for (int w=0; w < words; w++) {
      uint64_t aw = alive[w];
      while (aw) {
        int b = effCtz64(aw);
        aw &= aw - 1;
        int k = 64*w + b;
        if (v < 0 || deg[k] < deg[v]) v = k;
      }
    }
    perm[v] = nb - 1 - step;
    alive[v/64] &= ~(1ULL << (v%64));
    uint64_t* av = adj + (size_t)v*words;
    for (int w=0; w < words; w++) {
      uint64_t bits = av[w] & alive[w];
      while (bits) {
        int b = effCtz64(bits);
        bits &= bits - 1;
        int u = 64*w + b;
        uint64_t* au = adj + (size_t)u*words;
        int dg = 0;
        for (int x=0; x < words; x++) {
          au[x] |= av[x];
          if (x == u/64) au[x] &= ~(1ULL << (u%64));
          dg += effPopcount(au[x] & alive[x]);
        }
        deg[u] = dg;
      }
    }
  }
}

static int effCholFind(const int* L_rownnz, const int* L_rowadr, const int* L_colind, int r,
                       int col) {
  int adr = L_rowadr[r], nnz = L_rownnz[r];
  if (col == r) {
    return adr + nnz - 1;
  }
  int lo = 0, hi = nnz - 2;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    int cm = L_colind[adr+mid];
    if (cm == col) return adr + mid;
    if (cm < col) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

static void effAllow(int* amark, const int* eadr, const int* enbr, int k) {
  for (int j=eadr[k]; j < eadr[k+1]; j++) {
    amark[enbr[j]] = k;
  }
}

static int effAllowed(const int* amark, int ne, int k, int q) {
  return !ne || amark[q] == k;
}

// upper bound on the number of allowed flex vertex pairs (flex edges plus 1D rod hinge spans)
static size_t effCholFlexPairs(const mjModel* m) {
  size_t np = (size_t)m->nflexedge;
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_dim[f] == 1 && !m->flex_rigid[f] && m->flex_bendingadr[f] >= 0) {
      const mjtNum* bend = m->flex_bending + m->flex_bendingadr[f];
      for (int h=0; h < (int)bend[0]; h++) {
        np += mj_rodHingeVert(bend + 1 + mjROD_HINGE_SIZE*h) == 3 ? 1 : 3;
      }
    }
  }
  return np;
}

// the sorted, deduplicated symmetric CSR (eadr/enbr) of the np vertex pairs apair, on the stack
static void effPairCSR(mjData* d, int nb, const int* apair, int np, int** eadr, int** enbr) {
  int* ea = *eadr = effInts(d, nb + 1);
  int* en = *enbr = effInts(d, 2*(size_t)np);
  int* cur = effInts(d, nb);
  for (int k=0; k <= nb; k++) ea[k] = 0;
  for (int x=0; x < 2*np; x++) ea[apair[x]+1]++;
  for (int k=0; k < nb; k++) {
    ea[k+1] += ea[k];
    cur[k] = ea[k];
  }
  for (int x=0; x < np; x++) {
    en[cur[apair[2*x]]++] = apair[2*x+1];
    en[cur[apair[2*x+1]]++] = apair[2*x];
  }
  int ne = 0, start = 0;
  for (int k=0; k < nb; k++) {
    int end = ea[k+1];
    for (int x=start+1; x < end; x++) {
      int v = en[x], y = x;
      for (; y > start && en[y-1] > v; y--) en[y] = en[y-1];
      en[y] = v;
    }
    ea[k] = ne;
    for (int x=start; x < end; x++) {
      if (x == start || en[x] != en[x-1]) en[ne++] = en[x];
    }
    start = end;
  }
  ea[nb] = ne;
}

// pair (k, q) is in the sorted CSR (ea, en)
static int effPairIn(const int* ea, const int* en, int k, int q) {
  int lo = ea[k], hi = ea[k+1] - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (en[mid] == q) return 1;
    if (en[mid] < q) lo = mid + 1;
    else hi = mid - 1;
  }
  return 0;
}

// build the K vertex graph (kadr/knbr) and the allowed edge CSR (eadr/enbr: flex edges and 1D rod
// hinge spans between covered vertices plus those of the nx extra pairs xpair that are not flex
// pairs, *nnew of them, both ways, sorted and deduplicated)
static int effCholGraph(mjData* d, const mjModel* m, int nb, const int* dofid, const int* blk,
                        const int* K_rownnz, const int* K_rowadr, const int* K_colind,
                        const int* xpair, int nx, int** kadr, int** knbr, int** eadr,
                        int** enbr, int* nnew) {
  size_t kcap = 0;
  for (int k=0; k < nb; k++) kcap += K_rownnz[dofid[k]];
  int* ka = *kadr = effInts(d, nb + 1);
  int* kn = *knbr = effInts(d, kcap);
  ka[0] = 0;
  for (int k=0; k < nb; k++) {
    int i = dofid[k];
    ka[k+1] = ka[k];
    for (int j=K_rowadr[i]; j < K_rowadr[i] + K_rownnz[i]; j++) {
      int col = K_colind[j], q = blk[col];
      if (q >= 0 && q != k && col == dofid[q]) kn[ka[k+1]++] = q;
    }
  }
  int np = 0;
  int* apair = effInts(d, 2*(effCholFlexPairs(m) + nx));
  for (int f=0; f < m->nflex; f++) {
    for (int e=0; e < m->flex_edgenum[f]; e++) {
      int ke[2];
      for (int s=0; s < 2; s++) {
        int v = m->flex_edge[2*(m->flex_edgeadr[f]+e)+s];
        int b = m->flex_vertbodyid[m->flex_vertadr[f]+v];
        int da = (b >= 0 && m->body_dofnum[b] == 3) ? m->body_dofadr[b] : -1;
        ke[s] = (da >= 0 && blk[da] >= 0 && dofid[blk[da]] == da) ? blk[da] : -1;
      }
      if (ke[0] >= 0 && ke[1] >= 0 && ke[0] != ke[1]) {
        apair[2*np] = ke[0];
        apair[2*np+1] = ke[1];
        np++;
      }
    }
    if (m->flex_dim[f] == 1 && !m->flex_rigid[f] && m->flex_bendingadr[f] >= 0) {
      const mjtNum* bend = m->flex_bending + m->flex_bendingadr[f];
      // vertex spans not covered by flex edges: (0, 2), plus (1, 3) and (0, 3) for a hinge pair
      static const int span[3][2] = {{0, 2}, {1, 3}, {0, 3}};
      for (int h=0; h < (int)bend[0]; h++) {
        const mjtNum* hinge = bend + 1 + mjROD_HINGE_SIZE*h;
        int nr = mj_rodHingeVert(hinge);
        int kv[4];
        for (int s=0; s < nr; s++) {
          int v = (int)hinge[s];
          int b = m->flex_vertbodyid[m->flex_vertadr[f]+v];
          int da = (b >= 0 && m->body_dofnum[b] == 3) ? m->body_dofadr[b] : -1;
          kv[s] = (da >= 0 && blk[da] >= 0 && dofid[blk[da]] == da) ? blk[da] : -1;
        }
        int nspan = (nr == 3) ? 1 : 3;
        for (int p=0; p < nspan; p++) {
          int k0 = kv[span[p][0]], k1 = kv[span[p][1]];
          if (k0 >= 0 && k1 >= 0 && k0 != k1) {
            apair[2*np] = k0;
            apair[2*np+1] = k1;
            np++;
          }
        }
      }
    }
  }
  int nfe = 2*np;
  effPairCSR(d, nb, apair, np, eadr, enbr);

  // the extra pairs that are not flex pairs: the CSR again with them
  int nk = 0;
  for (int x=0; x < nx; x++) {
    int k = xpair[2*x], q = xpair[2*x+1];
    if (k != q && !effPairIn(*eadr, *enbr, k, q)) {
      apair[2*(np+nk)] = k;
      apair[2*(np+nk)+1] = q;
      nk++;
    }
  }
  if (nk) {
    effPairCSR(d, nb, apair, np + nk, eadr, enbr);
  }
  *nnew = nk;
  return nfe;
}

// single-pass symbolic analysis on the stack: builds the vertex graph, runs minimum-degree
// ordering if order is 1, or 2 and the extra pairs are not all flex pairs (else perm is the
// caller's), and computes the blocked symbolic Cholesky counts
typedef struct {
  int nb, nL, ne, nfe, nnew;
  int *eadr, *enbr, *rownnz, *rowadr, *colind, *scratch;
  int *L_rownnz, *L_rowadr, *BT_adr;
} mjEffCholPass;

static int effCholBegin(const mjModel* m, mjData* d, int nb, const int* dofid, const int* blk,
                        const int* K_rownnz, const int* K_rowadr, const int* K_colind,
                        const int* xpair, int nx, int order, int* perm, mjEffCholPass* p) {
  if (nb <= 0 || nb > EFF_CHOL_MAXVERT) {
    return 0;
  }
  int n = 3*nb, *kadr, *knbr;
  p->nb = nb;
  p->nfe = effCholGraph(d, m, nb, dofid, blk, K_rownnz, K_rowadr, K_colind, xpair, nx,
                        &kadr, &knbr, &p->eadr, &p->enbr, &p->nnew);
  p->ne = p->eadr[nb];
  int* amark = effInts(d, nb);
  int* mark = effInts(d, nb);
  int* adr = effInts(d, nb);
  int* cnt = effInts(d, nb);
  int* nbr = effInts(d, (size_t)kadr[nb] + p->ne);
  for (int k=0; k < nb; k++) amark[k] = mark[k] = -1;
  int tot = 0;
  for (int k=0; k < nb; k++) {
    adr[k] = tot;
    mark[k] = k;
    effAllow(amark, p->eadr, p->enbr, k);
    for (int j=kadr[k]; j < kadr[k+1]; j++) {
      int q = knbr[j];
      if (effAllowed(amark, p->ne, k, q) && mark[q] != k) {
        mark[q] = k;
        nbr[tot++] = q;
      }
    }
    for (int j=p->eadr[k]; j < p->eadr[k+1]; j++) {
      int q = p->enbr[j];
      if (mark[q] != k) {
        mark[q] = k;
        nbr[tot++] = q;
      }
    }
    cnt[k] = tot - adr[k];
  }
  if (order == 1 || (order == 2 && p->nnew)) {
    effCholMinDegree(d, nb, adr, cnt, nbr, perm);
  }
  p->rownnz = effInts(d, nb);
  p->rowadr = effInts(d, nb);
  p->colind = effInts(d, (size_t)kadr[nb] + p->ne);
  p->scratch = effInts(d, 3*(size_t)nb);
  for (int k=0; k < nb; k++) p->rownnz[perm[k]] = cnt[k];
  for (int pos=0, a=0; pos < nb; pos++) {
    p->rowadr[pos] = a;
    a += p->rownnz[pos];
  }
  for (int k=0; k < nb; k++) {
    int* row = p->colind + p->rowadr[perm[k]];
    for (int j=0; j < cnt[k]; j++) {
      int v = perm[nbr[adr[k]+j]], y = j;
      for (; y > 0 && row[y-1] > v; y--) row[y] = row[y-1];
      row[y] = v;
    }
  }
  p->L_rownnz = effInts(d, n);
  p->L_rowadr = effInts(d, n);
  p->BT_adr = effInts(d, nb + 1);
  p->nL = mju_cholFactorSymbolicBlocked(NULL, p->L_rownnz, p->L_rowadr, p->BT_adr, NULL, NULL,
                                        p->rownnz, p->rowadr, p->colind, nb, 3, p->scratch);
  return 1;
}

static void effCholFinish(const mjEffCholPass* p, mjEffChol* c) {
  int nb = p->nb, n = 3*nb;
  memcpy(c->L_rownnz, p->L_rownnz, sizeof(int)*n);
  memcpy(c->L_rowadr, p->L_rowadr, sizeof(int)*n);
  memcpy(c->BT_adr, p->BT_adr, sizeof(int)*(nb + 1));
  memcpy(c->eadr, p->eadr, sizeof(int)*(nb + 1));
  if (p->ne) {
    memcpy(c->enbr, p->enbr, sizeof(int)*p->ne);
  }
  mju_cholFactorSymbolicBlocked(c->L_colind, c->L_rownnz, c->L_rowadr, c->BT_adr, c->BT_row,
                                c->BT_off, p->rownnz, p->rowadr, p->colind, nb, 3, p->scratch);
}

// a constraint solve with this solver can fold its efc rows into the sparse factor
// (mj_effPrecFold): CG over a pyramidal cone, without sleep, which needs the island path
// (mj_effFoldMonolithic)
static int effFoldPossible(const mjModel* m, int solver) {
  return solver == mjSOL_CG && m->opt.cone != mjCONE_ELLIPTIC && !mjENABLED(mjENBL_SLEEP);
}

// return 1 if every covered flex qualifies for the sparse Cholesky factor: 1D rods and, if
// allow2d, 2D flexes, not interpolated (currently all-or-nothing across covered flexes since
// efm_dofid and nefmCvert index all covered triples together)
static int effChol12D(const mjModel* m, int allow2d) {
  for (int f=0; f < m->nflex; f++) {
    int covered = mjd_flexStiff_active(m, f, /*flg_bend=*/1, /*flg_stretch=*/1) ||
                  mjd_flexInterp_processed(m, f) || mj_effFlexContactPossible(m, f);
    if (!covered) {
      continue;
    }
    if (m->flex_interp[f]) {
      return 0;
    }
    if (m->flex_dim[f] != 1 && !(allow2d && m->flex_dim[f] == 2)) {
      return 0;
    }
  }
  return 1;
}

// the step uses the sparse factor. 2D flexes need a consumer that folds the contact rows into it:
// on its own the factor makes each constraint-solver iteration costlier than the iterations it
// saves, so 2D flexes take it under IPC or when CG folds, and keep the 3x3 blocks otherwise
static int effCholApplies(const mjModel* m, int nb, const int* dofid) {
  int allow2d = mjENABLED(mjENBL_IPC) || effFoldPossible(m, m->opt.solver);
  if (!nb || nb > EFF_CHOL_MAXVERT || !m->nefmCvert || m->efmC_perm[0] < 0 ||
      !effChol12D(m, allow2d)) {
    return 0;
  }
  if (mjd_flexInterpAssemblable(m)) {
    for (int f=0; f < m->nflex; f++) {
      if (mjd_flexInterp_processed(m, f)) {
        return 0;
      }
    }
  }
  return dofid[nb-1] + 3 <= m->nv;
}

static size_t effCholKcap(const mjData* d, int nb, const int* dofid) {
  size_t cap = 0;
  for (int k=0; k < nb; k++) cap += d->efm_K_rownnz[dofid[k]];
  return cap;
}

static size_t effCholAnalysisBytes(const mjModel* m, size_t nb, size_t kcap, size_t nx) {
  size_t ne = effCholFlexPairs(m) + nx;
  size_t nnbr = kcap + 2*ne;
  size_t words = (nb + 63) / 64;
  return mj_stackFrameBytes() + 2*effIntBytes(nb + 1) + effIntBytes(kcap) + 2*effIntBytes(2*ne) +
         (nx ? effIntBytes(nb + 1) + effIntBytes(2*ne) + effIntBytes(nb) : 0) +
         effIntBytes(nb) + 4*effIntBytes(nb) + effIntBytes(nnbr) + effBitBytes(nb*words) +
         effBitBytes(words) + effIntBytes(nb) + 2*effIntBytes(nb) + effIntBytes(nnbr) +
         effIntBytes(3*nb) + 2*effIntBytes(3*nb) + effIntBytes(nb + 1);
}

static size_t effFillBytes(size_t nb, size_t nv) {
  return mj_stackFrameBytes() + effNumBytes(9*nb) + effIntBytes(nv) + 5*effIntBytes(nb);
}

static size_t effCholFillBytes(const mjEffChol* c, int nv) {
  return effFillBytes(c->nb, nv);
}

static size_t effConScratchBytes(size_t nb) {
  return 2*effIntBytes(nb) + effNumBytes(3*nb);
}

static size_t effCholReserveBytes(int nb, int nv, int nL, size_t npack) {
  return effNumBytes(EFF_HDR + npack + (size_t)nL) + effNumBytes(EFF_HDR + (size_t)nL) +
         effConScratchBytes(nb) + effFillBytes(nb, nv);
}

// compute the model's minimum-degree vertex ordering for 1D rod and 2D flex metric stencils
void mj_effCholSetConst(mjModel* m, mjData* d) {
  int ncap = m->nefmCvert, nv = m->nv;
  if (!ncap) {
    return;
  }
  for (int t=0; t < ncap; t++) m->efmC_perm[t] = -1;
  // the ordering is structural: computed for every model that can use the factor, whatever the
  // solver options (effCholApplies decides per step)
  if (!effChol12D(m, /*allow2d=*/1)) {
    return;
  }
  mj_markStack(d);
  size_t avail = mj_stackBytesAvailable(d);
  size_t need = mj_stackFrameBytes() + 3*effIntBytes(nv) + effIntBytes(ncap + 1);
  int nb = 0, nK = 0, *K_rownnz = NULL, *K_rowadr = NULL, *blk = NULL, *dofid = NULL;
  size_t kcap = 0;
  if (avail >= need) {
    K_rownnz = effInts(d, nv);
    K_rowadr = effInts(d, nv);
    blk = effInts(d, nv);
    dofid = effInts(d, ncap + 1);
    nK = mjd_flexStiff_assemble(m, d, K_rownnz, K_rowadr, NULL, NULL, 0, 0, 1, 1, NULL);
    nb = effCoveredBlocks(nv, K_rownnz, NULL);
    if (!nb || nb > ncap || nb > EFF_CHOL_MAXVERT) {
      mj_freeStack(d);
      return;
    }
    effCoveredBlocks(nv, K_rownnz, dofid);
    if (dofid[nb-1] + 3 > nv) {
      mj_freeStack(d);
      return;
    }
    for (int k=0; k < nb; k++) kcap += K_rownnz[dofid[k]];
    need += effIntBytes(nK) + effIntBytes(nb) + effCholAnalysisBytes(m, nb, kcap, 0);
  }
  if (avail < need) {
    mj_freeStack(d);
    mj_warning(d, mjWARN_CNSTRFULL, d->narena);
    return;
  }
  int* K_colind = effInts(d, nK);
  int* perm = effInts(d, nb);
  mjd_flexStiff_assemble(m, d, K_rownnz, K_rowadr, K_colind, NULL, 0, 0, 1, 1, NULL);
  effCovered(nv, nb, dofid, blk);
  mjEffCholPass pass;
  if (effCholBegin(m, d, nb, dofid, blk, K_rownnz, K_rowadr, K_colind, NULL, 0, 1, perm, &pass)) {
    for (int k=0; k < nb; k++) m->efmC_perm[perm[k]] = dofid[k];
  }
  mj_freeStack(d);
}

// numeric factorization by 3x3 vertex blocks: H = L'L backward elimination in place in L (the
// rows of a vertex are read before they are written, and only rows already factored are read)
static int effCholNumericBlocked(const mjEffChol* c, mjtNum* restrict L, mjtNum mindiag,
                                 mjtNum* restrict dense) {
  int nb = c->nb, rank = c->n;
  const int *rowadr = c->L_rowadr, *rownnz = c->L_rownnz, *colind = c->L_colind;
  for (int p=nb-1; p >= 0; p--) {
    int r0 = 3*p, nob = rownnz[r0] - 1, used_dense = 0;
    const int* ci = colind + rowadr[r0];
    mjtNum* restrict L0 = L + rowadr[r0];
    mjtNum* restrict L1 = L + rowadr[r0+1];
    mjtNum* restrict L2 = L + rowadr[r0+2];
    mjtNum* restrict La[3] = {L0, L1, L2};
    mjtNum d00 = L0[nob], d10 = L1[nob], d11 = L1[nob+1];
    mjtNum d20 = L2[nob], d21 = L2[nob+1], d22 = L2[nob+2];
    for (int k=c->BT_adr[p]; k < c->BT_adr[p+1]; k++) {
      int qrow = 3*c->BT_row[k], off = c->BT_off[k];
      const mjtNum *Q0 = L + rowadr[qrow], *Q1 = L + rowadr[qrow+1], *Q2 = L + rowadr[qrow+2];
      const int* qci = colind + rowadr[qrow];
      mjtNum l00 = Q0[off], l01 = Q0[off+1], l02 = Q0[off+2];
      mjtNum l10 = Q1[off], l11 = Q1[off+1], l12 = Q1[off+2];
      mjtNum l20 = Q2[off], l21 = Q2[off+1], l22 = Q2[off+2];
      if (!used_dense && (!off || qci[0] == ci[nob-off])) {
        int d = nob - off;
        for (int i=0; i < off; i++) {
          mjtNum y0 = Q0[i], y1 = Q1[i], y2 = Q2[i];
          L0[d+i] -= l00*y0 + l10*y1 + l20*y2;
          L1[d+i] -= l01*y0 + l11*y1 + l21*y2;
          L2[d+i] -= l02*y0 + l12*y1 + l22*y2;
        }
      } else {
        if (!used_dense) {
          used_dense = 1;
          for (int i=0; i < nob; i++) {
            mjtNum* dd = dense + 3*ci[i];
            dd[0] = L0[i];
            dd[1] = L1[i];
            dd[2] = L2[i];
          }
        }
        for (int i=0; i < off; i++) {
          mjtNum y0 = Q0[i], y1 = Q1[i], y2 = Q2[i];
          mjtNum* dd = dense + 3*qci[i];
          dd[0] -= l00*y0 + l10*y1 + l20*y2;
          dd[1] -= l01*y0 + l11*y1 + l21*y2;
          dd[2] -= l02*y0 + l12*y1 + l22*y2;
        }
      }
      d00 -= l00*l00 + l10*l10 + l20*l20;
      d10 -= l01*l00 + l11*l10 + l21*l20;
      d20 -= l02*l00 + l12*l10 + l22*l20;
      d11 -= l01*l01 + l11*l11 + l21*l21;
      d21 -= l02*l01 + l12*l11 + l22*l21;
      d22 -= l02*l02 + l12*l12 + l22*l22;
    }
    if (used_dense) {
      for (int i=0; i < nob; i++) {
        mjtNum* dd = dense + 3*ci[i];
        L0[i] = dd[0];
        L1[i] = dd[1];
        L2[i] = dd[2];
        dd[0] = dd[1] = dd[2] = 0;
      }
    }
    L0[nob] = d00;
    L1[nob] = d10;
    L1[nob+1] = d11;
    L2[nob] = d20;
    L2[nob+1] = d21;
    L2[nob+2] = d22;
    for (int a=2; a >= 0; a--) {
      mjtNum* restrict Ra = La[a];
      mjtNum diag = Ra[nob+a];
      int deficient = diag < mindiag;
      if (deficient) {
        diag = mindiag;
        rank--;
      }
      mjtNum Lrr = mju_sqrt(diag), inv = 1.0 / Lrr;
      if (deficient) {
        mju_zero(Ra, nob + a);
      } else {
        for (int i=0; i < nob + a; i++) Ra[i] *= inv;
      }
      Ra[nob+a] = Lrr;
      for (int e=0; e < a; e++) {
        mjtNum le = Ra[nob+e];
        if (!le) continue;
        mjtNum* restrict Re = La[e];
        for (int i=0; i <= nob + e; i++) Re[i] -= le * Ra[i];
      }
    }
  }
  return rank;
}

// assemble the covered block of M + K into values, in the factor's pattern (adding to the contact
// terms there if prefilled, else zeroing it first), and factor it in place. A K block B_kq is
// exact where the factor has a block for the pair (an allowed pair or, without IPC, fill); where
// it has none, B_kq is dropped and replaced by its absolute row sums on the diagonal of k and its
// absolute column sums on that of q, which keeps the dropped part [D_r, -B; -B', D_c] diagonally
// dominant, so the assembled matrix stays positive definite and at least M + K
static void effCholFill(const mjModel* m, mjData* d, const mjEffChol* c, mjtNum* status,
                        mjtNum* values, int prefilled) {
  int nb = c->nb, n = c->n, nL = c->nL, nv = m->nv;
  mjtNum* H = values;
  *status = 0;
  mj_markStack(d);
  if (!prefilled) {
    mju_zero(H, nL);
  }
  mjtNum* scratch = mjSTACKALLOC(d, 3*n, mjtNum);
  mju_zero(scratch, 3*n);
  int* blk = effInts(d, nv);
  int* sc = effInts(d, 4*(size_t)nb);
  int *slot = sc, *qmark = sc + nb, *qlist = sc + 2*nb, *qpos = sc + 3*nb;
  effCovered(nv, nb, c->dofid, blk);
  for (int k=0; k < nb; k++) slot[k] = qmark[k] = -1;

  // under IPC, a K block is exact only on an allowed pair: exact fill entries make its barrier
  // solve take more CG iterations
  int* amark = NULL;
  if (mjENABLED(mjENBL_IPC)) {
    amark = effInts(d, nb);
    for (int k=0; k < nb; k++) amark[k] = -1;
  }

  // the three rows of a covered triple share their column pattern, with a 3x3 block per
  // neighbour (mj_effMulAdd)
  const int* K_rowadr = d->efm_K_rowadr;
  for (int k=0; k < nb; k++) {
    int p = c->perm[k], i0 = c->dofid[k];
    const int* vc = c->L_colind + c->L_rowadr[3*p];
    int nvc = (c->L_rownnz[3*p] - 1) / 3, nq = 0;
    for (int i=0; i < nvc; i++) slot[vc[3*i]/3] = i;
    if (amark) effAllow(amark, c->eadr, c->enbr, k);
    int ra[3], da[3];
    const mjtNum* Kv[3];
    for (int a=0; a < 3; a++) {
      ra[a] = c->L_rowadr[3*p+a];
      da[a] = ra[a] + 3*nvc;
      Kv[a] = d->efm_K_val + K_rowadr[i0+a];
    }
    const int* ci = d->efm_K_colind + K_rowadr[i0];
    int nn = d->efm_K_rownnz[i0] / 3;
    for (int j=0; j < nn; j++) {
      int q = blk[ci[3*j]];
      if (q < 0) continue;
      if (q == k) {
        for (int a=0; a < 3; a++) {
          for (int b=0; b <= a; b++) H[da[a]+b] += Kv[a][3*j+b];
        }
        continue;
      }

      // each pair is taken from the row of the vertex eliminated first (the larger perm)
      int pq = c->perm[q];
      if (pq > p) continue;
      int s = slot[pq];
      if (amark && !effAllowed(amark, c->ne, k, q)) s = -1;
      if (s >= 0) {
        for (int a=0; a < 3; a++) {
          mjtNum* h = H + ra[a] + 3*s;
          h[0] += Kv[a][3*j];
          h[1] += Kv[a][3*j+1];
          h[2] += Kv[a][3*j+2];
        }
      } else {
        if (qmark[q] != k) {
          qmark[q] = k;
          qlist[nq] = q;
          qpos[q] = nq++;
        }
        mjtNum* acc = scratch + 6*qpos[q];
        for (int a=0; a < 3; a++) {
          for (int b=0; b < 3; b++) {
            mjtNum av = mju_abs(Kv[a][3*j+b]);
            acc[a] += av;
            acc[3+b] += av;
          }
        }
      }
    }
    for (int a=0; a < 3; a++) {
      int i = i0 + a;
      for (int j=m->M_rowadr[i]; j < m->M_rowadr[i] + m->M_rownnz[i]; j++) {
        int col = m->M_colind[j];
        if (col >= i0) H[da[a]+col-i0] += d->M[j];
      }
    }
    for (int i=0; i < nvc; i++) slot[vc[3*i]/3] = -1;
    for (int idx=0; idx < nq; idx++) {
      int pk = 3*p, pq = 3*c->perm[qlist[idx]];
      mjtNum* acc = scratch + 6*idx;
      for (int a=0; a < 3; a++) {
        H[c->L_rowadr[pk+a] + c->L_rownnz[pk+a] - 1] += acc[a];
        H[c->L_rowadr[pq+a] + c->L_rownnz[pq+a] - 1] += acc[3+a];
      }
    }
    mju_zero(scratch, 6*nq);
  }
  mjtNum maxdiag = 0;
  for (int r=0; r < n; r++) {
    maxdiag = mju_max(maxdiag, H[c->L_rowadr[r]+c->L_rownnz[r]-1]);
  }
  if (effCholNumericBlocked(c, H, mju_max(mjMINVAL, EFF_CHOL_PIVOT*maxdiag), scratch) == n) {
    *status = 1;
  }
  mj_freeStack(d);
}

static void effCholEnsure(const mjModel* m, mjData* d) {
  int nb = d->nefmdof;
  if (!effCholReserved(d) || d->efm_L[9*nb] != EFF_CHOL_PENDING) {
    return;
  }
  d->efm_L[9*nb] = 0;
  mjEffChol c;
  effCholStep(d, &c);
  if (mj_stackBytesAvailable(d) >= effCholFillBytes(&c, m->nv)) {
    effCholFill(m, d, &c, d->efm_L + 9*nb, effCholStepValues(d, &c), /*prefilled=*/0);
  }
}

static void effCholSolveBlocked(mjtNum* res, const mjtNum* L, const mjtNum* vec,
                                const mjEffChol* c) {
  int nb = c->nb;
  const int *rowadr = c->L_rowadr, *rownnz = c->L_rownnz, *colind = c->L_colind;
  mju_copy(res, vec, 3*nb);
  for (int p=nb-1; p >= 0; p--) {
    int nob = rownnz[3*p] - 1;
    const mjtNum *L0 = L + rowadr[3*p], *L1 = L + rowadr[3*p+1], *L2 = L + rowadr[3*p+2];
    mjtNum* r = res + 3*p;
    mjtNum x2 = r[2] / L2[nob+2];
    mjtNum x1 = (r[1] - L2[nob+1]*x2) / L1[nob+1];
    mjtNum x0 = (r[0] - L2[nob]*x2 - L1[nob]*x1) / L0[nob];
    r[0] = x0;
    r[1] = x1;
    r[2] = x2;
    const int* ci = colind + rowadr[3*p];
    for (int j=0; j < nob; j += 3) {
      mjtNum* rq = res + ci[j];
      rq[0] -= L0[j]*x0 + L1[j]*x1 + L2[j]*x2;
      rq[1] -= L0[j+1]*x0 + L1[j+1]*x1 + L2[j+1]*x2;
      rq[2] -= L0[j+2]*x0 + L1[j+2]*x1 + L2[j+2]*x2;
    }
  }
  for (int p=0; p < nb; p++) {
    int nob = rownnz[3*p] - 1;
    const mjtNum *L0 = L + rowadr[3*p], *L1 = L + rowadr[3*p+1], *L2 = L + rowadr[3*p+2];
    const int* ci = colind + rowadr[3*p];
    mjtNum s0 = 0, s1 = 0, s2 = 0;
    for (int j=0; j < nob; j += 3) {
      const mjtNum* rq = res + ci[j];
      mjtNum y0 = rq[0], y1 = rq[1], y2 = rq[2];
      s0 += L0[j]*y0 + L0[j+1]*y1 + L0[j+2]*y2;
      s1 += L1[j]*y0 + L1[j+1]*y1 + L1[j+2]*y2;
      s2 += L2[j]*y0 + L2[j+1]*y1 + L2[j+2]*y2;
    }
    mjtNum* r = res + 3*p;
    mjtNum x0 = (r[0] - s0) / L0[nob];
    mjtNum x1 = (r[1] - s1 - L1[nob]*x0) / L1[nob+1];
    mjtNum x2 = (r[2] - s2 - L2[nob]*x0 - L2[nob+1]*x1) / L2[nob+2];
    r[0] = x0;
    r[1] = x1;
    r[2] = x2;
  }
}

static void effCholApply(mjData* d, const mjEffChol* c, mjtNum* x, const mjtNum* rhs,
                         const mjtNum* Ls) {
  int nb = c->nb, n = c->n;
  mj_markStack(d);
  mjtNum* bc = mjSTACKALLOC(d, n, mjtNum);
  mjtNum* xc = mjSTACKALLOC(d, n, mjtNum);
  for (int k=0; k < nb; k++) {
    int i = c->dofid[k], p = 3*c->perm[k];
    bc[p] = rhs[i];
    bc[p+1] = rhs[i+1];
    bc[p+2] = rhs[i+2];
  }
  effCholSolveBlocked(xc, Ls, bc, c);
  for (int k=0; k < nb; k++) {
    int i = c->dofid[k], p = 3*c->perm[k];
    x[i] = xc[p];
    x[i+1] = xc[p+1];
    x[i+2] = xc[p+2];
  }
  mj_freeStack(d);
}

enum { EFF_CON_COUNT, EFF_CON_COLLECT, EFF_CON_FOLD };

typedef struct {
  int op;
  const mjEffChol* c;
  const int* blk;
  int *slot, *vk, *pair;
  mjtNum *v, *H;
  size_t npair, maxpair, ntotal, ndiag;
} mjEffConFold;

static int effConPair(const mjEffChol* c, int k, int q, int* pos) {
  int pk = c->perm[k], pq = c->perm[q];
  int lo_p = pk > pq ? pk : pq, hi_p = pk > pq ? pq : pk;
  int j = effCholFind(c->L_rownnz, c->L_rowadr, c->L_colind, 3*lo_p, 3*hi_p);
  if (j < 0) {
    return 0;
  }
  int off = j - c->L_rowadr[3*lo_p];
  for (int a=0; a < 3; a++) {
    for (int b=0; b < 3; b++) {
      int s = pk > pq ? a : b, t = pk > pq ? b : a;
      pos[3*a+b] = c->L_rowadr[3*lo_p+s] + off + t;
    }
  }
  return 1;
}

static void effConTerm(mjEffConFold* f, mjtNum s, int nnz, const int* colind,
                       const mjtNum* val) {
  if (s <= 0) {
    return;
  }
  const mjEffChol* c = f->c;
  int ns = 0;
  for (int a=0; a < nnz; a++) {
    int i = colind ? colind[a] : a;
    int k = f->blk[i];
    if (k < 0 || !val[a]) continue;
    int sl = f->slot[k];
    if (sl < 0) {
      sl = f->slot[k] = ns;
      f->vk[ns++] = k;
      mju_zero3(f->v + 3*sl);
    }
    f->v[3*sl+i-c->dofid[k]] += val[a];
  }
  f->ndiag += ns;
  if (f->op == EFF_CON_COUNT) {
    f->ntotal += (size_t)ns * (ns - 1) / 2;
    for (int i=0; i < ns; i++) f->slot[f->vk[i]] = -1;
    return;
  }
  if (f->op == EFF_CON_FOLD) {
    for (int i=0; i < ns; i++) {
      int pk = 3*c->perm[f->vk[i]];
      const mjtNum* vk = f->v + 3*i;
      for (int a=0; a < 3; a++) {
        int dk = c->L_rowadr[pk+a] + c->L_rownnz[pk+a] - 1 - a;
        for (int b=0; b <= a; b++) {
          f->H[dk+b] += s * vk[a] * vk[b];
        }
      }
    }
  }
  int pos[9];
  for (int i=0; i < ns; i++) {
    for (int j=i+1; j < ns; j++) {
      int k = f->vk[i], q = f->vk[j];
      if (f->op == EFF_CON_COLLECT) {
        if (f->npair < f->maxpair) {
          f->pair[2*f->npair] = k < q ? k : q;
          f->pair[2*f->npair+1] = k < q ? q : k;
          f->npair++;
        }
      } else {
        const mjtNum *vk = f->v + 3*i, *vq = f->v + 3*j;
        if (effConPair(c, k, q, pos)) {
          for (int a=0; a < 3; a++) {
            for (int b=0; b < 3; b++) f->H[pos[3*a+b]] += s * vk[a] * vq[b];
          }
        } else {
          int pk = 3*c->perm[k], pq = 3*c->perm[q];
          for (int a=0; a < 3; a++) {
            int dk = c->L_rowadr[pk+a] + c->L_rownnz[pk+a] - 1 - a;
            int dq = c->L_rowadr[pq+a] + c->L_rownnz[pq+a] - 1 - a;
            for (int b=0; b <= a; b++) {
              f->H[dk+b] += s * vk[a] * vk[b];
              f->H[dq+b] += s * vq[a] * vq[b];
            }
          }
        }
      }
    }
  }
  for (int i=0; i < ns; i++) f->slot[f->vk[i]] = -1;
}

static void effConTerms(const mjModel* m, const mjData* d, mjEffConFold* f, int op,
                        const mjEffConRows* rows) {
  f->op = op;
  mjEffRank1Iter it = {.cls = op == EFF_CON_FOLD ? 0 : 2};
  for (int t=0; op != EFF_CON_FOLD && t < d->nefmT; t++) {
    int id = d->efm_tid[t], adr = m->ten_J_rowadr[id];
    effConTerm(f, 1, m->ten_J_rownnz[id], m->ten_J_colind + adr, d->ten_J + adr);
  }
  for (int i=0; op != EFF_CON_FOLD && d->efm_aid && i < m->nactuator; i++) {
    if (!mj_effActuatorPossible(m, i)) continue;
    for (int k=0; k < m->actuator_outnum[i]; k++) {
      int r = m->actuator_outadr[i] + k, adr = d->moment_rowadr[r];
      effConTerm(f, 1, d->moment_rownnz[r], d->moment_colind + adr, d->actuator_moment + adr);
    }
  }
  mjEffRank1 e;
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
    effConTerm(f, e.scale, e.nnz, e.colind, e.val);
  }
  for (int r=0; r < rows->nefc; r++) {
    mjtNum s = rows->D ? rows->D[r] : 1;
    if (!s) continue;
    if (rows->is_sparse) {
      int adr = rows->rowadr[r];
      effConTerm(f, s, rows->rownnz[r], rows->colind + adr, rows->J + adr);
    } else {
      effConTerm(f, s, m->nv, NULL, rows->J + (size_t)r*m->nv);
    }
  }
}

static void effConScratch(mjData* d, int nb, mjEffConFold* f) {
  f->slot = effInts(d, nb);
  f->vk = effInts(d, nb);
  f->v = mjSTACKALLOC(d, 3*nb, mjtNum);
  for (int k=0; k < nb; k++) f->slot[k] = -1;
}

// fold the efc rows and the rank-1 classes into the solve's copy of the sparse factor, on the
// step's analysis: pairs off its pattern are complemented on the diagonal (effConTerm)
void mj_effCholFoldFactor(const mjModel* m, mjData* d, const mjEffFold* fold, int nefc,
                          const mjtNum* efc_D, int is_sparse, const mjtNum* J,
                          const int* J_rownnz, const int* J_rowadr, const int* J_colind) {
  if (!fold || !fold->F) {
    return;
  }
  fold->F[0] = 0;
  if (!effCholReserved(d)) {
    return;
  }
  int nb = d->nefmdof, nv = m->nv;
  mjEffChol cs;
  effCholStep(d, &cs);
  if (mj_stackBytesAvailable(d) < mj_stackFrameBytes() + effIntBytes(nv) +
                                  effConScratchBytes(nb) + effCholFillBytes(&cs, nv)) {
    return;
  }
  mjEffConRows rows = {nefc, efc_D, is_sparse, J, J_rownnz, J_rowadr, J_colind};
  mj_markStack(d);
  int* blk = effInts(d, nv);
  effCovered(nv, nb, cs.dofid, blk);
  mjEffConFold con = {.c = &cs, .blk = blk, .H = fold->F + EFF_HDR};
  effConScratch(d, nb, &con);
  mju_zero(con.H, cs.nL);
  effConTerms(m, d, &con, EFF_CON_FOLD, &rows);

  // nothing folded: the copy is the metric's factor
  if (!con.ndiag) {
    fold->F[0] = EFF_CHOL_PENDING;
  } else {
    effCholFill(m, d, &cs, fold->F, con.H, /*prefilled=*/1);
  }
  mj_freeStack(d);
}

// the vertex pairs among the covered blocks that the terms a CG fold takes couple (effConTerms:
// the rank-1 classes and the efc rows, as mj_makeConstraint left them), with repeats, on the
// stack of d in the caller's frame. Returns their number, -1 when the stack cannot hold them
static int effCholConPairs(const mjModel* m, mjData* d, int nb, const int* dofid, const int* blk,
                           int** pairs) {
  *pairs = NULL;
  size_t frame = mj_stackFrameBytes(), con_bytes = effConScratchBytes(nb);
  if (mj_stackBytesAvailable(d) < frame + con_bytes) {
    return -1;
  }
  mjEffChol c0 = {.nb = nb, .dofid = dofid};
  mjEffConRows rows = {d->nefc, NULL, mj_isSparse(m), d->efc_J, d->efc_J_rownnz,
                       d->efc_J_rowadr, d->efc_J_colind};
  mjEffConFold con = {.c = &c0, .blk = blk};
  mj_markStack(d);
  effConScratch(d, nb, &con);
  effConTerms(m, d, &con, EFF_CON_COUNT, &rows);
  mj_freeStack(d);
  size_t ntotal = con.ntotal;
  if (!ntotal) {
    return 0;
  }
  if (ntotal > INT_MAX/2 || mj_stackBytesAvailable(d) < effIntBytes(2*ntotal) + frame + con_bytes) {
    return -1;
  }
  int* pair = effInts(d, 2*ntotal);
  mj_markStack(d);
  effConScratch(d, nb, &con);
  con.pair = pair;
  con.maxpair = ntotal;
  con.npair = 0;
  effConTerms(m, d, &con, EFF_CON_COLLECT, &rows);
  mj_freeStack(d);
  *pairs = pair;
  return (int)con.npair;
}

// analyse the step's efm_K pattern and allocate d->efm_L with the analysis and the factor tail
static mjtNum* effCholReserve(const mjModel* m, mjData* d, int nb, const int* dofid, int* ntail) {
  *ntail = 0;
  if (!effCholApplies(m, nb, dofid)) {
    return NULL;
  }
  int nv = m->nv;
  size_t kcap = effCholKcap(d, nb, dofid);
  size_t blocks_bytes = mj_stackBytes(sizeof(mjtNum)*9*nb, _Alignof(mjtNum));
  size_t avail = mj_stackBytesAvailable(d);
  size_t sym_bytes = mj_stackFrameBytes() + effIntBytes(nv) + effIntBytes(nb) +
                     effCholAnalysisBytes(m, nb, kcap, 0);
  if (avail < blocks_bytes + sym_bytes) {
    mj_warning(d, mjWARN_CNSTRFULL, d->narena);
    return NULL;
  }
  mj_markStack(d);
  int* blk = effInts(d, nv);
  int* perm = effInts(d, nb);
  effCovered(nv, nb, dofid, blk);
  int pos = 0;
  for (int t=0; t < m->nefmCvert && m->efmC_perm[t] >= 0; t++) {
    int k = m->efmC_perm[t] < nv ? blk[m->efmC_perm[t]] : -1;
    if (k >= 0 && dofid[k] == m->efmC_perm[t]) {
      perm[k] = pos++;
    }
  }

  // when the constraint solver can fold its efc rows into the factor (CG, IPC), the pattern also
  // takes the vertex pairs they couple, reordered by minimum degree when some are not flex pairs:
  // one analysis serves the metric's factor and the fold's (mj_effCholFoldFactor)
  int* xpair = NULL;
  int nx = 0;
  if (pos == nb && (effFoldPossible(m, m->opt.solver) || mjENABLED(mjENBL_IPC))) {
    nx = effCholConPairs(m, d, nb, dofid, blk, &xpair);
    if (nx < 0 || mj_stackBytesAvailable(d) < effCholAnalysisBytes(m, nb, kcap, nx)) {
      nx = 0;
      xpair = NULL;
    }
  }
  mjEffCholPass pass;
  int ok = (pos == nb) &&
           effCholBegin(m, d, nb, dofid, blk, d->efm_K_rownnz, d->efm_K_rowadr, d->efm_K_colind,
                        xpair, nx, /*order=*/2, perm, &pass) &&
           pass.nfe;
  if (!ok) {
    mj_freeStack(d);
    return NULL;
  }
  size_t npack = effPackNums(nb, pass.nL, pass.ne);
  if (avail < blocks_bytes + effCholReserveBytes(nb, nv, pass.nL, npack)) {
    mj_freeStack(d);
    mj_warning(d, mjWARN_CNSTRFULL, d->narena);
    return NULL;
  }
  int tail = EFF_HDR + (int)npack + pass.nL;
  if (mj_stackBytesAvailable(d) < effNumBytes(9*(size_t)nb + tail)) {
    mj_freeStack(d);
    mj_warning(d, mjWARN_CNSTRFULL, d->narena);
    return NULL;
  }
  mjtNum* B = (mjtNum*)effAlloc(d, sizeof(mjtNum)*(9*(size_t)nb + tail), _Alignof(mjtNum));
  if (!B) {
    mj_freeStack(d);
    return NULL;
  }
  B[9*nb] = EFF_CHOL_PENDING;
  int* A = (int*)(B + 9*nb + EFF_HDR);
  A[0] = pass.nL;
  A[1] = pass.ne;
  mjEffChol c;
  effPackLayout(A, nb, dofid, &c);
  memcpy(c.perm, perm, sizeof(int)*nb);
  effCholFinish(&pass, &c);
  mj_freeStack(d);
  *ntail = tail;
  return B;
}

static int effCholSolve(const mjModel* m, mjData* d, const mjEffFold* fold, mjtNum* x,
                        const mjtNum* b) {
  int nb = d->nefmdof;
  if (!effCholReserved(d)) {
    return 0;
  }
  mjEffChol cs;
  const mjtNum* Ls = NULL;
  if (fold) {
    if (fold->F && fold->F[0] == 1) {
      effCholStep(d, &cs);
      Ls = fold->F + EFF_HDR;
    } else if (!fold->F || fold->F[0] != EFF_CHOL_PENDING) {
      return 0;
    }
  }
  if (!Ls) {
    effCholEnsure(m, d);
    if (d->efm_L[9*nb] != 1) {
      return 0;
    }
    effCholStep(d, &cs);
    Ls = effCholStepValues(d, &cs);
  }
  effCholApply(d, &cs, x, b, Ls);
  return 1;
}

int mj_effCholFoldSize(const mjData* d) {
  if (!effCholReserved(d)) {
    return 0;
  }
  mjEffChol c;
  effCholStep(d, &c);
  return EFF_HDR + c.nL;
}

int mj_effFoldMonolithic(const mjModel* m, const mjData* d, int solver) {
  return d->efm_active && effFoldPossible(m, solver) && effCholReserved(d);
}

static size_t effCholFoldScratch(const mjModel* m, const mjData* d) {
  if (!effCholReserved(d)) {
    return 0;
  }
  int nb = d->nefmdof, nv = m->nv;
  return mj_stackFrameBytes() + effIntBytes(nv) + effConScratchBytes(nb) + effFillBytes(nb, nv);
}

static size_t effCholApplyScratch(const mjModel* m, const mjData* d) {
  if (!effCholReserved(d)) {
    return 0;
  }
  mjEffChol c;
  effCholStep(d, &c);
  return mjMAX(mj_stackFrameBytes() + 2*effNumBytes(c.n), effCholFillBytes(&c, m->nv));
}


// Build and factor the per-vertex 3x3 diagonal blocks of the flex part of (M + K), stored in
// d->efm_L, 9 numbers per covered vertex: O(n) to build and apply, approximate where the sparse
// factorization it replaces was exact. Both consumers use the blocks as a preconditioner: the CG
// constraint solver (Mgrad = Mtilde \ grad) and the qacc_smooth PCG in mj_effSolve, which
// supplies the accuracy. For 1D rods and 2D flexes the buffer can also hold a sparse factor of the
// covered block, which then replaces the blocks (effCholReserve).
static void effBlocks(const mjModel* m, mjData* d) {
  int nv = m->nv;

  // covered dofs come in contiguous triples (the 3 slide dofs of one flex point), but the first
  // one need not be at a multiple of 3: any joint declared before the flex shifts them. Walk the
  // covered rows rather than striding the dof index, which would straddle point boundaries.
  int nb = effCoveredBlocks(nv, d->efm_K_rownnz, NULL);
  d->nefmdof = 0;
  int* adr = (int*) effAlloc(d, sizeof(int)*(nb > 0 ? nb : 1), _Alignof(int));
  effCoveredBlocks(nv, d->efm_K_rownnz, adr);
  int ntail = 0;
  mjtNum* B = effCholReserve(m, d, nb, adr, &ntail);
  if (!B) {
    if (d->nisland > 0 && d->nefc > 0 && m->nflex > 0 && !mjENABLED(mjENBL_IPC)) {
      ntail = effIslandTail(m, d, nb);
    }
    size_t nB = 9*(size_t)nb + (size_t)ntail;
    B = (mjtNum*) effAlloc(d, sizeof(mjtNum)*(nB > 0 ? nB : 1), _Alignof(mjtNum));
    if (ntail) {
      B[9*nb] = EFF_ISLAND_TAG;
    }
  }
  for (int k = 0; k < nb; k++) {
    mjtNum* Bk = B + 9*k;
    effBlockRaw(m, d, adr[k], Bk);
    mju_cholFactor(Bk, 3, mjMINVAL);
  }
  d->efm_L = B;
  d->efm_dofid = adr;
  d->nefmdof = nb;
  d->nefmL = 9*nb + ntail;
}

// Apply the metric preconditioner: the per-step 3x3 blocks when they exist, else the constant
// bending factor from mj_setConst, on the dofs they cover; M^-1 on all other dofs. PCG requires
// symmetry, so covered and uncovered dofs must not see each other: zeroing the covered entries
// of the right-hand side before the qLD sweep keeps the uncovered rows from reading them.
// x = (L L') \ b for one 3x3 factor from mju_cholFactor, in mju_cholSolve's arithmetic; b may
// alias x
static inline void chol3Solve(mjtNum* x, const mjtNum* L, const mjtNum* b) {
  mjtNum r0 = b[0] / L[0];
  mjtNum r1 = (b[1] - L[3]*r0) / L[4];
  mjtNum r2 = (b[2] - (L[6]*r0 + L[7]*r1)) / L[8];
  r2 /= L[8];
  r1 = (r1 - L[7]*r2) / L[4];
  r0 -= L[3]*r1;
  r0 -= L[6]*r2;
  r0 /= L[0];
  x[0] = r0;
  x[1] = r1;
  x[2] = r2;
}

// x[U] = S \ bu on each component of the fold's dense factors, bu = b[U] gathered by the
// caller and overwritten
static void effDenseApply(const mjEffFold* fold, mjtNum* x, mjtNum* bu) {
  for (int c = 0; c < fold->ncomp; c++) {
    int adr = fold->Uadr[c], n = fold->Uadr[c+1] - adr;
    mju_cholSolve(bu + adr, fold->S + fold->Sadr[c], bu + adr, n);
  }
  for (int j = 0; j < fold->nu; j++) {
    x[fold->U[j]] = bu[j];
  }
}

// x = (M + K)_covered \ b on the covered dofs: the sparse factor of fold (or of the metric, when
// fold is NULL) if valid, else the 3x3 blocks L; b may alias x
static void effCovSolve(const mjModel* m, mjData* d, const mjtNum* L, const mjEffFold* fold,
                        mjtNum* x, const mjtNum* b) {
  if (effCholSolve(m, d, fold, x, b)) {
    return;
  }
  for (int k = 0; k < d->nefmdof; k++) {
    int i = d->efm_dofid[k];
    chol3Solve(x + i, L + 9*k, b + i);
  }
}

// fold: the solver's folded copy (mj_effPrecFold) with L its blocks, or NULL
static void effBlockApply(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b,
                          const mjtNum* L, const mjEffFold* fold) {
  int nv = m->nv;
  int nbd = m->nefm0dof;
  int flg_bend = nbd && !d->nefmdof;
  int flg_dense = fold && fold->S_valid;

  // the fold's dense factors on all the uncovered dofs, the sparse factor or the 3x3 blocks on the
  // rest: no backbone
  if (flg_dense && !fold->partial) {
    mj_markStack(d);
    mjtNum* bu = mjSTACKALLOC(d, fold->nu, mjtNum);
    for (int j = 0; j < fold->nu; j++) {
      bu[j] = b[fold->U[j]];   // before the covered solve writes x: b may alias x
    }
    effCovSolve(m, d, L, fold, x, b);
    effDenseApply(fold, x, bu);
    mj_freeStack(d);
    return;
  }

  // every dof a covered triple: the blocks are the whole preconditioner
  if (3*d->nefmdof == nv && !flg_bend) {
    effCovSolve(m, d, L, fold, x, b);
    return;
  }
  int need_stack = (x == b) || flg_dense || flg_bend;
  if (need_stack) {
    mj_markStack(d);
  }
  const mjtNum* rhs = b;
  if (x == b) {
    mjtNum* buf = mjSTACKALLOC(d, nv, mjtNum);
    mju_copy(buf, b, nv);   // b aliases x, which the sweep below overwrites
    rhs = buf;
  }

  // dofs no factor covers: backbone solve, using the qH backbone factor when diagonal
  // terms exist, else qLD
  mju_copy(x, rhs, nv);
  for (int k = 0; k < d->nefmdof; k++) {
    mju_zero(x + d->efm_dofid[k], 3);
  }
  if (flg_bend) {
    for (int i = 0; i < nbd; i++) {
      x[m->efm0_dofid[i]] = 0;
    }
  }
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < nv;
  if (d->efm_diag) {
    mj_solveLD(x, d->qH, d->qHDiagInv, sleep_filter ? d->nv_awake : nv, 1,
               m->M_rownnz, m->M_rowadr, m->M_colind, sleep_filter ? d->dof_awake_ind : NULL);
  } else {
    mj_solveLD(x, d->qLD, d->qLDiagInv, sleep_filter ? d->nv_awake : nv, 1,
               m->M_rownnz, m->M_rowadr, m->M_colind, sleep_filter ? d->dof_awake_ind : NULL);
  }

  // per-step stiffness: the sparse factor or the 3x3 blocks
  effCovSolve(m, d, L, fold, x, rhs);

  // the fold's dense factors where some uncovered dofs are in no component: each component is
  // a union of whole trees, which the backbone solve keeps apart, so overwriting its dofs
  // leaves the backbone solve on the other trees as it was
  if (flg_dense) {
    mjtNum* bu = mjSTACKALLOC(d, fold->nu, mjtNum);
    for (int j = 0; j < fold->nu; j++) {
      bu[j] = rhs[fold->U[j]];
    }
    effDenseApply(fold, x, bu);
  }

  // bending-only: exact (M + K_bend)^-1 on the dofs the constant factor covers
  if (flg_bend) {
    mjtNum* bfr = mjSTACKALLOC(d, nbd, mjtNum);
    mjtNum* bfz = mjSTACKALLOC(d, nbd, mjtNum);
    for (int i = 0; i < nbd; i++) {
      bfr[i] = rhs[m->efm0_dofid[i]];
    }
    mju_cholSolveSparse(bfz, m->efm0_L, bfr, nbd,
                        m->efm0_L_rownnz, m->efm0_L_rowadr, m->efm0_L_colind);
    for (int i = 0; i < nbd; i++) {
      x[m->efm0_dofid[i]] = bfz[i];
    }
  }
  if (need_stack) {
    mj_freeStack(d);
  }
}


// accurate solve of (M + K) x = b by PCG with the 3x3 block preconditioner, converging the
// relative residual to opt.tolerance; used for qacc_smooth. Reaching opt.iterations means the
// metric is too ill-conditioned for the blocks: warn (mjWARN_INERTIA, worst-residual dof) and
// return x under-converged.
void mj_effSolve(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b) {
  if (!d->efm_active) {
    mj_effPrec(m, d, x, b);
    return;
  }

  // backbone-only metric (no tendon, actuator or flex couplings): the qH solve inside the
  // block-apply is the exact metric solve, skip the iteration. Beyond the saved iterations,
  // the direct solve is block-decoupled across trees, which sleeping trees rely on
  if (!d->nefmT && !d->nefmA && !mj_effFlexAny(m)) {
    effBlockApply(m, d, x, b, d->efm_L, NULL);
    return;
  }
  int nv = m->nv;
  mj_markStack(d);
  mjtNum* r = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* z = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* p = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* Ap = mjSTACKALLOC(d, nv, mjtNum);
  mju_copy(r, b, nv);   // before zeroing x: b may alias x
  mju_zero(x, nv);
  mjtNum bn = mju_dot(r, r, nv);
  if (bn > mjMINVAL) {
    // converge the relative residual to opt.tolerance; both sides are squared norms
#ifdef mjUSESINGLE
    // float cannot reach a 1e-8 relative residual (eps ~1.2e-7): without a floor every step of
    // every covered model would run to opt.iterations and then warn.
    mjtNum tolerance = mju_max(m->opt.tolerance, 1e-5);
#else
    mjtNum tolerance = m->opt.tolerance;
#endif
    mjtNum tol = tolerance*tolerance*bn;
    int capped = 1;   // cleared by either exit below; still set means the cap was reached
    effBlockApply(m, d, z, r, d->efm_L, NULL);
    mju_copy(p, z, nv);
    mjtNum rz = mju_dot(r, z, nv);
    for (int it = 0; it < m->opt.iterations; it++) {
      mju_mulSymVecSparse(Ap, d->M, p, nv, m->M_rownnz, m->M_rowadr, m->M_colind);
      mj_effMulAdd(m, d, Ap, p, /*flg_contact=*/1);
      mjtNum pAp = mju_dot(p, Ap, nv);
      // curvature breakdown: the metric has no curvature along p, so no further progress is
      // possible and x is the best available. Not a budget failure, so it does not warn.
      if (pAp <= 0) { capped = 0; break; }
      mjtNum alpha = rz/pAp;
      mju_addToScl(x, p, alpha, nv);
      mju_addToScl(r, Ap, -alpha, nv);
      if (mju_dot(r, r, nv) < tol) { capped = 0; break; }
      effBlockApply(m, d, z, r, d->efm_L, NULL);
      mjtNum rznew = mju_dot(r, z, nv);
      mju_addScl(p, z, p, rznew/rz, nv);
      rz = rznew;
    }

    // Ran out of iterations with the residual still above tolerance: the metric is too
    // ill-conditioned for the blocks to solve within the budget. Blame the dof carrying the
    // largest residual.
    // mjWARN_INERTIA is the closest existing warning (reusing it avoids an ABI addition), but on
    // its own it points the user at their inertia when the cause is the flex stiffness, so say so
    // first. Gate on the same first-time condition mj_warning uses, or a model that fails every
    // step would print this thousands of times a second.
    if (capped && mju_dot(r, r, nv) >= tol) {
      int worst = 0;
      for (int i = 1; i < nv; i++) {
        if (mju_abs(r[i]) > mju_abs(r[worst])) worst = i;
      }
      if (!d->warning[mjWARN_INERTIA].number) {
        mju_warning("Flex stiffness is too ill-conditioned for the effective-metric block "
                    "preconditioner: the M+K solve ran out of iterations at a relative residual "
                    "of %.2e and qacc_smooth is under-converged. Reported as a singular inertia "
                    "below, because M+K is the effective inertia.",
                    mju_sqrt(mju_dot(r, r, nv)/bn));
      }
      mj_warning(d, mjWARN_INERTIA, worst);
    }
  }
  mj_freeStack(d);
}

void mj_effPrec(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b) {
  int nv = m->nv;

  // active metric: the prefactored 3x3 blocks are the preconditioner
  if (d->efm_active) {
    effBlockApply(m, d, x, b, d->efm_L, NULL);
    return;
  }

  // inactive metric: x = M \ b, which is exact
  if (x != b) {
    mju_copy(x, b, nv);
  }
  mj_solveLD(x, d->qLD, d->qLDiagInv, nv, 1, m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
}


//------------- dense uncovered-dof blocks of the CG preconditioner --------------------------------
// The 3x3 blocks precondition only the flex vertex triples; every other dof (the articulated
// trees, free bodies) gets the backbone solve of M + diag alone, without the rank-1 classes and
// the solve's efc rows (dof friction, limits, contacts on the trees), and those stiff rows then
// set the CG iteration count. The solver's fold therefore also carries dense factors of the
// uncovered block
//   S = M_uu + diag + fluid + rank-1 classes + J_u' D J_u   (rows with D > 0),
// applied as block Jacobi next to the 3x3 blocks; S is SPD whenever the unfolded matrices are.
// S is block diagonal over its connected components: M and the fluid blocks stay within a
// tree, so only the efc rows and the rank-1 terms join trees. Each component is factored on its
// own, which is exact for S at sum(n_c^3)/6 multiply-adds per solve, and the dofs of the
// components over the cap keep the backbone solve. The metric's own preconditioner has none:
// the qacc_smooth PCG keeps the backbone solve.

// largest component of the dense blocks: its factorization per solve costs n^3/6 multiply-adds
// (36M here), which a solve that the block saves iterations on still recovers
#define EFF_DENSE_MAXNU 600

// union-find root of tree t, halving the path
static int effRoot(int* par, int t) {
  while (par[t] != t) {
    par[t] = par[par[t]];
    t = par[t];
  }
  return t;
}


// join the trees of the uncovered dofs (cov[i] < 0) among the n columns ind
static void effJoin(const mjModel* m, const int* cov, int* par, const int* ind, int n) {
  int r0 = -1;
  for (int a=0; a < n; a++) {
    if (cov[ind[a]] >= 0) {
      continue;
    }
    int r = effRoot(par, m->dof_treeid[ind[a]]);
    if (r0 < 0) {
      r0 = r;
    } else if (r != r0) {
      par[r] = r0;
    }
  }
}


// the connected components of S over the uncovered dofs (cov[i] < 0): their trees, joined by the
// efc rows with D > 0 and the rank-1 terms, kept when of at most EFF_DENSE_MAXNU dofs. Sets
// comp[t], the component of tree t (numbered in the order of their first dof), -1: no uncovered
// dofs or over the cap. Returns the components, with *nu their dofs, *nS the mjtNums of their
// packed factors and *partial set if uncovered dofs were left out. par, cnt: ntree ints scratch
static int effComponents(const mjModel* m, const mjData* d, const int* cov,
                         int nefc, const mjtNum* efc_D, const int* J_rownnz,
                         const int* J_rowadr, const int* J_colind,
                         int* comp, int* par, int* cnt, int* nu, int* nS, int* partial) {
  int nv = m->nv, ntree = m->ntree;
  for (int t=0; t < ntree; t++) {
    par[t] = t;
    cnt[t] = 0;
    comp[t] = -1;
  }
  for (int i=0; i < nv; i++) {
    if (cov[i] < 0) {
      cnt[m->dof_treeid[i]]++;
    }
  }

  // join the trees each rank-1 term and efc row couples
  mjEffRank1Iter it = {0};
  mjEffRank1 e;
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
    effJoin(m, cov, par, e.colind, e.nnz);
  }
  for (int r=0; r < nefc; r++) {
    if (efc_D[r] > 0) {
      effJoin(m, cov, par, J_colind + J_rowadr[r], J_rownnz[r]);
    }
  }

  // component sizes, at the roots
  for (int t=0; t < ntree; t++) {
    int r = effRoot(par, t);
    if (r != t) {
      cnt[r] += cnt[t];
    }
  }

  // number the components by their first dof, those within the cap (-2: over it)
  int ncomp = 0;
  *nu = *nS = *partial = 0;
  for (int i=0; i < nv; i++) {
    if (cov[i] >= 0) {
      continue;
    }
    int r = effRoot(par, m->dof_treeid[i]);
    if (comp[r] != -1) {
      continue;
    }
    int n = cnt[r];
    if (n <= EFF_DENSE_MAXNU && *nS <= INT_MAX - n*n) {
      comp[r] = ncomp++;
      *nu += n;
      *nS += n*n;
    } else {
      comp[r] = -2;
      *partial = 1;
    }
  }
  for (int t=0; t < ntree; t++) {
    int c = comp[effRoot(par, t)];
    comp[t] = c < 0 ? -1 : c;
  }
  return ncomp;
}


// size the dense blocks of a fold over the given efc rows, 0: none
int mj_effFoldDenseSize(const mjModel* m, mjData* d, int nefc, const mjtNum* efc_D,
                        int is_sparse, const int* J_rownnz, const int* J_rowadr,
                        const int* J_colind, int* nu, int* ncomp) {
  *nu = *ncomp = 0;
  if (!d->nefmdof || !is_sparse || 3*d->nefmdof == m->nv) {
    return 0;
  }
  int ntree = m->ntree, nS, partial;
  mj_markStack(d);
  int* cov = mjSTACKALLOC(d, m->nv, int);
  int* tint = mjSTACKALLOC(d, 3*ntree, int);
  effCovered(m->nv, d->nefmdof, d->efm_dofid, cov);
  *ncomp = effComponents(m, d, cov, nefc, efc_D, J_rownnz, J_rowadr, J_colind,
                         tint, tint + ntree, tint + 2*ntree, nu, &nS, &partial);
  mj_freeStack(d);
  return nS;
}


// stack bytes of mj_effPrecFold and mj_effPrecBlocks beyond the fold's arrays, at most: the
// fold's frame (block additions, covered marks and, with a dense block, the components'
// scratch, which also bounds mj_effFoldDenseSize) or the apply's (the backbone right-hand side
// and the dense right-hand side, or the bending-only factor's vectors)
size_t mj_effFoldScratch(const mjModel* m, const mjData* d, int nu) {
  size_t frame = mj_stackFrameBytes();
  size_t sn = sizeof(mjtNum), an = _Alignof(mjtNum), si = sizeof(int), ai = _Alignof(int);
  size_t fold = frame + mj_stackBytes(sn*9*d->nefmdof, an) + mj_stackBytes(si*m->nv, ai) +
                (nu ? mj_stackBytes(si*3*m->ntree, ai) : 0);
  fold = mjMAX(fold, effCholFoldScratch(m, d));
  size_t apply = frame + mj_stackBytes(sn*m->nv, an) + mj_stackBytes(sn*nu, an) +
                 effCholApplyScratch(m, d);
  if (m->nefm0dof && !d->nefmdof) {
    apply += 2*mj_stackBytes(sn*m->nefm0dof, an);
  }
  return mjMAX(fold, apply);
}


// stack bytes of mj_effMulAdd and mj_effMulAddIsland, at most: the frames of the matrix-free
// flex operators they fall back to, one at a time, plus the island scatter/gather frame
size_t mj_effMulAddScratch(const mjModel* m, const mjData* d) {
  size_t frame = mj_stackFrameBytes(), bytes = 0;
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_interp[f] || m->flex_rigid[f]) continue;
    if (m->flex_dim[f] == 1 && m->flex_bendingadr[f] < 0) continue;
    if (!d->nefmK || !mj_flexSimple(m, f)) {
      // bending and stretch: the gathered vector and result
      size_t b = frame + 2*mj_stackBytes(sizeof(mjtNum)*3*m->flex_vertnum[f], _Alignof(mjtNum));
      bytes = mjMAX(bytes, b);
    }
  }
  if (!d->nefmK || !mjd_flexInterpAssemblable(m)) {
    size_t b = mjd_flexInterp_mulBytes(m);
    bytes = mjMAX(bytes, b);
  }
  if (bytes && !mjDISABLED(mjDSBL_ISLAND) && !mjENABLED(mjENBL_IPC)) {
    bytes += frame + 2*mj_stackBytes(sizeof(mjtNum)*m->nv, _Alignof(mjtNum));
  }
  return bytes;
}


// assemble and factor the dense blocks of the components into fold->S; dof i is at loc[i]
// within its component comp[dof_treeid[i]], loc[i] = -1 outside the components. The efc rows
// are in the sparse Jacobian. A component holds every uncovered dof its rows and rank-1 terms
// reach (they joined its trees), and M and the fluid blocks stay within a tree
static void effDenseBlock(const mjModel* m, const mjData* d, mjEffFold* fold, const int* loc,
                          const int* comp, int nefc, const mjtNum* efc_D, const mjtNum* J,
                          const int* J_rownnz, const int* J_rowadr, const int* J_colind) {
  mju_zero(fold->S, fold->nS);

  // M_uu (lower rows, ancestors then the diagonal), the diagonal classes and the fluid blocks
  for (int c=0; c < fold->ncomp; c++) {
    int n = fold->Uadr[c+1] - fold->Uadr[c];
    mjtNum* S = fold->S + fold->Sadr[c];
    for (int j=0; j < n; j++) {
      int i = fold->U[fold->Uadr[c] + j];
      for (int a=m->M_rowadr[i]; a < m->M_rowadr[i] + m->M_rownnz[i]; a++) {
        int cu = loc[m->M_colind[a]];
        if (cu < 0) {
          continue;
        }
        mjtNum v = d->M[a] + (d->efm_fluid ? d->efm_fluid[a] : 0);
        S[j*n + cu] += v;
        if (cu != j) {
          S[cu*n + j] += v;
        }
      }
      if (d->efm_diag) {
        S[j*n + j] += d->efm_diag[i];
      }
    }
  }

  // rank-1 classes
  mjEffRank1Iter it = {0};
  mjEffRank1 e;
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
    for (int a=0; a < e.nnz; a++) {
      int ia = e.colind[a], ua = loc[ia];
      if (ua < 0) {
        continue;
      }
      int c = comp[m->dof_treeid[ia]], n = fold->Uadr[c+1] - fold->Uadr[c];
      mjtNum* S = fold->S + fold->Sadr[c];
      for (int b=0; b < e.nnz; b++) {
        int ib = e.colind[b], ub = loc[ib];
        if (ub >= 0 && comp[m->dof_treeid[ib]] == c) {
          S[ua*n + ub] += e.scale * e.val[a] * e.val[b];
        }
      }
    }
  }

  // efc rows: J_u' D J_u
  for (int r=0; r < nefc; r++) {
    mjtNum D = efc_D[r];
    if (D <= 0) {
      continue;
    }
    int adr = J_rowadr[r], nnz = J_rownnz[r];
    for (int a=0; a < nnz; a++) {
      int ia = J_colind[adr+a], ua = loc[ia];
      if (ua < 0) {
        continue;
      }
      int c = comp[m->dof_treeid[ia]], n = fold->Uadr[c+1] - fold->Uadr[c];
      mjtNum* S = fold->S + fold->Sadr[c];
      mjtNum DJa = D * J[adr+a];
      for (int b=0; b < nnz; b++) {
        int ib = J_colind[adr+b], ub = loc[ib];
        if (ub >= 0 && comp[m->dof_treeid[ib]] == c) {
          S[ua*n + ub] += DJa * J[adr+b];
        }
      }
    }
  }

  // factor each component
  int valid = 1;
  for (int c=0; c < fold->ncomp; c++) {
    int n = fold->Uadr[c+1] - fold->Uadr[c];
    valid = valid && mju_cholFactor(fold->S + fold->Sadr[c], n, mjMINVAL) == n;
  }
  fold->S_valid = valid;
}


// fold the metric's rank-1 classes and the efc rows (quadratic zone) into a copy of the
// preconditioner blocks, factored into fold->L (9*nefmdof); only each term's per-vertex 3x3
// diagonal survives, as for the elastic part. With fold->nu, also the dense blocks of the
// components of the uncovered dofs (effDenseBlock). Returns 0 when nothing is covered, leaving
// fold untouched. Stack: one frame (mj_effFoldScratch)
int mj_effPrecFold(const mjModel* m, mjData* d, mjEffFold* fold,
                   int nefc, const mjtNum* efc_D, int is_sparse,
                   const mjtNum* J, const int* J_rownnz, const int* J_rowadr,
                   const int* J_colind) {
  if (!d->nefmdof) {
    return 0;
  }
  mj_markStack(d);
  int nv = m->nv;
  mjtNum* L = fold->L;
  mjtNum* Badd = mjSTACKALLOC(d, 9*d->nefmdof, mjtNum);
  int* blk = mjSTACKALLOC(d, nv, int);
  mju_zero(Badd, 9*d->nefmdof);
  effCovered(nv, d->nefmdof, d->efm_dofid, blk);

  // rank-1 classes: scale * v v', restricted to each covered block. A term's coupling between
  // DIFFERENT vertices is off-diagonal and cannot be represented here; only its self-terms land
  mjEffRank1Iter it = {0};
  mjEffRank1 e;
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
    for (int a=0; a < e.nnz; a++) {
      int ia = e.colind[a], k = blk[ia];
      if (k < 0) {
        continue;
      }
      int base = d->efm_dofid[k];
      for (int b=0; b < e.nnz; b++) {
        int ib = e.colind[b];
        if (blk[ib] == k) {
          Badd[9*k + 3*(ia-base) + (ib-base)] += e.scale * e.val[a] * e.val[b];
        }
      }
    }
  }

  // efc rows: the same blocks of J'*D*J, over every row the solve carries, active or not. The
  // rows are the pairs inside their wall, the likely active set, and rows switch state in
  // nearly every CG iteration, so a fold of the rows active at the start left the CG with a
  // preconditioner that matched neither iterate: on the reef knot the mean iterations per
  // solve fell from 212 to 59 when every row was folded
  for (int r=0; r < nefc; r++) {
    if (!efc_D[r]) {
      continue;
    }
    mjtNum D = efc_D[r];
    if (is_sparse) {
      int adr = J_rowadr[r], nnz = J_rownnz[r];
      for (int a=0; a < nnz; a++) {
        int ia = J_colind[adr+a], k = blk[ia];
        if (k < 0) {
          continue;
        }
        int base = d->efm_dofid[k];
        for (int b=0; b < nnz; b++) {
          int ib = J_colind[adr+b];
          if (blk[ib] == k) {
            Badd[9*k + 3*(ia-base) + (ib-base)] += D * J[adr+a] * J[adr+b];
          }
        }
      }
    } else {
      const mjtNum* Jr = J + (size_t)r*nv;
      for (int k=0; k < d->nefmdof; k++) {
        int base = d->efm_dofid[k];
        for (int a=0; a < 3; a++) {
          if (!Jr[base+a]) {
            continue;
          }
          for (int b=0; b < 3; b++) {
            Badd[9*k + 3*a + b] += D * Jr[base+a] * Jr[base+b];
          }
        }
      }
    }
  }

  for (int k=0; k < d->nefmdof; k++) {
    mjtNum* Bk = L + 9*k;
    effBlockRaw(m, d, d->efm_dofid[k], Bk);
    mju_addTo(Bk, Badd + 9*k, 9);
    mju_cholFactor(Bk, 3, mjMINVAL);
  }

  // dense blocks of the uncovered dofs: the components of the same rows mj_effFoldDenseSize
  // sized the fold for (a mismatch leaves the backbone solve), laid out by component in U
  fold->S_valid = 0;
  if (fold->nu && is_sparse) {
    int ntree = m->ntree, nu, nS, partial;
    int* tint = mjSTACKALLOC(d, 3*ntree, int);
    int* comp = tint;
    int* cur = tint + 2*ntree;
    int ncomp = effComponents(m, d, blk, nefc, efc_D, J_rownnz, J_rowadr, J_colind,
                              comp, tint + ntree, cur, &nu, &nS, &partial);
    if (nu == fold->nu && ncomp == fold->ncomp && nS == fold->nS) {
      // component offsets in U and S
      for (int c=0; c <= ncomp; c++) {
        fold->Uadr[c] = 0;
      }
      for (int i=0; i < nv; i++) {
        int c = blk[i] < 0 ? comp[m->dof_treeid[i]] : -1;
        if (c >= 0) {
          fold->Uadr[c+1]++;
        }
      }
      fold->Sadr[0] = 0;
      for (int c=0; c < ncomp; c++) {
        int n = fold->Uadr[c+1];
        fold->Uadr[c+1] += fold->Uadr[c];
        fold->Sadr[c+1] = fold->Sadr[c] + n*n;
        cur[c] = fold->Uadr[c];
      }

      // U by component, ascending within each; blk becomes the index within the component
      for (int i=0; i < nv; i++) {
        int c = blk[i] < 0 ? comp[m->dof_treeid[i]] : -1;
        if (c >= 0) {
          fold->U[cur[c]] = i;
          blk[i] = cur[c]++ - fold->Uadr[c];
        } else {
          blk[i] = -1;
        }
      }
      fold->partial = partial;
      effDenseBlock(m, d, fold, blk, comp, nefc, efc_D, J, J_rownnz, J_rowadr, J_colind);
    }
  }
  mj_freeStack(d);
  return 1;
}


// mj_effPrec against a solver-owned fold instead of the shared d->efm_L
void mj_effPrecBlocks(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b,
                      const mjEffFold* fold) {
  if (d->efm_active) {
    effBlockApply(m, d, x, b, fold->L, fold);
    return;
  }
  mj_effPrec(m, d, x, b);
}


// can this dof contribute a damping term to the metric diagonal (model-level check;
// values are velocity-dependent and computed in mj_effShift)
static int dofDampPossible(const mjModel* m, int i) {
  if (mjDISABLED(mjDSBL_DAMPER)) {
    return 0;
  }
  return m->dof_damping[i] > 0 ||
         !mju_isZero(m->dof_dampingpoly + mjNPOLY*i, mjNPOLY) ||
         m->jnt_actuatorid[m->dof_jntid[i]] != -1;
}


// fill efm_ck with h * d(spring force)/d(displacement) per dof, clamped PSD-safe;
// mirrors the joint-spring structure of mj_springdamper; returns any-nonzero
static int effDiagStiff(const mjModel* m, mjData* d) {
  int any = 0, njnt = m->njnt;
  mjtNum h = m->opt.timestep;
  mju_zero(d->efm_ck, m->nv);
  if (mjDISABLED(mjDSBL_SPRING)) {
    return 0;
  }

  for (int j=0; j < njnt; j++) {
    mjtNum stiffness = m->jnt_stiffness[j];
    const mjtNum* spoly = m->jnt_stiffnesspoly + mjNPOLY*j;
    if (stiffness == 0 && mju_isZero(spoly, mjNPOLY)) {
      continue;
    }

    int padr = m->jnt_qposadr[j];
    int dadr = m->jnt_dofadr[j];
    const mjtNum* qspring = m->jnt_plasticadr[j] >= 0 ?
                            d->plastic + m->jnt_plasticadr[j] :
                            m->qpos_spring + padr;
    mjtNum k;

    switch ((mjtJoint) m->jnt_type[j]) {
    case mjJNT_FREE:
      // translation: radial derivative of the spring, isotropic on the 3 slide dofs
      {
        mjtNum dif[3];
        mju_sub3(dif, d->qpos+padr, qspring);
        k = mju_max(0, mjd_xPolyForce(stiffness, spoly, mju_norm3(dif), mjNPOLY, 0));
        d->efm_ck[dadr] = d->efm_ck[dadr+1] = d->efm_ck[dadr+2] = h*k;
        any = any || k > 0;
      }

      // continue with rotations
      dadr += 3;
      padr += 3;
      qspring += 3;
      mjFALLTHROUGH;

    case mjJNT_BALL:
      // rotation: small-angle isotropic treatment of the quaternion spring
      {
        mjtNum dif[3], quat[4];
        mju_copy4(quat, d->qpos+padr);
        mju_normalize4(quat);
        mju_subQuat(dif, quat, qspring);
        k = mju_max(0, mjd_xPolyForce(stiffness, spoly, mju_norm3(dif), mjNPOLY, 0));
        d->efm_ck[dadr] = d->efm_ck[dadr+1] = d->efm_ck[dadr+2] = h*k;
        any = any || k > 0;
      }
      break;

    case mjJNT_SLIDE:
    case mjJNT_HINGE:
      {
        mjtNum x = d->qpos[padr] - qspring[0];
        k = mju_max(0, mjd_xPolyForce(stiffness, spoly, x, mjNPOLY, 0));
        d->efm_ck[dadr] = h*k;
        any = any || k > 0;
      }
      break;
    }
  }

  // twist hinges of 1D rods: series torsional stiffness of the chain
  for (int f=0; f < m->nflex; f++) {
    int dof;
    mjtNum k = mj_rodTwistStiffness(m, f, &dof);
    if (k > 0) {
      d->efm_ck[dof] += h*k;
      any = 1;
    }
  }

  return any;
}


// finalize efm_diag = h*D + h^2*K: the damping derivative is velocity-dependent,
// evaluated here at the velocity stage, matching mj_EulerSkip's coverage but clamped
static void effDiagDamp(const mjModel* m, mjData* d) {
  int nv = m->nv;
  mjtNum h = m->opt.timestep;
  int damper = !mjDISABLED(mjDSBL_DAMPER);

  for (int i=0; i < nv; i++) {
    mjtNum damp_deriv = 0;
    if (damper) {
      mjtNum poly[mjNPOLY];
      mju_copy(poly, m->dof_dampingpoly + mjNPOLY*i, mjNPOLY);
      mjtNum damping = m->dof_damping[i]
                       + mj_actuatorDamping(m, mjOBJ_JOINT, m->dof_jntid[i], poly);
      damp_deriv = mju_max(0, mjd_xPolyForce(damping, poly, d->qvel[i], mjNPOLY, 1));
    }
    d->efm_diag[i] = h*damp_deriv + h*d->efm_ck[i];
  }
}


// refresh the velocity-stage values of the active metric: the smooth-force shift
// c = -h*K*qvel and the diagonal (its damping part is velocity-dependent). No allocation
// here, mirroring the efc value refresh pattern. Called from the velocity stage only,
// after the derived velocities (ten_velocity, body velocities) it reads are computed;
// every consumer of the values runs at the actuation stage or later
void mj_effShift(const mjModel* m, mjData* d) {
  if (!d->efm_active) {
    return;
  }
  mjtNum h = m->opt.timestep;
  mju_zero(d->efm_c, m->nv);

  // flex stiffness shift, absent when the spring force is disabled
  if (!mjDISABLED(mjDSBL_SPRING)) {
    mjd_flexInterp_mul(m, d, d->efm_c, d->qvel, h, 0, d->flexelem_krot);
    mjd_flexBend_mul(m, d, d->efm_c, d->qvel, -h, 0);
    mjd_flexStretch_mul(m, d, d->efm_c, d->qvel, -h, 0);
  }

  // contact: -h*K_contact*v from the packed rank-1 rows (see effContactBuild), so the shift
  // and the operator apply one definition of contact. Each row's scale already carries the h^2
  // of the effective stiffness, so the shift's -h*K*v is -(1/h) * scale * row'(row.v)
  for (int adr=0; adr < d->nefmcon; ) {
    int nnz = d->efm_con_ind[adr];
    const int* colind = d->efm_con_ind + adr + 2;
    const mjtNum* val = d->efm_con_val + adr + 2;
    mjtNum dot = 0;
    for (int a=0; a < nnz; a++) {
      dot += val[a] * d->qvel[colind[a]];
    }
    dot *= -d->efm_con_val[adr] / h;
    for (int a=0; a < nnz; a++) {
      d->efm_c[colind[a]] += dot * val[a];
    }
    adr += 2 + nnz;
  }

  // per-dof diagonal classes
  if (d->efm_diag) {
    int nv = m->nv;
    effDiagDamp(m, d);

    // stiffness shift
    for (int i=0; i < nv; i++) {
      d->efm_c[i] -= d->efm_ck[i] * d->qvel[i];
    }
  }

  // tendon values: stiffness at the current deadband displacement, damping at the current
  // tendon velocity, both clamped PSD-safe, mirroring the passive-force expressions
  for (int t=0; t < d->nefmT; t++) {
    int i = d->efm_tid[t];

    mjtNum k = 0;
    if (!mjDISABLED(mjDSBL_SPRING)) {
      mjtNum length = d->ten_length[i];
      const mjtNum* lengthspring = m->tendon_plasticadr[i] >= 0 ?
                                   d->plastic + m->tendon_plasticadr[i] :
                                   m->tendon_lengthspring + 2*i;
      mjtNum lower = lengthspring[0];
      mjtNum upper = lengthspring[1];
      mjtNum x = (length > upper) ? length - upper : (length < lower) ? length - lower : 0;
      if (x) {
        k = mju_max(0, mjd_xPolyForce(m->tendon_stiffness[i],
                                      m->tendon_stiffnesspoly + mjNPOLY*i, x, mjNPOLY, 0));
      }
    }

    mjtNum b = 0;
    if (!mjDISABLED(mjDSBL_DAMPER)) {
      mjtNum dpoly[mjNPOLY];
      mju_copy(dpoly, m->tendon_dampingpoly + mjNPOLY*i, mjNPOLY);
      mjtNum damping = m->tendon_damping[i] + mj_actuatorDamping(m, mjOBJ_TENDON, i, dpoly);
      b = mju_max(0, mjd_xPolyForce(damping, dpoly, d->ten_velocity[i], mjNPOLY, 1));
    }

    d->efm_ts[t] = h*h*k + h*b;
    d->efm_tk[t] = h*k;

    // stiffness shift: c -= h*k * ten_velocity * J'
    if (k) {
      mjtNum ckv = d->efm_tk[t] * d->ten_velocity[i];
      int end = m->ten_J_rowadr[i] + m->ten_J_rownnz[i];
      for (int j=m->ten_J_rowadr[i]; j < end; j++) {
        d->efm_c[m->ten_J_colind[j]] -= ckv * d->ten_J[j];
      }
    }
  }


  // fluid drag blocks: assemble the drag-only passive-fluid velocity derivative into the
  // assemble the drag derivatives into stack scratch in qDeriv's shape (qDeriv itself is
  // user-facing), gather into M's sparsity pattern, scale by -h. Drag derivatives are
  // symmetric dissipative by construction; lift and added-mass terms are excluded and
  // integrate explicitly
  if (d->efm_fluid) {
    mj_markStack(d);
    mjtNum* scratch = mjSTACKALLOC(d, m->nD, mjtNum);
    mju_zero(scratch, m->nD);
    int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
    int nbody = sleep_filter ? d->nbody_awake : m->nbody;
    for (int b=0; b < nbody; b++) {
      int i = sleep_filter ? d->body_awake_ind[b] : b;
      if (m->body_mass[i] < mjMINVAL) {
        continue;
      }
      int use_ellipsoid_model = 0;
      for (int j=0; j < m->body_geomnum[i] && use_ellipsoid_model == 0; j++) {
        use_ellipsoid_model += (m->geom_fluid[mjNFLUID*(m->body_geomadr[i] + j)] > 0);
      }
      if (use_ellipsoid_model) {
        mjd_ellipsoidFluid(m, d, scratch, i, /*flg_dragonly=*/1);
      } else {
        mjd_inertiaBoxFluid(m, d, scratch, i);
      }
    }
    mju_gather(d->efm_fluid, scratch, m->mapD2M, m->nC);
    mju_scl(d->efm_fluid, d->efm_fluid, -m->opt.timestep, m->nC);
    mj_freeStack(d);
  }
}


// actuation-stage refresh of the metric: actuator gain scalars (ctrl- and state-dependent,
// so evaluated after mj_fwdActuation), their smooth-force shift, and the backbone factor
// qH = M + diag(h*D + h^2*K) + tendon and actuator diagonals. The backbone is the part of
// the metric inside M's kinematic-tree sparsity -- every class's diagonal projection plus
// the fluid blocks -- so mj_factorI factors it with no fill-in; it is the exact metric
// when no coupling terms exist, the preconditioner otherwise, and the only metric the dual
// solvers ever see (mj_makeY). Idempotent: actuation-stage objects are rebuilt from
// scratch, so re-running the stage (mj_forwardSkip) is safe. Under sleep, rows of sleeping
// trees may hold stale M values: harmless, M and its factor are block-diagonal by tree and
// only awake rows are ever gathered or solved. If flg_factor is 0 the backbone is assembled
// (efm_sdiag reads its diagonal) but not factored, leaving qH unfactored: inverse dynamics
// only multiplies by the metric, and needs the factor only for the exact constraint diagonal
void mj_effActuation(const mjModel* m, mjData* d, int flg_factor) {
  if (!d->efm_active) {
    return;
  }
  int nv = m->nv;
  mjtNum h = m->opt.timestep;

  // actuator gains, clamped to the stabilizing sign
  d->nefmA = 0;
  if (d->efm_aid) {
    mju_zero(d->efm_ca, nv);
    int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
    for (int i=0; i < m->nactuator; i++) {
      if (!mj_effActuatorPossible(m, i) || mjd_actuatorDerivSkip(m, d, i, sleep_filter)) {
        continue;
      }
      mjtNum gv = mju_max(0, -mjd_actuatorVelDeriv(m, d, i));
      mjtNum gp = mju_max(0, -mjd_actuatorLenDeriv(m, d, i));
      if (!gv && !gp) {
        continue;
      }
      int t = d->nefmA++;
      d->efm_aid[t] = i;
      d->efm_as[t] = h*h*gp + h*gv;
      d->efm_ak[t] = h*gp;

      // stiffness shift: ca -= h*gp * actuator_velocity * moment'
      if (gp) {
        int oadr = m->actuator_outadr[i];
        for (int k=0; k < m->actuator_outnum[i]; k++) {
          int r = oadr + k;
          mjtNum ckv = d->efm_ak[t] * d->actuator_velocity[r];
          int end = d->moment_rowadr[r] + d->moment_rownnz[r];
          for (int j=d->moment_rowadr[r]; j < end; j++) {
            d->efm_ca[d->moment_colind[j]] -= ckv * d->actuator_moment[j];
          }
        }
      }
    }
  }

  // factor the backbone qH = M + all metric diagonals
  if (d->efm_diag) {
    mju_copy(d->qH, d->M, m->nC);
    for (int i=0; i < nv; i++) {
      d->qH[m->M_rowadr[i] + m->M_rownnz[i] - 1] += d->efm_diag[i];
    }
    for (int t=0; t < d->nefmT; t++) {
      mjtNum s = d->efm_ts[t];
      if (!s) {
        continue;
      }
      int i = d->efm_tid[t];
      int end = m->ten_J_rowadr[i] + m->ten_J_rownnz[i];
      for (int j=m->ten_J_rowadr[i]; j < end; j++) {
        int c = m->ten_J_colind[j];
        d->qH[m->M_rowadr[c] + m->M_rownnz[c] - 1] += s * d->ten_J[j] * d->ten_J[j];
      }
    }
    for (int t=0; t < d->nefmA; t++) {
      mjtNum s = d->efm_as[t];
      int i = d->efm_aid[t];
      int oadr = m->actuator_outadr[i];
      for (int k=0; k < m->actuator_outnum[i]; k++) {
        int r = oadr + k;
        int end = d->moment_rowadr[r] + d->moment_rownnz[r];
        for (int j=d->moment_rowadr[r]; j < end; j++) {
          int c = d->moment_colind[j];
          d->qH[m->M_rowadr[c] + m->M_rownnz[c] - 1] +=
              s * d->actuator_moment[j] * d->actuator_moment[j];
        }
      }
    }
    // fluid drag blocks share M's sparsity: add the full pattern
    if (d->efm_fluid) {
      mju_addTo(d->qH, d->efm_fluid, m->nC);
    }

    // record the diagonal additions for metric-consistent regularization, then factor
    for (int i=0; i < nv; i++) {
      int diag = m->M_rowadr[i] + m->M_rownnz[i] - 1;
      d->efm_sdiag[i] = d->qH[diag] - d->M[diag];
    }
    if (flg_factor) {
      int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < nv;
      mj_factorI(d->qH, d->qHDiagInv, sleep_filter ? d->nv_awake : nv,
                 m->M_rownnz, m->M_rowadr, m->M_colind,
                 sleep_filter ? d->dof_awake_ind : NULL);
    }
  }
}


// island of flex f, or -1 if none of its bodies belong to an island. All awake trees of a
// metric-coupled flex share one island (unionConstraintTrees), so the first dynamic body in an
// island decides membership
static int flexIsland(const mjModel* m, const mjData* d, int f) {
  int num = m->flex_interp[f] ? m->flex_nodenum[f] : m->flex_vertnum[f];
  int adr = m->flex_interp[f] ? m->flex_nodeadr[f] : m->flex_vertadr[f];
  const int* bodyid = m->flex_interp[f] ? m->flex_nodebodyid : m->flex_vertbodyid;
  for (int j=0; j < num; j++) {
    int tree = m->body_treeid[bodyid[adr + j]];
    if (tree >= 0 && d->tree_island[tree] >= 0) {
      return d->tree_island[tree];
    }
  }
  return -1;
}


// does flex f require the matrix-free stiffness path (not covered by the CSR)?
static int flexMatrixFree(const mjModel* m, const mjData* d, int f, int interp_asm) {
  if (m->flex_interp[f]) {
    return !interp_asm && mjd_flexInterp_processed(m, f);
  }
  return mjd_flexStiff_active(m, f, /*flg_bend=*/1, /*flg_stretch=*/1) &&
         (!d->nefmK || !mj_flexSimple(m, f));
}


// build CSR-style per-island lists of covered triples, bending-only dofs, and matrix-free flexes
// in the tail of d->efm_L once per step, avoiding O(nefmdof + nefm0dof + nflex) scans per island
// per CG iteration
static void effIslandBuild(const mjModel* m, mjData* d) {
  int nisland = d->nisland;
  int nb = d->nefmdof;
  int ntail = effIslandTail(m, d, nb);
  if (!nb) {
    d->efm_L = EFMALLOC(mjtNum, ntail);
    d->nefmL = ntail;
  }

  mjtNum* tail = d->efm_L + 9*nb;
  mju_zero(tail, ntail);
  tail[0] = EFF_ISLAND_TAG;
  int* A = (int*)(tail + EFF_HDR);
  A[0] = nisland;
  A[1] = nb;
  A[2] = nb ? 0 : m->nefm0dof;
  A[3] = m->nflex;
  mjEffIsland isl;
  effIslandPackLayout(A, &isl);

  mju_zeroInt(isl.cov_adr, nisland + 1);
  for (int k=0; k < nb; k++) {
    int island = d->dof_island[d->efm_dofid[k]];
    if (island >= 0 && island < nisland) {
      isl.cov_adr[island + 1]++;
    }
  }
  for (int c=0; c < nisland; c++) {
    isl.cov_adr[c + 1] += isl.cov_adr[c];
  }
  for (int k=0; k < nb; k++) {
    int island = d->dof_island[d->efm_dofid[k]];
    if (island >= 0 && island < nisland) {
      isl.cov_ind[isl.cov_adr[island]++] = k;
    }
  }
  for (int c=nisland; c > 0; c--) {
    isl.cov_adr[c] = isl.cov_adr[c - 1];
  }
  isl.cov_adr[0] = 0;

  mju_zeroInt(isl.bend_adr, nisland + 1);
  if (!nb && m->nefm0dof) {
    for (int i=0; i < m->nefm0dof; i++) {
      int island = d->dof_island[m->efm0_dofid[i]];
      if (island >= 0 && island < nisland) {
        isl.bend_adr[island + 1]++;
      }
    }
    for (int c=0; c < nisland; c++) {
      isl.bend_adr[c + 1] += isl.bend_adr[c];
    }
    for (int i=0; i < m->nefm0dof; i++) {
      int island = d->dof_island[m->efm0_dofid[i]];
      if (island >= 0 && island < nisland) {
        isl.bend_ind[isl.bend_adr[island]++] = i;
      }
    }
    for (int c=nisland; c > 0; c--) {
      isl.bend_adr[c] = isl.bend_adr[c - 1];
    }
    isl.bend_adr[0] = 0;
  }

  mju_zeroInt(isl.flex_adr, nisland + 1);
  int interp_asm = d->nefmK && mjd_flexInterpAssemblable(m);
  for (int f=0; f < m->nflex; f++) {
    if (!flexMatrixFree(m, d, f, interp_asm)) {
      continue;
    }
    int island = flexIsland(m, d, f);
    if (island >= 0 && island < nisland) {
      isl.flex_adr[island + 1]++;
    }
  }
  for (int c=0; c < nisland; c++) {
    isl.flex_adr[c + 1] += isl.flex_adr[c];
  }
  for (int f=0; f < m->nflex; f++) {
    if (!flexMatrixFree(m, d, f, interp_asm)) {
      continue;
    }
    int island = flexIsland(m, d, f);
    if (island >= 0 && island < nisland) {
      isl.flex_ind[isl.flex_adr[island]++] = f;
    }
  }
  for (int c=nisland; c > 0; c--) {
    isl.flex_adr[c] = isl.flex_adr[c - 1];
  }
  isl.flex_adr[0] = 0;
}


// island-local metric product res += S*vec: the diagonal classes, fluid blocks, rank-1 terms,
// and flex stiffness of the island, with vectors in island-local dof coordinates
void mj_effMulAddIsland(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                        int island) {
  int nv = d->island_nv[island];
  int idofadr = d->island_idofadr[island];
  const int* idof2dof = d->map_idof2dof + idofadr;

  if (d->efm_diag) {
    for (int k=0; k < nv; k++) {
      res[k] += d->efm_diag[idof2dof[k]] * vec[k];
    }
  }

  // fluid drag blocks: M-row columns are ancestor dofs, guaranteed in the same island
  if (d->efm_fluid) {
    for (int k=0; k < nv; k++) {
      int i = idof2dof[k];
      int start = m->M_rowadr[i];
      int diag = start + m->M_rownnz[i] - 1;
      mjtNum acc = d->efm_fluid[diag] * vec[k];
      for (int a=start; a < diag; a++) {
        int kc = d->map_dof2idof[m->M_colind[a]] - idofadr;
        mjtNum F = d->efm_fluid[a];
        acc += F * vec[kc];
        res[kc] += F * vec[k];
      }
      res[k] += acc;
    }
  }

  // the island's rank-1 terms: every tree an entry touches shares the island (union-find
  // merges the support), so the first column decides membership
  mjEffRank1Iter it = {0};
  mjEffRank1 e;
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
    if (d->tree_island[m->dof_treeid[e.colind[0]]] != island) {
      continue;
    }
    mjtNum dot = 0;
    for (int j=0; j < e.nnz; j++) {
      dot += e.val[j] * vec[d->map_dof2idof[e.colind[j]] - idofadr];
    }
    dot *= e.scale;
    for (int j=0; j < e.nnz; j++) {
      res[d->map_dof2idof[e.colind[j]] - idofadr] += dot * e.val[j];
    }
  }

  mjEffIsland isl;
  int has_isl = effIslandStep(d, &isl);

  // flex stiffness CSR restricted to the island's covered triples.
  // Invariant: all dynamic trees of a metric-coupled flex are awake or asleep together (sleeping
  // acts on whole islands in mj_sleep and wakes whole cycles in mj_wakeIsland, and
  // unionConstraintTrees unions all awake trees of a metric-coupled flex into one island), so
  // every column in row i belongs to the same island as i.
  if (d->nefmK && d->nefmdof && has_isl) {
    const int* rowadr = d->efm_K_rowadr;
    const int* rownnz = d->efm_K_rownnz;
    const int* colind = d->efm_K_colind;
    const mjtNum* val = d->efm_K_val;
    int p_end = isl.cov_adr[island + 1];
    for (int p=isl.cov_adr[island]; p < p_end; p++) {
      int i = d->efm_dofid[isl.cov_ind[p]];
      int li = d->map_dof2idof[i] - idofadr;
      const mjtNum* v0 = val + rowadr[i];
      const mjtNum* v1 = val + rowadr[i+1];
      const mjtNum* v2 = val + rowadr[i+2];
      const int* ci = colind + rowadr[i];
      int nn = rownnz[i] / 3;
      mjtNum r0 = 0, r1 = 0, r2 = 0;
      for (int j=0; j < nn; j++) {
        int lc = d->map_dof2idof[ci[3*j]] - idofadr;
        if (lc < 0 || lc + 2 >= nv) {
          mjERROR("flex stiffness column outside island %d", island);  // SHOULD NOT OCCUR
        }
        mjtNum x0 = vec[lc], x1 = vec[lc+1], x2 = vec[lc+2];
        r0 += v0[3*j]*x0 + v0[3*j+1]*x1 + v0[3*j+2]*x2;
        r1 += v1[3*j]*x0 + v1[3*j+1]*x1 + v1[3*j+2]*x2;
        r2 += v2[3*j]*x0 + v2[3*j+1]*x1 + v2[3*j+2]*x2;
      }
      res[li] += r0;
      res[li+1] += r1;
      res[li+2] += r2;
    }
  } else if (d->nefmK) {
    for (int k=0; k < nv; k++) {
      int i = idof2dof[k];
      int nnz = d->efm_K_rownnz[i];
      if (!nnz) {
        continue;
      }
      const int* ci = d->efm_K_colind + d->efm_K_rowadr[i];
      const mjtNum* v = d->efm_K_val + d->efm_K_rowadr[i];
      mjtNum acc = 0;
      for (int j=0; j < nnz; j++) {
        int lc = d->map_dof2idof[ci[j]] - idofadr;
        if (lc < 0 || lc >= nv) {
          mjERROR("flex stiffness column outside island %d", island);  // SHOULD NOT OCCUR
        }
        acc += v[j] * vec[lc];
      }
      res[k] += acc;
    }
  }

  // matrix-free fallback for flexes in this island not covered by the CSR
  int has_mf = has_isl ? (isl.flex_adr[island] < isl.flex_adr[island + 1]) : 0;
  int interp_asm = d->nefmK && mjd_flexInterpAssemblable(m);
  if (!has_isl) {
    for (int f=0; f < m->nflex; f++) {
      if (flexMatrixFree(m, d, f, interp_asm) && flexIsland(m, d, f) == island) {
        has_mf = 1;
        break;
      }
    }
  }
  if (has_mf) {
    mjtNum h = m->opt.timestep;
    mjtNum s1 = mjDISABLED(mjDSBL_SPRING) ? 0 : h*h;
    mjtNum s2 = mjDISABLED(mjDSBL_DAMPER) ? 0 : h;
    mj_markStack(d);
    mjtNum* gvec = mjSTACKALLOC(d, m->nv, mjtNum);
    mjtNum* gres = mjSTACKALLOC(d, m->nv, mjtNum);
    for (int k=0; k < nv; k++) {
      int dof = idof2dof[k];
      gvec[dof] = vec[k];
      gres[dof] = 0;
    }
    int n = has_isl ? (isl.flex_adr[island + 1] - isl.flex_adr[island]) : m->nflex;
    for (int p=0; p < n; p++) {
      int f = has_isl ? isl.flex_ind[isl.flex_adr[island] + p] : p;
      if (!has_isl && (!flexMatrixFree(m, d, f, interp_asm) || flexIsland(m, d, f) != island)) {
        continue;
      }
      if (m->flex_interp[f]) {
        mjd_flexInterp_mulRange(m, d, gres, gvec, -s1, -s2, d->flexelem_krot, f, f+1);
      } else {
        mjd_flexBend_mulRange(m, d, gres, gvec, s1, s2, f, f+1);
        mjd_flexStretch_mulRange(m, d, gres, gvec, s1, s2, f, f+1);
      }
    }
    for (int k=0; k < nv; k++) {
      res[k] += gres[idof2dof[k]];
    }
    mj_freeStack(d);
  }
}


// apply the step's sparse factor on the covered triples of island (vectors in island-local dof
// coordinates); returns 1 if applied, 0 to fall back to the 3x3 blocks.
// Note: separate islands have no edges between them in L so zeroing rhs outside the island is
// exact, though solving the full factor and scanning nefmdof costs O(nefmdof) per island call.
static int effCholSolveIsland(const mjModel* m, mjData* d, int island, mjtNum* x,
                              const mjtNum* rhs) {
  int nb = d->nefmdof;
  if (!effCholReserved(d)) {
    return 0;
  }
  effCholEnsure(m, d);
  if (d->efm_L[9*nb] != 1) {
    return 0;
  }
  mjEffChol cs;
  effCholStep(d, &cs);
  if (mj_stackBytesAvailable(d) < mj_stackFrameBytes() + 2*effNumBytes(cs.n)) {
    return 0;
  }
  const mjtNum* Ls = effCholStepValues(d, &cs);
  int idofadr = d->island_idofadr[island];
  mj_markStack(d);
  mjtNum* bc = mjSTACKALLOC(d, cs.n, mjtNum);
  mjtNum* xc = mjSTACKALLOC(d, cs.n, mjtNum);
  for (int k=0; k < nb; k++) {
    int i = cs.dofid[k];
    if (d->dof_island[i] == island) {
      mju_copy3(bc + 3*cs.perm[k], rhs + (d->map_dof2idof[i] - idofadr));
    } else {
      mju_zero3(bc + 3*cs.perm[k]);
    }
  }
  effCholSolveBlocked(xc, Ls, bc, &cs);
  for (int k=0; k < nb; k++) {
    int i = cs.dofid[k];
    if (d->dof_island[i] == island) {
      mju_copy3(x + (d->map_dof2idof[i] - idofadr), xc + 3*cs.perm[k]);
    }
  }
  mj_freeStack(d);
  return 1;
}


// island-local metric preconditioner x ~= (M + B)_island^-1 b, vectors in island-local dof
// coordinates; qLD/qLDiagInv/M_* are the island's gathered backbone factor
void mj_effPrecIsland(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b, int island,
                      const mjtNum* qLD, const mjtNum* qLDiagInv,
                      const int* M_rownnz, const int* M_rowadr, const int* M_colind) {
  int nv = d->island_nv[island];
  int idofadr = d->island_idofadr[island];
  int nbd = m->nefm0dof;
  int flg_bend = nbd && !d->nefmdof;

  mjEffIsland isl;
  int has_isl = effIslandStep(d, &isl);
  int has_cov = has_isl && (isl.cov_adr[island] < isl.cov_adr[island + 1]);
  if (!has_isl) {
    for (int k=0; k < d->nefmdof; k++) {
      if (d->dof_island[d->efm_dofid[k]] == island) {
        has_cov = 1;
        break;
      }
    }
  }
  int has_bend = flg_bend && has_isl && (isl.bend_adr[island] < isl.bend_adr[island + 1]);

  // no covered or bending-only flex blocks in this island: the gathered backbone factor is all
  if (!has_cov && !has_bend) {
    if (x != b) {
      mju_copy(x, b, nv);
    }
    mj_solveLD(x, qLD, qLDiagInv, nv, 1, M_rownnz, M_rowadr, M_colind, NULL);
    return;
  }

  mj_markStack(d);
  mjtNum* rhs = mjSTACKALLOC(d, nv, mjtNum);
  mju_copy(rhs, b, nv);   // b may alias x

  // backbone solve on uncovered island dofs
  mju_copy(x, rhs, nv);
  if (has_cov && has_isl) {
    int p_end = isl.cov_adr[island + 1];
    for (int p=isl.cov_adr[island]; p < p_end; p++) {
      int i = d->efm_dofid[isl.cov_ind[p]];
      mju_zero(x + (d->map_dof2idof[i] - idofadr), 3);
    }
  } else if (has_cov) {
    for (int k=0; k < d->nefmdof; k++) {
      int i = d->efm_dofid[k];
      if (d->dof_island[i] == island) {
        mju_zero(x + (d->map_dof2idof[i] - idofadr), 3);
      }
    }
  }
  if (has_bend) {
    int p_end = isl.bend_adr[island + 1];
    for (int p=isl.bend_adr[island]; p < p_end; p++) {
      int gdof = m->efm0_dofid[isl.bend_ind[p]];
      x[d->map_dof2idof[gdof] - idofadr] = 0;
    }
  }
  mj_solveLD(x, qLD, qLDiagInv, nv, 1, M_rownnz, M_rowadr, M_colind, NULL);

  // sparse factor or per-step 3x3 blocks on the island's covered triples
  if (has_cov && !effCholSolveIsland(m, d, island, x, rhs)) {
    if (has_isl) {
      int p_end = isl.cov_adr[island + 1];
      for (int p=isl.cov_adr[island]; p < p_end; p++) {
        int k = isl.cov_ind[p];
        int li = d->map_dof2idof[d->efm_dofid[k]] - idofadr;
        chol3Solve(x + li, d->efm_L + 9*k, rhs + li);
      }
    } else {
      for (int k=0; k < d->nefmdof; k++) {
        int i = d->efm_dofid[k];
        if (d->dof_island[i] == island) {
          int li = d->map_dof2idof[i] - idofadr;
          chol3Solve(x + li, d->efm_L + 9*k, rhs + li);
        }
      }
    }
  }

  // bending-only constant factor restricted to the island
  if (has_bend) {
    mjtNum* bfr = mjSTACKALLOC(d, nbd, mjtNum);
    mjtNum* bfz = mjSTACKALLOC(d, nbd, mjtNum);
    mju_zero(bfr, nbd);
    int p_end = isl.bend_adr[island + 1];
    for (int p=isl.bend_adr[island]; p < p_end; p++) {
      int i = isl.bend_ind[p];
      int gdof = m->efm0_dofid[i];
      bfr[i] = rhs[d->map_dof2idof[gdof] - idofadr];
    }
    mju_cholSolveSparse(bfz, m->efm0_L, bfr, nbd,
                        m->efm0_L_rownnz, m->efm0_L_rowadr, m->efm0_L_colind);
    for (int p=isl.bend_adr[island]; p < p_end; p++) {
      int i = isl.bend_ind[p];
      int gdof = m->efm0_dofid[i];
      x[d->map_dof2idof[gdof] - idofadr] = bfz[i];
    }
  }
  mj_freeStack(d);
}


// publish passive flex contact as the metric's rank-1 contact class, one packed row per pair
// ([nnz, conid, colind...] / [scale, force, val...], see engine_derivative.h): matrix-free
// contact costs O(nnz) per pair where the stiffness CSR cost O(nnz^2), and the CSR's sparsity no
// longer grows with the contact set
static void effContactBuild(const mjModel* m, mjData* d, mjtNum scale) {
  d->nefmcon = 0;
  if (!d->ncon || !mjd_flexPassiveContact_any(m)) {
    return;
  }
  int nv = m->nv, issparse = mj_isSparse(m);
  mj_markStack(d);
  mjtNum* jacdif = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jac1 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jac2 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jacn = mjSTACKALLOC(d, 3*nv, mjtNum);
  int* chain = mjSTACKALLOC(d, nv, int);
  int* flex_ndyn = m->nflex ? mjSTACKALLOC(d, m->nflex, int) : NULL;
  for (int f = 0; f < m->nflex; f++) {
    flex_ndyn[f] = m->flex_interp[f] ? mjd_flexInterpDynNodes(m, f) : 0;
  }
  mjtNum* kcon = mjSTACKALLOC(d, d->ncon, mjtNum);

  // pass 1: packed length, two header entries plus nnz per row
  for (int i = 0; i < d->ncon; i++) {
    const mjContact* con = d->contact + i;
    if (con->exclude != 4) {
      kcon[i] = 0;
      continue;
    }
    kcon[i] = mjd_flexContactStiffness(m, d, con, flex_ndyn);
    if (kcon[i] <= 0) {
      continue;
    }
    int NV = mj_contactJacobian(m, d, con, con->dim, jacdif, NULL, jac1, jac2, NULL, NULL, chain);
    if (NV) {
      d->nefmcon += 2 + NV;   // an upper bound in a dense model, trimmed to `adr` below
    } else {
      d->contact[i].exclude = 3;   // affects no dofs, as the passive force used to mark it
    }
  }
  if (!d->nefmcon) {
    mj_freeStack(d);
    return;
  }
  d->efm_con_ind = EFMALLOC(int, d->nefmcon);
  d->efm_con_val = EFMALLOC(mjtNum, d->nefmcon);

  // pass 2: fill, rows packed back to back in both arrays
  int adr = 0;
  for (int i = 0; i < d->ncon; i++) {
    const mjContact* con = d->contact + i;
    mjtNum k = kcon[i];
    if (con->exclude != 4 || k <= 0) {
      continue;
    }
    int NV = mj_contactJacobian(m, d, con, con->dim, jacdif, NULL, jac1, jac2, NULL, NULL, chain);
    if (!NV) {
      continue;
    }
    mju_mulMatMat(jacn, con->frame, jacdif, con->dim > 1 ? 3 : 1, 3, NV);
    mjtNum s = mjd_flexContactSlack(k, con->dist, /*lam=*/0);
    int nnz = 0;
    for (int a = 0; a < NV; a++) {
      // a dense model gets no chain: every dof is returned, so compact the row on its nonzeros
      if (!issparse && jacn[a] == 0) {
        continue;
      }
      d->efm_con_ind[adr + 2 + nnz] = issparse ? chain[a] : a;
      d->efm_con_val[adr + 2 + nnz] = jacn[a];
      nnz++;
    }
    d->efm_con_ind[adr] = nnz;
    d->efm_con_ind[adr + 1] = i;
    d->efm_con_val[adr] = scale * k;
    d->efm_con_val[adr + 1] = -k*mjd_flexContactResidual(k, con->dist, s, /*lam=*/0);
    adr += 2 + nnz;
  }
  d->nefmcon = adr;   // the dense compaction can land below the bound counted above
  mj_freeStack(d);
}


// apply the published rows' forces, res += force * row: the passive stage takes its contact
// force from the rows the metric build published
void mj_effContactForce(const mjData* d, mjtNum* res) {
  const int* ind = d->efm_con_ind;
  const mjtNum* val = d->efm_con_val;
  for (int adr = 0; adr < d->nefmcon; ) {
    int nnz = ind[adr];
    mjtNum f = val[adr + 1];
    const int* colind = ind + adr + 2;
    const mjtNum* row = val + adr + 2;
    for (int a = 0; a < nnz; a++) {
      // fused multiply-add: rounding the product first moves the results by an ulp
      res[colind[a]] += f * row[a];
    }
    adr += 2 + nnz;
  }
}


// build the per-step implicit effective metric on the arena, or deactivate it. The gate
// decision (integrator=discrete) is the caller's: the metric module has no dependency on
// the solver configuration beyond what it is told here.
void mj_effBuild(const mjModel* m, mjData* d, int active, int flg_factor) {
  int nv = m->nv;
  d->efm_active = 0;
  d->nefmK = 0;
  d->nefmcon = 0;
  d->nefmT = 0;
  d->nefmdof = 0;
  d->nefmL = 0;
  d->nefmA = 0;
  d->efm_diag = NULL;
  d->efm_ck = NULL;
  d->efm_sdiag = NULL;
  d->efm_fluid = NULL;
  d->efm_tid = NULL;
  d->efm_ts = NULL;
  d->efm_tk = NULL;
  d->efm_aid = NULL;
  d->efm_as = NULL;
  d->efm_ak = NULL;
  d->efm_ca = NULL;
  if (!active) {
    return;
  }
  mjtNum h = m->opt.timestep;

  // corotated element stiffness cache (used by the shift and the matrix-free fallback)
  mju_zero(d->flexelem_krot, m->nflexstiffness);
  mjd_flexInterp_cacheKrot(m, d, d->flexelem_krot);

  // smooth-force shift c = h*K*qvel (values refreshed by mj_effShift in the velocity stage)
  d->efm_c = EFMALLOC(mjtNum, nv);

  // per-dof diagonal classes (joint damping/stiffness, joint-transmission actuator damping):
  // position-dependent stiffness assembled here; the velocity-dependent damping part and the
  // qH backbone factor are refreshed by mj_effShift in the velocity stage
  d->efm_ck = EFMALLOC(mjtNum, nv);
  int any_diag = effDiagStiff(m, d);
  for (int i=0; !any_diag && i < nv; i++) {
    any_diag = dofDampPossible(m, i);
  }

  // tendons with metric terms: id list here, values (velocity-dependent) in mj_effShift.
  // Sleeping tendons are excluded, matching the passive-force filter; the wake rule
  // (mj_wakeTendon) keeps metric-coupled tendon pairs awake or asleep together. Under the
  // dual solver the class is excluded (mj_effCouplings): an empty list removes its metric
  // terms and its shift together, so the tendon forces integrate explicitly
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
  if (m->ntendon && mj_effCouplings(m)) {
    d->efm_tid = EFMALLOC(int, m->ntendon);
    d->efm_ts  = EFMALLOC(mjtNum, m->ntendon);
    d->efm_tk  = EFMALLOC(mjtNum, m->ntendon);
    for (int i=0; i < m->ntendon; i++) {
      if (!mj_effTendonPossible(m, i)) {
        continue;
      }
      if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, i) != mjS_AWAKE) {
        continue;
      }
      d->efm_ts[d->nefmT] = 0;
      d->efm_tid[d->nefmT++] = i;
    }
  }

  // metric-possible actuators: allocation here, values (ctrl- and state-dependent) in
  // mj_effActuation at the actuation stage. Under the dual solver the class is excluded
  // (mj_effCouplings, as for tendons): no allocation, so mj_effActuation adds nothing
  int any_act = 0;
  if (mj_effCouplings(m)) {
    for (int i=0; i < m->nactuator; i++) {
      if (mj_effActuatorPossible(m, i)) {
        any_act = 1;
        break;
      }
    }
  }
  if (any_act) {
    d->efm_aid = EFMALLOC(int, m->nactuator);
    d->efm_as  = EFMALLOC(mjtNum, m->nactuator);
    d->efm_ak  = EFMALLOC(mjtNum, m->nactuator);
    d->efm_ca  = EFMALLOC(mjtNum, nv);
    mju_zero(d->efm_ca, nv);
  }

  // the tendon and actuator diagonals join the qH backbone: force the diagonal machinery on.
  // Fluid drag and passive flex contact enter only while their forces are applied: mj_passive
  // skips them, like every passive force, when both the spring and damper forces are disabled
  int passive = !(mjDISABLED(mjDSBL_SPRING) && mjDISABLED(mjDSBL_DAMPER));
  int any_fluid = passive && (m->opt.viscosity > 0 || m->opt.density > 0);
  d->efm_diag = (any_diag || d->nefmT || any_act || any_fluid) ? EFMALLOC(mjtNum, nv) : NULL;
  d->efm_sdiag = d->efm_diag ? EFMALLOC(mjtNum, nv) : NULL;
  d->efm_fluid = any_fluid ? EFMALLOC(mjtNum, m->nC) : NULL;

  // assemble the standard-flex part of B into CSR (constant during the step). With stretch or
  // assemblable interp present, assemble the FULL matrix (bending included): one CSR then
  // serves both the matvec and the per-step factor. Bending-only models keep the stencil
  // operator + the constant mj_setConst factor. The stiffness (s1) and damping (s2) parts enter
  // only when their forces are enabled; the structure does not depend on the flags
  mjtNum s1 = mjDISABLED(mjDSBL_SPRING) ? 0 : h*h;
  mjtNum s2 = mjDISABLED(mjDSBL_DAMPER) ? 0 : h;
  const mjtNum* krot = mjd_flexInterpAssemblable(m) ? d->flexelem_krot : NULL;
  d->efm_K_rownnz = EFMALLOC(int, nv);
  d->efm_K_rowadr = EFMALLOC(int, nv);

  // Newton consumes the metric as an explicit sparse matrix inside its Hessian: force full
  // CSR assembly, including bending-only flexes which otherwise keep the matrix-free stencil
  // plus the constant mj_setConst factor (non-assemblable interp flexes are rejected by
  // mj_checkDiscrete under Newton)
  int assemble_any = mjd_flexStiff_any(m, krot != NULL) || mjd_flexPassiveContact_any(m);
  if (m->opt.solver == mjSOL_NEWTON) {
    for (int f=0; !assemble_any && f < m->nflex; f++) {
      assemble_any = mjd_flexStiff_active(m, f, /*flg_bend=*/1, /*flg_stretch=*/1);
    }
  }
  if (assemble_any) {
    d->nefmK = mjd_flexStiff_assemble(m, d, d->efm_K_rownnz, d->efm_K_rowadr,
                                      NULL, NULL, s1, s2, /*bend*/ 1, /*stretch*/ 1, krot);
  }

  // passive flex contact is a rank-1 class, not CSR entries (see effContactBuild)
  if (passive) {
    effContactBuild(m, d, h*h);
  }
  if (d->nefmK) {
    d->efm_K_colind = EFMALLOC(int, d->nefmK);
    d->efm_K_val    = EFMALLOC(mjtNum, d->nefmK);
    mjd_flexStiff_assemble(m, d, d->efm_K_rownnz, d->efm_K_rowadr,
                           d->efm_K_colind, d->efm_K_val, s1, s2,
                           /*bend*/ 1, /*stretch*/ 1, krot);
    // per-step factor of the flex block of (M + K): the stiffness is constant during the
    // step, so one factorization here turns every preconditioner application into a direct
    // solve (the stiff flex block stops being iterated on). Consumers that only multiply
    // (inverse dynamics) skip it.
    if (flg_factor) {
      effBlocks(m, d);
    }

  } else {
    mju_zeroInt(d->efm_K_rownnz, nv);
    mju_zeroInt(d->efm_K_rowadr, nv);
  }

  // per-island flex/triple lists for mj_effMulAddIsland and mj_effPrecIsland
  if (d->nisland > 0 && d->nefc > 0 && m->nflex > 0 &&
      !mjENABLED(mjENBL_IPC) && !effCholReserved(d)) {
    effIslandBuild(m, d);
  }

  d->efm_active = 1;
}
