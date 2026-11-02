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

// The IPC contact mode of the discrete integrator: barrier-free augmented-Lagrangian contact for
// 2D flexes (Li et al., arXiv:2512.12151). A per-pair multiplier and a persistent active set
// replace the log barrier, and a CCD-bounded committed position keeps the configuration
// intersection-free. Elasticity is not solved here: elastic2d rides the effective metric and
// edge-equality rides the native constraint rows.

#include "engine/engine_ipc.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_collision_continuous.h"
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_smooth.h"
#include "engine/engine_core_util.h"
#include "engine/engine_derivative.h"
#include "engine/engine_forward.h"
#include "engine/engine_memory.h"
#include "engine/engine_metric.h"
#include "engine/engine_name.h"
#include "engine/engine_sensor.h"
#include "engine/engine_support.h"
#include "engine/engine_thread.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"

#define IPC_NPT 8                    // max vertices in one pair: point-triangle 1+3, edge-edge 2+2
#define IPC_ASET_AGE 25              // drop a never-engaged pair after this many quiet updates
#define IPC_EVICT 44                 // drop an engaged pair after this many quiet updates
#define IPC_DECAY 0.9                // stiffness decay per quiet update: scale = IPC_DECAY^age * mu
#define IPC_ALPHA_LB 1e-6            // advance xfree only when the CCD fraction exceeds this
#define IPC_STALL_MAX 64             // outer iterations without a feasible advance: the step froze
#define IPC_WS_TOL 1e-8              // round-off guard on the working set's residual test (m)

// The three constants below are absolute lengths and speeds, tuned on metre-scale models with
// millimetre-thick flexes at 1-2 ms steps; a model at another scale needs them rescaled.
// TODO(quaglino): compute the standoff per flex pair from the radii, as the engine's flex contact
// does with their sum, instead of a hard-coded cap, with a floor for a zero radius; the band then
// follows from it.
#define IPC_DELTACAP 0.001           // rest gap between midsurfaces (m): the standoff is the
                                     // thinner radius of the pair, capped at this
#define IPC_GHAT (3 * IPC_DELTACAP)  // detection band (m), narrowed per pair to the thinner radius
#define IPC_VEL_TOL 0.05             // step convergence (m/s): |dx|_inf <= IPC_VEL_TOL*h; a safety
                                     // criterion, the loop normally ends on a full line-search step

// Solver-side contact pair: the multiplier, the linearization at xfree, the active-set age and the
// slack. Its leading members mirror mjcFlexPair, so a pair is passed to the mjc_* API by cast.
typedef struct {
  mjcFlexPairType type;        // == mjcFlexPair.type
  int idx[4];                  // == mjcFlexPair.idx
  int gi;                      // == mjcFlexPair.g
  mjtNum lam;                  // AL multiplier
  mjtNum ld0, ln[3], lcw[4];   // per-outer linearization at xfree
  int liv[4], lniv;
  int cnt;                     // active-set state machine
} ipcCon;
#define IPC_PAIR(con) ((const mjcFlexPair*)(con))
// IPC_PAIR is a layout pun, so the shared prefix must stay put: a reorder of either struct
// would otherwise compile clean and silently mis-address every pair handed to the mjc_* API
_Static_assert(offsetof(ipcCon, type) == offsetof(mjcFlexPair, type), "ipcCon prefix: type");
_Static_assert(offsetof(ipcCon, idx) == offsetof(mjcFlexPair, idx), "ipcCon prefix: idx");
_Static_assert(offsetof(ipcCon, gi) == offsetof(mjcFlexPair, g), "ipcCon prefix: g");
_Static_assert(sizeof(((ipcCon*)0)->idx) == sizeof(((mjcFlexPair*)0)->idx), "ipcCon prefix: idx[]");

// per-pair AL stiffness, in force form: mjFLEXCONTACT_OMEGA2 times the smallest nonzero mass
// among the pair's vertices. A vertex pinned to a body carries mass 0 (its inertia is in the
// body), and a zero mu would put 0/0 in lam/mu; an all-pinned pair gets a tiny mu instead.
static mjtNum ipc_muPair(const ipcCon* con, const mjtNum* mass) {
  int vv[4], nvv;
  nvv = mjc_pairVerts(vv, IPC_PAIR(con));
  mjtNum mmin = 1e30;
  for (int q=0; q < nvv; q++) {
    mjtNum mv = mass[vv[q]];
    if (mv > 0 && mv < mmin) mmin = mv;
  }
  if (mmin >= 1e29) mmin = 1e-9;
  return mjFLEXCONTACT_OMEGA2 * mmin;
}


// cnt -> decay exponent c (AL normal-contact aging): c = cnt>=0 ? cnt : max(-cnt-6, 0).
static inline int ipc_cntExp(int cnt) { return cnt >= 0 ? cnt : (-cnt - 6 > 0 ? -cnt - 6 : 0); }

// sort two ints ascending in place
static inline void ipc_sort2(int* a, int* b) {
  if (*a > *b) {
    int t = *a;
    *a = *b;
    *b = t;
  }
}


// canonical pair indices: order-independent only within each role, as the roles differ per type
// (a lone vertex or geom feature vs. the flex points of a triangle or edge), so that permutations
// of one pair agree while distinct pairs sharing an index multiset do not
static void ipc_canonIdx(int id[4], const mjcFlexPair* con) {
  for (int k=0; k < 4; k++)
    id[k] = con->idx[k];
  switch (con->type) {
  case mjcFLEX_VERT_TRI:    // {vertex, triangle}
  case mjcGEOM_CORNER_TRI:  // {geom corner, triangle}
    ipc_sort2(id+1, id+2);
    ipc_sort2(id+2, id+3);
    ipc_sort2(id+1, id+2);
    break;
  case mjcFLEX_EDGE_EDGE:   // {edge, edge}, unordered edges of unordered endpoints
    ipc_sort2(id+0, id+1);
    ipc_sort2(id+2, id+3);
    if (id[0] > id[2] || (id[0] == id[2] && id[1] > id[3])) {
      int t0 = id[0], t1 = id[1];
      id[0] = id[2];
      id[1] = id[3];
      id[2] = t0;
      id[3] = t1;
    }
    break;
  case mjcFLEX_VERT_GEOM:   // {vertex}
    id[1] = id[2] = id[3] = 0;
    break;
  case mjcGEOM_EDGE_EDGE:   // {geom edge, flex edge}
    ipc_sort2(id+1, id+2);
    id[3] = 0;
    break;
  }
}


// per-pair hash for the active-set merge: contact type, geom and canonical indices
static uint64_t ipc_pairHash(const mjcFlexPair* con) {
  int id[4];
  ipc_canonIdx(id, con);
  uint64_t hh = (uint64_t)con->type * 1000003ull + (uint64_t)(con->g + 1);
  for (int k=0; k < 4; k++)
    hh = hh * 1000003ull + (uint64_t)(id[k] + 1);
  return hh;
}


// pair identity: same type, geom and canonical indices
static int ipc_pairEqual(const mjcFlexPair* a, const mjcFlexPair* b) {
  if (a->type != b->type || a->g != b->g) {
    return 0;
  }
  int ia[4], ib[4];
  ipc_canonIdx(ia, a);
  ipc_canonIdx(ib, b);
  return ia[0] == ib[0] && ia[1] == ib[1] && ia[2] == ib[2] && ia[3] == ib[3];
}


// Cross-step state lives per flex vertex: the multiplier (d->flexvert_lambda, the max over the
// pairs touching the vertex) and the contact age (d->flexvert_conage: < 0 loaded recently, > 0
// quiet updates since). A pair reads the min age over its vertices. The age drives the stiffness
// decay; eviction is decided in ipc_mergeActiveSet. Both key on the same signal, load at a vertex,
// so a pair is dropped at the bottom of its decay ramp rather than part way down, and mirror-image
// contacts stay symmetric (a pair's identity is an arbitrary closest-feature label; load is not).

// pair age: the min over its vertices, 0 when none carries one
static int ipc_conAge(const ipcCon* con, const int* age, const int* pt2vg) {
  int vv[4], nvv, a = INT_MAX;
  nvv = mjc_pairVerts(vv, IPC_PAIR(con));
  for (int q=0; q < nvv; q++) {
    int g = age[pt2vg[vv[q]]];
    if (g < a) a = g;
  }
  return a == INT_MAX ? 0 : a;
}


// advance the per-vertex age one update: loaded iff some active pair touches the vertex
static void ipc_ageStep(int* age, const ipcCon* aset, int naset, const int* pt2vg, int npt,
                        char* loaded) {
  for (int i=0; i < npt; i++) loaded[i] = 0;
  for (int c=0; c < naset; c++) {
    if (aset[c].lam <= 0) continue;
    int vv[4], nvv;
    nvv = mjc_pairVerts(vv, IPC_PAIR(&aset[c]));
    for (int q=0; q < nvv; q++) loaded[vv[q]] = 1;
  }
  for (int i=0; i < npt; i++) {
    int g = pt2vg[i];
    if (loaded[i]) age[g] = (age[g] == 0 || age[g] > 5) ? 0 : -1;
    else           age[g] += (age[g] >= 0) ? 1 : -1;
  }
}


// Multiplier and age update at the iterate x, reading the multiplier from before the update:
//   inactive (c_raw - lam/mu > 0): lam = 0, the quiet counter advances
//   active:                        lam -= c_raw*mu, the counter marks the pair loaded
// with no clamp on lam. Then sink lam into the per-vertex store for the next step's warm start.
static void ipc_flexLamUpdate(const mjtNum* x, const mjtNum* xfree, const mjtNum* rad, mjtNum ghat,
                              const mjtNum* mass, ipcCon* cand, int ncand, mjtNum* pal,
                              const int* pt2vg, int npt) {
  for (int c=0; c < ncand; c++) {
    ipcCon* con = &cand[c];
    mjtNum mu = ipc_muPair(con, mass);
    mjtNum lam0 = con->lam;
    mjtNum craw = con->ld0 - mjc_standoff(mjc_pairBand(IPC_PAIR(con), rad, ghat), IPC_DELTACAP);
    for (int p=0; p < con->lniv; p++) {
      int v = con->liv[p];
      for (int k=0; k < 3; k++)
        craw += con->lcw[p] * con->ln[k] * (x[3 * v + k] - xfree[3 * v + k]);
    }
    if (craw - lam0 / mu > 0) {  // inactive
      con->lam = 0;
      con->cnt += (con->cnt >= 0) ? 1 : -1;
    } else {  // active
      con->lam = lam0 - craw * mu;
      con->cnt = (con->cnt == 0 || con->cnt > 5) ? 0 : -1;
    }
  }
  for (int p=0; p < npt; p++)
    pal[pt2vg[p]] = 0;
  for (int c=0; c < ncand; c++) {
    if (cand[c].lam <= 0) continue;
    int vv[4], nvv;
    nvv = mjc_pairVerts(vv, IPC_PAIR(&cand[c]));
    for (int q=0; q < nvv; q++)
      if (cand[c].lam > pal[pt2vg[vv[q]]]) pal[pt2vg[vv[q]]] = cand[c].lam;
  }
}


// Merge the fresh candidates into the persistent active set: an existing pair is kept while it
// carries load or has not yet aged out (see below), and a candidate is added iff it was admitted
// (cadmit) and is not already present. New entries warm-start lam and the age from the per-vertex
// stores; the linearization fields are refreshed by the next outer iteration.
static void ipc_mergeActiveSet(mjData* d, ipcCon** aset, int* naset, const mjcFlexPair* cand,
                               int ncand, const int* cadmit, const mjtNum* pal, const int* conage,
                               const int* pt2vg) {
  // the merged set goes to a fresh arena block that replaces *aset, the old block being left to
  // the arena's next reset; its size is bounded by the old set plus the admitted candidates, both
  // known here, so the block never overflows
  int bound = *naset;
  for (int c=0; c < ncand; c++) bound += cadmit[c] ? 1 : 0;
  ipcCon* amerge = NULL;
  if (bound) {
    amerge = (ipcCon*)mj_arenaAllocByte(d, bound * sizeof(ipcCon), _Alignof(ipcCon));
    if (!amerge) {
      mjERROR("arena too small for the IPC active set (%d pairs): increase the model's memory",
              bound);
    }
  }
  const ipcCon* old = *aset;
  // presence hash over the merged set, open addressing keyed by pairHash and verified by
  // pairEqual (mtab: 1-based index into amerge, 0 = empty), per-call scratch on the mjData stack
  // sized by the merged-set bound: only the old set and the admitted candidates are inserted
  int cap = 1;
  while (cap < 2 * bound + 16)
    cap <<= 1;
  mj_markStack(d);
  uint64_t* mkey = mjSTACKALLOC(d, cap, uint64_t);
  int* mtab = mjSTACKALLOC(d, cap, int);
  uint64_t mask = (uint64_t)(cap - 1);
  for (int i=0; i < cap; i++)
    mtab[i] = 0;
  int nm = 0;
  // 1) keep existing pairs that still carry load
  for (int c=0; c < *naset; c++) {
    // A pair that has never carried load (cnt >= 0) is still raising its multiplier from zero, so
    // lam <= 0 cannot tell it from a finished pair: an engaged pair (cnt < 0) is kept for IPC_EVICT
    // quiet updates with its stiffness decaying (the paper removes by decay, not on the multiplier
    // reaching zero), and never-engaged pairs age out on their own counter.
    if ((old[c].lam <= 0 && old[c].cnt < -IPC_EVICT) || old[c].cnt > IPC_ASET_AGE) continue;
    uint64_t key = ipc_pairHash(IPC_PAIR(&old[c])), h = key & mask;
    while (mtab[h] != 0) {
      if (mkey[h] == key && ipc_pairEqual(IPC_PAIR(&amerge[mtab[h]-1]), IPC_PAIR(&old[c]))) break;
      h = (h + 1) & mask;
    }
    if (mtab[h] != 0) continue;
    mkey[h] = key;
    amerge[nm++] = old[c];
    mtab[h] = nm;
  }
  // 2) add new admitted broad-phase candidates not already present
  for (int c=0; c < ncand; c++) {
    if (!cadmit[c]) continue;
    uint64_t key = ipc_pairHash(IPC_PAIR(&cand[c])), h = key & mask;
    while (mtab[h] != 0) {
      if (mkey[h] == key && ipc_pairEqual(IPC_PAIR(&amerge[mtab[h]-1]), &cand[c])) break;
      h = (h + 1) & mask;
    }
    if (mtab[h] != 0) continue;  // dedup: pair already in the merged set
    ipcCon con = {0};
    con.type = cand[c].type;
    for (int q=0; q < 4; q++) con.idx[q] = cand[c].idx[q];
    con.gi = cand[c].g;
    int vv[4], nvv;
    nvv = mjc_pairVerts(vv, IPC_PAIR(&con));  // warm start from the per-vertex stores
    mjtNum s = 0;
    for (int q=0; q < nvv; q++)
      if (pal[pt2vg[vv[q]]] > s) s = pal[pt2vg[vv[q]]];
    con.lam = s;
    con.cnt = ipc_conAge(&con, conage, pt2vg);
    mkey[h] = key;
    amerge[nm++] = con;
    mtab[h] = nm;
  }
  *aset = amerge;
  *naset = nm;
  mj_freeStack(d);
}


// number of distinct pairs among cand and an optional old set, as deduplicated by the active-set
// merge; allocates from the arena, which is not released until the next reset (for testing only)
int mj_ipcCountDistinct(mjData* d, const mjcFlexPair* cand, int ncand, const mjcFlexPair* old,
                        int nold) {
  int npt = 1;
  for (int c=0; c < ncand; c++) {
    int vv[4], nvv = mjc_pairVerts(vv, &cand[c]);
    for (int q=0; q < nvv; q++)
      if (vv[q] >= npt) npt = vv[q] + 1;
  }
  for (int c=0; c < nold; c++) {
    int vv[4], nvv = mjc_pairVerts(vv, &old[c]);
    for (int q=0; q < nvv; q++)
      if (vv[q] >= npt) npt = vv[q] + 1;
  }
  mj_markStack(d);
  int* cadmit = mjSTACKALLOC(d, ncand, int);
  mjtNum* pal = mjSTACKALLOC(d, npt, mjtNum);
  int* conage = mjSTACKALLOC(d, npt, int);
  int* pt2vg = mjSTACKALLOC(d, npt, int);
  for (int c=0; c < ncand; c++) cadmit[c] = 1;
  for (int p=0; p < npt; p++) {
    pal[p] = 0;
    conage[p] = 0;
    pt2vg[p] = p;
  }
  ipcCon* aset = NULL;
  int naset = 0;
  if (old && nold > 0) {
    aset = mjSTACKALLOC(d, nold, ipcCon);
    naset = nold;
    for (int c=0; c < nold; c++) {
      ipcCon con = {0};
      con.type = old[c].type;
      for (int q=0; q < 4; q++) con.idx[q] = old[c].idx[q];
      con.gi = old[c].g;
      con.lam = 0;
      con.cnt = 0;
      aset[c] = con;
    }
  }
  ipc_mergeActiveSet(d, &aset, &naset, cand, ncand, cadmit, pal, conage, pt2vg);
  mj_freeStack(d);
  return naset;
}


// The merit is the inner solve's own objective at a trial state, in position units: the Gauss
// term 0.5*h^2*(a - a*)' M^ (a - a*) over every dof, with M^ the effective metric the solve
// minimizes in (mass, implicit damping and stiffness, its rank-1 classes) and a* = qacc_smooth,
// plus the native rows' cost and the contact cost below. One definition, shared with the solver
// through the metric and constraint-cost machinery, so the line search prices exactly the
// problem the solve was handed, the articulated trees included.
static mjtNum ipc_gaussCost(const mjModel* m, mjData* d, const mjtNum* qacc, mjtNum h) {
  int nv = m->nv;
  mj_markStack(d);
  mjtNum* da = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* Mda = mjSTACKALLOC(d, nv, mjtNum);
  mju_sub(da, qacc, d->qacc_smooth, nv);
  mj_mulM(m, d, Mda, da);  // M at q_n, as the solve's
  mj_effMulAdd(m, d, Mda, da, /*flg_contact=*/1);
  mjtNum E = 0.5 * h * h * mju_dot(da, Mda, nv);
  mj_freeStack(d);
  return E;
}


// contact cost of the active set at x, what its rows cost the solve: the one-sided AL quadratic
// of every pair over its linearized gap less the wall
static mjtNum ipc_contactCost(const mjtNum* x, const mjtNum* xfree, const mjtNum* mass,
                              const mjtNum* rad, mjtNum ghat, const ipcCon* acon, int nacon) {
  mjtNum E = 0;
  for (int c=0; c < nacon; c++) {
    const ipcCon* con = &acon[c];
    mjtNum mu = ipc_muPair(con, mass);
    mjtNum craw = con->ld0 - mjc_standoff(mjc_pairBand(IPC_PAIR(con), rad, ghat), IPC_DELTACAP);
    for (int p=0; p < con->lniv; p++) {
      int v = con->liv[p];
      for (int k=0; k < 3; k++)
        craw += con->lcw[p] * con->ln[k] * (x[3 * v + k] - xfree[3 * v + k]);
    }
    mjtNum dd = mjd_flexContactResidual(mu, craw, 0, con->lam);  // the wall at lam/mu
    if (dd < 0) {  // one-sided, as the pair's row: no cost outside the wall
      int cexp = ipc_cntExp(con->cnt);
      mjtNum scale = mu;
      for (int e=0; e < cexp; e++)
        scale *= IPC_DECAY;
      E += 0.5 * scale * dd * dd;
    }
  }
  return E;
}


// residual of a pair's row at x, in displacement units: the linearized gap less the wall.
// Negative means the row costs at x; the working set is seeded and rechecked on this
static mjtNum ipc_pairResidual(const ipcCon* con, const mjtNum* x, const mjtNum* xfree,
                               const mjtNum* mass, const mjtNum* rad, mjtNum ghat) {
  mjtNum r = con->ld0 - mjc_standoff(mjc_pairBand(IPC_PAIR(con), rad, ghat), IPC_DELTACAP) -
             con->lam / ipc_muPair(con, mass);
  for (int p=0; p < con->lniv; p++) {
    int v = con->liv[p];
    for (int k=0; k < 3; k++)
      r += con->lcw[p] * con->ln[k] * (x[3 * v + k] - xfree[3 * v + k]);
  }
  return r;
}


// The step's state is one generalized tangent over the model's dofs, carried as the velocity it
// ends the step with, w = v_n + h*a, so that the displacement over the step is h*w: nonzero on
// the dofs the step owns (the slide dofs of the free flex vertices and the dofs of the appended
// articulated trees), v_n elsewhere. Every other representation derives from it: the QP variable
// (ipc_tangentQacc) and the world points the contact geometry and the CCD work on (ipc_points);
// the accepted tangent is what the step commits, through the integrator's own two operations,
// qvel = w and qpos = q_n (+) h*w, so a step without contact reproduces the discrete integrator
// to the bit.

// the QP variable of a tangent: qacc = (w - v_n)/h on the owned dofs, qacc_smooth on the rest.
// Shared so every merit term prices one point.
static void ipc_tangentQacc(const mjModel* m, const mjData* d, const mjtNum* w, const char* own,
                            mjtNum h, mjtNum* qacc) {
  mju_copy(qacc, d->qacc_smooth, m->nv);
  for (int i=0; i < m->nv; i++)
    if (own[i]) qacc[i] = (w[i] - d->qvel[i]) / h;
}


static void ipc_pinnedPos(const mjModel* m, const mjData* d, int npt, const int* pjnv,
                          const int* pt2vg, mjtNum* x);

// the trees' configuration of a tangent, q_n (+) h*w over the articulated dofs, with forward
// kinematics, left in d->qpos and the kinematic fields (M stays at q_n)
static void ipc_treePose(const mjModel* m, mjData* d, const mjtNum* w, const mjtNum* qn,
                         mjtNum h, const char* own, const char* isflexdof, mjtNum* tmp) {
  for (int i=0; i < m->nv; i++)
    tmp[i] = (own[i] && !isflexdof[i]) ? w[i] : 0;
  mju_copy(d->qpos, qn, m->nq);
  mj_integratePos(m, d->qpos, tmp, h);
  mj_kinematics(m, d);
}


// world points of a tangent: a free vertex from its slide dofs through its body frame, a point
// pinned to an articulated body by forward kinematics at the trees' configuration, a point
// pinned to a static body where it is
static void ipc_points(const mjModel* m, mjData* d, const mjtNum* w, const mjtNum* xold,
                       const mjtNum* qn, mjtNum h, int npt, const int* fidx, const int* dofadr,
                       const int* pbody, const int* pjnv, const int* pt2vg, int npa,
                       const char* own, const char* isflexdof, mjtNum* tmp, mjtNum* x) {
  for (int v=0; v < npt; v++) {
    if (fidx[v] >= 0) {
      const mjtNum* R = d->xmat + 9 * pbody[v];  // slide dofs are in the body frame
      mjtNum dw[3];
      mju_mulMatVec3(dw, R, w + dofadr[v]);
      for (int c=0; c < 3; c++) x[3 * v + c] = xold[3 * v + c] + h * dw[c];
    } else {
      for (int c=0; c < 3; c++) x[3 * v + c] = xold[3 * v + c];
    }
  }
  if (npa) {
    ipc_treePose(m, d, w, qn, h, own, isflexdof, tmp);
    ipc_pinnedPos(m, d, npt, pjnv, pt2vg, x);
  }
}


// The active pairs as contact rows of the constraint solver, appended after the native
// rows for one solve. A row's Jacobian is the pair's gap gradient over its vertex dofs, in the
// h^2-scaled units of the reference; its stiffness is the AL penalty; its reference puts the wall
// at lam/mu, so the solver's state test (J*qacc < aref) is the AL activity test.
// The arrays live on the caller's stack frame and the mjData pointers are swapped in and restored
// around the solve; the arrays the solver does not read (position, margin, impedance) are not built.
typedef struct {
  int nefc, nJ;
  int *type, *id, *rownnz, *rowadr, *rowsuper, *colind, *state;
  mjtNum *J, *D, *R, *floss, *aref, *force, *b;
} ipcEfcRows;

static void ipc_efcPublish(const mjModel* m, mjData* d, ipcEfcRows* saved, int npair,
                           const int* esnpt, const int* esbase, const mjtNum* esw,
                           const mjtNum* esD, const mjtNum* esref, const int* esxadr,
                           const int* esxnum, const int* esxdof, const mjtNum* esxval) {
  int nv = m->nv, sparse = mj_isSparse(m);
  int nefc0 = d->nefc, nefc1 = nefc0 + npair, nJ0 = d->nJ, nJnew = 0;
  for (int c=0; c < npair; c++) nJnew += 3 * esnpt[c] + esxnum[c];
  int nJ1 = sparse ? nJ0 + nJnew : nefc1 * nv;
  saved->nefc = nefc0;          saved->nJ = nJ0;
  saved->type = d->efc_type;    saved->id = d->efc_id;
  saved->rownnz = d->efc_J_rownnz;  saved->rowadr = d->efc_J_rowadr;
  saved->rowsuper = d->efc_J_rowsuper;  saved->colind = d->efc_J_colind;
  saved->state = d->efc_state;  saved->J = d->efc_J;
  saved->D = d->efc_D;          saved->R = d->efc_R;
  saved->floss = d->efc_frictionloss;  saved->aref = d->efc_aref;
  saved->force = d->efc_force;  saved->b = d->efc_b;
  int n1 = nefc1 > 0 ? nefc1 : 1, nJa = nJ1 > 0 ? nJ1 : 1;
  int* type = mjSTACKALLOC(d, n1, int);
  int* id = mjSTACKALLOC(d, n1, int);
  int* rownnz = mjSTACKALLOC(d, n1, int);
  int* rowadr = mjSTACKALLOC(d, n1, int);
  int* rowsuper = mjSTACKALLOC(d, n1, int);
  int* colind = mjSTACKALLOC(d, sparse ? nJa : 1, int);
  int* state = mjSTACKALLOC(d, n1, int);
  mjtNum* J = mjSTACKALLOC(d, nJa, mjtNum);
  mjtNum* D = mjSTACKALLOC(d, n1, mjtNum);
  mjtNum* R = mjSTACKALLOC(d, n1, mjtNum);
  mjtNum* floss = mjSTACKALLOC(d, n1, mjtNum);
  mjtNum* aref = mjSTACKALLOC(d, n1, mjtNum);
  mjtNum* force = mjSTACKALLOC(d, n1, mjtNum);
  mjtNum* b = mjSTACKALLOC(d, n1, mjtNum);
  // the native rows, as they are
  if (nefc0) {
    memcpy(type, d->efc_type, nefc0 * sizeof(int));
    memcpy(id, d->efc_id, nefc0 * sizeof(int));
    memcpy(state, d->efc_state, nefc0 * sizeof(int));
    memcpy(D, d->efc_D, nefc0 * sizeof(mjtNum));
    memcpy(R, d->efc_R, nefc0 * sizeof(mjtNum));
    memcpy(floss, d->efc_frictionloss, nefc0 * sizeof(mjtNum));
    memcpy(aref, d->efc_aref, nefc0 * sizeof(mjtNum));
    memcpy(force, d->efc_force, nefc0 * sizeof(mjtNum));
    if (sparse) {
      memcpy(rownnz, d->efc_J_rownnz, nefc0 * sizeof(int));
      memcpy(rowadr, d->efc_J_rowadr, nefc0 * sizeof(int));
      memcpy(rowsuper, d->efc_J_rowsuper, nefc0 * sizeof(int));
      if (nJ0) {
        memcpy(colind, d->efc_J_colind, nJ0 * sizeof(int));
        memcpy(J, d->efc_J, nJ0 * sizeof(mjtNum));
      }
    } else {
      memcpy(J, d->efc_J, (size_t)nefc0 * nv * sizeof(mjtNum));
    }
  }
  // the pairs: one row each, column indices ascending within the row
  int adr = nJ0;
  for (int c=0; c < npair; c++) {
    int r = nefc0 + c, n = 3 * esnpt[c] + esxnum[c];
    type[r] = mjCNSTR_CONTACT_FRICTIONLESS;
    id[r] = -1;  // no mjContact behind an AL pair
    state[r] = mjCNSTRSTATE_SATISFIED;
    force[r] = 0;
    D[r] = esD[c];
    R[r] = 1 / esD[c];
    floss[r] = 0;
    aref[r] = esref[c];
    if (sparse) {
      rowadr[r] = adr;
      rowsuper[r] = 0;
      for (int q=0; q < esnpt[c]; q++)
        for (int k=0; k < 3; k++) {
          colind[adr + 3*q + k] = esbase[IPC_NPT*c + q] + k;
          J[adr + 3*q + k] = esw[3*IPC_NPT*c + 3*q + k];
        }
      for (int i=0; i < esxnum[c]; i++) {  // the body-chain columns of pinned corners
        colind[adr + 3*esnpt[c] + i] = esxdof[esxadr[c] + i];
        J[adr + 3*esnpt[c] + i] = esxval[esxadr[c] + i];
      }
      for (int i=1; i < n; i++) {  // insertion sort of (colind, J); n is a few dozen at most
        int ci = colind[adr + i], j = i - 1;
        mjtNum vi = J[adr + i];
        while (j >= 0 && colind[adr + j] > ci) {
          colind[adr + j + 1] = colind[adr + j];
          J[adr + j + 1] = J[adr + j];
          j--;
        }
        colind[adr + j + 1] = ci;
        J[adr + j + 1] = vi;
      }
      // merge repeated columns: the chains of two pinned corners on one body overlap
      int w = 0;
      for (int i=0; i < n; i++) {
        if (w > 0 && colind[adr + w - 1] == colind[adr + i]) {
          J[adr + w - 1] += J[adr + i];
        } else {
          colind[adr + w] = colind[adr + i];
          J[adr + w] = J[adr + i];
          w++;
        }
      }
      rownnz[r] = w;
      adr += w;
    } else {
      mjtNum* Jr = J + (size_t)r * nv;
      mju_zero(Jr, nv);
      for (int q=0; q < esnpt[c]; q++)
        for (int k=0; k < 3; k++)
          Jr[esbase[IPC_NPT*c + q] + k] = esw[3*IPC_NPT*c + 3*q + k];
      for (int i=0; i < esxnum[c]; i++)
        Jr[esxdof[esxadr[c] + i]] += esxval[esxadr[c] + i];
    }
  }
  d->efc_type = type;         d->efc_id = id;
  d->efc_J_rownnz = rownnz;   d->efc_J_rowadr = rowadr;
  d->efc_J_rowsuper = rowsuper;  d->efc_J_colind = colind;
  d->efc_state = state;       d->efc_J = J;
  d->efc_D = D;               d->efc_R = R;
  d->efc_frictionloss = floss;  d->efc_aref = aref;
  d->efc_force = force;       d->efc_b = b;
  d->nefc = nefc1;
  d->nJ = sparse ? adr : nJ1;
}


// generalized force of the published rows, J'*f over the rows appended after the native nefc0
static void ipc_efcRowForce(const mjModel* m, const mjData* d, int nefc0, mjtNum* res) {
  int nv = m->nv;
  mju_zero(res, nv);
  if (mj_isSparse(m)) {
    for (int r=nefc0; r < d->nefc; r++) {
      mjtNum f = d->efc_force[r];
      if (!f) continue;
      int adr = d->efc_J_rowadr[r], nnz = d->efc_J_rownnz[r];
      for (int a=0; a < nnz; a++)
        res[d->efc_J_colind[adr + a]] += d->efc_J[adr + a] * f;
    }
  } else {
    for (int r=nefc0; r < d->nefc; r++) {
      mjtNum f = d->efc_force[r];
      if (f) mju_addToScl(res, d->efc_J + (size_t)r * nv, f, nv);
    }
  }
}


static void ipc_efcRestore(mjData* d, const ipcEfcRows* saved) {
  d->efc_type = saved->type;         d->efc_id = saved->id;
  d->efc_J_rownnz = saved->rownnz;   d->efc_J_rowadr = saved->rowadr;
  d->efc_J_rowsuper = saved->rowsuper;  d->efc_J_colind = saved->colind;
  d->efc_state = saved->state;       d->efc_J = saved->J;
  d->efc_D = saved->D;               d->efc_R = saved->R;
  d->efc_frictionloss = saved->floss;  d->efc_aref = saved->aref;
  d->efc_force = saved->force;       d->efc_b = saved->b;
  d->nefc = saved->nefc;
  d->nJ = saved->nJ;
}


// merit cost of the efc rows: exactly what the QP minimizes over them, evaluated at the trial
// acceleration by mj_constraintUpdate and put in position units like every other merit term
static mjtNum ipc_efcCost(const mjModel* m, mjData* d, const mjtNum* qacc, mjtNum h,
                          int nefc_qp) {
  int nefc = nefc_qp;  // the rows the QP minimized
  if (!nefc) return 0;
  mj_markStack(d);
  mjtNum* jar = mjSTACKALLOC(d, nefc, mjtNum);
  int savednefc = d->nefc;
  d->nefc = nefc;  // mj_mulJacVec/mj_constraintUpdate read d->nefc internally
  mj_mulJacVec(m, d, jar, qacc);
  mju_subFrom(jar, d->efc_aref, nefc);
  mjtNum cost = 0;
  mj_constraintUpdate(m, d, jar, &cost, 0);
  d->nefc = savednefc;
  mj_freeStack(d);
  return h * h * cost;
}


// the merit at a trial state (tangent wt, its world points xt): the solve's objective
static mjtNum ipc_merit(const mjModel* m, mjData* d, const mjtNum* xt, const mjtNum* wt,
                        const char* own, const mjtNum* xfree, mjtNum h, int nefc_qp,
                        const mjtNum* mass, const mjtNum* rad, mjtNum ghat, const ipcCon* acon,
                        int nacon) {
  mj_markStack(d);
  mjtNum* qacc = mjSTACKALLOC(d, m->nv, mjtNum);
  ipc_tangentQacc(m, d, wt, own, h, qacc);
  mjtNum E = ipc_gaussCost(m, d, qacc, h) + ipc_efcCost(m, d, qacc, h, nefc_qp) +
             ipc_contactCost(xt, xfree, mass, rad, ghat, acon, nacon);
  mj_freeStack(d);
  return E;
}


// The step: minimize the incremental potential over the flex vertices and the appended
// articulated trees with penetration-free contact. Each outer iteration linearizes the active
// pairs at the committed position xfree, runs one discrete solve with the pairs published as the
// metric's contact class, line-searches the result on the merit, updates the multipliers, and
// advances xfree along the iterate as far as conservative CCD allows. Covers flex self-contact
// (vertex-triangle and edge-edge) and flex-vs-static-geom; rigid contact stays on the native
// constraint rows.

// world positions of the pinned points that ride articulated bodies (pjnv > 0) from the current
// body poses, placed as mj_flex places them
static void ipc_pinnedPos(const mjModel* m, const mjData* d, int npt, const int* pjnv,
                          const int* pt2vg, mjtNum* x) {
  for (int v=0; v < npt; v++) {
    if (pjnv[v] <= 0) continue;
    int vg = pt2vg[v], b = m->flex_vertbodyid[vg];
    mju_mulMatVec3(x + 3 * v, d->xmat + 9 * b, m->flex_vert + 3 * vg);
    mju_addTo3(x + 3 * v, d->xpos + 3 * b);
  }
}


// per-pair gap loops on the engine thread pool: either the active set's linearization (con != NULL:
// ld0, ln, liv, lcw, lniv of each pair) or the candidates' gaps (cand != NULL: gap[c]); each task
// covers a disjoint range of pairs, so the result is identical to the serial run
typedef struct {
  ipcCon* con;
  const mjcFlexPair* cand;
  mjtNum* gap;
  int n, chunk;
  const mjtNum* x;
  const mjtNum* gv;
  const mjtNum* ge;
  const mjtNum* rad;
  mjtNum ghat;
} ipcGapArg;

static void ipc_gapTask(const mjModel* m, mjData* d, void* arg, int thread_id, int task_id) {
  (void)thread_id;
  const ipcGapArg* a = (const ipcGapArg*)arg;
  int c0 = task_id * a->chunk, c1 = c0 + a->chunk < a->n ? c0 + a->chunk : a->n;
  for (int c=c0; c < c1; c++) {
    if (a->con) {
      ipcCon* w = a->con + c;
      w->ld0 = mjc_pairGap(IPC_PAIR(w), m, d, a->x, a->gv, a->ge, a->rad, w->ln, w->liv, w->lcw,
                           &w->lniv, a->ghat);
    } else {
      mjtNum nn[3], cw[4];
      int idv[4], ni;
      a->gap[c] = mjc_pairGap(a->cand + c, m, d, a->x, a->gv, a->ge, a->rad, nn, idv, cw, &ni,
                              a->ghat);
    }
  }
}

// run the gap loop described by a (chunk is set here) on d's thread pool
static void ipc_gapLoop(const mjModel* m, mjData* d, ipcGapArg* a) {
  int nthread = mju_numThread(d);
  if (nthread < 2 || a->n <= 0) {
    a->chunk = a->n;
    if (a->n > 0) ipc_gapTask(m, d, a, 0, 0);
    return;
  }
  int ntask = 8 * nthread;
  a->chunk = (a->n + ntask - 1) / ntask;
  if (a->chunk < 256) a->chunk = 256;
  mju_dispatch(m, d, ipc_gapTask, a, (a->n + a->chunk - 1) / a->chunk);
}


void mj_ipc(const mjModel* m, mjData* d) {
  mjtNum h = m->opt.timestep;
  // every dim-2 flex takes part; their vertices are concatenated in flex order and fxadr[k] is
  // the offset of flex flist[k]. A model with no dim-2 flex takes the same path with an empty
  // flex block: the appended articulated trees carry it, on the native efc rows.
  int nfd = 0;
  for (int i=0; i < m->nflex; i++)
    if (m->flex_dim[i] == 2) nfd++;

  // all per-step scratch lives on the mjData stack from here to the mj_freeStack at the end
  mj_markStack(d);
  int* flist = mjSTACKALLOC(d, (nfd > 0 ? nfd : 1), int);  // the dim-2 flex ids
  int* fxadr = mjSTACKALLOC(d, (nfd > 0 ? nfd : 1), int);  // free-point offset of each dim-2 flex
  int nfv = 0;                                       // total dim-2 flex verts (all flexes)
  for (int i=0, k = 0; i < m->nflex; i++)
    if (m->flex_dim[i] == 2) {
      flist[k] = i;
      fxadr[k] = nfv;
      nfv += m->flex_vertnum[i];
      k++;
    }
  // detection band; per-pair narrowing happens in mjc_pairBand and mjc_candidates
  mjtNum ghat = IPC_GHAT;
  // free point -> global flex vertex, and -> dim-2 flex slot
  int* pt2vg = mjSTACKALLOC(d, (nfv > 0 ? nfv : 1), int);
  int* pt2flex = mjSTACKALLOC(d, (nfv > 0 ? nfv : 1), int);
  for (int k=0; k < nfd; k++) {
    int va_k = m->flex_vertadr[flist[k]], nv_k = m->flex_vertnum[flist[k]];
    for (int lv=0; lv < nv_k; lv++) {
      pt2vg[fxadr[k] + lv] = va_k + lv;
      pt2flex[fxadr[k] + lv] = k;
    }
  }

  // the point array holds flex vertices only; every other body with dofs is an appended
  // articulated tree (below), a body carrying pinned vertices included: those follow it
  char* isflexvert = mjSTACKALLOC(d, (m->nbody > 0 ? m->nbody : 1), char);
  for (int b=0; b < m->nbody; b++)
    isflexvert[b] = 0;
  int npt = nfv;
  int* dofadr = mjSTACKALLOC(d, (npt > 0 ? npt : 1), int);
  int* fidx = mjSTACKALLOC(d, (npt > 0 ? npt : 1), int);   // free-point index, -1 when pinned
  mjtNum* mass = mjSTACKALLOC(d, (npt > 0 ? npt : 1), mjtNum);
  mjtNum* rad = mjSTACKALLOC(d, (npt > 0 ? npt : 1), mjtNum);
  int* pbody = mjSTACKALLOC(d, (npt > 0 ? npt : 1), int);  // body, for the slide-frame rotation
  // vertices per body over every flex: a body that carries more than one is a pinned attachment
  // whatever its joints, and its dofs are shared by the points it carries
  int* nvertbody = mjSTACKALLOC(d, (m->nbody > 0 ? m->nbody : 1), int);
  for (int b=0; b < m->nbody; b++)
    nvertbody[b] = 0;
  for (int vg=0; vg < m->nflexvert; vg++)
    if (m->flex_vertbodyid[vg] >= 0) nvertbody[m->flex_vertbodyid[vg]]++;
  int nfree = 0;
  for (int k=0; k < nfd; k++) {
    int fi = flist[k];
    mjtNum rk = m->flex_radius[fi];
    int va_k = m->flex_vertadr[fi], nv_k = m->flex_vertnum[fi];
    for (int lv=0; lv < nv_k; lv++) {
      int v = fxadr[k] + lv, bid = m->flex_vertbodyid[va_k + lv];
      dofadr[v] = -1;
      fidx[v] = -1;
      mass[v] = 0;
      rad[v] = rk;
      pbody[v] = bid;
      // a free vertex is one whose body IS the vertex: the three slide joints a flex vertex body
      // carries, exactly one vertex on it, sitting at its origin. Any other body holding a vertex
      // is a pinned attachment and takes the articulated path once for all the points it carries
      // (a ball joint has three dofs too; a cloth pinned to a body on slides shares that body's
      // dofs among its vertices)
      int ja = m->body_jntadr[bid];
      const mjtNum* lp = m->flex_vert + 3 * (va_k + lv);
      int slides = m->body_jntnum[bid] == 3 && m->jnt_type[ja] == mjJNT_SLIDE &&
                   m->jnt_type[ja + 1] == mjJNT_SLIDE && m->jnt_type[ja + 2] == mjJNT_SLIDE &&
                   nvertbody[bid] == 1 && lp[0] == 0 && lp[1] == 0 && lp[2] == 0;
      if (slides) {
        int da = m->body_dofadr[bid];
        dofadr[v] = da;
        fidx[v] = nfree++;
        mass[v] = mjd_flexVertMass(m, d, va_k + lv);
        isflexvert[bid] = 1;
      }
    }
  }
  // articulated trees: every tree with dofs that is not a flex vertex, integrated with the flex
  // block through the same tangent
  int ntree = m->ntree, na_artic = 0;
  char* isartictree = mjSTACKALLOC(d, (ntree > 0 ? ntree : 1), char);
  for (int t=0; t < ntree; t++)
    isartictree[t] = 0;
  for (int b=1; b < m->nbody; b++) {
    if (m->body_dofnum[b] == 0 || isflexvert[b]) continue;
    isartictree[m->dof_treeid[m->body_dofadr[b]]] = 1;
  }
  int* atid = mjSTACKALLOC(d, (ntree > 0 ? ntree : 1), int);  // articulated tree ids
  for (int t=0; t < ntree; t++)
    if (isartictree[t]) atid[na_artic++] = t;
  // the dofs the step owns: the free vertices' slides (the flex block) and the trees' dofs
  int nvv = (m->nv > 0 ? m->nv : 1), nown = 0;
  char* own = mjSTACKALLOC(d, nvv, char);
  char* isflexdof = mjSTACKALLOC(d, nvv, char);
  for (int i=0; i < m->nv; i++)
    own[i] = isflexdof[i] = 0;
  for (int v=0; v < npt; v++)
    if (fidx[v] >= 0)
      for (int c=0; c < 3; c++)
        own[dofadr[v] + c] = isflexdof[dofadr[v] + c] = 1;
  for (int a=0; a < na_artic; a++)
    for (int i=0; i < m->tree_dofnum[atid[a]]; i++)
      own[m->tree_dofadr[atid[a]] + i] = 1;
  for (int i=0; i < m->nv; i++)
    nown += own[i];
  int nstate = nfv;
  // Pinned points on articulated bodies ride them: their positions come from FK at every
  // configuration the step evaluates, and a pair they take part in reaches the body's dofs through
  // the point Jacobian, taken at q_n like the trees' tangents. Per point: the dof chain (pjdof,
  // pjnv entries from pjadr), the 3 x chain Jacobian (pjac) and the point's velocity (pvel). A tree
  // that carries such points advances with the cloth's CCD fraction (apinned).
  int* pjnv = mjSTACKALLOC(d, (npt > 0 ? npt : 1), int);
  int* pjadr = mjSTACKALLOC(d, (npt > 0 ? npt : 1), int);
  char* apinned = mjSTACKALLOC(d, (ntree > 0 ? ntree : 1), char);
  for (int t=0; t < ntree; t++)
    apinned[t] = 0;
  int* chain = mjSTACKALLOC(d, (m->nv > 0 ? m->nv : 1), int);
  int npj = 0, npa = 0, pjmax = 0;
  for (int v=0; v < npt; v++) {
    pjnv[v] = 0;
    pjadr[v] = -1;
    int t = m->body_treeid[pbody[v]];
    if (fidx[v] >= 0) {
      // a free vertex whose slide frame moves with a jointed tree is not supported: the flex
      // block integrates it in world coordinates with its frame fixed
      if (t >= 0 && isartictree[t]) {
        const char* fn = mj_id2name(m, mjOBJ_FLEX, flist[pt2flex[v]]);
        mj_freeStack(d);
        mjERROR("IPC mode: flex '%s' has free vertices under a jointed body; pin them or attach "
                "the flex to a static body", fn ? fn : "");
      }
      continue;
    }
    if (t < 0 || !isartictree[t]) continue;  // pinned to a static body: fixed
    pjnv[v] = mj_bodyChain(m, pbody[v], chain);
    if (pjnv[v] <= 0) continue;
    // the CCD sweeps straight segments, which a translating chain moves its points on; a
    // rotation moves them on arcs, so a hinge, ball or free joint on the chain is refused
    for (int j=0; j < pjnv[v]; j++) {
      int jid = m->dof_jntid[chain[j]];
      if (m->jnt_type[jid] != mjJNT_SLIDE) {
        const char* jn = mj_id2name(m, mjOBJ_JOINT, jid);
        const char* fn = mj_id2name(m, mjOBJ_FLEX, flist[pt2flex[v]]);
        const char* bn = mj_id2name(m, mjOBJ_BODY, pbody[v]);
        mj_freeStack(d);
        mjERROR("IPC mode: flex '%s' is pinned to body '%s', whose chain moves on joint '%s' (id "
                "%d), not a slide; pin flexes to static bodies or to bodies on slide joints only",
                fn ? fn : "", bn ? bn : "", jn ? jn : "", jid);
      }
    }
    apinned[t] = 1;
    npj += pjnv[v];
    npa++;
    if (pjnv[v] > pjmax) pjmax = pjnv[v];
  }
  // a point riding a moving body has no vertex mass of its own (a pinned vertex carries none), so
  // its pairs' stiffness scales with the body's subtree mass shared among the points it carries
  if (npa) {
    int* npin = mjSTACKALLOC(d, m->nbody, int);
    for (int b=0; b < m->nbody; b++) npin[b] = 0;
    for (int v=0; v < npt; v++)
      if (pjnv[v] > 0) npin[pbody[v]]++;
    for (int v=0; v < npt; v++)
      if (pjnv[v] > 0) mass[v] = m->body_subtreemass[pbody[v]] / npin[pbody[v]];
  }
  int* pjdof = mjSTACKALLOC(d, (npj > 0 ? npj : 1), int);
  mjtNum* pjac = mjSTACKALLOC(d, (npj > 0 ? 3 * npj : 1), mjtNum);
  mjtNum* pvel = mjSTACKALLOC(d, (npt > 0 ? 3 * npt : 1), mjtNum);
  for (int v=0, adr = 0; v < npt; v++) {
    if (pjnv[v] <= 0) continue;
    int NV = mj_bodyChain(m, pbody[v], chain);
    pjadr[v] = adr;
    for (int j=0; j < NV; j++) pjdof[adr + j] = chain[j];
    mj_jacSparse(m, d, pjac + 3 * adr, NULL, d->flexvert_xpos + 3 * pt2vg[v], pbody[v], NV,
                 chain, 0);
    for (int k=0; k < 3; k++) {
      mjtNum s = 0;
      for (int j=0; j < NV; j++) s += pjac[3 * adr + k * NV + j] * d->qvel[chain[j]];
      pvel[3 * v + k] = s;
    }
    adr += NV;
  }
  // the persistent active set, maintained across outer iterations by ipc_mergeActiveSet; the
  // injected rows and the merit are both built from it. It lives on the arena, sized exactly:
  // seeded from the first broad phase below, then replaced by each merge
  ipcCon* aset = NULL;
  int naset = 0;
  // static-geom sharp features (vertices, edges), collected once per step
  int gvcap = 1, gecap = 1;
  for (int gi=0; gi < m->ngeom; gi++) {
    if (m->geom_contype[gi] == 0 && m->geom_conaffinity[gi] == 0) continue;
    if (m->body_weldid[m->geom_bodyid[gi]] != 0) continue;  // static geoms only
    int type = m->geom_type[gi];
    if (type == mjGEOM_SPHERE) {
      gvcap += 1;  // the centre
    } else if (type == mjGEOM_CAPSULE) {
      gvcap += 2;  // the axis endpoints
      gecap += 1;  // the axis
    } else if (type == mjGEOM_BOX) {
      gvcap += 8;
      gecap += 12;
    } else if (type == mjGEOM_MESH) {
      int mid = m->geom_dataid[gi];
      gvcap += m->mesh_vertnum[mid];
      int pa = m->mesh_polyadr[mid], pn = m->mesh_polynum[mid];
      for (int p=0; p < pn; p++)
        gecap += m->mesh_polyvertnum[pa + p];  // upper bound (pre-dedup)
    }
  }
  int ngv = 0, nge = 0;
  mjtNum* gv = mjSTACKALLOC(d, 3 * gvcap, mjtNum);
  mjtNum* ge = mjSTACKALLOC(d, 6 * gecap, mjtNum);
  int* gvgeom = mjSTACKALLOC(d, gvcap, int);  // the owning geom of each feature (its masks)
  int* gegeom = mjSTACKALLOC(d, gecap, int);
  for (int gi=0; gi < m->ngeom; gi++) {
    if (m->geom_contype[gi] == 0 && m->geom_conaffinity[gi] == 0) continue;
    if (m->body_weldid[m->geom_bodyid[gi]] != 0) continue;  // static geoms only
    int nv0 = ngv, ne0 = nge;
    ngv += mjc_GeomVerts(m, gi, d->geom_xpos + 3 * gi, d->geom_xmat + 9 * gi, gv + 3 * ngv);
    nge += mjc_GeomEdges(m, gi, d->geom_xpos + 3 * gi, d->geom_xmat + 9 * gi, ge + 6 * nge);
    for (int c=nv0; c < ngv; c++) gvgeom[c] = gi;
    for (int c=ne0; c < nge; c++) gegeom[c] = gi;
  }
  // the tangents over the model's dofs, as end velocities: the iterate w, the committed wfree,
  // the free-flight predictor wtil = v + h*qacc_smooth, the line-search trial wn, the proposal
  // wprop, the direction ddw, and a scratch for the trees' configuration. The step starts at
  // rest, w = 0 (no displacement); the dofs it does not own carry v_n and are never moved
  mjtNum* w = mjSTACKALLOC(d, nvv, mjtNum);
  mjtNum* wfree = mjSTACKALLOC(d, nvv, mjtNum);
  mjtNum* wtil = mjSTACKALLOC(d, nvv, mjtNum);
  mjtNum* wn = mjSTACKALLOC(d, nvv, mjtNum);
  mjtNum* wprop = mjSTACKALLOC(d, nvv, mjtNum);
  mjtNum* ddw = mjSTACKALLOC(d, nvv, mjtNum);
  mjtNum* wtmp = mjSTACKALLOC(d, nvv, mjtNum);
  mjtNum* qn_a = mjSTACKALLOC(d, (m->nq > 0 ? m->nq : 1), mjtNum);  // q_n
  mju_copy(qn_a, d->qpos, m->nq);
  for (int i=0; i < m->nv; i++) {
    w[i] = wfree[i] = 0;
    wtil[i] = own[i] ? d->qvel[i] + h * d->qacc_smooth[i] : 0;
  }
  // their world images, sized 3*nstate (flex points 0..nfv-1) and indexed by state slot: the
  // iterate x, the committed intersection-free xfree, the predictor xtil, the trial xn and the
  // proposal xprop, all from xold (the points at q_n) through ipc_points
  mjtNum* x = mjSTACKALLOC(d, 3 * nstate, mjtNum);
  mjtNum* xfree = mjSTACKALLOC(d, 3 * nstate, mjtNum);
  mjtNum* xtil = mjSTACKALLOC(d, 3 * nstate, mjtNum);
  mjtNum* xold = mjSTACKALLOC(d, 3 * nstate, mjtNum);
  mjtNum* xn = mjSTACKALLOC(d, 3 * nstate, mjtNum);
  mjtNum* xprop = mjSTACKALLOC(d, 3 * nstate, mjtNum);
  mjtNum* dxp = mjSTACKALLOC(d, 3 * nstate, mjtNum);    // per-point sweep x - xfree for the CCD
  mjtNum* qfrc_rows = mjSTACKALLOC(d, m->nv, mjtNum);   // the contact rows' force of the last solve
  mju_zero(qfrc_rows, m->nv);
  const mjtNum* vx = d->flexvert_xpos;
  for (int v=0; v < npt; v++)
    for (int c=0; c < 3; c++)
      xold[3 * v + c] = vx[3 * pt2vg[v] + c];
  ipc_points(m, d, wtil, xold, qn_a, h, npt, fidx, dofadr, pbody, pjnv, pt2vg, npa, own, isflexdof,
             wtmp, xtil);
  // start from the committed, collision-free state
  ipc_points(m, d, w, xold, qn_a, h, npt, fidx, dofadr, pbody, pjnv, pt2vg, npa, own, isflexdof,
             wtmp, x);

  mjtNum ih2 = 1.0 / (h * h);
  char* ageload = mjSTACKALLOC(d, (npt > 0 ? npt : 1), char);  // ipc_ageStep temp
  // candidates over the predictor's sweep xold -> xtil: the base reach is the detection band, and
  // mjc_candidates extends every query by the travel of the points involved, so every pair that
  // can close during the step is captured
  mjtNum thresh = 3 * ghat;
  // static features move on the flex side only, so their base reach is the band itself; a wider
  // one on a many-piece convex bin produces hundreds of thousands of candidates
  mjtNum threshGeom = ghat;
  // the candidates live on the arena, pushed one at a time by the broad phase (mjc_candidates),
  // so every query yields an exactly sized array with nothing reserved up front
  mjcFlexPair* cand;
  int ncand = mjc_candidates(m, d, x, gv, ge, gvgeom, gegeom, ngv, nge, rad, thresh, threshGeom,
                             xold, xtil, ghat, nfv, npt, fidx, pjnv, flist, fxadr, nfd, pt2flex,
                             &cand);
  // with no candidate within the margin the predictor is collision-free and a better start
  // than xold
  if (ncand == 0) {
    for (int i=0; i < m->nv; i++)
      if (isflexdof[i]) w[i] = wtil[i];  // the trees keep their zero tangent
    ipc_points(m, d, w, xold, qn_a, h, npt, fidx, dofadr, pbody, pjnv, pt2vg, npa, own, isflexdof,
               wtmp, x);
  }
  // seed the persistent active set from the first broad phase, warm-starting lam and the age
  // from the per-vertex stores
  naset = ncand;
  aset = (ipcCon*)mj_arenaAllocByte(d, (naset ? naset : 1) * sizeof(ipcCon), _Alignof(ipcCon));
  if (!aset) {
    mj_freeStack(d);
    mjERROR("arena too small for the IPC active set (%d pairs): increase the model's memory",
            naset);
  }
  // the set's arena offset: every merge below moves the new set back here and the arena pointer
  // follows, so a step of many outer iterations holds one set and one candidate list at a time
  size_t aset_off = (size_t)((char*)aset - (char*)d->arena);
  for (int c=0; c < naset; c++) {
    ipcCon* a0 = &aset[c];
    memset(a0, 0, sizeof(ipcCon));
    a0->type = cand[c].type;
    for (int q=0; q < 4; q++) a0->idx[q] = cand[c].idx[q];
    a0->gi = cand[c].g;
    {
      int vv[4], npv = mjc_pairVerts(vv, IPC_PAIR(a0));
      mjtNum s = 0;
      for (int q=0; q < npv; q++)
        if (d->flexvert_lambda[pt2vg[vv[q]]] > s) s = d->flexvert_lambda[pt2vg[vv[q]]];
      a0->lam = s;
      a0->cnt = ipc_conAge(a0, d->flexvert_conage, pt2vg);
    }
  }
  for (int i=0; i < 3 * nstate; i++)
    xfree[i] = xold[i];
  // outer loop (the paper's Alg. 1): one discrete solve per iteration; beta accumulates the
  // fraction of the motion the committed position has absorbed
  mjtNum beta = 0.0;
  int inner_cap = 1;           // one solve per outer iteration
  int outer_cap = 1024;
  int stall = 0, stalled = 0;  // consecutive iterations without a feasible advance
  mjtNum last_ls_alpha = 1.0;  // accepted line-search step of the last iteration
  int nefc_qp = 0;             // nefc as the QP saw it; the merit scores exactly these rows
  for (int outer=0; outer < outer_cap && nown > 0; outer++) {  // nown: rigid-only runs too
    ipcCon* wcon = aset;  // the working set is the persistent active set
    int wncon = naset;
    // linearize the pairs at xfree (Eq. 10): c(x) is linear in x for this iteration
    {
      ipcGapArg ga = {wcon, NULL, NULL, wncon, 0, xfree, gv, ge, rad, ghat};
      ipc_gapLoop(m, d, &ga);
    }
    // x persists across outer iterations (the paper's warm start)
    int flex_converged_out = 0, artic_converged_out = 0, newton_converged_out = 0;
    for (int it=0; it < inner_cap && nown > 0; it++) {
      // one set: the injected rows and the merit are built from the same active set, which is
      // already the CCD-admitted one, so no proximity test decides membership
      // (IpcTest.SelfContactConservesMomentum guards the agreement)
      int nacon = wncon;
      // Inner solve: one discrete constraint solve with every active pair published as a
      // frictionless contact row of the constraint solver, whether or not the model also has
      // rigid bodies. In a pure-flex model the native contact rows are dropped, since the pairs
      // replace them; with rigid bodies they stay (rigid contact with friction). A pair's row is
      // unilateral: it costs nothing while the linearized gap is outside the wall and is the AL
      // quadratic inside it, and the solver decides which at every iteration and line-search
      // trial from the row's state, so a pair that is not touching cannot hold or pull. The
      // converged qacc is the search direction for the merit line search below.
      {
        int cap = nacon + 1;  // rows: one per pair
        // per-iteration frame: these arrays are sized by the live pair count
        mj_markStack(d);
        // per pair: dof triples and weights (esbase, esw), stiffness (esD) and reference (esref)
        int* esnpt = mjSTACKALLOC(d, cap, int);
        int* esbase = mjSTACKALLOC(d, IPC_NPT * cap, int);
        mjtNum* esw = mjSTACKALLOC(d, 3 * IPC_NPT * cap, mjtNum);
        mjtNum* esD = mjSTACKALLOC(d, cap, mjtNum);
        mjtNum* esref = mjSTACKALLOC(d, cap, mjtNum);
        // and, for pinned corners on articulated bodies, the body-chain columns (esxdof, esxval;
        // esxnum entries from esxadr), sized by the chains the live pairs' corners actually have
        int xcap = 1;
        if (pjmax > 0) {
          for (int c=0; c < nacon; c++)
            for (int p=0; p < wcon[c].lniv; p++)
              xcap += pjnv[wcon[c].liv[p]];
        }
        int* esxadr = mjSTACKALLOC(d, cap, int);
        int* esxnum = mjSTACKALLOC(d, cap, int);
        int* esxdof = mjSTACKALLOC(d, xcap, int);
        mjtNum* esxval = mjSTACKALLOC(d, xcap, mjtNum);
        mjtNum h2 = 1.0 / ih2;
        // The working set: only the pairs that can push are published. A pair starts in it when
        // its row is violated at the predictor or at the iterate, on the residual of
        // ipc_pairResidual: the linearized gap inside the wall (a pair that carried load sits
        // inside its wall by its force over mu; the multiplier alone does not qualify, since the
        // per-vertex warm start hands it to every pair of a loaded vertex); after each solve
        // every omitted pair is tested at the proposal with the same residual, the violated ones
        // join and the subproblem is solved again, warm-started from the proposal; every round
        // admits at least one pair, so the rounds end. With the linearization and the
        // multipliers fixed, an omitted pair the proposal does not violate has zero energy and
        // gradient there, so the restricted minimizer is the full one; the omitted pairs are
        // then checked once per solve instead of on every CG iteration.
        char* inws = mjSTACKALLOC(d, cap, char);
        for (int c=0; c < nacon; c++) {
          const ipcCon* con = &wcon[c];
          inws[c] = (ipc_pairResidual(con, xtil, xfree, mass, rad, ghat) < 0 ||
                     ipc_pairResidual(con, x, xfree, mass, rad, ghat) < 0);
        }
        mjtNum* ws_entry = mjSTACKALLOC(d, m->nv, mjtNum);
        mju_copy(ws_entry, d->qacc_warmstart, m->nv);
        for (;;) {
        int npair = 0, nx = 0;
        for (int c=0; c < nacon; c++) {
          if (!inws[c]) continue;
          ipcCon* con = &wcon[c];
          mjtNum mu = ipc_muPair(con, mass);
          int cexp = ipc_cntExp(con->cnt);
          mjtNum D = mu;
          for (int e=0; e < cexp; e++)
            D *= IPC_DECAY;
          mjtNum delta = mjc_standoff(mjc_pairBand(IPC_PAIR(con), rad, ghat), IPC_DELTACAP);
          mjtNum refc = -mjd_flexContactResidual(mu, con->ld0 - delta, 0, con->lam);
          int np = 0;
          esxadr[npair] = nx;
          // the reference is exact: the h*Rqv terms cancel per vertex against J*qacc
          for (int p=0; p < con->lniv; p++) {
            int v = con->liv[p];
            if (dofadr[v] < 0) {
              if (pjnv[v] > 0) {  // pinned to an articulated body: its chain's columns
                int NV = pjnv[v], a0 = pjadr[v];
                for (int j=0; j < NV; j++) {
                  mjtNum nj = 0;
                  for (int k=0; k < 3; k++) nj += con->ln[k] * pjac[3 * a0 + k * NV + j];
                  esxdof[nx] = pjdof[a0 + j];
                  esxval[nx] = con->lcw[p] * nj * h2;
                  nx++;
                }
                for (int k=0; k < 3; k++)
                  refc -= con->lcw[p] * con->ln[k] *
                          (xold[3 * v + k] + h * pvel[3 * v + k] - xfree[3 * v + k]);
              }
              continue;  // pinned to a static body: no dofs, and its reference term is zero
            }
            const mjtNum* R = d->xmat + 9 * pbody[v];
            const mjtNum* qv = d->qvel + dofadr[v];
            mjtNum Rtn[3], Rqv[3];
            mju_mulMatTVec3(Rtn, R, con->ln);
            mju_mulMatVec3(Rqv, R, qv);
            esbase[IPC_NPT*npair + np] = dofadr[v];
            for (int k=0; k < 3; k++) {
              esw[3*IPC_NPT*npair + 3*np + k] = con->lcw[p] * Rtn[k] * h2;
              refc -= con->lcw[p] * con->ln[k] * (xold[3 * v + k] + h * Rqv[k] - xfree[3 * v + k]);
            }
            np++;
          }
          // D = mu/h^2: the inertia is in acceleration units, 1/h^2 heavier than in position form
          esnpt[npair] = np;
          esD[npair] = D * ih2;
          esref[npair] = refc;   // the wall: J*qacc = ref puts the gap at lam/mu
          esxnum[npair] = nx - esxadr[npair];
          npair++;
        }
        int savednefc = d->nefc;
        nefc_qp = d->nefc;
        // The subproblem is solved by MuJoCo's nonlinear CG on the shared primal path, with no
        // factorization to go rank-deficient under stiff contact. A fully converged direction per
        // outer iteration is wasted, since the outer AL converges at the same rate with a loosely
        // solved subproblem; the iteration cap is the model's opt.iterations.
        // solver_niter is zeroed per solve; accumulate it so the step reports its total
        int niter[mjNISLAND];
        for (int i=0; i < mjNISLAND; i++) niter[i] = d->solver_niter[i];
        // publish the pairs as contact rows for the duration of the solve
        ipcEfcRows saved;
        ipc_efcPublish(m, d, &saved, npair, esnpt, esbase, esw, esD, esref, esxadr, esxnum, esxdof,
                       esxval);
        mj_fwdConstraintCG(m, d);  // CG over the monolithic problem, whatever the model's solver
        ipc_efcRowForce(m, d, saved.nefc, qfrc_rows);  // the pairs' share of the solve's force
        ipc_efcRestore(d, &saved);
        for (int i=0; i < mjNISLAND; i++) d->solver_niter[i] += niter[i];
        d->nefc = savednefc;
        // the proposal: the solve's acceleration as a tangent on the owned dofs, and its points
        for (int i=0; i < m->nv; i++)
          wprop[i] = own[i] ? d->qvel[i] + h * d->qacc[i] : 0;
        ipc_points(m, d, wprop, xold, qn_a, h, npt, fidx, dofadr, pbody, pjnv, pt2vg, npa, own,
                   isflexdof, wtmp, xprop);
        // the omitted pairs' most violated row at the proposal
        int nadd = 0;
        for (int c=0; c < nacon; c++) {
          if (inws[c]) continue;
          const ipcCon* con = &wcon[c];
          mjtNum r = ipc_pairResidual(con, xprop, xfree, mass, rad, ghat);
          if (r < -IPC_WS_TOL) {
            inws[c] = 1;
            nadd++;
          }
        }
        if (nadd == 0) break;
        mju_copy(d->qacc_warmstart, d->qacc, m->nv);  // the next round starts from this proposal
        }
        mju_copy(d->qacc_warmstart, ws_entry, m->nv);
        // the converged solution is a search direction, not a commit: the AL energy is finite,
        // so a converged step is large and must be line-searched
        for (int i=0; i < m->nv; i++)
          ddw[i] = own[i] ? wprop[i] - w[i] : 0;
        mj_freeStack(d);
      }
      if (na_artic)  // FK at q_n (+) h*w so rigid contacts read live poses; M stays at q_n
        ipc_treePose(m, d, w, qn_a, h, own, isflexdof, wtmp);
      // convergence per block on the solved direction: the articulated block must converge before
      // the outer loop may end (an unconverged rigid step injects energy through the explicit
      // contact reference), while the flex block may take the full-step bail (AL contact is
      // stable unconverged, and forcing it under load costs many outer iterations)
      int newton_converged = 0;
      {
        mjtNum maxf = 0, maxa = 0;
        for (int i=0; i < m->nv; i++) {
          if (!own[i]) continue;
          mjtNum a = ddw[i] < 0 ? -ddw[i] : ddw[i];
          if (isflexdof[i]) {
            if (a > maxf) maxf = a;
          } else if (a > maxa) {
            maxa = a;
          }
        }
        flex_converged_out = (maxf <= IPC_VEL_TOL);   // vacuously 1 with no flex dofs
        artic_converged_out = (maxa <= IPC_VEL_TOL);  // vacuously 1 with no articulated dofs
        newton_converged = flex_converged_out && artic_converged_out;
        newton_converged_out = newton_converged;
      }
      // line search on the merit, the solve's own objective at the trial state (ipc_merit). The
      // full linearized step overshoots into penetration and the CCD advance then collapses
      mjtNum E0 = ipc_merit(m, d, x, w, own, xfree, h, nefc_qp, mass, rad, ghat, wcon, wncon);
      mjtNum alpha = 1.0;
      // monotone decrease with slop, up to 8 halvings, always accepting the final trial
      for (int ls=0; ls < 8; ls++) {
        for (int i=0; i < m->nv; i++)
          wn[i] = own[i] ? w[i] + alpha * ddw[i] : 0;
        ipc_points(m, d, wn, xold, qn_a, h, npt, fidx, dofadr, pbody, pjnv, pt2vg, npa, own,
                   isflexdof, wtmp, xn);
        mjtNum Etr = ipc_merit(m, d, xn, wn, own, xfree, h, nefc_qp, mass, rad, ghat, wcon, wncon);
        if (Etr <= E0 + 1e-12 || newton_converged) break;
        alpha *= 0.5;
      }
      mju_copy(w, wn, m->nv);  // always accept the final trial
      for (int i=0; i < 3 * nstate; i++)
        x[i] = xn[i];
      last_ls_alpha = alpha;
    }
    // non-penetration advance, in this order: multiplier update on the persistent set, re-query
    // the candidates at xfree over the swept segment xfree -> x, CCD fraction, merge the admitted
    // candidates, advance xfree by that fraction
    ipc_flexLamUpdate(x, xfree, rad, ghat, mass, aset, naset, d->flexvert_lambda, pt2vg, npt);
    ipc_ageStep(d->flexvert_conage, aset, naset, pt2vg, npt, ageload);
    // the sweep x - xfree per point, the pinned points on articulated bodies included; a point
    // pinned to a static body does not move (its xfree can differ from x by round-off)
    for (int v=0; v < npt; v++)
      for (int c=0; c < 3; c++)
        dxp[3 * v + c] = (fidx[v] >= 0 || pjnv[v] > 0) ? x[3 * v + c] - xfree[3 * v + c] : 0;
    // re-query over the swept segment xfree -> x: the base reach is the band, extended per point by
    // its travel, so a pair the proposal crosses is a candidate however far it flings the point
    ncand = mjc_candidates(m, d, xfree, gv, ge, gvgeom, gegeom, ngv, nge, rad, 3 * ghat, 3 * ghat,
                           xfree, x, ghat, nfv, npt, fidx, pjnv, flist, fxadr, nfd, pt2flex, &cand);
    // per-query scratch: the gap at xfree, the closing flag from mjc_advance and the admission
    // mask handed to the merge
    mj_markStack(d);
    int nq = ncand > 0 ? ncand : 1;
    mjtNum* cgap = mjSTACKALLOC(d, nq, mjtNum);
    mjtNum* toi = mjSTACKALLOC(d, nq, mjtNum);
    int* appr = mjSTACKALLOC(d, nq, int);
    int* actc = mjSTACKALLOC(d, nq, int);
    {
      ipcGapArg ga = {NULL, cand, cgap, ncand, 0, xfree, gv, ge, rad, ghat};
      ipc_gapLoop(m, d, &ga);
    }
    // conservative-advancement CCD (mjc_advance): the largest fraction of xfree -> x that keeps
    // every gap above a fraction of its current value; appr flags the pairs that close this step
    // and toi is each pair's collision time along xfree -> x
    mjtNum ac =
        mjc_advance(m, d, xfree, dxp, gv, ge, rad, nfv, cand, ncand, cgap, pt2flex, appr, toi);
    // admit a candidate iff it is already distance-active or closes this step, and among the
    // closing ones only the earliest collision of at least one of its vertices (the paper's
    // filter): a primitive sweeping several layers admits the layer it meets first, not all
    for (int c=0; c < ncand; c++)
      actc[c] = (appr[c] || cgap[c] <= 0.0) ? 1 : 0;
    mjtNum* vmin = mjSTACKALLOC(d, npt > 0 ? npt : 1, mjtNum);
    for (int i=0; i < npt; i++) vmin[i] = 2.0;
    for (int c=0; c < ncand; c++) {
      if (!appr[c]) continue;
      int vv[4], npv = mjc_pairVerts(vv, IPC_PAIR(&cand[c]));
      for (int q=0; q < npv; q++)
        if (toi[c] < vmin[vv[q]]) vmin[vv[q]] = toi[c];
    }
    for (int c=0; c < ncand; c++) {
      if (!appr[c] || cgap[c] <= 0.0) continue;  // distance-active pairs stay admitted
      int vv[4], npv = mjc_pairVerts(vv, IPC_PAIR(&cand[c])), keep = 0;
      for (int q=0; q < npv; q++)
        if (toi[c] <= vmin[vv[q]] + 1e-12) keep = 1;
      actc[c] = keep;
    }
    ipc_mergeActiveSet(d, &aset, &naset, cand, ncand, actc, d->flexvert_lambda,
                       d->flexvert_conage, pt2vg);
    mj_freeStack(d);
    // reclaim this iteration's candidates and the old set (both dead now): move the merged set
    // down to the set's home offset and release everything above it
    if (naset) {
      memmove((char*)d->arena + aset_off, aset, naset * sizeof(ipcCon));
    }
    aset = (ipcCon*)((char*)d->arena + aset_off);
    d->parena = aset_off + naset * sizeof(ipcCon);
    // advance the committed tangent by the CCD fraction on the flex dofs and on a tree carrying
    // pinned points; the other trees take their iterate, as their contacts are native rows with
    // no CCD. The committed points follow the tangent.
    for (int i=0; i < m->nv; i++) {
      if (!own[i]) continue;
      if (!isflexdof[i] && !apinned[m->dof_treeid[i]])
        wfree[i] = w[i];
      else if (ac > IPC_ALPHA_LB)
        wfree[i] += ac * (w[i] - wfree[i]);
    }
    ipc_points(m, d, wfree, xold, qn_a, h, npt, fidx, dofadr, pbody, pjnv, pt2vg, npa, own,
               isflexdof, wtmp, xfree);
    beta = beta + (1.0 - beta) * ac;
    // terminate once the advance is complete and the full step was accepted or both blocks
    // converged. A partial advance commits only part of the motion while time still advances by h,
    // which reads as slow motion; partial advances are fine within the loop, and a frozen CCD
    // leaves through the stall path below
    if (beta >= 1.0 - 1e-6 && (last_ls_alpha >= 1.0 - 1e-9 || newton_converged_out))
      break;
    if (ac > IPC_ALPHA_LB)
      stall = 0;
    else if (++stall >= IPC_STALL_MAX) {
      stalled = 1;
      break;
    }  // CCD froze: warn below
  }
  // restore the position-dependent fields to q_n: the solve ran FK at trial configurations, and
  // a step leaves them at the pre-step configuration
  if (na_artic) {
    mju_copy(d->qpos, qn_a, m->nq);
    mj_kinematics(m, d);
    mj_comPos(m, d);
  }
  // the accepted tangent's endpoint q_n (+) h*wfree over every owned dof, slides and joints alike
  // (the operation ipc_treePose applied to the trial tangents), its velocity wfree, and the
  // effective acceleration (v_new - v_old)/h matching ipc_tangentQacc; a dof the step does not
  // own keeps its velocity and qacc_smooth
  mjtNum* qend = mjSTACKALLOC(d, (m->nq > 0 ? m->nq : 1), mjtNum);
  mjtNum* vend = mjSTACKALLOC(d, m->nv, mjtNum);
  mjtNum* aeff = mjSTACKALLOC(d, m->nv, mjtNum);
  mju_copy(qend, qn_a, m->nq);
  mju_copy(vend, d->qvel, m->nv);
  mju_copy(aeff, d->qacc_smooth, m->nv);
  for (int i=0; i < m->nv; i++)
    if (own[i]) {
      vend[i] = wfree[i];
      aeff[i] = (wfree[i] - d->qvel[i]) / h;
    } else {
      wfree[i] = 0;  // no displacement on a dof the step does not own
    }
  mj_integratePos(m, qend, wfree, h);
  // the constraint force behind it: the native rows evaluated at the effective acceleration and
  // the contact rows of the last solve (the committed step is that solve's proposal whenever the
  // line search and the CCD accepted it in full, which is every complete step)
  if (d->nefc) {
    mjtNum* jar = mjSTACKALLOC(d, d->nefc, mjtNum);
    mj_mulJacVec(m, d, jar, aeff);
    mju_subFrom(jar, d->efc_aref, d->nefc);
    mj_constraintUpdate(m, d, jar, NULL, 0);
  } else {
    mju_zero(d->qfrc_constraint, m->nv);
  }
  mju_addTo(d->qfrc_constraint, qfrc_rows, m->nv);
  // the acceleration-stage sensors: with a dim-2 flex in the model the forward pass skipped the
  // constraint stage (engine_forward.c) and computed them at the free-flight acceleration.
  // Recompute them at the pre-step state from the step's acceleration and constraint force, so a
  // step leaves them as the discrete integrator's forward pass would; a user or plugin sensor at
  // this stage is evaluated twice per step
  if (nfd) {
    mju_copy(d->qacc, aeff, m->nv);
    d->flg_rnepost = 0;  // cacc, if computed, was computed at the free-flight acceleration
    mj_sensorAcc(m, d);
  }
  // the shared commit: the acceleration, history and activations at the pre-step state, the
  // endpoint, time and plugins. The warm start stays as the rounds left it: every inner solve is
  // a different subproblem, so the step's acceleration is no guess for the next one
  mj_commit(m, d, d->act_dot, qend, vend, aeff);
  mj_freeStack(d);
  // an incomplete advance commits only part of the motion while time advances by h, which reads
  // as slow motion rather than as a failure: say so on either exit, the symptom is otherwise
  // silent
  if (beta < 1.0 - 1e-6) {
    if (stalled) {
      mju_warning("IPC: CCD stalled after %d iterations without a feasible advance; the step is "
                  "incomplete (advanced %.3f of the motion). Time = %.4f",
                  IPC_STALL_MAX, beta, d->time);
    } else {
      mju_warning("IPC: outer iteration limit reached; the step is incomplete (advanced %.3f of "
                  "the motion). Time = %.4f", beta, d->time);
    }
  }
}
