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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_METRIC_H_
#define MUJOCO_SRC_ENGINE_ENGINE_METRIC_H_

#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

// A published contact row in mjData.efm_con_ind / efm_con_val is a two-slot header and the entries:
//
//   ind = [nnz, conid, colind...]      val = [scale, force, val...]
//
// scale is the curvature the metric applies, force the pair's force along the row, conid the
// contact it came from. Apply the published forces: res += force * row over the rows.
MJAPI void mj_effContactForce(const mjData* d, mjtNum* res);

// actuation-stage refresh of the metric: actuator gains, their shift, the backbone (factored
// if flg_factor)
void mj_effActuation(const mjModel* m, mjData* d, int flg_factor);

// one rank-1 term of the metric: term = scale * val' * val over the sparse row
typedef struct {
  const mjtNum* val;   // sparse row values
  const int* colind;   // sparse row column indices
  int nnz;             // row nonzeros
  mjtNum scale;        // rank-1 scale
} mjEffRank1;

// iteration cursor over the metric's rank-1 producers (internal layout)
typedef struct {
  int cls;             // producer class
  int i;               // index within class
  int k;               // sub-row within index (multi-output actuators, rod hinges)
  int colind[2];       // inline column indices for 2-dof coupling rows
  mjtNum val[2];       // inline values for 2-dof coupling rows
} mjEffRank1Iter;

// yield the next live rank-1 term of the metric; init the cursor to {0}, returns 0 when
// exhausted. Producers: tendons, then actuator output rows; zero entries are skipped.
// A new metric class becomes a new case here, invisible to every consumer.
// flg_contact selects the passive-contact class, for callers that account for contact separately
int mj_effRank1Next(const mjModel* m, const mjData* d, mjEffRank1Iter* it,
                    mjEffRank1* e, int flg_contact);

// island-local metric product res += S*vec, vectors in island-local dof coordinates
void mj_effMulAddIsland(const mjModel* m, mjData* d, mjtNum* res,
                        const mjtNum* vec, int island);

// implicit effective metric Mtilde = M + (h^2+h*d)*K: per-step arena object (see mjdata.h efm_*)
// build (or deactivate, active==0); the gate decision belongs to the caller
MJAPI void mj_effBuild(const mjModel* m, mjData* d, int active, int flg_factor);

// refresh the metric's smooth-force shift c = h*K*qvel (values only, velocity stage)
MJAPI void mj_effShift(const mjModel* m, mjData* d);

// res += B*vec (the stiffness part of the metric; caller supplies the M part).
// flg_contact selects the passive-contact class, for callers that account for contact
// energy separately
MJAPI void mj_effMulAdd(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                        int flg_contact);

// solve (M + B) x = b by PCG preconditioned with mj_effPrec, to opt.tolerance on the relative
// residual; x = M^-1 b when the metric is inactive. Warns (mjWARN_INERTIA) if the iteration cap
// is reached before convergence, in which case x is returned under-converged.
MJAPI void mj_effSolve(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b);

// apply the metric preconditioner: x ~= (M + B)^-1 b, a cheap fixed linear operator, NOT a solve.
// Exact only when the metric is inactive (x = M^-1 b); otherwise approximate by construction.
MJAPI void mj_effPrec(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b);

// island-local metric preconditioner x ~= (M + B)_island^-1 b, vectors in
// island-local dof coordinates; L: folded blocks (mj_effPrecFold) or NULL for
// the step's own factor; qLD/qLDiagInv/M_* are the island's gathered backbone
// factor
void mj_effPrecIsland(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b,
                      int island, const mjtNum* L, const mjtNum* qLD,
                      const mjtNum* qLDiagInv,
                      const int* M_rownnz, const int* M_rowadr,
                      const int* M_colind);

// the model's vertex ordering of the sparse factor, into efmC_perm (from mj_setConst)
MJAPI void mj_effCholSetConst(mjModel* m, mjData* d);

// stack bytes mj_effMulAdd takes, at most (the matrix-free flex operators)
MJAPI size_t mj_effMulAddScratch(const mjModel* m, const mjData* d);

// stack bytes mj_effPrec and mj_effPrecIsland take, at most
MJAPI size_t mj_effPrecScratch(const mjModel* m, const mjData* d);

// number of covered 3x3 blocks mj_effPrecFold writes (island < 0: nefmdof; island >= 0: the
// island's covered triples, at most island_nv/3)
MJAPI int mj_effFoldBlocks(const mjData* d, int island);

// length of the per-dof diagonal fold tail for 1D DER rod twist chains (L + 9*mj_effFoldBlocks),
// 0 if none
int mj_effTwistDiagSize(const mjModel* m, const mjData* d, int island);

// stack bytes mj_effPrecFold takes for ncol columns (m->nv or island_nv[island])
MJAPI size_t mj_effFoldScratch(int ncol);

// fold the rank-1 classes and the efc rows (D > 0, active or not) into a solver-owned copy of the
// covered 3x3 blocks: L (9*mj_effFoldBlocks) receives, for each folded triple in ascending dof
// order, the factored 3x3 diagonal block of M + K + rank-1 classes + J'*D*J, with no sparse
// factorization. island >= 0: J has the island's local columns and only the island's triples are
// written compactly in L[0..9*nfold). Returns nfold (0, leaving L untouched, when no triple is
// folded or the stack cannot hold the fold's frame)
MJAPI int mj_effPrecFold(const mjModel* m, mjData* d, mjtNum* L, int island, int nefc,
                         const mjtNum* efc_D, int is_sparse, const mjtNum* J,
                         const int* J_rownnz, const int* J_rowadr, const int* J_colind);

// mj_effPrec with the folded blocks L (mj_effPrecFold) in place of the step's factor
MJAPI void mj_effPrecBlocks(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* b,
                            const mjtNum* L);

// a constraint solve with this solver runs monolithically: CG when the step's metric buffer holds
// the sparse factor, which leaves no room for the per-island lists, and sleep is off
int mj_effCGMonolithic(const mjModel* m, const mjData* d, int solver);


#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_METRIC_H_
