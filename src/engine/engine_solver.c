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

#include "engine/engine_solver.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjtype.h>
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_smooth.h"
#include "engine/engine_core_util.h"
#include "engine/engine_memory.h"
#include "engine/engine_macro.h"
#include "engine/engine_metric.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_sparse.h"


//---------------------------------- utility functions ---------------------------------------------

// save solver statistics
static void saveStats(const mjModel* m, mjData* d, int island, int iter,
                      mjtNum improvement, mjtNum gradient, mjtNum lineslope,
                      int nactive, int nchange, int neval, int nupdate) {
  // if island out of range, return
  if (island >= mjNISLAND) {
    return;
  }

  // if no islands, use first island
  island = mjMAX(0, island);

  // if iter out of range, return
  if (iter >= mjNSOLVER) {
    return;
  }

  // get mjSolverStat pointer
  mjSolverStat* stat = d->solver + island*mjNSOLVER + iter;

  // save stats
  stat->improvement = improvement;
  stat->gradient = gradient;
  stat->lineslope = lineslope;
  stat->nactive = nactive;
  stat->nchange = nchange;
  stat->neval = neval;
  stat->nupdate = nupdate;
}


// finalize dual solver: map to joint space
static void dualFinish(const mjModel* m, mjData* d) {
  // map constraint force to joint space
  mj_mulJacTVec(m, d, d->qfrc_constraint, d->efc_force);

  // compute constrained acceleration in joint space, in the solve metric: under the
  // effective metric, qacc_smooth and the constraint response both live in Mtilde
  // (mj_effSolve is the plain M solve when the metric is inactive)
  mj_effSolve(m, d, d->qacc, d->qfrc_constraint);
  mju_addTo(d->qacc, d->qacc_smooth, m->nv);
}


// PGS: map efc_force to joint space
void mj_dualFinish(const mjModel* m, mjData* d) {
  dualFinish(m, d);
}


// compute 1/diag(AR)
//   res[c] = 1 / AR[efclist[c], efclist[c]] for c = 0..nefc-1
//   efclist is NULL for monolithic (sequential) iteration
static void ARdiaginv(const mjModel* m, const mjData* d, mjtNum* res,
                      int nefc, const int* efclist, int flg_subR) {
  const mjtNum *AR = d->efc_AR;
  const mjtNum *R = d->efc_R;

  // sparse
  if (mj_isSparse(m)) {
    const int *rowadr = d->efc_AR_rowadr;
    const int *rownnz = d->efc_AR_rownnz;
    const int *colind = d->efc_AR_colind;

    for (int c=0; c < nefc; c++) {
      int i = efclist ? efclist[c] : c;
      int nnz = rownnz[i];
      for (int j=0; j < nnz; j++) {
        int adr = rowadr[i] + j;
        if (i == colind[adr]) {
          res[c] = 1 / (flg_subR ? mju_max(mjMINVAL, AR[adr] - R[i]) : AR[adr]);
          break;
        }
      }
    }
  }

  // dense
  else {
    int d_nefc = d->nefc;  // global nefc
    for (int c=0; c < nefc; c++) {
      int i = efclist ? efclist[c] : c;
      int adr = i * (d_nefc + 1);
      res[c] = 1 / (flg_subR ? mju_max(mjMINVAL, AR[adr] - R[i]) : AR[adr]);
    }
  }
}


// extract diagonal block from AR, clamp diag to 1e-10 if flg_subR
static void extractBlock(const mjModel* m, const mjData* d, mjtNum* Ac,
                         int start, int n, int flg_subR) {
  int nefc = d->nefc;
  const mjtNum *AR = d->efc_AR;

  // sparse
  if (mj_isSparse(m)) {
    const int* rownnz = d->efc_AR_rownnz;
    const int* rowadr = d->efc_AR_rowadr;
    const int* colind = d->efc_AR_colind;
    /*
            // GENERAL CASE
            mju_zero(Ac, n*n);
            for( j=0; j<n; j++ )
                for( k=0; k<rownnz[start+j]; k++ )
                {
                    int col = colind[rowadr[start+j]+k];
                    if( col>=start && col<start+n )
                        Ac[j*n+col-start] = AR[rowadr[start+j]+k];
                }
     */
    // assume full sub-matrix, find starting k: same for all rows
    int k;
    for (k=0; k < rownnz[start]; k++) {
      if (colind[rowadr[start]+k] == start) {
        break;
      }
    }

    // SHOULD NOT OCCUR
    if (k >= rownnz[start]) {
      mjERROR("internal error");
    }

    // copy rows
    for (int j=0; j < n; j++) {
      mju_copy(Ac+j*n, AR+rowadr[start+j]+k, n);
    }
  }

  // dense
  else {
    for (int j=0; j < n; j++) {
      mju_copy(Ac+j*n, AR+start+(start+j)*nefc, n);
    }
  }

  // subtract R from diagonal, clamp to 1e-10 from below
  if (flg_subR) {
    const mjtNum *R = d->efc_R;
    for (int j=0; j < n; j++) {
      Ac[j*(n+1)] -= R[start+j];
      Ac[j*(n+1)] = mju_max(1e-10, Ac[j*(n+1)]);
    }
  }
}


// compute residual for one block
static void residual(const mjModel* m, const mjData* d, mjtNum* res, int i, int dim, int flg_subR) {
  int nefc = d->nefc;

  // sparse
  if (mj_isSparse(m)) {
    for (int j=0; j < dim; j++) {
      res[j] = d->efc_b[i+j] + mju_dotSparse(d->efc_AR + d->efc_AR_rowadr[i+j],
                                             d->efc_force,
                                             d->efc_AR_rownnz[i+j],
                                             d->efc_AR_colind + d->efc_AR_rowadr[i+j]);
    }
  }

  // dense
  else {
    for (int j=0; j < dim; j++) {
      res[j] = d->efc_b[i+j] + mju_dot(d->efc_AR+(i+j)*nefc, d->efc_force, nefc);
    }
  }

  if (flg_subR) {
    for (int j=0; j < dim; j++) {
      res[j] -= d->efc_R[i+j]*d->efc_force[i+j];
    }
  }
}


// compute cost change
static mjtNum costChange(const mjtNum* A, mjtNum* force, const mjtNum* oldforce,
                         const mjtNum* res, int dim) {
  mjtNum change;

  // compute change
  if (dim == 1) {
    mjtNum delta = force[0] - oldforce[0];
    change = 0.5*delta*delta*A[0] + delta*res[0];
  } else {
    mjtNum delta[6];
    mju_sub(delta, force, oldforce, dim);
    change = 0.5*mju_mulVecMatVec(delta, A, delta, dim) + mju_dot(delta, res, dim);
  }

  // positive change: restore force
  if (change > 1e-10) {
    mju_copy(force, oldforce, dim);
    change = 0;
  }

  return change;
}


// PCG32 random number generator state
typedef struct {
  uint64_t state;
  uint64_t inc;
} pcg32_state;


// generate next 32-bit pseudorandom integer
static uint32_t pcg32_next(pcg32_state* rng) {
  uint64_t oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}


// Fisher-Yates shuffle of integer array
static void shuffle_int(int* array, int n, pcg32_state* rng) {
  for (int i = n - 1; i > 0; i--) {
    uint32_t j = pcg32_next(rng) % (i + 1);
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}


// set efc_state to dual constraint state; return nactive
//   iterates over efclist (or sequentially if NULL), classifies by ne/nf ranges
static int dualState(const mjData* d, int* state,
                     int ne, int nf, int nefc, const int* efclist) {
  const mjtNum* force = d->efc_force;
  const mjtNum* floss = d->efc_frictionloss;

  // equality and friction always active
  int nactive = ne + nf;

  // equality
  for (int c=0; c < ne; c++) {
    int i = efclist ? efclist[c] : c;
    state[i] = mjCNSTRSTATE_QUADRATIC;
  }

  // friction
  for (int c=ne; c < ne+nf; c++) {
    int i = efclist ? efclist[c] : c;
    if (force[i] <= -floss[i]) {
      state[i] = mjCNSTRSTATE_LINEARPOS;  // opposite of primal
    } else if (force[i] >= floss[i]) {
      state[i] = mjCNSTRSTATE_LINEARNEG;
    } else {
      state[i] = mjCNSTRSTATE_QUADRATIC;
    }
  }

  // limit and contact
  for (int c=ne+nf; c < nefc; c++) {
    int i = efclist ? efclist[c] : c;

    // non-negative
    if (d->efc_type[i] != mjCNSTR_CONTACT_ELLIPTIC) {
      if (force[i] <= 0) {
        state[i] = mjCNSTRSTATE_SATISFIED;
      } else {
        state[i] = mjCNSTRSTATE_QUADRATIC;
        nactive++;
      }
    }

    // elliptic
    else {
      // get contact dimensionality, friction, mu
      mjContact* con = d->contact + d->efc_id[i];
      int dim = con->dim, result = 0;
      mjtNum mu = con->mu, f[6];

      // f = map force to regular-cone space
      f[0] = force[i]/mu;
      for (int j=1; j < dim; j++) {
        f[j] = force[i+j]/con->friction[j-1];
      }

      // N = normal, T = norm of tangent vector
      mjtNum N = f[0];
      mjtNum T = mju_norm(f+1, dim-1);

      // top zone
      if (mu*N >= T) {
        result = mjCNSTRSTATE_SATISFIED;
      }

      // bottom zone
      else if (N+mu*T <= 0) {
        result = mjCNSTRSTATE_QUADRATIC;
        nactive += dim;
      }

      // middle zone
      else {
        result = mjCNSTRSTATE_CONE;
        nactive += dim;
      }

      // replicate state in all cone dimensions
      mju_fillInt(state+i, result, dim);

      // advance
      c += (dim-1);
    }
  }

  return nactive;
}


// update constraint state, return nactive and nchange
static int dualStateChange(const mjData* d, int* state, int* oldstate,
                           int ne, int nf, int nefc,
                           const int* efclist, int* nchange) {
  // save old state
  for (int c=0; c < nefc; c++) {
    int i = efclist ? efclist[c] : c;
    oldstate[c] = state[i];
  }

  // update state
  int nactive = dualState(d, state, ne, nf, nefc, efclist);

  // count state changes
  *nchange = 0;
  for (int c=0; c < nefc; c++) {
    int i = efclist ? efclist[c] : c;
    *nchange += (oldstate[c] != state[i]);
  }

  return nactive;
}


// project onto friction ellipsoid, write to force[i+1..i+dim-1]
// project tangential force onto friction ellipsoid: sum(f_t[j]^2/mu[j]^2) <= f_n^2
// if feasible is true, only scale down if outside the ellipsoid
// if feasible is false, always scale to the boundary
static void projectEllipsoid(mjtNum* friction, mjtNum normal, const mjtNum* mu,
                             int dim, int feasible) {
  mjtNum s = 0;
  for (int j=0; j < dim-1; j++) {
    s += friction[j]*friction[j] / (mu[j]*mu[j]);
  }

  mjtNum normal2 = normal*normal;
  if (!feasible || s > normal2) {
    mjtNum scl = mju_sqrt(normal2 / mju_max(mjMINVAL, s));
    for (int j=0; j < dim-1; j++) {
      friction[j] *= scl;
    }
  }
}


// solve QCQP and project onto friction ellipsoid, write to force[i+1..i+dim-1]
static void solveQCQP(mjtNum* force, int i, int dim,
                      mjtNum* Ac, mjtNum* bc, const mjtNum* mu) {
  int flg_active;
  mjtNum v[6];

  // solve
  if (dim == 3) {
    flg_active = mju_QCQP2(v, Ac, bc, mu, force[i]);
  } else if (dim == 4) {
    flg_active = mju_QCQP3(v, Ac, bc, mu, force[i]);
  } else {  // dim == 5
    flg_active = mju_QCQP(v, Ac, bc, mu, force[i], dim-1);
  }

  // on constraint: put v on ellipsoid, in case QCQP is approximate
  if (flg_active) {
    projectEllipsoid(v, force[i], mu, dim, /*feasible=*/0);
  }

  // assign
  mju_copy(force+i+1, v, dim-1);
}


// project contact force block onto friction cone (pyramidal or elliptic)
static void projectCone(mjtNum* force, const mjtNum* mu, int dim, int type) {
  // elliptic cone: project onto friction ellipsoid
  if (type == mjCNSTR_CONTACT_ELLIPTIC) {
    // clamp normal force
    if (force[0] < 0) {
      mju_zero(force, dim);
    } else {
      projectEllipsoid(force+1, force[0], mu, dim, /*feasible=*/1);
    }
  }

  // pyramidal or scalar: clamp to non-negative
  else {
    if (force[0] < 0) {
      force[0] = 0;
    }
  }
}


// global variable to toggle Nesterov momentum (for benchmarks/tests)
mjTHREADLOCAL int mj_nesterov_momentum = 1;


// inverse trace of the inertia matrix, over all dofs (island < 0) or over those of an island
static mjtNum inertiaScale(const mjModel* m, const mjData* d, int island) {
  int nv = island < 0 ? m->nv : d->island_nv[island];
  const int* dofs = island < 0 ? NULL : d->map_idof2dof + d->island_idofadr[island];
  mjtNum trace = 0;
  for (int i=0; i < nv; i++) {
    int dof = dofs ? dofs[i] : i;
    trace += d->M[m->M_rowadr[dof] + m->M_rownnz[dof] - 1];
  }
  return 1 / mju_max(mjMINVAL, trace);
}


//---------------------------- PGS solver ----------------------------------------------------------


// core PGS solver: iterates over constraints specified by efclist
//   island: island index for stats (use -1 for monolithic, mapped to 0)
//   ne, nf, nefc: constraint type counts
//   efclist: maps list position c to monolithic efc index (NULL for sequential)
static void solPGS(const mjModel* m, mjData* d, int island,
                   int ne, int nf, int nefc,
                   const int* efclist, int maxiter) {
  const mjtNum *floss = d->efc_frictionloss;
  mjtNum *force = d->efc_force;
  mj_markStack(d);
  mjtNum* ARinv = mjSTACKALLOC(d, nefc, mjtNum);
  int* oldstate = mjSTACKALLOC(d, 2*nefc, int);
  int* blockstart = oldstate + nefc;

  // Nesterov momentum
  mjtBool nesterov = (mj_nesterov_momentum != 0);
  mjtNum* force_prev = NULL;
  mjtNum* force_momentum = NULL;
  if (nesterov) {
    force_prev = mjSTACKALLOC(d, nefc, mjtNum);
    force_momentum = mjSTACKALLOC(d, nefc, mjtNum);
    mju_gather(force_prev, force, efclist, nefc);
  }

  int island_stat = mjMAX(0, island);  // island index for diagnostic stats
  mjtNum scale = inertiaScale(m, d, island);

  // precompute inverse diagonal of AR
  ARdiaginv(m, d, ARinv, nefc, efclist, 0);

  // initial constraint state
  dualState(d, d->efc_state, ne, nf, nefc, efclist);

  // build block-index array: one entry per constraint block
  int nblocks = 0;
  for (int c=0; c < nefc; ) {
    blockstart[nblocks++] = c;
    int i = efclist ? efclist[c] : c;
    if (d->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {
      c += d->contact[d->efc_id[i]].dim;
    } else {
      c++;
    }
  }

  // seed PCG32 RNG with a fixed seed
  pcg32_state rng;
  rng.state = 0;
  rng.inc = 1;
  pcg32_next(&rng);

  // main iteration
  int iter = 0;
  int nesterov_k = 0;  // Nesterov counter (resets on adaptive restart)
  while (iter < maxiter) {
    // Nesterov momentum extrapolation
    if (nesterov) {
      mjtNum beta = 0;
      if (iter > 0) {
        beta = (mjtNum)(nesterov_k - 1) / (mjtNum)(nesterov_k + 2);
      }

      // update with momentum, save pre-extrapolation value
      if (beta > 0) {
        for (int c=0; c < nefc; c++) {
          int i = efclist ? efclist[c] : c;
          mjtNum f_save = force[i];
          force[i] += beta*(force[i] - force_prev[c]);
          force_prev[c] = f_save;
        }

        // friction loss: project onto bounds
        for (int c=ne; c < ne+nf; c++) {
          int i = efclist ? efclist[c] : c;
          force[i] = mju_clip(force[i], -floss[i], floss[i]);
        }

        // contact force: project onto friction cone
        for (int c=ne+nf; c < nefc; ) {
          int i = efclist ? efclist[c] : c;
          int dim = 1;
          int type = d->efc_type[i];
          const mjtNum* mu = NULL;

          if (type == mjCNSTR_CONTACT_ELLIPTIC) {
            dim = d->contact[d->efc_id[i]].dim;
            mu = d->contact[d->efc_id[i]].friction;
          }

          projectCone(force+i, mu, dim, type);
          c += dim;
        }
      }

      // iter == 0 or beta <= 0 (nesterov_k <= 1): just save current force
      else {
        mju_gather(force_prev, force, efclist, nefc);
      }

      // save extrapolated point for gradient restart check
      mju_gather(force_momentum, force, efclist, nefc);
    }

    // clear improvement
    mjtNum improvement = 0;

    // shuffle constraint visitation order
    shuffle_int(blockstart, nblocks, &rng);

    // perform one sweep over constraint blocks
    for (int bi=0; bi < nblocks; bi++) {
      int c = blockstart[bi];
      int i = efclist ? efclist[c] : c;

      // get constraint dimensionality
      int dim;
      if (d->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {
        dim = d->contact[d->efc_id[i]].dim;
      } else {
        dim = 1;
      }

      // compute residual for this constraint
      mjtNum res[6];
      residual(m, d, res, i, dim, 0);

      // save old force
      mjtNum oldforce[6];
      mju_copy(oldforce, force+i, dim);

      // allocate AR submatrix, required later for costChage
      mjtNum Athis[36];

      // simple constraint
      if (d->efc_type[i] != mjCNSTR_CONTACT_ELLIPTIC) {
        // unconstrained minimum
        force[i] -= res[0]*ARinv[c];

        // impose interval and inequality constraints
        if (c >= ne && c < ne+nf) {
          if (force[i] < -floss[i]) {
            force[i] = -floss[i];
          } else if (force[i] > floss[i]) {
            force[i] = floss[i];
          }
        } else if (c >= ne+nf) {
          if (force[i] < 0) {
            force[i] = 0;
          }
        }
      }

      // elliptic cone constraint
      else {
        // get friction
        mjtNum *mu = d->contact[d->efc_id[i]].friction;

        //-------------------- perform normal or ray update

        // Athis = AR(this,this)
        extractBlock(m, d, Athis, i, dim, 0);

        // normal force too small: normal update
        if (force[i] < mjMINVAL) {
          // unconstrained minimum
          force[i] -= res[0]*ARinv[c];

          // clamp
          if (force[i] < 0) {
            force[i] = 0;
          }

          // clear friction (just in case)
          mju_zero(force+i+1, dim-1);
        }

        // ray update
        else {
          // v = ray
          mjtNum v[6];
          mju_copy(v, force+i, dim);

          // denom = v' * AR(this,this) * v
          mjtNum v1[6];
          mju_mulMatVec(v1, Athis, v, dim, dim);
          mjtNum denom = mju_dot(v, v1, dim);

          // avoid division by 0
          if (denom >= mjMINVAL) {
            // x = v' * res / denom
            mjtNum x = -mju_dot(v, res, dim) / denom;

            // make sure normal is non-negative
            if (force[i]+x*v[0] < 0) {
              x = -v[0]/force[i];
            }

            //  add x*v to f
            for (int j=0; j < dim; j++) {
              force[i+j] += x*v[j];
            }
          }
        }

        //-------------------- perform friction update, keep normal fixed

        // Ac = AR-submatrix; bc = b-subvector + Ac,rest * f_rest
        mjtNum bc[5], Ac[25];
        mju_copy(bc, res+1, dim-1);
        for (int j=0; j < dim-1; j++) {
          mju_copy(Ac+j*(dim-1), Athis+(j+1)*dim+1, dim-1);
          bc[j] -= mju_dot(Ac+j*(dim-1), oldforce+1, dim-1);
          bc[j] += Athis[(j+1)*dim]*(force[i]-oldforce[0]);
        }

        // guard for f_normal==0
        if (force[i] < mjMINVAL) {
          mju_zero(force+i+1, dim-1);
        }

        // QCQP
        else {
          solveQCQP(force, i, dim, Ac, bc, mu);
        }
      }

      // accumulate improvement
      if (dim == 1) {
        Athis[0] = 1/ARinv[c];
      }
      improvement -= costChange(Athis, force+i, oldforce, res, dim);
    }

    // update constraint state
    int nchange;
    int nactive = dualStateChange(d, d->efc_state, oldstate, ne, nf, nefc, efclist, &nchange);

    // scale improvement, save stats
    improvement *= scale;
    saveStats(m, d, island_stat, iter, improvement, 0, 0, nactive, nchange, 0, 0);

    // Nesterov gradient restart (O'Donoghue-Candès): reset when correction opposes extrapolation
    if (nesterov) {
      mjtBool restart = false;
      if (iter > 0) {
        mjtNum dot_corr_extr = 0;
        for (int c=0; c < nefc; c++) {
          int i = efclist ? efclist[c] : c;
          mjtNum correction = force[i] - force_momentum[c];
          mjtNum extrapolation = force_momentum[c] - force_prev[c];
          dot_corr_extr += correction * extrapolation;
        }
        restart = (dot_corr_extr < 0);
      }

      if (restart) {
        nesterov_k = 0;
      } else {
        nesterov_k++;
      }
    }

    // increment iteration count
    iter++;

    // terminate
    if (improvement < m->opt.tolerance) {
      break;
    }
  }

  // finalize statistics
  if (island_stat < mjNISLAND) {
    // update solver iterations
    d->solver_niter[island_stat] += iter;

    // set nnz
    if (mj_isSparse(m)) {
      d->solver_nnz[island_stat] = 0;
      for (int c=0; c < nefc; c++) {
        d->solver_nnz[island_stat] += d->efc_AR_rownnz[efclist ? efclist[c] : c];
      }
    } else {
      d->solver_nnz[island_stat] = nefc*nefc;
    }
  }

  mj_freeStack(d);
}


// PGS entry point (monolithic, no dualFinish — caller handles it)
void mj_solPGS(const mjModel* m, mjData* d, int maxiter) {
  solPGS(m, d, /*island=*/-1, d->ne, d->nf, d->nefc, /*efclist=*/NULL, maxiter);
}


// PGS entry point (one island)
void mj_solPGS_island(const mjModel* m, mjData* d, int island, int maxiter) {
  int ne = d->island_ne[island];
  int nf = d->island_nf[island];
  int nefc = d->island_nefc[island];
  int iefcadr = d->island_iefcadr[island];

  solPGS(m, d, island, ne, nf, nefc, d->map_iefc2efc + iefcadr, maxiter);
}


//---------------------------- NoSlip solver -------------------------------------------------------

// core NoSlip solver: iterates over constraints specified by efclist
//   island: island index for stats (use -1 for monolithic, mapped to 0)
//   ne, nf, nefc: constraint type counts
//   efclist: maps list position c to monolithic efc index (NULL for sequential)
static void solNoSlip(const mjModel* m, mjData* d, int island,
                      int ne, int nf, int nefc,
                      const int* efclist, int maxiter) {
  int dim, iter = 0;
  const mjtNum *floss = d->efc_frictionloss;
  mjtNum *force = d->efc_force;
  mjtNum *mu, improvement;
  mjtNum Ac[25], bc[5], res[5], oldforce[5], delta[5], mid, y, K0, K1;
  mjContact* con;
  mj_markStack(d);
  mjtNum* ARinv = mjSTACKALLOC(d, nefc, mjtNum);
  int* oldstate = mjSTACKALLOC(d, nefc, int);

  int island_stat = mjMAX(0, island);
  mjtNum scale = inertiaScale(m, d, island);

  // precompute inverse diagonal of A
  ARdiaginv(m, d, ARinv, nefc, efclist, 1);

  // initial constraint state
  dualState(d, d->efc_state, ne, nf, nefc, efclist);

  // main iteration
  while (iter < maxiter) {
    // clear improvement
    improvement = 0;

    // correct for cost change at iter 0
    if (iter == 0) {
      for (int c=0; c < nefc; c++) {
        int i = efclist ? efclist[c] : c;
        improvement += 0.5*force[i]*force[i]*d->efc_R[i];
      }
    }

    // perform one sweep: dry friction
    for (int c=ne; c < ne+nf; c++) {
      int i = efclist ? efclist[c] : c;

      // compute residual, save old
      residual(m, d, res, i, 1, 1);
      oldforce[0] = force[i];

      // unconstrained minimum
      force[i] -= res[0]*ARinv[c];

      // impose interval constraints
      if (force[i] < -floss[i]) {
        force[i] = -floss[i];
      } else if (force[i] > floss[i]) {
        force[i] = floss[i];
      }

      // add to improvement
      delta[0] = force[i] - oldforce[0];
      improvement -= 0.5*delta[0]*delta[0]/ARinv[c] + delta[0]*res[0];
    }

    // perform one sweep: contact friction
    for (int c=ne+nf; c < nefc; c++) {
      int i = efclist ? efclist[c] : c;

      // pyramidal contact
      if (d->efc_type[i] == mjCNSTR_CONTACT_PYRAMIDAL) {
        // get contact info
        con = d->contact + d->efc_id[i];
        dim = con->dim;
        mu = con->friction;

        // loop over pairs of opposing pyramid edges
        for (int j=i; j < i+2*(dim-1); j+=2) {
          // compute residual, save old
          residual(m, d, res, j, 2, 1);
          mju_copy(oldforce, force+j, 2);

          // Ac = AR-submatirx
          extractBlock(m, d, Ac, j, 2, 1);

          // bc = b-subvector + Ac,rest * f_rest
          mju_copy(bc, res, 2);
          for (int k=0; k < 2; k++) {
            bc[k] -= mju_dot(Ac+k*2, oldforce, 2);
          }

          // f0 = mid+y, f1 = mid-y
          mid = 0.5*(force[j]+force[j+1]);
          y = 0.5*(force[j]-force[j+1]);

          // K1 = A00 + A11 - 2*A01,  K0 = mid*A00 - mid*A11 + b0 - b1
          K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2];
          K0 = mid*(Ac[0] - Ac[3]) + bc[0] - bc[1];

          // guard against Ac==0
          if (K1 < mjMINVAL) {
            force[j] = force[j+1] = mid;
          }

          // otherwise minimize over y \in [-mid, mid]
          else {
            // unconstrained minimum
            y = -K0/K1;

            // clamp and assign
            if (y < -mid) {
              force[j] = 0;
              force[j+1] = 2*mid;
            } else if (y > mid) {
              force[j] = 2*mid;
              force[j+1] = 0;
            } else {
              force[j] = mid+y;
              force[j+1] = mid-y;
            }
          }

          // accumulate improvement
          improvement -= costChange(Ac, force+j, oldforce, res, 2);
        }

        // skip the rest of this contact
        c += 2*(dim-1)-1;
      }

      // elliptic contact
      else if (d->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {
        // get contact info
        con = d->contact + d->efc_id[i];
        dim = con->dim;
        mu = con->friction;

        // compute residual, save old
        residual(m, d, res, i+1, dim-1, 1);
        mju_copy(oldforce, force+i+1, dim-1);

        // Ac = AR-submatrix
        extractBlock(m, d, Ac, i+1, dim-1, 1);

        // bc = b-subvector + Ac,rest * f_rest
        mju_copy(bc, res, dim-1);
        for (int j=0; j < dim-1; j++) {
          bc[j] -= mju_dot(Ac+j*(dim-1), oldforce, dim-1);
        }

        // guard for f_normal==0
        if (force[i] < mjMINVAL) {
          mju_zero(force+i+1, dim-1);
        }

        // QCQP
        else {
          solveQCQP(force, i, dim, Ac, bc, mu);
        }

        // accumulate improvement
        improvement -= costChange(Ac, force+i+1, oldforce, res, dim-1);

        // skip the rest of this contact
        c += (dim-1);
      }
    }

    // update constraint state
    int nchange;
    int nactive = dualStateChange(d, d->efc_state, oldstate, ne, nf, nefc, efclist, &nchange);

    // scale improvement, save stats
    improvement *= scale;

    // save noslip stats after all the entries from regular solver
    if (island_stat < mjNISLAND) {
      int stats_iter = iter + d->solver_niter[island_stat];
      saveStats(m, d, island_stat, stats_iter, improvement, 0, 0, nactive, nchange, 0, 0);
    }

    // increment iteration count
    iter++;

    // terminate
    if (improvement < m->opt.noslip_tolerance) {
      break;
    }
  }

  // update solver iterations
  if (island_stat < mjNISLAND) {
    d->solver_niter[island_stat] += iter;
  }

  mj_freeStack(d);
}


// NoSlip entry point (monolithic, no dualFinish — caller handles it)
void mj_solNoSlip(const mjModel* m, mjData* d, int maxiter) {
  solNoSlip(m, d, /*island=*/-1, d->ne, d->nf, d->nefc, /*efclist=*/NULL, maxiter);
}


// NoSlip entry point (one island)
void mj_solNoSlip_island(const mjModel* m, mjData* d, int island, int maxiter) {
  int ne = d->island_ne[island];
  int nf = d->island_nf[island];
  int nefc = d->island_nefc[island];
  int iefcadr = d->island_iefcadr[island];

  solNoSlip(m, d, island, ne, nf, nefc, d->map_iefc2efc + iefcadr, maxiter);
}


//------------------------- Primal solvers ---------------------------------------------------------

// Primal context
typedef struct {
  int is_sparse;          // 1: sparse, 0: dense
  int is_elliptic;        // 1: elliptic, 0: pyramidal
  int island;             // current island index, -1 if monolithic

  // sizes
  int nv;                 // number of dofs
  int ne;                 // number of equalities
  int nf;                 // number of friction constraints
  int nefc;               // number of all constraints
  int nJ;                 // number of nonzeros in Jacobian

  // contact array
  mjContact* contact;

  // dof arrays
  const mjtNum* qfrc_smooth;
  const mjtNum* qacc_smooth;
  mjtNum* qfrc_constraint;
  mjtNum* qacc;

  // inertia
  int* M_rownnz;
  int* M_rowadr;
  int* M_colind;
  mjtNum* M;
  mjtNum* qLD;
  mjtNum* qLDiagInv;

  // efc arrays
  const mjtNum* efc_D;
  const mjtNum* efc_R;
  const mjtNum* efc_frictionloss;
  const mjtNum* efc_aref;
  const int* efc_id;
  const int* efc_type;
  mjtNum* efc_force;
  int* efc_state;

  // Jacobians
  int* J_rownnz;
  int* J_rowadr;
  int* J_rowsuper;
  int* J_colind;
  mjtNum* J;
  int* JT_rownnz;
  int* JT_rowadr;
  int* JT_rowsuper;
  int* JT_colind;
  mjtNum* JT;

  // common arrays (PrimalAllocate)
  mjtNum* Jaref;          // Jac*qacc - aref                              (nefc x 1)
  mjtNum* Jv;             // Jac*search                                   (nefc x 1)
  mjtNum* Ma;             // M*qacc                                       (nv x 1)
  mjtNum* Mv;             // M*search                                     (nv x 1)
  mjtNum* grad;           // gradient of master cost                      (nv x 1)
  mjtNum* Mgrad;          // M\grad or H\grad                             (nv x 1)
  mjtNum* search;         // linesearch vector                            (nv x 1)
  mjtNum* quad;           // quadratic polynomials for constraint costs   (nefc x 3)
  int* oldstate;          // previous constraint state                    (nefc x 1)

  // CG arrays (PrimalAllocate, CG only)
  mjtNum* gradold;        // previous gradient                            (nv x 1)
  mjtNum* Mgradold;       // previous preconditioned gradient             (nv x 1)
  mjtNum* graddif;        // grad - gradold                               (nv x 1)
  mjtNum* Mgraddif;       // M\(grad - gradold)                           (nv x 1)

  // Newton arrays, known-size (PrimalAllocate)
  mjtNum* D;              // constraint inertia                           (nefc x 1)
  mjtNum* cholupd;        // scratch for rank-1 Cholesky updates          (nv x 1)
  mjtNum* cholscratch;    // dense accumulator for the sparse kernels     (nv x 1)
  mjtNum* LTJ;            // L'*J for cone Cholesky updates               (6 x nv)
  int* H_rowadr;          // Hessian row addresses                        (nv x 1)
  int* H_rownnz;          // Hessian row nonzeros                         (nv x 1)
  int* HT_rownnz;         // Hessian transpose row nonzeros               (nv x 1)
  int* HT_rowadr;         // Hessian transpose row addresses              (nv x 1)
  int* L_rownnz;          // Hessian factor row nonzeros                  (nv x 1)
  int* L_rowadr;          // Hessian factor row addresses                 (nv x 1)
  int* LT_rownnz;         // Hessian factor transpose row nonzeros        (nv x 1)
  int* LT_rowadr;         // Hessian factor transpose row addresses       (nv x 1)

  // Newton arrays, computed-size (MakeHessian)
  int nH;                 // number of nonzeros in Hessian H
  int* H_colind;          // Hessian column indices                       (nH x 1)
  int* HT_colind;         // Hessian transpose column indices             (nH x 1)
  mjtNum* H;              // Hessian                                      (nH x 1)
  int nL;                 // number of nonzeros in Cholesky factor L
  int* L_colind;          // Cholesky factor column indices               (nL x 1)
  int* LT_colind;         // Cholesky factor transpose column indices     (nL x 1)
  int* LT_map;            // CSC-to-CSR index mapping                     (nL x 1)
  mjtNum* L;              // Cholesky factor                              (nL x 1)
  mjtNum* Lcone;          // Cholesky factor with cone contributions      (nL x 1)

  // implicit effective metric (built per step in mj_fwdPosition): when active, ctx.qfrc_smooth
  // is pre-shifted by +c and Ma/Mv carry the metric terms, so the stock objective/gradient/
  // linesearch formulas below operate in the Mtilde = M + h*D + h^2*K metric unchanged
  int flg_metric;         // effective metric active for this solve
  const mjModel* fm;      // model, for the metric calls
  mjData* fd;             // data, for the metric calls
  mjEffFold epfold;       // metric preconditioner with the contact class folded in
  mjEffFold* ep;          // &epfold when folded, or NULL

  // Newton: lower-triangle view of the metric matrix S = diag(h*D+h^2*K) + K_csr, built by
  // MakeMetricLower on the solver stack, merged into the Hessian alongside M
  int nS;                 // total nonzeros of the view
  int S_tcap;             // capacity of the rank-1 contribution, set by PrimalAllocate
  int* S_rownnz;          // view row nonzeros                            (nv x 1)
  int* S_rowadr;          // view row addresses                           (nv x 1)
  int* S_colind;          // view column indices                          (nS x 1)
  mjtNum* S_val;          // view values                                  (nS x 1)

  // globals
  mjtNum cost;            // constraint + Gauss cost
  mjtNum quadGauss[3];    // quadratic polynomial for Gauss cost
  mjtNum scale;           // scaling factor for improvement and gradient
  int nactive;            // number of active constraints
  int ncone;              // number of contacts in cone state
  int nupdate;            // number of Cholesky updates

  // cone fold gate (elliptic sparse)
  int* pathcost;          // reverse-etree path costs of L rows               (nv x 1)
  mjtSize Lflops;         // sum of squared L row nonzeros (2x factorization MACs)
  int cone_fold;          // cached fold decision (-1: recompute)

  // cone fold workspace, allocated on the first fold (elliptic sparse)
  mjtNum* Jmod;           // Jacobian, cone rows replaced by Lc'*Jc           (nJ x 1)
  mjtNum* JTmod;          // transpose of Jmod                                (nJ x 1)
  mjtNum* Dmod;           // constraint inertia, cone rows set to 1           (nefc x 1)
  mjtNum* Hcone;          // cone-inclusive Hessian                           (nH x 1)
  int* JTmod_rownnz;      // Jmod transpose row nonzeros                      (nv x 1)
  int* JTmod_rowadr;      // Jmod transpose row addresses                     (nv x 1)
  int* JTmod_colind;      // Jmod transpose column indices                    (nJ x 1)

  // linesearch diagnostics
  int LSiter;             // number of linesearch iterations
  int LSresult;           // linesearch result
  mjtNum LSslope;         // linesearch slope at solution
} mjPrimalContext;


// fold the efc rows in their quadratic zone (contact pairs among them) and the metric's rank-1
// classes into a solver-owned copy of the metric blocks (mj_effPrecFold): monolithic solve with a
// live metric, pyramidal cone. IPC folds for both primal solvers (Newton only for its
// certificate); otherwise CG folds whenever the step holds a sparse factor of the covered flex
// block (1D rods, 2D flexes), which mj_effFoldMonolithic keeps off the island path
static int effFoldWanted(const mjModel* m, const mjData* d, int island, int is_elliptic,
                         int flg_Newton) {
  if (!d->efm_active || island >= 0 || is_elliptic) {
    return 0;
  }
  return mjENABLED(mjENBL_IPC) || (!flg_Newton && mj_effFoldMonolithic(m, d, mjSOL_CG));
}


// set sizes and pointers to mjData arrays in mjPrimalContext
static void PrimalPointers(const mjModel* m, const mjData* d, mjPrimalContext* ctx, int island) {
  // clear everything
  memset(ctx, 0, sizeof(mjPrimalContext));

  // globals
  ctx->is_sparse = mj_isSparse(m);
  ctx->is_elliptic = (m->opt.cone == mjCONE_ELLIPTIC);
  ctx->contact = d->contact;
  ctx->island = island;
  ctx->cone_fold = -1;

  // set sizes and pointers (monolithic)
  if (island < 0) {
    // sizes
    ctx->nv               = m->nv;
    ctx->ne               = d->ne;
    ctx->nf               = d->nf;
    ctx->nefc             = d->nefc;
    ctx->nJ               = d->nJ;

    // dof arrays
    ctx->qfrc_smooth      = d->qfrc_smooth;
    ctx->qfrc_constraint  = d->qfrc_constraint;
    ctx->qacc_smooth      = d->qacc_smooth;
    ctx->qacc             = d->qacc;

    // inertia
    ctx->M_rownnz         = m->M_rownnz;
    ctx->M_rowadr         = m->M_rowadr;
    ctx->M_colind         = m->M_colind;
    ctx->M                = d->M;
    ctx->qLD              = d->qLD;
    ctx->qLDiagInv        = d->qLDiagInv;

    // efc arrays
    ctx->efc_D            = d->efc_D;
    ctx->efc_R            = d->efc_R;
    ctx->efc_frictionloss = d->efc_frictionloss;
    ctx->efc_aref         = d->efc_aref;
    ctx->efc_id           = d->efc_id;
    ctx->efc_type         = d->efc_type;
    ctx->efc_force        = d->efc_force;
    ctx->efc_state        = d->efc_state;

    // Jacobians
    ctx->J                = d->efc_J;
    if (ctx->is_sparse) {
      ctx->J_rownnz       = d->efc_J_rownnz;
      ctx->J_rowadr       = d->efc_J_rowadr;
      ctx->J_rowsuper     = d->efc_J_rowsuper;
      ctx->J_colind       = d->efc_J_colind;
    }
  }

  // set sizes and pointers (per-island)
  else {
    // sizes
    ctx->nv               = d->island_nv[island];
    ctx->ne               = d->island_ne[island];
    ctx->nf               = d->island_nf[island];
    ctx->nefc             = d->island_nefc[island];

    // dof arrays
    int idofadr           = d->island_idofadr[island];
    ctx->qfrc_smooth      = d->ifrc_smooth       + idofadr;
    ctx->qfrc_constraint  = d->ifrc_constraint   + idofadr;
    ctx->qacc_smooth      = d->iacc_smooth       + idofadr;
    ctx->qacc             = d->iacc              + idofadr;

    // efc arrays
    int iefcadr           = d->island_iefcadr[island];
    ctx->efc_D            = d->iefc_D            + iefcadr;
    ctx->efc_R            = d->iefc_R            + iefcadr;
    ctx->efc_frictionloss = d->iefc_frictionloss + iefcadr;
    ctx->efc_aref         = d->iefc_aref         + iefcadr;
    ctx->efc_id           = d->iefc_id           + iefcadr;
    ctx->efc_type         = d->iefc_type         + iefcadr;
    ctx->efc_force        = d->iefc_force        + iefcadr;
    ctx->efc_state        = d->iefc_state        + iefcadr;
  }
}


// allocate fixed-size arrays in mjPrimalContext
//  mj_{mark/free}Stack in calling function!
static void PrimalAllocate(const mjModel* m, mjData* d, mjPrimalContext* ctx, int flg_Newton) {
  // local sizes and flags
  int nv = ctx->nv;
  int nefc = ctx->nefc;
  int is_sparse = ctx->is_sparse;
  int is_elliptic = ctx->is_elliptic;
  int nJ = is_sparse ? d->nJ : 0;

  // compute island matrix sizes if needed
  int nC = 0;
  if (ctx->island >= 0) {
    // count nC: number of nonzeros in M block of island (always sparse)
    int island = ctx->island;
    int idofadr = d->island_idofadr[island];
    for (int i = 0; i < nv; i++) {
      int dof = d->map_idof2dof[idofadr + i];
      nC += m->M_rownnz[dof];
    }

    // count nJ: number of nonzeros in J block of island (sparse or dense)
    if (is_sparse) {
      nJ = 0;
      int iefcadr = d->island_iefcadr[island];
      for (int i = 0; i < nefc; i++) {
        int efc = d->map_iefc2efc[iefcadr + i];
        nJ += d->efc_J_rownnz[efc];
      }
    } else {
      nJ = nefc * nv;
    }
    ctx->nJ = nJ;
  }

  // discrete metric: size the lower-triangle view S_* (filled by MakeMetricLower)
  int nScap = 0;
  if (d->efm_active && flg_Newton) {
    int tcap = 0, tlow = 0;
    mjEffRank1Iter it = {0};
    mjEffRank1 e;
    while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
      if (ctx->island >= 0 && d->tree_island[m->dof_treeid[e.colind[0]]] != ctx->island) {
        continue;
      }
      tcap += e.nnz * e.nnz;
      tlow += e.nnz * (e.nnz + 1) / 2;
    }
    ctx->S_tcap = tcap;

    // the view is lower-triangle: at most half the symmetric CSR, the lower wedge of
    // each rank-1 outer product, the strictly-lower fluid pattern, one diagonal per row
    int kcap = d->nefmK / 2;
    if (ctx->island >= 0 && d->nefmK) {
      int ksum = 0;
      int idofadr = d->island_idofadr[ctx->island];
      for (int i = 0; i < nv; i++) {
        ksum += d->efm_K_rownnz[d->map_idof2dof[idofadr + i]];
      }
      kcap = ksum / 2;
    }
    nScap = kcap + tlow + nv + (d->efm_fluid ? m->nC - nv : 0);
  }

  // carve the mjtNum and int blocks: pass 0 totals the sizes, pass 1 hands out the
  // pointers, so a new array cannot be carved without also being counted
  size_t nNum = 0, nInt = 0;
  mjtNum* numblock = NULL;
  int* intblock = NULL;
  mjtNum* qfrc_eff = NULL;

#define CARVE_NUM(lval, count) \
  do {                         \
    if (pass) {                \
      (lval) = numblock;       \
      numblock += (count);     \
    } else                     \
      nNum += (count);         \
  } while (0)

#define CARVE_INT(lval, count) \
  do {                         \
    if (pass) {                \
      (lval) = intblock;       \
      intblock += (count);     \
    } else                     \
      nInt += (count);         \
  } while (0)

  // metric fold, with the dense uncovered-dof blocks for CG (Newton folds only for its
  // certificate, and its Hessian needs the stack). Sizing the blocks takes a frame that the
  // fold's scratch bounds
  int flg_fold = effFoldWanted(m, d, ctx->island, is_elliptic, flg_Newton);
  int nu = 0, ncomp = 0, nS = 0;
  if (flg_fold && !flg_Newton && mj_stackBytesAvailable(d) >= mj_effFoldScratch(m, d, 1)) {
    nS = mj_effFoldDenseSize(m, d, nefc, ctx->efc_D, is_sparse, ctx->J_rownnz, ctx->J_rowadr,
                             ctx->J_colind, &nu, &ncomp);
    if (!nS) {
      nu = ncomp = 0;
    }
  }

  // CG also folds the sparse factor of the covered dofs, when the metric has one
  int nF = flg_fold && !flg_Newton ? mj_effCholFoldSize(d) : 0;

  for (int pass = 0; pass < 2; pass++) {
    if (pass) {
      // keep the sparse factor's copy and the dense blocks only if the stack holds them with the
      // other arrays and the deepest frame the solve takes on top while they are alive: the fold,
      // the preconditioner apply or the metric product. Else drop the copy, then the blocks, with
      // a warning: the 3x3 blocks then precondition the covered dofs, the backbone solve the
      // uncovered ones
      if (nu || nF) {
        size_t fold = mj_effFoldScratch(m, d, nu), mul = mj_effMulAddScratch(m, d);
        size_t avail = mj_stackBytesAvailable(d), scratch = mjMAX(fold, mul);
        size_t need = mj_stackBytes(sizeof(mjtNum)*nNum, _Alignof(mjtNum)) +
                      mj_stackBytes(sizeof(int)*nInt, _Alignof(int)) + scratch;
        int dropped = 0;
        if (nF && need > avail) {
          nNum -= nF;
          nF = 0;
          dropped = 1;
          need = mj_stackBytes(sizeof(mjtNum)*nNum, _Alignof(mjtNum)) +
                 mj_stackBytes(sizeof(int)*nInt, _Alignof(int)) + scratch;
        }
        if (nu && need > avail) {
          nNum -= nS;
          nInt -= nu + 2*(ncomp + 1);
          nu = ncomp = nS = 0;
          dropped = 1;
        }
        if (dropped) {
          mj_warning(d, mjWARN_CNSTRFULL, d->narena);
        }
      }
      numblock = mjSTACKALLOC(d, nNum, mjtNum);
      intblock = mjSTACKALLOC(d, nInt, int);
    }

    // island matrices
    if (ctx->island >= 0) {
      // inertia
      CARVE_INT(ctx->M_rownnz,  nv);
      CARVE_INT(ctx->M_rowadr,  nv);
      CARVE_INT(ctx->M_colind,  nC);
      CARVE_NUM(ctx->M,         nC);
      CARVE_NUM(ctx->qLD,       nC);
      CARVE_NUM(ctx->qLDiagInv, nv);

      // Jacobians
      CARVE_NUM(ctx->J, nJ);
      if (is_sparse) {
        CARVE_INT(ctx->J_rownnz,   nefc);
        CARVE_INT(ctx->J_rowadr,   nefc);
        CARVE_INT(ctx->J_rowsuper, nefc);
        CARVE_INT(ctx->J_colind,   nJ);
      }
    }

    // common arrays
    CARVE_NUM(ctx->Jaref,  nefc);
    CARVE_NUM(ctx->Jv,     nefc);
    CARVE_NUM(ctx->Ma,     nv);
    CARVE_NUM(ctx->Mv,     nv);
    CARVE_NUM(ctx->grad,   nv);
    CARVE_NUM(ctx->Mgrad,  nv);
    CARVE_NUM(ctx->search, nv);
    CARVE_NUM(ctx->quad,   3*nefc);
    if (is_sparse) {
      CARVE_NUM(ctx->JT, nJ);
    }

    // solver-specific arrays
    if (flg_Newton) {
      CARVE_NUM(ctx->D,       nefc);
      CARVE_NUM(ctx->cholupd, nv);
      if (is_elliptic) {
        CARVE_NUM(ctx->LTJ, 6*nv);
      }
      if (is_sparse) {
        CARVE_NUM(ctx->cholscratch, nv);
      } else {
        ctx->nL = nv*nv;
        CARVE_NUM(ctx->L, ctx->nL);
        if (is_elliptic) {
          CARVE_NUM(ctx->Lcone, ctx->nL);
        }
      }
    } else {
      CARVE_NUM(ctx->gradold,  nv);
      CARVE_NUM(ctx->Mgradold, nv);
      CARVE_NUM(ctx->graddif,  nv);
      CARVE_NUM(ctx->Mgraddif, nv);
    }

    // constraint state and Jacobian transpose (sparse)
    // IPC: a solver-owned copy of the metric blocks with the contact class folded in
    // (mj_effPrecFold), or none; CG also folds the sparse factor's copy and the dense
    // uncovered-dof blocks when they fit
    ctx->ep = NULL;
    if (flg_fold) {
      ctx->ep = &ctx->epfold;
      CARVE_NUM(ctx->ep->L,    9*d->nefmdof);
      CARVE_NUM(ctx->ep->S,    nS);
      CARVE_INT(ctx->ep->U,    nu);
      CARVE_INT(ctx->ep->Uadr, nu ? ncomp + 1 : 0);
      CARVE_INT(ctx->ep->Sadr, nu ? ncomp + 1 : 0);
      CARVE_NUM(ctx->ep->F,    nF);
      ctx->ep->nu = nu;
      ctx->ep->ncomp = ncomp;
      ctx->ep->nS = nS;
      ctx->ep->partial = 0;
      ctx->ep->S_valid = 0;
      if (!nF) {
        ctx->ep->F = NULL;
      } else if (pass) {
        ctx->ep->F[0] = 0;
      }
    }

    CARVE_INT(ctx->oldstate, nefc);
    if (is_sparse) {
      CARVE_INT(ctx->JT_rownnz,   nv);
      CARVE_INT(ctx->JT_rowadr,   nv);
      CARVE_INT(ctx->JT_rowsuper, nv);
      CARVE_INT(ctx->JT_colind,   nJ);
    }

    // Newton sparse Hessian and factor sparsity
    if (flg_Newton && is_sparse) {
      CARVE_INT(ctx->H_rowadr,  nv);
      CARVE_INT(ctx->H_rownnz,  nv);
      CARVE_INT(ctx->HT_rownnz, nv);
      CARVE_INT(ctx->HT_rowadr, nv);
      CARVE_INT(ctx->L_rownnz,  nv);
      CARVE_INT(ctx->L_rowadr,  nv);
      CARVE_INT(ctx->LT_rownnz, nv);
      CARVE_INT(ctx->LT_rowadr, nv);
    }

    // discrete metric arrays
    if (d->efm_active) {
      if (flg_Newton) {
        CARVE_INT(ctx->S_rownnz, nv);
        CARVE_INT(ctx->S_rowadr, nv);
        CARVE_INT(ctx->S_colind, nScap);
        CARVE_NUM(ctx->S_val,    nScap);
      }
      CARVE_NUM(qfrc_eff, nv);
    }
  }

#undef CARVE_NUM
#undef CARVE_INT

  // populate island matrices if needed
  if (ctx->island >= 0) {
    int island = ctx->island;
    int idofadr = d->island_idofadr[island];
    int iefcadr = d->island_iefcadr[island];

    // under the discrete metric, gather the qH backbone factor of M + diag in place of
    // qLD: both are block-diagonal by tree, so the island block of the global factor is
    // the factor of the island block
    const mjtNum* gLD = (d->efm_active && d->efm_diag) ? d->qH : d->qLD;
    const mjtNum* gLDiagInv = (d->efm_active && d->efm_diag) ? d->qHDiagInv : d->qLDiagInv;
    mju_blockSparse(ctx->qLD, ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind,
                    gLD, m->M_rownnz, m->M_rowadr, m->M_colind,
                    nv, d->map_idof2dof + idofadr, d->map_dof2idof,
                    d->island_idofadr[island], 0, ctx->M, d->M);
    mju_gather(ctx->qLDiagInv, gLDiagInv, d->map_idof2dof + idofadr, nv);

    if (!is_sparse) {
      mju_block(ctx->J, d->efc_J, m->nv, nv, nefc,
                d->map_iefc2efc + iefcadr, d->map_idof2dof + idofadr);
    } else {
      mju_blockSparse(ctx->J, ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind,
                      d->efc_J, d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                      nefc, d->map_iefc2efc + iefcadr, d->map_dof2idof,
                      d->island_idofadr[island], 0, NULL, NULL);

      mju_superSparse(nefc, ctx->J_rowsuper, ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind);
    }
  }

  // sparse: compute Jacobian transpose
  if (is_sparse) {
    if (nefc) {
      int offset = ctx->J_rowadr[0];
      mju_transposeSparse(ctx->JT, ctx->J + offset, nefc, nv,
                          ctx->JT_rownnz, ctx->JT_rowadr, ctx->JT_colind, ctx->JT_rowsuper,
                          ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind + offset);
    } else {
      // no constraints (reachable via the flex CG dispatch): valid empty transpose structures
      mju_zeroInt(ctx->JT_rownnz, nv);
      mju_zeroInt(ctx->JT_rowadr, nv);
      mju_zeroInt(ctx->JT_rowsuper, nv);
    }
  }

  // implicit effective metric (built in mj_fwdPosition): route Ma/Mv/Mgrad through the
  // metric operators and shift the smooth force, so the stock objective/gradient/linesearch
  // formulas operate in the Mtilde = M + h*D + h^2*K metric; Newton additionally merges the
  // metric matrix into its Hessian (MakeHessian/FactorizeHessian). Island contexts see the
  // island-local view: gathered shift, island-mapped products, and the backbone factor of
  // M + diag gathered in place of qLD; flex never reaches the island path (monolithic solve
  // is forced for flex with metric terms)
  if (d->efm_active) {
    ctx->flg_metric = 1;
    ctx->fm = m;
    ctx->fd = d;
    if (ctx->island < 0) {
      mju_add(qfrc_eff, ctx->qfrc_smooth, d->efm_c, nv);
      if (d->efm_ca) {
        mju_addTo(qfrc_eff, d->efm_ca, nv);
      }
    } else {
      const int* idof2dof = d->map_idof2dof + d->island_idofadr[ctx->island];
      for (int k=0; k < nv; k++) {
        qfrc_eff[k] = ctx->qfrc_smooth[k] + d->efm_c[idof2dof[k]] +
                      (d->efm_ca ? d->efm_ca[idof2dof[k]] : 0);
      }
    }
    ctx->qfrc_smooth = qfrc_eff;
  }
}


// update efc_force, qfrc_constraint, cost-related
static void PrimalUpdateConstraint(mjPrimalContext* ctx, int flg_HessianCone) {
  int nefc = ctx->nefc, nv = ctx->nv;

  // update constraints
  mj_constraintUpdate_impl(ctx->ne, ctx->nf, ctx->nefc, ctx->efc_D, ctx->efc_R,
                           ctx->efc_frictionloss, ctx->Jaref, ctx->efc_type, ctx->efc_id,
                           ctx->contact, ctx->efc_state, ctx->efc_force,
                           &(ctx->cost), flg_HessianCone);

  // compute qfrc_constraint (dense or sparse)
  if (!ctx->is_sparse) {
    mju_mulMatTVec(ctx->qfrc_constraint, ctx->J, ctx->efc_force, nefc, nv);
  } else {
    mju_mulMatVecSparse(ctx->qfrc_constraint, ctx->JT, ctx->efc_force, nv,
                        ctx->JT_rownnz, ctx->JT_rowadr, ctx->JT_colind, ctx->JT_rowsuper);
  }

  // count active and cone
  ctx->nactive = 0;
  ctx->ncone = 0;
  for (int i=0; i < nefc; i++) {
    ctx->nactive += (ctx->efc_state[i] != mjCNSTRSTATE_SATISFIED);
    ctx->ncone += (ctx->efc_state[i] == mjCNSTRSTATE_CONE);
  }

  // add Gauss cost, set in quadratic[0]
  mjtNum Gauss = 0;
  for (int i=0; i < nv; i++) {
    Gauss += 0.5 * (ctx->Ma[i] - ctx->qfrc_smooth[i]) * (ctx->qacc[i] - ctx->qacc_smooth[i]);
  }

  ctx->quadGauss[0] = Gauss;
  ctx->cost += Gauss;
}


// update grad = M*qacc - qfrc_smooth - qfrc_constraint
static void PrimalUpdateGrad(mjPrimalContext* ctx) {
  int nv = ctx->nv;
  for (int i=0; i < nv; i++) {
    ctx->grad[i] = ctx->Ma[i] - ctx->qfrc_smooth[i] - ctx->qfrc_constraint[i];
  }
}


// the metric preconditioner, against the contact-folded blocks where they were built
static void PrimalMetricPrecond(mjPrimalContext* ctx, mjtNum* x, const mjtNum* b) {
  if (ctx->ep) {
    mj_effPrecBlocks(ctx->fm, ctx->fd, x, b, ctx->ep);
  } else {
    mj_effPrec(ctx->fm, ctx->fd, x, b);
  }
}


// update Mgrad;  Newton: Mgrad = H \ grad,  CG: Mgrad = M \ grad
static void PrimalUpdateMgrad(mjPrimalContext* ctx, int flg_Newton) {
  int nv = ctx->nv;

  // Newton: Mgrad = H \ grad
  if (flg_Newton) {
    if (ctx->is_sparse) {
      mju_cholSolveSparse(ctx->Mgrad, (ctx->ncone ? ctx->Lcone : ctx->L),
                          ctx->grad, nv, ctx->L_rownnz, ctx->L_rowadr, ctx->L_colind);
    } else {
      mju_cholSolve(ctx->Mgrad, (ctx->ncone ? ctx->Lcone : ctx->L), ctx->grad, nv);
    }
  }

  // CG monolithic: Mgrad = Mtilde \ grad (approximate metric preconditioner)
  else if (ctx->flg_metric && ctx->island < 0) {
    PrimalMetricPrecond(ctx, ctx->Mgrad, ctx->grad);
  }

  // CG island with metric: island-local metric preconditioner
  else if (ctx->flg_metric) {
    mj_effPrecIsland(ctx->fm, ctx->fd, ctx->Mgrad, ctx->grad, ctx->island,
                     ctx->qLD, ctx->qLDiagInv, ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind);
  }

  // CG without metric: Mgrad = M \ grad
  else {
    mju_copy(ctx->Mgrad, ctx->grad, nv);
    mj_solveLD(ctx->Mgrad, ctx->qLD, ctx->qLDiagInv, nv, 1,
               ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind, NULL);
  }
}


// update grad, Mgrad
static void PrimalUpdateGradient(mjPrimalContext* ctx, int flg_Newton) {
  PrimalUpdateGrad(ctx);
  PrimalUpdateMgrad(ctx, flg_Newton);
}


// prepare quadratic polynomials and contact cone quantities
static void PrimalPrepare(mjPrimalContext* ctx) {
  int nv = ctx->nv, nefc = ctx->nefc;
  const mjtNum* v = ctx->search;

  // Gauss: alpha^2*0.5*v'*M*v + alpha*v'*(Ma-qfrc_smooth) + 0.5*(a-qacc_smooth)'*(Ma-qfrc_smooth)
  //  quadGauss[0] already computed in PrimalUpdateConstraint
  ctx->quadGauss[1] = mju_dot(v, ctx->Ma, nv) - mju_dot(ctx->qfrc_smooth, v, nv);
  ctx->quadGauss[2] = 0.5*mju_dot(v, ctx->Mv, nv);

  // process constraints
  for (int i=0; i < nefc; i++) {
    // pointers to numeric data
    const mjtNum* Jv = ctx->Jv + i;
    const mjtNum* Jaref = ctx->Jaref + i;
    const mjtNum* D = ctx->efc_D + i;

    // pointer to this quadratic
    mjtNum* quad = ctx->quad + 3*i;

    // init with scalar quadratic
    mjtNum DJ0 = D[0]*Jaref[0];
    quad[0] = Jaref[0]*DJ0;
    quad[1] = Jv[0]*DJ0;
    quad[2] = Jv[0]*D[0]*Jv[0];

    // elliptic cone: extra processing
    if (ctx->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {
      // extract contact info
      const mjContact* con = ctx->contact + ctx->efc_id[i];
      int dim = con->dim;
      mjtNum UU = 0, mu = con->mu;
      const mjtNum* friction = con->friction;

      // complete vector quadratic (for bottom zone) and tangential norm at alpha=0
      for (int j=1; j < dim; j++) {
        mjtNum DJj = D[j]*Jaref[j];
        quad[0] += Jaref[j]*DJj;
        quad[1] += Jv[j]*DJj;
        quad[2] += Jv[j]*D[j]*Jv[j];
        mjtNum Uj = Jaref[j]*friction[j-1];
        UU += Uj*Uj;
      }

      // store cone quantities in quad[3-6], using the fact that dim>=3
      quad[3] = Jaref[0]*mu;
      quad[4] = Jv[0]*mu;
      quad[5] = mju_sqrt(UU);
      quad[6] = D[0] / ((mu*mu) * (1 + (mu*mu)));

      // advance to next constraint
      i += (dim-1);
    }

    // apply scaling
    quad[0] *= 0.5;
    quad[2] *= 0.5;
  }
}


// linesearch evaluation point
struct _mjPrimalPnt {
  mjtNum alpha;
  mjtNum cost;
  mjtNum deriv[2];
};
typedef struct _mjPrimalPnt mjPrimalPnt;


// Huber cost of a single friction constraint at a given point x
static mjtNum frictionCost(mjtNum x, mjtNum f, mjtNum Rf, mjtNum D) {
  // -bound < x < bound : quadratic
  if (-Rf < x && x < Rf) return 0.5*D*x*x;

  // x < -bound : linear negative
  else if (x <= -Rf) return f*(-0.5*Rf - x);

  // bound < x : linear positive
  else return f*(-0.5*Rf + x);
}


// Huber cost difference of a single friction constraint: cost(x) - cost(start)
static mjtNum frictionCostDif(mjtNum start, mjtNum x, mjtNum f, mjtNum Rf, mjtNum D) {
  int state_start = (-Rf < start && start < Rf) ? 0 : (start <= -Rf ? -1 : 1);
  int state_x = (-Rf < x && x < Rf) ? 0 : (x <= -Rf ? -1 : 1);

  // both quadratic
  if (state_start == 0 && state_x == 0) {
    return 0.5*D*(x - start)*(x + start);
  }

  // both linear negative
  if (state_start == -1 && state_x == -1) {
    return f*(start - x);
  }

  // both linear positive
  if (state_start == 1 && state_x == 1) {
    return f*(x - start);
  }

  // otherwise different zones: compute absolute costs and subtract
  return frictionCost(x, f, Rf, D) - frictionCost(start, f, Rf, D);
}


// compute cost difference of an elliptic cone at a given alpha: cost(alpha) - cost(0)
static mjtNum ellipticCostDif(const mjtNum* quad, mjtNum alpha, mjtNum mu,
                              mjtNum N, mjtNum T, mjtNum Tsqr_delta) {
  mjtNum U0 = quad[3], V0 = quad[4], T0 = quad[5], Dm = quad[6];

  // determine zone at alpha=0
  int zone0 = 0;
  if (T0 <= 0) {
    zone0 = (U0 < 0) ? 2 : 1;
  } else if (U0 >= mu*T0) {
    zone0 = 1;  // top zone
  } else if (mu*U0 + T0 <= 0) {
    zone0 = 2;  // bottom zone
  } else {
    zone0 = 3;  // middle zone
  }

  // determine zone at alpha
  int zone_alpha = 0;
  if (T <= 0) {
    zone_alpha = (N < 0) ? 2 : 1;  // bottom or top zone
  } else if (N >= mu*T) {
    zone_alpha = 1;  // top zone
  } else if (mu*N + T <= 0) {
    zone_alpha = 2;  // bottom zone
  } else {
    zone_alpha = 3;  // middle zone
  }

  // both top zone
  if (zone0 == 1 && zone_alpha == 1) {
    return 0;
  }

  // both bottom zone
  if (zone0 == 2 && zone_alpha == 2) {
    return alpha*alpha*quad[2] + alpha*quad[1];
  }

  // both middle zone: apply rationalized formula to avoid cancellation
  if (zone0 == 3 && zone_alpha == 3) {
    mjtNum T_delta = Tsqr_delta / (T + T0);
    mjtNum r_delta = alpha*V0 - mu*T_delta;
    mjtNum r0 = U0 - mu*T0;
    return 0.5*Dm*r_delta*(2*r0 + r_delta);
  }

  // CONE -> QUADRATIC (3 -> 2)
  if (zone0 == 3 && zone_alpha == 2) {
    mjtNum dq = alpha*(alpha*quad[2] + quad[1]);
    mjtNum boundary0 = mu*U0 + T0;
    mjtNum gap0 = 0.5*Dm*boundary0*boundary0;
    return dq + gap0;
  }

  // QUADRATIC -> CONE (2 -> 3)
  if (zone0 == 2 && zone_alpha == 3) {
    mjtNum dq = alpha*(alpha*quad[2] + quad[1]);
    mjtNum boundary = mu*N + T;
    mjtNum gap = 0.5*Dm*boundary*boundary;
    return dq - gap;
  }

  // SATISFIED -> QUADRATIC (1 -> 2)
  if (zone0 == 1 && zone_alpha == 2) {
    return alpha*alpha*quad[2] + alpha*quad[1] + quad[0];
  }

  // SATISFIED -> CONE (1 -> 3)
  if (zone0 == 1 && zone_alpha == 3) {
    mjtNum r = N - mu*T;
    return 0.5*Dm*r*r;
  }

  // CONE -> SATISFIED (3 -> 1)
  if (zone0 == 3 && zone_alpha == 1) {
    mjtNum r0 = U0 - mu*T0;
    return -0.5*Dm*r0*r0;
  }

  // QUADRATIC -> SATISFIED (2 -> 1)
  if (zone0 == 2 && zone_alpha == 1) {
    return -quad[0];
  }

  return 0;
}


// evaluate shifted linesearch cost: cost(alpha) - cost(0), and derivatives
static void PrimalEval(mjPrimalContext* ctx, mjPrimalPnt* p) {
  int ne = ctx->ne, nf = ctx->nf, nefc = ctx->nefc;

  // clear result
  mjtNum cost = 0, alpha = p->alpha;
  mjtNum deriv[2] = {0, 0};

  // init quad with Gauss, shifted: drop quadGauss[0]
  mjtNum quadTotal[3] = {0, ctx->quadGauss[1], ctx->quadGauss[2]};

  // process constraints
  for (int i=0; i < nefc; i++) {
    // equality: shifted quad (skip quad[0])
    if (i < ne) {
      quadTotal[1] += ctx->quad[3*i+1];
      quadTotal[2] += ctx->quad[3*i+2];
      continue;
    }

    // friction: compute cost(alpha) - cost(0) directly
    if (i < ne + nf) {
      // search point, friction loss, bound (Rf)
      mjtNum start = ctx->Jaref[i];
      mjtNum dir = ctx->Jv[i];
      mjtNum x = start + alpha*dir;
      mjtNum f = ctx->efc_frictionloss[i];
      mjtNum D = ctx->efc_D[i];
      mjtNum Rf = ctx->efc_R[i]*f;

      // cost delta
      cost += frictionCostDif(start, x, f, Rf, D);

      // -bound < x < bound : quadratic
      if (-Rf < x && x < Rf) {
        deriv[0] += D*x*dir;
        deriv[1] += D*dir*dir;
      }

      // x < -bound : linear negative
      else if (x <= -Rf) {
        deriv[0] += -f*dir;
      }

      // bound < x : linear positive
      else {
        deriv[0] += f*dir;
      }
      continue;
    }

    // limit and contact
    if (ctx->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {         // elliptic cone
      // extract contact info
      const mjContact* con = ctx->contact + ctx->efc_id[i];
      const mjtNum* Jaref = ctx->Jaref + i;
      const mjtNum* Jv = ctx->Jv + i;
      const mjtNum* friction = con->friction;
      mjtNum* quad = ctx->quad + 3*i;
      int dim = con->dim;
      mjtNum mu = con->mu;

      // unpack quad
      mjtNum U0 = quad[3];
      mjtNum V0 = quad[4];
      mjtNum Dm = quad[6];

      // normal and tangential components at alpha (formed componentwise before squaring)
      mjtNum N = U0 + alpha*V0;
      mjtNum V[5], W[5];
      mjtNum Tsqr = 0, WV = 0, Tsqr_delta = 0;
      for (int j=0; j < dim-1; j++) {
        mjtNum Uj = Jaref[j+1]*friction[j];
        mjtNum Vj = Jv[j+1]*friction[j];
        mjtNum Wj = Uj + alpha*Vj;
        V[j] = Vj;
        W[j] = Wj;
        Tsqr += Wj*Wj;
        WV += Wj*Vj;
        Tsqr_delta += alpha*Vj*(Uj + Wj);
      }
      mjtNum T = mju_sqrt(Tsqr);

      // shifted cost
      cost += ellipticCostDif(quad, alpha, mu, N, T, Tsqr_delta);

      // no tangential force : top or bottom zone
      if (Tsqr <= 0) {
        // bottom zone: quadratic derivatives
        if (N < 0) {
          deriv[0] += 2*alpha*quad[2] + quad[1];
          deriv[1] += 2*quad[2];
        }

        // top zone: nothing to do
      }

      // otherwise regular processing
      else {
        // N>=mu*T : top zone
        if (N >= mu*T) {
          // nothing to do
        }

        // mu*N+T<=0 : bottom zone
        else if (mu*N+T <= 0) {
          deriv[0] += 2*alpha*quad[2] + quad[1];
          deriv[1] += 2*quad[2];
        }

        // otherwise middle zone
        else {
          // unit tangent What = W / T
          mjtNum What[5];
          for (int j=0; j < dim-1; j++) {
            What[j] = W[j] / T;
          }

          // cross-product norm squared ||What ^ V||^2 via Lagrange identity to avoid cancellation
          mjtNum det = 0;
          for (int j=0; j < dim-1; j++) {
            for (int k=j+1; k < dim-1; k++) {
              mjtNum djk = What[j]*V[k] - What[k]*V[j];
              det += djk*djk;
            }
          }

          // derivatives
          mjtNum N1 = V0;
          mjtNum T1 = WV / T;
          deriv[0] += Dm*(N-mu*T)*(N1-mu*T1);
          deriv[1] += Dm*((N1-mu*T1)*(N1-mu*T1) + (N/T - mu)*(-mu*det));
        }
      }

      // advance to next constraint
      i += (dim-1);
    } else {                                                  // inequality
      // search point
      mjtNum start = ctx->Jaref[i];
      mjtNum x = start + alpha*ctx->Jv[i];
      mjtNum cost0 = start < 0 ? ctx->quad[3*i] : 0;

      // active
      if (x < 0) {
        // shifted quad: add quad[1], quad[2] and (quad[0] - cost0)
        quadTotal[0] += ctx->quad[3*i] - cost0;
        quadTotal[1] += ctx->quad[3*i+1];
        quadTotal[2] += ctx->quad[3*i+2];
      } else {
        cost -= cost0;
      }
    }
  }

  // add total quadratic (quadTotal[0] contains only shifted residuals)
  cost += alpha*alpha*quadTotal[2] + alpha*quadTotal[1] + quadTotal[0];
  deriv[0] += 2*alpha*quadTotal[2] + quadTotal[1];
  deriv[1] += 2*quadTotal[2];

  // check for convexity; SHOULD NOT OCCUR
  if (deriv[1] <= 0) {
    mju_warning("Linesearch objective is not convex");
    deriv[1] = mjMINVAL;
  }

  // assign and count
  p->cost = cost;
  p->deriv[0] = deriv[0];
  p->deriv[1] = deriv[1];
  ctx->LSiter++;
}


// update bracket point given 3 candidate points
static int updateBracket(mjPrimalContext* ctx,
                         mjPrimalPnt* p, const mjPrimalPnt candidates[3], mjPrimalPnt* pnext) {
  int flag = 0;
  for (int i=0; i < 3; i++) {
    // negative deriv
    if (p->deriv[0] < 0 && candidates[i].deriv[0] < 0 && p->deriv[0] < candidates[i].deriv[0]) {
      *p = candidates[i];
      flag = 1;
    }

    // positive deriv
    else if (p->deriv[0] > 0 &&
             candidates[i].deriv[0] > 0 &&
             p->deriv[0] > candidates[i].deriv[0]) {
      *p = candidates[i];
      flag = 2;
    }
  }

  // compute next point if updated
  if (flag) {
    pnext->alpha = p->alpha - p->deriv[0]/p->deriv[1];
    PrimalEval(ctx, pnext);
  }

  return flag;
}


// line search
static mjtNum PrimalSearch(mjPrimalContext* ctx, mjtNum tolerance, mjtNum ls_iterations,
                           mjtNum* improvement) {
  int nv = ctx->nv, nefc = ctx->nefc;
  mjPrimalPnt p0, p1, p2, pmid, p1next, p2next;

  // clear results
  ctx->LSiter = 0;
  ctx->LSresult = 0;
  ctx->LSslope = 1;       // means not computed
  *improvement = 0;

  // save search vector length, check
  mjtNum snorm = mju_norm(ctx->search, nv);
  if (snorm < mjMINVAL) {
    ctx->LSresult = 1;                          // search vector too small
    return 0;
  }

  // compute scaled gradtol and slope scaling
  mjtNum gtol = tolerance * snorm / ctx->scale;
  mjtNum slopescl = ctx->scale / snorm;

  // compute Mv = Mtilde * v
  mju_mulSymVecSparse(ctx->Mv, ctx->M, ctx->search, nv,
                      ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind);
  if (ctx->flg_metric) {
    if (ctx->island < 0) {
      mj_effMulAdd(ctx->fm, ctx->fd, ctx->Mv, ctx->search, /*flg_contact=*/1);
    } else {
      mj_effMulAddIsland(ctx->fm, ctx->fd, ctx->Mv, ctx->search, ctx->island);
    }
  }

  // compute Jv = J * search  (dense or sparse)
  if (!ctx->is_sparse) {
    mju_mulMatVec(ctx->Jv, ctx->J, ctx->search, nefc, nv);
  } else {
    mju_mulMatVecSparse(ctx->Jv, ctx->J, ctx->search, nefc,
                        ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind, ctx->J_rowsuper);
  }

  // prepare quadratics and cones
  PrimalPrepare(ctx);

  // init at alpha = 0, save
  p0.alpha = 0;
  PrimalEval(ctx, &p0);

  // always attempt one Newton step
  p1.alpha = p0.alpha - p0.deriv[0]/p0.deriv[1];
  PrimalEval(ctx, &p1);

  // check for initial convergence
  if (mju_abs(p1.deriv[0]) < gtol && (p1.alpha == 0 || p1.cost < 0)) {
    if (p1.alpha == 0) {
      ctx->LSresult = 2;  // no improvement, initial convergence
    } else {
      ctx->LSresult = 0;  // SUCCESS
    }
    ctx->LSslope = mju_abs(p1.deriv[0])*slopescl;
    *improvement = -p1.cost;
    return p1.alpha;
  }

  // save direction
  int dir = (p1.deriv[0] < 0 ? +1 : -1);

  // SANITY CHECKS
  /*
     // descent direction
     if( mju_dot(ctx->grad, ctx->search, m->nv)>=0 )
      printf("NOT A DESCENT:  grad %g   search %g   dot %g\n",
          mju_norm(ctx->grad, m->nv),
          mju_norm(ctx->search, m->nv),
          mju_dot(ctx->grad, ctx->search, m->nv));

     // 2nd derivative for Newton cone
     if( ctx->flg_Newton && ctx->ncone )
     {
      mjtNum dd = -p0.deriv[0]/p0.deriv[1];

      if( mju_abs(dd-1)>1e-6 )
          printf("2nd DERIVATIVE FAIL: d0  %g   d1  %g   alpha %g\n",
              p0.deriv[0], p0.deriv[1], dd);
     }

     // cost and gradient at 0: full-space vs. linesearch
     mjtNum grd = mju_dot(ctx->grad, ctx->search, m->nv);
     if( mju_abs(p0.cost-ctx->cost)/mjMAX(mjMINVAL,mju_abs(p0.cost+ctx->cost)) > 1e-6 ||
      mju_abs(p0.deriv[0]-grd)/mjMAX(mjMINVAL,mju_abs(p0.deriv[0]+grd)) > 1e-6 )
     {
      printf("LSiter = %d:\n", ctx->LSiter);
      printf("COST:  %g  %g  %g\n",
          p0.cost, ctx->cost,
          mju_abs(p0.cost-ctx->cost)/mjMAX(mjMINVAL,mju_abs(p0.cost+ctx->cost)));
      printf("GRAD:  %g  %g  %g\n",
          p0.deriv[0], grd,
          mju_abs(p0.deriv[0]-grd)/mjMAX(mjMINVAL,mju_abs(p0.deriv[0]+grd)));
     }
   */

  // one-sided search
  p2 = p0;
  int p2update = 1;
  while (p1.deriv[0]*dir <= -gtol && ctx->LSiter < ls_iterations) {
    // save current
    p2 = p1;
    p2update = 1;

    // move to Newton point w.r.t current
    p1.alpha -= p1.deriv[0]/p1.deriv[1];
    PrimalEval(ctx, &p1);

    // check for convergence
    if (mju_abs(p1.deriv[0]) < gtol && p1.cost < 0) {
      ctx->LSslope = mju_abs(p1.deriv[0])*slopescl;
      *improvement = -p1.cost;
      return p1.alpha;                          // SUCCESS
    }
  }

  // check for failure to bracket
  if (ctx->LSiter >= ls_iterations) {
    ctx->LSresult = 3;                          // could not bracket
    ctx->LSslope = mju_abs(p1.deriv[0])*slopescl;
    *improvement = -p1.cost;
    return p1.alpha;
  }

  // check for p2 update; SHOULD NOT OCCUR
  if (!p2update) {
    ctx->LSresult = 6;                          // no p2 update
    ctx->LSslope = mju_abs(p1.deriv[0])*slopescl;
    *improvement = -p1.cost;
    return p1.alpha;
  }

  // compute next-points for bracket
  p2next = p1;
  p1next.alpha = p1.alpha - p1.deriv[0]/p1.deriv[1];
  PrimalEval(ctx, &p1next);

  // bracketed search
  while (ctx->LSiter < ls_iterations) {
    // evaluate at midpoint
    pmid.alpha = 0.5*(p1.alpha + p2.alpha);
    PrimalEval(ctx, &pmid);

    // make list of candidates
    mjPrimalPnt candidates[3] = {p1next, p2next, pmid};

    // check candidates for convergence
    mjtNum bestcost = 0;
    int bestind = -1;
    for (int i=0; i < 3; i++) {
      if (mju_abs(candidates[i].deriv[0]) < gtol &&
          (bestind == -1 || candidates[i].cost < bestcost)) {
        bestcost = candidates[i].cost;
        bestind = i;
      }
    }
    if (bestind >= 0) {
      ctx->LSslope = mju_abs(candidates[bestind].deriv[0])*slopescl;
      *improvement = -candidates[bestind].cost;
      return candidates[bestind].alpha;       // SUCCESS
    }

    // update brackets
    int b1 = updateBracket(ctx, &p1, candidates, &p1next);
    int b2 = updateBracket(ctx, &p2, candidates, &p2next);

    // no update possible: numerical accuracy reached, use midpoint
    if (!b1 && !b2) {
      if (pmid.cost < 0) {
        ctx->LSresult = 0;  // SUCCESS
      } else {
        ctx->LSresult = 7;  // no improvement, could not bracket
      }

      ctx->LSslope = mju_abs(pmid.deriv[0])*slopescl;
      *improvement = -pmid.cost;
      return pmid.alpha;
    }
  }

  // choose bracket with best cost
  if (p1.cost <= p2.cost && p1.cost < 0) {
    ctx->LSresult = 4;                          // improvement but no convergence
    ctx->LSslope = mju_abs(p1.deriv[0])*slopescl;
    *improvement = -p1.cost;
    return p1.alpha;
  } else if (p2.cost <= p1.cost && p2.cost < 0) {
    ctx->LSresult = 4;                          // improvement but no convergence
    ctx->LSslope = mju_abs(p2.deriv[0])*slopescl;
    *improvement = -p2.cost;
    return p2.alpha;
  } else {
    ctx->LSresult = 5;                          // no improvement
    return 0;
  }
}


// build the lower-triangle view of the discrete metric matrix
// S = diag(h*D+h^2*K) + K_csr + tendon outer products: Newton consumes the metric
// explicitly inside the Hessian. The efm CSR stores the full symmetric pattern with
// sorted columns; the view keeps entries with col < row, folds all diagonals into one
// entry, and produces sorted unique columns per row. For island contexts rows and columns
// are island-local (the maps are monotone, so mapped columns stay sorted); the flex CSR
// never appears there (flex with metric terms forces a monolithic solve)
static void MakeMetricLower(mjData* d, mjPrimalContext* ctx) {
  const mjModel* m = ctx->fm;
  int nv = ctx->nv, island = ctx->island;
  int idofadr = island >= 0 ? d->island_idofadr[island] : 0;
  const int* idof2dof = island >= 0 ? d->map_idof2dof + idofadr : NULL;

  // the view S_* was carved by PrimalAllocate (which also sized the rank-1 contribution)
  // and lives for the whole solve: FactorizeHessian re-merges it on active-set recomputes.
  // Scratch below is freed before returning
  int tcap = ctx->S_tcap;
  mjEffRank1Iter it = {0};
  mjEffRank1 e;

  mj_markStack(d);

  // bucket the rank-1 entries by (local) row: count, carve, fill
  int* tint = mjSTACKALLOC(d, 2*nv + (tcap > 0 ? tcap : 1), int);
  mjtNum* tval = mjSTACKALLOC(d, tcap > 0 ? tcap : 1, mjtNum);
  int* trownnz = tint;
  int* trowadr = tint + nv;
  int* tcol = tint + 2*nv;
  mju_zeroInt(trownnz, nv);

  // count rank-1 entries per (local) row
  it = (mjEffRank1Iter){0};
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
    if (island >= 0 && d->tree_island[m->dof_treeid[e.colind[0]]] != island) {
      continue;
    }
    for (int j=0; j < e.nnz; j++) {
      int r = e.colind[j];
      int lr = island >= 0 ? d->map_dof2idof[r] - idofadr : r;
      if (lr < 0 || lr >= nv) {
        // island discovery unions metric couplings and the sleep policy keeps coupled
        // trees awake together, so a rank-1 term cannot straddle islands or sleep
        mjERROR("metric rank-1 term spans islands on dof %d", r);  // SHOULD NOT OCCUR
      }
      trownnz[lr] += e.nnz;
    }
  }
  // carve row addresses
  int adr = 0;
  for (int i=0; i < nv; i++) {
    trowadr[i] = adr;
    adr += trownnz[i];
    trownnz[i] = 0;
  }

  // fill the buckets
  it = (mjEffRank1Iter){0};
  while (mj_effRank1Next(m, d, &it, &e, /*flg_contact=*/1)) {
    if (island >= 0 && d->tree_island[m->dof_treeid[e.colind[0]]] != island) {
      continue;
    }
    for (int j=0; j < e.nnz; j++) {
      int r = e.colind[j];
      int lr = island >= 0 ? d->map_dof2idof[r] - idofadr : r;
      mjtNum sJr = e.scale * e.val[j];
      for (int a=0; a < e.nnz; a++) {
        int lc = island >= 0 ? d->map_dof2idof[e.colind[a]] - idofadr : e.colind[a];
        int w = trowadr[lr] + trownnz[lr]++;
        tcol[w] = lc;
        tval[w] = sJr * e.val[a];
      }
    }
  }

  // sort each bucketed row (insertion sort: per-tendon runs are already sorted and rows
  // are short)
  for (int i=0; i < nv; i++) {
    int start = trowadr[i], end = start + trownnz[i];
    for (int a=start+1; a < end; a++) {
      int c = tcol[a];
      mjtNum v = tval[a];
      int b = a - 1;
      while (b >= start && tcol[b] > c) {
        tcol[b+1] = tcol[b];
        tval[b+1] = tval[b];
        b--;
      }
      tcol[b+1] = c;
      tval[b+1] = v;
    }
  }

  // merge flex CSR, tendon buckets and diagonals into the lower view
  adr = 0;
  for (int i=0; i < nv; i++) {
    ctx->S_rowadr[i] = adr;
    int gdof = island >= 0 ? idof2dof[i] : i;
    mjtNum diag = d->efm_diag ? d->efm_diag[gdof] : 0;

    // three-way merge of the sorted flex row, the sorted tendon/actuator row and the
    // fluid M-pattern row, summing duplicates. Island dof->idof maps preserve order
    // across DOFs in the island, so mapped flex and fluid columns stay sorted
    int fa = 0, fn = 0;
    const int* fcol = NULL;
    const mjtNum* fval = NULL;
    if (d->nefmK) {
      fa = d->efm_K_rowadr[gdof];
      fn = d->efm_K_rownnz[gdof];
      fcol = d->efm_K_colind;
      fval = d->efm_K_val;
    }
    int fe = fa + fn;
    int ta = trowadr[i], te = ta + trownnz[i];
    int ga = 0, ge = 0;
    if (d->efm_fluid) {
      ga = m->M_rowadr[gdof];
      ge = ga + m->M_rownnz[gdof];
    }
    while (fa < fe || ta < te || ga < ge) {
      int col;
      mjtNum val = 0;
      int cf = fa < fe ? (island >= 0 ? d->map_dof2idof[fcol[fa]] - idofadr
                                      : fcol[fa]) : nv;
      int ct = ta < te ? tcol[ta] : nv;
      int cg = ga < ge ? (island >= 0 ? d->map_dof2idof[m->M_colind[ga]] - idofadr
                                      : m->M_colind[ga]) : nv;
      col = cf < ct ? cf : ct;
      col = cg < col ? cg : col;
      while (fa < fe &&
             (island >= 0 ? d->map_dof2idof[fcol[fa]] - idofadr
                          : fcol[fa]) == col) {
        val += fval[fa++];
      }
      while (ta < te && tcol[ta] == col) {
        val += tval[ta++];
      }
      while (ga < ge &&
             (island >= 0 ? d->map_dof2idof[m->M_colind[ga]] - idofadr
                          : m->M_colind[ga]) == col) {
        val += d->efm_fluid[ga++];
      }
      if (col < i) {
        ctx->S_colind[adr] = col;
        ctx->S_val[adr] = val;
        adr++;
      } else if (col == i) {
        diag += val;
        break;  // columns are strictly sorted: nothing at or before the diagonal remains
      } else {
        break;
      }
    }

    if (diag) {
      ctx->S_colind[adr] = i;
      ctx->S_val[adr] = diag;
      adr++;
    }
    ctx->S_rownnz[i] = adr - ctx->S_rowadr[i];
  }
  ctx->nS = adr;

  mj_freeStack(d);
}


// scratch stack bytes needed by mju_sqrMatTDSparseSymbolic + Numeric
static size_t primalSqrScratchBytes(int nv, int nefc) {
  size_t sym_bytes = mj_stackFrameBytes() +
                     2 * mj_stackBytes((size_t)nv * sizeof(int), _Alignof(int)) +
                     mj_stackBytes((size_t)nefc * sizeof(int), _Alignof(int));
  size_t num_bytes = mj_stackFrameBytes() +
                     mj_stackBytes(8 * (size_t)nv * sizeof(mjtNum), _Alignof(mjtNum));
  return mjMAX(sym_bytes, num_bytes);
}


// per-iteration scratch stack bytes needed by the primal solver main loop
static size_t primalLoopScratchBytes(const mjPrimalContext* ctx, const mjData* d) {
  size_t sqr_bytes = ctx->is_sparse ? primalSqrScratchBytes(ctx->nv, ctx->nefc) : 0;
  size_t mul_bytes = ctx->flg_metric ? mj_effMulAddScratch(ctx->fm, d) : 0;
  size_t prec_bytes =
      ctx->flg_metric ? mj_effFoldScratch(ctx->fm, d, ctx->ep ? ctx->ep->nu : 0) : 0;
  return mjMAX(sqr_bytes, mjMAX(mul_bytes, prec_bytes));
}


// allocate and compute Hessian given efc_state; return 1 on success, 0 if arena is exhausted
//  mj_{mark/free}Stack in caller function!
static int MakeHessian(mjData* d, mjPrimalContext* ctx) {
  int nv = ctx->nv, nefc = ctx->nefc;

  // per-iteration scratch needed by the solver main loop after MakeHessian returns
  size_t cg_bytes =
      ctx->LTJ ? 0 : mj_stackBytes(2 * (size_t)nv * sizeof(mjtNum), _Alignof(mjtNum));
  size_t loop_bytes = primalLoopScratchBytes(ctx, d);

  // compute constraint inertia
  for (int i=0; i < nefc; i++) {
    ctx->D[i] = ctx->efc_state[i] == mjCNSTRSTATE_QUADRATIC ? ctx->efc_D[i] : 0;
  }

  // discrete metric: build the lower-triangle view of S for merging into H
  ctx->nS = 0;
  if (ctx->flg_metric) {
    int tcap = ctx->S_tcap > 0 ? ctx->S_tcap : 1;
    size_t metric_bytes =
        mj_stackFrameBytes() +
        mj_stackBytes((2 * (size_t)nv + (size_t)tcap) * sizeof(int), _Alignof(int)) +
        mj_stackBytes((size_t)tcap * sizeof(mjtNum), _Alignof(mjtNum));
    if (mj_stackBytesAvailable(d) < mjMAX(metric_bytes, cg_bytes + loop_bytes)) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      return 0;
    }
    MakeMetricLower(d, ctx);
  }

  // sparse
  if (ctx->is_sparse) {
    if (mj_stackBytesAvailable(d) < cg_bytes + loop_bytes) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      return 0;
    }

    // count Hessian nonzeros, initialize rowadr, rownnz
    ctx->nH = mju_sqrMatTDSparseSymbolic(
        ctx->H_rownnz, ctx->H_rowadr, NULL, NULL,
        nefc, nv, ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind,
        ctx->JT_rownnz, ctx->JT_rowadr, ctx->JT_colind, ctx->JT_rowsuper, d);

    // add M nonzeros to Hessian total (unavoidable overcounting since H_colind is still unknown)
    ctx->nH += ctx->M_rowadr[nv - 1] + ctx->M_rownnz[nv - 1];

    // add discrete-metric nonzeros
    ctx->nH += ctx->nS;

    // check stack space for H, H_colind, HT_colind plus symbolic Cholesky counting scratch
    size_t H_bytes =
        mj_stackBytes((size_t)ctx->nH * sizeof(mjtNum), _Alignof(mjtNum)) +
        mj_stackBytes(2 * (size_t)ctx->nH * sizeof(int), _Alignof(int));
    size_t chol_count_bytes =
        mj_stackFrameBytes() + 2 * mj_stackBytes((size_t)nv * sizeof(int), _Alignof(int));
    if (mj_stackBytesAvailable(d) < H_bytes + cg_bytes + mjMAX(chol_count_bytes, loop_bytes)) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      return 0;
    }

    // nH is known: allocate H, H_colind, HT_colind
    ctx->H          = mjSTACKALLOC(d, ctx->nH, mjtNum);
    int* H_intblock = mjSTACKALLOC(d, 2*ctx->nH, int);
    ctx->H_colind   = H_intblock;
    ctx->HT_colind  = H_intblock + ctx->nH;

    // shift H row addresses to make room for M and the discrete metric
    int shift = 0;
    for (int r = 0; r < nv - 1; r++) {
      shift += ctx->M_rownnz[r] + (ctx->nS ? ctx->S_rownnz[r] : 0);
      ctx->H_rowadr[r + 1] += shift;
    }

    // compute H = J'*D*J: symbolic phase
    mju_sqrMatTDSparseSymbolic(
        ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, NULL,
        nefc, nv, ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind,
        ctx->JT_rownnz, ctx->JT_rowadr, ctx->JT_colind, ctx->JT_rowsuper, d);

    // compute H = J'*D*J: numeric phase
    mju_sqrMatTDSparseNumeric(
        ctx->H, nv, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind,
        NULL, ctx->J, ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind,
        ctx->JT, ctx->JT_rownnz, ctx->JT_rowadr, ctx->JT_colind,
        ctx->JT_rowsuper, ctx->D, d);

    // add mass matrix: H = J'*D*J + M
    mju_addToMatSparse(ctx->H, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, nv,
                       ctx->M, ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind);

    // add discrete metric: H = J'*D*J + Mtilde
    if (ctx->nS) {
      mju_addToMatSparse(ctx->H, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, nv,
                         ctx->S_val, ctx->S_rownnz, ctx->S_rowadr, ctx->S_colind);
    }

    // compute H' sparse structure (upper triangle, required for symbolic Cholesky)
    mju_transposeSparse(NULL, NULL, nv, nv, ctx->HT_rownnz, ctx->HT_rowadr, ctx->HT_colind, NULL,
                        ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind);

    // count total and row non-zeros of reverse-Cholesky factors L and LT
    ctx->nL = mju_cholFactorSymbolic(NULL, ctx->L_rownnz, ctx->L_rowadr, NULL,
                                     ctx->LT_rownnz, ctx->LT_rowadr, NULL,
                                     ctx->HT_rownnz, ctx->HT_rowadr, ctx->HT_colind,
                                     nv, d);

    // nL is known: check stack space, then allocate L_colind, LT_colind, LT_map, L, Lcone
    // L_colind + LT_colind + LT_map + pathcost
    size_t nL_int = 3*(size_t)ctx->nL + (ctx->is_elliptic ? (size_t)nv : 0);
    size_t nL_num = ctx->is_elliptic ? 2*(size_t)ctx->nL : (size_t)ctx->nL;  // L + Lcone
    size_t L_bytes =
        mj_stackBytes(nL_int * sizeof(int), _Alignof(int)) +
        mj_stackBytes(nL_num * sizeof(mjtNum), _Alignof(mjtNum));
    size_t chol_fill_bytes =
        mj_stackFrameBytes() + 3 * mj_stackBytes((size_t)nv * sizeof(int), _Alignof(int));
    if (mj_stackBytesAvailable(d) < L_bytes + mjMAX(chol_fill_bytes, loop_bytes)) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      return 0;
    }
    int* L_intblock    = mjSTACKALLOC(d, nL_int, int);
    mjtNum* L_numblock = mjSTACKALLOC(d, nL_num, mjtNum);
    ctx->L_colind      = L_intblock;
    ctx->LT_colind     = L_intblock + ctx->nL;
    ctx->LT_map        = L_intblock + 2*ctx->nL;
    ctx->pathcost      = ctx->is_elliptic ? L_intblock + 3*ctx->nL : NULL;
    ctx->L             = L_numblock;
    ctx->Lcone         = ctx->is_elliptic ? L_numblock + ctx->nL : NULL;

    // symbolic Cholesky: populate L_colind and LT structures
    mju_cholFactorSymbolic(ctx->L_colind, ctx->L_rownnz, ctx->L_rowadr,
                           ctx->LT_colind, ctx->LT_rownnz, ctx->LT_rowadr, ctx->LT_map,
                           ctx->HT_rownnz, ctx->HT_rowadr, ctx->HT_colind,
                           nv, d);

    // cone fold gate inputs, fixed for the solve since the L pattern is fixed:
    // pathcost[r] totals the L row nonzeros along the reverse-etree path from
    // row r (parent(r) = largest off-diagonal column of L row r), the cost of
    // a rank-1 update starting at column r; Lflops totals squared row nonzeros
    if (ctx->is_elliptic) {
      ctx->Lflops = 0;
      for (int r=0; r < nv; r++) {
        int nnz = ctx->L_rownnz[r];
        int parent = nnz >= 2 ? ctx->L_colind[ctx->L_rowadr[r] + nnz - 2] : -1;
        ctx->pathcost[r] = nnz + (parent >= 0 ? ctx->pathcost[parent] : 0);
        ctx->Lflops += (mjtSize)nnz * nnz;
      }
    }
  }

  // dense
  else {
    // compute H = M + J'*D*J
    mju_sqrMatTD_impl(ctx->L, ctx->J, ctx->D, nefc, nv, /*flg_upper=*/ 0);
    mju_addToSymSparse(ctx->L, ctx->M, ctx->nv,
                       ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind,
                       /*flg_upper=*/ 0);

    // add discrete metric (the view is lower+diagonal by construction)
    if (ctx->nS) {
      mju_addToSymSparse(ctx->L, ctx->S_val, nv,
                         ctx->S_rownnz, ctx->S_rowadr, ctx->S_colind,
                         /*flg_upper=*/ 0);
    }
  }

  return 1;
}


// forward declarations (for readability)
static void HessianCone(mjData* d, mjPrimalContext* ctx);
static void HessianConeUpdate(mjData* d, mjPrimalContext* ctx);

// factorize Hessian: L = chol(H), maybe (re)compute H given efc_state
static void FactorizeHessian(mjData* d, mjPrimalContext* ctx, int flg_recompute) {
  int nv = ctx->nv, nefc = ctx->nefc;

  // maybe compute constraint inertia
  if (flg_recompute) {
    for (int i=0; i < nefc; i++) {
      ctx->D[i] = ctx->efc_state[i] == mjCNSTRSTATE_QUADRATIC ? ctx->efc_D[i] : 0;
    }
  }

  // sparse
  if (ctx->is_sparse) {
    // maybe compute H = M + J'*D*J (sparsity pattern already in H_rownnz, H_rowadr, H_colind)
    if (flg_recompute) {
      // compute H = J'*D*J: numeric phase
      mju_sqrMatTDSparseNumeric(
          ctx->H, nv, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind,
          NULL, ctx->J, ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind,
          ctx->JT, ctx->JT_rownnz, ctx->JT_rowadr, ctx->JT_colind,
          ctx->JT_rowsuper, ctx->D, d);

      // add mass matrix: H = J'*D*J + C
      mju_addToMatSparse(ctx->H, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, nv,
                         ctx->M, ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind);

      // add discrete metric: H = J'*D*J + Mtilde
      if (ctx->nS) {
        mju_addToMatSparse(ctx->H, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, nv,
                           ctx->S_val, ctx->S_rownnz, ctx->S_rowadr, ctx->S_colind);
      }
    }

    // numeric sparse factorization: L = chol(H) using pre-computed sparsity pattern
    mju_cholFactorNumeric(
        ctx->L, nv, mjMINVAL,
        ctx->L_rownnz, ctx->L_rowadr, ctx->L_colind,
        ctx->LT_rownnz, ctx->LT_rowadr, ctx->LT_colind, ctx->LT_map,
        ctx->H, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, ctx->cholscratch);
  }

  // dense
  else {
    // maybe compute H = Mtilde + J'*D*J
    if (flg_recompute) {
      mju_sqrMatTD_impl(ctx->L, ctx->J, ctx->D, nefc, nv, /*flg_upper=*/ 0);
      mju_addToSymSparse(ctx->L, ctx->M, ctx->nv,
                         ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind,
                         /*flg_upper=*/ 0);
      if (ctx->nS) {
        mju_addToSymSparse(ctx->L, ctx->S_val, nv,
                           ctx->S_rownnz, ctx->S_rowadr, ctx->S_colind,
                           /*flg_upper=*/ 0);
      }
    }

    // factorize H
    mju_cholFactor(ctx->L, nv, mjMINVAL);
  }

  // add cones to factor if present
  if (ctx->ncone) {
    ctx->cone_fold = -1;
    HessianCone(d, ctx);
  }

  // mark full update
  ctx->nupdate = nefc;
}


// elliptic sparse case with many cones: rebuild Lcone with one factorization.
static void HessianConeFolded(mjData* d, mjPrimalContext* ctx) {
  int nv = ctx->nv, nefc = ctx->nefc;
  mjtNum local[36];

  // allocate on the first fold of this solve and reuse for the rest of it
  if (!ctx->Jmod) {
    size_t fold_bytes =
        2 * mj_stackBytes((size_t)ctx->nJ * sizeof(mjtNum), _Alignof(mjtNum)) +
        mj_stackBytes((size_t)nefc * sizeof(mjtNum), _Alignof(mjtNum)) +
        mj_stackBytes((size_t)ctx->nH * sizeof(mjtNum), _Alignof(mjtNum)) +
        2 * mj_stackBytes((size_t)nv * sizeof(int), _Alignof(int)) +
        mj_stackBytes((size_t)ctx->nJ * sizeof(int), _Alignof(int));
    size_t loop_bytes = primalLoopScratchBytes(ctx, d);
    if (mj_stackBytesAvailable(d) < fold_bytes + loop_bytes) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      ctx->cone_fold = 0;
      HessianConeUpdate(d, ctx);
      return;
    }
    ctx->Jmod         = mjSTACKALLOC(d, ctx->nJ, mjtNum);
    ctx->JTmod        = mjSTACKALLOC(d, ctx->nJ, mjtNum);
    ctx->Dmod         = mjSTACKALLOC(d, nefc, mjtNum);
    ctx->Hcone        = mjSTACKALLOC(d, ctx->nH, mjtNum);
    ctx->JTmod_rownnz = mjSTACKALLOC(d, nv, int);
    ctx->JTmod_rowadr = mjSTACKALLOC(d, nv, int);
    ctx->JTmod_colind = mjSTACKALLOC(d, ctx->nJ, int);
  }
  mjtNum* Jmod   = ctx->Jmod;
  mjtNum* JTmod  = ctx->JTmod;
  mjtNum* Dmod   = ctx->Dmod;
  mjtNum* Hcone  = ctx->Hcone;
  int* JT_rownnz = ctx->JTmod_rownnz;
  int* JT_rowadr = ctx->JTmod_rowadr;
  int* JT_colind = ctx->JTmod_colind;

  mju_copy(Jmod, ctx->J, ctx->nJ);
  for (int i=0; i < nefc; i++) {
    Dmod[i] = ctx->efc_state[i] == mjCNSTRSTATE_QUADRATIC ? ctx->efc_D[i] : 0;
  }

  // replace cone contact rows with Lc'*Jc, where Hc = Lc*Lc'
  for (int i=0; i < nefc; i++) {
    if (ctx->efc_state[i] == mjCNSTRSTATE_CONE) {
      int dim = ctx->contact[ctx->efc_id[i]].dim;
      mju_copy(local, ctx->contact[ctx->efc_id[i]].H, dim*dim);
      mju_cholFactor(local, dim, mjMINVAL);

      const int nnz = ctx->J_rownnz[i];
      for (int c=0; c < dim; c++) {
        mjtNum* dst = Jmod + ctx->J_rowadr[i+c];
        mju_zero(dst, nnz);
        for (int r=c; r < dim; r++) {
          mju_addToScl(dst, ctx->J + ctx->J_rowadr[i+r], local[r*dim+c], nnz);
        }
        Dmod[i+c] = 1;
      }
      ctx->nupdate += dim;
      i += (dim-1);
    }
  }

  // Lcone = chol(Jmod'*Dmod*Jmod + M), same sparsity as L
  mju_transposeSparse(JTmod, Jmod, nefc, nv, JT_rownnz, JT_rowadr, JT_colind, NULL,
                      ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind);
  mju_sqrMatTDSparseNumeric(Hcone, nv, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, NULL,
                            Jmod, ctx->J_rownnz, ctx->J_rowadr, ctx->J_colind,
                            JTmod, JT_rownnz, JT_rowadr, JT_colind,
                            ctx->JT_rowsuper, Dmod, d);
  mju_addToMatSparse(Hcone, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, nv,
                     ctx->M, ctx->M_rownnz, ctx->M_rowadr, ctx->M_colind);
  if (ctx->nS) {
    mju_addToMatSparse(Hcone, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind, nv,
                       ctx->S_val, ctx->S_rownnz, ctx->S_rowadr, ctx->S_colind);
  }
  mju_cholFactorNumeric(ctx->Lcone, nv, mjMINVAL,
                        ctx->L_rownnz, ctx->L_rowadr, ctx->L_colind,
                        ctx->LT_rownnz, ctx->LT_rowadr, ctx->LT_colind, ctx->LT_map,
                        Hcone, ctx->H_rownnz, ctx->H_rowadr, ctx->H_colind,
                        ctx->cholscratch);
}


// elliptic case, update path: Lcone = L, then dim rank-1 updates per cone contact
static void HessianConeUpdate(mjData* d, mjPrimalContext* ctx) {
  int nv = ctx->nv, nefc = ctx->nefc;
  mjtNum* LTJ = ctx->LTJ;
  mjtNum local[36];

  // start with Hcone = H
  mju_copy(ctx->Lcone, ctx->L, ctx->nL);

  // add contributions
  for (int i=0; i < nefc; i++) {
    if (ctx->efc_state[i] == mjCNSTRSTATE_CONE) {
      mjContact* con = ctx->contact + ctx->efc_id[i];
      int dim = con->dim;

      // Cholesky of local Hessian
      mju_copy(local, con->H, dim*dim);
      mju_cholFactor(local, dim, mjMINVAL);

      // sparse
      if (ctx->is_sparse) {
        // get nnz for row i (same for all rows in contact)
        const int nnz = ctx->J_rownnz[i];

        // compute LTJ = L'*J for this contact
        mju_zero(LTJ, dim*nnz);
        for (int r=0; r < dim; r++) {
          for (int c=0; c <= r; c++) {
            mju_addToScl(LTJ+c*nnz, ctx->J+ctx->J_rowadr[i+r], local[r*dim+c], nnz);
          }
        }

        // update
        for (int r=0; r < dim; r++) {
          mju_cholUpdateSparse(ctx->Lcone, LTJ+r*nnz, nv, 1,
                               ctx->L_rownnz, ctx->L_rowadr, ctx->L_colind, nnz,
                               ctx->J_colind+ctx->J_rowadr[i+r], ctx->cholscratch);
        }
      }

      // dense
      else {
        // compute LTJ = L'*J for this contact row
        mju_zero(LTJ, dim*nv);
        for (int r=0; r < dim; r++) {
          for (int c=0; c <= r; c++) {
            mju_addToScl(LTJ+c*nv, ctx->J+(i+r)*nv, local[r*dim+c], nv);
          }
        }

        // update
        for (int r=0; r < dim; r++) {
          mju_cholUpdate(ctx->Lcone, LTJ+r*nv, nv, 1);
        }
      }

      // count updates
      ctx->nupdate += dim;

      // advance to next constraint
      i += (dim-1);
    }
  }
}


// predict whether folding is faster than rank-1 updates for the current set, comparing flop counts:
// update path:   U = sum over cone contacts of dim * pathcost of the contact's last dof (a rank-1
//                update walks the reverse-etree path of its last dof; lower dofs merge into it)
// folded path:   Lflops/2 for the factorization plus S for J'*D*J over the D != 0 rows
// fold when 10 U > 3 (Lflops + 2 S): an update flop costs about twice a factorization flop
static int HessianConeFoldFaster(const mjPrimalContext* ctx) {
  int nefc = ctx->nefc;

  // U: update flops, S: J'*D*J flops
  mjtSize U = 0, S = 0;
  for (int i=0; i < nefc; i++) {
    int nnz = ctx->J_rownnz[i];
    if (ctx->efc_state[i] == mjCNSTRSTATE_QUADRATIC) {
      S += (mjtSize)nnz * nnz;
    } else if (ctx->efc_state[i] == mjCNSTRSTATE_CONE) {
      int dim = ctx->contact[ctx->efc_id[i]].dim;
      int lastcol = ctx->J_colind[ctx->J_rowadr[i] + nnz - 1];
      U += (mjtSize)dim * ctx->pathcost[lastcol];
      S += (mjtSize)dim * nnz * nnz;
      i += dim - 1;
    }
  }

  return 10*U > 3*(ctx->Lflops + 2*S);
}


// elliptic case: Hcone = H + cone_contributions
static void HessianCone(mjData* d, mjPrimalContext* ctx) {
  // sparse mode with many sliding contacts: one refactorization can beat
  // sum(dim) rank-1 updates per cone contact. The decision depends only on
  // sparsity patterns and efc_state, so it is cached until a state change.
  if (ctx->is_sparse) {
    if (ctx->cone_fold < 0) {
      ctx->cone_fold = HessianConeFoldFaster(ctx);
    }
    if (ctx->cone_fold) {
      HessianConeFolded(d, ctx);
      return;
    }
  }

  HessianConeUpdate(d, ctx);
}


// incremental update to Hessian factor due to changes in efc_state
static void HessianIncremental(mjData* d, mjPrimalContext* ctx, const int* oldstate) {
  int rank, nv = ctx->nv, nefc = ctx->nefc;
  mjtNum* cholupd = ctx->cholupd;

  // clear update counter
  ctx->nupdate = 0;

  // update H factorization
  for (int i=0; i < nefc; i++) {
    int flag_update = -1;

    // any state change invalidates the cached cone fold decision
    if (oldstate[i] != ctx->efc_state[i]) {
      ctx->cone_fold = -1;
    }

    // add quad
    if (oldstate[i] != mjCNSTRSTATE_QUADRATIC && ctx->efc_state[i] == mjCNSTRSTATE_QUADRATIC) {
      flag_update = 1;
    }

    // subtract quad
    else if (oldstate[i] == mjCNSTRSTATE_QUADRATIC && ctx->efc_state[i] != mjCNSTRSTATE_QUADRATIC) {
      flag_update = 0;
    }

    // perform update if flagged
    if (flag_update != -1) {
      // update with cholupd = J(i,:)*sqrt(D[i]))
      if (ctx->is_sparse) {
        // get nnz and adr of row i
        const int nnz = ctx->J_rownnz[i], adr = ctx->J_rowadr[i];

        // scale cholupd
        mju_scl(cholupd, ctx->J+adr, mju_sqrt(ctx->efc_D[i]), nnz);

        // sparse update or downdate
        rank = mju_cholUpdateSparse(ctx->L, cholupd, nv, flag_update,
                                    ctx->L_rownnz, ctx->L_rowadr, ctx->L_colind, nnz,
                                    ctx->J_colind+adr, ctx->cholscratch);
      } else {
        mju_scl(cholupd, ctx->J+i*nv, mju_sqrt(ctx->efc_D[i]), nv);
        rank = mju_cholUpdate(ctx->L, cholupd, nv, flag_update);
      }
      ctx->nupdate++;

      // recompute H directly if accuracy lost
      if (rank < nv) {
        FactorizeHessian(d, ctx, /*flg_recompute=*/1);

        // nothing else to do
        return;
      }
    }
  }

  // add cones if present
  if (ctx->ncone) {
    HessianCone(d, ctx);
  }
}


// driver
static void mj_solPrimal(const mjModel* m, mjData* d, int island, int maxiter, int flg_Newton) {
  int iter = 0;
  mjtNum alpha, beta;
  mjPrimalContext ctx;
  mj_markStack(d);

  // make context
  PrimalPointers(m, d, &ctx, island);

  // unconstrained metric island: mj_fwdAcceleration already solved
  // (M + S)*qacc_smooth = qfrc_smooth - qfrc_bias via mj_effSolve (which includes all metric
  // terms S), so with nefc == 0 the primal cost 1/2*(a - qacc_smooth)^T*Mtilde*(a - qacc_smooth)
  // is minimized directly at qacc = qacc_smooth
  if (island >= 0 && ctx.nefc == 0) {
    mju_copy(ctx.qacc, ctx.qacc_smooth, ctx.nv);
    mju_zero(ctx.qfrc_constraint, ctx.nv);
    if (island < mjNISLAND) {
      d->solver_niter[island] = 0;
    }
    mj_freeStack(d);
    return;
  }

  PrimalAllocate(m, d, &ctx, flg_Newton);

  // local copies
  int nv   = ctx.nv;
  int nefc = ctx.nefc;
  int* oldstate = ctx.oldstate;

  // compute Ma = Mtilde * qacc
  mju_mulSymVecSparse(ctx.Ma, ctx.M, ctx.qacc, nv,
                      ctx.M_rownnz, ctx.M_rowadr, ctx.M_colind);
  if (ctx.flg_metric) {
    if (ctx.island < 0) {
      mj_effMulAdd(m, d, ctx.Ma, ctx.qacc, /*flg_contact=*/1);
    } else {
      mj_effMulAddIsland(m, d, ctx.Ma, ctx.qacc, ctx.island);
    }
  }


  // compute Jaref = J * qacc - aref  (dense or sparse)
  if (!ctx.is_sparse) {
    mju_mulMatVec(ctx.Jaref, ctx.J, ctx.qacc, nefc, nv);
  } else {
    mju_mulMatVecSparse(ctx.Jaref, ctx.J, ctx.qacc, nefc,
                        ctx.J_rownnz, ctx.J_rowadr, ctx.J_colind, ctx.J_rowsuper);
  }
  mju_subFrom(ctx.Jaref, ctx.efc_aref, nefc);

  // first update
  PrimalUpdateConstraint(&ctx, flg_Newton & (m->opt.cone == mjCONE_ELLIPTIC));
  PrimalUpdateGrad(&ctx);

  // compute and save scaling factor: inverse trace of the inertia matrix
  mjtNum scale = inertiaScale(m, d, island);
  ctx.scale = scale;

  // IPC: fold the contact class and the efc rows into the solver-owned copy of the metric
  // blocks, so the CG preconditioner sees the contact curvature the operator carries
  if (ctx.ep && !mj_effPrecFold(m, d, ctx.ep, ctx.nefc, ctx.efc_D, ctx.is_sparse, ctx.J,
                                ctx.J_rownnz, ctx.J_rowadr, ctx.J_colind)) {
    ctx.ep = NULL;
  }

  // Mgrad = M \ grad: the CG preconditioned gradient, also the convergence certificate
  PrimalUpdateMgrad(&ctx, /*flg_Newton=*/0);

  // convergence certificate: the cost is strongly convex in the M-norm, bounding the
  // suboptimality by the duality gap at the current constraint forces:
  //   cost(qacc) - cost* <= 0.5 * grad'*M^-1*grad
  // if already below tolerance (e.g. good warmstart), skip the Hessian and the main loop
  int flg_gap = mju_max(0, 0.5*scale*mju_dot(ctx.grad, ctx.Mgrad, nv)) < m->opt.tolerance;

  // the gap bounds the *cost* suboptimality; on stiff constraints this permits force
  // errors of order sqrt(2*gap*stiffness). Newton solutions are characteristically
  // force-accurate, so Newton zero-iteration exits also require the gradient criterion;
  // CG solutions are characteristically cost-accurate and exit on the gap alone
  int flg_gradient = scale*mju_norm(ctx.grad, nv) < m->opt.tolerance;
  int flg_certificate = flg_gap && (!flg_Newton || flg_gradient);
  int flg_done = flg_certificate;

  // Newton: compute and factorize Hessian, Mgrad = H \ grad
  if (!flg_done && flg_Newton) {
    if (MakeHessian(d, &ctx)) {
      FactorizeHessian(d, &ctx, /*flg_recompute=*/0);
      PrimalUpdateMgrad(&ctx, /*flg_Newton=*/1);

      // Newton decrement already below tolerance: converged, skip the first line search
      // (gradient-gated like the certificate: H^-1 suppresses stiff-direction force errors)
      flg_done = flg_gradient &&
                 mju_max(0, 0.5*scale*mju_dot(ctx.grad, ctx.Mgrad, nv)) < m->opt.tolerance;
    } else {
      // arena exhausted during Hessian/Cholesky setup: fall back to CG
      flg_Newton = 0;
      flg_done = flg_gap;
      if (!flg_done) {
        // PrimalAllocate unconditionally carves cholupd[nv], cholscratch[nv] (sparse) /
        // L[nv*nv] (dense), and LTJ[6*nv] (elliptic) when flg_Newton is set; reuse them
        // for the 4 CG vectors now that Newton is disabled
        ctx.gradold  = ctx.cholupd;
        ctx.Mgradold = ctx.is_sparse ? ctx.cholscratch : ctx.L;
        mjtNum* cg_buf = ctx.LTJ ? ctx.LTJ : mjSTACKALLOC(d, 2*nv, mjtNum);
        ctx.graddif  = cg_buf;
        ctx.Mgraddif = cg_buf + nv;
      }
    }
  } else if (!flg_done && ctx.ep && ctx.ep->F) {
    mj_effCholFoldFactor(m, d, ctx.ep, ctx.nefc, ctx.efc_D, ctx.is_sparse, ctx.J, ctx.J_rownnz,
                         ctx.J_rowadr, ctx.J_colind);
    PrimalUpdateMgrad(&ctx, /*flg_Newton=*/0);
  }

  // start both with preconditioned gradient
  if (!flg_done) {
    mju_scl(ctx.search, ctx.Mgrad, -1, nv);
  }

  // main loop
  while (!flg_done && iter < maxiter) {
    // perform linesearch
    mjtNum ls_improvement;
    alpha = PrimalSearch(&ctx, m->opt.tolerance * m->opt.ls_tolerance, m->opt.ls_iterations,
                         &ls_improvement);

    // no improvement: done
    if (alpha == 0) {
      break;
    }

    // move to new solution
    mju_addToScl(ctx.qacc, ctx.search, alpha, nv);
    mju_addToScl(ctx.Ma, ctx.Mv, alpha, nv);
    mju_addToScl(ctx.Jaref, ctx.Jv, alpha, nefc);

    // save old
    if (!flg_Newton) {
      mju_copy(ctx.gradold, ctx.grad, nv);
      mju_copy(ctx.Mgradold, ctx.Mgrad, nv);
    }
    mju_copyInt(oldstate, ctx.efc_state, nefc);

    // update
    PrimalUpdateConstraint(&ctx, flg_Newton & (m->opt.cone == mjCONE_ELLIPTIC));
    if (flg_Newton) {
      HessianIncremental(d, &ctx, oldstate);
    }
    PrimalUpdateGradient(&ctx, flg_Newton);

    // count state changes
    int nchange = 0;
    for (int i=0; i < nefc; i++) {
      nchange += (ctx.efc_state[i] != oldstate[i]);
    }

    // scale improvement, gradient, save stats
    mjtNum improvement = scale * ls_improvement;
    mjtNum gradient = scale * mju_norm(ctx.grad, nv);
    saveStats(m, d, island, iter, improvement, gradient, ctx.LSslope,
              ctx.nactive, nchange, ctx.LSiter, ctx.nupdate);

    // Newton decrement: 0.5*grad'*H^-1*grad, the model's predicted improvement of the
    // next step; clamp to 0 so that tolerance == 0 keeps early termination disabled
    mjtNum decrement = flg_Newton ? mju_max(0, 0.5*scale*mju_dot(ctx.grad, ctx.Mgrad, nv)) : 0;

    // increment iteration count
    iter++;

    // termination
    if ((improvement > 0 && improvement < m->opt.tolerance) ||
         gradient < m->opt.tolerance ||
         (flg_Newton && decrement < m->opt.tolerance)) {
      break;
    }

    // update direction
    if (flg_Newton) {
      mju_scl(ctx.search, ctx.Mgrad, -1, nv);
    } else {
#ifndef mjCG_PRP
      // Hager-Zhang conjugate direction update
      mjtNum d_dot_y, y_dot_My, y_dot_Mgrad, d_dot_grad;
      mjtNum beta_hz;
      mjtNum d_norm, grad_norm, eta_k;
      const mjtNum eta = 0.01;

      // graddif = grad - gradold, Mgraddif = Mgrad - Mgradold
      mju_sub(ctx.graddif, ctx.grad, ctx.gradold, nv);
      mju_sub(ctx.Mgraddif, ctx.Mgrad, ctx.Mgradold, nv);

      // compute d'*y; restart to steepest descent if conjugacy is lost
      d_dot_y = mju_dot(ctx.search, ctx.graddif, nv);
      if (d_dot_y < mjMINVAL) {
        beta = 0;
      } else {
        // compute remaining inner products for the HZ formula
        y_dot_My = mju_dot(ctx.graddif, ctx.Mgraddif, nv);
        y_dot_Mgrad = mju_dot(ctx.graddif, ctx.Mgrad, nv);
        d_dot_grad = mju_dot(ctx.search, ctx.grad, nv);

        // primary Hager-Zhang beta coefficient
        beta_hz = (y_dot_Mgrad - 2*(y_dot_My/d_dot_y)*d_dot_grad) / d_dot_y;

        // dynamic truncation threshold to ensure d is not orthogonal to grad
        d_norm = mju_norm(ctx.search, nv);
        grad_norm = mju_norm(ctx.grad, nv);
        eta_k = -1.0 / mju_max(mjMINVAL, d_norm * mju_min(eta, grad_norm));

        // apply lower bound
        beta = mju_max(eta_k, beta_hz);
      }
#else
      // Polak-Ribiere-Plus conjugate direction update
      mju_sub(ctx.Mgraddif, ctx.Mgrad, ctx.Mgradold, nv);
      beta = mju_dot(ctx.grad, ctx.Mgraddif, nv) /
             mju_max(mjMINVAL, mju_dot(ctx.gradold, ctx.Mgradold, nv));

      // reset if negative
      if (beta < 0) {
        beta = 0;
      }
#endif

      // update
      for (int i=0; i < nv; i++) {
        ctx.search[i] = -ctx.Mgrad[i] + beta*ctx.search[i];
      }
    }
  }

  // finalize statistics
  if (island < mjNISLAND) {
    // if island is -1 (monolithic), clamp to 0
    int island_stat = island < 0 ? 0 : island;

    // update solver iterations
    d->solver_niter[island_stat] += iter;

    // set solver_nnz; if the certificate fired, no Hessian was built: report Jacobian nnz
    if (flg_Newton && !flg_certificate) {
      if (mj_isSparse(m)) {
        // two L factors if Lcone is present
        int num_factors = 1 + (ctx.Lcone != NULL);
        d->solver_nnz[island_stat] = num_factors * ctx.nL + ctx.nH;
      } else {
        d->solver_nnz[island_stat] = nv*nv;
      }
    } else {
      d->solver_nnz[island_stat] = ctx.nJ;
    }
  }

  mj_freeStack(d);
}


// CG entry point
void mj_solCG(const mjModel* m, mjData* d, int maxiter) {
  mj_solPrimal(m, d, /*island=*/-1, maxiter, /*flg_Newton=*/0);
}


// CG entry point (one island)
void mj_solCG_island(const mjModel* m, mjData* d, int island, int maxiter) {
  mj_solPrimal(m, d, island, maxiter, /*flg_Newton=*/0);
}


// Newton entry point
void mj_solNewton(const mjModel* m, mjData* d, int maxiter) {
  mj_solPrimal(m, d, /*island=*/-1, maxiter, /*flg_Newton=*/1);
}


// Newton entry point (one island)
void mj_solNewton_island(const mjModel* m, mjData* d, int island, int maxiter) {
  mj_solPrimal(m, d, island, maxiter, /*flg_Newton=*/1);
}
