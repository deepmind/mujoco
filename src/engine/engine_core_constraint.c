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

#include "engine/engine_core_constraint.h"

#include <stdio.h>
#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjxmacro.h>
#include "engine/engine_init.h"
#include "engine/engine_inline.h"
#include "engine/engine_core_util.h"
#include "engine/engine_core_smooth.h"
#include "engine/engine_memory.h"
#include "engine/engine_sleep.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"

#ifdef MEMORY_SANITIZER
  #include <sanitizer/msan_interface.h>
#endif

#ifdef mjUSEPLATFORMSIMD
  #if defined(__AVX__) && !defined(mjUSESINGLE)
    #define mjUSEAVX
  #endif  // defined(__AVX__) && !defined(mjUSESINGLE)
#endif  // mjUSEPLATFORMSIMD


//-------------------------- utility functions -----------------------------------------------------


// compute cell node Jacobians and combined chain for flex strain constraints
// npc: number of nodes per cell
// gindices: global indices of cell nodes in flex
// cell_node_jac: output array of size 3*npc*cell_nnz (allocated on stack)
// mj_{mark/free}Stack in calling function
static mjtNum* cell_pos_and_jac(const mjModel* m, mjData* d, int flex_id, int npc, const int* gindices,
                                int nv, const mjtNum* xpos_c, int* cell_chain, int* cell_nnz) {
  int* nstart = m->flex_nodeadr + flex_id;
  int* bodyid = m->flex_nodebodyid + *nstart;

  // build per-cell sparse chain: union of bodyChain for npc nodes
  *cell_nnz = 0;
  int* dof_used = mjSTACKALLOC(d, nv, int);
  int* temp_chain = mjSTACKALLOC(d, nv, int);
  mju_zeroInt(dof_used, nv);
  for (int n = 0; n < npc; n++) {
    int temp_nnz = mj_bodyChain(m, bodyid[gindices[n]], temp_chain);
    for (int k = 0; k < temp_nnz; k++) {
      dof_used[temp_chain[k]] = 1;
    }
  }
  for (int q = 0; q < nv; q++) {
    if (dof_used[q]) {
      cell_chain[(*cell_nnz)++] = q;
    }
  }

  // build per-cell node Jacobians: 3*npc x cell_nnz
  mjtNum* cell_node_jac = mjSTACKALLOC(d, 3*npc*(*cell_nnz), mjtNum);
  mju_zero(cell_node_jac, 3*npc*(*cell_nnz));
  int* chain_col = mjSTACKALLOC(d, nv, int);
  mjtNum* blk_jac = mjSTACKALLOC(d, 3*nv, mjtNum);
  for (int n = 0; n < npc; n++) {
    int body = bodyid[gindices[n]];
    int chain_n = mj_bodyChain(m, body, chain_col);
    mju_zero(blk_jac, 3*chain_n);
    mj_jacSparse(m, d, blk_jac, NULL, xpos_c + 3*n,
                 body, chain_n, chain_col, 0);
    // map node's sparse chain into cell_chain indexing
    for (int r = 0; r < 3; r++) {
      for (int k = 0; k < chain_n; k++) {
        // find chain_col[k] in cell_chain via linear scan (chain is short)
        for (int cc = 0; cc < *cell_nnz; cc++) {
          if (cell_chain[cc] == chain_col[k]) {
            cell_node_jac[(3*n + r)*(*cell_nnz) + cc] = blk_jac[r*chain_n + k];
            break;
          }
        }
      }
    }
  }

  return cell_node_jac;
}



// compute strain Jacobian from strain derivative w.r.t. cell-local node positions
// dSdx_local: input array of size 3*npc (dStrain/dNodePosition for cell nodes)
// cell_node_jac: input array of size 3*npc*cell_nnz (sparse Jacobians)
// strain_jac: output array of size cell_nnz (dStrain/dq)
static void cell_strain_jacobian(int npc, int cell_nnz,
                                 const mjtNum* dSdx_local,
                                 const mjtNum* cell_node_jac,
                                 mjtNum* strain_jac) {
  mju_zero(strain_jac, cell_nnz);
  for (int n = 0; n < npc; n++) {
    for (int c = 0; c < 3; c++) {
      mjtNum w = dSdx_local[3*n + c];
      if (w == 0) continue;
      int row = 3*n + c;
      for (int k = 0; k < cell_nnz; k++) {
        strain_jac[k] += w * cell_node_jac[row*cell_nnz + k];
      }
    }
  }
}


// allocate efc arrays on arena, return 1 on success, 0 on failure
static int arenaAllocEfc(const mjModel* m, mjData* d) {
#undef MJ_M
#define MJ_M(n) m->n
#undef MJ_D
#define MJ_D(n) d->n

  // move arena pointer to end of contact array
  d->parena = d->ncon * sizeof(mjContact);

  // poison remaining memory
#ifdef mjUSEASAN
  ASAN_POISON_MEMORY_REGION(
    (char*)d->arena + d->parena, d->narena - d->pstack - d->parena);
#endif

#define X(type, name, nr, nc)                                                 \
  d->name = mj_arenaAllocByte(d, sizeof(type) * (nr) * (nc), _Alignof(type)); \
  if (!d->name) {                                                             \
    mj_warning(d, mjWARN_CNSTRFULL, d->narena);                               \
    mj_clearEfc(d);                                                           \
    return 0;                                                                 \
  }

  MJDATA_ARENA_POINTERS_SOLVER
#undef X

#undef MJ_M
#define MJ_M(n) n
#undef MJ_D
#define MJ_D(n) n

  return 1;
}

// compute contact condim, solref, solimp, friction, adhesion
void mj_contactParam(const mjModel* m, int* condim, mjtNum* solref, mjtNum* solimp,
                     mjtNum* friction, mjtNum* adhesion, int g1, int g2, int f1, int f2) {
  mjtNum fri[3];

  // get parameters from geom1 or flex1
  int priority1 =           (f1 < 0) ? m->geom_priority[g1]     : m->flex_priority[f1];
  int condim1 =             (f1 < 0) ? m->geom_condim[g1]       : m->flex_condim[f1];
  mjtNum solmix1 =          (f1 < 0) ? m->geom_solmix[g1]       : m->flex_solmix[f1];
  const mjtNum* solref1 =   (f1 < 0) ? m->geom_solref+g1*mjNREF : m->flex_solref+f1*mjNREF;
  const mjtNum* solimp1 =   (f1 < 0) ? m->geom_solimp+g1*mjNIMP : m->flex_solimp+f1*mjNIMP;
  const mjtNum* friction1 = (f1 < 0) ? m->geom_friction+g1*3    : m->flex_friction+f1*3;
  mjtNum adhesion1 =        (f1 < 0) ? m->geom_adhesion[g1]     : 0;

  // get parameters from geom2 or flex2
  int priority2 =           (f2 < 0) ? m->geom_priority[g2]     : m->flex_priority[f2];
  int condim2 =             (f2 < 0) ? m->geom_condim[g2]       : m->flex_condim[f2];
  mjtNum solmix2 =          (f2 < 0) ? m->geom_solmix[g2]       : m->flex_solmix[f2];
  const mjtNum* solref2 =   (f2 < 0) ? m->geom_solref+g2*mjNREF : m->flex_solref+f2*mjNREF;
  const mjtNum* solimp2 =   (f2 < 0) ? m->geom_solimp+g2*mjNIMP : m->flex_solimp+f2*mjNIMP;
  const mjtNum* friction2 = (f2 < 0) ? m->geom_friction+g2*3    : m->flex_friction+f2*3;
  mjtNum adhesion2 =        (f2 < 0) ? m->geom_adhesion[g2]     : 0;

  // adhesion: each surface contributes its own attraction
  *adhesion = adhesion1 + adhesion2;

  // different priority: copy from item with higher priority
  if (priority1 > priority2) {
    *condim = condim1;
    mju_copy(solref, solref1, mjNREF);
    mju_copy(solimp, solimp1, mjNIMP);
    mju_copy(fri, friction1, 3);
    *adhesion = adhesion1;
  }
  else if (priority1 < priority2) {
    *condim = condim2;
    mju_copy(solref, solref2, mjNREF);
    mju_copy(solimp, solimp2, mjNIMP);
    mju_copy(fri, friction2, 3);
    *adhesion = adhesion2;
  }

  // same priority
  else {
    // condim: max
    *condim = mjMAX(condim1, condim2);

    // compute solver mix factor
    mjtNum mix;
    if (solmix1 >= mjMINVAL && solmix2 >= mjMINVAL) {
      mix = solmix1 / (solmix1 + solmix2);
    } else if (solmix1 < mjMINVAL && solmix2 < mjMINVAL) {
      mix = 0.5;
    } else if (solmix1 < mjMINVAL) {
      mix = 0.0;
    } else {
      mix = 1.0;
    }

    // reference standard: mix
    if (solref1[0] > 0 && solref2[0] > 0) {
      for (int i=0; i < mjNREF; i++) {
        solref[i] = mix*solref1[i] + (1-mix)*solref2[i];
      }
    }

    // reference direct: min
    else {
      for (int i=0; i < mjNREF; i++) {
        solref[i] = mju_min(solref1[i], solref2[i]);
      }
    }

    // impedance: mix
    for (int i=0; i < mjNIMP; i++) {
      solimp[i] = mix*solimp1[i] + (1-mix)*solimp2[i];
    }

    // friction: max
    for (int i=0; i < 3; i++) {
      fri[i] = mju_max(friction1[i], friction2[i]);
    }
  }

  // unpack 5D friction
  friction[0] = fri[0];
  friction[1] = fri[0];
  friction[2] = fri[1];
  friction[3] = fri[2];
  friction[4] = fri[2];

  // SHOULD NOT OCCUR
  if (*condim > 6 || *condim < 1) {
    mjERROR("Invalid condim value: %d", *condim);
  }
}


// assign/clamp contact friction parameters
void mj_assignFriction(const mjModel* m, mjtNum* target, const mjtNum* source) {
  if (mjENABLED(mjENBL_OVERRIDE)) {
    for (int i=0; i < 5; i++) {
      target[i] = mju_max(mjMINMU, m->opt.o_friction[i]);
    }
  } else {
    for (int i=0; i < 5; i++) {
      target[i] = mju_max(mjMINMU, source[i]);
    }
  }
}



// assign/override contact reference parameters
void mj_assignRef(const mjModel* m, mjtNum* target, const mjtNum* source) {
  if (mjENABLED(mjENBL_OVERRIDE)) {
    mju_copy(target, m->opt.o_solref, mjNREF);
  } else {
    mju_copy(target, source, mjNREF);
  }
}


// assign/override contact impedance parameters
void mj_assignImp(const mjModel* m, mjtNum* target, const mjtNum* source) {
  if (mjENABLED(mjENBL_OVERRIDE)) {
    mju_copy(target, m->opt.o_solimp, mjNIMP);
  } else {
    mju_copy(target, source, mjNIMP);
  }
}


// assign/override contact margin
mjtNum mj_assignMargin(const mjModel* m, mjtNum source) {
  if (mjENABLED(mjENBL_OVERRIDE)) {
    return m->opt.o_margin;
  } else {
    return source;
  }
}


// add contact to d->contact list; return 0 if success; 1 if buffer full
int mj_addContact(const mjModel* m, mjData* d, const mjContact* con) {
  // move arena pointer back to the end of the existing contact array and invalidate efc_ arrays
  mj_clearEfc(d);

  // copy contact
  mjContact* dst = mj_arenaAllocByte(d, sizeof(mjContact), _Alignof(mjContact));
  if (!dst) {
    mj_warning(d, mjWARN_CONTACTFULL, d->ncon);
    return 1;
  }
  *dst = *con;

  // increase counter, return success
  d->ncon++;
  return 0;
}


// add #size rows to constraint Jacobian; set pos, margin, frictionloss, type, id
static void mj_addConstraint(const mjModel* m, mjData* d,
                             const mjtNum* jac, const mjtNum* pos,
                             const mjtNum* margin, mjtNum frictionloss,
                             int size, int type, int id, int NV, const int* chain) {
  int empty, nv = m->nv, nefc = d->nefc;
  int *nnz = d->efc_J_rownnz, *adr = d->efc_J_rowadr, *ind = d->efc_J_colind;
  mjtNum *J = d->efc_J;

  // init empty guard for constraints other than contact
  if (type == mjCNSTR_CONTACT_FRICTIONLESS ||
      type == mjCNSTR_CONTACT_PYRAMIDAL ||
      type == mjCNSTR_CONTACT_ELLIPTIC) {
    empty = 0;
  } else {
    empty = 1;
  }

  // dense: copy entire Jacobian
  if (!mj_isSparse(m)) {
    // make sure jac is not empty
    if (empty) {
      for (int i=0; i < size*nv; i++) {
        if (jac[i]) {
          empty = 0;
          break;
        }
      }
    }

    // copy if not empty
    if (!empty) {
      mju_copy(J + nefc*nv, jac, size*nv);
    }
  }

  // sparse: copy chain
  else {
    // clamp NV (in case -1 was used in constraint construction)
    NV = mjMAX(0, NV);

    if (NV) {
      empty = 0;
    } else if (empty) {
      // all rows are empty, return early
      return;
    }

    // chain required in sparse mode
    if (NV && !chain) {
      mjERROR("called with dense arguments");
    }

    // process size elements
    for (int i=0; i < size; i++) {
      // set row address
      adr[nefc+i] = (nefc+i ? adr[nefc+i-1]+nnz[nefc+i-1] : 0);

      // set row descriptor
      nnz[nefc+i] = NV;

      // copy if not empty
      if (NV) {
        mju_copyInt(ind + adr[nefc+i], chain, NV);
        mju_copy(J + adr[nefc+i], jac + i*NV, NV);
      }
    }
  }

  // all rows empty: skip constraint
  if (empty) {
    return;
  }

  // set constraint pos, margin, frictionloss, type, id
  for (int i=0; i < size; i++) {
    d->efc_pos[nefc+i] = (pos ? pos[i] : 0);
    d->efc_margin[nefc+i] = (margin ? margin[i] : 0);
    d->efc_frictionloss[nefc+i] = frictionloss;
    d->efc_type[nefc+i] = type;
    d->efc_id[nefc+i] = id;
  }

  // increase counters
  d->nefc += size;
  if (type == mjCNSTR_EQUALITY) {
    d->ne += size;
  } else if (type == mjCNSTR_FRICTION_DOF || type == mjCNSTR_FRICTION_TENDON) {
    d->nf += size;
  } else if (type == mjCNSTR_LIMIT_JOINT || type == mjCNSTR_LIMIT_TENDON) {
    d->nl += size;
  }
}


// multiply Jacobian by vector
void mj_mulJacVec(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec) {
  // exit if no constraints
  if (!d->nefc) {
    return;
  }

  // sparse Jacobian
  if (mj_isSparse(m))
    mju_mulMatVecSparse(res, d->efc_J, vec, d->nefc,
                        d->efc_J_rownnz, d->efc_J_rowadr,
                        d->efc_J_colind, d->efc_J_rowsuper);

  // dense Jacobian
  else {
    mju_mulMatVec(res, d->efc_J, vec, d->nefc, m->nv);
  }
}


// multiply JacobianT by vector
void mj_mulJacTVec(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec) {
  // exit if no constraints
  if (!d->nefc) {
    return;
  }

  // sparse Jacobian
  if (mj_isSparse(m)) {
    mju_mulMatTVecSparse(res, d->efc_J, vec, d->nefc, m->nv,
                        d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind);
  }

  // dense Jacobian
  else {
    mju_mulMatTVec(res, d->efc_J, vec, d->nefc, m->nv);
  }
}


// compute global anchor points for connect/weld equality constraints
static void mj_equalityAnchors(const mjModel* m, const mjData* d, int eq_id,
                               mjtNum pos1[3], mjtNum pos2[3],
                               int* body1, int* body2) {
  mjtEq type = (mjtEq) m->eq_type[eq_id];
  int obj1 = m->eq_obj1id[eq_id];
  int obj2 = m->eq_obj2id[eq_id];

  if (m->eq_objtype[eq_id] == mjOBJ_BODY) {
    const mjtNum* data = m->eq_data + mjNEQDATA*eq_id;
    if (type == mjEQ_CONNECT) {
      mju_mulMatVec3(pos1, d->xmat + 9*obj1, data);
      mju_addTo3(pos1, d->xpos + 3*obj1);
      mju_mulMatVec3(pos2, d->xmat + 9*obj2, data + 3);
      mju_addTo3(pos2, d->xpos + 3*obj2);
    } else {
      // weld uses data+3*(1-j) for anchor
      mju_mulMatVec3(pos1, d->xmat + 9*obj1, data + 3);
      mju_addTo3(pos1, d->xpos + 3*obj1);
      mju_mulMatVec3(pos2, d->xmat + 9*obj2, data);
      mju_addTo3(pos2, d->xpos + 3*obj2);
    }
    *body1 = obj1;
    *body2 = obj2;
  } else {
    mju_copy3(pos1, d->site_xpos + 3*obj1);
    mju_copy3(pos2, d->site_xpos + 3*obj2);
    *body1 = m->site_bodyid[obj1];
    *body2 = m->site_bodyid[obj2];
  }
}


//--------------------- instantiate constraints by type --------------------------------------------

// equality constraints
void mj_instantiateEquality(const mjModel* m, mjData* d) {
  int issparse = mj_isSparse(m), nv = m->nv;
  int id[2], size, NV, NV2, *chain = NULL, *chain2 = NULL;
  int flex_edgeadr, flex_edgenum;
  mjtNum cpos[6], pos[2][3], ref[2], dif, deriv;
  mjtNum quat[4], quat1[4], quat2[4], quat3[4], axis[3];
  mjtNum *jac[2], *jacdif, *data;

  // disabled or no equality constraints: return
  if (mjDISABLED(mjDSBL_EQUALITY) || m->nemax == 0) {
    return;
  }

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

  mj_markStack(d);

  // allocate space
  jac[0] = mjSTACKALLOC(d, 6*nv, mjtNum);
  jac[1] = mjSTACKALLOC(d, 6*nv, mjtNum);
  jacdif = mjSTACKALLOC(d, 6*nv, mjtNum);
  if (issparse) {
    chain = mjSTACKALLOC(d, nv, int);
    chain2 = mjSTACKALLOC(d, nv, int);
  }

  // find active equality constraints
  for (int i=0; i < m->neq; i++) {
    // skip inactive
    if (!d->eq_active[i]) {
      continue;
    }

    // skip sleeping
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_EQUALITY, i) == mjS_ASLEEP) {
      continue;
    }

    // get constraint data
    data = m->eq_data + mjNEQDATA*i;
    id[0] = m->eq_obj1id[i];
    id[1] = m->eq_obj2id[i];
    size = 0;
    NV = 0;
    NV2 = 0;
    int body_id[2];

    // process according to type
    switch ((mjtEq) m->eq_type[i]) {
    case mjEQ_CONNECT:              // connect bodies with ball joint
      // find global points, body semantic
      mj_equalityAnchors(m, d, i, pos[0], pos[1], body_id, body_id + 1);

      // compute position error
      mju_sub3(cpos, pos[0], pos[1]);

      // compute Jacobian difference (opposite of contact: 0 - 1)
      NV = mj_jacDifPair(m, d, chain, body_id[1], body_id[0], pos[1], pos[0],
                         jac[1], jac[0], jacdif, NULL, NULL, NULL, issparse,
                         /*flg_skipcommon=*/0);

      // copy difference into jac[0]
      mju_copy(jac[0], jacdif, 3*NV);

      size = 3;
      break;

    case mjEQ_WELD:                 // fix relative position and orientation
      // find global points, body semantic
      mj_equalityAnchors(m, d, i, pos[0], pos[1], body_id, body_id + 1);

      // compute position error
      mju_sub3(cpos, pos[0], pos[1]);

      // get torquescale coefficient
      mjtNum torquescale = data[10];

      // compute error Jacobian (opposite of contact: 0 - 1)
      NV = mj_jacDifPair(m, d, chain, body_id[1], body_id[0], pos[1], pos[0],
                          jac[1], jac[0], jacdif,
                          jac[1]+3*nv, jac[0]+3*nv, jacdif+3*nv, issparse,
                          /*flg_skipcommon=*/0);

      // copy difference into jac[0], compress translation:rotation if sparse
      mju_copy(jac[0], jacdif, 3*NV);
      mju_copy(jac[0]+3*NV, jacdif+3*nv, 3*NV);

      // orientation, body semantic
      if (m->eq_objtype[i] == mjOBJ_BODY) {
        // compute orientation error: neg(q1) * q0 * relpose (axis components only)
        mjtNum* relpose = data+6;
        mji_mulQuat(quat, d->xquat+4*id[0], relpose);   // quat = q0*relpose
        mji_negQuat(quat1, d->xquat+4*id[1]);           // quat1 = neg(q1)
      }

      // orientation, site semantic
      else {
        mji_copy4(quat, d->site_xquat+4*id[0]);
        mji_negQuat(quat1, d->site_xquat+4*id[1]);
      }

      mji_mulQuat(quat2, quat1, quat);
      mji_scl3(cpos+3, quat2+1, torquescale);         // scale axis components by torquescale

      // correct rotation Jacobian: 0.5 * neg(q1) * (jac0-jac1) * q0 * relpose
      for (int j=0; j < NV; j++) {
        // axis = [jac0-jac1]_col(j)
        axis[0] = jac[0][3*NV+j];
        axis[1] = jac[0][4*NV+j];
        axis[2] = jac[0][5*NV+j];

        // apply formula
        mji_mulQuatAxis(quat2, quat1, axis);    // quat2 = neg(q1)*(jac0-jac1)
        mji_mulQuat(quat3, quat2, quat);        // quat3 = neg(q1)*(jac0-jac1)*q0*relpose

        // correct Jacobian
        jac[0][3*NV+j] = 0.5*quat3[1];
        jac[0][4*NV+j] = 0.5*quat3[2];
        jac[0][5*NV+j] = 0.5*quat3[3];
      }

      // scale rotational jacobian by torquescale
      mju_scl(jac[0]+3*NV, jac[0]+3*NV, torquescale, 3*NV);

      size = 6;
      break;

    case mjEQ_JOINT:                // couple joint values with cubic
    case mjEQ_TENDON:               // couple tendon lengths with cubic
      // get scalar positions and their Jacobians
      for (int j=0; j < 1+(id[1] >= 0); j++) {
        if (m->eq_type[i] == mjEQ_JOINT) {    // joint object
          pos[j][0] = d->qpos[m->jnt_qposadr[id[j]]];
          ref[j] = m->qpos0[m->jnt_qposadr[id[j]]];

          // make Jacobian: sparse or dense
          if (issparse) {
            // add first or second joint
            if (j == 0) {
              NV = 1;
              chain[0] = m->jnt_dofadr[id[j]];
              jac[j][0] = 1;
            } else {
              NV2 = 1;
              chain2[0] = m->jnt_dofadr[id[j]];
              jac[j][0] = 1;
            }
          } else {
            mju_zero(jac[j], nv);
            jac[j][m->jnt_dofadr[id[j]]] = 1;
          }
        } else {                            // tendon object
          pos[j][0] = d->ten_length[id[j]];
          ref[j] = m->tendon_length0[id[j]];

          // copy Jacobian: sparse or dense
          if (issparse) {
            if (j == 0) {
              NV = m->ten_J_rownnz[id[j]];
              mju_copyInt(chain, m->ten_J_colind+m->ten_J_rowadr[id[j]], NV);
              mju_copy(jac[j], d->ten_J+m->ten_J_rowadr[id[j]], NV);
            } else {
              NV2 = m->ten_J_rownnz[id[j]];
              mju_copyInt(chain2, m->ten_J_colind+m->ten_J_rowadr[id[j]], NV2);
              mju_copy(jac[j], d->ten_J+m->ten_J_rowadr[id[j]], NV2);
            }
          } else {
            mju_sparse2dense(jac[j], d->ten_J, 1, nv, m->ten_J_rownnz+id[j], m->ten_J_rowadr+id[j], m->ten_J_colind);
          }
        }
      }

      // both objects defined
      if (id[1] >= 0) {
        // compute position error
        dif = pos[1][0] - ref[1];
        cpos[0] = pos[0][0] - ref[0] - data[0] -
                  (data[1]*dif + data[2]*dif*dif + data[3]*dif*dif*dif + data[4]*dif*dif*dif*dif);

        // compute derivative
        deriv = data[1] + 2*data[2]*dif + 3*data[3]*dif*dif + 4*data[4]*dif*dif*dif;

        // compute Jacobian: sparse or dense
        if (issparse) {
          NV = mju_combineSparse(jac[0], jac[1], 1, -deriv, NV, NV2, chain, chain2);
        } else {
          mju_addToScl(jac[0], jac[1], -deriv, nv);
        }
      }

      // only one object defined
      else {
        // compute position error
        cpos[0] = pos[0][0] - ref[0] - data[0];

        // jac[0] already has the correct Jacobian
      }

      size = 1;
      break;

    case mjEQ_FLEXSTRAIN: {
      // each constraint represents a single element (3D cell or 2D face)
      int f = id[0];
      int nodenum = m->flex_nodenum[f];
      int interp = m->flex_interp[f];
      int order = interp < 0 ? -interp : interp;
      int shell_mode = (interp < 0);

      // skip if not interpolated (order == 0 or no nodes)
      if (!order || !nodenum) {
        break;
      }

      // only order 1 (trilinear) and 2 (quadratic) are supported
      if (order > 2) {
        mjERROR("flex strain constraints only support order 1 and 2, got %d", order);
      }

      int cx = m->flex_cellnum[3*f+0];
      int cy = m->flex_cellnum[3*f+1];
      int cz = m->flex_cellnum[3*f+2];
      int nstart = m->flex_nodeadr[f];
      int* bodyid = m->flex_nodebodyid + nstart;

      // nodes per element and element index
      int npe;
      int elem_idx;
      if (shell_mode) {
        npe = (order+1) * (order+1);
        elem_idx = (int)data[0];  // face element index
      } else {
        npe = (order+1) * (order+1) * (order+1);
        int ci = (int)data[0];
        int cj = (int)data[1];
        int ck = (int)data[2];
        elem_idx = ci * cy * cz + cj * cz + ck;
      }

      mj_markStack(d);

      // get element node indices
      int gindices[125];  // max npc = 125 for quadratic
      if (shell_mode) {
        mju_flexGatherFaceState(order, cx, cy, cz, elem_idx,
                                NULL, NULL, NULL, NULL, NULL, NULL, gindices, NULL);
      } else {
        int ci = (int)data[0], cj = (int)data[1], ck = (int)data[2];
        mju_flexGatherCellState(order, cy, cz, ci, cj, ck,
                                NULL, NULL, NULL, NULL, NULL, NULL, gindices, NULL);
      }

      // compute positions only for element nodes (npe << nodenum)
      mjtNum* xpos_e = mjSTACKALLOC(d, 3*npe, mjtNum);
      mjtNum* refpos_e = mjSTACKALLOC(d, 3*npe, mjtNum);
      for (int n = 0; n < npe; n++) {
        int gn = gindices[n];
        if (m->flex_centered[f] ||
            (m->flex_node[3*(gn + nstart)+0] == 0 &&
             m->flex_node[3*(gn + nstart)+1] == 0 &&
             m->flex_node[3*(gn + nstart)+2] == 0)) {
          mju_copy3(xpos_e + 3*n, d->xpos + 3*bodyid[gn]);
        } else {
          mju_mulMatVec3(xpos_e + 3*n, d->xmat + 9*bodyid[gn], m->flex_node + 3*(gn + nstart));
          mju_addTo3(xpos_e + 3*n, d->xpos + 3*bodyid[gn]);
        }
        mju_copy3(refpos_e + 3*n, m->flex_node0 + 3*(gn + nstart));
      }

      // compute corotational quaternion
      mjtNum elem_quat[4] = {1, 0, 0, 0};
      if (shell_mode) {
        // determine face normal axis from elem_idx
        int face_sizes[6] = {cy*cz, cy*cz, cx*cz, cx*cz, cx*cy, cx*cy};
        int face_normals[6] = {0, 0, 1, 1, 2, 2};
        int cumul = 0, normal_axis = 0;
        for (int ff = 0; ff < 6; ff++) {
          if (elem_idx < cumul + face_sizes[ff]) {
            normal_axis = face_normals[ff];
            break;
          }
          cumul += face_sizes[ff];
        }
        int na0 = (normal_axis + 1) % 3;
        int na1 = (normal_axis + 2) % 3;

        // compute corotational rotation from 2D deformation gradient at face center
        mjtNum d0 = refpos_e[3*(npe - 1) + na0] - refpos_e[na0];
        mjtNum d1 = refpos_e[3*(npe - 1) + na1] - refpos_e[na1];
        mjtNum t1[3] = {0, 0, 0}, t2[3] = {0, 0, 0};
        int idx = 0;
        for (int l0 = 0; l0 <= order; l0++) {
          for (int l1 = 0; l1 <= order; l1++) {
            mjtNum grad0 = mju_flexDphi(0.5, l0, order) * mju_flexPhi(0.5, l1, order);
            mjtNum grad1 = mju_flexPhi(0.5, l0, order) * mju_flexDphi(0.5, l1, order);
            for (int c = 0; c < 3; c++) {
              t1[c] += xpos_e[3*idx + c] * grad0;
              t2[c] += xpos_e[3*idx + c] * grad1;
            }
            idx++;
          }
        }
        if (d0 > mjMINVAL) mju_scl3(t1, t1, 1 / d0);
        if (d1 > mjMINVAL) mju_scl3(t2, t2, 1 / d1);
        mjtNum normal[3];
        mju_cross(normal, t1, t2);
        mjtNum mat[9] = {0};
        mjtNum* vecs[3];
        vecs[na0] = t1;
        vecs[na1] = t2;
        vecs[normal_axis] = normal;
        for (int col = 0; col < 3; col++) {
          mat[0*3 + col] = vecs[col][0];
          mat[1*3 + col] = vecs[col][1];
          mat[2*3 + col] = vecs[col][2];
        }
        mju_unit4(elem_quat);
        mju_mat2Rot(elem_quat, mat);
        mju_negQuat(elem_quat, elem_quat);
      } else {
        mjtNum center[3] = {0.5, 0.5, 0.5};
        mjtNum mat[9];
        mju_defGradient(mat, center, xpos_e, order);
        for (int col = 0; col < 3; col++) {
          mjtNum dc = refpos_e[3*(npe - 1) + col] - refpos_e[col];
          if (dc > mjMINVAL) {
            mjtNum inv = 1 / dc;
            mat[0*3 + col] *= inv;
            mat[1*3 + col] *= inv;
            mat[2*3 + col] *= inv;
          }
        }
        mju_unit4(elem_quat);
        mju_mat2Rot(elem_quat, mat);
        mju_negQuat(elem_quat, elem_quat);
      }

      // build per-element sparse chain and node Jacobians
      int* elem_chain = mjSTACKALLOC(d, nv, int);
      int elem_nnz = 0;
      mjtNum* elem_node_jac = cell_pos_and_jac(m, d, f, npe, gindices, nv, xpos_e, elem_chain,
                                               &elem_nnz);


      mjtNum* strain_jac = mjSTACKALLOC(d, elem_nnz, mjtNum);
      mjtNum* dSdx_local = mjSTACKALLOC(d, 3*npe, mjtNum);

      // for dense mode: allocate and zero a dense Jacobian buffer once
      mjtNum* dense_jac = NULL;
      if (!issparse) {
        dense_jac = mjSTACKALLOC(d, nv, mjtNum);
        mju_zero(dense_jac, nv);
      }

      // read eigenmode data from flex_stiffness
      int ndof_elem = 3 * npe;
      int stiffnessadr = m->flex_stiffnessadr[f];
      int neig = 0;
      const mjtNum* k_elem = NULL;
      if (stiffnessadr >= 0) {
        k_elem = m->flex_stiffness + stiffnessadr
               + elem_idx * ndof_elem * ndof_elem;
        neig = (int)k_elem[0];
      }

      // compute displacement in corotational frame
      mjtNum* displ_e = mjSTACKALLOC(d, ndof_elem, mjtNum);
      for (int n = 0; n < npe; n++) {
        // rotate xpos_e to corotational frame
        mjtNum xrot[3];
        mju_rotVecQuat(xrot, xpos_e + 3*n, elem_quat);
        displ_e[3*n + 0] = xrot[0] - refpos_e[3*n + 0];
        displ_e[3*n + 1] = xrot[1] - refpos_e[3*n + 1];
        displ_e[3*n + 2] = xrot[2] - refpos_e[3*n + 2];
      }

      // compute inverse quaternion for rotating eigenvectors to world frame
      mjtNum elem_quat_inv[4];
      mju_negQuat(elem_quat_inv, elem_quat);

      // loop over eigenmodes
      for (int eig = 0; eig < neig; eig++) {
        const mjtNum* eigvec = k_elem + 1 + eig * ndof_elem;

        // constraint residual: dot product of scaled eigenvector with displacement
        mjtNum residual = 0;
        for (int j = 0; j < ndof_elem; j++) {
          residual += eigvec[j] * displ_e[j];
        }
        cpos[0] = residual;

        // rotate eigenvector to world frame for Jacobian
        for (int n = 0; n < npe; n++) {
          mju_rotVecQuat(dSdx_local + 3*n, eigvec + 3*n, elem_quat_inv);
        }

        // contract with elem_node_jac to get sparse Jacobian
        cell_strain_jacobian(npe, elem_nnz, dSdx_local, elem_node_jac, strain_jac);

        if (issparse) {
          mj_addConstraint(m, d, strain_jac, cpos, 0, 0, 1, mjCNSTR_EQUALITY, i,
                           elem_nnz, elem_chain);
        } else {
          for (int k = 0; k < elem_nnz; k++) {
            dense_jac[elem_chain[k]] = strain_jac[k];
          }
          mj_addConstraint(m, d, dense_jac, cpos, 0, 0, 1, mjCNSTR_EQUALITY, i, 0, NULL);
          for (int k = 0; k < elem_nnz; k++) {
            dense_jac[elem_chain[k]] = 0;
          }
        }
      }

      mj_freeStack(d);
      break;
    }

  case mjEQ_FLEX:
    // edge constraint mode: add one constraint per non-rigid edge
    flex_edgeadr = m->flex_edgeadr[id[0]];
    flex_edgenum = m->flex_edgenum[id[0]];
    for (int e=flex_edgeadr; e < flex_edgeadr+flex_edgenum; e++) {
      // skip rigid
      if (m->flexedge_rigid[e]) {
        continue;
      }

      // position error
      cpos[0] = d->flexedge_length[e] - m->flexedge_length0[e];

      // add constraint: sparse or dense
      if (issparse) {
        mj_addConstraint(m, d, d->flexedge_J+m->flexedge_J_rowadr[e], cpos, 0, 0,
                          1, mjCNSTR_EQUALITY, i,
                          m->flexedge_J_rownnz[e],
                          m->flexedge_J_colind+m->flexedge_J_rowadr[e]);
      } else {
        mju_zero(jac[0], nv);  // reuse first row of jac[0]
        int rowadr = m->flexedge_J_rowadr[e];
        int rownnz = m->flexedge_J_rownnz[e];
        for (int k=0; k<rownnz; k++) {
          jac[0][m->flexedge_J_colind[rowadr+k]] = d->flexedge_J[rowadr+k];
        }
        mj_addConstraint(m, d, jac[0], cpos, 0, 0, 1, mjCNSTR_EQUALITY, i, 0, NULL);
      }
    }
    break;

    default:                    // SHOULD NOT OCCUR
      mjERROR("invalid equality constraint type %d", m->eq_type[i]);
    }

    // add constraint
    if (size) {
      int efcadr = d->nefc;
      mj_addConstraint(m, d, jac[0], cpos, 0, 0,
                       size, mjCNSTR_EQUALITY, i,
                       issparse ? NV : 0,
                       issparse ? chain : NULL);

      // set tendon_efcadr
      if (m->eq_type[i] == mjEQ_TENDON && d->nefc > efcadr) {
        if (d->tendon_efcadr[id[0]] == -1) {
          d->tendon_efcadr[id[0]] = efcadr;
        }
        if (id[1] >= 0 && d->tendon_efcadr[id[1]] == -1) {
          d->tendon_efcadr[id[1]] = efcadr;
        }
      }
    }
  }

  mj_freeStack(d);
}

// subtract Jdot*v correction from result vector for equality constraints
void mj_Jdotv(const mjModel* m, mjData* d, mjtNum* result) {
  int nv = m->nv, ne = d->ne;

  // nothing to do
  if (!ne || !nv) {
    return;
  }

  int issparse = mj_isSparse(m);

  mj_markStack(d);

  // allocate scratch for jacDot matrices (translational and rotational)
  int* chain = issparse ? mjSTACKALLOC(d, nv, int) : NULL;
  mjtNum* jacdot1 = NULL;
  mjtNum* jacdot2 = NULL;
  mjtNum* jacrdot1 = NULL;
  mjtNum* jacrdot2 = NULL;

  // iterate over equality constraint efc rows
  int row = 0;
  while (row < ne) {
    int eq_id = d->efc_id[row];
    mjtEq type = (mjtEq) m->eq_type[eq_id];

    // connect or weld: compute Jdot*v for translational part
    if (type == mjEQ_CONNECT || type == mjEQ_WELD) {
      mjtNum* data = m->eq_data + mjNEQDATA*eq_id;

      // allocate translational scratch on first connect or weld
      if (!jacdot1) {
        jacdot1 = mjSTACKALLOC(d, 3*nv, mjtNum);
        jacdot2 = mjSTACKALLOC(d, 3*nv, mjtNum);
      }

      // allocate rotational scratch on first weld
      if (type == mjEQ_WELD && !jacrdot1) {
        jacrdot1 = mjSTACKALLOC(d, 3*nv, mjtNum);
        jacrdot2 = mjSTACKALLOC(d, 3*nv, mjtNum);
      }

      // compute global anchor points and body ids
      int obj1 = m->eq_obj1id[eq_id];
      int obj2 = m->eq_obj2id[eq_id];
      mjtNum pos1[3], pos2[3];
      int body1, body2;
      mj_equalityAnchors(m, d, eq_id, pos1, pos2, &body1, &body2);

      // compute jacDot*v for each body point
      mjtNum jdv1[3], jdv2[3];
      mjtNum jrdv1[3] = {0}, jrdv2[3] = {0};
      if (issparse) {
        // get merged chain for the two bodies
        int NV = mj_mergeChain(m, chain, body1, body2, /*flg_skipcommon=*/0);

        if (NV) {
          // sparse: translational and rotational
          mjtNum* jacr1 = (type == mjEQ_WELD) ? jacrdot1 : NULL;
          mjtNum* jacr2 = (type == mjEQ_WELD) ? jacrdot2 : NULL;
          mj_jacDotSparse(m, d, jacdot1, jacr1, pos1, body1, NV, chain);
          mj_jacDotSparse(m, d, jacdot2, jacr2, pos2, body2, NV, chain);

          // translational jdv = jacDot * qvel
          mju_dotSparseX3(jdv1, jdv1+1, jdv1+2, jacdot1, jacdot1+NV, jacdot1+2*NV,
                          d->qvel, NV, chain);
          mju_dotSparseX3(jdv2, jdv2+1, jdv2+2, jacdot2, jacdot2+NV, jacdot2+2*NV,
                          d->qvel, NV, chain);

          // rotational jdv for welds
          if (type == mjEQ_WELD) {
            mju_dotSparseX3(jrdv1, jrdv1+1, jrdv1+2, jacrdot1, jacrdot1+NV, jacrdot1+2*NV,
                            d->qvel, NV, chain);
            mju_dotSparseX3(jrdv2, jrdv2+1, jrdv2+2, jacrdot2, jacrdot2+NV, jacrdot2+2*NV,
                            d->qvel, NV, chain);
          }
        } else {
          mju_zero3(jdv1);
          mju_zero3(jdv2);
        }
      } else {
        // dense: translational and rotational
        mjtNum* jacr1 = (type == mjEQ_WELD) ? jacrdot1 : NULL;
        mjtNum* jacr2 = (type == mjEQ_WELD) ? jacrdot2 : NULL;
        mj_jacDot(m, d, jacdot1, jacr1, pos1, body1);
        mj_jacDot(m, d, jacdot2, jacr2, pos2, body2);

        // translational jdv = jacDot * qvel
        mju_mulMatVec(jdv1, jacdot1, d->qvel, 3, nv);
        mju_mulMatVec(jdv2, jacdot2, d->qvel, 3, nv);

        // rotational jdv for welds
        if (type == mjEQ_WELD) {
          mju_mulMatVec(jrdv1, jacrdot1, d->qvel, 3, nv);
          mju_mulMatVec(jrdv2, jacrdot2, d->qvel, 3, nv);
        }
      }

      // subtract translational Jdot*v
      result[row+0] -= jdv1[0] - jdv2[0];
      result[row+1] -= jdv1[1] - jdv2[1];
      result[row+2] -= jdv1[2] - jdv2[2];

      // advance past translational rows
      row += 3;

      // weld: compute rotational Jdot*v
      if (type == mjEQ_WELD) {
        mjtNum torquescale = data[10];

        // get body quaternions and relpose, following mj_instantiateEquality
        mjtNum q0r[4], negq1[4];   // q0r = q0*relpose, negq1 = neg(q1)
        if (m->eq_objtype[eq_id] == mjOBJ_BODY) {
          mjtNum* relpose = data+6;
          mju_mulQuat(q0r, d->xquat+4*body1, relpose);
          mju_negQuat(negq1, d->xquat+4*body2);
        } else {
          mju_mulQuat(q0r, d->xquat+4*body1, m->site_quat+4*obj1);
          mjtNum qsite1[4];
          mju_mulQuat(qsite1, d->xquat+4*body2, m->site_quat+4*obj2);
          mju_negQuat(negq1, qsite1);
        }

        // angular velocities from cvel (first 3 components are angular)
        const mjtNum* omega1 = d->cvel+6*body1;
        const mjtNum* omega2 = d->cvel+6*body2;

        // relative angular velocity: domega = omega1 - omega2
        mjtNum domega[3];
        mju_sub3(domega, omega1, omega2);

        // quaternion derivatives: qdot = 0.5 * q * (0, omega)
        mjtNum qdot0[4];
        if (m->eq_objtype[eq_id] == mjOBJ_BODY) {
          mju_derivQuat(qdot0, d->xquat+4*body1, omega1);
        } else {
          mjtNum qfull0[4];
          mju_mulQuat(qfull0, d->xquat+4*body1, m->site_quat+4*obj1);
          mju_derivQuat(qdot0, qfull0, omega1);
        }
        mjtNum qdot0r[4];  // d/dt(q0 * relpose) = qdot0 * relpose
        if (m->eq_objtype[eq_id] == mjOBJ_BODY) {
          mju_mulQuat(qdot0r, qdot0, data+6);
        } else {
          mju_copy4(qdot0r, qdot0);
        }

        // neg(qdot1): d/dt(neg(q1)) = neg(qdot1)
        mjtNum negqdot1[4];
        if (m->eq_objtype[eq_id] == mjOBJ_BODY) {
          mjtNum qdot1[4];
          mju_derivQuat(qdot1, d->xquat+4*body2, omega2);
          mju_negQuat(negqdot1, qdot1);
        } else {
          mjtNum qfull1[4], qdot1[4];
          mju_mulQuat(qfull1, d->xquat+4*body2, m->site_quat+4*obj2);
          mju_derivQuat(qdot1, qfull1, omega2);
          mju_negQuat(negqdot1, qdot1);
        }

        // Jdot_rot * v differentiates: 0.5 * neg(q1) * (J0-J1)*v * q0*relpose
        // three terms from product rule:

        // djrdv = Jrdot0*v - Jrdot1*v (rotational jacDot difference * v)
        mjtNum djrdv[3];
        mju_sub3(djrdv, jrdv1, jrdv2);

        // term1: neg(qdot1) * domega * q0r
        mjtNum t1a[4], t1[4];
        mju_mulQuatAxis(t1a, negqdot1, domega);
        mju_mulQuat(t1, t1a, q0r);

        // term2: neg(q1) * djrdv * q0r
        mjtNum t2a[4], t2[4];
        mju_mulQuatAxis(t2a, negq1, djrdv);
        mju_mulQuat(t2, t2a, q0r);

        // term3: neg(q1) * domega * qdot0r
        mjtNum t3a[4], t3[4];
        mju_mulQuatAxis(t3a, negq1, domega);
        mju_mulQuat(t3, t3a, qdot0r);

        // combine: 0.5 * (term1 + term2 + term3), take vector part, scale
        result[row+0] -= 0.5 * (t1[1] + t2[1] + t3[1]) * torquescale;
        result[row+1] -= 0.5 * (t1[2] + t2[2] + t3[2]) * torquescale;
        result[row+2] -= 0.5 * (t1[3] + t2[3] + t3[3]) * torquescale;

        row += 3;
      }
    }

    // other types: advance past all rows with this efc_id
    else {
      while (row < ne && d->efc_id[row] == eq_id) {
        row++;
      }
    }
  }

  mj_freeStack(d);
}


// return number of constraint non-zeros, handle dense and dof-less cases
static inline int mj_addConstraintCount(const mjModel* m, int size, int NV) {
  // over count for dense allocation
  if (!mj_isSparse(m)) {
    return m->nv ? size : 0;
  }
  return mjMAX(0, NV) ? size : 0;
}


// frictional DOFs and tendons
// count_only: count constraints and Jacobian nonzeros without instantiating
static int mj_instantiateFriction(const mjModel* m, mjData* d, int count_only, int* nnz) {
  int nv = m->nv, issparse = mj_isSparse(m);
  int nf = 0;
  mjtNum* jac = NULL;

  // disabled: return
  if (mjDISABLED(mjDSBL_FRICTIONLOSS)) {
    return 0;
  }

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

  if (!count_only) {
    mj_markStack(d);

    // allocate Jacobian
    jac = mjSTACKALLOC(d, nv, mjtNum);
  }

  // find frictional DOFs
  for (int i=0; i < nv; i++) {
    // no friction loss: skip
    if (!m->dof_frictionloss[i]) {
      continue;
    }

    // sleeping tree: skip
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_DOF, i) == mjS_ASLEEP) {
      continue;
    }

    if (count_only) {
      nf += mj_addConstraintCount(m, 1, 1);
      if (nnz) *nnz += 1;
    } else {
      // prepare Jacobian: sparse or dense
      if (issparse) {
        jac[0] = 1;
      } else {
        mju_zero(jac, nv);
        jac[i] = 1;
      }

      // add constraint
      mj_addConstraint(m, d, jac, 0, 0, m->dof_frictionloss[i],
                        1, mjCNSTR_FRICTION_DOF, i,
                        issparse ? 1 : 0,
                        issparse ? &i : NULL);
    }
  }

  // find frictional tendons
  for (int i=0; i < m->ntendon; i++) {
    if (m->tendon_frictionloss[i] > 0) {
      // sleeping tendon: skip
      if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, i) == mjS_ASLEEP) {
        continue;
      }

      if (count_only) {
        nf += mj_addConstraintCount(m, 1, m->ten_J_rownnz[i]);
        if (nnz) *nnz += m->ten_J_rownnz[i];
      } else {
        int efcadr = d->nefc;
        // add constraint
        if (issparse) {
          mj_addConstraint(m, d, d->ten_J + m->ten_J_rowadr[i],
                           0, 0, m->tendon_frictionloss[i],
                           1, mjCNSTR_FRICTION_TENDON, i,
                           m->ten_J_rownnz[i],
                           m->ten_J_colind+m->ten_J_rowadr[i]);
        } else {
          mju_sparse2dense(jac, d->ten_J, 1, nv, m->ten_J_rownnz+i, m->ten_J_rowadr+i, m->ten_J_colind);
          mj_addConstraint(m, d, jac, 0, 0, m->tendon_frictionloss[i],
                           1, mjCNSTR_FRICTION_TENDON, i, 0, NULL);
        }
        // set tendon_efcadr
        if (d->nefc > efcadr && d->tendon_efcadr[i] == -1) {
          d->tendon_efcadr[i] = efcadr;
        }
      }
    }
  }

  if (!count_only) {
    mj_freeStack(d);
  }

  return nf;
}


// joint and tendon limits
// count_only: count constraints and Jacobian nonzeros without instantiating
static int mj_instantiateLimit(const mjModel* m, mjData* d, int count_only, int* nnz) {
  int nv = m->nv, issparse = mj_isSparse(m);
  int nl = 0;
  mjtNum margin, value, dist, angleAxis[3];
  mjtNum *jac = NULL;

  // disabled: return
  if (mjDISABLED(mjDSBL_LIMIT)) {
    return 0;
  }

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

  if (!count_only) {
    mj_markStack(d);

    // allocate Jacobian
    jac = mjSTACKALLOC(d, nv, mjtNum);
  }

  // find joint limits
  for (int i=0; i < m->njnt; i++) {
    // no limit: skip
    if (!m->jnt_limited[i]) {
      continue;
    }

    // sleeping tree: skip
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_JOINT, i) == mjS_ASLEEP) {
      continue;
    }

    // get margin
    margin = m->jnt_margin[i];

    // HINGE or SLIDE joint
    if (m->jnt_type[i] == mjJNT_SLIDE || m->jnt_type[i] == mjJNT_HINGE) {
      // get joint value
      value = d->qpos[m->jnt_qposadr[i]];

      // process lower and upper limits
      for (int side=-1; side <= 1; side+=2) {
        // compute distance (negative: penetration)
        dist = side * (m->jnt_range[2*i+(side+1)/2] - value);

        // detect joint limit
        if (dist < margin) {
          if (count_only) {
            nl += mj_addConstraintCount(m, 1, 1);
            if (nnz) *nnz += 1;
          } else {
            // prepare Jacobian: sparse or dense
            if (issparse) {
              jac[0] = -(mjtNum)side;
            } else {
              mju_zero(jac, nv);
              jac[m->jnt_dofadr[i]] = -(mjtNum)side;
            }

            // add constraint
            mj_addConstraint(m, d, jac, &dist, &margin, 0,
                             1, mjCNSTR_LIMIT_JOINT, i,
                             issparse ? 1 : 0,
                             issparse ? m->jnt_dofadr+i : NULL);
          }
        }
      }
    }

    // BALL joint
    else if (m->jnt_type[i] == mjJNT_BALL) {
      // convert joint quaternion to axis-angle
      int adr = m->jnt_qposadr[i];
      mjtNum quat[4] = {d->qpos[adr], d->qpos[adr+1], d->qpos[adr+2], d->qpos[adr+3]};
      mju_normalize4(quat);
      mju_quat2Vel(angleAxis, quat, 1);

      // get rotation angle, normalize
      value = mju_normalize3(angleAxis);

      // compute distance, using max of range (negative: penetration)
      dist = mju_max(m->jnt_range[2*i], m->jnt_range[2*i+1]) - value;

      // detect joint limit
      if (dist < margin) {
        if (count_only) {
          nl += mj_addConstraintCount(m, 1, 3);
          if (nnz) *nnz += 3;
        }

        // sparse
        else if (issparse) {
          // prepare dof index array
          int chain[3] = {
            m->jnt_dofadr[i] + 0,
            m->jnt_dofadr[i] + 1,
            m->jnt_dofadr[i] + 2
          };

          // prepare Jacobian
          mju_scl3(jac, angleAxis, -1);

          // add constraint
          mj_addConstraint(m, d, jac, &dist, &margin, 0,
                           1, mjCNSTR_LIMIT_JOINT, i, 3, chain);
        }

        // dense
        else {
          // prepare Jacobian
          mju_zero(jac, nv);
          mju_scl3(jac + m->jnt_dofadr[i], angleAxis, -1);

          // add constraint
          mj_addConstraint(m, d, jac, &dist, &margin, 0,
                           1, mjCNSTR_LIMIT_JOINT, i, 0, 0);
        }
      }
    }
  }

  // find tendon limits
  for (int i=0; i < m->ntendon; i++) {
    if (!m->tendon_limited[i]) {
      continue;
    }

    // sleeping tendon: skip
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, i) == mjS_ASLEEP) {
      continue;
    }

    // get value = length, margin
    value = d->ten_length[i];
    margin = m->tendon_margin[i];

    // process lower and upper limits
    for (int side=-1; side <= 1; side+=2) {
      // compute distance (negative: penetration)
      dist = side * (m->tendon_range[2*i+(side+1)/2] - value);

      // detect tendon limit
      if (dist < margin) {
        if (count_only) {
          nl += mj_addConstraintCount(m, 1, m->ten_J_rownnz[i]);
          if (nnz) *nnz += m->ten_J_rownnz[i];
        } else {
          // prepare Jacobian
          int efcadr = d->nefc;
          if (issparse) {
            mju_scl(jac, d->ten_J+m->ten_J_rowadr[i], -side, m->ten_J_rownnz[i]);
            mj_addConstraint(m, d, jac, &dist, &margin, 0,
                             1, mjCNSTR_LIMIT_TENDON, i,
                             m->ten_J_rownnz[i],
                             m->ten_J_colind+m->ten_J_rowadr[i]);
          } else {
            mju_sparse2dense(jac, d->ten_J, 1, nv, m->ten_J_rownnz+i, m->ten_J_rowadr+i, m->ten_J_colind);
            mju_scl(jac, jac, -side, nv);
            mj_addConstraint(m, d, jac, &dist, &margin, 0,
                             1, mjCNSTR_LIMIT_TENDON, i, 0, NULL);
          }
          // set tendon_efcadr
          if (d->nefc > efcadr && d->tendon_efcadr[i] == -1) {
            d->tendon_efcadr[i] = efcadr;
          }
        }
      }
    }
  }

  if (!count_only) {
    mj_freeStack(d);
  }

  return nl;
}


// return 1 if all bodies are welded together or have no dofs, 0 otherwise
static int noDofs(const mjModel* m, const int* body, int n) {
  int wid0 = m->body_treeid[body[0]] >= 0 ? m->body_weldid[body[0]] : 0;
  for (int i=1; i < n; i++) {
    int wid = m->body_treeid[body[i]] >= 0 ? m->body_weldid[body[i]] : 0;
    if (wid != wid0) {
      return 0;
    }
  }
  return 1;
}


// compute Jacobian for contact, return number of DOFs affected
int mj_contactJacobian(const mjModel* m, mjData* d, const mjContact* con, int dim,
                       mjtNum* jacdifp, mjtNum* jacdifr,
                       mjtNum* jac1p, mjtNum* jac2p,
                       mjtNum* jac1r, mjtNum* jac2r, int* chain) {
  // special case: single body on each side
  if ((con->geom[0] >= 0 || (con->vert[0] >= 0 && m->flex_interp[con->flex[0]] == 0 &&
                             !mj_rodHasTwist(m, con->flex[0]))) &&
      (con->geom[1] >= 0 || (con->vert[1] >= 0 && m->flex_interp[con->flex[1]] == 0 &&
                             !mj_rodHasTwist(m, con->flex[1])))) {
    // get bodies
    int bid[2];
    for (int side=0; side < 2; side++) {
      bid[side] = (con->geom[side] >= 0) ?
                  m->geom_bodyid[con->geom[side]] :
                  m->flex_vertbodyid[m->flex_vertadr[con->flex[side]] + con->vert[side]];
    }

    // no relative dofs: return 0
    if (noDofs(m, bid, 2)) {
      return 0;
    }

    // compute Jacobian differences, skipping common DOFs
    if (dim > 3) {
      return mj_jacDifPair(m, d, chain, bid[0], bid[1], con->pos, con->pos,
                           jac1p, jac2p, jacdifp, jac1r, jac2r, jacdifr, mj_isSparse(m), 1);
    } else {
      return mj_jacDifPair(m, d, chain, bid[0], bid[1], con->pos, con->pos,
                           jac1p, jac2p, jacdifp, NULL, NULL, NULL, mj_isSparse(m), 1);
    }
  }

  // general case: flex elements involved
  else {
    // get bodies and weights, negative sign for first side of contact
    int bid[2*mjMAXFLEXINTERPW];
    mjtNum bweight[2*mjMAXFLEXINTERPW];
    int nb = mj_contactBodyWeight(m, d, con, 0, bid, bweight);
    mju_scl(bweight, bweight, -1, nb);
    nb += mj_contactBodyWeight(m, d, con, 1, bid+nb, bweight+nb);

    // no relative dofs: return 0
    if (noDofs(m, bid, nb)) {
      return 0;
    }

    // combine weighted Jacobians
    int NV = mj_jacSum(m, d, chain, nb, bid, bweight, con->pos, jacdifp, jacdifr, dim > 3);
    if (NV > 0) {
      for (int side = 0; side < 2; side++) {
        if (con->flex[side] >= 0 && mj_rodHasTwist(m, con->flex[side])) {
          mj_rodContactJac(m, d, con, side, NV, mj_isSparse(m) ? chain : NULL,
                           jacdifp, dim > 3 ? jacdifr : NULL);
        }
      }
    }
    return NV;
  }
}


// frictionless and frictional contacts
void mj_instantiateContact(const mjModel* m, mjData* d) {
  int ispyramid = mj_isPyramidal(m), issparse = mj_isSparse(m), ncon = d->ncon;
  int dim, NV, nv = m->nv, *chain = NULL;
  mjContact* con;
  mjtNum cpos[6], cmargin[6], *jac, *jacdifp, *jacdifr, *jac1p, *jac2p, *jac1r, *jac2r;

  if (mjDISABLED(mjDSBL_CONTACT) || ncon == 0 || nv == 0) {
    return;
  }

  mj_markStack(d);

  // allocate Jacobian
  jac = mjSTACKALLOC(d, 6*nv, mjtNum);
  jacdifp = mjSTACKALLOC(d, 3*nv, mjtNum);
  jacdifr = mjSTACKALLOC(d, 3*nv, mjtNum);
  jac1p = mjSTACKALLOC(d, 3*nv, mjtNum);
  jac2p = mjSTACKALLOC(d, 3*nv, mjtNum);
  jac1r = mjSTACKALLOC(d, 3*nv, mjtNum);
  jac2r = mjSTACKALLOC(d, 3*nv, mjtNum);
  if (issparse) {
    chain = mjSTACKALLOC(d, nv, int);
  }

  // find contacts to be included
  for (int i=0; i < ncon; i++) {
    if (d->contact[i].exclude) {
      continue;
    }

    // get contact info, save efc_address
    con = d->contact + i;
    dim = con->dim;
    con->efc_address = d->nefc;
    NV = mj_contactJacobian(m, d, con, dim, jacdifp, jacdifr,
                            jac1p, jac2p, jac1r, jac2r, chain);

    // skip contact if no DOFs affected
    if (NV == 0) {
      con->efc_address = -1;
      con->exclude = 3;
      continue;
    }

    // rotate Jacobian differences to contact frame
    mju_mulMatMat(jac, con->frame, jacdifp, dim > 1 ? 3 : 1, 3, NV);
    if (dim > 3) {
      mju_mulMatMat(jac + 3*NV, con->frame, jacdifr, dim-3, 3, NV);
    }

    // make frictionless contact
    if (dim == 1) {
      // add constraint
      mj_addConstraint(m, d, jac, &(con->dist), &(con->includemargin), 0,
                       1, mjCNSTR_CONTACT_FRICTIONLESS, i,
                       issparse ? NV : 0,
                       issparse ? chain : NULL);
    }

    // make pyramidal friction cone
    else if (ispyramid) {
      // pos = dist
      cpos[0] = cpos[1] = con->dist;
      cmargin[0] = cmargin[1] = con->includemargin;

      // one pair per friction dimension
      for (int k=1; k < con->dim; k++) {
        // Jacobian for pair of opposing pyramid edges
        mju_addScl(jacdifp, jac, jac + k*NV, con->friction[k-1], NV);
        mju_addScl(jacdifp + NV, jac, jac + k*NV, -con->friction[k-1], NV);

        // add constraint
        mj_addConstraint(m, d, jacdifp, cpos, cmargin, 0,
                         2, mjCNSTR_CONTACT_PYRAMIDAL, i,
                         issparse ? NV : 0,
                         issparse ? chain : NULL);
      }
    }

    // make elliptic friction cone
    else {
      // normal pos = dist, all others 0
      mju_zero(cpos, con->dim);
      mju_zero(cmargin, con->dim);
      cpos[0] = con->dist;
      cmargin[0] = con->includemargin;

      // add constraint
      mj_addConstraint(m, d, jac, cpos, cmargin, 0,
                       con->dim, mjCNSTR_CONTACT_ELLIPTIC, i,
                       issparse ? NV : 0,
                       issparse ? chain : NULL);
    }
  }

  mj_freeStack(d);
}


//------------------------ compute constraint parameters -------------------------------------------

// compute diagApprox
void mj_diagApprox(const mjModel* m, mjData* d) {
  int id, dim, b1, b2, f, weldcnt = 0;
  int nefc = d->nefc;
  mjtNum tran, rot, fri, *dA = d->efc_diagA;
  mjContact* con = NULL;

  // loop over all constraints, compute approximate inverse inertia
  for (int i=0; i < nefc; i++) {
    // get constraint id
    id = d->efc_id[i];

    // process according to constraint type
    switch ((mjtConstraint) d->efc_type[i]) {
    case mjCNSTR_EQUALITY:
      // process according to equality-constraint type
      switch (m->eq_type[id]) {
      case mjEQ_CONNECT:
        b1 = m->eq_obj1id[id];
        b2 = m->eq_obj2id[id];

        // get body ids if using site semantics
        if (m->eq_objtype[id] == mjOBJ_SITE) {
          b1 = m->site_bodyid[b1];
          b2 = m->site_bodyid[b2];
        }

        // body translation
        dA[i] = m->body_invweight0[2*b1] + m->body_invweight0[2*b2];
        break;

      case mjEQ_WELD:  // distinguish translation and rotation inertia
        b1 = m->eq_obj1id[id];
        b2 = m->eq_obj2id[id];

        // get body ids if using site semantics
        if (m->eq_objtype[id] == mjOBJ_SITE) {
          b1 = m->site_bodyid[b1];
          b2 = m->site_bodyid[b2];
        }

        // body translation or rotation depending on weldcnt
        dA[i] = m->body_invweight0[2*b1 + (weldcnt > 2)] +
                m->body_invweight0[2*b2 + (weldcnt > 2)];
        weldcnt = (weldcnt + 1) % 6;
        break;

      case mjEQ_JOINT:
      case mjEQ_TENDON:
        // object 1 contribution
        dA[i] = (m->eq_type[id] == mjEQ_JOINT ?
                 m->dof_invweight0[m->jnt_dofadr[m->eq_obj1id[id]]] :
                 m->tendon_invweight0[m->eq_obj1id[id]]);

        // add object 2 contribution if present
        if (m->eq_obj2id[id] >= 0)
          dA[i] += (m->eq_type[id] == mjEQ_JOINT ?
                    m->dof_invweight0[m->jnt_dofadr[m->eq_obj2id[id]]] :
                    m->tendon_invweight0[m->eq_obj2id[id]]);
        break;

      case mjEQ_FLEX:
        // process all non-rigid edges for this flex
        f = m->eq_obj1id[id];
        int flex_edgeadr = m->flex_edgeadr[f];
        int flex_edgenum = m->flex_edgenum[f];
        for (int e=flex_edgeadr; e<flex_edgeadr+flex_edgenum; e++) {
          if (!m->flexedge_rigid[e]) {
            dA[i++] = m->flexedge_invweight0[e];
          }
        }

        // adjust constraint counter
        i--;
        break;

      case mjEQ_FLEXSTRAIN: {
        // strain constraints: use avg inv weight of element's nodes
        int flex_id = m->eq_obj1id[id];
        int nstart = m->flex_nodeadr[flex_id];
        int interp = m->flex_interp[flex_id];
        int order = interp < 0 ? -interp : interp;
        int is_shell = (interp < 0);

        int cx = m->flex_cellnum[3*flex_id+0];
        int cy = m->flex_cellnum[3*flex_id+1];
        int cz = m->flex_cellnum[3*flex_id+2];

        // nodes per element
        int npe;
        int elem_idx;
        if (is_shell) {
          npe = (order+1) * (order+1);
          elem_idx = (int)m->eq_data[mjNEQDATA*id + 0];
        } else {
          npe = (order+1) * (order+1) * (order+1);
          int ci_cell = (int)m->eq_data[mjNEQDATA*id + 0];
          int cj_cell = (int)m->eq_data[mjNEQDATA*id + 1];
          int ck_cell = (int)m->eq_data[mjNEQDATA*id + 2];
          elem_idx = ci_cell * cy * cz + cj_cell * cz + ck_cell;
        }

        // read neig from flex_stiffness
        int ndof_elem = 3 * npe;
        const mjtNum* k_elem = m->flex_stiffness + m->flex_stiffnessadr[flex_id]
                             + elem_idx * ndof_elem * ndof_elem;
        int nconstraint = (int)k_elem[0];

        // get element node indices
        int gindices[125];
        if (is_shell) {
          mju_flexGatherFaceState(order, cx, cy, cz, elem_idx,
                                  NULL, NULL, NULL, NULL, NULL, NULL, gindices, NULL);
        } else {
          int ci_cell = (int)m->eq_data[mjNEQDATA*id + 0];
          int cj_cell = (int)m->eq_data[mjNEQDATA*id + 1];
          int ck_cell = (int)m->eq_data[mjNEQDATA*id + 2];
          mju_flexGatherCellState(order, cy, cz, ci_cell, cj_cell, ck_cell,
                                  NULL, NULL, NULL, NULL, NULL, NULL, gindices, NULL);
        }

        mjtNum avg_invweight = 0;
        for (int n = 0; n < npe; n++) {
          int bodyid = m->flex_nodebodyid[nstart + gindices[n]];
          avg_invweight += m->body_invweight0[2*bodyid];
        }
        avg_invweight /= npe;
        for (int c = 0; c < nconstraint; c++) {
          dA[i++] = avg_invweight;
        }

        // adjust constraint counter
        i--;
        break;
      }

      default:
        mjERROR("unknown constraint type %d", d->efc_type[i]);    // SHOULD NOT OCCUR
      }
      break;

    case mjCNSTR_FRICTION_DOF:
      dA[i] = m->dof_invweight0[id];
      break;

    case mjCNSTR_LIMIT_JOINT:
      dA[i] = m->dof_invweight0[m->jnt_dofadr[id]];
      break;

    case mjCNSTR_FRICTION_TENDON:
    case mjCNSTR_LIMIT_TENDON:
      dA[i] = m->tendon_invweight0[id];
      break;

    case mjCNSTR_CONTACT_FRICTIONLESS:
    case mjCNSTR_CONTACT_PYRAMIDAL:
    case mjCNSTR_CONTACT_ELLIPTIC:
      // get contact info
      con = d->contact + id;
      dim = con->dim;

      // add the average translation and rotation components from both sides
      tran = rot = 0;
      for (int side=0; side < 2; side++) {
        // get bodies and weights
        int bid[mjMAXFLEXINTERPW];
        mjtNum bweight[mjMAXFLEXINTERPW];
        int nb = mj_contactBodyWeight(m, d, con, side, bid, bweight);

        // add weighted average over bodies
        for (int k=0; k < nb; k++) {
          tran += m->body_invweight0[2*bid[k]] * bweight[k];
          rot += m->body_invweight0[2*bid[k]+1] * bweight[k];
        }
      }

      // set frictionless
      if (d->efc_type[i] == mjCNSTR_CONTACT_FRICTIONLESS) {
        dA[i] = tran;
      }

      // set elliptical
      else if (d->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {
        for (int j=0; j < dim; j++) {
          dA[i+j] = (j < 3 ? tran : rot);
        }

        // processed dim elements in one i-loop iteration; advance counter
        i += (dim-1);
      }

      // set pyramidal
      else {
        for (int j=0; j < dim-1; j++) {
          fri = con->friction[j];
          dA[i+2*j] = dA[i+2*j+1] = tran + fri*fri*(j < 2 ? tran : rot);
        }

        // processed 2*dim-2 elements in one i-loop iteration; advance counter
        i += (2*dim-3);
      }
    }
  }
}


// get solref, solimp for specified constraint
static void getsolparam(const mjModel* m, const mjData* d, int i,
                        mjtNum* solref, mjtNum* solreffriction, mjtNum* solimp) {
  // get constraint id
  int id = d->efc_id[i];

  // clear solreffriction (applies only to contacts)
  mju_zero(solreffriction, mjNREF);

  // extract solver parameters from corresponding model element
  switch ((mjtConstraint) d->efc_type[i]) {
  case mjCNSTR_EQUALITY:
    mju_copy(solref, m->eq_solref+mjNREF*id, mjNREF);
    mju_copy(solimp, m->eq_solimp+mjNIMP*id, mjNIMP);
    break;

  case mjCNSTR_LIMIT_JOINT:
    mju_copy(solref, m->jnt_solref+mjNREF*id, mjNREF);
    mju_copy(solimp, m->jnt_solimp+mjNIMP*id, mjNIMP);
    break;

  case mjCNSTR_FRICTION_DOF:
    mju_copy(solref, m->dof_solref+mjNREF*id, mjNREF);
    mju_copy(solimp, m->dof_solimp+mjNIMP*id, mjNIMP);
    break;

  case mjCNSTR_LIMIT_TENDON:
    mju_copy(solref, m->tendon_solref_lim+mjNREF*id, mjNREF);
    mju_copy(solimp, m->tendon_solimp_lim+mjNIMP*id, mjNIMP);
    break;

  case mjCNSTR_FRICTION_TENDON:
    mju_copy(solref, m->tendon_solref_fri+mjNREF*id, mjNREF);
    mju_copy(solimp, m->tendon_solimp_fri+mjNIMP*id, mjNIMP);
    break;

  case mjCNSTR_CONTACT_FRICTIONLESS:
  case mjCNSTR_CONTACT_PYRAMIDAL:
  case mjCNSTR_CONTACT_ELLIPTIC:
    mju_copy(solref, d->contact[id].solref, mjNREF);
    mju_copy(solreffriction, d->contact[id].solreffriction, mjNREF);
    mju_copy(solimp, d->contact[id].solimp, mjNIMP);
  }

  // check reference format: standard or direct, cannot be mixed
  if ((solref[0] > 0) ^ (solref[1] > 0)) {
    mju_warning("mixed solref format, replacing with default");
    mj_defaultSolRefImp(solref, NULL);
  }

  // integrator safety: impose ref[0]>=2*timestep for standard format. Not applied under
  // the discrete integrator: implicitly treated rows (mj_makeImpedance) are stable at
  // any timeconst, and timeconst -> 0 is their rigid limit
  int metric = mj_isMetric(m);
  if (!mjDISABLED(mjDSBL_REFSAFE) && solref[0] > 0 && !metric) {
    solref[0] = mju_max(solref[0], 2*m->opt.timestep);
  }

  // check reference format: standard or direct, cannot be mixed
  if ((solreffriction[0] > 0) ^ (solreffriction[1] > 0)) {
    mju_warning("solreffriction values should have the same sign, replacing with default");
    mju_zero(solreffriction, mjNREF);  // default solreffriction is (0, 0)
  }

  // integrator safety: impose ref[0]>=2*timestep for standard format
  if (!mjDISABLED(mjDSBL_REFSAFE) && solreffriction[0] > 0 && !metric) {
    solreffriction[0] = mju_max(solreffriction[0], 2*m->opt.timestep);
  }

  // enforce constraints on solimp
  solimp[0] = mju_min(mjMAXIMP, mju_max(mjMINIMP, solimp[0]));
  solimp[1] = mju_min(mjMAXIMP, mju_max(mjMINIMP, solimp[1]));
  solimp[2] = mju_max(0, solimp[2]);
  solimp[3] = mju_min(mjMAXIMP, mju_max(mjMINIMP, solimp[3]));
  solimp[4] = mju_max(1, solimp[4]);
}


// get pos and dim for specified constraint
static void getposdim(const mjModel* m, const mjData* d, int i, mjtNum* pos, int* dim) {
  // get id of constraint-related object
  int id = d->efc_id[i];

  // set (dim, pos) for common case
  *dim = 1;
  *pos = d->efc_pos[i];

  // change (dim, distance) for special cases
  switch ((mjtConstraint) d->efc_type[i]) {
  case mjCNSTR_CONTACT_ELLIPTIC:
    *dim = d->contact[id].dim;
    break;

  case mjCNSTR_CONTACT_PYRAMIDAL:
    *dim = 2*(d->contact[id].dim-1);
    break;

  case mjCNSTR_EQUALITY:
    if (m->eq_type[id] == mjEQ_WELD) {
      *dim = 6;
      *pos = mju_norm(d->efc_pos+i, 6);
    } else if (m->eq_type[id] == mjEQ_CONNECT) {
      *dim = 3;
      *pos = mju_norm(d->efc_pos+i, 3);
    }
    break;
  default:
    // already handled
    break;
  }
}


// return a to the power of b, quick return for powers 1 and 2
// solimp[4] == 2 is the default, so these branches are common
static mjtNum power(mjtNum a, mjtNum b) {
  if (b == 1) {
    return a;
  } else if (b == 2) {
    return a*a;
  }
  return mju_pow(a, b);
}


// compute impedance and derivative for one constraint
static void getimpedance(const mjtNum* solimp, mjtNum pos, mjtNum margin,
                         mjtNum* imp, mjtNum* impP) {
  // flat function
  if (solimp[0] == solimp[1] || solimp[2] <= mjMINVAL) {
    *imp = 0.5*(solimp[0] + solimp[1]);
    *impP = 0;
    return;
  }

  // x = abs((pos-margin) / width)
  mjtNum x = (pos-margin) / solimp[2];
  mjtNum sgn = 1;
  if (x < 0) {
    x = -x;
    sgn = -1;
  }

  // fully saturated
  if (x >= 1 || x <= 0) {
    *imp = (x >= 1 ? solimp[1] : solimp[0]);
    *impP = 0;
    return;
  }

  // linear
  mjtNum y, yP;
  if (solimp[4] == 1) {
    y = x;
    yP = 1;
  }

  // y(x) = a*x^p if x<=midpoint
  else if (x <= solimp[3]) {
    mjtNum a = 1/power(solimp[3], solimp[4]-1);
    y = a*power(x, solimp[4]);
    yP = solimp[4] * a*power(x, solimp[4]-1);
  }

  // y(x) = 1-b*(1-x)^p if x>midpoint
  else {
    mjtNum b = 1/power(1-solimp[3], solimp[4]-1);
    y = 1-b*power(1-x, solimp[4]);
    yP = solimp[4] * b*power(1-x, solimp[4]-1);
  }

  // scale
  *imp = solimp[0] + y*(solimp[1]-solimp[0]);
  *impP = yP * sgn * (solimp[1]-solimp[0]) / solimp[2];
}


// implicit-row factor f = 1 + h*B + h^2*K*I of a constraint row (kbip = its efc_KBIP)
static inline mjtNum implicitFactor(const mjtNum* kbip, mjtNum h) {
  return 1 + h*kbip[1] + h*h*kbip[0]*kbip[2];
}


// compute efc_R, efc_D, efc_KBIP, adjust efc_diagA
void mj_makeImpedance(const mjModel* m, mjData* d) {
  int dim, nefc = d->nefc;
  mjtNum *R = d->efc_R, *KBIP = d->efc_KBIP;
  mjtNum pos, imp, impP, Rpy, solref[mjNREF], solreffriction[mjNREF], solimp[mjNIMP];
  int metric = mj_isMetric(m);
  mjtNum h = m->opt.timestep;

  // set efc_R, efc_KBIP
  for (int i=0; i < nefc; i++) {
    // get solref and solimp
    getsolparam(m, d, i, solref, solreffriction, solimp);

    // get pos and dim
    getposdim(m, d, i, &pos, &dim);

    // get imp and impP
    getimpedance(solimp, pos, d->efc_margin[i], &imp, &impP);

    // set R and KBIP for all constraint dimensions
    for (int j=0; j < dim; j++) {
      // R = (1-imp)/imp * diagApprox
      R[i+j] = mju_max(mjMINVAL, (1-imp)*d->efc_diagA[i+j]/imp);

      // constraint type
      int tp = d->efc_type[i+j];

      // elliptic contacts use solreffriction in non-normal directions, if non-zero
      int elliptic_friction = (tp == mjCNSTR_CONTACT_ELLIPTIC) && (j > 0);
      mjtNum* ref = elliptic_friction && (solreffriction[0] || solreffriction[1]) ?
                    solreffriction : solref;

      // friction: K = 0
      if (tp == mjCNSTR_FRICTION_DOF || tp == mjCNSTR_FRICTION_TENDON || elliptic_friction) {
        KBIP[4*(i+j)] = 0;
      }

      // standard: K = 1 / (d_width^2 * timeconst^2 * dampratio^2)
      else if (ref[0] > 0)
        KBIP[4*(i+j)] = 1 / mju_max(mjMINVAL, solimp[1]*solimp[1] * ref[0]*ref[0] * ref[1]*ref[1]);

      // direct: K = -solref[0] / d_width^2
      else {
        KBIP[4*(i+j)] = -ref[0] / mju_max(mjMINVAL, solimp[1]*solimp[1]);
      }

      // standard: B = 2 / (d_width*timeconst)
      if (ref[1] > 0) {
        KBIP[4*(i+j)+1] = 2 / mju_max(mjMINVAL, solimp[1]*ref[0]);
      }

      // direct: B = -solref[1] / d_width
      else {
        KBIP[4*(i+j)+1] = -ref[1] / mju_max(mjMINVAL, solimp[1]);
      }

      // I = imp, P = imp'
      KBIP[4*(i+j)+2] = imp;
      KBIP[4*(i+j)+3] = impP;

      // discrete: implicit row factor R <- R/f (matching reference in mj_referenceConstraint).
      // Floor at mjMAXIMP ceiling to keep weights well-conditioned as timeconst -> 0
      if (metric) {
        mjtNum* kbip = KBIP + 4*(i+j);

        // refsafe: a contact or limit row whose spring the step cannot resolve
        // (h^2*K*I > 1) rebounds on impact with restitution (h^2*K*I - 1)/f. Replace it
        // by the resolved row: timeconst shortened to the stiffest zero-restitution
        // value, damping ratio kept (K /= excess, B /= sqrt(excess)), damping bounded by
        // the impedance ceiling so that the statics stay exact
        if (!mjDISABLED(mjDSBL_REFSAFE) && ref[0] > 0 && kbip[0] > 0 &&
            (tp == mjCNSTR_LIMIT_JOINT          ||
             tp == mjCNSTR_LIMIT_TENDON         ||
             tp == mjCNSTR_CONTACT_FRICTIONLESS ||
             tp == mjCNSTR_CONTACT_PYRAMIDAL    ||
             tp == mjCNSTR_CONTACT_ELLIPTIC)) {
          mjtNum excess = (h*h)*kbip[0]*kbip[2];
          if (excess > 1) {
            mjtNum fmax = mjMAXIMP*(1-kbip[2]) / mju_max(mjMINVAL, kbip[2]*(1-mjMAXIMP));
            kbip[0] = 1 / (h*h*kbip[2]);
            kbip[1] = mju_min(kbip[1]/mju_sqrt(excess), mju_max(0, fmax-2)/h);
          }
        }

        mjtNum f = implicitFactor(kbip, h);
        mjtNum Rmin = mju_max(mjMINVAL, (1-mjMAXIMP)*d->efc_diagA[i+j]/mjMAXIMP);
        R[i+j] = mju_max(R[i+j]/f, Rmin);
      }
    }

    // skip the rest of this constraint
    i += (dim-1);
  }

  // frictional contacts: adjust R in friction dimensions, set contact master mu
  for (int i=d->ne+d->nf; i < nefc; i++) {
    if (d->efc_type[i] == mjCNSTR_CONTACT_PYRAMIDAL ||
        d->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {
      // extract id, dim, mu
      int id = d->efc_id[i];
      dim = d->contact[id].dim;
      mjtNum* friction = d->contact[id].friction;

      // set R[1] = R[0]/impratio
      R[i+1] = R[i]/mju_max(mjMINVAL, m->opt.impratio);

      // set mu of regularized cone = mu[1]*sqrt(R[1]/R[0])
      d->contact[id].mu = friction[0] * mju_sqrt(R[i+1]/R[i]);

      // elliptic
      if (d->efc_type[i] == mjCNSTR_CONTACT_ELLIPTIC) {
        // set remaining R's such that R[j]*mu[j]^2 = R[1]*mu[1]^2
        for (int j=1; j < dim-1; j++) {
          R[i+j+1] = R[i+1]*friction[0]*friction[0]/(friction[j]*friction[j]);
        }

        // skip the rest of this contact
        i += (dim-1);
      }

      // pyramidal: common R matching friction impedance of elliptic model
      else {
        // D0_el = 2*(dim-1)*D_py : normal match
        // D0_el = 2*mu^2*D_py    : friction match
        Rpy = 2*d->contact[id].mu*d->contact[id].mu*R[i];

        // assign Rpy to all pyramidal R
        for (int j=0; j < 2*(dim-1); j++) {
          R[i+j] = Rpy;
        }

        // skip the rest of this contact
        i += 2*(dim-1) - 1;
      }
    }
  }

  // set D = 1 / R
  for (int i=0; i < nefc; i++) {
    d->efc_D[i] = 1 / R[i];
  }

  // adjust diagA so that R = (1-imp)/imp * diagA
  for (int i=0; i < nefc; i++) {
    d->efc_diagA[i] = R[i] * KBIP[4*i+2] / (1-KBIP[4*i+2]);
  }
}


// forward declarations (defined in the projection section below)
static void mj_makeYSymbolic(const mjModel* m, mjData* d);
static void mj_makeYNumeric(const mjModel* m, mjData* d, int flg_diagexact);
static void mj_makeARSymbolic(const mjModel* m, mjData* d);
static void mj_makeARNumeric(const mjModel* m, mjData* d);

// compute constraint regularization in the current solve metric: efc_diagA (approximate,
// or exact under diagexact), then R, D, KBIP. Under the discrete integrator this runs at
// the actuation stage, where the effective metric is final; islands are discovered at the
// position stage, so their R/D copies are refreshed here. flg_AR: assemble the dual's AR
// (forward path); inverse dynamics never consumes it and passes 0
void mj_regularizeConstraint(const mjModel* m, mjData* d, int flg_AR) {
  int nefc = d->nefc, nv = m->nv;
  if (!nefc) {
    return;
  }

  // whitened Jacobian against the metric factor: exact diagonal, and/or the dual's AR.
  // Numeric phase only -- the pattern and its arena allocation are position-stage facts,
  // laid down by mj_projectConstraint; re-running this stage allocates nothing
  int isDual = mj_isDual(m);
  if (mjENABLED(mjENBL_DIAGEXACT) || (isDual && flg_AR)) {
    mj_makeYNumeric(m, d, mjENABLED(mjENBL_DIAGEXACT));
  }

  // approximate diagonal, corrected per row by the diagonal-class metric ratio
  if (!mjENABLED(mjENBL_DIAGEXACT)) {
    mj_diagApprox(m, d);
    if (mj_isMetric(m) && d->efm_sdiag) {
      int is_sparse = mj_isSparse(m);
      for (int r=0; r < nefc; r++) {
        mjtNum num = 0, den = 0;
        int rownnz = is_sparse ? d->efc_J_rownnz[r] : nv;
        int rowadr = is_sparse ? d->efc_J_rowadr[r] : r*nv;
        for (int a=0; a < rownnz; a++) {
          int c = is_sparse ? d->efc_J_colind[rowadr + a] : a;
          mjtNum J = d->efc_J[rowadr + a];
          if (!J) {
            continue;
          }
          mjtNum Mdiag = d->M[m->M_rowadr[c] + m->M_rownnz[c] - 1];
          num += J*J / mju_max(mjMINVAL, Mdiag + d->efm_sdiag[c]);
          den += J*J / mju_max(mjMINVAL, Mdiag);
        }
        if (den > mjMINVAL) {
          d->efc_diagA[r] *= num / den;
        }
      }
    }
  }

  // R, D, KBIP from diagA
  mj_makeImpedance(m, d);

  // assemble AR = Y*Y' + diag(R) for the dual solvers
  if (isDual && flg_AR) {
    mj_makeARNumeric(m, d);
  }

  // refresh island copies of R and D
  if (d->nisland) {
    mju_gather(d->iefc_D, d->efc_D, d->map_iefc2efc, nefc);
    mju_gather(d->iefc_R, d->efc_R, d->map_iefc2efc, nefc);
  }
}


//------------------------------------- constraint counting ----------------------------------------

// count the non-zero columns of the Jacobian returned by mj_jacSum
static int mj_jacSumCount(const mjModel* m, mjData* d, int* chain,
                          int n, const int* body) {
  int nv = m->nv, NV;

  mj_markStack(d);
  int* bodychain = mjSTACKALLOC(d, nv, int);
  int* tempchain = mjSTACKALLOC(d, nv, int);

  // set first
  NV = mj_bodyChain(m, body[0], chain);

  // accumulate remaining
  for (int i=1; i < n; i++) {
    // get body chain
    int bodyNV = mj_bodyChain(m, body[i], bodychain);
    if (!bodyNV) {
      continue;
    }

    // accumulate chains
    NV = mju_addChains(tempchain, nv, NV, bodyNV, chain, bodychain);
    if (NV) {
      mju_copyInt(chain, tempchain, NV);
    }
  }

  mj_freeStack(d);
  return NV;
}

// count equality constraints, count Jacobian nonzeros if nnz is not NULL
static int mj_ne(const mjModel* m, mjData* d, int* nnz) {
  int ne = 0, nnze = 0;
  int nv = m->nv, neq = m->neq;
  int id[2], size, NV, NV2, *chain = NULL, *chain2 = NULL;
  int issparse = (nnz != NULL);
  int flex_edgeadr, flex_edgenum;

  // disabled or no equality constraints: return
  if (mjDISABLED(mjDSBL_EQUALITY) || m->nemax == 0) {
    return 0;
  }

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

  mj_markStack(d);

  if (nnz) {
    chain = mjSTACKALLOC(d, nv, int);
    chain2 = mjSTACKALLOC(d, nv, int);
  }

  // pre-allocate buffer for cell body IDs (max npc = 125 for order=2)
  int* cell_bodies = nnz ? mjSTACKALLOC(d, 125, int) : NULL;

  // find active equality constraints
  for (int i=0; i < neq; i++) {
    // skip inactive
    if (!d->eq_active[i]) {
      continue;
    }

    // skip sleeping
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_EQUALITY, i) == mjS_ASLEEP) {
      continue;
    }

    id[0] = m->eq_obj1id[i];
    id[1] = m->eq_obj2id[i];
    size = 0;
    NV = 0;
    NV2 = 0;

    // process according to type
    switch ((mjtEq) m->eq_type[i]) {
    case mjEQ_CONNECT:
      size = 3;
      if (!nnz) {
        break;
      }

      // get body ids if using site semantics
      if (m->eq_objtype[i] == mjOBJ_SITE) {
        id[0] = m->site_bodyid[id[0]];
        id[1] = m->site_bodyid[id[1]];
      }

      NV = mj_jacDifPair(m, NULL, chain, id[1], id[0], NULL, NULL,
                         NULL, NULL, NULL, NULL, NULL, NULL, issparse,
                         /*flg_skipcommon=*/0);
      break;

    case mjEQ_WELD:
      size = 6;
      if (!nnz) {
        break;
      }

      // get body ids if using site semantics
      if (m->eq_objtype[i] == mjOBJ_SITE) {
        id[0] = m->site_bodyid[id[0]];
        id[1] = m->site_bodyid[id[1]];
      }

      NV = mj_jacDifPair(m, NULL, chain, id[1], id[0], NULL, NULL,
                         NULL, NULL, NULL, NULL, NULL, NULL, issparse,
                         /*flg_skipcommon=*/0);
      break;

    case mjEQ_JOINT:
    case mjEQ_TENDON:
      size = 1;
      if (!nnz) {
        break;
      }

      for (int j=0; j < 1+(id[1] >= 0); j++) {
        if (m->eq_type[i] == mjEQ_JOINT) {
          if (!j) {
            NV = 1;
            chain[0] = m->jnt_dofadr[id[j]];
          } else {
            NV2 = 1;
            chain2[0] = m->jnt_dofadr[id[j]];
          }
        } else {
          if (!j) {
            NV = m->ten_J_rownnz[id[j]];
            mju_copyInt(chain, m->ten_J_colind+m->ten_J_rowadr[id[j]], NV);
          } else {
            NV2 = m->ten_J_rownnz[id[j]];
            mju_copyInt(chain2, m->ten_J_colind+m->ten_J_rowadr[id[j]], NV2);
          }
        }
      }

      if (id[1] >= 0) {
        NV = mju_combineSparseCount(NV, NV2, chain, chain2);
      }
      break;

    case mjEQ_FLEX:
      flex_edgeadr = m->flex_edgeadr[id[0]];
      flex_edgenum = m->flex_edgenum[id[0]];

      // init with all edges, subtract rigid later
      size = flex_edgenum;

      // process edges of this flex
      for (int e=flex_edgeadr; e < flex_edgeadr+flex_edgenum; e++) {
        // rigid: reduce size and skip
        if (m->flexedge_rigid[e]) {
          size--;
          continue;
        }

        // accumulate NV if needed
        if (nnz) {
          int b1 = m->flex_vertbodyid[m->flex_vertadr[id[0]] + m->flex_edge[2*e]];
          int b2 = m->flex_vertbodyid[m->flex_vertadr[id[0]] + m->flex_edge[2*e+1]];
          NV += mj_jacDifPair(m, NULL, chain, b1, b2, NULL, NULL,
                              NULL, NULL, NULL, NULL, NULL, NULL, issparse,
                              /*flg_skipcommon=*/0);
        }
      }
      break;

    case mjEQ_FLEXSTRAIN: {
      // per-element strain constraints: each equality is one cell or face
      int f = id[0];
      int interp = m->flex_interp[f];
      int order = interp < 0 ? -interp : interp;
      int is_shell = (interp < 0);
      if (!order || !m->flex_nodenum[f]) {
        break;
      }

      int cx = m->flex_cellnum[3*f+0];
      int cy = m->flex_cellnum[3*f+1];
      int cz = m->flex_cellnum[3*f+2];

      int npe;
      int elem_idx;
      if (is_shell) {
        npe = (order+1) * (order+1);
        elem_idx = (int)m->eq_data[mjNEQDATA*i + 0];
      } else {
        npe = (order+1) * (order+1) * (order+1);
        int ci_cell = (int)m->eq_data[mjNEQDATA*i + 0];
        int cj_cell = (int)m->eq_data[mjNEQDATA*i + 1];
        int ck_cell = (int)m->eq_data[mjNEQDATA*i + 2];
        elem_idx = ci_cell * cy * cz + cj_cell * cz + ck_cell;
      }

      // read eigenmode count from flex_stiffness
      int ndof_elem = 3 * npe;
      size = 0;
      if (m->flex_stiffnessadr[f] >= 0) {
        const mjtNum* k_elem = m->flex_stiffness + m->flex_stiffnessadr[f]
                             + elem_idx * ndof_elem * ndof_elem;
        size = (int)k_elem[0];  // neig stored as first element
      }

      if (nnz) {
        // get element node body IDs
        int gindices[125];
        if (is_shell) {
          mju_flexGatherFaceState(order, cx, cy, cz, elem_idx,
                                  NULL, NULL, NULL, NULL, NULL, NULL, gindices, NULL);
        } else {
          int ci_cell = (int)m->eq_data[mjNEQDATA*i + 0];
          int cj_cell = (int)m->eq_data[mjNEQDATA*i + 1];
          int ck_cell = (int)m->eq_data[mjNEQDATA*i + 2];
          mju_flexGatherCellState(order, cy, cz, ci_cell, cj_cell, ck_cell,
                                  NULL, NULL, NULL, NULL, NULL, NULL, gindices, NULL);
        }
        int nstart = m->flex_nodeadr[f];
        for (int n = 0; n < npe; n++) {
          cell_bodies[n] = m->flex_nodebodyid[nstart + gindices[n]];
        }
        NV = mj_jacSumCount(m, d, chain, npe, cell_bodies);
        NV = size * NV;
      }
      break;
    }

    default:
      // might occur in case of the now-removed distance equality constraint
      mjERROR("unknown constraint type %d", m->eq_type[i]);    // SHOULD NOT OCCUR
    }

    // accumulate counts; flex NV already accumulated
    ne += mj_addConstraintCount(m, size, NV);
    if (m->eq_type[i] == mjEQ_FLEX || m->eq_type[i] == mjEQ_FLEXSTRAIN) {
      nnze += NV;
    } else {
      nnze += size*NV;
    }
  }

  if (nnz) {
    *nnz += nnze;
  }

  mj_freeStack(d);
  return ne;
}


// count contact constraints, count Jacobian nonzeros if nnz is not NULL
static int mj_nc(const mjModel* m, mjData* d, int* nnz) {
  int nnzc = 0, nc = 0;
  int ispyramid = mj_isPyramidal(m), ncon = d->ncon;

  if (mjDISABLED(mjDSBL_CONTACT) || !ncon) {
    return 0;
  }

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

  mj_markStack(d);
  int *chain = mjSTACKALLOC(d, m->nv, int);

  for (int i=0; i < ncon; i++) {
    mjContact* con = d->contact + i;

    // Passive path: flex-flex (including self-collision) and flex-vs-static-geometry, where every
    // dof is a metric-carried flex vertex so the Hessian is assembled in full. Flex-vs-moving-body
    // stays on the constraint solver. Passive if either flex asks for it.
    {
      int f0 = con->flex[0], f1 = con->flex[1];
      int wants = (f0 > -1 && mj_effFlexContactPossible(m, f0)) ||
                  (f1 > -1 && mj_effFlexContactPossible(m, f1));
      int ok = (f0 > -1 && f1 > -1);   // flex-flex, or a flex with itself
      for (int s = 0; s < 2 && !ok; s++) {
        if (con->flex[s] < 0 && con->geom[s] > -1) {
          ok = (m->body_weldid[m->geom_bodyid[con->geom[s]]] == 0);   // welded to the world
        }
      }
      if (wants && ok) {
        con->efc_address = -1;
        con->exclude = 4;
      }
    }

    // skip if excluded
    if (con->exclude) {
      continue;
    }

    // check for contact with sleeping tree; SHOULD NOT OCCUR
    if (sleep_filter) {
      int g1 = con->geom[0];
      int g2 = con->geom[1];
      if (g1 >= 0 && g2 >= 0) {
        int b1 = m->body_weldid[m->geom_bodyid[g1]];
        int b2 = m->body_weldid[m->geom_bodyid[g2]];
        int asleep1 = d->body_awake[b1] == mjS_ASLEEP;
        int asleep2 = d->body_awake[b2] == mjS_ASLEEP;
        if (asleep1 || asleep2) {
          mjERROR("contact %d involves sleeping geom %d", i, asleep1 ? g1 : g2);
        }
      }

      // check flex contact sides
      for (int side = 0; side < 2; side++) {
        if (con->geom[side] >= 0) continue;
        int b = mj_flexBody(m, con, side);
        if (d->body_awake[m->body_weldid[b]] == mjS_ASLEEP) {
          mjERROR("contact %d involves sleeping flex %d", i, con->flex[side]);
        }
      }
    }

    // compute NV only if nnz requested
    int NV = 0;
    if (nnz) {
      // single body on each side (geom-geom or flex vert-vert): skip common dofs
      if ((con->geom[0] >= 0 || (con->vert[0] >= 0 && m->flex_interp[con->flex[0]] == 0 &&
                                 !mj_rodHasTwist(m, con->flex[0]))) &&
          (con->geom[1] >= 0 || (con->vert[1] >= 0 && m->flex_interp[con->flex[1]] == 0 &&
                                 !mj_rodHasTwist(m, con->flex[1])))) {
        // get bodies
        int bid[2];
        for (int side=0; side < 2; side++) {
          bid[side] = (con->geom[side] >= 0) ?
                      m->geom_bodyid[con->geom[side]] :
                      m->flex_vertbodyid[m->flex_vertadr[con->flex[side]] + con->vert[side]];
        }
        if (!noDofs(m, bid, 2)) {
          NV = mj_jacDifPair(m, NULL, chain, bid[0], bid[1], NULL, NULL,
                             NULL, NULL, NULL, NULL, NULL, NULL, mj_isSparse(m), 1);
        }
      }

      // general case: flex elements involved
      else {
        // get bodies
        int bid[2*mjMAXFLEXINTERPW];
        int nb = mj_contactBodyWeight(m, d, con, 0, bid, NULL);
        nb += mj_contactBodyWeight(m, d, con, 1, bid+nb, NULL);

        // count non-zeros in merged chain
        if (!noDofs(m, bid, nb)) {
          NV = mj_jacSumCount(m, d, chain, nb, bid);
        }
      }
      if (!NV) {
        continue;
      }
    }

    // count according to friction type
    int dim = con->dim;
    if (dim == 1) {
      nc++;
      nnzc += NV;
    } else if (ispyramid) {
      nc += 2*(dim-1);
      nnzc += 2*(dim-1)*NV;
    } else {
      nc += dim;
      nnzc += dim*NV;
    }
  }

  if (nnz) {
    *nnz += nnzc;
  }

  mj_freeStack(d);
  return nc;
}


// pre-count Y_rownnz, Y_rowadr, return total nonzeros nY
//   Y has the sparsity of J * inv(L'), where L is the Cholesky factor of M
static int computeY_precount(int* Y_rownnz, int* Y_rowadr, int nefc, int nv,
                             const int* J_rownnz, const int* J_rowadr, const int* J_colind,
                             const int* M_rownnz, const int* M_rowadr, const int* M_colind,
                             int* marker) {
  mju_fillInt(marker, -1, nv);

  Y_rowadr[0] = 0;
  for (int r=0; r < nefc; r++) {
    int nnz = 0;  // nonzeros in row r of Y

    // traverse row r of J in reverse, count unique nonzeros
    int start = J_rowadr[r];
    int end = start + J_rownnz[r];
    for (int i=end-1; i >= start; i--) {
      int j = J_colind[i];

      // if dof j is marked, it was already counted by a child dof: skip it
      if (marker[j] == r) {
        continue;
      }

      // traverse row j of M, marking new unique nonzeros
      int nnzM = M_rownnz[j];
      int adrM = M_rowadr[j];
      for (int k=0; k < nnzM; k++) {
        int c = M_colind[adrM + k];
        if (marker[c] != r) {
          marker[c] = r;
          nnz++;
        }
      }
    }

    // update rownnz and rowadr
    Y_rownnz[r] = nnz;
    if (r < nefc - 1) {
      Y_rowadr[r+1] = Y_rowadr[r] + nnz;
    }
  }

  // total non-zeros in Y
  return Y_rowadr[nefc-1] + Y_rownnz[nefc-1];
}


// fill Y column indices and values from J, chaining up the pattern of M counted by
// computeY_precount; with Y == NULL, fill only the column indices (pattern-only)
static void computeY_fill(mjtNum* Y, int* Y_colind,
                          const int* Y_rownnz, const int* Y_rowadr, int nefc,
                          const mjtNum* J, const int* J_rownnz, const int* J_rowadr,
                          const int* J_colind,
                          const int* M_rownnz, const int* M_rowadr, const int* M_colind) {
  for (int r=0; r < nefc; r++) {
    // init row
    int end = Y_rowadr[r] + Y_rownnz[r];
    int adrJ = J_rowadr[r];
    int remainJ = J_rownnz[r];
    int nnzY = 0;

    // complete chain in reverse
    while (1) {
      // get previous dof in src and dst: the parent of the last added dof is the entry before
      // the diagonal in its row of M, none for simple dofs (diagonal rows)
      int prev_src = (remainJ > 0 ? J_colind[adrJ + remainJ - 1] : -1);
      int prev_dst = -1;
      if (nnzY > 0) {
        int j = Y_colind[end - nnzY];
        if (M_rownnz[j] > 1) {
          prev_dst = M_colind[M_rowadr[j] + M_rownnz[j] - 2];
        }
      }

      // both finished: break
      if (prev_src < 0 && prev_dst < 0) {
        break;
      }

      // add src
      else if (prev_src >= prev_dst) {
        nnzY++;
        remainJ--;
        Y_colind[end - nnzY] = prev_src;
        if (Y) {
          Y[end - nnzY] = J[adrJ + remainJ];
        }
      }

      // add dst
      else {
        nnzY++;
        Y_colind[end - nnzY] = prev_dst;
        if (Y) {
          Y[end - nnzY] = 0;
        }
      }
    }

    // compare with Y_rownnz: SHOULD NOT OCCUR
    if (nnzY != Y_rownnz[r]) {
      mjERROR("pre and post-count of Y_rownnz are not equal on row %d", r);
    }
  }
}


// in-place sparse back-substitution:  Y <- Y * M^{-1/2}
static void computeY_backsub(mjtNum* Y, const int* Y_rownnz, const int* Y_rowadr,
                             const int* Y_colind, int nefc,
                             const mjtNum* qLD, const int* M_rownnz, const int* M_rowadr,
                             const int* M_colind, const mjtNum* sqrtInvD) {
  for (int r=0; r < nefc; r++) {
    int nnzY = Y_rownnz[r];
    int adrY = Y_rowadr[r];

    // Y(r,:) <- inv(L') * Y(r,:), exploit sparsity of input vector
    for (int i=adrY + nnzY-1; i >= adrY; i--) {
      mjtNum val = Y[i];
      if (val == 0) {
        continue;
      }
      int j = Y_colind[i];
      int adrM = M_rowadr[j];
      mju_addToSclSparseInc(Y + adrY, qLD + adrM,
                            nnzY, Y_colind + adrY,
                            M_rownnz[j]-1, M_colind + adrM, -val);
    }

    // Y(r,:) <- sqrt(inv(D)) * Y(r,:)
    for (int i=adrY; i < adrY + nnzY; i++) {
      int j = Y_colind[i];
      Y[i] *= sqrtInvD[j];
    }
  }
}


//---------------------------- top-level API for constraint construction ---------------------------

// driver: call all functions above
void mj_makeConstraint(const mjModel* m, mjData* d) {
  // clear sizes
  d->ne = d->nf = d->nl = d->nefc = d->nJ = d->nA = d->nY = 0;

  // disabled or Jacobian not allocated: return
  if (mjDISABLED(mjDSBL_CONSTRAINT)) {
    return;
  }

  // precount sizes for constraint Jacobian matrices
  int *nnz = mj_isSparse(m) ? &(d->nJ) : NULL;
  int ne_allocated = mj_ne(m, d, nnz);
  int nf_allocated = mj_instantiateFriction(m, d, 1, nnz);
  int nl_allocated = mj_instantiateLimit(m, d, 1, nnz);
  int nc_allocated = mj_nc(m, d, nnz);
  int nefc_allocated = ne_allocated + nf_allocated + nl_allocated + nc_allocated;
  if (!mj_isSparse(m)) {
    d->nJ = nefc_allocated * m->nv;
  }
  d->nefc = nefc_allocated;

  // allocate efc arrays on arena
  if (!arenaAllocEfc(m, d)) {
    return;
  }

  // clear tendon_efcadr
  mju_fillInt(d->tendon_efcadr, -1, m->ntendon);

  // reset nefc for the instantiation functions, instantiate all elements of Jacobian
  d->nefc = 0;
  mj_instantiateEquality(m, d);
  mj_instantiateFriction(m, d, 0, NULL);
  mj_instantiateLimit(m, d, 0, NULL);
  mj_instantiateContact(m, d);

  // check sparse allocation
  if (mj_isSparse(m)) {
    if (d->ne != ne_allocated) {
      mjERROR("ne mis-allocation: found ne=%d but allocated %d", d->ne, ne_allocated);
    }

    if (d->nf != nf_allocated) {
      mjERROR("nf mis-allocation: found nf=%d but allocated %d", d->nf, nf_allocated);
    }

    if (d->nl != nl_allocated) {
      mjERROR("nl mis-allocation: found nl=%d but allocated %d", d->nl, nl_allocated);
    }

    // check that nefc was computed correctly
    if (d->nefc != nefc_allocated) {
      mjERROR("nefc mis-allocation: found nefc=%d but allocated %d", d->nefc, nefc_allocated);
    }

    // check that nJ was computed correctly
    if (d->nefc > 0) {
      int nJ = d->efc_J_rownnz[d->nefc - 1] + d->efc_J_rowadr[d->nefc - 1];
      if (d->nJ != nJ) {
        mjERROR("constraint Jacobian mis-allocation: found nJ=%d but allocated %d", nJ, d->nJ);
      }
    }
  } else if (d->nefc > nefc_allocated) {
    mjERROR("nefc under-allocation: found nefc=%d but allocated only %d", d->nefc, nefc_allocated);
  }

  // collect memory use statistics
  d->maxuse_con = mjMAX(d->maxuse_con, d->ncon);
  d->maxuse_efc = mjMAX(d->maxuse_efc, d->nefc);

  // no constraints: return
  if (!d->nefc) {
    return;
  }

  // compute supernodes of J
  if (mj_isSparse(m)) {
    mju_superSparse(d->nefc, d->efc_J_rowsuper, d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind);
  }

  // compute regularization; under the discrete integrator this happens at the
  // actuation stage (mj_regularizeConstraint), where the solve metric is final
  if (!mj_isMetric(m)) {
    // compute diagApprox
    mj_diagApprox(m, d);

    // compute KBIP, D, R, adjust diagA
    mj_makeImpedance(m, d);
  }
}


// symbolic phase of Y = J*M^{-1/2}: the sparsity pattern and its arena allocation.
// The pattern depends only on efc_J and the tree factor's pattern (M's), both fixed at
// the position stage, so this runs once per step in mj_projectConstraint
static void mj_makeYSymbolic(const mjModel* m, mjData* d) {
  int nefc = d->nefc, nv = m->nv;

  // sparse: pre-counted pattern
  if (mj_isSparse(m)) {
    // arena-allocate Y rownnz and rowadr
    d->efc_Y_rownnz = mj_arenaAllocByte(d, sizeof(int) * nefc, _Alignof(int));
    d->efc_Y_rowadr = mj_arenaAllocByte(d, sizeof(int) * nefc, _Alignof(int));
    if (!d->efc_Y_rownnz || !d->efc_Y_rowadr) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      mj_clearEfc(d);
      return;
    }

    // pre-count Y_rownnz, Y_rowadr, nY (total nonzeros)
    mj_markStack(d);
    int* marker = mjSTACKALLOC(d, nv, int);
    d->nY = computeY_precount(d->efc_Y_rownnz, d->efc_Y_rowadr, nefc, nv,
                              d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                              m->M_rownnz, m->M_rowadr, m->M_colind, marker);
    mj_freeStack(d);

    // arena-allocate values and column indices
    d->efc_Y = mj_arenaAllocByte(d, sizeof(mjtNum) * d->nY, _Alignof(mjtNum));
    d->efc_Y_colind = mj_arenaAllocByte(d, sizeof(int) * d->nY, _Alignof(int));
    if (!d->efc_Y || !d->efc_Y_colind) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      mj_clearEfc(d);
      return;
    }

    // under discrete the numeric phase is deferred to the actuation stage, but
    // mj_makeARSymbolic consumes the column indices now: fill the pattern here.
    // Classic integrators fill it in the numeric phase which follows immediately
    if (mj_isMetric(m)) {
      computeY_fill(NULL, d->efc_Y_colind, d->efc_Y_rownnz, d->efc_Y_rowadr, nefc,
                    NULL, d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                    m->M_rownnz, m->M_rowadr, m->M_colind);
    }
  }

  // dense: one block
  else {
    d->nY = nefc * nv;
    d->efc_Y = mj_arenaAllocByte(d, sizeof(mjtNum) * d->nY, _Alignof(mjtNum));
    if (!d->efc_Y) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      mj_clearEfc(d);
      return;
    }
  }
}


// numeric phase of Y = J*M^{-1/2}: refill values from J and backsubstitute against the
// current solve metric's factor; if flg_diagexact, overwrite efc_diagA with ||Y_i||^2.
// No allocation: safe to re-run at the actuation stage (mj_regularizeConstraint)
static void mj_makeYNumeric(const mjModel* m, mjData* d, int flg_diagexact) {
  int nefc = d->nefc, nv = m->nv;

  // symbolic phase did not run (option flags changed mid-step): nothing to fill
  if (!d->efc_Y) {
    return;
  }

  mj_markStack(d);

  // factor of the solve metric: the qH backbone under discrete, else qLD. The backbone is
  // all the dual solvers see: under PGS the coupling classes are excluded from the metric
  // (mj_effCouplings) so the backbone IS the metric; flex is rejected (mj_checkDiscrete);
  // under a primal solver the noslip post-pass consumes this AR as an approximation
  const mjtNum* factor = d->qLD;
  if (mj_effUseQH(d)) {
    factor = d->qH;
  }

  // inverse square root of D from the metric factor's LDL decomposition
  mjtNum* sqrtInvD = mjSTACKALLOC(d, nv, mjtNum);
  for (int i=0; i < nv; i++) {
    int diag = m->M_rowadr[i] + m->M_rownnz[i] - 1;
    sqrtInvD[i] = 1 / mju_sqrt(factor[diag]);
  }

  // sparse Y = backsubM2(J')'
  if (mj_isSparse(m)) {
    // fill in Y column indices, copy values from J (reconstructs every slot: re-entrant)
    computeY_fill(d->efc_Y, d->efc_Y_colind, d->efc_Y_rownnz, d->efc_Y_rowadr, nefc,
                  d->efc_J, d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind,
                  m->M_rownnz, m->M_rowadr, m->M_colind);

    // in-place sparse back-substitution:  Y <- Y * M^-1/2
    computeY_backsub(d->efc_Y, d->efc_Y_rownnz, d->efc_Y_rowadr,
                     d->efc_Y_colind, nefc,
                     factor, m->M_rownnz, m->M_rowadr, m->M_colind, sqrtInvD);

    // overwrite diagA with exact diagonal: diagA[i] = ||Y_i||^2
    if (flg_diagexact) {
      for (int i=0; i < nefc; i++) {
        int adr = d->efc_Y_rowadr[i];
        int nnz = d->efc_Y_rownnz[i];
        d->efc_diagA[i] = mju_dot(d->efc_Y+adr, d->efc_Y+adr, nnz);
      }
    }
  }

  // dense Y = backsubM2(J')'
  else {
    mj_solveM2_impl(d->efc_Y, d->efc_J, sqrtInvD, factor, m->nv, nefc,
                    m->M_rownnz, m->M_rowadr, m->M_colind, m->dof_simplenum);

    // overwrite diagA with exact diagonal: diagA[i] = ||Y_i||^2
    if (flg_diagexact) {
      for (int i=0; i < nefc; i++) {
        d->efc_diagA[i] = mju_dot(d->efc_Y+i*nv, d->efc_Y+i*nv, nv);
      }
    }
  }

  mj_freeStack(d);
}


// symbolic phase of AR = Y*Y' + diag(R): the sparsity pattern and its arena allocation.
// The pattern depends only on Y's pattern, so this runs once per step in
// mj_projectConstraint, after mj_makeYSymbolic
static void mj_makeARSymbolic(const mjModel* m, mjData* d) {
  int nefc = d->nefc, nv = m->nv;

  // sparse: symbolic square from Y's pattern
  if (mj_isSparse(m)) {
    // Y supernodes are identical to J supernodes
    const int* Y_rowsuper = d->efc_J_rowsuper;

    mj_markStack(d);

    // Y transposed, pattern only
    int* YT_rownnz = mjSTACKALLOC(d, nv, int);
    int* YT_rowadr = mjSTACKALLOC(d, nv, int);
    int* YT_colind = mjSTACKALLOC(d, d->nY, int);
    mju_transposeSparse(NULL, NULL, nefc, nv,
                        YT_rownnz, YT_rowadr, YT_colind, NULL,
                        d->efc_Y_rownnz, d->efc_Y_rowadr, d->efc_Y_colind);

    // allocate AR row nonzeros and addresses on arena
    d->efc_AR_rownnz = mj_arenaAllocByte(d, sizeof(int) * nefc, _Alignof(int));
    d->efc_AR_rowadr = mj_arenaAllocByte(d, sizeof(int) * nefc, _Alignof(int));
    if (!d->efc_AR_rownnz || !d->efc_AR_rowadr) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      mj_clearEfc(d);
      mj_freeStack(d);
      return;
    }

    int* diagind = mjSTACKALLOC(d, nefc, int);
    d->nA = mju_sqrMatTDSparseSymbolic(
        d->efc_AR_rownnz, d->efc_AR_rowadr, NULL, diagind,
        nv, nefc, YT_rownnz, YT_rowadr, YT_colind, NULL,
        d->efc_Y_rownnz, d->efc_Y_rowadr, d->efc_Y_colind, Y_rowsuper, d);

    // allocate A values and column indices on arena
    d->efc_AR = mj_arenaAllocByte(d, sizeof(mjtNum) * d->nA, _Alignof(mjtNum));
    d->efc_AR_colind = mj_arenaAllocByte(d, sizeof(int) * d->nA, _Alignof(int));
    if (!d->efc_AR || !d->efc_AR_colind) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      mj_clearEfc(d);
      mj_freeStack(d);
      return;
    }

    // A = Y * Y': symbolic phase
    mju_sqrMatTDSparseSymbolic(
        d->efc_AR_rownnz, d->efc_AR_rowadr, d->efc_AR_colind, diagind,
        nv, nefc, YT_rownnz, YT_rowadr, YT_colind, NULL,
        d->efc_Y_rownnz, d->efc_Y_rowadr, d->efc_Y_colind, Y_rowsuper, d);

    mj_freeStack(d);
  }

  // dense: one block
  else {
    d->nA = nefc * nefc;
    d->efc_AR = mj_arenaAllocByte(d, sizeof(mjtNum) * d->nA, _Alignof(mjtNum));
    if (!d->efc_AR) {
      mj_warning(d, mjWARN_CNSTRFULL, d->narena);
      mj_clearEfc(d);
      return;
    }
  }
}


// numeric phase of AR = Y*Y' + diag(R), into the pattern laid down by the symbolic
// phase. No allocation: safe to re-run at the actuation stage (mj_regularizeConstraint)
static void mj_makeARNumeric(const mjModel* m, mjData* d) {
  int nefc = d->nefc, nv = m->nv;

  // symbolic phase did not run (option flags changed mid-step): nothing to fill
  if (!d->efc_AR) {
    return;
  }

  mj_markStack(d);

  // sparse
  if (mj_isSparse(m)) {
    // Y supernodes are identical to J supernodes
    const int* Y_rowsuper = d->efc_J_rowsuper;

    // construct Y transposed
    int* YT_rownnz = mjSTACKALLOC(d, nv, int);
    int* YT_rowadr = mjSTACKALLOC(d, nv, int);
    int* YT_colind = mjSTACKALLOC(d, d->nY, int);
    mjtNum* YT = mjSTACKALLOC(d, d->nY, mjtNum);
    mju_transposeSparse(YT, d->efc_Y, nefc, nv,
                        YT_rownnz, YT_rowadr, YT_colind, NULL,
                        d->efc_Y_rownnz, d->efc_Y_rowadr, d->efc_Y_colind);

    // diagonal positions, recovered from the stored pattern (rows are sorted)
    int* diagind = mjSTACKALLOC(d, nefc, int);
    for (int i=0; i < nefc; i++) {
      int adr = d->efc_AR_rowadr[i];
      int end = adr + d->efc_AR_rownnz[i];
      while (adr < end && d->efc_AR_colind[adr] < i) {
        adr++;
      }
      diagind[i] = adr;
    }

    // A = Y * Y': numeric phase
    mju_sqrMatTDSparseNumeric(
        d->efc_AR, nefc, d->efc_AR_rownnz, d->efc_AR_rowadr,
        d->efc_AR_colind, diagind, YT, YT_rownnz, YT_rowadr,
        YT_colind, d->efc_Y, d->efc_Y_rownnz, d->efc_Y_rowadr,
        d->efc_Y_colind, Y_rowsuper, NULL, d);

    // AR = A + diag(R)
    for (int i=0; i < nefc; i++) {
      d->efc_AR[diagind[i]] += d->efc_R[i];
    }
  }

  // dense
  else {
    // construct YT on stack
    mjtNum* YT = mjSTACKALLOC(d, nv*nefc, mjtNum);
    mju_transpose(YT, d->efc_Y, nefc, nv);

    // AR = Y * Y'
    mju_sqrMatTD(d->efc_AR, YT, NULL, nv, nefc);

    // add R to diagonal of AR
    for (int r=0; r < nefc; r++) {
      d->efc_AR[r*(nefc+1)] += d->efc_R[r];
    }
  }

  mj_freeStack(d);
}


// compute efc_Y, optionally efc_diagA, optionally efc_AR
void mj_projectConstraint(const mjModel* m, mjData* d) {
  int nefc = d->nefc;

  // nothing to do
  if (!nefc) {
    return;
  }

  int isDual = mj_isDual(m);
  int diagexact = mjENABLED(mjENBL_DIAGEXACT);

  // under the discrete integrator the numeric phase runs at the actuation stage
  // (mj_regularizeConstraint), against the final effective metric. The patterns are
  // position-stage facts, so their symbolic phase and arena allocation happen here:
  // the arena's contract is that all allocation is position-stage, which keeps
  // re-evaluation from the velocity stage on (mj_forwardSkip, finite differences)
  // allocation-free
  if (mj_isMetric(m)) {
    if (isDual || diagexact) {
      mj_makeYSymbolic(m, d);
      if (isDual && d->nefc) {
        mj_makeARSymbolic(m, d);
      }
    }
    return;
  }

  // compute Y = J*M^{-1/2}; overwrite diagApprox if diagexact
  if (isDual || diagexact) {
    mj_makeYSymbolic(m, d);
    if (d->nefc) {
      mj_makeYNumeric(m, d, diagexact);
    }
  }

  // recompute impedance from exact diagonal
  if (diagexact && d->nefc) {
    mj_makeImpedance(m, d);

    // re-gather island D/R
    if (d->nisland) {
      mju_gather(d->iefc_D, d->efc_D, d->map_iefc2efc, d->nefc);
      mju_gather(d->iefc_R, d->efc_R, d->map_iefc2efc, d->nefc);
    }
  }

  // assemble AR for dual solver
  if (isDual && d->nefc) {
    mj_makeARSymbolic(m, d);
    if (d->nefc) {
      mj_makeARNumeric(m, d);
    }
  }
}


// add relative surface velocity of contacts to contact rows of efc_vel
static void mj_addSurfaceVel(const mjModel* m, mjData* d) {
  // no surface velocity: quick return
  if (!m->flg_surfacevel) {
    return;
  }

  int ispyramid = mj_isPyramidal(m);
  int ncon = d->ncon;

  // loop over contacts, add surface velocity to efc_vel
  for (int i=0; i < ncon; i++) {
    // get contact, skip excluded or frictionless
    const mjContact* con = d->contact + i;
    if (con->efc_address < 0 || con->dim <= 1) {
      continue;
    }

    // skip contact with no surface velocity
    const mjtNum* sv = con->surfacevel;
    if (!sv[0] && !sv[1] && !sv[2]) {
      continue;
    }

    // add to contact rows: (tangent1, tangent2, spin) map to directions k = 1, 2, 3
    int adr = con->efc_address, n = mjMIN(con->dim, 4);
    if (!ispyramid) {
      for (int k=1; k < n; k++) {
        d->efc_vel[adr + k] += sv[k-1];
      }
    } else {
      for (int k=1; k < n; k++) {
        mjtNum mu = con->friction[k-1];
        d->efc_vel[adr + 2*(k-1)]     += mu*sv[k-1];
        d->efc_vel[adr + 2*(k-1) + 1] -= mu*sv[k-1];
      }
    }
  }
}


// bias aref of adhesive contact rows: makes the compression branch of the net
// force-penetration curve independent of adhesion (exact translated-cone semantics)
static void mj_adhesionRef(const mjModel* m, mjData* d) {
  // no adhesion on any geom or pair: quick return
  if (!m->flg_adhesion) {
    return;
  }

  int ispyramid = mj_isPyramidal(m), ncon = d->ncon;

  for (int i=0; i < ncon; i++) {
    const mjContact* con = d->contact + i;
    if (!con->adhesion || con->efc_address < 0) {
      continue;
    }

    int adr = con->efc_address, dim = con->dim;
    if (dim == 1 || !ispyramid) {
      // normal row
      d->efc_aref[adr] += d->efc_R[adr] * con->adhesion;
    } else {
      // pyramid rows: the normal decomposes equally over the 2*(dim-1) edges
      mjtNum edge = con->adhesion / (2*(dim-1));
      for (int j=0; j < 2*(dim-1); j++) {
        d->efc_aref[adr+j] += d->efc_R[adr+j] * edge;
      }
    }
  }
}



// compute efc_vel
void mj_velocityConstraint(const mjModel* m, mjData* d) {
  mj_mulJacVec(m, d, d->efc_vel, d->qvel);
  mj_addSurfaceVel(m, d);
}


// compute efc_vel, efc_aref
void mj_referenceConstraint(const mjModel* m, mjData* d) {
  int nefc = d->nefc;
  const mjtNum* KBIP = d->efc_KBIP;
  int metric = mj_isMetric(m);
  mjtNum h = m->opt.timestep;

  // compute efc_vel
  mj_velocityConstraint(m, d);

  // compute aref = -B*vel - K*I*(pos-margin). Under the discrete integrator the row's spring-damper
  // is treated implicitly (backward Euler on the row): evaluated at the end-of-step state,
  // the position transported by h*vel
  mjtNum shift = metric ? h : 0;
  for (int i=0; i < nefc; i++) {
    const mjtNum* kbip = KBIP + 4*i;
    d->efc_aref[i] = -kbip[1]*d->efc_vel[i]
                     -kbip[0]*kbip[2]*(d->efc_pos[i]-d->efc_margin[i] + shift*d->efc_vel[i]);
  }

  // subtract Jdot*v correction for connect/weld equality constraints
  if (d->ne > 0) {
    mj_Jdotv(m, d, d->efc_aref);
  }

  // implicit rows: divide by the factor which scaled the row weight in mj_makeImpedance;
  // weight and reference are one identity, neither is valid alone
  if (metric) {
    for (int i=0; i < nefc; i++) {
      d->efc_aref[i] /= implicitFactor(KBIP + 4*i, h);
    }
  }

  // bias adhesive contact rows: a force offset (D*R*edge = edge), independent of the row factor,
  // so it is added after the division
  mj_adhesionRef(m, d);
}


//---------------------------- update constraint state ---------------------------------------------

// compute efc_state, efc_force
//  optional: cost(qacc) = s_hat(jar); cone Hessians
void mj_constraintUpdate_impl(int ne, int nf, int nefc,
                              const mjtNum* D, const mjtNum* R, const mjtNum* floss,
                              const mjtNum* jar, const int* type, const int* id,
                              mjContact* contact, int* state, mjtNum* force, mjtNum cost[1],
                              int flg_coneHessian) {
  mjtNum s = 0;

  // no constraints: clear cost, return
  if (!nefc) {
    if (cost) {
      *cost = 0;
    }
    return;
  }

  // compute unconstrained efc_force
  for (int i=0; i < nefc; i++) {
    force[i] = -D[i]*jar[i];
  }

  // update constraints
  for (int i=0; i < nefc; i++) {
    // ==== equality
    if (i < ne) {
      if (cost) {
        s += 0.5*D[i]*jar[i]*jar[i];
      }
      state[i] = mjCNSTRSTATE_QUADRATIC;
      continue;
    }

    // ==== friction
    if (i < ne + nf) {
      // linear negative
      if (jar[i] <= -R[i]*floss[i]) {
        if (cost) {
          s += -0.5*R[i]*floss[i]*floss[i] - floss[i]*jar[i];
        }

        force[i] = floss[i];
        state[i] = mjCNSTRSTATE_LINEARNEG;
      }

      // linear positive
      else if (jar[i] >= R[i]*floss[i]) {
        if (cost) {
          s += -0.5*R[i]*floss[i]*floss[i] + floss[i]*jar[i];
        }

        force[i] = -floss[i];
        state[i] = mjCNSTRSTATE_LINEARPOS;
      }

      // quadratic
      else {
        if (cost) {
          s += 0.5*D[i]*jar[i]*jar[i];
        }
        state[i] = mjCNSTRSTATE_QUADRATIC;
      }
      continue;
    }

    // ==== contact

    // non-negative constraint
    if (type[i] != mjCNSTR_CONTACT_ELLIPTIC) {
      // constraint is satisfied: no cost
      if (jar[i] >= 0) {
        force[i] = 0;

        state[i] = mjCNSTRSTATE_SATISFIED;
      }

      // quadratic
      else {
        if (cost) {
          s += 0.5*D[i]*jar[i]*jar[i];
        }
        state[i] = mjCNSTRSTATE_QUADRATIC;
      }
    }

    // contact with elliptic cone
    else {
      // get contact
      mjContact* con = contact + id[i];
      mjtNum mu = con->mu, *friction = con->friction;
      int dim = con->dim;

      // map to regular dual cone space
      mjtNum U[6];
      U[0] = jar[i]*mu;
      for (int j=1; j < dim; j++) {
        U[j] = jar[i+j]*friction[j-1];
      }

      // decompose into normal and tangent
      mjtNum N = U[0];
      mjtNum T = mju_norm(U+1, dim-1);

      // top zone
      if (N >= mu*T || (T <= 0 && N >= 0)) {
        mju_zero(force+i, dim);
        state[i] = mjCNSTRSTATE_SATISFIED;
      }

      // bottom zone
      else if (mu*N+T <= 0 || (T <= 0 && N < 0)) {
        if (cost) {
          for (int j=0; j < dim; j++) {
            s += 0.5*D[i+j]*jar[i+j]*jar[i+j];
          }
        }
        state[i] = mjCNSTRSTATE_QUADRATIC;
      }

      // middle zone
      else {
        // cost: 0.5*D0/(mu*mu*(1+mu*mu))*(N-mu*T)^2
        mjtNum Dm = D[i]/(mu*mu*(1+mu*mu));
        mjtNum NmT = N - mu*T;

        if (cost) {
          s += 0.5*Dm*NmT*NmT;
        }

        // unit tangent vector
        mjtNum Uhat[5];
        for (int j=1; j < dim; j++) {
          Uhat[j-1] = U[j] / T;
        }

        // force: - ds/djar = dU/djar * ds/dU  (dU/djar = diag(mu, friction))
        force[i] = -Dm*NmT*mu;
        for (int j=1; j < dim; j++) {
          force[i+j] = -force[i]*Uhat[j-1]*friction[j-1];
        }

        // set state
        state[i] = mjCNSTRSTATE_CONE;

        // cone Hessian
        if (flg_coneHessian) {
          // get Hessian pointer
          mjtNum* H = contact[id[i]].H;

          // set first row: (1, -mu * Uhat)
          H[0] = 1;
          for (int j=1; j < dim; j++) {
            H[j] = -mu*Uhat[j-1];
          }

          // set upper block: (mu*N/T) * Uhat*Uhat'
          mjtNum scl = mu*(N/T);
          for (int k=1; k < dim; k++) {
            for (int j=k; j < dim; j++) {
              H[k*dim+j] = scl*Uhat[j-1]*Uhat[k-1];
            }
          }

          // add to diagonal: (mu^2 - mu*N/T) * I
          scl = mu*mu - scl;
          for (int j=1; j < dim; j++) {
            H[j*(dim+1)] += scl;
          }

          // pre and post multiply by diag(mu, friction), scale by Dm
          for (int k=0; k < dim; k++) {
            scl = Dm * (k == 0 ? mu : friction[k-1]);
            for (int j=k; j < dim; j++) {
              H[k*dim+j] *= scl * (j == 0 ? mu : friction[j-1]);
            }
          }

          // make symmetric: copy upper into lower
          for (int k=0; k < dim; k++) {
            for (int j=k+1; j < dim; j++) {
              H[j*dim+k] = H[k*dim+j];
            }
          }
        }
      }

      // replicate state in all cone dimensions
      for (int j=1; j < dim; j++) {
        state[i+j] = state[i];
      }

      // advance to end of contact
      i += (dim-1);
    }
  }

  // assign cost
  if (cost) {
    *cost = s;
  }
}


// compute efc_state, efc_force, qfrc_constraint
// optional: cost(qacc) = s_hat(jar) where jar = Jac*qacc-aref; cone Hessians
void mj_constraintUpdate(const mjModel* m, mjData* d, const mjtNum* jar,
                         mjtNum cost[1], int flg_coneHessian) {
  mj_constraintUpdate_impl(d->ne, d->nf, d->nefc, d->efc_D, d->efc_R, d->efc_frictionloss,
                           jar, d->efc_type, d->efc_id, d->contact, d->efc_state, d->efc_force,
                           cost, flg_coneHessian);
  mj_mulJacTVec(m, d, d->qfrc_constraint, d->efc_force);
}
