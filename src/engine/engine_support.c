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

#include "engine/engine_support.h"

#include <inttypes.h>  // IWYU pragma: keep
#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_collision_convex.h"
#include "engine/engine_collision_driver.h"
#include "engine/engine_collision_gjk.h"
#include "engine/engine_collision_primitive.h"
#include "engine/engine_core_util.h"
#include "engine/engine_crossplatform.h"
#include "engine/engine_memory.h"
#include "engine/engine_memory.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"

#ifdef mjUSEPLATFORMSIMD
  #if defined(__AVX__) && !defined(mjUSESINGLE)
    #define mjUSEAVX
    #include "immintrin.h"
  #endif
#endif

//-------------------------- Constants -------------------------------------------------------------

 #define mjVERSION 3017001
#define mjVERSIONSTRING "3.17.1"

// names of disable flags
const char* mjDISABLESTRING[mjNDISABLE] = {
  "Constraint",
  "Equality",
  "Frictionloss",
  "Limit",
  "Contact",
  "Spring",
  "Damper",
  "Gravity",
  "Clampctrl",
  "Warmstart",
  "Filterparent",
  "Actuation",
  "Refsafe",
  "Sensor",
  "Midphase",
  "Eulerdamp",
  "AutoReset",
  "NativeCCD",
  "Island",
  "MultiCCD"
};


// names of enable flags
const char* mjENABLESTRING[mjNENABLE] = {
  "Override",
  "Energy",
  "Fwdinv",
  "InvDiscrete",
  "Sleep",
  "DiagExact",
  "IPC",
  "Speculative"
};


// names of timers
const char* mjTIMERSTRING[mjNTIMER]= {
  "step",
  "forward",
  "inverse",
  "position",
  "velocity",
  "actuation",
  "constraint",
  "advance",
  "pos_kinematics",
  "pos_inertia",
  "pos_collision",
  "pos_make",
  "pos_project",
  "col_broadphase",
  "col_narrowphase"
};


// names of log topics (index i corresponds to topic i+1)
const char* mjTOPICSTRING[mjNTOPIC] = {
  "Step timing",
  "Compile timing",
  "Sleep/wake"
};


// size of contact data fields
const int mjCONDATA_SIZE[mjNCONDATA] = {
  1,  // mjCONDATA_FOUND
  3,  // mjCONDATA_FORCE
  3,  // mjCONDATA_TORQUE
  1,  // mjCONDATA_DIST
  3,  // mjCONDATA_POS
  3,  // mjCONDATA_NORMAL
  3,  // mjCONDATA_TANGENT
  3,  // mjCONDATA_LINVEL
  3   // mjCONDATA_ANGVEL
};


// size of ray data fields
const int mjRAYDATA_SIZE[mjNRAYDATA] = {
  1,  // mjRAYDATA_DIST
  3,  // mjRAYDATA_DIR
  3,  // mjRAYDATA_ORIGIN
  3,  // mjRAYDATA_POINT
  3,  // mjRAYDATA_NORMAL
  1   // mjRAYDATA_DEPTH
};

//-------------------------- get/set state ---------------------------------------------------------

// return size of a single state element
static inline int mj_stateElemSize(const mjModel* m, mjtState sig) {
  switch (sig) {
  case mjSTATE_TIME:          return 1;
  case mjSTATE_QPOS:          return m->nq;
  case mjSTATE_QVEL:          return m->nv;
  case mjSTATE_ACT:           return m->na;
  case mjSTATE_HISTORY:       return m->nhistory;
  case mjSTATE_WARMSTART:     return m->nv;
  case mjSTATE_CTRL:          return m->nu;
  case mjSTATE_QFRC_APPLIED:  return m->nv;
  case mjSTATE_XFRC_APPLIED:  return 6*m->nbody;
  case mjSTATE_EQ_ACTIVE:     return m->neq;    // mjtByte, stored as mjtNum in state vector
  case mjSTATE_MOCAP_POS:     return 3*m->nmocap;
  case mjSTATE_MOCAP_QUAT:    return 4*m->nmocap;
  case mjSTATE_USERDATA:      return m->nuserdata;
  case mjSTATE_PLUGIN:        return m->npluginstate;
  case mjSTATE_PLASTIC:       return m->nplastic;
  default:
    mjERROR("invalid state element %u", sig);
    return 0;
  }
}


// return pointer to a single state element
static inline mjtNum* mj_stateElemPtr(const mjModel* m, mjData* d, mjtState sig) {
  switch (sig) {
  case mjSTATE_TIME:          return &d->time;
  case mjSTATE_QPOS:          return d->qpos;
  case mjSTATE_QVEL:          return d->qvel;
  case mjSTATE_ACT:           return d->act;
  case mjSTATE_HISTORY:       return d->history;
  case mjSTATE_WARMSTART:     return d->qacc_warmstart;
  case mjSTATE_CTRL:          return d->ctrl;
  case mjSTATE_QFRC_APPLIED:  return d->qfrc_applied;
  case mjSTATE_XFRC_APPLIED:  return d->xfrc_applied;
  case mjSTATE_MOCAP_POS:     return d->mocap_pos;
  case mjSTATE_MOCAP_QUAT:    return d->mocap_quat;
  case mjSTATE_USERDATA:      return d->userdata;
  case mjSTATE_PLUGIN:        return d->plugin_state;
  case mjSTATE_PLASTIC:       return d->plastic;
  default:
    mjERROR("invalid state element %u", sig);
    return NULL;
  }
}


static inline const mjtNum* mj_stateElemConstPtr(const mjModel* m, const mjData* d, mjtState sig) {
  return mj_stateElemPtr(m, (mjData*) d, sig);  // discard const qualifier from d
}


// get size of state signature
int mj_stateSize(const mjModel* m, int sig) {
  if (sig < 0) {
    mjERROR("invalid state signature %d < 0", sig);
    return 0;
  }

  if (sig >= (1<<mjNSTATE)) {
    mjERROR("invalid state signature %d >= 2^mjNSTATE", sig);
    return 0;
  }

  int size = 0;
  for (int i=0; i < mjNSTATE; i++) {
    mjtState element = 1<<i;
    if (element & sig) {
      size += mj_stateElemSize(m, element);
    }
  }

  return size;
}


// get state
void mj_getState(const mjModel* m, const mjData* d, mjtNum* state, int sig) {
  if (sig < 0) {
    mjERROR("invalid state signature %d < 0", sig);
    return;
  }

  if (sig >= (1<<mjNSTATE)) {
    mjERROR("invalid state signature %d >= 2^mjNSTATE", sig);
    return;
  }

  int adr = 0;
  for (int i=0; i < mjNSTATE; i++) {
    mjtState element = 1<<i;
    if (element & sig) {
      int size = mj_stateElemSize(m, element);

      // special handling of eq_active (mjtBool)
      if (element == mjSTATE_EQ_ACTIVE) {
        int neq = m->neq;
        for (int j=0; j < neq; j++) {
          state[adr++] = d->eq_active[j];
        }
      }

      // regular state components (mjtNum)
      else {
        const mjtNum* ptr = mj_stateElemConstPtr(m, d, element);
        mju_copy(state + adr, ptr, size);
        adr += size;
      }
    }
  }
}


// extract a sub-state from a state
void mj_extractState(const mjModel* m, const mjtNum* src, int srcsig, mjtNum* dst, int dstsig) {
  if (srcsig < 0) {
    mjERROR("invalid srcsig %d < 0", srcsig);
    return;
  }

  if (srcsig >= (1<<mjNSTATE)) {
    mjERROR("invalid srcsig %d >= 2^mjNSTATE", srcsig);
    return;
  }

  if ((srcsig & dstsig) != dstsig) {
    mjERROR("dstsig is not a subset of srcsig");
    return;
  }

  for (int i=0; i < mjNSTATE; i++) {
    mjtState element = 1<<i;
    if (element & srcsig) {
      int size = mj_stateElemSize(m, element);
      if (element & dstsig) {
        mju_copy(dst, src, size);
        dst += size;
      }
      src += size;
    }
  }
}


// set state
void mj_setState(const mjModel* m, mjData* d, const mjtNum* state, int sig) {
  if (sig < 0) {
    mjERROR("invalid state signature %d < 0", sig);
    return;
  }

  if (sig >= (1<<mjNSTATE)) {
    mjERROR("invalid state signature %d >= 2^mjNSTATE", sig);
    return;
  }

  int adr = 0;
  for (int i=0; i < mjNSTATE; i++) {
    mjtState element = 1<<i;
    if (element & sig) {
      int size = mj_stateElemSize(m, element);

      // special handling of eq_active (mjtBool)
      if (element == mjSTATE_EQ_ACTIVE) {
        int neq = m->neq;
        for (int j=0; j < neq; j++) {
          d->eq_active[j] = state[adr++];
        }
      }

      // regular state components (mjtNum)
      else {
        mjtNum* ptr = mj_stateElemPtr(m, d, element);
        mju_copy(ptr, state + adr, size);
        adr += size;
        if (element == mjSTATE_PLASTIC && d->flex_hessian_valid) {
          for (int f=0; f < m->nflex; f++) {
            if (m->flex_plasticadr[f] >= 0) {
              d->flex_hessian_valid[f] = 0;
            }
          }
        }
      }
    }
  }
}


// copy state from src to dst
void mj_copyState(const mjModel* m, const mjData* src, mjData* dst, int sig) {
  if (sig < 0) {
    mjERROR("invalid state signature %d < 0", sig);
    return;
  }

  if (sig >= (1<<mjNSTATE)) {
    mjERROR("invalid state signature %d >= 2^mjNSTATE", sig);
    return;
  }

  for (int i=0; i < mjNSTATE; i++) {
    mjtState element = 1<<i;
    if (element & sig) {
      int size = mj_stateElemSize(m, element);

      // special handling of eq_active (mjtBool)
      if (element == mjSTATE_EQ_ACTIVE) {
        int neq = m->neq;
        for (int j=0; j < neq; j++) {
          dst->eq_active[j] = src->eq_active[j];
        }
      }

      // regular state components (mjtNum)
      else {
        mjtNum* dst_ptr = mj_stateElemPtr(m, dst, element);
        const mjtNum* src_ptr = mj_stateElemConstPtr(m, src, element);
        mju_copy(dst_ptr, src_ptr, size);
        if (element == mjSTATE_PLASTIC && dst->flex_hessian_valid) {
          for (int f=0; f < m->nflex; f++) {
            if (m->flex_plasticadr[f] >= 0) {
              dst->flex_hessian_valid[f] = 0;
            }
          }
        }
      }
    }
  }
}


// copy current state to the k-th model keyframe
void mj_setKeyframe(mjModel* m, const mjData* d, int k) {
  // check keyframe index
  if (k >= m->nkey) {
    mjERROR("index must be smaller than %" PRId64 " (keyframes allocated in model)", m->nkey);
  }
  if (k < 0) {
    mjERROR("keyframe index cannot be negative");
  }

  // copy state to model keyframe
  m->key_time[k] = d->time;
  mju_copy(m->key_qpos + k*m->nq, d->qpos, m->nq);
  mju_copy(m->key_qvel + k*m->nv, d->qvel, m->nv);
  mju_copy(m->key_act + k*m->na, d->act, m->na);
  mju_copy(m->key_mpos + k*3*m->nmocap, d->mocap_pos, 3*m->nmocap);
  mju_copy(m->key_mquat + k*4*m->nmocap, d->mocap_quat, 4*m->nmocap);
  mju_copy(m->key_ctrl + k*m->nu, d->ctrl, m->nu);
  mju_copy(m->key_plastic + k*m->nplastic, d->plastic, m->nplastic);
}


//-------------------------- inertia functions -----------------------------------------------------

// convert sparse inertia matrix M into full matrix
void mj_fullM(const mjModel* m, const mjData* d, mjtNum* dst) {
  mju_sym2dense(dst, d->M, m->nv, m->M_rownnz, m->M_rowadr, m->M_colind);
}


// multiply vector by inertia matrix
void mj_mulM(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec) {
  mju_mulSymVecSparse(res, d->M, vec, m->nv, m->M_rownnz, m->M_rowadr, m->M_colind);
}


// multiply vector by M^(1/2)
void mj_mulM2(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec) {
  int  nv = m->nv;
  const mjtNum* qLD = d->qLD;

  mju_zero(res, nv);

  // res = L * vec
  for (int i=0; i < nv; i++) {
    // diagonal
    res[i] = vec[i];

    // non-simple: add off-diagonals
    if (!m->dof_simplenum[i]) {
      int adr = m->M_rowadr[i];
      res[i] += mju_dotSparse(qLD+adr, vec, m->M_rownnz[i] - 1, m->M_colind+adr);
    }
  }

  // res *= sqrt(D)
  for (int i=0; i < nv; i++) {
    int diag = m->M_rowadr[i] + m->M_rownnz[i] - 1;
    res[i] *= mju_sqrt(qLD[diag]);
  }
}


// add inertia matrix to destination matrix
//  destination can be sparse or dense when all int* are NULL
void mj_addM(const mjModel* m, mjData* d, mjtNum* dst,
             int* rownnz, int* rowadr, int* colind) {
  int nv = m->nv;

  // sparse
  if (rownnz && rowadr && colind) {
    mju_addToMatSparse(dst, rownnz, rowadr, colind, nv, d->M,
                       m->M_rownnz, m->M_rowadr, m->M_colind);
  }

  // dense
  else {
    mju_addToSymSparse(dst, d->M, nv, m->M_rownnz, m->M_rowadr, m->M_colind, /*flg_upper*/ 0);
  }
}


//-------------------------- perturbations ---------------------------------------------------------

// add Cartesian force and torque to qfrc_target
void mj_applyFT(const mjModel* m, mjData* d,
                const mjtNum force[3], const mjtNum torque[3],
                const mjtNum point[3], int body, mjtNum* qfrc_target) {
  int nv = m->nv;

  // allocate local variables
  mj_markStack(d);
  mjtNum* jacp = force ? mjSTACKALLOC(d, 3*nv, mjtNum) : NULL;
  mjtNum* jacr = torque ? mjSTACKALLOC(d, 3*nv, mjtNum) : NULL;
  mjtNum* qforce = mjSTACKALLOC(d, nv, mjtNum);

  // make sure body is in range
  if (body < 0 || body >= m->nbody) {
    mjERROR("invalid body %d", body);
  }

  // sparse case
  if (mj_isSparse(m)) {
    // construct chain and sparse Jacobians
    int* chain = mjSTACKALLOC(d, nv, int);
    int NV = mj_bodyChain(m, body, chain);
    mj_jacSparse(m, d, jacp, jacr, point, body, NV, chain, /*flg_skipcommon=*/0);

    // compute J'*f and accumulate
    if (force) {
      mju_mulMatTVec(qforce, jacp, force, 3, NV);
      for (int i=0; i < NV; i++) {
        qfrc_target[chain[i]] += qforce[i];
      }
    }
    if (torque) {
      mju_mulMatTVec(qforce, jacr, torque, 3, NV);
      for (int i=0; i < NV; i++) {
        qfrc_target[chain[i]] += qforce[i];
      }
    }
  }

  // dense case
  else {
    // compute Jacobians
    mj_jac(m, d, jacp, jacr, point, body);

    // compute J'*f and accumulate
    if (force) {
      mju_mulMatTVec(qforce, jacp, force, 3, nv);
      mju_addTo(qfrc_target, qforce, nv);
    }
    if (torque) {
      mju_mulMatTVec(qforce, jacr, torque, 3, nv);
      mju_addTo(qfrc_target, qforce, nv);
    }
  }

  mj_freeStack(d);
}


// accumulate xfrc_applied in qfrc
void mj_xfrcAccumulate(const mjModel* m, mjData* d, mjtNum* qfrc) {
  int nbody = m->nbody;
  const mjtNum *xfrc = d->xfrc_applied;

  // quick return if identically zero (efficient memcmp implementation)
  if (mju_isZeroByte((const unsigned char*)(xfrc+6), 6*(nbody-1)*sizeof(mjtNum))) {
    return;
  }

  // some non-zero wrenches, apply them
  for (int i=1; i < nbody; i++) {
    if (!mju_isZero(xfrc+6*i, 6)) {
      mj_applyFT(m, d, xfrc+6*i, xfrc+6*i+3, d->xipos+3*i, i, qfrc);
    }
  }
}



//-------------------------- miscellaneous ---------------------------------------------------------

// returns the smallest distance between two geoms (using nativeccd)
static mjtNum mj_geomDistanceCCD(const mjModel* m, mjData* d, int g1, int g2,
                                 mjtNum distmax, mjtNum fromto[6]) {
  mj_markStack(d);
  mjCCDConfig config;
  mjCCDStatus status;

  // set config
  config.max_iterations = m->opt.ccd_iterations;
  config.tolerance = m->opt.ccd_tolerance;
  config.npolygonmax = 0;
  config.nmeshdegmax = 0;
  config.max_contacts = 1;        // want contacts
  config.dist_cutoff = distmax;   // want geom distances
  config.buffer = mj_stackAllocByte(d, mjc_ccdSize(0, 0, config.max_iterations), sizeof(mjtNum));

  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m, d, g1, 0);
  mjc_initCCDObj(&obj2, m, d, g2, 0);

  mjtNum dist = mjc_ccd(&config, &status, &obj1, &obj2);
  mj_freeStack(d);

  // witness points are only computed if dist <= distmax
  if (fromto && status.nx > 0) {
    mju_copy3(fromto, status.x1);
    mju_copy3(fromto+3, status.x2);
  }

  // clamp dist to distmax as mjc_ccd returns DBL_MAX if dist > distmax
  return dist < distmax ? dist : distmax;
}


// returns the smallest distance between two geoms
mjtNum mj_geomDistance(const mjModel* m, mjData* d, int geom1, int geom2, mjtNum distmax,
                       mjtNum fromto[6]) {
  mjPreContact con[mjMAXCONPAIR];
  mjtNum dist = distmax;
  if (fromto) mju_zero(fromto, 6);

  // flip geom order if required
  int flip = m->geom_type[geom1] > m->geom_type[geom2];
  int g1 = flip ? geom2 : geom1;
  int g2 = flip ? geom1 : geom2;
  int type1 = m->geom_type[g1];
  int type2 = m->geom_type[g2];

  mjfCollision func = mjCOLLISIONFUNC[type1][type2];

  // call collision function if it exists
  if (!func) {
    return dist;
  }

  // use nativeccd if flag is enabled
  if (!mjDISABLED(mjDSBL_NATIVECCD)) {
    if (func == mjc_Convex || func == mjc_BoxBox) {
      return mj_geomDistanceCCD(m, d, geom1, geom2, distmax, fromto);
    }
  }

  // call collision function with distmax as margin
  int num = func(m, d, con, g1, g2, distmax);

  // find smallest distance
  int smallest = -1;
  for (int i=0; i < num; i++) {
    mjtNum dist_i = con[i].dist;
    if (dist_i < dist || (dist_i == dist && smallest < 0)) {
      dist = dist_i;
      smallest = i;
    }
  }

  // write fromto if given and a collision has been found
  if (fromto && smallest >= 0) {
    mjtNum sign = flip ? -1 : 1;
    mju_addScl3(fromto+0, con[smallest].pos, con[smallest].normal, -0.5*sign*dist);
    mju_addScl3(fromto+3, con[smallest].pos, con[smallest].normal, 0.5*sign*dist);
  }

  return dist;
}


// return 1 if point is inside a site (convex hull for meshes), 0 otherwise
int mj_insideSite(const mjModel* m, const mjData* d, int siteid, const mjtNum point[3]) {
  int type = m->site_type[siteid];
  const mjtNum* pos = d->site_xpos + 3*siteid;
  const mjtNum* mat = d->site_xmat + 9*siteid;
  const mjtNum* size = m->site_size + 3*siteid;

  if (type == mjGEOM_MESH) {
    int meshid = m->site_dataid[siteid];
    if (meshid < 0) {
      return 0;
    }
    int pn = m->mesh_polynum[meshid];
    if (pn <= 0) {
      return 0;
    }

    // rotate into local frame
    mjtNum vec[3], plocal[3], meshmat[9];
    mju_sub3(vec, point, pos);
    mju_mulMatTVec3(plocal, mat, vec);

    // transform from authored site frame to canonical mesh frame
    mju_sub3(vec, plocal, m->mesh_pos+3*meshid);
    mju_quat2Mat(meshmat, m->mesh_quat+4*meshid);
    mju_mulMatTVec3(plocal, meshmat, vec);

    // check bounding box
    if (mju_abs(plocal[0]) > size[0] ||
        mju_abs(plocal[1]) > size[1] ||
        mju_abs(plocal[2]) > size[2]) {
      return 0;
    }

    // check convex hull polygon face planes
    const float* vbase = m->mesh_vert + 3*m->mesh_vertadr[meshid];
    int polyadr = m->mesh_polyadr[meshid];
    for (int p=0; p < pn; p++) {
      const mjtNum* pnl = m->mesh_polynormal + 3*(polyadr + p);
      const float* v0 = vbase + 3*m->mesh_polyvert[m->mesh_polyvertadr[polyadr + p]];
      mjtNum c = pnl[0]*v0[0] + pnl[1]*v0[1] + pnl[2]*v0[2];
      mjtNum dd = pnl[0]*plocal[0] + pnl[1]*plocal[1] + pnl[2]*plocal[2] - c;
      if (dd > 0) {
        return 0;
      }
    }
    return 1;
  }

  return mju_insidePrimitive(pos, mat, size, (mjtGeom)type, point);
}


// compute velocity by finite-differencing two positions
void mj_differentiatePos(const mjModel* m, mjtNum* qvel, mjtNum dt,
                         const mjtNum* qpos1, const mjtNum* qpos2) {
  // loop over joints
  for (int j=0; j < m->njnt; j++) {
    // get addresses in qpos and qvel
    int padr = m->jnt_qposadr[j];
    int vadr = m->jnt_dofadr[j];

    switch ((mjtJoint) m->jnt_type[j]) {
    case mjJNT_FREE:
      for (int i=0; i < 3; i++) {
        qvel[vadr+i] = (qpos2[padr+i] - qpos1[padr+i]) / dt;
      }
      vadr += 3;
      padr += 3;

      // continue with rotations
      mjFALLTHROUGH;

    case mjJNT_BALL:
      // solve:  qpos1 * quat(qvel * dt) = qpos2
      mju_subQuat(qvel+vadr, qpos2+padr, qpos1+padr);
      mju_scl3(qvel+vadr, qvel+vadr, 1/dt);
      break;

    case mjJNT_HINGE:
    case mjJNT_SLIDE:
      qvel[vadr] = (qpos2[padr] - qpos1[padr]) / dt;
    }
  }
}


// integrate qpos with given qvel for given body indices
void mj_integratePosInd(const mjModel* m, mjtNum* qpos, const mjtNum* qvel, mjtNum dt,
                        const int* index, int nbody) {
  for (int b=1; b < nbody; b++) {
    int k = index ? index[b] : b;
    int start = m->body_jntadr[k];
    int end = start + m->body_jntnum[k];
    for (int j=start; j < end; j++) {
      // get addresses in qpos and qvel
      int padr = m->jnt_qposadr[j];
      int vadr = m->jnt_dofadr[j];

      switch ((mjtJoint) m->jnt_type[j]) {
      case mjJNT_FREE:
        // position update
        for (int i=0; i < 3; i++) {
          qpos[padr+i] += dt * qvel[vadr+i];
        }
        padr += 3;
        vadr += 3;

        // continue with rotation update
        mjFALLTHROUGH;

      case mjJNT_BALL:
        // quaternion update
        mju_quatIntegrate(qpos+padr, qvel+vadr, dt);
        break;

      case mjJNT_HINGE:
      case mjJNT_SLIDE:
        // scalar update: same for rotation and translation
        qpos[padr] += dt * qvel[vadr];
      }
    }
  }
}


// integrate qpos with given qvel
void mj_integratePos(const mjModel* m, mjtNum* qpos, const mjtNum* qvel, mjtNum dt) {
  mj_integratePosInd(m, qpos, qvel, dt, NULL, m->nbody);
}


// normalize all quaternions in qpos-type vector
void mj_normalizeQuat(const mjModel* m, mjtNum* qpos) {
  // find quaternion fields and normalize
  for (int i=0; i < m->njnt; i++) {
    if (m->jnt_type[i] == mjJNT_BALL || m->jnt_type[i] == mjJNT_FREE) {
      mju_normalize4(qpos+m->jnt_qposadr[i]+3*(m->jnt_type[i] == mjJNT_FREE));
    }
  }
}


// return 1 if actuator i is disabled, 0 otherwise
int mj_actuatorDisabled(const mjModel* m, int i) {
  int group = m->actuator_group[i];
  if (group < 0 || group > 30) {
    return 0;
  } else {
    return m->opt.disableactuator & (1 << group) ? 1 : 0;
  }
}


// returns the next activation given current act_dot, after clamping
mjtNum mj_nextActivation(const mjModel* m, const mjData* d,
                         int actuator_id, int act_adr, mjtNum act_dot) {
  mjtNum act = d->act[act_adr];
  int dyntype = m->actuator_dyntype[actuator_id];

  if (dyntype == mjDYN_FILTEREXACT) {
    // exact filter integration
    // act_dot(0) = (ctrl-act(0)) / tau
    // act(h) = act(0) + (ctrl-act(0)) (1 - exp(-h / tau))
    //        = act(0) + act_dot(0) * tau * (1 - exp(-h / tau))
    mjtNum tau = mju_max(mjMINVAL, m->actuator_dynprm[actuator_id*mjNDYN]);
    act = act + act_dot * tau * (1 - mju_exp(-m->opt.timestep / tau));
  } else if (dyntype == mjDYN_DCMOTOR) {
    const mjtNum* dynprm = m->actuator_dynprm + actuator_id * mjNDYN;
    const mjtNum* gainprm = m->actuator_gainprm + actuator_id * mjNGAIN;
    mjDCMotorSlots slots = mj_dcmotorSlots(dynprm, gainprm);

    int offset = act_adr - m->actuator_actadr[actuator_id];

    // current filter: exact integration
    if (offset == slots.current) {
      mjtNum te = mju_max(mjMINVAL, dynprm[0]);
      act = act + act_dot * te * (1 - mju_exp(-m->opt.timestep / te));
    }

    // LuGre bristle:  dz/dt = a*z + v  where a = -sigma0*|v|/g(v)
    else if (offset == slots.bristle) {
      const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*actuator_id;
      mjtNum F_C = biasprm[3];    // Coulomb friction
      mjtNum F_S = biasprm[4];    // static friction
      mjtNum v_S = biasprm[5];    // Stribeck velocity
      mjtNum sigma0 = dynprm[5];  // bristle stiffness
      mjtNum velocity = d->actuator_velocity[m->actuator_outadr[actuator_id]];
      mjtNum g = mj_lugreStribeck(velocity, F_C, F_S, v_S);

      // ZOH exact ZOH integration: z(h) = exp(ah)*z(0) + ((exp(ah)-1)/a)*v
      mjtNum a = -sigma0 * mju_abs(velocity) / mju_max(mjMINVAL, g);  // decay rate
      mjtNum h = m->opt.timestep;
      mjtNum exp_ah = mju_exp(a * h);                                 // state transition
      mjtNum int_h = mju_abs(a) > mjMINVAL ? (exp_ah - 1) / a : h;    // input integral
      act = exp_ah * act + int_h * velocity;
    }

    // integral state: Euler integration with anti-windup clamp
    else if (offset == slots.integral) {
      act = act + act_dot * m->opt.timestep;
      mjtNum Imax = dynprm[8];
      if (Imax > 0) {
        act = mju_clip(act, -Imax, Imax);
      }
    }

    // temperature and slew: Euler integration
    else {
      act = act + act_dot * m->opt.timestep;
    }
  }

  // otherwise Euler integration
  else {
    act = act + act_dot * m->opt.timestep;
  }

  // clamp to actrange unless DC motor
  if (dyntype != mjDYN_DCMOTOR && m->actuator_actlimited[actuator_id]) {
    const mjtNum* actrange = m->actuator_actrange + 2*actuator_id;
    act = mju_clip(act, actrange[0], actrange[1]);
  }

  return act;
}


// sum all body masses
mjtNum mj_getTotalmass(const mjModel* m) {
  mjtNum res = 0;

  for (int i=1; i < m->nbody; i++) {
    res += m->body_mass[i];
  }

  return res;
}


// scale all body masses and inertias to achieve specified total mass
void mj_setTotalmass(mjModel* m, mjtNum newmass) {
  // compute scale factor, avoid zeros
  mjtNum scale = mju_max(mjMINVAL, newmass / mju_max(mjMINVAL, mj_getTotalmass(m)));

  // scale all masses and inertias
  for (int i=1; i < m->nbody; i++) {
    m->body_mass[i] *= scale;
    m->body_inertia[3*i] *= scale;
    m->body_inertia[3*i+1] *= scale;
    m->body_inertia[3*i+2] *= scale;
  }

  // don't forget to call mj_set0 after changing masses
}


// version number
int mj_version(void) {
  return mjVERSION;
}


// current version of MuJoCo as a null-terminated string
const char* mj_versionString(void) {
  static const char versionstring[] = mjVERSIONSTRING;
  return versionstring;
}


// return total size of data in a contact sensor bitfield specification
int mju_condataSize(int dataspec) {
  int size = 0;
  for (int i=0; i < mjNCONDATA; i++) {
    if (dataspec & (1 << i)) {
      size += mjCONDATA_SIZE[i];
    }
  }
  return size;
}


// return total size of data in a rangefinder sensor bitfield specification
int mju_raydataSize(int dataspec) {
  int size = 0;
  for (int i=0; i < mjNRAYDATA; i++) {
    if (dataspec & (1 << i)) {
      size += mjRAYDATA_SIZE[i];
    }
  }
  return size;
}


// compute camera pixel parameters from model, output are:
//   pixel units: fx, fy (focal lengths), cx, cy (principal point)
//   length units: extent
void mju_camIntrinsics(const mjModel* m, int camid,
                       mjtNum* fx, mjtNum* fy, mjtNum* cx, mjtNum* cy, mjtNum* extent) {
  const int width = m->cam_resolution[2*camid];
  const int height = m->cam_resolution[2*camid+1];
  const float* sensorsize = m->cam_sensorsize + 2*camid;
  const float* intrinsic = m->cam_intrinsic + 4*camid;
  const mjtProjection projection = (mjtProjection)m->cam_projection[camid];

  switch (projection) {
  case mjPROJ_PERSPECTIVE:
    if (sensorsize[0] && sensorsize[1]) {
      // intrinsic-based perspective camera
      *fx = intrinsic[0] / sensorsize[0] * width;
      *fy = intrinsic[1] / sensorsize[1] * height;
      *cx = intrinsic[2] / sensorsize[0] * width;
      *cy = intrinsic[3] / sensorsize[1] * height;
    } else {
      // fovy-based perspective camera
      *fx = *fy = 0.5 / mju_tan(m->cam_fovy[camid] * mjPI / 360.0) * height;
      *cx = (mjtNum)width / 2.0;
      *cy = (mjtNum)height / 2.0;
    }
    break;
  case mjPROJ_ORTHOGRAPHIC:
    // orthographic: normalize pixel offset to [-1, 1]
    *fx = (mjtNum)width / 2.0;
    *fy = (mjtNum)height / 2.0;
    *cx = *fx;
    *cy = *fy;
    break;
  }

  // extent only used for orthographic cameras
  *extent = m->cam_fovy[camid];
}


// read delayed ctrl value for actuator at given time
const mjtNum* mj_readCtrl(const mjModel* m, const mjData* d, int id, mjtNum time,
                          mjtNum* result, int interp) {
  // validate actuator id
  if (id < 0 || id >= m->nactuator) {
    mjERROR("invalid actuator id %d", id);
    return NULL;
  }

  // no delay: return current ctrl value
  int nsample = m->actuator_history[2*id];
  if (nsample == 0) {
    return d->ctrl + m->actuator_ctrladr[id];
  }

  // resolve interpolation order: use model's interp if argument is -1
  if (interp < 0) interp = m->actuator_history[2*id+1];

  // get buffer pointer and read from history buffer
  int dim = m->actuator_ctrlnum[id];
  mjtNum delay = m->actuator_delay[id];
  const mjtNum* buf = d->history + m->actuator_historyadr[id];
  return mju_historyRead(buf, nsample, dim, result, time - delay, interp);
}


// read sensor value from history buffer at given time
const mjtNum* mj_readSensor(const mjModel* m, const mjData* d, int id, mjtNum time,
                            mjtNum* result, int interp) {
  // validate sensor id
  if (id < 0 || id >= m->nsensor) {
    mjERROR("invalid sensor id %d", id);
    return NULL;
  }

  // no history: return current sensor value
  int nsample = m->sensor_history[2*id];
  if (nsample == 0) {
    return d->sensordata + m->sensor_adr[id];
  }

  // resolve interpolation order: use model's interp if argument is -1
  if (interp < 0) interp = m->sensor_history[2*id+1];

  // get buffer pointer and read from history buffer
  int dim = m->sensor_dim[id];
  mjtNum delay = m->sensor_delay[id];
  const mjtNum* buf = d->history + m->sensor_historyadr[id];
  return mju_historyRead(buf, nsample, dim, result, time - delay, interp);
}


// initialize history buffer for actuator
void mj_initCtrlHistory(const mjModel* m, mjData* d, int id,
                        const mjtNum* times, const mjtNum* values) {
  // validate actuator id
  if (id < 0 || id >= m->nactuator) {
    mjERROR("invalid actuator id %d", id);
    return;
  }

  // check that actuator has a history buffer
  int nsample = m->actuator_history[2*id];
  if (nsample == 0) {
    mjERROR("actuator %d has no history buffer", id);
    return;
  }

  // get buffer pointer and dimension
  mjtNum* buf = d->history + m->actuator_historyadr[id];
  int dim = m->actuator_ctrlnum[id];

  // if times is NULL, use existing buffer times
  const mjtNum* buf_times = times ? times : buf + 2;

  // get existing user value (preserve it)
  mjtNum user = buf[0];

  // initialize history buffer
  mju_historyInit(buf, nsample, dim, buf_times, values, user);
}


// initialize history buffer for sensor
void mj_initSensorHistory(const mjModel* m, mjData* d, int id,
                          const mjtNum* times, const mjtNum* values, mjtNum phase) {
  // validate sensor id
  if (id < 0 || id >= m->nsensor) {
    mjERROR("invalid sensor id %d", id);
    return;
  }

  // check that sensor has a history buffer
  int nsample = m->sensor_history[2*id];
  if (nsample == 0) {
    mjERROR("sensor %d has no history buffer", id);
    return;
  }

  // get buffer pointer and dimension
  mjtNum* buf = d->history + m->sensor_historyadr[id];
  int dim = m->sensor_dim[id];

  // if times is NULL, use existing buffer times
  const mjtNum* buf_times = times ? times : buf + 2;

  // initialize history buffer with provided phase
  mju_historyInit(buf, nsample, dim, buf_times, values, phase);
}
