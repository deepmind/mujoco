// Copyright 2022 DeepMind Technologies Limited
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

#include "engine/engine_derivative.h"

#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjtype.h>
#include "engine/engine_core_util.h"
#include "engine/engine_crossplatform.h"
#include "engine/engine_inline.h"
#include "engine/engine_memory.h"
#include "engine/engine_passive.h"
#include "engine/engine_sleep.h"
#include "engine/engine_support.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"
#include "engine/engine_util_sparse.h"


//------------------------- derivatives of spatial algebra -----------------------------------------


// derivatives of cross product, Da and Db are 3x3
static void mjd_cross(const mjtNum a[3], const mjtNum b[3],
                      mjtNum* restrict Da, mjtNum* restrict Db) {
  // derivative w.r.t a
  if (Da) {
    mju_zero(Da, 9);
    Da[1] =  b[2];
    Da[2] = -b[1];
    Da[3] = -b[2];
    Da[5] =  b[0];
    Da[6] =  b[1];
    Da[7] = -b[0];
  }

  // derivative w.r.t b
  if (Db) {
    mju_zero(Db, 9);
    Db[1] = -a[2];
    Db[2] =  a[1];
    Db[3] =  a[2];
    Db[5] = -a[0];
    Db[6] = -a[1];
    Db[7] =  a[0];
  }
}


// derivative of mju_crossMotion w.r.t velocity
static void mjd_crossMotion_vel(mjtNum D[36], const mjtNum v[6]) {
  mju_zero(D, 36);

  // res[0] = -vel[2]*v[1] + vel[1]*v[2]
  D[0 + 2] = -v[1];
  D[0 + 1] = v[2];

  // res[1] =  vel[2]*v[0] - vel[0]*v[2]
  D[6 + 2] = v[0];
  D[6 + 0] = -v[2];

  // res[2] = -vel[1]*v[0] + vel[0]*v[1]
  D[12 + 1] = -v[0];
  D[12 + 0] = v[1];

  // res[3] = -vel[2]*v[4] + vel[1]*v[5] - vel[5]*v[1] + vel[4]*v[2]
  D[18 + 2] = -v[4];
  D[18 + 1] = v[5];
  D[18 + 5] = -v[1];
  D[18 + 4] = v[2];

  // res[4] =  vel[2]*v[3] - vel[0]*v[5] + vel[5]*v[0] - vel[3]*v[2]
  D[24 + 2] = v[3];
  D[24 + 0] = -v[5];
  D[24 + 5] = v[0];
  D[24 + 3] = -v[2];

  // res[5] = -vel[1]*v[3] + vel[0]*v[4] - vel[4]*v[0] + vel[3]*v[1]
  D[30 + 1] = -v[3];
  D[30 + 0] = v[4];
  D[30 + 4] = -v[0];
  D[30 + 3] = v[1];
}


// derivative of mju_crossForce w.r.t. velocity
static void mjd_crossForce_vel(mjtNum D[36], const mjtNum f[6]) {
  mju_zero(D, 36);

  // res[0] = -vel[2]*f[1] + vel[1]*f[2] - vel[5]*f[4] + vel[4]*f[5]
  D[0 + 2] = -f[1];
  D[0 + 1] = f[2];
  D[0 + 5] = -f[4];
  D[0 + 4] = f[5];

  // res[1] =  vel[2]*f[0] - vel[0]*f[2] + vel[5]*f[3] - vel[3]*f[5]
  D[6 + 2] = f[0];
  D[6 + 0] = -f[2];
  D[6 + 5] = f[3];
  D[6 + 3] = -f[5];

  // res[2] = -vel[1]*f[0] + vel[0]*f[1] - vel[4]*f[3] + vel[3]*f[4]
  D[12 + 1] = -f[0];
  D[12 + 0] = f[1];
  D[12 + 4] = -f[3];
  D[12 + 3] = f[4];

  // res[3] = -vel[2]*f[4] + vel[1]*f[5]
  D[18 + 2] = -f[4];
  D[18 + 1] = f[5];

  // res[4] =  vel[2]*f[3] - vel[0]*f[5]
  D[24 + 2] = f[3];
  D[24 + 0] = -f[5];

  // res[5] = -vel[1]*f[3] + vel[0]*f[4]
  D[30 + 1] = -f[3];
  D[30 + 0] = f[4];
}


// derivative of mju_crossForce w.r.t. force
static void mjd_crossForce_frc(mjtNum D[36], const mjtNum vel[6]) {
  mju_zero(D, 36);

  // res[0] = -vel[2]*f[1] + vel[1]*f[2] - vel[5]*f[4] + vel[4]*f[5]
  D[0 + 1] = -vel[2];
  D[0 + 2] = vel[1];
  D[0 + 4] = -vel[5];
  D[0 + 5] = vel[4];

  // res[1] =  vel[2]*f[0] - vel[0]*f[2] + vel[5]*f[3] - vel[3]*f[5]
  D[6 + 0] = vel[2];
  D[6 + 2] = -vel[0];
  D[6 + 3] = vel[5];
  D[6 + 5] = -vel[3];

  // res[2] = -vel[1]*f[0] + vel[0]*f[1] - vel[4]*f[3] + vel[3]*f[4]
  D[12 + 0] = -vel[1];
  D[12 + 1] = vel[0];
  D[12 + 3] = -vel[4];
  D[12 + 4] = vel[3];

  // res[3] = -vel[2]*f[4] + vel[1]*f[5]
  D[18 + 4] = -vel[2];
  D[18 + 5] = vel[1];

  // res[4] =  vel[2]*f[3] - vel[0]*f[5]
  D[24 + 3] = vel[2];
  D[24 + 5] = -vel[0];

  // res[5] = -vel[1]*f[3] + vel[0]*f[4]
  D[30 + 3] = -vel[1];
  D[30 + 4] = vel[0];
}


// derivative of mju_mulInertVec w.r.t vel
static void mjd_mulInertVec_vel(mjtNum D[36], const mjtNum i[10]) {
  mju_zero(D, 36);

  // res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5]
  D[0 + 0] = i[0];
  D[0 + 1] = i[3];
  D[0 + 2] = i[4];
  D[0 + 4] = -i[8];
  D[0 + 5] = i[7];

  // res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5]
  D[6 + 0] = i[3];
  D[6 + 1] = i[1];
  D[6 + 2] = i[5];
  D[6 + 3] = i[8];
  D[6 + 5] = -i[6];

  // res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4]
  D[12 + 0] = i[4];
  D[12 + 1] = i[5];
  D[12 + 2] = i[2];
  D[12 + 3] = -i[7];
  D[12 + 4] = i[6];

  // res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3]
  D[18 + 1] = i[8];
  D[18 + 2] = -i[7];
  D[18 + 3] = i[9];

  // res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4]
  D[24 + 2] = i[6];
  D[24 + 0] = -i[8];
  D[24 + 4] = i[9];

  // res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5]
  D[30 + 0] = i[7];
  D[30 + 1] = -i[6];
  D[30 + 5] = i[9];
}


// derivative of mju_subQuat w.r.t inputs
void mjd_subQuat(const mjtNum qa[4], const mjtNum qb[4], mjtNum Da[9], mjtNum Db[9]) {
  // no outputs, quick return
  if (!Da && !Db) {
    return;
  }

  // compute axis-angle quaternion difference
  mjtNum axis[3];
  mju_subQuat(axis, qa, qb);

  // normalize axis, get half-angle
  mjtNum half_angle = 0.5 * mju_normalize3(axis);

  // identity
  mjtNum Da_tmp[9] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  };

  // add term linear in cross product matrix K
  mjtNum K[9] = {
    0, -axis[2], axis[1],
    axis[2], 0, -axis[0],
    -axis[1], axis[0], 0
  };
  mju_addToScl(Da_tmp, K, half_angle, 9);

  // add term linear in K * K
  mjtNum KK[9];
  mji_mulMatMat3(KK, K, K);
  mjtNum coef = 1.0 - (half_angle < 6e-8 ? 1.0 : half_angle / mju_tan(half_angle));
  mju_addToScl(Da_tmp, KK, coef, 9);

  if (Da) {
    mju_copy9(Da, Da_tmp);
  }

  if (Db) {  // Db = -Da^T
    mju_transpose(Db, Da_tmp, 3, 3);
    mju_scl(Db, Db, -1.0, 9);
  }
}


// derivative of mju_quatIntegrate w.r.t scaled velocity
//  reference: https://arxiv.org/abs/1711.02508, Eq. 183
void mjd_quatIntegrate(const mjtNum vel[3], mjtNum scale,
                       mjtNum Dquat[9], mjtNum Dvel[9], mjtNum Dscale[3]) {
  // scaled velocity
  mjtNum s[3] = {scale*vel[0], scale*vel[1], scale*vel[2]};

  // 3 basis matrices
  mjtNum eye[9] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  };
  mjtNum cross[9] = {
    0,     s[2], -s[1],
   -s[2],  0,     s[0],
    s[1], -s[0],  0
  };
  mjtNum outer[9] = {
    s[0]*s[0], s[0]*s[1], s[0]*s[2],
    s[1]*s[0], s[1]*s[1], s[1]*s[2],
    s[2]*s[0], s[2]*s[1], s[2]*s[2]
  };

  // squared norm, norm of s
  mjtNum xx = mju_dot3(s, s);
  mjtNum x = mju_sqrt(xx);

  // 4 coefficients: a=cos(x), b=sin(x)/x, c=(1-cos(x))/x^2, d=(x-sin(x))/x^3
  mjtNum a = mju_cos(x);
  mjtNum b, c, d;

  // x is not small: use full expressions
  if (mju_abs(x) > 1.0/32) {
    b = mju_sin(x) / x;
    c = (1.0 - a) / xx;
    d = (1.0 - b) / xx;
  }

  // |x| <= 1/32: use 6th order Taylor expansion (Horner form)
  else {
    b =  1 + xx/6  * (xx/20 * (1 - xx/42) - 1);
    c = (1 + xx/12 * (xx/30 * (1 - xx/56) - 1)) / 2;
    d = (1 + xx/20 * (xx/42 * (1 - xx/72) - 1)) / 6;
  }

  // derivatives
  mjtNum Dvel_[9];
  for (int i=0; i < 9; i++) {
    if (Dquat)          Dquat[i] = a*eye[i] + b*cross[i] + c*outer[i];
    if (Dvel || Dscale) Dvel_[i] = b*eye[i] + c*cross[i] + d*outer[i];
  }
  if (Dvel) mju_copy9(Dvel, Dvel_);
  if (Dscale) mji_mulMatVec3(Dscale, Dvel_, vel);
}


//------------------------- dense derivatives of component functions -------------------------------
// no longer used, except in tests

// derivative of cvel, cdof_dot w.r.t qvel (dense version)
static void mjd_comVel_vel_dense(const mjModel* m, mjData* d, mjtNum* Dcvel, mjtNum* Dcdofdot) {
  int nv = m->nv, nbody = m->nbody;
  mjtNum mat[36];

  // clear Dcvel
  mju_zero(Dcvel, nbody*6*nv);

  // forward pass over bodies: accumulate Dcvel, set Dcdofdot
  for (int i=1; i < nbody; i++) {
    // Dcvel = Dcvel_parent
    mju_copy(Dcvel+i*6*nv, Dcvel+m->body_parentid[i]*6*nv, 6*nv);

    // Dcvel += D(cdof * qvel),  Dcdofdot = D(cvel x cdof)
    for (int j=m->body_dofadr[i]; j < m->body_dofadr[i]+m->body_dofnum[i]; j++) {
      switch ((mjtJoint) m->jnt_type[m->dof_jntid[j]]) {
      case mjJNT_FREE:
        // Dcdofdot = 0
        mju_zero(Dcdofdot+j*6*nv, 18*nv);

        // Dcvel += cdof * (D qvel)
        for (int k=0; k < 6; k++) {
          Dcvel[i*6*nv + k*nv + j+0] += d->cdof[(j+0)*6 + k];
          Dcvel[i*6*nv + k*nv + j+1] += d->cdof[(j+1)*6 + k];
          Dcvel[i*6*nv + k*nv + j+2] += d->cdof[(j+2)*6 + k];
        }

        // continue with rotations
        j += 3;
        mjFALLTHROUGH;

      case mjJNT_BALL:
        // Dcdofdot = D crossMotion(cvel, cdof)
        for (int k=0; k < 3; k++) {
          mjd_crossMotion_vel(mat, d->cdof+6*(j+k));
          mju_mulMatMat(Dcdofdot+(j+k)*6*nv, mat, Dcvel+i*6*nv, 6, 6, nv);
        }

        // Dcvel += cdof * (D qvel)
        for (int k=0; k < 6; k++) {
          Dcvel[i*6*nv + k*nv + j+0] += d->cdof[(j+0)*6 + k];
          Dcvel[i*6*nv + k*nv + j+1] += d->cdof[(j+1)*6 + k];
          Dcvel[i*6*nv + k*nv + j+2] += d->cdof[(j+2)*6 + k];
        }

        // adjust for 3-dof joint
        j += 2;
        break;

      default:
        // Dcdofdot = D crossMotion(cvel, cdof) * Dcvel
        mjd_crossMotion_vel(mat, d->cdof+6*j);
        mju_mulMatMat(Dcdofdot+j*6*nv, mat, Dcvel+i*6*nv, 6, 6, nv);

        // Dcvel += cdof * (D qvel)
        for (int k=0; k < 6; k++) {
          Dcvel[i*6*nv + k*nv + j] += d->cdof[j*6 + k];
        }
      }
    }
  }
}


// subtract (d qfrc_bias / d qvel) from qDeriv (dense version)
void mjd_rne_vel_dense(const mjModel* m, mjData* d) {
  int nv = m->nv, nbody = m->nbody;
  mjtNum mat[36], mat1[36], mat2[36], dmul[36], tmp[6];

  mj_markStack(d);
  mjtNum* Dcvel = mjSTACKALLOC(d, nbody*6*nv, mjtNum);
  mjtNum* Dcdofdot = mjSTACKALLOC(d, nv*6*nv, mjtNum);
  mjtNum* Dcacc = mjSTACKALLOC(d, nbody*6*nv, mjtNum);
  mjtNum* Dcfrcbody = mjSTACKALLOC(d, nbody*6*nv, mjtNum);
  mjtNum* row = mjSTACKALLOC(d, nv, mjtNum);

  // compute Dcvel and Dcdofdot
  mjd_comVel_vel_dense(m, d, Dcvel, Dcdofdot);

  // clear Dcacc
  mju_zero(Dcacc, nbody*6*nv);

  // forward pass over bodies: accumulate Dcacc, set Dcfrcbody
  for (int i=1; i < nbody; i++) {
    // Dcacc = Dcacc_parent
    mju_copy(Dcacc + i*6*nv, Dcacc + m->body_parentid[i]*6*nv, 6*nv);

    // Dcacc += D(cdofdot * qvel)
    for (int j=m->body_dofadr[i]; j < m->body_dofadr[i]+m->body_dofnum[i]; j++) {
      // Dcacc += cdofdot * (D qvel)
      for (int k=0; k < 6; k++) {
        Dcacc[i*6*nv + k*nv + j] += d->cdof_dot[j*6 + k];
      }

      // Dcacc += (D cdofdot) * qvel
      mju_addToScl(Dcacc+i*6*nv, Dcdofdot+j*6*nv, d->qvel[j], 6*nv);
    }

    //---------- Dcfrcbody = D(cinert * cacc + cvel x (cinert * cvel))

    // Dcfrcbody = (D mul / D cacc) * Dcacc
    mjd_mulInertVec_vel(dmul, d->cinert+10*i);
    mju_mulMatMat(Dcfrcbody+i*6*nv, dmul, Dcacc+i*6*nv, 6, 6, nv);

    // mat = (D cross / D cvel) + (D cross / D mul) * (D mul / D cvel)
    mju_mulInertVec(tmp, d->cinert+10*i, d->cvel+i*6);
    mjd_crossForce_vel(mat, tmp);
    mjd_crossForce_frc(mat1, d->cvel+i*6);
    mju_mulMatMat(mat2, mat1, dmul, 6, 6, 6);
    mju_addTo(mat, mat2, 36);

    // Dcfrcbody += mat * Dcvel  (use body 0 as temp)
    mju_mulMatMat(Dcfrcbody, mat, Dcvel+i*6*nv, 6, 6, nv);
    mju_addTo(Dcfrcbody+i*6*nv, Dcfrcbody, 6*nv);
  }

  // clear world Dcfrcbody, for style
  mju_zero(Dcfrcbody, 6*nv);

  // backward pass over bodies: accumulate Dcfrcbody
  for (int i=nbody-1; i > 0; i--) {
    if (m->body_parentid[i]) {
      mju_addTo(Dcfrcbody+m->body_parentid[i]*6*nv, Dcfrcbody+i*6*nv, 6*nv);
    }
  }

  // qDeriv -= D(cdof * cfrc_body)
  for (int i=0; i < nv; i++) {
    for (int k=0; k < 6; k++) {
      // compute D(cdof * cfrc_body), store in row
      mju_scl(row, Dcfrcbody + (m->dof_bodyid[i]*6+k)*nv, d->cdof[i*6+k], nv);

      // dense to sparse: qDeriv -= row
      int end = m->D_rowadr[i] + m->D_rownnz[i];
      for (int adr=m->D_rowadr[i]; adr < end; adr++) {
        d->qDeriv[adr] -= row[m->D_colind[adr]];
      }
    }
  }

  mj_freeStack(d);
}


//------------------------- sparse derivatives of component functions ------------------------------
// internal sparse format: dense body/dof x sparse dof x 6   (inner size is 6)

// copy sparse B-row from parent, shared ancestors only
static void copyFromParent(const mjModel* m, mjData* d, mjtNum* mat, int n) {
  // return if this is world or parent is world
  if (n == 0 || m->body_weldid[m->body_parentid[n]] == 0) {
    return;
  }

  // count dofs in ancestors
  int ndof = 0;
  int np = m->body_weldid[m->body_parentid[n]];
  while (np > 0) {
    // add self dofs
    ndof += m->body_dofnum[np];

    // advance to parent
    np = m->body_weldid[m->body_parentid[np]];
  }

  // copy: guaranteed to be at beginning of sparse array, due to sorting
  mju_copy(mat + 6*m->B_rowadr[n], mat + 6*m->B_rowadr[m->body_parentid[n]], 6*ndof);
}


// add sparse B-row to parent, all overlapping nonzeros
static void addToParent(const mjModel* m, mjData* d, mjtNum* mat, int n) {
  // return if this is world or parent is world
  if (n == 0 || m->body_weldid[m->body_parentid[n]] == 0) {
    return;
  }

  // find matching nonzeros
  int np = m->body_parentid[n];
  int i = 0, ip = 0;
  while (i < m->B_rownnz[n] && ip < m->B_rownnz[np]) {
    // columns match
    if (m->B_colind[m->B_rowadr[n] + i] == m->B_colind[m->B_rowadr[np] + ip]) {
      mju_addTo(mat + 6*(m->B_rowadr[np] + ip), mat + 6*(m->B_rowadr[n] + i), 6);

      // advance both
      i++;
      ip++;
    }

    // mismatch columns: advance parent
    else if (m->B_colind[m->B_rowadr[n] + i] > m->B_colind[m->B_rowadr[np] + ip]) {
      ip++;
    }

    // child nonzeroes must be subset of parent; SHOULD NOT OCCUR
    else {
      mjERROR("child nonzeroes must be subset of parent");
    }
  }
}


// derivative of cvel, cdof_dot w.r.t qvel
static void mjd_comVel_vel(const mjModel* m, mjData* d, mjtNum* Dcvel, mjtNum* Dcdofdot) {
  int nv = m->nv, nM = m->nM;
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;
  int* Badr = m->B_rowadr, * Dadr = m->D_rowadr;
  mjtNum mat[36], matT[36];   // 6x6 matrices

  // forward pass over bodies: accumulate Dcvel, set Dcdofdot
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // Dcvel = Dcvel_parent
    copyFromParent(m, d, Dcvel, i);

    // process all dofs of this body
    int doflast = m->body_dofadr[i] + m->body_dofnum[i];
    for (int j = m->body_dofadr[i]; j < doflast; j++) {
      // number of dof ancestors of dof j
      int Jadr = (j < nv - 1 ? m->dof_Madr[j + 1] : nM) - (m->dof_Madr[j] + 1);

      // Dcvel += D(cdof * qvel),  Dcdofdot = D(cvel x cdof)
      switch ((mjtJoint) m->jnt_type[m->dof_jntid[j]]) {
      case mjJNT_FREE:
        // Dcdofdot = 0 (already cleared)

        // Dcvel += cdof * D(qvel)
        mju_addTo(Dcvel + 6*(Badr[i] + Jadr + 0), d->cdof + 6*(j + 0), 6);
        mju_addTo(Dcvel + 6*(Badr[i] + Jadr + 1), d->cdof + 6*(j + 1), 6);
        mju_addTo(Dcvel + 6*(Badr[i] + Jadr + 2), d->cdof + 6*(j + 2), 6);

        // continue with rotations
        j += 3;
        Jadr += 3;
        mjFALLTHROUGH;

      case mjJNT_BALL:
        // Dcdofdot = Dcvel * D crossMotion(cvel, cdof)
        for (int dj=0; dj < 3; dj++) {
          mjd_crossMotion_vel(mat, d->cdof + 6 * (j + dj));
          mju_transpose(matT, mat, 6, 6);
          mju_mulMatMat(Dcdofdot + 6*Dadr[j + dj], Dcvel + 6*Badr[i], matT, Jadr + dj, 6, 6);
        }

        // Dcvel += cdof * (D qvel)
        mju_addTo(Dcvel + 6*(Badr[i] + Jadr + 0), d->cdof + 6*(j + 0), 6);
        mju_addTo(Dcvel + 6*(Badr[i] + Jadr + 1), d->cdof + 6*(j + 1), 6);
        mju_addTo(Dcvel + 6*(Badr[i] + Jadr + 2), d->cdof + 6*(j + 2), 6);

        // adjust for 3-dof joint
        j += 2;
        break;

      case mjJNT_HINGE:
      case mjJNT_SLIDE:
        // Dcdofdot = D crossMotion(cvel, cdof) * Dcvel
        mjd_crossMotion_vel(mat, d->cdof + 6 * j);
        mju_transpose(matT, mat, 6, 6);
        mju_mulMatMat(Dcdofdot + 6*Dadr[j], Dcvel + 6*Badr[i], matT, Jadr, 6, 6);

        // Dcvel += cdof * (D qvel)
        mju_addTo(Dcvel + 6*(Badr[i] + Jadr), d->cdof + 6*j, 6);
        break;

      default:
        mjERROR("unknown joint type");
      }
    }
  }
}


// subtract d qfrc_bias / d qvel from qDeriv
static void mjd_rne_vel(const mjModel* m, mjData* d) {
  int nM = m->nM;
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;
  int nparent = sleep_filter ? d->nparent_awake : m->nbody;
  int mnv = m->nv;
  int nv = sleep_filter ? d->nv_awake : mnv;

  const int* Badr = m->B_rowadr;
  const int* Dadr = m->D_rowadr;
  const int* Bnnz = m->B_rownnz;

  mjtNum mat[36], mat1[36], mat2[36], dmul[36], tmp[6];

  mj_markStack(d);
  mjtNum* Dcdofdot = mjSTACKALLOC(d, 6*m->nD, mjtNum);
  mjtNum* Dcvel = mjSTACKALLOC(d, 6*m->nB, mjtNum);
  mjtNum* Dcacc = mjSTACKALLOC(d, 6*m->nB, mjtNum);
  mjtNum* Dcfrcbody = mjSTACKALLOC(d, 6*m->nB, mjtNum);
  mjtNum* row = mjSTACKALLOC(d, m->nv, mjtNum);

  // clear
  if (!sleep_filter) {
    mju_zero(Dcdofdot,  6*m->nD);
    mju_zero(Dcvel,     6*m->nB);
    mju_zero(Dcacc,     6*m->nB);
    mju_zero(Dcfrcbody, 6*m->nB);
  } else {
    for (int i = 0; i < nv; i++) {
      int dof = d->dof_awake_ind[i];
      mju_zero(Dcdofdot + 6*m->D_rowadr[dof], 6*m->D_rownnz[dof]);
    }

    for (int i = 0; i < nbody; i++) {
      int body = d->body_awake_ind[i];
      int adr = 6*m->B_rowadr[body];
      int nnz = 6*m->B_rownnz[body];
      mju_zero(Dcvel     + adr, nnz);
      mju_zero(Dcacc     + adr, nnz);
      mju_zero(Dcfrcbody + adr, nnz);
    }
  }

  // compute Dcvel and Dcdofdot
  mjd_comVel_vel(m, d, Dcvel, Dcdofdot);

  // forward pass over bodies: accumulate Dcacc, set Dcfrcbody
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // Dcacc = Dcacc_parent
    copyFromParent(m, d, Dcacc, i);

    // process all dofs of this body
    int doflast = m->body_dofadr[i] + m->body_dofnum[i];
    for (int j=m->body_dofadr[i]; j < doflast; j++) {
      // number of dof ancestors of dof j
      int Jadr = (j < mnv - 1 ? m->dof_Madr[j + 1] : nM) - (m->dof_Madr[j] + 1);

      // Dcacc += cdofdot * (D qvel)
      mju_addTo(Dcacc + 6*(Badr[i] + Jadr), d->cdof_dot + 6*j, 6);

      // Dcacc += (D cdofdot) * qvel
      // Dcacc[row i] and Dcdofdot[row j] have identical sparsity
      mju_addToScl(Dcacc + 6*Badr[i], Dcdofdot + 6*Dadr[j], d->qvel[j], 6*Bnnz[i]);
    }

    //---------- Dcfrcbody = D(cinert * cacc + cvel x (cinert * cvel))

    // Dcfrcbody = (D mul / D cacc) * Dcacc
    mjd_mulInertVec_vel(dmul, d->cinert + 10*i);
    mju_transpose(mat1, dmul, 6, 6);
    mju_mulMatMat(Dcfrcbody + 6*Badr[i], Dcacc + 6*Badr[i], mat1, Bnnz[i], 6, 6);

    // mat = (D cross / D cvel) + (D cross / D mul) * (D mul / D cvel)
    mju_mulInertVec(tmp, d->cinert + 10*i, d->cvel + i*6);
    mjd_crossForce_vel(mat, tmp);
    mjd_crossForce_frc(mat1, d->cvel + i*6);
    mju_mulMatMat(mat2, mat1, dmul, 6, 6, 6);
    mju_addTo(mat, mat2, 36);

    // Dcfrcbody += mat * Dcvel  (use worldbody as temp)
    mju_transpose(mat1, mat, 6, 6);
    mju_mulMatMat(Dcfrcbody, Dcvel + 6*Badr[i], mat1, Bnnz[i], 6, 6);
    mju_addTo(Dcfrcbody + 6*Badr[i], Dcfrcbody, 6*Bnnz[i]);
  }

  // clear worldbody Dcfrcbody
  mju_zero(Dcfrcbody, 6*Bnnz[0]);

  // backward pass over bodies: accumulate Dcfrcbody
  for (int b=nparent-1; b > 0; b--) {
    int i = sleep_filter ? d->parent_awake_ind[b] : b;
    addToParent(m, d, Dcfrcbody, i);
  }

  // process all dofs, update qDeriv
  for (int v=0; v < nv; v++) {
    int j = sleep_filter ? d->dof_awake_ind[v] : v;

    // get body index
    int i = m->dof_bodyid[j];

    // qDeriv -= D(cdof * cfrc_body)
    mju_mulMatVec(row, Dcfrcbody + 6*Badr[i], d->cdof + 6*j, Bnnz[i], 6);
    mju_subFrom(d->qDeriv + Dadr[j], row, Bnnz[i]);
  }

  mj_freeStack(d);
}


// 3x3 sub-blocks of (d qfrc_bias / d qvel) for a free rigid subtree
//   outputs the two 3x3 blocks lin and rot such that the rotational columns
//   of the full 6x6 bias Jacobian B are  [-mass*lin; rot]  (linear columns are zero)
//
// derivation: let R = xmat, s = subtree_com - xpos, w = R*qvel[rot], and Iw be the
// composite world inertia about the subtree CoM. with qacc = 0,
// the CoM acceleration is w x (w x s) and the world bias force/torque at the CoM are
//   f = mass * w x (w x s),   tau = w x Iw*w
// projected onto the joint coordinates: bias = [f;  R'*(s x f + tau)]. differentiating
// w.r.t. the rotational dofs (through w = R*qvel[rot]), with K = [w x s]_x + [w]_x [s]_x:
//   d f / d w   = -mass * K            =>  lin = K * R
//   d tau / d w = [w]_x Iw - [Iw*w]_x  =>  rot = R' * (-mass*[s]_x K + d tau/d w) * R
static void freeBias_vel_blocks(mjtNum mass, const mjtNum R[9], const mjtNum Iw[9],
                                const mjtNum s[3], const mjtNum qvel_rot[3],
                                mjtNum lin[9], mjtNum rot[9]) {
  // world-frame angular velocity
  mjtNum w[3];
  mji_mulMatVec3(w, R, qvel_rot);

  // intermediate vectors: ws = w x s  (CoM offset velocity),  Iww = Iw * w  (angular momentum)
  mjtNum ws[3], Iww[3];
  mji_cross(ws, w, s);
  mji_mulMatVec3(Iww, Iw, w);

  // K = [w x s]_x + [w]_x [s]_x = s w^T - (w . s) I + [ws]_x
  mjtNum w_dot_s = w[0]*s[0] + w[1]*s[1] + w[2]*s[2];
  mjtNum K[9];
  K[0] = s[0]*w[0] - w_dot_s;
  K[1] = s[0]*w[1] - ws[2];
  K[2] = s[0]*w[2] + ws[1];

  K[3] = s[1]*w[0] + ws[2];
  K[4] = s[1]*w[1] - w_dot_s;
  K[5] = s[1]*w[2] - ws[0];

  K[6] = s[2]*w[0] - ws[1];
  K[7] = s[2]*w[1] + ws[0];
  K[8] = s[2]*w[2] - w_dot_s;

  // lin = K * R
  mji_mulMatMat3(lin, K, R);

  // C = -mass * [s]_x K + [w]_x Iw - [Iww]_x,  column by column
  // the last term (-[Iww]_x) is the negated cross-product matrix, added via ternaries
  mjtNum C[9];
  for (int c=0; c < 3; c++) {
    mjtNum s_x_K_row0 = s[1]*K[6+c] - s[2]*K[3+c];
    mjtNum s_x_K_row1 = s[2]*K[c] - s[0]*K[6+c];
    mjtNum s_x_K_row2 = s[0]*K[3+c] - s[1]*K[c];

    mjtNum w_x_Iw_row0 = w[1]*Iw[6+c] - w[2]*Iw[3+c];
    mjtNum w_x_Iw_row1 = w[2]*Iw[c] - w[0]*Iw[6+c];
    mjtNum w_x_Iw_row2 = w[0]*Iw[3+c] - w[1]*Iw[c];

    C[c]     = -mass * s_x_K_row0 + w_x_Iw_row0 + (c == 1 ? Iww[2] : (c == 2 ? -Iww[1] : 0));
    C[3 + c] = -mass * s_x_K_row1 + w_x_Iw_row1 + (c == 0 ? -Iww[2] : (c == 2 ? Iww[0] : 0));
    C[6 + c] = -mass * s_x_K_row2 + w_x_Iw_row2 + (c == 0 ? Iww[1] : (c == 1 ? -Iww[0] : 0));
  }

  // rot = R^T * C * R
  mjtNum tmp[9];
  mji_mulMatTMat3(tmp, R, C);
  mji_mulMatMat3(rot, tmp, R);
}


// 6x6 block B = d qfrc_bias / d qvel for a free rigid subtree
//   assembles the full 6x6 from the 3x3 sub-blocks computed by freeBias_vel_blocks
//   rows/cols ordered like the free joint dofs: [linear(3); rotational(3)]
//   linear columns are zero: the bias force does not depend on linear velocity
//   requires valid d->crb, computed by mj_crb
void mjd_freeBias_vel(const mjModel* m, const mjData* d, int jnt, mjtNum B[36]) {
  int body = m->jnt_bodyid[jnt];
  int adr = m->jnt_dofadr[jnt];
  const mjtNum* crb = d->crb + 10*body;
  mjtNum mass = crb[9];

  // composite world inertia about the subtree CoM, already accumulated by mj_crb
  mjtNum Iw[9] = {crb[0], crb[3], crb[4],
                  crb[3], crb[1], crb[5],
                  crb[4], crb[5], crb[2]};
  mjtNum s[3], lin[9], rot[9];
  mji_sub3(s, d->subtree_com + 3*body, d->xpos + 3*body);
  freeBias_vel_blocks(mass, d->xmat + 9*body, Iw, s, d->qvel + adr + 3, lin, rot);

  mju_zero(B, 36);
  for (int r=0; r < 3; r++) {
    for (int c=0; c < 3; c++) {
      B[6*(r+0) + 3+c] = -mass * lin[3*r+c];
      B[6*(r+3) + 3+c] = rot[3*r+c];
    }
  }
}


// return 1 if body is the root of a free rigid subtree (single free joint, fixed descendants)
mjtBool mj_isFreeBody(const mjModel* m, int body) {
  // must have exactly one joint, of free type
  if (m->body_jntnum[body] != 1 || m->jnt_type[m->body_jntadr[body]] != mjJNT_FREE) {
    return false;
  }

  int adr = m->jnt_dofadr[m->body_jntadr[body]];

  // descendants must not add degrees of freedom
  return m->tree_dofnum[m->dof_treeid[adr]] == 6;
}


// 6x6 block A = M - h * (d qfrc_smooth / d qvel) for the free joint of a rigid subtree
//   returns 1 and writes A if jnt is the free joint of an awake rigid subtree, 0 otherwise
//   requires valid d->qDeriv rows for the block, computed with flg_bias = 0; the bias
//   derivative excluded from qDeriv is added here via mjd_freeBias_vel
int mjd_freeMhat(const mjModel* m, const mjData* d, int jnt, mjtNum h, mjtNum A[36],
                 int flg_discrete) {
  int body = m->jnt_bodyid[jnt];
  int adr = m->jnt_dofadr[jnt];

  // must be a free rigid subtree, awake
  if (!mj_isFreeBody(m, body) || !d->tree_awake[m->dof_treeid[adr]]) {
    return 0;
  }

  // A = M block (gather from sparse lower triangle)
  mju_zero(A, 36);
  for (int r=0; r < 6; r++) {
    int rowadr = m->M_rowadr[adr+r];
    int rownnz = m->M_rownnz[adr+r];
    for (int k=0; k < rownnz; k++) {
      int c = m->M_colind[rowadr+k] - adr;
      A[6*r+c] = A[6*c+r] = d->M[rowadr+k];
    }
  }

  // discrete: the metric block is M + diag + fluid, assembled from the efm arrays
  // (qDeriv is not computed under discrete)
  if (flg_discrete) {
    // per-dof diagonal metric terms
    if (d->efm_diag) {
      for (int r=0; r < 6; r++) {
        A[6*r+r] += d->efm_diag[adr+r];
      }
    }

    // fluid drag blocks in this joint's rows of M's pattern
    if (d->efm_fluid) {
      for (int r=0; r < 6; r++) {
        int rowadr = m->M_rowadr[adr+r];
        int rownnz = m->M_rownnz[adr+r];
        for (int k=0; k < rownnz; k++) {
          int c = m->M_colind[rowadr+k] - adr;
          A[6*r+c] += d->efm_fluid[rowadr+k];
          if (c != r) {
            A[6*c+r] += d->efm_fluid[rowadr+k];
          }
        }
      }
    }
  }

  // implicit-family: A -= h * qDeriv block (actuator and passive derivatives)
  else {
    for (int r=0; r < 6; r++) {
      int rowadr = m->D_rowadr[adr+r];
      int rownnz = m->D_rownnz[adr+r];
      for (int k=0; k < rownnz; k++) {
        int c = m->D_colind[rowadr+k] - adr;
        A[6*r+c] -= h * d->qDeriv[rowadr+k];
      }
    }
  }

  // A -= h * d(qfrc_smooth)/d(qvel) for the bias term missing from qDeriv;
  // qfrc_smooth includes -qfrc_bias, so subtracting its derivative adds +h*B
  mjtNum B[36];
  mjd_freeBias_vel(m, d, jnt, B);
  mju_addToScl(A, B, h, 36);

  return 1;
}


// can this free rigid subtree take the local gyroscopic treatment under discrete:
// its rows of the solve must be decoupled -- no constraint Jacobian support, no flex
// CSR row, no tendon or actuator metric terms on its 6 dofs. Structural: reads the
// constraint Jacobian, valid from mj_makeConstraint on
int mjd_freeGyroPossible(const mjModel* m, const mjData* d, int jnt) {
  int adr = m->jnt_dofadr[jnt];
  if (m->jnt_type[jnt] != mjJNT_FREE || !mj_isFreeBody(m, m->jnt_bodyid[jnt])) {
    return 0;
  }
  for (int r=0; r < 6; r++) {
    if (d->nefmK && d->efm_K_rownnz[adr+r]) {
      return 0;
    }
  }

  // Matrix-free flex attachments can couple this free body without any CSR or constraint row.
  int tree = m->body_treeid[m->jnt_bodyid[jnt]];
  for (int f=0; f < m->nflex; f++) {
    if (!mj_effFlexPossible(m, f)) continue;
    const int* bodies = m->flex_interp[f] ? m->flex_nodebodyid : m->flex_vertbodyid;
    int vertadr = m->flex_interp[f] ? m->flex_nodeadr[f] : m->flex_vertadr[f];
    int num = m->flex_interp[f] ? m->flex_nodenum[f] : m->flex_vertnum[f];
    for (int v=vertadr; v < vertadr+num; v++) {
      if (m->body_treeid[bodies[v]] == tree) return 0;
    }
  }

  // constraint Jacobian support on any of the 6 dofs couples them to the solve
  int nefc = d->nefc;
  if (mj_isSparse(m)) {
    for (int i=0; i < nefc; i++) {
      int end = d->efc_J_rowadr[i] + d->efc_J_rownnz[i];
      for (int j=d->efc_J_rowadr[i]; j < end; j++) {
        int c = d->efc_J_colind[j];
        if (c >= adr+6) {
          break;
        }
        if (c >= adr) {
          return 0;
        }
      }
    }
  } else {
    int nv = m->nv;
    for (int i=0; i < nefc; i++) {
      const mjtNum* row = d->efc_J + nv*i + adr;
      for (int r=0; r < 6; r++) {
        if (row[r]) {
          return 0;
        }
      }
    }
  }

  // any tendon with metric terms touching these dofs couples them to other trees
  for (int t=0; t < d->nefmT; t++) {
    if (!d->efm_ts[t]) {
      continue;
    }
    int i = d->efm_tid[t];
    int end = m->ten_J_rowadr[i] + m->ten_J_rownnz[i];
    for (int j=m->ten_J_rowadr[i]; j < end; j++) {
      int c = m->ten_J_colind[j];
      if (c >= adr && c < adr+6) {
        return 0;
      }
    }
  }
  for (int t=0; t < d->nefmA; t++) {
    int i = d->efm_aid[t];
    int oadr = m->actuator_outadr[i];
    for (int k=0; k < m->actuator_outnum[i]; k++) {
      int r = oadr + k;
      int end = d->moment_rowadr[r] + d->moment_rownnz[r];
      for (int j=d->moment_rowadr[r]; j < end; j++) {
        int c = d->moment_colind[j];
        if (c >= adr && c < adr+6) {
          return 0;
        }
      }
    }
  }
  return 1;
}


//--------------------- utility functions for (d force / d vel) Jacobians --------------------------

// add J'*B*J to res, in qDeriv's sparsity layout
static void addJTBJ(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* J,
                    const mjtNum* B, int n) {
  int nv = m->nv;

  // allocate dense row
  mj_markStack(d);
  mjtNum* row = mjSTACKALLOC(d, nv, mjtNum);

  // process non-zero elements of B
  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      if (!B[i*n+j]) {
        continue;
      }
      // process non-zero elements of J(i,:)
      for (int k=0; k < nv; k++) {
        if (J[i*nv+k]) {
          // row = J(i,k)*B(i,j)*J(j,:)
          mju_scl(row, J+j*nv, J[i*nv+k] * B[i*n+j], nv);

          // add row to res(k,:)
          int rownnz_k = m->D_rownnz[k];
          for (int s=0; s < rownnz_k; s++) {
            int adr = m->D_rowadr[k] + s;
            res[adr] += row[m->D_colind[adr]];
          }
        }
      }
    }
  }

  mj_freeStack(d);
}


// add J'*B*J to res (in qDeriv's sparsity layout), sparse version
static void addJTBJSparse(
  const mjModel* m, mjData* d, mjtNum* res, const mjtNum* J,
  const mjtNum* B, int n, int offset,
  const int* J_rownnz, const int* J_rowadr, const int* J_colind) {

  // compute res(k,p) += sum_{i,j} ( J(i,k)*B(i,j)*J(j,p) )
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (!B[i*n+j]) {
        continue;
      }

      // loop over non-zero elements of J(i,:)
      int nnz_i = J_rownnz[offset+i];
      int adr_i = J_rowadr[offset+i];
      int nnz_j = J_rownnz[offset+j];
      int adr_j = J_rowadr[offset+j];
      for (int k = 0; k < nnz_i; k++) {
        int ik = adr_i + k;
        int colik = J_colind[ik];

        // res(k,:) += J(j,:) * J(i,k)*B(i,j)
        mju_addToSclSparseInc(res + m->D_rowadr[colik], J + adr_j,
                              m->D_rownnz[colik], m->D_colind + m->D_rowadr[colik],
                              nnz_j, J_colind + adr_j,
                              J[ik]*B[i*n+j]);
      }
    }
  }
}


//----------------------------- derivatives of actuator forces -------------------------------------

// derivative of mju_muscleGainLength w.r.t length
static mjtNum mjd_muscleGainLength(mjtNum length, mjtNum lmin, mjtNum lmax) {
  if (lmin <= length && length <= lmax) {
    // mid-ranges (maximum is at 1.0)
    mjtNum a = 0.5*(lmin+1);
    mjtNum b = 0.5*(1+lmax);

    if (length <= a) {
      // FL = 0.5*x*x
      mjtNum x = (length-lmin) / mju_max(mjMINVAL, a-lmin);
      return x / mju_max(mjMINVAL, a-lmin);
    } else if (length <= 1) {
      // FL = 1 - 0.5*x*x, x decreasing in length
      mjtNum x = (1-length) / mju_max(mjMINVAL, 1-a);
      return x / mju_max(mjMINVAL, 1-a);
    } else if (length <= b) {
      // FL = 1 - 0.5*x*x
      mjtNum x = (length-1) / mju_max(mjMINVAL, b-1);
      return -x / mju_max(mjMINVAL, b-1);
    } else {
      // FL = 0.5*x*x, x decreasing in length
      mjtNum x = (lmax-length) / mju_max(mjMINVAL, lmax-b);
      return -x / mju_max(mjMINVAL, lmax-b);
    }
  }

  return 0;
}


// derivative of mju_muscleGain w.r.t length
static mjtNum mjd_muscleGain_len(mjtNum len, mjtNum vel, const mjtNum lengthrange[2], mjtNum acc0,
                                 const mjtNum prm[9]) {
  // unpack parameters
  mjtNum range[2] = {prm[0], prm[1]};
  mjtNum force    = prm[2];
  mjtNum scale    = prm[3];
  mjtNum lmin     = prm[4];
  mjtNum lmax     = prm[5];
  mjtNum vmax     = prm[6];
  mjtNum fvmax    = prm[8];

  // scale force if negative
  if (force < 0) {
    force = scale / mju_max(mjMINVAL, acc0);
  }

  // optimum length
  mjtNum L0 = (lengthrange[1]-lengthrange[0]) / mju_max(mjMINVAL, range[1]-range[0]);

  // normalized length and velocity
  mjtNum L = range[0] + (len-lengthrange[0]) / mju_max(mjMINVAL, L0);
  mjtNum V = vel / mju_max(mjMINVAL, L0*vmax);

  // length curve derivative
  mjtNum dFL = mjd_muscleGainLength(L, lmin, lmax);

  // velocity curve
  mjtNum FV;
  mjtNum y = fvmax-1;
  if (V <= -1) {
    FV = 0;
  } else if (V <= 0) {
    FV = (V+1)*(V+1);
  } else if (V <= y) {
    FV = fvmax - (y-V)*(y-V) / mju_max(mjMINVAL, y);
  } else {
    FV = fvmax;
  }

  // compute dFL*FV and scale, make it negative
  return -force*dFL*FV/mju_max(mjMINVAL, L0);
}


// derivative of mju_muscleBias w.r.t length
static mjtNum mjd_muscleBias_len(mjtNum len, const mjtNum lengthrange[2], mjtNum acc0,
                                 const mjtNum prm[9]) {
  // unpack parameters
  mjtNum range[2] = {prm[0], prm[1]};
  mjtNum force    = prm[2];
  mjtNum scale    = prm[3];
  mjtNum lmax     = prm[5];
  mjtNum fpmax    = prm[7];

  // scale force if negative
  if (force < 0) {
    force = scale / mju_max(mjMINVAL, acc0);
  }

  // optimum length
  mjtNum L0 = (lengthrange[1]-lengthrange[0]) / mju_max(mjMINVAL, range[1]-range[0]);

  // normalized length
  mjtNum L = range[0] + (len-lengthrange[0]) / mju_max(mjMINVAL, L0);

  // derivative of the passive curve: half-quadratic to (L0+lmax)/2, linear beyond
  mjtNum b = 0.5*(1+lmax);
  mjtNum dFP;
  if (L <= 1) {
    dFP = 0;
  } else if (L <= b) {
    // FP = 0.5*x*x
    mjtNum x = (L-1) / mju_max(mjMINVAL, b-1);
    dFP = x / mju_max(mjMINVAL, b-1);
  } else {
    // FP = 0.5 + x
    dFP = 1 / mju_max(mjMINVAL, b-1);
  }

  // scale, make it negative
  return -force*fpmax*dFP/mju_max(mjMINVAL, L0);
}


// derivative of mju_muscleGain w.r.t velocity
static mjtNum mjd_muscleGain_vel(mjtNum len, mjtNum vel, const mjtNum lengthrange[2], mjtNum acc0,
                                 const mjtNum prm[9]) {
  // unpack parameters
  mjtNum range[2] = {prm[0], prm[1]};
  mjtNum force    = prm[2];
  mjtNum scale    = prm[3];
  mjtNum lmin     = prm[4];
  mjtNum lmax     = prm[5];
  mjtNum vmax     = prm[6];
  mjtNum fvmax    = prm[8];

  // scale force if negative
  if (force < 0) {
    force = scale / mju_max(mjMINVAL, acc0);
  }

  // optimum length
  mjtNum L0 = (lengthrange[1]-lengthrange[0]) / mju_max(mjMINVAL, range[1]-range[0]);

  // normalized length and velocity
  mjtNum L = range[0] + (len-lengthrange[0]) / mju_max(mjMINVAL, L0);
  mjtNum V = vel / mju_max(mjMINVAL, L0*vmax);

  // length curve
  mjtNum FL = mju_muscleGainLength(L, lmin, lmax);

  // velocity curve
  mjtNum dFV;
  mjtNum y = fvmax-1;
  if (V <= -1) {
    // FV = 0
    dFV = 0;
  } else if (V <= 0) {
    // FV = (V+1)*(V+1)
    dFV = 2*V + 2;
  } else if (V <= y) {
    // FV = fvmax - (y-V)*(y-V) / mju_max(mjMINVAL, y)
    dFV = (-2*V + 2*y) / mju_max(mjMINVAL, y);
  } else {
    // FV = fvmax
    dFV = 0;
  }

  // compute FVL and scale, make it negative
  return -force*FL*dFV/mju_max(mjMINVAL, L0*vmax);
}


//--------------------- utility functions for (d force / d pos) * vec Jacobians --------------------

// add J'*B*J*vec to res, sparse version
static void addJTBJ_mulSparse(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                              const int* J_rownnz, const int* J_rowadr, const int* J_colind,
                              const mjtNum* J, const mjtNum* B, int n) {
  // allocate temp vectors
  mj_markStack(d);
  mjtNum* Jv = mjSTACKALLOC(d, n, mjtNum);
  mjtNum* BJv = mjSTACKALLOC(d, n, mjtNum);

  // Jv = J*vec (Sparse Matrix-Vector Multiplication)
  mju_zero(Jv, n);
  for (int i=0; i < n; i++) {
    int nnz = J_rownnz[i];
    int adr = J_rowadr[i];
    for (int k=0; k < nnz; k++) {
      Jv[i] += J[adr + k] * vec[J_colind[adr + k]];
    }
  }

  // BJv = B*Jv (Dense Matrix-Vector Multiplication)
  mju_mulMatVec(BJv, B, Jv, n, n);

  // res += J'*BJv (Sparse Transpose Matrix-Vector Multiplication)
  for (int i=0; i < n; i++) {
    int nnz = J_rownnz[i];
    int adr = J_rowadr[i];
    mjtNum val = BJv[i];
    for (int k=0; k < nnz; k++) {
      res[J_colind[adr + k]] += J[adr + k] * val;
    }
  }

  mj_freeStack(d);
}


// upper bounds across all interpolated flexes: nodes per flex, nodes per element (3D cell or
// 2D face); max_npe == 0: none
static void flexInterpBounds(const mjModel* m, int* max_nodenum, int* max_npe) {
  *max_nodenum = 0;
  *max_npe = 0;
  for (int f = 0; f < m->nflex; f++) {
    if (!m->flex_interp[f]) continue;
    if (m->flex_rigid[f]) continue;
    int order = m->flex_interp[f];
    int shell_mode = order < 0;
    order = order < 0 ? -order : order;
    int npe;
    if (shell_mode) {
      npe = (order+1)*(order+1);
    } else {
      npe = (order+1)*(order+1)*(order+1);
    }
    if (npe > *max_npe) *max_npe = npe;
    if (m->flex_nodenum[f] > *max_nodenum) *max_nodenum = m->flex_nodenum[f];
  }
}


// stack bytes mjd_flexInterp_mul takes, at most: the kernel's frame below and the nested
// frame of addJTBJ_mulSparse on one element
size_t mjd_flexInterp_mulBytes(const mjModel* m) {
  int max_nodenum, max_npe;
  flexInterpBounds(m, &max_nodenum, &max_npe);
  if (max_npe == 0) {
    return 0;
  }
  size_t nv = m->nv, dim = 3*max_npe;
  size_t an = _Alignof(mjtNum), ai = _Alignof(int), sn = sizeof(mjtNum), si = sizeof(int);
  size_t frame = mj_stackFrameBytes();
  return frame +
         mj_stackBytes(sn*3*max_nodenum, an) + mj_stackBytes(sn*3*max_npe, an) +
         mj_stackBytes(sn*dim*dim, an) + 2*mj_stackBytes(si*dim, ai) +
         mj_stackBytes(sn*dim*nv, an) + mj_stackBytes(si*dim*nv, ai) +
         mj_stackBytes(si*nv, ai) + mj_stackBytes(sn*3*nv, an) + mj_stackBytes(sn*dim, an) +
         frame + 2*mj_stackBytes(sn*dim, an);
}


// shared kernel for flex interpolation derivatives, scale = s1 + s2*damping
//  res: output vector (res += J'*K*J*vec), or NULL for cache-only mode
//  vec: input vector, or NULL for cache-only mode
//  K_rot_cache: if non-NULL, use pre-cached K_rot values instead of computing them
//  K_rot_out: if non-NULL and K_rot_cache is NULL, store computed K_rot values here
// Stack: mjd_flexInterp_mulBytes
static void mjd_flexInterp_kernel(const mjModel* m, mjData* d,
                                  mjtNum* res, const mjtNum* vec, mjtNum s1, mjtNum s2,
                                  const mjtNum* K_rot_cache, mjtNum* K_rot_out,
                                  int first, int last) {
  int nv = m->nv;

  // compute upper bounds across all interpolated flexes
  int max_nodenum, max_npe;
  flexInterpBounds(m, &max_nodenum, &max_npe);

  // nothing to do
  if (max_npe == 0) {
    return;
  }

  int max_dim_c = 3 * max_npe;

  // single unconditional markStack
  mj_markStack(d);

  // per-flex node positions (upper bound)
  mjtNum* xpos = mjSTACKALLOC(d, 3*max_nodenum, mjtNum);

  // per-element arrays (upper bound)
  mjtNum* xpos_c = mjSTACKALLOC(d, 3*max_npe, mjtNum);
  mjtNum* K_rot_cell = mjSTACKALLOC(d, max_dim_c*max_dim_c, mjtNum);

  // sparse Jacobian for one cell (upper bound)
  int* J_rownnz = mjSTACKALLOC(d, max_dim_c, int);
  int* J_rowadr = mjSTACKALLOC(d, max_dim_c, int);
  mjtNum* J_val = mjSTACKALLOC(d, max_dim_c*nv, mjtNum);
  int* J_colind = mjSTACKALLOC(d, max_dim_c*nv, int);

  // temp allocations for chain
  int* chain_colind = mjSTACKALLOC(d, nv, int);
  mjtNum* blk_jac = mjSTACKALLOC(d, 3*nv, mjtNum);

  // temp buffer for fast path element result
  mjtNum* fast_tmp = mjSTACKALLOC(d, max_dim_c, mjtNum);

  // loop over flexes
  for (int f=first; f < last; f++) {
    // only process stiff, non-rigid flex_interp without strain constraints
    if (!mjd_flexInterp_processed(m, f)) {
      continue;
    }

    // get stiffness and damping
    int stiffnessadr = m->flex_stiffnessadr[f];
    mjtNum* K = m->flex_stiffness + stiffnessadr;

    // compute scale
    mjtNum damping = m->flex_damping[f];
    mjtNum scale = s1 + s2 * damping;

    // skip if scale is zero
    if (scale == 0) {
      continue;
    }

    int order = m->flex_interp[f];
    int shell_mode = order < 0;
    order = order < 0 ? -order : order;

    int cx = m->flex_cellnum[3*f+0];
    int cy = m->flex_cellnum[3*f+1];
    int cz = m->flex_cellnum[3*f+2];

    int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];

    // determine element type: 2D boundary quads (shell) or 3D cells (volume)
    int npe;
    int nelem_fe;
    if (shell_mode) {
      npe = (order+1)*(order+1);
      nelem_fe = 2*(cy*cz + cx*cz + cx*cy);
    } else {
      npe = (order+1)*(order+1)*(order+1);
      nelem_fe = cx * cy * cz;
    }
    int dim_e = 3 * npe;

    // gather raw node positions (unrotated)
    mju_flexGatherState(m, d, f, xpos, NULL);

    // check if centered fast path applies: centered, all nodes on simple slider
    // bodies (body_simple == 2 means diag M with sliders only, J = R_body per node)
    int use_fast_path = m->flex_centered[f];
    if (use_fast_path) {
      int nodenum_f = m->flex_nodenum[f];
      for (int n = 0; n < nodenum_f; n++) {
        if (m->body_simple[bodyid[n]] != 2) {
          use_fast_path = 0;
          break;
        }
      }
    }

    // loop over finite elements
    for (int fe = 0; fe < nelem_fe; fe++) {
      // get element stiffness
      mjtNum* k_elem = K + fe * 3*npe * 3*npe;

      // skip empty elements in volume mode: stiffness buffer is zero-initialized at compile time
      // (user_model.cc), and non-empty 3D cells have strictly positive diagonal; shells have no
      // empty cells and their x-normal faces (faces 0 and 1) have k_elem[0] == 0
      if (!shell_mode && k_elem[0] == 0) {
        continue;
      }

      // use cached K_rot or compute from scratch
      int gindices[125];  // element node indices (max npe = 125 for quadratic 3D)
      int krot_adr = stiffnessadr + fe * dim_e * dim_e;
      if (K_rot_cache) {
        // read K_rot from cache and apply scale
        for (int i = 0; i < dim_e*dim_e; i++) {
          K_rot_cell[i] = scale * K_rot_cache[krot_adr + i];
        }

        // recompute gindices (cheap index-only call, no rotation)
        if (shell_mode) {
          mju_flexGatherFaceState(order, cx, cy, cz, fe, NULL, NULL, NULL,
                                  NULL, NULL, NULL, gindices, NULL);
        } else {
          int ci = fe / (cy * cz);
          int cj = (fe / cz) % cy;
          int ck = fe % cz;
          mju_flexGatherCellState(order, cy, cz, ci, cj, ck, NULL, NULL, NULL,
                                  NULL, NULL, NULL, gindices, NULL);
        }
      } else {
        // gather element-local node positions and rotation
        const mjtNum* xpos0 = m->flex_node0 + 3*m->flex_nodeadr[f];
        mjtNum quat[4];
        if (shell_mode) {
          mju_flexGatherFaceState(order, cx, cy, cz, fe, xpos, NULL, xpos0,
                                  xpos_c, NULL, NULL, gindices, quat);
        } else {
          int ci = fe / (cy * cz);
          int cj = (fe / cz) % cy;
          int ck = fe % cz;
          mju_flexGatherCellState(order, cy, cz, ci, cj, ck, xpos, NULL, xpos0,
                                  xpos_c, NULL, NULL, gindices, quat);
        }

        // R = R_global2local, RT = R_local2global
        mjtNum R[9], RT[9];
        mju_quat2Mat(R, quat);
        mju_transpose(RT, R, 3, 3);

        // compute K_rot = RT * K_elem * R (block-wise)
        mju_zero(K_rot_cell, dim_e*dim_e);
        for (int a = 0; a < npe; a++) {
          for (int b = 0; b < npe; b++) {
            mjtNum blk[9], tmp[9];

            // get K_elem(a,b) 3x3 block
            int adr_cell = (3*a)*(3*npe) + 3*b;
            for (int r = 0; r < 3; r++) {
              for (int c = 0; c < 3; c++) {
                blk[3*r+c] = k_elem[adr_cell + r*(3*npe) + c];
              }
            }

            // tmp = K * R
            mji_mulMatMat3(tmp, blk, R);
            // blk = RT * tmp = RT * K * R
            mji_mulMatMat3(blk, RT, tmp);

            // store in K_rot_cell at (a, b)
            int adr_out = (3*a)*dim_e + 3*b;
            for (int r = 0; r < 3; r++) {
              for (int c = 0; c < 3; c++) {
                K_rot_cell[adr_out + r*dim_e + c] = scale * blk[3*r+c];
              }
            }
          }
        }

        // optionally store unscaled K_rot to output cache
        if (K_rot_out) {
          for (int i = 0; i < dim_e*dim_e; i++) {
            K_rot_out[krot_adr + i] = K_rot_cell[i] / scale;
          }
        }
      }

      // skip Jacobian construction and op when only caching (res == NULL)
      if (!res) {
        continue;
      }

      // fast path: centered flex with 3 translational DOFs per body
      // J = R_body (not I when body is rotated), so J'*K*J*vec = R^T*K*(R*vec)
      if (use_fast_path) {
        // apply R_body to vec for each node
        for (int n = 0; n < npe; n++) {
          int dof = m->body_dofadr[bodyid[gindices[n]]];
          mji_mulMatVec3(fast_tmp + 3*n, d->xmat + 9*bodyid[gindices[n]], vec + dof);
        }

        // compute R^T * K_rot * (R*vec) and add to res
        for (int a = 0; a < npe; a++) {
          int dof_a = m->body_dofadr[bodyid[gindices[a]]];
          mjtNum row[3] = {0, 0, 0};
          for (int b = 0; b < npe; b++) {
            int adr = (3*a)*dim_e + 3*b;
            for (int r = 0; r < 3; r++) {
              for (int c = 0; c < 3; c++) {
                row[r] += K_rot_cell[adr + r*dim_e + c] * fast_tmp[3*b + c];
              }
            }
          }
          // apply R^T and add to res
          mjtNum qfrc_loc[3];
          mji_mulMatTVec3(qfrc_loc, d->xmat + 9*bodyid[gindices[a]], row);
          for (int r = 0; r < 3; r++) {
            res[dof_a + r] += qfrc_loc[r];
          }
        }
      } else {
        // general path: construct sparse Jacobian for this element's nodes
        int current_adr = 0;
        for (int n = 0; n < npe; n++) {
          int bid = bodyid[gindices[n]];
          int chain_nnz = mj_bodyChain(m, bid, chain_colind);
          mj_jacSparse(m, d, blk_jac, NULL, xpos+3*gindices[n], bid,
                       chain_nnz, chain_colind, /*flg_skipcommon=*/0);

          for (int r = 0; r < 3; r++) {
            int row_idx = 3*n + r;
            J_rownnz[row_idx] = chain_nnz;
            J_rowadr[row_idx] = current_adr;

            for (int idx = 0; idx < chain_nnz; idx++) {
              J_colind[current_adr] = chain_colind[idx];
              J_val[current_adr] = blk_jac[r*chain_nnz + idx];
              current_adr++;
            }
          }
        }

        // res += J'*K_rot*J*vec
        addJTBJ_mulSparse(m, d, res, vec, J_rownnz, J_rowadr, J_colind,
                          J_val, K_rot_cell, dim_e);
      }

    }
  }

  mj_freeStack(d);
}


// compute res += (s1 + s2*damping) * J'*K*J * vec for interpolated flexes in [first, last)
void mjd_flexInterp_mulRange(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                             mjtNum s1, mjtNum s2, const mjtNum* K_rot_cache,
                             int first, int last) {
  mjd_flexInterp_kernel(m, d, res, vec, s1, s2, K_rot_cache, NULL, first, last);
}


// compute res += (s1 + s2*damping) * J'*K*J * vec, for all interpolated flexes
//   K_rot_cache: if non-NULL, use pre-cached K_rot (same layout as m->flex_stiffness)
void mjd_flexInterp_mul(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                        mjtNum s1, mjtNum s2, const mjtNum* K_rot_cache) {
  mjd_flexInterp_kernel(m, d, res, vec, s1, s2, K_rot_cache, NULL, 0, m->nflex);
}


// precompute unscaled K_rot for all elements into cache (same layout as m->flex_stiffness)
void mjd_flexInterp_cacheKrot(const mjModel* m, mjData* d, mjtNum* K_rot_out) {
  // use s1=1, s2=0 so scale=1 and K_rot_out gets unscaled values
  mjd_flexInterp_kernel(m, d, NULL, NULL, 1, 0, NULL, K_rot_out, 0, m->nflex);
}


// Edge bending stencil s*c*c' (s = cos(theta)*stiffness, Wardetzky cotangent vector c):
// negative semidefinite at rest creases sharper than 90 degrees (s < 0). Keep only convex
// stencils (trace s*|c|^2 >= 0) in the solver metric so it stays SPD.
static inline int flexBendConvex(const mjtNum* stencil) {
  return stencil[0] + stencil[5] + stencil[10] + stencil[15] >= 0;
}


// compute res += scale * K_bend * vec for standard (non-interp) flex bending, over the convex
// stencils (flexBendConvex)
//   scale = s1 + s2 * flex_damping[f]  per flex
//   for stiffness+damping: s1=h^2, s2=h  =>  scale = h^2 + h*damping
//   for stiffness only:    s1=h,   s2=0  =>  scale = h
void mjd_flexBend_mulRange(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                           mjtNum s1, mjtNum s2, int first, int last) {
  for (int f = first; f < last; f++) {
    // skip interp, rigid, or non-2D
    if (m->flex_interp[f] || m->flex_rigid[f] || m->flex_dim[f] != 2) {
      continue;
    }

    int bendingadr = m->flex_bendingadr[f];
    if (bendingadr < 0) {
      continue;
    }

    mjtNum scale = s1 + s2 * m->flex_damping[f];
    if (!scale) {
      continue;
    }

    const mjtNum* b = m->flex_bending + bendingadr;
    int vertnum = m->flex_vertnum[f];
    int edgenum = m->flex_edgenum[f];
    int edgeadr = m->flex_edgeadr[f];

    mj_markStack(d);
    mjtNum* wvec = mjSTACKALLOC(d, 3*vertnum, mjtNum);
    mjtNum* wres = mjSTACKALLOC(d, 3*vertnum, mjtNum);
    mju_zero(wres, 3*vertnum);

    mj_flexGather(m, d, f, wvec, vec);

    for (int e = 0; e < edgenum; e++) {
      const int* edge = m->flex_edge + 2*(e + edgeadr);
      const int* flap = m->flex_edgeflap + 2*(e + edgeadr);
      int v[4] = {edge[0], edge[1], flap[0], flap[1]};

      // skip boundary edges (no second flap vertex) and concave stencils
      if (v[3] == -1 || !flexBendConvex(b + 17*e)) {
        continue;
      }

      // apply 4x4 bending stencil in world space
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          mjtNum q = b[17*e + 4*i + j];
          for (int x = 0; x < 3; x++) {
            wres[3*v[i] + x] += q * wvec[3*v[j] + x];
          }
        }
      }
    }

    mj_flexScatter(m, d, f, res, wres, scale);

    mj_freeStack(d);
  }
}


void mjd_flexBend_mul(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                      mjtNum s1, mjtNum s2) {
  mjd_flexBend_mulRange(m, d, res, vec, s1, s2, 0, m->nflex);
}


// Jacobian J (nvrt x 3*nvrt) and weight W = s1*Wk + s2*damping*Wd of 1D rod hinge h of flex f, a
// single hinge (nvrt = 3, kb space) or a hinge pair (nvrt = 4, om space), so that the stiffness is
// J'WJ; returns 0 if the hinge is degenerate
// weights Wk, Wd (9 x nhinge each, kb space) of all hinges of the 1D rod chain of flex f, evaluated
// once on the mjData stack (the caller marks and frees it); NULL if the chain is isotropic or
// not handled by mj_rodChain
static mjtNum* rodChainWeights(const mjModel* m, mjData* d, int f) {
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  int nhinge = (int)b[0];
  if (nhinge <= 0 || b[1 + 3] >= 0 || b[1 + 5] == 0) {
    return NULL;
  }
  mjtNum* W = mjSTACKALLOC(d, 18*nhinge, mjtNum);
  if (!mj_rodChain(m, d, f, NULL, NULL, NULL, W, W + 9*nhinge, NULL)) {
    return NULL;
  }
  return W;
}


// chain: weights of the chain of flex f from rodChainWeights, or NULL
static int rodHingeWeight(const mjModel* m, const mjData* d, int f, int h, mjtNum s1, mjtNum s2,
                          const mjtNum* chain, int v[4], int* nvrt, mjtNum J[48],
                          mjtNum W[16]) {
  const mjtNum* hinge = m->flex_bending + m->flex_bendingadr[f] + 1 + mjROD_HINGE_SIZE*h;
  const mjtNum* xpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
  for (int i = 0; i < 4; i++) {
    v[i] = (int)hinge[i];
  }
  int n = mj_rodHingeVert(hinge);
  mjtNum Wk[16], Wd[16];
  if (n == 3) {
    mjtNum kb[3];
    if (!mj_rodHinge(xpos + 3*v[0], xpos + 3*v[1], xpos + 3*v[2], kb, J)) {
      return 0;
    }
    if (chain && hinge[5] != 0) {
      int nhinge = (int)m->flex_bending[m->flex_bendingadr[f]];
      mju_copy(Wk, chain + 9*h, 9);
      mju_copy(Wd, chain + 9*nhinge + 9*h, 9);
    } else {
      mj_rodHingeWeights(NULL, Wk, Wd, kb, hinge[4], hinge[6]);
    }
  } else {
    mjtNum om[4];
    if (!mj_rodPair(xpos + 3*v[0], xpos + 3*v[1], xpos + 3*v[2], xpos + 3*v[3], om, J)) {
      return 0;
    }
    mj_rodPairWeights(NULL, Wk, Wd, om, hinge[4], hinge[5], hinge + 6);
  }
  mjtNum sd = s2*m->flex_damping[f];
  for (int i = 0; i < n*n; i++) {
    W[i] = s1*Wk[i] + sd*Wd[i];
  }
  *nvrt = n;
  return 1;
}


// world-space stiffness (3*nvrt x 3*nvrt) of 1D rod hinge h of flex f: J'(s1*Wk + s2*damping*Wd)J,
// with block (i, j) at rows 3i..3i+2 and columns 3j..3j+2; returns 0 if the hinge is degenerate
static int rodHingeStiffness(const mjModel* m, const mjData* d, int f, int h,
                             mjtNum s1, mjtNum s2, const mjtNum* chain, int v[4], int* nvrt,
                             mjtNum K[144]) {
  mjtNum J[48], W[16];
  if (!rodHingeWeight(m, d, f, h, s1, s2, chain, v, nvrt, J, W)) {
    return 0;
  }

  // WJ = W * J (nvrt x 3*nvrt), K = J' * WJ (3*nvrt x 3*nvrt)
  int n = *nvrt;
  mjtNum WJ[48];
  mju_mulMatMat(WJ, W, J, n, n, 3*n);
  mju_mulMatTMat(K, J, WJ, n, 3*n, 3*n);
  return 1;
}


// compute res += K_rod * vec for 1D rod bending of flex f, applying J'(W(J vec)) per hinge
// without forming the hinge stiffness, see rodHingeWeight
static void flexRod_mul(const mjModel* m, mjData* d, int f, mjtNum* res, const mjtNum* vec,
                        mjtNum s1, mjtNum s2) {
  int nvert = m->flex_vertnum[f];
  int nhinge = (int)m->flex_bending[m->flex_bendingadr[f]];
  mj_markStack(d);
  mjtNum* wvec = mjSTACKALLOC(d, 3*nvert, mjtNum);
  mjtNum* wres = mjSTACKALLOC(d, 3*nvert, mjtNum);
  mj_flexGather(m, d, f, wvec, vec);
  mju_zero(wres, 3*nvert);
  const mjtNum* chain = rodChainWeights(m, d, f);

  for (int h = 0; h < nhinge; h++) {
    int v[4], n;
    mjtNum J[48], W[16];
    if (!rodHingeWeight(m, d, f, h, s1, s2, chain, v, &n, J, W)) {
      continue;
    }

    // u = J vec, w = W u
    mjtNum u[4] = {0}, w[4];
    for (int r = 0; r < n; r++) {
      for (int i = 0; i < n; i++) {
        u[r] += mju_dot3(J + 3*n*r + 3*i, wvec + 3*v[i]);
      }
    }
    mju_mulMatVec(w, W, u, n, n);

    // res += J' w
    for (int i = 0; i < n; i++) {
      for (int c = 0; c < 3; c++) {
        mjtNum sum = 0;
        for (int r = 0; r < n; r++) {
          sum += J[3*n*r + 3*i + c]*w[r];
        }
        wres[3*v[i]+c] += sum;
      }
    }
  }

  mj_flexScatter(m, d, f, res, wres, 1);
  mj_freeStack(d);
}


// does flex f carry 1D rod bending stiffness
static int flexRod_active(const mjModel* m, int f) {
  return m->flex_dim[f] == 1 && !m->flex_rigid[f] && m->flex_bendingadr[f] >= 0;
}


// compute res += (s1 + s2*flex_damping) * K_stretch * vec for standard flexes
// SNH uses its PSD-projected material Hessian; StVK keeps the material term
// and tensile geometric stiffness. For articulated attachments the pullback J'KJ
// omits derivatives of the attachment Jacobian.
// 1D flexes contribute their (state-dependent) rod bending stiffness here, see
// rodHingeStiffness, rather than with the constant bending stencils of 2D flexes.
void mjd_flexStretch_mulRange(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                              mjtNum s1, mjtNum s2, int first, int last) {
  for (int f = first; f < last; f++) {
    // 1D rod bending
    if (flexRod_active(m, f)) {
      flexRod_mul(m, d, f, res, vec, s1, s2);
      continue;
    }

    // skip interp, rigid, or 1D
    if (m->flex_interp[f] || m->flex_rigid[f] || m->flex_dim[f] < 2) {
      continue;
    }

    int stiffnessadr = m->flex_stiffnessadr[f];
    if (stiffnessadr < 0 || m->flex_stiffness[stiffnessadr] == 0) {
      continue;
    }

    mjtNum scale = s1 + s2 * m->flex_damping[f];
    if (!scale) {
      continue;
    }

    int nvert = m->flex_vertnum[f];
    mj_markStack(d);
    mjtNum* wvec = mjSTACKALLOC(d, 3*nvert, mjtNum);
    mjtNum* wres = mjSTACKALLOC(d, 3*nvert, mjtNum);
    mj_flexGather(m, d, f, wvec, vec);
    mju_zero(wres, 3*nvert);

    mj_flexHessian(m, d, f);
    mj_flexHessianMul(m, d, f, wres, wvec, scale);
    mj_flexScatter(m, d, f, res, wres, 1);
    mj_freeStack(d);
  }
}


void mjd_flexStretch_mul(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                         mjtNum s1, mjtNum s2) {
  mjd_flexStretch_mulRange(m, d, res, vec, s1, s2, 0, m->nflex);
}


// returns true if the interpolated flex is processed by mjd_flexInterp_mul
mjtBool mjd_flexInterp_processed(const mjModel* m, int f) {
  if (!m->flex_interp[f]) {
    return 0;
  }
  // strain constraints present: stiffness handled by the constraint solver
  if (m->flex_edgeequality[f] == 3) {
    return 0;
  }
  // element 0 may be an empty cell with zero stiffness
  if (m->flex_rigid[f] || !mj_flexHasStiffness(m, f)) {
    return 0;
  }
  return 1;
}


// can ALL operator-processed interp flexes be assembled to dof-level CSR? Requires every node
// to sit on a simple slider body (body_simple == 2: sliders only, so the point Jacobian is I3
// regardless of the node offset -- flex_centered is NOT required) or to be fixed (0 dofs, row
// dropped like a pinned vertex). All-or-nothing: mjd_flexInterp_mul applies every interp flex,
// so the CSR can replace the operator only if it covers them all.
mjtBool mjd_flexInterpAssemblable(const mjModel* m) {
  for (int f = 0; f < m->nflex; f++) {
    if (!mjd_flexInterp_processed(m, f)) {
      continue;
    }
    const int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];
    for (int n = 0; n < m->flex_nodenum[f]; n++) {
      int dofnum = m->body_dofnum[bodyid[n]];
      if (dofnum == 0) {
        continue;
      }
      if (m->body_simple[bodyid[n]] != 2 || dofnum != 3) {
        return 0;
      }
    }
  }
  return 1;
}


// element walk shared by the interp assembly passes: geometry of interp finite elements
// (mirrors mjd_flexInterp_kernel; index-only gather calls)
#define FLEXINTERP_WALK(f, body)                                             \
  {                                                                          \
    int order_ = m->flex_interp[f];                                          \
    int shell_ = order_ < 0;                                                 \
    order_ = order_ < 0 ? -order_ : order_;                                  \
    int cx_ = m->flex_cellnum[3 * (f) + 0];                                  \
    int cy_ = m->flex_cellnum[3 * (f) + 1];                                  \
    int cz_ = m->flex_cellnum[3 * (f) + 2];                                  \
    int npe = shell_ ? (order_ + 1) * (order_ + 1)                           \
                     : (order_ + 1) * (order_ + 1) * (order_ + 1);           \
    int nelem_fe_ =                                                          \
        shell_ ? 2 * (cy_ * cz_ + cx_ * cz_ + cx_ * cy_) : cx_ * cy_ * cz_;  \
    const mjtNum* K_ = m->flex_stiffness + m->flex_stiffnessadr[f];          \
    int gindices[125];                                                       \
    for (int fe = 0; fe < nelem_fe_; fe++) {                                 \
      if (!shell_ && K_[(size_t)fe * 3 * npe * 3 * npe] == 0) continue;      \
      if (shell_) {                                                          \
        mju_flexGatherFaceState(order_, cx_, cy_, cz_, fe, NULL, NULL, NULL, \
                                NULL, NULL, NULL, gindices, NULL);           \
      } else {                                                               \
        int ci_ = fe / (cy_ * cz_), cj_ = (fe / cz_) % cy_, ck_ = fe % cz_;  \
        mju_flexGatherCellState(order_, cy_, cz_, ci_, cj_, ck_, NULL, NULL, \
                                NULL, NULL, NULL, NULL, gindices, NULL);     \
      }                                                                      \
      body                                                                   \
    }                                                                        \
  }

// does any flex use the passive contact path? Distinct from elasticity: an empty CSR is valid for
// elastic models (matrix-free operators) but means "nothing" for a contact-only flex.
mjtBool mjd_flexPassiveContact_any(const mjModel* m) {
  for (int f = 0; f < m->nflex; f++) {
    if (mj_effFlexContactPossible(m, f)) {
      return 1;
    }
  }
  return 0;
}

// does ANY flex contribute assemblable implicit stiffness? (cheap existence check for the
// solver gate: stretch stiffness on a standard flex, or -- when Krot will be supplied -- an
// operator-processed interp flex)
mjtBool mjd_flexStiff_any(const mjModel* m, int flg_interp) {
  for (int f = 0; f < m->nflex; f++) {
    if (flg_interp && mjd_flexInterp_processed(m, f)) {
      return 1;
    }
    if (!m->flex_interp[f] && !m->flex_rigid[f] && m->flex_dim[f] >= 2 &&
        m->flex_stiffnessadr[f] >= 0 && m->flex_stiffness[m->flex_stiffnessadr[f]] != 0 &&
        mj_flexSimple(m, f)) {
      return 1;
    }
    if (flexRod_active(m, f) && mj_flexSimple(m, f)) {
      return 1;
    }
  }
  return 0;
}


// does this standard flex contribute implicit stiffness under the given term flags?
// 1D rod bending is state-dependent and therefore follows flg_stretch, not the constant
// bending stencils of flg_bend
mjtBool mjd_flexStiff_active(const mjModel* m, int f, int flg_bend, int flg_stretch) {
  if (flexRod_active(m, f)) {
    return flg_stretch && mj_flexSimple(m, f);
  }
  if (m->flex_interp[f] || m->flex_rigid[f] || m->flex_dim[f] < 2 ||
      !mj_flexSimple(m, f)) {
    return 0;
  }
  int bend = flg_bend && m->flex_bendingadr[f] >= 0;
  int stretch = flg_stretch && m->flex_stiffnessadr[f] >= 0 &&
                m->flex_stiffness[m->flex_stiffnessadr[f]] != 0;
  return bend || stretch;
}

// The flex-contact law (see engine_derivative.h). lam/k is taken as 0 when the multiplier is,
// so a massless pair (k = 0) yields no force rather than a NaN.
mjtNum mjd_flexContactSlack(mjtNum k, mjtNum gap, mjtNum lam) {
  mjtNum t = gap - (lam ? lam/k : 0);
  return t > 0 ? t : 0;
}


mjtNum mjd_flexContactResidual(mjtNum k, mjtNum gap, mjtNum s, mjtNum lam) {
  return gap - s - (lam ? lam/k : 0);
}


static mjtNum flexBodyMass(const mjModel* m, const mjData* d, int body) {
  if (body < 0) {
    return 0;
  }
  int b = m->body_weldid[body];
  if (m->body_dofnum[b] != 3) {
    return 0;
  }
  int da = m->body_dofadr[b];
  return d->M[m->M_rowadr[da] + m->M_rownnz[da] - 1];   // diagonal: the point mass
}


mjtNum mjd_flexVertMass(const mjModel* m, const mjData* d, int gv) {
  return flexBodyMass(m, d, m->flex_vertbodyid[gv]);
}


// count dynamic (3-dof) cage nodes of interpolated flex f
int mjd_flexInterpDynNodes(const mjModel* m, int f) {
  if (m->flex_centered[f]) {
    return m->flex_nodenum[f];
  }
  int nstart = m->flex_nodeadr[f];
  int ndyn = 0;
  for (int j = 0; j < m->flex_nodenum[f]; j++) {
    int b = m->flex_nodebodyid[nstart + j];
    if (b >= 0 && m->body_dofnum[m->body_weldid[b]] == 3) {
      ndyn++;
    }
  }
  return ndyn;
}


mjtNum mjd_flexContactStiffness(const mjModel* m, const mjData* d, const mjContact* con,
                                const int* flex_ndyn) {
  mjtNum mmin = 0;
  for (int side = 0; side < 2; side++) {
    int f = con->flex[side];
    if (f < 0) {
      continue;
    }
    if (m->flex_interp[f]) {
      // 1 cell has at most 27 nodes (quadratic), projected to at most 27 boundary nodes in TFI
      int body[27];
      mjtNum weight[27];
      int nb = mj_contactBodyWeight(m, d, con, side, body, weight);
      mjtNum inv_m = 0;
      for (int j = 0; j < nb; j++) {
        mjtNum mb = flexBodyMass(m, d, body[j]);
        if (mb > 0) {
          inv_m += weight[j] * weight[j] / mb;
        }
      }
      if (inv_m > 0) {
        int ndyn = flex_ndyn ? flex_ndyn[f] : mjd_flexInterpDynNodes(m, f);
        mjtNum scale = m->flex_vertnum[f] > 0
                       ? (mjtNum)ndyn / m->flex_vertnum[f]
                       : 1;
        mjtNum meff = scale / inv_m;
        if (mmin == 0 || meff < mmin) {
          mmin = meff;
        }
      }
      continue;
    }
    int gv[8], ngv = 0;
    if (con->vert[side] >= 0) {
      gv[ngv++] = m->flex_vertadr[f] + con->vert[side];
    } else if (con->elem[side] >= 0) {
      int nvrt = m->flex_dim[f] + 1;
      const int* e = m->flex_elem + m->flex_elemdataadr[f] + nvrt*con->elem[side];
      for (int j = 0; j < nvrt && ngv < 8; j++) {
        gv[ngv++] = m->flex_vertadr[f] + e[j];
      }
    }
    for (int j = 0; j < ngv; j++) {
      mjtNum mv = mjd_flexVertMass(m, d, gv[j]);
      if (mv > 0 && (mmin == 0 || mv < mmin)) {
        mmin = mv;
      }
    }
  }
  return mjFLEXCONTACT_OMEGA2 * mmin;   // 0 if every participant is massless: no stiffness, no NaN
}


// assemble the standard-flex implicit stiffness K = (s1 + s2*damping) * (K_bend + K_stretch)
// into dof-level CSR (same terms mjd_flexBend_mul / mjd_flexStretch_mul apply matrix-free; the
// matrix is constant during a solve, so assembling once and applying as a sparse matvec avoids
// re-walking stencils and re-unpacking metrics on every apply). Rows/columns exist only on the
// dofs of unpinned vertices of standard dim>=2 flexes with bending or stretch stiffness.
// Phase 1 (colind == NULL): fill rownnz/rowadr over nv, return total nnz.
// Phase 2: fill colind and, unless val is NULL (structure only), val (rownnz/rowadr must come from
// phase 1).
// Interp flexes are assembled iff Krot (the mjd_flexInterp_cacheKrot cache) is non-NULL and
// they qualify for the centered fast path (caller checks mjd_flexInterpAssemblable): rows on
// node body dofs, corotated 3x3 blocks, with the operator's NEGATED sign convention folded in
// so one CSR replaces all three matrix-free operators uniformly.
int mjd_flexStiff_assemble(const mjModel* m, mjData* d, int* rownnz, int* rowadr,
                           int* colind, mjtNum* val, mjtNum s1, mjtNum s2,
                           int flg_bend, int flg_stretch, const mjtNum* Krot) {
  int nv = m->nv;
  mj_markStack(d);

  // collect participating bodies: bodies with 3 dofs get one slot each
  int* bodyslot = mjSTACKALLOC(d, m->nbody, int);
  for (int b = 0; b < m->nbody; b++) {
    bodyslot[b] = -1;
  }
  for (int f = 0; f < m->nflex; f++) {
    // contact no longer forces vertex slots: a contact-only flex rides the rank-1 class
    if (!mjd_flexStiff_active(m, f, flg_bend, flg_stretch)) {
      continue;
    }
    for (int lv = 0; lv < m->flex_vertnum[f]; lv++) {
      int gv = m->flex_vertadr[f] + lv;
      int b = m->body_weldid[m->flex_vertbodyid[gv]];
      if (m->body_dofnum[b] == 3) {
        bodyslot[b] = 1;
      }
    }
  }

  // interp nodes participate when the caller supplies the K_rot cache (centered fast path)
  if (Krot) {
    for (int f = 0; f < m->nflex; f++) {
      if (!mjd_flexInterp_processed(m, f)) {
        continue;
      }
      const int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];
      for (int ln = 0; ln < m->flex_nodenum[f]; ln++) {
        int b = bodyid[ln];
        if (m->body_dofnum[b] == 3) {
          bodyslot[b] = 1;
        }
      }
    }
  }

  int nvert = 0;
  for (int b = 0; b < m->nbody; b++) {
    if (bodyslot[b] > 0) {
      bodyslot[b] = nvert++;
    }
  }

  int* vslot = mjSTACKALLOC(d, m->nflexvert > 0 ? m->nflexvert : 1, int);
  for (int i = 0; i < m->nflexvert; i++) {
    int b = m->flex_vertbodyid[i];
    vslot[i] = (b >= 0) ? bodyslot[m->body_weldid[b]] : -1;
  }

  int* nslot = mjSTACKALLOC(d, m->nflexnode > 0 ? (int)m->nflexnode : 1, int);
  for (int i = 0; i < m->nflexnode; i++) {
    int b = m->flex_nodebodyid[i];
    nslot[i] = (b >= 0) ? bodyslot[b] : -1;
  }

  if (!nvert) {
    mju_zeroInt(rownnz, nv);
    if (rowadr) mju_zeroInt(rowadr, nv);
    mj_freeStack(d);
    return 0;
  }
  int* vdof = mjSTACKALLOC(d, nvert, int);
  for (int b = 0; b < m->nbody; b++) {
    if (bodyslot[b] >= 0) {
      vdof[bodyslot[b]] = m->body_dofadr[b];
    }
  }

  // count stencil incidence per vertex slot (with duplicates)
  int* ncand = mjSTACKALLOC(d, nvert, int);
  mju_zeroInt(ncand, nvert);
  for (int f = 0; f < m->nflex; f++) {
    if (!mjd_flexStiff_active(m, f, flg_bend, flg_stretch)) {
      continue;
    }
    int dim = m->flex_dim[f], nvrt = dim + 1;
    // rod hinges: 3- or 4-vertex cliques
    if (flexRod_active(m, f)) {
      const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
      for (int h = 0; h < (int)b[0]; h++) {
        const mjtNum* hinge = b + 1 + mjROD_HINGE_SIZE*h;
        int nr = mj_rodHingeVert(hinge);
        for (int i = 0; i < nr; i++) {
          int si = vslot[m->flex_vertadr[f] + (int)hinge[i]];
          if (si >= 0) ncand[si] += nr;
        }
      }
      continue;
    }
    // bending stencils: 4-vertex flaps per edge
    if (flg_bend && m->flex_bendingadr[f] >= 0) {
      for (int e = 0; e < m->flex_edgenum[f]; e++) {
        const int* edge = m->flex_edge + 2*(e + m->flex_edgeadr[f]);
        const int* flap = m->flex_edgeflap + 2*(e + m->flex_edgeadr[f]);
        if (flap[1] == -1) continue;
        int v[4] = {edge[0], edge[1], flap[0], flap[1]};
        for (int i = 0; i < 4; i++) {
          int si = vslot[m->flex_vertadr[f] + v[i]];
          if (si >= 0) ncand[si] += 4;
        }
      }
    }
    // stretch elements: (dim+1)-vertex cliques
    if (flg_stretch && m->flex_stiffnessadr[f] >= 0 &&
        m->flex_stiffness[m->flex_stiffnessadr[f]] != 0) {
      const int* elem = m->flex_elem + m->flex_elemdataadr[f];
      for (int t = 0; t < m->flex_elemnum[f]; t++) {
        for (int i = 0; i < nvrt; i++) {
          int si = vslot[m->flex_vertadr[f] + elem[(dim+1)*t + i]];
          if (si >= 0) ncand[si] += nvrt;
        }
      }
    }
  }

  // interp stencils: npe-node element cliques (counting)
  if (Krot) {
    for (int f = 0; f < m->nflex; f++) {
      if (!mjd_flexInterp_processed(m, f)) {
        continue;
      }
      FLEXINTERP_WALK(f, {
        for (int i = 0; i < npe; i++) {
          int si = nslot[m->flex_nodeadr[f] + gindices[i]];
          if (si >= 0) ncand[si] += npe;
        }
      })
    }
  }

  // gather candidate neighbor lists (vertex slots, with duplicates)
  int* cadr = mjSTACKALLOC(d, nvert + 1, int);
  cadr[0] = 0;
  for (int s = 0; s < nvert; s++) {
    cadr[s+1] = cadr[s] + ncand[s];
  }
  int* cand = mjSTACKALLOC(d, cadr[nvert] > 0 ? cadr[nvert] : 1, int);
  mju_zeroInt(ncand, nvert);
  for (int f = 0; f < m->nflex; f++) {
    if (!mjd_flexStiff_active(m, f, flg_bend, flg_stretch)) {
      continue;
    }
    int dim = m->flex_dim[f], nvrt = dim + 1;
    if (flexRod_active(m, f)) {
      const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
      for (int h = 0; h < (int)b[0]; h++) {
        const mjtNum* hinge = b + 1 + mjROD_HINGE_SIZE*h;
        int nr = mj_rodHingeVert(hinge);
        for (int i = 0; i < nr; i++) {
          int si = vslot[m->flex_vertadr[f] + (int)hinge[i]];
          if (si < 0) continue;
          for (int j = 0; j < nr; j++) {
            int sj = vslot[m->flex_vertadr[f] + (int)hinge[j]];
            if (sj >= 0) cand[cadr[si] + ncand[si]++] = sj;
          }
        }
      }
      continue;
    }
    if (flg_bend && m->flex_bendingadr[f] >= 0) {
      for (int e = 0; e < m->flex_edgenum[f]; e++) {
        const int* edge = m->flex_edge + 2*(e + m->flex_edgeadr[f]);
        const int* flap = m->flex_edgeflap + 2*(e + m->flex_edgeadr[f]);
        if (flap[1] == -1) continue;
        int v[4] = {edge[0], edge[1], flap[0], flap[1]};
        for (int i = 0; i < 4; i++) {
          int si = vslot[m->flex_vertadr[f] + v[i]];
          if (si < 0) continue;
          for (int j = 0; j < 4; j++) {
            int sj = vslot[m->flex_vertadr[f] + v[j]];
            if (sj >= 0) cand[cadr[si] + ncand[si]++] = sj;
          }
        }
      }
    }
    if (flg_stretch && m->flex_stiffnessadr[f] >= 0 &&
        m->flex_stiffness[m->flex_stiffnessadr[f]] != 0) {
      const int* elem = m->flex_elem + m->flex_elemdataadr[f];
      for (int t = 0; t < m->flex_elemnum[f]; t++) {
        const int* vert = elem + (dim+1)*t;
        for (int i = 0; i < nvrt; i++) {
          int si = vslot[m->flex_vertadr[f] + vert[i]];
          if (si < 0) continue;
          for (int j = 0; j < nvrt; j++) {
            int sj = vslot[m->flex_vertadr[f] + vert[j]];
            if (sj >= 0) cand[cadr[si] + ncand[si]++] = sj;
          }
        }
      }
    }
  }

  // interp stencils: npe-node element cliques (filling)
  if (Krot) {
    for (int f = 0; f < m->nflex; f++) {
      if (!mjd_flexInterp_processed(m, f)) {
        continue;
      }
      FLEXINTERP_WALK(f, {
        for (int i = 0; i < npe; i++) {
          int si = nslot[m->flex_nodeadr[f] + gindices[i]];
          if (si < 0) continue;
          for (int j = 0; j < npe; j++) {
            int sj = nslot[m->flex_nodeadr[f] + gindices[j]];
            if (sj >= 0) cand[cadr[si] + ncand[si]++] = sj;
          }
        }
      })
    }
  }

  // passive contacts (filling)
  // per vertex: sort by neighbor dofadr, unique -> neighbor lists
  int* nadr = mjSTACKALLOC(d, nvert + 1, int);
  int* neigh = mjSTACKALLOC(d, cadr[nvert] > 0 ? cadr[nvert] : 1, int);
  nadr[0] = 0;
  for (int s = 0; s < nvert; s++) {
    int* c = cand + cadr[s];
    int n = ncand[s];
    // insertion sort by dof address (stencil-local lists are short)
    for (int i = 1; i < n; i++) {
      int key = c[i], kd = vdof[key], j = i - 1;
      while (j >= 0 && vdof[c[j]] > kd) {
        c[j+1] = c[j];
        j--;
      }
      c[j+1] = key;
    }
    int nn = 0;
    for (int i = 0; i < n; i++) {
      if (nn == 0 || neigh[nadr[s] + nn - 1] != c[i]) {
        neigh[nadr[s] + nn++] = c[i];
      }
    }
    nadr[s+1] = nadr[s] + nn;
  }

  // dof-level CSR structure: each vertex contributes 3 rows of 3*nneigh entries
  mju_zeroInt(rownnz, nv);
  int nnz = 0;
  for (int s = 0; s < nvert; s++) {
    int nn = nadr[s+1] - nadr[s];
    for (int k = 0; k < 3; k++) {
      rownnz[vdof[s] + k] = 3*nn;
    }
    nnz += 9*nn;
  }
  if (rowadr) {
    rowadr[0] = 0;
    for (int i = 1; i < nv; i++) {
      rowadr[i] = rowadr[i-1] + rownnz[i-1];
    }
  }

  // phase 1: structure only
  if (!colind) {
    mj_freeStack(d);
    return nnz;
  }

  // column indices (per vertex row: the 3 dofs of each neighbor, neighbor-dof-sorted)
  for (int s = 0; s < nvert; s++) {
    for (int k = 0; k < 3; k++) {
      int adr = rowadr[vdof[s] + k];
      for (int j = nadr[s]; j < nadr[s+1]; j++) {
        for (int c = 0; c < 3; c++) {
          colind[adr++] = vdof[neigh[j]] + c;
        }
      }
    }
  }

  // structure only
  if (!val) {
    mj_freeStack(d);
    return nnz;
  }
  mju_zero(val, nnz);

// block accumulation helper data: find neighbor position by binary search on dofadr
#define FLEXSTIFF_BLOCK(si, sj, pos)          \
  {                                           \
    int lo = nadr[si], hi = nadr[si + 1] - 1; \
    pos = -1;                                 \
    while (lo <= hi) {                        \
      int mid = (lo + hi) / 2;                \
      if (vdof[neigh[mid]] == vdof[sj]) {     \
        pos = mid - nadr[si];                 \
        break;                                \
      }                                       \
      if (vdof[neigh[mid]] < vdof[sj])        \
        lo = mid + 1;                         \
      else                                    \
        hi = mid - 1;                         \
    }                                         \
  }

  // values: bending stencils and Cartesian stretch blocks, rotated into the body frames
  for (int f = 0; f < m->nflex; f++) {
    if (!mjd_flexStiff_active(m, f, flg_bend, flg_stretch)) {
      continue;
    }
    mjtNum scale = s1 + s2*m->flex_damping[f];
    if (!scale) {
      continue;
    }

    // rod hinges: world 3x3 blocks rotated into the body frames
    if (flexRod_active(m, f)) {
      int va = m->flex_vertadr[f];
      mj_markStack((mjData*)d);
      const mjtNum* chain = rodChainWeights(m, (mjData*)d, f);
      for (int h = 0; h < (int)m->flex_bending[m->flex_bendingadr[f]]; h++) {
        int v[4], nr;
        mjtNum K[144];
        if (!rodHingeStiffness(m, d, f, h, s1, s2, chain, v, &nr, K)) {
          continue;
        }
        int ndof = 3*nr;
        for (int i = 0; i < nr; i++) {
          int si = vslot[va + v[i]];
          if (si < 0) continue;
          int bi = m->body_weldid[m->flex_vertbodyid[va + v[i]]];
          for (int j = 0; j < nr; j++) {
            int sj = vslot[va + v[j]];
            if (sj < 0) continue;
            int bj = m->body_weldid[m->flex_vertbodyid[va + v[j]]];
            mjtNum block[9], tmp[9], blkd[9];
            for (int r = 0; r < 3; r++) {
              mji_copy3(block + 3*r, K + ndof*(3*i+r) + 3*j);
            }
            mji_mulMatMat3(tmp, block, d->xmat + 9*bj);
            mji_mulMatTMat3(blkd, d->xmat + 9*bi, tmp);
            int pos;
            FLEXSTIFF_BLOCK(si, sj, pos);
            for (int r = 0; r < 3; r++) {
              mji_addTo3(val + rowadr[vdof[si]+r] + 3*pos, blkd + 3*r);
            }
          }
        }
      }
      mj_freeStack((mjData*)d);
      continue;
    }

    if (flg_bend && m->flex_bendingadr[f] >= 0) {
      const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
      for (int e = 0; e < m->flex_edgenum[f]; e++) {
        const int* edge = m->flex_edge + 2*(e + m->flex_edgeadr[f]);
        const int* flap = m->flex_edgeflap + 2*(e + m->flex_edgeadr[f]);
        if (flap[1] == -1) continue;
        if (!flexBendConvex(b + 17*e)) continue;   // values only: the pattern keeps the stencil
        int v[4] = {edge[0], edge[1], flap[0], flap[1]};
        for (int i = 0; i < 4; i++) {
          int si = vslot[m->flex_vertadr[f] + v[i]];
          if (si < 0) continue;
          int bi = m->body_weldid[m->flex_vertbodyid[m->flex_vertadr[f] + v[i]]];
          const mjtNum* qi = d->xquat + 4*bi;
          for (int j = 0; j < 4; j++) {
            int sj = vslot[m->flex_vertadr[f] + v[j]];
            if (sj < 0) continue;
            mjtNum q = scale*b[17*e + 4*i + j];
            if (!q) continue;
            int bj = m->body_weldid[m->flex_vertbodyid[m->flex_vertadr[f] + v[j]]];
            const mjtNum* qj = d->xquat + 4*bj;
            int pos;
            FLEXSTIFF_BLOCK(si, sj, pos);
            // fast path: when bi and bj share the same body orientation, R_bi^T * R_bj = I3
            if (bi == bj || (qi[0] == qj[0] && qi[1] == qj[1] &&
                             qi[2] == qj[2] && qi[3] == qj[3])) {
              val[rowadr[vdof[si] + 0] + 3*pos + 0] += q;
              val[rowadr[vdof[si] + 1] + 3*pos + 1] += q;
              val[rowadr[vdof[si] + 2] + 3*pos + 2] += q;
            } else {
              mjtNum blkd[9];
              mji_mulMatTMat3(blkd, d->xmat + 9*bi, d->xmat + 9*bj);
              for (int k = 0; k < 3; k++) {
                mji_addToScl3(val + rowadr[vdof[si] + k] + 3*pos, blkd + 3*k, q);
              }
            }
          }
        }
      }
    }

    if (flg_stretch && m->flex_stiffnessadr[f] >= 0 &&
        m->flex_stiffness[m->flex_stiffnessadr[f]] != 0) {
      mj_flexHessian(m, d, f);
      int va = m->flex_vertadr[f], ea = m->flex_edgeadr[f];
      int count = m->flex_vertnum[f] + 2*m->flex_edgenum[f];
      // transform each assembled vertex block into the destination slide axes once
      for (int b=0; b < count; b++) {
        int vi, vj;
        mjtNum block[9];
        if (b < m->flex_vertnum[f]) {
          vi = vj = b;
          const mjtNum* h = d->flexvert_hessian + 6*(va+b);
          block[0] = h[0];
          block[1] = block[3] = h[1];
          block[2] = block[6] = h[2];
          block[4] = h[3];
          block[5] = block[7] = h[4];
          block[8] = h[5];
        } else {
          int id = b-m->flex_vertnum[f], e = ea+id/2, transpose = id%2;
          vi = m->flex_edge[2*e+transpose];
          vj = m->flex_edge[2*e+1-transpose];
          const mjtNum* h = d->flexedge_hessian + 9*e;
          for (int r=0; r < 3; r++) {
            for (int c=0; c < 3; c++) {
              block[3*r+c] = transpose ? h[3*c+r] : h[3*r+c];
            }
          }
        }
        int si = vslot[va+vi], sj = vslot[va+vj];
        if (si < 0 || sj < 0) {
          continue;
        }

        // an unused flex vertex has no diagonal block in the sparse structure
        if (!rownnz[vdof[si]]) {
          continue;
        }

        int bi = m->body_weldid[m->flex_vertbodyid[va+vi]];
        int bj = m->body_weldid[m->flex_vertbodyid[va+vj]];
        mjtNum tmp[9], blkd[9];
        mji_mulMatMat3(tmp, block, d->xmat + 9*bj);
        mji_mulMatTMat3(blkd, d->xmat + 9*bi, tmp);
        int pos;
        FLEXSTIFF_BLOCK(si, sj, pos);
        for (int r=0; r < 3; r++) {
          mji_addToScl3(val + rowadr[vdof[si]+r] + 3*pos, blkd + 3*r, scale);
        }
      }
    }
  }

  // values: interp (corotated K_rot 3x3 node blocks per element). The interp operator's sign
  // convention is opposite to bend/stretch -- FlexBmulAdd calls it with negated scales -- so
  // the negation is folded in here and one CSR replaces all three operators uniformly.
  if (Krot) {
    for (int f = 0; f < m->nflex; f++) {
      if (!mjd_flexInterp_processed(m, f)) {
        continue;
      }
      mjtNum iscale = -(s1 + s2*m->flex_damping[f]);
      if (!iscale) {
        continue;
      }
      FLEXINTERP_WALK(f, {
        int dim_e = 3*npe;
        const mjtNum* kb = Krot + m->flex_stiffnessadr[f] + (size_t)fe*dim_e*dim_e;
        for (int i = 0; i < npe; i++) {
          int si = nslot[m->flex_nodeadr[f] + gindices[i]];
          if (si < 0) continue;
          int bi = m->flex_nodebodyid[m->flex_nodeadr[f] + gindices[i]];
          for (int j = 0; j < npe; j++) {
            int sj = nslot[m->flex_nodeadr[f] + gindices[j]];
            if (sj < 0) continue;
            int bj = m->flex_nodebodyid[m->flex_nodeadr[f] + gindices[j]];
            // extract K_rot 3x3 block for (i,j)
            mjtNum kb_ij[9];
            for (int r = 0; r < 3; r++) {
              for (int c = 0; c < 3; c++) {
                kb_ij[3*r+c] = iscale*kb[(3*i + r)*dim_e + 3*j + c];
              }
            }
            // apply R_bi^T * kb_ij * R_bj to transform to DOF space
            mjtNum tmp[9];
            mjtNum blk[9];
            mji_mulMatMat3(tmp, kb_ij, d->xmat + 9*bj);     // tmp = kb * R_bj
            mji_mulMatTMat3(blk, d->xmat + 9*bi, tmp);      // blk = R_bi^T * tmp
            int pos;
            FLEXSTIFF_BLOCK(si, sj, pos);
            for (int r = 0; r < 3; r++) {
              for (int c = 0; c < 3; c++) {
                val[rowadr[vdof[si] + r] + 3*pos + c] += blk[3*r+c];
              }
            }
          }
        }
      })
    }
  }
  #undef FLEXSTIFF_BLOCK
  #undef FLEXINTERP_WALK

  mj_freeStack(d);
  return nnz;
}




// unpack effective DC motor control inputs [pos, vel, ff, voltage] after delay, clamp, and slew
static void dcmotorInputs(const mjModel* m, const mjData* d, int i, mjtNum u4[4]) {
  int ctrlnum = m->actuator_ctrlnum[i];
  mjtNum u[4] = {0};
  if (ctrlnum > 0) {
    int uadr = m->actuator_ctrladr[i];
    if (m->actuator_delay[i]) {
      const mjtNum* ptr = mj_readCtrl(m, d, i, d->time, u, -1);
      if (ptr) {
        mju_copy(u, ptr, ctrlnum);
      }
    } else {
      mju_copy(u, d->ctrl + uadr, ctrlnum);
    }
    if (!mjDISABLED(mjDSBL_CLAMPCTRL)) {
      for (int k=0; k < ctrlnum; k++) {
        if (m->actuator_ctrllimited[uadr+k]) {
          u[k] = mju_clip(u[k], m->actuator_ctrlrange[2*(uadr+k)],
                          m->actuator_ctrlrange[2*(uadr+k)+1]);
        }
      }
    }
    mjtNum slew_s = m->actuator_dynprm[mjNDYN*i + 7];
    if (slew_s > 0 && m->actuator_actadr[i] >= 0) {
      mjtNum u_prev = d->act[m->actuator_actadr[i]];
      mjtNum slew = slew_s * m->opt.timestep;
      u[0] = mju_clip(u[0], u_prev - slew, u_prev + slew);
    }
  }
  int spec = m->actuator_ctrlspec[i];
  int adr = 0;
  u4[0] = (spec & mjINPUT_POS)     ? u[adr++] : 0;
  u4[1] = (spec & mjINPUT_VEL)     ? u[adr++] : 0;
  u4[2] = (spec & mjINPUT_FF)      ? u[adr++] : 0;
  u4[3] = (spec & mjINPUT_VOLTAGE) ? u[adr]   : 0;
}


// compute DC motor terminal voltage, optionally reporting Vmax controller saturation
static mjtNum dcmotorVoltage(const mjModel* m, const mjData* d, int i, int* v_clamped) {
  mjtNum u4[4];
  dcmotorInputs(m, d, i, u4);
  if (v_clamped) {
    *v_clamped = 0;
  }
  int spec = m->actuator_ctrlspec[i];
  mjtNum voltage = 0;
  if (m->actuator_ctrlnum[i] > 0 && (spec & (mjINPUT_POS | mjINPUT_VEL | mjINPUT_FF))) {
    const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
    const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*i;
    mjDCMotorSlots slots = mj_dcmotorSlots(dynprm, gainprm);
    int oadr = m->actuator_outadr[i];
    mjtNum x_I = (slots.integral >= 0) ? d->act[m->actuator_actadr[i] + slots.integral] : 0;
    mjtNum torque = gainprm[4]*(u4[0] - d->actuator_length[oadr]) +
                    gainprm[6]*(u4[1] - d->actuator_velocity[oadr]) +
                    gainprm[5]*x_I + u4[2];
    voltage = gainprm[0]/gainprm[1] * torque + gainprm[1]*d->actuator_velocity[oadr];
    mjtNum Vmax = gainprm[7];
    if (Vmax > 0 && mju_abs(voltage) >= Vmax) {
      if (v_clamped) {
        *v_clamped = 1;
      }
      voltage = mju_clip(voltage, -Vmax, Vmax);
    }
  }
  return voltage + u4[3];
}


// post-scaling pre-mechanical force of DC motor i (for tendon actfrcrange sum)
static mjtNum dcmotorPreMechForce(const mjModel* m, const mjData* d, int i) {
  int oadr = m->actuator_outadr[i];
  const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*i;
  const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
  mjtNum f = d->actuator_force[oadr];

  // mechanical forces are not added to disabled, plugin, or sleeping actuators (mj_fwdActuation)
  if (mj_actuatorDisabled(m, i) || m->actuator_plugin[i] >= 0 ||
      (mjENABLED(mjENBL_SLEEP) && mj_sleepState(m, d, mjOBJ_ACTUATOR, i) == mjS_ASLEEP)) {
    return f;
  }
  if (biasprm[0] != 0) {
    f -= biasprm[0] * mju_sin(biasprm[1]*d->actuator_length[oadr] + biasprm[2]);
  }
  if (dynprm[5] > 0) {
    mjDCMotorSlots slots = mj_dcmotorSlots(dynprm, m->actuator_gainprm + mjNGAIN*i);
    int adr = m->actuator_actadr[i] + slots.bristle;
    f += dynprm[5]*d->act[adr] + dynprm[6]*d->act_dot[adr];
  }
  return f;
}


// return 1 if tendon total actuator force is clamped by actfrcrange
static int tendonActFrcClamped(const mjModel* m, const mjData* d, int tendon_id) {
  if (!m->tendon_actfrclimited[tendon_id]) {
    return 0;
  }
  const mjtNum* range = m->tendon_actfrcrange + 2*tendon_id;
  mjtNum tendon_force = 0;
  for (int j=0; j < m->nactuator; j++) {
    if (m->actuator_trntype[j] == mjTRN_TENDON && m->actuator_trnid[2*j] == tendon_id) {
      tendon_force += (m->actuator_biastype[j] == mjBIAS_DCMOTOR)
                      ? dcmotorPreMechForce(m, d, j)
                      : d->actuator_force[m->actuator_outadr[j]];
    }
  }

  // relative tolerance for rounding in proportional force scaling (mj_fwdActuation)
#ifdef mjUSESINGLE
  const mjtNum tol = 1e-5f;
#else
  const mjtNum tol = 1e-10;
#endif
  return tendon_force <= range[0] + tol*mju_abs(range[0]) ||
         tendon_force >= range[1] - tol*mju_abs(range[1]);
}


// DC motor saturation flags: 1 electrical force clamped, 2 Vmax saturated, 4 dimax clamped
static int dcmotorSaturation(const mjModel* m, const mjData* d, int i) {
  int flags = 0;
  int v_clamped = 0;
  mjtNum V = dcmotorVoltage(m, d, i, &v_clamped);
  if (v_clamped) {
    flags |= 2;
  }

  // pre-clamp electrical force (before tendon/actuator force clamping and mechanics)
  const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
  const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*i;
  mjtNum K = gainprm[1];
  mjtNum f_elec;
  if (dynprm[0] > 0) {
    int act_adr = m->actuator_actadr[i] + m->actuator_actnum[i] - 1;
    mjtNum act = m->actuator_actearly[i]
                 ? mj_nextActivation(m, d, i, act_adr, d->act_dot[act_adr])
                 : d->act[act_adr];
    f_elec = K * act;
  } else {
    int oadr = m->actuator_outadr[i];
    mjtNum R = mj_dcmotorResistance(m, d, i);
    f_elec = (K / R) * (V - K * d->actuator_velocity[oadr]);
  }

  if (m->actuator_forcelimited[i]) {
    const mjtNum* range = m->actuator_forcerange + 2*i;
    if (f_elec <= range[0] || f_elec >= range[1]) {
      flags |= 1;
    }
  }
  if ((flags & 1) == 0 && m->actuator_trntype[i] == mjTRN_TENDON &&
      tendonActFrcClamped(m, d, m->actuator_trnid[2*i])) {
    flags |= 1;
  }

  if (dynprm[0] > 0 && dynprm[1] > 0) {
    int act_last = m->actuator_actadr[i] + m->actuator_actnum[i] - 1;
    if (mju_abs(d->act_dot[act_last]) >= dynprm[1]) {
      flags |= 4;
    }
  }

  return flags;
}


// return 1 if DC motor i's electrical force is clamped by forcerange or tendon actfrcrange
static int dcmotorElecClamped(const mjModel* m, const mjData* d, int i) {
  return (dcmotorSaturation(m, d, i) & 1) != 0;
}


// per-actuator skip conditions shared by the qDeriv and discrete-metric assemblers:
// disabled, sleeping, or force-clamped actuators contribute no derivative
int mjd_actuatorDerivSkip(const mjModel* m, const mjData* d, int i, int sleep_filter) {
  // skip if disabled
  if (mj_actuatorDisabled(m, i)) {
    return 1;
  }

  // skip if sleeping
  if (sleep_filter && mj_sleepState(m, d, mjOBJ_ACTUATOR, i) == mjS_ASLEEP) {
    return 1;
  }

  // DC motor: mechanical forces (cogging, LuGre) bypass electrical force clamping
  if (m->actuator_biastype[i] == mjBIAS_DCMOTOR) {
    int has_mech = (m->actuator_biasprm[mjNBIAS*i] != 0 ||
                    m->actuator_dynprm[mjNDYN*i + 6] > 0);
    if (!has_mech && dcmotorElecClamped(m, d, i)) {
      return 1;
    }
  } else {
    // skip if force is clamped by forcerange
    if (m->actuator_forcelimited[i]) {
      int oadr = m->actuator_outadr[i];
      const mjtNum* range = m->actuator_forcerange + 2*i;

      // SO3: force is norm-clamped (approximation: saturated force still varies tangentially)
      if (m->actuator_gaintype[i] == mjGAIN_SO3) {
        if (mju_norm3(d->actuator_force + oadr) >= range[1]) {
          return 1;
        }
      } else {
        mjtNum force = d->actuator_force[oadr];
        if (force <= range[0] || force >= range[1]) {
          return 1;
        }
      }
    }

    // skip if tendon total actuator force is clamped by actfrcrange
    if (m->actuator_trntype[i] == mjTRN_TENDON &&
        tendonActFrcClamped(m, d, m->actuator_trnid[2*i])) {
      return 1;
    }
  }

  // skip if joint total actuator force is clamped by actfrcrange
  // (keeping tendon/site actuator contributions on this dof is an approximation)
  if (m->actuator_trntype[i] == mjTRN_JOINT || m->actuator_trntype[i] == mjTRN_JOINTINPARENT) {
    int jnt_id = m->actuator_trnid[2*i];
    if (m->jnt_actfrclimited[jnt_id]) {
      const mjtNum* range = m->jnt_actfrcrange + 2*jnt_id;
      mjtNum jnt_force = d->qfrc_actuator[m->jnt_dofadr[jnt_id]];
      if (jnt_force <= range[0] || jnt_force >= range[1]) {
        return 1;
      }
    }
  }

  return 0;
}


// input of actuator i: ctrl, or (next-)activation
static mjtNum actuatorInput(const mjModel* m, const mjData* d, int i) {
  if (m->actuator_dyntype[i] == mjDYN_NONE) {
    int adr = m->actuator_ctrladr[i];
    mjtNum ctrl;
    if (m->actuator_delay[i]) {
      mjtNum ctrl_buf[4] = {0};
      const mjtNum* ptr = mj_readCtrl(m, d, i, d->time, ctrl_buf, -1);
      ctrl = ptr ? ptr[0] : ctrl_buf[0];
    } else {
      ctrl = d->ctrl[adr];
    }
    if (!mjDISABLED(mjDSBL_CLAMPCTRL) && m->actuator_ctrllimited[adr]) {
      ctrl = mju_clip(ctrl, m->actuator_ctrlrange[2*adr], m->actuator_ctrlrange[2*adr+1]);
    }
    return ctrl;
  }
  int act_adr = m->actuator_actadr[i] + m->actuator_actnum[i] - 1;
  mjtNum act = d->act[act_adr];

  // use next activation if actearly is set (matching forward pass)
  if (m->actuator_actearly[i]) {
    act = mj_nextActivation(m, d, i, act_adr, d->act_dot[act_adr]);
  }
  return act;
}


// d(force)/d(length) of actuator i: all gain and bias types
mjtNum mjd_actuatorLenDeriv(const mjModel* m, const mjData* d, int i) {
  int oadr = m->actuator_outadr[i];
  mjtNum bias_len = 0, gain_len = 0;

  // affine and SO3 servo bias
  if (m->actuator_biastype[i] == mjBIAS_AFFINE || m->actuator_biastype[i] == mjBIAS_SO3) {
    bias_len = (m->actuator_biasprm + mjNBIAS*i)[1];
  }

  // muscle bias: passive force-length curve
  else if (m->actuator_biastype[i] == mjBIAS_MUSCLE) {
    bias_len = mjd_muscleBias_len(d->actuator_length[oadr],
                                  m->actuator_lengthrange+2*oadr,
                                  m->actuator_acc0[oadr],
                                  m->actuator_biasprm + mjNBIAS*i);
  }

  // DC motor cogging torque (mechanical, added after force clamping)
  else if (m->actuator_biastype[i] == mjBIAS_DCMOTOR) {
    const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*i;
    mjtNum A = biasprm[0];
    if (A != 0) {
      mjtNum Np = biasprm[1];
      mjtNum phi = biasprm[2];
      bias_len = A * Np * mju_cos(Np * d->actuator_length[oadr] + phi);
    }
  }

  // affine gain
  if (m->actuator_gaintype[i] == mjGAIN_AFFINE) {
    gain_len = (m->actuator_gainprm + mjNGAIN*i)[1];
  }

  // muscle gain: active force-length curve
  else if (m->actuator_gaintype[i] == mjGAIN_MUSCLE) {
    gain_len = mjd_muscleGain_len(d->actuator_length[oadr],
                                  d->actuator_velocity[oadr],
                                  m->actuator_lengthrange+2*oadr,
                                  m->actuator_acc0[oadr],
                                  m->actuator_gainprm + mjNGAIN*i);
  }

  // DC motor controller position loop
  else if (m->actuator_gaintype[i] == mjGAIN_DCMOTOR) {
    int sat = dcmotorSaturation(m, d, i);
    if ((sat & 1) == 0) {
      const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
      const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*i;
      mjtNum te = dynprm[0];

      int v_clamped = (sat & 2) != 0;

      // controller length derivative dV/dl: torque-space kp through the tau->V map using
      // nameplate resistance; zeroed under Vmax saturation, O(h*ki) integral chain ignored
      mjtNum dVdl = 0;
      if (!v_clamped && (m->actuator_ctrlspec[i] & (mjINPUT_POS | mjINPUT_VEL | mjINPUT_FF))) {
        mjtNum R0 = mju_max(mjMINVAL, gainprm[0]);
        mjtNum K = gainprm[1];  // K > 0 on this path, enforced by the compiler
        dVdl = -gainprm[4]*R0/K;
      }

      if (te > 0) {
        if (m->actuator_actearly[i] && dVdl != 0 && (sat & 4) == 0) {
          // stateful current with actearly: d(K*next_act)/d(length) through act_dot
          mjtNum R = mj_dcmotorResistance(m, d, i);
          mjtNum K = gainprm[1];
          mjtNum s = 1 - mju_exp(-m->opt.timestep / te);
          bias_len += K * dVdl * s / R;
        }
      } else if (dVdl != 0) {
        // stateless: K/R * dVdl = -kp * R0 / R
        mjtNum R = mj_dcmotorResistance(m, d, i);
        mjtNum K = gainprm[1];
        bias_len += K * dVdl / R;
      }
    }
  }

  // force = gain .* [ctrl/act]
  if (gain_len != 0) {
    bias_len += gain_len * actuatorInput(m, d, i);
  }

  return bias_len;
}


// d(force)/d(velocity) of actuator i: all gain and bias types
mjtNum mjd_actuatorVelDeriv(const mjModel* m, const mjData* d, int i) {
  int oadr = m->actuator_outadr[i];
  mjtNum bias_vel = 0, gain_vel = 0;
  int sat_dc = 0;
  if (m->actuator_biastype[i] == mjBIAS_DCMOTOR ||
      m->actuator_gaintype[i] == mjGAIN_DCMOTOR) {
    sat_dc = dcmotorSaturation(m, d, i);
  }

  // affine bias
  if (m->actuator_biastype[i] == mjBIAS_AFFINE) {
    // extract bias info: prm = [const, kp, kv]
    bias_vel = (m->actuator_biasprm + mjNBIAS*i)[2];
  }

  // SO3 geodesic servo: kv term, applied to each output row below
  else if (m->actuator_biastype[i] == mjBIAS_SO3) {
    bias_vel = (m->actuator_biasprm + mjNBIAS*i)[2];
  }

  // DC motor bias (back-EMF)
  else if (m->actuator_biastype[i] == mjBIAS_DCMOTOR) {
    const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
    const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*i;
    if (dynprm[0] <= 0 && (sat_dc & 1) == 0) {
      mjtNum R = mj_dcmotorResistance(m, d, i);
      mjtNum K = gainprm[1];
      bias_vel -= K * K / R;
    }
  }

  // affine gain
  if (m->actuator_gaintype[i] == mjGAIN_AFFINE) {
    // extract bias info: prm = [const, kp, kv]
    gain_vel = (m->actuator_gainprm + mjNGAIN*i)[2];
  }

  // muscle gain
  else if (m->actuator_gaintype[i] == mjGAIN_MUSCLE) {
    gain_vel = mjd_muscleGain_vel(d->actuator_length[oadr],
                                  d->actuator_velocity[oadr],
                                  m->actuator_lengthrange+2*oadr,
                                  m->actuator_acc0[oadr],
                                  m->actuator_gainprm + mjNGAIN*i);
  }

  // DC motor controller damping and LuGre micro-damping
  else if (m->actuator_gaintype[i] == mjGAIN_DCMOTOR) {
    const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
    const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*i;
    mjtNum te = dynprm[0];

    if ((sat_dc & 1) == 0) {
      int v_clamped = (sat_dc & 2) != 0;

      // controller velocity derivative dV/dw: torque-space kd through the tau->V map using
      // nameplate resistance, plus back-EMF compensation K, which cancels the physical -K^2/R
      // back-EMF bias term so net damping is -kd*R0/R; zeroed under Vmax saturation
      mjtNum dVdw = 0;
      if (!v_clamped && (m->actuator_ctrlspec[i] & (mjINPUT_POS | mjINPUT_VEL | mjINPUT_FF))) {
        mjtNum R0 = mju_max(mjMINVAL, gainprm[0]);
        mjtNum K = gainprm[1];  // K > 0 on this path, enforced by the compiler
        dVdw = -gainprm[6]*R0/K + K;
      }

      if (te > 0) {
        if (m->actuator_actearly[i] && (sat_dc & 4) == 0) {
          // stateful current with actearly: d(K*next_act)/dω
          // includes both back-EMF (-K) and controller (dVdw) through act_dot
          mjtNum R = mj_dcmotorResistance(m, d, i);
          mjtNum K = gainprm[1];
          mjtNum s = 1 - mju_exp(-m->opt.timestep / te);
          bias_vel += K * (dVdw - K) * s / R;
        }
      } else if (dVdw != 0) {
        // stateless: controller terms only (back-EMF handled in bias block)
        mjtNum R = mj_dcmotorResistance(m, d, i);
        mjtNum K = gainprm[1];
        bias_vel += K * dVdw / R;
      }
    }

    // LuGre: force includes -sigma1*z_dot, z_dot = a*z + v
    // d(sigma1*z_dot)/dv = sigma1*(da/dv*z + 1), ignoring higher-order da/dv*z
    mjtNum sigma1 = dynprm[6];
    if (sigma1 > 0) {
      bias_vel -= sigma1;
    }
  }

  // force = gain .* [ctrl/act]
  if (gain_vel != 0) {
    bias_vel += gain_vel * actuatorInput(m, d, i);
  }

  return bias_vel;
}


// add (d qfrc_actuator / d qvel) to qDeriv
void mjd_actuator_vel(const mjModel* m, mjData* d) {
  int nactuator = m->nactuator;
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

  // disabled: nothing to add
  if (mjDISABLED(mjDSBL_ACTUATION)) {
    return;
  }

  // process actuators
  for (int i=0; i < nactuator; i++) {
    if (mjd_actuatorDerivSkip(m, d, i, sleep_filter)) {
      continue;
    }

    // combined scalar derivative
    mjtNum bias_vel = mjd_actuatorVelDeriv(m, d, i);

    // add, once per output row
    if (bias_vel != 0) {
      int oadr = m->actuator_outadr[i];
      for (int k=0; k < m->actuator_outnum[i]; k++) {
        addJTBJSparse(m, d, d->qDeriv, d->actuator_moment, &bias_vel, 1, oadr+k,
                      d->moment_rownnz, d->moment_rowadr, d->moment_colind);
      }
    }
  }
}


//----------------- utilities for ellipsoid-based fluid force derivatives --------------------------

static inline mjtNum pow2(const mjtNum val) {
  return val*val;
}


static inline mjtNum ellipsoid_max_moment(const mjtNum size[3], const int dir) {
  const mjtNum d0 = size[dir];
  const mjtNum d1 = size[(dir+1) % 3];
  const mjtNum d2 = size[(dir+2) % 3];
  return 8.0/15.0 * mjPI * d0 * pow2(pow2(mju_max(d1, d2)));
}


// add 3x3 matrix D to one of the four quadrants of the 6x6 matrix B
//   row_quad and col_quad should be either 0 or 1 (not checked)
static void addToQuadrant(mjtNum* restrict B, const mjtNum D[9], int col_quad, int row_quad) {
  int r = 3*row_quad, c = 3*col_quad;
  B[6*(c+0) + r+0] += D[0];
  B[6*(c+0) + r+1] += D[1];
  B[6*(c+0) + r+2] += D[2];
  B[6*(c+1) + r+0] += D[3];
  B[6*(c+1) + r+1] += D[4];
  B[6*(c+1) + r+2] += D[5];
  B[6*(c+2) + r+0] += D[6];
  B[6*(c+2) + r+1] += D[7];
  B[6*(c+2) + r+2] += D[8];
}


//----------------- components of ellipsoid-based fluid force derivatives --------------------------

// forces due to fluid mass moving with the body, B is 6x6
static void mjd_addedMassForces(
  mjtNum* restrict B, const mjtNum local_vels[6], const mjtNum fluid_density,
  const mjtNum virtual_mass[3], const mjtNum virtual_inertia[3]) {
  const mjtNum lin_vel[3] = {local_vels[3], local_vels[4], local_vels[5]};
  const mjtNum ang_vel[3] = {local_vels[0], local_vels[1], local_vels[2]};
  const mjtNum virtual_lin_mom[3] = {
    fluid_density * virtual_mass[0] * lin_vel[0],
    fluid_density * virtual_mass[1] * lin_vel[1],
    fluid_density * virtual_mass[2] * lin_vel[2]
  };
  const mjtNum virtual_ang_mom[3] = {
    fluid_density * virtual_inertia[0] * ang_vel[0],
    fluid_density * virtual_inertia[1] * ang_vel[1],
    fluid_density * virtual_inertia[2] * ang_vel[2]
  };
  mjtNum Da[9];
  mjtNum Db[9];

  // force[:3] += cross(virtual_ang_mom, ang_vel)
  mjd_cross(virtual_ang_mom, ang_vel, Da, Db);
  addToQuadrant(B, Db, 0, 0);
  for (int i=0; i < 9; ++i) {
    Da[i] *= fluid_density * virtual_inertia[i % 3];
  }
  addToQuadrant(B, Da, 0, 0);

  // force[:3] += cross(virtual_lin_mom, lin_vel)
  mjd_cross(virtual_lin_mom, lin_vel, Da, Db);
  addToQuadrant(B, Db, 0, 1);
  for (int i=0; i < 9; ++i) {
    Da[i] *= fluid_density * virtual_mass[i % 3];
  }
  addToQuadrant(B, Da, 0, 1);

  // force[3:] += cross(virtual_lin_mom, ang_vel)
  mjd_cross(virtual_lin_mom, ang_vel, Da, Db);
  addToQuadrant(B, Db, 1, 0);
  for (int i=0; i < 9; ++i) {
    Da[i] *= fluid_density * virtual_mass[i % 3];
  }
  addToQuadrant(B, Da, 1, 1);
}


// torque due to motion in the fluid, D is 3x3
static inline void mjd_viscous_torque(
  mjtNum* restrict D, const mjtNum lvel[6], const mjtNum fluid_density,
  const mjtNum fluid_viscosity, const mjtNum size[3],
  const mjtNum slender_drag_coef, const mjtNum ang_drag_coef) {
  const mjtNum d_max = mju_max(mju_max(size[0], size[1]), size[2]);
  const mjtNum d_min = mju_min(mju_min(size[0], size[1]), size[2]);
  const mjtNum d_mid = size[0] + size[1] + size[2] - d_max - d_min;
  // viscous force and torque in Stokes flow, analytical for spherical bodies
  const mjtNum eq_sphere_D = 2.0/3.0 * (size[0] + size[1] + size[2]);
  const mjtNum lin_visc_torq_coef = mjPI * eq_sphere_D*eq_sphere_D*eq_sphere_D;

  // moments of inertia used to compute angular quadratic drag
  const mjtNum I_max = 8.0/15.0 * mjPI * d_mid * (d_max*d_max)*(d_max*d_max);
  const mjtNum II[3] = {
    ellipsoid_max_moment(size, 0),
    ellipsoid_max_moment(size, 1),
    ellipsoid_max_moment(size, 2)
  };
  const mjtNum x = lvel[0], y = lvel[1], z = lvel[2];
  const mjtNum mom_coef[3] = {
    ang_drag_coef*II[0] + slender_drag_coef*(I_max - II[0]),
    ang_drag_coef*II[1] + slender_drag_coef*(I_max - II[1]),
    ang_drag_coef*II[2] + slender_drag_coef*(I_max - II[2])
  };
  const mjtNum mom_visc[3] = {
    x * mom_coef[0],
    y * mom_coef[1],
    z * mom_coef[2]
  };
  const mjtNum density = fluid_density / mju_max(mjMINVAL, mju_norm3(mom_visc));

  // -density * [x, y, z] * mom_coef^2
  const mjtNum mom_sq[3] = {
    -density * x * mom_coef[0] * mom_coef[0],
    -density * y * mom_coef[1] * mom_coef[1],
    -density * z * mom_coef[2] * mom_coef[2]
  };
  const mjtNum lin_coef = fluid_viscosity * lin_visc_torq_coef;

  // initialize
  mju_zero(D, 9);

  // set diagonal
  D[0] = D[4] = D[8] = x*mom_sq[0] + y*mom_sq[1] + z*mom_sq[2] - lin_coef;

  // add outer product
  mju_addToScl3(D, mom_sq, x);
  mju_addToScl3(D+3, mom_sq, y);
  mju_addToScl3(D+6, mom_sq, z);
}


// drag due to motion in the fluid, D is 3x3
static inline void mjd_viscous_drag(
  mjtNum* restrict D, const mjtNum lvel[6], const mjtNum fluid_density,
  const mjtNum fluid_viscosity, const mjtNum size[3],
  const mjtNum blunt_drag_coef, const mjtNum slender_drag_coef) {
  const mjtNum d_max = mju_max(mju_max(size[0], size[1]), size[2]);
  const mjtNum d_min = mju_min(mju_min(size[0], size[1]), size[2]);
  const mjtNum d_mid = size[0] + size[1] + size[2] - d_max - d_min;
  // viscous force and torque in Stokes flow, analytical for spherical bodies
  const mjtNum eq_sphere_D = 2.0/3.0 * (size[0] + size[1] + size[2]);
  const mjtNum A_max = mjPI * d_max * d_mid;

  const mjtNum inv_dmax = d_max > mjMINVAL ? 1.0 / d_max : 0.0;
  const mjtNum s0 = size[0] * inv_dmax;
  const mjtNum s1 = size[1] * inv_dmax;
  const mjtNum s2 = size[2] * inv_dmax;

  const mjtNum norm = mju_norm3(lvel+3);
  const mjtNum inv_norm = norm > mjMINVAL ? 1.0 / norm : 0.0;
  const mjtNum x = lvel[3] * inv_norm;
  const mjtNum y = lvel[4] * inv_norm;
  const mjtNum z = lvel[5] * inv_norm;
  const mjtNum xx = x*x, yy = y*y, zz = z*z, xy=x*y, yz=y*z, xz=x*z;

  const mjtNum a_raw = pow2(s1 * s2);
  const mjtNum b_raw = pow2(s2 * s0);
  const mjtNum c_raw = pow2(s0 * s1);
  const mjtNum proj_num = a_raw * xx + b_raw * yy + c_raw * zz;
  const mjtNum inv_proj_num = proj_num > mjMINVAL ? 1.0 / proj_num : 0.0;
  const mjtNum a = a_raw * inv_proj_num;
  const mjtNum b = b_raw * inv_proj_num;
  const mjtNum c = c_raw * inv_proj_num;
  const mjtNum aa = a*a, bb = b*b, cc = c*c;

  const mjtNum proj_denom = aa*xx + bb*yy + cc*zz;
  const mjtNum area_scale = d_max * d_max * mju_sqrt(proj_num);
  const mjtNum dA_coef = proj_denom > mjMINVAL ?
                         mjPI * area_scale / mju_sqrt(proj_denom) : 0.0;

  const mjtNum A_proj = mjPI * area_scale * mju_sqrt(proj_denom);

  const mjtNum lin_coef = fluid_viscosity * 3.0 * mjPI * eq_sphere_D;
  const mjtNum quad_coef = fluid_density * (
    A_proj*blunt_drag_coef + slender_drag_coef*(A_max - A_proj));
  const mjtNum Aproj_coef = fluid_density * norm * (blunt_drag_coef - slender_drag_coef);

  const mjtNum dAproj_dv[3] = {
    Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),
    Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),
    Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b))
  };

  // outer product
  D[0] = xx;  D[1] = xy;  D[2] = xz;
  D[3] = xy;  D[4] = yy;  D[5] = yz;
  D[6] = xz;  D[7] = yz;  D[8] = zz;

  // diag(D) += dot([x y z], [x y z])
  mjtNum inner = xx + yy + zz;
  D[0] += inner;
  D[4] += inner;
  D[8] += inner;

  // scale by -quad_coef*norm
  mju_scl(D, D, -quad_coef*norm, 9);

  // D += outer_product(-[x y z], dAproj_dv)
  mju_addToScl3(D+0, dAproj_dv, -x);
  mju_addToScl3(D+3, dAproj_dv, -y);
  mju_addToScl3(D+6, dAproj_dv, -z);

  // diag(D) -= lin_coef
  D[0] -= lin_coef;
  D[4] -= lin_coef;
  D[8] -= lin_coef;
}


// Kutta lift due to motion in the fluid, D is 3x3
static inline void mjd_kutta_lift(
  mjtNum* restrict D, const mjtNum lvel[6], const mjtNum fluid_density,
  const mjtNum size[3], const mjtNum kutta_lift_coef) {
  const mjtNum d_max = mju_max(mju_max(size[0], size[1]), size[2]);
  const mjtNum inv_dmax = d_max > mjMINVAL ? 1.0 / d_max : 0.0;
  const mjtNum s0 = size[0] * inv_dmax;
  const mjtNum s1 = size[1] * inv_dmax;
  const mjtNum s2 = size[2] * inv_dmax;

  const mjtNum norm = mju_norm3(lvel+3);
  const mjtNum inv_norm = norm > mjMINVAL ? 1.0 / norm : 0.0;
  const mjtNum x = lvel[3] * inv_norm;
  const mjtNum y = lvel[4] * inv_norm;
  const mjtNum z = lvel[5] * inv_norm;
  const mjtNum xx = x*x, yy = y*y, zz = z*z, xy=x*y, yz=y*z, xz=x*z;

  const mjtNum a_raw = pow2(s1 * s2);
  const mjtNum b_raw = pow2(s2 * s0);
  const mjtNum c_raw = pow2(s0 * s1);
  const mjtNum proj_num = a_raw * xx + b_raw * yy + c_raw * zz;
  const mjtNum inv_proj_num = proj_num > mjMINVAL ? 1.0 / proj_num : 0.0;
  const mjtNum a = a_raw * inv_proj_num;
  const mjtNum b = b_raw * inv_proj_num;
  const mjtNum c = c_raw * inv_proj_num;
  const mjtNum aa = a*a, bb = b*b, cc = c*c;

  const mjtNum proj_denom = aa * xx + bb * yy + cc * zz;
  const mjtNum area_scale = d_max * d_max * mju_sqrt(proj_num);
  const mjtNum df_denom = proj_denom > mjMINVAL ?
    mjPI * kutta_lift_coef * fluid_density * area_scale * norm / mju_sqrt(proj_denom) : 0.0;

  const mjtNum dfx_coef = yy * (a - b) + zz * (a - c);
  const mjtNum dfy_coef = xx * (b - a) + zz * (b - c);
  const mjtNum dfz_coef = xx * (c - a) + yy * (c - b);
  const mjtNum proj_term = proj_denom > mjMINVAL ? 1.0 / proj_denom : 0.0;
  const mjtNum cos_term = 1.0;

  // cosA = proj_num/(norm*proj_denom), A_proj = pi*sqrt(proj_denom/proj_num)
  // F = cosA * A_proj * (([a,b,c] * vel) \times vel) \times vel
  // derivative obtained with SymPy

  D[0] = a-a;  D[1] = b-a;  D[2] = c-a;
  D[3] = a-b;  D[4] = b-b;  D[5] = c-b;
  D[6] = a-c;  D[7] = b-c;  D[8] = c-c;
  mju_scl(D, D, 2.0, 9);

  const mjtNum inner_term[3] = {
    aa * proj_term - a + cos_term,
    bb * proj_term - b + cos_term,
    cc * proj_term - c + cos_term
  };
  mju_addToScl3(D + 0, inner_term, dfx_coef);
  mju_addToScl3(D + 3, inner_term, dfy_coef);
  mju_addToScl3(D + 6, inner_term, dfz_coef);

  D[0] *= xx;  D[1] *= xy;  D[2] *= xz;
  D[3] *= xy;  D[4] *= yy;  D[5] *= yz;
  D[6] *= xz;  D[7] *= yz;  D[8] *= zz;

  D[0] -= dfx_coef;
  D[4] -= dfy_coef;
  D[8] -= dfz_coef;

  mju_scl(D, D, df_denom, 9);
}


// Magnus force due to motion in the fluid, B is 6x6
static inline void mjd_magnus_force(
  mjtNum* restrict B, const mjtNum lvel[6], const mjtNum fluid_density,
  const mjtNum size[3], const mjtNum magnus_lift_coef) {
  const mjtNum volume = 4.0/3.0 * mjPI * size[0] * size[1] * size[2];

  // magnus_coef = magnus_lift_coef * fluid_density * volume
  const mjtNum magnus_coef = magnus_lift_coef * fluid_density * volume;

  mjtNum D_lin[9], D_ang[9];

  // premultiply by magnus_coef
  const mjtNum lin_vel[3] = {
    magnus_coef * lvel[3], magnus_coef * lvel[4], magnus_coef * lvel[5]
  };
  const mjtNum ang_vel[3] = {
    magnus_coef * lvel[0], magnus_coef * lvel[1], magnus_coef * lvel[2]
  };

  // force[3:] += magnus_coef * cross(ang_vel, lin_vel)
  mjd_cross(ang_vel, lin_vel, D_ang, D_lin);

  addToQuadrant(B, D_ang, 1, 0);
  addToQuadrant(B, D_lin, 1, 1);
}


//----------------- fluid force derivatives, ellipsoid and inertia-box models ----------------------

// fluid forces based on ellipsoid approximation
void mjd_ellipsoidFluid(const mjModel* m, mjData* d, mjtNum* res, int bodyid,
                        int flg_dragonly) {
  mj_markStack(d);

  int nv = m->nv;
  int nnz = nv;
  int rownnz[6], rowadr[6];
  mjtNum* J = mjSTACKALLOC(d, 6*nv, mjtNum);
  mjtNum* tmp = mjSTACKALLOC(d, 3*nv, mjtNum);
  int* colind = mjSTACKALLOC(d, 6*nv, int);
  int* colind_compressed = mjSTACKALLOC(d, 6*nv, int);

  mjtNum lvel[6], wind[6], lwind[6];
  mjtNum geom_interaction_coef, magnus_lift_coef, kutta_lift_coef;
  mjtNum semiaxes[3], virtual_mass[3], virtual_inertia[3];
  mjtNum blunt_drag_coef, slender_drag_coef, ang_drag_coef;

  if (mj_isSparse(m)) {
    // get sparse body Jacobian structure
    nnz = mj_bodyChain(m, bodyid, colind);

    // prepare rownnz, rowadr, colind for all 6 rows
    for (int i=0; i < 6; i++) {
      rownnz[i] = nnz;
      rowadr[i] = i == 0 ? 0 : rowadr[i-1] + nnz;
      for (int k=0; k < nnz; k++) {
        colind_compressed[i*nnz+k] = colind[k];
      }
    }
  }

  for (int j=0; j < m->body_geomnum[bodyid]; j++) {
    const int geomid = m->body_geomadr[bodyid] + j;

    mju_geomSemiAxes(semiaxes, m->geom_size + 3*geomid, m->geom_type[geomid]);

    readFluidGeomInteraction(
      m->geom_fluid + mjNFLUID*geomid, &geom_interaction_coef,
      &blunt_drag_coef, &slender_drag_coef, &ang_drag_coef,
      &kutta_lift_coef, &magnus_lift_coef,
      virtual_mass, virtual_inertia);

    // scales all forces, read from MJCF as boolean (0.0 or 1.0)
    if (geom_interaction_coef == 0.0) {
      continue;
    }

    // map from CoM-centered to local body-centered 6D velocity
    mj_objectVelocity(m, d, mjOBJ_GEOM, geomid, lvel, 1);
    // compute wind in local coordinates
    mju_zero(wind, 6);
    mju_copy3(wind+3, m->opt.wind);
    mju_transformSpatial(lwind, wind, 0,
                         d->geom_xpos + 3*geomid,  // Frame of ref's origin.
                         d->subtree_com + 3*m->body_rootid[bodyid],
                         d->geom_xmat + 9*geomid);  // Frame of ref's orientation.
    // subtract translational component from grom velocity
    mju_subFrom3(lvel+3, lwind+3);

    // get geom global Jacobian: rotation then translation
    if (mj_isSparse(m)) {
      mj_jacSparse(m, d, J+3*nnz, J, d->geom_xpos+3*geomid, m->geom_bodyid[geomid], nnz, colind,
                   /*flg_skipcommon=*/0);
    } else {
      mj_jacGeom(m, d, J+3*nv, J, geomid);
    }

    // rotate (compressed) Jacobian to local frame
    mju_mulMatTMat(tmp, d->geom_xmat+9*geomid, J, 3, 3, nnz);
    mju_copy(J, tmp, 3*nnz);
    mju_mulMatTMat(tmp, d->geom_xmat+9*geomid, J+3*nnz, 3, 3, nnz);
    mju_copy(J+3*nnz, tmp, 3*nnz);

    mjtNum B[36], D[9];
    mju_zero(B, 36);

    // lift and added-mass terms: asymmetric, excluded from the drag-only (metric) variant
    if (!flg_dragonly) {
      mjd_magnus_force(B, lvel, m->opt.density, semiaxes, magnus_lift_coef);

      mjd_kutta_lift(D, lvel, m->opt.density, semiaxes, kutta_lift_coef);
      addToQuadrant(B, D, 1, 1);
    }

    mjd_viscous_drag(D, lvel, m->opt.density, m->opt.viscosity, semiaxes,
                     blunt_drag_coef, slender_drag_coef);
    addToQuadrant(B, D, 1, 1);

    mjd_viscous_torque(D, lvel, m->opt.density, m->opt.viscosity, semiaxes,
                       slender_drag_coef, ang_drag_coef);
    addToQuadrant(B, D, 0, 0);

    if (!flg_dragonly) {
      mjd_addedMassForces(B, lvel, m->opt.density, virtual_mass, virtual_inertia);
    }

    // make B symmetric for the metric, or if integrator is IMPLICITFAST,
    // except for free rigid subtrees, including fluid geoms on fixed descendants
    if (flg_dragonly ||
        (m->opt.integrator == mjINT_IMPLICITFAST && !mj_isFreeBody(m, m->body_rootid[bodyid]))) {
      mju_symmetrize(B, B, 6);
    }

    if (mj_isSparse(m)) {
      addJTBJSparse(m, d, res, J, B, 6, 0, rownnz, rowadr, colind_compressed);
    } else {
      addJTBJ(m, d, res, J, B, 6);
    }
  }

  mj_freeStack(d);
}


// fluid forces based on inertia-box approximation
void mjd_inertiaBoxFluid(const mjModel* m, mjData* d, mjtNum* res, int i) {
  mj_markStack(d);

  int nv = m->nv;
  int rownnz[6], rowadr[6];
  mjtNum* J = mjSTACKALLOC(d, 6*nv, mjtNum);
  mjtNum* tmp = mjSTACKALLOC(d, 3*nv, mjtNum);
  int* colind = mjSTACKALLOC(d, 6*nv, int);

  mjtNum lvel[6], wind[6], lwind[6], box[3], B;
  mjtNum* inertia = m->body_inertia + 3*i;

  // equivalent inertia box
  box[0] = mju_sqrt(mju_max(mjMINVAL,
                            (inertia[1] + inertia[2] - inertia[0])) / m->body_mass[i] * 6.0);
  box[1] = mju_sqrt(mju_max(mjMINVAL,
                            (inertia[0] + inertia[2] - inertia[1])) / m->body_mass[i] * 6.0);
  box[2] = mju_sqrt(mju_max(mjMINVAL,
                            (inertia[0] + inertia[1] - inertia[2])) / m->body_mass[i] * 6.0);

  // map from CoM-centered to local body-centered 6D velocity
  mj_objectVelocity(m, d, mjOBJ_BODY, i, lvel, 1);

  // compute wind in local coordinates
  mju_zero(wind, 6);
  mju_copy3(wind+3, m->opt.wind);
  mju_transformSpatial(lwind, wind, 0, d->xipos+3*i,
                       d->subtree_com+3*m->body_rootid[i], d->ximat+9*i);

  // subtract translational component from body velocity
  mju_subFrom3(lvel+3, lwind+3);

  // init with dense
  int nnz = nv;

  // sparse Jacobian
  if (mj_isSparse(m)) {
    // get sparse body Jacobian structure
    nnz = mj_bodyChain(m, i, colind);

    // get sparse jacBodyCom
    mj_jacSparse(m, d, J+3*nnz, J, d->xipos+3*i, i, nnz, colind, /*flg_skipcommon=*/0);

    // prepare rownnz, rowadr, colind for all 6 rows
    rownnz[0] = nnz;
    rowadr[0] = 0;
    for (int j=1; j < 6; j++) {
      rownnz[j] = nnz;
      rowadr[j] = rowadr[j-1] + nnz;
      for (int k=0; k < nnz; k++) {
        colind[j*nnz+k] = colind[k];
      }
    }
  }

  // dense Jacobian
  else {
    mj_jacBodyCom(m, d, J+3*nv, J, i);
  }

  // rotate (compressed) Jacobian to local frame
  mju_mulMatTMat(tmp, d->ximat+9*i, J, 3, 3, nnz);
  mju_copy(J, tmp, 3*nnz);
  mju_mulMatTMat(tmp, d->ximat+9*i, J+3*nnz, 3, 3, nnz);
  mju_copy(J+3*nnz, tmp, 3*nnz);

  // add viscous force and torque
  if (m->opt.viscosity > 0) {
    // diameter of sphere approximation
    mjtNum diam = (box[0] + box[1] + box[2])/3.0;

    // mju_scl3(lfrc, lvel, -mjPI*diam*diam*diam*m->opt.viscosity)
    B = -mjPI*diam*diam*diam*m->opt.viscosity;
    for (int j=0; j < 3; j++) {
      if (mj_isSparse(m)) {
        addJTBJSparse(m, d, res, J, &B, 1, j, rownnz, rowadr, colind);
      } else {
        addJTBJ(m, d, res, J+j*nv, &B, 1);
      }
    }

    // mju_scl3(lfrc+3, lvel+3, -3.0*mjPI*diam*m->opt.viscosity);
    B = -3.0*mjPI*diam*m->opt.viscosity;
    for (int j=0; j < 3; j++) {
      if (mj_isSparse(m)) {
        addJTBJSparse(m, d, res, J, &B, 1, 3+j, rownnz, rowadr, colind);
      } else {
        addJTBJ(m, d, res, J+3*nv+j*nv, &B, 1);
      }
    }
  }

  // add lift and drag force and torque
  if (m->opt.density > 0) {
    // lfrc[0] -= m->opt.density*box[0]*(box[1]*box[1]*box[1]*box[1]+box[2]*box[2]*box[2]*box[2])*
    //            mju_abs(lvel[0])*lvel[0]/64.0;
    B = -m->opt.density*box[0]*(box[1]*box[1]*box[1]*box[1]+box[2]*box[2]*box[2]*box[2])*
        2*mju_abs(lvel[0])/64.0;
    if (mj_isSparse(m)) {
      addJTBJSparse(m, d, res, J, &B, 1, 0, rownnz, rowadr, colind);
    } else {
      addJTBJ(m, d, res, J, &B, 1);
    }

    // lfrc[1] -= m->opt.density*box[1]*(box[0]*box[0]*box[0]*box[0]+box[2]*box[2]*box[2]*box[2])*
    //            mju_abs(lvel[1])*lvel[1]/64.0;
    B = -m->opt.density*box[1]*(box[0]*box[0]*box[0]*box[0]+box[2]*box[2]*box[2]*box[2])*
        2*mju_abs(lvel[1])/64.0;
    if (mj_isSparse(m)) {
      addJTBJSparse(m, d, res, J, &B, 1, 1, rownnz, rowadr, colind);
    } else {
      addJTBJ(m, d, res, J+nv, &B, 1);
    }

    // lfrc[2] -= m->opt.density*box[2]*(box[0]*box[0]*box[0]*box[0]+box[1]*box[1]*box[1]*box[1])*
    //            mju_abs(lvel[2])*lvel[2]/64.0;
    B = -m->opt.density*box[2]*(box[0]*box[0]*box[0]*box[0]+box[1]*box[1]*box[1]*box[1])*
        2*mju_abs(lvel[2])/64.0;
    if (mj_isSparse(m)) {
      addJTBJSparse(m, d, res, J, &B, 1, 2, rownnz, rowadr, colind);
    } else {
      addJTBJ(m, d, res, J+2*nv, &B, 1);
    }

    // lfrc[3] -= 0.5*m->opt.density*box[1]*box[2]*mju_abs(lvel[3])*lvel[3];
    B = -0.5*m->opt.density*box[1]*box[2]*2*mju_abs(lvel[3]);
    if (mj_isSparse(m)) {
      addJTBJSparse(m, d, res, J, &B, 1, 3, rownnz, rowadr, colind);
    } else {
      addJTBJ(m, d, res, J+3*nv, &B, 1);
    }

    // lfrc[4] -= 0.5*m->opt.density*box[0]*box[2]*mju_abs(lvel[4])*lvel[4];
    B = -0.5*m->opt.density*box[0]*box[2]*2*mju_abs(lvel[4]);
    if (mj_isSparse(m)) {
      addJTBJSparse(m, d, res, J, &B, 1, 4, rownnz, rowadr, colind);
    } else {
      addJTBJ(m, d, res, J+4*nv, &B, 1);
    }

    // lfrc[5] -= 0.5*m->opt.density*box[0]*box[1]*mju_abs(lvel[5])*lvel[5];
    B = -0.5*m->opt.density*box[0]*box[1]*2*mju_abs(lvel[5]);
    if (mj_isSparse(m)) {
      addJTBJSparse(m, d, res, J, &B, 1, 5, rownnz, rowadr, colind);
    } else {
      addJTBJ(m, d, res, J+5*nv, &B, 1);
    }
  }

  mj_freeStack(d);
}


//------------------------- derivatives of passive forces ------------------------------------------

// add (d qfrc_passive / d qvel) to qDeriv
void mjd_passive_vel(const mjModel* m, mjData* d) {
  // all disabled: nothing to add
  if (mjDISABLED(mjDSBL_SPRING) && mjDISABLED(mjDSBL_DAMPER)) {
    return;
  }

  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;

  // fluid drag model, either body-level (inertia box) or geom-level (ellipsoid)
  if (m->opt.viscosity > 0 || m->opt.density > 0) {
    for (int b=0; b < nbody; b++) {
      int i = sleep_filter ? d->body_awake_ind[b] : b;

      if (m->body_mass[i] < mjMINVAL) {
        continue;
      }

      int use_ellipsoid_model = 0;
      // if any child geom uses the ellipsoid model, inertia-box model is disabled for parent body
      for (int j=0; j < m->body_geomnum[i] && use_ellipsoid_model == 0; j++) {
        const int geomid = m->body_geomadr[i] + j;
        use_ellipsoid_model += (m->geom_fluid[mjNFLUID*geomid] > 0);
      }
      if (use_ellipsoid_model) {
        mjd_ellipsoidFluid(m, d, d->qDeriv, i, /*flg_dragonly=*/0);
      } else {
        mjd_inertiaBoxFluid(m, d, d->qDeriv, i);
      }
    }
  }

  // disabled: nothing to add
  if (mjDISABLED(mjDSBL_DAMPER)) {
    return;
  }

  // dof damping
  int nv = m->nv;
  int nv_awake = sleep_filter ? d->nv_awake : nv;
  for (int j = 0; j < nv_awake; j++) {
    int i = sleep_filter ? d->dof_awake_ind[j] : j;
    mjtNum v = d->qvel[i];
    mjtNum poly[mjNPOLY];
    mju_copy(poly, m->dof_dampingpoly + mjNPOLY*i, mjNPOLY);
    mjtNum damping = m->dof_damping[i] + mj_actuatorDamping(m, mjOBJ_JOINT, m->dof_jntid[i], poly);
    int adr = m->D_rowadr[i] + m->D_diag[i];
    d->qDeriv[adr] -= mjd_xPolyForce(damping, poly, v, mjNPOLY, 1);
  }

  // flex_edgedamping is intentionally omitted: qDeriv uses tree-only D sparsity, which drops
  // off-diagonal blocks between flexcomp vertex trees and would break translational invariance

  // tendon damping
  int ntendon = m->ntendon;
  for (int i=0; i < ntendon; i++) {
    // skip tendon in one or two sleeping trees
    if (sleep_filter) {
      int treenum = m->tendon_treenum[i];
      int id1 = m->tendon_treeid[2*i];
      if (treenum == 1 && !d->tree_awake[id1]) continue;
      int id2 = m->tendon_treeid[2*i+1];
      if (treenum == 2 && !d->tree_awake[id1] && !d->tree_awake[id2]) continue;
    }

    mjtNum v = d->ten_velocity[i];
    mjtNum poly[mjNPOLY];
    mju_copy(poly, m->tendon_dampingpoly+mjNPOLY*i, mjNPOLY);
    mjtNum damping = m->tendon_damping[i] + mj_actuatorDamping(m, mjOBJ_TENDON, i, poly);
    mjtNum B = -mjd_xPolyForce(damping, poly, v, mjNPOLY, 1);

    if (!B) {
      continue;
    }

    // add sparse
    addJTBJSparse(m, d, d->qDeriv, d->ten_J, &B, 1, i,
                  m->ten_J_rownnz, m->ten_J_rowadr, m->ten_J_colind);
  }
}


//------------------------- main entry points ------------------------------------------------------

// analytical derivative of smooth forces w.r.t velocities:
//   d->qDeriv = d (qfrc_actuator + qfrc_passive - [qfrc_bias]) / d qvel
void mjd_smooth_vel(const mjModel* m, mjData* d, int flg_bias) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;

  // clear qDeriv
  if (!sleep_filter) {
    mju_zero(d->qDeriv, m->nD);
  } else {
    mju_zeroSparse(d->qDeriv, m->D_rownnz, m->D_rowadr, d->dof_awake_ind, d->nv_awake);
  }

  // qDeriv += d qfrc_actuator / d qvel
  mjd_actuator_vel(m, d);

  // qDeriv += d qfrc_passive / d qvel
  mjd_passive_vel(m, d);

  // qDeriv -= d qfrc_bias / d qvel; optional
  if (flg_bias) {
    mjd_rne_vel(m, d);
  }
}
