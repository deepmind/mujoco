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

#include "engine/engine_derivative_fd.h"

#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_core_util.h"
#include "engine/engine_forward.h"
#include "engine/engine_inverse.h"
#include "engine/engine_ipc.h"
#include "engine/engine_memory.h"
#include "engine/engine_macro.h"
#include "engine/engine_sensor.h"
#include "engine/engine_support.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"



//--------------------------- finite-differencing utility functions --------------------------------

// get state=[qpos; qvel; act] and optionally sensordata
static void getState(const mjModel* m, const mjData* d, mjtNum* state, mjtNum* sensordata) {
  mj_getState(m, d, state, mjSTATE_PHYSICS);
  if (sensordata) {
    mju_copy(sensordata, d->sensordata, m->nsensordata);
  }
}


// dx = (x2 - x1) / h
static void diff(mjtNum* restrict dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n) {
  mjtNum inv_h = 1/h;
  for (int i=0; i < n; i++) {
    dx[i] = inv_h * (x2[i] - x1[i]);
  }
}


// finite-difference two state vectors ds = (s2 - s1) / h
static void stateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2, mjtNum h) {
  int nq = m->nq, nv = m->nv, na = m->na;

  if (nq == nv) {
    diff(ds, s1, s2, h, nq+nv+na);
  } else {
    mj_differentiatePos(m, ds, h, s1, s2);
    diff(ds+nv, s1+nq, s2+nq, h, nv+na);
  }
}


// finite-difference two vectors, forward, backward or centered
static void clampedDiff(mjtNum* dx, const mjtNum* x, const mjtNum* x_plus, const mjtNum* x_minus,
                        mjtNum h, int nx) {
  if (x_plus && !x_minus) {
    // forward differencing
    diff(dx, x, x_plus, h, nx);
  } else if (!x_plus && x_minus) {
    // backward differencing
    diff(dx, x_minus, x, h, nx);
  } else if (x_plus && x_minus) {
    // centered differencing
    diff(dx, x_minus, x_plus, 2*h, nx);
  } else {
    // differencing failed, write zeros
    mju_zero(dx, nx);
  }
}


// finite-difference two state vectors, forward, backward or centered
static void clampedStateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s, const mjtNum* s_plus,
                             const mjtNum* s_minus, mjtNum h) {
  if (s_plus && !s_minus) {
    // forward differencing
    stateDiff(m, ds, s, s_plus, h);
  } else if (!s_plus && s_minus) {
    // backward differencing
    stateDiff(m, ds, s_minus, s, h);
  } else if (s_plus && s_minus) {
    // centered differencing
    stateDiff(m, ds, s_minus, s_plus, 2*h);
  } else {
    // differencing failed, write zeros
    mju_zero(ds, 2*m->nv + m->na);
  }
}


// check if two numbers are inside a given range
static int inRange(const mjtNum x1, const mjtNum x2, const mjtNum* range) {
  return x1 >= range[0] && x1 <= range[1] &&
         x2 >= range[0] && x2 <= range[1];
}


// return 1 if Euler damping matrix depends on qvel (nonlinear polynomial damping)
static int eulerDampDependsOnVel(const mjModel* m) {
  if (mjDISABLED(mjDSBL_EULERDAMP) || mjDISABLED(mjDSBL_DAMPER)) {
    return 0;
  }

  // check DOF damping polynomial
  for (int i=0; i < m->nv; i++) {
    if (!mju_isZero(m->dof_dampingpoly + mjNPOLY*i, mjNPOLY)) {
      return 1;
    }
  }

  // check joint actuator damping polynomial
  for (int i=0; i < m->nu; i++) {
    mjtTrn trntype = m->actuator_trntype[i];
    if (trntype == mjTRN_JOINT || trntype == mjTRN_JOINTINPARENT) {
      mjtNum gear = m->actuator_gear[6*m->actuator_outadr[i]];
      if (gear && !mju_isZero(m->actuator_dampingpoly + mjNPOLY*i, mjNPOLY)) {
        return 1;
      }
    }
  }

  return 0;
}


// advance simulation using control callback, skipstage is mjtStage
void mj_stepSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor) {
  TM_START;

  // common to all integrators
  mj_checkPos(m, d);
  mj_checkVel(m, d);
  mj_forwardSkip(m, d, skipstage, skipsensor);
  mj_checkAcc(m, d);

  // compare forward and inverse solutions if enabled
  if (mjENABLED(mjENBL_FWDINV)) {
    mj_compareFwdInv(m, d);
  }

  // use selected integrator
  switch ((mjtIntegrator) m->opt.integrator) {
  case mjINT_EULER: {
    int skipfactor = (skipstage > mjSTAGE_POS) ||
                     (skipstage == mjSTAGE_POS && !eulerDampDependsOnVel(m));
    mj_EulerSkip(m, d, skipfactor);
    break;
  }

  case mjINT_RK4:
    // ignore skipstage
    mj_RungeKutta(m, d, 4);
    break;

  case mjINT_IMPLICIT:
  case mjINT_IMPLICITFAST: {
    int skipfactor = (skipstage > mjSTAGE_VEL) ||
                     (skipstage == mjSTAGE_VEL &&
                      (mjDISABLED(mjDSBL_ACTUATION) || !m->nu));
    mj_implicitSkip(m, d, skipfactor);
    break;
  }

  case mjINT_DISCRETE:
    // the solve already performed the velocity update: no skip-dependent work
    if (mjENABLED(mjENBL_IPC)) {
      mj_ipc(m, d);
    } else {
      mj_discrete(m, d);
    }
    break;

  default:
    mjERROR("invalid integrator");
  }

  TM_END(mjTIMER_STEP);
}


// compute qfrc_inverse, optionally subtracting qfrc_actuator
static void inverseSkip(const mjModel* m, mjData* d, mjtStage stage, int skipsensor,
                        int flg_actuation, mjtNum* force) {
  // mirror mj_inverseSkip position and velocity stages before actuation
  if (flg_actuation) {
    if (stage < mjSTAGE_POS) {
      mj_invPosition(m, d);
      if (!skipsensor) {
        mj_sensorPos(m, d);
      }
      if (mjENABLED(mjENBL_ENERGY) && !d->flg_energypos) {
        mj_energyPos(m, d);
      }
    }
    if (stage < mjSTAGE_VEL) {
      mj_invVelocity(m, d);
      if (!skipsensor) {
        mj_sensorVel(m, d);
      }
      if (mjENABLED(mjENBL_ENERGY) && !d->flg_energyvel) {
        mj_energyVel(m, d);
      }
    }
    mj_fwdActuation(m, d);
    mj_inverseSkip(m, d, mjSTAGE_VEL, skipsensor);
    mju_sub(force, d->qfrc_inverse, d->qfrc_actuator, m->nv);
  } else {
    mj_inverseSkip(m, d, stage, skipsensor);
    mju_copy(force, d->qfrc_inverse, m->nv);
  }
}


//------------------------- derivatives of passive forces ------------------------------------------

// add forward fin-diff approximation of (d qfrc_passive / d qvel) to qDeriv
void mjd_passive_velFD(const mjModel* m, mjData* d, mjtNum eps) {
  int nv = m->nv;

  mj_markStack(d);
  mjtNum* qfrc_passive = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* fd = mjSTACKALLOC(d, nv, mjtNum);
  int* cnt = mjSTACKALLOC(d, nv, int);

  // clear row counters
  mju_zeroInt(cnt, nv);

  // save qfrc_passive, assume mj_fwdVelocity was called
  mju_copy(qfrc_passive, d->qfrc_passive, nv);

  // loop over dofs
  for (int i=0; i < nv; i++) {
    // save qvel[i]
    mjtNum saveqvel = d->qvel[i];

    // eval at qvel[i]+eps
    d->qvel[i] = saveqvel + eps;
    mj_fwdVelocity(m, d);

    // restore qvel[i]
    d->qvel[i] = saveqvel;

    // finite difference result in fd
    mju_sub(fd, d->qfrc_passive, qfrc_passive, nv);
    mju_scl(fd, fd, 1/eps, nv);

    // copy to i-th column of qDeriv
    for (int j=0; j < nv; j++) {
      int adr = m->D_rowadr[j] + cnt[j];
      if (cnt[j] < m->D_rownnz[j] && m->D_colind[adr] == i) {
        d->qDeriv[adr] = fd[j];
        cnt[j]++;
      }
    }
  }

  // restore
  mj_fwdVelocity(m, d);

  mj_freeStack(d);
}


//-------------------- derivatives of all smooth (unconstrained) forces ----------------------------

// centered finite difference approximation to mjd_smooth_vel
void mjd_smooth_velFD(const mjModel* m, mjData* d, mjtNum eps) {
  int nv = m->nv;

  mj_markStack(d);
  mjtNum* plus = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* minus = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* fd = mjSTACKALLOC(d, nv, mjtNum);
  int* cnt = mjSTACKALLOC(d, nv, int);

  // clear row counters
  mju_zeroInt(cnt, nv);

  // loop over dofs
  for (int i=0; i < nv; i++) {
    // save qvel[i]
    mjtNum saveqvel = d->qvel[i];

    // eval at qvel[i]+eps
    d->qvel[i] = saveqvel + eps;
    mj_fwdVelocity(m, d);
    mj_fwdActuation(m, d);
    mju_add(plus, d->qfrc_actuator, d->qfrc_passive, nv);
    mju_subFrom(plus, d->qfrc_bias, nv);

    // eval at qvel[i]-eps
    d->qvel[i] = saveqvel - eps;
    mj_fwdVelocity(m, d);
    mj_fwdActuation(m, d);
    mju_add(minus, d->qfrc_actuator, d->qfrc_passive, nv);
    mju_subFrom(minus, d->qfrc_bias, nv);

    // restore qvel[i]
    d->qvel[i] = saveqvel;

    // finite difference result in fd
    mju_sub(fd, plus, minus, nv);
    mju_scl(fd, fd, 0.5/eps, nv);

    // copy to sparse qDeriv
    for (int j=0; j < nv; j++) {
      if (cnt[j] < m->D_rownnz[j] && m->D_colind[m->D_rowadr[j]+cnt[j]] == i) {
        d->qDeriv[m->D_rowadr[j]+cnt[j]] = fd[j];
        cnt[j]++;
      }
    }
  }

  // make sure final row counters equal rownnz
  for (int i=0; i < nv; i++) {
    if (cnt[i] != m->D_rownnz[i]) {
      mjERROR("error in constructing FD sparse derivative");
    }
  }

  // restore
  mj_fwdVelocity(m, d);
  mj_fwdActuation(m, d);

  mj_freeStack(d);
}


//------------------------- main entry points ------------------------------------------------------


// finite differenced Jacobian of  (next_state, sensors) = mj_step(state, control)
//   all outputs are optional
//   output dimensions (transposed w.r.t Control Theory convention):
//     DyDq: (nv x 2*nv+na)
//     DyDv: (nv x 2*nv+na)
//     DyDa: (na x 2*nv+na)
//     DyDu: (nu x 2*nv+na)
//     DsDq: (nv x nsensordata)
//     DsDv: (nv x nsensordata)
//     DsDa: (na x nsensordata)
//     DsDu: (nu x nsensordata)
//   single-letter shortcuts:
//     inputs: q=qpos, v=qvel, a=act, u=ctrl
//     outputs: y=next_state (concatenated next qpos, qvel, act), s=sensordata
void mjd_stepFD(const mjModel* m, mjData* d, mjtNum eps, mjtBool flg_centered,
                mjtNum* DyDq, mjtNum* DyDv, mjtNum* DyDa, mjtNum* DyDu,
                mjtNum* DsDq, mjtNum* DsDv, mjtNum* DsDa, mjtNum* DsDu) {
  if (m->nhistory) {
    mjERROR("delays are not supported");
  }

  int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
  int ndx = 2*nv+na;  // row length of Dy Jacobians
  mj_markStack(d);

  // state to restore after finite differencing
  unsigned int restore_spec = mjSTATE_FULLPHYSICS | mjSTATE_CTRL;
  restore_spec |= mjDISABLED(mjDSBL_WARMSTART) ? 0 : mjSTATE_WARMSTART;

  mjtNum *fullstate  = mjSTACKALLOC(d, mj_stateSize(m, restore_spec), mjtNum);
  mjtNum *state      = mjSTACKALLOC(d, nq+nv+na, mjtNum);  // current state
  mjtNum *next       = mjSTACKALLOC(d, nq+nv+na, mjtNum);  // next state
  mjtNum *next_plus  = mjSTACKALLOC(d, nq+nv+na, mjtNum);  // forward-nudged next state
  mjtNum *next_minus = mjSTACKALLOC(d, nq+nv+na, mjtNum);  // backward-nudged next state

  // sensors
  int skipsensor = !DsDq && !DsDv && !DsDa && !DsDu;
  mjtNum *sensor       = skipsensor ? NULL : mjSTACKALLOC(d, ns, mjtNum);  // sensor values
  mjtNum *sensor_plus  = skipsensor ? NULL : mjSTACKALLOC(d, ns, mjtNum);  // forward-nudged
  mjtNum *sensor_minus = skipsensor ? NULL : mjSTACKALLOC(d, ns, mjtNum);  // backward-nudged

  // controls
  mjtNum *ctrl = mjSTACKALLOC(d, nu, mjtNum);

  // save current inputs
  mj_getState(m, d, fullstate, restore_spec);
  mju_copy(ctrl, d->ctrl, nu);
  getState(m, d, state, NULL);

  // step input
  mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);

  // save output
  getState(m, d, next, sensor);

  // restore input
  mj_setState(m, d, fullstate, restore_spec);

  // finite-difference controls: skip=mjSTAGE_VEL, handle ctrl at range limits
  if (DyDu || DsDu) {
    for (int i=0; i < nu; i++) {
      int limited = m->actuator_ctrllimited[i];
      // nudge forward, if possible given ctrlrange
      int nudge_fwd = !limited || inRange(ctrl[i], ctrl[i]+eps, m->actuator_ctrlrange+2*i);
      if (nudge_fwd) {
        // nudge forward
        d->ctrl[i] += eps;

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
        getState(m, d, next_plus, sensor_plus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
      }

      // nudge backward, if possible given ctrlrange
      int nudge_back = (flg_centered || !nudge_fwd) &&
                       (!limited || inRange(ctrl[i]-eps, ctrl[i], m->actuator_ctrlrange+2*i));
      if (nudge_back) {
        // nudge backward
        d->ctrl[i] -= eps;

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
      }

      // difference states
      if (DyDu) {
        clampedStateDiff(m, DyDu+i*ndx, next, nudge_fwd ? next_plus : NULL,
                         nudge_back ? next_minus : NULL, eps);
      }

      // difference sensors
      if (DsDu) {
        clampedDiff(DsDu+i*ns, sensor, nudge_fwd ? sensor_plus : NULL,
                    nudge_back ? sensor_minus : NULL, eps, ns);
      }
    }
  }

  // finite-difference activations: skip=mjSTAGE_VEL
  if (DyDa || DsDa) {
    for (int i=0; i < na; i++) {
      // nudge forward
      d->act[i] += eps;

      // step, get nudged output
      mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
      getState(m, d, next_plus, sensor_plus);

      // reset
      mj_setState(m, d, fullstate, restore_spec);

      // nudge backward
      if (flg_centered) {
        // nudge backward
        d->act[i] -= eps;

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
      }

      // difference states
      if (DyDa) {
        if (!flg_centered) {
          stateDiff(m, DyDa+i*ndx, next, next_plus, eps);
        } else {
          stateDiff(m, DyDa+i*ndx, next_minus, next_plus, 2*eps);
        }
      }

      // difference sensors
      if (DsDa) {
        if (!flg_centered) {
          diff(DsDa+i*ns, sensor, sensor_plus, eps, ns);
        } else {
          diff(DsDa+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
        }
      }
    }
  }


  // finite-difference velocities: skip=mjSTAGE_POS
  if (DyDv || DsDv) {
    for (int i=0; i < nv; i++) {
      // nudge forward
      d->qvel[i] += eps;

      // step, get nudged output
      mj_stepSkip(m, d, mjSTAGE_POS, skipsensor);
      getState(m, d, next_plus, sensor_plus);

      // reset
      mj_setState(m, d, fullstate, restore_spec);

      // nudge backward
      if (flg_centered) {
        // nudge
        d->qvel[i] -= eps;

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_POS, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
      }

      // difference states
      if (DyDv) {
        if (!flg_centered) {
          stateDiff(m, DyDv+i*ndx, next, next_plus, eps);
        } else {
          stateDiff(m, DyDv+i*ndx, next_minus, next_plus, 2*eps);
        }
      }

      // difference sensors
      if (DsDv) {
        if (!flg_centered) {
          diff(DsDv+i*ns, sensor, sensor_plus, eps, ns);
        } else {
          diff(DsDv+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
        }
      }
    }
  }

  // finite-difference positions: skip=mjSTAGE_NONE
  if (DyDq || DsDq) {
    mjtNum *dpos  = mjSTACKALLOC(d, nv, mjtNum);  // allocate position perturbation
    for (int i=0; i < nv; i++) {
      // nudge forward
      mju_zero(dpos, nv);
      dpos[i] = 1;
      mj_integratePos(m, d->qpos, dpos, eps);

      // step, get nudged output
      mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);
      getState(m, d, next_plus, sensor_plus);

      // reset
      mj_setState(m, d, fullstate, restore_spec);

      // nudge backward
      if (flg_centered) {
        // nudge backward
        mju_zero(dpos, nv);
        dpos[i] = 1;
        mj_integratePos(m, d->qpos, dpos, -eps);

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
      }

      // difference states
      if (DyDq) {
        if (!flg_centered) {
          stateDiff(m, DyDq+i*ndx, next, next_plus, eps);
        } else {
          stateDiff(m, DyDq+i*ndx, next_minus, next_plus, 2*eps);
        }
      }

      // difference sensors
      if (DsDq) {
        if (!flg_centered) {
          diff(DsDq+i*ns, sensor, sensor_plus, eps, ns);
        } else {
          diff(DsDq+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
        }
      }
    }
  }

  mj_freeStack(d);
}


// finite differenced transition matrices (control theory notation)
//   d(x_next) = A*dx + B*du
//   d(sensor) = C*dx + D*du
//   required output matrix dimensions:
//      A: (2*nv+na x 2*nv+na)
//      B: (2*nv+na x nu)
//      C: (nsensordata x 2*nv+na)
//      D: (nsensordata x nu)
void mjd_transitionFD(const mjModel* m, mjData* d, mjtNum eps, mjtBool flg_centered,
                      mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D) {
  if (m->opt.integrator == mjINT_RK4) {
    mjERROR("RK4 integrator is not supported");
  }
  if (m->nhistory) {
    mjERROR("delays are not supported");
  }
  if (mjENABLED(mjENBL_SLEEP)) {
    mjERROR("sleeping is not supported");
  }

  int nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
  int ndx = 2*nv+na;  // row length of state Jacobians

  // stepFD() offset pointers, initialised to NULL
  mjtNum *DyDq, *DyDv, *DyDa, *DsDq, *DsDv, *DsDa;
  DyDq = DyDv = DyDa = DsDq = DsDv = DsDa = NULL;

  mj_markStack(d);

  // allocate transposed matrices
  mjtNum *AT = A ? mjSTACKALLOC(d, ndx*ndx, mjtNum) : NULL;  // state-transition     (transposed)
  mjtNum *BT = B ? mjSTACKALLOC(d, nu*ndx, mjtNum) : NULL;   // control-transition   (transposed)
  mjtNum *CT = C ? mjSTACKALLOC(d, ndx*ns, mjtNum) : NULL;   // state-observation    (transposed)
  mjtNum *DT = D ? mjSTACKALLOC(d, nu*ns, mjtNum) : NULL;    // control-observation  (transposed)

  // set offset pointers
  if (A) {
    DyDq = AT;
    DyDv = AT+ndx*nv;
    DyDa = AT+ndx*2*nv;
  }

  if (C) {
    DsDq = CT;
    DsDv = CT + ns*nv;
    DsDa = CT + ns*2*nv;
  }

  // get Jacobians
  mjd_stepFD(m, d, eps, flg_centered, DyDq, DyDv, DyDa, BT, DsDq, DsDv, DsDa, DT);


  // transpose
  if (A) mju_transpose(A, AT, ndx, ndx);
  if (B) mju_transpose(B, BT, nu, ndx);
  if (C) mju_transpose(C, CT, ndx, ns);
  if (D) mju_transpose(D, DT, nu, ns);

  mj_freeStack(d);
}

// finite differenced Jacobians of (force, sensors) = mj_inverse(state, acceleration)
//   all outputs are optional
//   output dimensions (transposed w.r.t Control Theory convention):
//     DfDq: (nv x nv)
//     DfDv: (nv x nv)
//     DfDa: (nv x nv)
//     DsDq: (nv x nsensordata)
//     DsDv: (nv x nsensordata)
//     DsDa: (nv x nsensordata)
//     DmDq: (nv x nC)
//   single-letter shortcuts:
//     inputs: q=qpos, v=qvel, a=qacc
//     outputs: f=qfrc_inverse, s=sensordata, m=M
//   notes:
//     optionally compute mass matrix Jacobian DmDq
//     flg_actuation specifies whether to subtract qfrc_actuator from qfrc_inverse
void mjd_inverseFD(const mjModel* m, mjData* d, mjtNum eps, mjtBool flg_actuation,
                   mjtNum *DfDq, mjtNum *DfDv, mjtNum *DfDa,
                   mjtNum *DsDq, mjtNum *DsDv, mjtNum *DsDa,
                   mjtNum *DmDq) {
  int nq = m->nq, nv = m->nv, nC = m->nC, ns = m->nsensordata;

  if (m->opt.integrator == mjINT_RK4) {
    mjERROR("RK4 integrator is not supported");
  }

  if (m->opt.noslip_iterations) {
    mjERROR("noslip solver is not supported");
  }

  if (mjENABLED(mjENBL_SLEEP)) {
    mjERROR("sleeping is not supported");
  }

  // skip sensor computations if no sensor Jacobians requested
  int skipsensor = !DsDq && !DsDv && !DsDa;

  // local vectors
  mj_markStack(d);
  mjtNum *pos        = mjSTACKALLOC(d, nq, mjtNum);                      // position
  mjtNum *force      = mjSTACKALLOC(d, nv, mjtNum);                      // force
  mjtNum *force_plus = mjSTACKALLOC(d, nv, mjtNum);                      // nudged force
  mjtNum *sensor     = skipsensor ? NULL : mjSTACKALLOC(d, ns, mjtNum);  // sensor values
  mjtNum *mass       = DmDq ? mjSTACKALLOC(d, nC, mjtNum) : NULL;        // mass matrix

  // save current positions
  mju_copy(pos, d->qpos, nq);

  // center point outputs
  inverseSkip(m, d, mjSTAGE_NONE, skipsensor, flg_actuation, force);
  if (sensor) mju_copy(sensor, d->sensordata, ns);
  if (mass) mju_copy(mass, d->M, nC);

  // acceleration: skip = mjSTAGE_VEL
  if (DfDa || DsDa) {
    for (int i=0; i < nv; i++) {
      // nudge acceleration
      mjtNum tmp = d->qacc[i];
      d->qacc[i] += eps;

      // inverse dynamics, get force output
      inverseSkip(m, d, mjSTAGE_VEL, skipsensor, flg_actuation, force_plus);

      // restore
      d->qacc[i] = tmp;

      // row of force Jacobian
      if (DfDa) diff(DfDa + i*nv, force, force_plus, eps, nv);

      // row of sensor Jacobian
      if (DsDa) diff(DsDa + i*ns, sensor, d->sensordata, eps, ns);
    }
  }

  // velocity: skip = mjSTAGE_POS
  if (DfDv || DsDv) {
    for (int i=0; i < nv; i++) {
      // nudge velocity
      mjtNum tmp = d->qvel[i];
      d->qvel[i] += eps;

      // inverse dynamics, get force output
      inverseSkip(m, d, mjSTAGE_POS, skipsensor, flg_actuation, force_plus);

      // restore
      d->qvel[i] = tmp;

      // row of force Jacobian
      if (DfDv) diff(DfDv + i*nv, force, force_plus, eps, nv);

      // row of sensor Jacobian
      if (DsDv) diff(DsDv + i*ns, sensor, d->sensordata, eps, ns);
    }
  }

  // position: skip = mjSTAGE_NONE
  if (DfDq || DsDq || DmDq) {
    mjtNum *dpos  = mjSTACKALLOC(d, nv, mjtNum);  // allocate position perturbation
    for (int i=0; i < nv; i++) {
      // nudge
      mju_zero(dpos, nv);
      dpos[i] = 1.0;
      mj_integratePos(m, d->qpos, dpos, eps);

      // inverse dynamics, get force output
      inverseSkip(m, d, mjSTAGE_NONE, skipsensor, flg_actuation, force_plus);

      // restore
      mju_copy(d->qpos, pos, nq);

      // row of force Jacobian
      if (DfDq) diff(DfDq + i*nv, force, force_plus, eps, nv);

      // row of sensor Jacobian
      if (DsDq) diff(DsDq + i*ns, sensor, d->sensordata, eps, ns);

      // row of inertia Jacobian
      if (DmDq) diff(DmDq + i*nC, mass, d->M, eps, nC);
    }
  }

  mj_freeStack(d);
}
