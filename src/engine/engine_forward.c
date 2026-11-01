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

#include "engine/engine_forward.h"

#include <stddef.h>
#include <stdio.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjplugin.h>
#include "engine/engine_callback.h"
#include "engine/engine_collision_driver.h"
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_smooth.h"
#include "engine/engine_core_util.h"
#include "engine/engine_derivative.h"
#include "engine/engine_inverse.h"
#include "engine/engine_io.h"
#include "engine/engine_ipc.h"
#include "engine/engine_island.h"
#include "engine/engine_macro.h"
#include "engine/engine_memory.h"
#include "engine/engine_metric.h"
#include "engine/engine_passive.h"
#include "engine/engine_plugin.h"
#include "engine/engine_sensor.h"
#include "engine/engine_sleep.h"
#include "engine/engine_solver.h"
#include "engine/engine_support.h"
#include "engine/engine_inline.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"
#include "engine/engine_thread.h"



//--------------------------- check values ---------------------------------------------------------

// check positions, reset if bad
void mj_checkPos(const mjModel* m, mjData* d) {
  int nq = m->nq;
  const mjtNum* qpos = d->qpos;
  for (int i=0; i < nq; i++) {
    if (mju_isBad(qpos[i])) {
      mj_warning(d, mjWARN_BADQPOS, i);
      if (!mjDISABLED(mjDSBL_AUTORESET)) {
        mj_resetData(m, d);
        d->warning[mjWARN_BADQPOS].number++;
        d->warning[mjWARN_BADQPOS].lastinfo = i;
      }
      return;
    }
  }
}


// check velocities, reset if bad
void mj_checkVel(const mjModel* m, mjData* d) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  int nv = sleep_filter ? d->nv_awake : m->nv;

  for (int j=0; j < nv; j++) {
    int i = sleep_filter ? d->dof_awake_ind[j] : j;

    if (mju_isBad(d->qvel[i])) {
      mj_warning(d, mjWARN_BADQVEL, i);
      if (!mjDISABLED(mjDSBL_AUTORESET)) {
        mj_resetData(m, d);
        d->warning[mjWARN_BADQVEL].number++;
        d->warning[mjWARN_BADQVEL].lastinfo = i;
      }
      return;
    }
  }
}


// check accelerations, reset if bad
void mj_checkAcc(const mjModel* m, mjData* d) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  int nv = sleep_filter ? d->nv_awake : m->nv;

  for (int j=0; j < nv; j++) {
    int i = sleep_filter ? d->dof_awake_ind[j] : j;

    if (mju_isBad(d->qacc[i])) {
      mj_warning(d, mjWARN_BADQACC, i);
      if (!mjDISABLED(mjDSBL_AUTORESET)) {
        mj_resetData(m, d);
        d->warning[mjWARN_BADQACC].number++;
        d->warning[mjWARN_BADQACC].lastinfo = i;
        mj_forward(m, d);
      }
      return;
    }
  }
}


//-------------------------- solver components -----------------------------------------------------

// kinematics-related computations
void mj_fwdKinematics(const mjModel* m, mjData* d) {
  mj_kinematics(m, d);
  mj_comPos(m, d);
  mj_camlight(m, d);
  mj_flex(m, d);
  mj_tendon(m, d);
  if (mj_wakeTendon(m, d)) {
    mj_updateSleep(m, d);
  }
}

// position-dependent computations
void mj_fwdPosition(const mjModel* m, mjData* d) {
  TM_START1;

  // clear position-dependent flags for lazy evaluation
  d->flg_energypos = 0;

  TM_START;
  mj_fwdKinematics(m, d);

  TM_END(mjTIMER_POS_KINEMATICS);

  // inertia, timed internally (POS_INERTIA)
  mj_makeM(m, d);
  mj_factorM(m, d);

  // collision, timed internally (POS_COLLISION)
  mj_collision(m, d);

  if (mj_wakeCollision(m, d)) {
    mj_updateSleep(m, d);
    mj_collision(m, d);
  }

  if (mj_wakeEquality(m, d)) {
    mj_updateSleep(m, d);
  }

  TM_RESTART;
  mj_makeConstraint(m, d);
  mj_island(m, d);
  TM_END(mjTIMER_POS_MAKE);

  TM_RESTART;
  mj_projectConstraint(m, d);
  TM_END(mjTIMER_POS_PROJECT);

  TM_RESTART;
  mj_transmission(m, d);
  TM_ADD(mjTIMER_POS_KINEMATICS);

  // implicit effective metric Mtilde = M + K: build (or deactivate) for this step. Arena
  // lifetime and skip semantics mirror the constraint data: built once per position stage,
  // value-refreshed in the velocity stage, consumed downstream.
  mj_effBuild(m, d, mj_isMetric(m), /*flg_factor=*/1);

  TM_END1(mjTIMER_POSITION);
}


// velocity-dependent computations
void mj_fwdVelocity(const mjModel* m, mjData* d) {
  TM_START;

  // clear velocity-dependent flags for lazy evaluation
  d->flg_subtreevel = 0;
  d->flg_energyvel = 0;

  // flexedge velocity: skip interp and rigid flexes (edge Jacobians are zero)
  mju_zero(d->flexedge_velocity, m->nflexedge);
  for (int f = 0; f < m->nflex; f++) {
    if (m->flex_rigid[f] || m->flex_interp[f]) continue;
    int adr = m->flex_edgeadr[f];
    int num = m->flex_edgenum[f];
    mju_mulMatVecSparse(d->flexedge_velocity + adr, d->flexedge_J, d->qvel, num,
                        m->flexedge_J_rownnz + adr, m->flexedge_J_rowadr + adr,
                        m->flexedge_J_colind, NULL);
  }

  // tendon velocity: always sparse
  mju_mulMatVecSparse(d->ten_velocity, d->ten_J, d->qvel, m->ntendon,
                      m->ten_J_rownnz, m->ten_J_rowadr, m->ten_J_colind, NULL);

  // actuator velocity: always sparse
  if (!mjDISABLED(mjDSBL_ACTUATION)) {
    mju_mulMatVecSparse(d->actuator_velocity, d->actuator_moment, d->qvel, m->nout,
                        d->moment_rownnz, d->moment_rowadr, d->moment_colind, NULL);
  } else {
    mju_zero(d->actuator_velocity, m->nout);
  }

  // com-based velocities, passive forces, constraint references
  mj_comVel(m, d);
  mj_passive(m, d);

  // under discrete, constraint references are computed at the actuation stage where the
  // metric is final (regularization, then references, which consume R via adhesion);
  // constraint velocities are needed now for velocity sensors
  if (mj_isMetric(m)) {
    mj_velocityConstraint(m, d);
  } else {
    mj_referenceConstraint(m, d);
  }

  // compute qfrc_bias with abbreviated RNE (without acceleration)
  mj_rne(m, d, 0, d->qfrc_bias);

  // add bias force due to tendon armature
  mj_tendonBias(m, d, d->qfrc_bias);

  // refresh the metric's velocity-stage values
  mj_effShift(m, d);


  TM_END(mjTIMER_VELOCITY);
}


// unpack servo-family inputs from control block in canonical order [pos, vel, ff]
// absent input: setpoint 0
static void unpackServoInputs(const mjtNum* u, int spec, mjtNum out[4]) {
  int adr = 0;
  out[0] = (spec & mjINPUT_POS)     ? u[adr++] : 0;
  out[1] = (spec & mjINPUT_VEL)     ? u[adr++] : 0;
  out[2] = (spec & mjINPUT_FF)      ? u[adr++] : 0;
  out[3] = (spec & mjINPUT_VOLTAGE) ? u[adr]   : 0;
}


// helper for DC motor: computes control voltage from PID state
static mjtNum dcmotorVoltage(const mjtNum* u, int spec, mjtNum length, mjtNum velocity,
                             mjtNum x_I, const mjtNum* gainprm) {
  mjtNum voltage = 0;

  // unpack present inputs in canonical order [pos, vel, ff, voltage]; absent input: 0
  mjtNum u4[4];
  unpackServoInputs(u, spec, u4);

  // on-board controller: torque-space PID + torque feedforward
  if (spec & (mjINPUT_POS | mjINPUT_VEL | mjINPUT_FF)) {
    mjtNum kp = gainprm[4];  // proportional gain
    mjtNum ki = gainprm[5];  // integral gain
    mjtNum kd = gainprm[6];  // derivative gain
    mjtNum torque = kp*(u4[0] - length) + kd*(u4[1] - velocity) + ki*x_I + u4[2];

    // torque mode is current control: V = R/K * torque + K * velocity, the second
    // term compensating back-EMF; the compiler requires K > 0 on this path
    mjtNum R = gainprm[0];
    mjtNum K = gainprm[1];
    voltage = R/K * torque + K*velocity;

    // driver supply limit
    mjtNum Vmax = gainprm[7];
    if (Vmax > 0) voltage = mju_clip(voltage, -Vmax, Vmax);
  }

  // raw terminal voltage input: downstream of the controller, unclamped
  return voltage + u4[3];
}


// clamp vector to range
static void clampVec(mjtNum* vec, const mjtNum* range, const mjtBool* limited, int n,
                      const int* index) {
  for (int i=0; i < n; i++) {
    int j = index ? index[i] : i;
    if (limited[i]) {
      vec[j] = mju_clip(vec[j], range[2*i], range[2*i + 1]);
    }
  }
}


// expmap (axis-angle) vector to quaternion
static void expmap2Quat(mjtNum quat[4], const mjtNum v[3]) {
  mjtNum angle = mju_norm3(v);
  if (angle < mjMINVAL) {
    quat[0] = 1;
    quat[1] = quat[2] = quat[3] = 0;
  } else {
    mjtNum axis[3] = {v[0]/angle, v[1]/angle, v[2]/angle};
    mji_axisAngle2Quat(quat, axis, angle);
  }
}


// period of the rotational transmission for wrap-eligible servo actuators, 0 otherwise
static mjtNum wrapPeriod(const mjModel* m, int i) {
  // servo shape: fixed gain, affine bias, matching kp, setpoint input
  mjtDyn dyntype = m->actuator_dyntype[i];
  int servo = m->actuator_gaintype[i] == mjGAIN_FIXED  &&
              m->actuator_biastype[i] == mjBIAS_AFFINE &&
              m->actuator_gainprm[mjNGAIN*i] == -m->actuator_biasprm[mjNBIAS*i+1] &&
              (dyntype == mjDYN_NONE || dyntype == mjDYN_INTEGRATOR);

  // PID shape: kp and kv are single-sourced in the affine bias
  int pid = m->actuator_gaintype[i] == mjGAIN_PID;

  if (!servo && !pid) {
    return 0;
  }

  const mjtNum* gear = m->actuator_gear+6*m->actuator_outadr[i];
  mjtTrn trntype = m->actuator_trntype[i];

  // site transmission with refsite and purely rotational gear
  if (trntype == mjTRN_SITE && m->actuator_trnid[2*i+1] >= 0 &&
      !gear[0] && !gear[1] && !gear[2]) {
    return 2*mjPI * mju_norm3(gear+3);
  }

  // joint transmission on a ball joint
  if ((trntype == mjTRN_JOINT || trntype == mjTRN_JOINTINPARENT) &&
      m->jnt_type[m->actuator_trnid[2*i]] == mjJNT_BALL) {
    return 2*mjPI * mju_norm3(gear);
  }

  return 0;
}


// representative of setpoint u nearest to length, given period
static mjtNum wrapSetpoint(mjtNum u, mjtNum length, mjtNum period) {
  mjtNum err = u - length;
  return u - period * mju_round(err / period);
}


// slew-rate-limit setpoint u given previous effective setpoint u_prev, write act_dot
// period > 0: wrap u to the representative nearest u_prev before limiting
static mjtNum slewLimit(mjtNum u, mjtNum u_prev, mjtNum slew_s, mjtNum dt,
                        mjtNum period, mjtNum* act_dot) {
  if (period > 0) {
    u = wrapSetpoint(u, u_prev, period);
  }
  mjtNum slew = slew_s * dt;
  mjtNum u_eff = mju_clip(u, u_prev - slew, u_prev + slew);
  *act_dot = (u_eff - u_prev) / dt;
  return u_eff;
}


// (qpos, qvel, ctrl, act) => (qfrc_actuator, actuator_force, act_dot)
void mj_fwdActuation(const mjModel* m, mjData* d) {
  TM_START;
  int nv = m->nv, nu = m->nu, nactuator = m->nactuator, nout = m->nout, ntendon = m->ntendon;
  mjtNum gain, bias, tau;
  mjtNum *force = d->actuator_force;

  // clear actuator_force
  mju_zero(force, nout);

  int sleep_filter = mjENABLED(mjENBL_SLEEP);

  // disabled or no actuation: return
  if (nactuator == 0 || mjDISABLED(mjDSBL_ACTUATION)) {
    mju_zero(d->qfrc_actuator, nv);
    TM_END(mjTIMER_ACTUATION);
    return;
  }

  // any tendon transmission targets with force limits
  int tendon_frclimited = 0;

  // local copy of ctrl
  mj_markStack(d);
  mjtNum *ctrl = mjSTACKALLOC(d, nu, mjtNum);

  // copy controls, overwrite delayed actuators from history buffer
  mju_copy(ctrl, d->ctrl, nu);
  for (int i = 0; i < nactuator; i++) {
    if (m->actuator_delay[i]) {
      int adr = m->actuator_ctrladr[i];
      const mjtNum* ptr = mj_readCtrl(m, d, i, d->time, ctrl + adr, -1);
      if (ptr) mju_copy(ctrl + adr, ptr, m->actuator_ctrlnum[i]);
    }
  }

  // clamp local copy
  if (!mjDISABLED(mjDSBL_CLAMPCTRL)) {
    clampVec(ctrl, m->actuator_ctrlrange, m->actuator_ctrllimited, nu, NULL);
  }

  // check controls, set all to 0 if any are bad
  for (int i=0; i < nu; i++) {
    if (mju_isBad(ctrl[i])) {
      mj_warning(d, mjWARN_BADCTRL, i);
      mju_zero(ctrl, nu);
      break;
    }
  }

  // act_dot for stateful actuators
  for (int i=0; i < nactuator; i++) {
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_ACTUATOR, i) == mjS_ASLEEP) {
      continue;
    }

    int act_first = m->actuator_actadr[i];
    if (act_first < 0) {
      continue;
    }

    // addresses of the actuator's input and output blocks
    int uadr = m->actuator_ctrladr[i];
    int oadr = m->actuator_outadr[i];

    // zero act_dot for actuator plugins
    int actnum = m->actuator_actnum[i];
    if (actnum) {
      mju_zero(d->act_dot + act_first, actnum);
    }

    // extract info
    const mjtNum* dynprm = m->actuator_dynprm + i*mjNDYN;
    mjtDyn dyntype = m->actuator_dyntype[i];

    // index into the last element in act. For most actuators it's also the
    // first element, but actuator plugins might store their own state in act
    int act_last = act_first + actnum - 1;

    // compute act_dot according to dynamics type
    switch (dyntype) {
    case mjDYN_INTEGRATOR: {        // simple integrator, one per control
      int num = m->actuator_ctrlnum[i];
      for (int j=0; j < num; j++) {
        d->act_dot[act_last-num+1+j] = ctrl[uadr+j];
      }
      break;
    }

    case mjDYN_FILTER:              // linear filter: dynprm = tau
    case mjDYN_FILTEREXACT:
      tau = mju_max(mjMINVAL, dynprm[0]);
      d->act_dot[act_last] = (ctrl[uadr] - d->act[act_last]) / tau;
      break;

    case mjDYN_MUSCLE:              // muscle model: dynprm = (tau_act, tau_deact)
      d->act_dot[act_last] = mju_muscleDynamics(ctrl[uadr], d->act[act_last], dynprm);
      break;

    case mjDYN_PID: {               // PID controller states, slot order: slew, integral
      int adr = act_first;
      mjtNum period = wrapPeriod(m, i);

      // slew rate limiting of the position setpoint
      mjtNum slew_s = dynprm[1];
      if (slew_s > 0) {
        ctrl[uadr] = slewLimit(ctrl[uadr], d->act[adr], slew_s, m->opt.timestep,
                               period, d->act_dot + adr);
        adr++;
      }

      // integral of the position error
      if (m->actuator_gainprm[mjNGAIN*i] > 0) {
        mjtNum err = ctrl[uadr] - d->actuator_length[oadr];

        // rotational transmission: error on the circle
        if (period > 0) {
          err -= period*mju_round(err/period);
        }

        // anti-windup: stop accumulating beyond imax
        mjtNum imax = dynprm[0];
        if (imax > 0) {
          mjtNum z = d->act[adr];
          if (z >= imax) {
            err = mju_min(err, 0);
          } else if (z <= -imax) {
            err = mju_max(err, 0);
          }
        }
        d->act_dot[adr] = err;
      }
      break;
    }

    case mjDYN_DCMOTOR: {           // DC motor: up to 5 optional states
      const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*i;

      // verify allocated state size matches parameters; SHOULD NOT OCCUR
      if (mj_dcmotorSlots(dynprm, gainprm).num_slots != actnum) {
        mjERROR("inconsistent state array dimension in DC motor (actuator %d)", i);
      }

      int adr = act_first;
      mjtNum velocity = d->actuator_velocity[oadr];
      mjtNum R = gainprm[0];   // resistance
      mjtNum K = gainprm[1];   // motor constant
      mjtNum ki = gainprm[5];  // integral gain
      mjtNum te = dynprm[0];   // electrical time constant

      // slot order: slew, integral, temperature, bristle, current

      // controller state: slew rate limiting
      mjtNum slew_s = dynprm[7];  // slew rate limit
      if (slew_s > 0) {
        ctrl[uadr] = slewLimit(ctrl[uadr], d->act[adr], slew_s, m->opt.timestep,
                               0, d->act_dot + adr);
        adr++;
      }

      // controller state: integral of the position error (setpoint mode only)
      mjtNum x_I = 0;
      if (ki > 0) {
        x_I = d->act[adr];
        mjtNum Imax = dynprm[8];   // integral clamp
        mjtNum act_dot = ctrl[uadr] - d->actuator_length[oadr];

        // clamp act_dot based on integral state
        if (Imax > 0) {
          if (x_I >= Imax) {
            act_dot = mju_min(act_dot, 0);
          } else if (x_I <= -Imax) {
            act_dot = mju_max(act_dot, 0);
          }
        }
        d->act_dot[adr] = act_dot;
        adr++;
      }

      // compute physical voltage to feed into current and temperature equations
      mjtNum V = dcmotorVoltage(ctrl + uadr, m->actuator_ctrlspec[i],
                                d->actuator_length[oadr], velocity, x_I, gainprm);

      // temperature: dT/dt = (R*i^2 - T/RT) / C, where T = delta above ambient
      mjtNum RT = dynprm[2];  // thermal resistance
      if (RT > 0) {
        mjtNum C = dynprm[3];       // thermal capacitance
        mjtNum Ta = dynprm[4];      // ambient temperature
        mjtNum alpha = gainprm[2];  // temperature coefficient
        mjtNum T0 = gainprm[3];     // reference temperature
        mjtNum T = d->act[adr];     // temperature rise above ambient
        R *= 1 + alpha * (T + Ta - T0);

        // get current: from act_last if stateful, from (V - K*omega)/R if stateless
        mjtNum current = (te > 0) ? d->act[act_last] : (V - K * velocity) / R;
        d->act_dot[adr] = (R*current*current - T / RT) / C;
        adr++;
      }

      // LuGre bristle state: dz/dt = v - sigma0 * |v| / g(v) * z
      mjtNum sigma0 = dynprm[5];  // bristle stiffness
      if (sigma0 > 0) {
        const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*i;
        mjtNum F_C = biasprm[3];  // Coulomb friction
        mjtNum F_S = biasprm[4];  // static friction
        mjtNum v_S = biasprm[5];  // Stribeck velocity
        mjtNum z = d->act[adr];   // bristle state
        mjtNum g = mj_lugreStribeck(velocity, F_C, F_S, v_S);
        mjtNum a = -sigma0 * mju_abs(velocity) / mju_max(mjMINVAL, g);
        d->act_dot[adr] = a * z + velocity;
        adr++;
      }

      // current state: di/dt = (V/R - K/R*omega - i) / te
      if (te > 0) {
        mjtNum dimax = dynprm[1];  // current rate limit (di/dt)_max
        mjtNum i_dot = (V/R - K/R*velocity - d->act[act_last]) / te;
        if (dimax > 0) {
          i_dot = mju_clip(i_dot, -dimax, dimax);
        }
        d->act_dot[act_last] = i_dot;
      }
      break;
    }

    default:                        // user dynamics
      if (mjcb_act_dyn) {
        if (actnum == 1) {
          // scalar activation dynamics, get act_dot
          d->act_dot[act_last] = mjcb_act_dyn(m, d, i);
        } else {
          // higher-order dynamics, mjcb_act_dyn writes into act_dot directly
          mjcb_act_dyn(m, d, i);
        }
      }
    }
  }

  // get act_dot from actuator plugins
  if (m->nplugin) {
    const int nslot = mjp_pluginCount();
    for (int i=0; i < m->nplugin; i++) {
      const int slot = m->plugin[i];
      const mjpPlugin* plugin = mjp_getPluginAtSlotUnsafe(slot, nslot);
      if (!plugin) {
        mjERROR("invalid plugin slot: %d", slot);
      }
      if (plugin->capabilityflags & mjPLUGIN_ACTUATOR) {
        if (plugin->actuator_act_dot) {
          plugin->actuator_act_dot(m, d, i);
        }
      }
    }
  }

  // force = gain .* [ctrl/act] + bias
  for (int i=0; i < nactuator; i++) {
    // skip if sleeping
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_ACTUATOR, i) == mjS_ASLEEP) {
      continue;
    }

    // skip if disabled
    if (mj_actuatorDisabled(m, i)) {
      continue;
    }

    // skip actuator plugins -- these are handled after builtin actuator types
    if (m->actuator_plugin[i] >= 0) {
      continue;
    }

    // addresses of the actuator's input and output blocks
    int uadr = m->actuator_ctrladr[i];
    int oadr = m->actuator_outadr[i];

    // SO(3) geodesic servo: 3 or 4 inputs and 3 outputs on an SO3 transmission
    if (m->actuator_gaintype[i] == mjGAIN_SO3) {
      mjtNum q_tgt[4];

      // quat input: normalize ctrl directly (zero maps to the identity)
      if (m->actuator_ctrlspec[i] == mjCHART_QUAT) {
        mju_copy4(q_tgt, ctrl + uadr);
        mju_normalize4(q_tgt);
      }

      // expmap input: ctrl block (position) or act block (integrator)
      else {
        mjtNum u[3];
        if (m->actuator_dyntype[i] == mjDYN_NONE) {
          mju_copy3(u, ctrl + uadr);
        } else {
          int act_adr = m->actuator_actadr[i];
          if (m->actuator_actearly[i]) {
            for (int k=0; k < 3; k++) {
              u[k] = mj_nextActivation(m, d, i, act_adr+k, d->act_dot[act_adr+k]);
            }
          } else {
            mju_copy3(u, d->act + act_adr);
          }
        }
        expmap2Quat(q_tgt, u);
      }

      // error rotation from current to target: e = log(q_cur^-1 * q_tgt), in the local frame
      // of the transmission, matching the frame of the moment rows and of actuator_velocity
      // note: the force is invariant to the setpoint representative (exp is ray-periodic),
      // so no wrapping is required; act is re-anchored at integration time in mj_advance
      mjtNum q_cur[4], e[3];
      expmap2Quat(q_cur, d->actuator_length + oadr);
      mji_subQuat(e, q_tgt, q_cur);

      // output force: kp * error + constant - kv * velocity
      mjtNum kp = m->actuator_gainprm[mjNGAIN*i];
      const mjtNum* prm = m->actuator_biasprm + mjNBIAS*i;
      for (int k=0; k < 3; k++) {
        force[oadr+k] = kp*e[k] + prm[0] + prm[2]*d->actuator_velocity[oadr+k];
      }
      continue;
    }

    // check for tendon transmission with force limits
    if (ntendon && !tendon_frclimited && m->actuator_trntype[i] == mjTRN_TENDON) {
      tendon_frclimited = m->tendon_actfrclimited[m->actuator_trnid[2*i]];
    }

    // extract info
    const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
    const mjtNum* gainprm = m->actuator_gainprm + mjNGAIN*i;
    mjtGain gaintype = m->actuator_gaintype[i];
    int actnum = m->actuator_actnum[i];

    // handle SISO actuators according to gain type
    switch (gaintype) {
    case mjGAIN_FIXED:              // fixed gain: prm = gain
      gain = gainprm[0];
      break;

    case mjGAIN_PID:                // PID servo: input side handled below, state side in bias
      gain = 0;
      break;

    case mjGAIN_AFFINE:             // affine: prm = [const, kp, kv]
      gain = gainprm[0] + gainprm[1]*d->actuator_length[oadr] +
             gainprm[2]*d->actuator_velocity[oadr];
      break;

    case mjGAIN_MUSCLE:             // muscle gain
      gain = mju_muscleGain(d->actuator_length[oadr],
                            d->actuator_velocity[oadr],
                            m->actuator_lengthrange+2*oadr,
                            m->actuator_acc0[oadr],
                            gainprm);
      break;

    case mjGAIN_DCMOTOR: {          // DC motor: gain = K or K/R
      mjtNum R = mj_dcmotorResistance(m, d, i);  // motor resistance
      mjtNum K = gainprm[1];                     // motor constant
      mjDCMotorSlots slots = mj_dcmotorSlots(dynprm, gainprm);

      // verify allocated state size matches parameters; SHOULD NOT OCCUR
      if (slots.num_slots != actnum) {
        mjERROR("inconsistent state array dimension in DC motor (actuator %d)", i);
      }

      // stateful current: gain = K, force = K * act[last] (generic path)
      // stateless: gain = K/R, force = K/R * ctrl (condition below)
      gain = (dynprm[0] > 0) ? K : K / R;

      // controller: compute voltage, override ctrl[uadr] for force computation
      // (pure raw-voltage motor reads ctrl directly; empty block reads as 0 below)
      if (m->actuator_ctrlspec[i] != mjINPUT_VOLTAGE && m->actuator_ctrlnum[i] > 0) {
        mjtNum x_I = (slots.integral >= 0) ? d->act[m->actuator_actadr[i]+slots.integral] : 0;
        ctrl[uadr] = dcmotorVoltage(ctrl + uadr, m->actuator_ctrlspec[i],
                                    d->actuator_length[oadr],
                                    d->actuator_velocity[oadr], x_I, gainprm);
      }
      break;
    }

    case mjGAIN_SO3:                // handled above via early continue
      mjERROR("mjGAIN_SO3 reached SISO switch (actuator %d)", i);
      break;

    default:                        // user gain
      if (mjcb_act_gain) {
        gain = mjcb_act_gain(m, d, i);
      } else {
        gain = 1;
      }
    }

    // set force = gain .* [ctrl/act]

    // DC motor without current state: use ctrl even if other activations exist
    int dcmotor_no_current = (gaintype == mjGAIN_DCMOTOR && dynprm[0] <= 0);

    // PID servo: force = kp*(qref - l) + kv*(vref - l_dot) [+ ff] [+ ki*z]
    // input-side terms computed here; state-side terms added by the affine bias below
    if (gaintype == mjGAIN_PID) {
      const mjtNum* prm = m->actuator_biasprm + mjNBIAS*i;

      // unpack present inputs in canonical order [pos, vel, ff]; absent input: setpoint 0
      mjtNum u4[4];
      unpackServoInputs(ctrl + uadr, m->actuator_ctrlspec[i], u4);
      mjtNum qref = u4[0], vref = u4[1], ff = u4[2];

      // position setpoint: representative nearest the length on rotational transmissions
      mjtNum period = wrapPeriod(m, i);
      if (period > 0) {
        qref = wrapSetpoint(qref, d->actuator_length[oadr], period);
      }

      // kp and kv are single-sourced in the affine bias parameters
      force[oadr] = -prm[1]*qref - prm[2]*vref + ff;

      // integral state (last slot): force += ki * z
      if (actnum && gainprm[0] > 0) {
        int act_adr = m->actuator_actadr[i] + actnum - 1;
        mjtNum z = m->actuator_actearly[i]
                       ? mj_nextActivation(m, d, i, act_adr, d->act_dot[act_adr])
                       : d->act[act_adr];
        force[oadr] += gainprm[0]*z;
      }
    }
    else if (actnum == 0 || dcmotor_no_current) {
      // empty input block (passive dcmotor): input is 0
      mjtNum input = m->actuator_ctrlnum[i] ? ctrl[uadr] : 0;

      // rotational setpoint: use representative nearest the length (local, no state change)
      mjtNum period = wrapPeriod(m, i);
      if (period > 0) {
        input = wrapSetpoint(input, d->actuator_length[oadr], period);
      }
      force[oadr] = gain * input;
    } else {
      // use last activation variable associated with actuator i
      int act_adr = m->actuator_actadr[i] + actnum - 1;

      mjtNum act;
      if (m->actuator_actearly[i]) {
        act = mj_nextActivation(m, d, i, act_adr, d->act_dot[act_adr]);
      } else {
        act = d->act[act_adr];
      }

      // rotational setpoint: use representative nearest the length (local, no state change)
      mjtNum period = wrapPeriod(m, i);
      if (period > 0) {
        act = wrapSetpoint(act, d->actuator_length[oadr], period);
      }
      force[oadr] = gain * act;
    }

    // extract bias info
    const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*i;
    mjtBias biastype = m->actuator_biastype[i];

    // handle according to bias type
    switch (biastype) {
    case mjBIAS_NONE:               // none
      bias = 0.0;
      break;

    case mjBIAS_AFFINE:             // affine: biasprm = [const, kp, kv]
      bias = biasprm[0] + biasprm[1]*d->actuator_length[oadr] +
             biasprm[2]*d->actuator_velocity[oadr];
      break;

    case mjBIAS_MUSCLE:             // muscle passive force
      bias =  mju_muscleBias(d->actuator_length[oadr],
                             m->actuator_lengthrange+2*oadr,
                             m->actuator_acc0[oadr],
                             biasprm);
      break;

    case mjBIAS_DCMOTOR: {          // DC motor: back-EMF only (current-limited)
      bias = 0;

      // back-EMF (stateless only; for stateful current it's in the ODE)
      mjtNum te = m->actuator_dynprm[mjNDYN*i];  // electrical time constant
      if (te <= 0) {
        mjtNum K = gainprm[1];   // motor constant
        bias -= gain * K * d->actuator_velocity[oadr];
      }
      break;
    }

    default:                        // user bias
      if (mjcb_act_bias) {
        bias = mjcb_act_bias(m, d, i);
      } else {
        bias = 0;
      }
    }

    // add bias
    force[oadr] += bias;
  }

  // handle actuator plugins
  if (m->nplugin) {
    const int nslot = mjp_pluginCount();
    for (int i=0; i < m->nplugin; i++) {
      const int slot = m->plugin[i];
      const mjpPlugin* plugin = mjp_getPluginAtSlotUnsafe(slot, nslot);
      if (!plugin) {
        mjERROR("invalid plugin slot: %d", slot);
      }
      if (plugin->capabilityflags & mjPLUGIN_ACTUATOR) {
        if (!plugin->compute) {
          mjERROR("`compute` is a null function pointer for plugin at slot %d", slot);
        }
        plugin->compute(m, d, i, mjPLUGIN_ACTUATOR);
      }
    }
  }

  // clamp tendon total actuator force
  if (tendon_frclimited) {
    // compute total force for each tendon
    mjtNum* tendon_total_force = mjSTACKALLOC(d, ntendon, mjtNum);
    mju_zero(tendon_total_force, ntendon);
    for (int i=0; i < nactuator; i++) {
      if (m->actuator_trntype[i] == mjTRN_TENDON) {
        int tendon_id = m->actuator_trnid[2*i];
        if (m->tendon_actfrclimited[tendon_id]) {
          tendon_total_force[tendon_id] += force[m->actuator_outadr[i]];
        }
      }
    }

    // scale tendon actuator forces if limited and outside range
    for (int i=0; i < nactuator; i++) {
      if (m->actuator_trntype[i] != mjTRN_TENDON) {
        continue;
      }
      int tendon_id = m->actuator_trnid[2*i];
      mjtNum tendon_force = tendon_total_force[tendon_id];
      if (m->tendon_actfrclimited[tendon_id] && tendon_force) {
        const mjtNum* range = m->tendon_actfrcrange + 2 * tendon_id;
        if (tendon_force < range[0]) {
          force[m->actuator_outadr[i]] *= range[0] / tendon_force;
        } else if (tendon_force > range[1]) {
          force[m->actuator_outadr[i]] *= range[1] / tendon_force;
        }
      }
    }
  }

  // clamp actuator_force
  for (int i=0; i < nactuator; i++) {
    if (!m->actuator_forcelimited[i]) {
      continue;
    }
    const mjtNum* range = m->actuator_forcerange + 2*i;
    mjtNum* f = force + m->actuator_outadr[i];

    // SO3: clamp the norm of the output torque, preserving its direction
    if (m->actuator_gaintype[i] == mjGAIN_SO3) {
      mjtNum norm = mju_norm3(f);
      if (norm > range[1]) {
        mju_scl3(f, f, range[1]/norm);
      }
    }

    // otherwise: clamp each output
    else {
      int outnum = m->actuator_outnum[i];
      for (int j=0; j < outnum; j++) {
        f[j] = mju_clip(f[j], range[0], range[1]);
      }
    }
  }

  // add DC motor mechanical forces (not subject to current limits)
  for (int i=0; i < nactuator; i++) {
    if (m->actuator_biastype[i] != mjBIAS_DCMOTOR) {
      continue;
    }
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_ACTUATOR, i) == mjS_ASLEEP) {
      continue;
    }
    if (mj_actuatorDisabled(m, i) || m->actuator_plugin[i] >= 0) {
      continue;
    }

    const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*i;
    const mjtNum* dynprm = m->actuator_dynprm + mjNDYN*i;
    int oadr = m->actuator_outadr[i];

    // cogging torque
    mjtNum A = biasprm[0];
    if (A != 0) {
      mjtNum Np = biasprm[1];
      mjtNum phi = biasprm[2];
      force[oadr] += A * mju_sin(Np*d->actuator_length[oadr] + phi);
    }

    // LuGre friction
    mjtNum sigma0 = dynprm[5];
    if (sigma0 > 0) {
      mjtNum sigma1 = dynprm[6];
      mjDCMotorSlots slots = mj_dcmotorSlots(dynprm, m->actuator_gainprm + mjNGAIN*i);
      int adr = m->actuator_actadr[i] + slots.bristle;
      mjtNum z = d->act[adr];
      mjtNum z_dot = d->act_dot[adr];
      force[oadr] -= sigma0 * z + sigma1 * z_dot;
    }
  }

  // qfrc_actuator = moment' * force
  mju_mulMatTVecSparse(d->qfrc_actuator, d->actuator_moment, force, nout, nv,
                       d->moment_rownnz, d->moment_rowadr, d->moment_colind);

  // actuator-level gravity compensation
  if (m->flg_gravcomp && !mjDISABLED(mjDSBL_GRAVITY) && mju_norm3(m->opt.gravity)) {
    // number of dofs for each joint type: {mjJNT_FREE, mjJNT_BALL, mjJNT_SLIDE, mjJNT_HINGE}
    static const int jnt_dofnum[4] = {6, 3, 1, 1};
    int njnt = m->njnt;
    for (int i=0; i < njnt; i++) {
      // skip if gravcomp added as passive force
      if (!m->jnt_actgravcomp[i]) {
        continue;
      }

      // add gravcomp force
      int dofnum = jnt_dofnum[m->jnt_type[i]];
      int dofadr = m->jnt_dofadr[i];
      mju_addTo(d->qfrc_actuator + dofadr, d->qfrc_gravcomp + dofadr, dofnum);
    }
  }

  // clamp qfrc_actuator to joint-level actuator force limits
  clampVec(d->qfrc_actuator, m->jnt_actfrcrange, m->jnt_actfrclimited, m->njnt, m->jnt_dofadr);

  mj_freeStack(d);
  TM_END(mjTIMER_ACTUATION);
}


// add up all non-constraint forces, compute qacc_smooth
void mj_fwdAcceleration(const mjModel* m, mjData* d) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  int nv;
  const int* index;

  // qfrc_smooth = qfrc_passive - qfrc_bias + qfrc_applied + qfrc_actuator
  if (!sleep_filter) {
    nv = m->nv;
    index = NULL;
    mju_sub(d->qfrc_smooth, d->qfrc_passive, d->qfrc_bias, nv);
    mju_addTo(d->qfrc_smooth, d->qfrc_applied, nv);
    mju_addTo(d->qfrc_smooth, d->qfrc_actuator, nv);
  } else {
    nv = d->nv_awake;
    index = d->dof_awake_ind;
    mju_subInd(d->qfrc_smooth, d->qfrc_passive, d->qfrc_bias, index, nv);
    mju_addToInd(d->qfrc_smooth, d->qfrc_applied, index, nv);
    mju_addToInd(d->qfrc_smooth, d->qfrc_actuator, index, nv);
  }

  // qfrc_smooth += project(xfrc_applied)
  mj_xfrcAccumulate(m, d, d->qfrc_smooth);

  // implicit effective metric (built in mj_fwdPosition): the smooth acceleration is that of
  // the linearly-implicit dynamics, (M + K)*qacc_smooth = qfrc_smooth + c, so the constraint
  // solver, the no-constraint shortcut and the warmstart all see one consistent metric.
  if (d->efm_active) {
    mj_markStack(d);
    mjtNum* qfrc_eff = mjSTACKALLOC(d, m->nv, mjtNum);
    if (!sleep_filter) {
      mju_add(qfrc_eff, d->qfrc_smooth, d->efm_c, nv);
      if (d->efm_ca) {
        mju_addTo(qfrc_eff, d->efm_ca, nv);
      }
      mj_effSolve(m, d, d->qacc_smooth, qfrc_eff);
    } else {
      // awake dofs only: sleep islands follow the metric couplings, so the metric has no
      // terms across the awake/asleep boundary and zeroed asleep entries remain exactly
      // zero under the solve; scatter back only the awake results, asleep stay frozen
      mju_zero(qfrc_eff, m->nv);
      mju_addInd(qfrc_eff, d->qfrc_smooth, d->efm_c, index, nv);
      if (d->efm_ca) {
        mju_addToInd(qfrc_eff, d->efm_ca, index, nv);
      }
      mjtNum* qacc_eff = mjSTACKALLOC(d, m->nv, mjtNum);
      mj_effSolve(m, d, qacc_eff, qfrc_eff);
      mju_copyInd(d->qacc_smooth, qacc_eff, index, nv);
    }
    mj_freeStack(d);
    return;
  }

  // copy for in-place solve: qacc_smooth = qfrc_smooth
  if (!sleep_filter) {
    mju_copy(d->qacc_smooth, d->qfrc_smooth, nv);
  } else {
    mju_copyInd(d->qacc_smooth, d->qfrc_smooth, index, nv);
  }

  // qacc_smooth = M \ qfrc_smooth
  mj_solveLD(d->qacc_smooth, d->qLD, d->qLDiagInv, nv, 1,
             m->M_rownnz, m->M_rowadr, m->M_colind, index);
}


// warmstart/init solver
static void warmstart(const mjModel* m, mjData* d) {
  int nv = m->nv, nefc = d->nefc;

  // warmstart with best of (qacc_warmstart, qacc_smooth)
  if (!mjDISABLED(mjDSBL_WARMSTART)) {
    mj_markStack(d);
    mjtNum* jar = mjSTACKALLOC(d, nefc, mjtNum);

    // start with qacc = qacc_warmstart
    mju_copy(d->qacc, d->qacc_warmstart, nv);

    // compute jar(qacc_warmstart)
    mj_mulJacVec(m, d, jar, d->qacc_warmstart);
    mju_subFrom(jar, d->efc_aref, nefc);

    // update constraints, save cost(qacc_warmstart)
    mjtNum cost_warmstart;
    mj_constraintUpdate(m, d, jar, &cost_warmstart, 0);

    // PGS
    if (m->opt.solver == mjSOL_PGS) {
      // cost(force_warmstart)
      mjtNum PGS_warmstart = mju_dot(d->efc_force, d->efc_b, nefc);
      mjtNum* ARf = mjSTACKALLOC(d, nefc, mjtNum);
      if (mj_isSparse(m))
        mju_mulMatVecSparse(ARf, d->efc_AR, d->efc_force, nefc,
                            d->efc_AR_rownnz, d->efc_AR_rowadr,
                            d->efc_AR_colind, NULL);
      else {
        mju_mulMatVec(ARf, d->efc_AR, d->efc_force, nefc, nefc);
      }
      PGS_warmstart += 0.5*mju_dot(d->efc_force, ARf, nefc);

      // use zero if better
      if (PGS_warmstart > 0) {
        mju_zero(d->efc_force, nefc);
        mju_zero(d->qfrc_constraint, nv);
      }
    }

    // non-PGS
    else {
      // add Gauss to cost(qacc_warmstart)
      mjtNum* da = mjSTACKALLOC(d, nv, mjtNum);
      mjtNum* Mda = mjSTACKALLOC(d, nv, mjtNum);
      mju_sub(da, d->qacc_warmstart, d->qacc_smooth, nv);
      mj_mulM(m, d, Mda, da);
      if (d->efm_active) {
        mj_effMulAdd(m, d, Mda, da, /*flg_contact=*/1);
      }
      cost_warmstart += 0.5 * mju_dot(da, Mda, nv);

      // cost(qacc_smooth)
      mjtNum cost_smooth;
      mj_constraintUpdate(m, d, d->efc_b, &cost_smooth, 0);

      // use qacc_smooth if better
      if (cost_warmstart > cost_smooth) {
        mju_copy(d->qacc, d->qacc_smooth, nv);
      }
    }

    // have island structure: unconstrained qacc = qacc_smooth
    if (d->nisland > 0) {
      // loop over unconstrained dofs in map_idof2dof[nidof, nv)
      for (int i=d->nidof; i < nv; i++) {
        int dof = d->map_idof2dof[i];
        d->qacc[dof] = d->qacc_smooth[dof];
      }
    }

    mj_freeStack(d);
  }

  // coldstart with qacc = qacc_smooth, efc_force = 0
  else {
    mju_copy(d->qacc, d->qacc_smooth, nv);
    mju_zero(d->efc_force, nefc);
  }
}


// mju_dispatch callback: solve one island
static void solveIslandTask(const mjModel* m, mjData* d, void* arg, int thread_id, int island) {
  if (m->opt.solver == mjSOL_NEWTON) {
    mj_solNewton_island(m, d, island, m->opt.iterations);
  } else if (m->opt.solver == mjSOL_CG) {
    mj_solCG_island(m, d, island, m->opt.iterations);
  } else {
    mj_solPGS_island(m, d, island, m->opt.iterations);
  }
}


// discrete: reinstate the free-body gyroscopic derivative for decoupled standalone free
// bodies, re-solving their rows of qacc against the local unsymmetric metric. Guarded by
// mjd_freeGyroPossible: no constraint Jacobian support and no off-diagonal metric coupling
// on the 6 dofs, so the overwrite leaves all other rows and the constraint solution intact
static void mj_discreteGyro(const mjModel* m, mjData* d) {
  if (m->opt.integrator != mjINT_DISCRETE || !d->efm_active) {
    return;
  }
  for (int j=0; j < m->njnt; j++) {
    mjtNum A[36], rhs[6], x[6];

    // skip if the joint cannot take the decoupled local treatment
    if (!mjd_freeGyroPossible(m, d, j) ||
        !mjd_freeMhat(m, d, j, m->opt.timestep, A, /*flg_discrete=*/1)) {
      continue;
    }

    // rhs = qfrc_smooth + smooth-force shifts
    int adr = m->jnt_dofadr[j];
    mju_add(rhs, d->qfrc_smooth + adr, d->efm_c + adr, 6);
    if (d->efm_ca) {
      mju_addTo(rhs, d->efm_ca + adr, 6);
    }

    // overwrite the joint's rows of qacc with the local solve
    int pivot[6];
    if (mju_factorLU6(A, pivot)) {
      mju_solveLU6(x, A, rhs, pivot);
      mji_copy6(d->qacc + adr, x);
    }
  }
}


// compute efc_b, efc_force, qfrc_constraint; update qacc
static void fwdConstraint(const mjModel* m, mjData* d, mjtSolver solver, int flg_island) {
  TM_START;
  int nv = m->nv, nefc = d->nefc, nisland = d->nisland, nidof;

  // always clear qfrc_constraint
  mju_zero(d->qfrc_constraint, nv);

  // no constraints: copy unconstrained acc, clear forces, return
  // (with the effective metric active, qacc_smooth is already the implicit answer)
  if (!nefc) {
    mju_copy(d->qacc, d->qacc_smooth, nv);
    mju_zeroInt(d->solver_niter, mjNISLAND);
    mj_discreteGyro(m, d);
    TM_END(mjTIMER_CONSTRAINT);
    return;
  }

  // compute efc_b = J*qacc_smooth - aref
  mj_mulJacVec(m, d, d->efc_b, d->qacc_smooth);
  mju_subFrom(d->efc_b, d->efc_aref, nefc);

  // warmstart solver
  warmstart(m, d);
  mju_zeroInt(d->solver_niter, mjNISLAND);

  // check if islands are supported: IPC folds its subproblem monolithically, and so does CG when it
  // folds into the step's sparse flex factor
  int islands_supported = flg_island && nisland > 0 && !mjENABLED(mjENBL_IPC) &&
                          !mj_effFoldMonolithic(m, d, solver);

  // run solver over constraint islands
  if (islands_supported) {
    switch ((mjtSolver) solver) {
    case mjSOL_PGS:
      mju_dispatch(m, d, solveIslandTask, NULL, nisland);
      break;

    case mjSOL_CG:
    case mjSOL_NEWTON:
      // copy inputs to islands (vel+acc deps, pos-dependent already copied in mj_island)
      nidof = d->nidof;
      mju_gather(d->ifrc_smooth,     d->qfrc_smooth,     d->map_idof2dof, nidof);
      mju_gather(d->ifrc_constraint, d->qfrc_constraint, d->map_idof2dof, nidof);
      mju_gather(d->iacc_smooth,     d->qacc_smooth,     d->map_idof2dof, nidof);
      mju_gather(d->iacc,            d->qacc,            d->map_idof2dof, nidof);
      mju_gather(d->iefc_force,      d->efc_force,       d->map_iefc2efc, nefc);
      mju_gather(d->iefc_aref,       d->efc_aref,        d->map_iefc2efc, nefc);

      mju_dispatch(m, d, solveIslandTask, NULL, nisland);

      // copy back solver outputs (scatter dofs since ni <= nv)
      mju_scatter(d->qacc,            d->iacc,            d->map_idof2dof, nidof);
      mju_scatter(d->qfrc_constraint, d->ifrc_constraint, d->map_idof2dof, nidof);
      mju_gather(d->efc_force, d->iefc_force, d->map_efc2iefc, nefc);
      break;
    }

    // run noslip solver per island if enabled
    if (m->opt.noslip_iterations > 0) {
      for (int island=0; island < nisland; island++) {
        mj_solNoSlip_island(m, d, island, m->opt.noslip_iterations);
      }
    }
  }

  // run solver over all constraints (monolithic)
  else {
    switch ((mjtSolver) solver) {
    case mjSOL_PGS:                     // PGS
      mj_solPGS(m, d, m->opt.iterations);
      break;

    case mjSOL_CG:                      // CG
      mj_solCG(m, d, m->opt.iterations);
      break;

    case mjSOL_NEWTON:                  // Newton
      mj_solNewton(m, d, m->opt.iterations);
      break;
    }

    // run noslip solver if enabled
    if (m->opt.noslip_iterations > 0) {
      mj_solNoSlip(m, d, m->opt.noslip_iterations);
    }
  }

  // dual solvers: map efc_force to joint space (always monolithic)
  if (solver == mjSOL_PGS || m->opt.noslip_iterations > 0) {
    mj_dualFinish(m, d);
  }

  mj_discreteGyro(m, d);
  TM_END(mjTIMER_CONSTRAINT);
}

// forward constraint solve, using the model's own solver and island policy
void mj_fwdConstraint(const mjModel* m, mjData* d) {
  // check for invalid solver type, on the entry callers invoke (mj_fwdConstraintCG pins a valid
  // one); the condition mirrors fwdConstraint's early-out
  if (d->nefc && m->opt.solver != mjSOL_PGS &&
      m->opt.solver != mjSOL_CG && m->opt.solver != mjSOL_NEWTON) {
    mjERROR("unknown solver type %d", m->opt.solver);
  }
  fwdConstraint(m, d, (mjtSolver)m->opt.solver, !mjDISABLED(mjDSBL_ISLAND));
}


// forward constraint solve pinned to matrix-free CG over the monolithic problem, for the IPC
// mode's inner subproblem: the metric operators have no island-local form, and engine_ipc
// should not have to mutate m->opt to say so
void mj_fwdConstraintCG(const mjModel* m, mjData* d) {
  fwdConstraint(m, d, mjSOL_CG, 0);
}


//-------------------------- state advancement and integration  ------------------------------------

// the work every integrator does at the pre-step state before it writes the new one: history
// buffers, activations, sleep
//   act_dot: activation derivatives
static void advanceStart(const mjModel* m, mjData* d, const mjtNum* act_dot) {
  int nactuator = m->nactuator, nsensor = m->nsensor;

  // advance history buffers
  if (m->nhistory > 0) {
    // advance ctrl history buffers
    for (int i = 0; i < nactuator; i++) {
      int nsample = m->actuator_history[2*i];
      if (nsample == 0) continue;

      // get history buffer pointer and insert ctrl at current time
      int dim = m->actuator_ctrlnum[i];
      mjtNum* buf = d->history + m->actuator_historyadr[i];
      mjtNum* slot = mju_historyInsert(buf, nsample, dim, d->time);
      mju_copy(slot, d->ctrl + m->actuator_ctrladr[i], dim);
    }

    // advance sensor history buffers
    for (int i = 0; i < nsensor; i++) {
      int nsample = m->sensor_history[2*i];
      if (nsample == 0) continue;

      // get history buffer parameters
      int dim = m->sensor_dim[i];
      mjtNum* buf = d->history + m->sensor_historyadr[i];
      mjtNum delay = m->sensor_delay[i];
      mjtNum interval = m->sensor_interval[2*i];

      if (interval > 0) {
        // interval mode: if condition is satisfied, compute; otherwise copy
        mjtNum time_prev = buf[0];  // first slot stores previous sensor tick
        if (time_prev + interval <= d->time) {
          buf[0] += interval;  // advance by exact interval (continuous time)
          mjtNum* slot = mju_historyInsert(buf, nsample, dim, d->time);
          if (delay > 0) {
            // have delay, compute sensor
            mj_computeSensor(m, d, i, slot);
          } else {
            // no delay, copy from sensordata (already computed)
            mju_copy(slot, d->sensordata + m->sensor_adr[i], dim);
          }
        }
      } else if (delay > 0) {
        // delay-only mode: always compute and insert
        mjtNum* slot = mju_historyInsert(buf, nsample, dim, d->time);
        mj_computeSensor(m, d, i, slot);
      } else {
        // history-only mode: copy from sensordata (already computed)
        mjtNum* slot = mju_historyInsert(buf, nsample, dim, d->time);
        mju_copy(slot, d->sensordata + m->sensor_adr[i], dim);
      }
    }
  }

  // advance activations
  if (m->na && !mjDISABLED(mjDSBL_ACTUATION)) {
    for (int i=0; i < nactuator; i++) {
      int actadr = m->actuator_actadr[i];
      int actadr_end = actadr + m->actuator_actnum[i];
      for (int j=actadr; j < actadr_end; j++) {
        // if disabled, set act_dot to 0
        d->act[j] = mj_nextActivation(m, d, i, j, mj_actuatorDisabled(m, i) ? 0 : act_dot[j]);
      }
    }

    // rotational setpoints stored in act: replace with an equivalent bounded representative,
    // like the actrange clamp above, this is a projection applied at integration time
    for (int i=0; i < nactuator; i++) {
      if (m->actuator_dyntype[i] != mjDYN_INTEGRATOR) {
        continue;
      }

      // per-axis servo: wrap act to the representative nearest the length
      mjtNum period = wrapPeriod(m, i);
      if (period > 0) {
        int adr = m->actuator_actadr[i] + m->actuator_actnum[i] - 1;
        d->act[adr] = wrapSetpoint(d->act[adr], d->actuator_length[m->actuator_outadr[i]], period);
      }

      // SO3 servo: re-anchor the act setpoint to the canonical representative
      else if (m->actuator_gaintype[i] == mjGAIN_SO3) {
        int adr = m->actuator_actadr[i];
        mjtNum angle = mju_norm3(d->act + adr);
        if (angle > mjPI) {
          mjtNum scale = (angle - 2*mjPI*mju_round(angle/(2*mjPI))) / angle;
          for (int k=0; k < 3; k++) {
            d->act[adr+k] *= scale;
          }
        }
      }
    }
  }

  // put islands to sleep according to velocity tolerance
  if (mj_sleep(m, d)) {
    // if any trees put to sleep (qvel set to 0), recompute all velocity-dependent quantities
    mj_forwardSkip(m, d, mjSTAGE_POS, 0);

    // update sleep indices
    mj_updateSleep(m, d);
  }
}


// Gram determinant and optional gradient for 6 squared edge lengths in mj_stretchEdges[1] order
static mjtNum gramDet3D(mjtNum grad[6], const mjtNum q[6]) {
  mjtNum a = q[0], b = q[2], c = q[4];
  mjtNum d = 0.5 * (q[0] + q[2] - q[1]);
  mjtNum e = 0.5 * (q[0] + q[4] - q[5]);
  mjtNum f = 0.5 * (q[2] + q[4] - q[3]);
  mjtNum c0 = b*c - f*f, c1 = a*c - e*e, c2 = a*b - d*d;
  mjtNum c3 = e*f - c*d, c4 = d*f - b*e, c5 = d*e - a*f;
  if (grad) {
    grad[0] = c0 + c3 + c4;
    grad[1] = -c3;
    grad[2] = c1 + c3 + c5;
    grad[3] = -c5;
    grad[4] = c2 + c4 + c5;
    grad[5] = -c4;
  }
  return a*c0 + d*c3 + e*c4;
}


// the work every integrator does after it wrote the new state: time, plugin states, plastic springs
static void advanceFinish(const mjModel* m, mjData* d, const mjtNum* qvel) {
  // advance time
  d->time += m->opt.timestep;

  // advance plugin states
  if (m->nplugin) {
    const int nslot = mjp_pluginCount();
    for (int i = 0; i < m->nplugin; ++i) {
      const int slot = m->plugin[i];
      const mjpPlugin* plugin = mjp_getPluginAtSlotUnsafe(slot, nslot);
      if (!plugin) {
        mjERROR("invalid plugin slot: %d", slot);
      }
      if (plugin->advance) {
        plugin->advance(m, d, i);
      }
    }
  }

  // advance plastic spring references (return mapping)
  if (m->nplastic && !mjDISABLED(mjDSBL_SPRING)) {
    int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

    // joints
    for (int j = 0; j < m->njnt; j++) {
      if (m->jnt_plasticadr[j] < 0) {
        continue;
      }
      if (sleep_filter && d->body_awake[m->jnt_bodyid[j]] != mjS_AWAKE) {
        continue;
      }
      mjtNum* qspring = d->plastic + m->jnt_plasticadr[j];
      int padr = m->jnt_qposadr[j];
      mjtNum yield = mju_max(0, m->jnt_springyield[j]);

      mjtJoint type = (mjtJoint) m->jnt_type[j];
      if (type == mjJNT_SLIDE || type == mjJNT_HINGE) {
        mjtNum x = d->qpos[padr] - qspring[0];
        if (x > yield) {
          qspring[0] = d->qpos[padr] - yield;
        } else if (x < -yield) {
          qspring[0] = d->qpos[padr] + yield;
        }
      } else if (type == mjJNT_BALL) {
        mjtNum dif[3], quat[4];
        mju_copy4(quat, d->qpos + padr);
        mju_normalize4(quat);
        mju_subQuat(dif, quat, qspring);
        mjtNum r = mju_norm3(dif);
        if (r > yield) {
          mju_quatIntegrate(qspring, dif, (r - yield) / r);
          mju_normalize4(qspring);
        }
      }
    }

    // tendons
    for (int i = 0; i < m->ntendon; i++) {
      if (m->tendon_plasticadr[i] < 0) {
        continue;
      }
      if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, i) != mjS_AWAKE) {
        continue;
      }
      mjtNum* lengthspring = d->plastic + m->tendon_plasticadr[i];
      mjtNum yield = mju_max(0, m->tendon_springyield[i]);
      mjtNum L_next = 0;

      int adr = m->tendon_adr[i];
      if (m->wrap_type[adr] == mjWRAP_JOINT) {
        int num = m->tendon_num[i];
        for (int j = 0; j < num; j++) {
          int jntid = m->wrap_objid[adr + j];
          L_next += m->wrap_prm[adr + j] * d->qpos[m->jnt_qposadr[jntid]];
        }
      } else {
        int rownnz = m->ten_J_rownnz[i];
        int rowadr = m->ten_J_rowadr[i];
        L_next = d->ten_length[i] + m->opt.timestep *
                 mju_dotSparse(d->ten_J + rowadr, qvel, rownnz, m->ten_J_colind + rowadr);
      }

      mjtNum lower = lengthspring[0];
      mjtNum upper = lengthspring[1];
      mjtNum x = (L_next > upper) ? L_next - upper : (L_next < lower) ? L_next - lower : 0;
      if (x > yield) {
        mjtNum delta = x - yield;
        if (lower != 0 || upper <= 0) {
          lengthspring[0] += delta;
        }
        lengthspring[1] += delta;
      } else if (x < -yield) {
        mjtNum delta = x + yield;
        lengthspring[0] += delta;
        lengthspring[1] += delta;
      }
    }

    // flexes
    for (int f = 0; f < m->nflex; f++) {
      if (m->flex_plasticadr[f] < 0) {
        continue;
      }
      if (sleep_filter && mj_sleepState(m, d, mjOBJ_FLEX, f) != mjS_AWAKE) {
        continue;
      }

      mjtNum yield = mju_max(0, m->flex_yield[f]);
      mjtNum* plastic = d->plastic + m->flex_plasticadr[f];

      // interpolated flex: cell-center / face-center corotational strain return mapping
      if (m->flex_interp[f] != 0) {
        if (m->flex_edgeequality[f] == 3 || !mj_flexHasStiffness(m, f)) {
          continue;
        }
        int order = m->flex_interp[f];
        int shell_mode = order < 0;
        order = order < 0 ? -order : order;
        int cx = m->flex_cellnum[3*f + 0];
        int cy = m->flex_cellnum[3*f + 1];
        int cz = m->flex_cellnum[3*f + 2];
        int npe = shell_mode ? (order + 1) * (order + 1)
                             : (order + 1) * (order + 1) * (order + 1);
        int nelem_fe = shell_mode ? 2 * (cy*cz + cx*cz + cx*cy) : cx * cy * cz;
        int nodenum = m->flex_nodenum[f];
        int nstart = m->flex_nodeadr[f];
        const int* bodyid = m->flex_nodebodyid + nstart;
        const mjtNum* xpos0 = m->flex_node0 + 3 * nstart;
        const mjtNum* k = m->flex_stiffness + m->flex_stiffnessadr[f];

        mj_markStack(d);
        mjtNum* xpos_g = mjSTACKALLOC(d, 3 * nodenum, mjtNum);
        mjtNum* xpos_e = mjSTACKALLOC(d, 3 * npe, mjtNum);
        mjtNum* xpos0_e = mjSTACKALLOC(d, 3 * npe, mjtNum);
        mjtNum* displ_e = mjSTACKALLOC(d, 3 * npe, mjtNum);

        for (int i = 0; i < nodenum; i++) {
          int bid = bodyid[i];
          if (bid < 0) {
            mji_zero3(xpos_g + 3*i);
            continue;
          }
          if (m->flex_centered[f] ||
              (m->flex_node[3*(i + nstart) + 0] == 0 &&
               m->flex_node[3*(i + nstart) + 1] == 0 &&
               m->flex_node[3*(i + nstart) + 2] == 0)) {
            mji_copy3(xpos_g + 3*i, d->xpos + 3*bid);
          } else {
            mju_mulMatVec3(xpos_g + 3*i, d->xmat + 9*bid, m->flex_node + 3*(i + nstart));
            mji_addTo3(xpos_g + 3*i, d->xpos + 3*bid);
          }
          int body = m->body_weldid[bid];
          if (!m->body_dofnum[body]) {
            continue;
          }
          int da = m->body_dofadr[body];
          mjtNum vnode[3] = {0, 0, 0};
          if (m->body_simple[body] == 2) {
            for (int j = 0; j < m->body_dofnum[body]; j++) {
              mji_addToScl3(vnode, d->cdof + 6*(da + j) + 3, qvel[da + j]);
            }
          } else {
            mjtNum offset[3];
            mji_sub3(offset, xpos_g + 3*i, d->subtree_com + 3*m->body_rootid[body]);
            for (int j = da + m->body_dofnum[body] - 1; j >= 0; j = m->dof_parentid[j]) {
              mjtNum column[3];
              mji_cross(column, d->cdof + 6*j, offset);
              mji_addTo3(column, d->cdof + 6*j + 3);
              mji_addToScl3(vnode, column, qvel[j]);
            }
          }
          mji_addToScl3(xpos_g + 3*i, vnode, m->opt.timestep);
        }

        for (int fe = 0; fe < nelem_fe; fe++) {
          const mjtNum* k_elem = k + fe * 3*npe * 3*npe;
          if (!shell_mode && k_elem[0] == 0) {
            continue;
          }

          mjtNum quat[4];
          if (shell_mode) {
            mju_flexGatherFaceState(order, cx, cy, cz, fe, xpos_g, NULL, xpos0,
                                    xpos_e, NULL, xpos0_e, NULL, quat);
          } else {
            int ci = fe / (cy * cz);
            int cj = (fe / cz) % cy;
            int ck = fe % cz;
            mju_flexGatherCellState(order, cy, cz, ci, cj, ck, xpos_g, NULL,
                                    xpos0, xpos_e, NULL, xpos0_e, NULL, quat);
          }

          for (int n = 0; n < npe; n++) {
            mju_rotVecQuat(xpos_e + 3*n, xpos_e + 3*n, quat);
            mji_sub3(displ_e + 3*n, xpos_e + 3*n, xpos0_e + 3*n);
          }

          if (shell_mode) {
            int normal_axis = (fe < 2*cy*cz) ? 0 : (fe < 2*(cy*cz + cx*cz)) ? 1 : 2;
            int axis0 = (normal_axis + 1) % 3;
            int axis1 = (normal_axis + 2) % 3;
            mjtNum d0 = xpos0_e[3*(npe - 1) + axis0] - xpos0_e[axis0];
            mjtNum d1 = xpos0_e[3*(npe - 1) + axis1] - xpos0_e[axis1];
            if (d0 <= mjMINVAL || d1 <= mjMINVAL) {
              continue;
            }
            mjtNum t1[3] = {0, 0, 0}, t2[3] = {0, 0, 0};
            int idx = 0;
            for (int l0 = 0; l0 <= order; l0++) {
              for (int l1 = 0; l1 <= order; l1++) {
                mjtNum grad0 = mju_flexDphi(0.5, l0, order) * mju_flexPhi(0.5, l1, order);
                mjtNum grad1 = mju_flexPhi(0.5, l0, order) * mju_flexDphi(0.5, l1, order);
                for (int c = 0; c < 3; c++) {
                  t1[c] += displ_e[3*idx + c] * grad0;
                  t2[c] += displ_e[3*idx + c] * grad1;
                }
                idx++;
              }
            }
            mjtNum* sp = plastic + 3 * fe;
            mjtNum e0 = t1[axis0] / d0 - sp[0];
            mjtNum e1 = t2[axis1] / d1 - sp[1];
            mjtNum g01 = (t2[axis0] / d1 + t1[axis1] / d0) - sp[2];
            mjtNum alpha = 0.5 * (e0 + e1);
            mjtNum alpha_pos = mju_max(0, alpha);
            mjtNum dev0 = 0.5 * (e0 - e1);
            mjtNum dev1 = -dev0;
            mjtNum dev_norm_sq = 2 * dev0 * dev0 + 0.5 * g01 * g01;
            mjtNum eps_eq = mju_sqrt((2 * dev_norm_sq) / 3 + alpha_pos * alpha_pos);
            if (eps_eq > yield) {
              mjtNum scale = 1 - yield / eps_eq;
              sp[0] += scale * (dev0 + alpha_pos);
              sp[1] += scale * (dev1 + alpha_pos);
              sp[2] += scale * g01;
              mjtNum fnorm_sq = sp[0]*sp[0] + sp[1]*sp[1] + 0.5 * sp[2]*sp[2];
              if (fnorm_sq > 0.1225) {
                mjtNum clamp_scale = 0.35 / mju_sqrt(fnorm_sq);
                sp[0] *= clamp_scale;
                sp[1] *= clamp_scale;
                sp[2] *= clamp_scale;
              }
            }
          } else {
            mjtNum dx = xpos0_e[3*(npe - 1) + 0] - xpos0_e[0];
            mjtNum dy = xpos0_e[3*(npe - 1) + 1] - xpos0_e[1];
            mjtNum dz = xpos0_e[3*(npe - 1) + 2] - xpos0_e[2];
            if (dx <= mjMINVAL || dy <= mjMINVAL || dz <= mjMINVAL) {
              continue;
            }
            mjtNum p_center[3] = {0.5, 0.5, 0.5};
            mjtNum grad_d[9];
            mju_defGradient(grad_d, p_center, displ_e, order);
            mjtNum* sp = plastic + 6 * fe;
            mjtNum e_el[6] = {
              grad_d[0] / dx - sp[0],
              grad_d[4] / dy - sp[1],
              grad_d[8] / dz - sp[2],
              (grad_d[1] / dy + grad_d[3] / dx) - sp[3],
              (grad_d[2] / dz + grad_d[6] / dx) - sp[4],
              (grad_d[5] / dz + grad_d[7] / dy) - sp[5]
            };
            mjtNum mean = (e_el[0] + e_el[1] + e_el[2]) / 3;
            mjtNum dev[6] = {
              e_el[0] - mean,
              e_el[1] - mean,
              e_el[2] - mean,
              e_el[3],
              e_el[4],
              e_el[5]
            };
            mjtNum dev_norm_sq = dev[0]*dev[0] + dev[1]*dev[1] + dev[2]*dev[2] +
                                 0.5 * (dev[3]*dev[3] + dev[4]*dev[4] + dev[5]*dev[5]);
            mjtNum eps_eq = mju_sqrt((2 * dev_norm_sq) / 3);
            if (eps_eq > yield) {
              mjtNum scale = 1 - yield / eps_eq;
              for (int c = 0; c < 6; c++) {
                sp[c] += scale * dev[c];
              }
              mjtNum fnorm_sq = sp[0]*sp[0] + sp[1]*sp[1] + sp[2]*sp[2] +
                                0.5 * (sp[3]*sp[3] + sp[4]*sp[4] + sp[5]*sp[5]);
              if (fnorm_sq > 0.1225) {
                mjtNum clamp_scale = 0.35 / mju_sqrt(fnorm_sq);
                for (int c = 0; c < 6; c++) {
                  sp[c] *= clamp_scale;
                }
              }
            }
          }
        }
        mj_freeStack(d);
        continue;
      }

      if (m->flex_dim[f] < 2) {
        continue;
      }

      int dim = m->flex_dim[f];
      int nedge = (dim == 2) ? 3 : 6;
      int vertnum = m->flex_vertnum[f];
      int edgenum = m->flex_edgenum[f];
      int elemnum = m->flex_elemnum[f];

      mj_markStack(d);
      mjtNum* xpos = mjSTACKALLOC(d, 3 * vertnum, mjtNum);
      mj_flexGather(m, d, f, xpos, qvel);
      mju_addScl(xpos, d->flexvert_xpos + 3 * m->flex_vertadr[f], xpos,
                 m->opt.timestep, 3 * vertnum);

      // element stretch return mapping (von Mises / J2 deviatoric strain)
      int stiffnessadr = m->flex_stiffnessadr[f];
      int stretch_pnum = 0;
      if (stiffnessadr >= 0 && m->flex_stiffness[stiffnessadr] != 0 && elemnum > 0) {
        stretch_pnum = nedge * elemnum;
        const mjtNum* k = m->flex_stiffness + stiffnessadr;
        const int* eelem = m->flex_elemedge + m->flex_elemedgeadr[f];
        const int* edges = m->flex_edge + 2 * m->flex_edgeadr[f];
        const mjtNum* rest = m->flexedge_length0 + m->flex_edgeadr[f];
        int stride = (dim == 2) ? 21 : 24;

        mjtNum* deformed_sq = mjSTACKALLOC(d, edgenum, mjtNum);
        for (int e = 0; e < edgenum; e++) {
          mjtNum diff[3];
          mju_sub3(diff, xpos + 3 * edges[2*e + 1], xpos + 3 * edges[2*e]);
          deformed_sq[e] = mju_dot3(diff, diff);
        }

        for (int t = 0; t < elemnum; t++) {
          const mjtNum* packed = k + stride * t;
          const int* te = eelem + nedge * t;
          mjtNum* splastic = plastic + nedge * t;

          mjtNum g[6], s_el[6];
          for (int e = 0; e < nedge; e++) {
            int idx = te[e];
            g[e] = rest[idx] * rest[idx];
            s_el[e] = deformed_sq[idx] - g[e] - splastic[e];
          }

          mjtNum metric[36], f_vec[6], h_vec[6];
          mj_stretchElasticity(metric, f_vec, packed, s_el, nedge);
          mj_stretchTension(h_vec, metric, g, nedge);

          mjtNum g_h = mju_dot(g, h_vec, nedge);
          if (mju_abs(g_h) <= mjMINVAL) {
            continue;
          }
          mjtNum alpha = mju_dot(g, f_vec, nedge) / g_h;

          mjtNum s_dev[6], f_dev[6];
          for (int e = 0; e < nedge; e++) {
            s_dev[e] = s_el[e] - alpha * g[e];
            f_dev[e] = f_vec[e] - alpha * h_vec[e];
          }

          // compute tr(P * K_t) where P_ab = (e0_a . e0_b)^2
          mjtNum trPK = 0;
          for (int a = 0; a < nedge; a++) {
            trPK += g[a] * g[a] * metric[nedge * a + a];
          }
          if (dim == 2) {
            mjtNum d01 = g[0] + g[1] - g[2];
            mjtNum d02 = g[0] + g[2] - g[1];
            mjtNum d12 = g[1] + g[2] - g[0];
            trPK += 0.5 * (d01 * d01 * metric[3*0 + 1] +
                           d02 * d02 * metric[3*0 + 2] +
                           d12 * d12 * metric[3*1 + 2]);
          } else {
            static const int faces[4][3] = {
              {0, 1, 2}, {0, 4, 5}, {1, 3, 5}, {2, 3, 4}
            };
            for (int fi = 0; fi < 4; fi++) {
              int a = faces[fi][0], b = faces[fi][1], c = faces[fi][2];
              mjtNum dab = g[a] + g[b] - g[c];
              mjtNum dac = g[a] + g[c] - g[b];
              mjtNum dbc = g[b] + g[c] - g[a];
              trPK += 0.5 * (dab * dab * metric[6*a + b] +
                             dac * dac * metric[6*a + c] +
                             dbc * dbc * metric[6*b + c]);
            }
            mjtNum s0 = g[0] + g[3];
            mjtNum s1 = g[1] + g[4];
            mjtNum s2 = g[2] + g[5];
            mjtNum d03 = s1 - s2, d14 = s2 - s0, d25 = s0 - s1;
            trPK += 0.5 * (d03 * d03 * metric[6*0 + 3] +
                           d14 * d14 * metric[6*1 + 4] +
                           d25 * d25 * metric[6*2 + 5]);
          }

          mjtNum mu_t = (trPK - g_h / dim) / (nedge - 1);
          if (mu_t <= mjMINVAL) {
            continue;
          }

          mjtNum dev_norm_sq = mju_max(0, mju_dot(s_dev, f_dev, nedge) / (4 * mu_t));
          mjtNum alpha_pos = (dim == 2) ? mju_max(0, alpha) : 0;
          mjtNum eps_eq = mju_sqrt((2 * dev_norm_sq) / 3 + alpha_pos * alpha_pos);
          if (eps_eq > yield) {
            mjtNum scale = 1 - yield / eps_eq;
            if (dim == 2) {
              for (int e = 0; e < 3; e++) {
                splastic[e] += scale * (s_dev[e] + alpha_pos * g[e]);
                splastic[e] = mju_max(splastic[e], -0.99 * g[e]);
              }
            } else {
              mjtNum sp_dev[6], grad[6];
              for (int e = 0; e < 6; e++) {
                sp_dev[e] = splastic[e] + scale * s_dev[e];
              }
              mjtNum beta = mju_dot(h_vec, sp_dev, 6) / g_h;
              for (int e = 0; e < 6; e++) {
                sp_dev[e] -= beta * g[e];
              }

              // solve det(G(u*g + sp_dev)) = det(G(g)) via depressed cubic c3*u^3 + c1*u + c0 - c3 = 0
              mjtNum c3 = gramDet3D(NULL, g);
              mjtNum c0 = gramDet3D(grad, sp_dev);
              mjtNum c1 = mju_dot(grad, g, 6);
              mjtNum u = 1;
              if (c3 > mjMINVAL) {
                mjtNum m2 = mju_max(0, -c1 / (3 * c3));
                if (m2 > mjMINVAL) {
                  mjtNum m_val = mju_sqrt(m2);
                  mjtNum R = (c3 - c0) / (2 * c3 * m_val * m2);
                  if (R >= 1) {
                    mjtNum w_cbrt = mju_pow(R + mju_sqrt(mju_max(0, R*R - 1)), 1.0 / 3.0);
                    u = m_val * (w_cbrt + 1 / w_cbrt);
                  } else {
                    u = 2 * m_val * mju_cos(mju_acos(mju_max(-1, R)) / 3);
                  }
                }
              }
              for (int e = 0; e < 6; e++) {
                splastic[e] = mju_max(sp_dev[e] + (u - 1) * g[e], -0.99 * g[e]);
              }
            }
            d->flex_hessian_valid[f] = 0;
          }
        }
      }

      // 2D shell bending return mapping in dihedral angle space
      int bendingadr = m->flex_bendingadr[f];
      if (dim == 2 && bendingadr >= 0 && stiffnessadr >= 0) {
        // 2D shell thickness is stored in stiffness[6] (stiffness[0:6] holds the 2D stretch metric)
        mjtNum thickness = m->flex_stiffness[stiffnessadr + 6];
        if (thickness > mjMINVAL) {
          mjtNum* bend_plastic = plastic + stretch_pnum;
          const mjtNum* b = m->flex_bending + bendingadr;
          const mjtNum* rest = m->flexedge_length0 + m->flex_edgeadr[f];
          mjtNum kappa_yield = 2 * yield / thickness;

          for (int e = 0; e < edgenum; e++) {
            const int* edge = m->flex_edge + 2 * (e + m->flex_edgeadr[f]);
            const int* flap = m->flex_edgeflap + 2 * (e + m->flex_edgeadr[f]);
            int v[4] = {edge[0], edge[1], flap[0], flap[1]};
            if (v[3] == -1) {
              continue;
            }
            const mjtNum* be = b + 17 * e;
            mjtNum b23 = be[11];
            if (b23 <= mjMINVAL) {
              continue;
            }

            mjtNum ed[3][3], frc2[3], frc3[3];
            mji_sub3(ed[0], xpos + 3*v[1], xpos + 3*v[0]);
            mji_sub3(ed[1], xpos + 3*v[2], xpos + 3*v[0]);
            mji_sub3(ed[2], xpos + 3*v[3], xpos + 3*v[0]);
            mji_cross(frc2, ed[2], ed[0]);
            mji_cross(frc3, ed[0], ed[1]);

            mjtNum two_a1 = mju_norm3(frc3);
            mjtNum two_a2 = mju_norm3(frc2);
            mjtNum l0 = rest[e];
            mjtNum denom = two_a1 * two_a2;
            if (denom <= mjMINVAL || l0 <= mjMINVAL) {
              continue;
            }

            mjtNum omega = mju_dot3(ed[2], frc3);
            mjtNum theta = mju_atan2(omega * l0, mju_dot3(frc2, frc3));
            mjtNum theta_p = mju_atan2(-(be[16] + bend_plastic[e]) * l0, b23);
            mjtNum theta_yield = 0.5 * (two_a1 + two_a2) / l0 * kappa_yield;
            mjtNum theta_el = theta - theta_p;
            if (theta_el > theta_yield || theta_el < -theta_yield) {
              theta_p = (theta_el > theta_yield) ? theta - theta_yield : theta + theta_yield;
              theta_p = mju_clip(theta_p, -1.3, 1.3);
              bend_plastic[e] = -(b23 / l0) * mju_tan(theta_p) - be[16];
            }
          }
        }
      }

      mj_freeStack(d);
    }
  }
}


// advance state and time
//   act_dot: activation derivatives
//   qacc:    acceleration used to update d->qvel (d->qvel += h*qacc)
//   qvel:    optional velocity used for position integration; if NULL, use d->qvel
void mj_advance(const mjModel* m, mjData* d,
                const mjtNum* act_dot, const mjtNum* qacc, const mjtNum* qvel) {
  advanceStart(m, d, act_dot);

  // advance velocities
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
  if (sleep_filter) {
    mju_addToSclInd(d->qvel, qacc, d->dof_awake_ind, m->opt.timestep, d->nv_awake);
  } else {
    mju_addToScl(d->qvel, qacc, m->opt.timestep, m->nv);
  }

  // advance positions with qvel if given, d->qvel otherwise (semi-implicit)
  const int* index = sleep_filter ? d->body_awake_ind : NULL;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;
  const mjtNum* v = qvel ? qvel : d->qvel;
  mj_integratePosInd(m, d->qpos, v, m->opt.timestep, index, nbody);

  advanceFinish(m, d, v);

  // save qacc for next step warmstart
  mju_copy(d->qacc_warmstart, d->qacc, m->nv);
}


// advance state and time to a verified endpoint, for an integrator that solved the step at the
// position level: the effective acceleration is published first, as it is the step's acceleration
// at the pre-step state (a delayed acceleration sensor reads it there); then the pre-step work,
// then the endpoint and its velocity replace the state in full, then time and plugins. The
// warm start is left to the caller
//   act_dot: activation derivatives
//   qpos:    the endpoint
//   qvel:    the velocity at the endpoint
//   qacc:    the effective acceleration of the step, (qvel - d->qvel)/h on the dofs it moved
void mj_commit(const mjModel* m, mjData* d, const mjtNum* act_dot,
               const mjtNum* qpos, const mjtNum* qvel, const mjtNum* qacc) {
  mju_copy(d->qacc, qacc, m->nv);
  advanceStart(m, d, act_dot);
  mju_copy(d->qpos, qpos, m->nq);
  mju_copy(d->qvel, qvel, m->nv);
  advanceFinish(m, d, d->qvel);
}

// Euler integrator, semi-implicit in velocity, possibly skipping factorisation
void mj_EulerSkip(const mjModel* m, mjData* d, int skipfactor) {
  TM_START;
  mj_markStack(d);
  mjtNum* qfrc = mjSTACKALLOC(d, m->nv, mjtNum);
  mjtNum* qacc = mjSTACKALLOC(d, m->nv, mjtNum);

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  int nv = sleep_filter ? d->nv_awake : m->nv;
  const int* dof_awake_ind = sleep_filter ? d->dof_awake_ind : NULL;

  // check for dof damping if disable flag is not set
  int dof_damping = 0;
  if (!mjDISABLED(mjDSBL_EULERDAMP) && !mjDISABLED(mjDSBL_DAMPER)) {
    for (int v=0; v < nv; v++) {
      int i = sleep_filter ? dof_awake_ind[v] : v;
      if (m->dof_damping[i] > 0 ||
          !mju_isZero(m->dof_dampingpoly + mjNPOLY*i, mjNPOLY) ||
          m->jnt_actuatorid[m->dof_jntid[i]] != -1) {
        dof_damping = 1;
        break;
      }
    }
  }

  // no damping or disabled: explicit velocity integration
  if (!dof_damping) {
    if (sleep_filter) {
      mju_copyInd(qacc, d->qacc, dof_awake_ind, nv);
    } else {
      mju_copy(qacc, d->qacc, nv);
    }
  }

  // damping: integrate implicitly
  else {
    if (!skipfactor) {
      // qH = M
      if (sleep_filter) {
        mju_copySparse(d->qH, d->M, m->M_rownnz, m->M_rowadr, dof_awake_ind, d->nv_awake);
      } else {
        mju_copy(d->qH, d->M, m->nC);
      }

      // qH += h*diag(B)
      for (int v=0; v < nv; v++) {
        int i = sleep_filter ? dof_awake_ind[v] : v;
        mjtNum qv = d->qvel[i];
        mjtNum poly[mjNPOLY];
        mju_copy(poly, m->dof_dampingpoly + mjNPOLY*i, mjNPOLY);
        mjtNum damping = m->dof_damping[i]
                         + mj_actuatorDamping(m, mjOBJ_JOINT, m->dof_jntid[i], poly);
        mjtNum damp_deriv = mjd_xPolyForce(damping, poly, qv, mjNPOLY, 1);
        d->qH[m->M_rowadr[i] + m->M_rownnz[i] - 1] += m->opt.timestep * damp_deriv;
      }

      // factorize in-place; warn if a near-singular pivot was clamped
      int clamped = mj_factorI(d->qH, d->qHDiagInv, nv,
                               m->M_rownnz, m->M_rowadr, m->M_colind, dof_awake_ind);
      if (clamped >= 0) {
        mj_warning(d, mjWARN_INERTIA, clamped);
      }
    }

    // solve
    if (sleep_filter) {
      mju_addInd(qfrc, d->qfrc_smooth, d->qfrc_constraint, dof_awake_ind, nv);
      mju_copyInd(qacc, qfrc, dof_awake_ind, nv);
    } else {
      mju_add(qfrc, d->qfrc_smooth, d->qfrc_constraint, nv);
      mju_copy(qacc, qfrc, nv);
    }
    mj_solveLD(qacc, d->qH, d->qHDiagInv, nv, 1,
               m->M_rownnz, m->M_rowadr, m->M_colind, dof_awake_ind);
  }

  // advance state and time
  mj_advance(m, d, d->act_dot, qacc, NULL);

  mj_freeStack(d);

  TM_END(mjTIMER_ADVANCE);
}


// Euler integrator, semi-implicit in velocity
void mj_Euler(const mjModel* m, mjData* d) {
  mj_EulerSkip(m, d, 0);
}


// discrete integrator: the solver's qacc is the step map, advance directly
void mj_discrete(const mjModel* m, mjData* d) {
  TM_START;
  mj_advance(m, d, d->act_dot, d->qacc, NULL);
  TM_END(mjTIMER_ADVANCE);
}


// RK4 tableau
const mjtNum RK4_A[9] = {
  0.5,    0,      0,
  0,      0.5,    0,
  0,      0,      1
};

const mjtNum RK4_B[4] = {
  1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0
};


// Runge-Kutta-Munthe-Kaas (RKMK) correction: orientations are written as q = q0 * exp(rot) and
//  RK integrates the rotation vector rot, an ODE on a vector space, so the order of the tableau is
//  kept; the rate of rot is not the angular velocity but dexp^{-1}_{-rot} of it, and summing the
//  uncorrected stage angular velocities makes orientations only second-order accurate;
//  map the rotational dofs of ball and free joints in vel to rotation vector rates, given the stage
//  position increment velocity dqvel (rotation vector = h*dqvel); other dofs are unchanged
static void rkCorrectVel(const mjModel* m, mjtNum* vel, const mjtNum* dqvel, mjtNum h) {
  for (int j=0; j < m->njnt; j++) {
    int vadr = m->jnt_dofadr[j];
    int type = m->jnt_type[j];
    if (type == mjJNT_FREE) {
      vadr += 3;
    } else if (type != mjJNT_BALL) {
      continue;
    }
    mjtNum rot[3] = {h*dqvel[vadr], h*dqvel[vadr+1], h*dqvel[vadr+2]};
    mju_derivExpmap(vel+vadr, rot, vel+vadr);
  }
}


// Runge Kutta explicit order-N integrator
//  (A,B) is the tableau, C is set to row_sum(A)
void mj_RungeKutta(const mjModel* m, mjData* d, int N) {
  int nv = m->nv, nq = m->nq, na = m->na;
  mjtNum h = m->opt.timestep, time = d->time;
  mjtNum C[9], T[9], *X[10], *F[10], *dX;
  const mjtNum* A = (N == 4 ? RK4_A : 0);
  const mjtNum* B = (N == 4 ? RK4_B : 0);

  // check order
  if (!A) {
    mjERROR("supported RK orders: N=4");
  }

  // allocate space for intermediate solutions
  mj_markStack(d);
  dX = mjSTACKALLOC(d, 2*nv+na, mjtNum);
  for (int i=0; i < N; i++) {
    X[i] = mjSTACKALLOC(d, nq+nv+na, mjtNum);
    F[i] = mjSTACKALLOC(d, nv+na, mjtNum);
  }

  // precompute C and T;  C,T,A have size (N-1)
  for (int i=1; i < N; i++) {
    // C(i) = sum_j A(i,j)
    C[i-1] = 0;
    for (int j=0; j < i; j++) {
      C[i-1] += A[(i-1)*(N-1)+j];
    }

    // compute T
    T[i-1] = d->time + C[i-1]*h;
  }

  // init X[0], F[0]; mj_forward() was already called
  mju_copy(X[0], d->qpos, nq);
  mju_copy(X[0]+nq, d->qvel, nv);  // rotation vector is 0: no RKMK correction
  mju_copy(F[0], d->qacc, nv);
  if (na) {
    mju_copy(X[0]+nq+nv, d->act, na);
    mju_copy(F[0]+nv, d->act_dot, na);
  }
  mjtNum *ten_length0 = NULL, *ten_J0 = NULL;
  if (m->nplastic && m->ntendon) {
    ten_length0 = mjSTACKALLOC(d, m->ntendon, mjtNum);
    mju_copy(ten_length0, d->ten_length, m->ntendon);
    if (m->nJten) {
      ten_J0 = mjSTACKALLOC(d, m->nJten, mjtNum);
      mju_copy(ten_J0, d->ten_J, m->nJten);
    }
  }
  mjtNum *flexvert_xpos0 = NULL, *cdof0 = NULL, *subtree_com0 = NULL;
  if (m->nplastic && m->nflexvert) {
    flexvert_xpos0 = mjSTACKALLOC(d, 3 * m->nflexvert, mjtNum);
    mju_copy(flexvert_xpos0, d->flexvert_xpos, 3 * m->nflexvert);
    if (m->nv) {
      cdof0 = mjSTACKALLOC(d, 6 * m->nv, mjtNum);
      mju_copy(cdof0, d->cdof, 6 * m->nv);
    }
    if (m->nbody) {
      subtree_com0 = mjSTACKALLOC(d, 3 * m->nbody, mjtNum);
      mju_copy(subtree_com0, d->subtree_com, 3 * m->nbody);
    }
  }

  // compute the remaining X[i], F[i]
  for (int i=1; i < N; i++) {
    // compute dX
    mju_zero(dX, 2*nv+na);
    for (int j=0; j < i; j++) {
      mju_addToScl(dX, X[j]+nq, A[(i-1)*(N-1)+j], nv);
      mju_addToScl(dX+nv, F[j], A[(i-1)*(N-1)+j], nv+na);
    }

    // compute X[i] = X[0] '+' dX
    mju_copy(X[i], X[0], nq+nv+na);
    mj_integratePos(m, X[i], dX, h);
    mju_addToScl(X[i]+nq, dX+nv, h, nv+na);

    // set X[i], T[i-1] in mjData
    mju_copy(d->qpos, X[i], nq);
    mju_copy(d->qvel, X[i]+nq, nv);
    if (na) {
      mju_copy(d->act, X[i]+nq+nv, na);
    }
    d->time = T[i-1];

    // RKMK: d->qvel keeps the stage velocity; X[i]+nq is only read below as a position rate,
    //  map its angular velocities through dexp^{-1}_{-h*dX} in place
    rkCorrectVel(m, X[i]+nq, dX, h);

    // evaluate F[i]
    mj_forwardSkip(m, d, mjSTAGE_NONE, 1);  // 1: do not recompute sensors and energy
    mju_copy(F[i], d->qacc, nv);
    if (na) {
      mju_copy(F[i]+nv, d->act_dot, na);
    }
  }

  // compute dX for final update (using B instead of A)
  mju_zero(dX, 2*nv+na);
  for (int j=0; j < N; j++) {
    mju_addToScl(dX, X[j]+nq, B[j], nv);
    mju_addToScl(dX+nv, F[j], B[j], nv+na);
  }

  // reset state and time
  d->time = time;
  mju_copy(d->qpos, X[0], nq);
  mju_copy(d->qvel, X[0]+nq, nv);
  mju_copy(d->act, X[0]+nq+nv, na);
  if (ten_length0) {
    mju_copy(d->ten_length, ten_length0, m->ntendon);
    if (ten_J0) {
      mju_copy(d->ten_J, ten_J0, m->nJten);
    }
  }
  if (flexvert_xpos0) {
    mju_copy(d->flexvert_xpos, flexvert_xpos0, 3 * m->nflexvert);
    if (cdof0) {
      mju_copy(d->cdof, cdof0, 6 * m->nv);
    }
    if (subtree_com0) {
      mju_copy(d->subtree_com, subtree_com0, 3 * m->nbody);
    }
  }

  // effective acceleration of the step (RK weighted combination), read by delayed sensors
  mju_copy(d->qacc, dX+nv, nv);

  // advance state and time
  mj_advance(m, d, dX+2*nv, dX+nv, dX);

  // warm start the next step from the last stage's acceleration, the most recent solve
  mju_copy(d->qacc_warmstart, F[N-1], nv);

  mj_freeStack(d);
}


// runtime option validation for the discrete integrator
void mj_checkDiscrete(const mjModel* m) {
  // the IPC contact mode is a mode of the discrete integrator: its inner solves are the
  // discrete solve, so no other integrator can host it
  if (mjENABLED(mjENBL_IPC) && m->opt.integrator != mjINT_DISCRETE) {
    mjERROR("flag ipc requires integrator='discrete'");
  }

  // the inner solves are matrix-free CG: the metric operators are never factored, which keeps
  // them well defined under stiff contact
  if (mjENABLED(mjENBL_IPC) && m->opt.solver != mjSOL_CG) {
    mjERROR("flag ipc requires solver='CG'");
  }

  // inverse dynamics is not supported under the flag (mj_discreteAcc), and this option runs it
  // on every step
  if (mjENABLED(mjENBL_IPC) && mjENABLED(mjENBL_FWDINV)) {
    mjERROR("flag ipc does not support flag fwdinv: inverse dynamics is not available");
  }

  // the IPC step appends every articulated tree to its solve and writes the result back, with no
  // notion of a sleeping tree; sleeping bodies are also invisible to its flex broad phase
  if (mjENABLED(mjENBL_IPC) && mjENABLED(mjENBL_SLEEP)) {
    mjERROR("flag ipc does not support flag sleep");
  }

  // SNH and passive flex contact require the discrete integrator's effective metric
  if (!mj_isMetric(m)) {
    for (int f=0; f < m->nflex; f++) {
      int stiffnessadr = m->flex_stiffnessadr[f];
      if (!m->flex_interp[f] && m->flex_dim[f] == 3 && stiffnessadr >= 0 &&
          m->flex_stiffness[stiffnessadr+21] != 0) {
        mjERROR("stable Neo-Hookean elasticity requires integrator='discrete'");
      }
      if (mj_effFlexContactPossible(m, f)) {
        mjERROR("passive flex contact requires an integrator with the effective metric: "
                "set integrator='discrete'");
      }
    }

    // until the discrete integrator, flex elasticity was integrated implicitly under
    // implicit and implicitfast; those models need the metric. Euler and RK4 always
    // integrated it explicitly and still do
    if (m->opt.integrator == mjINT_IMPLICIT || m->opt.integrator == mjINT_IMPLICITFAST) {
      for (int f=0; f < m->nflex; f++) {
        if (mj_effFlexStiffPossible(m, f)) {
          mjERROR("flex elasticity is no longer integrated implicitly under "
                  "integrator='implicit' and 'implicitfast': set "
                  "integrator='discrete', or 'Euler' with a small timestep");
        }
      }
    }
    return;
  }
  for (int f=0; f < m->nflex; f++) {
    if (!m->flex_rigid[f] && (m->flex_edgestiffness[f] > 0 || m->flex_edgedamping[f] > 0)) {
      mjERROR("discrete integrator: flex edge stiffness and damping are not supported; "
              "use <elasticity> or <edge equality> instead");
    }
  }
  if (m->opt.solver == mjSOL_NEWTON) {
    for (int f=0; f < m->nflex; f++) {
      if (!m->flex_interp[f] && mj_effFlexStiffPossible(m, f) && !mj_flexSimple(m, f)) {
        mjERROR("discrete integrator: flex with general attachments requires solver='CG'");
      }
    }
  }
  if (m->opt.solver == mjSOL_NEWTON && !mjd_flexInterpAssemblable(m)) {
    mjERROR("discrete integrator: interpolated flex with non-simple nodes not yet "
            "supported with the Newton solver");
  }
  if (m->opt.solver == mjSOL_PGS || m->opt.noslip_iterations > 0) {
    // the dual solvers assemble the constraint-space AR from an explicit factorization of
    // the metric, and the only factor is the qH backbone. Under PGS the tendon and
    // actuator classes are therefore excluded from the metric and integrate explicitly
    // (mj_effCouplings), and the backbone IS the metric; under a primal solver, the
    // noslip post-pass consumes the backbone AR as an approximation. Flex cannot be
    // excluded: explicit flex elasticity is the divergence the metric exists to prevent
    if (mj_effFlexAny(m)) {
      mjERROR("discrete integrator: PGS and noslip not yet supported with flex");
    }
  }
  if (mjENABLED(mjENBL_SLEEP)) {
    if (mjDISABLED(mjDSBL_ISLAND)) {
      mjERROR("discrete integrator: sleep without islands not yet supported");
    }
  }
}


// fully implicit in velocity, possibly skipping factorization
void mj_implicitSkip(const mjModel* m, mjData* d, int skipfactor) {
  TM_START;
  int nD = m->nD, nC = m->nC, njnt = m->njnt;

  mj_markStack(d);
  mjtNum* qfrc = mjSTACKALLOC(d, m->nv, mjtNum);
  mjtNum* qacc = mjSTACKALLOC(d, m->nv, mjtNum);

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  int nv = sleep_filter ? d->nv_awake : m->nv;
  const int* dof_awake_ind = sleep_filter ? d->dof_awake_ind : NULL;

  // set qfrc = qfrc_smooth + qfrc_constraint
  if (sleep_filter) {
    mju_addInd(qfrc, d->qfrc_smooth, d->qfrc_constraint, dof_awake_ind, nv);
  } else {
    mju_add(qfrc, d->qfrc_smooth, d->qfrc_constraint, nv);
  }

  // factorization
  if (!skipfactor) {
    // implicit
    if (m->opt.integrator == mjINT_IMPLICIT) {
      // compute analytical derivative qDeriv
      mjd_smooth_vel(m, d, /* flg_bias = */ 1);

      // gather qLU <- M (lower to full)
      mju_gatherMasked(d->qLU, d->M, m->mapM2D, nD);

      // set qLU = M - dt*qDeriv
      mju_addToScl(d->qLU, d->qDeriv, -m->opt.timestep, nD);
    }

    // implicitfast
    else if (m->opt.integrator == mjINT_IMPLICITFAST) {
      // compute analytical derivative qDeriv; skip rne derivative
      mjd_smooth_vel(m, d, /* flg_bias = */ 0);

      // modified mass matrix: gather qH <- qDeriv (full to lower)
      mju_gather(d->qH, d->qDeriv, m->mapD2M, nC);

      // set qH = M - dt*qDeriv
      mju_addScl(d->qH, d->M, d->qH, -m->opt.timestep, nC);

      // standalone free bodies: reset qH block rows to M; their qDeriv rows may be asymmetric and
      // are handled by the local LU solve; we reset to keep LTL well-defined
      for (int j=0; j < njnt; j++) {
        if (m->jnt_type[j] != mjJNT_FREE || !mj_isFreeBody(m, m->jnt_bodyid[j])) {
          continue;
        }
        int adr = m->jnt_dofadr[j];
        for (int r=0; r < 6; r++) {
          mju_copy(d->qH + m->M_rowadr[adr+r], d->M + m->M_rowadr[adr+r], m->M_rownnz[adr+r]);
        }
      }
    } else {
      mjERROR("integrator must be implicit or implicitfast");
    }

    // standard factorization (implicit / implicitfast)
    int clamped;
    if (m->opt.integrator == mjINT_IMPLICIT) {
      int* scratch = mjSTACKALLOC(d, nv, int);
      clamped = mju_factorLUSparse(d->qLU, nv, scratch,
                                   m->D_rownnz, m->D_rowadr, m->D_colind, dof_awake_ind);
    } else {
      clamped = mj_factorI(d->qH, d->qHDiagInv, nv,
                           m->M_rownnz, m->M_rowadr, m->M_colind, dof_awake_ind);
    }

    // warn if a near-singular pivot was clamped
    if (clamped >= 0) {
      mj_warning(d, mjWARN_INERTIA, clamped);
    }
  }

  // standard sparse solve
  if (m->opt.integrator == mjINT_IMPLICIT) {
    mju_solveLUSparse(qacc, d->qLU, qfrc, nv, m->D_rownnz, m->D_rowadr, m->D_diag, m->D_colind,
                      dof_awake_ind);
  } else {
    // implicitfast
    if (sleep_filter) {
      mju_copyInd(qacc, qfrc, dof_awake_ind, nv);
    } else {
      mju_copy(qacc, qfrc, nv);
    }
    mj_solveLD(qacc, d->qH, d->qHDiagInv, nv, 1, m->M_rownnz, m->M_rowadr, m->M_colind, dof_awake_ind);
  }

  // implicitfast: local unsymmetric solve for standalone free bodies
  // adds the bias (gyroscopic) derivative, dropped from the global symmetric solve; the
  // 6x6 block of M - h*D is decoupled from the rest of the system (D sparsity is tree-local),
  // so overwriting these rows of qacc leaves all other DOFs unaffected
  if (m->opt.integrator == mjINT_IMPLICITFAST) {
    for (int j=0; j < m->njnt; j++) {
      mjtNum A[36];
      if (!mjd_freeMhat(m, d, j, m->opt.timestep, A, /*flg_discrete=*/0)) {
        continue;
      }

      // solve A * qacc_block = qfrc_block
      int adr = m->jnt_dofadr[j];
      int pivot[6];
      if (mju_factorLU6(A, pivot)) {
        mjtNum x[6];  // local vector for guaranteed memory alignment
        mju_solveLU6(x, A, qfrc+adr, pivot);
        mji_copy6(qacc+adr, x);
      }
    }
  }

  // advance state and time
  mj_advance(m, d, d->act_dot, qacc, NULL);

  mj_freeStack(d);

  TM_END(mjTIMER_ADVANCE);
}


// fully implicit in velocity
void mj_implicit(const mjModel* m, mjData* d) {
  mj_implicitSkip(m, d, 0);
}


//-------------------------- top-level API ---------------------------------------------------------

// under the ipc flag a model with a dim-2 flex takes its constraint solves inside mj_ipc, so the
// forward pipeline skips its own (mj_ipc would discard it); a rigid-only model keeps it
static int ipcSkipConstraint(const mjModel* m) {
  if (!mjENABLED(mjENBL_IPC)) return 0;
  for (int i = 0; i < m->nflex; i++) {
    if (m->flex_dim[i] == 2) return 1;
  }
  return 0;
}


// the constraint stage, unless the IPC mode owns it (ipcSkipConstraint): then qacc stays at the
// free-flight acceleration, the integrator computes the step and recomputes the acceleration-stage
// sensors from it (mj_ipc)
static void fwdConstraintStage(const mjModel* m, mjData* d) {
  if (ipcSkipConstraint(m)) {
    mju_copy(d->qacc, d->qacc_smooth, m->nv);
    mju_zeroInt(d->solver_niter, mjNISLAND);
  } else {
    mj_fwdConstraint(m, d);
  }
}


// forward dynamics with skip; skipstage is mjtStage
void mj_forwardSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor) {
  TM_START;

  // validate option combinations for the discrete integrator
  mj_checkDiscrete(m);

  // position-dependent
  if (skipstage < mjSTAGE_POS) {
    mj_fwdPosition(m, d);

    if (!skipsensor) {
      mj_sensorPos(m, d);
    }

    if (!d->flg_energypos) {
      if (mjENABLED(mjENBL_ENERGY)) {
        mj_energyPos(m, d);
      } else {
        d->energy[0] = d->energy[1] = 0;
      }
    }
  }

  // velocity-dependent
  if (skipstage < mjSTAGE_VEL) {
    mj_fwdVelocity(m, d);

    if (!skipsensor) {
      mj_sensorVel(m, d);
    }

    if (mjENABLED(mjENBL_ENERGY) && !d->flg_energyvel) {
      mj_energyVel(m, d);
    }
  }

  // acceleration-dependent
  if (mjcb_control && !mjDISABLED(mjDSBL_ACTUATION)) {
    mjcb_control(m, d);
  }

  mj_fwdActuation(m, d);
  mj_effActuation(m, d, /*flg_factor=*/1);
  if (mj_isMetric(m)) {
    mj_regularizeConstraint(m, d, /*flg_AR=*/1);
    mj_referenceConstraint(m, d);
  }
  mj_fwdAcceleration(m, d);
  fwdConstraintStage(m, d);
  if (!skipsensor) {
    d->flg_rnepost = 0;  // clear flag for lazy evaluation
    mj_sensorAcc(m, d);
  }

  TM_END(mjTIMER_FORWARD);
}


// forward dynamics
void mj_forward(const mjModel* m, mjData* d) {
  mj_forwardSkip(m, d, mjSTAGE_NONE, 0);
}


// advance simulation using control callback
void mj_step(const mjModel* m, mjData* d) {
  TM_START;

  // common to all integrators
  mj_checkPos(m, d);
  mj_checkVel(m, d);
  // under the ipc flag with a flex the constraint stage is skipped (ipcSkipConstraint): qacc_smooth
  // is the free-flight acceleration mj_ipc linearizes about, and mj_ipc runs the solves itself and
  // recomputes the acceleration-stage sensors from its result
  mj_forward(m, d);
  mj_checkAcc(m, d);

  // compare forward and inverse solutions if enabled
  if (mjENABLED(mjENBL_FWDINV)) {
    mj_compareFwdInv(m, d);
  }

  // use selected integrator
  switch ((mjtIntegrator) m->opt.integrator) {
  case mjINT_EULER:
    mj_Euler(m, d);
    break;

  case mjINT_RK4:
    mj_RungeKutta(m, d, 4);
    break;

  case mjINT_IMPLICIT:
  case mjINT_IMPLICITFAST:
    mj_implicit(m, d);
    break;

  case mjINT_DISCRETE:
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


// advance simulation in two phases: before input is set by user
void mj_step1(const mjModel* m, mjData* d) {
  TM_START;
  mj_checkDiscrete(m);
  mj_checkPos(m, d);
  mj_checkVel(m, d);
  mj_fwdPosition(m, d);
  mj_sensorPos(m, d);

  if (!d->flg_energypos) {
    if (mjENABLED(mjENBL_ENERGY)) {
      mj_energyPos(m, d);
    } else {
      d->energy[0] = d->energy[1] = 0;
    }
  }

  mj_fwdVelocity(m, d);
  mj_sensorVel(m, d);
  if (mjENABLED(mjENBL_ENERGY) && !d->flg_energyvel) {
    mj_energyVel(m, d);
  }

  if (mjcb_control) {
    mjcb_control(m, d);
  }
  TM_END(mjTIMER_STEP);
}


//   >>>>   user can modify ctrl and q/xfrc_applied between step1 and step2   <<<<


// advance simulation in two phases: after input is set by user
void mj_step2(const mjModel* m, mjData* d) {
  TM_START;
  mj_fwdActuation(m, d);
  mj_effActuation(m, d, /*flg_factor=*/1);
  if (mj_isMetric(m)) {
    mj_regularizeConstraint(m, d, /*flg_AR=*/1);
    mj_referenceConstraint(m, d);
  }
  mj_fwdAcceleration(m, d);
  fwdConstraintStage(m, d);
  d->flg_rnepost = 0;  // clear flag for lazy evaluation
  mj_sensorAcc(m, d);
  mj_checkAcc(m, d);

  // compare forward and inverse solutions if enabled
  if (mjENABLED(mjENBL_FWDINV)) {
    mj_compareFwdInv(m, d);
  }

  // integrate with Euler, implicit or discrete; RK4 defaults to Euler
  if (m->opt.integrator == mjINT_DISCRETE) {
    if (mjENABLED(mjENBL_IPC)) {
      mj_ipc(m, d);
    } else {
      mj_discrete(m, d);
    }
  } else if (m->opt.integrator == mjINT_IMPLICIT || m->opt.integrator == mjINT_IMPLICITFAST) {
    mj_implicit(m, d);
  } else {
    mj_Euler(m, d);
  }

  d->timer[mjTIMER_STEP].number--;
  TM_END(mjTIMER_STEP);
}
