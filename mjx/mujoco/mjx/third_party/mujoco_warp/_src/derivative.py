# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Optional

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import history
from mujoco.mjx.third_party.mujoco_warp._src import math
from mujoco.mjx.third_party.mujoco_warp._src import passive
from mujoco.mjx.third_party.mujoco_warp._src import smooth
from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src import util_misc
from mujoco.mjx.third_party.mujoco_warp._src.passive import _rod_hinge
from mujoco.mjx.third_party.mujoco_warp._src.passive import _rod_pad44
from mujoco.mjx.third_party.mujoco_warp._src.passive import build_efm_contact
from mujoco.mjx.third_party.mujoco_warp._src.passive import ellipsoid_max_moment
from mujoco.mjx.third_party.mujoco_warp._src.passive import geom_semiaxes
from mujoco.mjx.third_party.mujoco_warp._src.passive import rod_stencil
from mujoco.mjx.third_party.mujoco_warp._src.passive import rod_stencil_jacobian
from mujoco.mjx.third_party.mujoco_warp._src.support import next_act
from mujoco.mjx.third_party.mujoco_warp._src.types import FLEX_INTERP_MAX_NODES
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import ROD_HINGE_SIZE
from mujoco.mjx.third_party.mujoco_warp._src.types import BiasType
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DeterminismType
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import DynType
from mujoco.mjx.third_party.mujoco_warp._src.types import EnableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import GainType
from mujoco.mjx.third_party.mujoco_warp._src.types import IntegratorType
from mujoco.mjx.third_party.mujoco_warp._src.types import JointType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import TrnType
from mujoco.mjx.third_party.mujoco_warp._src.types import mat43
from mujoco.mjx.third_party.mujoco_warp._src.types import vec6
from mujoco.mjx.third_party.mujoco_warp._src.types import vec10
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def _jnt_actfrc_clamped(
  # Model:
  jnt_dofadr: wp.array[int],
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  # Data in:
  qfrc_actuator_in: wp.array2d[float],
  # In:
  worldid: int,
  actid: int,
) -> bool:
  trntype = actuator_trntype[actid]
  if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:
    jntid = actuator_trnid[actid][0]
    if jnt_actfrclimited[jntid]:
      jnt_force = qfrc_actuator_in[worldid, jnt_dofadr[jntid]]
      actfrcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]
      if jnt_force <= actfrcrange[0] or jnt_force >= actfrcrange[1]:
        return True
  return False


@wp.func
def _tendon_actfrc_clamped(
  # Model:
  tendon_actfrclimited: wp.array[bool],
  tendon_actfrcrange: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  # Data in:
  act_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # In:
  worldid: int,
  actid: int,
) -> bool:
  if actuator_trntype[actid] == TrnType.TENDON:
    tenid = actuator_trnid[actid][0]
    if tendon_actfrclimited[tenid]:
      ten_force = float(0.0)
      for j in range(actuator_force_in.shape[1]):
        if actuator_trntype[j] == TrnType.TENDON and actuator_trnid[j][0] == tenid:
          f = actuator_force_in[worldid, j]
          if actuator_biastype[j] == BiasType.DCMOTOR:
            f -= util_misc.dcmotor_mech_force(
              act_in,
              act_dot_in,
              worldid,
              actuator_actadr[j],
              actuator_length_in[worldid, j],
              actuator_dynprm[worldid % actuator_dynprm.shape[0], j],
              actuator_gainprm[worldid % actuator_gainprm.shape[0], j],
              actuator_biasprm[worldid % actuator_biasprm.shape[0], j],
            )
          ten_force += f
      actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]
      tol = 1.0e-5
      if ten_force <= actfrcrange[0] + tol * wp.abs(actfrcrange[0]) or ten_force >= actfrcrange[1] - tol * wp.abs(
        actfrcrange[1]
      ):
        return True
  return False


@wp.func
def _dcmotor_saturation(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  tendon_actfrclimited: wp.array[bool],
  tendon_actfrcrange: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_ctrladr: wp.array[int],
  actuator_ctrlnum: wp.array[int],
  actuator_ctrlspec: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # In:
  worldid: int,
  actid: int,
  R: float,
) -> int:
  """DC motor saturation flags: 1 electrical force clamped, 2 Vmax saturated, 4 dimax clamped."""
  h = opt_timestep[worldid % opt_timestep.shape[0]]
  dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]
  gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], actid]
  K = gainprm[1]
  te = dynprm[0]
  actadr = actuator_actadr[actid]
  velocity = actuator_velocity_in[worldid, actid]
  slots = util_misc.dcmotor_slots(dynprm, gainprm)
  flags = int(0)

  # effective first control: clamped, then slew-limited (matching forward)
  uadr = actuator_ctrladr[actid]
  ctrlnum = actuator_ctrlnum[actid]
  spec = actuator_ctrlspec[actid]
  u_first = float(0.0)
  if ctrlnum > 0:
    u_first = ctrl_in[worldid, uadr]
    if actuator_ctrllimited[uadr] and not (opt_disableflags & DisableBit.CLAMPCTRL):
      ctrlrange = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], uadr]
      u_first = wp.clamp(u_first, ctrlrange[0], ctrlrange[1])
    if actadr >= 0 and slots[0] >= 0:
      u_prev = act_in[worldid, actadr + slots[0]]
      slew = dynprm[7] * h
      u_first = wp.clamp(u_first, u_prev - slew, u_prev + slew)

  # terminal voltage, with controller Vmax saturation
  V = float(0.0)
  if ctrlnum > 0:
    if (spec & 7) != 0:
      x_I = float(0.0)
      if actadr >= 0 and slots[1] >= 0:
        x_I = act_in[worldid, actadr + slots[1]]
      u4 = util_misc.unpack_servo_inputs(ctrl_in, worldid, uadr, spec, u_first)
      length = actuator_length_in[worldid, actid]
      torque = gainprm[4] * (u4[0] - length) + gainprm[6] * (u4[1] - velocity) + gainprm[5] * x_I + u4[2]
      V = gainprm[0] / K * torque + K * velocity
      Vmax = gainprm[7]
      if Vmax > 0.0 and wp.abs(V) >= Vmax:
        flags = flags | 2
        V = wp.clamp(V, -Vmax, Vmax)
      V += u4[3]
    else:
      V = u_first

  # pre-mechanical electrical force, current-rate saturation
  f_elec = float(0.0)
  if te > 0.0:
    act_last = actadr + actuator_actnum[actid] - 1
    current = act_in[worldid, act_last]
    current_dot = act_dot_in[worldid, act_last]
    if actuator_actearly[actid]:
      current += current_dot * te * (1.0 - wp.exp(-h / te))
      if actuator_actlimited[actid]:
        actrange = actuator_actrange[worldid % actuator_actrange.shape[0], actid]
        current = wp.clamp(current, actrange[0], actrange[1])
    f_elec = K * current
    dimax = dynprm[1]
    if dimax > 0.0 and wp.abs(current_dot) >= dimax:
      flags = flags | 4
  else:
    f_elec = K / R * (V - K * velocity)

  if actuator_forcelimited[actid]:
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
    if f_elec <= forcerange[0] or f_elec >= forcerange[1]:
      flags = flags | 1

  if (flags & 1) == 0 and _tendon_actfrc_clamped(
    tendon_actfrclimited,
    tendon_actfrcrange,
    actuator_trntype,
    actuator_biastype,
    actuator_actadr,
    actuator_trnid,
    actuator_dynprm,
    actuator_gainprm,
    actuator_biasprm,
    act_in,
    act_dot_in,
    actuator_length_in,
    actuator_force_in,
    worldid,
    actid,
  ):
    flags = flags | 1

  return flags


@wp.kernel
def _qderiv_actuator_passive_vel(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_dofadr: wp.array[int],
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  tendon_actfrclimited: wp.array[bool],
  tendon_actfrcrange: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_ctrladr: wp.array[int],
  actuator_ctrlnum: wp.array[int],
  actuator_ctrlspec: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  actuator_acc0: wp.array2d[float],
  actuator_lengthrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  qfrc_actuator_in: wp.array2d[float],
  # Out:
  vel_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  actuator_gainprm_id = worldid % actuator_gainprm.shape[0]
  actuator_biasprm_id = worldid % actuator_biasprm.shape[0]
  actuator_dynprm_id = worldid % actuator_dynprm.shape[0]
  opt_timestep_id = worldid % opt_timestep.shape[0]

  bias = float(0.0)
  gaintype = actuator_gaintype[actid]
  biastype = actuator_biastype[actid]
  R_dcmotor = float(0.0)
  sat_dcmotor = int(0)
  if gaintype == GainType.DCMOTOR or biastype == BiasType.DCMOTOR:
    dynprm_dc = actuator_dynprm[actuator_dynprm_id, actid]
    gainprm_dc = actuator_gainprm[actuator_gainprm_id, actid]
    R_dcmotor = util_misc.dcmotor_resistance(act_in, worldid, actuator_actadr[actid], dynprm_dc, gainprm_dc)
    sat_dcmotor = _dcmotor_saturation(
      opt_timestep,
      opt_disableflags,
      tendon_actfrclimited,
      tendon_actfrcrange,
      actuator_trntype,
      actuator_biastype,
      actuator_ctrladr,
      actuator_ctrlnum,
      actuator_ctrlspec,
      actuator_actadr,
      actuator_actnum,
      actuator_trnid,
      actuator_dynprm,
      actuator_gainprm,
      actuator_biasprm,
      actuator_actlimited,
      actuator_actrange,
      actuator_actearly,
      actuator_forcelimited,
      actuator_forcerange,
      actuator_ctrllimited,
      actuator_ctrlrange,
      act_in,
      ctrl_in,
      act_dot_in,
      actuator_length_in,
      actuator_velocity_in,
      actuator_force_in,
      worldid,
      actid,
      R_dcmotor,
    )

  if gaintype == GainType.AFFINE:
    gain = actuator_gainprm[actuator_gainprm_id, actid][2]
  elif gaintype == GainType.MUSCLE:
    gain = util_misc.muscle_gain_vel_deriv(
      actuator_length_in[worldid, actid],
      actuator_velocity_in[worldid, actid],
      actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid],
      actuator_acc0[worldid % actuator_acc0.shape[0], actid],
      actuator_gainprm[actuator_gainprm_id, actid],
    )
  elif gaintype == GainType.DCMOTOR:
    gain = 0.0
    dynprm = actuator_dynprm[actuator_dynprm_id, actid]
    gainprm = actuator_gainprm[actuator_gainprm_id, actid]
    te = dynprm[0]
    K = gainprm[1]

    # electrical terms vanish when electrical force is clamped by forcerange or tendon actfrcrange
    if (sat_dcmotor & 1) == 0:
      # controller velocity derivative dV/dω using nameplate resistance, zero under Vmax
      dVdw = 0.0
      if (actuator_ctrlspec[actid] & 7) != 0 and (sat_dcmotor & 2) == 0:
        R0 = wp.max(MJ_MINVAL, gainprm[0])
        dVdw = -gainprm[6] * R0 / K + K

      if te > 0.0:
        # stateful current with actearly: d(K*next_act)/dω, zero under dimax
        # includes both back-EMF (-K) and controller (dVdw) through act_dot
        if actuator_actearly[actid] and (sat_dcmotor & 4) == 0:
          s = 1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te)
          bias += K * (dVdw - K) * s / R_dcmotor
      elif dVdw != 0.0:
        # stateless: controller terms only (back-EMF handled in bias block)
        bias += K * dVdw / R_dcmotor

    # LuGre: force includes -sigma1*z_dot, z_dot = a*z + v
    # d(sigma1*z_dot)/dv = sigma1*(da/dv*z + 1), ignoring higher-order da/dv*z
    sigma1 = dynprm[6]
    if sigma1 > 0.0:
      bias -= sigma1
  else:
    gain = 0.0

  if biastype == BiasType.AFFINE:
    bias += actuator_biasprm[actuator_biasprm_id, actid][2]
  elif biastype == BiasType.DCMOTOR:
    if actuator_dynprm[actuator_dynprm_id, actid][0] <= 0.0 and (sat_dcmotor & 1) == 0:
      K = actuator_gainprm[actuator_gainprm_id, actid][1]
      bias -= K * K / R_dcmotor

  if bias == 0.0 and gain == 0.0:
    vel_out[worldid, actid] = 0.0
    return

  # skip if force is clamped by forcerange (DC motor: electrical force, handled above)
  if actuator_forcelimited[actid] and gaintype != GainType.DCMOTOR and biastype != BiasType.DCMOTOR:
    force = actuator_force_in[worldid, actid]
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
    if force <= forcerange[0] or force >= forcerange[1]:
      vel_out[worldid, actid] = 0.0
      return

  # skip if tendon total actuator force is clamped by actfrcrange
  if (
    gaintype != GainType.DCMOTOR
    and biastype != BiasType.DCMOTOR
    and _tendon_actfrc_clamped(
      tendon_actfrclimited,
      tendon_actfrcrange,
      actuator_trntype,
      actuator_biastype,
      actuator_actadr,
      actuator_trnid,
      actuator_dynprm,
      actuator_gainprm,
      actuator_biasprm,
      act_in,
      act_dot_in,
      actuator_length_in,
      actuator_force_in,
      worldid,
      actid,
    )
  ):
    vel_out[worldid, actid] = 0.0
    return

  # skip if joint total actuator force is clamped by actfrcrange
  # (keeping tendon/site actuator contributions on this dof is an approximation)
  if _jnt_actfrc_clamped(
    jnt_dofadr,
    jnt_actfrclimited,
    jnt_actfrcrange,
    actuator_trntype,
    actuator_trnid,
    qfrc_actuator_in,
    worldid,
    actid,
  ):
    vel_out[worldid, actid] = 0.0
    return

  vel = float(bias)
  if actuator_dyntype[actid] != DynType.NONE:
    if gain != 0.0:
      act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1

      # use next activation if actearly is set (matching forward pass)
      if actuator_actearly[actid]:
        act = next_act(
          opt_timestep[opt_timestep_id],
          actuator_dyntype[actid],
          actuator_dynprm[actuator_dynprm_id, actid],
          actuator_actrange[worldid % actuator_actrange.shape[0], actid],
          act_in[worldid, act_adr],
          act_dot_in[worldid, act_adr],
          1.0,
          actuator_actlimited[actid],
        )
      else:
        act = act_in[worldid, act_adr]

      vel += gain * act
  else:
    if gain != 0.0:
      vel += gain * ctrl_in[worldid, actid]

  vel_out[worldid, actid] = vel


@wp.func
def _nonzero_mask(x: float) -> float:
  """Returns 1.0 for non-zero input, 0.0 otherwise."""
  if x != 0.0:
    return 1.0
  return 0.0


@cache_kernel
def _qderiv_actuator_passive_actuation_sparse(nM: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nM

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    M_elemid: wp.array2d[int],
    # Data in:
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    # In:
    vel_in: wp.array2d[float],
    # Out:
    qDeriv_out: wp.array2d[float],
  ):
    worldid, actid = wp.tid()

    vel = vel_in[worldid, actid]
    if vel == 0.0:
      return

    rownnz = moment_rownnz_in[worldid, actid]
    rowadr = moment_rowadr_in[worldid, actid]

    for i in range(rownnz):
      rowadri = rowadr + i
      moment_i = actuator_moment_in[worldid, rowadri]
      if moment_i == 0.0:
        continue
      dofi = moment_colind_in[worldid, rowadri]

      for j in range(i + 1):
        rowadrj = rowadr + j
        moment_j = actuator_moment_in[worldid, rowadrj]
        if moment_j == 0.0:
          continue
        dofj = moment_colind_in[worldid, rowadrj]

        elemid = M_elemid[dofi, dofj]
        if elemid >= 0:
          contrib = moment_i * moment_j * vel
          wp.atomic_add(qDeriv_out[worldid], elemid, contrib)

  return kernel


@wp.kernel
def _qderiv_actuator_passive(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  M_elemid: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  M_in: wp.array2d[float],
  # In:
  Mi: wp.array[int],
  Mj: wp.array[int],
  qDeriv_in: wp.array2d[float],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  worldid, elemid = wp.tid()

  dofiid = Mi[elemid]
  dofjid = Mj[elemid]

  # Off-pattern (dofiid, dofjid) pairs have no CSR entry (madr < 0).
  madr = M_elemid[dofiid, dofjid]
  if madr < 0:
    return

  qderiv = qDeriv_in[worldid, madr]

  if not (opt_disableflags & DisableBit.DAMPER) and dofiid == dofjid:
    damping, dpoly = util_misc.dof_damping_total(
      jnt_actuatorid,
      dof_jntid,
      dof_damping,
      dof_dampingpoly,
      actuator_trntype,
      actuator_trnid,
      actuator_damping,
      actuator_dampingpoly,
      actuator_gear,
      worldid,
      dofiid,
    )
    v = qvel_in[worldid, dofiid]
    qderiv -= util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  qDeriv_out[worldid, madr] = M_in[worldid, madr] - qderiv


# TODO(team): improve performance with tile operations?
@wp.kernel
def _qderiv_tendon_damping(
  # Model:
  ntendon: int,
  opt_timestep: wp.array[float],
  tendon_actuatorid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  M_elemid: wp.array2d[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # In:
  Mi: wp.array[int],
  Mj: wp.array[int],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  worldid, elemid = wp.tid()
  dofiid = Mi[elemid]
  dofjid = Mj[elemid]

  # Off-pattern (dofiid, dofjid) pairs have no CSR entry (madr < 0).
  madr = M_elemid[dofiid, dofjid]
  if madr < 0:
    return

  qderiv = float(0.0)
  for tenid in range(ntendon):
    damping, dpoly = util_misc.tendon_damping_total(
      tendon_actuatorid,
      tendon_damping,
      tendon_dampingpoly,
      actuator_trntype,
      actuator_trnid,
      actuator_damping,
      actuator_dampingpoly,
      actuator_gear,
      worldid,
      tenid,
    )
    if damping == 0.0 and dpoly[0] == 0.0 and dpoly[1] == 0.0:
      continue

    rownnz = ten_J_rownnz[tenid]
    rowadr = ten_J_rowadr[tenid]
    Ji = float(0.0)
    Jj = float(0.0)
    for k in range(rownnz):
      if Ji != 0.0 and Jj != 0.0:
        break
      sparseid = rowadr + k
      colind = ten_J_colind[sparseid]
      if colind == dofiid:
        Ji = ten_J_in[worldid, sparseid]
      if colind == dofjid:
        Jj = ten_J_in[worldid, sparseid]

    v = ten_velocity_in[worldid, tenid]
    qderiv -= Ji * Jj * util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  qDeriv_out[worldid, madr] -= qderiv


@wp.kernel
def deriv_rne_cvel_cdof_dot(
  # Model:
  body_parentid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_dofadr: wp.array[int],
  jnt_type: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  body_tree_: wp.array[int],
  # Out:
  Dcvel_out: wp.array3d[wp.spatial_vector],
  Dcdof_dot_out: wp.array3d[wp.spatial_vector],
):
  """Forward pass: compute d(cvel)/d(qvel_k) and d(cdof_dot)/d(qvel_k).

  Mirrors the accumulation order of comvel for each joint type.

  Dcdof_dot for rotation DOFs of free joints (dofid+0..2) is zero because the
  forward pass sets cdof_dot[dofid+0..2] = 0.  The Dcdof_dot array is
  zero-initialized so no explicit write is needed.
  """
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  dofadr = body_dofadr[bodyid]
  jntid = body_jntadr[bodyid]
  jntnum = body_jntnum[bodyid]
  pid = body_parentid[bodyid]

  cdof = cdof_in[worldid]

  # Initialize from parent
  cvel_k = Dcvel_out[worldid, pid, dofid]

  if jntnum == 0:
    Dcvel_out[worldid, bodyid, dofid] = cvel_k
    return

  dof_i = dofadr

  for j in range(jntid, jntid + jntnum):
    jnttype = jnt_type[j]

    if jnttype == 0:  # FREE
      # rotation DOFs (dof_i+0..2) contribute to cvel
      if dofid >= dof_i and dofid < dof_i + 3:
        cvel_k += cdof[dofid]

      # cdof_dot for rotation DOFs is zero (set in forward kinematics),
      # so Dcdof_dot for rotation DOFs is zero (from wp.zeros init)

      # derivative of cdof_dot for translation DOFs 3,4,5
      Dcdof_dot_out[worldid, dof_i + 3, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 3])
      Dcdof_dot_out[worldid, dof_i + 4, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 4])
      Dcdof_dot_out[worldid, dof_i + 5, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 5])

      # translation DOFs (dof_i+3..5) contribute to cvel
      if dofid >= dof_i + 3 and dofid < dof_i + 6:
        cvel_k += cdof[dofid]

      dof_i += 6

    elif jnttype == 1:  # BALL
      Dcdof_dot_out[worldid, dof_i + 0, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 0])
      Dcdof_dot_out[worldid, dof_i + 1, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 1])
      Dcdof_dot_out[worldid, dof_i + 2, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 2])

      if dofid >= dof_i and dofid < dof_i + 3:
        cvel_k += cdof[dofid]

      dof_i += 3
    else:  # HINGE or SLIDE
      Dcdof_dot_out[worldid, dof_i, dofid] = math.motion_cross(cvel_k, cdof[dof_i])

      if dofid == dof_i:
        cvel_k += cdof[dof_i]

      dof_i += 1

  Dcvel_out[worldid, bodyid, dofid] = cvel_k


@wp.kernel
def deriv_rne_cacc_cfrcbody_forward(
  # Model:
  body_parentid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cinert_in: wp.array2d[vec10],
  cvel_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  body_tree_: wp.array[int],
  Dcvel_in: wp.array3d[wp.spatial_vector],
  Dcdof_dot_in: wp.array3d[wp.spatial_vector],
  # Out:
  Dcacc_out: wp.array3d[wp.spatial_vector],
  Dcfrcbody_out: wp.array3d[wp.spatial_vector],
):
  """Forward pass: compute d(cacc)/d(qvel_k) and d(cfrc_body)/d(qvel_k)."""
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  dofadr = body_dofadr[bodyid]
  dofnum = body_dofnum[bodyid]
  pid = body_parentid[bodyid]

  qvel = qvel_in[worldid]

  dcacc = Dcacc_out[worldid, pid, dofid]

  for j in range(dofadr, dofadr + dofnum):
    # Term 1: d(cdof_dot * qvel)/d(qvel_k) when j == dofid
    if j == dofid:
      dcacc += cdof_dot_in[worldid, j]

    # Term 2: cdof_dot depends on cvel which depends on qvel_k
    dcdofdot = Dcdof_dot_in[worldid, j, dofid]
    dcacc += dcdofdot * qvel[j]

  Dcacc_out[worldid, bodyid, dofid] = dcacc

  # d(cfrc_body)/d(qvel_k)
  cinert = cinert_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  dcvel = Dcvel_in[worldid, bodyid, dofid]

  # term1 = cinert * d(cacc)/d(qvel_k)
  term1 = math.inert_vec(cinert, dcacc)

  # term2 = d(cvel x* (cinert * cvel))/d(qvel_k)
  cinert_cvel = math.inert_vec(cinert, cvel)
  cinert_dcvel = math.inert_vec(cinert, dcvel)
  term2 = math.motion_cross_force(dcvel, cinert_cvel) + math.motion_cross_force(cvel, cinert_dcvel)

  Dcfrcbody_out[worldid, bodyid, dofid] = term1 + term2


@cache_kernel
def deriv_rne_cfrcbody_backward(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    # In:
    body_tree_: wp.array[int],
    Dcfrcbody_in: wp.array3d[wp.spatial_vector],
    # Out:
    Dcfrcbody_out: wp.array3d[wp.spatial_vector],
  ):
    """Backward pass: accumulate d(cfrc_body) from children to parents."""
    worldid, nodeid, dofid = wp.tid()
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]

    # body_tree never contains bodyid=0 (worldbody), so pid >= 0 is always valid.
    # Siblings at the same level may share a parent; atomic_add handles this.
    val = Dcfrcbody_in[worldid, bodyid, dofid]
    wp.atomic_add(Dcfrcbody_out[worldid, pid], dofid, val)

  return kernel


@cache_kernel
def deriv_rne_body2jnt_sparse(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    dof_bodyid: wp.array[int],
    # Data in:
    cdof_in: wp.array2d[wp.spatial_vector],
    # In:
    timestep: wp.array[float],
    Di: wp.array[int],
    Dj: wp.array[int],
    Dcfrcbody_in: wp.array3d[wp.spatial_vector],
    # Out:
    qDeriv_out: wp.array2d[float],
  ):
    """Project body-space RNE derivatives into joint-space qDeriv (sparse)."""
    worldid, elemid = wp.tid()
    dt = timestep[worldid % timestep.shape[0]]

    i = Di[elemid]
    j = Dj[elemid]

    body_i = dof_bodyid[i]
    dcfrc = Dcfrcbody_in[worldid, body_i, j]
    term = wp.dot(cdof_in[worldid, i], dcfrc)

    wp.atomic_add(qDeriv_out[worldid], elemid, dt * term)

  return kernel


def deriv_rne_vel(m: Model, d: Data, out: wp.array2d[float]):
  """Compute RNE velocity derivatives and add to the output.

  Implements the analytical derivative of inverse-dynamics Coriolis/centrifugal
  forces with respect to joint velocities.

  Args:
    m: The model (device).
    d: The data (device).
    out: D-structure output array (nworld, nD) to accumulate RNE terms into.
  """
  # TODO(team): consider caching these allocations
  Dcvel = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcdof_dot = wp.zeros((d.nworld, m.nv, m.nv), dtype=wp.spatial_vector)
  Dcacc = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcfrcbody = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)

  # Forward pass 1: compute Dcvel and Dcdof_dot
  for body_tree in m.body_tree:
    wp.launch(
      deriv_rne_cvel_cdof_dot,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[
        m.body_parentid,
        m.body_jntnum,
        m.body_jntadr,
        m.body_dofadr,
        m.jnt_type,
        d.cdof,
        body_tree,
      ],
      outputs=[Dcvel, Dcdof_dot],
    )

  # Forward pass 2: compute Dcacc and Dcfrcbody
  for body_tree in m.body_tree:
    wp.launch(
      deriv_rne_cacc_cfrcbody_forward,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[
        m.body_parentid,
        m.body_dofnum,
        m.body_dofadr,
        d.qvel,
        d.cinert,
        d.cvel,
        d.cdof_dot,
        body_tree,
        Dcvel,
        Dcdof_dot,
      ],
      outputs=[Dcacc, Dcfrcbody],
    )

  # Backward pass: accumulate Dcfrcbody from children to parents
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  Dcfrcbody_in = wp.empty_like(Dcfrcbody) if det_atomics else Dcfrcbody
  for body_tree in reversed(m.body_tree):
    if det_atomics:
      wp.copy(Dcfrcbody_in, Dcfrcbody)
    wp.launch(
      deriv_rne_cfrcbody_backward(det_atomics),
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[m.body_parentid, body_tree, Dcfrcbody_in],
      outputs=[Dcfrcbody],
    )

  # Project body-space derivatives into joint-space qDeriv (always sparse D-structure)
  wp.launch(
    deriv_rne_body2jnt_sparse(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.qD_fullm_i.size),
    inputs=[m.dof_bodyid, d.cdof, m.opt.timestep, m.qD_fullm_i, m.qD_fullm_j, Dcfrcbody],
    outputs=[out],
  )


@wp.func
def _geom_ellipsoid_fluid_B(
  # In:
  semiaxes: wp.vec3,
  blunt_drag_coef: float,
  slender_drag_coef: float,
  ang_drag_coef: float,
  kutta_lift_coef: float,
  magnus_lift_coef: float,
  virtual_mass: wp.vec3,
  virtual_inertia: wp.vec3,
  ang_vel: wp.vec3,
  lin_vel: wp.vec3,
  density: float,
  viscosity: float,
  flg_dragonly: bool,
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
  # B = [[B00, B01], [B10, B11]] where rows are [ang; lin], cols are [ang; lin]
  B00 = wp.mat33(0.0)  # torque wrt ang_vel
  B01 = wp.mat33(0.0)  # torque wrt lin_vel
  B10 = wp.mat33(0.0)  # force wrt ang_vel
  B11 = wp.mat33(0.0)  # force wrt lin_vel

  d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])
  d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])
  d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min
  inv_dmax = wp.where(d_max > MJ_MINVAL, 1.0 / d_max, 0.0)
  s0 = semiaxes[0] * inv_dmax
  s1 = semiaxes[1] * inv_dmax
  s2 = semiaxes[2] * inv_dmax

  norm = wp.length(lin_vel)
  inv_norm = wp.where(norm > MJ_MINVAL, 1.0 / norm, 0.0)
  v = lin_vel * inv_norm
  x = v[0]
  y = v[1]
  z = v[2]
  xx = x * x
  yy = y * y
  zz = z * z

  s12_sq = (s1 * s2) * (s1 * s2)
  s20_sq = (s2 * s0) * (s2 * s0)
  s01_sq = (s0 * s1) * (s0 * s1)

  proj_num = s12_sq * xx + s20_sq * yy + s01_sq * zz
  inv_proj_num = wp.where(proj_num > MJ_MINVAL, 1.0 / proj_num, 0.0)
  a = s12_sq * inv_proj_num
  b = s20_sq * inv_proj_num
  c = s01_sq * inv_proj_num
  aa = a * a
  bb = b * b
  cc = c * c

  proj_denom = aa * xx + bb * yy + cc * zz
  area_scale = d_max * d_max * wp.sqrt(proj_num)

  if not flg_dragonly:
    skew_lin_vel = wp.skew(lin_vel)
    skew_ang_vel = wp.skew(ang_vel)
    if density > 0.0:
      # --- added mass forces ---
      density_vm = density * virtual_mass
      density_vi = density * virtual_inertia
      virtual_lin_mom = wp.cw_mul(density_vm, lin_vel)
      virtual_ang_mom = wp.cw_mul(density_vi, ang_vel)
      skew_virtual_lin_mom = wp.skew(virtual_lin_mom)

      # torque += cross(virtual_ang_mom, ang_vel) -> B00
      B00 += wp.skew(virtual_ang_mom) - skew_ang_vel @ wp.diag(density_vi)

      # torque += cross(virtual_lin_mom, lin_vel) -> B01
      B01 += skew_virtual_lin_mom - skew_lin_vel @ wp.diag(density_vm)

      # force += cross(virtual_lin_mom, ang_vel) -> B10
      B10 += skew_virtual_lin_mom
      # Da = d/d(vlm via lin) = _skew_neg(ang), scaled by density*vm -> B11
      B11 += -skew_ang_vel @ wp.diag(density_vm)

    # --- Magnus force: force += magnus_coef * cross(ang_vel, lin_vel) ---
    volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]
    magnus_coef = magnus_lift_coef * density * volume
    B10 -= skew_lin_vel * magnus_coef
    B11 += skew_ang_vel * magnus_coef

    # --- Kutta lift (3x3 -> B11) ---
    df_denom = wp.where(
      proj_denom > MJ_MINVAL,
      wp.pi * kutta_lift_coef * density * area_scale * norm / wp.sqrt(proj_denom),
      0.0,
    )

    dfx_coef = yy * (a - b) + zz * (a - c)
    dfy_coef = xx * (b - a) + zz * (b - c)
    dfz_coef = xx * (c - a) + yy * (c - b)
    proj_term = wp.where(proj_denom > MJ_MINVAL, 1.0 / proj_denom, 0.0)
    cos_term = 1.0

    D = wp.skew(wp.vec3(b - c, c - a, a - b)) * 2.0

    df_coef = wp.vec3(dfx_coef, dfy_coef, dfz_coef)
    inner_term = wp.vec3(
      aa * proj_term - a + cos_term,
      bb * proj_term - b + cos_term,
      cc * proj_term - c + cos_term,
    )

    D += wp.outer(df_coef, inner_term)

    V = wp.diag(v)
    D = V @ D @ V - wp.diag(df_coef)

    D *= df_denom
    B11 += D

  # --- viscous drag (3x3 -> B11) ---
  eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])
  A_max = wp.pi * d_max * d_mid

  A_proj = wp.pi * area_scale * wp.sqrt(proj_denom)

  lin_coef = viscosity * wp.static(3.0 * wp.pi) * eq_sphere_D
  quad_coef = density * (A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj))
  Aproj_coef = density * norm * (blunt_drag_coef - slender_drag_coef)
  dA_coef = wp.where(proj_denom > MJ_MINVAL, wp.pi * area_scale / wp.sqrt(proj_denom), 0.0)

  dAproj_dv = wp.vec3(
    Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),
    Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),
    Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b)),
  )

  inner = xx + yy + zz
  D = (wp.outer(v, v) + wp.diag(wp.vec3(inner))) * (-quad_coef * norm)
  D -= wp.outer(v, dAproj_dv)
  D -= wp.diag(wp.vec3(lin_coef))

  B11 += D

  # --- viscous torque (3x3 -> B00) ---
  lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D
  I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * d_max * d_max * d_max * d_max
  II = wp.vec3(
    ellipsoid_max_moment(semiaxes, 0),
    ellipsoid_max_moment(semiaxes, 1),
    ellipsoid_max_moment(semiaxes, 2),
  )

  mom_coef = wp.vec3(
    ang_drag_coef * II[0] + slender_drag_coef * (I_max - II[0]),
    ang_drag_coef * II[1] + slender_drag_coef * (I_max - II[1]),
    ang_drag_coef * II[2] + slender_drag_coef * (I_max - II[2]),
  )

  mom_visc = wp.cw_mul(ang_vel, mom_coef)
  norm_mom = wp.length(mom_visc)
  density_scaled = math.safe_div(density, norm_mom)

  mom_sq = -density_scaled * wp.cw_mul(wp.cw_mul(ang_vel, mom_coef), mom_coef)

  torq_lin_coef = viscosity * lin_visc_torq_coef
  diag_val = wp.dot(ang_vel, mom_sq) - torq_lin_coef

  D = wp.outer(ang_vel, mom_sq) + wp.diag(wp.vec3(diag_val))
  B00 += D

  return B00, B01, B10, B11


@wp.func
def _deriv_ellipsoid_fluid(
  # Model:
  opt_integrator: int,
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  bodyid: int,
  rootid: int,
  geomadr: int,
  geomnum: int,
  cdof_i: wp.spatial_vector,
  cdof_j: wp.spatial_vector,
  wind: wp.vec3,
  density: float,
  viscosity: float,
) -> float:
  """Compute one body's ellipsoid fluid derivative contribution for a DOF pair.

  Returns the scalar J_i^T @ B @ J_j contribution accumulated across geoms.
  """
  flg_dragonly = opt_integrator == IntegratorType.DISCRETE
  is_symmetrize = flg_dragonly or opt_integrator == IntegratorType.IMPLICITFAST

  # Body kinematics
  xipos = xipos_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  ang_global = wp.spatial_top(cvel)
  lin_global = wp.spatial_bottom(cvel)
  subtree_root = subtree_com_in[worldid, rootid]
  lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

  qderiv_contrib = float(0.0)

  cdof_ang_i = wp.vec3(cdof_i[0], cdof_i[1], cdof_i[2])
  cdof_lin_i = wp.vec3(cdof_i[3], cdof_i[4], cdof_i[5])
  cdof_ang_j = wp.vec3(cdof_j[0], cdof_j[1], cdof_j[2])
  cdof_lin_j = wp.vec3(cdof_j[3], cdof_j[4], cdof_j[5])

  for g in range(geomnum):
    geomid = geomadr + g
    coef = geom_fluid[geomid, 0]
    if coef <= 0.0:
      continue

    size = geom_size[worldid % geom_size.shape[0], geomid]
    semiaxes = geom_semiaxes(size, geom_type[geomid])
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_rotT = wp.transpose(geom_rot)
    geom_pos = geom_xpos_in[worldid, geomid]

    # compute local velocity
    lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)
    l_ang = geom_rotT @ ang_global
    l_lin = geom_rotT @ lin_point

    if wind[0] != 0.0 or wind[1] != 0.0 or wind[2] != 0.0:
      l_lin -= geom_rotT @ wind

    ang_vel = l_ang
    lin_vel = l_lin

    # read fluid coefficients
    blunt_drag_coef = geom_fluid[geomid, 1]
    slender_drag_coef = geom_fluid[geomid, 2]
    ang_drag_coef = geom_fluid[geomid, 3]
    kutta_lift_coef = geom_fluid[geomid, 4]
    magnus_lift_coef = geom_fluid[geomid, 5]
    virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
    virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])

    B00, B01, B10, B11 = _geom_ellipsoid_fluid_B(
      semiaxes,
      blunt_drag_coef,
      slender_drag_coef,
      ang_drag_coef,
      kutta_lift_coef,
      magnus_lift_coef,
      virtual_mass,
      virtual_inertia,
      ang_vel,
      lin_vel,
      density,
      viscosity,
      flg_dragonly=flg_dragonly,
    )

    # symmetrize for implicitfast or discrete
    if is_symmetrize:
      B00 = 0.5 * (B00 + wp.transpose(B00))
      B11 = 0.5 * (B11 + wp.transpose(B11))
      B01_sym = 0.5 * (B01 + wp.transpose(B10))
      B01 = B01_sym
      B10 = wp.transpose(B01_sym)

    # --- Jacobian transformation: J_i^T @ B @ J_j ---
    offset = geom_pos - subtree_root

    jac_p_i = cdof_lin_i + wp.cross(cdof_ang_i, offset)
    la_i = geom_rotT @ cdof_ang_i
    ll_i = geom_rotT @ jac_p_i

    jac_p_j = cdof_lin_j + wp.cross(cdof_ang_j, offset)
    la_j = geom_rotT @ cdof_ang_j
    ll_j = geom_rotT @ jac_p_j

    # B @ J_j = [B00 @ la_j + B01 @ ll_j; B10 @ la_j + B11 @ ll_j]
    Bj_ang = B00 @ la_j + B01 @ ll_j
    Bj_lin = B10 @ la_j + B11 @ ll_j

    # J_i^T @ (B @ J_j) = la_i . Bj_ang + ll_i . Bj_lin
    qderiv_contrib += wp.dot(la_i, Bj_ang) + wp.dot(ll_i, Bj_lin)

  return qderiv_contrib


@cache_kernel
def _qderiv_ellipsoid_fluid(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_wind: wp.array[wp.vec3],
    opt_density: wp.array[float],
    opt_viscosity: wp.array[float],
    opt_integrator: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_geomnum: wp.array[int],
    body_geomadr: wp.array[int],
    dof_bodyid: wp.array[int],
    geom_type: wp.array[int],
    geom_size: wp.array2d[wp.vec3],
    geom_fluid: wp.array2d[float],
    body_fluid_ellipsoid_adr: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    M_elemid: wp.array2d[int],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    cvel_in: wp.array2d[wp.spatial_vector],
    # In:
    Mi: wp.array[int],
    Mj: wp.array[int],
    # Out:
    qDeriv_out: wp.array2d[float],
  ):
    """Compute ellipsoid fluid force derivative contribution to qDeriv.

    Parallelized over (world, fluid_body, elem). For each fluid body and DOF
    pair, computes the 6x6 derivative matrix B in local geom frame via
    _deriv_ellipsoid_fluid and accumulates J_i^T @ B @ J_j into qDeriv.
    """
    worldid, fluid_idx, elemid = wp.tid()

    bodyid = body_fluid_ellipsoid_adr[fluid_idx]

    dofiid = Mi[elemid]
    dofjid = Mj[elemid]

    madr = M_elemid[dofiid, dofjid]
    if madr < 0:
      return

    # dofiid is the "deeper" DOF (Mi >= Mj in tree ordering).
    # Any body that has dofiid in its chain also has dofjid.
    bodyid_i = dof_bodyid[dofiid]

    if bodyid_i == 0:
      return

    if body_isdofancestor[bodyid, dofiid] == 0:
      return

    wind = opt_wind[worldid % opt_wind.shape[0]]
    density = opt_density[worldid % opt_density.shape[0]]
    viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    if density <= 0.0 and viscosity <= 0.0:
      return

    cdof_i = cdof_in[worldid, dofiid]
    cdof_j = cdof_in[worldid, dofjid]

    contrib = _deriv_ellipsoid_fluid(
      opt_integrator,
      geom_type,
      geom_size,
      geom_fluid,
      xipos_in,
      geom_xpos_in,
      geom_xmat_in,
      subtree_com_in,
      cvel_in,
      worldid,
      bodyid,
      body_rootid[bodyid],
      body_geomadr[bodyid],
      body_geomnum[bodyid],
      cdof_i,
      cdof_j,
      wind,
      density,
      viscosity,
    )

    contrib *= timestep

    if contrib != 0.0:
      wp.atomic_add(qDeriv_out[worldid], madr, -contrib)

  return kernel


@wp.func
def _deriv_box_fluid(
  # Model:
  opt_integrator: int,
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  # In:
  worldid: int,
  bodyid: int,
  lvel: wp.spatial_vector,
  density: float,
  viscosity: float,
) -> wp.spatial_matrix:
  B = wp.spatial_matrix(0.0)

  mass = body_mass[worldid % body_mass.shape[0], bodyid]
  inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]
  scl = 6.0 / mass

  # Equivalent inertia box
  box = wp.vec3(
    wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl),
    wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl),
    wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl),
  )

  # Viscous force and torque
  if viscosity > 0.0:
    diam = (box[0] + box[1] + box[2]) * wp.static(1.0 / 3.0)

    # Rotational viscosity
    visc_rot = -wp.pi * diam * diam * diam * viscosity
    B[0, 0] += visc_rot
    B[1, 1] += visc_rot
    B[2, 2] += visc_rot

    # Translational viscosity
    visc_lin = wp.static(-3.0 * wp.pi) * diam * viscosity
    B[3, 3] += visc_lin
    B[4, 4] += visc_lin
    B[5, 5] += visc_lin

  # Lift and drag force and torque
  if density > 0.0:
    term0 = box[1] * box[1] * box[1] * box[1] + box[2] * box[2] * box[2] * box[2]
    term1 = box[0] * box[0] * box[0] * box[0] + box[2] * box[2] * box[2] * box[2]
    term2 = box[0] * box[0] * box[0] * box[0] + box[1] * box[1] * box[1] * box[1]

    inv_32 = wp.static(1.0 / 32.0)
    B[0, 0] -= density * box[0] * term0 * wp.abs(lvel[0]) * inv_32
    B[1, 1] -= density * box[1] * term1 * wp.abs(lvel[1]) * inv_32
    B[2, 2] -= density * box[2] * term2 * wp.abs(lvel[2]) * inv_32

    B[3, 3] -= density * box[1] * box[2] * wp.abs(lvel[3])
    B[4, 4] -= density * box[0] * box[2] * wp.abs(lvel[4])
    B[5, 5] -= density * box[0] * box[1] * wp.abs(lvel[5])

  if opt_integrator == IntegratorType.IMPLICITFAST or opt_integrator == IntegratorType.DISCRETE:
    B = 0.5 * (B + wp.transpose(B))

  return B


@wp.func
def _get_jac_column_local(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  point_global: wp.vec3,
  bodyid: int,
  dofid: int,
  worldid: int,
  b_imat: wp.mat33,
) -> wp.spatial_vector:
  offset = point_global - subtree_com_in[worldid, body_rootid[bodyid]]
  cdof_val = cdof_in[worldid, dofid]
  cdof_ang = wp.spatial_top(cdof_val)
  cdof_lin = wp.spatial_bottom(cdof_val)

  jacp = cdof_lin + wp.cross(cdof_ang, offset)
  jacr = cdof_ang

  b_imat_T = wp.transpose(b_imat)
  jacp_loc = b_imat_T @ jacp
  jacr_loc = b_imat_T @ jacr
  return wp.spatial_vector(jacr_loc, jacp_loc)


@cache_kernel
def _qderiv_box_fluid(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_wind: wp.array[wp.vec3],
    opt_density: wp.array[float],
    opt_viscosity: wp.array[float],
    opt_integrator: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mass: wp.array2d[float],
    body_inertia: wp.array2d[wp.vec3],
    dof_bodyid: wp.array[int],
    body_fluid_box_adr: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    M_elemid: wp.array2d[int],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    ximat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    cvel_in: wp.array2d[wp.spatial_vector],
    # In:
    Mi: wp.array[int],
    Mj: wp.array[int],
    # Out:
    qDeriv_out: wp.array2d[float],
  ):
    worldid, fluid_idx, elemid = wp.tid()

    bodyid = body_fluid_box_adr[fluid_idx]

    dofiid = Mi[elemid]
    dofjid = Mj[elemid]

    madr = M_elemid[dofiid, dofjid]
    if madr < 0:
      return

    bodyid_i = dof_bodyid[dofiid]

    if bodyid_i == 0:
      return

    if body_isdofancestor[bodyid, dofiid] == 0:
      return

    wind = opt_wind[worldid % opt_wind.shape[0]]
    density = opt_density[worldid % opt_density.shape[0]]
    viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    if density <= 0.0 and viscosity <= 0.0:
      return

    # Body velocity and kinematics
    b_ipos = xipos_in[worldid, bodyid]
    b_imat = ximat_in[worldid, bodyid]
    subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]

    vel_subtree = cvel_in[worldid, bodyid]
    v_subtree_ang = wp.vec3(vel_subtree[0], vel_subtree[1], vel_subtree[2])
    v_subtree_lin = wp.vec3(vel_subtree[3], vel_subtree[4], vel_subtree[5])

    lin_com = v_subtree_lin - wp.cross(b_ipos - subtree_root, v_subtree_ang)
    b_imat_T = wp.transpose(b_imat)
    v_local_ang = b_imat_T @ v_subtree_ang
    v_local_lin = b_imat_T @ lin_com
    wind_local = b_imat_T @ wind

    lvel = wp.spatial_vector(v_local_ang, v_local_lin - wind_local)

    B = _deriv_box_fluid(
      opt_integrator,
      body_mass,
      body_inertia,
      worldid,
      bodyid,
      lvel,
      density,
      viscosity,
    )

    # Jacobian transformation: J_i^T @ B @ J_j
    J_i = _get_jac_column_local(
      body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofiid, worldid, b_imat
    )
    J_j = _get_jac_column_local(
      body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofjid, worldid, b_imat
    )

    contrib = wp.dot(J_i, B @ J_j) * timestep

    if contrib != 0.0:
      wp.atomic_add(qDeriv_out[worldid], madr, -contrib)

  return kernel


@event_scope
def deriv_smooth_vel(m: Model, d: Data, out: wp.array2d[float]):
  """Analytical derivative of smooth forces w.r.t. velocities.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    out: M - dt * qDeriv (derivatives of smooth forces w.r.t velocities).
  """
  Mi = m.M_fullm_i
  Mj = m.M_fullm_j

  if not (m.opt.disableflags & (DisableBit.ACTUATION | DisableBit.DAMPER)):
    # TODO(team): only clear elements not set by _qderiv_actuator_passive
    out.zero_()
    if m.nactuator > 0 and not (m.opt.disableflags & DisableBit.ACTUATION):
      if m.nhistory > 0:
        ctrl = wp.empty((d.nworld, m.nu), dtype=float)
        history.read_ctrl_delayed(m, d, ctrl)
      else:
        ctrl = d.ctrl
      vel = wp.empty((d.nworld, m.nactuator), dtype=float)
      wp.launch(
        _qderiv_actuator_passive_vel,
        dim=(d.nworld, m.nactuator),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.jnt_dofadr,
          m.jnt_actfrclimited,
          m.jnt_actfrcrange,
          m.tendon_actfrclimited,
          m.tendon_actfrcrange,
          m.actuator_trntype,
          m.actuator_dyntype,
          m.actuator_gaintype,
          m.actuator_biastype,
          m.actuator_ctrladr,
          m.actuator_ctrlnum,
          m.actuator_ctrlspec,
          m.actuator_actadr,
          m.actuator_actnum,
          m.actuator_trnid,
          m.actuator_dynprm,
          m.actuator_gainprm,
          m.actuator_biasprm,
          m.actuator_actlimited,
          m.actuator_actrange,
          m.actuator_actearly,
          m.actuator_forcelimited,
          m.actuator_forcerange,
          m.actuator_ctrllimited,
          m.actuator_ctrlrange,
          m.actuator_acc0,
          m.actuator_lengthrange,
          d.act,
          ctrl,
          d.act_dot,
          d.actuator_length,
          d.actuator_velocity,
          d.actuator_force,
          d.qfrc_actuator,
        ],
        outputs=[vel],
      )
      # out (qDeriv) is in M-structure.
      det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
      wp.launch(
        _qderiv_actuator_passive_actuation_sparse(m.nM if det_atomics else 0, det_atomics),
        dim=(d.nworld, m.nactuator),
        inputs=[
          m.M_elemid,
          d.moment_rownnz,
          d.moment_rowadr,
          d.moment_colind,
          d.actuator_moment,
          vel,
        ],
        outputs=[out],
      )
    wp.launch(
      _qderiv_actuator_passive,
      dim=(d.nworld, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.jnt_actuatorid,
        m.dof_jntid,
        m.dof_damping,
        m.dof_dampingpoly,
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_damping,
        m.actuator_dampingpoly,
        m.actuator_gear,
        m.M_elemid,
        d.qvel,
        d.M,
        Mi,
        Mj,
        out,
      ],
      outputs=[out],
    )
  else:
    # TODO(team): directly utilize M for these settings
    wp.copy(out, d.M)

  if not (m.opt.disableflags & DisableBit.DAMPER):
    # flex_edgedamping is omitted: tree-only M sparsity drops inter-tree flexcomp blocks
    wp.launch(
      _qderiv_tendon_damping,
      dim=(d.nworld, Mi.size),
      inputs=[
        m.ntendon,
        m.opt.timestep,
        m.tendon_actuatorid,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_damping,
        m.actuator_dampingpoly,
        m.actuator_gear,
        m.M_elemid,
        d.ten_J,
        d.ten_velocity,
        Mi,
        Mj,
      ],
      outputs=[out],
    )
  if m.has_fluid:
    wp.launch(
      _qderiv_ellipsoid_fluid(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, m.body_fluid_ellipsoid_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_geomnum,
        m.body_geomadr,
        m.dof_bodyid,
        m.geom_type,
        m.geom_size,
        m.geom_fluid,
        m.body_fluid_ellipsoid_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.geom_xpos,
        d.geom_xmat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[out],
    )
    wp.launch(
      _qderiv_box_fluid(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, m.body_fluid_box_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_inertia,
        m.dof_bodyid,
        m.body_fluid_box_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.ximat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[out],
    )


@wp.kernel
def _eff_diag_stiff(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  qpos_spring: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_plasticadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  # Data in:
  qpos_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  # Data out:
  efm_diag_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  dofid = jnt_dofadr[jntid]
  jnttype = jnt_type[jntid]
  h = opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_disableflags & DisableBit.SPRING:
    return

  stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]
  spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]
  if stiffness == 0.0 and spoly[0] == 0.0 and spoly[1] == 0.0:
    return

  qposid = jnt_qposadr[jntid]
  qpos_spring_id = worldid % qpos_spring.shape[0]
  padr = jnt_plasticadr[jntid]

  if jnttype == JointType.FREE:
    if padr >= 0:
      ref_pos = wp.vec3(plastic_in[worldid, padr + 0], plastic_in[worldid, padr + 1], plastic_in[worldid, padr + 2])
      ref = wp.quat(
        plastic_in[worldid, padr + 3],
        plastic_in[worldid, padr + 4],
        plastic_in[worldid, padr + 5],
        plastic_in[worldid, padr + 6],
      )
    else:
      ref_pos = wp.vec3(
        qpos_spring[qpos_spring_id, qposid + 0],
        qpos_spring[qpos_spring_id, qposid + 1],
        qpos_spring[qpos_spring_id, qposid + 2],
      )
      ref = wp.quat(
        qpos_spring[qpos_spring_id, qposid + 3],
        qpos_spring[qpos_spring_id, qposid + 4],
        qpos_spring[qpos_spring_id, qposid + 5],
        qpos_spring[qpos_spring_id, qposid + 6],
      )
    dif = wp.vec3(
      qpos_in[worldid, qposid + 0] - ref_pos[0],
      qpos_in[worldid, qposid + 1] - ref_pos[1],
      qpos_in[worldid, qposid + 2] - ref_pos[2],
    )
    r = wp.length(dif)
    k = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r, 0))
    for i in range(3):
      efm_diag_out[worldid, dofid + i] = h * k

    rot = wp.normalize(
      wp.quat(
        qpos_in[worldid, qposid + 3],
        qpos_in[worldid, qposid + 4],
        qpos_in[worldid, qposid + 5],
        qpos_in[worldid, qposid + 6],
      )
    )
    dif_rot = math.quat_sub(rot, ref)
    r_rot = wp.length(dif_rot)
    k_rot = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r_rot, 0))
    for i in range(3):
      efm_diag_out[worldid, dofid + 3 + i] = h * k_rot

  elif jnttype == JointType.BALL:
    rot = wp.normalize(
      wp.quat(
        qpos_in[worldid, qposid + 0],
        qpos_in[worldid, qposid + 1],
        qpos_in[worldid, qposid + 2],
        qpos_in[worldid, qposid + 3],
      )
    )
    if padr >= 0:
      ref = wp.quat(
        plastic_in[worldid, padr + 0],
        plastic_in[worldid, padr + 1],
        plastic_in[worldid, padr + 2],
        plastic_in[worldid, padr + 3],
      )
    else:
      ref = wp.quat(
        qpos_spring[qpos_spring_id, qposid + 0],
        qpos_spring[qpos_spring_id, qposid + 1],
        qpos_spring[qpos_spring_id, qposid + 2],
        qpos_spring[qpos_spring_id, qposid + 3],
      )
    dif_rot = math.quat_sub(rot, ref)
    r_rot = wp.length(dif_rot)
    k_rot = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r_rot, 0))
    for i in range(3):
      efm_diag_out[worldid, dofid + i] = h * k_rot

  else:
    ref_scalar = plastic_in[worldid, padr] if padr >= 0 else qpos_spring[qpos_spring_id, qposid]
    x = qpos_in[worldid, qposid] - ref_scalar
    k = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, x, 0))
    efm_diag_out[worldid, dofid] = h * k


@wp.kernel
def _eff_rod_ktheta_stiff(
  # Model:
  opt_timestep: wp.array[float],
  jnt_dofadr: wp.array[int],
  flex_rigid: wp.array[bool],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # Data in:
  flex_rod_chain_ok_in: wp.array2d[bool],
  flex_rodedge_ktheta_in: wp.array2d[float],
  # Data out:
  efm_diag_out: wp.array2d[float],
):
  """Adds 1D DER rod twist-angle bending stiffness h*ktheta to efm_diag."""
  worldid, flexid = wp.tid()
  if flex_rigid[flexid] or not flex_rod_has_twist[flexid] or not flex_rod_chain_ok_in[worldid, flexid]:
    return
  h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
  eadr = flex_rod_edgeadr[flexid]
  nedge = flex_rod_edgenum[flexid]
  for e in range(nedge):
    jnt = flex_rodedge_jnt[eadr + e]
    if jnt >= 0:
      efm_diag_out[worldid, jnt_dofadr[jnt]] += h_dt * flex_rodedge_ktheta_in[worldid, eadr + e]


@wp.kernel
def _eff_rod_twist_shift(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_dofadr: wp.array[int],
  flex_bendingadr: wp.array[int],
  flex_bending: wp.array[float],
  flex_rigid: wp.array[bool],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  # Data out:
  efm_c_out: wp.array2d[float],
):
  """Applies 1D DER rod torsional coupling velocity shift -h*2*kt*(qvel[dof1] - qvel[dof0])."""
  worldid, flexid = wp.tid()
  if (opt_disableflags & DisableBit.SPRING) != 0:
    return
  if flex_rigid[flexid] or not support.rod_has_twist(flex_rod_has_twist, flexid):
    return
  h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
  b_adr = flex_bendingadr[flexid]
  nhinge = int(flex_bending[b_adr])
  eadr = flex_rod_edgeadr[flexid]
  for hi in range(nhinge):
    kt = flex_bending[b_adr + 1 + ROD_HINGE_SIZE * hi + 8]
    if kt <= 0.0:
      continue
    jnt0 = flex_rodedge_jnt[eadr + hi]
    jnt1 = flex_rodedge_jnt[eadr + hi + 1]
    dof0 = jnt_dofadr[jnt0] if jnt0 >= 0 else -1
    dof1 = jnt_dofadr[jnt1] if jnt1 >= 0 else -1
    if dof0 < 0 and dof1 < 0:
      continue
    w0 = qvel_in[worldid, dof0] if dof0 >= 0 else 0.0
    w1 = qvel_in[worldid, dof1] if dof1 >= 0 else 0.0
    ckv = h_dt * 2.0 * kt * (w1 - w0)
    if dof1 >= 0:
      efm_c_out[worldid, dof1] -= ckv
    if dof0 >= 0:
      efm_c_out[worldid, dof0] += ckv


@wp.kernel
def _eff_add_rod_twist_qH_diag(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_dofadr: wp.array[int],
  flex_bendingadr: wp.array[int],
  flex_bending: wp.array[float],
  flex_damping: wp.array[float],
  flex_rigid: wp.array[bool],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # Data out:
  qH_out: wp.array2d[float],
):
  """Adds 1D DER rod torsional coupling diagonals to qH."""
  worldid, flexid = wp.tid()
  if flex_rigid[flexid] or not support.rod_has_twist(flex_rod_has_twist, flexid):
    return
  h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
  s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else h_dt * h_dt
  s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else h_dt
  scale_f = s1 + s2 * flex_damping[flexid]
  b_adr = flex_bendingadr[flexid]
  nhinge = int(flex_bending[b_adr])
  if scale_f <= 0.0 or nhinge <= 0:
    return
  eadr = flex_rod_edgeadr[flexid]
  for hi in range(nhinge):
    kt = flex_bending[b_adr + 1 + ROD_HINGE_SIZE * hi + 8]
    if kt <= 0.0:
      continue
    jnt0 = flex_rodedge_jnt[eadr + hi]
    jnt1 = flex_rodedge_jnt[eadr + hi + 1]
    dof0 = jnt_dofadr[jnt0] if jnt0 >= 0 else -1
    dof1 = jnt_dofadr[jnt1] if jnt1 >= 0 else -1
    if dof0 < 0 and dof1 < 0:
      continue
    scale = scale_f * 2.0 * kt
    if dof0 >= 0:
      qH_out[worldid, M_rowadr[dof0] + M_rownnz[dof0] - 1] += scale
    if dof1 >= 0:
      qH_out[worldid, M_rowadr[dof1] + M_rownnz[dof1] - 1] += scale


@cache_kernel
def _eff_mul_add_rod_twist(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    jnt_dofadr: wp.array[int],
    flex_bendingadr: wp.array[int],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_jnt: wp.array[int],
    # In:
    vec_in: wp.array2d[float],
    skip_in: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, flexid = wp.tid()
    if wp.static(check_skip):
      if skip_in[worldid]:
        return
    if flex_rigid[flexid] or not support.rod_has_twist(flex_rod_has_twist, flexid):
      return
    h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
    s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else h_dt * h_dt
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else h_dt
    scale_f = s1 + s2 * flex_damping[flexid]
    b_adr = flex_bendingadr[flexid]
    nhinge = int(flex_bending[b_adr])
    if scale_f <= 0.0 or nhinge <= 0:
      return
    eadr = flex_rod_edgeadr[flexid]
    for hi in range(nhinge):
      kt = flex_bending[b_adr + 1 + ROD_HINGE_SIZE * hi + 8]
      if kt <= 0.0:
        continue
      jnt0 = flex_rodedge_jnt[eadr + hi]
      jnt1 = flex_rodedge_jnt[eadr + hi + 1]
      dof0 = jnt_dofadr[jnt0] if jnt0 >= 0 else -1
      dof1 = jnt_dofadr[jnt1] if jnt1 >= 0 else -1
      if dof0 < 0 and dof1 < 0:
        continue
      v0 = vec_in[worldid, dof0] if dof0 >= 0 else 0.0
      v1 = vec_in[worldid, dof1] if dof1 >= 0 else 0.0
      dv = scale_f * 2.0 * kt * (v1 - v0)
      if dof1 >= 0:
        res_out[worldid, dof1] += dv
      if dof0 >= 0:
        res_out[worldid, dof0] -= dv

  return kernel


@cache_kernel
def _eff_twist_solve(has_tw_diag: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    jnt_dofadr: wp.array[int],
    dof_treeid: wp.array[int],
    tree_dofnum: wp.array[int],
    flex_bendingadr: wp.array[int],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    flex_rod_has_twist: wp.array[bool],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_jnt: wp.array[int],
    # Data in:
    qH_in: wp.array2d[float],
    # In:
    rhs_in: wp.array2d[float],
    tw_diag_in: wp.array2d[float],
    # Data out:
    flex_rodedge_tridiag_out: wp.array2d[wp.vec2],
    # Out:
    res_out: wp.array2d[float],
  ):
    """Exact O(nedge) symmetric tridiagonal LDL^T solve on 1D DER rod twist chains."""
    worldid, flexid = wp.tid()
    if flex_rigid[flexid] or not support.rod_has_twist(flex_rod_has_twist, flexid):
      return
    h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
    s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else h_dt * h_dt
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else h_dt
    scale_f = s1 + s2 * flex_damping[flexid]
    b_adr = flex_bendingadr[flexid]
    nhinge = int(flex_bending[b_adr])
    if scale_f <= 0.0 or nhinge <= 0:
      return

    eadr = flex_rod_edgeadr[flexid]
    nedge = flex_rod_edgenum[flexid]
    e0 = int(-1)
    e1 = int(-2)
    for e in range(nedge):
      jnt = flex_rodedge_jnt[eadr + e]
      dof = jnt_dofadr[jnt] if jnt >= 0 else -1
      if dof < 0:
        if e0 >= 0 and e1 < e0:
          e1 = e - 1
        continue
      if e1 >= e0 and e0 >= 0:
        return
      if e0 < 0:
        e0 = e
      if M_rownnz[dof] != 1 or tree_dofnum[dof_treeid[dof]] != 1:
        return
    if e0 < 0:
      return
    if e1 < e0:
      e1 = nedge - 1

    for e in range(e0, e1 + 1):
      dof = jnt_dofadr[flex_rodedge_jnt[eadr + e]]
      res_out[worldid, dof] = rhs_in[worldid, dof]
      d_extra = tw_diag_in[worldid, dof] if wp.static(has_tw_diag) else 0.0
      diag_e = wp.max(MJ_MINVAL, qH_in[worldid, M_rowadr[dof]] + d_extra)
      off_e = float(0.0)
      if e < e1:
        kt = flex_bending[b_adr + 1 + ROD_HINGE_SIZE * e + 8]
        if kt > 0.0:
          off_e = -scale_f * 2.0 * kt
      flex_rodedge_tridiag_out[worldid, eadr + e] = wp.vec2(diag_e, off_e)

    for e in range(e0, e1):
      td = flex_rodedge_tridiag_out[worldid, eadr + e]
      diag_k = td[0]
      lk = td[1] / diag_k
      flex_rodedge_tridiag_out[worldid, eadr + e] = wp.vec2(diag_k, lk)
      td_next = flex_rodedge_tridiag_out[worldid, eadr + e + 1]
      diag_next = wp.max(MJ_MINVAL, td_next[0] - lk * lk * diag_k)
      flex_rodedge_tridiag_out[worldid, eadr + e + 1] = wp.vec2(diag_next, td_next[1])
      dof_k = jnt_dofadr[flex_rodedge_jnt[eadr + e]]
      dof_next = jnt_dofadr[flex_rodedge_jnt[eadr + e + 1]]
      res_out[worldid, dof_next] = res_out[worldid, dof_next] - lk * res_out[worldid, dof_k]

    dof_last = jnt_dofadr[flex_rodedge_jnt[eadr + e1]]
    res_out[worldid, dof_last] = res_out[worldid, dof_last] / flex_rodedge_tridiag_out[worldid, eadr + e1][0]
    e_back = int(e1 - 1)
    while e_back >= e0:
      td = flex_rodedge_tridiag_out[worldid, eadr + e_back]
      dof_b = jnt_dofadr[flex_rodedge_jnt[eadr + e_back]]
      dof_b1 = jnt_dofadr[flex_rodedge_jnt[eadr + e_back + 1]]
      res_out[worldid, dof_b] = res_out[worldid, dof_b] / td[0] - td[1] * res_out[worldid, dof_b1]
      e_back -= 1

  return kernel


@wp.kernel
def _eff_diag_damp_shift(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # Data in:
  qvel_in: wp.array2d[float],
  # Data out:
  efm_c_out: wp.array2d[float],
  efm_diag_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  h = opt_timestep[worldid % opt_timestep.shape[0]]

  damp_deriv = float(0.0)
  if not (opt_disableflags & DisableBit.DAMPER):
    damping, dpoly = util_misc.dof_damping_total(
      jnt_actuatorid,
      dof_jntid,
      dof_damping,
      dof_dampingpoly,
      actuator_trntype,
      actuator_trnid,
      actuator_damping,
      actuator_dampingpoly,
      actuator_gear,
      worldid,
      dofid,
    )
    v = qvel_in[worldid, dofid]
    damp_deriv = wp.max(0.0, util_misc._poly_force_deriv(damping, dpoly, v, 1))

  ck = efm_diag_out[worldid, dofid]
  efm_diag_out[worldid, dofid] = h * damp_deriv + h * ck
  efm_c_out[worldid, dofid] = -ck * qvel_in[worldid, dofid]


@cache_kernel
def _eff_tendon_shift(max_ten_J_rownnz: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_ten_J_rownnz

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    tendon_actuatorid: wp.array[int],
    tendon_plasticadr: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_stiffness: wp.array2d[float],
    tendon_stiffnesspoly: wp.array2d[wp.vec2],
    tendon_damping: wp.array2d[float],
    tendon_dampingpoly: wp.array2d[wp.vec2],
    tendon_lengthspring: wp.array2d[wp.vec2],
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_damping: wp.array2d[float],
    actuator_dampingpoly: wp.array2d[wp.vec2],
    actuator_gear: wp.array2d[wp.spatial_vector],
    # Data in:
    plastic_in: wp.array2d[float],
    ten_J_in: wp.array2d[float],
    ten_length_in: wp.array2d[float],
    ten_velocity_in: wp.array2d[float],
    # Data out:
    efm_c_out: wp.array2d[float],
    efm_ts_out: wp.array2d[float],
  ):
    worldid, tenid = wp.tid()
    h = opt_timestep[worldid % opt_timestep.shape[0]]

    k = float(0.0)
    if not (opt_disableflags & DisableBit.SPRING):
      length = ten_length_in[worldid, tenid]
      padr = tendon_plasticadr[tenid]
      if padr >= 0:
        lower = plastic_in[worldid, padr + 0]
        upper = plastic_in[worldid, padr + 1]
      else:
        range_spring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]
        lower = range_spring[0]
        upper = range_spring[1]
      x = float(0.0)
      if length > upper:
        x = length - upper
      elif length < lower:
        x = length - lower
      if x != 0.0:
        stiff = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]
        spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]
        k = wp.max(0.0, util_misc._poly_force_deriv(stiff, spoly, x, 0))

    b = float(0.0)
    if not (opt_disableflags & DisableBit.DAMPER):
      damping, dpoly = util_misc.tendon_damping_total(
        tendon_actuatorid,
        tendon_damping,
        tendon_dampingpoly,
        actuator_trntype,
        actuator_trnid,
        actuator_damping,
        actuator_dampingpoly,
        actuator_gear,
        worldid,
        tenid,
      )
      v = ten_velocity_in[worldid, tenid]
      b = wp.max(0.0, util_misc._poly_force_deriv(damping, dpoly, v, 1))

    ts = h * h * k + h * b
    tk = h * k
    efm_ts_out[worldid, tenid] = ts

    if k > 0.0:
      ckv = tk * ten_velocity_in[worldid, tenid]
      rowadr = ten_J_rowadr[tenid]
      rownnz = ten_J_rownnz[tenid]
      for j in range(rownnz):
        sparseid = rowadr + j
        col = ten_J_colind[sparseid]
        val = ten_J_in[worldid, sparseid]
        wp.atomic_sub(efm_c_out, worldid, col, ckv * val)

  return kernel


@cache_kernel
def _eff_actuator_actuation(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    jnt_dofadr: wp.array[int],
    jnt_actfrclimited: wp.array[bool],
    jnt_actfrcrange: wp.array2d[wp.vec2],
    tendon_actfrclimited: wp.array[bool],
    tendon_actfrcrange: wp.array2d[wp.vec2],
    actuator_trntype: wp.array[int],
    actuator_dyntype: wp.array[int],
    actuator_gaintype: wp.array[int],
    actuator_biastype: wp.array[int],
    actuator_ctrladr: wp.array[int],
    actuator_ctrlnum: wp.array[int],
    actuator_ctrlspec: wp.array[int],
    actuator_actadr: wp.array[int],
    actuator_actnum: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_dynprm: wp.array2d[vec10],
    actuator_gainprm: wp.array2d[vec10],
    actuator_biasprm: wp.array2d[vec10],
    actuator_actlimited: wp.array[bool],
    actuator_actrange: wp.array2d[wp.vec2],
    actuator_actearly: wp.array[bool],
    actuator_forcelimited: wp.array[bool],
    actuator_forcerange: wp.array2d[wp.vec2],
    actuator_ctrllimited: wp.array[bool],
    actuator_ctrlrange: wp.array2d[wp.vec2],
    actuator_acc0: wp.array2d[float],
    actuator_lengthrange: wp.array2d[wp.vec2],
    # Data in:
    act_in: wp.array2d[float],
    ctrl_in: wp.array2d[float],
    act_dot_in: wp.array2d[float],
    actuator_length_in: wp.array2d[float],
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    actuator_velocity_in: wp.array2d[float],
    actuator_force_in: wp.array2d[float],
    qfrc_actuator_in: wp.array2d[float],
    # Data out:
    efm_ca_out: wp.array2d[float],
    efm_as_out: wp.array2d[float],
  ):
    worldid, actid = wp.tid()
    h = opt_timestep[worldid % opt_timestep.shape[0]]

    if opt_disableflags & DisableBit.ACTUATION:
      return

    gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], actid]
    biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], actid]
    dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]

    bias_len = float(0.0)
    gain_len = float(0.0)
    bias_vel = float(0.0)
    gain_vel = float(0.0)

    # DC motor: hot resistance and saturation flags (1 force, 2 Vmax, 4 dimax)
    gaintype = actuator_gaintype[actid]
    biastype = actuator_biastype[actid]
    R_dc = float(0.0)
    sat_dc = int(0)
    if gaintype == GainType.DCMOTOR or biastype == BiasType.DCMOTOR:
      R_dc = util_misc.dcmotor_resistance(act_in, worldid, actuator_actadr[actid], dynprm, gainprm)
      sat_dc = _dcmotor_saturation(
        opt_timestep,
        opt_disableflags,
        tendon_actfrclimited,
        tendon_actfrcrange,
        actuator_trntype,
        actuator_biastype,
        actuator_ctrladr,
        actuator_ctrlnum,
        actuator_ctrlspec,
        actuator_actadr,
        actuator_actnum,
        actuator_trnid,
        actuator_dynprm,
        actuator_gainprm,
        actuator_biasprm,
        actuator_actlimited,
        actuator_actrange,
        actuator_actearly,
        actuator_forcelimited,
        actuator_forcerange,
        actuator_ctrllimited,
        actuator_ctrlrange,
        act_in,
        ctrl_in,
        act_dot_in,
        actuator_length_in,
        actuator_velocity_in,
        actuator_force_in,
        worldid,
        actid,
        R_dc,
      )

    if biastype == BiasType.AFFINE:
      bias_len += biasprm[1]
      bias_vel += biasprm[2]
    elif biastype == BiasType.MUSCLE:
      bias_len += util_misc.muscle_bias_len_deriv(
        actuator_length_in[worldid, actid],
        actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid],
        actuator_acc0[worldid % actuator_acc0.shape[0], actid],
        biasprm,
      )
    elif biastype == BiasType.DCMOTOR:
      # cogging torque (mechanical, added after force clamping)
      A = biasprm[0]
      if A != 0.0:
        Np = biasprm[1]
        bias_len += A * Np * wp.cos(Np * actuator_length_in[worldid, actid] + biasprm[2])
      if dynprm[0] <= 0.0 and (sat_dc & 1) == 0:
        K = gainprm[1]
        bias_vel -= math.safe_div(K * K, R_dc)

    if gaintype == GainType.AFFINE:
      gain_len = gainprm[1]
      gain_vel = gainprm[2]
    elif gaintype == GainType.MUSCLE:
      lenrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid]
      acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], actid]
      gain_len = util_misc.muscle_gain_len_deriv(
        actuator_length_in[worldid, actid],
        actuator_velocity_in[worldid, actid],
        lenrange,
        acc0,
        gainprm,
      )
      gain_vel = util_misc.muscle_gain_vel_deriv(
        actuator_length_in[worldid, actid],
        actuator_velocity_in[worldid, actid],
        lenrange,
        acc0,
        gainprm,
      )
    elif gaintype == GainType.DCMOTOR:
      te = dynprm[0]
      R0 = gainprm[0]
      K = gainprm[1]

      # electrical terms vanish when electrical force is clamped by forcerange or tendon actfrcrange
      if (sat_dc & 1) == 0:
        # controller derivatives through the tau->V map, zero under Vmax
        dVdl = float(0.0)
        dVdw = float(0.0)
        if (actuator_ctrlspec[actid] & 7) != 0 and (sat_dc & 2) == 0:
          dVdl = math.safe_div(-gainprm[4] * R0, K)
          dVdw = math.safe_div(-gainprm[6] * R0, K) + K

        if te > 0.0:
          # stateful current with actearly, zero under dimax
          if actuator_actearly[actid] and (sat_dc & 4) == 0:
            s = 1.0 - wp.exp(-h / te)
            bias_len += math.safe_div(K * dVdl * s, R_dc)
            bias_vel += math.safe_div(K * (dVdw - K) * s, R_dc)
        else:
          if dVdl != 0.0:
            bias_len += math.safe_div(K * dVdl, R_dc)
          if dVdw != 0.0:
            bias_vel += math.safe_div(K * dVdw, R_dc)

      sigma1 = dynprm[6]
      if sigma1 > 0.0:
        bias_vel -= sigma1

    if gain_len != 0.0 or gain_vel != 0.0:
      u_val = float(0.0)
      if actuator_dyntype[actid] != DynType.NONE:
        act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1
        if actuator_actearly[actid]:
          u_val = next_act(
            h,
            actuator_dyntype[actid],
            dynprm,
            actuator_actrange[worldid % actuator_actrange.shape[0], actid],
            act_in[worldid, act_adr],
            act_dot_in[worldid, act_adr],
            1.0,
            actuator_actlimited[actid],
          )
        else:
          u_val = act_in[worldid, act_adr]
      else:
        cadr = actuator_ctrladr[actid]
        if cadr >= 0:
          u_val = ctrl_in[worldid, cadr]
          if not (opt_disableflags & DisableBit.CLAMPCTRL) and actuator_ctrllimited[cadr]:
            crange = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], cadr]
            u_val = wp.clamp(u_val, crange[0], crange[1])
      bias_len += gain_len * u_val
      bias_vel += gain_vel * u_val

    gp = wp.max(0.0, -bias_len)
    gv = wp.max(0.0, -bias_vel)
    if gp == 0.0 and gv == 0.0:
      efm_as_out[worldid, actid] = 0.0
      return

    # DC motor: electrical force clamping handled above, mechanical terms are unclamped
    if actuator_forcelimited[actid] and gaintype != GainType.DCMOTOR and biastype != BiasType.DCMOTOR:
      force = actuator_force_in[worldid, actid]
      forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
      if force <= forcerange[0] or force >= forcerange[1]:
        efm_as_out[worldid, actid] = 0.0
        return

    if (
      gaintype != GainType.DCMOTOR
      and biastype != BiasType.DCMOTOR
      and _tendon_actfrc_clamped(
        tendon_actfrclimited,
        tendon_actfrcrange,
        actuator_trntype,
        actuator_biastype,
        actuator_actadr,
        actuator_trnid,
        actuator_dynprm,
        actuator_gainprm,
        actuator_biasprm,
        act_in,
        act_dot_in,
        actuator_length_in,
        actuator_force_in,
        worldid,
        actid,
      )
    ):
      efm_as_out[worldid, actid] = 0.0
      return

    # skip if joint total actuator force is clamped by actfrcrange
    # (keeping tendon/site actuator contributions on this dof is an approximation)
    if _jnt_actfrc_clamped(
      jnt_dofadr,
      jnt_actfrclimited,
      jnt_actfrcrange,
      actuator_trntype,
      actuator_trnid,
      qfrc_actuator_in,
      worldid,
      actid,
    ):
      efm_as_out[worldid, actid] = 0.0
      return

    as_val = h * h * gp + h * gv
    ak_val = h * gp
    efm_as_out[worldid, actid] = as_val

    if gp > 0.0:
      ckv = ak_val * actuator_velocity_in[worldid, actid]
      rowadr = moment_rowadr_in[worldid, actid]
      rownnz = moment_rownnz_in[worldid, actid]
      for j in range(rownnz):
        sparseid = rowadr + j
        col = moment_colind_in[worldid, sparseid]
        val = actuator_moment_in[worldid, sparseid]
        wp.atomic_sub(efm_ca_out, worldid, col, ckv * val)

  return kernel


@wp.kernel
def _eff_init_sdiag(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  efm_diag_in: wp.array2d[float],
  # Data out:
  qH_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  diag_elem = M_rowadr[dofid] + M_rownnz[dofid] - 1
  qH_out[worldid, diag_elem] += efm_diag_in[worldid, dofid]


@cache_kernel
def _eff_add_tendon_qH_diag(max_ten_J_rownnz: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_ten_J_rownnz

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    # Data in:
    ten_J_in: wp.array2d[float],
    efm_ts_in: wp.array2d[float],
    # Data out:
    qH_out: wp.array2d[float],
  ):
    worldid, t = wp.tid()
    ts = efm_ts_in[worldid, t]
    if ts == 0.0:
      return
    rowadr = ten_J_rowadr[t]
    rownnz = ten_J_rownnz[t]
    for j in range(rownnz):
      sparseid = rowadr + j
      dof = ten_J_colind[sparseid]
      val = ten_J_in[worldid, sparseid]
      term = ts * val * val
      diag_elem = M_rowadr[dof] + M_rownnz[dof] - 1
      wp.atomic_add(qH_out, worldid, diag_elem, term)

  return kernel


@cache_kernel
def _eff_add_actuator_qH_diag(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    # Data in:
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    efm_as_in: wp.array2d[float],
    # Data out:
    qH_out: wp.array2d[float],
  ):
    worldid, a = wp.tid()
    as_val = efm_as_in[worldid, a]
    if as_val == 0.0:
      return
    rowadr = moment_rowadr_in[worldid, a]
    rownnz = moment_rownnz_in[worldid, a]
    for j in range(rownnz):
      sparseid = rowadr + j
      dof = moment_colind_in[worldid, sparseid]
      val = actuator_moment_in[worldid, sparseid]
      term = as_val * val * val
      diag_elem = M_rowadr[dof] + M_rownnz[dof] - 1
      wp.atomic_add(qH_out, worldid, diag_elem, term)

  return kernel


@wp.kernel
def _eff_add_fluid_qH(
  # Data in:
  efm_fluid_in: wp.array2d[float],
  # Data out:
  qH_out: wp.array2d[float],
):
  worldid, madr = wp.tid()
  qH_out[worldid, madr] += efm_fluid_in[worldid, madr]


@cache_kernel
def _eff_mul_fluid(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    M_mulm_rowadr: wp.array[int],
    M_mulm_col: wp.array[int],
    M_mulm_madr: wp.array[int],
    # Data in:
    efm_fluid_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip_in: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()
    if wp.static(check_skip):
      if skip_in[worldid]:
        return

    acc = float(0.0)
    start = M_mulm_rowadr[dofid]
    end = M_mulm_rowadr[dofid + 1]
    for k in range(start, end):
      col = M_mulm_col[k]
      madr = M_mulm_madr[k]
      acc += efm_fluid_in[worldid, madr] * vec_in[worldid, col]

    res_out[worldid, dofid] += acc

  return kernel


@cache_kernel
def _eff_mul_diag(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    efm_diag_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    res_out[worldid, dofid] += efm_diag_in[worldid, dofid] * vec_in[worldid, dofid]

  return kernel


@cache_kernel
def _eff_mul_diag_and_csr(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    efm_K_rownnz: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_colind: wp.array[int],
    # Data in:
    efm_diag_in: wp.array2d[float],
    efm_K_val_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip_in: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()
    if wp.static(check_skip):
      if skip_in[worldid]:
        return
    s = efm_diag_in[worldid, dofid] * vec_in[worldid, dofid]
    rownnz = efm_K_rownnz[dofid]
    if rownnz > 0:
      rowadr = efm_K_rowadr[dofid]
      for j in range(rownnz):
        adr = rowadr + j
        col = efm_K_colind[adr]
        s += efm_K_val_in[worldid, adr] * vec_in[worldid, col]
    res_out[worldid, dofid] += s

  return kernel


@cache_kernel
def _eff_mul_add_tendon(check_skip: bool, max_ten_J_rownnz: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_ten_J_rownnz

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    # Data in:
    ten_J_in: wp.array2d[float],
    efm_ts_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, t = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return

    s = efm_ts_in[worldid, t]
    if s == 0.0:
      return

    rowadr = ten_J_rowadr[t]
    rownnz = ten_J_rownnz[t]
    dot = float(0.0)
    for j in range(rownnz):
      sparseid = rowadr + j
      dot += ten_J_in[worldid, sparseid] * vec_in[worldid, ten_J_colind[sparseid]]
    dot *= s
    for j in range(rownnz):
      sparseid = rowadr + j
      wp.atomic_add(res_out, worldid, ten_J_colind[sparseid], dot * ten_J_in[worldid, sparseid])

  return kernel


@cache_kernel
def _eff_mul_add_actuator(check_skip: bool, nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Data in:
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    efm_as_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, a = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return

    s = efm_as_in[worldid, a]
    if s == 0.0:
      return

    rowadr = moment_rowadr_in[worldid, a]
    rownnz = moment_rownnz_in[worldid, a]
    dot = float(0.0)
    for j in range(rownnz):
      sparseid = rowadr + j
      dot += actuator_moment_in[worldid, sparseid] * vec_in[worldid, moment_colind_in[worldid, sparseid]]
    dot *= s
    for j in range(rownnz):
      sparseid = rowadr + j
      wp.atomic_add(res_out, worldid, moment_colind_in[worldid, sparseid], dot * actuator_moment_in[worldid, sparseid])

  return kernel


@wp.kernel
def _eff_rhs(
  # Model:
  opt_enableflags: int,
  dof_treeid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  qfrc_smooth_in: wp.array2d[float],
  efm_c_in: wp.array2d[float],
  efm_ca_in: wp.array2d[float],
  # Out:
  rhs_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if (opt_enableflags & EnableBit.SLEEP) and tree_awake_in[worldid, dof_treeid[dofid]] == 0:
    rhs_out[worldid, dofid] = 0.0
    return
  rhs_out[worldid, dofid] = qfrc_smooth_in[worldid, dofid] + efm_c_in[worldid, dofid] + efm_ca_in[worldid, dofid]


@wp.func
def _find_csr_col(rowadr: int, rownnz: int, target_col: int, colind: wp.array[int]) -> int:
  lo = rowadr
  hi = rowadr + rownnz - 1
  while lo <= hi:
    mid = (lo + hi) // 2
    c = colind[mid]
    if c == target_col:
      return mid
    elif c < target_col:
      lo = mid + 1
    else:
      hi = mid - 1
  return -1


@cache_kernel
def _eff_flex_stretch_stiff_vert(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    efm_K_rownnz: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_colind: wp.array[int],
    flex_vertflexid: wp.array[int],
    flex_simple: wp.array[bool],
    # Data in:
    xmat_in: wp.array2d[wp.mat33],
    flexvert_hessian_in: wp.array2d[vec6],
    # Data out:
    efm_K_val_out: wp.array2d[float],
  ):
    worldid, vertid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    f = flex_vertflexid[vertid]
    if f < 0 or flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2 or not flex_simple[f]:
      return

    stiffness_adr = flex_stiffnessadr[f]
    if stiffness_adr < 0 or flex_stiffness[stiffness_adr] == 0.0:
      return

    s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
    scale = s1 + s2 * flex_damping[f]
    if scale == 0.0:
      return

    bi = body_weldid[flex_vertbodyid[vertid]]
    if body_dofnum[bi] != 3:
      return
    dof_i = body_dofadr[bi]
    if efm_K_rownnz[dof_i] == 0:
      return

    h = flexvert_hessian_in[worldid, vertid]
    block = wp.mat33(h[0], h[1], h[2], h[1], h[3], h[4], h[2], h[4], h[5])
    R_bi = xmat_in[worldid, bi]
    blkd = scale * (wp.transpose(R_bi) * block * R_bi)

    for r in range(3):
      row = dof_i + r
      idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_i, efm_K_colind)
      if idx >= 0:
        for c in range(3):
          wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])

  return kernel


@cache_kernel
def _eff_flex_stretch_stiff_edge(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    efm_K_rownnz: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_colind: wp.array[int],
    flex_edgeflexid: wp.array[int],
    flex_simple: wp.array[bool],
    # Data in:
    xmat_in: wp.array2d[wp.mat33],
    flexedge_hessian_in: wp.array2d[wp.mat33],
    # Data out:
    efm_K_val_out: wp.array2d[float],
  ):
    worldid, edgeid, side = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    f = flex_edgeflexid[edgeid]
    if f < 0 or flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2 or not flex_simple[f]:
      return

    stiffness_adr = flex_stiffnessadr[f]
    if stiffness_adr < 0 or flex_stiffness[stiffness_adr] == 0.0:
      return

    s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
    scale = s1 + s2 * flex_damping[f]
    if scale == 0.0:
      return

    va = flex_vertadr[f]
    edge = flex_edge[edgeid]
    vi = edge[side]
    vj = edge[1 - side]

    bi = body_weldid[flex_vertbodyid[va + vi]]
    bj = body_weldid[flex_vertbodyid[va + vj]]
    if body_dofnum[bi] != 3 or body_dofnum[bj] != 3:
      return

    dof_i = body_dofadr[bi]
    dof_j = body_dofadr[bj]
    if efm_K_rownnz[dof_i] == 0:
      return

    h = flexedge_hessian_in[worldid, edgeid]
    block = wp.transpose(h) if side == 1 else h
    blkd = scale * (wp.transpose(xmat_in[worldid, bi]) * block * xmat_in[worldid, bj])

    for r in range(3):
      row = dof_i + r
      idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)
      if idx >= 0:
        for c in range(3):
          wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])

  return kernel


@cache_kernel
def _eff_flex_bend_stiff(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_edgeadr: wp.array[int],
    flex_edgenum: wp.array[int],
    flex_bendingadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_edgeflap: wp.array[wp.vec2i],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_centered: wp.array[bool],
    efm_K_rownnz: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_colind: wp.array[int],
    flex_edgeflexid: wp.array[int],
    flex_simple: wp.array[bool],
    # Data in:
    xquat_in: wp.array2d[wp.quat],
    xmat_in: wp.array2d[wp.mat33],
    # Data out:
    efm_K_val_out: wp.array2d[float],
  ):
    worldid, edgeid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    f = flex_edgeflexid[edgeid]

    if f < 0:
      return

    if flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] != 2 or not flex_simple[f]:
      return

    bendingadr = flex_bendingadr[f]
    if bendingadr < 0:
      return

    s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
    scale = s1 + s2 * flex_damping[f]
    if scale == 0.0:
      return

    local_edgeid = edgeid - flex_edgeadr[f]
    vbase = flex_vertadr[f]

    edge = flex_edge[edgeid]
    flap = flex_edgeflap[edgeid]

    if flap[1] == -1:
      return

    verts = wp.vec4i(edge[0], edge[1], flap[0], flap[1])
    bend_offset = bendingadr + 17 * local_edgeid

    dofs = wp.vec4i(-1, -1, -1, -1)
    bodies = wp.vec4i(-1, -1, -1, -1)
    for k in range(4):
      bk = body_weldid[flex_vertbodyid[vbase + verts[k]]]
      bodies[k] = bk
      if body_dofnum[bk] == 3:
        dofs[k] = body_dofadr[bk]

    for i in range(4):
      dof_i = dofs[i]
      if dof_i >= 0 and efm_K_rownnz[dof_i] > 0:
        bi = bodies[i]
        R_bi_T = wp.transpose(xmat_in[worldid, bi])
        qi = xquat_in[worldid, bi]
        for j in range(4):
          dof_j = dofs[j]
          if dof_j >= 0:
            q = scale * flex_bending[bend_offset + 4 * i + j]
            if q != 0.0:
              bj = bodies[j]
              if bi == bj or qi == xquat_in[worldid, bj]:
                for r in range(3):
                  row = dof_i + r
                  idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)
                  if idx >= 0:
                    wp.atomic_add(efm_K_val_out, worldid, idx + r, q)
              else:
                blkd = q * (R_bi_T * xmat_in[worldid, bj])
                for r in range(3):
                  row = dof_i + r
                  idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)
                  if idx >= 0:
                    for c in range(3):
                      wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])

  return kernel


@cache_kernel
def _eff_flex_rod_stiff(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_bendingadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    efm_K_rownnz: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_colind: wp.array[int],
    flex_simple: wp.array[bool],
    flex_rodhinge_map: wp.array[wp.vec2i],
    flex_rod_is_chain: wp.array[bool],
    flex_rod_hingeadr: wp.array[int],
    # Data in:
    xmat_in: wp.array2d[wp.mat33],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    flex_rod_chain_ok_in: wp.array2d[bool],
    flex_rodhinge_Wk_in: wp.array2d[wp.mat33],
    flex_rodhinge_Wd_in: wp.array2d[wp.mat33],
    # Data out:
    efm_K_val_out: wp.array2d[float],
  ):
    """Adds the rod hinge stiffness J'(h^2 Wk + h damping Wd)J in the vertex body frames."""
    worldid, hingeid = wp.tid()
    mapping = flex_rodhinge_map[hingeid]
    f = mapping[0]
    adr = mapping[1]
    if not flex_simple[f]:
      return

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
    sd = s2 * flex_damping[f]
    if s1 == 0.0 and sd == 0.0:
      return

    ok, verts, J0, J1, J2, J3, _, Wk, Wd = rod_stencil(flex_bending, flexvert_xpos_in, worldid, flex_vertadr[f], adr)
    if not ok:
      return
    ok_chain = bool(False)
    if verts[3] < 0 and flex_rod_is_chain[f] and flex_rod_chain_ok_in[worldid, f]:
      ok_chain = bool(True)
      hidx = flex_rod_hingeadr[f] + (adr - flex_bendingadr[f] - 1) // ROD_HINGE_SIZE
      Wk = _rod_pad44(flex_rodhinge_Wk_in[worldid, hidx])
      Wd = _rod_pad44(flex_rodhinge_Wd_in[worldid, hidx])
    W = s1 * Wk + sd * Wd
    nvert = 3 if verts[3] < 0 else 4

    j_psi = mat43()
    wt = float(0.0)
    if wp.static(ROD_HINGE_SIZE == 10) and verts[3] < 0 and ok_chain:
      wt_cand = (s1 + sd) * 2.0 * flex_bending[adr + 8]
      if wt_cand > 0.0:
        x0 = flexvert_xpos_in[worldid, verts[0]]
        x1 = flexvert_xpos_in[worldid, verts[1]]
        x2 = flexvert_xpos_in[worldid, verts[2]]
        ok_kb, kb_t, _A0_t, _A1_t = _rod_hinge(x0, x1, x2)
        l0 = wp.length(x1 - x0)
        l1 = wp.length(x2 - x1)
        if ok_kb and l0 >= MJ_MINVAL and l1 >= MJ_MINVAL:
          inv0 = 0.5 / l0
          inv1 = 0.5 / l1
          j_psi[0] = -inv0 * kb_t
          j_psi[1] = (inv0 - inv1) * kb_t
          j_psi[2] = inv1 * kb_t
          wt = wt_cand

    dofs = wp.vec4i(-1)
    bodies = wp.vec4i(-1)
    for i in range(nvert):
      bodyid = flex_vertbodyid[verts[i]]
      if bodyid >= 0:
        bi = body_weldid[bodyid]
        bodies[i] = bi
        if body_dofnum[bi] == 3:
          dofs[i] = body_dofadr[bi]

    for i in range(4):
      dof_i = dofs[i]
      if dof_i >= 0 and efm_K_rownnz[dof_i] > 0:
        JiT_W = wp.transpose(rod_stencil_jacobian(J0, J1, J2, J3, i)) * W
        R_bi_T = wp.transpose(xmat_in[worldid, bodies[i]])
        for j in range(4):
          dof_j = dofs[j]
          if dof_j >= 0:
            world_blk = JiT_W * rod_stencil_jacobian(J0, J1, J2, J3, j)
            if wt > 0.0:
              world_blk += wt * wp.outer(j_psi[i], j_psi[j])
            blkd = R_bi_T * world_blk * xmat_in[worldid, bodies[j]]
            for r in range(3):
              row = dof_i + r
              idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)
              if idx >= 0:
                for c in range(3):
                  wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])

  return kernel


@cache_kernel
def _eff_flex_interp_stiff(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 9 * FLEX_INTERP_MAX_NODES * FLEX_INTERP_MAX_NODES

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_node0: wp.array[wp.vec3],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_centered: wp.array[bool],
    efm_K_rownnz: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_colind: wp.array[int],
    flex_faceadr: wp.array[int],
    flex_cell_map: wp.array[wp.vec4i],
    flex_face: wp.array2d[int],
    # Data in:
    xmat_in: wp.array2d[wp.mat33],
    flexnode_xpos_in: wp.array2d[wp.vec3],
    face_quat_in: wp.array2d[wp.quat],
    # Data out:
    efm_K_val_out: wp.array2d[float],
  ):
    worldid, cellid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    mapping = flex_cell_map[cellid]
    f = mapping[0]
    ci = mapping[1]
    cj = mapping[2]
    ck = mapping[3]

    order = flex_interp[f]
    if order == 0:
      return
    shell_mode = order < 0
    order_abs = -order if shell_mode else order

    stiffness_adr_base = flex_stiffnessadr[f]
    if stiffness_adr_base < 0:
      return

    s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
    iscale = -(s1 + s2 * flex_damping[f])
    if iscale == 0.0:
      return

    npc = (order_abs + 1) * (order_abs + 1) if shell_mode else (order_abs + 1) * (order_abs + 1) * (order_abs + 1)
    ndof_cell = 3 * npc
    cellnum = flex_cellnum[f]
    cy = cellnum[1]
    cz = cellnum[2]
    nstart = flex_nodeadr[f]
    ny_g = cy * order_abs + 1
    nz_g = cz * order_abs + 1

    cell_idx = ci if shell_mode else ci * cy * cz + cj * cz + ck
    k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell
    if not shell_mode and flex_stiffness[k_base] == 0.0:
      return

    if shell_mode:
      cell_quat = face_quat_in[worldid, flex_faceadr[f] + ci]
    else:
      cell_quat = support.compute_interp_cell_quat(
        flex_node0, flexnode_xpos_in, order_abs, ci, cj, ck, cy, cz, ny_g, nz_g, nstart, worldid
      )
    R = wp.quat_to_matrix(cell_quat)
    RT = wp.transpose(R)

    op1 = order_abs + 1
    for idx_i in range(npc):
      if shell_mode:
        node_i = flex_face[flex_faceadr[f] + ci, idx_i]
      else:
        li_i = idx_i // (op1 * op1)
        lj_i = (idx_i // op1) % op1
        lk_i = idx_i % op1
        gidx_i = (ci * order_abs + li_i) * ny_g * nz_g + (cj * order_abs + lj_i) * nz_g + (ck * order_abs + lk_i)
        node_i = nstart + gidx_i
      bi = flex_nodebodyid[node_i]
      if body_dofnum[bi] != 3:
        continue
      dof_i = body_dofadr[bi]
      T_i_T = wp.transpose(xmat_in[worldid, bi]) * R

      for idx_j in range(npc):
        if shell_mode:
          node_j = flex_face[flex_faceadr[f] + ci, idx_j]
        else:
          li_j = idx_j // (op1 * op1)
          lj_j = (idx_j // op1) % op1
          lk_j = idx_j % op1
          gidx_j = (ci * order_abs + li_j) * ny_g * nz_g + (cj * order_abs + lj_j) * nz_g + (ck * order_abs + lk_j)
          node_j = nstart + gidx_j
        bj = flex_nodebodyid[node_j]
        if body_dofnum[bj] != 3:
          continue
        dof_j = body_dofadr[bj]
        T_j = RT * xmat_in[worldid, bj]

        K_ij = wp.mat33(0.0)
        for r in range(3):
          for c in range(3):
            K_ij[r, c] = flex_stiffness[k_base + (3 * idx_i + r) * ndof_cell + (3 * idx_j + c)]

        blkd = iscale * (T_i_T * K_ij * T_j)
        for r in range(3):
          row = dof_i + r
          rowadr = efm_K_rowadr[row]
          rownnz = efm_K_rownnz[row]
          for c in range(3):
            col = dof_j + c
            idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)
            if idx >= 0:
              wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])

  return kernel


@wp.kernel
def _eff_add_efmK_qH_diag(
  # Model:
  body_parentid: wp.array[int],
  body_weldid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  jnt_type: wp.array[int],
  jnt_axis: wp.array2d[wp.vec3],
  dof_bodyid: wp.array[int],
  dof_treeid: wp.array[int],
  tree_dofnum: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  # Data in:
  efm_K_val_in: wp.array2d[float],
  # In:
  qH_eff_in: wp.array2d[float],
  # Out:
  qH_eff_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  rownnz = efm_K_rownnz[dofid]
  if rownnz <= 0 or tree_dofnum[dof_treeid[dofid]] != 3:
    return
  b = body_weldid[dof_bodyid[dofid]]
  b_dofadr = body_dofadr[b]
  if body_dofnum[b] != 3 or dofid < b_dofadr or dofid >= b_dofadr + 3:
    return
  is_simple = body_simple[b] == 2
  if not is_simple:
    if body_jntnum[b] == 3 and body_dofnum[body_weldid[body_parentid[b]]] == 0:
      jadr = body_jntadr[b]
      ax_w = worldid % jnt_axis.shape[0]
      is_simple = True
      for j in range(3):
        if jnt_type[jadr + j] != int(JointType.SLIDE):
          is_simple = False
        ax = jnt_axis[ax_w, jadr + j]
        for k in range(3):
          target = 1.0 if j == k else 0.0
          if wp.abs(ax[k] - target) > MJ_MINVAL:
            is_simple = False
  if not is_simple:
    return
  idx = _find_csr_col(efm_K_rowadr[dofid], rownnz, dofid, efm_K_colind)
  if idx >= 0:
    diag = M_rowadr[dofid] + M_rownnz[dofid] - 1
    qH_eff_out[worldid, diag] = qH_eff_in[worldid, diag] + efm_K_val_in[worldid, idx]


@wp.kernel
def _eff_zero_efm0_dofs(
  # Model:
  efm0_dofid: wp.array[int],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  vec_out[worldid, efm0_dofid[k]] = 0.0


@wp.kernel
def _eff_prec_efm0_solve(
  # Model:
  nefm0dof: int,
  efm0_dofid: wp.array[int],
  efm0_L_rownnz: wp.array[int],
  efm0_L_rowadr: wp.array[int],
  efm0_L_colind: wp.array[int],
  efm0_L: wp.array[float],
  # In:
  vec_in: wp.array2d[float],
  # Out:
  res_out: wp.array2d[float],
):
  worldid = wp.tid()
  for r in range(nefm0dof):
    dof_r = efm0_dofid[r]
    res_out[worldid, dof_r] = vec_in[worldid, dof_r]

  i = nefm0dof - 1
  while i >= 0:
    adr = efm0_L_rowadr[i]
    nnz = efm0_L_rownnz[i]
    diag_idx = adr + nnz - 1
    dof_i = efm0_dofid[i]
    val_i = res_out[worldid, dof_i] / efm0_L[diag_idx]
    res_out[worldid, dof_i] = val_i
    if val_i != 0.0:
      for j in range(nnz - 1):
        c = efm0_L_colind[adr + j]
        res_out[worldid, efm0_dofid[c]] -= efm0_L[adr + j] * val_i
    i -= 1

  for i in range(nefm0dof):
    adr = efm0_L_rowadr[i]
    nnz = efm0_L_rownnz[i]
    diag_idx = adr + nnz - 1
    dof_i = efm0_dofid[i]
    acc = res_out[worldid, dof_i]
    for j in range(nnz - 1):
      c = efm0_L_colind[adr + j]
      acc -= efm0_L[adr + j] * res_out[worldid, efm0_dofid[c]]
    res_out[worldid, dof_i] = acc / efm0_L[diag_idx]


@wp.kernel
def _zero_sleeping_dofs(
  # Model:
  dof_treeid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if tree_awake_in[worldid, dof_treeid[dofid]] == 0:
    vec_out[worldid, dofid] = 0.0


@cache_kernel
def _eff_flex_interp_mul(check_skip: bool, is_shift: bool, nv: int, deterministic: bool = False):
  """Applies corotated interpolated flex stiffness/damping operator to a DOF vector."""
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = FLEX_INTERP_MAX_NODES * nv

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_simple: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_node0: wp.array[wp.vec3],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_faceadr: wp.array[int],
    flex_cell_map: wp.array[wp.vec4i],
    flex_face: wp.array2d[int],
    # Data in:
    xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexnode_xpos_in: wp.array2d[wp.vec3],
    face_quat_in: wp.array2d[wp.quat],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, cellid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    if timestep == 0.0:
      return

    mapping = flex_cell_map[cellid]
    f = mapping[0]
    ci = mapping[1]
    cj = mapping[2]
    ck = mapping[3]

    order = flex_interp[f]
    if order == 0:
      return
    shell_mode = order < 0
    order_abs = -order if shell_mode else order

    stiffness_adr_base = flex_stiffnessadr[f]
    if stiffness_adr_base < 0:
      return

    if wp.static(is_shift):
      if opt_disableflags & DisableBit.SPRING:
        return
      scale = timestep
    else:
      s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else -timestep * timestep
      s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else -timestep
      scale = s1 + s2 * flex_damping[f]
      if scale == 0.0:
        return

    npc = (order_abs + 1) * (order_abs + 1) if shell_mode else (order_abs + 1) * (order_abs + 1) * (order_abs + 1)
    ndof_cell = 3 * npc
    cellnum = flex_cellnum[f]
    cy = cellnum[1]
    cz = cellnum[2]
    nstart = flex_nodeadr[f]
    ny_g = cy * order_abs + 1
    nz_g = cz * order_abs + 1

    cell_idx = ci if shell_mode else ci * cy * cz + cj * cz + ck
    k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell
    if not shell_mode and flex_stiffness[k_base] == 0.0:
      return

    if shell_mode:
      cell_quat = face_quat_in[worldid, flex_faceadr[f] + ci]
    else:
      cell_quat = support.compute_interp_cell_quat(
        flex_node0, flexnode_xpos_in, order_abs, ci, cj, ck, cy, cz, ny_g, nz_g, nstart, worldid
      )
    cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])

    op1 = order_abs + 1
    for idx_i in range(npc):
      if shell_mode:
        node_i = flex_face[flex_faceadr[f] + ci, idx_i]
      else:
        li_i = idx_i // (op1 * op1)
        lj_i = (idx_i // op1) % op1
        lk_i = idx_i % op1
        gidx_i = (ci * order_abs + li_i) * ny_g * nz_g + (cj * order_abs + lj_i) * nz_g + (ck * order_abs + lk_i)
        node_i = nstart + gidx_i
      bi = flex_nodebodyid[node_i]
      if bi == 0:
        continue

      frc_corot = wp.vec3(0.0, 0.0, 0.0)
      for idx_j in range(npc):
        if shell_mode:
          node_j = flex_face[flex_faceadr[f] + ci, idx_j]
        else:
          li_j = idx_j // (op1 * op1)
          lj_j = (idx_j // op1) % op1
          lk_j = idx_j % op1
          gidx_j = (ci * order_abs + li_j) * ny_g * nz_g + (cj * order_abs + lj_j) * nz_g + (ck * order_abs + lk_j)
          node_j = nstart + gidx_j
        bj = flex_nodebodyid[node_j]
        if bj == 0:
          continue

        v_world_j = wp.vec3(0.0, 0.0, 0.0)
        if body_simple[bj] == 2 and body_dofnum[bj] == 3:
          dof_j = body_dofadr[bj]
          v_loc_j = wp.vec3(vec_in[worldid, dof_j], vec_in[worldid, dof_j + 1], vec_in[worldid, dof_j + 2])
          v_world_j = xmat_in[worldid, bj] * v_loc_j
        else:
          pt_j = flexnode_xpos_in[worldid, node_j]
          offset_j = pt_j - subtree_com_in[worldid, body_rootid[bj]]
          cur_b = bj
          while cur_b > 0:
            d_adr = body_dofadr[cur_b]
            d_num = body_dofnum[cur_b]
            for d_idx in range(d_adr, d_adr + d_num):
              cdof = cdof_in[worldid, d_idx]
              jacp = wp.spatial_bottom(cdof) + wp.cross(wp.spatial_top(cdof), offset_j)
              v_world_j += jacp * vec_in[worldid, d_idx]
            cur_b = body_parentid[cur_b]

        v_corot_j = wp.quat_rotate(cell_quat_inv, v_world_j)

        for r in range(3):
          row = 3 * idx_i + r
          acc = float(0.0)
          for c in range(3):
            col = 3 * idx_j + c
            acc += flex_stiffness[k_base + row * ndof_cell + col] * v_corot_j[c]
          frc_corot[r] += acc

      frc_world = wp.quat_rotate(cell_quat, frc_corot)
      if body_simple[bi] == 2 and body_dofnum[bi] == 3:
        dof_i = body_dofadr[bi]
        frc_loc = wp.transpose(xmat_in[worldid, bi]) * frc_world
        for r in range(3):
          wp.atomic_add(res_out, worldid, dof_i + r, scale * frc_loc[r])
      else:
        pt_i = flexnode_xpos_in[worldid, node_i]
        offset_i = pt_i - subtree_com_in[worldid, body_rootid[bi]]
        cur_b = bi
        while cur_b > 0:
          d_adr = body_dofadr[cur_b]
          d_num = body_dofnum[cur_b]
          for d_idx in range(d_adr, d_adr + d_num):
            cdof = cdof_in[worldid, d_idx]
            jacp = wp.spatial_bottom(cdof) + wp.cross(wp.spatial_top(cdof), offset_i)
            wp.atomic_add(res_out, worldid, d_idx, scale * wp.dot(jacp, frc_world))
          cur_b = body_parentid[cur_b]

  return kernel


@wp.kernel(module="unique")
def _pcg_init_tiled(
  # Model:
  nv: int,
  nactuator: int,
  ntendon: int,
  opt_enableflags: int,
  dof_treeid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  efm_ts_in: wp.array2d[float],
  efm_as_in: wp.array2d[float],
  # In:
  rhs_in: wp.array2d[float],
  z0_in: wp.array2d[float],
  flex_any: bool,
  tendon_any: bool,
  actuator_any: bool,
  # Data out:
  qacc_out: wp.array2d[float],
  # Out:
  efm_r_out: wp.array2d[float],
  efm_p_out: wp.array2d[float],
  efm_rz_out: wp.array[float],
  efm_bn_out: wp.array[float],
  efm_done_out: wp.array[bool],
  efm_nsolving_out: wp.array[int],
):
  worldid, tid = wp.tid()
  BLOCK_DIM = wp.block_dim()
  sleep_enabled = (opt_enableflags & EnableBit.SLEEP) != 0

  active = flex_any
  if not active and tendon_any:
    for t in range(ntendon):
      if efm_ts_in[worldid, t] != 0.0:
        active = True
        break
  if not active and actuator_any:
    for a in range(nactuator):
      if efm_as_in[worldid, a] != 0.0:
        active = True
        break

  if not active:
    for dofid in range(tid, nv, BLOCK_DIM):
      z = z0_in[worldid, dofid]
      if sleep_enabled and tree_awake_in[worldid, dof_treeid[dofid]] == 0:
        z = 0.0
      qacc_out[worldid, dofid] = z
      efm_r_out[worldid, dofid] = 0.0
      efm_p_out[worldid, dofid] = 0.0
    if tid == 0:
      efm_bn_out[worldid] = 0.0
      efm_rz_out[worldid] = 0.0
      efm_done_out[worldid] = True
      wp.atomic_sub(efm_nsolving_out, 0, 1)  # kernel_analyzer: ignore[atomic]
    return

  local_bn = float(0.0)
  local_rz = float(0.0)
  for dofid in range(tid, nv, BLOCK_DIM):
    b = rhs_in[worldid, dofid]
    z = z0_in[worldid, dofid]
    if sleep_enabled and tree_awake_in[worldid, dof_treeid[dofid]] == 0:
      b = 0.0
      z = 0.0
    efm_r_out[worldid, dofid] = b
    efm_p_out[worldid, dofid] = z
    qacc_out[worldid, dofid] = 0.0
    local_bn += b * b
    local_rz += b * z

  bn_tile = wp.tile(local_bn, preserve_type=True)
  bn_sum = wp.tile_reduce(wp.add, bn_tile)
  rz_tile = wp.tile(local_rz, preserve_type=True)
  rz_sum = wp.tile_reduce(wp.add, rz_tile)
  if tid == 0:
    bn = bn_sum[0]
    efm_bn_out[worldid] = bn
    efm_rz_out[worldid] = rz_sum[0]
    done = bn <= MJ_MINVAL
    efm_done_out[worldid] = done
    if done:
      wp.atomic_sub(efm_nsolving_out, 0, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel(module="unique")
def _pcg_pAp_step_and_check_tiled(
  # Model:
  nv: int,
  opt_tolerance: wp.array[float],
  # In:
  efm_p_in: wp.array2d[float],
  efm_Ap_in: wp.array2d[float],
  efm_rz_in: wp.array[float],
  efm_bn_in: wp.array[float],
  efm_done_in: wp.array[bool],
  # Data out:
  qacc_out: wp.array2d[float],
  # Out:
  efm_r_out: wp.array2d[float],
  efm_done_out: wp.array[bool],
  efm_nsolving_out: wp.array[int],
):
  worldid, tid = wp.tid()
  if efm_done_in[worldid]:
    return

  local_pAp = float(0.0)
  BLOCK_DIM = wp.block_dim()
  for dofid in range(tid, nv, BLOCK_DIM):
    local_pAp += efm_p_in[worldid, dofid] * efm_Ap_in[worldid, dofid]

  pAp_tile = wp.tile(local_pAp, preserve_type=True)
  pAp_sum = wp.tile_reduce(wp.add, pAp_tile)
  pAp = pAp_sum[0]
  if pAp <= 0.0:
    if tid == 0:
      efm_done_out[worldid] = True
      wp.atomic_sub(efm_nsolving_out, 0, 1)  # kernel_analyzer: ignore[atomic]
    return

  alpha = efm_rz_in[worldid] / pAp

  local_rr = float(0.0)
  for dofid in range(tid, nv, BLOCK_DIM):
    p = efm_p_in[worldid, dofid]
    ap = efm_Ap_in[worldid, dofid]
    qacc_out[worldid, dofid] += alpha * p
    r = efm_r_out[worldid, dofid] - alpha * ap
    efm_r_out[worldid, dofid] = r
    local_rr += r * r

  rr_tile = wp.tile(local_rr, preserve_type=True)
  rr_sum = wp.tile_reduce(wp.add, rr_tile)
  if tid == 0:
    tol_user = opt_tolerance[worldid % opt_tolerance.shape[0]]
    tol_base = wp.max(tol_user, 1e-5)
    tol = tol_base * tol_base * efm_bn_in[worldid]
    if rr_sum[0] < tol:
      efm_done_out[worldid] = True
      wp.atomic_sub(efm_nsolving_out, 0, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel(module="unique")
def _pcg_beta_and_p_update_tiled(
  # Model:
  nv: int,
  # In:
  efm_r_in: wp.array2d[float],
  efm_z_in: wp.array2d[float],
  efm_rz_in: wp.array[float],
  efm_done_in: wp.array[bool],
  max_iter: int,
  # Out:
  efm_p_out: wp.array2d[float],
  efm_rz_out: wp.array[float],
  efm_nsolving_out: wp.array[int],
  efm_iter_out: wp.array[int],
):
  worldid, tid = wp.tid()
  if worldid == 0 and tid == 0:
    it = efm_iter_out[0] + 1
    efm_iter_out[0] = it
    if it >= max_iter:
      efm_nsolving_out[0] = 0

  if efm_done_in[worldid]:
    return

  local_rznew = float(0.0)
  BLOCK_DIM = wp.block_dim()
  for dofid in range(tid, nv, BLOCK_DIM):
    local_rznew += efm_r_in[worldid, dofid] * efm_z_in[worldid, dofid]

  rz_tile = wp.tile(local_rznew, preserve_type=True)
  rz_sum = wp.tile_reduce(wp.add, rz_tile)
  rznew = rz_sum[0]
  rz = efm_rz_in[worldid]
  beta = math.safe_div(rznew, rz)
  if tid == 0:
    efm_rz_out[worldid] = rznew

  for dofid in range(tid, nv, BLOCK_DIM):
    efm_p_out[worldid, dofid] = efm_z_in[worldid, dofid] + beta * efm_p_out[worldid, dofid]


@cache_kernel
def _eff_contact_shift(max_dof: int, deterministic: bool = False):
  """Applies passive contact smooth-force velocity shift c += -h*k*Jn^T*(Jn*v)."""
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_dof

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    # Data in:
    qvel_in: wp.array2d[float],
    contact_worldid_in: wp.array[int],
    # In:
    efm_con_dof_in: wp.array2d[int],
    efm_con_val_in: wp.array2d[float],
    efm_con_scale_in: wp.array[float],
    efm_con_nnz_in: wp.array[int],
    # Data out:
    efm_c_out: wp.array2d[float],
  ):
    cid = wp.tid()
    nnz = efm_con_nnz_in[cid]
    if nnz == 0:
      return
    scale = efm_con_scale_in[cid]
    if scale == float(0.0):
      return
    worldid = contact_worldid_in[cid]
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    neg_scale_dt = -scale / timestep
    dot = float(0.0)
    for a in range(nnz):
      dof = efm_con_dof_in[cid, a]
      dot += efm_con_val_in[cid, a] * qvel_in[worldid, dof]
    scaled_dot = dot * neg_scale_dt
    for a in range(nnz):
      dof = efm_con_dof_in[cid, a]
      wp.atomic_add(efm_c_out, worldid, dof, scaled_dot * efm_con_val_in[cid, a])

  return kernel


@cache_kernel
def _eff_mul_add_contact(check_skip: bool, max_dof: int, deterministic: bool = False):
  """Multiplies vector by passive contact rank-1 metric curvature h^2*k*Jn^T*Jn."""
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_dof

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Data in:
    contact_worldid_in: wp.array[int],
    # In:
    efm_con_dof_in: wp.array2d[int],
    efm_con_val_in: wp.array2d[float],
    efm_con_scale_in: wp.array[float],
    efm_con_nnz_in: wp.array[int],
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    cid = wp.tid()
    nnz = efm_con_nnz_in[cid]
    if nnz == 0:
      return
    worldid = contact_worldid_in[cid]
    if wp.static(check_skip):
      if skip[worldid]:
        return
    scale = efm_con_scale_in[cid]
    if scale == float(0.0):
      return
    dot = float(0.0)
    for a in range(nnz):
      dof = efm_con_dof_in[cid, a]
      dot += efm_con_val_in[cid, a] * vec_in[worldid, dof]
    dot *= scale
    for a in range(nnz):
      dof = efm_con_dof_in[cid, a]
      wp.atomic_add(res_out, worldid, dof, dot * efm_con_val_in[cid, a])

  return kernel


@event_scope
def eff_build(m: Model, d: Data):
  """Builds the implicit effective metric for the current position state."""
  if m.opt.integrator != IntegratorType.DISCRETE:
    return

  d.efm_c.zero_()
  d.efm_ca.zero_()
  d.efm_diag.zero_()
  d.efm_ts.zero_()
  d.efm_as.zero_()

  if m.nflex > 0 and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    passive.flex_hessian(m, d)
    if m.nflexrodedge > 0:
      passive.eval_rod_chains(m, d)

  if m.nefmK > 0:
    d.efm_K_val.zero_()

    det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
    wp.launch(
      _eff_flex_stretch_stiff_vert(det_atomics),
      dim=(d.nworld, m.nflexvert),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_dim,
        m.flex_interp,
        m.flex_stiffnessadr,
        m.flex_vertbodyid,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_rigid,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.flex_vertflexid,
        m.flex_simple,
        d.xmat,
        d.flexvert_hessian,
      ],
      outputs=[d.efm_K_val],
    )

    wp.launch(
      _eff_flex_stretch_stiff_edge(det_atomics),
      dim=(d.nworld, m.nflexedge, 2),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_stiffnessadr,
        m.flex_vertbodyid,
        m.flex_edge,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_rigid,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.flex_edgeflexid,
        m.flex_simple,
        d.xmat,
        d.flexedge_hessian,
      ],
      outputs=[d.efm_K_val],
    )

    wp.launch(
      _eff_flex_bend_stiff(det_atomics),
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_edgenum,
        m.flex_bendingadr,
        m.flex_vertbodyid,
        m.flex_edge,
        m.flex_edgeflap,
        m.flex_bending,
        m.flex_damping,
        m.flex_rigid,
        m.flex_centered,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.flex_edgeflexid,
        m.flex_simple,
        d.xquat,
        d.xmat,
      ],
      outputs=[d.efm_K_val],
    )

    wp.launch(
      _eff_flex_rod_stiff(det_atomics),
      dim=(d.nworld, m.nflexrodhinge),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_vertadr,
        m.flex_bendingadr,
        m.flex_vertbodyid,
        m.flex_bending,
        m.flex_damping,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.flex_simple,
        m.flex_rodhinge_map,
        m.flex_rod_is_chain,
        m.flex_rod_hingeadr,
        d.xmat,
        d.flexvert_xpos,
        d.flex_rod_chain_ok,
        d.flex_rodhinge_Wk,
        d.flex_rodhinge_Wd,
      ],
      outputs=[d.efm_K_val],
    )

    if m.flex_interp_assemblable:
      wp.launch(
        _eff_flex_interp_stiff(det_atomics),
        dim=(d.nworld, m.nflexintcell),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.body_dofnum,
          m.body_dofadr,
          m.flex_interp,
          m.flex_cellnum,
          m.flex_nodeadr,
          m.flex_stiffnessadr,
          m.flex_nodebodyid,
          m.flex_node0,
          m.flex_stiffness,
          m.flex_damping,
          m.flex_centered,
          m.efm_K_rownnz,
          m.efm_K_rowadr,
          m.efm_K_colind,
          m.flex_faceadr,
          m.flex_cell_map,
          m.flex_face,
          d.xmat,
          d.flexnode_xpos,
          d.face_quat,
        ],
        outputs=[d.efm_K_val],
      )


@event_scope
def eff_shift(m: Model, d: Data):
  """Refreshes velocity-stage terms and computes the smooth force shift c = -h*K*v."""
  if m.opt.integrator != IntegratorType.DISCRETE:
    return

  d.efm_diag.zero_()

  wp.launch(
    _eff_diag_stiff,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_plasticadr,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      d.qpos,
      d.plastic,
    ],
    outputs=[d.efm_diag],
  )

  # ktheta was cached by passive.eval_rod_chains, which passive() runs before this in fwd_velocity
  if m.nflexrodedge > 0 and not (m.opt.disableflags & DisableBit.SPRING):
    wp.launch(
      _eff_rod_ktheta_stiff,
      dim=(d.nworld, m.nflex),
      inputs=[
        m.opt.timestep,
        m.jnt_dofadr,
        m.flex_rigid,
        m.flex_rod_has_twist,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
        d.flex_rod_chain_ok,
        d.flex_rodedge_ktheta,
      ],
      outputs=[d.efm_diag],
    )

  wp.launch(
    _eff_diag_damp_shift,
    dim=(d.nworld, m.nv),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.jnt_actuatorid,
      m.dof_jntid,
      m.dof_damping,
      m.dof_dampingpoly,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_damping,
      m.actuator_dampingpoly,
      m.actuator_gear,
      d.qvel,
    ],
    outputs=[d.efm_c, d.efm_diag],
  )

  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  wp.launch(
    _eff_tendon_shift(m.max_ten_J_rownnz if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.tendon_actuatorid,
      m.tendon_plasticadr,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.tendon_stiffness,
      m.tendon_stiffnesspoly,
      m.tendon_damping,
      m.tendon_dampingpoly,
      m.tendon_lengthspring,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_damping,
      m.actuator_dampingpoly,
      m.actuator_gear,
      d.plastic,
      d.ten_J,
      d.ten_length,
      d.ten_velocity,
    ],
    outputs=[d.efm_c, d.efm_ts],
  )

  if not (m.opt.disableflags & DisableBit.SPRING):
    passive.flex_hessian(m, d)
    passive.flex_bend_mul(m, d, d.efm_c, d.qvel, s1_in=-1.0, s2_in=0.0, use_timestep=True, non_simple_only=False)
    passive.flex_rod_mul(m, d, d.efm_c, d.qvel, s1_in=-1.0, s2_in=0.0, use_timestep=True, non_simple_only=False)
    passive.flex_stretch_mul(m, d, d.efm_c, d.qvel, s1_in=-1.0, s2_in=0.0, use_timestep=True, non_simple_only=False)
    if m.nflexrodedge > 0:
      wp.launch(
        _eff_rod_twist_shift,
        dim=(d.nworld, m.nflex),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.jnt_dofadr,
          m.flex_bendingadr,
          m.flex_bending,
          m.flex_rigid,
          m.flex_rod_has_twist,
          m.flex_rod_edgeadr,
          m.flex_rod_edgenum,
          m.flex_rodedge_jnt,
          d.qvel,
        ],
        outputs=[d.efm_c],
      )

    wp.launch(
      _eff_flex_interp_mul(False, True, m.nv if det_atomics else 0, det_atomics),
      dim=(d.nworld, m.nflexintcell),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_parentid,
        m.body_rootid,
        m.body_dofnum,
        m.body_dofadr,
        m.body_simple,
        m.flex_interp,
        m.flex_cellnum,
        m.flex_nodeadr,
        m.flex_stiffnessadr,
        m.flex_nodebodyid,
        m.flex_node0,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_faceadr,
        m.flex_cell_map,
        m.flex_face,
        d.xmat,
        d.subtree_com,
        d.cdof,
        d.flexnode_xpos,
        d.face_quat,
        d.qvel,
        m.body_is_free,
      ],
      outputs=[d.efm_c],
    )

  if m.has_flex_passive and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = build_efm_contact(m, d)
    wp.launch(
      _eff_contact_shift(m.efm_con_max_dof if det_atomics else 0, det_atomics),
      dim=d.naconmax,
      inputs=[
        m.opt.timestep,
        d.qvel,
        d.contact.worldid,
        efm_con_dof,
        efm_con_val,
        efm_con_scale,
        efm_con_nnz,
      ],
      outputs=[d.efm_c],
    )

  d.efm_fluid.zero_()
  if m.has_fluid and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    Mi = m.M_fullm_i
    Mj = m.M_fullm_j
    wp.launch(
      _qderiv_ellipsoid_fluid(det_atomics),
      dim=(d.nworld, m.body_fluid_ellipsoid_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_geomnum,
        m.body_geomadr,
        m.dof_bodyid,
        m.geom_type,
        m.geom_size,
        m.geom_fluid,
        m.body_fluid_ellipsoid_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.geom_xpos,
        d.geom_xmat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[d.efm_fluid],
    )
    wp.launch(
      _qderiv_box_fluid(det_atomics),
      dim=(d.nworld, m.body_fluid_box_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_inertia,
        m.dof_bodyid,
        m.body_fluid_box_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.ximat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[d.efm_fluid],
    )


@event_scope
def eff_actuation(m: Model, d: Data):
  """Evaluates actuation-stage metric terms and factors the backbone qH."""
  if m.opt.integrator != IntegratorType.DISCRETE:
    return

  if m.nhistory > 0:
    ctrl = wp.empty((d.nworld, m.nu), dtype=float)
    history.read_ctrl_delayed(m, d, ctrl)
  else:
    ctrl = d.ctrl

  d.efm_ca.zero_()
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  wp.launch(
    _eff_actuator_actuation(m.nv if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.nactuator),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.jnt_dofadr,
      m.jnt_actfrclimited,
      m.jnt_actfrcrange,
      m.tendon_actfrclimited,
      m.tendon_actfrcrange,
      m.actuator_trntype,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_ctrladr,
      m.actuator_ctrlnum,
      m.actuator_ctrlspec,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_trnid,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_actlimited,
      m.actuator_actrange,
      m.actuator_actearly,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      m.actuator_ctrllimited,
      m.actuator_ctrlrange,
      m.actuator_acc0,
      m.actuator_lengthrange,
      d.act,
      ctrl,
      d.act_dot,
      d.actuator_length,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_velocity,
      d.actuator_force,
      d.qfrc_actuator,
    ],
    outputs=[d.efm_ca, d.efm_as],
  )

  wp.copy(d.qH, d.M)
  wp.launch(
    _eff_init_sdiag,
    dim=(d.nworld, m.nv),
    inputs=[
      m.M_rownnz,
      m.M_rowadr,
      d.efm_diag,
    ],
    outputs=[d.qH],
  )
  wp.launch(
    _eff_add_tendon_qH_diag(m.max_ten_J_rownnz if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.M_rownnz,
      m.M_rowadr,
      d.ten_J,
      d.efm_ts,
    ],
    outputs=[d.qH],
  )
  wp.launch(
    _eff_add_actuator_qH_diag(m.nv if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.nactuator),
    inputs=[
      m.M_rownnz,
      m.M_rowadr,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.efm_as,
    ],
    outputs=[d.qH],
  )

  if m.has_fluid and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    wp.launch(
      _eff_add_fluid_qH,
      dim=(d.nworld, m.nC),
      inputs=[d.efm_fluid],
      outputs=[d.qH],
    )

  if m.nflexrodedge > 0 and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    wp.launch(
      _eff_add_rod_twist_qH_diag,
      dim=(d.nworld, m.nflex),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.jnt_dofadr,
        m.flex_bendingadr,
        m.flex_bending,
        m.flex_damping,
        m.flex_rigid,
        m.M_rownnz,
        m.M_rowadr,
        m.flex_rod_has_twist,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
      ],
      outputs=[d.qH],
    )

  if smooth.skip_qacc_smooth(m, d):
    return

  # Factor qH into qHLD and qHDiagInv (adding efm_K diagonal into the factor only,
  # keeping d.qH = M + efm_sdiag so regularize_constraint and Newton Hessian assembly
  # do not double-count efm_K).
  if m.nefmK > 0:
    qH_eff = wp.clone(d.qH)
    wp.launch(
      _eff_add_efmK_qH_diag,
      dim=(d.nworld, m.nv),
      inputs=[
        m.body_parentid,
        m.body_weldid,
        m.body_jntnum,
        m.body_jntadr,
        m.body_dofnum,
        m.body_dofadr,
        m.body_simple,
        m.jnt_type,
        m.jnt_axis,
        m.dof_bodyid,
        m.dof_treeid,
        m.tree_dofnum,
        m.M_rownnz,
        m.M_rowadr,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        d.efm_K_val,
        qH_eff,
      ],
      outputs=[qH_eff],
    )
  else:
    qH_eff = d.qH
  if m.M_tiles:
    smooth._factor_blocks(m, d, qH_eff, d.qHLD, d.qHDiagInv)
  if d.qHLD.shape[1] > m.qLD_block_total:
    smooth._factor_i_sparse(m, d, qH_eff, d.qHLD[:, m.qLD_block_total :], d.qHDiagInv)


def eff_mul_m_scratch(m: Model, d: Data) -> tuple[Optional[tuple], Optional[tuple]]:
  """Builds the eff_mul_m efm_con and flex_scratch arguments.

  eff_mul_m allocates them on each call if they are not passed, which is not allowed inside
  wp.capture_while: build them before the loop instead.
  """
  efm_con = build_efm_contact(m, d) if m.has_flex_passive else None
  flex_scratch = passive.flex_stretch_scratch(m, d) if m.nflex and m.nflexvert else None
  return efm_con, flex_scratch


@event_scope
def eff_mul_m(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  skip: Optional[wp.array] = None,
  efm_con: Optional[tuple] = None,
  flex_scratch: Optional[tuple] = None,
):
  """Multiplies vector by effective metric M_hat = M + h*D + h^2*K.

  Inside wp.capture_while, where memory cannot be allocated, pass efm_con and flex_scratch
  from eff_mul_m_scratch, built before the loop.
  """
  support.mul_m(m, d, res, vec, skip=skip)
  if m.opt.integrator == IntegratorType.DISCRETE:
    check_skip = skip is not None
    skip_arr = skip if check_skip else m.body_is_free
    if m.nefmK > 0:
      wp.launch(
        _eff_mul_diag_and_csr(check_skip),
        dim=(d.nworld, m.nv),
        inputs=[
          m.efm_K_rownnz,
          m.efm_K_rowadr,
          m.efm_K_colind,
          d.efm_diag,
          d.efm_K_val,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    else:
      wp.launch(
        _eff_mul_diag(check_skip),
        dim=(d.nworld, m.nv),
        inputs=[d.efm_diag, vec, skip_arr],
        outputs=[res],
      )
    det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
    wp.launch(
      _eff_mul_add_tendon(check_skip, m.max_ten_J_rownnz if det_atomics else 0, det_atomics),
      dim=(d.nworld, m.ntendon),
      inputs=[
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        d.ten_J,
        d.efm_ts,
        vec,
        skip_arr,
      ],
      outputs=[res],
    )
    wp.launch(
      _eff_mul_add_actuator(check_skip, m.nv if det_atomics else 0, det_atomics),
      dim=(d.nworld, m.nactuator),
      inputs=[
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
        d.actuator_moment,
        d.efm_as,
        vec,
        skip_arr,
      ],
      outputs=[res],
    )
    if m.nflexrodedge > 0 and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
      wp.launch(
        _eff_mul_add_rod_twist(check_skip),
        dim=(d.nworld, m.nflex),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.jnt_dofadr,
          m.flex_bendingadr,
          m.flex_bending,
          m.flex_damping,
          m.flex_rigid,
          m.flex_rod_has_twist,
          m.flex_rod_edgeadr,
          m.flex_rod_edgenum,
          m.flex_rodedge_jnt,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    if m.has_flex_passive:
      if efm_con is None:
        efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = build_efm_contact(m, d)
      else:
        efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = efm_con
      wp.launch(
        _eff_mul_add_contact(check_skip, m.efm_con_max_dof if det_atomics else 0, det_atomics),
        dim=d.naconmax,
        inputs=[
          d.contact.worldid,
          efm_con_dof,
          efm_con_val,
          efm_con_scale,
          efm_con_nnz,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    if m.has_fluid and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
      wp.launch(
        _eff_mul_fluid(check_skip),
        dim=(d.nworld, m.nv),
        inputs=[
          m.M_mulm_rowadr,
          m.M_mulm_col,
          m.M_mulm_madr,
          d.efm_fluid,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    if (m.nefmK == 0 or m.has_non_simple_flex) and not (
      (m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)
    ):
      passive.flex_bend_mul(m, d, res, vec, s1_in=1.0, s2_in=1.0, use_timestep=True, non_simple_only=(m.nefmK > 0), skip=skip)
      passive.flex_rod_mul(m, d, res, vec, s1_in=1.0, s2_in=1.0, use_timestep=True, non_simple_only=(m.nefmK > 0), skip=skip)
      passive.flex_stretch_mul(
        m,
        d,
        res,
        vec,
        s1_in=1.0,
        s2_in=1.0,
        use_timestep=True,
        non_simple_only=(m.nefmK > 0),
        skip=skip,
        scratch=flex_scratch,
      )
    if not m.flex_interp_assemblable and not (
      (m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)
    ):
      wp.launch(
        _eff_flex_interp_mul(check_skip, False, m.nv if det_atomics else 0, det_atomics),
        dim=(d.nworld, m.nflexintcell),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.body_parentid,
          m.body_rootid,
          m.body_dofnum,
          m.body_dofadr,
          m.body_simple,
          m.flex_interp,
          m.flex_cellnum,
          m.flex_nodeadr,
          m.flex_stiffnessadr,
          m.flex_nodebodyid,
          m.flex_node0,
          m.flex_stiffness,
          m.flex_damping,
          m.flex_faceadr,
          m.flex_cell_map,
          m.flex_face,
          d.xmat,
          d.subtree_com,
          d.cdof,
          d.flexnode_xpos,
          d.face_quat,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        _zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[res],
      )


@cache_kernel
def _eff_fold_efc_sparse(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    # Out:
    tw_diag_out: wp.array2d[float],
  ):
    worldid, r = wp.tid()
    if r >= nefc_in[worldid]:
      return
    D = efc_D_in[worldid, r]
    if D == 0.0:
      return
    adr = efc_J_rowadr_in[worldid, r]
    nnz = efc_J_rownnz_in[worldid, r]
    for a in range(nnz):
      ia = efc_J_colind_in[worldid, 0, adr + a]
      ja = efc_J_in[worldid, 0, adr + a]
      wp.atomic_add(tw_diag_out, worldid, ia, D * ja * ja)

  return kernel


@wp.kernel
def _eff_fold_tw_diag_efc_dense(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  # Out:
  tw_diag_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  nefc = wp.min(nefc_in[worldid], efc_D_in.shape[1])
  acc = float(0.0)
  for r in range(nefc):
    D = efc_D_in[worldid, r]
    if D != 0.0:
      j = efc_J_in[worldid, r, dofid]
      if j != 0.0:
        acc += D * j * j
  tw_diag_out[worldid, dofid] += acc


@cache_kernel
def _eff_fold_contact_rank1(max_dof: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_dof

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Data in:
    contact_worldid_in: wp.array[int],
    nacon_in: wp.array[int],
    # In:
    efm_con_dof_in: wp.array2d[int],
    efm_con_val_in: wp.array2d[float],
    efm_con_scale_in: wp.array[float],
    efm_con_nnz_in: wp.array[int],
    # Out:
    tw_diag_out: wp.array2d[float],
  ):
    cid = wp.tid()
    if cid >= nacon_in[0]:
      return
    scale = efm_con_scale_in[cid]
    if scale == 0.0:
      return
    worldid = contact_worldid_in[cid]
    nnz = efm_con_nnz_in[cid]
    for a in range(nnz):
      ia = efm_con_dof_in[cid, a]
      va = efm_con_val_in[cid, a]
      wp.atomic_add(tw_diag_out, worldid, ia, scale * va * va)

  return kernel


@wp.kernel
def _eff_zero_covered_dofs(
  # Model:
  efm_dofid: wp.array[int],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  i = efm_dofid[k]
  vec_out[worldid, i] = 0.0
  vec_out[worldid, i + 1] = 0.0
  vec_out[worldid, i + 2] = 0.0


@wp.kernel
def _eff_invert_covered_fold_diag(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  efm_dofid: wp.array[int],
  # Data in:
  qH_in: wp.array2d[float],
  efm_K_val_in: wp.array2d[float],
  # In:
  tw_diag_in: wp.array2d[float],
  # Out:
  tw_diag_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  i = efm_dofid[k]
  for r in range(3):
    row = i + r
    diag = qH_in[worldid, M_rowadr[row] + M_rownnz[row] - 1] + tw_diag_in[worldid, row]
    rownnz = efm_K_rownnz[row]
    if rownnz > 0:
      idx = _find_csr_col(efm_K_rowadr[row], rownnz, row, efm_K_colind)
      if idx >= 0:
        diag += efm_K_val_in[worldid, idx]
    tw_diag_out[worldid, row] = 1.0 / wp.max(MJ_MINVAL, diag)


@wp.kernel
def _eff_apply_covered_fold_diag(
  # Model:
  efm_dofid: wp.array[int],
  # In:
  vec_in: wp.array2d[float],
  tw_diag_in: wp.array2d[float],
  # Out:
  res_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  i = efm_dofid[k]
  res_out[worldid, i] = vec_in[worldid, i] * tw_diag_in[worldid, i]
  res_out[worldid, i + 1] = vec_in[worldid, i + 1] * tw_diag_in[worldid, i + 1]
  res_out[worldid, i + 2] = vec_in[worldid, i + 2] * tw_diag_in[worldid, i + 2]


@event_scope
def eff_prec_fold(
  m: Model,
  d: Data,
  tw_diag: Optional[wp.array] = None,
):
  """Folds active/inactive efc and passive-contact diagonals into tw_diag for covered flex triples and 1D rod twist."""
  if tw_diag is None:
    return
  tw_diag.zero_()
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)

  if m.is_sparse:
    wp.launch(
      _eff_fold_efc_sparse(m.nv if det_atomics else 0, det_atomics),
      dim=(d.nworld, d.njmax),
      inputs=[
        d.nefc,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
      ],
      outputs=[tw_diag],
    )
  else:
    wp.launch(
      _eff_fold_tw_diag_efc_dense,
      dim=(d.nworld, m.nv),
      inputs=[d.nefc, d.efc.J, d.efc.D],
      outputs=[tw_diag],
    )

  if m.has_flex_passive and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = build_efm_contact(m, d)
    wp.launch(
      _eff_fold_contact_rank1(m.efm_con_max_dof if det_atomics else 0, det_atomics),
      dim=d.naconmax,
      inputs=[
        d.contact.worldid,
        d.nacon,
        efm_con_dof,
        efm_con_val,
        efm_con_scale,
        efm_con_nnz,
      ],
      outputs=[tw_diag],
    )

  if m.nefmdof > 0:
    wp.launch(
      _eff_invert_covered_fold_diag,
      dim=(d.nworld, m.nefmdof),
      inputs=[
        m.M_rownnz,
        m.M_rowadr,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.efm_dofid,
        d.qH,
        d.efm_K_val,
        tw_diag,
      ],
      outputs=[tw_diag],
    )


@event_scope
def eff_prec(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  x_scratch: Optional[wp.array] = None,
  tw_diag: Optional[wp.array] = None,
):
  """Preconditions residual vector using backbone qHLD (plus folded diagonal, efm0_L, or 1D rod twist solve)."""
  has_rod_twist = m.nflexrodedge > 0 and not (
    (m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)
  )
  has_tw_diag = tw_diag is not None
  tw_in = tw_diag if has_tw_diag else d.qH
  if m.efm0_active:
    if m.nefm0dof < m.nv:
      wp.copy(res, vec)
      wp.launch(
        _eff_zero_efm0_dofs,
        dim=(d.nworld, m.nefm0dof),
        inputs=[m.efm0_dofid],
        outputs=[res],
      )
      smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, res, res, x_scratch=x_scratch)
      if has_rod_twist:
        wp.launch(
          _eff_twist_solve(has_tw_diag),
          dim=(d.nworld, m.nflex),
          inputs=[
            m.opt.timestep,
            m.opt.disableflags,
            m.jnt_dofadr,
            m.dof_treeid,
            m.tree_dofnum,
            m.flex_bendingadr,
            m.flex_bending,
            m.flex_damping,
            m.flex_rigid,
            m.M_rownnz,
            m.M_rowadr,
            m.flex_rod_has_twist,
            m.flex_rod_edgeadr,
            m.flex_rod_edgenum,
            m.flex_rodedge_jnt,
            d.qH,
            vec,
            tw_in,
          ],
          outputs=[d.flex_rodedge_tridiag, res],
        )
    wp.launch(
      _eff_prec_efm0_solve,
      dim=d.nworld,
      inputs=[
        m.nefm0dof,
        m.efm0_dofid,
        m.efm0_L_rownnz,
        m.efm0_L_rowadr,
        m.efm0_L_colind,
        m.efm0_L,
        vec,
      ],
      outputs=[res],
    )
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        _zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[res],
      )
    return

  if has_tw_diag and m.nefmdof > 0:
    if 3 * m.nefmdof == m.nv and not has_rod_twist:
      wp.launch(
        _eff_apply_covered_fold_diag,
        dim=(d.nworld, m.nefmdof),
        inputs=[m.efm_dofid, vec, tw_diag],
        outputs=[res],
      )
      if m.opt.enableflags & EnableBit.SLEEP:
        wp.launch(
          _zero_sleeping_dofs,
          dim=(d.nworld, m.nv),
          inputs=[m.dof_treeid, d.tree_awake],
          outputs=[res],
        )
      return

    rhs = wp.clone(vec) if res.ptr == vec.ptr else vec
    if res.ptr != rhs.ptr:
      wp.copy(res, rhs)
    wp.launch(
      _eff_zero_covered_dofs,
      dim=(d.nworld, m.nefmdof),
      inputs=[m.efm_dofid],
      outputs=[res],
    )
    smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, res, res, x_scratch=x_scratch)
    if has_rod_twist:
      wp.launch(
        _eff_twist_solve(has_tw_diag),
        dim=(d.nworld, m.nflex),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.jnt_dofadr,
          m.dof_treeid,
          m.tree_dofnum,
          m.flex_bendingadr,
          m.flex_bending,
          m.flex_damping,
          m.flex_rigid,
          m.M_rownnz,
          m.M_rowadr,
          m.flex_rod_has_twist,
          m.flex_rod_edgeadr,
          m.flex_rod_edgenum,
          m.flex_rodedge_jnt,
          d.qH,
          rhs,
          tw_in,
        ],
        outputs=[d.flex_rodedge_tridiag, res],
      )
    wp.launch(
      _eff_apply_covered_fold_diag,
      dim=(d.nworld, m.nefmdof),
      inputs=[m.efm_dofid, rhs, tw_diag],
      outputs=[res],
    )
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        _zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[res],
      )
    return

  smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, res, vec, x_scratch=x_scratch)
  if has_rod_twist:
    wp.launch(
      _eff_twist_solve(has_tw_diag),
      dim=(d.nworld, m.nflex),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.jnt_dofadr,
        m.dof_treeid,
        m.tree_dofnum,
        m.flex_bendingadr,
        m.flex_bending,
        m.flex_damping,
        m.flex_rigid,
        m.M_rownnz,
        m.M_rowadr,
        m.flex_rod_has_twist,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
        d.qH,
        vec,
        tw_in,
      ],
      outputs=[d.flex_rodedge_tridiag, res],
    )
  if m.opt.enableflags & EnableBit.SLEEP:
    wp.launch(
      _zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[res],
    )


@event_scope
def eff_solve(m: Model, d: Data, qacc: wp.array2d[float], qfrc: Optional[wp.array] = None):
  """Solves the effective metric system M_hat * qacc = qfrc_eff."""
  efm_r = wp.empty((d.nworld, m.nv), dtype=float)
  efm_z = wp.empty((d.nworld, m.nv), dtype=float)
  efm_p = wp.empty((d.nworld, m.nv), dtype=float)
  efm_Ap = wp.empty((d.nworld, m.nv), dtype=float)
  efm_rz = wp.empty(d.nworld, dtype=float)
  efm_bn = wp.empty(d.nworld, dtype=float)
  efm_done = wp.empty(d.nworld, dtype=bool)
  efm_nsolving = wp.empty(1, dtype=int)
  efm_iter = wp.empty(1, dtype=int)

  if qfrc is None:
    rhs = efm_r
    wp.launch(
      _eff_rhs,
      dim=(d.nworld, m.nv),
      inputs=[
        m.opt.enableflags,
        m.dof_treeid,
        d.tree_awake,
        d.qfrc_smooth,
        d.efm_c,
        d.efm_ca,
      ],
      outputs=[rhs],
    )
  else:
    rhs = qfrc

  has_spring_or_damper = not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER))
  has_efm_tendon = (m.has_tendon_stiffness and not (m.opt.disableflags & DisableBit.SPRING)) or (
    m.has_tendon_damping and not (m.opt.disableflags & DisableBit.DAMPER)
  )
  has_efm_actuator = m.has_efm_actuator and not (m.opt.disableflags & DisableBit.ACTUATION)
  has_flex_any = has_spring_or_damper and (
    m.nefmK > 0
    or m.nflexrodhinge > 0
    or m.has_non_simple_flex
    or m.efm0_active
    or m.has_flex_passive
    or not m.flex_interp_assemblable
  )
  if not has_flex_any and not has_efm_tendon and not has_efm_actuator:
    smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, qacc, rhs)
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        _zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[qacc],
      )
    return

  efm_nsolving.fill_(d.nworld)
  efm_iter.zero_()
  # allocated here since _pcg_iteration may run inside wp.capture_while
  efm_con, flex_scratch = eff_mul_m_scratch(m, d)
  solve_ld_scratch = (
    wp.empty((d.nworld, m.nv), dtype=float)
    if (m.opt.deterministic & DeterminismType.ATOMICS) and d.qHLD.shape[1] > m.qLD_block_total
    else None
  )

  eff_prec(m, d, efm_z, rhs, x_scratch=solve_ld_scratch)
  wp.launch_tiled(
    _pcg_init_tiled,
    dim=d.nworld,
    inputs=[
      m.nv,
      m.nactuator,
      m.ntendon,
      m.opt.enableflags,
      m.dof_treeid,
      d.tree_awake,
      d.efm_ts,
      d.efm_as,
      rhs,
      efm_z,
      has_flex_any,
      has_efm_tendon,
      has_efm_actuator,
    ],
    outputs=[qacc, efm_r, efm_p, efm_rz, efm_bn, efm_done, efm_nsolving],
    block_dim=m.block_dim.eff_pcg,
  )

  max_it = m.opt.iterations

  def _pcg_iteration():
    eff_mul_m(m, d, efm_Ap, efm_p, skip=efm_done, efm_con=efm_con, flex_scratch=flex_scratch)
    wp.launch_tiled(
      _pcg_pAp_step_and_check_tiled,
      dim=d.nworld,
      inputs=[
        m.nv,
        m.opt.tolerance,
        efm_p,
        efm_Ap,
        efm_rz,
        efm_bn,
        efm_done,
      ],
      outputs=[
        qacc,
        efm_r,
        efm_done,
        efm_nsolving,
      ],
      block_dim=m.block_dim.eff_pcg,
    )
    eff_prec(m, d, efm_z, efm_r, x_scratch=solve_ld_scratch)
    wp.launch_tiled(
      _pcg_beta_and_p_update_tiled,
      dim=d.nworld,
      inputs=[
        m.nv,
        efm_r,
        efm_z,
        efm_rz,
        efm_done,
        max_it,
      ],
      outputs=[
        efm_p,
        efm_rz,
        efm_nsolving,
        efm_iter,
      ],
      block_dim=m.block_dim.eff_pcg,
    )

  if max_it > 0:
    if m.opt.graph_conditional and wp.get_device().is_cuda:
      wp.capture_while(efm_nsolving, while_body=_pcg_iteration)
    else:
      for _ in range(max_it):
        _pcg_iteration()
