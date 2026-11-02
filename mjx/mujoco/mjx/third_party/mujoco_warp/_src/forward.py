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

from typing import Optional, Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import collision_driver
from mujoco.mjx.third_party.mujoco_warp._src import constraint
from mujoco.mjx.third_party.mujoco_warp._src import derivative
from mujoco.mjx.third_party.mujoco_warp._src import history
from mujoco.mjx.third_party.mujoco_warp._src import island
from mujoco.mjx.third_party.mujoco_warp._src import math
from mujoco.mjx.third_party.mujoco_warp._src import passive
from mujoco.mjx.third_party.mujoco_warp._src import sensor
from mujoco.mjx.third_party.mujoco_warp._src import sleep
from mujoco.mjx.third_party.mujoco_warp._src import smooth
from mujoco.mjx.third_party.mujoco_warp._src import solver
from mujoco.mjx.third_party.mujoco_warp._src import util_misc
from mujoco.mjx.third_party.mujoco_warp._src.support import mat33_to_quat_polar
from mujoco.mjx.third_party.mujoco_warp._src.support import next_act
from mujoco.mjx.third_party.mujoco_warp._src.support import xfrc_accumulate
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
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
from mujoco.mjx.third_party.mujoco_warp._src.types import ObjType
from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
from mujoco.mjx.third_party.mujoco_warp._src.types import SleepState
from mujoco.mjx.third_party.mujoco_warp._src.types import SolverType
from mujoco.mjx.third_party.mujoco_warp._src.types import TrnType
from mujoco.mjx.third_party.mujoco_warp._src.types import WrapType
from mujoco.mjx.third_party.mujoco_warp._src.types import mat66
from mujoco.mjx.third_party.mujoco_warp._src.types import vec6
from mujoco.mjx.third_party.mujoco_warp._src.types import vec10
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


def check_discrete(m: Model):
  """Validate model configuration for discrete integrator."""
  if m.opt.integrator != IntegratorType.DISCRETE:
    if m.has_flex_snh:
      raise ValueError("stable Neo-Hookean elasticity requires integrator='discrete'")
    if m.has_flex_passive:
      raise ValueError("passive flex contact requires an integrator with the effective metric: set integrator='discrete'")
    if m.opt.integrator in (IntegratorType.IMPLICIT, IntegratorType.IMPLICITFAST) and (m.nefmK > 0 or m.has_non_simple_flex):
      raise ValueError(
        "flex elasticity is no longer integrated implicitly under integrator='implicit' and 'implicitfast': "
        "set integrator='discrete', or 'Euler' with a small timestep"
      )
    return
  if m.has_flex_edge_stiff_damp:
    raise ValueError(
      "discrete integrator: flex edge stiffness and damping are not supported; use <elasticity> or <edge equality> instead"
    )
  if m.has_unsupported_flex_interp:
    raise NotImplementedError("Discrete integrator: quadratic (abs(flex_interp) >= 2) interpolated flexes are not supported")
  if m.opt.solver == SolverType.NEWTON and m.has_non_simple_flex:
    raise ValueError("discrete integrator: flex with general attachments requires solver='CG'")
  if m.opt.solver == SolverType.NEWTON and not m.flex_interp_assemblable:
    raise ValueError(
      "Discrete integrator: Newton solver requires interpolated flex nodes to be on 3-DOF slider bodies; use solver='CG'"
    )
  if m.opt.enableflags & EnableBit.SLEEP:
    if m.opt.disableflags & DisableBit.ISLAND:
      raise ValueError("Discrete integrator: sleep without islands not yet supported")
    if m.nefmK > 0 or m.has_non_simple_flex or m.has_flex_passive or not m.flex_interp_assemblable:
      raise ValueError("Discrete integrator: sleep with flex not yet supported")


@wp.kernel
def _next_position(
  # Model:
  opt_timestep: wp.array[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  qvel_scale_in: float,
  # Data out:
  qpos_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  jnttype = jnt_type[jntid]
  qpos_adr = jnt_qposadr[jntid]
  dof_adr = jnt_dofadr[jntid]
  qpos = qpos_in[worldid]
  qpos_next = qpos_out[worldid]
  qvel = qvel_in[worldid]

  if jnttype == JointType.FREE:
    qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
    qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_new = qpos_pos + timestep * qvel_lin

    qpos_quat = wp.quat(
      qpos[qpos_adr + 3],
      qpos[qpos_adr + 4],
      qpos[qpos_adr + 5],
      qpos[qpos_adr + 6],
    )
    qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_new[0]
    qpos_next[qpos_adr + 1] = qpos_new[1]
    qpos_next[qpos_adr + 2] = qpos_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[0]
    qpos_next[qpos_adr + 4] = qpos_quat_new[1]
    qpos_next[qpos_adr + 5] = qpos_quat_new[2]
    qpos_next[qpos_adr + 6] = qpos_quat_new[3]

  elif jnttype == JointType.BALL:
    qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
    qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_quat_new[0]
    qpos_next[qpos_adr + 1] = qpos_quat_new[1]
    qpos_next[qpos_adr + 2] = qpos_quat_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[3]

  else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
    qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in


@wp.kernel
def _next_velocity(
  # Model:
  opt_timestep: wp.array[float],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # In:
  qacc_scale_in: float,
  # Data out:
  qvel_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep


@wp.func
def _wrap_period(
  # Model:
  jnt_type: wp.array[int],
  # In:
  trntype: int,
  dyntype: int,
  gaintype: int,
  biastype: int,
  trnid: wp.vec2i,
  gainprm: vec10,
  biasprm: vec10,
  gear: wp.spatial_vector,
) -> float:
  # servo shape: fixed gain, affine bias, matching kp, setpoint input
  is_servo = (
    gaintype == GainType.FIXED
    and biastype == BiasType.AFFINE
    and gainprm[0] == -biasprm[1]
    and (
      dyntype == DynType.NONE
      or dyntype == DynType.INTEGRATOR
      or dyntype == DynType.FILTER
      or dyntype == DynType.FILTEREXACT
    )
  )
  if not is_servo:
    return 0.0

  # site transmission with refsite and purely rotational gear
  if trntype == TrnType.SITE and trnid[1] >= 0 and gear[0] == 0.0 and gear[1] == 0.0 and gear[2] == 0.0:
    rot_gear = wp.vec3(gear[3], gear[4], gear[5])
    return wp.static(2.0 * wp.pi) * wp.length(rot_gear)

  # joint transmission on a ball joint
  if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:
    if jnt_type[trnid[0]] == JointType.BALL:
      lin_gear = wp.vec3(gear[0], gear[1], gear[2])
      return wp.static(2.0 * wp.pi) * wp.length(lin_gear)

  return 0.0


@wp.kernel
def _next_activation(
  # Model:
  opt_timestep: wp.array[float],
  jnt_type: wp.array[int],
  actuator_trntype: wp.array[int],
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # Data in:
  act_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  # In:
  act_dot_scale: float,
  limit: bool,
  # Data out:
  act_out: wp.array2d[float],
):
  worldid, uid = wp.tid()
  opt_timestep_id = worldid % opt_timestep.shape[0]
  actuator_dynprm_id = worldid % actuator_dynprm.shape[0]
  actuator_actrange_id = worldid % actuator_actrange.shape[0]
  actuator_gainprm_id = worldid % actuator_gainprm.shape[0]
  actuator_biasprm_id = worldid % actuator_biasprm.shape[0]

  actadr = actuator_actadr[uid]
  actnum = actuator_actnum[uid]
  dyntype = actuator_dyntype[uid]

  if dyntype == DynType.DCMOTOR:
    dynprm = actuator_dynprm[actuator_dynprm_id, uid]
    gainprm = actuator_gainprm[actuator_gainprm_id, uid]
    biasprm = actuator_biasprm[actuator_biasprm_id, uid]
    slots = util_misc.dcmotor_slots(dynprm, gainprm)

    for j in range(actadr, actadr + actnum):
      offset = j - actadr
      act = act_in[worldid, j]
      act_dot = act_dot_in[worldid, j]

      if offset == slots[4]:  # current
        R = gainprm[0]
        te = wp.max(MJ_MINVAL, dynprm[0])
        act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te))
      elif offset == slots[3]:  # bristle
        F_C = biasprm[3]
        F_S = biasprm[4]
        v_S = biasprm[5]
        sigma0 = dynprm[5]
        velocity = actuator_velocity_in[worldid, uid]
        g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)

        a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)
        h = opt_timestep[opt_timestep_id]
        exp_ah = wp.exp(a * h)
        int_h = h
        if wp.abs(a) > MJ_MINVAL:
          int_h = (exp_ah - 1.0) / a
        act = exp_ah * act + int_h * velocity
      elif offset == slots[1]:  # integral
        act = act + act_dot * opt_timestep[opt_timestep_id]
        Imax = dynprm[8]
        if Imax > 0.0:
          act = wp.clamp(act, -Imax, Imax)
      else:  # temperature and slew
        act = act + act_dot * opt_timestep[opt_timestep_id]

      act_out[worldid, j] = act
  else:
    for j in range(actadr, actadr + actnum):
      act = next_act(
        opt_timestep[opt_timestep_id],
        dyntype,
        actuator_dynprm[actuator_dynprm_id, uid],
        actuator_actrange[actuator_actrange_id, uid],
        act_in[worldid, j],
        act_dot_in[worldid, j],
        act_dot_scale,
        limit and actuator_actlimited[uid],
      )
      if limit and (dyntype == DynType.INTEGRATOR or dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT):
        gaintype = actuator_gaintype[uid]
        biastype = actuator_biastype[uid]
        gainprm = actuator_gainprm[actuator_gainprm_id, uid]
        biasprm = actuator_biasprm[actuator_biasprm_id, uid]
        trntype = actuator_trntype[uid]
        trnid = actuator_trnid[uid]
        gear = actuator_gear[worldid % actuator_gear.shape[0], uid]
        period = _wrap_period(
          jnt_type,
          trntype,
          dyntype,
          gaintype,
          biastype,
          trnid,
          gainprm,
          biasprm,
          gear,
        )
        if period > 0.0:
          length = actuator_length_in[worldid, uid]
          err = act - length
          act -= period * wp.round(err / period)

      act_out[worldid, j] = act


@cache_kernel
def _next_time_builder(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def _next_time(
    # Model:
    opt_timestep: wp.array[float],
    is_sparse: bool,
    # Data in:
    nefc_in: wp.array[int],
    time_in: wp.array[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    nworld_in: int,
    naconmax_in: int,
    njmax_in: int,
    njmax_nnz_in: int,
    nacon_in: wp.array[int],
    ncollision_in: wp.array[int],
    # Data out:
    time_out: wp.array[float],
    overflow_out: wp.array[int],
  ):
    worldid = wp.tid()
    time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]
    nefc = nefc_in[worldid]

    if nefc > njmax_in:
      if wp.static(bool(warn_overflow & OverflowType.NEFC)):
        wp.printf(
          "nefc overflow - please increase njmax beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NEFC (or = 0 for all)\n",
          njmax_in,
        )
      overflow_out[worldid] = overflow_out[worldid] | OverflowType.NEFC
    elif nefc > 0 and is_sparse:
      efcid = wp.min(nefc, njmax_in) - 1
      efc_nnz = efc_J_rowadr_in[worldid, efcid] + efc_J_rownnz_in[worldid, efcid]
      if efc_nnz > njmax_nnz_in:
        if wp.static(bool(warn_overflow & OverflowType.NJMAX_NNZ)):
          wp.printf(
            "njmax_nnz overflow - please increase njmax_nnz beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NJMAX_NNZ (or = 0 for all)\n",
            njmax_nnz_in,
          )
        overflow_out[worldid] = overflow_out[worldid] | OverflowType.NJMAX_NNZ

    ncollision = ncollision_in[0]
    if ncollision > naconmax_in:
      if worldid == 0 and wp.static(bool(warn_overflow & OverflowType.BROADPHASE)):
        wp.printf(
          "broadphase overflow - please increase nconmax beyond %u or naconmax beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.BROADPHASE (or = 0 for all)\n",
          naconmax_in // nworld_in,
          naconmax_in,
        )
      overflow_out[worldid] = overflow_out[worldid] | OverflowType.BROADPHASE

    nacon = nacon_in[0]
    if nacon > naconmax_in:
      if worldid == 0 and wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)):
        wp.printf(
          "narrowphase overflow - please increase nconmax beyond %u or naconmax beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",
          naconmax_in // nworld_in,
          naconmax_in,
        )
      overflow_out[worldid] = overflow_out[worldid] | OverflowType.NARROWPHASE

  return _next_time


@wp.kernel
def _next_rod_twist(
  # Model:
  qpos0: wp.array2d[float],
  body_parentid: wp.array[int],
  body_pos: wp.array2d[wp.vec3],
  body_quat: wp.array2d[wp.quat],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_bendingadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_vert: wp.array[wp.vec3],
  flex_bending: wp.array[float],
  # Data in:
  qpos_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  plastic_out: wp.array2d[float],
):
  """Carries the twist reference of 1D rods to the configuration of this step, see mjROD_STATE_SIZE."""
  worldid, flexid = wp.tid()
  padr = flex_plasticadr[flexid]
  bendingadr = flex_bendingadr[flexid]
  if flex_dim[flexid] != 1 or padr < 0 or bendingadr < 0:
    return
  ok, _g, _Wk, _Wd, _f0, _f1, _f2, _tb0, _tdof, _tfrc, state_t, state_a, state_psi = passive._rod_chain_hinge(
    qpos0,
    body_parentid,
    body_pos,
    body_quat,
    jnt_qposadr,
    jnt_dofadr,
    flex_vertadr,
    flex_bendingadr,
    flex_plasticadr,
    flex_vertbodyid,
    flex_vert,
    flex_bending,
    qpos_in,
    plastic_in,
    xpos_in,
    xquat_in,
    xmat_in,
    flexvert_xpos_in,
    worldid,
    flexid,
    bendingadr + 1,
    False,
    False,
  )
  if not ok:
    return
  for i in range(3):
    plastic_out[worldid, padr + i] = state_t[i]
    plastic_out[worldid, padr + 3 + i] = state_a[i]
  plastic_out[worldid, padr + 6] = state_psi


@wp.kernel
def _next_plastic_joint(
  # Model:
  opt_enableflags: int,
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  jnt_plasticadr: wp.array[int],
  jnt_springyield: wp.array2d[float],
  # Data in:
  qpos_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  body_awake_in: wp.array2d[int],
  # Data out:
  plastic_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  padr = jnt_plasticadr[jntid]
  if padr < 0:
    return
  if (opt_enableflags & EnableBit.SLEEP) and body_awake_in[worldid, jnt_bodyid[jntid]] != SleepState.AWAKE:
    return
  yield_ = wp.max(0.0, jnt_springyield[worldid % jnt_springyield.shape[0], jntid])

  qadr = jnt_qposadr[jntid]
  jnttype = jnt_type[jntid]

  if jnttype == JointType.SLIDE or jnttype == JointType.HINGE:
    x = qpos_in[worldid, qadr] - plastic_in[worldid, padr]
    if x > yield_:
      plastic_out[worldid, padr] = qpos_in[worldid, qadr] - yield_
    elif x < -yield_:
      plastic_out[worldid, padr] = qpos_in[worldid, qadr] + yield_
  elif jnttype == JointType.BALL:
    quat = wp.normalize(
      wp.quat(
        qpos_in[worldid, qadr + 0],
        qpos_in[worldid, qadr + 1],
        qpos_in[worldid, qadr + 2],
        qpos_in[worldid, qadr + 3],
      )
    )
    qspring = wp.quat(
      plastic_in[worldid, padr + 0],
      plastic_in[worldid, padr + 1],
      plastic_in[worldid, padr + 2],
      plastic_in[worldid, padr + 3],
    )
    dif_rot = math.quat_sub(quat, qspring)
    r_rot = wp.length(dif_rot)
    if r_rot > yield_:
      qspring_new = math.quat_integrate(qspring, dif_rot, (r_rot - yield_) / r_rot)
      plastic_out[worldid, padr + 0] = qspring_new[0]
      plastic_out[worldid, padr + 1] = qspring_new[1]
      plastic_out[worldid, padr + 2] = qspring_new[2]
      plastic_out[worldid, padr + 3] = qspring_new[3]


@wp.kernel
def _next_plastic_tendon(
  # Model:
  opt_timestep: wp.array[float],
  opt_enableflags: int,
  jnt_qposadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  dof_treeid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  tendon_plasticadr: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_springyield: wp.array2d[float],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  wrap_prm: wp.array[float],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  ten_J_in: wp.array2d[float],
  ten_length_in: wp.array2d[float],
  body_awake_in: wp.array2d[int],
  tree_awake_in: wp.array2d[int],
  # Data out:
  plastic_out: wp.array2d[float],
):
  worldid, tenid = wp.tid()
  padr = tendon_plasticadr[tenid]
  if padr < 0:
    return
  yield_ = wp.max(0.0, tendon_springyield[worldid % tendon_springyield.shape[0], tenid])

  adr = tendon_adr[tenid]
  L_next = float(0.0)
  if wrap_type[adr] == WrapType.JOINT:
    num = tendon_num[tenid]
    if opt_enableflags & EnableBit.SLEEP:
      awake = bool(False)
      for j in range(num):
        if body_awake_in[worldid, jnt_bodyid[wrap_objid[adr + j]]] == SleepState.AWAKE:
          awake = True
      if not awake:
        return
    for j in range(num):
      L_next += wrap_prm[adr + j] * qpos_in[worldid, jnt_qposadr[wrap_objid[adr + j]]]
  else:
    rownnz = ten_J_rownnz[tenid]
    rowadr = ten_J_rowadr[tenid]
    if opt_enableflags & EnableBit.SLEEP:
      awake = bool(False)
      for k in range(rownnz):
        t = dof_treeid[ten_J_colind[rowadr + k]]
        if t >= 0 and tree_awake_in[worldid, t] != 0:
          awake = True
      if not awake:
        return
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    dL_dt = float(0.0)
    for k in range(rownnz):
      sparseid = rowadr + k
      dL_dt += ten_J_in[worldid, sparseid] * qvel_in[worldid, ten_J_colind[sparseid]]
    L_next = ten_length_in[worldid, tenid] + timestep * dL_dt

  lower = plastic_in[worldid, padr + 0]
  upper = plastic_in[worldid, padr + 1]
  x = wp.where(L_next > upper, L_next - upper, wp.where(L_next < lower, L_next - lower, 0.0))
  if x > yield_:
    delta = x - yield_
    if lower != 0.0 or upper <= 0.0:
      plastic_out[worldid, padr + 0] = lower + delta
    plastic_out[worldid, padr + 1] = upper + delta
  elif x < -yield_:
    delta = x + yield_
    plastic_out[worldid, padr + 0] = lower + delta
    plastic_out[worldid, padr + 1] = upper + delta


@wp.func
def _face_trpk(g: vec6, metric: passive.mat66, a: int, b: int, c: int) -> float:
  dab = g[a] + g[b] - g[c]
  dac = g[a] + g[c] - g[b]
  dbc = g[b] + g[c] - g[a]
  return 0.5 * (dab * dab * metric[a, b] + dac * dac * metric[a, c] + dbc * dbc * metric[b, c])


@wp.func
def _gram_det_3d(q: vec6) -> Tuple[float, vec6]:
  a = q[0]
  b = q[2]
  c = q[4]
  d = 0.5 * (q[0] + q[2] - q[1])
  e = 0.5 * (q[0] + q[4] - q[5])
  f = 0.5 * (q[2] + q[4] - q[3])
  c0 = b * c - f * f
  c1 = a * c - e * e
  c2 = a * b - d * d
  c3 = e * f - c * d
  c4 = d * f - b * e
  c5 = d * e - a * f
  grad = vec6(c0 + c3 + c4, -c3, c1 + c3 + c5, -c5, c2 + c4 + c5, -c4)
  return a * c0 + d * c3 + e * c4, grad


@wp.kernel
def _next_plastic_flex_stretch(
  # Model:
  opt_timestep: wp.array[float],
  opt_enableflags: int,
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  dof_parentid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_edgeadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_elemedgeadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_elemedge: wp.array[int],
  flexedge_length0: wp.array[float],
  flex_stiffness: wp.array[float],
  flex_yield: wp.array[float],
  flex_elemflexid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  body_awake_in: wp.array2d[int],
  # Data out:
  plastic_out: wp.array2d[float],
  flex_hessian_valid_out: wp.array2d[bool],
):
  worldid, elemid = wp.tid()
  f = flex_elemflexid[elemid]
  padr_base = flex_plasticadr[f]
  if padr_base < 0:
    return
  yield_ = wp.max(0.0, flex_yield[f])
  dim = flex_dim[f]
  if (dim != 2 and dim != 3) or flex_interp[f] != 0:
    return
  stiffness_adr_base = flex_stiffnessadr[f]
  if stiffness_adr_base < 0 or flex_stiffness[stiffness_adr_base] == 0.0:
    return

  local_elemid = elemid - flex_elemadr[f]
  nedge = 3 if dim == 2 else 6
  stride = 21 if dim == 2 else 24
  padr = padr_base + local_elemid * nedge
  vbase = flex_vertadr[f]
  ebase = flex_edgeadr[f]
  elem_data_adr = flex_elemdataadr[f] + local_elemid * (dim + 1)
  ee_base = flex_elemedgeadr[f] + local_elemid * nedge
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_enableflags & EnableBit.SLEEP:
    any_awake = bool(False)
    for v in range(dim + 1):
      gvert = vbase + flex_elem[elem_data_adr + v]
      bid = flex_vertbodyid[gvert]
      if bid >= 0 and body_awake_in[worldid, bid] == SleepState.AWAKE:
        any_awake = bool(True)
    if not any_awake:
      return

  xpos_next = passive.mat43()
  for v in range(dim + 1):
    vi = flex_elem[elem_data_adr + v]
    gvert = vbase + vi
    vel = passive.flex_gather_vert(
      body_rootid,
      body_weldid,
      body_dofnum,
      body_dofadr,
      body_simple,
      dof_parentid,
      flex_vertbodyid,
      subtree_com_in,
      cdof_in,
      flexvert_xpos_in,
      qvel_in,
      worldid,
      gvert,
    )
    xpos_next[v] = flexvert_xpos_in[worldid, gvert] + timestep * vel

  edges = passive.stretch_edge_endpoints(dim)
  g = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  s_el = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for e in range(nedge):
    dx = xpos_next[edges[e, 0]] - xpos_next[edges[e, 1]]
    L_sq = wp.dot(dx, dx)
    L0 = flexedge_length0[ebase + flex_elemedge[ee_base + e]]
    g_e = L0 * L0
    g[e] = g_e
    s_el[e] = L_sq - g_e - plastic_in[worldid, padr + e]

  stiff_adr = stiffness_adr_base + stride * local_elemid
  metric, f_vec = passive.stretch_elasticity(flex_stiffness, stiff_adr, s_el, nedge)
  h_vec = passive.stretch_tension(metric, g, nedge)

  g_h = wp.dot(g, h_vec)
  if wp.abs(g_h) <= MJ_MINVAL:
    return
  alpha = wp.dot(g, f_vec) / g_h

  s_dev = s_el - alpha * g
  f_dev = f_vec - alpha * h_vec

  trPK = float(0.0)
  for a in range(nedge):
    trPK += g[a] * g[a] * metric[a, a]
  if dim == 2:
    trPK += _face_trpk(g, metric, 0, 1, 2)
  else:
    trPK += _face_trpk(g, metric, 0, 1, 2)
    trPK += _face_trpk(g, metric, 0, 4, 5)
    trPK += _face_trpk(g, metric, 1, 3, 5)
    trPK += _face_trpk(g, metric, 2, 3, 4)
    s0 = g[0] + g[3]
    s1 = g[1] + g[4]
    s2 = g[2] + g[5]
    d03 = s1 - s2
    d14 = s2 - s0
    d25 = s0 - s1
    trPK += 0.5 * (d03 * d03 * metric[0, 3] + d14 * d14 * metric[1, 4] + d25 * d25 * metric[2, 5])

  mu_t = (trPK - g_h / float(dim)) / float(nedge - 1)
  if mu_t <= MJ_MINVAL:
    return

  dev_norm_sq = wp.max(0.0, wp.dot(s_dev, f_dev) / (4.0 * mu_t))
  alpha_pos = wp.max(0.0, alpha) if dim == 2 else 0.0
  eps_eq = wp.sqrt((2.0 * dev_norm_sq) / 3.0 + alpha_pos * alpha_pos)
  if eps_eq > yield_:
    scale = 1.0 - yield_ / eps_eq
    if dim == 2:
      for e in range(3):
        s_p = plastic_in[worldid, padr + e] + scale * (s_dev[e] + alpha_pos * g[e])
        plastic_out[worldid, padr + e] = wp.max(s_p, -0.99 * g[e])
    else:
      sp_dev = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      for e in range(6):
        sp_dev[e] = plastic_in[worldid, padr + e] + scale * s_dev[e]
      beta = wp.dot(h_vec, sp_dev) / g_h
      sp_dev = sp_dev - beta * g

      c3, _ = _gram_det_3d(g)
      c0, grad = _gram_det_3d(sp_dev)
      c1 = wp.dot(grad, g)
      u = float(1.0)
      if c3 > MJ_MINVAL:
        m2 = wp.max(0.0, -c1 / (3.0 * c3))
        if m2 > MJ_MINVAL:
          m_val = wp.sqrt(m2)
          R = (c3 - c0) / (2.0 * c3 * m_val * m2)
          if R >= 1.0:
            w_cbrt = wp.pow(R + wp.sqrt(wp.max(0.0, R * R - 1.0)), 1.0 / 3.0)
            u = m_val * (w_cbrt + 1.0 / w_cbrt)
          else:
            u = 2.0 * m_val * wp.cos(wp.acos(wp.max(-1.0, R)) / 3.0)
      for e in range(6):
        plastic_out[worldid, padr + e] = wp.max(sp_dev[e] + (u - 1.0) * g[e], -0.99 * g[e])
    flex_hessian_valid_out[worldid, f] = False


@wp.kernel
def _next_plastic_flex_bend(
  # Model:
  opt_timestep: wp.array[float],
  opt_enableflags: int,
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  dof_parentid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_edgeadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_bendingadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_edge: wp.array[wp.vec2i],
  flex_edgeflap: wp.array[wp.vec2i],
  flexedge_length0: wp.array[float],
  flex_stiffness: wp.array[float],
  flex_bending: wp.array[float],
  flex_yield: wp.array[float],
  flex_edgeflexid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  body_awake_in: wp.array2d[int],
  # Data out:
  plastic_out: wp.array2d[float],
):
  worldid, edgeid = wp.tid()
  f = flex_edgeflexid[edgeid]
  padr_base = flex_plasticadr[f]
  if padr_base < 0:
    return
  if flex_dim[f] != 2 or flex_interp[f] != 0:
    return
  bending_adr_base = flex_bendingadr[f]
  stiffness_adr_base = flex_stiffnessadr[f]
  if bending_adr_base < 0 or stiffness_adr_base < 0:
    return
  thickness = flex_stiffness[stiffness_adr_base + 6]
  if thickness <= MJ_MINVAL:
    return
  flap = flex_edgeflap[edgeid]
  if flap[1] == -1:
    return

  eid = edgeid - flex_edgeadr[f]
  bbase = bending_adr_base + 17 * eid
  b23 = flex_bending[bbase + 11]
  if b23 <= MJ_MINVAL:
    return

  edge = flex_edge[edgeid]
  vbase = flex_vertadr[f]
  v = wp.vec4i(vbase + edge[0], vbase + edge[1], vbase + flap[0], vbase + flap[1])

  if opt_enableflags & EnableBit.SLEEP:
    any_awake = bool(False)
    for j in range(4):
      bid = flex_vertbodyid[v[j]]
      if bid >= 0 and body_awake_in[worldid, bid] == SleepState.AWAKE:
        any_awake = bool(True)
    if not any_awake:
      return

  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  vpos = passive.mat43()
  for j in range(4):
    vel = passive.flex_gather_vert(
      body_rootid,
      body_weldid,
      body_dofnum,
      body_dofadr,
      body_simple,
      dof_parentid,
      flex_vertbodyid,
      subtree_com_in,
      cdof_in,
      flexvert_xpos_in,
      qvel_in,
      worldid,
      v[j],
    )
    vpos[j] = flexvert_xpos_in[worldid, v[j]] + timestep * vel

  ed0 = vpos[1] - vpos[0]
  ed1 = vpos[2] - vpos[0]
  ed2 = vpos[3] - vpos[0]
  frc2 = wp.cross(ed2, ed0)
  frc3 = wp.cross(ed0, ed1)

  two_a1 = wp.length(frc3)
  two_a2 = wp.length(frc2)
  l0 = flexedge_length0[edgeid]
  denom = two_a1 * two_a2
  if denom <= MJ_MINVAL or l0 <= MJ_MINVAL:
    return

  yield_ = wp.max(0.0, flex_yield[f])
  kappa_yield = 2.0 * yield_ / thickness
  omega = wp.dot(ed2, frc3)
  theta = wp.atan2(omega * l0, wp.dot(frc2, frc3))
  theta_yield = 0.5 * (two_a1 + two_a2) / l0 * kappa_yield

  stretch_pnum = 3 * flex_elemnum[f] if flex_stiffness[stiffness_adr_base] != 0.0 else 0
  padr = padr_base + stretch_pnum + eid
  b16_0 = flex_bending[bbase + 16]
  b_p = plastic_in[worldid, padr]
  theta_p = wp.atan2(-(b16_0 + b_p) * l0, b23)
  theta_el = theta - theta_p
  if theta_el > theta_yield or theta_el < -theta_yield:
    theta_p = theta - theta_yield if theta_el > theta_yield else theta + theta_yield
    theta_p = wp.clamp(theta_p, -1.3, 1.3)
    plastic_out[worldid, padr] = -(b23 / l0) * wp.tan(theta_p) - b16_0


@wp.kernel
def _next_plastic_flex_interp(
  # Model:
  opt_timestep: wp.array[float],
  opt_enableflags: int,
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  dof_parentid: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_node0: wp.array[wp.vec3],
  flex_stiffness: wp.array[float],
  flex_yield: wp.array[float],
  flex_faceadr: wp.array[int],
  flex_cell_map: wp.array[wp.vec4i],
  flex_face: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  body_awake_in: wp.array2d[int],
  flexnode_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  plastic_out: wp.array2d[float],
):
  worldid, cellid = wp.tid()
  mapping = flex_cell_map[cellid]
  f = mapping[0]
  ci = mapping[1]
  cj = mapping[2]
  ck = mapping[3]

  padr_base = flex_plasticadr[f]
  if padr_base < 0:
    return
  order = flex_interp[f]
  if order == 0:
    return
  shell_mode = order < 0
  order_abs = -order if shell_mode else order
  npc = (order_abs + 1) * (order_abs + 1) if shell_mode else (order_abs + 1) * (order_abs + 1) * (order_abs + 1)
  ndof_cell = 3 * npc

  cellnum = flex_cellnum[f]
  cx = cellnum[0]
  cy = cellnum[1]
  cz = cellnum[2]
  nstart = flex_nodeadr[f]
  ny_g = cy * order_abs + 1
  nz_g = cz * order_abs + 1

  stiffness_adr_base = flex_stiffnessadr[f]
  if stiffness_adr_base < 0:
    return
  cell_idx = ci if shell_mode else ci * cy * cz + cj * cz + ck
  k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell
  if not shell_mode and flex_stiffness[k_base] == 0.0:
    return

  op1 = order_abs + 1
  if opt_enableflags & EnableBit.SLEEP:
    any_awake = bool(False)
    for idx_j in range(npc):
      if shell_mode:
        node_j = flex_face[flex_faceadr[f] + ci, idx_j]
      else:
        li_j = idx_j // (op1 * op1)
        lj_j = (idx_j // op1) % op1
        lk_j = idx_j % op1
        gidx_j = (ci * order_abs + li_j) * ny_g * nz_g + (cj * order_abs + lj_j) * nz_g + (ck * order_abs + lk_j)
        node_j = nstart + gidx_j
      bid = flex_nodebodyid[node_j]
      if bid >= 0 and body_awake_in[worldid, bid] == SleepState.AWAKE:
        any_awake = bool(True)
    if not any_awake:
      return

  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  f0 = wp.vec3(0.0)
  f1 = wp.vec3(0.0)
  f2 = wp.vec3(0.0)
  for idx_j in range(npc):
    if shell_mode:
      node_j = flex_face[flex_faceadr[f] + ci, idx_j]
    else:
      li_j = idx_j // (op1 * op1)
      lj_j = (idx_j // op1) % op1
      lk_j = idx_j % op1
      gidx_j = (ci * order_abs + li_j) * ny_g * nz_g + (cj * order_abs + lj_j) * nz_g + (ck * order_abs + lk_j)
      node_j = nstart + gidx_j
    vel_j = passive.flex_gather_vert(
      body_rootid,
      body_weldid,
      body_dofnum,
      body_dofadr,
      body_simple,
      dof_parentid,
      flex_nodebodyid,
      subtree_com_in,
      cdof_in,
      flexnode_xpos_in,
      qvel_in,
      worldid,
      node_j,
    )
    xpos_j = flexnode_xpos_in[worldid, node_j] + timestep * vel_j
    if shell_mode:
      l0 = idx_j // op1
      l1 = idx_j % op1
      dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))
      dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))
      phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5
      phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5
      f0 += xpos_j * (dphi0 * phi1)
      f1 += xpos_j * (phi0 * dphi1)
    else:
      dphi_x = float(li_j - 1) if order_abs == 2 else (-1.0 + 2.0 * float(li_j))
      dphi_y = float(lj_j - 1) if order_abs == 2 else (-1.0 + 2.0 * float(lj_j))
      dphi_z = float(lk_j - 1) if order_abs == 2 else (-1.0 + 2.0 * float(lk_j))
      phi_x = wp.where(li_j == 1, 1.0, 0.0) if order_abs == 2 else 0.5
      phi_y = wp.where(lj_j == 1, 1.0, 0.0) if order_abs == 2 else 0.5
      phi_z = wp.where(lk_j == 1, 1.0, 0.0) if order_abs == 2 else 0.5
      f0 += xpos_j * (dphi_x * phi_y * phi_z)
      f1 += xpos_j * (phi_x * dphi_y * phi_z)
      f2 += xpos_j * (phi_x * phi_y * dphi_z)

  yield_ = wp.max(0.0, flex_yield[f])
  if shell_mode:
    normal_axis = 0 if ci < 2 * cy * cz else (1 if ci < 2 * (cy * cz + cx * cz) else 2)
    axis0 = (normal_axis + 1) % 3
    axis1 = (normal_axis + 2) % 3
    node_0 = flex_face[flex_faceadr[f] + ci, 0]
    node_last = flex_face[flex_faceadr[f] + ci, npc - 1]
    dref = flex_node0[node_last] - flex_node0[node_0]
    d0 = dref[axis0]
    d1 = dref[axis1]
    if d0 <= MJ_MINVAL or d1 <= MJ_MINVAL:
      return
    t1_phys = f0 / d0
    t2_phys = f1 / d1
    normal = wp.cross(t1_phys, t2_phys)
    F = wp.mat33(0.0)
    if normal_axis == 0:
      F = wp.mat33(normal[0], t1_phys[0], t2_phys[0], normal[1], t1_phys[1], t2_phys[1], normal[2], t1_phys[2], t2_phys[2])
    elif normal_axis == 1:
      F = wp.mat33(t2_phys[0], normal[0], t1_phys[0], t2_phys[1], normal[1], t1_phys[1], t2_phys[2], normal[2], t1_phys[2])
    else:
      F = wp.mat33(t1_phys[0], t2_phys[0], normal[0], t1_phys[1], t2_phys[1], normal[1], t1_phys[2], t2_phys[2], normal[2])
    cell_quat = mat33_to_quat_polar(F)
    cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])
    t1_rot = wp.quat_rotate(cell_quat_inv, f0)
    t2_rot = wp.quat_rotate(cell_quat_inv, f1)
    padr = padr_base + 3 * cell_idx
    sp0 = plastic_in[worldid, padr + 0]
    sp1 = plastic_in[worldid, padr + 1]
    sp2 = plastic_in[worldid, padr + 2]
    e0 = (t1_rot[axis0] / d0 - 1.0) - sp0
    e1 = (t2_rot[axis1] / d1 - 1.0) - sp1
    g01 = (t2_rot[axis0] / d1 + t1_rot[axis1] / d0) - sp2
    alpha = 0.5 * (e0 + e1)
    alpha_pos = wp.max(0.0, alpha)
    dev0 = 0.5 * (e0 - e1)
    dev1 = -dev0
    dev_norm_sq = 2.0 * dev0 * dev0 + 0.5 * g01 * g01
    eps_eq = wp.sqrt((2.0 * dev_norm_sq) / 3.0 + alpha_pos * alpha_pos)
    if eps_eq > yield_:
      scale = 1.0 - yield_ / eps_eq
      sp0 += scale * (dev0 + alpha_pos)
      sp1 += scale * (dev1 + alpha_pos)
      sp2 += scale * g01
      fnorm_sq = sp0 * sp0 + sp1 * sp1 + 0.5 * sp2 * sp2
      if fnorm_sq > 0.1225:
        clamp_scale = 0.35 / wp.sqrt(fnorm_sq)
        sp0 *= clamp_scale
        sp1 *= clamp_scale
        sp2 *= clamp_scale
      plastic_out[worldid, padr + 0] = sp0
      plastic_out[worldid, padr + 1] = sp1
      plastic_out[worldid, padr + 2] = sp2
  else:
    node_0 = nstart + (ci * order_abs) * ny_g * nz_g + (cj * order_abs) * nz_g + (ck * order_abs)
    node_last = nstart + ((ci + 1) * order_abs) * ny_g * nz_g + ((cj + 1) * order_abs) * nz_g + ((ck + 1) * order_abs)
    dref = flex_node0[node_last] - flex_node0[node_0]
    dx = dref[0]
    dy = dref[1]
    dz = dref[2]
    if dx <= MJ_MINVAL or dy <= MJ_MINVAL or dz <= MJ_MINVAL:
      return
    f0_phys = f0 / dx
    f1_phys = f1 / dy
    f2_phys = f2 / dz
    F = wp.mat33(f0_phys[0], f1_phys[0], f2_phys[0], f0_phys[1], f1_phys[1], f2_phys[1], f0_phys[2], f1_phys[2], f2_phys[2])
    cell_quat = mat33_to_quat_polar(F)
    cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])
    f0_rot = wp.quat_rotate(cell_quat_inv, f0)
    f1_rot = wp.quat_rotate(cell_quat_inv, f1)
    f2_rot = wp.quat_rotate(cell_quat_inv, f2)
    padr = padr_base + 6 * cell_idx
    sp0 = plastic_in[worldid, padr + 0]
    sp1 = plastic_in[worldid, padr + 1]
    sp2 = plastic_in[worldid, padr + 2]
    sp3 = plastic_in[worldid, padr + 3]
    sp4 = plastic_in[worldid, padr + 4]
    sp5 = plastic_in[worldid, padr + 5]
    e0 = (f0_rot[0] / dx - 1.0) - sp0
    e1 = (f1_rot[1] / dy - 1.0) - sp1
    e2 = (f2_rot[2] / dz - 1.0) - sp2
    g01 = (f1_rot[0] / dy + f0_rot[1] / dx) - sp3
    g02 = (f2_rot[0] / dz + f0_rot[2] / dx) - sp4
    g12 = (f2_rot[1] / dz + f1_rot[2] / dy) - sp5
    mean = (e0 + e1 + e2) / 3.0
    dev0 = e0 - mean
    dev1 = e1 - mean
    dev2 = e2 - mean
    dev_norm_sq = dev0 * dev0 + dev1 * dev1 + dev2 * dev2 + 0.5 * (g01 * g01 + g02 * g02 + g12 * g12)
    eps_eq = wp.sqrt((2.0 * dev_norm_sq) / 3.0)
    if eps_eq > yield_:
      scale = 1.0 - yield_ / eps_eq
      sp0 += scale * dev0
      sp1 += scale * dev1
      sp2 += scale * dev2
      sp3 += scale * g01
      sp4 += scale * g02
      sp5 += scale * g12
      fnorm_sq = sp0 * sp0 + sp1 * sp1 + sp2 * sp2 + 0.5 * (sp3 * sp3 + sp4 * sp4 + sp5 * sp5)
      if fnorm_sq > 0.1225:
        clamp_scale = 0.35 / wp.sqrt(fnorm_sq)
        sp0 *= clamp_scale
        sp1 *= clamp_scale
        sp2 *= clamp_scale
        sp3 *= clamp_scale
        sp4 *= clamp_scale
        sp5 *= clamp_scale
      plastic_out[worldid, padr + 0] = sp0
      plastic_out[worldid, padr + 1] = sp1
      plastic_out[worldid, padr + 2] = sp2
      plastic_out[worldid, padr + 3] = sp3
      plastic_out[worldid, padr + 4] = sp4
      plastic_out[worldid, padr + 5] = sp5


def _advance(m: Model, d: Data, qacc: wp.array, qvel: Optional[wp.array] = None):
  """Advance state and time given activation derivatives and acceleration."""
  # TODO(team): can we assume static timesteps?

  # advance activations
  wp.launch(
    _next_activation,
    dim=(d.nworld, m.nactuator),
    inputs=[
      m.opt.timestep,
      m.jnt_type,
      m.actuator_trntype,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_trnid,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_actlimited,
      m.actuator_actrange,
      m.actuator_gear,
      d.act,
      d.act_dot,
      d.actuator_length,
      d.actuator_velocity,
      1.0,
      True,
    ],
    outputs=[d.act],
  )

  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, d.qvel, qacc, 1.0],
    outputs=[d.qvel],
  )

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  qvel_in = qvel or d.qvel

  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, qvel_in, 1.0],
    outputs=[d.qpos],
  )

  # carry the twist reference of 1D rods to the configuration of this step (flexvert_xpos)
  if m.nplastic > 0 and m.nflexrodhinge > 0:
    wp.launch(
      _next_rod_twist,
      dim=(d.nworld, m.nflex),
      inputs=[
        m.qpos0,
        m.body_parentid,
        m.body_pos,
        m.body_quat,
        m.jnt_qposadr,
        m.jnt_dofadr,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_bendingadr,
        m.flex_plasticadr,
        m.flex_vertbodyid,
        m.flex_vert,
        m.flex_bending,
        d.qpos,
        d.plastic,
        d.xpos,
        d.xquat,
        d.xmat,
        d.flexvert_xpos,
      ],
      outputs=[d.plastic],
    )

  # update plastic spring references via return mapping
  if m.nplastic > 0 and not (m.opt.disableflags & DisableBit.SPRING):
    wp.launch(
      _next_plastic_joint,
      dim=(d.nworld, m.njnt),
      inputs=[
        m.opt.enableflags,
        m.jnt_type,
        m.jnt_qposadr,
        m.jnt_bodyid,
        m.jnt_plasticadr,
        m.jnt_springyield,
        d.qpos,
        d.plastic,
        d.body_awake,
      ],
      outputs=[d.plastic],
    )
    wp.launch(
      _next_plastic_tendon,
      dim=(d.nworld, m.ntendon),
      inputs=[
        m.opt.timestep,
        m.opt.enableflags,
        m.jnt_qposadr,
        m.jnt_bodyid,
        m.dof_treeid,
        m.tendon_adr,
        m.tendon_num,
        m.tendon_plasticadr,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_springyield,
        m.wrap_type,
        m.wrap_objid,
        m.wrap_prm,
        d.qpos,
        qvel_in,
        d.plastic,
        d.ten_J,
        d.ten_length,
        d.body_awake,
        d.tree_awake,
      ],
      outputs=[d.plastic],
    )
    wp.launch(
      _next_plastic_flex_stretch,
      dim=(d.nworld, m.nflexelem),
      inputs=[
        m.opt.timestep,
        m.opt.enableflags,
        m.body_rootid,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.body_simple,
        m.dof_parentid,
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_elemadr,
        m.flex_elemdataadr,
        m.flex_stiffnessadr,
        m.flex_elemedgeadr,
        m.flex_plasticadr,
        m.flex_vertbodyid,
        m.flex_elem,
        m.flex_elemedge,
        m.flexedge_length0,
        m.flex_stiffness,
        m.flex_yield,
        m.flex_elemflexid,
        qvel_in,
        d.plastic,
        d.subtree_com,
        d.cdof,
        d.flexvert_xpos,
        d.body_awake,
      ],
      outputs=[d.plastic, d.flex_hessian_valid],
    )
    wp.launch(
      _next_plastic_flex_bend,
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.opt.timestep,
        m.opt.enableflags,
        m.body_rootid,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.body_simple,
        m.dof_parentid,
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_elemnum,
        m.flex_stiffnessadr,
        m.flex_bendingadr,
        m.flex_plasticadr,
        m.flex_vertbodyid,
        m.flex_edge,
        m.flex_edgeflap,
        m.flexedge_length0,
        m.flex_stiffness,
        m.flex_bending,
        m.flex_yield,
        m.flex_edgeflexid,
        qvel_in,
        d.plastic,
        d.subtree_com,
        d.cdof,
        d.flexvert_xpos,
        d.body_awake,
      ],
      outputs=[d.plastic],
    )
    wp.launch(
      _next_plastic_flex_interp,
      dim=(d.nworld, m.nflexintcell),
      inputs=[
        m.opt.timestep,
        m.opt.enableflags,
        m.body_rootid,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.body_simple,
        m.dof_parentid,
        m.flex_interp,
        m.flex_cellnum,
        m.flex_nodeadr,
        m.flex_stiffnessadr,
        m.flex_plasticadr,
        m.flex_nodebodyid,
        m.flex_node0,
        m.flex_stiffness,
        m.flex_yield,
        m.flex_faceadr,
        m.flex_cell_map,
        m.flex_face,
        qvel_in,
        d.plastic,
        d.subtree_com,
        d.cdof,
        d.body_awake,
        d.flexnode_xpos,
      ],
      outputs=[d.plastic],
    )

  # advance history buffers before time advance
  history.insert_ctrl_history(m, d)

  wp.launch(
    _next_time_builder(int(m.opt.warn_overflow)),
    dim=d.nworld,
    inputs=[
      m.opt.timestep,
      m.is_sparse,
      d.nefc,
      d.time,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.nworld,
      d.naconmax,
      d.njmax,
      d.njmax_nnz,
      d.nacon,
      d.ncollision,
    ],
    outputs=[d.time, d.overflow],
  )

  wp.copy(d.qacc_warmstart, d.qacc)

  sleep_enabled = bool(m.opt.enableflags & EnableBit.SLEEP) and not bool(m.opt.disableflags & DisableBit.ISLAND)
  if sleep_enabled:
    sleep.sleep(m, d)
    fwd_velocity(m, d)
    sleep.update_sleep(m, d)


@wp.kernel
def _compute_damping_deriv(
  # Model:
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
  # Out:
  deriv_out: wp.array2d[float],
):
  worldid, tid = wp.tid()
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
    tid,
  )
  v = qvel_in[worldid, tid]
  deriv_out[worldid, tid] = util_misc._poly_force_deriv(damping, dpoly, v, 1)


@wp.kernel
def _euler_damp_qfrc(
  # Model:
  opt_timestep: wp.array[float],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # In:
  damp_deriv: wp.array2d[float],
  # Out:
  M_integration_out: wp.array2d[float],
):
  worldid, tid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  adr = M_rowadr[tid] + M_rownnz[tid] - 1
  M_integration_out[worldid, adr] += timestep * damp_deriv[worldid, tid]


@event_scope
def euler(m: Model, d: Data):
  """Euler integrator, semi-implicit in velocity."""
  # integrate damping implicitly
  if not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER)):
    qacc = wp.empty((d.nworld, m.nv), dtype=float)

    # Compute damping derivative
    damp_deriv = wp.empty((d.nworld, m.nv), dtype=float)
    wp.launch(
      _compute_damping_deriv,
      dim=(d.nworld, m.nv),
      inputs=[
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
      outputs=[damp_deriv],
    )

    # Clone M, add the damping to the diagonal, and factor-solve. factor_solve_i factors each block
    # per-block (packed dense and/or sparse LDL); the scratch qLD matches d.qLD.
    M = wp.clone(d.M)
    qLD = wp.empty_like(d.qLD)
    qLDiagInv = wp.empty((d.nworld, m.nv), dtype=float)
    wp.launch(
      _euler_damp_qfrc,
      dim=(d.nworld, m.nv),
      inputs=[m.opt.timestep, m.M_rownnz, m.M_rowadr, damp_deriv],
      outputs=[M],
    )
    smooth.factor_solve_i(m, d, M, qLD, qLDiagInv, qacc, d.efc.Ma)
    _advance(m, d, qacc)
  else:
    _advance(m, d, d.qacc)


def _rk_perturb_state(
  m: Model,
  d: Data,
  scale: float,
  qpos_t0: wp.array2d[float],
  qvel_t0: wp.array2d[float],
  qvel_pos: wp.array2d[float],
  act_t0: Optional[wp.array] = None,
):
  # position, integrated with the (corrected) previous stage velocity
  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos_t0, qvel_pos, scale],
    outputs=[d.qpos],
  )

  # velocity
  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, qvel_t0, d.qacc, scale],
    outputs=[d.qvel],
  )

  # activation
  if m.na and act_t0 is not None:
    wp.launch(
      _next_activation,
      dim=(d.nworld, m.nactuator),
      inputs=[
        m.opt.timestep,
        m.jnt_type,
        m.actuator_trntype,
        m.actuator_dyntype,
        m.actuator_gaintype,
        m.actuator_biastype,
        m.actuator_actadr,
        m.actuator_actnum,
        m.actuator_trnid,
        m.actuator_dynprm,
        m.actuator_gainprm,
        m.actuator_biasprm,
        m.actuator_actlimited,
        m.actuator_actrange,
        m.actuator_gear,
        act_t0,
        d.act_dot,
        d.actuator_length,
        d.actuator_velocity,
        scale,
        False,
      ],
      outputs=[d.act],
    )


@wp.kernel
def _rk_correct_velocity(
  # Model:
  opt_timestep: wp.array[float],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  # In:
  qvel_stage_in: wp.array2d[float],
  scale: float,
  # Out:
  qvel_stage_out: wp.array2d[float],
):
  # Runge-Kutta-Munthe-Kaas (RKMK) correction: maps the stage angular velocities of ball and free joints to the
  # rates of their rotation vectors rot = timestep * scale * (previous corrected stage velocity), in place
  worldid, jntid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  jnttype = jnt_type[jntid]
  dofadr = jnt_dofadr[jntid]

  if jnttype == JointType.HINGE or jnttype == JointType.SLIDE:
    qvel_stage_out[worldid, dofadr] = qvel_in[worldid, dofadr]
    return

  rotadr = dofadr
  if jnttype == JointType.FREE:
    rotadr = dofadr + 3

  qvel_prev = wp.vec3(qvel_stage_in[worldid, rotadr], qvel_stage_in[worldid, rotadr + 1], qvel_stage_in[worldid, rotadr + 2])
  rot = timestep * scale * qvel_prev
  vel = wp.vec3(qvel_in[worldid, rotadr], qvel_in[worldid, rotadr + 1], qvel_in[worldid, rotadr + 2])
  res = math.deriv_expmap(rot, vel)

  if jnttype == JointType.FREE:
    for i in range(3):
      qvel_stage_out[worldid, dofadr + i] = qvel_in[worldid, dofadr + i]
  for i in range(3):
    qvel_stage_out[worldid, rotadr + i] = res[i]


@wp.kernel
def _rk_accumulate_velocity_acceleration(
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # In:
  scale: float,
  # Data out:
  qvel_out: wp.array2d[float],
  qacc_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]
  qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]


@wp.kernel
def _rk_accumulate_activation_velocity(
  # Data in:
  act_dot_in: wp.array2d[float],
  # In:
  scale: float,
  # Data out:
  act_dot_out: wp.array2d[float],
):
  worldid, actid = wp.tid()
  act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]


def _rk_accumulate(
  m: Model,
  d: Data,
  scale: float,
  qvel_stage: wp.array2d[float],
  qvel_rk: wp.array2d[float],
  qacc_rk: wp.array2d[float],
  act_dot_rk: Optional[wp.array] = None,
):
  """Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4."""
  wp.launch(
    _rk_accumulate_velocity_acceleration,
    dim=(d.nworld, m.nv),
    inputs=[qvel_stage, d.qacc, scale],
    outputs=[qvel_rk, qacc_rk],
  )

  if m.na and act_dot_rk is not None:
    wp.launch(
      _rk_accumulate_activation_velocity,
      dim=(d.nworld, m.na),
      inputs=[d.act_dot, scale],
      outputs=[act_dot_rk],
    )


@event_scope
def rungekutta4(m: Model, d: Data):
  """Runge-Kutta explicit order 4 integrator."""
  # RK4 tableau
  A = [0.5, 0.5, 1.0]  # diagonal only
  B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

  qpos_t0 = wp.clone(d.qpos)
  qvel_t0 = wp.clone(d.qvel)
  qvel_rk = wp.zeros((d.nworld, m.nv), dtype=float)
  qacc_rk = wp.zeros((d.nworld, m.nv), dtype=float)

  if m.na:
    act_t0 = wp.clone(d.act)
    act_dot_rk = wp.zeros((d.nworld, m.na), dtype=float)
  else:
    act_t0 = None
    act_dot_rk = None

  save_len = bool(
    m.na
    and not (m.opt.disableflags & DisableBit.ACTUATION)
    and m.has_actuator_wrap
  )
  save_vel = bool(
    m.na
    and not (m.opt.disableflags & DisableBit.ACTUATION)
    and m.has_actuator_dcmotor
  )
  actuator_length_t0 = wp.clone(d.actuator_length) if save_len else None
  actuator_velocity_t0 = wp.clone(d.actuator_velocity) if save_vel else None

  if m.nplastic > 0 and m.ntendon > 0:
    ten_length_t0 = wp.clone(d.ten_length)
    ten_J_t0 = wp.clone(d.ten_J)
  else:
    ten_length_t0 = None
    ten_J_t0 = None

  if m.nplastic > 0 and m.nflexvert > 0:
    flexvert_xpos_t0 = wp.clone(d.flexvert_xpos)
    flexnode_xpos_t0 = wp.clone(d.flexnode_xpos) if m.nflexnode > 0 else None
    cdof_t0 = wp.clone(d.cdof)
    subtree_com_t0 = wp.clone(d.subtree_com)
    xpos_t0 = wp.clone(d.xpos)
    xquat_t0 = wp.clone(d.xquat)
    xmat_t0 = wp.clone(d.xmat)
  else:
    flexvert_xpos_t0 = None
    flexnode_xpos_t0 = None
    cdof_t0 = None
    subtree_com_t0 = None
    xpos_t0 = None
    xquat_t0 = None
    xmat_t0 = None

  # stage velocities with the RKMK correction (rates of the rotation vectors of ball and free joints); the first
  # stage has a zero rotation vector, so it needs no correction
  qvel_stage = wp.clone(d.qvel)

  _rk_accumulate(m, d, B[0], qvel_stage, qvel_rk, qacc_rk, act_dot_rk)

  for i in range(3):
    a, b = float(A[i]), B[i + 1]
    _rk_perturb_state(m, d, a, qpos_t0, qvel_t0, qvel_stage, act_t0)
    _forward(m, d, skip_sensor=True)
    wp.launch(
      _rk_correct_velocity,
      dim=(d.nworld, m.njnt),
      inputs=[m.opt.timestep, m.jnt_type, m.jnt_dofadr, d.qvel, qvel_stage, a],
      outputs=[qvel_stage],
    )
    _rk_accumulate(m, d, b, qvel_stage, qvel_rk, qacc_rk, act_dot_rk)

  wp.copy(d.qpos, qpos_t0)
  wp.copy(d.qvel, qvel_t0)

  # last stage's acceleration, the most recent solve: warm start for the next step
  qacc_last = wp.clone(d.qacc)

  if m.na:
    wp.copy(d.act, act_t0)
    wp.copy(d.act_dot, act_dot_rk)

  if actuator_length_t0 is not None:
    wp.copy(d.actuator_length, actuator_length_t0)
  if actuator_velocity_t0 is not None:
    wp.copy(d.actuator_velocity, actuator_velocity_t0)

  if ten_length_t0 is not None:
    wp.copy(d.ten_length, ten_length_t0)
    wp.copy(d.ten_J, ten_J_t0)

  if flexvert_xpos_t0 is not None:
    wp.copy(d.flexvert_xpos, flexvert_xpos_t0)
    if flexnode_xpos_t0 is not None:
      wp.copy(d.flexnode_xpos, flexnode_xpos_t0)
    wp.copy(d.cdof, cdof_t0)
    wp.copy(d.subtree_com, subtree_com_t0)
    wp.copy(d.xpos, xpos_t0)
    wp.copy(d.xquat, xquat_t0)
    wp.copy(d.xmat, xmat_t0)

  _advance(m, d, qacc_rk, qvel_rk)

  # effective acceleration of the step (RK weighted combination)
  wp.copy(d.qacc, qacc_rk)
  wp.copy(d.qacc_warmstart, qacc_last)


@wp.kernel
def _map_m2d(
  # Model:
  mapM2D: wp.array[int],
  # In:
  qH_M: wp.array2d[float],
  # Data out:
  qLU_out: wp.array2d[float],
):
  # Scatter qH_M (M-structure) into the D-structure qLU via mapM2D.
  worldid, elemid = wp.tid()
  m_idx = mapM2D[elemid]
  if m_idx >= 0:
    qLU_out[worldid, elemid] = qH_M[worldid, m_idx]
  else:
    qLU_out[worldid, elemid] = 0.0


@wp.kernel
def _implicit_free_body_reset_m(
  # Model:
  body_dofadr: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  body_freeadr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  # Data out:
  qH_out: wp.array2d[float],
):
  worldid, freeid = wp.tid()
  bodyid = body_freeadr[freeid]

  dof_adr = body_dofadr[bodyid]
  for r in range(6):
    row = dof_adr + r
    start = M_rowadr[row]
    nnz = M_rownnz[row]
    for k in range(nnz):
      qH_out[worldid, start + k] = M_in[worldid, start + k]


@wp.func
def _project_spatial_B(
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  dof_adr: int,
  geom_rotT: wp.mat33,
  offset: wp.vec3,
  B00: wp.mat33,
  B01: wp.mat33,
  B10: wp.mat33,
  B11: wp.mat33,
) -> mat66:
  c0 = cdof_in[worldid, dof_adr + 0]
  c1 = cdof_in[worldid, dof_adr + 1]
  c2 = cdof_in[worldid, dof_adr + 2]
  c3 = cdof_in[worldid, dof_adr + 3]
  c4 = cdof_in[worldid, dof_adr + 4]
  c5 = cdof_in[worldid, dof_adr + 5]

  ll0 = geom_rotT @ wp.vec3(c0[3], c0[4], c0[5])
  ll1 = geom_rotT @ wp.vec3(c1[3], c1[4], c1[5])
  ll2 = geom_rotT @ wp.vec3(c2[3], c2[4], c2[5])

  a3 = wp.vec3(c3[0], c3[1], c3[2])
  a4 = wp.vec3(c4[0], c4[1], c4[2])
  a5 = wp.vec3(c5[0], c5[1], c5[2])

  p3 = wp.vec3(c3[3], c3[4], c3[5]) + wp.cross(a3, offset)
  p4 = wp.vec3(c4[3], c4[4], c4[5]) + wp.cross(a4, offset)
  p5 = wp.vec3(c5[3], c5[4], c5[5]) + wp.cross(a5, offset)

  la3 = geom_rotT @ a3
  la4 = geom_rotT @ a4
  la5 = geom_rotT @ a5

  ll3 = geom_rotT @ p3
  ll4 = geom_rotT @ p4
  ll5 = geom_rotT @ p5

  ll_trans = wp.mat33(
    ll0[0],
    ll1[0],
    ll2[0],
    ll0[1],
    ll1[1],
    ll2[1],
    ll0[2],
    ll1[2],
    ll2[2],
  )
  la_rot = wp.mat33(
    la3[0],
    la4[0],
    la5[0],
    la3[1],
    la4[1],
    la5[1],
    la3[2],
    la4[2],
    la5[2],
  )
  ll_rot = wp.mat33(
    ll3[0],
    ll4[0],
    ll5[0],
    ll3[1],
    ll4[1],
    ll5[1],
    ll3[2],
    ll4[2],
    ll5[2],
  )

  ll_trans_T = wp.transpose(ll_trans)
  la_rot_T = wp.transpose(la_rot)
  ll_rot_T = wp.transpose(ll_rot)

  T_lin = B11 @ ll_trans
  T_rot = B10 @ la_rot + B11 @ ll_rot

  top_left = ll_trans_T @ T_lin
  top_right = ll_trans_T @ T_rot

  bot_left = la_rot_T @ (B01 @ ll_trans) + ll_rot_T @ T_lin
  bot_right = la_rot_T @ (B00 @ la_rot + B01 @ ll_rot) + ll_rot_T @ T_rot

  out = mat66(0.0)
  for r in range(3):
    for c in range(3):
      out[r, c] = top_left[r, c]
      out[r, 3 + c] = top_right[r, c]
      out[3 + r, c] = bot_left[r, c]
      out[3 + r, 3 + c] = bot_right[r, c]
  return out


@wp.kernel
def _implicit_free_body(
  # Model:
  nbody: int,
  opt_timestep: wp.array[float],
  opt_wind: wp.array[wp.vec3],
  opt_density: wp.array[float],
  opt_viscosity: wp.array[float],
  opt_integrator: int,
  opt_disableflags: int,
  opt_enableflags: int,
  body_rootid: wp.array[int],
  body_dofadr: wp.array[int],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_treeid: wp.array[int],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  body_fluid_ellipsoid: wp.array[bool],
  body_freeadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  crb_in: wp.array2d[vec10],
  M_in: wp.array2d[float],
  tree_awake_in: wp.array2d[int],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  solve: bool,
  vec_in: wp.array2d[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, freeid = wp.tid()
  bodyid = body_freeadr[freeid]

  dof_adr = body_dofadr[bodyid]

  # Sleep guard matching mjd_freeMhat: skip if sleeping
  treeid = dof_treeid[dof_adr]
  if (opt_enableflags & EnableBit.SLEEP) and (tree_awake_in[worldid, treeid] == 0):
    return

  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  # 1. A = M block (gather from sparse lower triangle)
  A = mat66(0.0)
  for r in range(6):
    row = dof_adr + r
    start = M_rowadr[row]
    nnz = M_rownnz[row]
    for k in range(nnz):
      c = M_colind[start + k] - dof_adr
      val = M_in[worldid, start + k]
      A[r, c] = val
      A[c, r] = val

  # 2. Add joint damping (with damper disable check and polynomial damping)
  if not (opt_disableflags & DisableBit.DAMPER):
    jntid = dof_jntid[dof_adr]
    act_damp, act_dpoly = util_misc.actuator_target_damping(
      actuator_trntype,
      actuator_trnid,
      actuator_damping,
      actuator_dampingpoly,
      actuator_gear,
      worldid,
      ObjType.JOINT,
      jntid,
      jnt_actuatorid[jntid],
    )
    for r in range(6):
      dof = dof_adr + r
      damp = dof_damping[worldid % dof_damping.shape[0], dof] + act_damp
      dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dof] + act_dpoly
      v = qvel_in[worldid, dof]
      A[r, r] += timestep * util_misc._poly_force_deriv(damp, dpoly, v, 1)

  # 3. Add gyroscopic bias velocity derivative
  crb = crb_in[worldid, bodyid]
  subtree_mass = crb[9]
  R = xmat_in[worldid, bodyid]
  Iw = wp.mat33(
    crb[0],
    crb[3],
    crb[4],
    crb[3],
    crb[1],
    crb[5],
    crb[4],
    crb[5],
    crb[2],
  )
  s = subtree_com_in[worldid, bodyid] - xpos_in[worldid, bodyid]
  qvel_rot = wp.vec3(
    qvel_in[worldid, dof_adr + 3],
    qvel_in[worldid, dof_adr + 4],
    qvel_in[worldid, dof_adr + 5],
  )
  lin, rot = math.free_bias_vel_blocks(subtree_mass, R, Iw, s, qvel_rot)
  h_mass = -timestep * subtree_mass
  for r in range(3):
    for c in range(3):
      A[r, 3 + c] += h_mass * lin[r, c]
      A[3 + r, 3 + c] += timestep * rot[r, c]

  # 4. Fluid force derivatives
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
  wind = opt_wind[worldid % opt_wind.shape[0]]

  if density > 0.0 or viscosity > 0.0:
    subtree_root = subtree_com_in[worldid, bodyid]

    for b in range(bodyid, nbody):
      if body_rootid[b] != bodyid:
        break

      if body_fluid_ellipsoid[b]:
        xipos = xipos_in[worldid, b]
        cvel = cvel_in[worldid, b]
        ang_global = wp.spatial_top(cvel)
        lin_global = wp.spatial_bottom(cvel)
        lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

        geomadr = body_geomadr[b]
        geomnum = body_geomnum[b]

        for g in range(geomnum):
          geomid = geomadr + g
          coef = geom_fluid[geomid, 0]
          if coef <= 0.0:
            continue

          size = geom_size[worldid % geom_size.shape[0], geomid]
          semiaxes = passive.geom_semiaxes(size, geom_type[geomid])
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

          blunt_drag_coef = geom_fluid[geomid, 1]
          slender_drag_coef = geom_fluid[geomid, 2]
          ang_drag_coef = geom_fluid[geomid, 3]
          kutta_lift_coef = geom_fluid[geomid, 4]
          magnus_lift_coef = geom_fluid[geomid, 5]
          virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
          virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])

          # Compute 6x6 spatial B matrix once per geom (unsymmetrized for free body)
          B00, B01, B10, B11 = derivative._geom_ellipsoid_fluid_B(
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
            False,
          )

          # 3x3 block projection of J^T @ B @ J
          offset = geom_pos - subtree_root
          J_T_B_J = _project_spatial_B(cdof_in, worldid, dof_adr, geom_rotT, offset, B00, B01, B10, B11)
          for r in range(6):
            for c in range(6):
              A[r, c] -= timestep * J_T_B_J[r, c]

      elif body_mass[worldid % body_mass.shape[0], b] >= MJ_MINVAL:
        b_ipos = xipos_in[worldid, b]
        b_imat = ximat_in[worldid, b]

        cvel = cvel_in[worldid, b]
        v_subtree_ang = wp.vec3(cvel[0], cvel[1], cvel[2])
        v_subtree_lin = wp.vec3(cvel[3], cvel[4], cvel[5])

        lin_com = v_subtree_lin - wp.cross(b_ipos - subtree_root, v_subtree_ang)
        b_imat_T = wp.transpose(b_imat)
        v_local_ang = b_imat_T @ v_subtree_ang
        v_local_lin = b_imat_T @ (lin_com - wind)

        lvel = wp.spatial_vector(
          v_local_ang[0],
          v_local_ang[1],
          v_local_ang[2],
          v_local_lin[0],
          v_local_lin[1],
          v_local_lin[2],
        )

        B_box = derivative._deriv_box_fluid(
          opt_integrator,
          body_mass,
          body_inertia,
          worldid,
          b,
          lvel,
          density,
          viscosity,
        )
        B00 = wp.diag(wp.vec3(B_box[0, 0], B_box[1, 1], B_box[2, 2]))
        B11 = wp.diag(wp.vec3(B_box[3, 3], B_box[4, 4], B_box[5, 5]))
        zero33 = wp.mat33(0.0)
        offset_box = b_ipos - subtree_root
        J_T_B_J = _project_spatial_B(cdof_in, worldid, dof_adr, b_imat_T, offset_box, B00, zero33, zero33, B11)
        for r in range(6):
          for c in range(6):
            A[r, c] -= timestep * J_T_B_J[r, c]

  # Solve A * vec_out = vec_in
  if solve:
    A_fact, pivot, ok = math.lu_factor_6x6(A)
    if ok:
      b_vec = vec6(0.0)
      for r in range(6):
        b_vec[r] = vec_in[worldid, dof_adr + r]
      x = math.lu_solve_6x6(A_fact, pivot, b_vec)
      for r in range(6):
        vec_out[worldid, dof_adr + r] = x[r]
    return

  # or multiply vec_out = A * vec_in
  b_vec = vec6(0.0)
  for r in range(6):
    b_vec[r] = vec_in[worldid, dof_adr + r]
  for r in range(6):
    Ax = float(0.0)
    for c in range(6):
      Ax += A[r, c] * b_vec[c]
    vec_out[worldid, dof_adr + r] = Ax


def implicit_free_body(m: Model, d: Data, vec_in: wp.array2d[float], vec_out: wp.array2d[float], solve: bool):
  """Solves or multiplies by the implicit mass matrix of standalone free bodies.

  Overwrites the free-body rows of vec_out with A^-1 * vec_in (solve) or A * vec_in, where A is
  the free-body block of M - h * D including the gyroscopic bias derivative.
  """
  wp.launch(
    _implicit_free_body,
    dim=(d.nworld, m.body_freeadr.size),
    inputs=[
      m.nbody,
      m.opt.timestep,
      m.opt.wind,
      m.opt.density,
      m.opt.viscosity,
      m.opt.integrator,
      m.opt.disableflags,
      m.opt.enableflags,
      m.body_rootid,
      m.body_dofadr,
      m.body_geomnum,
      m.body_geomadr,
      m.body_mass,
      m.body_inertia,
      m.jnt_actuatorid,
      m.dof_jntid,
      m.dof_treeid,
      m.dof_damping,
      m.dof_dampingpoly,
      m.geom_type,
      m.geom_size,
      m.geom_fluid,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_damping,
      m.actuator_dampingpoly,
      m.actuator_gear,
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.body_fluid_ellipsoid,
      m.body_freeadr,
      d.qvel,
      d.xpos,
      d.xmat,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.subtree_com,
      d.cdof,
      d.crb,
      d.M,
      d.tree_awake,
      d.cvel,
      solve,
      vec_in,
    ],
    outputs=[vec_out],
  )


@cache_kernel
def _discrete_free_gyro_kernel(has_fluid: bool, has_flex_passive: bool, is_inverse: bool, is_sparse: bool):
  @wp.kernel(module="unique")
  def _discrete_free_gyro(
    # Model:
    nactuator: int,
    ntendon: int,
    opt_timestep: wp.array[float],
    opt_enableflags: int,
    body_dofadr: wp.array[int],
    dof_treeid: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    M_colind: wp.array[int],
    efm_K_rownnz: wp.array[int],
    body_freeadr: wp.array[int],
    # Data in:
    nefc_in: wp.array[int],
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    xpos_in: wp.array2d[wp.vec3],
    xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    crb_in: wp.array2d[vec10],
    M_in: wp.array2d[float],
    tree_awake_in: wp.array2d[int],
    qfrc_smooth_in: wp.array2d[float],
    efm_c_in: wp.array2d[float],
    efm_diag_in: wp.array2d[float],
    efm_fluid_in: wp.array2d[float],
    efm_ca_in: wp.array2d[float],
    efm_ts_in: wp.array2d[float],
    efm_as_in: wp.array2d[float],
    contact_worldid_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    nacon_in: wp.array[int],
    # In:
    efm_con_dof_in: wp.array2d[int],
    efm_con_scale_in: wp.array[float],
    efm_con_nnz_in: wp.array[int],
    # Out:
    out: wp.array2d[float],
  ):
    worldid, freeid = wp.tid()
    bodyid = body_freeadr[freeid]
    dof_adr = body_dofadr[bodyid]

    # Sleep guard matching mjd_freeMhat: skip if sleeping
    treeid = dof_treeid[dof_adr]
    if (opt_enableflags & EnableBit.SLEEP) and (tree_awake_in[worldid, treeid] == 0):
      return

    # Decoupling guards (mjd_freeGyroPossible): skip if K, constraints, or rank-1 touch DOFs
    for r in range(6):
      if efm_K_rownnz[dof_adr + r] != 0:
        return
    nefc = nefc_in[worldid]
    for i in range(nefc):
      if wp.static(is_sparse):
        radr = efc_J_rowadr_in[worldid, i]
        rnnz = efc_J_rownnz_in[worldid, i]
        for j in range(rnnz):
          c = efc_J_colind_in[worldid, 0, radr + j]
          if c >= dof_adr and c < dof_adr + 6:
            if efc_J_in[worldid, 0, radr + j] != 0.0:
              return
      else:
        for r in range(6):
          if efc_J_in[worldid, i, dof_adr + r] != 0.0:
            return
    for t in range(ntendon):
      if efm_ts_in[worldid, t] != 0.0:
        radr = ten_J_rowadr[t]
        for j in range(ten_J_rownnz[t]):
          c = ten_J_colind[radr + j]
          if c >= dof_adr and c < dof_adr + 6:
            return
    for a in range(nactuator):
      if efm_as_in[worldid, a] != 0.0:
        radr = moment_rowadr_in[worldid, a]
        for j in range(moment_rownnz_in[worldid, a]):
          c = moment_colind_in[worldid, radr + j]
          if c >= dof_adr and c < dof_adr + 6:
            return
    if wp.static(has_flex_passive):
      nacon = nacon_in[0]
      for cid in range(nacon):
        if contact_worldid_in[cid] == worldid and efm_con_scale_in[cid] != 0.0:
          nnz = efm_con_nnz_in[cid]
          for j in range(nnz):
            c = efm_con_dof_in[cid, j]
            if c >= dof_adr and c < dof_adr + 6:
              return

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    crb = crb_in[worldid, bodyid]
    subtree_mass = crb[9]
    Iw = wp.mat33(
      crb[0],
      crb[3],
      crb[4],
      crb[3],
      crb[1],
      crb[5],
      crb[4],
      crb[5],
      crb[2],
    )
    qvel_rot = wp.vec3(
      qvel_in[worldid, dof_adr + 3],
      qvel_in[worldid, dof_adr + 4],
      qvel_in[worldid, dof_adr + 5],
    )

    A = mat66(0.0)
    for r in range(6):
      row = dof_adr + r
      start = M_rowadr[row]
      nnz = M_rownnz[row]
      for k in range(nnz):
        c = M_colind[start + k] - dof_adr
        val = M_in[worldid, start + k]
        A[r, c] = val
        A[c, r] = val
      A[r, r] += efm_diag_in[worldid, row]

    if wp.static(has_fluid):
      for r in range(6):
        row = dof_adr + r
        start = M_rowadr[row]
        nnz = M_rownnz[row]
        for k in range(nnz):
          c = M_colind[start + k] - dof_adr
          val = efm_fluid_in[worldid, start + k]
          A[r, c] += val
          if c != r:
            A[c, r] += val

    s = subtree_com_in[worldid, bodyid] - xpos_in[worldid, bodyid]
    lin, rot = math.free_bias_vel_blocks(subtree_mass, xmat_in[worldid, bodyid], Iw, s, qvel_rot)
    h_mass = -timestep * subtree_mass
    for r in range(3):
      for c in range(3):
        A[r, 3 + c] += h_mass * lin[r, c]
        A[3 + r, 3 + c] += timestep * rot[r, c]

    if wp.static(is_inverse):
      for r in range(6):
        row_sum = float(0.0)
        for c in range(6):
          row_sum += A[r, c] * qacc_in[worldid, dof_adr + c]
        out[worldid, dof_adr + r] = row_sum
    else:
      A_fact, pivot, ok = math.lu_factor_6x6(A)
      if ok:
        b_vec = vec6(0.0)
        for r in range(6):
          dof = dof_adr + r
          b_vec[r] = qfrc_smooth_in[worldid, dof] + efm_c_in[worldid, dof] + efm_ca_in[worldid, dof]
        x_vec = math.lu_solve_6x6(A_fact, pivot, b_vec)
        for r in range(6):
          out[worldid, dof_adr + r] = x_vec[r]

  return _discrete_free_gyro


def _launch_discrete_free_gyro(m: Model, d: Data, out: wp.array2d[float], qacc: wp.array2d[float], is_inverse: bool):
  if m.opt.integrator != IntegratorType.DISCRETE or not m.flex_interp_assemblable or m.has_non_simple_flex:
    return
  has_fluid = m.has_fluid and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER))
  if m.has_flex_passive and m.body_freeadr.size > 0:
    efm_con_dof, _, efm_con_scale, _, efm_con_nnz = passive.build_efm_contact(m, d)
  else:
    efm_con_dof = wp.empty((0, 0), dtype=int, device=d.qacc.device)
    efm_con_scale = wp.empty(0, dtype=float, device=d.qacc.device)
    efm_con_nnz = wp.empty(0, dtype=int, device=d.qacc.device)
  wp.launch(
    _discrete_free_gyro_kernel(has_fluid, m.has_flex_passive, is_inverse, m.is_sparse),
    dim=(d.nworld, m.body_freeadr.size),
    inputs=[
      m.nactuator,
      m.ntendon,
      m.opt.timestep,
      m.opt.enableflags,
      m.body_dofadr,
      m.dof_treeid,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.efm_K_rownnz,
      m.body_freeadr,
      d.nefc,
      d.qvel,
      qacc,
      d.xpos,
      d.xmat,
      d.subtree_com,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.crb,
      d.M,
      d.tree_awake,
      d.qfrc_smooth,
      d.efm_c,
      d.efm_diag,
      d.efm_fluid,
      d.efm_ca,
      d.efm_ts,
      d.efm_as,
      d.contact.worldid,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.nacon,
      efm_con_dof,
      efm_con_scale,
      efm_con_nnz,
    ],
    outputs=[out],
  )


def implicit_qlu(m: Model, d: Data, qLU: Optional[wp.array2d[float]] = None):
  """Sets qLU (d.qLU if None) = M - dt * qDeriv in D-structure, including RNE derivatives."""
  qLU = d.qLU if qLU is None else qLU

  # Smooth velocity derivatives into M-structure
  qH_M = wp.empty((d.nworld, m.nC), dtype=float)
  derivative.deriv_smooth_vel(m, d, qH_M)

  # Map qH_M (M-structure) to qLU (D-structure) via mapM2D.
  wp.launch(
    _map_m2d,
    dim=(d.nworld, m.nD),
    inputs=[m.mapM2D, qH_M],
    outputs=[qLU],
  )

  # Compute RNE derivatives, scale by timestep, and add in-place to qLU.
  # Note: inverse dynamics Coriolis enters equations of motion as -qfrc_bias,
  # so M - dt * df/dv = M + dt * d(qfrc_bias)/dv.
  derivative.deriv_rne_vel(m, d, qLU)


@event_scope
def implicit(m: Model, d: Data):
  """Integrates fully implicit in velocity."""
  if m.opt.integrator == IntegratorType.IMPLICIT:
    implicit_qlu(m, d)

    # Factorize and solve: qacc = qLU \ Ma
    qacc = wp.empty((d.nworld, m.nv), dtype=float)
    smooth.factor_solve_lu(m, d, d.qLU, qacc, d.efc.Ma)
    implicit_free_body(m, d, d.efc.Ma, qacc, solve=True)
    _advance(m, d, qacc)
  elif (m.opt.disableflags & (DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)) != (
    DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  ):
    # qDeriv is in M-structure; the scratch qLD matches d.qLD (per-block).
    qDeriv = wp.empty((d.nworld, m.nC), dtype=float)
    qLD = wp.empty_like(d.qLD)
    qLDiagInv = wp.empty((d.nworld, m.nv), dtype=float)
    derivative.deriv_smooth_vel(m, d, qDeriv)
    wp.launch(
      _implicit_free_body_reset_m,
      dim=(d.nworld, m.body_freeadr.size),
      inputs=[
        m.body_dofadr,
        m.M_rownnz,
        m.M_rowadr,
        m.body_freeadr,
        d.M,
      ],
      outputs=[qDeriv],
    )
    qacc = wp.empty((d.nworld, m.nv), dtype=float)
    smooth.factor_solve_i(m, d, qDeriv, qLD, qLDiagInv, qacc, d.efc.Ma)
    implicit_free_body(m, d, d.efc.Ma, qacc, solve=True)
    _advance(m, d, qacc)
  else:
    _advance(m, d, d.qacc)


@event_scope
def fwd_kinematics(m: Model, d: Data):
  """Kinematics-dependent computations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
  """
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.flex(m, d)
  smooth.tendon(m, d)

  sleep_enabled = bool(m.opt.enableflags & EnableBit.SLEEP) and not bool(m.opt.disableflags & DisableBit.ISLAND)
  if sleep_enabled and m.ntendon > 0:
    sleep.wake_tendon(m, d)
    sleep.update_sleep_trees(m, d)


@event_scope
def fwd_position(m: Model, d: Data, factorize: bool = True):
  """Position-dependent computations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize inertia matrix.
  """
  fwd_kinematics(m, d)

  sleep_enabled = bool(m.opt.enableflags & EnableBit.SLEEP) and not bool(m.opt.disableflags & DisableBit.ISLAND)

  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  if factorize:
    smooth.factor_m(m, d)
  if m.opt.run_collision_detection:
    if sleep_enabled:
      # pass 1
      collision_driver.collision(m, d)
      # wake any sleeping tree touched by an awake one
      sleep.wake_collision(m, d)
      # snapshot the awake state pass 1 used, before update_sleep overwrites it. a body is "newly
      # awakened" if it was asleep here but awake after update_sleep below.
      awake_prev = wp.clone(d.body_awake)
      sleep.update_sleep(m, d)
      # pass 2: passing awake_prev runs the incremental pass, emitting only pairs involving a
      # newly-awakened body and appending them to the pass-1 buffer.
      collision_driver.collision(m, d, awake_prev=awake_prev)
    else:
      collision_driver.collision(m, d)

  constraint.make_constraint(m, d)

  if sleep_enabled:
    if m.neq > 0:
      sleep.wake_equality(m, d)
    sleep.update_sleep(m, d)

  if sleep_enabled:
    island.island(m, d)
  smooth.transmission(m, d)
  if m.opt.integrator == IntegratorType.DISCRETE:
    derivative.eff_build(m, d)


@wp.kernel
def _actuator_velocity(
  # Data in:
  qvel_in: wp.array2d[float],
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # Data out:
  actuator_velocity_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  vel = float(0.0)
  for i in range(rownnz):
    sparseid = rowadr + i
    colind = moment_colind_in[worldid, sparseid]
    vel += actuator_moment_in[worldid, sparseid] * qvel_in[worldid, colind]

  actuator_velocity_out[worldid, actid] = vel


@wp.kernel
def _tendon_velocity(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  ten_J_in: wp.array2d[float],
  # Data out:
  ten_velocity_out: wp.array2d[float],
):
  worldid, tenid = wp.tid()

  velocity = float(0.0)
  rownnz = ten_J_rownnz[tenid]
  rowadr = ten_J_rowadr[tenid]
  for i in range(rownnz):
    sparseid = rowadr + i
    J = ten_J_in[worldid, sparseid]
    if J != 0.0:
      colind = ten_J_colind[sparseid]
      velocity += J * qvel_in[worldid, colind]

  ten_velocity_out[worldid, tenid] = velocity


@event_scope
def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""
  wp.launch(
    _actuator_velocity,
    dim=(d.nworld, m.nactuator),
    inputs=[d.qvel, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment],
    outputs=[d.actuator_velocity],
  )

  wp.launch(
    _tendon_velocity,
    dim=(d.nworld, m.ntendon),
    inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.qvel, d.ten_J],
    outputs=[d.ten_velocity],
  )

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)
  smooth.tendon_bias(m, d, d.qfrc_bias)
  if m.opt.integrator == IntegratorType.DISCRETE:
    derivative.eff_shift(m, d)


@wp.kernel
def _actuator_force(
  # Model:
  na: int,
  ntendon: int,
  opt_timestep: wp.array[float],
  jnt_type: wp.array[int],
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
  actuator_gear: wp.array2d[wp.spatial_vector],
  actuator_acc0: wp.array2d[float],
  actuator_lengthrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  # In:
  dsbl_clampctrl: int,
  # Data out:
  act_dot_out: wp.array2d[float],
  actuator_force_out: wp.array2d[float],
):
  worldid, uid = wp.tid()

  actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]

  uadr = actuator_ctrladr[uid]
  ctrlnum = actuator_ctrlnum[uid]
  ctrl = 0.0
  if ctrlnum > 0:
    ctrl = ctrl_in[worldid, uadr]
    if actuator_ctrllimited[uadr] and not dsbl_clampctrl:
      ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uadr]
      ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
  ctrl_act = ctrl
  u_first = ctrl

  dyntype = actuator_dyntype[uid]
  gaintype = actuator_gaintype[uid]
  biastype = actuator_biastype[uid]
  trntype = actuator_trntype[uid]
  trnid = actuator_trnid[uid]
  gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]
  biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]
  gear = actuator_gear[worldid % actuator_gear.shape[0], uid]
  period = _wrap_period(
    jnt_type,
    trntype,
    dyntype,
    gaintype,
    biastype,
    trnid,
    gainprm,
    biasprm,
    gear,
  )

  act_first = actuator_actadr[uid]
  if na and act_first >= 0:
    act_last = act_first + actuator_actnum[uid] - 1
    dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]

    if dyntype == DynType.INTEGRATOR:
      act_dot = ctrl
    elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:
      act = act_in[worldid, act_last]
      # rotational setpoint: filter toward the representative nearest the current act
      u_filter = ctrl
      if period > 0.0:
        u_filter -= period * wp.round((u_filter - act) / period)
      act_dot = (u_filter - act) / wp.max(dynprm[0], MJ_MINVAL)
    elif dyntype == DynType.MUSCLE:
      dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]
      act = act_in[worldid, act_last]
      act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)
    elif dyntype == DynType.DCMOTOR:
      slots = util_misc.dcmotor_slots(dynprm, gainprm)
      adr = act_first

      act_dot = 0.0

      # slew rate
      if slots[0] >= 0:
        u_prev = act_in[worldid, adr]
        slew_s = dynprm[7]
        slew = slew_s * opt_timestep[worldid % opt_timestep.shape[0]]
        u_eff = wp.clamp(u_first, u_prev - slew, u_prev + slew)
        act_dot = (u_eff - u_prev) / opt_timestep[worldid % opt_timestep.shape[0]]
        act_dot_out[worldid, adr] = act_dot
        u_first = u_eff
        adr += 1

      # integral
      x_I = 0.0
      if slots[1] >= 0:
        x_I = act_in[worldid, adr]
        Imax = dynprm[8]
        act_dot = u_first - actuator_length_in[worldid, uid]

        if Imax > 0.0:
          if x_I >= Imax:
            act_dot = wp.min(act_dot, 0.0)
          elif x_I <= -Imax:
            act_dot = wp.max(act_dot, 0.0)

        act_dot_out[worldid, adr] = act_dot
        adr += 1

      # voltage
      V = 0.0
      if ctrlnum > 0:
        V = util_misc.dcmotor_voltage(
          ctrl_in,
          worldid,
          uadr,
          actuator_ctrlspec[uid],
          u_first,
          actuator_length_in[worldid, uid],
          actuator_velocity_in[worldid, uid],
          x_I,
          gainprm,
        )

      # temperature
      R = gainprm[0]
      K = gainprm[1]
      te = wp.max(MJ_MINVAL, dynprm[0])

      if slots[2] >= 0:
        RT = dynprm[2]
        C = dynprm[3]
        Ta = dynprm[4]
        alpha = gainprm[2]
        T0 = gainprm[3]
        T = act_in[worldid, adr]
        R_eff = R * (1.0 + alpha * (T + Ta - T0))

        current = (V - K * actuator_velocity_in[worldid, uid]) / R_eff
        if slots[4] >= 0:
          current = act_in[worldid, act_last]

        act_dot = (R_eff * current * current - T / RT) / C
        act_dot_out[worldid, adr] = act_dot
        adr += 1
        R = R_eff

      # bristle
      if slots[3] >= 0:
        sigma0 = dynprm[5]
        biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]
        F_C = biasprm[3]
        F_S = biasprm[4]
        v_S = biasprm[5]
        z = act_in[worldid, adr]
        g = util_misc.lugre_stribeck(actuator_velocity_in[worldid, uid], F_C, F_S, v_S)
        a = -sigma0 * wp.abs(actuator_velocity_in[worldid, uid]) / wp.max(MJ_MINVAL, g)
        act_dot = a * z + actuator_velocity_in[worldid, uid]
        act_dot_out[worldid, adr] = act_dot
        adr += 1

      # current
      if slots[4] >= 0:
        dimax = dynprm[1]
        act_dot = (V / R - K / R * actuator_velocity_in[worldid, uid] - act_in[worldid, act_last]) / te
        if dimax > 0.0:
          act_dot = wp.clamp(act_dot, -dimax, dimax)
        act_dot_out[worldid, act_last] = act_dot

    elif dyntype == DynType.USER:
      act_dot = 0.0  # set by act_dyn_callback
    else:  # DynType.NONE
      act_dot = 0.0

    act_dot_out[worldid, act_last] = act_dot

    if actuator_actearly[uid]:
      if dyntype == DynType.INTEGRATOR or dyntype == DynType.NONE or dyntype == DynType.DCMOTOR:
        act = act_in[worldid, act_last]

      if dyntype == DynType.DCMOTOR:
        gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]
        slots = util_misc.dcmotor_slots(dynprm, gainprm)
        offset = actuator_actnum[uid] - 1

        if offset == slots[4]:  # current
          te = wp.max(MJ_MINVAL, dynprm[0])
          ctrl_act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[worldid % opt_timestep.shape[0]] / te))
        elif offset == slots[3]:  # bristle
          sigma0 = dynprm[5]
          biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]
          F_C = biasprm[3]
          F_S = biasprm[4]
          v_S = biasprm[5]
          velocity = actuator_velocity_in[worldid, uid]
          g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)
          a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)
          h = opt_timestep[worldid % opt_timestep.shape[0]]
          exp_ah = wp.exp(a * h)
          int_h = h
          if wp.abs(a) > MJ_MINVAL:
            int_h = (exp_ah - 1.0) / a
          ctrl_act = exp_ah * act + int_h * velocity
        elif offset == slots[1]:  # integral
          ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]
          Imax = dynprm[8]
          if Imax > 0.0:
            ctrl_act = wp.clamp(ctrl_act, -Imax, Imax)
        else:  # temperature or slew or default
          ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]
      else:
        ctrl_act = next_act(
          opt_timestep[worldid % opt_timestep.shape[0]],
          dyntype,
          dynprm,
          actuator_actrange[worldid % actuator_actrange.shape[0], uid],
          act,
          act_dot,
          1.0,
          actuator_actlimited[uid],
        )
    else:
      ctrl_act = act_in[worldid, act_last]

  length = actuator_length_in[worldid, uid]
  velocity = actuator_velocity_in[worldid, uid]

  # gain
  dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]

  gain = 0.0
  if gaintype == GainType.FIXED:
    gain = gainprm[0]
  elif gaintype == GainType.AFFINE:
    gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity
  elif gaintype == GainType.MUSCLE:
    acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]
    lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]
    gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)
  elif gaintype == GainType.DCMOTOR:
    K = gainprm[1]
    te = dynprm[0]
    R = util_misc.dcmotor_resistance(act_in, worldid, act_first, dynprm, gainprm)

    gain = K if te > 0.0 else K / R

    if te <= 0.0:
      if ctrlnum == 0:
        ctrl_act = 0.0
      elif (actuator_ctrlspec[uid] & 7) != 0:
        x_I = 0.0
        if na and act_first >= 0:
          slots = util_misc.dcmotor_slots(dynprm, gainprm)
          if slots[1] >= 0:
            x_I = act_in[worldid, act_first + slots[1]]
        ctrl_act = util_misc.dcmotor_voltage(
          ctrl_in,
          worldid,
          uadr,
          actuator_ctrlspec[uid],
          u_first,
          length,
          velocity,
          x_I,
          gainprm,
        )
      else:
        ctrl_act = u_first
  # GainType.USER: gain stays 0, modified by act_gain_callback

  # bias
  bias = 0.0  # BiasType.NONE or BiasType.USER (modified by act_bias_callback)
  if biastype == BiasType.AFFINE:
    bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity
  elif biastype == BiasType.MUSCLE:
    acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]
    lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]
    bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)
  elif biastype == BiasType.DCMOTOR:
    if dynprm[0] <= 0.0:
      K = gainprm[1]
      bias -= gain * K * velocity

  if period > 0.0:
    err = ctrl_act - length
    ctrl_act -= period * wp.round(err / period)

  force = gain * ctrl_act + bias

  if not (ntendon > 0 and trntype == TrnType.TENDON):
    if actuator_forcelimited[uid]:
      forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]
      force = wp.clamp(force, forcerange[0], forcerange[1])

    # add DC motor mechanical forces (not subject to current limits)
    if biastype == BiasType.DCMOTOR:
      # cogging torque
      A = biasprm[0]
      if A != 0.0:
        Np = biasprm[1]
        phi = biasprm[2]
        force += A * wp.sin(Np * length + phi)

      # LuGre friction
      sigma0 = dynprm[5]
      if na and act_first >= 0 and sigma0 > 0.0:
        sigma1 = dynprm[6]
        slots = util_misc.dcmotor_slots(dynprm, gainprm)
        adr = act_first + slots[3]  # slots[3] is bristle
        z = act_in[worldid, adr]
        z_dot = act_dot_out[worldid, adr]
        force -= sigma0 * z + sigma1 * z_dot

  actuator_force_out[worldid, uid] = force


@cache_kernel
def _tendon_actuator_force(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    # Data in:
    actuator_force_in: wp.array2d[float],
    # Out:
    ten_actfrc_out: wp.array2d[float],
  ):
    worldid, actid = wp.tid()

    if actuator_trntype[actid] == TrnType.TENDON:
      tenid = actuator_trnid[actid][0]
      # TODO(team): only compute for tendons with force limits?
      wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])

  return kernel


@wp.kernel
def _tendon_actuator_force_clamp(
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
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # In:
  ten_actfrc_in: wp.array2d[float],
  # Data out:
  actuator_force_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  if actuator_trntype[actid] != TrnType.TENDON:
    return

  tenid = actuator_trnid[actid][0]
  ten_limited = tendon_actfrclimited[tenid]
  frc_limited = actuator_forcelimited[actid]
  is_dcmotor = actuator_biastype[actid] == BiasType.DCMOTOR
  if not ten_limited and not frc_limited and not is_dcmotor:
    return

  force = actuator_force_in[worldid, actid]
  if ten_limited:
    ten_actfrc = ten_actfrc_in[worldid, tenid]
    if ten_actfrc != 0.0:
      actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]
      if ten_actfrc < actfrcrange[0]:
        force *= actfrcrange[0] / ten_actfrc
      elif ten_actfrc > actfrcrange[1]:
        force *= actfrcrange[1] / ten_actfrc

  if frc_limited:
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
    force = wp.clamp(force, forcerange[0], forcerange[1])

  # add DC motor mechanical forces (not subject to current or tendon force limits)
  if is_dcmotor:
    force += util_misc.dcmotor_mech_force(
      act_in,
      act_dot_in,
      worldid,
      actuator_actadr[actid],
      actuator_length_in[worldid, actid],
      actuator_dynprm[worldid % actuator_dynprm.shape[0], actid],
      actuator_gainprm[worldid % actuator_gainprm.shape[0], actid],
      actuator_biasprm[worldid % actuator_biasprm.shape[0], actid],
    )

  actuator_force_out[worldid, actid] = force


@cache_kernel
def _qfrc_actuator(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Data in:
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    actuator_force_in: wp.array2d[float],
    # Data out:
    qfrc_actuator_out: wp.array2d[float],
  ):
    worldid, actid = wp.tid()

    rownnz = moment_rownnz_in[worldid, actid]
    rowadr = moment_rowadr_in[worldid, actid]
    act_force = actuator_force_in[worldid, actid]

    for i in range(rownnz):
      sparseid = rowadr + i
      colind = moment_colind_in[worldid, sparseid]
      qfrc = actuator_moment_in[worldid, sparseid] * act_force
      wp.atomic_add(qfrc_actuator_out[worldid], colind, qfrc)

  return kernel


@wp.kernel
def _qfrc_actuator_gravcomp_limits(
  # Model:
  jnt_actfrclimited: wp.array[bool],
  jnt_actgravcomp: wp.array[int],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  dof_jntid: wp.array[int],
  # Data in:
  qfrc_gravcomp_in: wp.array2d[float],
  qfrc_actuator_in: wp.array2d[float],
  # In:
  gravity_enabled: bool,
  # Data out:
  qfrc_actuator_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  jntid = dof_jntid[dofid]

  qfrc = qfrc_actuator_in[worldid, dofid]

  # actuator-level gravity compensation, skip if added as passive force
  if gravity_enabled and jnt_actgravcomp[jntid]:
    qfrc += qfrc_gravcomp_in[worldid, dofid]

  # limits
  if jnt_actfrclimited[jntid]:
    frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]
    qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])

  qfrc_actuator_out[worldid, dofid] = qfrc


@event_scope
def fwd_actuation(m: Model, d: Data):
  """Actuation-dependent computations."""
  if not m.nactuator or (m.opt.disableflags & DisableBit.ACTUATION):
    d.act_dot.zero_()
    d.qfrc_actuator.zero_()
    d.actuator_force.zero_()
  else:
    # read delayed ctrl (or direct copy if no delay)
    if m.nhistory > 0:
      ctrl = wp.empty((d.nworld, m.nu), dtype=float)
      history.read_ctrl_delayed(m, d, ctrl)
    else:
      ctrl = d.ctrl

    wp.launch(
      _actuator_force,
      dim=(d.nworld, m.nactuator),
      inputs=[
        m.na,
        m.ntendon,
        m.opt.timestep,
        m.jnt_type,
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
        m.actuator_gear,
        m.actuator_acc0,
        m.actuator_lengthrange,
        d.act,
        ctrl,
        d.actuator_length,
        d.actuator_velocity,
        m.opt.disableflags & DisableBit.CLAMPCTRL,
      ],
      outputs=[d.act_dot, d.actuator_force],
    )

    if m.callback.act_dyn:
      m.callback.act_dyn(m, d)
    if m.callback.act_gain:
      m.callback.act_gain(m, d)
    if m.callback.act_bias:
      m.callback.act_bias(m, d)

    det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
    if m.ntendon:
      # total actuator force at tendon
      ten_actfrc = wp.zeros((d.nworld, m.ntendon), dtype=float)
      wp.launch(
        _tendon_actuator_force(det_atomics),
        dim=(d.nworld, m.nactuator),
        inputs=[m.actuator_trntype, m.actuator_trnid, d.actuator_force],
        outputs=[ten_actfrc],
      )

      wp.launch(
        _tendon_actuator_force_clamp,
        dim=(d.nworld, m.nactuator),
        inputs=[
          m.tendon_actfrclimited,
          m.tendon_actfrcrange,
          m.actuator_trntype,
          m.actuator_biastype,
          m.actuator_actadr,
          m.actuator_trnid,
          m.actuator_dynprm,
          m.actuator_gainprm,
          m.actuator_biasprm,
          m.actuator_forcelimited,
          m.actuator_forcerange,
          d.act,
          d.act_dot,
          d.actuator_length,
          d.actuator_force,
          ten_actfrc,
        ],
        outputs=[d.actuator_force],
      )

    # TODO(team): optimize performance
    d.qfrc_actuator.zero_()
    wp.launch(
      _qfrc_actuator(m.nv if det_atomics else 0, det_atomics),
      dim=(d.nworld, m.nactuator),
      inputs=[
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
        d.actuator_moment,
        d.actuator_force,
      ],
      outputs=[d.qfrc_actuator],
    )
    gravity_enabled = not (m.opt.disableflags & DisableBit.GRAVITY)
    wp.launch(
      _qfrc_actuator_gravcomp_limits,
      dim=(d.nworld, m.nv),
      inputs=[
        m.jnt_actfrclimited,
        m.jnt_actgravcomp,
        m.jnt_actfrcrange,
        m.dof_jntid,
        d.qfrc_gravcomp,
        d.qfrc_actuator,
        gravity_enabled,
      ],
      outputs=[d.qfrc_actuator],
    )
  if m.opt.integrator == IntegratorType.DISCRETE:
    derivative.eff_actuation(m, d)
    constraint.regularize_constraint(m, d)


@cache_kernel
def _qfrc_smooth(enable_sleep: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    body_treeid: wp.array[int],
    dof_bodyid: wp.array[int],
    # Data in:
    qfrc_applied_in: wp.array2d[float],
    tree_awake_in: wp.array2d[int],
    qfrc_bias_in: wp.array2d[float],
    qfrc_passive_in: wp.array2d[float],
    qfrc_actuator_in: wp.array2d[float],
    # Data out:
    qfrc_smooth_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()

    if wp.static(enable_sleep):
      bodyid = dof_bodyid[dofid]
      tree = body_treeid[bodyid]
      if tree >= 0 and tree_awake_in[worldid, tree] == 0:
        qfrc_smooth_out[worldid, dofid] = 0.0
        return

    qfrc_smooth_out[worldid, dofid] = (
      qfrc_passive_in[worldid, dofid]
      - qfrc_bias_in[worldid, dofid]
      + qfrc_actuator_in[worldid, dofid]
      + qfrc_applied_in[worldid, dofid]
    )

  return kernel


@event_scope
def fwd_acceleration(m: Model, d: Data, factorize: bool = False):
  """Add up all non-constraint forces, compute qacc_smooth.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize inertia matrix.
  """
  enable_sleep = bool(m.opt.enableflags & EnableBit.SLEEP) and not bool(m.opt.disableflags & DisableBit.ISLAND)
  wp.launch(
    _qfrc_smooth(enable_sleep),
    dim=(d.nworld, m.nv),
    inputs=[
      m.body_treeid,
      m.dof_bodyid,
      d.qfrc_applied,
      d.tree_awake,
      d.qfrc_bias,
      d.qfrc_passive,
      d.qfrc_actuator,
    ],
    outputs=[d.qfrc_smooth],
  )
  xfrc_accumulate(m, d, d.qfrc_smooth)

  if m.opt.integrator == IntegratorType.DISCRETE:
    derivative.eff_solve(m, d, d.qacc_smooth)
  elif enable_sleep:
    # update the active-DOF set (needs contacts from fwd_position) and solve
    # the smooth acceleration in compacted dense space.
    island.update_active_dofs(m, d)
    solver.smooth_solve_compact(m, d)
  elif factorize:
    smooth.factor_solve_i(m, d, d.M, d.qLD, d.qLDiagInv, d.qacc_smooth, d.qfrc_smooth)
  else:
    smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


def _energy_pos(m: Model, d: Data):
  if m.opt.enableflags & EnableBit.ENERGY:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()


def _energy_vel(m: Model, d: Data):
  if m.opt.enableflags & EnableBit.ENERGY:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)


def _forward(m: Model, d: Data, skip_sensor: bool = False):
  check_discrete(m)
  sleep_enabled = bool(m.opt.enableflags & EnableBit.SLEEP) and not bool(m.opt.disableflags & DisableBit.ISLAND)
  if sleep_enabled:
    sleep.wake(m, d)
    sleep.update_sleep(m, d)

  fwd_position(m, d, factorize=False)
  if not skip_sensor:
    d.sensordata.zero_()
    sensor.sensor_pos(m, d)
  _energy_pos(m, d)

  fwd_velocity(m, d)
  if not skip_sensor:
    sensor.sensor_vel(m, d)
  _energy_vel(m, d)

  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)
  fwd_actuation(m, d)
  fwd_acceleration(m, d, factorize=True)

  solver.solve(m, d)
  if m.opt.integrator == IntegratorType.DISCRETE:
    _launch_discrete_free_gyro(m, d, d.qacc, d.qacc, False)
  if m.opt.run_rne_postconstraint or (
    not skip_sensor and not (m.opt.disableflags & DisableBit.SENSOR) and m.sensor_rne_postconstraint
  ):
    smooth.rne_postconstraint(m, d)
  if not skip_sensor:
    sensor.sensor_acc(m, d, skip_rne_postconstraint=True)


@event_scope
def forward(m: Model, d: Data):
  """Forward dynamics."""
  _forward(m, d)


@event_scope
def discrete(m: Model, d: Data):
  """Advance simulation using discrete integrator."""
  _advance(m, d, d.qacc)


@event_scope
def step(m: Model, d: Data):
  """Advance simulation."""
  forward(m, d)

  if m.opt.integrator == IntegratorType.EULER:
    euler(m, d)
  elif m.opt.integrator == IntegratorType.RK4:
    rungekutta4(m, d)
  elif m.opt.integrator in (IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT):
    implicit(m, d)
  elif m.opt.integrator == IntegratorType.DISCRETE:
    discrete(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")


@event_scope
def step1(m: Model, d: Data):
  """Advance simulation in two phases: before input is set by user."""
  check_discrete(m)
  fwd_position(m, d)
  d.sensordata.zero_()
  sensor.sensor_pos(m, d)

  _energy_pos(m, d)

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  _energy_vel(m, d)

  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)


@event_scope
def step2(m: Model, d: Data):
  """Advance simulation in two phases: after input is set by user."""
  fwd_actuation(m, d)
  fwd_acceleration(m, d)
  solver.solve(m, d)
  if m.opt.integrator == IntegratorType.DISCRETE:
    _launch_discrete_free_gyro(m, d, d.qacc, d.qacc, False)
  if m.opt.run_rne_postconstraint or (not (m.opt.disableflags & DisableBit.SENSOR) and m.sensor_rne_postconstraint):
    smooth.rne_postconstraint(m, d)
  sensor.sensor_acc(m, d, skip_rne_postconstraint=True)

  # integrate with Euler, implicitfast, or discrete
  if m.opt.integrator in (IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT):
    implicit(m, d)
  elif m.opt.integrator == IntegratorType.DISCRETE:
    discrete(m, d)
  else:
    # note: RK4 defaults to Euler
    euler(m, d)
