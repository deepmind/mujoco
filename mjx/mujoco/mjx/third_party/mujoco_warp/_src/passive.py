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

from mujoco.mjx.third_party.mujoco_warp._src import math
from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src import util_misc
from mujoco.mjx.third_party.mujoco_warp._src.types import FLEX_INTERP_MAX_NODES
from mujoco.mjx.third_party.mujoco_warp._src.types import FLEX_STIFFNESS_3D
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import ROD_HINGE_SIZE
from mujoco.mjx.third_party.mujoco_warp._src.types import ContactType
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DeterminismType
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import IntegratorType
from mujoco.mjx.third_party.mujoco_warp._src.types import JointType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import ObjType
from mujoco.mjx.third_party.mujoco_warp._src.types import mat43
from mujoco.mjx.third_party.mujoco_warp._src.types import mat63
from mujoco.mjx.third_party.mujoco_warp._src.types import mat66
from mujoco.mjx.third_party.mujoco_warp._src.types import vec6
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _pow2(val: float) -> float:
  return val * val


@wp.func
def _pow4(val: float) -> float:
  sq = val * val
  return sq * sq


@wp.func
def geom_semiaxes(size: wp.vec3, geom_type: int) -> wp.vec3:  # kernel_analyzer: ignore
  if geom_type == GeomType.SPHERE:
    r = size[0]
    return wp.vec3(r, r, r)

  if geom_type == GeomType.CAPSULE:
    radius = size[0]
    half_length = size[1]
    return wp.vec3(radius, radius, half_length + radius)

  if geom_type == GeomType.CYLINDER:
    radius = size[0]
    half_length = size[1]
    return wp.vec3(radius, radius, half_length)

  # ellipsoid, box, mesh, sdf -> use size directly
  return size


@wp.func
def ellipsoid_max_moment(size: wp.vec3, dir: int) -> float:
  d0 = size[dir]
  d1 = size[(dir + 1) % 3]
  d2 = size[(dir + 2) % 3]
  return wp.static(8.0 / 15.0 * wp.pi) * d0 * _pow4(wp.max(d1, d2))


@wp.kernel
def _spring_damper_dof_passive(
  # Model:
  opt_disableflags: int,
  qpos_spring: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_actuatorid: wp.array[int],
  jnt_plasticadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  # Data out:
  qfrc_spring_out: wp.array2d[float],
  qfrc_damper_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  dofid = jnt_dofadr[jntid]
  jnttype = jnt_type[jntid]
  stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]
  spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]
  has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not (opt_disableflags & DisableBit.SPRING)

  ndof = 1
  if jnttype == JointType.FREE:
    ndof = 6
  elif jnttype == JointType.BALL:
    ndof = 3
  if opt_disableflags & DisableBit.DAMPER:
    for i in range(ndof):
      qfrc_damper_out[worldid, dofid + i] = 0.0
  else:
    dof_damping_id = worldid % dof_damping.shape[0]
    dof_dampingpoly_id = worldid % dof_dampingpoly.shape[0]
    act_damping, act_dpoly = util_misc.actuator_target_damping(
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
    for i in range(ndof):
      damping = dof_damping[dof_damping_id, dofid + i] + act_damping
      dpoly = dof_dampingpoly[dof_dampingpoly_id, dofid + i] + act_dpoly
      if damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0:
        v = qvel_in[worldid, dofid + i]
        qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)
      else:
        qfrc_damper_out[worldid, dofid + i] = 0.0

  if not has_stiffness:
    for i in range(ndof):
      qfrc_spring_out[worldid, dofid + i] = 0.0
    return
  qposid = jnt_qposadr[jntid]
  qpos_spring_id = worldid % qpos_spring.shape[0]
  padr = jnt_plasticadr[jntid]

  if jnttype == JointType.FREE:
    # spring
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
    k = util_misc._poly_force(stiffness, spoly, r, 0)
    qfrc_spring_out[worldid, dofid + 0] = -k * dif[0]
    qfrc_spring_out[worldid, dofid + 1] = -k * dif[1]
    qfrc_spring_out[worldid, dofid + 2] = -k * dif[2]

    rot = wp.quat(
      qpos_in[worldid, qposid + 3],
      qpos_in[worldid, qposid + 4],
      qpos_in[worldid, qposid + 5],
      qpos_in[worldid, qposid + 6],
    )
    rot = wp.normalize(rot)
    dif = math.quat_sub(rot, ref)
    r_rot = wp.length(dif)
    k_rot = util_misc._poly_force(stiffness, spoly, r_rot, 0)
    qfrc_spring_out[worldid, dofid + 3] = -k_rot * dif[0]
    qfrc_spring_out[worldid, dofid + 4] = -k_rot * dif[1]
    qfrc_spring_out[worldid, dofid + 5] = -k_rot * dif[2]

  elif jnttype == JointType.BALL:
    # spring
    rot = wp.quat(
      qpos_in[worldid, qposid + 0],
      qpos_in[worldid, qposid + 1],
      qpos_in[worldid, qposid + 2],
      qpos_in[worldid, qposid + 3],
    )
    rot = wp.normalize(rot)
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
    dif = math.quat_sub(rot, ref)
    r = wp.length(dif)
    k = util_misc._poly_force(stiffness, spoly, r, 0)
    qfrc_spring_out[worldid, dofid + 0] = -k * dif[0]
    qfrc_spring_out[worldid, dofid + 1] = -k * dif[1]
    qfrc_spring_out[worldid, dofid + 2] = -k * dif[2]

  else:  # mjJNT_SLIDE, mjJNT_HINGE
    # spring
    ref_scalar = plastic_in[worldid, padr] if padr >= 0 else qpos_spring[qpos_spring_id, qposid]
    fdif = qpos_in[worldid, qposid] - ref_scalar
    qfrc_spring_out[worldid, dofid] = -fdif * util_misc._poly_force(stiffness, spoly, fdif, 0)


@cache_kernel
def _spring_damper_tendon_passive(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
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
    # In:
    dsbl_spring: bool,
    dsbl_damper: bool,
    # Data out:
    qfrc_spring_out: wp.array2d[float],
    qfrc_damper_out: wp.array2d[float],
  ):
    worldid, tenid, dofid_sparse = wp.tid()

    stiffness = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]
    spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]
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

    has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not dsbl_spring
    has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not dsbl_damper

    if not has_stiffness and not has_damping:
      return

    rownnz = ten_J_rownnz[tenid]
    if dofid_sparse >= rownnz:
      return
    rowadr = ten_J_rowadr[tenid]
    sparseid = rowadr + dofid_sparse
    J = ten_J_in[worldid, sparseid]
    dofid = ten_J_colind[sparseid]

    if has_stiffness:
      # compute spring force along tendon
      length = ten_length_in[worldid, tenid]
      padr = tendon_plasticadr[tenid]
      if padr >= 0:
        lower = plastic_in[worldid, padr + 0]
        upper = plastic_in[worldid, padr + 1]
      else:
        lengthspring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]
        lower = lengthspring[0]
        upper = lengthspring[1]

      x = wp.where(
        length > upper,
        length - upper,
        wp.where(length < lower, length - lower, 0.0),
      )
      frc_spring = -x * util_misc._poly_force(stiffness, spoly, x, 0)

      # transform to joint torque
      wp.atomic_add(qfrc_spring_out[worldid], dofid, J * frc_spring)

    if has_damping:
      # compute damper force along tendon
      v = ten_velocity_in[worldid, tenid]
      frc_damper = -v * util_misc._poly_force(damping, dpoly, v, 1)

      # transform to joint torque
      wp.atomic_add(qfrc_damper_out[worldid], dofid, J * frc_damper)

  return kernel


@cache_kernel
def _spring_damper_flexedge_passive(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    flexedge_length0: wp.array[float],
    flex_edgestiffness: wp.array[float],
    flex_edgedamping: wp.array[float],
    flexedge_rigid: wp.array[bool],
    flexedge_J_rownnz: wp.array[int],
    flexedge_J_rowadr: wp.array[int],
    flexedge_J_colind: wp.array[int],
    flex_edgeflexid: wp.array[int],
    # Data in:
    flexedge_J_in: wp.array2d[float],
    flexedge_length_in: wp.array2d[float],
    flexedge_velocity_in: wp.array2d[float],
    # In:
    dsbl_spring: bool,
    dsbl_damper: bool,
    # Data out:
    qfrc_spring_out: wp.array2d[float],
    qfrc_damper_out: wp.array2d[float],
  ):
    worldid, edgeid = wp.tid()

    if flexedge_rigid[edgeid]:
      return

    f = flex_edgeflexid[edgeid]

    stiffness = float(0.0)
    if not dsbl_spring:
      stiffness = flex_edgestiffness[f]

    damping = float(0.0)
    if not dsbl_damper:
      damping = flex_edgedamping[f]

    if stiffness == 0.0 and damping == 0.0:
      return

    rownnz = flexedge_J_rownnz[edgeid]
    if rownnz == 0:
      return

    frc_spring = float(0.0)
    if stiffness != 0.0:
      frc_spring = stiffness * (flexedge_length0[edgeid] - flexedge_length_in[worldid, edgeid])

    frc_damper = float(0.0)
    if damping != 0.0:
      frc_damper = -damping * flexedge_velocity_in[worldid, edgeid]

    if frc_spring == 0.0 and frc_damper == 0.0:
      return

    rowadr = flexedge_J_rowadr[edgeid]
    for k in range(rownnz):
      sparseid = rowadr + k
      colind = flexedge_J_colind[sparseid]
      J = flexedge_J_in[worldid, sparseid]
      if frc_spring != 0.0:
        wp.atomic_add(qfrc_spring_out[worldid], colind, J * frc_spring)
      if frc_damper != 0.0:
        wp.atomic_add(qfrc_damper_out[worldid], colind, J * frc_damper)

  return kernel


@cache_kernel
def _gravity_force(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_gravity: wp.array[wp.vec3],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mass: wp.array2d[float],
    body_gravcomp: wp.array2d[float],
    dof_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    # Data out:
    qfrc_gravcomp_out: wp.array2d[float],
  ):
    worldid, bodyid, dofid = wp.tid()
    bodyid += 1  # skip world body
    gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]
    gravity = opt_gravity[worldid % opt_gravity.shape[0]]

    if gravcomp:
      force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp
      pos = xipos_in[worldid, bodyid]
      jac, _ = support.jac_dof(
        body_parentid,
        body_rootid,
        dof_bodyid,
        body_isdofancestor,
        subtree_com_in,
        cdof_in,
        pos,
        bodyid,
        dofid,
        worldid,
      )

      wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))

  return kernel


@wp.kernel
def _fluid_force(
  # Model:
  opt_wind: wp.array[wp.vec3],
  opt_density: wp.array[float],
  opt_viscosity: wp.array[float],
  body_rootid: wp.array[int],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  body_fluid_ellipsoid: wp.array[bool],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # Out:
  fluid_applied_out: wp.array2d[wp.spatial_vector],
):
  """Computes body-space fluid forces for both inertia-box and ellipsoid models."""
  worldid, bodyid = wp.tid()
  zero_force = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))

  if bodyid == 0:
    fluid_applied_out[worldid, bodyid] = zero_force
    return

  # skip bodies with negligible mass
  mass = body_mass[worldid % body_mass.shape[0], bodyid]
  if mass < MJ_MINVAL:
    fluid_applied_out[worldid, bodyid] = zero_force
    return

  wind = opt_wind[worldid % opt_wind.shape[0]]
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]

  # Body kinematics
  xipos = xipos_in[worldid, bodyid]
  rot = ximat_in[worldid, bodyid]
  rotT = wp.transpose(rot)
  cvel = cvel_in[worldid, bodyid]
  ang_global = wp.spatial_top(cvel)
  lin_global = wp.spatial_bottom(cvel)
  subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]
  lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

  if body_fluid_ellipsoid[bodyid]:
    force_global = wp.vec3(0.0)
    torque_global = wp.vec3(0.0)

    start = body_geomadr[bodyid]
    count = body_geomnum[bodyid]

    for i in range(count):
      geomid = start + i
      coef = geom_fluid[geomid, 0]
      if coef <= 0.0:
        continue

      size = geom_size[worldid % geom_size.shape[0], geomid]
      semiaxes = geom_semiaxes(size, geom_type[geomid])
      geom_rot = geom_xmat_in[worldid, geomid]
      geom_rotT = wp.transpose(geom_rot)
      geom_pos = geom_xpos_in[worldid, geomid]

      lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)

      l_ang = geom_rotT @ ang_global
      l_lin = geom_rotT @ lin_point

      if wind[0] or wind[1] or wind[2]:
        l_lin -= geom_rotT @ wind

      lfrc_torque = wp.vec3(0.0)
      lfrc_force = wp.vec3(0.0)

      if density > 0.0:
        # added-mass forces and torques
        virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
        virtual_inertia = wp.vec3(
          geom_fluid[geomid, 9],
          geom_fluid[geomid, 10],
          geom_fluid[geomid, 11],
        )

        virtual_lin_mom = wp.vec3(
          density * virtual_mass[0] * l_lin[0],
          density * virtual_mass[1] * l_lin[1],
          density * virtual_mass[2] * l_lin[2],
        )
        virtual_ang_mom = wp.vec3(
          density * virtual_inertia[0] * l_ang[0],
          density * virtual_inertia[1] * l_ang[1],
          density * virtual_inertia[2] * l_ang[2],
        )

        added_mass_force = wp.cross(virtual_lin_mom, l_ang)
        added_mass_torque = wp.cross(virtual_lin_mom, l_lin) + wp.cross(virtual_ang_mom, l_ang)

        lfrc_force += added_mass_force
        lfrc_torque += added_mass_torque

      # lift force orthogonal to velocity from Kutta-Joukowski theorem
      magnus_coef = geom_fluid[geomid, 5]
      kutta_coef = geom_fluid[geomid, 4]
      blunt_drag_coef = geom_fluid[geomid, 1]
      slender_drag_coef = geom_fluid[geomid, 2]
      ang_drag_coef = geom_fluid[geomid, 3]

      volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]
      d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])
      d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])
      d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min
      A_max = wp.pi * d_max * d_mid

      lin_speed = wp.length(l_lin)
      inv_speed = wp.where(lin_speed > MJ_MINVAL, 1.0 / lin_speed, 0.0)
      v = l_lin * inv_speed
      inv_dmax = wp.where(d_max > MJ_MINVAL, 1.0 / d_max, 0.0)
      s = semiaxes * inv_dmax

      magnus_force = wp.cross(l_ang, l_lin) * (magnus_coef * density * volume)

      s12_sq = _pow2(s[1] * s[2])
      s20_sq = _pow2(s[2] * s[0])
      s01_sq = _pow2(s[0] * s[1])

      proj_num = s12_sq * _pow2(v[0]) + s20_sq * _pow2(v[1]) + s01_sq * _pow2(v[2])
      inv_proj_num = wp.where(proj_num > MJ_MINVAL, 1.0 / proj_num, 0.0)
      a = s12_sq * inv_proj_num
      b = s20_sq * inv_proj_num
      c = s01_sq * inv_proj_num
      proj_denom = a * a * v[0] * v[0] + b * b * v[1] * v[1] + c * c * v[2] * v[2]

      A_proj = wp.pi * d_max * d_max * wp.sqrt(proj_num * proj_denom)
      cos_alpha = wp.where(proj_denom > MJ_MINVAL, 1.0 / proj_denom, 0.0)

      norm = wp.vec3(a * v[0], b * v[1], c * v[2])

      kutta_force = wp.vec3(0.0)
      if density > 0.0 and kutta_coef != 0.0 and lin_speed > MJ_MINVAL:
        kutta_circ = wp.cross(norm, l_lin) * (kutta_coef * density * cos_alpha * A_proj)
        kutta_force = wp.cross(kutta_circ, l_lin)

      eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])
      lin_visc_force_coef = wp.static(3.0 * wp.pi) * eq_sphere_D
      lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D

      I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * _pow4(d_max)
      II0 = ellipsoid_max_moment(semiaxes, 0)
      II1 = ellipsoid_max_moment(semiaxes, 1)
      II2 = ellipsoid_max_moment(semiaxes, 2)

      mom_visc = wp.vec3(
        l_ang[0] * (ang_drag_coef * II0 + slender_drag_coef * (I_max - II0)),
        l_ang[1] * (ang_drag_coef * II1 + slender_drag_coef * (I_max - II1)),
        l_ang[2] * (ang_drag_coef * II2 + slender_drag_coef * (I_max - II2)),
      )

      drag_lin_coef = viscosity * lin_visc_force_coef + density * lin_speed * (
        A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj)
      )
      drag_ang_coef = viscosity * lin_visc_torq_coef + density * wp.length(mom_visc)

      lfrc_torque -= drag_ang_coef * l_ang
      lfrc_force += magnus_force + kutta_force - drag_lin_coef * l_lin

      lfrc_torque *= coef
      lfrc_force *= coef

      # map force/torque from local to world frame: lfrc -> bfrc
      geom_force = geom_rot @ lfrc_force
      force_global += geom_force
      torque_global += geom_rot @ lfrc_torque + wp.cross(geom_pos - xipos, geom_force)

    fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)
    return

  l_ang = rotT @ ang_global
  l_lin = rotT @ lin_com

  if wind[0] or wind[1] or wind[2]:
    l_lin -= rotT @ wind

  lfrc_torque = wp.vec3(0.0)
  lfrc_force = wp.vec3(0.0)

  has_viscosity = viscosity > 0.0
  has_density = density > 0.0

  if has_viscosity or has_density:
    inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]
    mass = body_mass[worldid % body_mass.shape[0], bodyid]
    scl = 6.0 / mass
    box0 = wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl)
    box1 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl)
    box2 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl)

  if has_viscosity:
    diam = (box0 + box1 + box2) / 3.0
    lfrc_torque = -l_ang * wp.pow(diam, 3.0) * wp.pi * viscosity
    lfrc_force = -3.0 * l_lin * diam * wp.pi * viscosity

  if has_density:
    lfrc_force -= wp.vec3(
      0.5 * density * box1 * box2 * wp.abs(l_lin[0]) * l_lin[0],
      0.5 * density * box0 * box2 * wp.abs(l_lin[1]) * l_lin[1],
      0.5 * density * box0 * box1 * wp.abs(l_lin[2]) * l_lin[2],
    )

    scl = density / 64.0
    box0_pow4 = wp.pow(box0, 4.0)
    box1_pow4 = wp.pow(box1, 4.0)
    box2_pow4 = wp.pow(box2, 4.0)
    lfrc_torque -= wp.vec3(
      box0 * (box1_pow4 + box2_pow4) * wp.abs(l_ang[0]) * l_ang[0] * scl,
      box1 * (box0_pow4 + box2_pow4) * wp.abs(l_ang[1]) * l_ang[1] * scl,
      box2 * (box0_pow4 + box1_pow4) * wp.abs(l_ang[2]) * l_ang[2] * scl,
    )

  torque_global = rot @ lfrc_torque
  force_global = rot @ lfrc_force

  fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)


def _fluid(m: Model, d: Data):
  fluid_applied = wp.empty((d.nworld, m.nbody), dtype=wp.spatial_vector)

  wp.launch(
    _fluid_force,
    dim=(d.nworld, m.nbody),
    inputs=[
      m.opt.wind,
      m.opt.density,
      m.opt.viscosity,
      m.body_rootid,
      m.body_geomnum,
      m.body_geomadr,
      m.body_mass,
      m.body_inertia,
      m.geom_type,
      m.geom_size,
      m.geom_fluid,
      m.body_fluid_ellipsoid,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.subtree_com,
      d.cvel,
    ],
    outputs=[fluid_applied],
  )

  support.apply_ft(m, d, fluid_applied, d.qfrc_fluid, False)


@cache_kernel
def _qfrc_adhesion(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 2 * nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    dof_bodyid: wp.array[int],
    geom_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_geom_in: wp.array[wp.vec2i],
    contact_worldid_in: wp.array[int],
    contact_adhesion_in: wp.array[float],
    nacon_in: wp.array[int],
    # Data out:
    qfrc_adhesion_out: wp.array2d[float],
  ):
    cid = wp.tid()
    if cid >= nacon_in[0]:
      return

    adhesion = contact_adhesion_in[cid]
    if adhesion == 0.0:
      return

    worldid = contact_worldid_in[cid]
    geoms = contact_geom_in[cid]
    g1 = geoms[0]
    g2 = geoms[1]
    body1 = body_weldid[geom_bodyid[g1]] if g1 >= 0 else -1
    body2 = body_weldid[geom_bodyid[g2]] if g2 >= 0 else -1

    pos = contact_pos_in[cid]
    frame = contact_frame_in[cid]
    normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
    force = adhesion * normal

    if body1 > 0:
      b = body1
      while b > 0:
        dofadr = body_dofadr[b]
        dofnum = body_dofnum[b]
        for dofid in range(dofadr, dofadr + dofnum):
          jacp, _ = support.jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com_in,
            cdof_in,
            pos,
            body1,
            dofid,
            worldid,
          )
          wp.atomic_add(qfrc_adhesion_out, worldid, dofid, wp.dot(jacp, force))
        b = body_parentid[b]

    if body2 > 0:
      b = body2
      while b > 0:
        dofadr = body_dofadr[b]
        dofnum = body_dofnum[b]
        for dofid in range(dofadr, dofadr + dofnum):
          jacp, _ = support.jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com_in,
            cdof_in,
            pos,
            body2,
            dofid,
            worldid,
          )
          wp.atomic_add(qfrc_adhesion_out, worldid, dofid, wp.dot(jacp, -force))
        b = body_parentid[b]

  return kernel


@cache_kernel
def _qfrc_passive_kernel(has_fluid: bool, flg_adhesion: bool, gravity_enabled: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    jnt_actgravcomp: wp.array[int],
    dof_jntid: wp.array[int],
    # Data in:
    qfrc_spring_in: wp.array2d[float],
    qfrc_damper_in: wp.array2d[float],
    qfrc_gravcomp_in: wp.array2d[float],
    qfrc_fluid_in: wp.array2d[float],
    qfrc_adhesion_in: wp.array2d[float],
    # Data out:
    qfrc_passive_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()
    qfrc_passive = qfrc_spring_in[worldid, dofid]
    qfrc_passive += qfrc_damper_in[worldid, dofid]

    # add gravcomp unless added by actuators
    if wp.static(gravity_enabled):
      if not jnt_actgravcomp[dof_jntid[dofid]]:
        qfrc_passive += qfrc_gravcomp_in[worldid, dofid]

    # add fluid force
    if wp.static(has_fluid):
      qfrc_passive += qfrc_fluid_in[worldid, dofid]

    # add adhesion force
    if wp.static(flg_adhesion):
      qfrc_passive += qfrc_adhesion_in[worldid, dofid]

    qfrc_passive_out[worldid, dofid] = qfrc_passive

  return kernel


@wp.func
def _snh_cubic(tension: vec6, s: vec6, gamma: float) -> vec6:
  a = s[0]
  b = s[2]
  c = s[4]
  d = 0.5 * (s[0] + s[2] - s[1])
  e = 0.5 * (s[0] + s[4] - s[5])
  f = 0.5 * (s[2] + s[4] - s[3])
  cof0 = b * c - f * f
  cof1 = a * c - e * e
  cof2 = a * b - d * d
  cof3 = e * f - c * d
  cof4 = d * f - b * e
  cof5 = d * e - a * f
  scale = 2.0 * gamma
  return vec6(
    tension[0] + scale * (cof0 + cof3 + cof4),
    tension[1] - scale * cof3,
    tension[2] + scale * (cof1 + cof3 + cof5),
    tension[3] - scale * cof5,
    tension[4] + scale * (cof2 + cof4 + cof5),
    tension[5] - scale * cof4,
  )


@wp.func
def _snh_volume(edgevec: mat63, inv_det_dm: float) -> tuple[float, mat43]:
  a = -edgevec[0]
  b = edgevec[2]
  c = -edgevec[4]
  bxc = wp.cross(b, c)
  cxa = wp.cross(c, a)
  axb = wp.cross(a, b)
  g1 = bxc * inv_det_dm
  g2 = cxa * inv_det_dm
  g3 = axb * inv_det_dm
  g0 = -(g1 + g2 + g3)
  grad = mat43()
  grad[0] = g0
  grad[1] = g1
  grad[2] = g2
  grad[3] = g3
  return wp.dot(a, bxc) * inv_det_dm, grad


@wp.func
def _snh_cubic_metric(metric: mat66, s: vec6, gamma: float) -> mat66:
  a = s[0]
  b = s[2]
  c = s[4]
  d = 0.5 * (s[0] + s[2] - s[1])
  e = 0.5 * (s[0] + s[4] - s[5])
  f = 0.5 * (s[2] + s[4] - s[3])

  basis = wp.matrix(
    1.0,
    0.0,
    0.0,
    0.5,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.5,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.5,
    0.0,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.5,
    0.0,
    0.0,
    1.0,
    0.0,
    0.5,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.5,
    0.0,
    shape=(6, 6),
    dtype=float,
  )
  for j in range(6):
    h0 = basis[j, 0]
    h1 = basis[j, 1]
    h2 = basis[j, 2]
    h3 = basis[j, 3]
    h4 = basis[j, 4]
    h5 = basis[j, 5]
    dc0 = h1 * c + b * h2 - 2.0 * f * h5
    dc1 = h0 * c + a * h2 - 2.0 * e * h4
    dc2 = h0 * b + a * h1 - 2.0 * d * h3
    dc3 = h4 * f + e * h5 - h2 * d - c * h3
    dc4 = h3 * f + d * h5 - h1 * e - b * h4
    dc5 = h3 * e + d * h4 - h0 * f - a * h5
    dg = vec6(dc0 + dc3 + dc4, -dc3, dc1 + dc3 + dc5, -dc5, dc2 + dc4 + dc5, -dc4)
    for i in range(j + 1):
      value = 2.0 * gamma * dg[i]
      metric[i, j] += value
      if i != j:
        metric[j, i] += value
  return metric


@wp.func
def _snh_positive3(A: wp.mat33) -> bool:
  scale = float(0.0)
  for row in range(3):
    for col in range(3):
      scale = wp.max(scale, wp.abs(A[row, col]))
  tol = 1.0e-5 * scale
  if A[0, 0] <= tol:
    return False
  pivot = A[1, 1] - A[0, 1] * (A[0, 1] / A[0, 0])
  if pivot <= tol:
    return False
  cross = A[1, 2] - A[0, 1] * (A[0, 2] / A[0, 0])
  return A[2, 2] - A[0, 2] * (A[0, 2] / A[0, 0]) - cross * (cross / pivot) > tol


@wp.func
def _snh_eigen3(A: wp.mat33) -> tuple[wp.vec3, wp.mat33]:
  scale = float(0.0)
  for row in range(3):
    for col in range(3):
      scale = wp.max(scale, wp.abs(A[row, col]))
  Q = wp.identity(3, dtype=float)
  if scale == 0.0:
    return wp.vec3(0.0, 0.0, 0.0), Q
  D = A * (1.0 / scale)
  if D[0, 1] == 0.0 and D[0, 2] == 0.0 and D[1, 2] == 0.0:
    return wp.vec3(A[0, 0], A[1, 1], A[2, 2]), Q

  mean = (D[0, 0] + D[1, 1] + D[2, 2]) * (1.0 / 3.0)
  B = D
  B[0, 0] -= mean
  B[1, 1] -= mean
  B[2, 2] -= mean
  bscale = float(0.0)
  for row in range(3):
    for col in range(3):
      bscale = wp.max(bscale, wp.abs(B[row, col]))
  B = B * (1.0 / bscale)
  p = wp.sqrt(
    (
      B[0, 0] * B[0, 0]
      + B[1, 1] * B[1, 1]
      + B[2, 2] * B[2, 2]
      + 2.0 * (B[0, 1] * B[0, 1] + B[0, 2] * B[0, 2] + B[1, 2] * B[1, 2])
    )
    * (1.0 / 6.0)
  )
  B = B * (1.0 / p)
  determinant = (
    B[0, 0] * (B[1, 1] * B[2, 2] - B[1, 2] * B[1, 2])
    - B[0, 1] * (B[0, 1] * B[2, 2] - B[0, 2] * B[1, 2])
    + B[0, 2] * (B[0, 1] * B[1, 2] - B[0, 2] * B[1, 1])
  )
  r_val = 0.5 * determinant

  root = 2.0 * wp.cos(wp.acos(wp.min(1.0, wp.abs(r_val))) * (1.0 / 3.0))
  if r_val < 0.0:
    root = -root
  B[0, 0] -= root
  B[1, 1] -= root
  B[2, 2] -= root

  row0 = B[0]
  row1 = B[1]
  row2 = B[2]
  cr0 = wp.cross(row0, row1)
  cr1 = wp.cross(row0, row2)
  cr2 = wp.cross(row1, row2)
  n0 = wp.dot(cr0, cr0)
  n1 = wp.dot(cr1, cr1)
  n2 = wp.dot(cr2, cr2)
  best_cr = cr0
  best_n = n0
  if n1 > best_n:
    best_cr = cr1
    best_n = n1
  if n2 > best_n:
    best_cr = cr2
    best_n = n2
  q = best_cr * (1.0 / wp.sqrt(best_n))

  if wp.abs(q[0]) > wp.abs(q[1]):
    inv = 1.0 / wp.sqrt(q[0] * q[0] + q[2] * q[2])
    u = wp.vec3(-q[2] * inv, 0.0, q[0] * inv)
  else:
    inv = 1.0 / wp.sqrt(q[1] * q[1] + q[2] * q[2])
    u = wp.vec3(0.0, q[2] * inv, -q[1] * inv)
  v = wp.cross(q, u)
  Du = D * u
  Dv = D * v
  Dq = D * q

  a = wp.dot(u, Du)
  b = wp.dot(u, Dv)
  c = wp.dot(v, Dv)
  cosine = 1.0
  sine = 0.0
  t = 0.0
  if b != 0.0:
    delta = 0.5 * (c - a)
    wscale = wp.max(wp.abs(delta), wp.abs(b))
    x = delta / wscale
    y = b / wscale
    t = (y if x >= 0.0 else -y) / (wp.abs(x) + wp.sqrt(x * x + y * y))
    cosine = 1.0 / wp.sqrt(1.0 + t * t)
    sine = t * cosine

  value = wp.vec3(wp.dot(q, Dq) * scale, (a - t * b) * scale, (c + t * b) * scale)
  c1 = cosine * u - sine * v
  c2 = sine * u + cosine * v
  Q = wp.mat33(q[0], c1[0], c2[0], q[1], c1[1], c2[1], q[2], c1[2], c2[2])
  return value, Q


@wp.func
def _snh_jacobi_rot(A: wp.mat33, V: wp.mat33, p: int, q: int) -> tuple[wp.mat33, wp.mat33, bool]:
  pp = A[0, p] * A[0, p] + A[1, p] * A[1, p] + A[2, p] * A[2, p]
  qq = A[0, q] * A[0, q] + A[1, q] * A[1, q] + A[2, q] * A[2, q]
  pq = A[0, p] * A[0, q] + A[1, p] * A[1, q] + A[2, p] * A[2, q]
  if wp.abs(pq) <= 2.0e-6 * wp.sqrt(pp) * wp.sqrt(qq):
    return A, V, True
  delta = 0.5 * (qq - pp)
  t = (pq if delta >= 0.0 else -pq) / (wp.abs(delta) + wp.sqrt(delta * delta + pq * pq))
  c = 1.0 / wp.sqrt(1.0 + t * t)
  s = t * c
  for row in range(3):
    arp = A[row, p]
    arq = A[row, q]
    A[row, p] = c * arp - s * arq
    A[row, q] = s * arp + c * arq
    vrp = V[row, p]
    vrq = V[row, q]
    V[row, p] = c * vrp - s * vrq
    V[row, q] = s * vrp + c * vrq
  return A, V, False


@wp.func
def _snh_swap_cols(A: wp.mat33, V: wp.mat33, s: wp.vec3, p: int, q: int) -> tuple[wp.mat33, wp.mat33, wp.vec3]:
  if s[q] > s[p]:
    sp = s[p]
    s[p] = s[q]
    s[q] = sp
    for row in range(3):
      arp = A[row, p]
      A[row, p] = A[row, q]
      A[row, q] = arp
      vrp = V[row, p]
      V[row, p] = V[row, q]
      V[row, q] = vrp
  return A, V, s


@wp.func
def _snh_svd(F: wp.mat33) -> tuple[wp.mat33, wp.vec3, wp.mat33]:
  scale = float(0.0)
  for row in range(3):
    for col in range(3):
      scale = wp.max(scale, wp.abs(F[row, col]))
  A = F * (1.0 / scale) if scale > 0.0 else wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  V = wp.identity(3, dtype=float)
  for _ in range(24):
    A, V, d01 = _snh_jacobi_rot(A, V, 0, 1)
    A, V, d02 = _snh_jacobi_rot(A, V, 0, 2)
    A, V, d12 = _snh_jacobi_rot(A, V, 1, 2)
    if d01 and d02 and d12:
      break

  s = wp.vec3(
    A[0, 0] * A[0, 0] + A[1, 0] * A[1, 0] + A[2, 0] * A[2, 0],
    A[0, 1] * A[0, 1] + A[1, 1] * A[1, 1] + A[2, 1] * A[2, 1],
    A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2] + A[2, 2] * A[2, 2],
  )
  A, V, s = _snh_swap_cols(A, V, s, 0, 1)
  A, V, s = _snh_swap_cols(A, V, s, 0, 2)
  A, V, s = _snh_swap_cols(A, V, s, 1, 2)

  if wp.determinant(V) < 0.0:
    for r in range(3):
      V[r, 2] = -V[r, 2]
      A[r, 2] = -A[r, 2]

  u0 = wp.vec3(A[0, 0], A[1, 0], A[2, 0])
  u1 = wp.vec3(A[0, 1], A[1, 1], A[2, 1])
  u2 = wp.vec3(A[0, 2], A[1, 2], A[2, 2])
  s0 = wp.length(u0)
  if s0 == 0.0:
    return wp.identity(3, dtype=float), wp.vec3(0.0, 0.0, 0.0), V

  u0 = u0 * (1.0 / s0)
  u1 = u1 - wp.dot(u0, u1) * u0
  norm = wp.length(u1)
  if norm > 1.0e-6 * s0:
    u1 = u1 * (1.0 / norm)
  else:
    axis = int(0)
    if wp.abs(u0[1]) < wp.abs(u0[axis]):
      axis = 1
    if wp.abs(u0[2]) < wp.abs(u0[axis]):
      axis = 2
    u1 = wp.vec3(
      (1.0 if axis == 0 else 0.0) - u0[axis] * u0[0],
      (1.0 if axis == 1 else 0.0) - u0[axis] * u0[1],
      (1.0 if axis == 2 else 0.0) - u0[axis] * u0[2],
    )
    u1 = wp.normalize(u1)

  u2 = wp.cross(u0, u1)
  a1 = wp.vec3(A[0, 1], A[1, 1], A[2, 1])
  a2 = wp.vec3(A[0, 2], A[1, 2], A[2, 2])
  sigma = wp.vec3(s0, wp.dot(u1, a1), wp.dot(u2, a2)) * scale
  U = wp.mat33(u0[0], u1[0], u2[0], u0[1], u1[1], u2[1], u0[2], u1[2], u2[2])
  return U, sigma, V


@wp.func
def stretch_edge_endpoints(dim: int):
  return wp.where(
    dim == 3,
    wp.matrix(0, 1, 1, 2, 2, 0, 2, 3, 0, 3, 1, 3, shape=(6, 2), dtype=int),
    wp.matrix(1, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),
  )


@wp.func
def stretch_edge_vectors(vert_xpos: mat43, dim: int) -> mat63:
  nedge = 3 if dim == 2 else 6
  edges = stretch_edge_endpoints(dim)
  edgevec = mat63()
  for e in range(nedge):
    edgevec[e] = vert_xpos[edges[e, 0]] - vert_xpos[edges[e, 1]]
  return edgevec


@wp.func
def stretch_elongation(
  # Model:
  flex_elemedge: wp.array[int],
  flexedge_length0: wp.array[float],
  # Data in:
  plastic_in: wp.array2d[float],
  flexedge_length_in: wp.array2d[float],
  # In:
  worldid: int,
  elemedge_adr: int,
  edge_adr: int,
  padr: int,
  nedge: int,
) -> vec6:
  elongation = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for e in range(nedge):
    e_idx = edge_adr + flex_elemedge[elemedge_adr + e]
    deformed = flexedge_length_in[worldid, e_idx]
    reference = flexedge_length0[e_idx]
    val = deformed * deformed - reference * reference
    if padr >= 0:
      val -= plastic_in[worldid, padr + e]
    elongation[e] = val
  return elongation


@wp.func
def stretch_metric(
  # Model:
  flex_stiffness: wp.array[float],
  # In:
  stiffness_adr: int,
  nedge: int,
) -> mat66:
  metric = mat66(0.0)
  idx = int(0)
  for e1 in range(nedge):
    for e2 in range(e1, nedge):
      val = flex_stiffness[stiffness_adr + idx]
      metric[e1, e2] = val
      metric[e2, e1] = val
      idx += 1
  return metric


@wp.func
def stretch_tension(metric: mat66, elongation: vec6, nedge: int) -> vec6:
  tension = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for e1 in range(nedge):
    t_val = float(0.0)
    for e2 in range(nedge):
      t_val += metric[e1, e2] * elongation[e2]
    tension[e1] = t_val
  return tension


@wp.func
def stretch_elasticity(
  # Model:
  flex_stiffness: wp.array[float],
  # In:
  stiffness_adr: int,
  elongation: vec6,
  nedge: int,
) -> tuple[mat66, vec6]:
  metric = stretch_metric(flex_stiffness, stiffness_adr, nedge)
  tension = stretch_tension(metric, elongation, nedge)
  return metric, tension


@wp.func
def stretch_stiffness(
  # Model:
  flex_elemedge: wp.array[int],
  flexedge_length0: wp.array[float],
  flex_stiffness: wp.array[float],
  # Data in:
  plastic_in: wp.array2d[float],
  flexedge_length_in: wp.array2d[float],
  # In:
  worldid: int,
  stiffness_adr: int,
  elemedge_adr: int,
  edge_adr: int,
  padr: int,
  nedge: int,
) -> tuple[mat66, vec6]:
  elongation = stretch_elongation(
    flex_elemedge,
    flexedge_length0,
    plastic_in,
    flexedge_length_in,
    worldid,
    elemedge_adr,
    edge_adr,
    padr,
    nedge,
  )
  metric, tension = stretch_elasticity(flex_stiffness, stiffness_adr, elongation, nedge)
  for e in range(nedge):
    tension[e] = wp.max(tension[e], 0.0)
  return metric, tension


@wp.func
def stretch_stiffness_block(
  # In:
  metric: mat66,
  tension: vec6,
  edgevec: mat63,
  dim: int,
  i: int,
  j: int,
  scale: float,
) -> wp.mat33:
  nedge = 3 if dim == 2 else 6
  edges = stretch_edge_endpoints(dim)
  blk = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for a in range(nedge):
    sa = 1.0 if i == edges[a, 0] else (-1.0 if i == edges[a, 1] else 0.0)
    if sa == 0.0:
      continue
    for b in range(nedge):
      sb = 1.0 if j == edges[b, 0] else (-1.0 if j == edges[b, 1] else 0.0)
      if sb == 0.0:
        continue
      w = 2.0 * scale * metric[a, b] * sa * sb
      for r in range(3):
        for c in range(3):
          blk[r, c] += w * edgevec[a, r] * edgevec[b, c]
  geo = float(0.0)
  for a in range(nedge):
    sa = 1.0 if i == edges[a, 0] else (-1.0 if i == edges[a, 1] else 0.0)
    sb = 1.0 if j == edges[a, 0] else (-1.0 if j == edges[a, 1] else 0.0)
    if sa != 0.0 and sb != 0.0:
      geo += tension[a] * sa * sb
  geo *= scale
  blk[0, 0] += geo
  blk[1, 1] += geo
  blk[2, 2] += geo
  return blk


@wp.func
def snh_project(
  # Model:
  flex_vert0: wp.array[wp.vec3],
  flex_stiffness: wp.array[float],
  # In:
  stiffness_adr: int,
  vbase: int,
  vert: wp.vec4i,
  size: wp.vec3,
  edgevec: mat63,
  elongation: vec6,
) -> tuple[bool, wp.mat33, mat43, wp.mat33, wp.vec3, wp.vec3]:
  v0 = flex_vert0[vbase + vert[0]]
  v1 = flex_vert0[vbase + vert[1]] - v0
  v2 = flex_vert0[vbase + vert[2]] - v0
  v3 = flex_vert0[vbase + vert[3]] - v0
  rest0 = wp.vec3(2.0 * size[0] * v1[0], 2.0 * size[1] * v1[1], 2.0 * size[2] * v1[2])
  rest1 = wp.vec3(2.0 * size[0] * v2[0], 2.0 * size[1] * v2[1], 2.0 * size[2] * v2[2])
  rest2 = wp.vec3(2.0 * size[0] * v3[0], 2.0 * size[1] * v3[1], 2.0 * size[2] * v3[2])

  k21 = flex_stiffness[stiffness_adr + 21]
  k22 = flex_stiffness[stiffness_adr + 22]
  k23 = flex_stiffness[stiffness_adr + 23]

  grad0 = wp.cross(rest1, rest2) * k23
  grad1 = wp.cross(rest2, rest0) * k23
  grad2 = wp.cross(rest0, rest1) * k23

  F = -wp.outer(edgevec[0], grad0) + wp.outer(edgevec[2], grad1) - wp.outer(edgevec[4], grad2)
  rotation, sigma, V = _snh_svd(F)
  VT = wp.transpose(V)

  hg1 = VT * grad0
  hg2 = VT * grad1
  hg3 = VT * grad2
  hg0 = -(hg1 + hg2 + hg3)
  gradient = mat43()
  gradient[0] = hg0
  gradient[1] = hg1
  gradient[2] = hg2
  gradient[3] = hg3

  metric, tension = stretch_elasticity(flex_stiffness, stiffness_adr, elongation, 6)
  tension = _snh_cubic(tension, elongation, k21)
  metric = _snh_cubic_metric(metric, elongation, k21)

  vertex = mat43()
  vertex[0] = wp.vec3(0.0, 0.0, 0.0)
  vertex[1] = VT * rest0
  vertex[2] = VT * rest1
  vertex[3] = VT * rest2

  edges = stretch_edge_endpoints(3)
  reference = mat63()
  sq0 = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  sq1 = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  sq2 = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for e in range(6):
    ref_e = vertex[edges[e, 0]] - vertex[edges[e, 1]]
    reference[e] = ref_e
    sq0[e] = ref_e[0] * ref_e[0]
    sq1[e] = ref_e[1] * ref_e[1]
    sq2[e] = ref_e[2] * ref_e[2]

  prod0 = metric * sq0
  prod1 = metric * sq1
  prod2 = metric * sq2
  geo = wp.vec3(wp.dot(tension, sq0), wp.dot(tension, sq1), wp.dot(tension, sq2))

  J = sigma[0] * sigma[1] * sigma[2]
  volume_stiffness = 2.0 * k22
  pressure = volume_stiffness * (J - 1.0)
  cof = wp.vec3(sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1])

  a00 = 2.0 * sigma[0] * sigma[0] * wp.dot(sq0, prod0) + volume_stiffness * cof[0] * cof[0] + geo[0]
  a11 = 2.0 * sigma[1] * sigma[1] * wp.dot(sq1, prod1) + volume_stiffness * cof[1] * cof[1] + geo[1]
  a22 = 2.0 * sigma[2] * sigma[2] * wp.dot(sq2, prod2) + volume_stiffness * cof[2] * cof[2] + geo[2]
  a01 = 2.0 * sigma[0] * sigma[1] * wp.dot(sq0, prod1) + volume_stiffness * cof[0] * cof[1] + pressure * sigma[2]
  a02 = 2.0 * sigma[0] * sigma[2] * wp.dot(sq0, prod2) + volume_stiffness * cof[0] * cof[2] + pressure * sigma[1]
  a12 = 2.0 * sigma[1] * sigma[2] * wp.dot(sq1, prod2) + volume_stiffness * cof[1] * cof[2] + pressure * sigma[0]
  A = wp.mat33(a00, a01, a02, a01, a11, a12, a02, a12, a22)

  if _snh_positive3(A):
    stretch = A
  else:
    eig, Q = _snh_eigen3(A)
    ev0 = wp.max(0.0, eig[0])
    ev1 = wp.max(0.0, eig[1])
    ev2 = wp.max(0.0, eig[2])
    QD = wp.mat33(
      Q[0, 0] * ev0,
      Q[0, 1] * ev1,
      Q[0, 2] * ev2,
      Q[1, 0] * ev0,
      Q[1, 1] * ev1,
      Q[1, 2] * ev2,
      Q[2, 0] * ev0,
      Q[2, 1] * ev1,
      Q[2, 2] * ev2,
    )
    stretch = QD * wp.transpose(Q)

  symmetric = wp.vec3(0.0, 0.0, 0.0)
  skew = wp.vec3(0.0, 0.0, 0.0)
  pair_i = wp.vec3i(0, 0, 1)
  pair_j = wp.vec3i(1, 2, 2)
  for mode in range(3):
    ii = pair_i[mode]
    jj = pair_j[mode]
    direction = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for e in range(6):
      direction[e] = reference[e, ii] * reference[e, jj]
    response = metric * direction
    material = wp.dot(direction, response)
    geometric = 0.5 * (geo[ii] + geo[jj])
    sum_s = sigma[ii] + sigma[jj]
    diff_s = sigma[ii] - sigma[jj]
    cross_s = pressure * sigma[3 - ii - jj]
    symmetric[mode] = wp.max(0.0, sum_s * sum_s * material + geometric - cross_s)
    skew[mode] = wp.max(0.0, diff_s * diff_s * material + geometric + cross_s)

  active = (
    stretch[0, 0] > 0.0
    or stretch[1, 1] > 0.0
    or stretch[2, 2] > 0.0
    or symmetric[0] > 0.0
    or symmetric[1] > 0.0
    or symmetric[2] > 0.0
    or skew[0] > 0.0
    or skew[1] > 0.0
    or skew[2] > 0.0
  )
  return active, rotation, gradient, stretch, symmetric, skew


@wp.func
def snh_projected_block(
  # In:
  rotation: wp.mat33,
  gradient: mat43,
  stretch: wp.mat33,
  symmetric: wp.vec3,
  skew: wp.vec3,
  i: int,
  j: int,
) -> wp.mat33:
  a = gradient[i]
  b = gradient[j]
  s0 = 0.5 * (symmetric[0] + skew[0])
  d0 = 0.5 * (symmetric[0] - skew[0])
  s1 = 0.5 * (symmetric[1] + skew[1])
  d1 = 0.5 * (symmetric[1] - skew[1])
  s2 = 0.5 * (symmetric[2] + skew[2])
  d2 = 0.5 * (symmetric[2] - skew[2])

  local = wp.mat33(
    stretch[0, 0] * a[0] * b[0] + s0 * a[1] * b[1] + s1 * a[2] * b[2],
    stretch[0, 1] * a[0] * b[1] + d0 * a[1] * b[0],
    stretch[0, 2] * a[0] * b[2] + d1 * a[2] * b[0],
    stretch[1, 0] * a[1] * b[0] + d0 * a[0] * b[1],
    stretch[1, 1] * a[1] * b[1] + s0 * a[0] * b[0] + s2 * a[2] * b[2],
    stretch[1, 2] * a[1] * b[2] + d2 * a[2] * b[1],
    stretch[2, 0] * a[2] * b[0] + d1 * a[0] * b[2],
    stretch[2, 1] * a[2] * b[1] + d2 * a[1] * b[2],
    stretch[2, 2] * a[2] * b[2] + s1 * a[0] * b[0] + s2 * a[1] * b[1],
  )
  return rotation * local * wp.transpose(rotation)


@wp.func
def flex_gather_vert(
  # Model:
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  dof_parentid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  vec_in: wp.array2d[float],
  worldid: int,
  gvert: int,
) -> wp.vec3:
  res = wp.vec3(0.0, 0.0, 0.0)
  bodyid = flex_vertbodyid[gvert]
  if bodyid < 0:
    return res
  body = body_weldid[bodyid]
  dofnum = body_dofnum[body]
  if dofnum == 0:
    return res
  da = body_dofadr[body]
  if body_simple[body] == 2:
    for j in range(dofnum):
      axis = wp.spatial_bottom(cdof_in[worldid, da + j])
      res += axis * vec_in[worldid, da + j]
    return res
  offset = flexvert_xpos_in[worldid, gvert] - subtree_com_in[worldid, body_rootid[body]]
  i = da + dofnum - 1
  while i >= 0:
    cdof = cdof_in[worldid, i]
    column = wp.spatial_bottom(cdof) + wp.cross(wp.spatial_top(cdof), offset)
    res += column * vec_in[worldid, i]
    i = dof_parentid[i]
  return res


@wp.func
def flex_scatter_vert(
  # Model:
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  dof_parentid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  worldid: int,
  gvert: int,
  f_world: wp.vec3,
  scale: float,
  # Out:
  res_out: wp.array2d[float],
):
  if f_world[0] == 0.0 and f_world[1] == 0.0 and f_world[2] == 0.0:
    return
  bodyid = flex_vertbodyid[gvert]
  if bodyid < 0:
    return
  body = body_weldid[bodyid]
  dofnum = body_dofnum[body]
  if dofnum == 0:
    return
  da = body_dofadr[body]
  if body_simple[body] == 2:
    for j in range(dofnum):
      axis = wp.spatial_bottom(cdof_in[worldid, da + j])
      wp.atomic_add(res_out, worldid, da + j, scale * wp.dot(axis, f_world))
    return
  offset = flexvert_xpos_in[worldid, gvert] - subtree_com_in[worldid, body_rootid[body]]
  i = da + dofnum - 1
  while i >= 0:
    cdof = cdof_in[worldid, i]
    column = wp.spatial_bottom(cdof) + wp.cross(wp.spatial_top(cdof), offset)
    wp.atomic_add(res_out, worldid, i, scale * wp.dot(column, f_world))
    i = dof_parentid[i]


@wp.func
def _rod_hinge(xa: wp.vec3, xv: wp.vec3, xb: wp.vec3) -> tuple[bool, wp.vec3, wp.mat33, wp.mat33]:
  """Curvature binormal of a 1D rod hinge and its derivatives, see Bergou et al. 2008.

  kb = 2 e0 x e1 / (|e0||e1| + e0.e1), with e0 = xv - xa and e1 = xb - xv. Returns whether the
  hinge is valid, kb, A0 = d(kb)/d(e0) and A1 = d(kb)/d(e1); the Jacobian with respect to
  (xa, xv, xb) is [-A0, A0 - A1, A1], see rod_stencil.
  """
  e0 = xv - xa
  e1 = xb - xv
  l0 = wp.length(e0)
  l1 = wp.length(e1)
  denom = l0 * l1 + wp.dot(e0, e1)
  if l0 < MJ_MINVAL or l1 < MJ_MINVAL or denom < MJ_MINVAL:
    return False, wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0)
  inv_denom = 1.0 / denom
  kb = (2.0 * inv_denom) * wp.cross(e0, e1)

  # gradients of denom with respect to e0 and e1
  g0 = (l1 / l0) * e0 + e1
  g1 = (l0 / l1) * e1 + e0

  A0 = inv_denom * (-2.0 * wp.skew(e1) - wp.outer(kb, g0))
  A1 = inv_denom * (2.0 * wp.skew(e0) - wp.outer(kb, g1))
  return True, kb, A0, A1


# smoothing of curved rod references at zero curvature, relative to the reference curvature
_ROD_SMOOTH = 1.0e-4


@wp.func
def _rod_hinge_weights(kb: wp.vec3, k: float, kb0: float) -> tuple[wp.vec3, wp.mat33, wp.mat33]:
  """Gradient, Hessian, and damping weight of a single rod hinge in kb space.

  The energy is k |kb|^2 for a straight reference (kb0 = 0), with gradient 2k kb and Hessian 2k I,
  and k (|kb| - kb0)^2 for a curved one, smoothed at kb = 0, with the Hessian projected to PSD. The
  damping weight 2k n n', n = kb / |kb|, damps only the rate of |kb|, so rigid rotations are not
  damped; it is isotropic at kb = 0.
  """
  r = wp.length(kb)
  I3 = wp.identity(n=3, dtype=float)
  Wd = (2.0 * k) * I3
  nn = wp.mat33(0.0)
  if r >= MJ_MINVAL:
    n = kb / r
    nn = wp.outer(n, n)
    Wd = (2.0 * k) * nn

  if kb0 == 0.0:
    return (2.0 * k) * kb, (2.0 * k) * I3, Wd

  # energy k (rho - gam)^2 kb0^2, with rho = sqrt((|kb| / kb0)^2 + e^2) and gam = sqrt(1 + e^2)
  e2 = wp.static(_ROD_SMOOTH * _ROD_SMOOTH)
  gam = wp.static((1.0 + _ROD_SMOOTH * _ROD_SMOOTH) ** 0.5)
  c = r / kb0
  rho2 = c * c + e2
  rho = wp.sqrt(rho2)
  s = 1.0 - gam / rho
  g = (2.0 * k * s) * kb

  # Hessian: tangential 2k s and radial 2k (1 - gam e^2 / rho^3), clamped to PSD
  Wk = wp.mat33(0.0)
  if r >= MJ_MINVAL:
    tang = wp.max(0.0, s)
    rad = wp.max(0.0, 1.0 - gam * e2 / (rho * rho2))
    Wk = (2.0 * k) * (tang * (I3 - nn) + rad * nn)
  return g, Wk, Wd


@wp.func
def _rod_pair(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> tuple[bool, wp.vec4, mat43, mat43, mat43, mat43]:
  """Curvature binormals of a rod hinge pair in a frame orthogonal to their shared edge.

  Returns whether the pair is valid, om = (kba.u, kba.v, kbb.u, kbb.v) for the hinges
  (x0, x1, x2) and (x1, x2, x3) and axes (u, v) orthogonal to the edge (x1, x2), and the 4x3
  Jacobian blocks d(om)/dx of the four vertices. The axes are held fixed: the hinge pair energy
  does not depend on their rotation about the edge.
  """
  oka, kba, A0, A1 = _rod_hinge(x0, x1, x2)
  okb, kbb, B0, B1 = _rod_hinge(x1, x2, x3)
  edge = x2 - x1
  ledge = wp.length(edge)
  if not oka or not okb or ledge < MJ_MINVAL:
    return False, wp.vec4(0.0), mat43(), mat43(), mat43(), mat43()

  u, v = math.orthogonals(edge / ledge)
  om = wp.vec4(wp.dot(kba, u), wp.dot(kba, v), wp.dot(kbb, u), wp.dot(kbb, v))

  # rows u'd(kb)/dx and v'd(kb)/dx of the Jacobians [-A0, A0 - A1, A1] and [-B0, B0 - B1, B1]
  ua0 = wp.transpose(A0) * u
  va0 = wp.transpose(A0) * v
  ua1 = wp.transpose(A1) * u
  va1 = wp.transpose(A1) * v
  ub0 = wp.transpose(B0) * u
  vb0 = wp.transpose(B0) * v
  ub1 = wp.transpose(B1) * u
  vb1 = wp.transpose(B1) * v
  z3 = wp.vec3(0.0)

  J0 = mat43()
  J0[0] = -ua0
  J0[1] = -va0
  J0[2] = z3
  J0[3] = z3

  J1 = mat43()
  J1[0] = ua0 - ua1
  J1[1] = va0 - va1
  J1[2] = -ub0
  J1[3] = -vb0

  J2 = mat43()
  J2[0] = ua1
  J2[1] = va1
  J2[2] = ub0 - ub1
  J2[3] = vb0 - vb1

  J3 = mat43()
  J3[0] = z3
  J3[1] = z3
  J3[2] = ub1
  J3[3] = vb1

  return True, om, J0, J1, J2, J3


@wp.func
def _rod_pair_weights(om: wp.vec4, w0: float, w1: float, prm: wp.vec4) -> tuple[wp.vec4, wp.mat44, wp.mat44]:
  """Gradient, Hessian, and damping weight of a rod hinge pair in om space.

  The energy is w0 (|oma| - r0a)^2 + w1 (|omb| - r0b)^2 + kt |R(dphi0) nta - ntb|^2, smoothed at
  om = 0, with the Hessian projected to PSD. The damping weight is orthogonal to rotations about
  the shared edge, so rigid rotations are not damped.
  """
  r0a = prm[0]
  r0b = prm[1]
  tc = prm[2]
  ts = prm[3]
  kt = wp.sqrt(tc * tc + ts * ts)
  if w0 <= 0.0 and w1 <= 0.0 and kt <= 0.0:
    return wp.vec4(0.0), wp.mat44(0.0), wp.mat44(0.0)

  e2 = wp.static(_ROD_SMOOTH * _ROD_SMOOTH)
  gam = wp.static((1.0 + _ROD_SMOOTH * _ROD_SMOOTH) ** 0.5)

  # hinge a: smoothed curvature magnitude and unit binormal
  ra = wp.sqrt(om[0] * om[0] + om[1] * om[1])
  na0 = 0.0
  na1 = 0.0
  if ra >= MJ_MINVAL:
    na0 = om[0] / ra
    na1 = om[1] / ra
  ca = wp.where(r0a > 0.0, ra / r0a, 0.0)
  rho2a = ca * ca + e2
  rhoa = wp.sqrt(rho2a)
  sa = wp.where(r0a > 0.0, 1.0 - gam / rhoa, 1.0)

  # hinge b: smoothed curvature magnitude and unit binormal
  rb = wp.sqrt(om[2] * om[2] + om[3] * om[3])
  nb0 = 0.0
  nb1 = 0.0
  if rb >= MJ_MINVAL:
    nb0 = om[2] / rb
    nb1 = om[3] / rb
  cb = wp.where(r0b > 0.0, rb / r0b, 0.0)
  rho2b = cb * cb + e2
  rhob = wp.sqrt(rho2b)
  sb = wp.where(r0b > 0.0, 1.0 - gam / rhob, 1.0)

  # smoothed unit binormals nta = alpa * oma, ntb = alpb * omb and their 2x2 Jacobians Ma, Mb
  alpa = wp.where(r0a > 0.0, gam / (rhoa * r0a), 0.0)
  alpb = wp.where(r0b > 0.0, gam / (rhob * r0b), 0.0)
  nta0 = alpa * om[0]
  nta1 = alpa * om[1]
  ntb0 = alpb * om[2]
  ntb1 = alpb * om[3]

  rad_ma = wp.where(ra >= MJ_MINVAL, e2 / rho2a, 1.0)
  na_perp0 = wp.where(ra >= MJ_MINVAL, na1 * na1, 1.0)
  na_perp1 = wp.where(ra >= MJ_MINVAL, na0 * na0, 1.0)
  Ma00 = alpa * (na_perp0 + rad_ma * na0 * na0)
  Ma01 = alpa * (rad_ma - 1.0) * na0 * na1
  Ma11 = alpa * (na_perp1 + rad_ma * na1 * na1)

  rad_mb = wp.where(rb >= MJ_MINVAL, e2 / rho2b, 1.0)
  nb_perp0 = wp.where(rb >= MJ_MINVAL, nb1 * nb1, 1.0)
  nb_perp1 = wp.where(rb >= MJ_MINVAL, nb0 * nb0, 1.0)
  Mb00 = alpb * (nb_perp0 + rad_mb * nb0 * nb0)
  Mb01 = alpb * (rad_mb - 1.0) * nb0 * nb1
  Mb11 = alpb * (nb_perp1 + rad_mb * nb1 * nb1)

  # gradient: radial bending + dihedral torsion kt |R(dphi0) nta - ntb|^2
  tba0 = tc * ntb0 + ts * ntb1
  tba1 = -ts * ntb0 + tc * ntb1
  tab0 = tc * nta0 - ts * nta1
  tab1 = ts * nta0 + tc * nta1
  da0 = kt * nta0 - tba0
  da1 = kt * nta1 - tba1
  db0 = kt * ntb0 - tab0
  db1 = kt * ntb1 - tab1
  Mada0 = wp.where(
    ra >= MJ_MINVAL,
    alpa * ((-na1 * tba0 + na0 * tba1) * na1 + rad_ma * (na0 * da0 + na1 * da1) * na0),
    Ma00 * da0 + Ma01 * da1,
  )
  Mada1 = wp.where(
    ra >= MJ_MINVAL,
    alpa * ((na1 * tba0 - na0 * tba1) * na0 + rad_ma * (na0 * da0 + na1 * da1) * na1),
    Ma01 * da0 + Ma11 * da1,
  )
  Mbdb0 = wp.where(
    rb >= MJ_MINVAL,
    alpb * ((-nb1 * tab0 + nb0 * tab1) * nb1 + rad_mb * (nb0 * db0 + nb1 * db1) * nb0),
    Mb00 * db0 + Mb01 * db1,
  )
  Mbdb1 = wp.where(
    rb >= MJ_MINVAL,
    alpb * ((nb1 * tab0 - nb0 * tab1) * nb0 + rad_mb * (nb0 * db0 + nb1 * db1) * nb1),
    Mb01 * db0 + Mb11 * db1,
  )
  g = wp.vec4(
    2.0 * w0 * sa * om[0] + 2.0 * Mada0,
    2.0 * w0 * sa * om[1] + 2.0 * Mada1,
    2.0 * w1 * sb * om[2] + 2.0 * Mbdb0,
    2.0 * w1 * sb * om[3] + 2.0 * Mbdb1,
  )

  # Hessian: clamped radial bending + Gauss-Newton torsion 2 kt Jr' Jr (PSD, exact at reference)
  tanga = wp.where(r0a > 0.0, wp.max(0.0, sa), 1.0)
  rada = wp.where(r0a > 0.0, wp.max(0.0, 1.0 - gam * e2 / (rhoa * rho2a)), 1.0)
  tangb = wp.where(r0b > 0.0, wp.max(0.0, sb), 1.0)
  radb = wp.where(r0b > 0.0, wp.max(0.0, 1.0 - gam * e2 / (rhob * rho2b)), 1.0)

  Wka00 = wp.where(
    r0a > 0.0 and ra < MJ_MINVAL,
    0.0,
    2.0 * w0 * (tanga * na_perp0 + rada * na0 * na0),
  )
  Wka01 = wp.where(r0a > 0.0 and ra < MJ_MINVAL, 0.0, 2.0 * w0 * (rada - tanga) * na0 * na1)
  Wka11 = wp.where(
    r0a > 0.0 and ra < MJ_MINVAL,
    0.0,
    2.0 * w0 * (tanga * na_perp1 + rada * na1 * na1),
  )

  Wkb00 = wp.where(
    r0b > 0.0 and rb < MJ_MINVAL,
    0.0,
    2.0 * w1 * (tangb * nb_perp0 + radb * nb0 * nb0),
  )
  Wkb01 = wp.where(r0b > 0.0 and rb < MJ_MINVAL, 0.0, 2.0 * w1 * (radb - tangb) * nb0 * nb1)
  Wkb11 = wp.where(
    r0b > 0.0 and rb < MJ_MINVAL,
    0.0,
    2.0 * w1 * (tangb * nb_perp1 + radb * nb1 * nb1),
  )

  Ma2_00 = Ma00 * Ma00 + Ma01 * Ma01
  Ma2_01 = Ma00 * Ma01 + Ma01 * Ma11
  Ma2_11 = Ma01 * Ma01 + Ma11 * Ma11

  Mb2_00 = Mb00 * Mb00 + Mb01 * Mb01
  Mb2_01 = Mb00 * Mb01 + Mb01 * Mb11
  Mb2_11 = Mb01 * Mb01 + Mb11 * Mb11

  MaTt00 = Ma00 * tc - Ma01 * ts
  MaTt01 = Ma00 * ts + Ma01 * tc
  MaTt10 = Ma01 * tc - Ma11 * ts
  MaTt11 = Ma01 * ts + Ma11 * tc

  Kab00 = -2.0 * (MaTt00 * Mb00 + MaTt01 * Mb01)
  Kab01 = -2.0 * (MaTt00 * Mb01 + MaTt01 * Mb11)
  Kab10 = -2.0 * (MaTt10 * Mb00 + MaTt11 * Mb01)
  Kab11 = -2.0 * (MaTt10 * Mb01 + MaTt11 * Mb11)

  Wk = wp.mat44(
    Wka00 + 2.0 * kt * Ma2_00,
    Wka01 + 2.0 * kt * Ma2_01,
    Kab00,
    Kab01,
    Wka01 + 2.0 * kt * Ma2_01,
    Wka11 + 2.0 * kt * Ma2_11,
    Kab10,
    Kab11,
    Kab00,
    Kab10,
    Wkb00 + 2.0 * kt * Mb2_00,
    Wkb01 + 2.0 * kt * Mb2_01,
    Kab01,
    Kab11,
    Wkb01 + 2.0 * kt * Mb2_01,
    Wkb11 + 2.0 * kt * Mb2_11,
  )

  # damping: radial curvature rates + dihedral twist rate, orthogonal to edge rotations
  Wda00 = wp.where(ra < MJ_MINVAL, 2.0 * w0, 2.0 * w0 * na0 * na0)
  Wda01 = wp.where(ra < MJ_MINVAL, 0.0, 2.0 * w0 * na0 * na1)
  Wda11 = wp.where(ra < MJ_MINVAL, 2.0 * w0, 2.0 * w0 * na1 * na1)

  Wdb00 = wp.where(rb < MJ_MINVAL, 2.0 * w1, 2.0 * w1 * nb0 * nb0)
  Wdb01 = wp.where(rb < MJ_MINVAL, 0.0, 2.0 * w1 * nb0 * nb1)
  Wdb11 = wp.where(rb < MJ_MINVAL, 2.0 * w1, 2.0 * w1 * nb1 * nb1)

  qd = wp.vec4(0.0)
  if ra >= MJ_MINVAL and rb >= MJ_MINVAL and r0a > 0.0 and r0b > 0.0:
    scla_d = alpa * (cb / rhob)
    sclb_d = alpb * (ca / rhoa)
    qd = wp.vec4(scla_d * na1, -scla_d * na0, -sclb_d * nb1, sclb_d * nb0)

  Wd = wp.mat44(
    Wda00,
    Wda01,
    0.0,
    0.0,
    Wda01,
    Wda11,
    0.0,
    0.0,
    0.0,
    0.0,
    Wdb00,
    Wdb01,
    0.0,
    0.0,
    Wdb01,
    Wdb11,
  ) + (2.0 * kt) * wp.outer(qd, qd)

  return g, Wk, Wd


@wp.func
def _rod_rows(a: wp.mat33) -> mat43:
  """Pads a 3x3 single hinge Jacobian block to the 4x3 block of a rod hinge record."""
  res = mat43()
  res[0] = a[0]
  res[1] = a[1]
  res[2] = a[2]
  return res


@wp.func
def _rod_pad44(a: wp.mat33) -> wp.mat44:
  """Pads a 3x3 single hinge weight to the 4x4 weight of a rod hinge record."""
  res = wp.mat44(0.0)
  for i in range(3):
    for j in range(3):
      res[i, j] = a[i, j]
  return res


@wp.kernel
def _eval_rod_chains(
  # Model:
  qpos0: wp.array2d[float],
  jnt_qposadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_bendingadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_bending: wp.array[float],
  flex_rod_is_chain: wp.array[bool],
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_hingeadr: wp.array[int],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  flex_rodedge_jnt: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  flex_rod_chain_valid_in: wp.array2d[bool],
  flex_rod_ref_ok_in: wp.array2d[bool],
  flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
  flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
  flex_rod_tail_loc_in: wp.array3d[wp.vec3],
  flex_rodhinge_om0_in: wp.array2d[wp.vec2],
  # Data out:
  flex_rod_chain_ok_out: wp.array2d[bool],
  flex_rodhinge_g_out: wp.array2d[wp.vec3],
  flex_rodhinge_Wk_out: wp.array2d[wp.mat33],
  flex_rodhinge_Wd_out: wp.array2d[wp.mat33],
  flex_rodedge_ktheta_out: wp.array2d[float],
  flex_rodedge_grad_out: wp.array2d[float],
  flex_rodedge_frc_out: wp.array2d[wp.vec3],
  flex_rod_end_torque_out: wp.array2d[wp.spatial_vector],
  flex_rod_chain_valid_out: wp.array2d[bool],
):
  """Evaluates 1D DER rod chains (mj_rodChain) in two streaming passes with zero edge cap.

  Runs once per position state: smooth.flex invalidates flex_rod_chain_valid, and later calls
  return early until it does again.
  """
  worldid, f = wp.tid()
  if flex_rod_chain_valid_in[worldid, f]:
    return
  flex_rod_chain_valid_out[worldid, f] = True
  z3 = wp.vec3(0.0)
  flex_rod_chain_ok_out[worldid, f] = False
  flex_rod_end_torque_out[worldid, f] = wp.spatial_vector(z3, z3)
  if wp.static(ROD_HINGE_SIZE != 10) or not flex_rod_is_chain[f] or not flex_rod_ref_ok_in[worldid, f]:
    return

  bendingadr = flex_bendingadr[f]
  hadr = flex_rod_hingeadr[f]
  eadr = flex_rod_edgeadr[f]
  nedge = flex_rod_edgenum[f]
  nhinge = nedge - 1
  if bendingadr < 0 or hadr < 0 or eadr < 0 or nhinge <= 0:
    return

  va = flex_vertadr[f]
  padr = flex_plasticadr[f]
  root_clamped = flex_rod_root_clamped[f]
  tail_clamped = flex_rod_tail_clamped[f]

  # Pass 1: validate all edge directors, hinge parallel transports, and curvature binormals.
  ok0, _anc_p, _tp_p, t_prev, u_prev, _vp_p, _l_p = support.rod_chain_edge_director(
    flex_vertadr,
    flex_plasticadr,
    flex_vertbodyid,
    flex_rod_root_clamped,
    flex_rod_tail_clamped,
    flex_rod_edgeadr,
    flex_rod_edgenum,
    flex_rodedge_vert,
    plastic_in,
    xmat_in,
    flexvert_xpos_in,
    flex_rod_frame0_loc_in,
    flex_rod_frame0_ref_in,
    flex_rod_tail_loc_in,
    worldid,
    f,
    0,
    z3,
    z3,
  )
  if not ok0:
    return

  has_ktheta = bool(False)
  for h in range(nhinge):
    ok1, _anc_c, _tp_c, t_cur, u_cur, _vp_c, _l_c = support.rod_chain_edge_director(
      flex_vertadr,
      flex_plasticadr,
      flex_vertbodyid,
      flex_rod_root_clamped,
      flex_rod_tail_clamped,
      flex_rod_edgeadr,
      flex_rod_edgenum,
      flex_rodedge_vert,
      plastic_in,
      xmat_in,
      flexvert_xpos_in,
      flex_rod_frame0_loc_in,
      flex_rod_frame0_ref_in,
      flex_rod_tail_loc_in,
      worldid,
      f,
      h + 1,
      t_prev,
      u_prev,
    )
    if not ok1:
      return
    ok_pth, _ut, _vt = support.rod_parallel_transport(t_prev, t_cur, u_prev)
    if not ok_pth:
      return
    b_h = bendingadr + 1 + ROD_HINGE_SIZE * h
    gv0 = va + int(flex_bending[b_h + 0])
    gv1 = va + int(flex_bending[b_h + 1])
    gv2 = va + int(flex_bending[b_h + 2])
    ok_kb, _kb_c, _A0_c, _A1_c = _rod_hinge(
      flexvert_xpos_in[worldid, gv0],
      flexvert_xpos_in[worldid, gv1],
      flexvert_xpos_in[worldid, gv2],
    )
    if not ok_kb:
      return
    if flex_bending[b_h + 5] != 0.0 or flex_bending[b_h + 6] > 0.0 or flex_bending[b_h + 9] != 0.0:
      has_ktheta = bool(True)
    t_prev = t_cur
    u_prev = u_cur

  # Pass 2: stream along hinges, writing per-hinge weights and per-edge forces/stiffnesses.
  q0_id = worldid % qpos0.shape[0]
  _ok0, anc0, tp0, t0, u0, v0, l0 = support.rod_chain_edge_director(
    flex_vertadr,
    flex_plasticadr,
    flex_vertbodyid,
    flex_rod_root_clamped,
    flex_rod_tail_clamped,
    flex_rod_edgeadr,
    flex_rod_edgenum,
    flex_rodedge_vert,
    plastic_in,
    xmat_in,
    flexvert_xpos_in,
    flex_rod_frame0_loc_in,
    flex_rod_frame0_ref_in,
    flex_rod_tail_loc_in,
    worldid,
    f,
    0,
    z3,
    z3,
  )
  jnt0 = flex_rodedge_jnt[eadr + 0]
  theta0 = (qpos_in[worldid, jnt_qposadr[jnt0]] - qpos0[q0_id, jnt_qposadr[jnt0]]) if jnt0 >= 0 else 0.0
  grad0 = float(0.0)
  ktheta0 = float(0.0)
  ge0 = z3
  tau_b0 = z3
  tau_bT = z3

  for h in range(nhinge):
    e1 = h + 1
    _ok1, anc1, tp1, t1, u1, v1, l1 = support.rod_chain_edge_director(
      flex_vertadr,
      flex_plasticadr,
      flex_vertbodyid,
      flex_rod_root_clamped,
      flex_rod_tail_clamped,
      flex_rod_edgeadr,
      flex_rod_edgenum,
      flex_rodedge_vert,
      plastic_in,
      xmat_in,
      flexvert_xpos_in,
      flex_rod_frame0_loc_in,
      flex_rod_frame0_ref_in,
      flex_rod_tail_loc_in,
      worldid,
      f,
      e1,
      t0,
      u0,
    )
    jnt1 = flex_rodedge_jnt[eadr + e1]
    theta1 = (qpos_in[worldid, jnt_qposadr[jnt1]] - qpos0[q0_id, jnt_qposadr[jnt1]]) if jnt1 >= 0 else 0.0
    grad1 = float(0.0)
    ktheta1 = float(0.0)
    ge1 = z3

    _ok_pth, u_trans, v_trans = support.rod_parallel_transport(t0, t1, u0)
    dpsi_h = wp.atan2(wp.dot(u1, v_trans), wp.dot(u1, u_trans))
    if padr >= 0 and anc0 and anc1:
      psi_prev = plastic_in[worldid, padr + 6 * nedge + h]
      jump = dpsi_h - psi_prev
      dpsi_h -= 2.0 * wp.pi * wp.round(jump / (2.0 * wp.pi))

    b_h = bendingadr + 1 + ROD_HINGE_SIZE * h
    v0_h = va + int(flex_bending[b_h + 0])
    v1_h = va + int(flex_bending[b_h + 1])
    v2_h = va + int(flex_bending[b_h + 2])
    _ok_kb, kb_h, _A0_h, _A1_h = _rod_hinge(
      flexvert_xpos_in[worldid, v0_h],
      flexvert_xpos_in[worldid, v1_h],
      flexvert_xpos_in[worldid, v2_h],
    )

    k1 = flex_bending[b_h + 4]
    k2 = k1 + flex_bending[b_h + 5]
    kt = flex_bending[b_h + 8]
    kd1 = wp.sqrt(k1 * k2)
    om0_h = flex_rodhinge_om0_in[worldid, hadr + h]
    w1_0 = om0_h[0]
    w2_0 = om0_h[1]

    c0 = wp.cos(theta0)
    s0 = wp.sin(theta0)
    m1_0 = c0 * u0 + s0 * v0
    m2_0 = -s0 * u0 + c0 * v0
    om0_0 = wp.dot(kb_h, u0)
    om0_1 = wp.dot(kb_h, v0)
    w1_s0 = c0 * om0_0 + s0 * om0_1
    w2_s0 = -s0 * om0_0 + c0 * om0_1
    dw1_0 = w1_s0 - w1_0
    dw2_0 = w2_s0 - w2_0
    grad0 += k1 * dw1_0 * w2_s0 - k2 * dw2_0 * w1_s0
    if has_ktheta:
      ktheta0 += wp.max(0.0, k1 * (w2_s0 * w2_s0 - dw1_0 * w1_s0) + k2 * (w1_s0 * w1_s0 - dw2_0 * w2_s0))

    c1 = wp.cos(theta1)
    s1 = wp.sin(theta1)
    m1_1 = c1 * u1 + s1 * v1
    m2_1 = -s1 * u1 + c1 * v1
    om1_0 = wp.dot(kb_h, u1)
    om1_1 = wp.dot(kb_h, v1)
    w1_s1 = c1 * om1_0 + s1 * om1_1
    w2_s1 = -s1 * om1_0 + c1 * om1_1
    dw1_1 = w1_s1 - w1_0
    dw2_1 = w2_s1 - w2_0
    grad1 += k1 * dw1_1 * w2_s1 - k2 * dw2_1 * w1_s1
    if has_ktheta:
      ktheta1 += wp.max(0.0, k1 * (w2_s1 * w2_s1 - dw1_1 * w1_s1) + k2 * (w1_s1 * w1_s1 - dw2_1 * w2_s1))

    dtheta = theta1 - theta0 + dpsi_h
    tau_h = 2.0 * kt * dtheta
    grad0 -= tau_h
    grad1 += tau_h
    ge0 += (tau_h / (2.0 * l0)) * kb_h
    ge1 += (tau_h / (2.0 * l1)) * kb_h

    g_kb = (k1 * dw1_0) * m1_0 + (k2 * dw2_0) * m2_0 + (k1 * dw1_1) * m1_1 + (k2 * dw2_1) * m2_1
    Wk_h = k1 * (wp.outer(m1_0, m1_0) + wp.outer(m1_1, m1_1)) + k2 * (wp.outer(m2_0, m2_0) + wp.outer(m2_1, m2_1))
    Wd_iso = kd1 * (wp.outer(m1_0, m1_0) + wp.outer(m1_1, m1_1)) + k2 * (wp.outer(m2_0, m2_0) + wp.outer(m2_1, m2_1))
    r_kb = wp.length(kb_h)
    if r_kb < MJ_MINVAL:
      Wd_h = Wd_iso
    else:
      n_kb = kb_h / r_kb
      keff2 = wp.dot(n_kb, Wd_iso * n_kb)
      Wd_h = keff2 * wp.outer(n_kb, n_kb)

    flex_rodhinge_g_out[worldid, hadr + h] = g_kb
    flex_rodhinge_Wk_out[worldid, hadr + h] = Wk_h
    flex_rodhinge_Wd_out[worldid, hadr + h] = Wd_h

    if grad0 != 0.0:
      d0_edge = 1.0 + wp.dot(tp0, t0)
      denom0 = l0 * d0_edge
      if denom0 >= MJ_MINVAL:
        ge0 += (-grad0 / denom0) * wp.cross(tp0, t0)
      if h == 0 and root_clamped and d0_edge >= MJ_MINVAL:
        tau_b0 = (-grad0 / d0_edge) * (tp0 + t0)

    flex_rodedge_grad_out[worldid, eadr + h] = grad0
    flex_rodedge_frc_out[worldid, eadr + h] = ge0
    flex_rodedge_ktheta_out[worldid, eadr + h] = ktheta0

    anc0 = anc1
    tp0 = tp1
    t0 = t1
    u0 = u1
    v0 = v1
    l0 = l1
    theta0 = theta1
    grad0 = grad1
    ktheta0 = ktheta1
    ge0 = ge1

  if grad0 != 0.0:
    dT_edge = 1.0 + wp.dot(tp0, t0)
    denom_last = l0 * dT_edge
    if denom_last >= MJ_MINVAL:
      ge0 += (-grad0 / denom_last) * wp.cross(tp0, t0)
    if tail_clamped and dT_edge >= MJ_MINVAL:
      tau_bT = (-grad0 / dT_edge) * (tp0 + t0)

  flex_rodedge_grad_out[worldid, eadr + nhinge] = grad0
  flex_rodedge_frc_out[worldid, eadr + nhinge] = ge0
  flex_rodedge_ktheta_out[worldid, eadr + nhinge] = ktheta0
  flex_rod_end_torque_out[worldid, f] = wp.spatial_vector(tau_b0, tau_bT)
  flex_rod_chain_ok_out[worldid, f] = True


def eval_rod_chains(m: Model, d: Data):
  """Evaluates all 1D DER rod chains into cached Data arrays, once per position state.

  smooth.flex computes the per-world rod reference frames and invalidates the cache; the first
  consumer (eff_build or passive) evaluates the chains and later calls return early.
  """
  if m.nflex == 0:
    return
  if m.nflexrodedge == 0:
    d.flex_rod_chain_ok.zero_()
    return
  wp.launch(
    _eval_rod_chains,
    dim=(d.nworld, m.nflex),
    inputs=[
      m.qpos0,
      m.jnt_qposadr,
      m.flex_vertadr,
      m.flex_bendingadr,
      m.flex_plasticadr,
      m.flex_vertbodyid,
      m.flex_bending,
      m.flex_rod_is_chain,
      m.flex_rod_root_clamped,
      m.flex_rod_tail_clamped,
      m.flex_rod_hingeadr,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_vert,
      m.flex_rodedge_jnt,
      d.qpos,
      d.plastic,
      d.xmat,
      d.flexvert_xpos,
      d.flex_rod_chain_valid,
      d.flex_rod_ref_ok,
      d.flex_rod_frame0_loc,
      d.flex_rod_frame0_ref,
      d.flex_rod_tail_loc,
      d.flex_rodhinge_om0,
    ],
    outputs=[
      d.flex_rod_chain_ok,
      d.flex_rodhinge_g,
      d.flex_rodhinge_Wk,
      d.flex_rodhinge_Wd,
      d.flex_rodedge_ktheta,
      d.flex_rodedge_grad,
      d.flex_rodedge_frc,
      d.flex_rod_end_torque,
      d.flex_rod_chain_valid,
    ],
  )


@wp.func
def rod_stencil(
  # Model:
  flex_bending: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  worldid: int,
  vbase: int,
  adr: int,
) -> tuple[bool, wp.vec4i, mat43, mat43, mat43, mat43, wp.vec4, wp.mat44, wp.mat44]:
  """Vertices, Jacobian, and weights of the 1D rod hinge record at flex_bending[adr].

  A record is a single hinge [v0, v1, v2, -1, k, 0, kb0, 0, 0, 0] (energy in kb space, last
  vertex -1 and last Jacobian row 0) or a hinge pair [v0, v1, v2, v3, w0, w1, om0] (energy in om
  space), see _rod_hinge_weights and _rod_pair_weights. Returns whether the record is valid, its
  global vertex ids, the Jacobian blocks J0..J3 of its vertices, and the gradient g, Hessian Wk,
  and damping weight Wd, so that the forces are -J'g and the stiffness is J'WkJ.
  """
  v0 = vbase + int(flex_bending[adr])
  v1 = vbase + int(flex_bending[adr + 1])
  v2 = vbase + int(flex_bending[adr + 2])
  x0 = flexvert_xpos_in[worldid, v0]
  x1 = flexvert_xpos_in[worldid, v1]
  x2 = flexvert_xpos_in[worldid, v2]

  # straight hinges only: [v0, v1, v2, k]
  if wp.static(ROD_HINGE_SIZE == 4):
    v3 = -1
    k = flex_bending[adr + 3]
    kb0 = 0.0
  else:
    v3 = int(flex_bending[adr + 3])
    k = flex_bending[adr + 4]
    kb0 = flex_bending[adr + 6]

  if v3 < 0:
    ok, kb, A0, A1 = _rod_hinge(x0, x1, x2)
    if not ok:
      return (
        False,
        wp.vec4i(-1),
        mat43(),
        mat43(),
        mat43(),
        mat43(),
        wp.vec4(0.0),
        wp.mat44(0.0),
        wp.mat44(0.0),
      )
    gh, Wkh, Wdh = _rod_hinge_weights(kb, k, kb0)
    return (
      True,
      wp.vec4i(v0, v1, v2, -1),
      _rod_rows(-A0),
      _rod_rows(A0 - A1),
      _rod_rows(A1),
      mat43(),
      wp.vec4(gh[0], gh[1], gh[2], 0.0),
      _rod_pad44(Wkh),
      _rod_pad44(Wdh),
    )

  v3 += vbase
  ok, om, J0, J1, J2, J3 = _rod_pair(x0, x1, x2, flexvert_xpos_in[worldid, v3])
  if not ok:
    return (
      False,
      wp.vec4i(-1),
      mat43(),
      mat43(),
      mat43(),
      mat43(),
      wp.vec4(0.0),
      wp.mat44(0.0),
      wp.mat44(0.0),
    )
  om0 = wp.vec4(
    flex_bending[adr + 6],
    flex_bending[adr + 7],
    flex_bending[adr + 8],
    flex_bending[adr + 9],
  )
  g, Wk, Wd = _rod_pair_weights(om, flex_bending[adr + 4], flex_bending[adr + 5], om0)
  return True, wp.vec4i(v0, v1, v2, v3), J0, J1, J2, J3, g, Wk, Wd


@wp.func
def rod_stencil_jacobian(J0: mat43, J1: mat43, J2: mat43, J3: mat43, i: int) -> mat43:
  """Block i (0..3) of the Jacobian of a 1D rod hinge record, see rod_stencil."""
  if i == 0:
    return J0
  if i == 1:
    return J1
  if i == 2:
    return J2
  return J3


@cache_kernel
def _flex_gather_kernel(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_simple: wp.array[int],
    dof_parentid: wp.array[int],
    flex_vertbodyid: wp.array[int],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[wp.vec3],
  ):
    worldid, gvert = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    res_out[worldid, gvert] = flex_gather_vert(
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
      vec_in,
      worldid,
      gvert,
    )

  return kernel


@cache_kernel
def _flex_scatter_kernel(check_skip: bool, nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_simple: wp.array[int],
    dof_parentid: wp.array[int],
    flex_vertbodyid: wp.array[int],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    vec_in: wp.array2d[wp.vec3],
    scale: float,
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, gvert = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    flex_scatter_vert(
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
      worldid,
      gvert,
      vec_in[worldid, gvert],
      scale,
      res_out,
    )

  return kernel


def flex_gather(
  m: Model,
  d: Data,
  res: wp.array2d[wp.vec3],
  vec: wp.array2d[float],
  skip: Optional[wp.array] = None,
):
  """Gathers generalized vector into world-space flex vertex coordinates."""
  if m.nflexvert == 0:
    return
  check_skip = skip is not None
  skip_arr = skip if check_skip else m.body_is_free
  wp.launch(
    _flex_gather_kernel(check_skip),
    dim=(d.nworld, m.nflexvert),
    inputs=[
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_simple,
      m.dof_parentid,
      m.flex_vertbodyid,
      d.subtree_com,
      d.cdof,
      d.flexvert_xpos,
      vec,
      skip_arr,
    ],
    outputs=[res],
  )


def flex_scatter(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[wp.vec3],
  scale: float = 1.0,
  skip: Optional[wp.array] = None,
):
  """Scatters world-space flex vertex forces into generalized coordinates."""
  if m.nflexvert == 0 or scale == 0.0:
    return
  check_skip = skip is not None
  skip_arr = skip if check_skip else m.body_is_free
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  wp.launch(
    _flex_scatter_kernel(check_skip, m.nv if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.nflexvert),
    inputs=[
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_simple,
      m.dof_parentid,
      m.flex_vertbodyid,
      d.subtree_com,
      d.cdof,
      d.flexvert_xpos,
      vec,
      scale,
      skip_arr,
    ],
    outputs=[res],
  )


@wp.func
def _flex_hessian_active(
  # Model:
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_stiffness: wp.array[float],
  flex_rigid: wp.array[bool],
  # In:
  f: int,
) -> bool:
  if f < 0:
    return False
  if flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2:
    return False
  stiffness_adr = flex_stiffnessadr[f]
  if stiffness_adr < 0:
    return False
  return flex_stiffness[stiffness_adr] != 0.0


@wp.kernel
def _flex_hessian_clear(
  # Model:
  nflexvert: int,
  nflexedge: int,
  flex_edgeflexid: wp.array[int],
  flex_vertflexid: wp.array[int],
  # Data in:
  flex_hessian_valid_in: wp.array2d[bool],
  # Data out:
  flexvert_hessian_out: wp.array2d[vec6],
  flexedge_hessian_out: wp.array2d[wp.mat33],
):
  worldid, tid = wp.tid()
  if tid < nflexvert:
    f_v = flex_vertflexid[tid]
    if f_v >= 0 and not flex_hessian_valid_in[worldid, f_v]:
      flexvert_hessian_out[worldid, tid] = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  if tid < nflexedge:
    f_e = flex_edgeflexid[tid]
    if f_e >= 0 and not flex_hessian_valid_in[worldid, f_e]:
      flexedge_hessian_out[worldid, tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@cache_kernel
def _flex_hessian_elem(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_edgeadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_elemedgeadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_plasticmode: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_elem: wp.array[int],
    flex_elemedge: wp.array[int],
    flex_vert0: wp.array[wp.vec3],
    flexedge_length0: wp.array[float],
    flex_size: wp.array[wp.vec3],
    flex_stiffness: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_elemflexid: wp.array[int],
    # Data in:
    plastic_in: wp.array2d[float],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    flex_hessian_valid_in: wp.array2d[bool],
    flexedge_length_in: wp.array2d[float],
    # Data out:
    flexedge_hessian_out: wp.array2d[wp.mat33],
  ):
    worldid, elemid = wp.tid()
    f = flex_elemflexid[elemid]
    if flex_hessian_valid_in[worldid, f]:
      return
    if not _flex_hessian_active(
      flex_dim,
      flex_interp,
      flex_stiffnessadr,
      flex_stiffness,
      flex_rigid,
      f,
    ):
      return

    local_elemid = elemid - flex_elemadr[f]
    dim = flex_dim[f]
    nvrt = dim + 1
    nedge = 3 if dim == 2 else 6
    elem_data_adr = flex_elemdataadr[f] + local_elemid * nvrt
    vbase = flex_vertadr[f]
    ebase = flex_edgeadr[f]
    ee_base = flex_elemedgeadr[f] + local_elemid * nedge
    stiffness_adr = flex_stiffnessadr[f] + local_elemid * (FLEX_STIFFNESS_3D if dim == 3 else 21)
    snh = wp.static(FLEX_STIFFNESS_3D == 24) and dim == 3 and flex_stiffness[stiffness_adr + 21] != 0.0
    padr = flex_plasticadr[f] + local_elemid * nedge if (flex_plasticmode[f] & 2) != 0 else -1

    elem_verts = wp.vec4i(-1, -1, -1, -1)
    vert_xpos = mat43()
    for v in range(nvrt):
      vert_idx = flex_elem[elem_data_adr + v]
      elem_verts[v] = vert_idx
      vert_xpos[v] = flexvert_xpos_in[worldid, vbase + vert_idx]

    edgevec = stretch_edge_vectors(vert_xpos, dim)

    metric = mat66(0.0)
    tension = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rotation = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    proj_grad = mat43()
    stretch = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    symmetric = wp.vec3(0.0, 0.0, 0.0)
    skew = wp.vec3(0.0, 0.0, 0.0)

    if snh:
      elongation = stretch_elongation(
        flex_elemedge,
        flexedge_length0,
        plastic_in,
        flexedge_length_in,
        worldid,
        ee_base,
        ebase,
        padr,
        6,
      )
      active, rotation, proj_grad, stretch, symmetric, skew = snh_project(
        flex_vert0,
        flex_stiffness,
        stiffness_adr,
        vbase,
        elem_verts,
        flex_size[f],
        edgevec,
        elongation,
      )
      if not active:
        return
    else:
      metric, tension = stretch_stiffness(
        flex_elemedge,
        flexedge_length0,
        flex_stiffness,
        plastic_in,
        flexedge_length_in,
        worldid,
        stiffness_adr,
        ee_base,
        ebase,
        padr,
        nedge,
      )

    edges = stretch_edge_endpoints(dim)
    for e in range(6):
      if e < nedge:
        i = edges[e, 0]
        j = edges[e, 1]
        edge_id = ebase + flex_elemedge[ee_base + e]
        endpoints = flex_edge[edge_id]
        if endpoints[0] != elem_verts[i]:
          swap = i
          i = j
          j = swap
        if snh:
          block = snh_projected_block(rotation, proj_grad, stretch, symmetric, skew, i, j)
        else:
          block = stretch_stiffness_block(metric, tension, edgevec, dim, i, j, 1.0)
        wp.atomic_add(flexedge_hessian_out, worldid, edge_id, block)

  return kernel


@cache_kernel
def _flex_hessian_diag(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_stiffness: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_edgeflexid: wp.array[int],
    # Data in:
    flex_hessian_valid_in: wp.array2d[bool],
    flexedge_hessian_in: wp.array2d[wp.mat33],
    # Data out:
    flexvert_hessian_out: wp.array2d[vec6],
  ):
    worldid, edgeid = wp.tid()
    f = flex_edgeflexid[edgeid]
    if flex_hessian_valid_in[worldid, f]:
      return
    if not _flex_hessian_active(
      flex_dim,
      flex_interp,
      flex_stiffnessadr,
      flex_stiffness,
      flex_rigid,
      f,
    ):
      return

    va = flex_vertadr[f]
    vert = flex_edge[edgeid]
    block = flexedge_hessian_in[worldid, edgeid]
    d0 = vec6(
      -block[0, 0],
      -block[0, 1],
      -block[0, 2],
      -block[1, 1],
      -block[1, 2],
      -block[2, 2],
    )
    d1 = vec6(
      -block[0, 0],
      -block[1, 0],
      -block[2, 0],
      -block[1, 1],
      -block[2, 1],
      -block[2, 2],
    )
    wp.atomic_add(flexvert_hessian_out, worldid, va + vert[0], d0)
    wp.atomic_add(flexvert_hessian_out, worldid, va + vert[1], d1)

  return kernel


def flex_hessian(m: Model, d: Data):
  """Caches the unscaled Cartesian stretch Hessian for all flexes."""
  if m.nflex == 0:
    return
  nclear = max(m.nflexvert, m.nflexedge)
  if nclear > 0:
    wp.launch(
      _flex_hessian_clear,
      dim=(d.nworld, nclear),
      inputs=[
        m.nflexvert,
        m.nflexedge,
        m.flex_edgeflexid,
        m.flex_vertflexid,
        d.flex_hessian_valid,
      ],
      outputs=[d.flexvert_hessian, d.flexedge_hessian],
    )
  if m.nflexelem > 0:
    det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
    wp.launch(
      _flex_hessian_elem(det_atomics),
      dim=(d.nworld, m.nflexelem),
      inputs=[
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_elemadr,
        m.flex_elemdataadr,
        m.flex_stiffnessadr,
        m.flex_elemedgeadr,
        m.flex_plasticadr,
        m.flex_plasticmode,
        m.flex_edge,
        m.flex_elem,
        m.flex_elemedge,
        m.flex_vert0,
        m.flexedge_length0,
        m.flex_size,
        m.flex_stiffness,
        m.flex_rigid,
        m.flex_elemflexid,
        d.plastic,
        d.flexvert_xpos,
        d.flex_hessian_valid,
        d.flexedge_length,
      ],
      outputs=[d.flexedge_hessian],
    )
    wp.launch(
      _flex_hessian_diag(det_atomics),
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_stiffnessadr,
        m.flex_edge,
        m.flex_stiffness,
        m.flex_rigid,
        m.flex_edgeflexid,
        d.flex_hessian_valid,
        d.flexedge_hessian,
      ],
      outputs=[d.flexvert_hessian],
    )
  d.flex_hessian_valid.fill_(True)


@wp.func
def _flex_stretch_mul_scale(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_stiffness: wp.array[float],
  flex_damping: wp.array[float],
  flex_rigid: wp.array[bool],
  flex_simple: wp.array[bool],
  # In:
  worldid: int,
  f: int,
  s1_in: float,
  s2_in: float,
  use_timestep: bool,
  non_simple_only: bool,
  snh_only: bool,
) -> float:
  if f < 0:
    return 0.0
  if flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2:
    return 0.0
  if non_simple_only and flex_simple[f]:
    return 0.0
  stiffness_adr = flex_stiffnessadr[f]
  if stiffness_adr < 0:
    return 0.0
  if flex_stiffness[stiffness_adr] == 0.0:
    return 0.0
  if snh_only:
    if not wp.static(FLEX_STIFFNESS_3D == 24) or flex_dim[f] != 3 or flex_stiffness[stiffness_adr + 21] == 0.0:
      return 0.0
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  if use_timestep:
    if s1_in < 0.0:
      s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else s1_in * timestep
    else:
      s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else s1_in * timestep * timestep
    s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else s2_in * timestep
    return s1 + s2 * flex_damping[f]
  if s1_in == 0.0 and timestep <= 0.0:
    return 0.0
  return s1_in + s2_in * flex_damping[f]


@cache_kernel
def _flex_hessian_mul_vert(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_vertflexid: wp.array[int],
    flex_simple: wp.array[bool],
    # Data in:
    flexvert_hessian_in: wp.array2d[vec6],
    # In:
    vec_in: wp.array2d[wp.vec3],
    skip: wp.array[bool],
    s1_in: float,
    s2_in: float,
    use_timestep: bool,
    non_simple_only: bool,
    snh_only: bool,
    # Out:
    res_out: wp.array2d[wp.vec3],
  ):
    worldid, vertid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    f = flex_vertflexid[vertid]
    scale = _flex_stretch_mul_scale(
      opt_timestep,
      opt_disableflags,
      flex_dim,
      flex_interp,
      flex_stiffnessadr,
      flex_stiffness,
      flex_damping,
      flex_rigid,
      flex_simple,
      worldid,
      f,
      s1_in,
      s2_in,
      use_timestep,
      non_simple_only,
      snh_only,
    )
    if scale == 0.0:
      res_out[worldid, vertid] = wp.vec3(0.0, 0.0, 0.0)
      return
    a = flexvert_hessian_in[worldid, vertid]
    x = vec_in[worldid, vertid]
    res_out[worldid, vertid] = scale * wp.vec3(
      a[0] * x[0] + a[1] * x[1] + a[2] * x[2],
      a[1] * x[0] + a[3] * x[1] + a[4] * x[2],
      a[2] * x[0] + a[4] * x[1] + a[5] * x[2],
    )

  return kernel


@cache_kernel
def _flex_hessian_mul_edge(check_skip: bool, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_edgeflexid: wp.array[int],
    flex_simple: wp.array[bool],
    # Data in:
    flexedge_hessian_in: wp.array2d[wp.mat33],
    # In:
    vec_in: wp.array2d[wp.vec3],
    skip: wp.array[bool],
    s1_in: float,
    s2_in: float,
    use_timestep: bool,
    non_simple_only: bool,
    snh_only: bool,
    # Out:
    res_out: wp.array2d[wp.vec3],
  ):
    worldid, edgeid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    f = flex_edgeflexid[edgeid]
    scale = _flex_stretch_mul_scale(
      opt_timestep,
      opt_disableflags,
      flex_dim,
      flex_interp,
      flex_stiffnessadr,
      flex_stiffness,
      flex_damping,
      flex_rigid,
      flex_simple,
      worldid,
      f,
      s1_in,
      s2_in,
      use_timestep,
      non_simple_only,
      snh_only,
    )
    if scale == 0.0:
      return
    va = flex_vertadr[f]
    v = flex_edge[edgeid]
    v0 = va + v[0]
    v1 = va + v[1]
    block = flexedge_hessian_in[worldid, edgeid]
    x = vec_in[worldid, v0]
    y = vec_in[worldid, v1]
    wp.atomic_add(res_out, worldid, v0, scale * (block * y))
    wp.atomic_add(res_out, worldid, v1, scale * (wp.transpose(block) * x))

  return kernel


def flex_hessian_mul(
  m: Model,
  d: Data,
  res: wp.array2d[wp.vec3],
  vec: wp.array2d[wp.vec3],
  s1_in: float = 1.0,
  s2_in: float = 0.0,
  use_timestep: bool = False,
  non_simple_only: bool = False,
  snh_only: bool = False,
  skip: Optional[wp.array] = None,
):
  """Computes res = (s1 + s2*flex_damping) * H * vec in world vertex coordinates."""
  if m.nflexvert == 0:
    return
  check_skip = skip is not None
  skip_arr = skip if check_skip else m.body_is_free
  wp.launch(
    _flex_hessian_mul_vert(check_skip),
    dim=(d.nworld, m.nflexvert),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.flex_dim,
      m.flex_interp,
      m.flex_stiffnessadr,
      m.flex_stiffness,
      m.flex_damping,
      m.flex_rigid,
      m.flex_vertflexid,
      m.flex_simple,
      d.flexvert_hessian,
      vec,
      skip_arr,
      s1_in,
      s2_in,
      use_timestep,
      non_simple_only,
      snh_only,
    ],
    outputs=[res],
  )
  wp.launch(
    _flex_hessian_mul_edge(check_skip, bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.nflexedge),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.flex_dim,
      m.flex_interp,
      m.flex_vertadr,
      m.flex_stiffnessadr,
      m.flex_edge,
      m.flex_stiffness,
      m.flex_damping,
      m.flex_rigid,
      m.flex_edgeflexid,
      m.flex_simple,
      d.flexedge_hessian,
      vec,
      skip_arr,
      s1_in,
      s2_in,
      use_timestep,
      non_simple_only,
      snh_only,
    ],
    outputs=[res],
  )


def flex_stretch_scratch(m: Model, d: Data) -> tuple[wp.array2d[wp.vec3], wp.array2d[wp.vec3]]:
  """Allocates the per-vertex scratch arrays used by flex_stretch_mul."""
  wvec = wp.empty((d.nworld, m.nflexvert), dtype=wp.vec3)
  wres = wp.empty((d.nworld, m.nflexvert), dtype=wp.vec3)
  return wvec, wres


def flex_stretch_mul(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  s1_in: float = 1.0,
  s2_in: float = 0.0,
  use_timestep: bool = True,
  non_simple_only: bool = False,
  snh_only: bool = False,
  skip: Optional[wp.array] = None,
  scratch: Optional[tuple[wp.array2d[wp.vec3], wp.array2d[wp.vec3]]] = None,
):
  """Adds (s1 + s2*flex_damping) * J^T * K_stretch * J * vec to res.

  Pass scratch (from flex_stretch_scratch) when calling inside wp.capture_while, where memory
  cannot be allocated.
  """
  if m.nflex == 0 or m.nflexvert == 0:
    return
  wvec, wres = scratch if scratch is not None else flex_stretch_scratch(m, d)
  flex_gather(m, d, wvec, vec, skip=skip)
  flex_hessian_mul(
    m,
    d,
    wres,
    wvec,
    s1_in=s1_in,
    s2_in=s2_in,
    use_timestep=use_timestep,
    non_simple_only=non_simple_only,
    snh_only=snh_only,
    skip=skip,
  )
  flex_scatter(m, d, res, wres, scale=1.0, skip=skip)


@cache_kernel
def _flex_bend_mul(check_skip: bool, nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 4 * nv

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
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
    flex_bendingadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_edgeflap: wp.array[wp.vec2i],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_edgeflexid: wp.array[int],
    flex_simple: wp.array[bool],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    s1_in: float,
    s2_in: float,
    use_timestep: bool,
    non_simple_only: bool,
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, edgeid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    f = flex_edgeflexid[edgeid]
    if f < 0:
      return
    if flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] != 2:
      return
    if non_simple_only and flex_simple[f]:
      return
    bendingadr = flex_bendingadr[f]
    if bendingadr < 0:
      return

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    if use_timestep:
      if s1_in < 0.0:
        s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else s1_in * timestep
      else:
        s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else s1_in * timestep * timestep
      s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else s2_in * timestep
      scale = s1 + s2 * flex_damping[f]
    else:
      scale = s1_in + s2_in * flex_damping[f]
    if scale == 0.0:
      return

    flap = flex_edgeflap[edgeid]
    if flap[1] == -1:
      return
    edge = flex_edge[edgeid]
    vbase = flex_vertadr[f]
    local_edgeid = edgeid - flex_edgeadr[f]
    bend_offset = bendingadr + 17 * local_edgeid
    verts = wp.vec4i(edge[0], edge[1], flap[0], flap[1])

    wvec = mat43()
    for j in range(4):
      wvec[j] = flex_gather_vert(
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
        vec_in,
        worldid,
        vbase + verts[j],
      )

    for i in range(4):
      vw = wp.vec3(0.0, 0.0, 0.0)
      for j in range(4):
        q = flex_bending[bend_offset + 4 * i + j]
        vw += q * wvec[j]
      flex_scatter_vert(
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
        worldid,
        vbase + verts[i],
        vw,
        scale,
        res_out,
      )

  return kernel


def flex_bend_mul(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  s1_in: float = 1.0,
  s2_in: float = 0.0,
  use_timestep: bool = True,
  non_simple_only: bool = False,
  skip: Optional[wp.array] = None,
):
  """Adds (s1 + s2*flex_damping) * J^T * K_bend * J * vec to res."""
  if m.nflex == 0 or m.nflexedge == 0:
    return
  check_skip = skip is not None
  skip_arr = skip if check_skip else m.body_is_free
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  wp.launch(
    _flex_bend_mul(check_skip, m.nv if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.nflexedge),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
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
      m.flex_bendingadr,
      m.flex_vertbodyid,
      m.flex_edge,
      m.flex_edgeflap,
      m.flex_bending,
      m.flex_damping,
      m.flex_rigid,
      m.flex_edgeflexid,
      m.flex_simple,
      d.subtree_com,
      d.cdof,
      d.flexvert_xpos,
      vec,
      skip_arr,
      s1_in,
      s2_in,
      use_timestep,
      non_simple_only,
    ],
    outputs=[res],
  )


@cache_kernel
def _flex_rod_mul(check_skip: bool, nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 4 * nv

  @wp.kernel(module="unique", module_options=module_options, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_simple: wp.array[int],
    dof_parentid: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    flex_simple: wp.array[bool],
    flex_rodhinge_map: wp.array[wp.vec2i],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    flex_rod_chain_ok_in: wp.array2d[bool],
    flex_rodhinge_Wk_in: wp.array2d[wp.mat33],
    flex_rodhinge_Wd_in: wp.array2d[wp.mat33],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    s1_in: float,
    s2_in: float,
    use_timestep: bool,
    non_simple_only: bool,
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, hingeid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    mapping = flex_rodhinge_map[hingeid]
    f = mapping[0]
    adr = mapping[1]
    if non_simple_only and flex_simple[f]:
      return

    if use_timestep:
      timestep = opt_timestep[worldid % opt_timestep.shape[0]]
      if s1_in < 0.0:
        s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else s1_in * timestep
      else:
        s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else s1_in * timestep * timestep
      s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else s2_in * timestep
    else:
      s1 = s1_in
      s2 = s2_in
    sd = s2 * flex_damping[f]
    if s1 == 0.0 and sd == 0.0:
      return

    ok, verts, J0, J1, J2, J3, _, Wk, Wd = rod_stencil(flex_bending, flexvert_xpos_in, worldid, flex_vertadr[f], adr)
    if not ok:
      return
    ok_chain = verts[3] < 0 and flex_rod_chain_ok_in[worldid, f]
    if ok_chain:
      Wk = _rod_pad44(flex_rodhinge_Wk_in[worldid, hingeid])
      Wd = _rod_pad44(flex_rodhinge_Wd_in[worldid, hingeid])
    W = s1 * Wk + sd * Wd
    nvert = 3 if verts[3] < 0 else 4

    # J' W J vec, without forming the hinge stiffness
    vel_v = mat43()
    u = wp.vec4(0.0)
    for i in range(nvert):
      vel_i = flex_gather_vert(
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
        vec_in,
        worldid,
        verts[i],
      )
      vel_v[i] = vel_i
      u += rod_stencil_jacobian(J0, J1, J2, J3, i) * vel_i
    w = W * u

    j_psi = mat43()
    dot_psi = float(0.0)
    if wp.static(ROD_HINGE_SIZE == 10) and verts[3] < 0 and ok_chain:
      wt = (s1 + sd) * 2.0 * flex_bending[adr + 8]
      if wt > 0.0:
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
          dot_psi = wt * (wp.dot(j_psi[0], vel_v[0]) + wp.dot(j_psi[1], vel_v[1]) + wp.dot(j_psi[2], vel_v[2]))

    for i in range(nvert):
      flex_scatter_vert(
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
        worldid,
        verts[i],
        wp.transpose(rod_stencil_jacobian(J0, J1, J2, J3, i)) * w + dot_psi * j_psi[i],
        1.0,
        res_out,
      )

  return kernel


def flex_rod_mul(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  s1_in: float = 1.0,
  s2_in: float = 0.0,
  use_timestep: bool = True,
  non_simple_only: bool = False,
  skip: Optional[wp.array] = None,
):
  """Adds J^T * (s1 * Wk + s2 * flex_damping * Wd) * J * vec to res for 1D rod bending hinges.

  With use_timestep, s1 is scaled by timestep^2 (by timestep when negative) and s2 by timestep.
  """
  check_skip = skip is not None
  skip_arr = skip if check_skip else m.body_is_free
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  wp.launch(
    _flex_rod_mul(check_skip, m.nv if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.nflexrodhinge),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_simple,
      m.dof_parentid,
      m.flex_vertadr,
      m.flex_vertbodyid,
      m.flex_bending,
      m.flex_damping,
      m.flex_simple,
      m.flex_rodhinge_map,
      d.subtree_com,
      d.cdof,
      d.flexvert_xpos,
      d.flex_rod_chain_ok,
      d.flex_rodhinge_Wk,
      d.flex_rodhinge_Wd,
      vec,
      skip_arr,
      s1_in,
      s2_in,
      use_timestep,
      non_simple_only,
    ],
    outputs=[res],
  )


@cache_kernel
def _flex_elasticity(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_edgeadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_elemedgeadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_plasticmode: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_elemedge: wp.array[int],
    flexedge_length0: wp.array[float],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_elemflexid: wp.array[int],
    # Data in:
    plastic_in: wp.array2d[float],
    xipos_in: wp.array2d[wp.vec3],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    flexedge_length_in: wp.array2d[float],
    flexedge_velocity_in: wp.array2d[float],
    # In:
    dsbl_spring: bool,
    dsbl_damper: bool,
    # Out:
    flex_spring_body_force_out: wp.array2d[wp.spatial_vector],
    flex_damper_body_force_out: wp.array2d[wp.spatial_vector],
  ):
    worldid, elemid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    f = flex_elemflexid[elemid]
    local_elemid = elemid - flex_elemadr[f]

    stiffness_adr_base = flex_stiffnessadr[f]
    if stiffness_adr_base < 0:
      return
    if flex_stiffness[stiffness_adr_base] == 0.0:
      return

    if timestep > 0.0 and not dsbl_damper:
      kD = flex_damping[f] / timestep
    else:
      kD = 0.0
    if dsbl_spring and kD == 0.0:
      return

    dim = flex_dim[f]
    nvert = dim + 1
    nedge = 3 if dim == 2 else 6

    stiffness_adr = stiffness_adr_base + local_elemid * (FLEX_STIFFNESS_3D if dim == 3 else 21)
    snh = wp.static(FLEX_STIFFNESS_3D == 24) and dim == 3 and flex_stiffness[stiffness_adr + 21] != 0.0
    if snh and dsbl_spring:
      return

    edges = wp.where(
      dim == 1,
      wp.matrix(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),
      stretch_edge_endpoints(dim),
    )

    elem_data_adr = flex_elemdataadr[f] + local_elemid * (dim + 1)
    vbase = flex_vertadr[f]

    vert_body = wp.vec4i(-1, -1, -1, -1)
    vert_xpos = mat43()
    for v in range(nvert):
      vert = flex_elem[elem_data_adr + v]
      gvert = vbase + vert
      bodyid = flex_vertbodyid[gvert]
      if v == 0 and bodyid < 0:
        return
      vert_body[v] = bodyid
      vert_xpos[v] = flexvert_xpos_in[worldid, gvert]

    edgevec = mat63()
    for e in range(nedge):
      edgevec[e] = vert_xpos[edges[e, 0]] - vert_xpos[edges[e, 1]]

    elemedge_adr = flex_elemedgeadr[f] + local_elemid * nedge
    edge_adr = flex_edgeadr[f]
    padr = flex_plasticadr[f] + local_elemid * nedge if (flex_plasticmode[f] & 2) != 0 else -1

    elongation_spring = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if not dsbl_spring:
      elongation_spring = stretch_elongation(
        flex_elemedge,
        flexedge_length0,
        plastic_in,
        flexedge_length_in,
        worldid,
        elemedge_adr,
        edge_adr,
        padr,
        nedge,
      )

    metric, tension_spring = stretch_elasticity(flex_stiffness, stiffness_adr, elongation_spring, nedge)
    if snh:
      tension_spring = _snh_cubic(tension_spring, elongation_spring, flex_stiffness[stiffness_adr + 21])

    tension_damper = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if kD > 0.0 and not snh:
      elongation_damper = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      for e in range(nedge):
        e_idx = edge_adr + flex_elemedge[elemedge_adr + e]
        deformed = flexedge_length_in[worldid, e_idx]
        vel = flexedge_velocity_in[worldid, e_idx]
        dL = vel * timestep
        elongation_damper[e] = dL * (2.0 * deformed - dL) * kD
      tension_damper = stretch_tension(metric, elongation_damper, nedge)

    force_spring = wp.matrix(0.0, shape=(6, 3))
    force_damper = wp.matrix(0.0, shape=(6, 3))
    for ed2 in range(nedge):
      s_val = tension_spring[ed2]
      d_val = tension_damper[ed2]
      for i in range(2):
        vi = edges[ed2, i]
        for x in range(3):
          g_val = edgevec[ed2, x] if i == 0 else -edgevec[ed2, x]
          if not dsbl_spring:
            force_spring[vi, x] -= s_val * g_val
          if kD > 0.0 and not snh:
            force_damper[vi, x] -= d_val * g_val

    if snh and not dsbl_spring:
      J, vol_grad = _snh_volume(edgevec, flex_stiffness[stiffness_adr + 23])
      pressure = 2.0 * flex_stiffness[stiffness_adr + 22] * (J - 1.0)
      for v in range(4):
        gv = vol_grad[v]
        for x in range(3):
          force_spring[v, x] -= pressure * gv[x]

    for v in range(4):
      if v < nvert:
        bodyid = vert_body[v]
        node_pos = vert_xpos[v]
        body_xipos = xipos_in[worldid, bodyid]
        offset = body_xipos - node_pos
        if not dsbl_spring:
          frc_s = force_spring[v]
          spatial_frc_s = wp.spatial_vector(frc_s, -wp.cross(offset, frc_s))
          wp.atomic_add(flex_spring_body_force_out, worldid, bodyid, spatial_frc_s)
        if kD > 0.0 and not snh:
          frc_d = force_damper[v]
          spatial_frc_d = wp.spatial_vector(frc_d, -wp.cross(offset, frc_d))
          wp.atomic_add(flex_damper_body_force_out, worldid, bodyid, spatial_frc_d)

  return kernel


@cache_kernel
def _flex_rod_bending(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 12 * nv + 4

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_simple: wp.array[int],
    jnt_dofadr: wp.array[int],
    dof_parentid: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    flex_rodhinge_map: wp.array[wp.vec2i],
    flex_rod_root_clamped: wp.array[bool],
    flex_rod_tail_clamped: wp.array[bool],
    flex_rod_hingeadr: wp.array[int],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_vert: wp.array[wp.vec2i],
    flex_rodedge_jnt: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    cvel_in: wp.array2d[wp.spatial_vector],
    flex_rod_chain_ok_in: wp.array2d[bool],
    flex_rodhinge_g_in: wp.array2d[wp.vec3],
    flex_rodhinge_Wd_in: wp.array2d[wp.mat33],
    flex_rodedge_grad_in: wp.array2d[float],
    flex_rodedge_frc_in: wp.array2d[wp.vec3],
    flex_rod_end_torque_in: wp.array2d[wp.spatial_vector],
    # In:
    dsbl_spring: bool,
    dsbl_damper: bool,
    # Data out:
    qfrc_spring_out: wp.array2d[float],
    qfrc_damper_out: wp.array2d[float],
  ):
    """Passive forces of 1D rod bending, see rod_stencil; damping excludes rigid rotations."""
    worldid, hingeid = wp.tid()
    mapping = flex_rodhinge_map[hingeid]
    f = mapping[0]
    adr = mapping[1]

    damping = flex_damping[f]
    has_spring = not dsbl_spring
    has_damper = not dsbl_damper and damping != 0.0
    if not has_spring and not has_damper:
      return

    va = flex_vertadr[f]
    ok, verts, J0, J1, J2, J3, g, _, Wd = rod_stencil(flex_bending, flexvert_xpos_in, worldid, va, adr)
    if not ok:
      return
    ok_chain = verts[3] < 0 and flex_rod_chain_ok_in[worldid, f]
    f_pt0 = wp.vec3(0.0)
    f_pt1 = wp.vec3(0.0)
    f_pt2 = wp.vec3(0.0)
    tau_b0 = wp.vec3(0.0)
    tau_bT = wp.vec3(0.0)
    twist_dof0 = int(-1)
    twist_frc0 = float(0.0)
    twist_dof1 = int(-1)
    twist_frc1 = float(0.0)
    h_idx = hingeid - flex_rod_hingeadr[f]
    eadr = flex_rod_edgeadr[f]
    nhinge = flex_rod_edgenum[f] - 1
    if ok_chain:
      g_c = flex_rodhinge_g_in[worldid, hingeid]
      g = wp.vec4(g_c[0], g_c[1], g_c[2], 0.0)
      Wd = _rod_pad44(flex_rodhinge_Wd_in[worldid, hingeid])
      if has_spring:
        ge_h = flex_rodedge_frc_in[worldid, eadr + h_idx]
        f_pt0 += ge_h
        f_pt1 -= ge_h
        jnt_h = flex_rodedge_jnt[eadr + h_idx]
        if jnt_h >= 0:
          twist_dof0 = jnt_dofadr[jnt_h]
          twist_frc0 = -flex_rodedge_grad_in[worldid, eadr + h_idx]
        if h_idx == 0:
          tau_b0 = wp.spatial_top(flex_rod_end_torque_in[worldid, f])
        if h_idx == nhinge - 1:
          ge_last = flex_rodedge_frc_in[worldid, eadr + nhinge]
          f_pt1 += ge_last
          f_pt2 -= ge_last
          jnt_last = flex_rodedge_jnt[eadr + nhinge]
          if jnt_last >= 0:
            twist_dof1 = jnt_dofadr[jnt_last]
            twist_frc1 = -flex_rodedge_grad_in[worldid, eadr + nhinge]
          tau_bT = wp.spatial_bottom(flex_rod_end_torque_in[worldid, f])
    nvert = 3 if verts[3] < 0 else 4

    # damper weight Wd * J * qvel and frame-indifferent torsional Rayleigh damping
    wd = wp.vec4(0.0)
    f_dmp = mat43()
    tau_b0_d = wp.vec3(0.0)
    tau_bT_d = wp.vec3(0.0)
    b0_chain = int(-1)
    bT_chain = int(-1)
    dmp_dof0 = int(-1)
    dmp_frc0 = float(0.0)
    dmp_dof1 = int(-1)
    dmp_frc1 = float(0.0)
    if has_damper:
      vvel = mat43()
      udot = wp.vec4(0.0)
      for i in range(nvert):
        vel_i = flex_gather_vert(
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
          verts[i],
        )
        vvel[i] = vel_i
        udot += rod_stencil_jacobian(J0, J1, J2, J3, i) * vel_i
      wd = Wd * udot

      if wp.static(ROD_HINGE_SIZE == 10) and verts[3] < 0 and ok_chain:
        kt = flex_bending[adr + 8]
        jnt0 = flex_rodedge_jnt[eadr + h_idx]
        jnt1 = flex_rodedge_jnt[eadr + h_idx + 1]
        if kt > 0.0 and (jnt0 >= 0 or jnt1 >= 0):
          b0_chain = flex_vertbodyid[va + flex_rodedge_vert[eadr + 0][0]]
          bT_chain = flex_vertbodyid[va + flex_rodedge_vert[eadr + nhinge][1]]
          root_clamped = flex_rod_root_clamped[f]
          tail_clamped = flex_rod_tail_clamped[f]
          x0 = flexvert_xpos_in[worldid, verts[0]]
          x1 = flexvert_xpos_in[worldid, verts[1]]
          x2 = flexvert_xpos_in[worldid, verts[2]]
          e0 = x1 - x0
          e1 = x2 - x1
          l0 = wp.length(e0)
          l1 = wp.length(e1)
          ok_kb, kb_d, _A0_d, _A1_d = _rod_hinge(x0, x1, x2)
          if ok_kb and l0 >= MJ_MINVAL and l1 >= MJ_MINVAL:
            w0 = float(0.0)
            if jnt0 >= 0:
              w0 = qvel_in[worldid, jnt_dofadr[jnt0]]
            elif h_idx == 0 and root_clamped and b0_chain > 0:
              w0 = wp.dot(wp.spatial_top(cvel_in[worldid, b0_chain]), e0) / l0
            tail = (h_idx == nhinge - 1) and tail_clamped and (bT_chain > 0)
            w1 = float(0.0)
            if jnt1 >= 0:
              w1 = qvel_in[worldid, jnt_dofadr[jnt1]]
            elif tail:
              w1 = wp.dot(wp.spatial_top(cvel_in[worldid, bT_chain]), e1) / l1
            dv0 = vvel[1] - vvel[0]
            dv1 = vvel[2] - vvel[1]
            gamma_dot = (w1 - w0) + wp.dot(kb_d, dv0) / (2.0 * l0) + wp.dot(kb_d, dv1) / (2.0 * l1)
            tau_unscaled = 2.0 * kt * gamma_dot
            tau_d = damping * tau_unscaled
            if jnt1 >= 0:
              dmp_dof1 = jnt_dofadr[jnt1]
              dmp_frc1 = -tau_d
            elif tail:
              tau_bT_d = (-tau_d / l1) * e1
            if jnt0 >= 0:
              dmp_dof0 = jnt_dofadr[jnt0]
              dmp_frc0 = tau_d
            elif h_idx == 0 and root_clamped and b0_chain > 0:
              tau_b0_d = (tau_d / l0) * e0
            f_dmp[0] = (tau_unscaled / (2.0 * l0)) * kb_d
            f_dmp[1] = (-tau_unscaled / (2.0 * l0) + tau_unscaled / (2.0 * l1)) * kb_d
            f_dmp[2] = (-tau_unscaled / (2.0 * l1)) * kb_d

    # vertex forces -J' g and -J' wd
    for i in range(nvert):
      JT = wp.transpose(rod_stencil_jacobian(J0, J1, J2, J3, i))
      if has_spring:
        f_pt = wp.vec3(0.0)
        if i == 0:
          f_pt = f_pt0
        elif i == 1:
          f_pt = f_pt1
        elif i == 2:
          f_pt = f_pt2
        flex_scatter_vert(
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
          worldid,
          verts[i],
          -(JT * g) + f_pt,
          1.0,
          qfrc_spring_out,
        )
      if has_damper:
        flex_scatter_vert(
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
          worldid,
          verts[i],
          -(JT * wd) + f_dmp[i],
          damping,
          qfrc_damper_out,
        )
    if has_spring and twist_dof0 >= 0:
      wp.atomic_add(qfrc_spring_out, worldid, twist_dof0, twist_frc0)
    if has_spring and twist_dof1 >= 0:
      wp.atomic_add(qfrc_spring_out, worldid, twist_dof1, twist_frc1)
    if has_spring and wp.dot(tau_b0, tau_b0) > 0.0:
      bw = body_weldid[flex_vertbodyid[verts[0]]]
      dof_i = body_dofadr[bw] + body_dofnum[bw] - 1
      while dof_i >= 0:
        wp.atomic_add(
          qfrc_spring_out,
          worldid,
          dof_i,
          wp.dot(wp.spatial_top(cdof_in[worldid, dof_i]), tau_b0),
        )
        dof_i = dof_parentid[dof_i]
    if has_spring and wp.dot(tau_bT, tau_bT) > 0.0:
      bw_T = body_weldid[flex_vertbodyid[verts[2]]]
      dof_T = body_dofadr[bw_T] + body_dofnum[bw_T] - 1
      while dof_T >= 0:
        wp.atomic_add(
          qfrc_spring_out,
          worldid,
          dof_T,
          wp.dot(wp.spatial_top(cdof_in[worldid, dof_T]), tau_bT),
        )
        dof_T = dof_parentid[dof_T]
    if has_damper and dmp_dof0 >= 0:
      wp.atomic_add(qfrc_damper_out, worldid, dmp_dof0, dmp_frc0)
    if has_damper and dmp_dof1 >= 0:
      wp.atomic_add(qfrc_damper_out, worldid, dmp_dof1, dmp_frc1)
    if has_damper and b0_chain > 0 and wp.dot(tau_b0_d, tau_b0_d) > 0.0:
      bw_d = body_weldid[b0_chain]
      dof_d = body_dofadr[bw_d] + body_dofnum[bw_d] - 1
      while dof_d >= 0:
        wp.atomic_add(
          qfrc_damper_out,
          worldid,
          dof_d,
          wp.dot(wp.spatial_top(cdof_in[worldid, dof_d]), tau_b0_d),
        )
        dof_d = dof_parentid[dof_d]
    if has_damper and bT_chain > 0 and wp.dot(tau_bT_d, tau_bT_d) > 0.0:
      bw_Td = body_weldid[bT_chain]
      dof_Td = body_dofadr[bw_Td] + body_dofnum[bw_Td] - 1
      while dof_Td >= 0:
        wp.atomic_add(
          qfrc_damper_out,
          worldid,
          dof_Td,
          wp.dot(wp.spatial_top(cdof_in[worldid, dof_Td]), tau_bT_d),
        )
        dof_Td = dof_parentid[dof_Td]

  return kernel


@cache_kernel
def _flex_bending(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_edgeadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_bendingadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_plasticmode: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_edgeflap: wp.array[wp.vec2i],
    flex_stiffness: wp.array[float],
    flex_bending: wp.array[float],
    flex_damping: wp.array[float],
    flex_edgeflexid: wp.array[int],
    # Data in:
    plastic_in: wp.array2d[float],
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    cvel_in: wp.array2d[wp.spatial_vector],
    # In:
    dsbl_spring: bool,
    dsbl_damper: bool,
    # Out:
    flex_spring_body_force_out: wp.array2d[wp.spatial_vector],
    flex_damper_body_force_out: wp.array2d[wp.spatial_vector],
  ):
    worldid, edgeid = wp.tid()

    f = flex_edgeflexid[edgeid]
    eid = edgeid - flex_edgeadr[f]

    bendingadr = flex_bendingadr[f]
    if bendingadr < 0:
      return

    if flex_dim[f] != 2 or flex_interp[f] != 0:
      return

    flap = flex_edgeflap[edgeid]
    if flap[1] == -1:
      return

    has_spring = not dsbl_spring
    damping = flex_damping[f]
    has_damper = not dsbl_damper and damping > 0.0
    if not has_spring and not has_damper:
      return

    edge = flex_edge[edgeid]
    vertadr = flex_vertadr[f]
    v = wp.vec4i(
      vertadr + edge[0],
      vertadr + edge[1],
      vertadr + flap[0],
      vertadr + flap[1],
    )

    vpos = mat43()
    bodyids = wp.vec4i()
    has_dof = wp.vec4i(0, 0, 0, 0)
    for j in range(4):
      vpos[j] = flexvert_xpos_in[worldid, v[j]]
      bodyid_j = flex_vertbodyid[v[j]]
      bodyids[j] = bodyid_j
      if bodyid_j >= 0:
        if body_dofnum[body_weldid[bodyid_j]] > 0:
          has_dof[j] = 1

    bbase = bendingadr + 17 * eid
    c16 = flex_bending[bbase + 16]
    padr_base = flex_plasticadr[f]
    if (flex_plasticmode[f] & 1) != 0:
      stretch_pnum = 3 * flex_elemnum[f] if (flex_plasticmode[f] & 2) != 0 else 0
      c16 += plastic_in[worldid, padr_base + stretch_pnum + eid]
    frc = mat43()
    if has_spring and c16 != 0.0:
      ed0 = vpos[1] - vpos[0]
      ed1 = vpos[2] - vpos[0]
      ed2 = vpos[3] - vpos[0]

      frc[1] = wp.cross(ed1, ed2)
      frc[2] = wp.cross(ed2, ed0)
      frc[3] = wp.cross(ed0, ed1)
      frc[0] = -(frc[1] + frc[2] + frc[3])

    # Gather velocities if damping is enabled
    vel = mat43()
    if has_damper:
      for j in range(4):
        if has_dof[j]:
          bodyid_j = bodyids[j]
          cvel_j = cvel_in[worldid, bodyid_j]
          omega_j = wp.spatial_top(cvel_j)
          vcom_j = wp.spatial_bottom(cvel_j)
          com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]
          r_j = vpos[j] - com_j
          vel[j] = vcom_j + wp.cross(omega_j, r_j)

    for i in range(4):
      if has_dof[i]:
        acc_spring = wp.vec3(0.0)
        acc_damper = wp.vec3(0.0)
        for j in range(4):
          coeff = flex_bending[bbase + 4 * i + j]
          if has_spring:
            acc_spring += coeff * vpos[j]
          if has_damper:
            acc_damper += coeff * vel[j]

        bodyid = bodyids[i]
        node_pos = vpos[i]
        body_xipos = xipos_in[worldid, bodyid]
        offset = body_xipos - node_pos

        if has_spring:
          frc_s = -(acc_spring + c16 * frc[i])
          spatial_frc_s = wp.spatial_vector(frc_s, -wp.cross(offset, frc_s))
          wp.atomic_add(flex_spring_body_force_out, worldid, bodyid, spatial_frc_s)

        if has_damper:
          frc_d = -acc_damper * damping
          spatial_frc_d = wp.spatial_vector(frc_d, -wp.cross(offset, frc_d))
          wp.atomic_add(flex_damper_body_force_out, worldid, bodyid, spatial_frc_d)

  return kernel


@cache_kernel
def _flex_passive_interp(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = FLEX_INTERP_MAX_NODES

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    nflex: int,
    body_rootid: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_node: wp.array[wp.vec3],
    flex_node0: wp.array[wp.vec3],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_edgeequality: wp.array[int],
    flex_centered: wp.array[bool],
    flex_faceadr: wp.array[int],
    flex_cell_map: wp.array[wp.vec4i],
    flex_face: wp.array2d[int],
    # Data in:
    plastic_in: wp.array2d[float],
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cvel_in: wp.array2d[wp.spatial_vector],
    flexnode_xpos_in: wp.array2d[wp.vec3],
    face_quat_in: wp.array2d[wp.quat],
    # In:
    dsbl_spring: bool,
    dsbl_damper: bool,
    # Out:
    flex_spring_body_force_out: wp.array2d[wp.spatial_vector],
    flex_damper_body_force_out: wp.array2d[wp.spatial_vector],
    displ_scratch_out: wp.array3d[wp.vec3],
    vel_corot_scratch_out: wp.array3d[wp.vec3],
  ):
    """Corotational passive forces for interpolated flex (trilinear/quadratic)."""
    worldid, cellid = wp.tid()

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

    npc = (order_abs + 1) * (order_abs + 1) if shell_mode else (order_abs + 1) * (order_abs + 1) * (order_abs + 1)
    ndof_cell = 3 * npc

    cellnum = flex_cellnum[f]
    cy = cellnum[1]
    cz = cellnum[2]
    nstart = flex_nodeadr[f]
    ny_g = cy * order_abs + 1
    nz_g = cz * order_abs + 1

    # Cell stiffness matrix address
    stiffness_adr_base = flex_stiffnessadr[f]
    if stiffness_adr_base < 0:
      return

    cell_idx = ci if shell_mode else ci * cy * cz + cj * cz + ck
    k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell

    # Skip empty cells (zero stiffness) in volume mode; shells have no empty cells
    # and their x-normal faces (faces 0 and 1) have flex_stiffness[k_base] == 0
    if not shell_mode and flex_stiffness[k_base] == 0.0:
      return

    if shell_mode:
      cell_quat = face_quat_in[worldid, flex_faceadr[f] + ci]
    else:
      cell_quat = support.compute_interp_cell_quat(
        flex_node0,
        flexnode_xpos_in,
        order_abs,
        ci,
        cj,
        ck,
        cy,
        cz,
        ny_g,
        nz_g,
        nstart,
        worldid,
      )

    # mju_negQuat: conjugate (R⁻¹) — negate xyz, keep w
    cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])

    # Pre-compute displacements and velocities in corotational frame
    # (matches C: rotate all positions/velocities once, then K*u)
    op1 = order_abs + 1
    node_0 = (
      flex_face[flex_faceadr[f] + ci, 0]
      if shell_mode
      else nstart + (ci * order_abs) * ny_g * nz_g + (cj * order_abs) * nz_g + (ck * order_abs)
    )
    refpos_0 = flex_node0[node_0]
    padr_base = flex_plasticadr[f]
    cx = cellnum[0]
    normal_axis = 0 if ci < 2 * cy * cz else (1 if ci < 2 * (cy * cz + cx * cz) else 2)
    axis0 = (normal_axis + 1) % 3
    axis1 = (normal_axis + 2) % 3
    sp0 = float(0.0)
    sp1 = float(0.0)
    sp2 = float(0.0)
    sp3 = float(0.0)
    sp4 = float(0.0)
    sp5 = float(0.0)
    if not dsbl_spring and padr_base >= 0:
      if shell_mode:
        padr = padr_base + 3 * cell_idx
        sp0 = plastic_in[worldid, padr + 0]
        sp1 = plastic_in[worldid, padr + 1]
        sp2 = plastic_in[worldid, padr + 2]
      else:
        padr = padr_base + 6 * cell_idx
        sp0 = plastic_in[worldid, padr + 0]
        sp1 = plastic_in[worldid, padr + 1]
        sp2 = plastic_in[worldid, padr + 2]
        sp3 = plastic_in[worldid, padr + 3]
        sp4 = plastic_in[worldid, padr + 4]
        sp5 = plastic_in[worldid, padr + 5]
    for idx_j in range(npc):
      if shell_mode:
        node_j = flex_face[flex_faceadr[f] + ci, idx_j]
      else:
        li_j = idx_j // (op1 * op1)
        lj_j = (idx_j // op1) % op1
        lk_j = idx_j % op1
        gidx_j = (ci * order_abs + li_j) * ny_g * nz_g + (cj * order_abs + lj_j) * nz_g + (ck * order_abs + lk_j)
        node_j = nstart + gidx_j

      xpos_j = flexnode_xpos_in[worldid, node_j]

      if not dsbl_spring:
        refpos_j = flex_node0[node_j]
        xrot_j = wp.quat_rotate(cell_quat_inv, xpos_j)
        displ_j = xrot_j - refpos_j
        if padr_base >= 0:
          dx0 = refpos_j - refpos_0
          if shell_mode:
            displ_j[axis0] -= sp0 * dx0[axis0] + 0.5 * sp2 * dx0[axis1]
            displ_j[axis1] -= 0.5 * sp2 * dx0[axis0] + sp1 * dx0[axis1]
          else:
            displ_j[0] -= sp0 * dx0[0] + 0.5 * sp3 * dx0[1] + 0.5 * sp4 * dx0[2]
            displ_j[1] -= 0.5 * sp3 * dx0[0] + sp1 * dx0[1] + 0.5 * sp5 * dx0[2]
            displ_j[2] -= 0.5 * sp4 * dx0[0] + 0.5 * sp5 * dx0[1] + sp2 * dx0[2]
        displ_scratch_out[worldid, cellid, idx_j] = displ_j

      if not dsbl_damper:
        bodyid_j = flex_nodebodyid[node_j]
        cvel_j = cvel_in[worldid, bodyid_j]
        omega_j = wp.spatial_top(cvel_j)
        vcom_j = wp.spatial_bottom(cvel_j)
        com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]
        r_j = xpos_j - com_j
        vel_world_j = vcom_j + wp.cross(omega_j, r_j)
        vel_corot_scratch_out[worldid, cellid, idx_j] = wp.quat_rotate(cell_quat_inv, vel_world_j)

    # Compute K*displacement and K*velocity per output node, then scatter forces
    for idx_i in range(npc):
      if shell_mode:
        node_i = flex_face[flex_faceadr[f] + ci, idx_i]
      else:
        li_i = idx_i // (op1 * op1)
        lj_i = (idx_i // op1) % op1
        lk_i = idx_i % op1
        gidx_i = (ci * order_abs + li_i) * ny_g * nz_g + (cj * order_abs + lj_i) * nz_g + (ck * order_abs + lk_i)
        node_i = nstart + gidx_i
      bodyid_i = flex_nodebodyid[node_i]

      frc_spring = wp.vec3(0.0)
      frc_damper = wp.vec3(0.0)

      for comp_i in range(3):
        row = idx_i * 3 + comp_i
        val_spring = float(0.0)
        val_damper = float(0.0)

        for idx_j in range(npc):
          for comp_j in range(3):
            col = idx_j * 3 + comp_j
            K_ij = flex_stiffness[k_base + row * ndof_cell + col]

            if not dsbl_spring:
              val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]

            if not dsbl_damper:
              val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]

        frc_spring[comp_i] = val_spring
        frc_damper[comp_i] = val_damper

      # Rotate forces back to world frame (R)
      frc_spring_world = wp.quat_rotate(cell_quat, frc_spring)
      frc_damper_world = wp.quat_rotate(cell_quat, frc_damper)

      # Scale damper force by damping coefficient
      frc_damper_world = frc_damper_world * flex_damping[f]

      # Apply forces to body
      node_pos = flexnode_xpos_in[worldid, node_i]
      body_xipos = xipos_in[worldid, bodyid_i]

      offset = body_xipos - node_pos
      if not dsbl_spring:
        spatial_frc_s = wp.spatial_vector(frc_spring_world, -wp.cross(offset, frc_spring_world))
        wp.atomic_add(flex_spring_body_force_out, worldid, bodyid_i, spatial_frc_s)

      if not dsbl_damper:
        spatial_frc_d = wp.spatial_vector(frc_damper_world, -wp.cross(offset, frc_damper_world))
        wp.atomic_add(flex_damper_body_force_out, worldid, bodyid_i, spatial_frc_d)

  return kernel


@cache_kernel
def _flex_passive_bend_interp(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.func
  def _apply_face_forces(
    # Model:
    flex_nodebodyid: wp.array[int],
    flex_face: wp.array2d[int],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    flexnode_xpos_in: wp.array2d[wp.vec3],
    # In:
    face_id: int,
    local_coords: wp.vec2,
    wt1: wp.vec3,
    wt2: wp.vec3,
    stiffness_scale: float,
    order_abs: int,
    worldid: int,
    # Out:
    body_force_out: wp.array2d[wp.spatial_vector],
  ):
    idx = int(0)
    for l0 in range(3):
      if l0 <= order_abs:
        for l1 in range(3):
          if l1 <= order_abs:
            g0 = support.dphi2D(local_coords[0], l0, local_coords[1], l1, order_abs, 0)
            g1 = support.dphi2D(local_coords[0], l0, local_coords[1], l1, order_abs, 1)

            gidx = flex_face[face_id, idx]

            frc = (wt2 * g0 - wt1 * g1) * stiffness_scale

            bid = flex_nodebodyid[gidx]
            node_pos = flexnode_xpos_in[worldid, gidx]

            body_xipos = xipos_in[worldid, bid]
            offset = body_xipos - node_pos
            spatial_frc = wp.spatial_vector(frc, -wp.cross(offset, frc))
            wp.atomic_add(body_force_out, worldid, bid, spatial_frc)
            idx += 1

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    nflex: int,
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_nodenum: wp.array[int],
    flex_bendingadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_node: wp.array[wp.vec3],
    flex_bending: wp.array[float],
    flex_centered: wp.array[bool],
    flex_faceadr: wp.array[int],
    flex_bend_interp_map: wp.array[wp.vec2i],
    flex_face: wp.array2d[int],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    flexnode_xpos_in: wp.array2d[wp.vec3],
    face_xpos_in: wp.array3d[wp.vec3],
    face_quat_in: wp.array2d[wp.quat],
    # Out:
    flex_spring_body_force_out: wp.array2d[wp.spatial_vector],
  ):
    worldid, bend_edge_id = wp.tid()

    mapping = flex_bend_interp_map[bend_edge_id]
    f = mapping[0]
    e = mapping[1]

    order = flex_interp[f]
    order_abs = -order
    bendingadr = flex_bendingadr[f]

    cellnum = flex_cellnum[f]
    cx = cellnum[0]
    cy = cellnum[1]
    cz = cellnum[2]
    nstart = flex_nodeadr[f]

    edata_base = bendingadr + 1 + e * 10
    fe_A = int(flex_bending[edata_base + 0])
    fe_B = int(flex_bending[edata_base + 1])
    local_A = wp.vec2(flex_bending[edata_base + 2], flex_bending[edata_base + 3])
    local_B = wp.vec2(flex_bending[edata_base + 4], flex_bending[edata_base + 5])
    stiffness = flex_bending[edata_base + 6]
    dn0 = wp.vec3(
      flex_bending[edata_base + 7],
      flex_bending[edata_base + 8],
      flex_bending[edata_base + 9],
    )

    if stiffness <= 0.0:
      return

    # Look up cached face data instead of recomputing
    face_id_A = flex_faceadr[f] + fe_A
    face_id_B = flex_faceadr[f] + fe_B

    quat_A = face_quat_in[worldid, face_id_A]
    quat_B = face_quat_in[worldid, face_id_B]

    # Compute deformed normals at edge midpoint
    t1_A = wp.vec3(0.0)
    t2_A = wp.vec3(0.0)
    t1_B = wp.vec3(0.0)
    t2_B = wp.vec3(0.0)

    idx = int(0)
    for l0 in range(3):
      if l0 > order_abs:
        continue
      for l1 in range(3):
        if l1 > order_abs:
          continue
        pos_A = face_xpos_in[worldid, face_id_A, idx]
        pos_B = face_xpos_in[worldid, face_id_B, idx]

        grad0_A = support.flex_dphi(local_A[0], l0, order_abs) * support.flex_phi(local_A[1], l1, order_abs)
        grad1_A = support.flex_phi(local_A[0], l0, order_abs) * support.flex_dphi(local_A[1], l1, order_abs)

        grad0_B = support.flex_dphi(local_B[0], l0, order_abs) * support.flex_phi(local_B[1], l1, order_abs)
        grad1_B = support.flex_phi(local_B[0], l0, order_abs) * support.flex_dphi(local_B[1], l1, order_abs)

        t1_A += pos_A * grad0_A
        t2_A += pos_A * grad1_A

        t1_B += pos_B * grad0_B
        t2_B += pos_B * grad1_B
        idx += 1

    n_A = wp.cross(t1_A, t2_A)
    n_B = wp.cross(t1_B, t2_B)

    len_A = wp.length(n_A)
    len_B = wp.length(n_B)

    if len_A < MJ_MINVAL or len_B < MJ_MINVAL:
      return

    inv_A = 1.0 / len_A
    inv_B = 1.0 / len_B
    n_A_norm = n_A * inv_A
    n_B_norm = n_B * inv_B

    # Average face quaternions
    if wp.dot(quat_A, quat_B) < 0.0:
      quat_B = -quat_B
    quat_avg = quat_A + quat_B
    quat_avg = wp.normalize(quat_avg)

    # rotate dn0 using average quat
    dn0_rot = wp.quat_rotate(quat_avg, dn0)

    # residual: r = (n_A - n_B) - dn0_rot
    r = n_A_norm - n_B_norm - dn0_rot

    # Compute projection and cross products
    dot_A = wp.dot(n_A_norm, r)
    w_A = (r - n_A_norm * dot_A) * inv_A

    dot_B = wp.dot(n_B_norm, r)
    w_B = (r - n_B_norm * dot_B) * inv_B

    wAt2 = wp.cross(w_A, t2_A)
    wAt1 = wp.cross(w_A, t1_A)
    wBt2 = wp.cross(w_B, t2_B)
    wBt1 = wp.cross(w_B, t1_B)

    # Apply forces for Face A
    _apply_face_forces(
      flex_nodebodyid,
      flex_face,
      xipos_in,
      flexnode_xpos_in,
      face_id_A,
      local_A,
      wAt1,
      wAt2,
      stiffness,
      order_abs,
      worldid,
      flex_spring_body_force_out,
    )

    # Apply forces for Face B (negative stiffness)
    _apply_face_forces(
      flex_nodebodyid,
      flex_face,
      xipos_in,
      flexnode_xpos_in,
      face_id_B,
      local_B,
      wBt1,
      wBt2,
      -stiffness,
      order_abs,
      worldid,
      flex_spring_body_force_out,
    )

  return kernel


@wp.func
def _flex_vert_mass(
  # Model:
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  # In:
  worldid: int,
  gv: int,
) -> float:
  b = body_weldid[flex_vertbodyid[gv]]
  if body_dofnum[b] != 3:
    return float(0.0)
  da = body_dofadr[b]
  diag_idx = M_rowadr[da] + M_rownnz[da] - 1
  return M_in[worldid, diag_idx]


@wp.func
def _flex_participant_mass(
  # Model:
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  # In:
  worldid: int,
  flex_id: int,
  elem_id: int,
  vert_id: int,
  mmin_in: float,
) -> float:
  mmin = mmin_in
  if flex_id >= 0:
    if vert_id >= 0:
      gv = flex_vertadr[flex_id] + vert_id
      mv = _flex_vert_mass(
        body_weldid,
        body_dofnum,
        body_dofadr,
        flex_vertbodyid,
        M_rownnz,
        M_rowadr,
        M_in,
        worldid,
        gv,
      )
      if mv > 0.0 and (mmin == 0.0 or mv < mmin):
        mmin = mv
    elif elem_id >= 0:
      dim = flex_dim[flex_id]
      nvrt = dim + 1
      base_e = flex_elemdataadr[flex_id] + nvrt * elem_id
      for j in range(nvrt):
        gv = flex_vertadr[flex_id] + flex_elem[base_e + j]
        mv = _flex_vert_mass(
          body_weldid,
          body_dofnum,
          body_dofadr,
          flex_vertbodyid,
          M_rownnz,
          M_rowadr,
          M_in,
          worldid,
          gv,
        )
        if mv > 0.0 and (mmin == 0.0 or mv < mmin):
          mmin = mv
  return mmin


@wp.func
def _flex_contact_stiffness(
  # Model:
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  # In:
  worldid: int,
  flex0: int,
  flex1: int,
  elem0: int,
  elem1: int,
  vert0: int,
  vert1: int,
) -> float:
  mmin = _flex_participant_mass(
    body_weldid,
    body_dofnum,
    body_dofadr,
    flex_dim,
    flex_vertadr,
    flex_elemdataadr,
    flex_vertbodyid,
    flex_elem,
    M_rownnz,
    M_rowadr,
    M_in,
    worldid,
    flex0,
    elem0,
    vert0,
    0.0,
  )
  mmin = _flex_participant_mass(
    body_weldid,
    body_dofnum,
    body_dofadr,
    flex_dim,
    flex_vertadr,
    flex_elemdataadr,
    flex_vertbodyid,
    flex_elem,
    M_rownnz,
    M_rowadr,
    M_in,
    worldid,
    flex1,
    elem1,
    vert1,
    mmin,
  )
  # 5.0e7 is mjFLEXCONTACT_OMEGA2: squared natural frequency of the passive flex contact law,
  # scaling the minimum nonzero participant vertex mass to form the pair stiffness.
  return 5.0e7 * mmin


@wp.func
def _add_con_dof(
  max_dof: int,
  # In:
  cid: int,
  dof: int,
  val: float,
  nnz: int,
  # Out:
  efm_con_dof_out: wp.array2d[int],
  efm_con_val_out: wp.array2d[float],
) -> int:
  for idx in range(nnz):
    if efm_con_dof_out[cid, idx] == dof:
      efm_con_val_out[cid, idx] += val
      return nnz
  if nnz >= max_dof:
    wp.printf("Warning: efm_con_max_dof overflow (%d >= %d)\n", nnz, max_dof)
    return nnz
  efm_con_dof_out[cid, nnz] = dof
  efm_con_val_out[cid, nnz] = val
  return nnz + 1


@wp.kernel
def _eff_contact_build(
  # Model:
  opt_timestep: wp.array[float],
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_parentid: wp.array[int],
  geom_type: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  efm_con_max_dof: int,
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  plastic_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  M_in: wp.array2d[float],
  contact_dist_in: wp.array[float],
  contact_pos_in: wp.array[wp.vec3],
  contact_frame_in: wp.array[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  efm_con_dof_out: wp.array2d[int],
  efm_con_val_out: wp.array2d[float],
  efm_con_scale_out: wp.array[float],
  efm_con_force_out: wp.array[float],
  efm_con_nnz_out: wp.array[int],
):
  """Builds rank-1 passive flex contact metric terms, normal Jacobians, and forces."""
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return

  if not bool(contact_type_in[cid] & int(ContactType.PASSIVE)):
    efm_con_nnz_out[cid] = 0
    efm_con_force_out[cid] = float(0.0)
    efm_con_scale_out[cid] = float(0.0)
    return

  worldid = contact_worldid_in[cid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  geom_pair = contact_geom_in[cid]
  flex_pair = contact_flex_in[cid]
  elem_pair = contact_elem_in[cid]
  vert_pair = contact_vert_in[cid]
  flex0 = flex_pair[0]
  flex1 = flex_pair[1]
  elem0 = elem_pair[0]
  elem1 = elem_pair[1]
  vert0 = vert_pair[0]
  vert1 = vert_pair[1]

  k = _flex_contact_stiffness(
    body_weldid,
    body_dofnum,
    body_dofadr,
    flex_dim,
    flex_vertadr,
    flex_elemdataadr,
    flex_vertbodyid,
    flex_elem,
    M_rownnz,
    M_rowadr,
    M_in,
    worldid,
    flex0,
    flex1,
    elem0,
    elem1,
    vert0,
    vert1,
  )

  if k <= float(0.0):
    efm_con_nnz_out[cid] = 0
    efm_con_force_out[cid] = float(0.0)
    efm_con_scale_out[cid] = float(0.0)
    return

  gap = contact_dist_in[cid]
  pos = contact_pos_in[cid]
  normal = contact_frame_in[cid][0]

  nnz = int(0)

  for side in range(2):
    fs = flex_pair[side]
    if fs < 0:
      continue
    vs = vert_pair[side]
    es = elem_pair[side]
    sgn = 1.0 if side == 1 else -1.0
    if vs >= 0:
      gv = flex_vertadr[fs] + vs
      b = body_weldid[flex_vertbodyid[gv]]
      if body_dofnum[b] == 3:
        da = body_dofadr[b]
        n_loc = wp.transpose(xmat_in[worldid, b]) @ normal
        for ax in range(3):
          nnz = _add_con_dof(
            efm_con_max_dof,
            cid,
            da + ax,
            sgn * n_loc[ax],
            nnz,
            efm_con_dof_out,
            efm_con_val_out,
          )
    elif es >= 0:
      dims = flex_dim[fs]
      nvrts = dims + 1
      bases = flex_elemdataadr[fs] + nvrts * es
      ws = support.flex_elem_weights(
        flex_dim,
        flex_vertadr,
        flex_elemdataadr,
        flex_elem,
        flexvert_xpos_in,
        fs,
        es,
        pos,
        worldid,
      )
      for j in range(nvrts):
        gv = flex_vertadr[fs] + flex_elem[bases + j]
        b = body_weldid[flex_vertbodyid[gv]]
        if body_dofnum[b] == 3:
          da = body_dofadr[b]
          n_loc = wp.transpose(xmat_in[worldid, b]) @ normal
          for ax in range(3):
            nnz = _add_con_dof(
              efm_con_max_dof,
              cid,
              da + ax,
              sgn * ws[j] * n_loc[ax],
              nnz,
              efm_con_dof_out,
              efm_con_val_out,
            )

    (
      dof_twist,
      dp_twist,
      _dr_twist,
      active_rod,
      b0_rod,
      b1_rod,
      x0_rod,
      x1_rod,
      w0_rod,
      w1_rod,
      inv_le_rod,
      t_rod,
      dvec_rod,
      t_cross_r_rod,
      g_psi_rod,
    ) = support.rod_contact_side_params(
      jnt_dofadr,
      geom_type,
      flex_vertadr,
      flex_elemadr,
      flex_elemnum,
      flex_elemdataadr,
      flex_plasticadr,
      flex_vertbodyid,
      flex_elem,
      flex_rod_has_twist,
      flex_rod_edgeadr,
      flex_rod_edgenum,
      flex_rodedge_jnt,
      flex_elem_rodedge,
      flex_vert_rodelem,
      plastic_in,
      flexvert_xpos_in,
      worldid,
      side,
      geom_pair,
      flex_pair,
      elem_pair,
      vert_pair,
      gap,
      pos,
      normal,
    )
    if dof_twist >= 0:
      nnz = _add_con_dof(
        efm_con_max_dof,
        cid,
        dof_twist,
        wp.dot(normal, dp_twist),
        nnz,
        efm_con_dof_out,
        efm_con_val_out,
      )
    if active_rod:
      for ep in range(2):
        bk = b1_rod if ep == 1 else b0_rod
        xk = x1_rod if ep == 1 else x0_rod
        wk = w1_rod if ep == 1 else w0_rod
        esgn = 1.0 if ep == 1 else -1.0
        xk_off = xk - subtree_com_in[worldid, body_rootid[bk]]
        con_minus_xk = pos - xk
        bw = body_weldid[bk]
        da_k = body_dofadr[bw] + body_dofnum[bw] - 1
        while da_k >= 0:
          dp_k, _dr_k = support.rod_contact_dof_delta(
            cdof_in[worldid, da_k],
            xk_off,
            con_minus_xk,
            t_rod,
            dvec_rod,
            t_cross_r_rod,
            g_psi_rod,
            sgn * wk,
            sgn * esgn,
            inv_le_rod,
          )
          nnz = _add_con_dof(
            efm_con_max_dof,
            cid,
            da_k,
            wp.dot(normal, dp_k),
            nnz,
            efm_con_dof_out,
            efm_con_val_out,
          )
          da_k = dof_parentid[da_k]

  # Compact zero entries
  real_nnz = int(0)
  for idx in range(nnz):
    if wp.abs(efm_con_val_out[cid, idx]) > 1e-12:
      if real_nnz != idx:
        efm_con_dof_out[cid, real_nnz] = efm_con_dof_out[cid, idx]
        efm_con_val_out[cid, real_nnz] = efm_con_val_out[cid, idx]
      real_nnz += 1

  efm_con_nnz_out[cid] = real_nnz
  if real_nnz > 0:
    efm_con_force_out[cid] = -k * wp.min(gap, float(0.0))
    efm_con_scale_out[cid] = timestep * timestep * k
  else:
    efm_con_force_out[cid] = float(0.0)
    efm_con_scale_out[cid] = float(0.0)


def build_efm_contact(
  m: Model, d: Data, rebuild: bool = True
) -> tuple[wp.array2d[int], wp.array2d[float], wp.array[float], wp.array[float], wp.array[int]]:
  """Allocates and builds rank-1 passive flex contact metric terms."""
  efm_con_dof = wp.zeros((d.naconmax, m.efm_con_max_dof), dtype=int)
  efm_con_val = wp.zeros((d.naconmax, m.efm_con_max_dof), dtype=float)
  efm_con_scale = wp.zeros(d.naconmax, dtype=float)
  efm_con_force = wp.zeros(d.naconmax, dtype=float)
  efm_con_nnz = wp.zeros(d.naconmax, dtype=int)
  if (
    rebuild
    and m.has_flex_passive
    and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER))
  ):
    wp.launch(
      _eff_contact_build,
      dim=d.naconmax,
      inputs=[
        m.opt.timestep,
        m.body_rootid,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.jnt_dofadr,
        m.dof_parentid,
        m.geom_type,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_elemdataadr,
        m.flex_plasticadr,
        m.flex_vertbodyid,
        m.flex_elem,
        m.M_rownnz,
        m.M_rowadr,
        m.efm_con_max_dof,
        m.flex_rod_has_twist,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
        m.flex_elem_rodedge,
        m.flex_vert_rodelem,
        d.plastic,
        d.xmat,
        d.subtree_com,
        d.cdof,
        d.flexvert_xpos,
        d.M,
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.geom,
        d.contact.flex,
        d.contact.elem,
        d.contact.vert,
        d.contact.worldid,
        d.contact.type,
        d.nacon,
      ],
      outputs=[
        efm_con_dof,
        efm_con_val,
        efm_con_scale,
        efm_con_force,
        efm_con_nnz,
      ],
    )
  return efm_con_dof, efm_con_val, efm_con_scale, efm_con_force, efm_con_nnz


@cache_kernel
def _eff_contact_force(max_dof: int, deterministic: bool = False):
  """Applies passive contact normal repulsion forces into qfrc_spring."""
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_dof

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Data in:
    contact_worldid_in: wp.array[int],
    # In:
    efm_con_dof_in: wp.array2d[int],
    efm_con_val_in: wp.array2d[float],
    efm_con_force_in: wp.array[float],
    efm_con_nnz_in: wp.array[int],
    # Data out:
    qfrc_spring_out: wp.array2d[float],
  ):
    cid = wp.tid()
    nnz = efm_con_nnz_in[cid]
    if nnz == 0:
      return
    f = efm_con_force_in[cid]
    if f == float(0.0):
      return
    worldid = contact_worldid_in[cid]
    for a in range(nnz):
      dof = efm_con_dof_in[cid, a]
      val = efm_con_val_in[cid, a]
      wp.atomic_add(qfrc_spring_out, worldid, dof, f * val)

  return kernel


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  dsbl_spring = m.opt.disableflags & DisableBit.SPRING
  dsbl_damper = m.opt.disableflags & DisableBit.DAMPER

  if dsbl_spring and dsbl_damper:
    d.qfrc_spring.zero_()
    d.qfrc_damper.zero_()
    d.qfrc_gravcomp.zero_()
    d.qfrc_fluid.zero_()
    d.qfrc_passive.zero_()
    return

  wp.launch(
    _spring_damper_dof_passive,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.disableflags,
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_actuatorid,
      m.jnt_plasticadr,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      m.dof_damping,
      m.dof_dampingpoly,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_damping,
      m.actuator_dampingpoly,
      m.actuator_gear,
      d.qpos,
      d.qvel,
      d.plastic,
    ],
    outputs=[d.qfrc_spring, d.qfrc_damper],
  )

  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  if m.ntendon:
    wp.launch(
      _spring_damper_tendon_passive(det_atomics),
      dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
      inputs=[
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
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        d.qfrc_spring,
        d.qfrc_damper,
      ],
    )

  if not (dsbl_spring and dsbl_damper):
    wp.launch(
      _spring_damper_flexedge_passive(m.nv if det_atomics else 0, det_atomics),
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.flexedge_length0,
        m.flex_edgestiffness,
        m.flex_edgedamping,
        m.flexedge_rigid,
        m.flexedge_J_rownnz,
        m.flexedge_J_rowadr,
        m.flexedge_J_colind,
        m.flex_edgeflexid,
        d.flexedge_J,
        d.flexedge_length,
        d.flexedge_velocity,
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        d.qfrc_spring,
        d.qfrc_damper,
      ],
    )
    if m.nflexrodhinge > 0:
      eval_rod_chains(m, d)
      wp.launch(
        _flex_rod_bending(m.nv if det_atomics else 0, det_atomics),
        dim=(d.nworld, m.nflexrodhinge),
        inputs=[
          m.body_rootid,
          m.body_weldid,
          m.body_dofnum,
          m.body_dofadr,
          m.body_simple,
          m.jnt_dofadr,
          m.dof_parentid,
          m.flex_vertadr,
          m.flex_vertbodyid,
          m.flex_bending,
          m.flex_damping,
          m.flex_rodhinge_map,
          m.flex_rod_root_clamped,
          m.flex_rod_tail_clamped,
          m.flex_rod_hingeadr,
          m.flex_rod_edgeadr,
          m.flex_rod_edgenum,
          m.flex_rodedge_vert,
          m.flex_rodedge_jnt,
          d.qvel,
          d.subtree_com,
          d.cdof,
          d.flexvert_xpos,
          d.cvel,
          d.flex_rod_chain_ok,
          d.flex_rodhinge_g,
          d.flex_rodhinge_Wd,
          d.flex_rodedge_grad,
          d.flex_rodedge_frc,
          d.flex_rod_end_torque,
          dsbl_spring,
          dsbl_damper,
        ],
        outputs=[d.qfrc_spring, d.qfrc_damper],
      )

  flex_spring_body_force = None
  flex_damper_body_force = None
  if m.nflex > 0:
    flex_spring_body_force = wp.zeros((d.nworld, m.nbody), dtype=wp.spatial_vector, device=d.qfrc_spring.device)
    flex_damper_body_force = wp.zeros((d.nworld, m.nbody), dtype=wp.spatial_vector, device=d.qfrc_spring.device)

  if not dsbl_spring or not dsbl_damper:
    wp.launch(
      _flex_elasticity(det_atomics),
      dim=(d.nworld, m.nflexelem),
      inputs=[
        m.opt.timestep,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_elemadr,
        m.flex_elemdataadr,
        m.flex_stiffnessadr,
        m.flex_elemedgeadr,
        m.flex_plasticadr,
        m.flex_plasticmode,
        m.flex_vertbodyid,
        m.flex_elem,
        m.flex_elemedge,
        m.flexedge_length0,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_elemflexid,
        d.plastic,
        d.xipos,
        d.flexvert_xpos,
        d.flexedge_length,
        d.flexedge_velocity,
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        flex_spring_body_force,
        flex_damper_body_force,
      ],
    )

    wp.launch(
      _flex_bending(det_atomics),
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.body_rootid,
        m.body_weldid,
        m.body_dofnum,
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_elemnum,
        m.flex_stiffnessadr,
        m.flex_bendingadr,
        m.flex_plasticadr,
        m.flex_plasticmode,
        m.flex_vertbodyid,
        m.flex_edge,
        m.flex_edgeflap,
        m.flex_stiffness,
        m.flex_bending,
        m.flex_damping,
        m.flex_edgeflexid,
        d.plastic,
        d.xipos,
        d.subtree_com,
        d.flexvert_xpos,
        d.cvel,
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        flex_spring_body_force,
        flex_damper_body_force,
      ],
    )
    wp.launch(
      _flex_passive_bend_interp(det_atomics),
      dim=(d.nworld, m.nflexbend_interp),
      inputs=[
        m.nflex,
        m.flex_interp,
        m.flex_cellnum,
        m.flex_nodeadr,
        m.flex_nodenum,
        m.flex_bendingadr,
        m.flex_nodebodyid,
        m.flex_node,
        m.flex_bending,
        m.flex_centered,
        m.flex_faceadr,
        m.flex_bend_interp_map,
        m.flex_face,
        d.xipos,
        d.flexnode_xpos,
        d.face_xpos,
        d.face_quat,
      ],
      outputs=[flex_spring_body_force],
    )

  gravity_enabled = not (m.opt.disableflags & DisableBit.GRAVITY)
  d.qfrc_gravcomp.zero_()
  if gravity_enabled:
    wp.launch(
      _gravity_force(det_atomics),
      dim=(d.nworld, m.nbody - 1, m.nv),
      inputs=[
        m.opt.gravity,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_gravcomp,
        m.dof_bodyid,
        m.body_isdofancestor,
        d.xipos,
        d.subtree_com,
        d.cdof,
      ],
      outputs=[d.qfrc_gravcomp],
    )

  # Launch passive interp kernel for interpolated flex (trilinear/quadratic)
  if m.nflex and m.nflexintcell > 0:
    displ_scratch = wp.empty((d.nworld, m.nflexintcell, FLEX_INTERP_MAX_NODES), dtype=wp.vec3)
    vel_corot_scratch = wp.empty((d.nworld, m.nflexintcell, FLEX_INTERP_MAX_NODES), dtype=wp.vec3)
    wp.launch(
      _flex_passive_interp(det_atomics),
      dim=(d.nworld, m.nflexintcell),
      inputs=[
        m.nflex,
        m.body_rootid,
        m.flex_interp,
        m.flex_cellnum,
        m.flex_nodeadr,
        m.flex_stiffnessadr,
        m.flex_plasticadr,
        m.flex_nodebodyid,
        m.flex_node,
        m.flex_node0,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_edgeequality,
        m.flex_centered,
        m.flex_faceadr,
        m.flex_cell_map,
        m.flex_face,
        d.plastic,
        d.xipos,
        d.subtree_com,
        d.cvel,
        d.flexnode_xpos,
        d.face_quat,
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        flex_spring_body_force,
        flex_damper_body_force,
        displ_scratch,
        vel_corot_scratch,
      ],
    )

  if m.nflex > 0:
    if not dsbl_spring:
      support.apply_ft(m, d, flex_spring_body_force, d.qfrc_spring, True)
    if not dsbl_damper:
      support.apply_ft(m, d, flex_damper_body_force, d.qfrc_damper, True)
    if m.has_flex_snh and not dsbl_damper:
      flex_hessian(m, d)
      flex_stretch_mul(
        m,
        d,
        d.qfrc_damper,
        d.qvel,
        s1_in=0.0,
        s2_in=-1.0,
        use_timestep=False,
        snh_only=True,
      )
    if m.opt.integrator == IntegratorType.DISCRETE and m.has_flex_passive:
      efm_con_dof, efm_con_val, _, efm_con_force, efm_con_nnz = build_efm_contact(m, d, rebuild=True)
      wp.launch(
        _eff_contact_force(m.efm_con_max_dof if det_atomics else 0, det_atomics),
        dim=d.naconmax,
        inputs=[
          d.contact.worldid,
          efm_con_dof,
          efm_con_val,
          efm_con_force,
          efm_con_nnz,
        ],
        outputs=[d.qfrc_spring],
      )

  if m.has_fluid:
    _fluid(m, d)

  d.qfrc_adhesion.zero_()
  if m.flg_adhesion and (not (m.opt.disableflags & DisableBit.CONTACT)) and m.nv > 0:
    wp.launch(
      _qfrc_adhesion(m.nv if det_atomics else 0, det_atomics),
      dim=d.naconmax,
      inputs=[
        m.body_parentid,
        m.body_rootid,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.dof_bodyid,
        m.geom_bodyid,
        m.body_isdofancestor,
        d.subtree_com,
        d.cdof,
        d.contact.pos,
        d.contact.frame,
        d.contact.geom,
        d.contact.worldid,
        d.contact.adhesion,
        d.nacon,
      ],
      outputs=[
        d.qfrc_adhesion,
      ],
    )

  wp.launch(
    _qfrc_passive_kernel(m.has_fluid, m.flg_adhesion, gravity_enabled),
    dim=(d.nworld, m.nv),
    inputs=[
      m.jnt_actgravcomp,
      m.dof_jntid,
      d.qfrc_spring,
      d.qfrc_damper,
      d.qfrc_gravcomp,
      d.qfrc_fluid,
      d.qfrc_adhesion,
    ],
    outputs=[
      d.qfrc_passive,
    ],
  )

  if m.callback.passive:
    m.callback.passive(m, d)
