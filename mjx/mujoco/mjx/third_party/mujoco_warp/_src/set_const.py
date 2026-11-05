# Copyright 2026 The Newton Developers
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


import warnings

import mujoco
import numpy as np
import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import math as mjmath
from mujoco.mjx.third_party.mujoco_warp._src import smooth
from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src.types import TrnType
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"default_grid_stride": False})


# TODO: Update kernel_analyzer to support Model fields as outputs so it can be enabled here.
# kernel_analyzer: off
@wp.kernel
def _init_subtreemass(
  body_mass_in: wp.array2d[float],
  body_subtreemass_out: wp.array2d[float],
):
  worldid, bodyid = wp.tid()
  body_mass_id = worldid % body_mass_in.shape[0]
  body_subtreemass_id = worldid % body_subtreemass_out.shape[0]
  body_subtreemass_out[body_subtreemass_id, bodyid] = body_mass_in[body_mass_id, bodyid]


@cache_kernel
def _accumulate_subtreemass(deterministic: bool = False):
  module_options = {"enable_backward": False, "deterministic": wp.DeterministicMode.RUN_TO_RUN} if deterministic else {}

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    body_parentid: wp.array[int],
    body_subtreemass_in: wp.array2d[float],
    body_tree_: wp.array[int],
    body_subtreemass_out: wp.array2d[float],
  ):
    worldid, nodeid = wp.tid()
    body_subtreemass_id = worldid % body_subtreemass_in.shape[0]
    bodyid = body_tree_[nodeid]
    parentid = body_parentid[bodyid]
    if bodyid != 0:
      wp.atomic_add(body_subtreemass_out, body_subtreemass_id, parentid, body_subtreemass_in[body_subtreemass_id, bodyid])

  return kernel


@wp.kernel
def _copy_qpos0_to_qpos(
  qpos0: wp.array2d[float],
  qpos_out: wp.array2d[float],
):
  worldid, i = wp.tid()
  qpos0_id = worldid % qpos0.shape[0]
  qpos_out[worldid, i] = qpos0[qpos0_id, i]


@wp.kernel
def _copy_body_to_mocap(
  body_mocapid: wp.array[int],
  body_pos: wp.array2d[wp.vec3],
  body_quat: wp.array2d[wp.quat],
  mocap_pos_out: wp.array2d[wp.vec3],
  mocap_quat_out: wp.array2d[wp.quat],
):
  worldid, bodyid = wp.tid()
  mocapid = body_mocapid[bodyid]
  if mocapid >= 0:
    mocap_pos_out[worldid, mocapid] = body_pos[worldid % body_pos.shape[0], bodyid]
    mocap_quat_out[worldid, mocapid] = body_quat[worldid % body_quat.shape[0], bodyid]


@wp.kernel
def _copy_tendon_length0(
  ten_length_in: wp.array2d[float],
  tendon_length0_out: wp.array2d[float],
):
  worldid, tenid = wp.tid()
  tendon_length0_id = worldid % tendon_length0_out.shape[0]
  tendon_length0_out[tendon_length0_id, tenid] = ten_length_in[worldid, tenid]


@wp.kernel
def _compute_eq_data0(
  # Model:
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  xmat_in: wp.array2d[wp.mat33],
  # Out:
  eq_data_out: wp.array2d[types.vec12],
):
  """Compute the data of equality constraints between bodies which follows from qpos0.

  The anchor of connects in the second body, and the relative pose of welds which are marked in
  data[11] for it to be computed. Kinematics must have been evaluated at qpos0.
  """
  worldid, eqid = wp.tid()

  if eq_objtype[eqid] != int(types.ObjType.BODY.value):
    return

  eqtype = eq_type[eqid]
  eq_data_id = worldid % eq_data_out.shape[0]
  data = eq_data_out[eq_data_id, eqid]
  obj1id = eq_obj1id[eqid]
  obj2id = eq_obj2id[eqid]

  if eqtype == int(types.EqType.CONNECT.value):
    # data[0:3] = anchor in body1 local frame; map to global frame
    anchor1 = wp.vec3(data[0], data[1], data[2])
    pos = xpos_in[worldid, obj1id] + xmat_in[worldid, obj1id] @ anchor1

    # data[3:6] = anchor position in body2 local frame
    anchor2 = wp.transpose(xmat_in[worldid, obj2id]) @ (pos - xpos_in[worldid, obj2id])
    data[3] = anchor2[0]
    data[4] = anchor2[1]
    data[5] = anchor2[2]
    eq_data_out[eq_data_id, eqid] = data
  elif eqtype == int(types.EqType.WELD.value) and data[11] != 0.0:
    # data[0:3] = anchor in body2 local frame; map to global frame
    anchor2 = wp.vec3(data[0], data[1], data[2])
    pos = xpos_in[worldid, obj2id] + xmat_in[worldid, obj2id] @ anchor2

    # data[3:6] = anchor position in body1 local frame
    anchor1 = wp.transpose(xmat_in[worldid, obj1id]) @ (pos - xpos_in[worldid, obj1id])
    data[3] = anchor1[0]
    data[4] = anchor1[1]
    data[5] = anchor1[2]

    # data[6:10] = neg(xquat1) * xquat2 = "xquat2 - xquat1" in body1 local frame
    relquat = mjmath.mul_quat(mjmath.quat_inv(xquat_in[worldid, obj1id]), xquat_in[worldid, obj2id])
    data[6] = relquat[0]
    data[7] = relquat[1]
    data[8] = relquat[2]
    data[9] = relquat[3]
    eq_data_out[eq_data_id, eqid] = data


@wp.kernel
def _compute_geom_bound(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  hfield_size: wp.array[wp.vec4],
  # Out:
  geom_size_out: wp.array2d[wp.vec3],
  geom_aabb_out: wp.array3d[wp.vec3],
  geom_rbound_out: wp.array2d[float],
):
  """Compute the bounding box and sphere of geoms whose bounds follow from their size.

  The size of a height field geom follows its height field. Mesh and SDF geoms keep the bounds of
  their mesh.
  """
  worldid, geomid = wp.tid()
  gtype = geom_type[geomid]
  if gtype == int(types.GeomType.MESH.value) or gtype == int(types.GeomType.SDF.value):
    return

  geom_size_id = worldid % geom_size_out.shape[0]
  size = geom_size_out[geom_size_id, geomid]
  lo = wp.vec3(0.0)
  hi = wp.vec3(0.0)
  rbound = float(0.0)

  if gtype == int(types.GeomType.HFIELD.value):
    hsize = hfield_size[geom_dataid[worldid % geom_dataid.shape[0], geomid]]
    geom_size_out[geom_size_id, geomid] = wp.vec3(hsize[0], hsize[1], 0.25 * hsize[2] + 0.5 * hsize[3])
    lo = wp.vec3(-hsize[0], -hsize[1], -hsize[3])
    hi = wp.vec3(hsize[0], hsize[1], hsize[2])
    rbound = wp.sqrt(hsize[0] * hsize[0] + hsize[1] * hsize[1] + wp.max(hsize[2] * hsize[2], hsize[3] * hsize[3]))
  elif gtype == int(types.GeomType.PLANE.value):
    lo = wp.vec3(-types.MJ_MAXVAL)
    hi = wp.vec3(types.MJ_MAXVAL, types.MJ_MAXVAL, 0.0)
  else:
    if gtype == int(types.GeomType.SPHERE.value):
      hi = wp.vec3(size[0])
      rbound = size[0]
    elif gtype == int(types.GeomType.CAPSULE.value):
      hi = wp.vec3(size[0], size[0], size[0] + size[1])
      rbound = size[0] + size[1]
    elif gtype == int(types.GeomType.CYLINDER.value):
      hi = wp.vec3(size[0], size[0], size[1])
      rbound = wp.sqrt(size[0] * size[0] + size[1] * size[1])
    elif gtype == int(types.GeomType.ELLIPSOID.value):
      hi = size
      rbound = wp.max(wp.max(size[0], size[1]), size[2])
    elif gtype == int(types.GeomType.BOX.value):
      hi = size
      rbound = wp.length(size)
    else:
      hi = size
    lo = -hi

  geom_aabb_id = worldid % geom_aabb_out.shape[0]
  geom_aabb_out[geom_aabb_id, geomid, 0] = 0.5 * (hi + lo)
  geom_aabb_out[geom_aabb_id, geomid, 1] = 0.5 * (hi - lo)
  geom_rbound_out[worldid % geom_rbound_out.shape[0], geomid] = rbound


@wp.kernel
def _set_unit_vector(
  dofid_target: int,
  unit_vec_out: wp.array2d[float],
):
  worldid = wp.tid()
  nv = unit_vec_out.shape[1]
  for i in range(nv):
    if i == dofid_target:
      unit_vec_out[worldid, i] = 1.0
    else:
      unit_vec_out[worldid, i] = 0.0


@wp.kernel
def _extract_dof_A_diag(
  dofid: int,
  result_vec_in: wp.array2d[float],
  dof_A_diag_out: wp.array2d[float],
):
  worldid = wp.tid()
  dof_A_diag_id = worldid % dof_A_diag_out.shape[0]
  dof_A_diag_out[dof_A_diag_id, dofid] = result_vec_in[worldid, dofid]


@wp.kernel
def _finalize_dof_invweight0(
  dof_jntid: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_A_diag_in: wp.array2d[float],
  dof_invweight0_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  dof_invweight0_id = worldid % dof_invweight0_out.shape[0]
  dof_A_diag_id = worldid % dof_A_diag_in.shape[0]

  jntid = dof_jntid[dofid]
  jtype = jnt_type[jntid]
  dofadr = jnt_dofadr[jntid]

  if jtype == int(types.JointType.FREE.value):
    # FREE joint: 6 DOFs, average first 3 (trans) and last 3 (rot) separately
    if dofid < dofadr + 3:
      avg = wp.static(1.0 / 3.0) * (
        dof_A_diag_in[dof_A_diag_id, dofadr + 0]
        + dof_A_diag_in[dof_A_diag_id, dofadr + 1]
        + dof_A_diag_in[dof_A_diag_id, dofadr + 2]
      )
    else:
      avg = wp.static(1.0 / 3.0) * (
        dof_A_diag_in[dof_A_diag_id, dofadr + 3]
        + dof_A_diag_in[dof_A_diag_id, dofadr + 4]
        + dof_A_diag_in[dof_A_diag_id, dofadr + 5]
      )
    dof_invweight0_out[dof_invweight0_id, dofid] = avg
  elif jtype == int(types.JointType.BALL.value):
    # BALL joint: 3 DOFs, average all
    avg = wp.static(1.0 / 3.0) * (
      dof_A_diag_in[dof_A_diag_id, dofadr + 0]
      + dof_A_diag_in[dof_A_diag_id, dofadr + 1]
      + dof_A_diag_in[dof_A_diag_id, dofadr + 2]
    )
    dof_invweight0_out[dof_invweight0_id, dofid] = avg
  else:
    # HINGE/SLIDE: 1 DOF, no averaging
    dof_invweight0_out[dof_invweight0_id, dofid] = dof_A_diag_in[dof_A_diag_id, dofid]


@wp.kernel
def _compute_body_jac_row(
  nv: int,
  bodyid_target: int,
  row_idx: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_dofadr: wp.array[int],
  body_dofnum: wp.array[int],
  dof_parentid: wp.array[int],
  subtree_com_in: wp.array2d[wp.vec3],
  xipos_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  body_jac_row_out: wp.array2d[float],
):
  worldid = wp.tid()

  for i in range(nv):
    body_jac_row_out[worldid, i] = 0.0

  bodyid = bodyid_target
  while bodyid > 0 and body_dofnum[bodyid] == 0:
    bodyid = body_parentid[bodyid]

  if bodyid == 0:
    return

  # Compute offset from point (xipos) to subtree_com of root body
  point = xipos_in[worldid, bodyid_target]
  offset = point - subtree_com_in[worldid, body_rootid[bodyid_target]]

  # Get last dof that affects this body
  dofid = body_dofadr[bodyid] + body_dofnum[bodyid] - 1

  # Backward pass over dof ancestor chain
  while dofid >= 0:
    cdof = cdof_in[worldid, dofid]
    cdof_ang = wp.spatial_top(cdof)
    cdof_lin = wp.spatial_bottom(cdof)

    if row_idx < 3:
      tmp = wp.cross(cdof_ang, offset)
      if row_idx == 0:
        body_jac_row_out[worldid, dofid] = cdof_lin[0] + tmp[0]
      elif row_idx == 1:
        body_jac_row_out[worldid, dofid] = cdof_lin[1] + tmp[1]
      else:
        body_jac_row_out[worldid, dofid] = cdof_lin[2] + tmp[2]
    else:
      if row_idx == 3:
        body_jac_row_out[worldid, dofid] = cdof_ang[0]
      elif row_idx == 4:
        body_jac_row_out[worldid, dofid] = cdof_ang[1]
      else:
        body_jac_row_out[worldid, dofid] = cdof_ang[2]

    dofid = dof_parentid[dofid]


@wp.kernel
def _compute_body_A_diag_entry(
  nv: int,
  bodyid_target: int,
  row_idx: int,
  body_jac_row_in: wp.array2d[float],
  result_vec_in: wp.array2d[float],
  body_A_diag_out: wp.array3d[float],
):
  worldid = wp.tid()
  body_A_diag_id = worldid % body_A_diag_out.shape[0]
  # A[row,row] = J[row] · inv(M) · J[row]' = J[row] · result_vec
  dot_prod = float(0.0)
  for i in range(nv):
    dot_prod += body_jac_row_in[worldid, i] * result_vec_in[worldid, i]
  body_A_diag_out[body_A_diag_id, bodyid_target, row_idx] = dot_prod


@wp.kernel
def _finalize_body_invweight0(
  body_weldid: wp.array[int],
  body_A_diag_in: wp.array3d[float],
  body_invweight0_out: wp.array2d[wp.vec2],
):
  worldid, bodyid = wp.tid()
  body_invweight0_id = worldid % body_invweight0_out.shape[0]
  body_A_diag_id = worldid % body_A_diag_in.shape[0]

  # World body and static bodies have zero invweight
  if bodyid == 0 or body_weldid[bodyid] == 0:
    body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(0.0, 0.0)
    return

  # Average diagonal: trans = (A[0,0]+A[1,1]+A[2,2])/3, rot = (A[3,3]+A[4,4]+A[5,5])/3
  inv_trans = wp.static(1.0 / 3.0) * (
    body_A_diag_in[body_A_diag_id, bodyid, 0]
    + body_A_diag_in[body_A_diag_id, bodyid, 1]
    + body_A_diag_in[body_A_diag_id, bodyid, 2]
  )
  inv_rot = wp.static(1.0 / 3.0) * (
    body_A_diag_in[body_A_diag_id, bodyid, 3]
    + body_A_diag_in[body_A_diag_id, bodyid, 4]
    + body_A_diag_in[body_A_diag_id, bodyid, 5]
  )

  # Prevent degenerate constraints: if one component is near zero, use the other as fallback
  if inv_trans < mujoco.mjMINVAL and inv_rot > mujoco.mjMINVAL:
    inv_trans = inv_rot  # use rotation as fallback for translation
  elif inv_rot < mujoco.mjMINVAL and inv_trans > mujoco.mjMINVAL:
    inv_rot = inv_trans  # use translation as fallback for rotation

  body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(inv_trans, inv_rot)


@wp.kernel
def _copy_tendon_jacobian(
  tenid_target: int,
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  ten_J_in: wp.array2d[float],
  ten_J_vec_out: wp.array2d[float],
):
  worldid = wp.tid()
  rownnz = ten_J_rownnz[tenid_target]
  rowadr = ten_J_rowadr[tenid_target]
  for i in range(rownnz):
    colind = ten_J_colind[rowadr + i]
    ten_J_vec_out[worldid, colind] = ten_J_in[worldid, rowadr + i]


@wp.kernel
def _compute_tendon_dot_product(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # In:
  tenid_target: int,
  ten_J_in: wp.array2d[float],
  result_vec_in: wp.array2d[float],
  # Out:
  tendon_invweight0_out: wp.array2d[float],
):
  worldid = wp.tid()
  tendon_invweight0_id = worldid % tendon_invweight0_out.shape[0]
  dot_prod = float(0.0)

  rownnz = ten_J_rownnz[tenid_target]
  rowadr = ten_J_rowadr[tenid_target]
  for i in range(rownnz):
    sparseid = rowadr + i
    colind = ten_J_colind[sparseid]
    dot_prod += ten_J_in[worldid, sparseid] * result_vec_in[worldid, colind]

  tendon_invweight0_out[tendon_invweight0_id, tenid_target] = dot_prod


@wp.kernel
def _compute_cam_pos0(
  cam_bodyid: wp.array[int],
  cam_targetbodyid: wp.array[int],
  cam_pos: wp.array2d[wp.vec3],
  cam_quat: wp.array2d[wp.quat],
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  subtree_com_in: wp.array2d[wp.vec3],
  cam_pos0_out: wp.array2d[wp.vec3],
  cam_poscom0_out: wp.array2d[wp.vec3],
  cam_mat0_out: wp.array2d[wp.mat33],
):
  worldid, camid = wp.tid()
  bodyid = cam_bodyid[camid]
  targetid = cam_targetbodyid[camid]
  xpos = xpos_in[worldid, bodyid]
  xquat = xquat_in[worldid, bodyid]

  # Fixed-mode camera pose at qpos0
  rel_pos = mjmath.rot_vec_quat(cam_pos[worldid % cam_pos.shape[0], camid], xquat)
  cam_xpos = xpos + rel_pos
  cam_xmat = mjmath.quat_to_mat(mjmath.mul_quat(xquat, cam_quat[worldid % cam_quat.shape[0], camid]))

  cam_pos0_out[worldid % cam_pos0_out.shape[0], camid] = rel_pos
  comid = wp.where(targetid >= 0, targetid, bodyid)
  cam_poscom0_out[worldid % cam_poscom0_out.shape[0], camid] = cam_xpos - subtree_com_in[worldid, comid]
  cam_mat0_out[worldid % cam_mat0_out.shape[0], camid] = cam_xmat


@wp.kernel
def _compute_light_pos0(
  light_bodyid: wp.array[int],
  light_targetbodyid: wp.array[int],
  light_pos: wp.array2d[wp.vec3],
  light_dir: wp.array2d[wp.vec3],
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  subtree_com_in: wp.array2d[wp.vec3],
  light_pos0_out: wp.array2d[wp.vec3],
  light_poscom0_out: wp.array2d[wp.vec3],
  light_dir0_out: wp.array2d[wp.vec3],
):
  worldid, lightid = wp.tid()
  bodyid = light_bodyid[lightid]
  targetid = light_targetbodyid[lightid]
  xpos = xpos_in[worldid, bodyid]
  xquat = xquat_in[worldid, bodyid]

  # Fixed-mode light pose at qpos0
  rel_pos = mjmath.rot_vec_quat(light_pos[worldid % light_pos.shape[0], lightid], xquat)
  light_xpos = xpos + rel_pos
  light_xdir = wp.normalize(mjmath.rot_vec_quat(light_dir[worldid % light_dir.shape[0], lightid], xquat))

  light_pos0_out[worldid % light_pos0_out.shape[0], lightid] = rel_pos
  comid = wp.where(targetid >= 0, targetid, bodyid)
  light_poscom0_out[worldid % light_poscom0_out.shape[0], lightid] = light_xpos - subtree_com_in[worldid, comid]
  light_dir0_out[worldid % light_dir0_out.shape[0], lightid] = light_xdir


@wp.kernel
def _copy_actuator_moment(
  actid_target: int,
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  act_moment_vec_out: wp.array2d[float],
):
  worldid = wp.tid()
  nv = act_moment_vec_out.shape[1]
  for i in range(nv):
    act_moment_vec_out[worldid, i] = 0.0
  rownnz = moment_rownnz_in[worldid, actid_target]
  rowadr = moment_rowadr_in[worldid, actid_target]
  for i in range(rownnz):
    sparseid = rowadr + i
    col = moment_colind_in[worldid, sparseid]
    act_moment_vec_out[worldid, col] = actuator_moment_in[worldid, sparseid]


@wp.kernel
def _compute_actuator_acc0(
  actid_target: int,
  nv: int,
  result_vec_in: wp.array2d[float],
  actuator_acc0_out: wp.array2d[float],
):
  worldid = wp.tid()
  norm_sq = float(0.0)
  for i in range(nv):
    norm_sq += result_vec_in[worldid, i] * result_vec_in[worldid, i]
  actuator_acc0_out[worldid, actid_target] = wp.sqrt(norm_sq)


@wp.kernel
def _set_length_range(
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_gear: wp.array2d[wp.spatial_vector],
  jnt_limited: wp.array[int],
  jnt_range: wp.array2d[wp.vec2],
  tendon_limited: wp.array[int],
  tendon_range: wp.array2d[wp.vec2],
  ntendon: int,
  actuator_lengthrange_out: wp.array2d[wp.vec2],
):
  worldid, actid = wp.tid()
  trntype = actuator_trntype[actid]
  id0 = actuator_trnid[actid][0]
  gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]

  lr = wp.vec2(0.0, 0.0)

  if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:
    if jnt_limited[id0]:
      rng = jnt_range[worldid % jnt_range.shape[0], id0]
      if gear0 > 0.0:
        lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)
      else:
        lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)
  elif trntype == TrnType.TENDON:
    if ntendon > 0 and tendon_limited[id0]:
      rng = tendon_range[worldid % tendon_range.shape[0], id0]
      if gear0 > 0.0:
        lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)
      else:
        lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)

  actuator_lengthrange_out[worldid, actid] = lr


# kernel_analyzer: on


def _set_bound(m: types.Model):
  """Compute geom bounding boxes and spheres."""
  nworld_geom = np.max([m.geom_dataid.shape[0], m.geom_size.shape[0], m.geom_aabb.shape[0], m.geom_rbound.shape[0]])
  wp.launch(
    _compute_geom_bound,
    dim=(nworld_geom, m.ngeom),
    inputs=[m.geom_type, m.geom_dataid, m.hfield_size],
    outputs=[m.geom_size, m.geom_aabb, m.geom_rbound],
  )


def _set_fixed(m: types.Model):
  """Compute fixed quantities (do not depend on qpos0)."""
  nworld_subtreemass = m.body_subtreemass.shape[0]
  wp.launch(_init_subtreemass, dim=(nworld_subtreemass, m.nbody), inputs=[m.body_mass], outputs=[m.body_subtreemass])
  det_atomics = bool(m.opt.deterministic & types.DeterminismType.ATOMICS)
  subtreemass_in = wp.empty_like(m.body_subtreemass) if det_atomics else m.body_subtreemass
  for i in reversed(range(len(m.body_tree))):
    body_tree = m.body_tree[i]
    if det_atomics:
      wp.copy(subtreemass_in, m.body_subtreemass)
    wp.launch(
      _accumulate_subtreemass(det_atomics),
      dim=(nworld_subtreemass, body_tree.size),
      inputs=[m.body_parentid, subtreemass_in, body_tree],
      outputs=[m.body_subtreemass],
    )


def _kinematics0(m: types.Model, d: types.Data):
  """Run kinematics and tendon lengths in qpos0."""
  wp.launch(_copy_qpos0_to_qpos, dim=(d.nworld, m.nq), inputs=[m.qpos0], outputs=[d.qpos])
  if m.nmocap:
    wp.launch(
      _copy_body_to_mocap,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_mocapid, m.body_pos, m.body_quat],
      outputs=[d.mocap_pos, d.mocap_quat],
    )
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.tendon(m, d)


def _set_kinematics0(m: types.Model, d: types.Data):
  """Compute kinematic quantities in qpos0 (requires _kinematics0)."""
  wp.launch(_copy_tendon_length0, dim=(m.tendon_length0.shape[0], m.ntendon), inputs=[d.ten_length], outputs=[m.tendon_length0])

  wp.launch(
    _compute_eq_data0,
    dim=(m.eq_data.shape[0], m.neq),
    inputs=[m.eq_type, m.eq_obj1id, m.eq_obj2id, m.eq_objtype, d.xpos, d.xquat, d.xmat],
    outputs=[m.eq_data],
  )

  nworld_cam = np.max([m.cam_pos0.shape[0], m.cam_poscom0.shape[0], m.cam_mat0.shape[0]])
  wp.launch(
    _compute_cam_pos0,
    dim=(nworld_cam, m.ncam),
    inputs=[m.cam_bodyid, m.cam_targetbodyid, m.cam_pos, m.cam_quat, d.xpos, d.xquat, d.subtree_com],
    outputs=[m.cam_pos0, m.cam_poscom0, m.cam_mat0],
  )

  nworld_light = np.max([m.light_pos0.shape[0], m.light_poscom0.shape[0], m.light_dir0.shape[0]])
  wp.launch(
    _compute_light_pos0,
    dim=(nworld_light, m.nlight),
    inputs=[m.light_bodyid, m.light_targetbodyid, m.light_pos, m.light_dir, d.xpos, d.xquat, d.subtree_com],
    outputs=[m.light_pos0, m.light_poscom0, m.light_dir0],
  )


def _set_inertia0(m: types.Model, d: types.Data):
  """Compute inertial quantities in qpos0 (requires _kinematics0)."""
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  smooth.factor_m(m, d)
  smooth.transmission(m, d)

  # dof_invweight0: computed per joint with averaging for multi-DOF joints
  # FREE: 6 DOFs, trans gets mean(A[0:3]), rot gets mean(A[3:6])
  # BALL: 3 DOFs, all get mean(A[0:3])
  # HINGE/SLIDE: 1 DOF, gets A[0,0]
  if m.nv > 0:
    unit_vec = wp.zeros((d.nworld, m.nv), dtype=float)
    result_vec = wp.zeros((d.nworld, m.nv), dtype=float)
    dof_A_diag = wp.zeros((d.nworld, m.nv), dtype=float)

    # TODO(team): more efficient approach instead of looping over nv?
    for dofid in range(m.nv):
      wp.launch(_set_unit_vector, dim=d.nworld, inputs=[dofid], outputs=[unit_vec])
      smooth.solve_m(m, d, result_vec, unit_vec)
      wp.launch(_extract_dof_A_diag, dim=d.nworld, inputs=[dofid, result_vec], outputs=[dof_A_diag])

    wp.launch(
      _finalize_dof_invweight0,
      dim=(m.dof_invweight0.shape[0], m.nv),
      inputs=[m.dof_jntid, m.jnt_type, m.jnt_dofadr, dof_A_diag],
      outputs=[m.dof_invweight0],
    )

  # body_invweight0: computed as mean diagonal of J * inv(M) * J'
  # where J is the 6xnv body Jacobian (3 rows translation, 3 rows rotation)
  if m.nv > 0:
    body_jac_row = wp.zeros((d.nworld, m.nv), dtype=float)
    body_result_vec = wp.zeros((d.nworld, m.nv), dtype=float)
    body_A_diag = wp.zeros((d.nworld, m.nbody, 6), dtype=float)

    # TODO(team): more efficient approach instead of nested iterations?
    for bodyid in range(1, m.nbody):
      for row_idx in range(6):
        wp.launch(
          _compute_body_jac_row,
          dim=d.nworld,
          inputs=[
            m.nv,
            bodyid,
            row_idx,
            m.body_parentid,
            m.body_rootid,
            m.body_dofadr,
            m.body_dofnum,
            m.dof_parentid,
            d.subtree_com,
            d.xipos,
            d.cdof,
          ],
          outputs=[body_jac_row],
        )
        smooth.solve_m(m, d, body_result_vec, body_jac_row)
        wp.launch(
          _compute_body_A_diag_entry,
          dim=d.nworld,
          inputs=[m.nv, bodyid, row_idx, body_jac_row, body_result_vec],
          outputs=[body_A_diag],
        )

    wp.launch(
      _finalize_body_invweight0,
      dim=(m.body_invweight0.shape[0], m.nbody),
      inputs=[m.body_weldid, body_A_diag],
      outputs=[m.body_invweight0],
    )
  else:
    m.body_invweight0.zero_()

  # tendon_invweight0[t] = J_t * inv(M) * J_t'
  if m.ntendon > 0 and m.nv > 0:
    ten_J_vec = wp.empty((d.nworld, m.nv), dtype=float)
    ten_result_vec = wp.empty((d.nworld, m.nv), dtype=float)

    for tenid in range(m.ntendon):
      ten_J_vec.zero_()
      wp.launch(
        _copy_tendon_jacobian,
        dim=d.nworld,
        inputs=[tenid, m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.ten_J],
        outputs=[ten_J_vec],
      )
      smooth.solve_m(m, d, ten_result_vec, ten_J_vec)
      wp.launch(
        _compute_tendon_dot_product,
        dim=m.tendon_invweight0.shape[0],
        inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, tenid, d.ten_J, ten_result_vec],
        outputs=[m.tendon_invweight0],
      )
  else:
    m.tendon_invweight0.zero_()

  # actuator_acc0[i] = ||inv(M) * actuator_moment[i]|| - acceleration from unit actuator force
  if m.nactuator > 0 and m.nv > 0:
    act_moment_vec = wp.zeros((d.nworld, m.nv), dtype=float)
    act_result_vec = wp.zeros((d.nworld, m.nv), dtype=float)

    for actid in range(m.nactuator):
      wp.launch(
        _copy_actuator_moment,
        dim=d.nworld,
        inputs=[actid, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment],
        outputs=[act_moment_vec],
      )
      smooth.solve_m(m, d, act_result_vec, act_moment_vec)
      wp.launch(
        _compute_actuator_acc0, dim=m.actuator_acc0.shape[0], inputs=[actid, m.nv, act_result_vec], outputs=[m.actuator_acc0]
      )
  else:
    m.actuator_acc0.zero_()


def _restore(m: types.Model, d: types.Data, groups: int):
  """Restore state fields in d to correspond to d.qpos."""
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.tendon(m, d)
  if groups & types.SetConstGroup.INERT0:
    smooth.crb(m, d)
    smooth.tendon_armature(m, d)
    smooth.factor_m(m, d)
    smooth.transmission(m, d)


def set_const(
  m: types.Model,
  d: types.Data,
  groups: int = types.SetConstGroup.ALL,
  restore: bool = True,
):
  """Recomputes qpos0-dependent and fixed constant model fields.

  This function propagates changes from some model fields to derived fields,
  allowing modifications that would otherwise be unsafe. It should be called
  after modifying model parameters at runtime.

  Model fields that can be modified safely with set_const:

  ==================================  ==============================================
  Field                               Notes
  ==================================  ==============================================
  qpos0
  body_mass, body_inertia,            Mass and inertia are usually scaled together
  body_ipos, body_iquat               since inertia is sum(m * r^2).
  body_pos, body_quat                 Unsafe for static bodies (invalidates BVH).
  dof_armature, tendon_armature
  actuator_armature, actuator_gear    Actuator armature is added to joint and tendon
                                      armature. Not if changing from/to zero, since
                                      jnt_actuatorid and tendon_actuatorid are set
                                      by put_model.
  eq_data                             Connect anchors, marked weld poses.
  geom_size, hfield_size              Bounds of primitive and height field geoms.
  cam_pos, cam_quat                   Camera pose in body frame.
  light_pos, light_dir                Light position and direction in body frame.
  tendon_stiffness, tendon_damping    Only if changing from/to zero.
  ==================================  ==============================================

  For selective updates, pass SetConstGroup flags to groups:

  ====================  =================================================
  Modified Field        Groups
  ====================  =================================================
  body_mass             SetConstGroup.ALL
  geom_size             SetConstGroup.FIXED (or ALL if tendon wraps geom)
  cam_pos, cam_quat     SetConstGroup.KIN0
  light_pos, light_dir  SetConstGroup.KIN0
  eq_data               SetConstGroup.KIN0
  body_inertia          SetConstGroup.INERT0
  dof_armature          SetConstGroup.INERT0
  qpos0                 SetConstGroup.KIN0 | SetConstGroup.INERT0
  ====================  =================================================

  Computes:
    - Fixed quantities (SetConstGroup.FIXED):
      - body_subtreemass: mass of body and all descendants
      - geom_aabb, geom_rbound, geom_size (hfield): geom bounds
    - Kinematic qpos0 quantities (SetConstGroup.KIN0):
      - tendon_length0: tendon resting lengths
      - eq_data: connect anchors and marked weld poses (eq_data[11])
      - cam_pos0, cam_poscom0, cam_mat0: camera references
      - light_pos0, light_poscom0, light_dir0: light references
    - Inertial qpos0 quantities (SetConstGroup.INERT0):
      - dof_invweight0: inverse inertia for DOFs
      - body_invweight0: inverse spatial inertia for bodies
      - tendon_invweight0: inverse weight for tendons
      - actuator_acc0: acceleration from unit actuator force

  Skips:
    - actuator_length0: not in mjwarp
    - dof_length, flexedge_length0, flexedge_invweight0: unbatched in mjwarp
    - tendon_lengthspring (from qpos_spring): SPRING group unsupported in mjwarp

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    groups: Bitmask of SetConstGroup flags selecting which groups to compute.
    restore: Whether to restore state fields to correspond to d.qpos.
  """
  if groups & types.SetConstGroup.FIXED:
    _set_bound(m)
    _set_fixed(m)

  if groups & (types.SetConstGroup.KIN0 | types.SetConstGroup.INERT0):
    qpos_saved = wp.clone(d.qpos)
    mocap_pos_saved = wp.clone(d.mocap_pos) if m.nmocap else None
    mocap_quat_saved = wp.clone(d.mocap_quat) if m.nmocap else None

    # the constants do not depend on the contacts of d, which transmission averages for adhesion
    # actuators: hide them while the constants are computed, as mj_setConst clears them
    nacon_saved = wp.clone(d.nacon) if m.nacttrnbody else None
    if m.nacttrnbody:
      d.nacon.zero_()

    _kinematics0(m, d)
    if groups & types.SetConstGroup.KIN0:
      _set_kinematics0(m, d)
    if groups & types.SetConstGroup.INERT0:
      _set_inertia0(m, d)

    wp.copy(d.qpos, qpos_saved)
    if m.nmocap:
      wp.copy(d.mocap_pos, mocap_pos_saved)
      wp.copy(d.mocap_quat, mocap_quat_saved)
    if m.nacttrnbody:
      wp.copy(d.nacon, nacon_saved)

    if restore:
      _restore(m, d, groups)


# TODO(team): remove in future version
def set_const_fixed(m: types.Model, d: types.Data):
  """Recomputes qpos0-independent constant model fields (deprecated)."""
  warnings.warn(
    "set_const_fixed is deprecated, use set_const(m, d, groups=SetConstGroup.FIXED) instead.",
    DeprecationWarning,
    stacklevel=2,
  )
  set_const(m, d, groups=types.SetConstGroup.FIXED, restore=False)


# TODO(team): remove in future version
def set_const_0(m: types.Model, d: types.Data, restore: bool = True):
  """Recomputes qpos0-dependent constant model fields (deprecated)."""
  warnings.warn(
    "set_const_0 is deprecated, use set_const(m, d, groups=SetConstGroup.KIN0 | SetConstGroup.INERT0) instead.",
    DeprecationWarning,
    stacklevel=2,
  )
  set_const(
    m,
    d,
    groups=types.SetConstGroup.KIN0 | types.SetConstGroup.INERT0,
    restore=restore,
  )


def set_length_range(m: types.Model, d: types.Data, index: int = -1):
  """Compute feasible actuator length ranges from joint/tendon limits.

  For joint and tendon transmissions with limits, copies the range directly
  from jnt_range or tendon_range scaled by gear. Actuators without limits
  keep (0, 0). This covers the common robotics use case; simulation-based
  computation for general transmissions is not yet implemented.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object (unused, kept for API compatibility with MuJoCo C).
    index: Actuator index to compute for, or -1 for all actuators.
  """
  if m.nu == 0:
    return

  wp.launch(
    _set_length_range,
    dim=(d.nworld, m.nu),
    inputs=[
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_gear,
      m.jnt_limited,
      m.jnt_range,
      m.tendon_limited,
      m.tendon_range,
      m.ntendon,
    ],
    outputs=[m.actuator_lengthrange],
  )
