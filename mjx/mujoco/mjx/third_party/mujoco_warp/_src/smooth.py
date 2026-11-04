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

from mujoco.mjx.third_party.mujoco_warp._src import math
from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src import util_misc
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import Q_LD_BLOCK_COMPACT
from mujoco.mjx.third_party.mujoco_warp._src.types import Q_LD_BLOCK_SPARSE
from mujoco.mjx.third_party.mujoco_warp._src.types import ROD_HINGE_SIZE
from mujoco.mjx.third_party.mujoco_warp._src.types import CamLightType
from mujoco.mjx.third_party.mujoco_warp._src.types import ConeType
from mujoco.mjx.third_party.mujoco_warp._src.types import ConstraintType
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DeterminismType
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import EqType
from mujoco.mjx.third_party.mujoco_warp._src.types import JointType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import ObjType
from mujoco.mjx.third_party.mujoco_warp._src.types import TendonType
from mujoco.mjx.third_party.mujoco_warp._src.types import TileSet
from mujoco.mjx.third_party.mujoco_warp._src.types import TrnType
from mujoco.mjx.third_party.mujoco_warp._src.types import WrapType
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.types import vec10
from mujoco.mjx.third_party.mujoco_warp._src.types import vec12
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _kinematics_branch(
  # Model:
  qpos0: wp.array2d[float],
  body_parentid: wp.array[int],
  body_mocapid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_pos: wp.array2d[wp.vec3],
  body_quat: wp.array2d[wp.quat],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_pos: wp.array2d[wp.vec3],
  jnt_axis: wp.array2d[wp.vec3],
  body_branches: wp.array[int],
  body_branch_start: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  mocap_pos_in: wp.array2d[wp.vec3],
  mocap_quat_in: wp.array2d[wp.quat],
  # Data out:
  xpos_out: wp.array2d[wp.vec3],
  xquat_out: wp.array2d[wp.quat],
  xanchor_out: wp.array2d[wp.vec3],
  xaxis_out: wp.array2d[wp.vec3],
):
  worldid, branchid = wp.tid()

  start = body_branch_start[branchid]
  end = body_branch_start[branchid + 1]

  qpos = qpos_in[worldid]

  for i in range(start, end):
    bodyid = body_branches[i]
    pid = body_parentid[bodyid]
    jntadr = body_jntadr[bodyid]
    jntnum = body_jntnum[bodyid]

    if jntnum == 1:
      jnt_type_ = jnt_type[jntadr]
      if jnt_type_ == JointType.FREE:
        qadr = jnt_qposadr[jntadr]
        xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
        xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
        xquat = wp.normalize(xquat)

        xpos_out[worldid, bodyid] = xpos
        xquat_out[worldid, bodyid] = xquat
        xanchor_out[worldid, jntadr] = xpos
        xaxis_out[worldid, jntadr] = jnt_axis[worldid % jnt_axis.shape[0], jntadr]
        continue

    # regular or no joints
    # apply fixed translation and rotation relative to parent
    jnt_pos_id = worldid % jnt_pos.shape[0]
    pid = body_parentid[bodyid]

    # mocap bodies have world body as parent
    mocapid = body_mocapid[bodyid]
    if mocapid >= 0:
      xpos = mocap_pos_in[worldid, mocapid]
      xquat = mocap_quat_in[worldid, mocapid]
    else:
      xpos = body_pos[worldid % body_pos.shape[0], bodyid]
      xquat = body_quat[worldid % body_quat.shape[0], bodyid]

    if pid >= 0:
      xpos = math.rot_vec_quat(xpos, xquat_out[worldid, pid]) + xpos_out[worldid, pid]
      xquat = math.mul_quat(xquat_out[worldid, pid], xquat)

    for _ in range(jntnum):
      qadr = jnt_qposadr[jntadr]
      jnt_type_ = jnt_type[jntadr]
      jnt_axis_ = jnt_axis[worldid % jnt_axis.shape[0], jntadr]
      xanchor = math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat) + xpos
      xaxis = math.rot_vec_quat(jnt_axis_, xquat)

      if jnt_type_ == JointType.BALL:
        qloc = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        qloc = wp.normalize(qloc)
        xquat = math.mul_quat(xquat, qloc)
        # correct for off-center rotation
        xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)
      elif jnt_type_ == JointType.SLIDE:
        xpos += xaxis * (qpos[qadr] - qpos0[worldid % qpos0.shape[0], qadr])
      elif jnt_type_ == JointType.HINGE:
        qpos0_ = qpos0[worldid % qpos0.shape[0], qadr]
        qloc_ = math.axis_angle_to_quat(jnt_axis_, qpos[qadr] - qpos0_)
        xquat = math.mul_quat(xquat, qloc_)
        # correct for off-center rotation
        xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)

      xanchor_out[worldid, jntadr] = xanchor
      xaxis_out[worldid, jntadr] = xaxis
      jntadr += 1

    xquat = wp.normalize(xquat)
    xpos_out[worldid, bodyid] = xpos
    xquat_out[worldid, bodyid] = xquat


@wp.kernel
def _compute_body_inertial_frames(
  # Model:
  body_ipos: wp.array2d[wp.vec3],
  body_iquat: wp.array2d[wp.quat],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  # Data out:
  xipos_out: wp.array2d[wp.vec3],
  ximat_out: wp.array2d[wp.mat33],
):
  worldid, bodyid = wp.tid()
  xpos = xpos_in[worldid, bodyid]
  xquat = xquat_in[worldid, bodyid]
  xipos_out[worldid, bodyid] = xpos + math.rot_vec_quat(body_ipos[worldid % body_ipos.shape[0], bodyid], xquat)
  ximat_out[worldid, bodyid] = math.quat_to_mat(math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], bodyid]))


@wp.kernel
def _compute_body_matrices(
  # Data in:
  xquat_in: wp.array2d[wp.quat],
  # Data out:
  xmat_out: wp.array2d[wp.mat33],
):
  worldid, bodyid = wp.tid()
  xmat_out[worldid, bodyid] = math.quat_to_mat(xquat_in[worldid, bodyid])


@wp.kernel
def _geom_local_to_global(
  # Model:
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_mocapid: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_pos: wp.array2d[wp.vec3],
  geom_quat: wp.array2d[wp.quat],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  # Data out:
  geom_xpos_out: wp.array2d[wp.vec3],
  geom_xmat_out: wp.array2d[wp.mat33],
):
  worldid, geomid = wp.tid()
  bodyid = geom_bodyid[geomid]

  if body_weldid[bodyid] == 0 and body_mocapid[body_rootid[bodyid]] == -1:
    # geoms attached to the world are static (unless they are descended from mcocap bodies)
    # for such static geoms, geom_xpos and geom_xquat are computed only once during make_data
    return

  xpos = xpos_in[worldid, bodyid]
  xquat = xquat_in[worldid, bodyid]
  geom_xpos_out[worldid, geomid] = xpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], xquat)
  geom_xmat_out[worldid, geomid] = math.quat_to_mat(math.mul_quat(xquat, geom_quat[worldid % geom_quat.shape[0], geomid]))


@wp.kernel
def _site_local_to_global(
  # Model:
  site_bodyid: wp.array[int],
  site_pos: wp.array2d[wp.vec3],
  site_quat: wp.array2d[wp.quat],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  # Data out:
  site_xpos_out: wp.array2d[wp.vec3],
  site_xmat_out: wp.array2d[wp.mat33],
):
  worldid, siteid = wp.tid()
  bodyid = site_bodyid[siteid]
  xpos = xpos_in[worldid, bodyid]
  xquat = xquat_in[worldid, bodyid]
  site_xpos_out[worldid, siteid] = xpos + math.rot_vec_quat(site_pos[worldid % site_pos.shape[0], siteid], xquat)
  site_xmat_out[worldid, siteid] = math.quat_to_mat(math.mul_quat(xquat, site_quat[worldid % site_quat.shape[0], siteid]))


@wp.kernel
def _flex_vertices(
  # Model:
  nflex: int,
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_vertcell: wp.array[int],
  flex_vert: wp.array[wp.vec3],
  flex_vert0: wp.array[wp.vec3],
  flex_centered: wp.array[bool],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  flexnode_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  flexvert_xpos_out: wp.array2d[wp.vec3],
):
  worldid, vertid = wp.tid()

  for f in range(nflex):
    locid = vertid - flex_vertadr[f]
    if locid >= 0 and locid < flex_vertnum[f]:
      break

  order = flex_interp[f]
  if order != 0:
    # Interpolated flex: vertex position = weighted sum of node positions
    coord = flex_vert0[vertid]
    cn = flex_cellnum[f]
    cx = cn[0]
    cy = cn[1]
    cz = cn[2]

    ux = coord[0] * float(cx)
    uy = coord[1] * float(cy)
    uz = coord[2] * float(cz)

    # Cell lookup: find containing cell
    ci = wp.clamp(int(wp.floor(ux)), 0, cx - 1)
    cj = wp.clamp(int(wp.floor(uy)), 0, cy - 1)
    ck = wp.clamp(int(wp.floor(uz)), 0, cz - 1)

    # If precomputed active cell differs from natural cell, redirect and extrapolate
    cell = flex_vertcell[vertid]
    if cell >= 0 and cell != ci * cy * cz + cj * cz + ck:
      ck = cell % cz
      cj = (cell // cz) % cy
      ci = cell // (cy * cz)
      local_x = wp.clamp(ux - float(ci), -0.02, 1.02)
      local_y = wp.clamp(uy - float(cj), -0.02, 1.02)
      local_z = wp.clamp(uz - float(ck), -0.02, 1.02)
    else:
      local_x = wp.clamp(ux - float(ci), 0.0, 1.0)
      local_y = wp.clamp(uy - float(cj), 0.0, 1.0)
      local_z = wp.clamp(uz - float(ck), 0.0, 1.0)
    local = wp.vec3(local_x, local_y, local_z)

    # Node grid dimensions
    ny_g = cy + 1
    nz_g = cz + 1
    nstart = flex_nodeadr[f]

    # Accumulate weighted node positions
    result = wp.vec3(0.0, 0.0, 0.0)
    for li in range(2):
      for lj in range(2):
        for lk in range(2):
          w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)
          gi = ci + li
          gj = cj + lj
          gk = ck + lk
          node_idx = gi * ny_g * nz_g + gj * nz_g + gk
          result += w * flexnode_xpos_in[worldid, nstart + node_idx]

    flexvert_xpos_out[worldid, vertid] = result
  else:
    # Non-interpolated flex: vertex position from single body
    bodyid = flex_vertbodyid[vertid]
    xpos = xpos_in[worldid, bodyid]

    if flex_centered[f]:
      flexvert_xpos_out[worldid, vertid] = xpos
    else:
      xmat = xmat_in[worldid, bodyid]
      local_pos = flex_vert[vertid]
      flexvert_xpos_out[worldid, vertid] = xmat @ local_pos + xpos


@wp.kernel
def _flex_nodes(
  # Model:
  nflex: int,
  flex_nodeadr: wp.array[int],
  flex_nodenum: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_node: wp.array[wp.vec3],
  flex_centered: wp.array[bool],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  # Data out:
  flexnode_xpos_out: wp.array2d[wp.vec3],
):
  worldid, nodeid = wp.tid()

  for f in range(nflex):
    locid = nodeid - flex_nodeadr[f]
    if locid >= 0 and locid < flex_nodenum[f]:
      break

  bodyid = flex_nodebodyid[nodeid]
  xpos = xpos_in[worldid, bodyid]

  if flex_centered[f]:
    flexnode_xpos_out[worldid, nodeid] = xpos
  else:
    local_pos = flex_node[nodeid]
    if local_pos[0] == 0.0 and local_pos[1] == 0.0 and local_pos[2] == 0.0:
      flexnode_xpos_out[worldid, nodeid] = xpos
    else:
      xmat = xmat_in[worldid, bodyid]
      flexnode_xpos_out[worldid, nodeid] = xmat @ local_pos + xpos


@wp.kernel
def _flex_edges(
  # Model:
  body_rootid: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_edge: wp.array[wp.vec2i],
  flexedge_J_rownnz: wp.array[int],
  flexedge_J_rowadr: wp.array[int],
  flexedge_J_colind: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  flex_edgeflexid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  flexedge_J_out: wp.array2d[float],
  flexedge_length_out: wp.array2d[float],
  flexedge_velocity_out: wp.array2d[float],
):
  worldid, edgeid = wp.tid()
  f = flex_edgeflexid[edgeid]

  vbase = flex_vertadr[f]
  v = flex_edge[edgeid]
  vbase0 = vbase + v[0]
  vbase1 = vbase + v[1]

  pos1 = flexvert_xpos_in[worldid, vbase0]
  pos2 = flexvert_xpos_in[worldid, vbase1]
  vec = pos2 - pos1
  edge, edge_length = math.normalize_with_norm(vec)
  flexedge_length_out[worldid, edgeid] = edge_length

  b1 = flex_vertbodyid[vbase0]
  b2 = flex_vertbodyid[vbase1]

  rownnz = flexedge_J_rownnz[edgeid]
  if b1 < 0 or b2 < 0 or rownnz == 0:
    flexedge_velocity_out[worldid, edgeid] = 0.0
    return

  rowadr = flexedge_J_rowadr[edgeid]
  root1 = body_rootid[b1]
  root2 = body_rootid[b2]
  com1 = subtree_com_in[worldid, root1]
  com2 = com1 if root2 == root1 else subtree_com_in[worldid, root2]
  offset1 = pos1 - com1
  offset2 = pos2 - com2
  vel = float(0.0)

  for i in range(rownnz):
    dofid = flexedge_J_colind[rowadr + i]
    cdof = cdof_in[worldid, dofid]
    cdof_ang = wp.spatial_top(cdof)
    cdof_lin = wp.spatial_bottom(cdof)
    J_val = float(0.0)
    if body_isdofancestor[b1, dofid] != 0:
      jacp1 = cdof_lin + wp.cross(cdof_ang, offset1)
      J_val -= wp.dot(jacp1, edge)
    if body_isdofancestor[b2, dofid] != 0:
      jacp2 = cdof_lin + wp.cross(cdof_ang, offset2)
      J_val += wp.dot(jacp2, edge)
    flexedge_J_out[worldid, rowadr + i] = J_val
    vel += J_val * qvel_in[worldid, dofid]

  flexedge_velocity_out[worldid, edgeid] = vel


@event_scope
def kinematics(m: Model, d: Data):
  """Computes forward kinematics for all bodies, sites, geoms, and flexible elements.

  This function updates the global positions and orientations of all bodies, as well as the
  derived positions and orientations of geoms, sites, and flexible elements, based on the
  current joint positions and any attached mocap bodies.
  """
  wp.launch(
    _kinematics_branch,
    dim=(d.nworld, m.nbranch),
    inputs=[
      m.qpos0,
      m.body_parentid,
      m.body_mocapid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_pos,
      m.body_quat,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_pos,
      m.jnt_axis,
      m.body_branches,
      m.body_branch_start,
      d.qpos,
      d.mocap_pos,
      d.mocap_quat,
    ],
    outputs=[d.xpos, d.xquat, d.xanchor, d.xaxis],
  )

  wp.launch(
    _compute_body_matrices,
    dim=(d.nworld, m.nbody),
    inputs=[d.xquat],
    outputs=[d.xmat],
  )

  wp.launch(
    _compute_body_inertial_frames,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_ipos, m.body_iquat, d.xpos, d.xquat],
    outputs=[d.xipos, d.ximat],
  )

  wp.launch(
    _geom_local_to_global,
    dim=(d.nworld, m.ngeom),
    inputs=[m.body_rootid, m.body_weldid, m.body_mocapid, m.geom_bodyid, m.geom_pos, m.geom_quat, d.xpos, d.xquat],
    outputs=[d.geom_xpos, d.geom_xmat],
  )

  wp.launch(
    _site_local_to_global,
    dim=(d.nworld, m.nsite),
    inputs=[m.site_bodyid, m.site_pos, m.site_quat, d.xpos, d.xquat],
    outputs=[d.site_xpos, d.site_xmat],
  )


@wp.kernel
def _flex_face_kinematics(
  # Model:
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_node0: wp.array[wp.vec3],
  flex_face_map: wp.array[wp.vec2i],
  flex_face: wp.array2d[int],
  # Data in:
  flexnode_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  face_xpos_out: wp.array3d[wp.vec3],
  face_quat_out: wp.array2d[wp.quat],
):
  worldid, face_id = wp.tid()

  mapping = flex_face_map[face_id]
  f = mapping[0]
  face_elem_idx = mapping[1]

  order = flex_interp[f]
  order_abs = -order
  cellnum = flex_cellnum[f]
  cx = cellnum[0]
  cy = cellnum[1]
  cz = cellnum[2]
  t1 = wp.vec3(0.0)
  t2 = wp.vec3(0.0)
  gidx0 = int(-1)
  gidx1 = int(-1)
  for local_idx in range(9):
    gidx = flex_face[face_id, local_idx]
    if gidx != -1:
      if local_idx == 0:
        gidx0 = gidx
      gidx1 = gidx
      node_pos = flexnode_xpos_in[worldid, gidx]

      face_xpos_out[worldid, face_id, local_idx] = node_pos

      l0 = local_idx // (order_abs + 1)
      l1 = local_idx % (order_abs + 1)

      dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))
      dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))
      phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5
      phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5

      grad0 = dphi0 * phi1
      grad1 = phi0 * dphi1

      t1 += node_pos * grad0
      t2 += node_pos * grad1
    else:
      face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)

  normal_axis, _, _, _, _, _ = support.get_face_metadata(cx, cy, cz, face_elem_idx, order_abs)
  axis0 = (normal_axis + 1) % 3
  axis1 = (normal_axis + 2) % 3
  if gidx0 != -1 and gidx1 != -1:
    dref = flex_node0[gidx1] - flex_node0[gidx0]
    d0 = dref[axis0]
    d1 = dref[axis1]
    if d0 > MJ_MINVAL:
      t1 = t1 / d0
    if d1 > MJ_MINVAL:
      t2 = t2 / d1

  normal = wp.cross(t1, t2)

  F = wp.mat33(0.0)
  if normal_axis == 0:
    F = wp.mat33(
      normal[0],
      t1[0],
      t2[0],
      normal[1],
      t1[1],
      t2[1],
      normal[2],
      t1[2],
      t2[2],
    )
  elif normal_axis == 1:
    F = wp.mat33(
      t2[0],
      normal[0],
      t1[0],
      t2[1],
      normal[1],
      t1[1],
      t2[2],
      normal[2],
      t1[2],
    )
  else:
    F = wp.mat33(
      t1[0],
      t2[0],
      normal[0],
      t1[1],
      t2[1],
      normal[1],
      t1[2],
      t2[2],
      normal[2],
    )

  face_quat_out[worldid, face_id] = support.mat33_to_quat_polar(F)


@wp.func
def _rod_body_pose0(
  # Model:
  body_parentid: wp.array[int],
  body_pos: wp.array2d[wp.vec3],
  body_quat: wp.array2d[wp.quat],
  # In:
  worldid: int,
  bodyid: int,
) -> Tuple[wp.vec3, wp.quat]:
  """Nominal pose of a body with all joints at zero (rodBodyPose0)."""
  pos_id = worldid % body_pos.shape[0]
  quat_id = worldid % body_quat.shape[0]
  pos = wp.vec3(0.0)
  quat = wp.quat(1.0, 0.0, 0.0, 0.0)
  cur = bodyid
  while cur > 0:
    q = body_quat[quat_id, cur]
    pos = math.rot_vec_quat(pos, q) + body_pos[pos_id, cur]
    quat = math.mul_quat(q, quat)
    cur = body_parentid[cur]
  return pos, quat


@wp.func
def _rod_vert_pos0(
  # Model:
  body_parentid: wp.array[int],
  body_pos: wp.array2d[wp.vec3],
  body_quat: wp.array2d[wp.quat],
  flex_vertbodyid: wp.array[int],
  flex_vert: wp.array[wp.vec3],
  # In:
  worldid: int,
  vertid: int,
) -> wp.vec3:
  """Nominal world position of a flex vertex with all joints at zero (rodVertPos0)."""
  pos, quat = _rod_body_pose0(body_parentid, body_pos, body_quat, worldid, flex_vertbodyid[vertid])
  return pos + math.rot_vec_quat(flex_vert[vertid], quat)


@wp.kernel
def _flex_rod_ref(
  # Model:
  body_parentid: wp.array[int],
  body_pos: wp.array2d[wp.vec3],
  body_quat: wp.array2d[wp.quat],
  flex_vertadr: wp.array[int],
  flex_bendingadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_vert: wp.array[wp.vec3],
  flex_bending: wp.array[float],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_hingeadr: wp.array[int],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  # Data out:
  flex_rod_chain_valid_out: wp.array2d[bool],
  flex_rod_ref_ok_out: wp.array2d[bool],
  flex_rod_frame0_loc_out: wp.array3d[wp.vec3],
  flex_rod_frame0_ref_out: wp.array3d[wp.vec3],
  flex_rod_tail_loc_out: wp.array3d[wp.vec3],
  flex_rodhinge_om0_out: wp.array2d[wp.vec2],
):
  """Reference frames and curvatures of open 1D DER rods at the nominal body poses of this world.

  Mirrors rodChain, rodRootDirector and rodTailDirector, which recompute them from body_pos and
  body_quat; also invalidates the cached rod chain evaluation (passive.eval_rod_chains).
  """
  worldid, f = wp.tid()
  z3 = wp.vec3(0.0)
  flex_rod_chain_valid_out[worldid, f] = False
  flex_rod_ref_ok_out[worldid, f] = False
  for i in range(2):
    flex_rod_frame0_loc_out[worldid, f, i] = z3
    flex_rod_frame0_ref_out[worldid, f, i] = z3
    flex_rod_tail_loc_out[worldid, f, i] = z3

  nedge = flex_rod_edgenum[f]
  nhinge = nedge - 1
  if wp.static(ROD_HINGE_SIZE != 10) or nhinge <= 0:
    return

  hadr = flex_rod_hingeadr[f]
  if hadr >= 0:
    for h in range(nhinge):
      flex_rodhinge_om0_out[worldid, hadr + h] = wp.vec2(0.0)

  va = flex_vertadr[f]
  eadr = flex_rod_edgeadr[f]
  bendingadr = flex_bendingadr[f]

  # first edge: qpos0 frame in the frame of its first body (rodRootDirector)
  verts0 = flex_rodedge_vert[eadr]
  _p0, q0 = _rod_body_pose0(body_parentid, body_pos, body_quat, worldid, flex_vertbodyid[va + verts0[0]])
  R0 = math.quat_to_mat(q0)
  xa = _rod_vert_pos0(body_parentid, body_pos, body_quat, flex_vertbodyid, flex_vert, worldid, va + verts0[0])
  xv = _rod_vert_pos0(body_parentid, body_pos, body_quat, flex_vertbodyid, flex_vert, worldid, va + verts0[1])
  t0_loc = wp.transpose(R0) @ (xv - xa)
  n_t0 = wp.length(t0_loc)
  if n_t0 < MJ_MINVAL:
    return
  t0_loc = t0_loc / n_t0
  frame_u, frame_v = math.orthogonals(t0_loc)
  psi0 = float(0.0)
  if flex_bending[bendingadr + 1 + 9] < 0.0:
    psi0 = 0.5 * wp.pi
  c0 = wp.cos(psi0)
  s0 = wp.sin(psi0)
  u0_loc = c0 * frame_u + s0 * frame_v
  v0_loc = -s0 * frame_u + c0 * frame_v
  t_ref = R0 @ t0_loc
  u_ref = R0 @ u0_loc
  v_prev = R0 @ v0_loc
  flex_rod_frame0_loc_out[worldid, f, 0] = t0_loc
  flex_rod_frame0_loc_out[worldid, f, 1] = u0_loc
  flex_rod_frame0_ref_out[worldid, f, 0] = t_ref
  flex_rod_frame0_ref_out[worldid, f, 1] = u_ref

  # reference Bishop frame along the qpos0 chain, for curved references or a clamped tail (rodChain)
  tail_clamped = flex_rod_tail_clamped[f]
  need_ref_chain = tail_clamped
  for h in range(nhinge):
    if flex_bending[bendingadr + 1 + ROD_HINGE_SIZE * h + 6] > 0.0:
      need_ref_chain = True
  if need_ref_chain:
    for h in range(nhinge):
      xb = _rod_vert_pos0(
        body_parentid, body_pos, body_quat, flex_vertbodyid, flex_vert, worldid, va + flex_rodedge_vert[eadr + h + 1][1]
      )
      if hadr >= 0 and flex_bending[bendingadr + 1 + ROD_HINGE_SIZE * h + 6] > 0.0:
        e0 = xv - xa
        e1 = xb - xv
        l0 = wp.length(e0)
        l1 = wp.length(e1)
        denom = l0 * l1 + wp.dot(e0, e1)
        if l0 >= MJ_MINVAL and l1 >= MJ_MINVAL and denom >= MJ_MINVAL:
          kb = (2.0 / denom) * wp.cross(e0, e1)
          flex_rodhinge_om0_out[worldid, hadr + h] = wp.vec2(wp.dot(kb, u_ref), wp.dot(kb, v_prev))
      t_next = xb - xv
      n_next = wp.length(t_next)
      if n_next < MJ_MINVAL:
        return
      t_next = t_next / n_next
      ok_pt, u_next, v_next = support.rod_parallel_transport(t_ref, t_next, u_ref)
      if not ok_pt:
        return
      t_ref = t_next
      u_ref = u_next
      v_prev = v_next
      xa = xv
      xv = xb

    # clamped last edge: qpos0 frame in the frame of its body (rodTailDirector)
    if tail_clamped:
      _pT, qT = _rod_body_pose0(
        body_parentid, body_pos, body_quat, worldid, flex_vertbodyid[va + flex_rodedge_vert[eadr + nhinge][1]]
      )
      RT = math.quat_to_mat(qT)
      flex_rod_tail_loc_out[worldid, f, 0] = wp.transpose(RT) @ t_ref
      flex_rod_tail_loc_out[worldid, f, 1] = wp.transpose(RT) @ u_ref

  flex_rod_ref_ok_out[worldid, f] = True


@event_scope
def flex(m: Model, d: Data):
  if m.nflex > 0:
    d.flex_hessian_valid.zero_()
  # Compute node positions first (needed for interpolated vertex positions)
  wp.launch(
    _flex_nodes,
    dim=(d.nworld, m.nflexnode),
    inputs=[
      m.nflex,
      m.flex_nodeadr,
      m.flex_nodenum,
      m.flex_nodebodyid,
      m.flex_node,
      m.flex_centered,
      d.xpos,
      d.xmat,
    ],
    outputs=[d.flexnode_xpos],
  )

  wp.launch(
    _flex_vertices,
    dim=(d.nworld, m.nflexvert),
    inputs=[
      m.nflex,
      m.flex_interp,
      m.flex_cellnum,
      m.flex_nodeadr,
      m.flex_vertadr,
      m.flex_vertnum,
      m.flex_vertbodyid,
      m.flex_vertcell,
      m.flex_vert,
      m.flex_vert0,
      m.flex_centered,
      d.xpos,
      d.xmat,
      d.flexnode_xpos,
    ],
    outputs=[d.flexvert_xpos],
  )
  wp.launch(
    _flex_edges,
    dim=(d.nworld, m.nflexedge),
    inputs=[
      m.body_rootid,
      m.flex_vertadr,
      m.flex_vertbodyid,
      m.flex_edge,
      m.flexedge_J_rownnz,
      m.flexedge_J_rowadr,
      m.flexedge_J_colind,
      m.body_isdofancestor,
      m.flex_edgeflexid,
      d.qvel,
      d.subtree_com,
      d.cdof,
      d.flexvert_xpos,
    ],
    outputs=[
      d.flexedge_J,
      d.flexedge_length,
      d.flexedge_velocity,
    ],
  )

  wp.launch(
    _flex_face_kinematics,
    dim=(d.nworld, m.nflexface),
    inputs=[
      m.flex_interp,
      m.flex_cellnum,
      m.flex_node0,
      m.flex_face_map,
      m.flex_face,
      d.flexnode_xpos,
    ],
    outputs=[
      d.face_xpos,
      d.face_quat,
    ],
  )

  if m.nflexrodedge > 0:
    wp.launch(
      _flex_rod_ref,
      dim=(d.nworld, m.nflex),
      inputs=[
        m.body_parentid,
        m.body_pos,
        m.body_quat,
        m.flex_vertadr,
        m.flex_bendingadr,
        m.flex_vertbodyid,
        m.flex_vert,
        m.flex_bending,
        m.flex_rod_tail_clamped,
        m.flex_rod_hingeadr,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_vert,
      ],
      outputs=[
        d.flex_rod_chain_valid,
        d.flex_rod_ref_ok,
        d.flex_rod_frame0_loc,
        d.flex_rod_frame0_ref,
        d.flex_rod_tail_loc,
        d.flex_rodhinge_om0,
      ],
    )


@wp.kernel
def _subtree_com_init(
  # Model:
  body_mass: wp.array2d[float],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  # Data out:
  subtree_com_out: wp.array2d[wp.vec3],
):
  worldid, bodyid = wp.tid()
  subtree_com_out[worldid, bodyid] = xipos_in[worldid, bodyid] * body_mass[worldid % body_mass.shape[0], bodyid]


@cache_kernel
def _subtree_com_acc(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    # In:
    body_tree_: wp.array[int],
    # Data out:
    subtree_com_out: wp.array2d[wp.vec3],
  ):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if bodyid != 0:
      wp.atomic_add(subtree_com_out, worldid, pid, subtree_com_in[worldid, bodyid])

  return kernel


@wp.kernel
def _subtree_div(
  # Model:
  body_subtreemass: wp.array2d[float],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  subtree_com_out: wp.array2d[wp.vec3],
):
  worldid, bodyid = wp.tid()
  com = subtree_com_in[worldid, bodyid]
  mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]
  if mass != 0.0:
    subtree_com_out[worldid, bodyid] = com / mass


@wp.kernel
def _cinert(
  # Model:
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  cinert_out: wp.array2d[vec10],
):
  worldid, bodyid = wp.tid()
  mat = ximat_in[worldid, bodyid]
  inert = body_inertia[worldid % body_inertia.shape[0], bodyid]
  mass = body_mass[worldid % body_mass.shape[0], bodyid]
  dif = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]
  # express inertia in com-based frame (mju_inertCom)

  res = vec10()
  # res_rot = mat * diag(inert) * mat'
  tmp = mat @ wp.diag(inert) @ wp.transpose(mat)
  res[0] = tmp[0, 0]
  res[1] = tmp[1, 1]
  res[2] = tmp[2, 2]
  res[3] = tmp[0, 1]
  res[4] = tmp[0, 2]
  res[5] = tmp[1, 2]
  # res_rot -= mass * dif_cross * dif_cross
  res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])
  res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])
  res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])
  res[3] -= mass * dif[0] * dif[1]
  res[4] -= mass * dif[0] * dif[2]
  res[5] -= mass * dif[1] * dif[2]
  # res_tran = mass * dif
  res[6] = mass * dif[0]
  res[7] = mass * dif[1]
  res[8] = mass * dif[2]
  # res_mass = mass
  res[9] = mass

  cinert_out[worldid, bodyid] = res


@wp.kernel
def _cdof(
  # Model:
  body_rootid: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  # Data in:
  xmat_in: wp.array2d[wp.mat33],
  xanchor_in: wp.array2d[wp.vec3],
  xaxis_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  cdof_out: wp.array2d[wp.spatial_vector],
):
  worldid, jntid = wp.tid()
  bodyid = jnt_bodyid[jntid]
  dofid = jnt_dofadr[jntid]
  jnt_type_ = jnt_type[jntid]
  xaxis = xaxis_in[worldid, jntid]
  xmat = wp.transpose(xmat_in[worldid, bodyid])

  # compute com-anchor vector
  offset = subtree_com_in[worldid, body_rootid[bodyid]] - xanchor_in[worldid, jntid]

  res = cdof_out[worldid]
  if jnt_type_ == JointType.FREE:
    res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    # I_3 rotation in child frame (assume no subsequent rotations)
    res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
    res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
    res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
  elif jnt_type_ == JointType.BALL:  # ball
    # I_3 rotation in child frame (assume no subsequent rotations)
    res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
    res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
    res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
  elif jnt_type_ == JointType.SLIDE:
    res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
  elif jnt_type_ == JointType.HINGE:  # hinge
    res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))


@event_scope
def com_pos(m: Model, d: Data):
  """Computes subtree center of mass positions.

  Transforms inertia and motion to global frame centered at subtree CoM. Accumulates the
  mass-weighted positions up the kinematic tree, divides by total mass, and computes composite
  inertias and motion degrees of freedom in the subtree CoM frame.
  """
  wp.launch(_subtree_com_init, dim=(d.nworld, m.nbody), inputs=[m.body_mass, d.xipos], outputs=[d.subtree_com])

  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  subtree_com_in = wp.empty_like(d.subtree_com) if det_atomics else d.subtree_com
  for i in reversed(range(len(m.body_tree))):
    body_tree = m.body_tree[i]
    if det_atomics:
      wp.copy(subtree_com_in, d.subtree_com)
    wp.launch(
      _subtree_com_acc(det_atomics),
      dim=(d.nworld, body_tree.size),
      inputs=[m.body_parentid, subtree_com_in, body_tree],
      outputs=[d.subtree_com],
    )

  wp.launch(_subtree_div, dim=(d.nworld, m.nbody), inputs=[m.body_subtreemass, d.subtree_com], outputs=[d.subtree_com])
  wp.launch(
    _cinert,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
    outputs=[d.cinert],
  )
  wp.launch(
    _cdof,
    dim=(d.nworld, m.njnt),
    inputs=[m.body_rootid, m.jnt_type, m.jnt_dofadr, m.jnt_bodyid, d.xmat, d.xanchor, d.xaxis, d.subtree_com],
    outputs=[d.cdof],
  )


@wp.kernel
def _cam_local_to_global(
  # Model:
  cam_mode: wp.array[int],
  cam_bodyid: wp.array[int],
  cam_targetbodyid: wp.array[int],
  cam_pos: wp.array2d[wp.vec3],
  cam_quat: wp.array2d[wp.quat],
  cam_poscom0: wp.array2d[wp.vec3],
  cam_pos0: wp.array2d[wp.vec3],
  cam_mat0: wp.array2d[wp.mat33],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  cam_xpos_out: wp.array2d[wp.vec3],
  cam_xmat_out: wp.array2d[wp.mat33],
):
  worldid, camid = wp.tid()
  cam_pos_id = worldid % cam_pos.shape[0]
  cam_quat_id = worldid % cam_quat.shape[0]
  is_target_cam = (cam_mode[camid] == CamLightType.TARGETBODY) or (cam_mode[camid] == CamLightType.TARGETBODYCOM)
  invalid_target = is_target_cam and (cam_targetbodyid[camid] < 0)
  if invalid_target:
    bodyid = cam_bodyid[camid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)
    cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))
  elif cam_mode[camid] == CamLightType.TRACK:
    cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]
    body_xpos = xpos_in[worldid, cam_bodyid[camid]]
    cam_xpos_out[worldid, camid] = body_xpos + cam_pos0[worldid % cam_pos0.shape[0], camid]
  elif cam_mode[camid] == CamLightType.TRACKCOM:
    cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]
    cam_xpos_out[worldid, camid] = (
      subtree_com_in[worldid, cam_bodyid[camid]] + cam_poscom0[worldid % cam_poscom0.shape[0], camid]
    )
  elif cam_mode[camid] == CamLightType.TARGETBODY or cam_mode[camid] == CamLightType.TARGETBODYCOM:
    bodyid = cam_bodyid[camid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)
    pos = xpos_in[worldid, cam_targetbodyid[camid]]
    if cam_mode[camid] == CamLightType.TARGETBODYCOM:
      pos = subtree_com_in[worldid, cam_targetbodyid[camid]]
    # zaxis = -desired camera direction, in global frame
    mat_3 = wp.normalize(cam_xpos_out[worldid, camid] - pos)
    # xaxis: orthogonal to zaxis and to (0,0,1)
    mat_1 = wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), mat_3))
    mat_2 = wp.normalize(wp.cross(mat_3, mat_1))
    # fmt: off
    cam_xmat_out[worldid, camid] = wp.mat33(
      mat_1[0], mat_2[0], mat_3[0],
      mat_1[1], mat_2[1], mat_3[1],
      mat_1[2], mat_2[2], mat_3[2]
    )
    # fmt: on
  else:
    bodyid = cam_bodyid[camid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)
    cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))


@wp.kernel
def _light_local_to_global(
  # Model:
  light_mode: wp.array[int],
  light_bodyid: wp.array[int],
  light_targetbodyid: wp.array[int],
  light_pos: wp.array2d[wp.vec3],
  light_dir: wp.array2d[wp.vec3],
  light_poscom0: wp.array2d[wp.vec3],
  light_pos0: wp.array2d[wp.vec3],
  light_dir0: wp.array2d[wp.vec3],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  light_xpos_out: wp.array2d[wp.vec3],
  light_xdir_out: wp.array2d[wp.vec3],
):
  worldid, lightid = wp.tid()
  light_pos_id = worldid % light_pos.shape[0]
  light_dir_id = worldid % light_dir.shape[0]
  is_target_light = (light_mode[lightid] == CamLightType.TARGETBODY) or (light_mode[lightid] == CamLightType.TARGETBODYCOM)
  invalid_target = is_target_light and (light_targetbodyid[lightid] < 0)
  if invalid_target:
    bodyid = light_bodyid[lightid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)
    light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)
    return
  elif light_mode[lightid] == CamLightType.TRACK:
    light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]
    body_xpos = xpos_in[worldid, light_bodyid[lightid]]
    light_xpos_out[worldid, lightid] = body_xpos + light_pos0[worldid % light_pos0.shape[0], lightid]
  elif light_mode[lightid] == CamLightType.TRACKCOM:
    light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]
    light_xpos_out[worldid, lightid] = (
      subtree_com_in[worldid, light_bodyid[lightid]] + light_poscom0[worldid % light_poscom0.shape[0], lightid]
    )
  elif light_mode[lightid] == CamLightType.TARGETBODY or light_mode[lightid] == CamLightType.TARGETBODYCOM:
    bodyid = light_bodyid[lightid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)
    pos = xpos_in[worldid, light_targetbodyid[lightid]]
    if light_mode[lightid] == CamLightType.TARGETBODYCOM:
      pos = subtree_com_in[worldid, light_targetbodyid[lightid]]
    light_xdir_out[worldid, lightid] = pos - light_xpos_out[worldid, lightid]
  else:
    bodyid = light_bodyid[lightid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)
    light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)

  light_xdir_out[worldid, lightid] = wp.normalize(light_xdir_out[worldid, lightid])


@event_scope
def camlight(m: Model, d: Data):
  """Computes camera and light positions and orientations.

  Updates the global positions and orientations for all cameras and lights in the model,
  including special handling for tracking and target modes.
  """
  wp.launch(
    _cam_local_to_global,
    dim=(d.nworld, m.ncam),
    inputs=[
      m.cam_mode,
      m.cam_bodyid,
      m.cam_targetbodyid,
      m.cam_pos,
      m.cam_quat,
      m.cam_poscom0,
      m.cam_pos0,
      m.cam_mat0,
      d.xpos,
      d.xquat,
      d.subtree_com,
    ],
    outputs=[d.cam_xpos, d.cam_xmat],
  )
  wp.launch(
    _light_local_to_global,
    dim=(d.nworld, m.nlight),
    inputs=[
      m.light_mode,
      m.light_bodyid,
      m.light_targetbodyid,
      m.light_pos,
      m.light_dir,
      m.light_poscom0,
      m.light_pos0,
      m.light_dir0,
      d.xpos,
      d.xquat,
      d.subtree_com,
    ],
    outputs=[d.light_xpos, d.light_xdir],
  )


@cache_kernel
def _crb_accumulate(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    # Data in:
    crb_in: wp.array2d[vec10],
    # In:
    body_tree_: wp.array[int],
    # Data out:
    crb_out: wp.array2d[vec10],
  ):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if pid == 0:
      return
    wp.atomic_add(crb_out, worldid, pid, crb_in[worldid, bodyid])

  return kernel


@wp.kernel
def _M(
  # Model:
  jnt_actuatorid: wp.array[int],
  dof_bodyid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_parentid: wp.array[int],
  dof_armature: wp.array2d[float],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array2d[float],
  actuator_gear: wp.array2d[wp.spatial_vector],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  crb_in: wp.array2d[vec10],
  # Data out:
  M_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  bodyid = dof_bodyid[dofid]
  rowadr = M_rowadr[dofid]
  madr_ij = rowadr + M_rownnz[dofid] - 1

  # precompute buf = crb_body_i * cdof_i
  cdof = cdof_in[worldid, dofid]
  buf = math.inert_vec(crb_in[worldid, bodyid], cdof)

  # M(i,i): armature inertia plus the dof's own term
  M_out[worldid, madr_ij] = util_misc.dof_armature_total(
    jnt_actuatorid, dof_jntid, dof_armature, actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, dofid
  ) + wp.dot(cdof, buf)

  # sparse backward pass over the ancestors stored in this row; rows of simple dofs are
  # diagonal-only, so stop at the row start instead of walking into the previous row
  madr_ij -= 1
  dofid = dof_parentid[dofid]
  while dofid >= 0 and madr_ij >= rowadr:
    M_out[worldid, madr_ij] = wp.dot(cdof_in[worldid, dofid], buf)
    madr_ij -= 1
    dofid = dof_parentid[dofid]


@event_scope
def crb(m: Model, d: Data):
  """Computes composite rigid body inertias for each body and the joint-space inertia matrix.

  Accumulates composite rigid body inertias up the kinematic tree and computes the
  joint-space inertia matrix in either sparse or dense format, depending on model options.
  """
  wp.copy(d.crb, d.cinert)

  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  crb_in = wp.empty_like(d.crb) if det_atomics else d.crb
  for i in reversed(range(len(m.body_tree))):
    body_tree = m.body_tree[i]
    if det_atomics:
      wp.copy(crb_in, d.crb)
    wp.launch(
      _crb_accumulate(det_atomics),
      dim=(d.nworld, body_tree.size),
      inputs=[m.body_parentid, crb_in, body_tree],
      outputs=[d.crb],
    )

  d.M.zero_()
  wp.launch(
    _M,
    dim=(d.nworld, m.nv),
    inputs=[
      m.jnt_actuatorid,
      m.dof_bodyid,
      m.dof_jntid,
      m.dof_parentid,
      m.dof_armature,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_armature,
      m.actuator_gear,
      m.M_rownnz,
      m.M_rowadr,
      d.cdof,
      d.crb,
    ],
    outputs=[d.M],
  )


@cache_kernel
def _tendon_armature(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    dof_parentid: wp.array[int],
    tendon_actuatorid: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_armature: wp.array2d[float],
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_armature: wp.array2d[float],
    actuator_gear: wp.array2d[wp.spatial_vector],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    # Data in:
    ten_J_in: wp.array2d[float],
    # Data out:
    M_out: wp.array2d[float],
  ):
    worldid, tenid, dofid = wp.tid()

    armature = util_misc.tendon_armature_total(
      tendon_actuatorid, tendon_armature, actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, tenid
    )

    if armature == 0.0:
      return

    rownnz = ten_J_rownnz[tenid]
    if dofid >= rownnz:
      return
    rowadr = ten_J_rowadr[tenid]
    dofid_sparse = dofid
    sparseid = rowadr + dofid_sparse
    dofid = ten_J_colind[sparseid]
    ten_Ji = ten_J_in[worldid, sparseid]

    if ten_Ji == 0.0:
      return

    # Walk the row's entries from the diagonal backward over ancestors.
    madr_ij = M_rowadr[dofid] + M_rownnz[dofid] - 1

    # sparse backward pass over ancestors
    dofidi = dofid
    ptr = dofid_sparse
    while dofid >= 0:
      if dofid == dofidi:
        ten_Jj = ten_Ji
      else:
        # scan pointer backward to find matching colind entry
        while ptr >= 0:
          sparseid = rowadr + ptr
          if ten_J_colind[sparseid] <= dofid:
            break
          ptr -= 1
        if ptr >= 0 and ten_J_colind[sparseid] == dofid:
          ten_Jj = ten_J_in[worldid, sparseid]
        else:
          ten_Jj = float(0.0)

      Mij = armature * ten_Jj * ten_Ji

      wp.atomic_add(M_out[worldid], madr_ij, Mij)
      madr_ij -= 1

      dofid = dof_parentid[dofid]

  return kernel


@event_scope
def tendon_armature(m: Model, d: Data):
  """Add tendon armature to M."""
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  wp.launch(
    _tendon_armature(m.nv if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
    inputs=[
      m.dof_parentid,
      m.tendon_actuatorid,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.tendon_armature,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_armature,
      m.actuator_gear,
      m.M_rownnz,
      m.M_rowadr,
      d.ten_J,
    ],
    outputs=[d.M],
  )


@cache_kernel
def _qLD_acc(nv: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = nv

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    # In:
    qLD_updates_: wp.array[wp.vec3i],
    L_in: wp.array2d[float],
    # Out:
    L_out: wp.array2d[float],
  ):
    worldid, nodeid = wp.tid()
    update = qLD_updates_[nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    Madr_i = M_rowadr[i]  # Address of row being updated
    diag_k = M_rowadr[k] + M_rownnz[k] - 1  # Address of diagonal element of k
    # tmp = M(k,i) / M(k,k)
    tmp = L_in[worldid, Madr_ki] / L_in[worldid, diag_k]
    for j in range(M_rownnz[i]):
      # M(i,j) -= M(k,j) * tmp
      wp.atomic_sub(L_out[worldid], Madr_i + j, L_in[worldid, M_rowadr[k] + j] * tmp)
    if wp.static(not deterministic):
      # M(k,i) = tmp
      L_out[worldid, Madr_ki] = tmp

  return kernel


@wp.kernel
def _qLD_scale(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # In:
  qLD_updates_: wp.array[wp.vec3i],
  L_in: wp.array2d[float],
  # Out:
  L_out: wp.array2d[float],
):
  worldid, nodeid = wp.tid()
  update = qLD_updates_[nodeid]
  k, Madr_ki = update[1], update[2]
  diag_k = M_rowadr[k] + M_rownnz[k] - 1
  L_out[worldid, Madr_ki] = L_in[worldid, Madr_ki] / L_in[worldid, diag_k]


@wp.kernel
def _qLDiag_div(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # In:
  L_in: wp.array2d[float],
  # Out:
  D_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  diag_i = M_rowadr[dofid] + M_rownnz[dofid] - 1  # Address of diagonal element of i
  D_out[worldid, dofid] = 1.0 / L_in[worldid, diag_i]


def _factor_i_sparse(m: Model, d: Data, M: wp.array2d[float], L: wp.array2d[float], D: wp.array2d[float]):
  """Sparse L'*D*L factorization of inertia-like matrix M, assumed spd."""
  wp.copy(L, M)

  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  L_in = wp.empty_like(L) if det_atomics else L
  for i in reversed(range(len(m.qLD_updates))):
    qLD_updates = m.qLD_updates[i]
    if det_atomics:
      wp.copy(L_in, L)
    wp.launch(
      _qLD_acc(m.nv if det_atomics else 0, det_atomics),
      dim=(d.nworld, qLD_updates.size),
      inputs=[m.M_rownnz, m.M_rowadr, qLD_updates, L_in],
      outputs=[L],
    )
    if det_atomics:
      wp.launch(
        _qLD_scale,
        dim=(d.nworld, qLD_updates.size),
        inputs=[m.M_rownnz, m.M_rowadr, qLD_updates, L_in],
        outputs=[L],
      )

  wp.launch(_qLDiag_div, dim=(d.nworld, m.nv), inputs=[m.M_rownnz, m.M_rowadr, L], outputs=[D])


@cache_kernel
def _small_cholesky_factorize_block(block_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    M_rowadr: wp.array[int],
    qLD_block_adr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    # In:
    block_dof: wp.array[int],
    # Out:
    D_out: wp.array2d[float],
    L_out: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]
    size = wp.static(block_size)
    matrix_adr = M_rowadr[start]

    factor_adr = qLD_block_adr[start]
    if factor_adr == Q_LD_BLOCK_COMPACT:
      for i in range(wp.static(block_size)):
        D_out[worldid, start + i] = 1.0 / M_in[worldid, matrix_adr + i]
    else:
      for i in range(wp.static(block_size)):
        value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]
        for k in range(i):
          factor = L_out[worldid, factor_adr + k * size + i]
          value -= factor * factor

        diagonal_value = wp.sqrt(value)
        L_out[worldid, factor_adr + i * size + i] = diagonal_value
        diagonal_inv = 1.0 / diagonal_value

        for j in range(i + 1, size):
          value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]
          for k in range(i):
            value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]
          L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv

  return kernel


@cache_kernel
def _tile_cholesky_factorize_block(tile: TileSet):
  # One diagonal block of `block_size` dofs per (world, block) tile group. tile_load_indexed gathers
  # the block's dense slots from CSR via a precomputed per-slot index tile (block_elemid, laid out
  # [block, slot]); structurally absent pairs carry an out-of-bounds index that reads as 0.
  # Tile shapes must be compile-time constants, so the densify is inlined per kernel (sharing via a
  # wp.func is not possible) -- keep it in sync with _tile_cholesky_factorize_solve_block.
  block_size = tile.size
  block_area = block_size * block_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    # In:
    block_elemid: wp.array[int],
    block_dof: wp.array[int],
    # Out:
    L_out: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]

    idx = wp.tile_load(block_elemid, shape=(block_area,), offset=(blk * block_area,), storage="shared")
    block = wp.tile_load_indexed(M_in[worldid], idx, shape=(block_area,), storage="shared")

    L = wp.tile_reshape(block, (block_size, block_size))
    wp.tile_cholesky_inplace(L, fill_mode="upper")
    wp.tile_store(L_out[worldid], wp.tile_reshape(L, (block_area,)), offset=(qLD_block_adr[start],))

  return kernel


def _factor_blocks(
  m: Model,
  d: Data,
  M: wp.array2d[float],
  L: wp.array2d[float],
  D: wp.array2d[float],
):
  for tile in m.M_tiles:
    if tile.elemid.size == 0:
      wp.launch(
        _small_cholesky_factorize_block(tile.size),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.M_rowadr, m.qLD_block_adr, M, tile.adr],
        outputs=[D, L],
        block_dim=m.block_dim.small_cholesky,
      )
    else:
      wp.launch_tiled(
        _tile_cholesky_factorize_block(tile),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.qLD_block_adr, M, tile.elemid, tile.adr],
        outputs=[L],
        block_dim=m.block_dim.cholesky_factorize,
      )


@event_scope
def factor_m(m: Model, d: Data):
  """Factorization of inertia-like matrix M, assumed spd.

  Compact blocks use reciprocal diagonals, full small blocks use scalar Cholesky, larger dense
  blocks use tile Cholesky, and oversized blocks use sparse LDL.
  """
  if m.M_tiles:
    _factor_blocks(m, d, d.M, d.qLD, d.qLDiagInv)
  if d.qLD.shape[1] > m.qLD_block_total:
    _factor_i_sparse(m, d, d.M, d.qLD[:, m.qLD_block_total :], d.qLDiagInv)


@wp.kernel
def _cacc_world(
  # In:
  gravity: wp.array[wp.vec3],
  # Data out:
  cacc_out: wp.array2d[wp.spatial_vector],
):
  worldid = wp.tid()
  cacc_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -gravity[worldid % gravity.shape[0]])


def _rne_cacc_world(m: Model, d: Data):
  if m.opt.disableflags & DisableBit.GRAVITY:
    d.cacc.zero_()
  else:
    wp.launch(_cacc_world, dim=[d.nworld], inputs=[m.opt.gravity], outputs=[d.cacc])


@wp.kernel
def _cacc_branch(
  # Model:
  body_parentid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_branches: wp.array[int],
  body_branch_start: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  flg_acc: bool,
  # Data out:
  cacc_out: wp.array2d[wp.spatial_vector],
):
  worldid, branchid = wp.tid()

  start = body_branch_start[branchid]
  end = body_branch_start[branchid + 1]

  bodyid = body_branches[start]
  pid = body_parentid[bodyid]
  local_cacc = cacc_out[worldid, pid]
  for i in range(start, end):
    bodyid = body_branches[i]
    dofnum = body_dofnum[bodyid]
    dofadr = body_dofadr[bodyid]
    for j in range(dofnum):
      local_cacc += cdof_dot_in[worldid, dofadr + j] * qvel_in[worldid, dofadr + j]
      if flg_acc:
        local_cacc += cdof_in[worldid, dofadr + j] * qacc_in[worldid, dofadr + j]
    cacc_out[worldid, bodyid] = local_cacc


def _rne_cacc_forward(m: Model, d: Data, flg_acc: bool = False):
  wp.launch(
    _cacc_branch,
    dim=(d.nworld, m.nbranch),
    inputs=[
      m.body_parentid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_branches,
      m.body_branch_start,
      d.qvel,
      d.qacc,
      d.cdof,
      d.cdof_dot,
      flg_acc,
    ],
    outputs=[d.cacc],
  )


@wp.kernel
def _cfrc(
  # Data in:
  cinert_in: wp.array2d[vec10],
  cvel_in: wp.array2d[wp.spatial_vector],
  cacc_in: wp.array2d[wp.spatial_vector],
  cfrc_ext_in: wp.array2d[wp.spatial_vector],
  # In:
  flg_cfrc_ext: bool,
  # Data out:
  cfrc_int_out: wp.array2d[wp.spatial_vector],
):
  worldid, bodyid = wp.tid()
  if bodyid == 0:
    cfrc_int_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return
  cacc = cacc_in[worldid, bodyid]
  cinert = cinert_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  frc = math.inert_vec(cinert, cacc)
  frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))
  if flg_cfrc_ext:
    frc -= cfrc_ext_in[worldid, bodyid]

  cfrc_int_out[worldid, bodyid] = frc


def _rne_cfrc(m: Model, d: Data, flg_cfrc_ext: bool = False):
  wp.launch(_cfrc, dim=[d.nworld, m.nbody], inputs=[d.cinert, d.cvel, d.cacc, d.cfrc_ext, flg_cfrc_ext], outputs=[d.cfrc_int])


@cache_kernel
def _cfrc_backward(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    # Data in:
    cfrc_int_in: wp.array2d[wp.spatial_vector],
    # In:
    body_tree_: wp.array[int],
    # Data out:
    cfrc_int_out: wp.array2d[wp.spatial_vector],
  ):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if bodyid != 0:
      wp.atomic_add(cfrc_int_out[worldid], pid, cfrc_int_in[worldid, bodyid])

  return kernel


def _rne_cfrc_backward(m: Model, d: Data):
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  cfrc_int_in = wp.empty_like(d.cfrc_int) if det_atomics else d.cfrc_int
  for body_tree in reversed(m.body_tree):
    if det_atomics:
      wp.copy(cfrc_int_in, d.cfrc_int)
    wp.launch(
      _cfrc_backward(det_atomics),
      dim=[d.nworld, body_tree.size],
      inputs=[m.body_parentid, cfrc_int_in, body_tree],
      outputs=[d.cfrc_int],
    )


@wp.kernel
def _qfrc_bias(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  cfrc_int_in: wp.array2d[wp.spatial_vector],
  # Data out:
  qfrc_bias_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  bodyid = dof_bodyid[dofid]
  qfrc_bias_out[worldid, dofid] = wp.dot(cdof_in[worldid, dofid], cfrc_int_in[worldid, bodyid])


@event_scope
def rne(m: Model, d: Data, flg_acc: bool = False):
  """Computes inverse dynamics using the recursive Newton-Euler algorithm.

  Computes the bias forces (`qfrc_bias`) and internal forces (`cfrc_int`) for the current state,
  including the effects of gravity and optionally joint accelerations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    flg_acc: If True, includes joint accelerations in the computation.
  """
  _rne_cacc_world(m, d)
  _rne_cacc_forward(m, d, flg_acc=flg_acc)
  _rne_cfrc(m, d)
  _rne_cfrc_backward(m, d)
  wp.launch(_qfrc_bias, dim=[d.nworld, m.nv], inputs=[m.dof_bodyid, d.cdof, d.cfrc_int], outputs=[d.qfrc_bias])


@wp.kernel
def _cfrc_ext(
  # Model:
  body_rootid: wp.array[int],
  # Data in:
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  cfrc_ext_out: wp.array2d[wp.spatial_vector],
):
  worldid, bodyid = wp.tid()
  if bodyid == 0:
    cfrc_ext_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  else:
    xfrc_applied = xfrc_applied_in[worldid, bodyid]
    subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
    xipos = xipos_in[worldid, bodyid]
    cfrc_ext_out[worldid, bodyid] = support.transform_force(xfrc_applied, subtree_com - xipos)


@wp.kernel
def _count_equality_constraints(
  # Model:
  eq_type: wp.array[int],
  # Data in:
  ne_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  # Out:
  ne_connect_out: wp.array[int],
  ne_weld_out: wp.array[int],
):
  """Counts connect and weld equality constraints from efc data."""
  worldid, efcid = wp.tid()

  # Only process rows within the equality constraint range
  if efcid >= ne_in[worldid]:
    return

  # Get the equality constraint ID and its type
  eq_id = efc_id_in[worldid, efcid]
  eq_constraint_type = eq_type[eq_id]

  # Count by type (each connect has 3 rows, each weld has 6 rows)
  if eq_constraint_type == EqType.CONNECT:
    wp.atomic_add(ne_connect_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
  elif eq_constraint_type == EqType.WELD:
    wp.atomic_add(ne_weld_out, worldid, 1)  # kernel_analyzer: ignore[atomic]


@cache_kernel
def _cfrc_ext_equality(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    site_bodyid: wp.array[int],
    site_pos: wp.array2d[wp.vec3],
    site_quat: wp.array2d[wp.quat],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_objtype: wp.array[int],
    eq_data: wp.array2d[vec12],
    # Data in:
    xpos_in: wp.array2d[wp.vec3],
    xquat_in: wp.array2d[wp.quat],
    xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    efc_id_in: wp.array2d[int],
    efc_force_in: wp.array2d[float],
    # In:
    ne_connect_in: wp.array[int],
    ne_weld_in: wp.array[int],
    # Data out:
    cfrc_ext_out: wp.array2d[wp.spatial_vector],
  ):
    worldid, eqid = wp.tid()

    ne_connect = ne_connect_in[worldid]
    ne_weld = ne_weld_in[worldid]
    num_connect = ne_connect // 3

    if eqid >= num_connect + ne_weld // 6:
      return

    is_connect = eqid < num_connect
    if is_connect:
      efcid = 3 * eqid
    else:
      efcid = 6 * eqid - ne_connect

    cfrc_force = wp.vec3(
      efc_force_in[worldid, efcid + 0],
      efc_force_in[worldid, efcid + 1],
      efc_force_in[worldid, efcid + 2],
    )

    id = efc_id_in[worldid, efcid]
    eq_data_ = eq_data[worldid % eq_data.shape[0], id]
    body_semantic = eq_objtype[id] == ObjType.BODY

    obj1 = eq_obj1id[id]
    obj2 = eq_obj2id[id]

    if body_semantic:
      bodyid1 = obj1
      bodyid2 = obj2
    else:
      bodyid1 = site_bodyid[obj1]
      bodyid2 = site_bodyid[obj2]

    if is_connect:
      cfrc_torque = wp.vec3(0.0, 0.0, 0.0)  # no torque from connect
    else:
      force_torque = wp.vec3(
        efc_force_in[worldid, efcid + 3],
        efc_force_in[worldid, efcid + 4],
        efc_force_in[worldid, efcid + 5],
      )
      if body_semantic:
        relpose = wp.quat(eq_data_[6], eq_data_[7], eq_data_[8], eq_data_[9])
        q0 = math.mul_quat(xquat_in[worldid, bodyid1], relpose)
        q1 = xquat_in[worldid, bodyid2]
      else:
        site_quat_id = worldid % site_quat.shape[0]
        q0 = math.mul_quat(xquat_in[worldid, bodyid1], site_quat[site_quat_id, obj1])
        q1 = math.mul_quat(xquat_in[worldid, bodyid2], site_quat[site_quat_id, obj2])

      quat = math.mul_quat(math.quat_mul_axis(q1, force_torque), math.quat_inv(q0))
      cfrc_torque = (0.5 * eq_data_[10]) * wp.vec3(quat[1], quat[2], quat[3])

    # body 1
    if bodyid1:
      if body_semantic:
        if is_connect:
          offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])
        else:
          offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])
      else:
        offset = site_pos[worldid % site_pos.shape[0], obj1]

      # transform point on body1: local -> global
      pos = xmat_in[worldid, bodyid1] @ offset + xpos_in[worldid, bodyid1]

      # subtree CoM-based torque_force vector
      newpos = subtree_com_in[worldid, body_rootid[bodyid1]]

      dif = newpos - pos
      cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)

      # apply (opposite for body 1)
      wp.atomic_add(cfrc_ext_out[worldid], bodyid1, cfrc_com)

    # body 2
    if bodyid2:
      if body_semantic:
        if is_connect:
          offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])
        else:
          offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])
      else:
        offset = site_pos[worldid % site_pos.shape[0], obj2]

      # transform point on body2: local -> global
      pos = xmat_in[worldid, bodyid2] @ offset + xpos_in[worldid, bodyid2]

      # subtree CoM-based torque_force vector
      newpos = subtree_com_in[worldid, body_rootid[bodyid2]]

      dif = newpos - pos
      cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)

      # apply
      wp.atomic_sub(cfrc_ext_out[worldid], bodyid2, cfrc_com)

  return kernel


@wp.func
def transform_force(force: wp.vec3, torque: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:
  torque -= wp.cross(offset, force)
  return wp.spatial_vector(torque, force)


@cache_kernel
def _cfrc_ext_contact(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_cone: int,
    body_rootid: wp.array[int],
    geom_bodyid: wp.array[int],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    contact_adhesion_in: wp.array[float],
    efc_force_in: wp.array2d[float],
    njmax_in: int,
    nacon_in: wp.array[int],
    # Data out:
    cfrc_ext_out: wp.array2d[wp.spatial_vector],
  ):
    contactid = wp.tid()

    if contactid >= nacon_in[0]:
      return

    geom = contact_geom_in[contactid]
    id1 = geom_bodyid[geom[0]]
    id2 = geom_bodyid[geom[1]]

    if id1 == 0 and id2 == 0:
      return

    worldid = contact_worldid_in[contactid]

    # contact force in world frame
    force = support.contact_force_fn(
      opt_cone,
      contact_frame_in,
      contact_friction_in,
      contact_dim_in,
      contact_efc_address_in,
      contact_adhesion_in,
      efc_force_in,
      njmax_in,
      nacon_in,
      worldid,
      contactid,
      to_world_frame=True,
    )

    pos = contact_pos_in[contactid]

    # contact force on bodies
    if id1:
      com1 = subtree_com_in[worldid, body_rootid[id1]]
      wp.atomic_sub(cfrc_ext_out[worldid], id1, support.transform_force(force, com1 - pos))

    if id2:
      com2 = subtree_com_in[worldid, body_rootid[id2]]
      wp.atomic_add(cfrc_ext_out[worldid], id2, support.transform_force(force, com2 - pos))

  return kernel


@cache_kernel
def _cfrc_ext_tendon_constraint(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    eq_type: wp.array[int],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_data: wp.array2d[vec12],
    tendon_range: wp.array2d[wp.vec2],
    tendon_length0: wp.array2d[float],
    # Data in:
    nefc_in: wp.array[int],
    ten_length_in: wp.array2d[float],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_force_in: wp.array2d[float],
    # Out:
    ten_frc_out: wp.array2d[float],
  ):
    worldid, efcid = wp.tid()
    if efcid >= nefc_in[worldid]:
      return

    efc_force = efc_force_in[worldid, efcid]
    if efc_force == 0.0:
      return

    efc_type = efc_type_in[worldid, efcid]

    if efc_type == ConstraintType.EQUALITY:
      eqid = efc_id_in[worldid, efcid]
      if eq_type[eqid] != EqType.TENDON:
        return
      obj1 = eq_obj1id[eqid]
      obj2 = eq_obj2id[eqid]
      wp.atomic_add(ten_frc_out[worldid], obj1, efc_force)
      if obj2 >= 0:
        dif = ten_length_in[worldid, obj2] - tendon_length0[worldid % tendon_length0.shape[0], obj2]
        eq_data_ = eq_data[worldid % eq_data.shape[0], eqid]
        deriv = eq_data_[1] + 2.0 * eq_data_[2] * dif + 3.0 * eq_data_[3] * dif * dif + 4.0 * eq_data_[4] * dif * dif * dif
        wp.atomic_sub(ten_frc_out[worldid], obj2, deriv * efc_force)
      return

    if efc_type == ConstraintType.FRICTION_TENDON:
      tenid = efc_id_in[worldid, efcid]
      wp.atomic_add(ten_frc_out[worldid], tenid, efc_force)
      return

    if efc_type == ConstraintType.LIMIT_TENDON:
      tenid = efc_id_in[worldid, efcid]
      tenrange = tendon_range[worldid % tendon_range.shape[0], tenid]
      length = ten_length_in[worldid, tenid]
      dist_min = length - tenrange[0]
      dist_max = tenrange[1] - length
      scl = wp.where(dist_min < dist_max, 1.0, -1.0)
      wp.atomic_add(ten_frc_out[worldid], tenid, scl * efc_force)

  return kernel


@cache_kernel
def _cfrc_ext_tendon_actuator(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_gear: wp.array2d[wp.spatial_vector],
    # Data in:
    actuator_force_in: wp.array2d[float],
    # Out:
    ten_frc_out: wp.array2d[float],
  ):
    worldid, actid = wp.tid()
    if actuator_trntype[actid] != TrnType.TENDON:
      return

    force = actuator_force_in[worldid, actid]
    if force == 0.0:
      return
    tenid = actuator_trnid[actid][0]
    gear = actuator_gear[worldid % actuator_gear.shape[0], actid][0]
    wp.atomic_add(ten_frc_out[worldid], tenid, force * gear)

  return kernel


@cache_kernel
def _cfrc_ext_spatial_tendon(nwrap: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 4 * nwrap

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    geom_bodyid: wp.array[int],
    site_bodyid: wp.array[int],
    tendon_type: wp.array[int],
    tendon_adr: wp.array[int],
    tendon_num: wp.array[int],
    tendon_actuatorid: wp.array[int],
    tendon_plasticadr: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_stiffness: wp.array2d[float],
    tendon_stiffnesspoly: wp.array2d[wp.vec2],
    tendon_damping: wp.array2d[float],
    tendon_dampingpoly: wp.array2d[wp.vec2],
    tendon_armature: wp.array2d[float],
    tendon_lengthspring: wp.array2d[wp.vec2],
    wrap_type: wp.array[int],
    wrap_objid: wp.array[int],
    wrap_prm: wp.array[float],
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_damping: wp.array2d[float],
    actuator_dampingpoly: wp.array2d[wp.vec2],
    actuator_armature: wp.array2d[float],
    actuator_gear: wp.array2d[wp.spatial_vector],
    # Data in:
    plastic_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    ten_wrapadr_in: wp.array2d[int],
    ten_wrapnum_in: wp.array2d[int],
    ten_J_in: wp.array2d[float],
    ten_length_in: wp.array2d[float],
    wrap_obj_in: wp.array2d[wp.vec2i],
    wrap_xpos_in: wp.array2d[wp.spatial_vector],
    ten_velocity_in: wp.array2d[float],
    # In:
    ten_bias_coef_in: wp.array2d[float],
    dsbl_spring: bool,
    dsbl_damper: bool,
    ten_frc_in: wp.array2d[float],
    # Data out:
    cfrc_ext_out: wp.array2d[wp.spatial_vector],
  ):
    worldid, tenid = wp.tid()

    # fixed tendon: acts through the joints
    if tendon_type[tenid] == TendonType.FIXED:
      return

    frc = ten_frc_in[worldid, tenid]

    # ten_frc += spring and damper
    if not dsbl_spring:
      stiffness = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]
      spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]
      if stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0:
        length = ten_length_in[worldid, tenid]
        padr = tendon_plasticadr[tenid]
        if padr >= 0:
          lower = plastic_in[worldid, padr + 0]
          upper = plastic_in[worldid, padr + 1]
        else:
          lengthspring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]
          lower = lengthspring[0]
          upper = lengthspring[1]
        x = wp.where(length > upper, length - upper, wp.where(length < lower, length - lower, 0.0))
        frc += -x * util_misc._poly_force(stiffness, spoly, x, 0)

    if not dsbl_damper:
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
      if damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0:
        v = ten_velocity_in[worldid, tenid]
        frc += -v * util_misc._poly_force(damping, dpoly, v, 1)

    # ten_frc -= armature * tendon acceleration: reaction of the armature inertia
    armature = util_misc.tendon_armature_total(
      tendon_actuatorid, tendon_armature, actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, tenid
    )
    if armature != 0.0:
      acc = ten_bias_coef_in[worldid, tenid]
      rownnz = ten_J_rownnz[tenid]
      rowadr = ten_J_rowadr[tenid]
      for i in range(rownnz):
        sparseid = rowadr + i
        acc += ten_J_in[worldid, sparseid] * qacc_in[worldid, ten_J_colind[sparseid]]
      frc -= armature * acc

    if frc == 0.0:
      return

    adr = tendon_adr[tenid]
    num = tendon_num[tenid]
    p = ten_wrapadr_in[worldid, tenid]
    pend = p + ten_wrapnum_in[worldid, tenid]
    scaled_frc = frc
    prevbody = int(-1)
    prevpnt = wp.vec3(0.0)
    prevcom = wp.vec3(0.0)

    for j_offset in range(num):
      j = adr + j_offset
      wtype = wrap_type[j]
      objid = wrap_objid[j]

      # pulley: divides the force in the next branch, skip its marker in the path
      if wtype == WrapType.PULLEY:
        scaled_frc = frc / wrap_prm[j]
        prevbody = -1
        p += 1
        continue

      # site: one point; geom: two points if the tendon wraps around it
      npnt = int(0)
      body = int(0)
      if wtype == WrapType.SITE:
        npnt = 1
        body = site_bodyid[objid]
      else:
        body = geom_bodyid[objid]
        if p < pend and wrap_obj_in[worldid, p // 2][p % 2] == objid:
          npnt = 2

      com = wp.vec3(0.0)
      if npnt > 0 and body != 0:
        com = subtree_com_in[worldid, body_rootid[body]]

      for k in range(2):
        if k >= npnt:
          continue

        wrap_pos = wrap_xpos_in[worldid, p // 2]
        pnt = wp.where(p % 2 == 0, wp.spatial_top(wrap_pos), wp.spatial_bottom(wrap_pos))
        if prevbody >= 0 and body != prevbody:
          diff = pnt - prevpnt
          vec, norm = math.normalize_with_norm(diff)
          if norm > 0.0:
            force = vec * scaled_frc
            if body != 0:
              wp.atomic_add(cfrc_ext_out[worldid], body, wp.spatial_vector(-wp.cross(com - pnt, force), force))
            if prevbody != 0:
              wp.atomic_sub(
                cfrc_ext_out[worldid],
                prevbody,
                wp.spatial_vector(-wp.cross(prevcom - prevpnt, force), force),
              )
        prevbody = body
        prevpnt = pnt
        prevcom = com
        p += 1

  return kernel


@event_scope
def rne_postconstraint(m: Model, d: Data):
  """Computes the recursive Newton-Euler algorithm after constraints are applied.

  Computes `cacc`, `cfrc_ext`, and `cfrc_int`, including the effects of applied forces, equality
  constraints, contacts, and spatial tendons.
  """
  # cfrc_ext = perturb
  wp.launch(
    _cfrc_ext,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, d.xfrc_applied, d.xipos, d.subtree_com],
    outputs=[d.cfrc_ext],
  )

  # Equality constraint forces - only if model has equality constraints
  if m.neq > 0:
    # Allocate inline counters and count from efc data
    ne_connect = wp.zeros((d.nworld,), dtype=int)
    ne_weld = wp.zeros((d.nworld,), dtype=int)

    wp.launch(
      _count_equality_constraints,
      dim=(d.nworld, d.njmax),  # TODO(team): launch over max equality constraints
      inputs=[m.eq_type, d.ne, d.efc.type, d.efc.id],
      outputs=[ne_connect, ne_weld],
    )

    wp.launch(
      _cfrc_ext_equality(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, m.neq),
      inputs=[
        m.body_rootid,
        m.site_bodyid,
        m.site_pos,
        m.site_quat,
        m.eq_obj1id,
        m.eq_obj2id,
        m.eq_objtype,
        m.eq_data,
        d.xpos,
        d.xquat,
        d.xmat,
        d.subtree_com,
        d.efc.id,
        d.efc.force,
        ne_connect,
        ne_weld,
      ],
      outputs=[d.cfrc_ext],
    )

  # cfrc_ext += contacts
  wp.launch(
    _cfrc_ext_contact(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.naconmax,),
    inputs=[
      m.opt.cone,
      m.body_rootid,
      m.geom_bodyid,
      d.subtree_com,
      d.contact.pos,
      d.contact.frame,
      d.contact.friction,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
      d.contact.adhesion,
      d.efc.force,
      d.njmax,
      d.nacon,
    ],
    outputs=[d.cfrc_ext],
  )

  # cfrc_ext += spatial tendons
  if m.ntendon > 0:
    ten_frc = wp.zeros((d.nworld, m.ntendon), dtype=float)

    wp.launch(
      _cfrc_ext_tendon_constraint(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, d.njmax),
      inputs=[
        m.eq_type,
        m.eq_obj1id,
        m.eq_obj2id,
        m.eq_data,
        m.tendon_range,
        m.tendon_length0,
        d.nefc,
        d.ten_length,
        d.efc.type,
        d.efc.id,
        d.efc.force,
      ],
      outputs=[ten_frc],
    )

    wp.launch(
      _cfrc_ext_tendon_actuator(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
      dim=(d.nworld, m.nu),
      inputs=[
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_gear,
        d.actuator_force,
      ],
      outputs=[ten_frc],
    )

    ten_bias_coef = wp.zeros((d.nworld, m.ntendon), dtype=float)
    if m.nJten > 0:
      ten_Jdot = wp.zeros((d.nworld, m.nJten), dtype=float)
      wp.launch(
        _tendon_dot,
        dim=(d.nworld, m.ntendon),
        inputs=[
          m.body_parentid,
          m.body_rootid,
          m.body_dofnum,
          m.body_dofadr,
          m.jnt_type,
          m.jnt_dofadr,
          m.dof_jntid,
          m.site_bodyid,
          m.tendon_type,
          m.tendon_adr,
          m.tendon_num,
          m.tendon_actuatorid,
          m.ten_J_rownnz,
          m.ten_J_rowadr,
          m.ten_J_colind,
          m.tendon_armature,
          m.wrap_type,
          m.wrap_objid,
          m.wrap_prm,
          m.actuator_trntype,
          m.actuator_trnid,
          m.actuator_armature,
          m.actuator_gear,
          d.site_xpos,
          d.subtree_com,
          d.cdof,
          d.cvel,
          d.cdof_dot,
        ],
        outputs=[ten_Jdot],
      )

      wp.launch(
        _tendon_bias_coef(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
        dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
        inputs=[
          m.tendon_actuatorid,
          m.ten_J_rownnz,
          m.ten_J_rowadr,
          m.ten_J_colind,
          m.tendon_armature,
          m.actuator_trntype,
          m.actuator_trnid,
          m.actuator_armature,
          m.actuator_gear,
          d.qvel,
          ten_Jdot,
        ],
        outputs=[ten_bias_coef],
      )

    dsbl_spring = bool(m.opt.disableflags & DisableBit.SPRING)
    dsbl_damper = bool(m.opt.disableflags & DisableBit.DAMPER)

    det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
    wp.launch(
      _cfrc_ext_spatial_tendon(m.nwrap if det_atomics else 0, det_atomics),
      dim=(d.nworld, m.ntendon),
      inputs=[
        m.body_rootid,
        m.geom_bodyid,
        m.site_bodyid,
        m.tendon_type,
        m.tendon_adr,
        m.tendon_num,
        m.tendon_actuatorid,
        m.tendon_plasticadr,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_stiffness,
        m.tendon_stiffnesspoly,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.tendon_armature,
        m.tendon_lengthspring,
        m.wrap_type,
        m.wrap_objid,
        m.wrap_prm,
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_damping,
        m.actuator_dampingpoly,
        m.actuator_armature,
        m.actuator_gear,
        d.plastic,
        d.qacc,
        d.subtree_com,
        d.ten_wrapadr,
        d.ten_wrapnum,
        d.ten_J,
        d.ten_length,
        d.wrap_obj,
        d.wrap_xpos,
        d.ten_velocity,
        ten_bias_coef,
        dsbl_spring,
        dsbl_damper,
        ten_frc,
      ],
      outputs=[d.cfrc_ext],
    )

  # forward pass over bodies: compute cacc, cfrc_int
  _rne_cacc_world(m, d)
  _rne_cacc_forward(m, d, flg_acc=True)

  # cfrc_body = cinert * cacc + cvel x (cinert * cvel)
  _rne_cfrc(m, d, flg_cfrc_ext=True)

  # backward pass over bodies: accumulate cfrc_int from children
  _rne_cfrc_backward(m, d)


@wp.func
def _accumulate_jac_dot_chain(
  # Model:
  body_parentid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_jntid: wp.array[int],
  ten_J_colind: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  offset: wp.vec3,
  pvel_lin: wp.vec3,
  dpnt: wp.vec3,
  dvel: wp.vec3,
  bodyid: int,
  rowadr: int,
  rownnz: int,
  scale: float,
  worldid: int,
  # Out:
  ten_Jdot_out: wp.array2d[float],
):
  """Walk body chain from bodyid to root, accumulate Jdot contributions."""
  ptr = rownnz - 1
  bid = bodyid
  while bid > 0:
    bdofadr = body_dofadr[bid]
    bdofnum = body_dofnum[bid]
    # iterate DOFs in this body in descending order
    for k_rev in range(bdofnum):
      dof = bdofadr + bdofnum - 1 - k_rev
      # scan pointer backward to find matching colind entry
      while ptr >= 0:
        sparseid = rowadr + ptr
        if ten_J_colind[sparseid] <= dof:
          break
        ptr -= 1
      if ptr >= 0 and ten_J_colind[sparseid] == dof:
        cdof = cdof_in[worldid, dof]
        cdof_ang = wp.spatial_top(cdof)
        cdof_lin = wp.spatial_bottom(cdof)
        cdof_dot = cdof_dot_in[worldid, dof]

        # quaternion override: use cvel of DOF's body (which is bid)
        dofjntid = dof_jntid[dof]
        jnttype = jnt_type[dofjntid]
        jntdofadr = jnt_dofadr[dofjntid]
        if (jnttype == JointType.BALL) or ((jnttype == JointType.FREE) and dof >= jntdofadr + 3):
          cdof_dot = math.motion_cross(cvel_in[worldid, bid], cdof)

        cdof_dot_ang = wp.spatial_top(cdof_dot)
        cdof_dot_lin = wp.spatial_bottom(cdof_dot)

        # jacp_dot (from jac_dot_dof)
        jacp_dot = cdof_dot_lin + wp.cross(cdof_dot_ang, offset) + wp.cross(cdof_ang, pvel_lin)

        # jacp (from jac_dof)
        jacp = cdof_lin + wp.cross(cdof_ang, offset)

        # combined: dot(jacdot, dpnt) + dot(jac, dvel)
        Jdot = (wp.dot(jacp_dot, dpnt) + wp.dot(jacp, dvel)) * scale
        if Jdot != 0.0:
          ten_Jdot_out[worldid, sparseid] += Jdot
    bid = body_parentid[bid]


@wp.kernel
def _tendon_dot(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_jntid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_type: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  tendon_actuatorid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_armature: wp.array2d[float],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  wrap_prm: wp.array[float],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array2d[float],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # Out:
  ten_Jdot_out: wp.array2d[float],
):
  worldid, tenid = wp.tid()

  armature = util_misc.tendon_armature_total(
    tendon_actuatorid, tendon_armature, actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, tenid
  )
  if armature == 0.0:
    return

  # fixed tendon has zero Jdot
  if tendon_type[tenid] == TendonType.FIXED:
    return

  # process spatial tendon
  adr = tendon_adr[tenid]
  divisor = float(1.0)
  num = tendon_num[tenid]
  j = int(0)
  while j < num - 1:
    # get 1st and 2nd object
    type0 = wrap_type[adr + j + 0]
    type1 = wrap_type[adr + j + 1]
    id0 = wrap_objid[adr + j + 0]
    id1 = wrap_objid[adr + j + 1]

    # pulley
    pulley = WrapType.PULLEY
    if (type0 == pulley) or (type1 == pulley):
      # get divisor
      if type0 == pulley:
        divisor = wrap_prm[adr + j]

      j += 1
      continue

    # init sequence; assume it start with site
    wpnt0 = site_xpos_in[worldid, id0]

    wbody0 = site_bodyid[id0]
    cvel0 = cvel_in[worldid, wbody0]
    subtree_com0 = subtree_com_in[worldid, body_rootid[wbody0]]
    offset0 = wpnt0 - subtree_com0
    pvel_lin0 = wp.spatial_bottom(cvel0) - wp.cross(offset0, wp.spatial_top(cvel0))

    # second object is geom: process site-geom-site
    if (type1 == WrapType.SPHERE) or (type1 == WrapType.CYLINDER):
      # TODO(team): derivatives of util_misc.wrap
      return

    # complete sequence
    wbody1 = site_bodyid[id1]
    wpnt1 = site_xpos_in[worldid, id1]

    cvel1 = cvel_in[worldid, wbody1]
    subtree_com1 = subtree_com_in[worldid, body_rootid[wbody1]]
    offset1 = wpnt1 - subtree_com1
    pvel_lin1 = wp.spatial_bottom(cvel1) - wp.cross(offset1, wp.spatial_top(cvel1))

    # accumulate moments if consecutive points are in different bodies
    if wbody0 != wbody1:
      # dpnt = 3D position difference, normalize
      dpnt, norm = math.normalize_with_norm(wpnt1 - wpnt0)

      # dvel = d / dt (dpnt)
      wvel0 = wp.spatial_bottom(cvel0) - wp.cross(wpnt0 - subtree_com0, wp.spatial_top(cvel0))
      wvel1 = wp.spatial_bottom(cvel1) - wp.cross(wpnt1 - subtree_com1, wp.spatial_top(cvel1))
      dvel = wvel1 - wvel0
      dot = wp.dot(dpnt, dvel)
      dvel += dpnt * (-dot)
      if norm > MJ_MINVAL:
        dvel /= norm
      else:
        dvel = wp.vec3(0.0)

      rownnz = ten_J_rownnz[tenid]
      rowadr = ten_J_rowadr[tenid]
      inv_divisor = math.safe_div(float(1.0), divisor)

      # body0 contributes with negative sign, body1 with positive
      _accumulate_jac_dot_chain(
        body_parentid,
        body_dofnum,
        body_dofadr,
        jnt_type,
        jnt_dofadr,
        dof_jntid,
        ten_J_colind,
        cdof_in,
        cvel_in,
        cdof_dot_in,
        offset0,
        pvel_lin0,
        dpnt,
        dvel,
        wbody0,
        rowadr,
        rownnz,
        -inv_divisor,
        worldid,
        ten_Jdot_out,
      )
      _accumulate_jac_dot_chain(
        body_parentid,
        body_dofnum,
        body_dofadr,
        jnt_type,
        jnt_dofadr,
        dof_jntid,
        ten_J_colind,
        cdof_in,
        cvel_in,
        cdof_dot_in,
        offset1,
        pvel_lin1,
        dpnt,
        dvel,
        wbody1,
        rowadr,
        rownnz,
        inv_divisor,
        worldid,
        ten_Jdot_out,
      )

    # TODO(team): j += 2 if geom wrapping
    j += 1


@cache_kernel
def _tendon_bias_coef(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    tendon_actuatorid: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_armature: wp.array2d[float],
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_armature: wp.array2d[float],
    actuator_gear: wp.array2d[wp.spatial_vector],
    # Data in:
    qvel_in: wp.array2d[float],
    # In:
    ten_Jdot_in: wp.array2d[float],
    # Out:
    ten_bias_coef_out: wp.array2d[float],
  ):
    worldid, tenid, dofid_sparse = wp.tid()

    armature = util_misc.tendon_armature_total(
      tendon_actuatorid, tendon_armature, actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, tenid
    )
    if armature == 0.0:
      return

    rownnz = ten_J_rownnz[tenid]
    if dofid_sparse >= rownnz:
      return
    rowadr = ten_J_rowadr[tenid]
    sparseid = rowadr + dofid_sparse
    ten_Jdot = ten_Jdot_in[worldid, sparseid]
    if ten_Jdot == 0.0:
      return

    dofid = ten_J_colind[sparseid]
    wp.atomic_add(ten_bias_coef_out[worldid], tenid, ten_Jdot * qvel_in[worldid, dofid])

  return kernel


@cache_kernel
def _tendon_bias_qfrc(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    tendon_actuatorid: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    tendon_armature: wp.array2d[float],
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_armature: wp.array2d[float],
    actuator_gear: wp.array2d[wp.spatial_vector],
    # Data in:
    ten_J_in: wp.array2d[float],
    # In:
    ten_bias_coef_in: wp.array2d[float],
    # Out:
    qfrc_out: wp.array2d[float],
  ):
    worldid, tenid, dofid = wp.tid()

    armature = util_misc.tendon_armature_total(
      tendon_actuatorid, tendon_armature, actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, tenid
    )
    if armature == 0.0:
      return

    rownnz = ten_J_rownnz[tenid]
    if dofid >= rownnz:
      return
    rowadr = ten_J_rowadr[tenid]
    sparseid = rowadr + dofid
    ten_J = ten_J_in[worldid, sparseid]

    if ten_J == 0.0:
      return

    dofid = ten_J_colind[sparseid]

    wp.atomic_add(qfrc_out[worldid], dofid, ten_J * armature * ten_bias_coef_in[worldid, tenid])

  return kernel


@event_scope
def tendon_bias(m: Model, d: Data, qfrc: wp.array2d[float]):
  """Add bias force due to tendon armature.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    qfrc: Force.
  """
  # time derivative of tendon Jacobian
  ten_Jdot = wp.zeros((d.nworld, m.nJten), dtype=float)
  wp.launch(
    _tendon_dot,
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.body_parentid,
      m.body_rootid,
      m.body_dofnum,
      m.body_dofadr,
      m.jnt_type,
      m.jnt_dofadr,
      m.dof_jntid,
      m.site_bodyid,
      m.tendon_type,
      m.tendon_adr,
      m.tendon_num,
      m.tendon_actuatorid,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.tendon_armature,
      m.wrap_type,
      m.wrap_objid,
      m.wrap_prm,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_armature,
      m.actuator_gear,
      d.site_xpos,
      d.subtree_com,
      d.cdof,
      d.cvel,
      d.cdof_dot,
    ],
    outputs=[ten_Jdot],
  )

  # tendon bias force coefficients
  ten_bias_coef = wp.zeros((d.nworld, m.ntendon), dtype=float)
  wp.launch(
    _tendon_bias_coef(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
    inputs=[
      m.tendon_actuatorid,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.tendon_armature,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_armature,
      m.actuator_gear,
      d.qvel,
      ten_Jdot,
    ],
    outputs=[ten_bias_coef],
  )

  wp.launch(
    _tendon_bias_qfrc(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
    inputs=[
      m.tendon_actuatorid,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.tendon_armature,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_armature,
      m.actuator_gear,
      d.ten_J,
      ten_bias_coef,
    ],
    outputs=[qfrc],
  )


@wp.kernel
def _comvel_root(cvel_out: wp.array2d[wp.spatial_vector]):
  worldid, elementid = wp.tid()
  cvel_out[worldid, 0][elementid] = 0.0


@wp.kernel
def _comvel_branch(
  # Model:
  body_parentid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_dofadr: wp.array[int],
  jnt_type: wp.array[int],
  body_branches: wp.array[int],
  body_branch_start: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # Data out:
  cvel_out: wp.array2d[wp.spatial_vector],
  cdof_dot_out: wp.array2d[wp.spatial_vector],
):
  worldid, branchid = wp.tid()

  start = body_branch_start[branchid]
  end = body_branch_start[branchid + 1]

  qvel = qvel_in[worldid]
  cdof = cdof_in[worldid]

  for i in range(start, end):
    bodyid = body_branches[i]
    pid = body_parentid[bodyid]
    cvel = cvel_out[worldid, pid]
    dofid = body_dofadr[bodyid]
    jntid = body_jntadr[bodyid]
    jntnum = body_jntnum[bodyid]

    if jntnum == 0:
      cvel_out[worldid, bodyid] = cvel
      continue

    for j in range(jntid, jntid + jntnum):
      jnttype = jnt_type[j]

      if jnttype == JointType.FREE:
        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        cdof_dot_out[worldid, dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cdof_dot_out[worldid, dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cdof_dot_out[worldid, dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])
        cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])
        cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])

        cvel += cdof[dofid + 3] * qvel[dofid + 3]
        cvel += cdof[dofid + 4] * qvel[dofid + 4]
        cvel += cdof[dofid + 5] * qvel[dofid + 5]

        dofid += 6
      elif jnttype == JointType.BALL:
        cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])
        cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])
        cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])

        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        dofid += 3
      else:
        cdof_dot_out[worldid, dofid] = math.motion_cross(cvel, cdof[dofid])
        cvel += cdof[dofid] * qvel[dofid]

        dofid += 1

    cvel_out[worldid, bodyid] = cvel


@event_scope
def com_vel(m: Model, d: Data):
  """Computes the spatial velocities (cvel) and the derivative `cdof_dot` for all bodies.

  Propagates velocities down the kinematic tree, updating the spatial velocity and
  derivative for each body.
  """
  wp.launch(_comvel_root, dim=(d.nworld, 6), inputs=[], outputs=[d.cvel])

  wp.launch(
    _comvel_branch,
    dim=(d.nworld, m.nbranch),
    inputs=[
      m.body_parentid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_dofadr,
      m.jnt_type,
      m.body_branches,
      m.body_branch_start,
      d.qvel,
      d.cdof,
    ],
    outputs=[d.cvel, d.cdof_dot],
  )


@cache_kernel
def _transmission(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    nv: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_parentid: wp.array[int],
    site_bodyid: wp.array[int],
    site_quat: wp.array2d[wp.quat],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    actuator_trntype: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_cranklength: wp.array2d[float],
    actuator_gear: wp.array2d[wp.spatial_vector],
    body_isdofancestor: wp.array2d[int],
    # Data in:
    qpos_in: wp.array2d[float],
    xquat_in: wp.array2d[wp.quat],
    site_xpos_in: wp.array2d[wp.vec3],
    site_xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    ten_J_in: wp.array2d[float],
    ten_length_in: wp.array2d[float],
    # In:
    moment_nnz: wp.array[int],
    # Data out:
    actuator_length_out: wp.array2d[float],
    moment_rownnz_out: wp.array2d[int],
    moment_rowadr_out: wp.array2d[int],
    moment_colind_out: wp.array2d[int],
    actuator_moment_out: wp.array2d[float],
  ):
    worldid, actid = wp.tid()
    trntype = actuator_trntype[actid]
    actuator_gear_id = worldid % actuator_gear.shape[0]
    gear = actuator_gear[actuator_gear_id, actid]
    if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:
      qpos = qpos_in[worldid]
      jntid = actuator_trnid[actid][0]
      jnt_typ = jnt_type[jntid]
      qadr = jnt_qposadr[jntid]
      vadr = jnt_dofadr[jntid]
      if jnt_typ == JointType.FREE:
        moment_rownnz_out[worldid, actid] = 6
        rowadr = wp.atomic_add(moment_nnz, worldid, 6)
        moment_rowadr_out[worldid, actid] = rowadr
        moment_colind_out[worldid, rowadr + 0] = vadr + 0
        moment_colind_out[worldid, rowadr + 1] = vadr + 1
        moment_colind_out[worldid, rowadr + 2] = vadr + 2
        moment_colind_out[worldid, rowadr + 3] = vadr + 3
        moment_colind_out[worldid, rowadr + 4] = vadr + 4
        moment_colind_out[worldid, rowadr + 5] = vadr + 5
        actuator_length_out[worldid, actid] = 0.0
        if trntype == TrnType.JOINTINPARENT:
          quat = wp.normalize(wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6]))
          quat_neg = math.quat_inv(quat)
          gearaxis = math.rot_vec_quat(wp.spatial_bottom(gear), quat_neg)
          actuator_moment_out[worldid, rowadr + 0] = gear[0]
          actuator_moment_out[worldid, rowadr + 1] = gear[1]
          actuator_moment_out[worldid, rowadr + 2] = gear[2]
          actuator_moment_out[worldid, rowadr + 3] = gearaxis[0]
          actuator_moment_out[worldid, rowadr + 4] = gearaxis[1]
          actuator_moment_out[worldid, rowadr + 5] = gearaxis[2]
        else:
          actuator_moment_out[worldid, rowadr + 0] = gear[0]
          actuator_moment_out[worldid, rowadr + 1] = gear[1]
          actuator_moment_out[worldid, rowadr + 2] = gear[2]
          actuator_moment_out[worldid, rowadr + 3] = gear[3]
          actuator_moment_out[worldid, rowadr + 4] = gear[4]
          actuator_moment_out[worldid, rowadr + 5] = gear[5]
      elif jnt_typ == JointType.BALL:
        q = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        q = wp.normalize(q)
        axis_angle = math.quat_to_vel(q)
        gearaxis = wp.spatial_top(gear)  # [:3]
        if trntype == TrnType.JOINTINPARENT:
          quat_neg = math.quat_inv(q)
          gearaxis = math.rot_vec_quat(gearaxis, quat_neg)
        actuator_length_out[worldid, actid] = wp.dot(axis_angle, gearaxis)

        nnz = 3
        moment_rownnz_out[worldid, actid] = nnz
        rowadr = wp.atomic_add(moment_nnz, worldid, nnz)
        moment_rowadr_out[worldid, actid] = rowadr

        for i in range(3):
          sparseid = rowadr + i
          moment_colind_out[worldid, sparseid] = vadr + i
          actuator_moment_out[worldid, sparseid] = gearaxis[i]
      elif jnt_typ == JointType.SLIDE or jnt_typ == JointType.HINGE:
        actuator_length_out[worldid, actid] = qpos[qadr] * gear[0]

        nnz = 1
        moment_rownnz_out[worldid, actid] = nnz
        rowadr = wp.atomic_add(moment_nnz, worldid, nnz)
        moment_rowadr_out[worldid, actid] = rowadr
        moment_colind_out[worldid, rowadr] = vadr
        actuator_moment_out[worldid, rowadr] = gear[0]
      else:
        wp.printf("unrecognized joint type")
    elif trntype == TrnType.SLIDERCRANK:
      # get data
      trnid = actuator_trnid[actid]
      id = trnid[0]
      idslider = trnid[1]
      gear0 = gear[0]
      rod = actuator_cranklength[worldid % actuator_cranklength.shape[0], actid]
      site_xmat = site_xmat_in[worldid, idslider]
      axis = wp.vec3(site_xmat[0, 2], site_xmat[1, 2], site_xmat[2, 2])
      site_xpos_id = site_xpos_in[worldid, id]
      site_xpos_idslider = site_xpos_in[worldid, idslider]
      vec = site_xpos_id - site_xpos_idslider

      # compute length and determinant
      #  length = a' * v - sqrt(det);  det = (a' * v)^2 + r^2 - v' * v
      av = wp.dot(vec, axis)
      det = av * av + rod * rod - wp.dot(vec, vec)
      ok = 1
      if det <= 0.0:
        ok = 0
        sdet = 0.0
        length = av
      else:
        sdet = wp.sqrt(det)
        length = av - sdet

      actuator_length_out[worldid, actid] = length * gear0

      # compute derivatives of length w.r.t. vec and axis
      if ok == 1:
        scale = 1.0 - math.safe_div(av, sdet)
        dldv = axis * scale + math.safe_div(vec, sdet)
        dlda = vec * scale
      else:
        dldv = axis
        dlda = vec

      # count dofs
      b1 = body_weldid[site_bodyid[id]]
      b2 = body_weldid[site_bodyid[idslider]]
      da1_init = int(-1)
      da2_init = int(-1)
      if b1 > 0:
        da1_init = body_dofadr[b1] + body_dofnum[b1] - 1
      if b2 > 0:
        da2_init = body_dofadr[b2] + body_dofnum[b2] - 1

      da1 = da1_init
      da2 = da2_init
      ndof = int(0)
      while da1 >= 0 or da2 >= 0:
        da = wp.max(da1, da2)
        ndof += 1
        if da1 == da:
          da1 = dof_parentid[da1]
        if da2 == da:
          da2 = dof_parentid[da2]

      moment_rownnz_out[worldid, actid] = ndof
      rowadr = wp.atomic_add(moment_nnz, worldid, ndof)
      moment_rowadr_out[worldid, actid] = rowadr

      # traverse dofs
      da1 = da1_init
      da2 = da2_init

      ptr = ndof - 1
      while da1 >= 0 or da2 >= 0:
        da = wp.max(da1, da2)

        # get Jacobians of axis(jacA) and vec(jac)
        jacp, jacr = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          site_xpos_idslider,
          site_bodyid[idslider],
          da,
          worldid,
        )
        jacS = jacp
        jacA = wp.cross(jacr, axis)
        jac, _ = support.jac_dof(
          body_parentid,
          body_rootid,
          dof_bodyid,
          body_isdofancestor,
          subtree_com_in,
          cdof_in,
          site_xpos_id,
          site_bodyid[id],
          da,
          worldid,
        )
        jac -= jacS

        # apply the chain rule
        moment = wp.dot(dlda, jacA) + wp.dot(dldv, jac)
        sparseid = rowadr + ptr
        moment_colind_out[worldid, sparseid] = da
        actuator_moment_out[worldid, sparseid] = moment * gear0
        ptr -= 1

        if da1 == da:
          da1 = dof_parentid[da1]
        if da2 == da:
          da2 = dof_parentid[da2]
    elif trntype == TrnType.TENDON:
      tenid = actuator_trnid[actid][0]

      gear0 = gear[0]
      actuator_length_out[worldid, actid] = ten_length_in[worldid, tenid] * gear0

      rownnz_ten = ten_J_rownnz[tenid]
      rowadr_ten = ten_J_rowadr[tenid]

      rowadr_mom = wp.atomic_add(moment_nnz, worldid, rownnz_ten)
      moment_rownnz_out[worldid, actid] = rownnz_ten
      moment_rowadr_out[worldid, actid] = rowadr_mom

      for k in range(rownnz_ten):
        sparseid_ten = rowadr_ten + k
        sparseid_mom = rowadr_mom + k
        moment_colind_out[worldid, sparseid_mom] = ten_J_colind[sparseid_ten]
        actuator_moment_out[worldid, sparseid_mom] = ten_J_in[worldid, sparseid_ten] * gear0
    elif trntype == TrnType.BODY:
      # cannot compute meaningful length, set to zero
      actuator_length_out[worldid, actid] = 0.0

      # initialize moment
      rowadr = wp.atomic_add(moment_nnz, worldid, nv)
      moment_rownnz_out[worldid, actid] = nv
      moment_rowadr_out[worldid, actid] = rowadr
      for i in range(nv):
        sparseid = rowadr + i
        moment_colind_out[worldid, sparseid] = i
        actuator_moment_out[worldid, sparseid] = 0.0

      # moment computed by _transmission_body_moment and _transmission_body_moment_scale
    elif trntype == TrnType.SITE:
      trnid = actuator_trnid[actid]
      siteid = trnid[0]
      refid = trnid[1]

      gear = actuator_gear[actuator_gear_id, actid]
      site_quat_id = worldid % site_quat.shape[0]
      gear_translation = wp.spatial_top(gear)
      gear_rotational = wp.spatial_bottom(gear)

      # reference site undefined
      if refid == -1:
        # wrench: gear expressed in global frame
        site_xmat = site_xmat_in[worldid, siteid]
        wrench_translation = site_xmat @ gear_translation
        wrench_rotation = site_xmat @ gear_rotational

        # count dofs
        b1 = body_weldid[site_bodyid[siteid]]
        da_init = int(-1)
        if b1 > 0:
          da_init = body_dofadr[b1] + body_dofnum[b1] - 1

        da = da_init
        ndof = int(0)
        while da >= 0:
          ndof += 1
          da = dof_parentid[da]

        moment_rownnz_out[worldid, actid] = ndof
        rowadr = wp.atomic_add(moment_nnz, worldid, ndof)
        moment_rowadr_out[worldid, actid] = rowadr
        actuator_length_out[worldid, actid] = 0.0

        # traverse dofs
        da = da_init
        ptr = ndof - 1
        while da >= 0:
          jacp, jacr = support.jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com_in,
            cdof_in,
            site_xpos_in[worldid, siteid],
            site_bodyid[siteid],
            da,
            worldid,
          )
          moment = wp.dot(jacp, wrench_translation) + wp.dot(jacr, wrench_rotation)
          sparseid = rowadr + ptr
          moment_colind_out[worldid, sparseid] = da
          actuator_moment_out[worldid, sparseid] = moment
          ptr -= 1
          da = dof_parentid[da]
      # reference site defined
      else:
        # initialize last dof address for each body
        bodyid = site_bodyid[siteid]
        bodyrefid = site_bodyid[refid]
        b0 = body_weldid[bodyid]
        b1 = body_weldid[bodyrefid]
        dofadr0 = body_dofadr[b0] + body_dofnum[b0] - 1
        dofadr1 = body_dofadr[b1] + body_dofnum[b1] - 1

        # find common ancestral dof, if any
        dofadr_common = -1
        if dofadr0 >= 0 and dofadr1 >= 0:
          # traverse up the tree until common ancestral dof is found
          while dofadr0 != dofadr1:
            if dofadr0 < dofadr1:
              dofadr1 = dof_parentid[dofadr1]
            else:
              dofadr0 = dof_parentid[dofadr0]

            if dofadr0 == -1 or dofadr1 == -1:
              # reached tree root, no common ancestral dof
              break

          # found common ancestral dof
          if dofadr0 == dofadr1:
            dofadr_common = dofadr0

        translational_transmission = not (gear[0] == 0.0 and gear[1] == 0.0 and gear[2] == 0.0)
        rotational_transmission = not (gear[3] == 0.0 and gear[4] == 0.0 and gear[5] == 0.0)

        site_xpos = site_xpos_in[worldid, siteid]
        ref_xpos = site_xpos_in[worldid, refid]
        ref_xmat = site_xmat_in[worldid, refid]

        length = float(0.0)

        if translational_transmission:
          # vec: site position in reference site frame
          vec = wp.transpose(ref_xmat) @ (site_xpos - ref_xpos)
          length += wp.dot(vec, gear_translation)

          wrench_translation = ref_xmat @ gear_translation

        if rotational_transmission:
          # get site and refsite quats from parent bodies (avoid converting matrix to quat)
          quat = math.mul_quat(xquat_in[worldid, bodyid], site_quat[site_quat_id, siteid])
          refquat = math.mul_quat(xquat_in[worldid, bodyrefid], site_quat[site_quat_id, refid])

          # convert difference to expmap (axis-angle)
          vec = math.quat_sub(quat, refquat)
          length += wp.dot(vec, gear_rotational)

          wrench_rotation = ref_xmat @ gear_rotational

        actuator_length_out[worldid, actid] = length

        # count dofs
        da1_init = int(-1)
        da2_init = int(-1)
        if b0 > 0:
          da1_init = body_dofadr[b0] + body_dofnum[b0] - 1
        if b1 > 0:
          da2_init = body_dofadr[b1] + body_dofnum[b1] - 1

        da1 = da1_init
        da2 = da2_init
        ndof = int(0)
        while da1 >= 0 or da2 >= 0:
          da = wp.max(da1, da2)
          if da1 == da and da2 == da:
            break
          ndof += 1
          if da1 == da:
            da1 = dof_parentid[da1]
          if da2 == da:
            da2 = dof_parentid[da2]

        moment_rownnz_out[worldid, actid] = ndof
        rowadr = wp.atomic_add(moment_nnz, worldid, ndof)
        moment_rowadr_out[worldid, actid] = rowadr

        # traverse dofs
        da1 = da1_init
        da2 = da2_init

        ptr = ndof - 1
        while da1 >= 0 or da2 >= 0:
          da = wp.max(da1, da2)
          if da1 == da and da2 == da:
            break

          jacp, jacr = support.jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com_in,
            cdof_in,
            site_xpos,
            site_bodyid[siteid],
            da,
            worldid,
          )
          jacpref, jacrref = support.jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com_in,
            cdof_in,
            ref_xpos,
            site_bodyid[refid],
            da,
            worldid,
          )

          moment = float(0.0)
          if translational_transmission:
            moment += wp.dot(jacp - jacpref, wrench_translation)
          if rotational_transmission:
            moment += wp.dot(jacr - jacrref, wrench_rotation)

          sparseid = rowadr + ptr
          moment_colind_out[worldid, sparseid] = da
          actuator_moment_out[worldid, sparseid] = moment
          ptr -= 1

          if da1 == da:
            da1 = dof_parentid[da1]
          if da2 == da:
            da2 = dof_parentid[da2]
    else:
      wp.printf("unhandled transmission type %d\n", trntype)

  return kernel


@cache_kernel
def _transmission_body_moment(nmaxpyramid: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 2 * nmaxpyramid

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_cone: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    dof_bodyid: wp.array[int],
    geom_bodyid: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    body_isdofancestor: wp.array2d[int],
    actuator_trntype_body_adr: wp.array[int],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    moment_rowadr_in: wp.array2d[int],
    contact_dist_in: wp.array[float],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_includemargin_in: wp.array[float],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    nacon_in: wp.array[int],
    # In:
    efc_is_sparse: bool,
    # Data out:
    actuator_moment_out: wp.array2d[float],
    # Out:
    actuator_trntype_body_ncon_out: wp.array2d[int],
  ):
    trnbodyid, conid, dofid = wp.tid()
    actid = actuator_trntype_body_adr[trnbodyid]
    bodyid = actuator_trnid[actid][0]

    if conid >= nacon_in[0]:
      return

    worldid = contact_worldid_in[conid]

    # get geom ids
    geom = contact_geom_in[conid]
    g1 = geom[0]
    g2 = geom[1]

    # contact involving flex, continue
    if g1 < 0 or g2 < 0:
      return

    # get body ids
    b1 = geom_bodyid[g1]
    b2 = geom_bodyid[g2]

    # irrelevant contact, continue
    if b1 != bodyid and b2 != bodyid:
      return

    contact_exclude = int(contact_dist_in[conid] >= contact_includemargin_in[conid])

    if dofid == 0:
      wp.atomic_add(actuator_trntype_body_ncon_out[worldid], trnbodyid, 1)

    rowadr = moment_rowadr_in[worldid, actid]

    # mark contact normals in efc_force
    if contact_exclude == 0:
      contact_dim = contact_dim_in[conid]
      contact_efc_address = contact_efc_address_in[conid]

      if contact_dim == 1 or opt_cone == ConeType.ELLIPTIC:
        efcid0 = contact_efc_address[0]
        if efc_is_sparse:
          rownnz = efc_J_rownnz_in[worldid, efcid0]
          if dofid < rownnz:
            efc_rowadr = efc_J_rowadr_in[worldid, efcid0]
            efc_sparseid = efc_rowadr + dofid
            colind = efc_J_colind_in[worldid, 0, efc_sparseid]
            wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid])
          else:
            return
        else:
          colind = dofid
          wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid0, dofid])
      else:
        npyramid = contact_dim - 1  # number of frictional directions
        efc_force = 0.5 / float(npyramid)

        for j in range(2 * npyramid):
          efcid = contact_efc_address[j]
          if efc_is_sparse:
            rownnz = efc_J_rownnz_in[worldid, efcid]
            if dofid < rownnz:
              efc_rowadr = efc_J_rowadr_in[worldid, efcid]
              efc_sparseid = efc_rowadr + dofid
              colind = efc_J_colind_in[worldid, 0, efc_sparseid]
              wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid] * efc_force)
            else:
              return
          else:
            colind = dofid
            wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid, dofid] * efc_force)

    # excluded contact in gap: get Jacobian, accumulate
    elif contact_exclude == 1:
      contact_pos = contact_pos_in[conid]
      contact_frame = contact_frame_in[conid]
      normal = wp.vec3(contact_frame[0, 0], contact_frame[0, 1], contact_frame[0, 2])

      # get Jacobian difference
      efcid0 = contact_efc_address_in[conid][0]
      if efc_is_sparse and efcid0 >= 0:
        # contact has valid efc row: use sparse pattern
        if dofid >= efc_J_rownnz_in[worldid, efcid0]:
          return
        sparseid = efc_J_rowadr_in[worldid, efcid0] + dofid
        colind = efc_J_colind_in[worldid, 0, sparseid]
      else:
        # excluded contact with no efc row or dense: use dofid directly
        colind = dofid

      jacp1, _ = support.jac_dof(
        body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, contact_pos, b1, colind, worldid
      )
      jacp2, _ = support.jac_dof(
        body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, contact_pos, b2, colind, worldid
      )

      jacdif = jacp2 - jacp1

      # project Jacobian along the normal of the contact frame
      wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, wp.dot(normal, jacdif))

  return kernel


@wp.kernel
def _transmission_body_moment_scale(
  # Model:
  actuator_trntype_body_adr: wp.array[int],
  # Data in:
  moment_rowadr_in: wp.array2d[int],
  # In:
  actuator_trntype_body_ncon_in: wp.array2d[int],
  # Data out:
  actuator_moment_out: wp.array2d[float],
):
  worldid, trnbodyid, dofid = wp.tid()

  ncon = actuator_trntype_body_ncon_in[worldid, trnbodyid]

  if ncon > 0:
    actid = actuator_trntype_body_adr[trnbodyid]
    rowadr = moment_rowadr_in[worldid, actid]
    actuator_moment_out[worldid, rowadr + dofid] /= -float(ncon)


@event_scope
def transmission(m: Model, d: Data):
  """Computes actuator/transmission lengths and moments.

  Updates the actuator length and moments for all actuators in the model, including joint
  and tendon transmissions.
  """
  # TODO(team): investigate pre-computing moment_rownnz, moment_rowadr, moment_colind
  moment_nnz = wp.zeros((d.nworld,), dtype=int)

  wp.launch(
    _transmission(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.nactuator),
    inputs=[
      m.nv,
      m.body_parentid,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.dof_bodyid,
      m.dof_parentid,
      m.site_bodyid,
      m.site_quat,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_cranklength,
      m.actuator_gear,
      m.body_isdofancestor,
      d.qpos,
      d.xquat,
      d.site_xpos,
      d.site_xmat,
      d.subtree_com,
      d.cdof,
      d.ten_J,
      d.ten_length,
      moment_nnz,
    ],
    outputs=[d.actuator_length, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment],
  )

  if m.nacttrnbody:
    # compute moments
    ncon = wp.zeros((d.nworld, m.nacttrnbody), dtype=int)

    det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
    wp.launch(
      _transmission_body_moment(m.nmaxpyramid if det_atomics else 0, det_atomics),
      dim=(m.nacttrnbody, d.naconmax, m.nv),
      inputs=[
        m.opt.cone,
        m.body_parentid,
        m.body_rootid,
        m.dof_bodyid,
        m.geom_bodyid,
        m.actuator_trnid,
        m.body_isdofancestor,
        m.actuator_trntype_body_adr,
        d.subtree_com,
        d.cdof,
        d.moment_rowadr,
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.includemargin,
        d.contact.dim,
        d.contact.geom,
        d.contact.efc_address,
        d.contact.worldid,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.nacon,
        m.is_sparse,
      ],
      outputs=[d.actuator_moment, ncon],
    )

    # scale moments
    wp.launch(
      _transmission_body_moment_scale,
      dim=(d.nworld, m.nacttrnbody, m.nv),
      inputs=[m.actuator_trntype_body_adr, d.moment_rowadr, ncon],
      outputs=[d.actuator_moment],
    )


@wp.func
def _solve_LD_sparse_copy_fn(
  # Model:
  qLD_block_adr: wp.array[int],
  # In:
  y_in: wp.array2d[float],
  worldid: int,
  dofid: int,
  # Out:
  x_out: wp.array2d[float],
):
  if qLD_block_adr[dofid] == Q_LD_BLOCK_SPARSE:
    x_out[worldid, dofid] = y_in[worldid, dofid]


@wp.func
def _solve_LD_sparse_diag_fn(
  # Model:
  qLD_block_adr: wp.array[int],
  # In:
  D_in: wp.array2d[float],
  x_in: wp.array2d[float],
  worldid: int,
  dofid: int,
  # Out:
  x_out: wp.array2d[float],
):
  if qLD_block_adr[dofid] == Q_LD_BLOCK_SPARSE:
    x_out[worldid, dofid] = x_in[worldid, dofid] * D_in[worldid, dofid]


@wp.func
def _solve_LD_sparse_update_fn(
  # In:
  L_in: wp.array2d[float],
  x_in: wp.array2d[float],
  update: wp.vec3i,
  worldid: int,
  up: bool,
  # Out:
  x_out: wp.array2d[float],
):
  i, k, Madr_ki = update[0], update[1], update[2]
  if up:
    wp.atomic_sub(x_out[worldid], i, L_in[worldid, Madr_ki] * x_in[worldid, k])
  else:
    wp.atomic_sub(x_out[worldid], k, L_in[worldid, Madr_ki] * x_in[worldid, i])


@wp.kernel
def _solve_LD_sparse_copy(
  # Model:
  qLD_block_adr: wp.array[int],
  # In:
  y_in: wp.array2d[float],
  # Out:
  x_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  _solve_LD_sparse_copy_fn(qLD_block_adr, y_in, worldid, dofid, x_out)


@wp.kernel
def _solve_LD_sparse_diag(
  # Model:
  qLD_block_adr: wp.array[int],
  # In:
  D_in: wp.array2d[float],
  x_in: wp.array2d[float],
  # Out:
  x_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  _solve_LD_sparse_diag_fn(qLD_block_adr, D_in, x_in, worldid, dofid, x_out)


@cache_kernel
def _solve_LD_sparse_level(up: bool, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # In:
    L_in: wp.array2d[float],
    qLD_updates_: wp.array[wp.vec3i],
    x_in: wp.array2d[float],
    # Out:
    x_out: wp.array2d[float],
  ):
    worldid, nodeid = wp.tid()
    _solve_LD_sparse_update_fn(L_in, x_in, qLD_updates_[nodeid], worldid, wp.static(up), x_out)

  return kernel


@cache_kernel
def _solve_LD_sparse_fused(nv: int, nlevels: int):
  """Fused sparse backsubstitution: UP + diag + DOWN in one kernel."""

  @wp.func_native(snippet="WP_TILE_SYNC();")
  def _syncthreads():
    pass

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # In:
    L: wp.array2d[float],
    D: wp.array2d[float],
    all_updates: wp.array[wp.vec3i],
    level_offsets: wp.array[int],
    y: wp.array2d[float],
    # Out:
    x_out: wp.array2d[float],
  ):
    worldid, tid = wp.tid()
    NV = wp.static(nv)
    NLEVELS = wp.static(nlevels)
    BLOCK_DIM = wp.block_dim()

    # Copy y to x_out for sparse-block dofs only.
    for dofid in range(tid, NV, BLOCK_DIM):
      _solve_LD_sparse_copy_fn(qLD_block_adr, y, worldid, dofid, x_out)
    _syncthreads()

    # Forward substitution (all_updates only references sparse-block dofs)
    for level in range(NLEVELS):
      level_idx = NLEVELS - 1 - level
      level_offset = level_offsets[level_idx]
      level_size = level_offsets[level_idx + 1] - level_offset

      for u in range(tid, level_size, BLOCK_DIM):
        _solve_LD_sparse_update_fn(
          L, x_out, all_updates[level_offset + u], worldid, True, x_out
        )  # kernel_analyzer: ignore[atomic]
      _syncthreads()

    # Diagonal multiply (sparse-block dofs only)
    for dofid in range(tid, NV, BLOCK_DIM):
      _solve_LD_sparse_diag_fn(qLD_block_adr, D, x_out, worldid, dofid, x_out)
    _syncthreads()

    # Backward substitution
    for level in range(NLEVELS):
      level_idx = level
      level_offset = level_offsets[level_idx]
      level_size = level_offsets[level_idx + 1] - level_offset

      for u in range(tid, level_size, BLOCK_DIM):
        _solve_LD_sparse_update_fn(
          L, x_out, all_updates[level_offset + u], worldid, False, x_out
        )  # kernel_analyzer: ignore[atomic]
      _syncthreads()

  return kernel


def _solve_LD_sparse(
  m: Model,
  d: Data,
  L: wp.array2d[float],
  D: wp.array2d[float],
  x: wp.array2d[float],
  y: wp.array2d[float],
  x_scratch: Optional[wp.array] = None,
):
  """Computes sparse backsubstitution: x = inv(L'*D*L)*y."""
  if m.opt.deterministic & DeterminismType.ATOMICS:
    x_in = x_scratch if x_scratch is not None else wp.empty_like(x)
    wp.launch(_solve_LD_sparse_copy, dim=(d.nworld, m.nv), inputs=[m.qLD_block_adr, y], outputs=[x])
    for qLD_updates in reversed(m.qLD_updates):
      wp.copy(x_in, x)
      wp.launch(
        _solve_LD_sparse_level(True, True),
        dim=(d.nworld, qLD_updates.size),
        inputs=[L, qLD_updates, x_in],
        outputs=[x],
      )
    wp.launch(_solve_LD_sparse_diag, dim=(d.nworld, m.nv), inputs=[m.qLD_block_adr, D, x], outputs=[x])
    for qLD_updates in m.qLD_updates:
      wp.copy(x_in, x)
      wp.launch(
        _solve_LD_sparse_level(False, True),
        dim=(d.nworld, qLD_updates.size),
        inputs=[L, qLD_updates, x_in],
        outputs=[x],
      )
    return

  nlevels = len(m.qLD_updates)
  if wp.get_device().is_cuda:
    dim_block = m.block_dim.solve_LD_sparse_fused
  else:
    # Fallback for CPU
    dim_block = 1

  wp.launch(
    _solve_LD_sparse_fused(m.nv, nlevels),
    dim=(d.nworld, dim_block),
    inputs=[m.qLD_block_adr, L, D, m.qLD_all_updates, m.qLD_level_offsets, y],
    outputs=[x],
    block_dim=dim_block,
  )


@wp.func
def _small_cholesky_solve(
  # In:
  block_size: int,
  worldid: int,
  factor_adr: int,
  start: int,
  L_in: wp.array2d[float],
  y_in: wp.array2d[float],
  # Out:
  x_out: wp.array2d[float],
):
  for i in range(block_size):
    value = y_in[worldid, start + i]
    for k in range(i):
      value -= L_in[worldid, factor_adr + k * block_size + i] * x_out[worldid, start + k]
    x_out[worldid, start + i] = value / L_in[worldid, factor_adr + i * block_size + i]

  for reverse_i in range(block_size):
    i = block_size - 1 - reverse_i
    value = x_out[worldid, start + i]
    for k in range(i + 1, block_size):
      value -= L_in[worldid, factor_adr + i * block_size + k] * x_out[worldid, start + k]
    x_out[worldid, start + i] = value / L_in[worldid, factor_adr + i * block_size + i]


@cache_kernel
def _small_cholesky_solve_block(block_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # In:
    block_dof: wp.array[int],
    D_in: wp.array2d[float],
    L_in: wp.array2d[float],
    y_in: wp.array2d[float],
    # Out:
    x_out: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]
    size = wp.static(block_size)
    factor_adr = qLD_block_adr[start]
    if factor_adr == Q_LD_BLOCK_COMPACT:
      for i in range(wp.static(block_size)):
        x_out[worldid, start + i] = D_in[worldid, start + i] * y_in[worldid, start + i]
    else:
      _small_cholesky_solve(size, worldid, factor_adr, start, L_in, y_in, x_out)

  return kernel


@cache_kernel
def _tile_cholesky_solve_block(tile: TileSet):
  # One diagonal block per (world, block) thread group; no densify, so a 2D grid suffices.
  block_size = tile.size
  block_area = block_size * block_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # In:
    block_dof: wp.array[int],
    L_in: wp.array2d[float],
    y: wp.array2d[float],
    # Out:
    x: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]

    L = wp.tile_reshape(
      wp.tile_load(L_in[worldid], shape=(block_area,), offset=(qLD_block_adr[start],)), (block_size, block_size)
    )
    rhs = wp.tile_load(y[worldid], shape=(block_size,), offset=(start,))
    sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
    wp.tile_store(x[worldid], sol, offset=(start,))

  return kernel


def _solve_blocks(
  m: Model,
  d: Data,
  L: wp.array2d[float],
  D: wp.array2d[float],
  x: wp.array2d[float],
  y: wp.array2d[float],
):
  for tile in m.M_tiles:
    if tile.elemid.size == 0:
      wp.launch(
        _small_cholesky_solve_block(tile.size),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.qLD_block_adr, tile.adr, D, L, y],
        outputs=[x],
        block_dim=m.block_dim.small_cholesky,
      )
    else:
      # The triangular back-substitution is largely sequential, so large blocks prefer fewer threads
      # for better occupancy while moderate blocks still want a couple warps (16/27->64, 60->32).
      block_dim = m.block_dim.cholesky_solve if tile.size <= 40 else 32
      wp.launch_tiled(
        _tile_cholesky_solve_block(tile),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.qLD_block_adr, tile.adr, L, y],
        outputs=[x],
        block_dim=block_dim,
      )


def solve_LD(
  m: Model,
  d: Data,
  L: wp.array2d[float],
  D: wp.array2d[float],
  x: wp.array2d[float],
  y: wp.array2d[float],
  x_scratch: Optional[wp.array] = None,
):
  """Computes backsubstitution for the inertia factorization.

  Compact blocks use reciprocal diagonals, full small blocks use scalar Cholesky, dense blocks use
  tile Cholesky, and sparse blocks use the LDL region.

  Args:
    m: The model containing factorization and sparsity information.
    d: The data object containing workspace and factorization results.
    L: The factor: packed dense region followed by the nC LDL region.
    D: Reciprocal diagonal for compact and sparse blocks.
    x: Output array for the solution.
    y: Input right-hand side array.
    x_scratch: Preallocated scratch array for deterministic sparse backsubstitution.
  """
  if m.M_tiles:
    _solve_blocks(m, d, L, D, x, y)
  if L.shape[1] > m.qLD_block_total:
    _solve_LD_sparse(m, d, L[:, m.qLD_block_total :], D, x, y, x_scratch=x_scratch)


@event_scope
def solve_m(m: Model, d: Data, x: wp.array2d[float], y: wp.array2d[float], x_scratch: Optional[wp.array] = None):
  """Computes backsubstitution: x = qLD * y.

  Args:
    m: The model containing inertia and factorization information.
    d: The data object containing factorization results.
    x: Output array for the solution.
    y: Input right-hand side array.
    x_scratch: Preallocated scratch array for deterministic sparse backsubstitution.
  """
  solve_LD(m, d, d.qLD, d.qLDiagInv, x, y, x_scratch=x_scratch)


@cache_kernel
def _tile_cholesky_factorize_solve_block(tile: TileSet):
  # Fused factor+solve: densify the block, factor it, and back-substitute in one launch (avoids
  # re-loading the factor). Grid/densify structure matches _tile_cholesky_factorize_block.
  block_size = tile.size
  block_area = block_size * block_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    # In:
    block_elemid: wp.array[int],
    block_dof: wp.array[int],
    y: wp.array2d[float],
    x: wp.array2d[float],
    # Out:
    L_out: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]

    # Densify the block (see _tile_cholesky_factorize_block for the gather rationale).
    idx = wp.tile_load(block_elemid, shape=(block_area,), offset=(blk * block_area,), storage="shared")
    block = wp.tile_load_indexed(M_in[worldid], idx, shape=(block_area,), storage="shared")

    L = wp.tile_reshape(block, (block_size, block_size))
    wp.tile_cholesky_inplace(L, fill_mode="upper")
    wp.tile_store(L_out[worldid], wp.tile_reshape(L, (block_area,)), offset=(qLD_block_adr[start],))

    rhs = wp.tile_load(y[worldid], shape=(block_size,), offset=(start,))
    sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
    wp.tile_store(x[worldid], sol, offset=(start,))

  return kernel


@cache_kernel
def _small_cholesky_factorize_solve_block(block_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    M_rowadr: wp.array[int],
    qLD_block_adr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    # In:
    block_dof: wp.array[int],
    y: wp.array2d[float],
    # Out:
    D_out: wp.array2d[float],
    x_out: wp.array2d[float],
    L_out: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]
    size = wp.static(block_size)
    matrix_adr = M_rowadr[start]

    factor_adr = qLD_block_adr[start]
    if factor_adr == Q_LD_BLOCK_COMPACT:
      for i in range(wp.static(block_size)):
        inverse = 1.0 / M_in[worldid, matrix_adr + i]
        D_out[worldid, start + i] = inverse
        x_out[worldid, start + i] = inverse * y[worldid, start + i]
    else:
      for i in range(wp.static(block_size)):
        diagonal_value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]
        rhs_value = y[worldid, start + i]
        for k in range(i):
          factor = L_out[worldid, factor_adr + k * size + i]
          diagonal_value -= factor * factor
          rhs_value -= factor * x_out[worldid, start + k]

        diagonal_factor = wp.sqrt(diagonal_value)
        L_out[worldid, factor_adr + i * size + i] = diagonal_factor
        diagonal_inv = 1.0 / diagonal_factor
        x_out[worldid, start + i] = rhs_value * diagonal_inv

        for j in range(i + 1, size):
          value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]
          for k in range(i):
            value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]
          L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv

      for reverse_i in range(wp.static(block_size)):
        i = size - 1 - reverse_i
        value = x_out[worldid, start + i]
        for k in range(i + 1, size):
          value -= L_out[worldid, factor_adr + i * size + k] * x_out[worldid, start + k]
        x_out[worldid, start + i] = value / L_out[worldid, factor_adr + i * size + i]

  return kernel


def _factor_solve_blocks(
  m: Model,
  d: Data,
  M: wp.array2d[float],
  L: wp.array2d[float],
  D: wp.array2d[float],
  x: wp.array2d[float],
  y: wp.array2d[float],
):
  for tile in m.M_tiles:
    if tile.elemid.size == 0:
      wp.launch(
        _small_cholesky_factorize_solve_block(tile.size),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.M_rowadr, m.qLD_block_adr, M, tile.adr, y],
        outputs=[D, x, L],
        block_dim=m.block_dim.small_cholesky,
      )
    else:
      wp.launch_tiled(
        _tile_cholesky_factorize_solve_block(tile),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.qLD_block_adr, M, tile.elemid, tile.adr, y],
        outputs=[x, L],
        block_dim=m.block_dim.cholesky_factorize_solve,
      )


def factor_solve_i(m, d, M, L, D, x, y):
  """Factorizes and solves the inertia-like linear system.

  Compact blocks use reciprocal diagonals, full small blocks use scalar Cholesky, dense blocks use
  tile Cholesky, and sparse blocks use LDL. Factorizes M and solves for x.

  Args:
    m: The model containing factorization and sparsity information.
    d: The data object containing workspace and factorization results.
    M: The inertia-like matrix to factorize (CSR, length nC).
    L: Output factor: packed dense region followed by the nC LDL region (sized like d.qLD).
    D: Output reciprocal diagonal for compact and sparse blocks.
    x: Output array for the solution.
    y: Input right-hand side array.
  """
  if m.M_tiles:
    _factor_solve_blocks(m, d, M, L, D, x, y)
  if L.shape[1] > m.qLD_block_total:
    L_ldl = L[:, m.qLD_block_total :]
    _factor_i_sparse(m, d, M, L_ldl, D)
    _solve_LD_sparse(m, d, L_ldl, D, x, y)


@cache_kernel
def _factor_solve_lu_sparse_fused(nv: int):
  """Fused sparse LU factorization and solve in a single kernel."""

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    D_rownnz: wp.array[int],
    D_rowadr: wp.array[int],
    D_diag: wp.array[int],
    D_colind: wp.array[int],
    # In:
    qfrc: wp.array2d[float],
    # Data out:
    qacc_out: wp.array2d[float],
    qLU_out: wp.array2d[float],
  ):
    worldid = wp.tid()
    NV = wp.static(nv)

    # Phase 1: LU factorization (in-place on qLU_out)
    for i in range(NV):
      qacc_out[worldid, i] = float(D_rownnz[i])

    # process diagonal elements from n-1 down to 0
    for r_rev in range(NV):
      i = NV - 1 - r_rev

      rem_i = int(qacc_out[worldid, i])
      rowadr_i = D_rowadr[i]
      ii = rowadr_i + rem_i - 1
      qacc_out[worldid, i] = float(rem_i - 1)

      # cache diagonal element for row i
      LUii = qLU_out[worldid, ii]

      # rows j above i (j < i), processed from i-1 down to 0
      for c in range(i):
        j = i - 1 - c

        # get address of last remaining element of row j
        rem_j = int(qacc_out[worldid, j])
        rowadr_j = D_rowadr[j]
        ji = rowadr_j + rem_j - 1

        # process row j if (j,i) is non-zero
        if D_colind[ji] == i:
          # adjust remaining counter
          rem_j = rem_j - 1
          qacc_out[worldid, j] = float(rem_j)

          # (j,i) = (j,i) / (i,i)
          LUji = qLU_out[worldid, ji] / LUii
          qLU_out[worldid, ji] = LUji

          # (j,k) = (j,k) - (i,k) * (j,i) for k < i
          icnt = rowadr_i
          jcnt = rowadr_j
          jend = rowadr_j + rem_j
          while jcnt < jend:
            col_i = D_colind[icnt]
            col_j = D_colind[jcnt]
            if col_i == col_j:
              qLU_out[worldid, jcnt] = qLU_out[worldid, jcnt] - qLU_out[worldid, icnt] * LUji
              icnt = icnt + 1
              jcnt = jcnt + 1
            elif col_i > col_j:
              jcnt = jcnt + 1
            else:
              icnt = icnt + 1

    # Phase 2: LU solve (backward + forward substitution)

    # Backward substitution: solve (U+I)*qacc = qfrc
    for k_rev in range(NV):
      i = NV - 1 - k_rev

      diag_i = D_diag[i]
      rowadr_i = D_rowadr[i]
      d1 = diag_i + 1
      nnz_upper = D_rownnz[i] - d1

      acc = qfrc[worldid, i]
      for j in range(nnz_upper):
        adr_j = rowadr_i + d1 + j
        col = D_colind[adr_j]
        acc = acc - qLU_out[worldid, adr_j] * qacc_out[worldid, col]
      qacc_out[worldid, i] = acc

    # Forward substitution: solve L*qacc = qacc
    for i in range(NV):
      diag_i = D_diag[i]
      rowadr_i = D_rowadr[i]

      acc = qacc_out[worldid, i]
      for j in range(diag_i):
        adr_j = rowadr_i + j
        col = D_colind[adr_j]
        acc = acc - qLU_out[worldid, adr_j] * qacc_out[worldid, col]

      qacc_out[worldid, i] = acc / qLU_out[worldid, rowadr_i + diag_i]

  return kernel


@event_scope
def factor_solve_lu(m: Model, d: Data, qLU: wp.array2d[float], qacc: wp.array2d[float], qfrc: wp.array2d[float]):
  r"""Factorize and solve non-symmetric implicit system: qacc = A \\ qfrc.

  qLU is overwritten in-place with the LU factors, then used to solve for qacc.

  Args:
    m: The model containing D-structure sparsity information.
    d: The data object.
    qLU: array containing the system matrix, overwritten with LU factors.
    qacc: output array for the solution.
    qfrc: input right-hand side.
  """
  wp.launch(
    _factor_solve_lu_sparse_fused(m.nv),
    dim=(d.nworld,),
    inputs=[m.D_rownnz, m.D_rowadr, m.D_diag, m.D_colind, qfrc],
    outputs=[qacc, qLU],
  )


@wp.kernel
def _subtree_vel_forward(
  # Model:
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # Data out:
  subtree_linvel_out: wp.array2d[wp.vec3],
  subtree_angmom_out: wp.array2d[wp.vec3],
  # Out:
  subtree_bodyvel_out: wp.array2d[wp.spatial_vector],
):
  worldid, bodyid = wp.tid()
  body_mass_id = worldid % body_mass.shape[0]
  body_inertia_id = worldid % body_inertia.shape[0]

  cvel = cvel_in[worldid, bodyid]
  ang = wp.spatial_top(cvel)
  lin = wp.spatial_bottom(cvel)
  xipos = xipos_in[worldid, bodyid]
  ximat = ximat_in[worldid, bodyid]
  subtree_com_root = subtree_com_in[worldid, body_rootid[bodyid]]

  # update linear velocity
  lin -= wp.cross(xipos - subtree_com_root, ang)

  subtree_linvel_out[worldid, bodyid] = body_mass[body_mass_id, bodyid] * lin
  dv = wp.transpose(ximat) @ ang
  dv[0] *= body_inertia[body_inertia_id, bodyid][0]
  dv[1] *= body_inertia[body_inertia_id, bodyid][1]
  dv[2] *= body_inertia[body_inertia_id, bodyid][2]
  subtree_angmom_out[worldid, bodyid] = ximat @ dv
  subtree_bodyvel_out[worldid, bodyid] = wp.spatial_vector(ang, lin)


@cache_kernel
def _linear_momentum(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_subtreemass: wp.array2d[float],
    # Data in:
    subtree_linvel_in: wp.array2d[wp.vec3],
    # In:
    body_tree_: wp.array[int],
    # Data out:
    subtree_linvel_out: wp.array2d[wp.vec3],
  ):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    linvel = subtree_linvel_in[worldid, bodyid]
    if bodyid:
      pid = body_parentid[bodyid]
      wp.atomic_add(subtree_linvel_out[worldid], pid, linvel)
    if wp.static(not deterministic):
      subtree_linvel_out[worldid, bodyid] = linvel / wp.max(
        MJ_MINVAL, body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]
      )

  return kernel


@wp.kernel
def _linear_momentum_div(
  # Model:
  body_subtreemass: wp.array2d[float],
  # Data in:
  subtree_linvel_in: wp.array2d[wp.vec3],
  # Data out:
  subtree_linvel_out: wp.array2d[wp.vec3],
):
  worldid, bodyid = wp.tid()
  subtree_linvel_out[worldid, bodyid] = subtree_linvel_in[worldid, bodyid] / wp.max(
    MJ_MINVAL, body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]
  )


@wp.kernel
def _angular_momentum_body(
  # Model:
  body_mass: wp.array2d[float],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  subtree_linvel_in: wp.array2d[wp.vec3],
  subtree_angmom_in: wp.array2d[wp.vec3],
  # In:
  subtree_bodyvel_in: wp.array2d[wp.spatial_vector],
  # Data out:
  subtree_angmom_out: wp.array2d[wp.vec3],
):
  worldid, bodyid = wp.tid()
  if bodyid == 0:
    return
  dx = xipos_in[worldid, bodyid] - subtree_com_in[worldid, bodyid]
  dv = wp.spatial_bottom(subtree_bodyvel_in[worldid, bodyid]) - subtree_linvel_in[worldid, bodyid]
  dp = dv * body_mass[worldid % body_mass.shape[0], bodyid]
  subtree_angmom_out[worldid, bodyid] = subtree_angmom_in[worldid, bodyid] + wp.cross(dx, dp)


@cache_kernel
def _angular_momentum(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    subtree_linvel_in: wp.array2d[wp.vec3],
    subtree_angmom_in: wp.array2d[wp.vec3],
    # In:
    subtree_bodyvel_in: wp.array2d[wp.spatial_vector],
    body_tree_: wp.array[int],
    # Data out:
    subtree_angmom_out: wp.array2d[wp.vec3],
  ):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]

    if bodyid == 0:
      return

    pid = body_parentid[bodyid]

    com = subtree_com_in[worldid, bodyid]
    com_parent = subtree_com_in[worldid, pid]
    linvel = subtree_linvel_in[worldid, bodyid]
    linvel_parent = subtree_linvel_in[worldid, pid]  # Data field
    subtreemass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]

    if wp.static(deterministic):
      angmom = subtree_angmom_in[worldid, bodyid]
    else:
      xipos = xipos_in[worldid, bodyid]
      vel = subtree_bodyvel_in[worldid, bodyid]
      mass = body_mass[worldid % body_mass.shape[0], bodyid]

      # momentum wrt body i
      dx = xipos - com
      dv = wp.spatial_bottom(vel) - linvel
      dp = dv * mass
      dL = wp.cross(dx, dp)

      # add to subtree i
      angmom = subtree_angmom_in[worldid, bodyid] + dL
      subtree_angmom_out[worldid, bodyid] = angmom

    # momentum wrt parent
    dx = com - com_parent
    dv = linvel - linvel_parent
    dv *= subtreemass
    dL_parent = wp.cross(dx, dv)

    # add to parent
    wp.atomic_add(subtree_angmom_out[worldid], pid, angmom + dL_parent)

  return kernel


def subtree_vel(m: Model, d: Data):
  """Computes subtree linear velocity and angular momentum.

  Computes the linear momentum and angular momentum for each subtree, accumulating
  contributions up the kinematic tree.
  """
  subtree_bodyvel = wp.empty((d.nworld, m.nbody), dtype=wp.spatial_vector)

  # bodywise quantities
  wp.launch(
    _subtree_vel_forward,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com, d.cvel],
    outputs=[d.subtree_linvel, d.subtree_angmom, subtree_bodyvel],
  )

  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  subtree_vec_in = wp.empty_like(d.subtree_linvel) if det_atomics else None

  # sum body linear momentum recursively up the kinematic tree
  for body_tree in reversed(m.body_tree):
    if det_atomics:
      wp.copy(subtree_vec_in, d.subtree_linvel)
    wp.launch(
      _linear_momentum(det_atomics),
      dim=[d.nworld, body_tree.size],
      inputs=[m.body_parentid, m.body_subtreemass, subtree_vec_in if det_atomics else d.subtree_linvel, body_tree],
      outputs=[d.subtree_linvel],
    )
  if det_atomics:
    wp.launch(
      _linear_momentum_div,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_subtreemass, d.subtree_linvel],
      outputs=[d.subtree_linvel],
    )
    wp.launch(
      _angular_momentum_body,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_mass, d.xipos, d.subtree_com, d.subtree_linvel, d.subtree_angmom, subtree_bodyvel],
      outputs=[d.subtree_angmom],
    )

  for body_tree in reversed(m.body_tree):
    if det_atomics:
      wp.copy(subtree_vec_in, d.subtree_angmom)
    wp.launch(
      _angular_momentum(det_atomics),
      dim=[d.nworld, body_tree.size],
      inputs=[
        m.body_parentid,
        m.body_mass,
        m.body_subtreemass,
        d.xipos,
        d.subtree_com,
        d.subtree_linvel,
        subtree_vec_in if det_atomics else d.subtree_angmom,
        subtree_bodyvel,
        body_tree,
      ],
      outputs=[d.subtree_angmom],
    )


@cache_kernel
def _joint_tendon(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    wrap_objid: wp.array[int],
    wrap_prm: wp.array[float],
    tendon_jnt_adr: wp.array[int],
    wrap_jnt_adr: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    # Data out:
    ten_J_out: wp.array2d[float],
    ten_length_out: wp.array2d[float],
  ):
    worldid, wrapid = wp.tid()

    tenid = tendon_jnt_adr[wrapid]
    wrapjntid = wrap_jnt_adr[wrapid]
    wrapobjid = wrap_objid[wrapjntid]
    prm = wrap_prm[wrapjntid]

    # add to length
    L = prm * qpos_in[worldid, jnt_qposadr[wrapobjid]]
    wp.atomic_add(ten_length_out[worldid], tenid, L)

    # add to moment
    dofadr = jnt_dofadr[wrapobjid]
    rowadr = ten_J_rowadr[tenid]
    rownnz = ten_J_rownnz[tenid]
    for k in range(rownnz):
      if ten_J_colind[rowadr + k] == dofadr:
        ten_J_out[worldid, rowadr + k] = prm
        break

  return kernel


@wp.func
def _accumulate_jac_chain(
  # Model:
  body_parentid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  offset: wp.vec3,
  vec: wp.vec3,
  bodyid: int,
  rowadr: int,
  rownnz: int,
  scale: float,
  worldid: int,
  # Data out:
  ten_J_out: wp.array2d[float],
):
  """Walk body chain from bodyid to root, accumulate Jacobian contributions."""
  ptr = rownnz - 1
  bid = bodyid
  while bid > 0:
    bdofadr = body_dofadr[bid]
    bdofnum = body_dofnum[bid]
    # iterate DOFs in this body in descending order
    for k_rev in range(bdofnum):
      dof = bdofadr + bdofnum - 1 - k_rev
      # scan pointer backward to find matching colind entry
      while ptr >= 0:
        sparseid = rowadr + ptr
        if ten_J_colind[sparseid] <= dof:
          break
        ptr -= 1
      if ptr >= 0 and ten_J_colind[sparseid] == dof:
        cdof = cdof_in[worldid, dof]
        cdof_ang = wp.spatial_top(cdof)
        cdof_lin = wp.spatial_bottom(cdof)
        jacp = cdof_lin + wp.cross(cdof_ang, offset)
        J = wp.dot(jacp, vec) * scale
        if J != 0.0:
          wp.atomic_add(ten_J_out[worldid], sparseid, J)
    bid = body_parentid[bid]


@cache_kernel
def _spatial_site_tendon(max_ten_J_rownnz: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 2 * max_ten_J_rownnz

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    site_bodyid: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    wrap_objid: wp.array[int],
    tendon_site_pair_adr: wp.array[int],
    wrap_site_pair_adr: wp.array[int],
    wrap_pulley_scale: wp.array[float],
    # Data in:
    site_xpos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    # Data out:
    ten_J_out: wp.array2d[float],
    ten_length_out: wp.array2d[float],
  ):
    worldid, elementid = wp.tid()

    # site pairs
    site_pair_adr = wrap_site_pair_adr[elementid]
    tenid = tendon_site_pair_adr[elementid]

    # pulley scaling
    pulley_scale = wrap_pulley_scale[site_pair_adr]

    id0 = wrap_objid[site_pair_adr + 0]
    id1 = wrap_objid[site_pair_adr + 1]

    pnt0 = site_xpos_in[worldid, id0]
    pnt1 = site_xpos_in[worldid, id1]
    dif = pnt1 - pnt0
    vec, length = math.normalize_with_norm(dif)
    wp.atomic_add(ten_length_out[worldid], tenid, length * pulley_scale)

    if length < MJ_MINVAL:
      vec = wp.vec3(1.0, 0.0, 0.0)

    body0 = site_bodyid[id0]
    body1 = site_bodyid[id1]
    if body0 != body1:
      rownnz = ten_J_rownnz[tenid]
      rowadr = ten_J_rowadr[tenid]
      offset0 = pnt0 - subtree_com_in[worldid, body_rootid[body0]]
      offset1 = pnt1 - subtree_com_in[worldid, body_rootid[body1]]
      _accumulate_jac_chain(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        offset0,
        vec,
        body0,
        rowadr,
        rownnz,
        -pulley_scale,
        worldid,
        ten_J_out,
      )
      _accumulate_jac_chain(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        offset1,
        vec,
        body1,
        rowadr,
        rownnz,
        pulley_scale,
        worldid,
        ten_J_out,
      )

  return kernel


@cache_kernel
def _spatial_geom_tendon(max_ten_J_rownnz: int, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = 4 * max_ten_J_rownnz

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_size: wp.array2d[wp.vec3],
    site_bodyid: wp.array[int],
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    wrap_type: wp.array[int],
    wrap_objid: wp.array[int],
    wrap_prm: wp.array[float],
    tendon_geom_adr: wp.array[int],
    wrap_geom_adr: wp.array[int],
    wrap_pulley_scale: wp.array[float],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    site_xpos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    # Data out:
    ten_J_out: wp.array2d[float],
    ten_length_out: wp.array2d[float],
    # Out:
    wrap_geom_xpos_out: wp.array2d[wp.spatial_vector],
  ):
    worldid, elementid = wp.tid()
    wrap_adr = wrap_geom_adr[elementid]
    tenid = tendon_geom_adr[elementid]

    # pulley scaling
    pulley_scale = wrap_pulley_scale[wrap_adr]

    # site-geom-site
    wrap_objid_site0 = wrap_objid[wrap_adr - 1]
    wrap_objid_geom = wrap_objid[wrap_adr + 0]
    wrap_objid_site1 = wrap_objid[wrap_adr + 1]

    # get site positions before and after geom
    site_pnt0 = site_xpos_in[worldid, wrap_objid_site0]
    site_pnt1 = site_xpos_in[worldid, wrap_objid_site1]

    # get geom information
    geom_xpos = geom_xpos_in[worldid, wrap_objid_geom]
    geom_xmat = geom_xmat_in[worldid, wrap_objid_geom]
    geomsize = geom_size[worldid % geom_size.shape[0], wrap_objid_geom][0]
    geom_type = wrap_type[wrap_adr]

    # get body ids for site-geom-site instances
    bodyid_site0 = site_bodyid[wrap_objid_site0]
    bodyid_geom = geom_bodyid[wrap_objid_geom]
    bodyid_site1 = site_bodyid[wrap_objid_site1]

    # find wrap object sidesite (if it exists)
    sideid = int(wp.round(wrap_prm[wrap_adr]))
    if sideid >= 0:
      side = site_xpos_in[worldid, sideid]
    else:
      side = wp.vec3(MJ_MAXVAL)

    # compute geom wrap length and connect points (if wrap occurs)
    length_geomgeom, geom_pnt0, geom_pnt1 = util_misc.wrap(
      site_pnt0, site_pnt1, geom_xpos, geom_xmat, geomsize, geom_type, side
    )

    # store geom points
    wrap_geom_xpos_out[worldid, elementid] = wp.spatial_vector(geom_pnt0, geom_pnt1)

    rownnz = ten_J_rownnz[tenid]
    rowadr = ten_J_rowadr[tenid]

    if length_geomgeom >= 0.0:
      dif_sitegeom = geom_pnt0 - site_pnt0
      dif_geomsite = site_pnt1 - geom_pnt1
      vec_sitegeom, length_sitegeom = math.normalize_with_norm(dif_sitegeom)
      vec_geomsite, length_geomsite = math.normalize_with_norm(dif_geomsite)

      # length
      length_sitegeomsite = length_sitegeom + length_geomgeom + length_geomsite

      if length_sitegeomsite:
        wp.atomic_add(ten_length_out[worldid], tenid, length_sitegeomsite * pulley_scale)

      # moment
      if length_sitegeom < MJ_MINVAL:
        vec_sitegeom = wp.vec3(1.0, 0.0, 0.0)

      if length_geomsite < MJ_MINVAL:
        vec_geomsite = wp.vec3(1.0, 0.0, 0.0)

      dif_body_sitegeom = bodyid_site0 != bodyid_geom
      dif_body_geomsite = bodyid_geom != bodyid_site1

      # site-geom segment
      if dif_body_sitegeom:
        offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]
        offset_geom0 = geom_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_geom]]
        _accumulate_jac_chain(
          body_parentid,
          body_dofnum,
          body_dofadr,
          ten_J_colind,
          cdof_in,
          offset_site0,
          vec_sitegeom,
          bodyid_site0,
          rowadr,
          rownnz,
          -pulley_scale,
          worldid,
          ten_J_out,
        )
        _accumulate_jac_chain(
          body_parentid,
          body_dofnum,
          body_dofadr,
          ten_J_colind,
          cdof_in,
          offset_geom0,
          vec_sitegeom,
          bodyid_geom,
          rowadr,
          rownnz,
          pulley_scale,
          worldid,
          ten_J_out,
        )

      # geom-site segment
      if dif_body_geomsite:
        offset_geom1 = geom_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_geom]]
        offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]
        _accumulate_jac_chain(
          body_parentid,
          body_dofnum,
          body_dofadr,
          ten_J_colind,
          cdof_in,
          offset_geom1,
          vec_geomsite,
          bodyid_geom,
          rowadr,
          rownnz,
          -pulley_scale,
          worldid,
          ten_J_out,
        )
        _accumulate_jac_chain(
          body_parentid,
          body_dofnum,
          body_dofadr,
          ten_J_colind,
          cdof_in,
          offset_site1,
          vec_geomsite,
          bodyid_site1,
          rowadr,
          rownnz,
          pulley_scale,
          worldid,
          ten_J_out,
        )
    else:
      dif_sitesite = site_pnt1 - site_pnt0
      vec_sitesite, length_sitesite = math.normalize_with_norm(dif_sitesite)

      # length
      if length_sitesite:
        wp.atomic_add(ten_length_out[worldid], tenid, length_sitesite * pulley_scale)

      # moment
      if length_sitesite < MJ_MINVAL:
        vec_sitesite = wp.vec3(1.0, 0.0, 0.0)

      if bodyid_site0 != bodyid_site1:
        offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]
        offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]
        _accumulate_jac_chain(
          body_parentid,
          body_dofnum,
          body_dofadr,
          ten_J_colind,
          cdof_in,
          offset_site0,
          vec_sitesite,
          bodyid_site0,
          rowadr,
          rownnz,
          -pulley_scale,
          worldid,
          ten_J_out,
        )
        _accumulate_jac_chain(
          body_parentid,
          body_dofnum,
          body_dofadr,
          ten_J_colind,
          cdof_in,
          offset_site1,
          vec_sitesite,
          bodyid_site1,
          rowadr,
          rownnz,
          pulley_scale,
          worldid,
          ten_J_out,
        )

  return kernel


@wp.kernel
def _spatial_tendon_wrap(
  # Model:
  ntendon: int,
  tendon_type: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  # In:
  wrap_geom_xpos_in: wp.array2d[wp.spatial_vector],
  # Data out:
  ten_wrapadr_out: wp.array2d[int],
  ten_wrapnum_out: wp.array2d[int],
  wrap_obj_out: wp.array2d[wp.vec2i],
  wrap_xpos_out: wp.array2d[wp.spatial_vector],
):
  worldid = wp.tid()

  wrapcount = int(0)
  wrapgeomid = int(0)
  for i in range(ntendon):
    ten_wrapadr_out[worldid, i] = wrapcount

    # process fixed tendon
    if tendon_type[i] == TendonType.FIXED:
      continue

    # process spatial tendon
    adr = tendon_adr[i]
    tendonnum = tendon_num[i]
    wrapnum = int(0)
    j = int(0)
    while j < tendonnum - 1:
      # get 1st and 2nd object
      type0 = wrap_type[adr + j + 0]
      type1 = wrap_type[adr + j + 1]
      id0 = wrap_objid[adr + j + 0]
      id1 = wrap_objid[adr + j + 1]

      # pulley
      pulley0 = type0 == WrapType.PULLEY
      if pulley0 or type1 == WrapType.PULLEY:
        if pulley0:
          row = wrapcount // 2
          col = wrapcount % 2
          wrap_xpos_out[worldid, row][3 * col + 0] = 0.0
          wrap_xpos_out[worldid, row][3 * col + 1] = 0.0
          wrap_xpos_out[worldid, row][3 * col + 2] = 0.0

          wrap_obj_out[worldid, row][col] = -2

          wrapnum += 1
          wrapcount += 1

        # move to next
        j += 1
        continue

      # init sequence; assume it starts with site
      wpnt_site0 = site_xpos_in[worldid, id0]

      # second object is geom: process site-geom-site
      if type1 == WrapType.SPHERE or type1 == WrapType.CYLINDER:
        wrap_geom_xpos = wrap_geom_xpos_in[worldid, wrapgeomid]
        wpnt_geom0 = wp.spatial_top(wrap_geom_xpos)
        wrapgeomid += 1

        wrapid = id1
        id1 = wrap_objid[adr + j + 2]
        if wp.norm_l2(wpnt_geom0) < MJ_MAXVAL:
          wpnt_geom1 = wp.spatial_bottom(wrap_geom_xpos)
          wpnt_site1 = site_xpos_in[worldid, id1]

          # assign to wrap
          row0 = (wrapcount + 0) // 2
          col0 = (wrapcount + 0) % 2
          row1 = (wrapcount + 1) // 2
          col1 = (wrapcount + 1) % 2
          row2 = (wrapcount + 2) // 2
          col2 = (wrapcount + 2) % 2
          row3 = (wrapcount + 3) // 2
          col3 = (wrapcount + 3) % 2

          wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]
          wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]
          wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]

          wrap_xpos_out[worldid, row1][3 * col1 + 0] = wpnt_geom0[0]
          wrap_xpos_out[worldid, row1][3 * col1 + 1] = wpnt_geom0[1]
          wrap_xpos_out[worldid, row1][3 * col1 + 2] = wpnt_geom0[2]

          wrap_xpos_out[worldid, row2][3 * col2 + 0] = wpnt_geom1[0]
          wrap_xpos_out[worldid, row2][3 * col2 + 1] = wpnt_geom1[1]
          wrap_xpos_out[worldid, row2][3 * col2 + 2] = wpnt_geom1[2]

          wrap_xpos_out[worldid, row3][3 * col3 + 0] = wpnt_site1[0]
          wrap_xpos_out[worldid, row3][3 * col3 + 1] = wpnt_site1[1]
          wrap_xpos_out[worldid, row3][3 * col3 + 2] = wpnt_site1[2]

          wrap_obj_out[worldid, row0][col0] = -1
          wrap_obj_out[worldid, row1][col1] = wrapid
          wrap_obj_out[worldid, row2][col2] = wrapid

          wrapnum += 3
          wrapcount += 3
          j += 2

        else:
          row0 = (wrapcount + 0) // 2
          col0 = (wrapcount + 0) % 2

          wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]
          wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]
          wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]

          wrap_obj_out[worldid, row0][col0] = -1

          wrapnum += 1
          wrapcount += 1
          j += 2

      else:
        row0 = (wrapcount + 0) // 2
        col0 = (wrapcount + 0) % 2

        wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]
        wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]
        wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]

        wrap_obj_out[worldid, row0][col0] = -1

        wrapnum += 1
        wrapcount += 1
        j += 1

      # assign last site before pulley or tendon end
      if adr + j + 1 < wrap_type.shape[0]:
        last_before_pulley = wrap_type[adr + j + 1] == WrapType.PULLEY
      else:
        last_before_pulley = False

      if j == tendonnum - 1 or last_before_pulley:
        row0 = (wrapcount + 0) // 2
        col0 = (wrapcount + 0) % 2

        wpnt_site1 = site_xpos_in[worldid, id1]
        wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site1[0]
        wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site1[1]
        wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site1[2]

        wrap_obj_out[worldid, row0][col0] = -1
        wrapnum += 1
        wrapcount += 1

    ten_wrapnum_out[worldid, i] = wrapnum


@event_scope
def tendon(m: Model, d: Data):
  """Computes tendon lengths and moments.

  Updates the tendon length and moment arrays for all tendons in the model, including joint,
  site, and geom tendons.
  """
  if not m.ntendon:
    return

  d.ten_length.zero_()
  d.ten_J.zero_()

  # Cartesian 3D points for geom wrap points
  wrap_geom_xpos = wp.empty((d.nworld, m.nwrap), dtype=wp.spatial_vector)

  # process joint tendons
  wp.launch(
    _joint_tendon(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.wrap_jnt_adr.size),
    inputs=[
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.wrap_objid,
      m.wrap_prm,
      m.tendon_jnt_adr,
      m.wrap_jnt_adr,
      d.qpos,
    ],
    outputs=[d.ten_J, d.ten_length],
  )

  spatial_site = m.wrap_site_pair_adr.size > 0
  spatial_geom = m.wrap_geom_adr.size > 0

  if spatial_site or spatial_geom:
    d.wrap_xpos.zero_()
    d.wrap_obj.zero_()

  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  # process spatial site tendons
  wp.launch(
    _spatial_site_tendon(m.max_ten_J_rownnz if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.wrap_site_pair_adr.size),
    inputs=[
      m.body_parentid,
      m.body_rootid,
      m.body_dofnum,
      m.body_dofadr,
      m.site_bodyid,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.wrap_objid,
      m.tendon_site_pair_adr,
      m.wrap_site_pair_adr,
      m.wrap_pulley_scale,
      d.site_xpos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[d.ten_J, d.ten_length],
  )

  # process spatial geom tendons
  wp.launch(
    _spatial_geom_tendon(m.max_ten_J_rownnz if det_atomics else 0, det_atomics),
    dim=(d.nworld, m.wrap_geom_adr.size),
    inputs=[
      m.body_parentid,
      m.body_rootid,
      m.body_dofnum,
      m.body_dofadr,
      m.geom_bodyid,
      m.geom_size,
      m.site_bodyid,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.wrap_type,
      m.wrap_objid,
      m.wrap_prm,
      m.tendon_geom_adr,
      m.wrap_geom_adr,
      m.wrap_pulley_scale,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[d.ten_J, d.ten_length, wrap_geom_xpos],
  )

  if spatial_site or spatial_geom:
    wp.launch(
      _spatial_tendon_wrap,
      dim=d.nworld,
      inputs=[m.ntendon, m.tendon_type, m.tendon_adr, m.tendon_num, m.wrap_type, m.wrap_objid, d.site_xpos, wrap_geom_xpos],
      outputs=[d.ten_wrapadr, d.ten_wrapnum, d.wrap_obj, d.wrap_xpos],
    )
