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
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive_core import tri_weights
from mujoco.mjx.third_party.mujoco_warp._src.math import motion_cross
from mujoco.mjx.third_party.mujoco_warp._src.types import FLEX_INTERP_PHYS_JAC
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import ConeType
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DynType
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import JointType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import State
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.types import vec10
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


# TODO(team): kernel analyzer array slice?
@wp.func
def next_act(
  # Model:
  opt_timestep: float,  # kernel_analyzer: ignore
  actuator_dyntype: int,  # kernel_analyzer: ignore
  actuator_dynprm: vec10,  # kernel_analyzer: ignore
  actuator_actrange: wp.vec2,  # kernel_analyzer: ignore
  # Data In:
  act_in: float,  # kernel_analyzer: ignore
  act_dot_in: float,  # kernel_analyzer: ignore
  # In:
  act_dot_scale: float,
  clamp: bool,
) -> float:
  # advance actuation
  if actuator_dyntype == DynType.FILTEREXACT:
    tau = wp.max(MJ_MINVAL, actuator_dynprm[0])
    act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))
  elif actuator_dyntype == DynType.USER:
    return act_in
  else:
    act = act_in + act_dot_scale * act_dot_in * opt_timestep

  # clamp to actrange
  if clamp:
    act = wp.clamp(act, actuator_actrange[0], actuator_actrange[1])

  return act


@wp.func
def mat33_to_quat_polar(F: wp.mat33) -> wp.quat:
  cell_quat = wp.quat(0.0, 0.0, 0.0, 1.0)
  for _iter in range(50):
    rot = wp.quat_to_matrix(cell_quat)
    rot_t = wp.transpose(rot)
    col1_rot = rot_t[0]
    col2_rot = rot_t[1]
    col3_rot = rot_t[2]
    F_t = wp.transpose(F)
    col1_mat = F_t[0]
    col2_mat = F_t[1]
    col3_mat = F_t[2]

    omega = wp.cross(col1_rot, col1_mat) + wp.cross(col2_rot, col2_mat) + wp.cross(col3_rot, col3_mat)
    trace = wp.dot(col1_rot, col1_mat) + wp.dot(col2_rot, col2_mat) + wp.dot(col3_rot, col3_mat)
    denom = wp.abs(trace) + 1.0e-10
    omega = omega / denom

    w = wp.length(omega)
    if w < 1.0e-6:
      R_t_F = rot_t @ F
      sym = 0.5 * (R_t_F + wp.transpose(R_t_F))

      # fast path: T = tr(S)*I - S is strictly positive definite iff rotation maximizes tr(R'*M)
      T00 = sym[1, 1] + sym[2, 2]
      T11 = sym[0, 0] + sym[2, 2]
      T22 = sym[0, 0] + sym[1, 1]
      detT = (
        T00 * (T11 * T22 - sym[1, 2] * sym[1, 2])
        - sym[0, 1] * sym[0, 1] * T22
        - sym[0, 2] * sym[0, 2] * T11
        - 2.0 * sym[0, 1] * sym[0, 2] * sym[1, 2]
      )
      scale = wp.abs(sym[0, 0]) + wp.abs(sym[1, 1]) + wp.abs(sym[2, 2]) + MJ_MINVAL
      tol = 1.0e-6 * scale
      tol2 = tol * scale
      tol3 = tol2 * scale
      if T00 > tol and (T00 * T11 - sym[0, 1] * sym[0, 1]) > tol2 and detT > tol3:
        break

      eigval, eigvec = math.eig3(sym)
      lambda_max = eigval[0]
      tol = 1.0e-6 * (wp.abs(lambda_max) + wp.abs(trace) + MJ_MINVAL)
      if lambda_max <= trace + tol:
        break

      # stationary point (minimum or saddle): flip 180 degrees about rot * top eigenvector, which
      # strictly increases the trace
      v_max = eigvec[:, 0]
      axis = wp.normalize(rot @ v_max)
      qflip = wp.quat(axis[0], axis[1], axis[2], 0.0)
      cell_quat = wp.normalize(qflip * cell_quat)
      continue

    axis = omega / w
    half_w = 0.5 * w
    qrot = wp.quat(
      axis[0] * wp.sin(half_w),
      axis[1] * wp.sin(half_w),
      axis[2] * wp.sin(half_w),
      wp.cos(half_w),
    )
    cell_quat = wp.normalize(qrot * cell_quat)
  return cell_quat


@wp.func
def compute_interp_cell_quat(
  # Model:
  flex_node0: wp.array[wp.vec3],
  # Data in:
  flexnode_xpos_in: wp.array2d[wp.vec3],
  # In:
  order: int,
  ci: int,
  cj: int,
  ck: int,
  cy: int,
  cz: int,
  ny_g: int,
  nz_g: int,
  nstart: int,
  worldid: int,
) -> wp.quat:
  """Computes corotational cell quaternion from deformation gradient at cell center."""
  npc = (order + 1) * (order + 1) * (order + 1)
  F = wp.mat33(0.0)
  idx = int(0)
  for li in range(order + 1):
    for lj in range(order + 1):
      for lk in range(order + 1):
        if idx < npc:
          gi = ci * order + li
          gj = cj * order + lj
          gk = ck * order + lk
          gidx = gi * ny_g * nz_g + gj * nz_g + gk

          node_pos = flexnode_xpos_in[worldid, nstart + gidx]

          dphi_x = float(li - 1) if order == 2 else (-1.0 + 2.0 * float(li))
          dphi_y = float(lj - 1) if order == 2 else (-1.0 + 2.0 * float(lj))
          dphi_z = float(lk - 1) if order == 2 else (-1.0 + 2.0 * float(lk))
          phi_x = wp.where(li == 1, 1.0, 0.0) if order == 2 else 0.5
          phi_y = wp.where(lj == 1, 1.0, 0.0) if order == 2 else 0.5
          phi_z = wp.where(lk == 1, 1.0, 0.0) if order == 2 else 0.5

          grad_x = dphi_x * phi_y * phi_z
          grad_y = phi_x * dphi_y * phi_z
          grad_z = phi_x * phi_y * dphi_z

          for r in range(3):
            F[r, 0] += node_pos[r] * grad_x
            F[r, 1] += node_pos[r] * grad_y
            F[r, 2] += node_pos[r] * grad_z

          idx += 1

  if wp.static(FLEX_INTERP_PHYS_JAC):
    node_0 = nstart + (ci * order) * ny_g * nz_g + (cj * order) * nz_g + (ck * order)
    node_last = nstart + ((ci + 1) * order) * ny_g * nz_g + ((cj + 1) * order) * nz_g + ((ck + 1) * order)
    dref = flex_node0[node_last] - flex_node0[node_0]
    for c in range(3):
      dc = dref[c]
      if dc > MJ_MINVAL:
        inv_dc = 1.0 / dc
        F[0, c] *= inv_dc
        F[1, c] *= inv_dc
        F[2, c] *= inv_dc

  return mat33_to_quat_polar(F)


@wp.func
def rod_parallel_transport(
  # In:
  t_from: wp.vec3,
  t_to: wp.vec3,
  u_in: wp.vec3,
) -> Tuple[bool, wp.vec3, wp.vec3]:
  """Parallel-transports unit normal u_in orthogonal to t_from onto t_to; False if antipodal."""
  denom = 1.0 + wp.dot(t_from, t_to)
  if denom < MJ_MINVAL:
    return False, wp.vec3(0.0), wp.vec3(0.0)
  u_raw = u_in - (wp.dot(t_to, u_in) / denom) * (t_from + t_to)
  nu = wp.length(u_raw)
  if nu < MJ_MINVAL:
    return False, wp.vec3(0.0), wp.vec3(0.0)
  u_out = u_raw / nu
  v_out = wp.cross(t_to, u_out)
  return True, u_out, v_out


@wp.func
def rod_has_twist(
  # Model:
  flex_rod_has_twist: wp.array[bool],
  # In:
  f: int,
) -> bool:
  """Returns True if flex f is a 1D open DER chain with per-edge twist DOFs or box section."""
  return bool(flex_rod_has_twist[f])


@wp.func
def rod_box_size(
  # Model:
  flex_rod_is_box: wp.array[bool],
  flex_rod_box_size: wp.array[wp.vec2],
  # In:
  f: int,
) -> Tuple[bool, wp.vec2]:
  """Returns (True, (a_half, b_half)) if flex f is a 1D DER rod with a box cross-section."""
  return bool(flex_rod_is_box[f]), flex_rod_box_size[f]


@wp.func
def rod_clamped_ends(
  # Model:
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  # In:
  f: int,
) -> Tuple[bool, bool]:
  """Returns (root_clamped, tail_clamped) for 1D DER rod flex f."""
  return bool(flex_rod_root_clamped[f]), bool(flex_rod_tail_clamped[f])


@wp.func
def rod_chain_twist_jnt(
  # Model:
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # In:
  f: int,
  c: int,
) -> int:
  """Twist hinge joint ID of chain edge c of 1D DER rod f, or -1 if none."""
  nedge = flex_rod_edgenum[f]
  if c < 0 or c >= nedge:
    return -1
  return flex_rodedge_jnt[flex_rod_edgeadr[f] + c]


@wp.func
def rod_chain_twist_dof(
  # Model:
  jnt_dofadr: wp.array[int],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # In:
  f: int,
  c: int,
) -> int:
  """DOF address of the twist hinge of chain edge c of 1D DER rod f, or -1 if none."""
  jnt = rod_chain_twist_jnt(flex_rod_edgeadr, flex_rod_edgenum, flex_rodedge_jnt, f, c)
  return jnt_dofadr[jnt] if jnt >= 0 else -1


@wp.func
def rod_edge_twist_jnt(
  # Model:
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  # In:
  f: int,
  elem: int,
) -> Tuple[int, float]:
  """Returns (jnt_id, sgn_chain) for 1D element elem of DER rod flex f (mj_rodEdgeTwistJnt)."""
  if elem < 0 or elem >= flex_elemnum[f]:
    return -1, 1.0
  info = flex_elem_rodedge[flex_elemadr[f] + elem]
  c = info[0]
  sgn = float(info[1])
  if c < 0 or c >= flex_rod_edgenum[f]:
    return -1, 1.0
  return flex_rodedge_jnt[flex_rod_edgeadr[f] + c], sgn


@wp.func
def rod_vert_elem(
  # Model:
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_vert_rodelem: wp.array[int],
  # In:
  f: int,
  v: int,
) -> int:
  """Element of 1D flex f incident to flex-local vertex v, or -1 if none (mj_rodVertElem)."""
  if v < 0 or v >= flex_vertnum[f]:
    return -1
  return flex_vert_rodelem[flex_vertadr[f] + v]


@wp.func
def rod_edge_state(
  # Data in:
  plastic_in: wp.array2d[float],
  # In:
  worldid: int,
  padr: int,
  c: int,
) -> Tuple[bool, wp.vec3, wp.vec3]:
  """Time-parallel state (t_old, u_old) of chain edge c in d.plastic; False if uninitialized."""
  if padr < 0:
    return False, wp.vec3(0.0), wp.vec3(0.0)
  base = padr + 6 * c
  t_s = wp.vec3(
    plastic_in[worldid, base + 0],
    plastic_in[worldid, base + 1],
    plastic_in[worldid, base + 2],
  )
  u_s = wp.vec3(
    plastic_in[worldid, base + 3],
    plastic_in[worldid, base + 4],
    plastic_in[worldid, base + 5],
  )
  if wp.dot(t_s, t_s) > 0.25 and wp.dot(u_s, u_s) > 0.25:
    return True, t_s, u_s
  return False, wp.vec3(0.0), wp.vec3(0.0)


@wp.func
def rod_edge_director(
  # Model:
  flex_vertadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  # Data in:
  plastic_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
  flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
  flex_rod_tail_loc_in: wp.array3d[wp.vec3],
  # In:
  worldid: int,
  f: int,
  c: int,
) -> Tuple[bool, wp.vec3, wp.vec3, wp.vec3]:
  """Current tangent t and director (u, v) of chain edge c of 1D rod f (rodEdgeDirector)."""
  nedge = flex_rod_edgenum[f]
  if c < 0 or c >= nedge:
    return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
  nhinge = nedge - 1
  va = flex_vertadr[f]
  eadr = flex_rod_edgeadr[f]
  padr = flex_plasticadr[f]
  root_clamped = flex_rod_root_clamped[f]
  tail_clamped = flex_rod_tail_clamped[f]

  t_out = wp.vec3(0.0)
  u_out = wp.vec3(0.0)
  v_out = wp.vec3(0.0)

  k = c
  while k >= 0:
    verts_k = flex_rodedge_vert[eadr + k]
    ek = flexvert_xpos_in[worldid, va + verts_k[1]] - flexvert_xpos_in[worldid, va + verts_k[0]]
    lk = wp.length(ek)
    if lk < MJ_MINVAL:
      return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
    t_out = ek / lk

    if k == 0 and root_clamped:
      t0_loc = flex_rod_frame0_loc_in[worldid, f, 0]
      if wp.dot(t0_loc, t0_loc) < 0.25:
        return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
      b0 = flex_vertbodyid[va + verts_k[0]]
      R_b0 = xmat_in[worldid, b0]
      t0 = R_b0 @ t0_loc
      u0 = R_b0 @ flex_rod_frame0_loc_in[worldid, f, 1]
      ok_pt, u_out, v_out = rod_parallel_transport(t0, t_out, u0)
      if not ok_pt:
        return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
      break

    if k == nhinge and tail_clamped:
      tT_loc = flex_rod_tail_loc_in[worldid, f, 0]
      if wp.dot(tT_loc, tT_loc) < 0.25:
        return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
      bT = flex_vertbodyid[va + verts_k[1]]
      R_bT = xmat_in[worldid, bT]
      tT = R_bT @ tT_loc
      uT = R_bT @ flex_rod_tail_loc_in[worldid, f, 1]
      ok_pt, u_out, v_out = rod_parallel_transport(tT, t_out, uT)
      if not ok_pt:
        return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
      break

    has_st, t_s, u_s = rod_edge_state(plastic_in, worldid, padr, k)
    if has_st:
      ok_pt, u_out, v_out = rod_parallel_transport(t_s, t_out, u_s)
      if ok_pt:
        break

    if k == 0:
      t0_ref = flex_rod_frame0_ref_in[worldid, f, 0]
      u0_ref = flex_rod_frame0_ref_in[worldid, f, 1]
      ok_pt = bool(False)
      if wp.dot(t0_ref, t0_ref) > 0.25:
        ok_pt, u_out, v_out = rod_parallel_transport(t0_ref, t_out, u0_ref)
      if not ok_pt:
        u_out, v_out = math.orthogonals(t_out)
      break

    k -= 1

  e = k + 1
  while e <= c:
    verts_e = flex_rodedge_vert[eadr + e]
    ee = flexvert_xpos_in[worldid, va + verts_e[1]] - flexvert_xpos_in[worldid, va + verts_e[0]]
    le = wp.length(ee)
    if le < MJ_MINVAL:
      return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
    t_e = ee / le
    ok_pt, u_e, v_e = rod_parallel_transport(t_out, t_e, u_out)
    if not ok_pt:
      u_e, v_e = math.orthogonals(t_e)
    t_out = t_e
    u_out = u_e
    v_out = v_e
    e += 1

  return True, t_out, u_out, v_out


@wp.func
def rod_chain_edge_director(
  # Model:
  flex_vertadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  # Data in:
  plastic_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
  flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
  flex_rod_tail_loc_in: wp.array3d[wp.vec3],
  # In:
  worldid: int,
  f: int,
  e: int,
  t_prev_edge: wp.vec3,
  u_prev_edge: wp.vec3,
) -> Tuple[bool, bool, wp.vec3, wp.vec3, wp.vec3, wp.vec3, float]:
  """One streaming step of mj_rodChain's per-edge reference director propagation.

  Returns (ok, anchored, t_prev, t_e, u_e, v_e, l_e) for chain edge e of rod f, given edge e-1's
  current director (t_prev_edge, u_prev_edge) when e > 0.
  """
  z3 = wp.vec3(0.0)
  eadr = flex_rod_edgeadr[f]
  nedge = flex_rod_edgenum[f]
  nhinge = nedge - 1
  va = flex_vertadr[f]
  verts_e = flex_rodedge_vert[eadr + e]
  de = flexvert_xpos_in[worldid, va + verts_e[1]] - flexvert_xpos_in[worldid, va + verts_e[0]]
  le = wp.length(de)
  if le < MJ_MINVAL:
    return False, False, z3, z3, z3, z3, 0.0
  te = de / le

  if (e == 0 and flex_rod_root_clamped[f]) or (e == nhinge and flex_rod_tail_clamped[f]):
    if e == 0:
      tc_loc = flex_rod_frame0_loc_in[worldid, f, 0]
      if wp.dot(tc_loc, tc_loc) < 0.25:
        return False, False, z3, z3, z3, z3, 0.0
      bc = flex_vertbodyid[va + verts_e[0]]
      R_bc = xmat_in[worldid, bc]
      tc = R_bc @ tc_loc
      uc = R_bc @ flex_rod_frame0_loc_in[worldid, f, 1]
    else:
      tc_loc = flex_rod_tail_loc_in[worldid, f, 0]
      if wp.dot(tc_loc, tc_loc) < 0.25:
        return False, False, z3, z3, z3, z3, 0.0
      bc = flex_vertbodyid[va + verts_e[1]]
      R_bc = xmat_in[worldid, bc]
      tc = R_bc @ tc_loc
      uc = R_bc @ flex_rod_tail_loc_in[worldid, f, 1]
    ok_pt, ue, ve = rod_parallel_transport(tc, te, uc)
    if not ok_pt:
      return False, False, z3, z3, z3, z3, 0.0
    return True, True, tc, te, ue, ve, le

  has_st, t_s, u_s = rod_edge_state(plastic_in, worldid, flex_plasticadr[f], e)
  if has_st:
    ok_pt, ue, ve = rod_parallel_transport(t_s, te, u_s)
    if ok_pt:
      return True, True, t_s, te, ue, ve, le

  ok_pt = bool(False)
  ue = z3
  ve = z3
  if e > 0:
    ok_pt, ue, ve = rod_parallel_transport(t_prev_edge, te, u_prev_edge)
  else:
    t0_ref = flex_rod_frame0_ref_in[worldid, f, 0]
    u0_ref = flex_rod_frame0_ref_in[worldid, f, 1]
    if wp.dot(t0_ref, t0_ref) > 0.25:
      ok_pt, ue, ve = rod_parallel_transport(t0_ref, te, u0_ref)
  if not ok_pt:
    ue, ve = math.orthogonals(te)
  return True, False, te, te, ue, ve, le


@wp.func
def rod_box_elem(
  # Model:
  qpos0: wp.array2d[float],
  jnt_qposadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_rod_is_box: wp.array[bool],
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_box_size: wp.array[wp.vec2],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  # Data in:
  qpos_in: wp.array2d[float],
  plastic_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
  flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
  flex_rod_tail_loc_in: wp.array3d[wp.vec3],
  # In:
  worldid: int,
  f: int,
  e: int,
) -> Tuple[bool, wp.vec3, wp.mat33, wp.vec3]:
  """Computes world-space box pose and half-sizes for 1D element e of a box-section DER rod."""
  zero_mat = wp.mat33(0.0)
  is_box, box_2d = rod_box_size(flex_rod_is_box, flex_rod_box_size, f)
  if not is_box or e < 0 or e >= flex_elemnum[f]:
    return False, wp.vec3(0.0), zero_mat, wp.vec3(0.0)

  info = flex_elem_rodedge[flex_elemadr[f] + e]
  c_target = info[0]
  sgn = float(info[1])
  if c_target < 0 or c_target >= flex_rod_edgenum[f]:
    return False, wp.vec3(0.0), zero_mat, wp.vec3(0.0)

  ok_dir, t_curr, u_c, v_c = rod_edge_director(
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
    c_target,
  )
  if not ok_dir:
    return False, wp.vec3(0.0), zero_mat, wp.vec3(0.0)

  va = flex_vertadr[f]
  ed = flex_elemdataadr[f] + 2 * e
  x0 = flexvert_xpos_in[worldid, va + flex_elem[ed + 0]]
  x1 = flexvert_xpos_in[worldid, va + flex_elem[ed + 1]]
  le = wp.length(x1 - x0)
  if le < MJ_MINVAL:
    return False, wp.vec3(0.0), zero_mat, wp.vec3(0.0)

  pos = 0.5 * (x0 + x1)
  size = wp.vec3(box_2d[0], box_2d[1], 0.5 * le)

  theta_c = 0.0
  jnt_c = flex_rodedge_jnt[flex_rod_edgeadr[f] + c_target]
  if jnt_c >= 0:
    qadr = jnt_qposadr[jnt_c]
    qpos0_id = worldid % qpos0.shape[0]
    theta_c = qpos_in[worldid, qadr] - qpos0[qpos0_id, qadr]

  ct = wp.cos(theta_c)
  st = wp.sin(theta_c)
  m1 = ct * u_c + st * v_c
  m2 = -st * u_c + ct * v_c
  mat = wp.mat33(
    m1[0],
    sgn * m2[0],
    sgn * t_curr[0],
    m1[1],
    sgn * m2[1],
    sgn * t_curr[1],
    m1[2],
    sgn * m2[2],
    sgn * t_curr[2],
  )
  return True, pos, mat, size


@wp.func
def rod_contact_side_params(
  # Model:
  jnt_dofadr: wp.array[int],
  geom_type: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  plastic_in: wp.array2d[float],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  worldid: int,
  side: int,
  geom: wp.vec2i,
  flex: wp.vec2i,
  elem: wp.vec2i,
  vert: wp.vec2i,
  con_dist: float,
  con_pos: wp.vec3,
  con_normal: wp.vec3,
) -> Tuple[
  int,
  wp.vec3,
  wp.vec3,
  bool,
  int,
  int,
  wp.vec3,
  wp.vec3,
  float,
  float,
  float,
  wp.vec3,
  wp.vec3,
  wp.vec3,
  wp.vec3,
]:
  """Precomputes per-(contact, side) rod lever-arm Jacobian parameters (mj_rodContactJac)."""
  z3 = wp.vec3(0.0)
  f = flex[side]
  if f < 0 or flex_elemnum[f] <= 0 or not rod_has_twist(flex_rod_has_twist, f):
    return -1, z3, z3, False, -1, -1, z3, z3, 0.0, 0.0, 0.0, z3, z3, z3, z3

  va = flex_vertadr[f]
  elem_id = elem[side] if elem[side] >= 0 else (flex_vert_rodelem[va + vert[side]] if vert[side] >= 0 else -1)
  if elem_id < 0 or elem_id >= flex_elemnum[f]:
    return -1, z3, z3, False, -1, -1, z3, z3, 0.0, 0.0, 0.0, z3, z3, z3, z3

  ed = flex_elemdataadr[f] + 2 * elem_id
  ev0 = flex_elem[ed + 0]
  ev1 = flex_elem[ed + 1]
  if flex[1 - side] == f and (vert[1 - side] == ev0 or vert[1 - side] == ev1):
    return -1, z3, z3, False, -1, -1, z3, z3, 0.0, 0.0, 0.0, z3, z3, z3, z3

  x0 = flexvert_xpos_in[worldid, va + ev0]
  x1 = flexvert_xpos_in[worldid, va + ev1]
  evec = x1 - x0
  le = wp.length(evec)
  if le < MJ_MINVAL:
    return -1, z3, z3, False, -1, -1, z3, z3, 0.0, 0.0, 0.0, z3, z3, z3, z3

  inv_le = 1.0 / le
  t = evec * inv_le
  sgn = 1.0 if side == 1 else -1.0

  w0 = 0.0
  w1 = 0.0
  dvec = z3
  if vert[side] >= 0:
    w1 = 1.0 if vert[side] == ev1 else 0.0
    w0 = 1.0 - w1
    xk = x1 if w1 > 0.0 else x0
    dvec = con_pos - xk
    other_geom = geom[1 - side]
    if other_geom >= 0 and geom_type[other_geom] == int(GeomType.PLANE):
      dvec = dvec + (0.5 * sgn * con_dist) * con_normal
  else:
    ap = con_pos - x0
    w1 = wp.clamp(wp.dot(ap, t) * inv_le, 0.0, 1.0)
    w0 = 1.0 - w1
    p_axis = w0 * x0 + w1 * x1
    dvec = con_pos - p_axis

  # full lever arm dvec = r_perp + a t from the axis point to the surface point
  r_perp = dvec - wp.dot(dvec, t) * t
  t_cross_r = wp.cross(t, r_perp)

  jnt_e, sgn_chain = rod_edge_twist_jnt(
    flex_elemadr,
    flex_elemnum,
    flex_rod_edgeadr,
    flex_rod_edgenum,
    flex_rodedge_jnt,
    flex_elem_rodedge,
    f,
    elem_id,
  )
  dof_twist = jnt_dofadr[jnt_e] if jnt_e >= 0 else -1
  tsgn = sgn * sgn_chain
  dp_twist = tsgn * t_cross_r
  dr_twist = tsgn * t

  b0 = flex_vertbodyid[va + ev0]
  b1 = flex_vertbodyid[va + ev1]
  if b0 == b1:
    return dof_twist, dp_twist, dr_twist, False, -1, -1, z3, z3, 0.0, 0.0, 0.0, z3, z3, z3, z3

  c_target = flex_elem_rodedge[flex_elemadr[f] + elem_id][0]
  g_psi = z3
  if c_target >= 0:
    has_st, t_s, _u_s = rod_edge_state(plastic_in, worldid, flex_plasticadr[f], c_target)
    if has_st:
      t_chain = sgn_chain * t
      denom = le * (1.0 + wp.dot(t_s, t_chain))
      if denom >= MJ_MINVAL:
        g_psi = (-1.0 / denom) * wp.cross(t_s, t_chain)

  return (
    dof_twist,
    dp_twist,
    dr_twist,
    True,
    b0,
    b1,
    x0,
    x1,
    w0,
    w1,
    inv_le,
    t,
    dvec,
    t_cross_r,
    g_psi,
  )


@wp.func
def rod_contact_dof_delta(
  # In:
  dof_axis: wp.spatial_vector,
  xk_offset: wp.vec3,
  con_minus_xk: wp.vec3,
  t: wp.vec3,
  dvec: wp.vec3,
  t_cross_r: wp.vec3,
  g_psi: wp.vec3,
  sgn_wk: float,
  sgn_esgn: float,
  inv_le: float,
) -> Tuple[wp.vec3, wp.vec3]:
  """Per-DOF (delta_jacp, delta_jacr) from one endpoint of a 1D DER rod element.

  The surface point at lever arm dvec = r_perp + a t moves with the cross-section, adding the
  longitudinal velocity (t x tdot) x dvec = (t (dvec . jv) - a jv) * (sgn_esgn / le).
  """
  cdof_ang = wp.spatial_top(dof_axis)
  cdof_lin = wp.spatial_bottom(dof_axis)
  jv = cdof_lin + wp.cross(cdof_ang, xk_offset)
  dpsi = sgn_esgn * wp.dot(g_psi, jv)
  coeff_t = -sgn_esgn * wp.dot(dvec, jv) * inv_le
  coeff_j = sgn_esgn * wp.dot(dvec, t) * inv_le
  dp = -sgn_wk * wp.cross(cdof_ang, con_minus_xk) + coeff_t * t + coeff_j * jv + dpsi * t_cross_r
  dr = -sgn_wk * cdof_ang + (sgn_esgn * inv_le) * wp.cross(t, jv) + dpsi * t
  return dp, dr


@wp.func
def _append_body_weight_zero(
  # In:
  body_ids: wp.vec4i,
  nb: int,
  b: int,
) -> Tuple[wp.vec4i, int]:
  if b < 0 or nb >= 4:
    return body_ids, nb
  for k in range(4):
    if k < nb and body_ids[k] == b:
      return body_ids, nb
  if nb == 0:
    return wp.vec4i(b, -1, -1, -1), 1
  if nb == 1:
    return wp.vec4i(body_ids[0], b, -1, -1), 2
  if nb == 2:
    return wp.vec4i(body_ids[0], body_ids[1], b, -1), 3
  return wp.vec4i(body_ids[0], body_ids[1], body_ids[2], b), 4


@cache_kernel
def mul_m_kernel(check_skip: bool):
  @wp.kernel(module="unique", grid_stride=False)
  def _mul_m(
    # Model:
    M_mulm_rowadr: wp.array[int],
    M_mulm_col: wp.array[int],
    M_mulm_madr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    # In:
    vec: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res: wp.array2d[float],
  ):
    """Sparse matmul: one thread per DOF, gather-based (no atomics)."""
    worldid, dofid = wp.tid()

    if wp.static(check_skip):
      if skip[worldid]:
        return

    # Gather all contributions (diagonal + off-diagonal).
    acc = float(0.0)
    start = M_mulm_rowadr[dofid]
    end = M_mulm_rowadr[dofid + 1]
    for k in range(start, end):
      col = M_mulm_col[k]
      madr = M_mulm_madr[k]
      acc += M_in[worldid, madr] * vec[worldid, col]

    res[worldid, dofid] = acc

  return _mul_m


@cache_kernel
def mul_m_dense(nv: int, check_skip: bool):
  @wp.kernel(module="unique", grid_stride=False)
  def _mul_m_dense(
    # Data in:
    M_in: wp.array3d[float],  # kernel_analyzer: ignore
    # In:
    vec: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res: wp.array2d[float],
  ):
    """Dense matmul for the compact active-DOF inertia block (nworld, nv, nv)."""
    worldid, i = wp.tid()

    if wp.static(check_skip):
      if skip[worldid]:
        return

    acc = float(0.0)
    for j in range(wp.static(nv)):
      acc += M_in[worldid, i, j] * vec[worldid, j]
    res[worldid, i] = acc

  return _mul_m_dense


@event_scope
def mul_m(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  skip: Optional[wp.array] = None,
  M: Optional[wp.array] = None,
):
  """Multiply vectors by inertia matrix; optionally skip per world.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    res: Result: M @ vec.
    vec: Input vector to multiply by M.
    skip: Per-world bitmask to skip computing output.
    M: Input matrix: M @ vec.
  """
  check_skip = skip is not None
  skip = skip or wp.empty(0, dtype=bool)

  if M is None:
    M = d.M

  if M.ndim == 3:
    # Dense compact active-DOF block (nworld, nv, nv) used by the compact solver.
    wp.launch(
      mul_m_dense(m.nv, check_skip),
      dim=(d.nworld, m.nv),
      inputs=[M, vec, skip],
      outputs=[res],
    )
  else:
    wp.launch(
      mul_m_kernel(check_skip),
      dim=(d.nworld, m.nv),
      inputs=[m.M_mulm_rowadr, m.M_mulm_col, m.M_mulm_madr, M, vec, skip],
      outputs=[res],
    )


@wp.kernel
def _apply_ft(
  # Model:
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_subtreenum: wp.array[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  ft_in: wp.array2d[wp.spatial_vector],
  flg_add: bool,
  # Out:
  qfrc_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  cdof = cdof_in[worldid, dofid]
  rotational_cdof = wp.vec3(cdof[0], cdof[1], cdof[2])
  jac = wp.spatial_vector(cdof[3], cdof[4], cdof[5], cdof[0], cdof[1], cdof[2])

  dofbodyid = dof_bodyid[dofid]
  subtree_com = subtree_com_in[worldid, body_rootid[dofbodyid]]
  accumul = float(0.0)

  # Bodies in a subtree are contiguous in MuJoCo's depth-first order.
  for bodyid in range(dofbodyid, dofbodyid + body_subtreenum[dofbodyid]):
    ft_body = ft_in[worldid, bodyid]
    if ft_body == wp.spatial_vector():
      continue
    offset = xipos_in[worldid, bodyid] - subtree_com
    cross_term = wp.cross(rotational_cdof, offset)
    accumul += wp.dot(jac, ft_body) + wp.dot(cross_term, wp.spatial_top(ft_body))

  if flg_add:
    qfrc_out[worldid, dofid] += accumul
  else:
    qfrc_out[worldid, dofid] = accumul


def apply_ft(m: Model, d: Data, ft: wp.array2d[wp.spatial_vector], qfrc: wp.array2d[float], flg_add: bool):
  wp.launch(
    kernel=_apply_ft,
    dim=(d.nworld, m.nv),
    inputs=[m.body_rootid, m.dof_bodyid, m.body_subtreenum, d.xipos, d.subtree_com, d.cdof, ft, flg_add],
    outputs=[qfrc],
  )


@event_scope
def xfrc_accumulate(m: Model, d: Data, qfrc: wp.array2d[float]):
  """Map applied forces at each body via Jacobians to dof space and accumulate.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    qfrc: Total applied force mapped to dof space.
  """
  apply_ft(m, d, d.xfrc_applied, qfrc, True)


@wp.func
def _decode_pyramid(njmax_in: int, pyramid: wp.array[float], efc_address: int, mu: vec5, condim: int) -> wp.spatial_vector:
  """Converts pyramid representation to contact force."""
  force = wp.spatial_vector()

  if condim == 1:
    force[0] = pyramid[efc_address]
    return force

  force[0] = float(0.0)
  for i in range(condim - 1):
    adr = 2 * i + efc_address
    if adr < njmax_in:
      dir1 = pyramid[adr]
    else:
      dir1 = 0.0
    if adr + 1 < njmax_in:
      dir2 = pyramid[adr + 1]
    else:
      dir2 = 0.0
    force[0] += dir1 + dir2
    force[i + 1] = (dir1 - dir2) * mu[i]

  return force


@wp.func
def contact_force_fn(
  # Model:
  opt_cone: int,
  # Data in:
  contact_frame_in: wp.array[wp.mat33],
  contact_friction_in: wp.array[vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_adhesion_in: wp.array[float],
  efc_force_in: wp.array2d[float],
  njmax_in: int,
  nacon_in: wp.array[int],
  # In:
  worldid: int,
  contact_id: int,
  to_world_frame: bool,
) -> wp.spatial_vector:
  """Extract 6D force:torque for one contact, in contact frame by default."""
  force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  condim = contact_dim_in[contact_id]
  efc_address = contact_efc_address_in[contact_id, 0]

  if contact_id >= 0 and contact_id <= nacon_in[0] and efc_address >= 0:
    if opt_cone == ConeType.PYRAMIDAL:
      force = _decode_pyramid(
        njmax_in,
        efc_force_in[worldid],
        efc_address,
        contact_friction_in[contact_id],
        condim,
      )
    else:
      for i in range(condim):
        if contact_efc_address_in[contact_id, i] < njmax_in:
          force[i] = efc_force_in[worldid, contact_efc_address_in[contact_id, i]]

    # report net interface force: solver cone force minus adhesive pull
    force[0] -= contact_adhesion_in[contact_id]

  if to_world_frame:
    # Transform both top and bottom parts of spatial vector by the full contact frame matrix
    t = wp.spatial_top(force) @ contact_frame_in[contact_id]
    b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]
    force = wp.spatial_vector(t, b)

  return force


@wp.kernel
def contact_force_kernel(
  # Model:
  opt_cone: int,
  # Data in:
  contact_frame_in: wp.array[wp.mat33],
  contact_friction_in: wp.array[vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_adhesion_in: wp.array[float],
  efc_force_in: wp.array2d[float],
  njmax_in: int,
  nacon_in: wp.array[int],
  # In:
  contact_ids: wp.array[int],
  to_world_frame: bool,
  # Out:
  out: wp.array[wp.spatial_vector],
):
  tid = wp.tid()

  contactid = contact_ids[tid]

  if contactid >= nacon_in[0]:
    return

  worldid = contact_worldid_in[contactid]

  out[tid] = contact_force_fn(
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
    to_world_frame,
  )


def contact_force(m: Model, d: Data, contact_ids: wp.array[int], to_world_frame: bool, force: wp.array[wp.spatial_vector]):
  """Compute forces for contacts in Data.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    contact_ids: IDs for each contact.
    to_world_frame: If True, map force from contact to world frame.
    force: Contact forces.
  """
  wp.launch(
    contact_force_kernel,
    dim=contact_ids.size,
    inputs=[
      m.opt.cone,
      d.contact.frame,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.contact.worldid,
      d.contact.adhesion,
      d.efc.force,
      d.njmax,
      d.nacon,
      contact_ids,
      to_world_frame,
    ],
    outputs=[force],
  )


@wp.func
def transform_force(force: wp.vec3, torque: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:
  return wp.spatial_vector(torque - wp.cross(offset, force), force)


@wp.func
def transform_force(frc: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector:
  force = wp.spatial_top(frc)
  torque = wp.spatial_bottom(frc)
  return transform_force(force, torque, offset)


@wp.func
def _compute_jacp(cdof_clip: wp.spatial_vector, offset: wp.vec3, affect: int) -> wp.vec3:
  if affect == 0:
    return wp.vec3(0.0)
  cdof_lin = wp.spatial_bottom(cdof_clip)
  cdof_ang = wp.spatial_top(cdof_clip)
  return cdof_lin + wp.cross(cdof_ang, offset)


@wp.func
def _compute_jacr(cdof_clip: wp.spatial_vector, affect: int) -> wp.vec3:
  if affect == 0:
    return wp.vec3(0.0)
  return wp.spatial_top(cdof_clip)


@wp.func
def jac_dof(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  point: wp.vec3,
  bodyid: int,
  dofid: int,
  worldid: int,
) -> Tuple[wp.vec3, wp.vec3]:
  if body_isdofancestor[bodyid, dofid] == 0:
    return wp.vec3(0.0), wp.vec3(0.0)

  offset = point - wp.vec3(subtree_com_in[worldid, body_rootid[bodyid]])

  cdof = cdof_in[worldid, dofid]
  cdof_ang = wp.spatial_top(cdof)
  cdof_lin = wp.spatial_bottom(cdof)

  jacp = cdof_lin + wp.cross(cdof_ang, offset)
  jacr = cdof_ang

  return jacp, jacr


@cache_kernel
def _make_jac_kernel(has_jacp: bool, has_jacr: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def _jac(
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    dof_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    # Data in:
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    # In:
    point_in: wp.array[wp.vec3],
    bodyid_in: wp.array[int],
    # Out:
    jacp_out: wp.array3d[float],
    jacr_out: wp.array3d[float],
  ):
    worldid, dofid = wp.tid()

    jacp_val, jacr_val = jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      point_in[worldid],
      bodyid_in[worldid],
      dofid,
      worldid,
    )

    if wp.static(has_jacp):
      jacp_out[worldid, 0, dofid] = jacp_val[0]
      jacp_out[worldid, 1, dofid] = jacp_val[1]
      jacp_out[worldid, 2, dofid] = jacp_val[2]

    if wp.static(has_jacr):
      jacr_out[worldid, 0, dofid] = jacr_val[0]
      jacr_out[worldid, 1, dofid] = jacr_val[1]
      jacr_out[worldid, 2, dofid] = jacr_val[2]

  return _jac


@event_scope
def jac(
  m: Model,
  d: Data,
  jacp: wp.array | None,  # wp.array3d[float]
  jacr: wp.array | None,  # wp.array3d[float]
  point: wp.array[wp.vec3],
  body: wp.array[int],
):
  """Compute translational and rotational Jacobian for point on body.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state (device).
    jacp: Output translational Jacobian (optional).
    jacr: Output rotational Jacobian (optional).
    point: 3D point in global coordinates.
    body: Body ID for each world.
  """
  kernel = _make_jac_kernel(jacp is not None, jacr is not None)

  jacp_arr = jacp or wp.empty((0, 0, 0), dtype=float)
  jacr_arr = jacr or wp.empty((0, 0, 0), dtype=float)

  wp.launch(
    kernel,
    dim=(d.nworld, m.nv),
    inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor, d.subtree_com, d.cdof, point, body],
    outputs=[jacp_arr, jacr_arr],
  )


@wp.func
def jac_dot_dof(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_bodyid: wp.array[int],
  dof_jntid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  point: wp.vec3,
  bodyid: int,
  dofid: int,
  worldid: int,
) -> Tuple[wp.vec3, wp.vec3]:
  if body_isdofancestor[bodyid, dofid] == 0:
    return wp.vec3(0.0), wp.vec3(0.0)

  com = subtree_com_in[worldid, body_rootid[bodyid]]
  offset = point - com

  # transform spatial
  cvel = cvel_in[worldid, bodyid]
  pvel_lin = wp.spatial_bottom(cvel) - wp.cross(offset, wp.spatial_top(cvel))

  cdof = cdof_in[worldid, dofid]
  cdof_dot = cdof_dot_in[worldid, dofid]

  # check for quaternion
  dofjntid = dof_jntid[dofid]
  jnttype = jnt_type[dofjntid]
  jntdofadr = jnt_dofadr[dofjntid]

  if (jnttype == JointType.BALL) or ((jnttype == JointType.FREE) and dofid >= jntdofadr + 3):
    # compute cdof_dot for quaternion (use current body cvel)
    cvel = cvel_in[worldid, dof_bodyid[dofid]]
    cdof_dot = motion_cross(cvel, cdof)

  cdof_dot_ang = wp.spatial_top(cdof_dot)
  cdof_dot_lin = wp.spatial_bottom(cdof_dot)

  # construct translational Jacobian (correct for rotation)
  # first correction term, account for varying cdof
  correction1 = wp.cross(cdof_dot_ang, offset)

  # second correction term, account for point translational velocity
  correction2 = wp.cross(wp.spatial_top(cdof), pvel_lin)

  jacp = cdof_dot_lin + correction1 + correction2
  jacr = cdof_dot_ang

  return jacp, jacr


def get_state(m: Model, d: Data, state: wp.array2d[float], sig: int, active: Optional[wp.array] = None):
  """Copy concatenated state components specified by sig from Data into state.

  The bits of the integer sig correspond to element fields of State.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output information (device).
    state: Concatenation of state components.
    sig: Bitflag specifying state components.
    active: Per-world bitmask for getting state.
  """
  if sig >= (1 << State.NSTATE):
    raise ValueError(f"invalid state signature {sig} >= 2^mjNSTATE")

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def _get_state(
    # Model:
    nq: int,
    nv: int,
    nu: int,
    na: int,
    nbody: int,
    neq: int,
    nmocap: int,
    nplastic: int,
    nuserdata: int,
    nhistory: int,
    # Data in:
    time_in: wp.array[float],
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    act_in: wp.array2d[float],
    history_in: wp.array2d[float],
    plastic_in: wp.array2d[float],
    qacc_warmstart_in: wp.array2d[float],
    ctrl_in: wp.array2d[float],
    qfrc_applied_in: wp.array2d[float],
    xfrc_applied_in: wp.array2d[wp.spatial_vector],
    eq_active_in: wp.array2d[bool],
    mocap_pos_in: wp.array2d[wp.vec3],
    mocap_quat_in: wp.array2d[wp.quat],
    userdata_in: wp.array2d[float],
    # In:
    sig_in: int,
    active_in: wp.array[bool],
    # Out:
    state_out: wp.array2d[float],
  ):
    worldid = wp.tid()

    if wp.static(active is not None):
      if not active_in[worldid]:
        return

    adr = int(0)
    for i in range(State.NSTATE.value):
      element = 1 << i
      if element & sig_in:
        if element == State.TIME:
          state_out[worldid, adr] = time_in[worldid]
          adr += 1
        elif element == State.QPOS:
          for j in range(nq):
            state_out[worldid, adr + j] = qpos_in[worldid, j]
          adr += nq
        elif element == State.QVEL:
          for j in range(nv):
            state_out[worldid, adr + j] = qvel_in[worldid, j]
          adr += nv
        elif element == State.ACT:
          for j in range(na):
            state_out[worldid, adr + j] = act_in[worldid, j]
          adr += na
        elif element == State.HISTORY:
          for j in range(nhistory):
            state_out[worldid, adr + j] = history_in[worldid, j]
          adr += nhistory
        elif element == State.WARMSTART:
          for j in range(nv):
            state_out[worldid, adr + j] = qacc_warmstart_in[worldid, j]
          adr += nv
        elif element == State.CTRL:
          for j in range(nu):
            state_out[worldid, adr + j] = ctrl_in[worldid, j]
          adr += nu
        elif element == State.QFRC_APPLIED:
          for j in range(nv):
            state_out[worldid, adr + j] = qfrc_applied_in[worldid, j]
          adr += nv
        elif element == State.XFRC_APPLIED:
          for j in range(nbody):
            xfrc = xfrc_applied_in[worldid, j]
            state_out[worldid, adr + 0] = xfrc[0]
            state_out[worldid, adr + 1] = xfrc[1]
            state_out[worldid, adr + 2] = xfrc[2]
            state_out[worldid, adr + 3] = xfrc[3]
            state_out[worldid, adr + 4] = xfrc[4]
            state_out[worldid, adr + 5] = xfrc[5]
            adr += 6
        elif element == State.EQ_ACTIVE:
          for j in range(neq):
            state_out[worldid, adr + j] = float(eq_active_in[worldid, j])
          adr += neq
        elif element == State.MOCAP_POS:
          for j in range(nmocap):
            pos = mocap_pos_in[worldid, j]
            state_out[worldid, adr + 0] = pos[0]
            state_out[worldid, adr + 1] = pos[1]
            state_out[worldid, adr + 2] = pos[2]
            adr += 3
        elif element == State.MOCAP_QUAT:
          for j in range(nmocap):
            quat = mocap_quat_in[worldid, j]
            state_out[worldid, adr + 0] = quat[0]
            state_out[worldid, adr + 1] = quat[1]
            state_out[worldid, adr + 2] = quat[2]
            state_out[worldid, adr + 3] = quat[3]
            adr += 4
        elif element == State.USERDATA:
          for j in range(nuserdata):
            state_out[worldid, adr + j] = userdata_in[worldid, j]
          adr += nuserdata
        elif element == State.PLASTIC:
          for j in range(nplastic):
            state_out[worldid, adr + j] = plastic_in[worldid, j]
          adr += nplastic

  wp.launch(
    _get_state,
    dim=d.nworld,
    inputs=[
      m.nq,
      m.nv,
      m.nu,
      m.na,
      m.nbody,
      m.neq,
      m.nmocap,
      m.nplastic,
      m.nuserdata,
      m.nhistory,
      d.time,
      d.qpos,
      d.qvel,
      d.act,
      d.history,
      d.plastic,
      d.qacc_warmstart,
      d.ctrl,
      d.qfrc_applied,
      d.xfrc_applied,
      d.eq_active,
      d.mocap_pos,
      d.mocap_quat,
      d.userdata,
      int(sig),
      active or wp.ones(d.nworld, dtype=bool),
    ],
    outputs=[state],
  )


def set_state(m: Model, d: Data, state: wp.array2d[float], sig: int, active: Optional[wp.array] = None):
  """Copy concatenated state components specified by sig from state into Data.

  The bits of the integer sig correspond to element fields of State.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output information (device).
    state: Concatenation of state components.
    sig: Bitflag specifying state components.
    active: Per-world bitmask for setting state.
  """
  if sig >= (1 << State.NSTATE):
    raise ValueError(f"invalid state signature {sig} >= 2^mjNSTATE")

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def _set_state(
    # Model:
    nq: int,
    nv: int,
    nu: int,
    na: int,
    nbody: int,
    nflex: int,
    neq: int,
    nmocap: int,
    nplastic: int,
    nuserdata: int,
    nhistory: int,
    flex_plasticadr: wp.array[int],
    # In:
    sig_in: int,
    active_in: wp.array[bool],
    state_in: wp.array2d[float],
    # Data out:
    time_out: wp.array[float],
    qpos_out: wp.array2d[float],
    qvel_out: wp.array2d[float],
    act_out: wp.array2d[float],
    history_out: wp.array2d[float],
    plastic_out: wp.array2d[float],
    qacc_warmstart_out: wp.array2d[float],
    ctrl_out: wp.array2d[float],
    qfrc_applied_out: wp.array2d[float],
    xfrc_applied_out: wp.array2d[wp.spatial_vector],
    eq_active_out: wp.array2d[bool],
    mocap_pos_out: wp.array2d[wp.vec3],
    mocap_quat_out: wp.array2d[wp.quat],
    userdata_out: wp.array2d[float],
    flex_hessian_valid_out: wp.array2d[bool],
  ):
    worldid = wp.tid()

    if wp.static(active is not None):
      if not active_in[worldid]:
        return

    adr = int(0)
    for i in range(State.NSTATE.value):
      element = 1 << i
      if element & sig_in:
        if element == State.TIME:
          time_out[worldid] = state_in[worldid, adr]
          adr += 1
        elif element == State.QPOS:
          for j in range(nq):
            qpos_out[worldid, j] = state_in[worldid, adr + j]
          adr += nq
        elif element == State.QVEL:
          for j in range(nv):
            qvel_out[worldid, j] = state_in[worldid, adr + j]
          adr += nv
        elif element == State.ACT:
          for j in range(na):
            act_out[worldid, j] = state_in[worldid, adr + j]
          adr += na
        elif element == State.HISTORY:
          for j in range(nhistory):
            history_out[worldid, j] = state_in[worldid, adr + j]
          adr += nhistory
        elif element == State.WARMSTART:
          for j in range(nv):
            qacc_warmstart_out[worldid, j] = state_in[worldid, adr + j]
          adr += nv
        elif element == State.CTRL:
          for j in range(nu):
            ctrl_out[worldid, j] = state_in[worldid, adr + j]
          adr += nu
        elif element == State.QFRC_APPLIED:
          for j in range(nv):
            qfrc_applied_out[worldid, j] = state_in[worldid, adr + j]
          adr += nv
        elif element == State.XFRC_APPLIED:
          for j in range(nbody):
            xfrc = wp.spatial_vector(
              state_in[worldid, adr + 0],
              state_in[worldid, adr + 1],
              state_in[worldid, adr + 2],
              state_in[worldid, adr + 3],
              state_in[worldid, adr + 4],
              state_in[worldid, adr + 5],
            )
            xfrc_applied_out[worldid, j] = xfrc
            adr += 6
        elif element == State.EQ_ACTIVE:
          for j in range(neq):
            eq_active_out[worldid, j] = bool(state_in[worldid, adr + j])
          adr += neq
        elif element == State.MOCAP_POS:
          for j in range(nmocap):
            pos = wp.vec3(
              state_in[worldid, adr + 0],
              state_in[worldid, adr + 1],
              state_in[worldid, adr + 2],
            )
            mocap_pos_out[worldid, j] = pos
            adr += 3
        elif element == State.MOCAP_QUAT:
          for j in range(nmocap):
            quat = wp.quat(
              state_in[worldid, adr + 0],
              state_in[worldid, adr + 1],
              state_in[worldid, adr + 2],
              state_in[worldid, adr + 3],
            )
            mocap_quat_out[worldid, j] = quat
            adr += 4
        elif element == State.USERDATA:
          for j in range(nuserdata):
            userdata_out[worldid, j] = state_in[worldid, adr + j]
          adr += nuserdata
        elif element == State.PLASTIC:
          for j in range(nplastic):
            plastic_out[worldid, j] = state_in[worldid, adr + j]
          adr += nplastic
          for f in range(nflex):
            if flex_plasticadr[f] >= 0:
              flex_hessian_valid_out[worldid, f] = False

  wp.launch(
    _set_state,
    dim=d.nworld,
    inputs=[
      m.nq,
      m.nv,
      m.nu,
      m.na,
      m.nbody,
      m.nflex,
      m.neq,
      m.nmocap,
      m.nplastic,
      m.nuserdata,
      m.nhistory,
      m.flex_plasticadr,
      int(sig),
      active or wp.ones(d.nworld, dtype=bool),
      state,
    ],
    outputs=[
      d.time,
      d.qpos,
      d.qvel,
      d.act,
      d.history,
      d.plastic,
      d.qacc_warmstart,
      d.ctrl,
      d.qfrc_applied,
      d.xfrc_applied,
      d.eq_active,
      d.mocap_pos,
      d.mocap_quat,
      d.userdata,
      d.flex_hessian_valid,
    ],
  )


@wp.func
def _phi(s: float, i: int) -> float:
  """1D trilinear basis function (order=1 only).

  phi(s, 0) = 1 - s
  phi(s, 1) = s
  """
  if i == 0:
    return 1.0 - s
  return s


@wp.func
def eval_basis_trilinear(local: wp.vec3, node_idx: int) -> float:
  """Evaluate trilinear basis function for node_idx at local coords [0,1]^3.

  For order=1 (trilinear), node_idx encodes (i,j,k) via bits:
    k = node_idx & 1, j = (node_idx >> 1) & 1, i = (node_idx >> 2) & 1
  """
  k = node_idx & 1
  j = (node_idx >> 1) & 1
  i = (node_idx >> 2) & 1
  return _phi(local[0], i) * _phi(local[1], j) * _phi(local[2], k)


MAX_INTERP_BODIES = 64
vec_interp_body = wp.types.vector(length=MAX_INTERP_BODIES, dtype=int)
vec_interp_weight = wp.types.vector(length=MAX_INTERP_BODIES, dtype=float)


@wp.func
def add_interp_weight(
  # In:
  nb: int,
  body: vec_interp_body,
  weight: vec_interp_weight,
  b: int,
  w: float,
) -> tuple[int, vec_interp_body, vec_interp_weight]:
  """Appends weight to body id, merging duplicates by summing weights."""
  if b < 0:
    return nb, body, weight
  for i in range(nb):
    if body[i] == b:
      weight[i] += w
      return nb, body, weight
  if nb < MAX_INTERP_BODIES:
    body[nb] = b
    weight[nb] = w
    return nb + 1, body, weight
  return nb, body, weight


@wp.func
def add_interp_point_weights(
  # Model:
  body_weldid: wp.array[int],
  flex_nodebodyid: wp.array[int],
  # In:
  coord: wp.vec3,
  scale: float,
  cn: wp.vec3i,
  nstart: int,
  is_shell: bool,
  use_weld: bool,
  local_nb: int,
  local_bodies: vec_interp_body,
  local_weights: vec_interp_weight,
) -> tuple[int, vec_interp_body, vec_interp_weight]:
  """Accumulates trilinear cell node body weights at reference coord (with Coons TFI in shell mode)."""
  if scale == 0.0:
    return local_nb, local_bodies, local_weights
  cx = cn[0]
  cy = cn[1]
  cz = cn[2]
  ci = wp.clamp(int(wp.floor(coord[0] * float(cx))), 0, cx - 1)
  cj = wp.clamp(int(wp.floor(coord[1] * float(cy))), 0, cy - 1)
  ck = wp.clamp(int(wp.floor(coord[2] * float(cz))), 0, cz - 1)
  local = wp.vec3(
    wp.clamp(coord[0] * float(cx) - float(ci), 0.0, 1.0),
    wp.clamp(coord[1] * float(cy) - float(cj), 0.0, 1.0),
    wp.clamp(coord[2] * float(cz) - float(ck), 0.0, 1.0),
  )
  cy1 = cy + 1
  cz1 = cz + 1
  n_corners = int(8)
  three = int(3)
  for c_idx in range(n_corners):
    basis = eval_basis_trilinear(local, c_idx)
    if basis != 0.0:
      nw = scale * basis
      gi = ci + ((c_idx >> 2) & 1)
      gj = cj + ((c_idx >> 1) & 1)
      gk = ck + (c_idx & 1)
      is_int = is_shell and gi > 0 and gi < cx and gj > 0 and gj < cy and gk > 0 and gk < cz
      idx_x = wp.vec3i(0, cx, gi)
      idx_y = wp.vec3i(0, cy, gj)
      idx_z = wp.vec3i(0, cz, gk)
      s = float(gi) / float(cx)
      t = float(gj) / float(cy)
      u = float(gk) / float(cz)
      wx = wp.vec3(1.0 - s, s, -1.0)
      wy = wp.vec3(1.0 - t, t, -1.0)
      wz = wp.vec3(1.0 - u, u, -1.0)
      a_start = int(0) if is_int else int(2)
      for a in range(a_start, three):
        for b in range(a_start, three):
          for c in range(a_start, three):
            if is_int and a == 2 and b == 2 and c == 2:
              continue
            w_tfi = nw * wx[a] * wy[b] * wz[c] if is_int else nw
            nidx = nstart + (idx_x[a] * cy1 + idx_y[b]) * cz1 + idx_z[c]
            bid = flex_nodebodyid[nidx]
            if use_weld:
              bid = body_weldid[bid]
            local_nb, local_bodies, local_weights = add_interp_weight(
              local_nb, local_bodies, local_weights, bid, w_tfi
            )
  return local_nb, local_bodies, local_weights


@wp.func
def add_flex_interp_weights(
  # Model:
  body_weldid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  f: int,
  elem_id: int,
  vert_id: int,
  con_pos: wp.vec3,
  worldid: int,
  scale: float,
  use_weld: bool,
  local_nb: int,
  local_bodies: vec_interp_body,
  local_weights: vec_interp_weight,
) -> tuple[int, vec_interp_body, vec_interp_weight]:
  """Accumulates interpolated flex contact side bodies and weights at the blended reference coord."""
  coord = wp.vec3(0.0)
  if vert_id >= 0:
    coord = flex_vert0[flex_vertadr[f] + vert_id]
  elif elem_id >= 0:
    dim_f = flex_dim[f]
    edata_adr = flex_elemdataadr[f] + elem_id * (dim_f + 1)
    vert_adr_f = flex_vertadr[f]
    w_elem = flex_elem_weights(
      flex_dim, flex_vertadr, flex_elemdataadr, flex_elem, flexvert_xpos_in, f, elem_id, con_pos, worldid
    )
    for vi in range(dim_f + 1):
      v_idx = flex_elem[edata_adr + vi]
      coord += w_elem[vi] * flex_vert0[vert_adr_f + v_idx]
  else:
    return local_nb, local_bodies, local_weights

  return add_interp_point_weights(
    body_weldid,
    flex_nodebodyid,
    coord,
    scale,
    flex_cellnum[f],
    flex_nodeadr[f],
    flex_interp[f] < 0,
    use_weld,
    local_nb,
    local_bodies,
    local_weights,
  )


@wp.func
def _top4_interp(
  # In:
  local_nb: int,
  local_bodies: vec_interp_body,
  local_weights: vec_interp_weight,
) -> tuple[wp.vec4i, wp.vec4]:
  selected_b = wp.vec4i(-1, -1, -1, -1)
  selected_W = wp.vec4(0.0, 0.0, 0.0, 0.0)
  lw = local_weights
  n_top = int(4)
  for p in range(n_top):
    max_w = float(-1.0)
    max_b = int(-1)
    max_k = int(-1)
    for k in range(local_nb):
      w = lw[k]
      if w > max_w:
        max_w = w
        max_b = local_bodies[k]
        max_k = k
    if max_k >= 0 and max_w > 0.0:
      lw[max_k] = -1.0
      selected_b[p] = max_b
      selected_W[p] = max_w

  sum_W = selected_W[0] + selected_W[1] + selected_W[2] + selected_W[3]
  if sum_W != 0.0:
    selected_W = wp.vec4(
      selected_W[0] / sum_W,
      selected_W[1] / sum_W,
      selected_W[2] / sum_W,
      selected_W[3] / sum_W,
    )
  return selected_b, selected_W


@wp.func
def get_face_metadata(
  # In:
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  order_abs: int,
) -> Tuple[int, int, int, int, int, int]:
  size01 = cellnum_y * cellnum_z
  size23 = cellnum_x * cellnum_z
  size45 = cellnum_x * cellnum_y

  face_id = 0
  within_face = 0

  if face_elem_idx < size01:
    face_id = 0
    within_face = face_elem_idx
  elif face_elem_idx < 2 * size01:
    face_id = 1
    within_face = face_elem_idx - size01
  elif face_elem_idx < 2 * size01 + size23:
    face_id = 2
    within_face = face_elem_idx - 2 * size01
  elif face_elem_idx < 2 * size01 + 2 * size23:
    face_id = 3
    within_face = face_elem_idx - 2 * size01 - size23
  elif face_elem_idx < 2 * size01 + 2 * size23 + size45:
    face_id = 4
    within_face = face_elem_idx - 2 * size01 - 2 * size23
  else:
    face_id = 5
    within_face = face_elem_idx - 2 * size01 - 2 * size23 - size45

  normal_axis = face_id // 2

  c1 = 0
  if face_id == 0 or face_id == 1:
    c1 = cellnum_z
  elif face_id == 2 or face_id == 3:
    c1 = cellnum_x
  else:
    c1 = cellnum_y

  fixed_dim = 0
  if normal_axis == 0:
    fixed_dim = cellnum_x
  elif normal_axis == 1:
    fixed_dim = cellnum_y
  else:
    fixed_dim = cellnum_z
  g_fixed = (face_id % 2) * fixed_dim * order_abs

  q0 = within_face // c1
  q1 = within_face % c1

  ny_g = cellnum_y * order_abs + 1
  nz_g = cellnum_z * order_abs + 1

  return normal_axis, g_fixed, q0, q1, ny_g, nz_g


@wp.func
def gather_face_node_index_fast(
  # In:
  normal_axis: int,
  g_fixed: int,
  q0: int,
  q1: int,
  ny_g: int,
  nz_g: int,
  local_idx: int,
  order_abs: int,
) -> int:
  l0 = local_idx // (order_abs + 1)
  l1 = local_idx % (order_abs + 1)

  g = wp.vec3i(0, 0, 0)
  if normal_axis == 0:
    g = wp.vec3i(g_fixed, q0 * order_abs + l0, q1 * order_abs + l1)
  elif normal_axis == 1:
    g = wp.vec3i(q1 * order_abs + l1, g_fixed, q0 * order_abs + l0)
  else:
    g = wp.vec3i(q0 * order_abs + l0, q1 * order_abs + l1, g_fixed)

  return g[0] * ny_g * nz_g + g[1] * nz_g + g[2]


@wp.func
def gather_face_node_index(
  # In:
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  local_idx: int,
  order_abs: int,
) -> int:
  normal_axis, g_fixed, q0, q1, ny_g, nz_g = get_face_metadata(cellnum_x, cellnum_y, cellnum_z, face_elem_idx, order_abs)
  return gather_face_node_index_fast(normal_axis, g_fixed, q0, q1, ny_g, nz_g, local_idx, order_abs)


@wp.func
def compute_interp_face_quat(
  # Model:
  flex_node0: wp.array[wp.vec3],
  # Data in:
  flexnode_xpos_in: wp.array2d[wp.vec3],
  # In:
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  nstart: int,
  order_abs: int,
  worldid: int,
) -> wp.quat:
  normal_axis, g_fixed, q0, q1, ny_g, nz_g = get_face_metadata(
    cellnum_x,
    cellnum_y,
    cellnum_z,
    face_elem_idx,
    order_abs,
  )

  t1 = wp.vec3(0.0)
  t2 = wp.vec3(0.0)

  npc = (order_abs + 1) * (order_abs + 1)
  gidx0 = int(0)
  gidx1 = int(0)

  for local_idx in range(9):
    if local_idx < npc:
      gidx = gather_face_node_index_fast(
        normal_axis,
        g_fixed,
        q0,
        q1,
        ny_g,
        nz_g,
        local_idx,
        order_abs,
      )
      if local_idx == 0:
        gidx0 = gidx
      gidx1 = gidx
      node_pos = flexnode_xpos_in[worldid, nstart + gidx]

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

  if wp.static(FLEX_INTERP_PHYS_JAC):
    axis0 = (normal_axis + 1) % 3
    axis1 = (normal_axis + 2) % 3
    dref = flex_node0[nstart + gidx1] - flex_node0[nstart + gidx0]
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

  return mat33_to_quat_polar(F)


@wp.func
def flex_phi(s: float, i: int, order: int) -> float:
  return 1.0 - s if i == 0 else s


@wp.func
def flex_dphi(s: float, i: int, order: int) -> float:
  return -1.0 if i == 0 else 1.0


@wp.func
def dphi2D(s0: float, l0: int, s1: float, l1: int, order: int, direction: int) -> float:
  if direction == 0:
    return flex_dphi(s0, l0, order) * flex_phi(s1, l1, order)
  else:
    return flex_phi(s0, l0, order) * flex_dphi(s1, l1, order)


@wp.func
def _tet_weights(p: wp.vec3, x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> wp.vec4:
  """Barycentric weights of the closest point on tetrahedron x0 x1 x2 x3 to p."""
  # solve p - x0 = sum_i w_i (x_i - x0) with Cramer's rule, done if p is inside
  e1 = x1 - x0
  e2 = x2 - x0
  e3 = x3 - x0
  ep = p - x0
  c = wp.cross(e2, e3)
  det = wp.dot(e1, c)
  if wp.abs(det) > MJ_MINVAL:
    w1 = wp.dot(ep, c) / det
    w2 = wp.dot(e1, wp.cross(ep, e3)) / det
    w3 = wp.dot(e1, wp.cross(e2, ep)) / det
    w0 = 1.0 - w1 - w2 - w3
    if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0 and w3 >= 0.0:
      return wp.vec4(w0, w1, w2, w3)

  # p is outside: closest point on the faces, face i is opposite vertex i
  w = wp.vec4(0.25)
  mindist = float(MJ_MAXVAL)

  wf = tri_weights(p, x1, x2, x3)
  dist = wp.length(wf[0] * x1 + wf[1] * x2 + wf[2] * x3 - p)
  if dist < mindist:
    mindist = dist
    w = wp.vec4(0.0, wf[0], wf[1], wf[2])

  wf = tri_weights(p, x0, x2, x3)
  dist = wp.length(wf[0] * x0 + wf[1] * x2 + wf[2] * x3 - p)
  if dist < mindist:
    mindist = dist
    w = wp.vec4(wf[0], 0.0, wf[1], wf[2])

  wf = tri_weights(p, x0, x1, x3)
  dist = wp.length(wf[0] * x0 + wf[1] * x1 + wf[2] * x3 - p)
  if dist < mindist:
    mindist = dist
    w = wp.vec4(wf[0], wf[1], 0.0, wf[2])

  wf = tri_weights(p, x0, x1, x2)
  dist = wp.length(wf[0] * x0 + wf[1] * x1 + wf[2] * x2 - p)
  if dist < mindist:
    w = wp.vec4(wf[0], wf[1], wf[2], 0.0)

  return w


@wp.func
def flex_elem_weights(
  # Model:
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  flexid: int,
  elemid: int,
  pos: wp.vec3,
  worldid: int,
) -> wp.vec4:
  """Barycentric weights of the point on a flex element closest to pos (see mj_elemVertWeight).

  Args:
    flex_dim: Flex dimensions.
    flex_vertadr: First vertex address of each flex.
    flex_elemdataadr: First element vertex id address of each flex.
    flex_elem: Element vertex ids.
    flexvert_xpos_in: Cartesian flex vertex positions.
    flexid: Flex id.
    elemid: Element id, local to the flex.
    pos: Contact position.
    worldid: World id.

  Returns:
    Weights of the dim + 1 element vertices, in element order, zero-padded to 4.
  """
  dim = flex_dim[flexid]
  vertadr = flex_vertadr[flexid]
  edata = flex_elemdataadr[flexid] + elemid * (dim + 1)

  x0 = flexvert_xpos_in[worldid, vertadr + flex_elem[edata]]
  x1 = flexvert_xpos_in[worldid, vertadr + flex_elem[edata + 1]]

  w = wp.vec4(0.0)
  if dim == 1:
    ab = x1 - x0
    len2 = wp.dot(ab, ab)
    t = float(0.0)
    if len2 > MJ_MINVAL:
      t = wp.clamp(wp.dot(pos - x0, ab) / len2, 0.0, 1.0)
    w = wp.vec4(1.0 - t, t, 0.0, 0.0)
  elif dim == 2:
    x2 = flexvert_xpos_in[worldid, vertadr + flex_elem[edata + 2]]
    wt = tri_weights(pos, x0, x1, x2)
    w = wp.vec4(wt[0], wt[1], wt[2], 0.0)
  else:
    x2 = flexvert_xpos_in[worldid, vertadr + flex_elem[edata + 2]]
    x3 = flexvert_xpos_in[worldid, vertadr + flex_elem[edata + 3]]
    w = _tet_weights(pos, x0, x1, x2, x3)

  # degenerate element: uniform weights
  for i in range(4):
    if i <= dim and (not wp.isfinite(w[i]) or wp.abs(w[i]) > MJ_MAXVAL):
      inv = 1.0 / float(dim + 1)
      w = wp.vec4(inv, inv, wp.where(dim >= 2, inv, 0.0), wp.where(dim == 3, inv, 0.0))

  return w


@wp.func
def contact_bodies_and_weights(
  # Model:
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  conid: int,
  side: int,
  geom: wp.vec2i,
  flex: wp.vec2i,
  elem: wp.vec2i,
  vert: wp.vec2i,
  con_pos: wp.vec3,
  worldid: int,
) -> Tuple[wp.vec4i, wp.vec4]:
  geom_id = geom[side]
  flex_id = flex[side]
  elem_id = elem[side]
  vert_id = vert[side]

  # Rigid Geom Side
  if geom_id >= 0:
    return wp.vec4i(geom_bodyid[geom_id], -1, -1, -1), wp.vec4(1.0, 0.0, 0.0, 0.0)

  # Interpolated flex contact (vertex or element, volume or shell)
  if flex_interp[flex_id] != 0:
    local_bodies = vec_interp_body(-1)
    local_weights = vec_interp_weight(0.0)
    local_nb = int(0)
    local_nb, local_bodies, local_weights = add_flex_interp_weights(
      flex_nodebodyid,
      flex_dim,
      flex_interp,
      flex_cellnum,
      flex_nodeadr,
      flex_vertadr,
      flex_elemdataadr,
      flex_nodebodyid,
      flex_elem,
      flex_vert0,
      flexvert_xpos_in,
      flex_id,
      elem_id,
      vert_id,
      con_pos,
      worldid,
      1.0,
      False,
      local_nb,
      local_bodies,
      local_weights,
    )
    return _top4_interp(local_nb, local_bodies, local_weights)

  # Plane-Vertex or Vertex-only flex contact
  flex_vert_start = flex_vertadr[flex_id]
  if vert_id >= 0:
    body = flex_vertbodyid[flex_vert_start + vert_id]
    body_ids = wp.vec4i(body, -1, -1, -1)
    weights = wp.vec4(1.0, 0.0, 0.0, 0.0)
    nb = int(1)
    if flex_elemnum[flex_id] > 0 and rod_has_twist(flex_rod_has_twist, flex_id):
      e_rod = flex_vert_rodelem[flex_vert_start + vert_id]
      v_other = vert[1 - side] if flex[1 - side] == flex_id else -1
      if e_rod >= 0:
        ed = flex_elemdataadr[flex_id] + 2 * e_rod
        ev0 = flex_elem[ed + 0]
        ev1 = flex_elem[ed + 1]
        if ev0 != v_other and ev1 != v_other:
          body_ids, nb = _append_body_weight_zero(body_ids, nb, flex_vertbodyid[flex_vert_start + ev0])
          body_ids, nb = _append_body_weight_zero(body_ids, nb, flex_vertbodyid[flex_vert_start + ev1])
          jnt_e, _sgn_v = rod_edge_twist_jnt(
            flex_elemadr,
            flex_elemnum,
            flex_rod_edgeadr,
            flex_rod_edgenum,
            flex_rodedge_jnt,
            flex_elem_rodedge,
            flex_id,
            e_rod,
          )
          if jnt_e >= 0:
            body_ids, nb = _append_body_weight_zero(body_ids, nb, jnt_bodyid[jnt_e])
    return body_ids, weights

  dim = flex_dim[flex_id]
  if dim < 1 or dim > 3:
    return wp.vec4i(-1, -1, -1, -1), wp.vec4(0.0, 0.0, 0.0, 0.0)

  # barycentric weights of the closest point on the element
  elem_weight = flex_elem_weights(
    flex_dim, flex_vertadr, flex_elemdataadr, flex_elem, flexvert_xpos_in, flex_id, elem_id, con_pos, worldid
  )

  elem_data_start = flex_elemdataadr[flex_id] + elem_id * (dim + 1)
  v0 = flex_elem[elem_data_start + 0]
  v1 = flex_elem[elem_data_start + 1]
  b0 = flex_vertbodyid[flex_vert_start + v0]
  b1 = flex_vertbodyid[flex_vert_start + v1]

  if dim == 1:
    body_ids = wp.vec4i(b0, b1, -1, -1)
    weights = wp.vec4(elem_weight[0], elem_weight[1], 0.0, 0.0)
    nb = int(2)

    if flex_elemnum[flex_id] > 0 and rod_has_twist(flex_rod_has_twist, flex_id):
      v_other = vert[1 - side] if flex[1 - side] == flex_id else -1
      if v0 != v_other and v1 != v_other:
        jnt_e, _sgn_e = rod_edge_twist_jnt(
          flex_elemadr,
          flex_elemnum,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_id,
          elem_id,
        )
        if jnt_e >= 0:
          body_ids, nb = _append_body_weight_zero(body_ids, nb, jnt_bodyid[jnt_e])

    return body_ids, weights

  elif dim == 2:
    v2 = flex_elem[elem_data_start + 2]
    b2 = flex_vertbodyid[flex_vert_start + v2]
    return wp.vec4i(b0, b1, b2, -1), wp.vec4(elem_weight[0], elem_weight[1], elem_weight[2], 0.0)

  else:
    v2 = flex_elem[elem_data_start + 2]
    v3 = flex_elem[elem_data_start + 3]
    b2 = flex_vertbodyid[flex_vert_start + v2]
    b3 = flex_vertbodyid[flex_vert_start + v3]
    return wp.vec4i(b0, b1, b2, b3), elem_weight


# TODO(flex): Add interpolated flex self-collision support. Currently, interpolated
# flex contacts only handle flex-vs-geom contacts (e.g., flex on ground plane).
# Self-collision requires distributing both sides of the contact across cell nodes.
