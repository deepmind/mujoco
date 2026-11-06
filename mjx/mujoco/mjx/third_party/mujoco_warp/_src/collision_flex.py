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
"""Flex collision detection (geom vs flex triangles)."""

import dataclasses
import math

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import collision_primitive_core
from mujoco.mjx.third_party.mujoco_warp._src import support
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import Geom
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import sap_binary_search
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import sap_range
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import ccd
from mujoco.mjx.third_party.mujoco_warp._src.math import make_frame
from mujoco.mjx.third_party.mujoco_warp._src.math import orthogonals
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAX_EPAFACES
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAX_EPAHORIZON
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXCONPAIR
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINMU
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import ContactType
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import IntegratorType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
from mujoco.mjx.third_party.mujoco_warp._src.types import mat63
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_FPS_BLOCK_SIZE: int = 64
ENABLE_SAT_PREFILTER: bool = True
_FLEX_CONTACT_DEDUP_TOLERANCE: float = 1e-3


@wp.func
def _flex_element_aabb_filter(
  # In:
  box1_min: wp.vec3,
  box1_max: wp.vec3,
  box2_min: wp.vec3,
  box2_max: wp.vec3,
):
  """Return True if the two AABBs do NOT intersect (discard pair)."""
  if box1_max[0] < box2_min[0] or box1_min[0] > box2_max[0]:
    return True
  if box1_max[1] < box2_min[1] or box1_min[1] > box2_max[1]:
    return True
  if box1_max[2] < box2_min[2] or box1_min[2] > box2_max[2]:
    return True
  return False


@wp.func
def _axis_separated(
  # In:
  p0: wp.vec3,
  p1: wp.vec3,
  p2: wp.vec3,
  q0: wp.vec3,
  q1: wp.vec3,
  q2: wp.vec3,
  ax: wp.vec3,
  cutoff_sq: float,
) -> bool:
  min1 = wp.min(wp.dot(p0, ax), wp.min(wp.dot(p1, ax), wp.dot(p2, ax)))
  max1 = wp.max(wp.dot(p0, ax), wp.max(wp.dot(p1, ax), wp.dot(p2, ax)))
  min2 = wp.min(wp.dot(q0, ax), wp.min(wp.dot(q1, ax), wp.dot(q2, ax)))
  max2 = wp.max(wp.dot(q0, ax), wp.max(wp.dot(q1, ax), wp.dot(q2, ax)))
  diff = wp.max(min1 - max2, min2 - max1)
  if diff > 0.0:
    lax_sq = wp.length_sq(ax)
    if lax_sq > 1e-12 and diff * diff > cutoff_sq * lax_sq:
      return True
  return False


@wp.func
def _point_segment_axis(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> wp.vec3:
  ab = b - a
  ab_len_sq = wp.length_sq(ab)
  if ab_len_sq > 1e-12:
    ap = p - a
    t = wp.clamp(wp.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
    return ap - t * ab
  return p - a


@wp.func
def _triangle_sat_separated(
  # In:
  p0: wp.vec3,
  p1: wp.vec3,
  p2: wp.vec3,
  q0: wp.vec3,
  q1: wp.vec3,
  q2: wp.vec3,
  cutoff_sq: float,
) -> bool:
  """Returns True if two triangles inflated by cutoff are separated (no contact possible)."""
  e1_0 = p1 - p0
  e1_1 = p2 - p1
  e1_2 = p0 - p2

  # Face normal of triangle 1
  n1 = wp.cross(e1_0, e1_1)
  p_proj = wp.dot(p0, n1)
  q0_d = wp.dot(q0, n1)
  q1_d = wp.dot(q1, n1)
  q2_d = wp.dot(q2, n1)
  diff1 = wp.max(p_proj - wp.max(q0_d, wp.max(q1_d, q2_d)), wp.min(q0_d, wp.min(q1_d, q2_d)) - p_proj)
  if diff1 > 0.0:
    ln1_sq = wp.length_sq(n1)
    if ln1_sq > 1e-12 and diff1 * diff1 > cutoff_sq * ln1_sq:
      return True

  # Face normal of triangle 2
  e2_0 = q1 - q0
  e2_1 = q2 - q1
  e2_2 = q0 - q2
  n2 = wp.cross(e2_0, e2_1)
  q_proj = wp.dot(q0, n2)
  p0_d = wp.dot(p0, n2)
  p1_d = wp.dot(p1, n2)
  p2_d = wp.dot(p2, n2)
  diff2 = wp.max(wp.min(p0_d, wp.min(p1_d, p2_d)) - q_proj, q_proj - wp.max(p0_d, wp.max(p1_d, p2_d)))
  if diff2 > 0.0:
    ln2_sq = wp.length_sq(n2)
    if ln2_sq > 1e-12 and diff2 * diff2 > cutoff_sq * ln2_sq:
      return True

  # Edge-edge cross products (3x3 = 9 axes)
  for i in range(3):
    e1 = e1_0
    if i == 1:
      e1 = e1_1
    elif i == 2:
      e1 = e1_2
    for j in range(3):
      e2 = e2_0
      if j == 1:
        e2 = e2_1
      elif j == 2:
        e2 = e2_2
      if _axis_separated(p0, p1, p2, q0, q1, q2, wp.cross(e1, e2), cutoff_sq):
        return True

  # In-plane edge normals for triangle 1 (3 axes)
  for i in range(3):
    e = e1_0
    if i == 1:
      e = e1_1
    elif i == 2:
      e = e1_2
    if _axis_separated(p0, p1, p2, q0, q1, q2, wp.cross(e, n1), cutoff_sq):
      return True

  # In-plane edge normals for triangle 2 (3 axes)
  for i in range(3):
    e = e2_0
    if i == 1:
      e = e2_1
    elif i == 2:
      e = e2_2
    if _axis_separated(p0, p1, p2, q0, q1, q2, wp.cross(e, n2), cutoff_sq):
      return True

  # Vertex-to-vertex axes (3x3 = 9 axes)
  for i in range(3):
    p = p0
    if i == 1:
      p = p1
    elif i == 2:
      p = p2
    for j in range(3):
      q = q0
      if j == 1:
        q = q1
      elif j == 2:
        q = q2
      if _axis_separated(p0, p1, p2, q0, q1, q2, p - q, cutoff_sq):
        return True

  # Vertex-to-edge axes: vertices of T1 to edges of T2 (9 axes)
  for i in range(3):
    p = p0
    if i == 1:
      p = p1
    elif i == 2:
      p = p2
    for j in range(3):
      qa = q0
      qb = q1
      if j == 1:
        qa = q1
        qb = q2
      elif j == 2:
        qa = q2
        qb = q0
      if _axis_separated(p0, p1, p2, q0, q1, q2, _point_segment_axis(p, qa, qb), cutoff_sq):
        return True

  # Vertex-to-edge axes: vertices of T2 to edges of T1 (9 axes)
  for i in range(3):
    q = q0
    if i == 1:
      q = q1
    elif i == 2:
      q = q2
    for j in range(3):
      pa = p0
      pb = p1
      if j == 1:
        pa = p1
        pb = p2
      elif j == 2:
        pa = p2
        pb = p0
      if _axis_separated(p0, p1, p2, q0, q1, q2, _point_segment_axis(q, pa, pb), cutoff_sq):
        return True

  return False


@wp.kernel
def _flex_broadphase_bounds(
  # Model:
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_radius: wp.array[float],
  flex_rod_is_box: wp.array[bool],
  flex_rod_box_size: wp.array[wp.vec2],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  flex_aabb_min_out: wp.array2d[wp.vec3],
  flex_aabb_max_out: wp.array2d[wp.vec3],
):
  worldid, flexid = wp.tid()

  start = flex_vertadr[flexid]
  num = flex_vertnum[flexid]
  if num == 0:
    return

  min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
  max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)

  for i in range(num):
    pos = flexvert_xpos_in[worldid, start + i]
    min_bound = wp.min(min_bound, pos)
    max_bound = wp.max(max_bound, pos)

  margin = flex_margin[flexid] + flex_gap[flexid]
  rad = flex_radius[flexid]
  is_box, box_2d = support.rod_box_size(flex_rod_is_box, flex_rod_box_size, flexid)
  if is_box:
    rad = wp.max(rad, wp.length(box_2d))
  bound = rad + margin
  inflate = wp.vec3(bound, bound, bound)

  flex_aabb_min_out[worldid, flexid] = min_bound - inflate
  flex_aabb_max_out[worldid, flexid] = max_bound + inflate


# TODO(team): generalize into a shared contact parameter mixing function
#   (mj_contactParam) that works for both geom-geom and geom-flex contacts.
@wp.func
def _mix_flex_contact_params(
  # In:
  a_condim: int,
  a_priority: int,
  a_solmix: float,
  a_solref: wp.vec2,
  a_solimp: vec5,
  a_friction: wp.vec3,
  a_gap: float,
  b_condim: int,
  b_priority: int,
  b_solmix: float,
  b_solref: wp.vec2,
  b_solimp: vec5,
  b_friction: wp.vec3,
  b_gap: float,
) -> tuple[int, float, wp.vec2, vec5, vec5]:
  """Mix contact parameters between geom and flex, matching mj_contactParam."""
  gap = a_gap + b_gap

  if a_priority > b_priority:
    condim = a_condim
    solref = a_solref
    solimp = a_solimp
    fri = a_friction
  elif a_priority < b_priority:
    condim = b_condim
    solref = b_solref
    solimp = b_solimp
    fri = b_friction
  else:
    # same priority
    condim = wp.max(a_condim, b_condim)

    # compute solver mix factor
    if a_solmix >= MJ_MINVAL and b_solmix >= MJ_MINVAL:
      mix = a_solmix / (a_solmix + b_solmix)
    elif a_solmix < MJ_MINVAL and b_solmix < MJ_MINVAL:
      mix = 0.5
    elif a_solmix < MJ_MINVAL:
      mix = 0.0
    else:
      mix = 1.0

    # solref: mix if both standard, min if either direct
    if a_solref[0] > 0.0 and b_solref[0] > 0.0:
      solref = wp.vec2(
        mix * a_solref[0] + (1.0 - mix) * b_solref[0],
        mix * a_solref[1] + (1.0 - mix) * b_solref[1],
      )
    else:
      solref = wp.vec2(
        wp.min(a_solref[0], b_solref[0]),
        wp.min(a_solref[1], b_solref[1]),
      )

    # solimp: mix
    solimp = vec5(
      mix * a_solimp[0] + (1.0 - mix) * b_solimp[0],
      mix * a_solimp[1] + (1.0 - mix) * b_solimp[1],
      mix * a_solimp[2] + (1.0 - mix) * b_solimp[2],
      mix * a_solimp[3] + (1.0 - mix) * b_solimp[3],
      mix * a_solimp[4] + (1.0 - mix) * b_solimp[4],
    )

    # friction: max
    fri = wp.vec3(
      wp.max(a_friction[0], b_friction[0]),
      wp.max(a_friction[1], b_friction[1]),
      wp.max(a_friction[2], b_friction[2]),
    )

  # unpack 5D friction with MJ_MINMU floor
  friction = vec5(
    wp.max(MJ_MINMU, fri[0]),
    wp.max(MJ_MINMU, fri[0]),
    wp.max(MJ_MINMU, fri[1]),
    wp.max(MJ_MINMU, fri[2]),
    wp.max(MJ_MINMU, fri[2]),
  )

  return condim, gap, solref, solimp, friction


@wp.func
def _write_candidate(
  # In:
  max_candidates: int,
  dist: float,
  pos: wp.vec3,
  nrm: wp.vec3,
  geom: int,
  flexid1: int,
  flexid2: int,
  elemid: int,
  vertid: int,
  worldid: int,
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  if dist >= MJ_MAXVAL:
    return

  candid = wp.atomic_add(ncand_out, 0, 1)  # kernel_analyzer: ignore[atomic]
  if candid >= max_candidates:
    if warn_overflow:
      wp.printf(
        "flex candidate overflow - please increase naconmax beyond %u\n"
        "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",
        max_candidates,
      )
    wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NARROWPHASE))
    return

  cand_dist_out[candid] = dist
  cand_pos_out[candid] = pos
  cand_nrm_out[candid] = nrm
  if geom >= 0:
    cand_geom_out[candid] = wp.vec2i(geom, -1)
    cand_flex_out[candid] = wp.vec2i(flexid1, flexid2)
    cand_elem_out[candid] = wp.vec2i(-1, elemid)
    cand_vert_out[candid] = wp.vec2i(-1, vertid)
  elif geom == -2:
    cand_geom_out[candid] = wp.vec2i(-1, -1)
    cand_flex_out[candid] = wp.vec2i(flexid1, flexid2)
    cand_elem_out[candid] = wp.vec2i(elemid, vertid)
    cand_vert_out[candid] = wp.vec2i(-1, -1)
  else:
    cand_geom_out[candid] = wp.vec2i(-1, -1)
    cand_flex_out[candid] = wp.vec2i(flexid1, flexid2)
    cand_elem_out[candid] = wp.vec2i(-1, elemid)
    cand_vert_out[candid] = wp.vec2i(vertid, -1)
  cand_worldid_out[candid] = worldid


@wp.func
def _collide_geom_capsule_detect(
  # In:
  max_candidates: int,
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size_val: wp.vec3,
  cap_pos: wp.vec3,
  cap_axis: wp.vec3,
  cap_radius: float,
  cap_half_len: float,
  margin: float,
  geomid: int,
  flexid: int,
  elemid: int,
  vertex_id: int,
  worldid: int,
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  if gtype == int(GeomType.SPHERE):
    sphere_radius = size_val[0]
    dist, contact_pos, nrm = collision_primitive_core.sphere_capsule(
      pos, sphere_radius, cap_pos, cap_axis, cap_radius, cap_half_len
    )
    if dist < margin:
      _write_candidate(
        max_candidates,
        dist,
        contact_pos,
        nrm,
        geomid,
        -1,
        flexid,
        elemid,
        vertex_id,
        worldid,
        warn_overflow,
        overflow_out,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        ncand_out,
      )
    return

  dists = wp.vec2(collision_primitive_core.MJ_MAXVAL, collision_primitive_core.MJ_MAXVAL)
  poss = collision_primitive_core.mat23f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  nrms = collision_primitive_core.mat23f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  if gtype == int(GeomType.CAPSULE):
    g_radius = size_val[0]
    g_half_len = size_val[1]
    g_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    dists, poss, nrms = collision_primitive_core.capsule_capsule(
      pos, g_axis, g_radius, g_half_len, cap_pos, cap_axis, cap_radius, cap_half_len
    )
  elif gtype == int(GeomType.BOX):
    dists, poss, box_nrms = collision_primitive_core.capsule_box(
      cap_pos, cap_axis, cap_radius, cap_half_len, pos, rot, size_val
    )
    nrms = -box_nrms

  for i in range(2):
    if dists[i] < margin:
      _write_candidate(
        max_candidates,
        dists[i],
        poss[i],
        nrms[i],
        geomid,
        -1,
        flexid,
        elemid,
        vertex_id,
        worldid,
        warn_overflow,
        overflow_out,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        ncand_out,
      )


@wp.func
def _collide_geom_box_rod_detect(
  # In:
  max_candidates: int,
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size_val: wp.vec3,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  margin: float,
  geomid: int,
  flexid: int,
  elemid: int,
  vertex_id: int,
  worldid: int,
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  if gtype == int(GeomType.SPHERE):
    dist, contact_pos, nrm = collision_primitive_core.sphere_box(pos, size_val[0], box_pos, box_rot, box_size)
    if dist < margin:
      _write_candidate(
        max_candidates,
        dist,
        contact_pos,
        nrm,
        geomid,
        -1,
        flexid,
        elemid,
        vertex_id,
        worldid,
        warn_overflow,
        overflow_out,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        ncand_out,
      )
    return

  if gtype == int(GeomType.CAPSULE):
    g_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    dists, poss, nrms = collision_primitive_core.capsule_box(pos, g_axis, size_val[0], size_val[1], box_pos, box_rot, box_size)
    for i in range(2):
      if dists[i] < margin:
        _write_candidate(
          max_candidates,
          dists[i],
          poss[i],
          nrms[i],
          geomid,
          -1,
          flexid,
          elemid,
          vertex_id,
          worldid,
          warn_overflow,
          overflow_out,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          ncand_out,
        )
    return

  if gtype == int(GeomType.BOX):
    ncon, b_dists, b_poss, b_nrm = collision_primitive_core.box_box(pos, rot, size_val, box_pos, box_rot, box_size, margin)
    for i in range(ncon):
      if b_dists[i] < margin:
        _write_candidate(
          max_candidates,
          b_dists[i],
          b_poss[i],
          b_nrm,
          geomid,
          -1,
          flexid,
          elemid,
          vertex_id,
          worldid,
          warn_overflow,
          overflow_out,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          ncand_out,
        )


@wp.func
def _collide_geom_triangle_detect(
  # In:
  max_candidates: int,
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size_val: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
  geomid: int,
  flexid: int,
  elemid: int,
  vertex_id: int,
  worldid: int,
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  if gtype == int(GeomType.SPHERE):
    sphere_radius = size_val[0]
    dist, contact_pos, nrm = collision_primitive_core.sphere_triangle(pos, sphere_radius, t1, t2, t3, tri_radius, margin)
    _write_candidate(
      max_candidates,
      dist,
      contact_pos,
      nrm,
      geomid,
      -1,
      flexid,
      elemid,
      vertex_id,
      worldid,
      warn_overflow,
      overflow_out,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      ncand_out,
    )
    return

  if gtype == int(GeomType.CAPSULE):
    cap_radius = size_val[0]
    cap_half_len = size_val[1]
    cap_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    cap_dists, cap_poss, cap_nrms = collision_primitive_core.capsule_triangle(
      pos, cap_axis, cap_radius, cap_half_len, t1, t2, t3, tri_radius, margin
    )
    for i in range(5):
      _write_candidate(
        max_candidates,
        cap_dists[i],
        cap_poss[i],
        cap_nrms[i],
        geomid,
        -1,
        flexid,
        elemid,
        vertex_id,
        worldid,
        warn_overflow,
        overflow_out,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        ncand_out,
      )
    return

  if gtype == int(GeomType.BOX):
    box_dists, box_poss, box_nrms = collision_primitive_core.box_triangle(pos, rot, size_val, t1, t2, t3, tri_radius, margin)
    for i in range(11):
      _write_candidate(
        max_candidates,
        box_dists[i],
        box_poss[i],
        box_nrms[i],
        geomid,
        -1,
        flexid,
        elemid,
        vertex_id,
        worldid,
        warn_overflow,
        overflow_out,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        ncand_out,
      )
    return

  if gtype == int(GeomType.CYLINDER):
    cyl_radius = size_val[0]
    cyl_half_height = size_val[1]
    cyl_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    cyl_dists, cyl_poss, cyl_nrms = collision_primitive_core.cylinder_triangle(
      pos, cyl_axis, cyl_radius, cyl_half_height, t1, t2, t3, tri_radius
    )
    for i in range(2):
      if cyl_dists[i] < margin:
        _write_candidate(
          max_candidates,
          cyl_dists[i],
          cyl_poss[i],
          cyl_nrms[i],
          geomid,
          -1,
          flexid,
          elemid,
          vertex_id,
          worldid,
          warn_overflow,
          overflow_out,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          ncand_out,
        )


@wp.func
def _collide_mesh_convex(
  # Model:
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_pos: wp.array[wp.vec3],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  # In:
  max_candidates: int,
  mesh_geom_pos: wp.vec3,
  geom_rot: wp.mat33,
  geom_size_val: wp.vec3,
  did: int,
  geom2: Geom,
  geom2_type: int,
  geom2_pos: wp.vec3,
  geom2_bounding_radius: float,
  geom2_radius: float,
  margin: float,
  geomid: int,
  flexid: int,
  elemid: int,
  vertid: int,
  worldid: int,
  epa_vert: wp.array[wp.vec3],
  epa_vert_index: wp.array[int],
  epa_face: wp.array[int],
  epa_pr: wp.array[wp.vec3],
  epa_norm2: wp.array[float],
  epa_horizon: wp.array[int],
  tolerance: float,
  ccd_iterations: int,
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  # Construct Mesh Geom (geom1)
  geom1 = Geom()
  geom1.pos = mesh_geom_pos
  geom1.rot = geom_rot
  geom1.size = geom_size_val
  geom1.margin = 0.0
  geom1.index = -1
  geom1.vertadr = wp.where(did >= 0, mesh_vertadr[did], -1)
  geom1.vertnum = wp.where(did >= 0, mesh_vertnum[did], -1)
  geom1.graphadr = wp.where(did >= 0, mesh_graphadr[did], -1)
  geom1.vert = mesh_vert
  geom1.graph = mesh_graph

  geom_center = mesh_geom_pos
  if did >= 0:
    geom_center = mesh_geom_pos + geom_rot @ mesh_pos[did]

  r_geom = wp.length(geom_size_val)
  if wp.length(geom2_pos - geom_center) <= r_geom + geom2_bounding_radius + margin + geom2_radius + 0.04:
    dist, ncontact, w1, w2, idx = ccd(
      tolerance,
      margin + geom2_radius,
      ccd_iterations,
      ccd_iterations,
      geom1,
      geom2,
      int(GeomType.MESH),
      geom2_type,
      mesh_geom_pos,
      geom2_pos,
      epa_vert,
      epa_vert_index,
      epa_face,
      epa_pr,
      epa_norm2,
      epa_horizon,
      warn_overflow,
      worldid,
      overflow_out,
    )

    if ncontact > 0 and dist < margin + geom2_radius:
      diff = (w1 - w2) if dist < 0.0 else (w2 - w1)
      gjk_normal = wp.normalize(diff)
      if wp.length_sq(diff) < 1e-12:
        gjk_normal = wp.normalize(geom2_pos - geom_center)

      best_normal = gjk_normal
      if did >= 0:
        w1_local = wp.transpose(geom_rot) @ (w1 - mesh_geom_pos)
        v_offset = mesh_vertadr[did]

        sv0 = int(0)
        if idx >= 0:
          f_verts = wp.vec3i(epa_face[idx] & 0x3FF, (epa_face[idx] >> 10) & 0x3FF, (epa_face[idx] >> 20) & 0x3FF)
          sv0 = epa_vert_index[2 * f_verts[0]]
        else:
          min_v_dist = float(1e10)
          for vi in range(mesh_vertnum[did]):
            d_v = wp.length_sq(w1_local - mesh_vert[v_offset + vi])
            if d_v < min_v_dist:
              min_v_dist = d_v
              sv0 = vi

        min_plane_dist = float(1e10)
        best_normal_local = wp.transpose(geom_rot) @ gjk_normal
        best_poly_idx = int(-1)

        v_global_idx = v_offset + sv0
        polymap_start = mesh_polymapadr[v_global_idx]
        npolygons = mesh_polymapnum[v_global_idx]

        for k in range(npolygons):
          poly_idx = mesh_polymap[polymap_start + k]
          normal_local = mesh_polynormal[poly_idx]
          v0_local_idx = mesh_polyvert[mesh_polyvertadr[poly_idx]]
          v0_mesh_local = mesh_vert[v_offset + v0_local_idx]

          dist_to_plane = wp.abs(wp.dot(w1_local - v0_mesh_local, normal_local))
          if dist_to_plane < min_plane_dist:
            min_plane_dist = dist_to_plane
            best_normal_local = normal_local
            best_poly_idx = poly_idx

        if best_poly_idx >= 0:
          vert_start = mesh_polyvertadr[best_poly_idx]
          v0_idx = mesh_polyvert[vert_start]
          v1_idx = mesh_polyvert[vert_start + 1]
          v2_idx = mesh_polyvert[vert_start + 2]
          v0_mesh_local = mesh_vert[v_offset + v0_idx]
          v1_mesh_local = mesh_vert[v_offset + v1_idx]
          v2_mesh_local = mesh_vert[v_offset + v2_idx]

          if min_plane_dist <= 0.005 and _inside_triangle(w1_local, v0_mesh_local, v1_mesh_local, v2_mesh_local, 0.05):
            best_normal = wp.normalize(geom_rot @ best_normal_local)

      normal = wp.where(wp.dot(best_normal, gjk_normal) >= 0.0, best_normal, -best_normal)
      contact_pos = 0.5 * (w1 + w2)

      _write_candidate(
        max_candidates,
        dist - geom2_radius,
        contact_pos,
        normal,
        geomid,
        -1,
        flexid,
        elemid,
        vertid,
        worldid,
        warn_overflow,
        overflow_out,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        ncand_out,
      )


@cache_kernel
def _flex_plane_narrowphase(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ngeom: int,
    qpos0: wp.array2d[float],
    jnt_qposadr: wp.array[int],
    geom_type: wp.array[int],
    geom_contype: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_margin: wp.array2d[float],
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_margin: wp.array[float],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_radius: wp.array[float],
    flex_vertflexid: wp.array[int],
    flex_rod_is_box: wp.array[bool],
    flex_rod_root_clamped: wp.array[bool],
    flex_rod_tail_clamped: wp.array[bool],
    flex_rod_box_size: wp.array[wp.vec2],
    flex_rod_edgeadr: wp.array[int],
    flex_rod_edgenum: wp.array[int],
    flex_rodedge_vert: wp.array[wp.vec2i],
    flex_rodedge_jnt: wp.array[int],
    flex_elem_rodedge: wp.array[wp.vec2i],
    flex_vert_rodelem: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    plastic_in: wp.array2d[float],
    xmat_in: wp.array2d[wp.mat33],
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    flex_aabb_min_in: wp.array2d[wp.vec3],
    flex_aabb_max_in: wp.array2d[wp.vec3],
    flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
    flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
    flex_rod_tail_loc_in: wp.array3d[wp.vec3],
    # In:
    max_candidates: int,
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    cand_dist_out: wp.array[float],
    cand_pos_out: wp.array[wp.vec3],
    cand_nrm_out: wp.array[wp.vec3],
    cand_geom_out: wp.array[wp.vec2i],
    cand_flex_out: wp.array[wp.vec2i],
    cand_elem_out: wp.array[wp.vec2i],
    cand_vert_out: wp.array[wp.vec2i],
    cand_worldid_out: wp.array[int],
    ncand_out: wp.array[int],
  ):
    worldid, vertid = wp.tid()

    flexid = flex_vertflexid[vertid]
    radius = flex_radius[flexid]
    flex_margin_val = flex_margin[flexid]
    local_vertid = vertid - flex_vertadr[flexid]

    elemnum = flex_elemnum[flexid]
    is_box = bool(False)
    box_2d = wp.vec2(0.0, 0.0)
    if elemnum > 0:
      is_box, box_2d = support.rod_box_size(flex_rod_is_box, flex_rod_box_size, flexid)
    bound = wp.length(box_2d) if is_box else radius

    ok_box = bool(False)
    box_evaluated = bool(False)
    bmat = wp.mat33(0.0)

    vert = flexvert_xpos_in[worldid, vertid]
    flex_aabb_min = flex_aabb_min_in[worldid, flexid]
    flex_aabb_max = flex_aabb_max_in[worldid, flexid]
    flex_center = 0.5 * (flex_aabb_min + flex_aabb_max)
    flex_half_size = 0.5 * (flex_aabb_max - flex_aabb_min)

    for geomid in range(ngeom):
      if geom_type[geomid] != int(GeomType.PLANE):
        continue

      g_contype = geom_contype[geomid]
      g_conaffinity = geom_conaffinity[geomid]
      f_contype = flex_contype[flexid]
      f_conaffinity = flex_conaffinity[flexid]
      if not ((g_contype & f_conaffinity) or (f_contype & g_conaffinity)):
        continue

      margin = geom_margin[worldid % geom_margin.shape[0], geomid] + flex_margin_val
      geom_pos = geom_xpos_in[worldid, geomid]
      geom_rot = geom_xmat_in[worldid, geomid]
      plane_normal = wp.vec3(geom_rot[0, 2], geom_rot[1, 2], geom_rot[2, 2])

      # Coarse AABB vs plane test
      proj_half = (
        wp.abs(flex_half_size[0] * plane_normal[0])
        + wp.abs(flex_half_size[1] * plane_normal[1])
        + wp.abs(flex_half_size[2] * plane_normal[2])
      )
      diff_center = flex_center - geom_pos
      dist_center = wp.dot(diff_center, plane_normal)
      if dist_center - proj_half > margin:
        continue

      diff = vert - geom_pos
      signed_dist = wp.dot(diff, plane_normal)
      if signed_dist > margin + bound:
        continue

      if is_box and not box_evaluated:
        box_evaluated = True
        e_rod = flex_vert_rodelem[vertid]
        if e_rod >= 0:
          ok_box, _mid_box, bmat, _hsize_box = support.rod_box_elem(
            qpos0,
            jnt_qposadr,
            flex_vertadr,
            flex_elemadr,
            flex_elemnum,
            flex_elemdataadr,
            flex_plasticadr,
            flex_vertbodyid,
            flex_elem,
            flex_rod_is_box,
            flex_rod_root_clamped,
            flex_rod_tail_clamped,
            flex_rod_box_size,
            flex_rod_edgeadr,
            flex_rod_edgenum,
            flex_rodedge_vert,
            flex_rodedge_jnt,
            flex_elem_rodedge,
            qpos_in,
            plastic_in,
            xmat_in,
            flexvert_xpos_in,
            flex_rod_frame0_loc_in,
            flex_rod_frame0_ref_in,
            flex_rod_tail_loc_in,
            worldid,
            flexid,
            e_rod,
          )

      if ok_box:
        m1 = wp.vec3(bmat[0, 0], bmat[1, 0], bmat[2, 0])
        m2 = wp.vec3(bmat[0, 1], bmat[1, 1], bmat[2, 1])
        n1 = wp.dot(plane_normal, m1)
        n2 = wp.dot(plane_normal, m2)
        s1 = -1.0 if n1 >= 0.0 else 1.0
        s2 = -1.0 if n2 >= 0.0 else 1.0
        c1_0 = s1 * box_2d[0]
        c1_1 = c1_0
        c2_0 = s2 * box_2d[1]
        c2_1 = c2_0
        if box_2d[0] * wp.abs(n1) <= box_2d[1] * wp.abs(n2):
          c1_1 = -c1_0
        else:
          c2_1 = -c2_0
        c1_vec = wp.vec2(c1_0, c1_1)
        c2_vec = wp.vec2(c2_0, c2_1)
        for k in range(2):
          corner = c1_vec[k] * m1 + c2_vec[k] * m2
          ldist = wp.dot(corner, plane_normal)
          cdist = signed_dist + ldist
          if cdist < margin and not (k == 1 and ldist > 0.0):
            contact_pos = vert + corner - (0.5 * cdist) * plane_normal
            _write_candidate(
              max_candidates,
              cdist,
              contact_pos,
              plane_normal,
              geomid,
              -1,
              flexid,
              -1,
              local_vertid,
              worldid,
              wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
              overflow_out,
              cand_dist_out,
              cand_pos_out,
              cand_nrm_out,
              cand_geom_out,
              cand_flex_out,
              cand_elem_out,
              cand_vert_out,
              cand_worldid_out,
              ncand_out,
            )
        continue

      dist = signed_dist - radius

      if dist < margin:
        contact_pos = vert - plane_normal * (dist * 0.5 + radius)
        _write_candidate(
          max_candidates,
          dist,
          contact_pos,
          plane_normal,
          geomid,
          -1,
          flexid,
          -1,
          local_vertid,
          worldid,
          wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
          overflow_out,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          ncand_out,
        )

  return kernel


@wp.func
def _inside_triangle(
  # In:
  P: wp.vec3,
  A: wp.vec3,
  B: wp.vec3,
  C: wp.vec3,
  tol: float,
) -> bool:
  v0 = B - A
  v1 = C - A
  v2 = P - A
  d00 = wp.dot(v0, v0)
  d01 = wp.dot(v0, v1)
  d11 = wp.dot(v1, v1)
  d20 = wp.dot(v2, v0)
  d21 = wp.dot(v2, v1)
  denom = d00 * d11 - d01 * d01
  if wp.abs(denom) < 1e-12:
    return False
  v = (d11 * d20 - d01 * d21) / denom
  w = (d00 * d21 - d01 * d20) / denom
  u = 1.0 - v - w
  return u >= -tol and v >= -tol and w >= -tol and u <= 1.0 + tol and v <= 1.0 + tol and w <= 1.0 + tol


@wp.func
def _exclude_self_collision(
  # Model:
  flex_vertbodyid: wp.array[int],
  # In:
  v1: wp.vec4i,
  n1: int,
  v2: wp.vec4i,
  n2: int,
  vert_adr: int,
) -> bool:
  for i in range(n1):
    idx1 = v1[i]
    if idx1 >= 0:
      b1 = flex_vertbodyid[vert_adr + idx1]
      for j in range(n2):
        idx2 = v2[j]
        if idx1 == idx2:
          return True
        if idx2 >= 0 and b1 >= 0:
          b2 = flex_vertbodyid[vert_adr + idx2]
          if b1 == b2:
            return True
  return False


@wp.func
def _get_element_vertices(
  # Model:
  flex_elem: wp.array[int],
  # In:
  dim: int,
  elem_data_idx: int,
) -> wp.vec4i:
  v0 = flex_elem[elem_data_idx]
  v1 = flex_elem[elem_data_idx + 1]
  v2 = int(-1)
  v3 = int(-1)
  if dim >= 2:
    v2 = flex_elem[elem_data_idx + 2]
  if dim >= 3:
    v3 = flex_elem[elem_data_idx + 3]
  return wp.vec4i(v0, v1, v2, v3)


@wp.func
def _elem_active(
  # Model:
  flex_activelayers: wp.array[int],
  flex_dim: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemlayer: wp.array[int],
  # In:
  flexid: int,
  elemid: int,
) -> bool:
  if flex_dim[flexid] < 3:
    return True
  return flex_elemlayer[flex_elemadr[flexid] + elemid] < flex_activelayers[flexid]


@wp.kernel
def _flex_sap_project(
  # Model:
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_activelayers: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_elemlayer: wp.array[int],
  flex_radius: wp.array[float],
  flex_elemflexid: wp.array[int],
  flex_rod_is_box: wp.array[bool],
  flex_rod_box_size: wp.array[wp.vec2],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  # In:
  nelem: int,
  direction: wp.vec3,
  # Out:
  projection_lower_out: wp.array2d[float],
  projection_upper_out: wp.array2d[float],
  sort_index_out: wp.array2d[int],
  elem_aabb_lower_out: wp.array2d[wp.vec3],
  elem_aabb_upper_out: wp.array2d[wp.vec3],
  segmented_index_out: wp.array[int],
):
  worldid, elemid = wp.tid()

  flexid = flex_elemflexid[elemid]

  # Initialize sort index
  sort_index_out[worldid, elemid] = elemid

  # Compute AABB from vertex positions
  dim = flex_dim[flexid]
  vert_adr = flex_vertadr[flexid]
  elem_adr = flex_elemadr[flexid]
  e = elemid - elem_adr
  elem_data_idx = flex_elemdataadr[flexid] + e * (dim + 1)

  v = _get_element_vertices(flex_elem, dim, elem_data_idx)

  p0 = flexvert_xpos_in[worldid, vert_adr + v[0]]
  p1 = flexvert_xpos_in[worldid, vert_adr + v[1]]

  aabb_min = wp.min(p0, p1)
  aabb_max = wp.max(p0, p1)

  if dim >= 2:
    p2 = flexvert_xpos_in[worldid, vert_adr + v[2]]
    aabb_min = wp.min(aabb_min, p2)
    aabb_max = wp.max(aabb_max, p2)
  if dim >= 3:
    p3 = flexvert_xpos_in[worldid, vert_adr + v[3]]
    aabb_min = wp.min(aabb_min, p3)
    aabb_max = wp.max(aabb_max, p3)

  radius = flex_radius[flexid]
  is_box, box_2d = support.rod_box_size(flex_rod_is_box, flex_rod_box_size, flexid)
  if is_box:
    radius = wp.max(radius, wp.length(box_2d))
  margin = flex_margin[flexid]
  gap = flex_gap[flexid]
  rbound = radius + margin + gap
  inflate = wp.vec3(rbound, rbound, rbound)
  aabb_min = aabb_min - inflate
  aabb_max = aabb_max + inflate

  elem_aabb_lower_out[worldid, elemid] = aabb_min
  elem_aabb_upper_out[worldid, elemid] = aabb_max

  # Project AABB onto direction to get 1D interval
  center = 0.5 * (aabb_min + aabb_max)
  halfsize = 0.5 * (aabb_max - aabb_min)
  proj_center = wp.dot(direction, center)
  proj_radius = wp.abs(direction[0]) * halfsize[0] + wp.abs(direction[1]) * halfsize[1] + wp.abs(direction[2]) * halfsize[2]

  if not _elem_active(flex_activelayers, flex_dim, flex_elemadr, flex_elemlayer, flexid, e):
    projection_lower_out[worldid, elemid] = MJ_MAXVAL
    projection_upper_out[worldid, elemid] = MJ_MAXVAL
  else:
    projection_lower_out[worldid, elemid] = proj_center - proj_radius
    projection_upper_out[worldid, elemid] = proj_center + proj_radius

  # Segmented sort boundaries
  if elemid == 0:
    segmented_index_out[worldid] = worldid * nelem
    if worldid == nworld_in - 1:
      segmented_index_out[nworld_in] = nworld_in * nelem


@cache_kernel
def _flex_sap_sweep(is_self: bool, warn_overflow: int, enable_sat: bool = True):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_margin: wp.array[float],
    flex_gap: wp.array[float],
    flex_selfcollide: wp.array[int],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_radius: wp.array[float],
    flex_elemflexid: wp.array[int],
    # Data in:
    flexvert_xpos_in: wp.array2d[wp.vec3],
    flex_aabb_min_in: wp.array2d[wp.vec3],
    flex_aabb_max_in: wp.array2d[wp.vec3],
    # In:
    nelem: int,
    sort_index_in: wp.array2d[int],
    cumulative_sum_in: wp.array[int],
    nsweep_in: int,
    aabb_lower_in: wp.array2d[wp.vec3],
    aabb_upper_in: wp.array2d[wp.vec3],
    max_pairs: int,
    # Data out:
    ncollision_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    collision_pair_out: wp.array[wp.vec2i],
    collision_worldid_out: wp.array[int],
  ):
    worldelemid = wp.tid()

    nworldelem = cumulative_sum_in.shape[0]
    nworkpackages = cumulative_sum_in[nworldelem - 1]

    while worldelemid < nworkpackages:
      i = sap_binary_search(cumulative_sum_in, worldelemid, 0, nworldelem)
      j = i + worldelemid + 1
      if i > 0:
        j -= cumulative_sum_in[i - 1]

      worldid = i // nelem
      i = i % nelem
      j = j % nelem

      elem1 = sort_index_in[worldid, i]
      elem2 = sort_index_in[worldid, j]

      flexid1 = flex_elemflexid[elem1]
      flexid2 = flex_elemflexid[elem2]

      if wp.static(is_self):
        if flexid1 != flexid2 or flex_selfcollide[flexid1] == 0 or (flex_contype[flexid1] & flex_conaffinity[flexid1]) == 0:
          worldelemid += nsweep_in
          continue
      else:
        if flexid1 == flexid2:
          worldelemid += nsweep_in
          continue

      if elem1 > elem2:
        tmpelem = elem1
        elem1 = elem2
        elem2 = tmpelem
        tmpid = flexid1
        flexid1 = flexid2
        flexid2 = tmpid

      worldelemid += nsweep_in

      if not wp.static(is_self):
        if _flex_element_aabb_filter(
          flex_aabb_min_in[worldid, flexid1],
          flex_aabb_max_in[worldid, flexid1],
          flex_aabb_min_in[worldid, flexid2],
          flex_aabb_max_in[worldid, flexid2],
        ):
          continue

        contype1 = flex_contype[flexid1]
        conaffinity1 = flex_conaffinity[flexid1]
        contype2 = flex_contype[flexid2]
        conaffinity2 = flex_conaffinity[flexid2]
        if not ((contype1 & conaffinity2) or (contype2 & conaffinity1)):
          continue

      lower1 = aabb_lower_in[worldid, elem1]
      upper1 = aabb_upper_in[worldid, elem1]
      lower2 = aabb_lower_in[worldid, elem2]
      upper2 = aabb_upper_in[worldid, elem2]

      if _flex_element_aabb_filter(lower1, upper1, lower2, upper2):
        continue

      dim1 = flex_dim[flexid1]
      vert_adr1 = flex_vertadr[flexid1]
      elem_adr1 = flex_elemadr[flexid1]
      e1 = elem1 - elem_adr1
      elem_data_idx1 = flex_elemdataadr[flexid1] + e1 * (dim1 + 1)
      v1_indices = _get_element_vertices(flex_elem, dim1, elem_data_idx1)

      if wp.static(is_self):
        e2 = elem2 - elem_adr1
        elem_data_idx2 = flex_elemdataadr[flexid1] + e2 * (dim1 + 1)
        v2_indices = _get_element_vertices(flex_elem, dim1, elem_data_idx2)
        if _exclude_self_collision(
          flex_vertbodyid,
          v1_indices,
          dim1 + 1,
          v2_indices,
          dim1 + 1,
          vert_adr1,
        ):
          continue
      else:
        dim2 = flex_dim[flexid2]
        vert_adr2 = flex_vertadr[flexid2]
        elem_adr2 = flex_elemadr[flexid2]
        e2 = elem2 - elem_adr2
        elem_data_idx2 = flex_elemdataadr[flexid2] + e2 * (dim2 + 1)
        v2_indices = _get_element_vertices(flex_elem, dim2, elem_data_idx2)

        shared_body = bool(False)
        for ii in range(dim1 + 1):
          idx1 = v1_indices[ii]
          if idx1 >= 0:
            b1 = flex_vertbodyid[vert_adr1 + idx1]
            for jj in range(dim2 + 1):
              idx2 = v2_indices[jj]
              if idx2 >= 0 and b1 >= 0:
                b2 = flex_vertbodyid[vert_adr2 + idx2]
                if b1 == b2:
                  shared_body = bool(True)
                  break
            if shared_body:
              break

        if shared_body:
          continue

      if dim1 == 2:
        dim2 = dim1 if wp.static(is_self) else flex_dim[flexid2]
        if dim2 == 2 and wp.static(enable_sat):
          r1 = flex_radius[flexid1]
          cutoff = float(2.0) * r1
          if not wp.static(is_self):
            cutoff = (
              r1 + flex_radius[flexid2] + flex_margin[flexid1] + flex_margin[flexid2] + flex_gap[flexid1] + flex_gap[flexid2]
            )
          cutoff_sq = cutoff * cutoff
          p0 = flexvert_xpos_in[worldid, vert_adr1 + v1_indices[0]]
          p1 = flexvert_xpos_in[worldid, vert_adr1 + v1_indices[1]]
          p2 = flexvert_xpos_in[worldid, vert_adr1 + v1_indices[2]]
          vadr2 = vert_adr1 if wp.static(is_self) else flex_vertadr[flexid2]
          q0 = flexvert_xpos_in[worldid, vadr2 + v2_indices[0]]
          q1 = flexvert_xpos_in[worldid, vadr2 + v2_indices[1]]
          q2 = flexvert_xpos_in[worldid, vadr2 + v2_indices[2]]
          if _triangle_sat_separated(p0, p1, p2, q0, q1, q2, cutoff_sq):
            continue

      idx = wp.atomic_add(ncollision_out, 0, 1)  # kernel_analyzer: ignore[atomic]
      if idx >= max_pairs:
        if wp.static(bool(warn_overflow & OverflowType.BROADPHASE)):
          wp.printf(
            "Flex SAP buffer overflow - please increase naconmax beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.BROADPHASE (or = 0 for all)\n",
            max_pairs,
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.BROADPHASE))
        return

      collision_pair_out[idx] = wp.vec2i(elem1, elem2)
      collision_worldid_out[idx] = worldid

  return kernel


@cache_kernel
def _flex_narrowphase(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_ccd_tolerance: wp.array[float],
    qpos0: wp.array2d[float],
    jnt_qposadr: wp.array[int],
    flex_margin: wp.array[float],
    flex_gap: wp.array[float],
    flex_activelayers: wp.array[int],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_elemlayer: wp.array[int],
    flex_radius: wp.array[float],
    flex_elemflexid: wp.array[int],
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
    naccdmax_in: int,
    ncollision_in: wp.array[int],
    flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
    flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
    flex_rod_tail_loc_in: wp.array3d[wp.vec3],
    # In:
    max_candidates: int,
    gjk_iterations: int,
    epa_iterations: int,
    collision_pair_in: wp.array[wp.vec2i],
    collision_worldid_in: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    workspace_verts_out: wp.array[wp.vec3],
    epa_vert_out: wp.array2d[wp.vec3],
    epa_vert_index_out: wp.array2d[int],
    epa_face_out: wp.array2d[int],
    epa_pr_out: wp.array2d[wp.vec3],
    epa_norm2_out: wp.array2d[float],
    epa_horizon_out: wp.array2d[int],
    cand_dist_out: wp.array[float],
    cand_pos_out: wp.array[wp.vec3],
    cand_nrm_out: wp.array[wp.vec3],
    cand_geom_out: wp.array[wp.vec2i],
    cand_flex_out: wp.array[wp.vec2i],
    cand_elem_out: wp.array[wp.vec2i],
    cand_vert_out: wp.array[wp.vec2i],
    cand_worldid_out: wp.array[int],
    ncand_out: wp.array[int],
  ):
    """Process broadphase-identified flex pairs through narrowphase."""
    pairid = wp.tid()

    # Check bounds
    if pairid >= ncollision_in[0]:
      return

    pair = collision_pair_in[pairid]
    elem1_global = pair[0]
    elem2_global = pair[1]
    worldid = collision_worldid_in[pairid]

    flexid1 = flex_elemflexid[elem1_global]
    flexid2 = flex_elemflexid[elem2_global]

    radius1 = flex_radius[flexid1]
    radius2 = flex_radius[flexid2]
    dim1 = flex_dim[flexid1]
    dim2 = flex_dim[flexid2]
    vert_adr1 = flex_vertadr[flexid1]
    vert_adr2 = flex_vertadr[flexid2]
    elem_adr1 = flex_elemadr[flexid1]
    elem_adr2 = flex_elemadr[flexid2]

    e1 = elem1_global - elem_adr1
    e2 = elem2_global - elem_adr2

    if not _elem_active(flex_activelayers, flex_dim, flex_elemadr, flex_elemlayer, flexid1, e1):
      return
    if not _elem_active(flex_activelayers, flex_dim, flex_elemadr, flex_elemlayer, flexid2, e2):
      return
    elem_data_idx1 = flex_elemdataadr[flexid1] + e1 * (dim1 + 1)
    elem_data_idx2 = flex_elemdataadr[flexid2] + e2 * (dim2 + 1)

    v1_indices = _get_element_vertices(flex_elem, dim1, elem_data_idx1)
    v2_indices = _get_element_vertices(flex_elem, dim2, elem_data_idx2)

    # Workspace for this pair (8 slots per pair)
    offset1 = pairid * 8
    for idx in range(dim1 + 1):
      workspace_verts_out[offset1 + idx] = flexvert_xpos_in[worldid, vert_adr1 + v1_indices[idx]]

    mix_margin = 0.0
    if flexid1 != flexid2:
      mix_margin = flex_margin[flexid1] + flex_margin[flexid2] + flex_gap[flexid1] + flex_gap[flexid2]

    if dim1 == 1 and dim2 == 1:
      box1, bpos1, bmat1, bsize1 = support.rod_box_elem(
        qpos0,
        jnt_qposadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_plasticadr,
        flex_vertbodyid,
        flex_elem,
        flex_rod_is_box,
        flex_rod_root_clamped,
        flex_rod_tail_clamped,
        flex_rod_box_size,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_vert,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        qpos_in,
        plastic_in,
        xmat_in,
        flexvert_xpos_in,
        flex_rod_frame0_loc_in,
        flex_rod_frame0_ref_in,
        flex_rod_tail_loc_in,
        worldid,
        flexid1,
        e1,
      )
      box2, bpos2, bmat2, bsize2 = support.rod_box_elem(
        qpos0,
        jnt_qposadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_plasticadr,
        flex_vertbodyid,
        flex_elem,
        flex_rod_is_box,
        flex_rod_root_clamped,
        flex_rod_tail_clamped,
        flex_rod_box_size,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_vert,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        qpos_in,
        plastic_in,
        xmat_in,
        flexvert_xpos_in,
        flex_rod_frame0_loc_in,
        flex_rod_frame0_ref_in,
        flex_rod_tail_loc_in,
        worldid,
        flexid2,
        e2,
      )
      if box1 and box2:
        ncon_bb, bb_dists, bb_poss, bb_nrm = collision_primitive_core.box_box(
          bpos1, bmat1, bsize1, bpos2, bmat2, bsize2, mix_margin
        )
        for c in range(ncon_bb):
          d_val = bb_dists[c]
          if d_val <= mix_margin:
            _write_candidate(
              max_candidates,
              d_val,
              bb_poss[c],
              bb_nrm,
              -2,
              flexid1,
              flexid2,
              e1,
              e2,
              worldid,
              wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
              overflow_out,
              cand_dist_out,
              cand_pos_out,
              cand_nrm_out,
              cand_geom_out,
              cand_flex_out,
              cand_elem_out,
              cand_vert_out,
              cand_worldid_out,
              ncand_out,
            )
        return

      p0 = workspace_verts_out[offset1]
      p1 = workspace_verts_out[offset1 + 1]
      cap1_pos = 0.5 * (p0 + p1)
      cap1_axis = wp.normalize(p1 - p0)
      cap1_half_len = 0.5 * wp.length(p1 - p0)

      p2_0 = flexvert_xpos_in[worldid, vert_adr2 + v2_indices[0]]
      p2_1 = flexvert_xpos_in[worldid, vert_adr2 + v2_indices[1]]
      cap2_pos = 0.5 * (p2_0 + p2_1)
      cap2_axis = wp.normalize(p2_1 - p2_0)
      cap2_half_len = 0.5 * wp.length(p2_1 - p2_0)

      if box1:
        contact_dist, contact_pos, box_nrms = collision_primitive_core.capsule_box(
          cap2_pos,
          cap2_axis,
          radius2,
          cap2_half_len,
          bpos1,
          bmat1,
          bsize1,
        )
        contact_normal = -box_nrms
      elif box2:
        contact_dist, contact_pos, contact_normal = collision_primitive_core.capsule_box(
          cap1_pos,
          cap1_axis,
          radius1,
          cap1_half_len,
          bpos2,
          bmat2,
          bsize2,
        )
      else:
        contact_dist, contact_pos, contact_normal = collision_primitive_core.capsule_capsule(
          cap1_pos,
          cap1_axis,
          radius1,
          cap1_half_len,
          cap2_pos,
          cap2_axis,
          radius2,
          cap2_half_len,
        )

      for c in range(2):
        d_val = contact_dist[c]
        if d_val <= mix_margin:
          _write_candidate(
            max_candidates,
            d_val,
            contact_pos[c],
            contact_normal[c],
            -2,
            flexid1,
            flexid2,
            e1,
            e2,
            worldid,
            wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
            overflow_out,
            cand_dist_out,
            cand_pos_out,
            cand_nrm_out,
            cand_geom_out,
            cand_flex_out,
            cand_elem_out,
            cand_vert_out,
            cand_worldid_out,
            ncand_out,
          )
    else:
      if pairid >= naccdmax_in:
        if wp.static(bool(warn_overflow & OverflowType.CCD)):
          wp.printf(
            "CCD overflow in flex narrowphase - please increase naccdmax beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",
            naccdmax_in,
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.CCD))
        return

      # GJK/EPA narrowphase
      offset2 = pairid * 8 + 4
      for idx in range(dim2 + 1):
        workspace_verts_out[offset2 + idx] = flexvert_xpos_in[worldid, vert_adr2 + v2_indices[idx]]

      geom1 = Geom()
      geom1.pos = wp.vec3(0.0)
      geom1.rot = wp.identity(n=3, dtype=float)
      geom1.size = wp.vec3(0.0)
      geom1.margin = 2.0 * radius1 + mix_margin
      geom1.vert = workspace_verts_out
      geom1.vertadr = offset1
      geom1.vertnum = dim1 + 1
      geom1.graphadr = -1
      geom1.index = -1

      geom2 = Geom()
      geom2.pos = wp.vec3(0.0)
      geom2.rot = wp.identity(n=3, dtype=float)
      geom2.size = wp.vec3(0.0)
      geom2.margin = 2.0 * radius2 + mix_margin
      geom2.vert = workspace_verts_out
      geom2.vertadr = offset2
      geom2.vertnum = dim2 + 1
      geom2.graphadr = -1
      geom2.index = -1

      center1 = wp.vec3(0.0)
      for idx in range(dim1 + 1):
        center1 += workspace_verts_out[offset1 + idx]
      center1 = center1 / float(dim1 + 1)

      center2 = wp.vec3(0.0)
      for idx in range(dim2 + 1):
        center2 += workspace_verts_out[offset2 + idx]
      center2 = center2 / float(dim2 + 1)

      tol = opt_ccd_tolerance[0 % opt_ccd_tolerance.shape[0]]

      dist, ncontact, w1, w2, _ = ccd(
        tol,
        radius1 + radius2 + mix_margin,
        gjk_iterations,
        epa_iterations,
        geom1,
        geom2,
        int(GeomType.MESH),
        int(GeomType.MESH),
        center1,
        center2,
        epa_vert_out[pairid],
        epa_vert_index_out[pairid],
        epa_face_out[pairid],
        epa_pr_out[pairid],
        epa_norm2_out[pairid],
        epa_horizon_out[pairid],
        wp.static(bool(warn_overflow & OverflowType.EPA_HORIZON)),
        worldid,
        overflow_out,
      )

      if ncontact > 0 and dist < 0.0:
        phys_dist = dist + mix_margin
        pos = 0.5 * (w1 + w2)
        nrm = wp.normalize(w1 - w2)
        _write_candidate(
          max_candidates,
          phys_dist,
          pos,
          nrm,
          -2,
          flexid1,
          flexid2,
          e1,
          e2,
          worldid,
          wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
          overflow_out,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          ncand_out,
        )

  return kernel


@cache_kernel
def _flex_narrowphase_elem_detect(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    ngeom: int,
    opt_ccd_tolerance: wp.array[float],
    qpos0: wp.array2d[float],
    jnt_qposadr: wp.array[int],
    geom_type: wp.array[int],
    geom_contype: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_aabb: wp.array3d[wp.vec3],
    geom_margin: wp.array2d[float],
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_margin: wp.array[float],
    flex_activelayers: wp.array[int],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_plasticadr: wp.array[int],
    flex_vertbodyid: wp.array[int],
    flex_elem: wp.array[int],
    flex_elemlayer: wp.array[int],
    flex_radius: wp.array[float],
    mesh_vertadr: wp.array[int],
    mesh_vertnum: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_vert: wp.array[wp.vec3],
    mesh_graph: wp.array[int],
    mesh_pos: wp.array[wp.vec3],
    mesh_polynormal: wp.array[wp.vec3],
    mesh_polyvertadr: wp.array[int],
    mesh_polyvert: wp.array[int],
    mesh_polymapadr: wp.array[int],
    mesh_polymapnum: wp.array[int],
    mesh_polymap: wp.array[int],
    flex_elemflexid: wp.array[int],
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
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    naccdmax_in: int,
    flex_aabb_min_in: wp.array2d[wp.vec3],
    flex_aabb_max_in: wp.array2d[wp.vec3],
    flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
    flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
    flex_rod_tail_loc_in: wp.array3d[wp.vec3],
    # In:
    epa_vert: wp.array2d[wp.vec3],
    epa_vert_index: wp.array2d[int],
    epa_face: wp.array2d[int],
    epa_pr: wp.array2d[wp.vec3],
    epa_norm2: wp.array2d[float],
    epa_horizon: wp.array2d[int],
    nccd: wp.array[int],
    ccd_iterations: int,
    max_candidates: int,
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    cand_dist_out: wp.array[float],
    cand_pos_out: wp.array[wp.vec3],
    cand_nrm_out: wp.array[wp.vec3],
    cand_geom_out: wp.array[wp.vec2i],
    cand_flex_out: wp.array[wp.vec2i],
    cand_elem_out: wp.array[wp.vec2i],
    cand_vert_out: wp.array[wp.vec2i],
    cand_worldid_out: wp.array[int],
    ncand_out: wp.array[int],
  ):
    worldid, elemid = wp.tid()

    flexid = flex_elemflexid[elemid]
    f_dim = flex_dim[flexid]

    # skip inactive elements of 3D flexes, see mj_isElemActive
    if f_dim == 3 and flex_elemlayer[elemid] >= flex_activelayers[flexid]:
      return

    local_elemid = elemid - flex_elemadr[flexid]
    vert_adr = flex_vertadr[flexid]
    elem_radius = flex_radius[flexid]
    elem_margin = flex_margin[flexid]

    geom2 = Geom()
    geom2.pos = wp.vec3(0.0, 0.0, 0.0)
    geom2.rot = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    geom2.margin = 0.0
    geom2.index = -1

    centroid = wp.vec3(0.0, 0.0, 0.0)
    r_elem = float(0.0)
    cap_axis = wp.vec3(0.0, 0.0, 1.0)
    geom2_type = int(GeomType.TRIANGLE)
    elem_min = wp.vec3(0.0, 0.0, 0.0)
    elem_max = wp.vec3(0.0, 0.0, 0.0)
    t1 = wp.vec3(0.0, 0.0, 0.0)
    t2 = wp.vec3(0.0, 0.0, 0.0)
    t3 = wp.vec3(0.0, 0.0, 0.0)
    p0 = wp.vec3(0.0, 0.0, 0.0)
    p1 = wp.vec3(0.0, 0.0, 0.0)
    p2 = wp.vec3(0.0, 0.0, 0.0)
    p3 = wp.vec3(0.0, 0.0, 0.0)
    v0 = int(-1)
    v1 = int(-1)
    v2 = int(-1)
    v3 = int(-1)
    ok_box = False
    bpos = wp.vec3(0.0, 0.0, 0.0)
    bmat = wp.mat33(0.0)
    bsize = wp.vec3(0.0, 0.0, 0.0)

    if f_dim == 1:
      edata_idx = flex_elemdataadr[flexid] + local_elemid * 2
      v0 = flex_elem[edata_idx]
      v1 = flex_elem[edata_idx + 1]

      t1 = flexvert_xpos_in[worldid, vert_adr + v0]
      t2 = flexvert_xpos_in[worldid, vert_adr + v1]

      centroid = 0.5 * (t1 + t2)
      seg = t1 - t2
      seg_len = wp.length(seg)
      r_elem = 0.5 * seg_len
      if seg_len > MJ_MINVAL:
        cap_axis = seg / seg_len
      b_ax, c_ax = orthogonals(cap_axis)

      geom2.pos = centroid
      geom2.rot = wp.matrix_from_cols(b_ax, c_ax, cap_axis)
      geom2.size = wp.vec3(elem_radius, r_elem, 0.0)
      geom2_type = int(GeomType.CAPSULE)

      ok_box, bpos, bmat, bsize = support.rod_box_elem(
        qpos0,
        jnt_qposadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_plasticadr,
        flex_vertbodyid,
        flex_elem,
        flex_rod_is_box,
        flex_rod_root_clamped,
        flex_rod_tail_clamped,
        flex_rod_box_size,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_vert,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        qpos_in,
        plastic_in,
        xmat_in,
        flexvert_xpos_in,
        flex_rod_frame0_loc_in,
        flex_rod_frame0_ref_in,
        flex_rod_tail_loc_in,
        worldid,
        flexid,
        local_elemid,
      )
      pad = elem_radius
      if ok_box:
        pad = wp.max(pad, wp.sqrt(bsize[0] * bsize[0] + bsize[1] * bsize[1]))

      elem_min = wp.min(t1, t2) - wp.vec3(pad, pad, pad)
      elem_max = wp.max(t1, t2) + wp.vec3(pad, pad, pad)
    elif f_dim == 2:
      edata_idx = flex_elemdataadr[flexid] + local_elemid * 3
      v0 = flex_elem[edata_idx]
      v1 = flex_elem[edata_idx + 1]
      v2 = flex_elem[edata_idx + 2]

      t1 = flexvert_xpos_in[worldid, vert_adr + v0]
      t2 = flexvert_xpos_in[worldid, vert_adr + v1]
      t3 = flexvert_xpos_in[worldid, vert_adr + v2]

      geom2.rot = wp.mat33(t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], t3[0], t3[1], t3[2])
      geom2_type = int(GeomType.TRIANGLE)
      centroid = (t1 + t2 + t3) * wp.static(1.0 / 3.0)
      d1 = wp.length(t1 - centroid)
      d2 = wp.length(t2 - centroid)
      d3 = wp.length(t3 - centroid)
      r_elem = wp.max(d1, wp.max(d2, d3))

      elem_min = wp.min(t1, wp.min(t2, t3)) - wp.vec3(elem_radius, elem_radius, elem_radius)
      elem_max = wp.max(t1, wp.max(t2, t3)) + wp.vec3(elem_radius, elem_radius, elem_radius)
    elif f_dim == 3:
      edata_idx = flex_elemdataadr[flexid] + local_elemid * 4
      v0 = flex_elem[edata_idx]
      v1 = flex_elem[edata_idx + 1]
      v2 = flex_elem[edata_idx + 2]
      v3 = flex_elem[edata_idx + 3]

      p0 = flexvert_xpos_in[worldid, vert_adr + v0]
      p1 = flexvert_xpos_in[worldid, vert_adr + v1]
      p2 = flexvert_xpos_in[worldid, vert_adr + v2]
      p3 = flexvert_xpos_in[worldid, vert_adr + v3]

      geom2.polyvert = mat63(
        p0[0],
        p0[1],
        p0[2],
        p1[0],
        p1[1],
        p1[2],
        p2[0],
        p2[1],
        p2[2],
        p3[0],
        p3[1],
        p3[2],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
      )
      geom2_type = int(GeomType.FLEX)
      geom2.margin = 2.0 * elem_radius
      centroid = (p0 + p1 + p2 + p3) * wp.static(0.25)
      d0 = wp.length(p0 - centroid)
      d1 = wp.length(p1 - centroid)
      d2 = wp.length(p2 - centroid)
      d3 = wp.length(p3 - centroid)
      r_elem = wp.max(d0, wp.max(d1, wp.max(d2, d3)))

      elem_min = wp.min(p0, wp.min(p1, p2))
      elem_min = wp.min(elem_min, p3) - wp.vec3(elem_radius, elem_radius, elem_radius)
      elem_max = wp.max(p0, wp.max(p1, p2))
      elem_max = wp.max(elem_max, p3) + wp.vec3(elem_radius, elem_radius, elem_radius)
    else:
      return

    b0 = flex_vertbodyid[vert_adr + v0]
    b1 = flex_vertbodyid[vert_adr + v1]
    b2 = int(-1)
    b3 = int(-1)
    if f_dim >= 2:
      b2 = flex_vertbodyid[vert_adr + v2]
    if f_dim == 3:
      b3 = flex_vertbodyid[vert_adr + v3]

    f_contype = flex_contype[flexid]
    f_conaffinity = flex_conaffinity[flexid]
    flex_aabb_min_val = flex_aabb_min_in[worldid, flexid]
    flex_aabb_max_val = flex_aabb_max_in[worldid, flexid]

    for geomid in range(ngeom):
      gtype = geom_type[geomid]
      if (
        gtype != int(GeomType.SPHERE)
        and gtype != int(GeomType.CAPSULE)
        and gtype != int(GeomType.BOX)
        and gtype != int(GeomType.CYLINDER)
        and gtype != int(GeomType.ELLIPSOID)
        and gtype != int(GeomType.MESH)
      ):
        continue

      g_contype = geom_contype[geomid]
      g_conaffinity = geom_conaffinity[geomid]
      if not ((g_contype & f_conaffinity) or (f_contype & g_conaffinity)):
        continue

      # skip if element has vertices on the same body as geom
      b = geom_bodyid[geomid]
      if b >= 0 and (b == b0 or b == b1 or b == b2 or b == b3):
        continue

      geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
      margin = geom_margin_val + elem_margin

      geom_pos = geom_xpos_in[worldid, geomid]
      geom_rot = geom_xmat_in[worldid, geomid]
      geom_size_val = geom_size[worldid % geom_size.shape[0], geomid]

      # Stage 1: Coarse flex object AABB vs Geom world AABB check
      aabb_id = worldid % geom_aabb.shape[0]
      geom_center_local = geom_aabb[aabb_id, geomid, 0]
      geom_half_size_local = geom_aabb[aabb_id, geomid, 1]
      geom_center_global = geom_rot @ geom_center_local + geom_pos
      geom_half_size_global = wp.vec3(
        wp.abs(geom_rot[0, 0]) * geom_half_size_local[0]
        + wp.abs(geom_rot[0, 1]) * geom_half_size_local[1]
        + wp.abs(geom_rot[0, 2]) * geom_half_size_local[2],
        wp.abs(geom_rot[1, 0]) * geom_half_size_local[0]
        + wp.abs(geom_rot[1, 1]) * geom_half_size_local[1]
        + wp.abs(geom_rot[1, 2]) * geom_half_size_local[2],
        wp.abs(geom_rot[2, 0]) * geom_half_size_local[0]
        + wp.abs(geom_rot[2, 1]) * geom_half_size_local[1]
        + wp.abs(geom_rot[2, 2]) * geom_half_size_local[2],
      )
      inflate = wp.vec3(margin, margin, margin)
      geom_box_min = geom_center_global - geom_half_size_global - inflate
      geom_box_max = geom_center_global + geom_half_size_global + inflate

      if _flex_element_aabb_filter(geom_box_min, geom_box_max, flex_aabb_min_val, flex_aabb_max_val):
        continue

      # Stage 2: Element AABB vs Geom world AABB check
      if _flex_element_aabb_filter(geom_box_min, geom_box_max, elem_min, elem_max):
        continue

      # sphere/capsule : tetrahedron, analytic (falls back to ccd for degenerate tetrahedra)
      if f_dim == 3 and (gtype == int(GeomType.SPHERE) or gtype == int(GeomType.CAPSULE)):
        cap_len = wp.where(gtype == int(GeomType.CAPSULE), geom_size_val[1], 0.0)
        cap_axis_g = geom_rot[:, 2]
        valid, cdist, cpos, cnrm = collision_primitive_core.capsule_tetrahedron(
          geom_pos - cap_len * cap_axis_g, geom_pos + cap_len * cap_axis_g, geom_size_val[0], p0, p1, p2, p3, elem_radius
        )
        if valid:
          if cdist < margin:
            _write_candidate(
              max_candidates,
              cdist,
              cpos,
              cnrm,
              geomid,
              -1,
              flexid,
              local_elemid,
              -1,
              worldid,
              wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
              overflow_out,
              cand_dist_out,
              cand_pos_out,
              cand_nrm_out,
              cand_geom_out,
              cand_flex_out,
              cand_elem_out,
              cand_vert_out,
              cand_worldid_out,
              ncand_out,
            )
          continue

      if gtype == int(GeomType.MESH):
        ccdid = wp.atomic_add(nccd, 0, 1)  # kernel_analyzer: ignore[atomic]
        if ccdid >= naccdmax_in:
          if wp.static(bool(warn_overflow & OverflowType.CCD)):
            wp.printf(
              "CCD overflow in flex narrowphase - please increase naccdmax beyond %u\n"
              "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",
              naccdmax_in,
            )
          wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.CCD))
          continue

        did = geom_dataid[worldid % geom_dataid.shape[0], geomid]
        tolerance = opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]]

        geom2_r = elem_radius if f_dim == 2 else 0.0
        geom2_rbound = r_elem if f_dim == 2 else (r_elem + elem_radius)
        _collide_mesh_convex(
          mesh_vertadr,
          mesh_vertnum,
          mesh_graphadr,
          mesh_vert,
          mesh_graph,
          mesh_pos,
          mesh_polynormal,
          mesh_polyvertadr,
          mesh_polyvert,
          mesh_polymapadr,
          mesh_polymapnum,
          mesh_polymap,
          max_candidates,
          geom_pos,
          geom_rot,
          geom_size_val,
          did,
          geom2,
          geom2_type,
          centroid,
          geom2_rbound,
          geom2_r,
          margin,
          geomid,
          flexid,
          local_elemid,
          -1,
          worldid,
          epa_vert[ccdid],
          epa_vert_index[ccdid],
          epa_face[ccdid],
          epa_pr[ccdid],
          epa_norm2[ccdid],
          epa_horizon[ccdid],
          tolerance,
          ccd_iterations,
          wp.static(bool(warn_overflow & OverflowType.EPA_HORIZON)),
          overflow_out,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          ncand_out,
        )

      elif f_dim == 1 and (gtype == int(GeomType.SPHERE) or gtype == int(GeomType.CAPSULE) or gtype == int(GeomType.BOX)):
        if ok_box:
          _collide_geom_box_rod_detect(
            max_candidates,
            gtype,
            geom_pos,
            geom_rot,
            geom_size_val,
            bpos,
            bmat,
            bsize,
            margin,
            geomid,
            flexid,
            local_elemid,
            -1,
            worldid,
            wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
            overflow_out,
            cand_dist_out,
            cand_pos_out,
            cand_nrm_out,
            cand_geom_out,
            cand_flex_out,
            cand_elem_out,
            cand_vert_out,
            cand_worldid_out,
            ncand_out,
          )
        else:
          _collide_geom_capsule_detect(
            max_candidates,
            gtype,
            geom_pos,
            geom_rot,
            geom_size_val,
            centroid,
            cap_axis,
            elem_radius,
            r_elem,
            margin,
            geomid,
            flexid,
            local_elemid,
            -1,
            worldid,
            wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
            overflow_out,
            cand_dist_out,
            cand_pos_out,
            cand_nrm_out,
            cand_geom_out,
            cand_flex_out,
            cand_elem_out,
            cand_vert_out,
            cand_worldid_out,
            ncand_out,
          )

      elif f_dim == 2 and (
        gtype == int(GeomType.SPHERE)
        or gtype == int(GeomType.CAPSULE)
        or gtype == int(GeomType.BOX)
        or gtype == int(GeomType.CYLINDER)
      ):
        _collide_geom_triangle_detect(
          max_candidates,
          gtype,
          geom_pos,
          geom_rot,
          geom_size_val,
          t1,
          t2,
          t3,
          elem_radius,
          margin,
          geomid,
          flexid,
          local_elemid,
          -1,
          worldid,
          wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
          overflow_out,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          ncand_out,
        )

      else:
        ccdid = wp.atomic_add(nccd, 0, 1)  # kernel_analyzer: ignore[atomic]
        if ccdid >= naccdmax_in:
          if wp.static(bool(warn_overflow & OverflowType.CCD)):
            wp.printf(
              "CCD overflow in flex narrowphase - please increase naccdmax beyond %u\n"
              "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",
              naccdmax_in,
            )
          wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.CCD))
          continue

        tolerance = opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]]

        geom1 = Geom()
        geom1.pos = geom_pos
        geom1.rot = geom_rot
        geom1.size = geom_size_val
        geom1.margin = 0.0
        geom1.index = -1

        r_geom = wp.length(geom_size_val)
        if wp.length(centroid - geom_pos) <= r_geom + r_elem + margin + elem_radius + 0.04:
          ccd_cutoff = (margin + elem_radius) if f_dim == 2 else margin
          dist, ncontact, w1, w2, idx = ccd(
            tolerance,
            ccd_cutoff,
            ccd_iterations,
            ccd_iterations,
            geom1,
            geom2,
            gtype,
            geom2_type,
            geom_pos,
            centroid,
            epa_vert[ccdid],
            epa_vert_index[ccdid],
            epa_face[ccdid],
            epa_pr[ccdid],
            epa_norm2[ccdid],
            epa_horizon[ccdid],
            wp.static(bool(warn_overflow & OverflowType.EPA_HORIZON)),
            worldid,
            overflow_out,
          )

          cand_dist = (dist - elem_radius) if f_dim == 2 else dist
          if ncontact > 0 and cand_dist < margin:
            diff = (w1 - w2) if dist < 0.0 else (w2 - w1)
            normal = wp.normalize(diff)
            if wp.length_sq(diff) < 1e-12:
              normal = wp.normalize(centroid - geom_pos)

            contact_pos = 0.5 * (w1 + w2)
            if f_dim == 2:
              contact_pos -= 0.5 * elem_radius * normal

            _write_candidate(
              max_candidates,
              cand_dist,
              contact_pos,
              normal,
              geomid,
              -1,
              flexid,
              local_elemid,
              -1,
              worldid,
              wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)),
              overflow_out,
              cand_dist_out,
              cand_pos_out,
              cand_nrm_out,
              cand_geom_out,
              cand_flex_out,
              cand_elem_out,
              cand_vert_out,
              cand_worldid_out,
              ncand_out,
            )

  return kernel


@wp.kernel
def _compute_filter_key(
  # Model:
  ngeom: int,
  nflex: int,
  # In:
  ncand: wp.array[int],
  cand_geom: wp.array[wp.vec2i],
  cand_flex: wp.array[wp.vec2i],
  cand_pos: wp.array[wp.vec3],
  cand_worldid: wp.array[int],
  # Out:
  key_out: wp.array[wp.int64],
  val_out: wp.array[int],
):
  """Compute sort key for candidate grouping.

  Groups candidates by (worldid, flex_id, geom_id) in high bits and spatial projection in low bits.
  """
  i = wp.tid()
  if i >= ncand[0]:
    key_out[i] = wp.int64(9223372036854775807)  # (largest int64 value): sort unused entries to end
    val_out[i] = i
    return

  worldid = cand_worldid[i]
  flex_id = cand_flex[i][1]
  geom_id = cand_geom[i][0]
  flex1 = cand_flex[i][0]

  if geom_id >= 0:
    group = geom_id * nflex + flex_id
  else:
    f1 = wp.min(flex1, flex_id)
    f2 = wp.max(flex1, flex_id)
    group = ngeom * nflex + f1 * nflex + f2

  group_id = worldid * (ngeom * nflex + nflex * nflex) + group
  is_self = int(flex1 == flex_id) if geom_id < 0 else 0

  group_key = (wp.int64(group_id) << wp.int64(1)) | wp.int64(is_self)

  # Spatial projection key: project position onto diagonal vector u = (1, 1, 1) / sqrt(3).
  # Clamped monotonic 1D integer mapping with 1 um resolution.
  p = cand_pos[i]
  s = (p[0] + p[1] + p[2]) * wp.static(1.0 / math.sqrt(3.0))
  spatial_key = wp.clamp(wp.int64(s * 1000000.0) + wp.int64(100000000), wp.int64(0), wp.int64(2147483647))

  key_out[i] = (group_key << wp.int64(32)) | spatial_key
  val_out[i] = i


@wp.func
def _compare_candidate(
  # In:
  dist_i: float,
  dist_j: float,
  cand_idx_i: int,
  cand_idx_j: int,
) -> bool:
  """Returns True if candidate i is dominated by (and should yield to) candidate j."""
  if dist_j < dist_i:
    return True
  if dist_j == dist_i and cand_idx_j < cand_idx_i:
    return True
  return False


# Note: MuJoCo Warp uses proximity deduplication (1 mm threshold) instead of
# Farthest Point Sampling (FPS) limited to mjMAXCONPAIR for flex-geom collisions.
# Flex self-collision and flex-flex collision apply proximity deduplication before FPS.
# TODO(thowell): Figure out an FPS version that utilizes a configurable mj_maxconpair.
@wp.kernel
def _filter_flex_candidates_sorted(
  # In:
  ncand: wp.array[int],
  epsilon: float,
  sort_key: wp.array[wp.int64],
  sort_val: wp.array[int],
  cand_dist: wp.array[float],
  cand_pos: wp.array[wp.vec3],
  # Out:
  cand_active_out: wp.array[int],
):
  """Filter duplicate candidates using sorted order.

  After sorting by group key, candidates in the same group are contiguous.
  Each candidate only compares with neighbors sharing the same key, reducing
  complexity from O(n^2) to O(n * k) where k is the average group size.
  """
  si = wp.tid()
  ncand_limit = wp.min(ncand[0], cand_active_out.shape[0])
  if si >= ncand_limit:
    if si < cand_active_out.shape[0]:
      cand_active_out[sort_val[si]] = 0
    return

  i = sort_val[si]
  my_key = sort_key[si]
  my_group = my_key >> wp.int64(32)
  my_spatial = my_key & wp.int64(0x7FFFFFFF)
  eps_key = wp.int64(wp.ceil(epsilon * 1000000.0)) + wp.int64(1)
  pos_i = cand_pos[i]
  dist_i = cand_dist[i]
  eps2 = epsilon * epsilon

  keep = int(1)

  # Compare with same-key neighbors (backward)
  j = si - 1
  while j >= 0:
    key_j = sort_key[j]
    if (key_j >> wp.int64(32)) != my_group:
      break
    spatial_j = key_j & wp.int64(0x7FFFFFFF)
    if my_spatial - spatial_j >= eps_key:
      break

    oj = sort_val[j]
    pos_j = cand_pos[oj]
    diff = pos_i - pos_j
    if wp.dot(diff, diff) < eps2:
      if _compare_candidate(dist_i, cand_dist[oj], i, oj):
        keep = 0
        break
    j -= 1

  # Compare with same-key neighbors (forward)
  if keep == 1:
    j = si + 1
    while j < ncand_limit:
      key_j = sort_key[j]
      if (key_j >> wp.int64(32)) != my_group:
        break
      spatial_j = key_j & wp.int64(0x7FFFFFFF)
      if spatial_j - my_spatial >= eps_key:
        break

      oj = sort_val[j]
      pos_j = cand_pos[oj]
      diff = pos_i - pos_j
      if wp.dot(diff, diff) < eps2:
        if _compare_candidate(dist_i, cand_dist[oj], i, oj):
          keep = 0
          break
      j += 1

  cand_active_out[i] = keep


@cache_kernel
def _write_filtered_contacts(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_integrator: int,
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_condim: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_priority: wp.array[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    flex_condim: wp.array[int],
    flex_priority: wp.array[int],
    flex_solmix: wp.array[float],
    flex_solref: wp.array[wp.vec2],
    flex_solimp: wp.array[vec5],
    flex_friction: wp.array[wp.vec3],
    flex_margin: wp.array[float],
    flex_gap: wp.array[float],
    flex_passive: wp.array[int],
    flex_dim: wp.array[int],
    flex_interp: wp.array[int],
    # Data in:
    naconmax_in: int,
    # In:
    ncand: wp.array[int],
    cand_dist: wp.array[float],
    cand_pos: wp.array[wp.vec3],
    cand_nrm: wp.array[wp.vec3],
    cand_geom: wp.array[wp.vec2i],
    cand_flex: wp.array[wp.vec2i],
    cand_elem: wp.array[wp.vec2i],
    cand_vert: wp.array[wp.vec2i],
    cand_worldid: wp.array[int],
    cand_active: wp.array[int],
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_flex_out: wp.array[wp.vec2i],
    contact_elem_out: wp.array[wp.vec2i],
    contact_vert_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    contact_type_out: wp.array[int],
    contact_geomcollisionid_out: wp.array[int],
    nacon_out: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
  ):
    i = wp.tid()
    if i >= ncand[0]:
      return

    if cand_active[i] == 0:
      return

    geomid = cand_geom[i][0]
    worldid = cand_worldid[i]

    condim = int(0)
    margin = float(0.0)
    gap = float(0.0)
    solref = wp.vec2(0.0, 0.0)
    solimp = vec5(0.0, 0.0, 0.0, 0.0, 0.0)
    friction = vec5(0.0, 0.0, 0.0, 0.0, 0.0)

    if geomid >= 0:
      flexid = cand_flex[i][1]
      geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
      tri_margin = flex_margin[flexid]
      margin = geom_margin_val + tri_margin

      condim, gap, solref, solimp, friction = _mix_flex_contact_params(
        geom_condim[geomid],
        geom_priority[geomid],
        geom_solmix[worldid % geom_solmix.shape[0], geomid],
        geom_solref[worldid % geom_solref.shape[0], geomid],
        geom_solimp[worldid % geom_solimp.shape[0], geomid],
        geom_friction[worldid % geom_friction.shape[0], geomid],
        geom_gap[worldid % geom_gap.shape[0], geomid],
        flex_condim[flexid],
        flex_priority[flexid],
        flex_solmix[flexid],
        flex_solref[flexid],
        flex_solimp[flexid],
        flex_friction[flexid],
        flex_gap[flexid],
      )
    else:
      flex1 = cand_flex[i][0]
      flex2 = cand_flex[i][1]
      is_self = flex1 == flex2
      if is_self:
        margin = float(0.0)
        gap1 = float(0.0)
        gap2 = float(0.0)
      else:
        margin = flex_margin[flex1] + flex_margin[flex2]
        gap1 = flex_gap[flex1]
        gap2 = flex_gap[flex2]
      mixed_condim, gap, solref, solimp, friction = _mix_flex_contact_params(
        flex_condim[flex1],
        flex_priority[flex1],
        flex_solmix[flex1],
        flex_solref[flex1],
        flex_solimp[flex1],
        flex_friction[flex1],
        gap1,
        flex_condim[flex2],
        flex_priority[flex2],
        flex_solmix[flex2],
        flex_solref[flex2],
        flex_solimp[flex2],
        flex_friction[flex2],
        gap2,
      )

      if cand_vert[i][0] >= 0 and flex_dim[flex1] == 3:
        condim = 1
      else:
        condim = mixed_condim

    if cand_dist[i] >= margin:
      return

    id_ = wp.atomic_add(nacon_out, 0, 1)  # kernel_analyzer: ignore[atomic]
    if id_ >= naconmax_in:
      if wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)):
        wp.printf(
          "flex contact overflow - please increase naconmax beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",
          naconmax_in,
        )
      wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NARROWPHASE))
      return

    contact_dist_out[id_] = cand_dist[i]
    contact_pos_out[id_] = cand_pos[i]
    contact_frame_out[id_] = make_frame(cand_nrm[i])
    if geomid >= 0 and geom_type[geomid] == int(GeomType.PLANE):
      contact_includemargin_out[id_] = margin - gap
    else:
      contact_includemargin_out[id_] = margin
    contact_friction_out[id_] = friction
    contact_solref_out[id_] = solref
    contact_solreffriction_out[id_] = solref
    contact_solimp_out[id_] = solimp
    contact_dim_out[id_] = condim
    contact_geom_out[id_] = cand_geom[i]
    contact_flex_out[id_] = cand_flex[i]
    contact_elem_out[id_] = cand_elem[i]
    contact_vert_out[id_] = cand_vert[i]
    for j in range(contact_efc_address_out.shape[1]):
      contact_efc_address_out[id_, j] = -1
    contact_worldid_out[id_] = cand_worldid[i]
    f0 = cand_flex[i][0]
    f1 = cand_flex[i][1]
    g0 = cand_geom[i][0]
    g1 = cand_geom[i][1]
    wants = (
      (f0 >= 0 and flex_passive[f0] != 0 and flex_interp[f0] == 0 and flex_dim[f0] >= 2)
      or (f1 >= 0 and flex_passive[f1] != 0 and flex_interp[f1] == 0 and flex_dim[f1] >= 2)
    ) and opt_integrator == int(IntegratorType.DISCRETE)
    ok = (f0 >= 0 or (g0 >= 0 and body_weldid[geom_bodyid[g0]] == 0)) and (
      f1 >= 0 or (g1 >= 0 and body_weldid[geom_bodyid[g1]] == 0)
    )
    is_passive = wants and ok
    contact_type_out[id_] = int(ContactType.PASSIVE) if is_passive else int(ContactType.CONSTRAINT)
    contact_geomcollisionid_out[id_] = 0

  return kernel


@wp.kernel
def _populate_active_sorted(
  # In:
  ncand: wp.array[int],
  sort_val: wp.array[int],
  cand_active: wp.array[int],
  # Out:
  cand_active_sorted_out: wp.array[int],
):
  si = wp.tid()
  ncand_limit = wp.min(ncand[0], cand_active_sorted_out.shape[0])
  if si < ncand_limit:
    cand_active_sorted_out[si] = cand_active[sort_val[si]]
  else:
    if si < cand_active_sorted_out.shape[0]:
      cand_active_sorted_out[si] = 0


@wp.kernel
def _find_group_starts(
  # In:
  ncand: wp.array[int],
  sort_key: wp.array[wp.int64],
  # Out:
  flex_group_temp_out: wp.array[int],
):
  si = wp.tid()
  ncand_limit = wp.min(ncand[0], flex_group_temp_out.shape[0])
  if si >= ncand_limit:
    return

  if si == 0:
    flex_group_temp_out[si] = 1
  else:
    my_key = sort_key[si]
    my_group = my_key >> wp.int64(32)
    prev_key = sort_key[si - 1]
    prev_group = prev_key >> wp.int64(32)
    if my_group != prev_group:
      flex_group_temp_out[si] = 1
    else:
      flex_group_temp_out[si] = 0


@cache_kernel
def _populate_group_starts(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    flex_group_temp_in: wp.array[int],
    flex_group_ids_in: wp.array[int],
    ncand: wp.array[int],
    filter_val_in: wp.array[int],
    cand_worldid: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    flex_group_start_indices_out: wp.array[int],
    flex_num_groups_out: wp.array[int],
  ):
    si = wp.tid()
    ncand_limit = wp.min(ncand[0], flex_group_temp_in.shape[0])
    if si >= ncand_limit:
      return

    if flex_group_temp_in[si] == 1:
      g = flex_group_ids_in[si] - 1
      if g >= flex_group_start_indices_out.shape[0]:
        if wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)):
          wp.printf(
            "flex candidate group overflow - please increase naconmax beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",
            flex_group_start_indices_out.shape[0],
          )
        worldid = cand_worldid[filter_val_in[si]]
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NARROWPHASE))
      else:
        flex_group_start_indices_out[g] = si

    if si == ncand_limit - 1:
      flex_num_groups_out[0] = flex_group_ids_in[si]

  return kernel


@wp.func
def _tie_break_fps(
  # In:
  curr_idx: int,
  sel_idx: int,
  cand_elem: wp.array[wp.vec2i],
):
  if sel_idx < 0:
    return True

  elem1_curr = cand_elem[curr_idx][0]
  elem2_curr = cand_elem[curr_idx][1]

  elem1_sel = cand_elem[sel_idx][0]
  elem2_sel = cand_elem[sel_idx][1]

  if elem1_curr != elem1_sel:
    return elem1_curr < elem1_sel
  if elem2_curr != elem2_sel:
    return elem2_curr < elem2_sel
  return curr_idx < sel_idx


@wp.kernel
def _parallel_fps_find_seed(
  # In:
  flex_group_start_indices_in: wp.array[int],
  flex_num_groups_in: wp.array[int],
  ncand: wp.array[int],
  cand_active_sorted: wp.array[int],
  sort_val: wp.array[int],
  cand_dist: wp.array[float],
  cand_elem: wp.array[wp.vec2i],
  cand_geom: wp.array[wp.vec2i],
  # Out:
  scratch_dist_out: wp.array2d[float],
  scratch_cidx_out: wp.array2d[int],
  scratch_count_out: wp.array2d[int],
):
  g, tid = wp.tid()
  if g >= flex_num_groups_in[0] or ncand[0] <= MJ_MAXCONPAIR:
    return

  ncand_limit = wp.min(ncand[0], cand_active_sorted.shape[0])
  g_start = flex_group_start_indices_in[g]
  if g_start < 0 or g_start >= ncand_limit:
    scratch_count_out[g, tid] = 0
    scratch_cidx_out[g, tid] = -1
    return

  first_cand_idx = sort_val[g_start]
  if cand_geom[first_cand_idx][0] >= 0:
    scratch_count_out[g, tid] = 0
    scratch_cidx_out[g, tid] = -1
    return

  g_end = ncand_limit
  if g < flex_num_groups_in[0] - 1:
    g_end = wp.min(ncand_limit, flex_group_start_indices_in[g + 1])

  local_active = int(0)
  min_d = float(1e10)
  sel_cidx = int(-1)

  for si in range(g_start + tid, g_end, wp.static(_FPS_BLOCK_SIZE)):
    if cand_active_sorted[si] == 1:
      local_active += 1
      c_idx = sort_val[si]
      d_val = cand_dist[c_idx]
      if d_val < min_d:
        min_d = d_val
        sel_cidx = c_idx
      elif d_val == min_d:
        if _tie_break_fps(c_idx, sel_cidx, cand_elem):
          sel_cidx = c_idx

  scratch_dist_out[g, tid] = min_d
  scratch_cidx_out[g, tid] = sel_cidx
  scratch_count_out[g, tid] = local_active


@wp.kernel
def _parallel_fps_init_condition(
  # Out:
  fps_groups_active_out: wp.array[int],
  fps_iter_out: wp.array[int],
  fps_condition_out: wp.array[int],
):
  fps_groups_active_out[0] = 0
  fps_iter_out[0] = 0
  fps_condition_out[0] = 0


@wp.kernel
def _parallel_fps_check_condition(
  # In:
  ncand: wp.array[int],
  fps_groups_active_in: wp.array[int],
  # Out:
  fps_condition_out: wp.array[int],
):
  if ncand[0] > wp.static(MJ_MAXCONPAIR) and fps_groups_active_in[0] > 0:
    fps_condition_out[0] = 1
  else:
    fps_condition_out[0] = 0


@wp.kernel
def _parallel_fps_step_condition(
  # In:
  fps_groups_active_in: wp.array[int],
  # Out:
  fps_iter_out: wp.array[int],
  fps_condition_out: wp.array[int],
):
  k = fps_iter_out[0] + 1
  fps_iter_out[0] = k
  if k >= wp.static(MJ_MAXCONPAIR - 1) or fps_groups_active_in[0] <= 0:
    fps_condition_out[0] = 0
  else:
    fps_condition_out[0] = 1


@wp.kernel
def _parallel_fps_resolve_seed(
  # In:
  flex_num_groups_in: wp.array[int],
  ncand: wp.array[int],
  cand_pos: wp.array[wp.vec3],
  cand_elem: wp.array[wp.vec2i],
  scratch_dist_in: wp.array2d[float],
  scratch_cidx_in: wp.array2d[int],
  scratch_count_in: wp.array2d[int],
  # Out:
  selected_cidx_out: wp.array[int],
  selected_pos_out: wp.array[wp.vec3],
  fps_groups_active_out: wp.array[int],
):
  g = wp.tid()
  selected_cidx_out[g] = -1
  if g >= flex_num_groups_in[0] or ncand[0] <= MJ_MAXCONPAIR:
    return

  total_active = int(0)
  for t in range(wp.static(_FPS_BLOCK_SIZE)):
    total_active += scratch_count_in[g, t]

  if total_active <= MJ_MAXCONPAIR:
    selected_cidx_out[g] = -1
    return

  wp.atomic_add(fps_groups_active_out, 0, 1)  # kernel_analyzer: ignore[atomic]

  min_d = float(1e10)
  sel_cidx = int(-1)
  for t in range(wp.static(_FPS_BLOCK_SIZE)):
    c_idx = scratch_cidx_in[g, t]
    if c_idx >= 0:
      d_val = scratch_dist_in[g, t]
      if d_val < min_d:
        min_d = d_val
        sel_cidx = c_idx
      elif d_val == min_d:
        if _tie_break_fps(c_idx, sel_cidx, cand_elem):
          sel_cidx = c_idx

  selected_cidx_out[g] = sel_cidx
  if sel_cidx >= 0:
    selected_pos_out[g] = cand_pos[sel_cidx]


@wp.kernel
def _parallel_fps_init_dist_and_find_max(
  # In:
  flex_group_start_indices_in: wp.array[int],
  flex_num_groups_in: wp.array[int],
  ncand: wp.array[int],
  cand_active_sorted: wp.array[int],
  sort_val: wp.array[int],
  cand_pos: wp.array[wp.vec3],
  cand_elem: wp.array[wp.vec2i],
  selected_cidx: wp.array[int],
  selected_pos: wp.array[wp.vec3],
  # Out:
  fps_min_dist_out: wp.array[float],
  cand_active_out: wp.array[int],
  scratch_dist_out: wp.array2d[float],
  scratch_cidx_out: wp.array2d[int],
):
  g, tid = wp.tid()
  if g >= flex_num_groups_in[0]:
    return

  seed_idx = selected_cidx[g]
  if seed_idx < 0:
    scratch_dist_out[g, tid] = float(-1e10)
    scratch_cidx_out[g, tid] = -1
    return

  g_start = flex_group_start_indices_in[g]
  ncand_limit = wp.min(ncand[0], cand_active_sorted.shape[0])
  g_end = ncand_limit
  if g < flex_num_groups_in[0] - 1:
    g_end = wp.min(ncand_limit, flex_group_start_indices_in[g + 1])

  seed_p = selected_pos[g]
  max_d = float(-1e10)
  sel_cidx = int(-1)

  for si in range(g_start + tid, g_end, wp.static(_FPS_BLOCK_SIZE)):
    if cand_active_sorted[si] == 1:
      c_idx = sort_val[si]
      if c_idx == seed_idx:
        cand_active_out[c_idx] = 1
        fps_min_dist_out[c_idx] = -1e10
      else:
        cand_active_out[c_idx] = 0
        d = wp.length(cand_pos[c_idx] - seed_p)
        fps_min_dist_out[c_idx] = d
        if d > max_d:
          max_d = d
          sel_cidx = c_idx
        elif d == max_d:
          if _tie_break_fps(c_idx, sel_cidx, cand_elem):
            sel_cidx = c_idx

  scratch_dist_out[g, tid] = max_d
  scratch_cidx_out[g, tid] = sel_cidx


@wp.kernel
def _parallel_fps_resolve_max(
  # In:
  flex_num_groups_in: wp.array[int],
  cand_pos: wp.array[wp.vec3],
  cand_elem: wp.array[wp.vec2i],
  scratch_dist_in: wp.array2d[float],
  scratch_cidx_in: wp.array2d[int],
  # Out:
  selected_cidx_out: wp.array[int],
  selected_pos_out: wp.array[wp.vec3],
  cand_active_out: wp.array[int],
  fps_min_dist_out: wp.array[float],
  fps_groups_active_out: wp.array[int],
):
  g = wp.tid()
  if g >= flex_num_groups_in[0]:
    return

  if selected_cidx_out[g] < 0:
    return

  max_d = float(-1e10)
  sel_cidx = int(-1)
  for t in range(wp.static(_FPS_BLOCK_SIZE)):
    c_idx = scratch_cidx_in[g, t]
    if c_idx >= 0:
      md = scratch_dist_in[g, t]
      if md > max_d:
        max_d = md
        sel_cidx = c_idx
      elif md == max_d:
        if _tie_break_fps(c_idx, sel_cidx, cand_elem):
          sel_cidx = c_idx

  if sel_cidx >= 0 and max_d > 0.0:
    selected_cidx_out[g] = sel_cidx
    selected_pos_out[g] = cand_pos[sel_cidx]
    cand_active_out[sel_cidx] = 1
    fps_min_dist_out[sel_cidx] = -1e10
  else:
    selected_cidx_out[g] = -1
    wp.atomic_sub(fps_groups_active_out, 0, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _parallel_fps_update_and_find_max(
  # In:
  flex_group_start_indices_in: wp.array[int],
  flex_num_groups_in: wp.array[int],
  ncand: wp.array[int],
  cand_active_sorted: wp.array[int],
  sort_val: wp.array[int],
  cand_pos: wp.array[wp.vec3],
  cand_elem: wp.array[wp.vec2i],
  selected_cidx: wp.array[int],
  selected_pos: wp.array[wp.vec3],
  # Out:
  fps_min_dist_out: wp.array[float],
  scratch_dist_out: wp.array2d[float],
  scratch_cidx_out: wp.array2d[int],
):
  g, tid = wp.tid()
  if g >= flex_num_groups_in[0]:
    return

  new_cidx = selected_cidx[g]
  if new_cidx < 0:
    scratch_dist_out[g, tid] = float(-1e10)
    scratch_cidx_out[g, tid] = -1
    return

  g_start = flex_group_start_indices_in[g]
  ncand_limit = wp.min(ncand[0], cand_active_sorted.shape[0])
  g_end = ncand_limit
  if g < flex_num_groups_in[0] - 1:
    g_end = wp.min(ncand_limit, flex_group_start_indices_in[g + 1])

  new_p = selected_pos[g]

  max_d = float(-1e10)
  sel_cidx = int(-1)

  for si in range(g_start + tid, g_end, wp.static(_FPS_BLOCK_SIZE)):
    if cand_active_sorted[si] == 1:
      c_idx = sort_val[si]
      md = fps_min_dist_out[c_idx]
      if md > 0.0:
        d_new = wp.length(cand_pos[c_idx] - new_p)
        md = wp.min(md, d_new)
        fps_min_dist_out[c_idx] = md

        if md > max_d:
          max_d = md
          sel_cidx = c_idx
        elif md == max_d:
          if _tie_break_fps(c_idx, sel_cidx, cand_elem):
            sel_cidx = c_idx

  scratch_dist_out[g, tid] = max_d
  scratch_cidx_out[g, tid] = sel_cidx


def flex_broadphase_aabb(m: Model, d: Data):
  """Precompute dynamic flex object bounding boxes."""
  wp.launch(
    _flex_broadphase_bounds,
    dim=(d.nworld, m.nflex),
    inputs=[
      m.flex_margin,
      m.flex_gap,
      m.flex_vertadr,
      m.flex_vertnum,
      m.flex_radius,
      m.flex_rod_is_box,
      m.flex_rod_box_size,
      d.flexvert_xpos,
    ],
    outputs=[
      d.flex_aabb_min,
      d.flex_aabb_max,
    ],
  )


@dataclasses.dataclass
class FlexWorkspace:
  """GPU workspace buffers for flex collision detection, EPA CCD, and contact filtering."""

  # Candidate buffers
  dist: wp.array
  pos: wp.array
  nrm: wp.array
  geom: wp.array
  flex: wp.array
  elem: wp.array
  vert: wp.array
  worldid: wp.array
  ncand: wp.array

  # Filtering & FPS buffers
  filter_key: wp.array
  filter_val: wp.array
  cand_active: wp.array

  # EPA Narrowphase buffers
  epa_vert: wp.array
  epa_vert_index: wp.array
  epa_face: wp.array
  epa_pr: wp.array
  epa_norm2: wp.array
  epa_horizon: wp.array

  # Optional FPS & CCD buffers
  cand_active_sorted: wp.array | None = None
  flex_group_temp: wp.array | None = None
  flex_group_ids: wp.array | None = None
  flex_group_start_indices: wp.array | None = None
  flex_fps_min_dist: wp.array | None = None
  flex_num_groups: wp.array | None = None
  fps_scratch_dist: wp.array | None = None
  fps_scratch_cidx: wp.array | None = None
  fps_scratch_count: wp.array | None = None
  fps_selected_cidx: wp.array | None = None
  fps_selected_pos: wp.array | None = None
  fps_groups_active: wp.array | None = None
  fps_condition: wp.array | None = None
  fps_iter: wp.array | None = None
  nccd: wp.array | None = None


def _allocate_flex_workspace(m: Model, d: Data) -> FlexWorkspace:
  epa_iterations = m.opt.ccd_iterations
  has_epa = m.nmesh > 0 or m.has_ellipsoid_geom or m.has_flex_selfcollide or m.nflex > 1 or m.has_1d_flex or m.has_3d_flex
  capacity = d.naccdmax if has_epa else 1

  needs_nccd = m.nmesh > 0 or m.has_ellipsoid_geom or m.has_1d_flex or m.has_3d_flex
  nccd = wp.zeros(1, dtype=int) if needs_nccd else None

  has_fps = m.has_flex_selfcollide or m.nflex > 1
  if has_fps:
    world_stride = m.ngeom * m.nflex + m.nflex * m.nflex
    nmax_groups = d.nworld * world_stride
    cand_active_sorted = wp.empty(d.naconmax, dtype=int)
    flex_group_temp = wp.empty(d.naconmax, dtype=int)
    flex_group_ids = wp.empty(d.naconmax, dtype=int)
    flex_group_start_indices = wp.full(nmax_groups, -1, dtype=int)
    flex_fps_min_dist = wp.empty(d.naconmax, dtype=float)
    flex_num_groups = wp.zeros(1, dtype=int)
    fps_scratch_dist = wp.empty((nmax_groups, _FPS_BLOCK_SIZE), dtype=float)
    fps_scratch_cidx = wp.empty((nmax_groups, _FPS_BLOCK_SIZE), dtype=int)
    fps_scratch_count = wp.empty((nmax_groups, _FPS_BLOCK_SIZE), dtype=int)
    fps_selected_cidx = wp.full(nmax_groups, -1, dtype=int)
    fps_selected_pos = wp.empty(nmax_groups, dtype=wp.vec3)
    fps_groups_active = wp.zeros(1, dtype=int)
    fps_condition = wp.zeros(1, dtype=int)
    fps_iter = wp.zeros(1, dtype=int)
  else:
    cand_active_sorted = None
    flex_group_temp = None
    flex_group_ids = None
    flex_group_start_indices = None
    flex_fps_min_dist = None
    flex_num_groups = None
    fps_scratch_dist = None
    fps_scratch_cidx = None
    fps_scratch_count = None
    fps_selected_cidx = None
    fps_selected_pos = None
    fps_groups_active = None
    fps_condition = None
    fps_iter = None

  return FlexWorkspace(
    dist=wp.empty(d.naconmax, dtype=float),
    pos=wp.empty(d.naconmax, dtype=wp.vec3),
    nrm=wp.empty(d.naconmax, dtype=wp.vec3),
    geom=wp.empty(d.naconmax, dtype=wp.vec2i),
    flex=wp.empty(d.naconmax, dtype=wp.vec2i),
    elem=wp.empty(d.naconmax, dtype=wp.vec2i),
    vert=wp.empty(d.naconmax, dtype=wp.vec2i),
    worldid=wp.empty(d.naconmax, dtype=int),
    ncand=wp.zeros(1, dtype=int),
    filter_key=wp.empty(d.naconmax * 2, dtype=wp.int64),
    filter_val=wp.empty(d.naconmax * 2, dtype=int),
    cand_active=wp.empty(d.naconmax, dtype=int),
    cand_active_sorted=cand_active_sorted,
    flex_group_temp=flex_group_temp,
    flex_group_ids=flex_group_ids,
    flex_group_start_indices=flex_group_start_indices,
    flex_fps_min_dist=flex_fps_min_dist,
    flex_num_groups=flex_num_groups,
    fps_scratch_dist=fps_scratch_dist,
    fps_scratch_cidx=fps_scratch_cidx,
    fps_scratch_count=fps_scratch_count,
    fps_selected_cidx=fps_selected_cidx,
    fps_selected_pos=fps_selected_pos,
    fps_groups_active=fps_groups_active,
    fps_condition=fps_condition,
    fps_iter=fps_iter,
    epa_vert=wp.empty(shape=(capacity, 10 + 2 * epa_iterations), dtype=wp.vec3),
    epa_vert_index=wp.empty(shape=(capacity, 10 + 2 * epa_iterations), dtype=int),
    epa_face=wp.empty(shape=(capacity, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=int),
    epa_pr=wp.empty(shape=(capacity, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=wp.vec3),
    epa_norm2=wp.empty(shape=(capacity, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=float),
    epa_horizon=wp.empty(shape=(capacity, MJ_MAX_EPAHORIZON), dtype=int),
    nccd=nccd,
  )


def _run_filter_flex_fps(
  m: Model,
  d: Data,
  ws: FlexWorkspace,
  nmax_groups: int,
):
  """Applies Far Point Sampling to limit contact points per group to MJ_MAXCONPAIR in parallel."""
  wp.launch(
    _parallel_fps_init_condition,
    dim=1,
    inputs=[],
    outputs=[
      ws.fps_groups_active,
      ws.fps_iter,
      ws.fps_condition,
    ],
  )
  wp.launch(
    _parallel_fps_find_seed,
    dim=(nmax_groups, _FPS_BLOCK_SIZE),
    inputs=[
      ws.flex_group_start_indices,
      ws.flex_num_groups,
      ws.ncand,
      ws.cand_active_sorted,
      ws.filter_val,
      ws.dist,
      ws.elem,
      ws.geom,
      ws.fps_scratch_dist,
      ws.fps_scratch_cidx,
      ws.fps_scratch_count,
    ],
  )
  wp.launch(
    _parallel_fps_resolve_seed,
    dim=nmax_groups,
    inputs=[
      ws.flex_num_groups,
      ws.ncand,
      ws.pos,
      ws.elem,
      ws.fps_scratch_dist,
      ws.fps_scratch_cidx,
      ws.fps_scratch_count,
    ],
    outputs=[
      ws.fps_selected_cidx,
      ws.fps_selected_pos,
      ws.fps_groups_active,
    ],
  )
  wp.launch(
    _parallel_fps_check_condition,
    dim=1,
    inputs=[ws.ncand, ws.fps_groups_active],
    outputs=[ws.fps_condition],
  )
  wp.launch(
    _parallel_fps_init_dist_and_find_max,
    dim=(nmax_groups, _FPS_BLOCK_SIZE),
    inputs=[
      ws.flex_group_start_indices,
      ws.flex_num_groups,
      ws.ncand,
      ws.cand_active_sorted,
      ws.filter_val,
      ws.pos,
      ws.elem,
      ws.fps_selected_cidx,
      ws.fps_selected_pos,
      ws.flex_fps_min_dist,
      ws.cand_active,
      ws.fps_scratch_dist,
      ws.fps_scratch_cidx,
    ],
  )

  def _fps_iteration():
    wp.launch(
      _parallel_fps_resolve_max,
      dim=nmax_groups,
      inputs=[
        ws.flex_num_groups,
        ws.pos,
        ws.elem,
        ws.fps_scratch_dist,
        ws.fps_scratch_cidx,
      ],
      outputs=[
        ws.fps_selected_cidx,
        ws.fps_selected_pos,
        ws.cand_active,
        ws.flex_fps_min_dist,
        ws.fps_groups_active,
      ],
    )
    wp.launch(
      _parallel_fps_update_and_find_max,
      dim=(nmax_groups, _FPS_BLOCK_SIZE),
      inputs=[
        ws.flex_group_start_indices,
        ws.flex_num_groups,
        ws.ncand,
        ws.cand_active_sorted,
        ws.filter_val,
        ws.pos,
        ws.elem,
        ws.fps_selected_cidx,
        ws.fps_selected_pos,
        ws.flex_fps_min_dist,
        ws.fps_scratch_dist,
        ws.fps_scratch_cidx,
      ],
    )
    wp.launch(
      _parallel_fps_step_condition,
      dim=1,
      inputs=[ws.fps_groups_active],
      outputs=[
        ws.fps_iter,
        ws.fps_condition,
      ],
    )

  if m.opt.graph_conditional:
    wp.capture_while(ws.fps_condition, while_body=_fps_iteration)
  else:
    for _ in range(1, MJ_MAXCONPAIR):
      _fps_iteration()


def _filter_and_write_contacts(
  m: Model,
  d: Data,
  ws: FlexWorkspace,
  enable_fps: bool = False,
):
  """Deduplicates candidates, optionally applies FPS filtering, and writes contacts to d.contact."""
  wp.launch(
    _compute_filter_key,
    dim=d.naconmax,
    inputs=[
      m.ngeom,
      m.nflex,
      ws.ncand,
      ws.geom,
      ws.flex,
      ws.pos,
      ws.worldid,
    ],
    outputs=[
      ws.filter_key,
      ws.filter_val,
    ],
  )

  wp.utils.radix_sort_pairs(ws.filter_key, ws.filter_val, d.naconmax)

  wp.launch(
    _filter_flex_candidates_sorted,
    dim=d.naconmax,
    inputs=[
      ws.ncand,
      _FLEX_CONTACT_DEDUP_TOLERANCE,
      ws.filter_key,
      ws.filter_val,
      ws.dist,
      ws.pos,
    ],
    outputs=[ws.cand_active],
  )

  if enable_fps:
    wp.launch(
      _populate_active_sorted,
      dim=d.naconmax,
      inputs=[ws.ncand, ws.filter_val, ws.cand_active],
      outputs=[ws.cand_active_sorted],
    )

    wp.launch(
      _find_group_starts,
      dim=d.naconmax,
      inputs=[
        ws.ncand,
        ws.filter_key,
      ],
      outputs=[ws.flex_group_temp],
    )

    wp.utils.array_scan(ws.flex_group_temp, ws.flex_group_ids, True)

    world_stride = m.ngeom * m.nflex + m.nflex * m.nflex
    nmax_groups = d.nworld * world_stride

    wp.launch(
      _populate_group_starts(int(m.opt.warn_overflow)),
      dim=d.naconmax,
      inputs=[
        ws.flex_group_temp,
        ws.flex_group_ids,
        ws.ncand,
        ws.filter_val,
        ws.worldid,
      ],
      outputs=[
        d.overflow,
        ws.flex_group_start_indices,
        ws.flex_num_groups,
      ],
    )

    _run_filter_flex_fps(m, d, ws, nmax_groups)

  wp.launch(
    _write_filtered_contacts(int(m.opt.warn_overflow)),
    dim=d.naconmax,
    inputs=[
      m.opt.integrator,
      m.body_weldid,
      m.geom_type,
      m.geom_condim,
      m.geom_bodyid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.flex_condim,
      m.flex_priority,
      m.flex_solmix,
      m.flex_solref,
      m.flex_solimp,
      m.flex_friction,
      m.flex_margin,
      m.flex_gap,
      m.flex_passive,
      m.flex_dim,
      m.flex_interp,
      d.naconmax,
      ws.ncand,
      ws.dist,
      ws.pos,
      ws.nrm,
      ws.geom,
      ws.flex,
      ws.elem,
      ws.vert,
      ws.worldid,
      ws.cand_active,
    ],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.flex,
      d.contact.elem,
      d.contact.vert,
      d.contact.efc_address,
      d.contact.worldid,
      d.contact.type,
      d.contact.geomcollisionid,
      d.nacon,
      d.overflow,
    ],
  )


def _detect_plane_flex_candidates(
  m: Model,
  d: Data,
  ws: FlexWorkspace,
):
  """Detect candidates between plane geoms and flex vertices."""
  if m.nflexvert == 0 or not m.has_plane_geom:
    return

  wp.launch(
    _flex_plane_narrowphase(int(m.opt.warn_overflow)),
    dim=(d.nworld, m.nflexvert),
    inputs=[
      m.ngeom,
      m.qpos0,
      m.jnt_qposadr,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_margin,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_margin,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_plasticadr,
      m.flex_vertbodyid,
      m.flex_elem,
      m.flex_radius,
      m.flex_vertflexid,
      m.flex_rod_is_box,
      m.flex_rod_root_clamped,
      m.flex_rod_tail_clamped,
      m.flex_rod_box_size,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_vert,
      m.flex_rodedge_jnt,
      m.flex_elem_rodedge,
      m.flex_vert_rodelem,
      d.qpos,
      d.plastic,
      d.xmat,
      d.geom_xpos,
      d.geom_xmat,
      d.flexvert_xpos,
      d.flex_aabb_min,
      d.flex_aabb_max,
      d.flex_rod_frame0_loc,
      d.flex_rod_frame0_ref,
      d.flex_rod_tail_loc,
      d.naconmax,
    ],
    outputs=[
      d.overflow,
      ws.dist,
      ws.pos,
      ws.nrm,
      ws.geom,
      ws.flex,
      ws.elem,
      ws.vert,
      ws.worldid,
      ws.ncand,
    ],
  )


def _detect_elem_geom_candidates(
  m: Model,
  d: Data,
  ws: FlexWorkspace,
):
  """Detect candidates between 1D/2D/3D flex elements and geoms."""
  epa_iterations = m.opt.ccd_iterations
  wp.launch(
    _flex_narrowphase_elem_detect(int(m.opt.warn_overflow)),
    dim=(d.nworld, m.nflexelem),
    inputs=[
      m.ngeom,
      m.opt.ccd_tolerance,
      m.qpos0,
      m.jnt_qposadr,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_size,
      m.geom_aabb,
      m.geom_margin,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_margin,
      m.flex_activelayers,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_plasticadr,
      m.flex_vertbodyid,
      m.flex_elem,
      m.flex_elemlayer,
      m.flex_radius,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_graphadr,
      m.mesh_vert,
      m.mesh_graph,
      m.mesh_pos,
      m.mesh_polynormal,
      m.mesh_polyvertadr,
      m.mesh_polyvert,
      m.mesh_polymapadr,
      m.mesh_polymapnum,
      m.mesh_polymap,
      m.flex_elemflexid,
      m.flex_rod_is_box,
      m.flex_rod_root_clamped,
      m.flex_rod_tail_clamped,
      m.flex_rod_box_size,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_vert,
      m.flex_rodedge_jnt,
      m.flex_elem_rodedge,
      d.qpos,
      d.plastic,
      d.xmat,
      d.geom_xpos,
      d.geom_xmat,
      d.flexvert_xpos,
      d.naccdmax,
      d.flex_aabb_min,
      d.flex_aabb_max,
      d.flex_rod_frame0_loc,
      d.flex_rod_frame0_ref,
      d.flex_rod_tail_loc,
      ws.epa_vert,
      ws.epa_vert_index,
      ws.epa_face,
      ws.epa_pr,
      ws.epa_norm2,
      ws.epa_horizon,
      ws.nccd,
      epa_iterations,
      d.naconmax,
    ],
    outputs=[
      d.overflow,
      ws.dist,
      ws.pos,
      ws.nrm,
      ws.geom,
      ws.flex,
      ws.elem,
      ws.vert,
      ws.worldid,
      ws.ncand,
    ],
  )


def _run_flex_sap_sort(
  m: Model,
  d: Data,
) -> tuple[wp.array, wp.array, wp.array, wp.array]:
  """Performs SAP projection, segmented sorting, and cumulative range scan for flex elements."""
  nelem = m.nflexelem
  nworldelem = d.nworld * nelem

  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  sap_lower = wp.empty((d.nworld, nelem, 2), dtype=float)
  sap_upper = wp.empty((d.nworld, nelem), dtype=float)
  sap_sort_index = wp.empty((d.nworld, nelem, 2), dtype=int)
  sap_range_arr = wp.empty((d.nworld, nelem), dtype=int)
  sap_cumsum = wp.empty((d.nworld, nelem), dtype=int)
  sap_seg_index = wp.empty(d.nworld + 1, dtype=int)
  elem_aabb_lower = wp.empty((d.nworld, nelem), dtype=wp.vec3)
  elem_aabb_upper = wp.empty((d.nworld, nelem), dtype=wp.vec3)

  wp.launch(
    _flex_sap_project,
    dim=(d.nworld, nelem),
    inputs=[
      m.flex_margin,
      m.flex_gap,
      m.flex_activelayers,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemdataadr,
      m.flex_elem,
      m.flex_elemlayer,
      m.flex_radius,
      m.flex_elemflexid,
      m.flex_rod_is_box,
      m.flex_rod_box_size,
      d.flexvert_xpos,
      d.nworld,
      nelem,
      direction,
    ],
    outputs=[
      sap_lower.reshape((-1, nelem)),
      sap_upper,
      sap_sort_index.reshape((-1, nelem)),
      elem_aabb_lower,
      elem_aabb_upper,
      sap_seg_index,
    ],
  )

  wp.utils.segmented_sort_pairs(
    sap_lower.reshape((-1, nelem)),
    sap_sort_index.reshape((-1, nelem)),
    nworldelem,
    sap_seg_index,
  )

  wp.launch(
    sap_range,
    dim=(d.nworld, nelem),
    inputs=[
      nelem,
      sap_lower.reshape((-1, nelem)),
      sap_upper,
      sap_sort_index.reshape((-1, nelem)),
    ],
    outputs=[
      sap_range_arr,
    ],
  )

  wp.utils.array_scan(
    sap_range_arr.reshape(-1),
    sap_cumsum.reshape(-1),
    True,
  )

  return (
    sap_sort_index.reshape((-1, nelem)),
    sap_cumsum.reshape(-1),
    elem_aabb_lower,
    elem_aabb_upper,
  )


def _run_flex_narrowphase(
  m: Model,
  d: Data,
  ctx,
  ws: FlexWorkspace,
):
  """Executes narrowphase collision detection for element pairs."""
  epa_iterations = m.opt.ccd_iterations
  workspace_verts = wp.empty(d.naconmax * 8, dtype=wp.vec3)
  wp.launch(
    _flex_narrowphase(int(m.opt.warn_overflow)),
    dim=d.naconmax,
    inputs=[
      m.opt.ccd_tolerance,
      m.qpos0,
      m.jnt_qposadr,
      m.flex_margin,
      m.flex_gap,
      m.flex_activelayers,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_plasticadr,
      m.flex_vertbodyid,
      m.flex_elem,
      m.flex_elemlayer,
      m.flex_radius,
      m.flex_elemflexid,
      m.flex_rod_is_box,
      m.flex_rod_root_clamped,
      m.flex_rod_tail_clamped,
      m.flex_rod_box_size,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_vert,
      m.flex_rodedge_jnt,
      m.flex_elem_rodedge,
      d.qpos,
      d.plastic,
      d.xmat,
      d.flexvert_xpos,
      d.naccdmax,
      d.ncollision,
      d.flex_rod_frame0_loc,
      d.flex_rod_frame0_ref,
      d.flex_rod_tail_loc,
      d.naconmax,
      m.opt.ccd_iterations,
      epa_iterations,
      ctx.collision_pair,
      ctx.collision_worldid,
    ],
    outputs=[
      d.overflow,
      workspace_verts,
      ws.epa_vert,
      ws.epa_vert_index,
      ws.epa_face,
      ws.epa_pr,
      ws.epa_norm2,
      ws.epa_horizon,
      ws.dist,
      ws.pos,
      ws.nrm,
      ws.geom,
      ws.flex,
      ws.elem,
      ws.vert,
      ws.worldid,
      ws.ncand,
    ],
  )


def _flex_geom_collision(
  m: Model,
  d: Data,
  ws: FlexWorkspace,
):
  """Detect and write collisions between rigid geoms and flex objects."""
  if m.nflex == 0:
    return

  ws.ncand.zero_()
  if ws.flex_num_groups is not None:
    ws.flex_num_groups.zero_()

  # Plane collisions (flex vertices vs infinite planes)
  _detect_plane_flex_candidates(m, d, ws)

  # 1D cable, 2D cloth, and 3D softbody element collisions (elements vs rigid geoms)
  _detect_elem_geom_candidates(m, d, ws)

  # Contact writing pass
  _filter_and_write_contacts(m, d, ws, enable_fps=False)


def _flex_sap_collision(
  m: Model,
  d: Data,
  ctx,
  ws: FlexWorkspace,
  is_self: bool,
  sap_data: tuple[wp.array, wp.array, wp.array, wp.array] | None = None,
  enable_sat: bool | None = None,
):
  """Detect and write flex self or flex-flex collision contacts (broadphase and narrowphase)."""
  if is_self and not m.has_flex_selfcollide:
    return
  if not is_self and m.nflex <= 1:
    return

  if enable_sat is None:
    enable_sat = ENABLE_SAT_PREFILTER

  ws.ncand.zero_()
  if ws.flex_num_groups is not None:
    ws.flex_num_groups.zero_()

  if sap_data is None:
    sap_data = _run_flex_sap_sort(m, d)
  sap_sort_index, sap_cumsum, elem_aabb_lower, elem_aabb_upper = sap_data
  nsweep = 5 * d.nworld * m.nflexelem
  d.ncollision.zero_()

  wp.launch(
    _flex_sap_sweep(is_self, int(m.opt.warn_overflow), enable_sat),
    dim=nsweep,
    inputs=[
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_margin,
      m.flex_gap,
      m.flex_selfcollide,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemdataadr,
      m.flex_vertbodyid,
      m.flex_elem,
      m.flex_radius,
      m.flex_elemflexid,
      d.flexvert_xpos,
      d.flex_aabb_min,
      d.flex_aabb_max,
      m.nflexelem,
      sap_sort_index,
      sap_cumsum,
      nsweep,
      elem_aabb_lower,
      elem_aabb_upper,
      d.naconmax,
    ],
    outputs=[
      d.ncollision,
      d.overflow,
      ctx.collision_pair,
      ctx.collision_worldid,
    ],
  )

  _run_flex_narrowphase(
    m,
    d,
    ctx,
    ws,
  )

  _filter_and_write_contacts(m, d, ws, enable_fps=True)


@event_scope
def flex_collision(m: Model, d: Data, ctx, enable_sat: bool | None = None):
  """Runs collision detection for all flex collisions."""
  if m.nflex == 0 or m.nflexelem == 0:
    return

  if enable_sat is None:
    enable_sat = ENABLE_SAT_PREFILTER

  # Update dynamic flex object bounding boxes
  flex_broadphase_aabb(m, d)

  ws = _allocate_flex_workspace(m, d)

  # Compute SAP projection and segmented sort once if needed by self or flex-flex collision
  sap_data = None
  needs_self_sap = m.has_flex_selfcollide
  needs_flex_flex_sap = m.nflex > 1
  if needs_self_sap or needs_flex_flex_sap:
    sap_data = _run_flex_sap_sort(m, d)

  # 1. Flex-Geom Collision (Primitive and CCD)
  _flex_geom_collision(m, d, ws)

  # 2. Flex Self-Collision (Broadphase and Narrowphase)
  _flex_sap_collision(m, d, ctx, ws, is_self=True, sap_data=sap_data, enable_sat=enable_sat)

  # 3. Flex-Flex Collision (Broadphase and Narrowphase)
  _flex_sap_collision(m, d, ctx, ws, is_self=False, sap_data=sap_data, enable_sat=enable_sat)
