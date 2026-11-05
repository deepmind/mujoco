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

from mujoco.mjx.third_party.mujoco_warp._src.collision_convex import convex_narrowphase
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import CollisionContext
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import create_collision_context
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import sap_binary_search
from mujoco.mjx.third_party.mujoco_warp._src.collision_core import sap_range
from mujoco.mjx.third_party.mujoco_warp._src.collision_flex import flex_collision
from mujoco.mjx.third_party.mujoco_warp._src.collision_primitive import primitive_narrowphase
from mujoco.mjx.third_party.mujoco_warp._src.collision_sdf import sdf_narrowphase
from mujoco.mjx.third_party.mujoco_warp._src.math import upper_tri_index
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import BroadphaseFilter
from mujoco.mjx.third_party.mujoco_warp._src.types import BroadphaseType
from mujoco.mjx.third_party.mujoco_warp._src.types import CollisionType
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DeterminismType
from mujoco.mjx.third_party.mujoco_warp._src.types import DisableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import EnableBit
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import SleepState
from mujoco.mjx.third_party.mujoco_warp._src.types import mat23
from mujoco.mjx.third_party.mujoco_warp._src.types import mat63
from mujoco.mjx.third_party.mujoco_warp._src.types import vec6
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Corresponding table to MuJoCo's mjCOLLISIONFUNC table in engine_collision_driver.c
MJ_COLLISION_TABLE = {
  (GeomType.PLANE, GeomType.SPHERE): CollisionType.PRIMITIVE,
  (GeomType.PLANE, GeomType.CAPSULE): CollisionType.PRIMITIVE,
  (GeomType.PLANE, GeomType.ELLIPSOID): CollisionType.PRIMITIVE,
  (GeomType.PLANE, GeomType.CYLINDER): CollisionType.PRIMITIVE,
  (GeomType.PLANE, GeomType.BOX): CollisionType.PRIMITIVE,
  (GeomType.PLANE, GeomType.MESH): CollisionType.PRIMITIVE,
  (GeomType.HFIELD, GeomType.SPHERE): CollisionType.CONVEX,
  (GeomType.HFIELD, GeomType.CAPSULE): CollisionType.CONVEX,
  (GeomType.HFIELD, GeomType.ELLIPSOID): CollisionType.CONVEX,
  (GeomType.HFIELD, GeomType.CYLINDER): CollisionType.CONVEX,
  (GeomType.HFIELD, GeomType.BOX): CollisionType.CONVEX,
  (GeomType.HFIELD, GeomType.MESH): CollisionType.CONVEX,
  (GeomType.SPHERE, GeomType.SPHERE): CollisionType.PRIMITIVE,
  (GeomType.SPHERE, GeomType.CAPSULE): CollisionType.PRIMITIVE,
  (GeomType.SPHERE, GeomType.ELLIPSOID): CollisionType.CONVEX,
  (GeomType.SPHERE, GeomType.CYLINDER): CollisionType.PRIMITIVE,
  (GeomType.SPHERE, GeomType.BOX): CollisionType.PRIMITIVE,
  (GeomType.SPHERE, GeomType.MESH): CollisionType.CONVEX,
  (GeomType.CAPSULE, GeomType.CAPSULE): CollisionType.PRIMITIVE,
  (GeomType.CAPSULE, GeomType.ELLIPSOID): CollisionType.CONVEX,
  (GeomType.CAPSULE, GeomType.CYLINDER): CollisionType.CONVEX,
  (GeomType.CAPSULE, GeomType.BOX): CollisionType.PRIMITIVE,
  (GeomType.CAPSULE, GeomType.MESH): CollisionType.CONVEX,
  (GeomType.ELLIPSOID, GeomType.ELLIPSOID): CollisionType.CONVEX,
  (GeomType.ELLIPSOID, GeomType.CYLINDER): CollisionType.CONVEX,
  (GeomType.ELLIPSOID, GeomType.BOX): CollisionType.CONVEX,
  (GeomType.ELLIPSOID, GeomType.MESH): CollisionType.CONVEX,
  (GeomType.CYLINDER, GeomType.CYLINDER): CollisionType.CONVEX,
  (GeomType.CYLINDER, GeomType.BOX): CollisionType.CONVEX,
  (GeomType.CYLINDER, GeomType.MESH): CollisionType.CONVEX,
  (GeomType.BOX, GeomType.BOX): CollisionType.CONVEX,  # overwritten by NATIVECCD disable flag
  (GeomType.BOX, GeomType.MESH): CollisionType.CONVEX,
  (GeomType.MESH, GeomType.MESH): CollisionType.CONVEX,
}


# TODO(team): Implement narrowphase flex collision support for:
#             - HFIELD
#             - SDF
MJ_FLEX_COLLISION_TABLE = {
  (GeomType.PLANE, GeomType.FLEX): CollisionType.PRIMITIVE,
  (GeomType.SPHERE, GeomType.FLEX): CollisionType.PRIMITIVE,
  (GeomType.CAPSULE, GeomType.FLEX): CollisionType.PRIMITIVE,
  (GeomType.BOX, GeomType.FLEX): CollisionType.PRIMITIVE,
  (GeomType.CYLINDER, GeomType.FLEX): CollisionType.PRIMITIVE,
  (GeomType.MESH, GeomType.FLEX): CollisionType.CONVEX,
  (GeomType.ELLIPSOID, GeomType.FLEX): CollisionType.CONVEX,
}


@wp.func
def _plane_filter(
  size1: float, size2: float, margin1: float, margin2: float, xpos1: wp.vec3, xpos2: wp.vec3, xmat1: wp.mat33, xmat2: wp.mat33
) -> bool:
  if size1 == 0.0:
    # geom1 is a plane
    dist = wp.dot(xpos2 - xpos1, wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))
    return dist <= size2 + margin1 + margin2
  elif size2 == 0.0:
    # geom2 is a plane
    dist = wp.dot(xpos1 - xpos2, wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))
    return dist <= size1 + margin1 + margin2

  return True


@wp.func
def _sphere_filter(size1: float, size2: float, margin1: float, margin2: float, xpos1: wp.vec3, xpos2: wp.vec3) -> bool:
  bound = size1 + size2 + margin1 + margin2
  dif = xpos2 - xpos1
  dist_sq = wp.dot(dif, dif)
  return dist_sq <= bound * bound


# TODO(team): improve performance by precomputing bounding box
@wp.func
def _aabb_filter(
  # In:
  center1: wp.vec3,
  center2: wp.vec3,
  size1: wp.vec3,
  size2: wp.vec3,
  margin1: float,
  margin2: float,
  xpos1: wp.vec3,
  xpos2: wp.vec3,
  xmat1: wp.mat33,
  xmat2: wp.mat33,
) -> bool:
  """Axis aligned boxes collision.

  references: see Ericson, Real-time Collision Detection section 4.2.
              filterBox: filter contact based on global AABBs.
  """
  center1 = xmat1 @ center1 + xpos1
  center2 = xmat2 @ center2 + xpos2

  margin = margin1 + margin2

  max_x1 = -MJ_MAXVAL
  max_y1 = -MJ_MAXVAL
  max_z1 = -MJ_MAXVAL
  min_x1 = MJ_MAXVAL
  min_y1 = MJ_MAXVAL
  min_z1 = MJ_MAXVAL

  max_x2 = -MJ_MAXVAL
  max_y2 = -MJ_MAXVAL
  max_z2 = -MJ_MAXVAL
  min_x2 = MJ_MAXVAL
  min_y2 = MJ_MAXVAL
  min_z2 = MJ_MAXVAL

  sign = wp.vec2(-1.0, 1.0)

  for i in range(2):
    for j in range(2):
      for k in range(2):
        corner1 = wp.vec3(sign[i] * size1[0], sign[j] * size1[1], sign[k] * size1[2])
        pos1 = xmat1 @ corner1

        corner2 = wp.vec3(sign[i] * size2[0], sign[j] * size2[1], sign[k] * size2[2])
        pos2 = xmat2 @ corner2

        if pos1[0] > max_x1:
          max_x1 = pos1[0]

        if pos1[1] > max_y1:
          max_y1 = pos1[1]

        if pos1[2] > max_z1:
          max_z1 = pos1[2]

        if pos1[0] < min_x1:
          min_x1 = pos1[0]

        if pos1[1] < min_y1:
          min_y1 = pos1[1]

        if pos1[2] < min_z1:
          min_z1 = pos1[2]

        if pos2[0] > max_x2:
          max_x2 = pos2[0]

        if pos2[1] > max_y2:
          max_y2 = pos2[1]

        if pos2[2] > max_z2:
          max_z2 = pos2[2]

        if pos2[0] < min_x2:
          min_x2 = pos2[0]

        if pos2[1] < min_y2:
          min_y2 = pos2[1]

        if pos2[2] < min_z2:
          min_z2 = pos2[2]

  if center1[0] + max_x1 + margin < center2[0] + min_x2:
    return False
  if center1[1] + max_y1 + margin < center2[1] + min_y2:
    return False
  if center1[2] + max_z1 + margin < center2[2] + min_z2:
    return False
  if center2[0] + max_x2 + margin < center1[0] + min_x1:
    return False
  if center2[1] + max_y2 + margin < center1[1] + min_y1:
    return False
  if center2[2] + max_z2 + margin < center1[2] + min_z1:
    return False

  return True


# TODO(team): improve performance by precomputing bounding box
@wp.func
def _obb_filter(
  # In:
  center1: wp.vec3,
  center2: wp.vec3,
  size1: wp.vec3,
  size2: wp.vec3,
  margin1: float,
  margin2: float,
  xpos1: wp.vec3,
  xpos2: wp.vec3,
  xmat1: wp.mat33,
  xmat2: wp.mat33,
) -> bool:
  """Oriented bounding boxes collision (see Gottschalk et al.), see mj_collideOBB."""
  margin = margin1 + margin2

  xcenter = mat23()
  normal = mat63()
  proj = wp.vec2()
  radius = wp.vec2()

  # compute centers in local coordinates
  xcenter[0] = xmat1 @ center1 + xpos1
  xcenter[1] = xmat2 @ center2 + xpos2

  # compute normals in global coordinates
  normal[0] = wp.vec3(xmat1[0, 0], xmat1[1, 0], xmat1[2, 0])
  normal[1] = wp.vec3(xmat1[0, 1], xmat1[1, 1], xmat1[2, 1])
  normal[2] = wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2])
  normal[3] = wp.vec3(xmat2[0, 0], xmat2[1, 0], xmat2[2, 0])
  normal[4] = wp.vec3(xmat2[0, 1], xmat2[1, 1], xmat2[2, 1])
  normal[5] = wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2])

  # check intersections
  for j in range(2):
    for k in range(3):
      for i in range(2):
        proj[i] = wp.dot(xcenter[i], normal[3 * j + k])
        if i == 0:
          size = size1
        else:
          size = size2

        # fmt: off
        radius[i] = (
            wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))
          + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))
          + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))
        )
        # fmt: on
      if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):
        return False

  return True


def _broadphase_filter(
  opt_broadphase_filter: int,
  ngeom_aabb: int,
  ngeom_rbound: int,
  ngeom_margin: int,
  ngeom_gap: int,
  npair_margin: int,
  npair_gap: int,
):
  @wp.func
  def func(
    # Model:
    geom_aabb: wp.array3d[wp.vec3],
    geom_rbound: wp.array2d[float],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    # In:
    geom1: int,
    geom2: int,
    pairid: int,
    worldid: int,
  ) -> bool:
    # 1: plane
    # 2: sphere
    # 4: aabb
    # 8: obb

    aabb_id = worldid % ngeom_aabb if wp.static(ngeom_aabb > 1) else 0
    center1, center2 = geom_aabb[aabb_id, geom1, 0], geom_aabb[aabb_id, geom2, 0]  # kernel_analyzer: ignore
    size1, size2 = geom_aabb[aabb_id, geom1, 1], geom_aabb[aabb_id, geom2, 1]  # kernel_analyzer: ignore

    rbound_id = worldid % ngeom_rbound if wp.static(ngeom_rbound > 1) else 0
    rbound1, rbound2 = geom_rbound[rbound_id, geom1], geom_rbound[rbound_id, geom2]  # kernel_analyzer: ignore
    effective_margin1 = float(0.0)
    effective_margin2 = float(0.0)
    if pairid >= 0:
      # explicit contact pairs use the pair margin and gap (the filters use the sum of the margins)
      pair_margin_id = worldid % npair_margin if wp.static(npair_margin > 1) else 0
      pair_gap_id = worldid % npair_gap if wp.static(npair_gap > 1) else 0
      effective_margin1 = pair_margin[pair_margin_id, pairid] + pair_gap[pair_gap_id, pairid]  # kernel_analyzer: ignore
    else:
      margin_id = worldid % ngeom_margin if wp.static(ngeom_margin > 1) else 0
      margin1, margin2 = geom_margin[margin_id, geom1], geom_margin[margin_id, geom2]  # kernel_analyzer: ignore
      gap_id = worldid % ngeom_gap if wp.static(ngeom_gap > 1) else 0
      gap1, gap2 = geom_gap[gap_id, geom1], geom_gap[gap_id, geom2]  # kernel_analyzer: ignore
      effective_margin1 = margin1 + gap1
      effective_margin2 = margin2 + gap2
    xpos1, xpos2 = geom_xpos_in[worldid, geom1], geom_xpos_in[worldid, geom2]
    xmat1, xmat2 = geom_xmat_in[worldid, geom1], geom_xmat_in[worldid, geom2]

    if rbound1 == 0.0 or rbound2 == 0.0:
      if wp.static(opt_broadphase_filter & BroadphaseFilter.PLANE):
        return _plane_filter(rbound1, rbound2, effective_margin1, effective_margin2, xpos1, xpos2, xmat1, xmat2)
    else:
      if wp.static(opt_broadphase_filter & BroadphaseFilter.SPHERE):
        if not _sphere_filter(rbound1, rbound2, effective_margin1, effective_margin2, xpos1, xpos2):
          return False
      if wp.static(opt_broadphase_filter & BroadphaseFilter.AABB):
        if not _aabb_filter(center1, center2, size1, size2, effective_margin1, effective_margin2, xpos1, xpos2, xmat1, xmat2):
          return False
      if wp.static(opt_broadphase_filter & BroadphaseFilter.OBB):
        if not _obb_filter(center1, center2, size1, size2, effective_margin1, effective_margin2, xpos1, xpos2, xmat1, xmat2):
          return False

    return True

  return func


@wp.func
def _mesh_missing(
  # Model:
  geom_dataid: wp.array2d[int],
  # In:
  geom: int,
  geomtype: int,
  dataid_id: int,
) -> bool:
  # with per-world meshes (batched geom_dataid), a mesh geom with dataid -1 is absent from a world
  return (geomtype == GeomType.MESH) and (geom_dataid[dataid_id, geom] < 0)  # kernel_analyzer: ignore


@wp.func
def _add_geom_pair(
  # Data in:
  naconmax_in: int,
  # In:
  geom1: int,
  geom2: int,
  type1: int,
  type2: int,
  worldid: int,
  pairid: wp.vec2i,
  # Data out:
  ncollision_out: wp.array[int],
  # Out:
  collision_pair_out: wp.array[wp.vec2i],
  collision_pairid_out: wp.array[wp.vec2i],
  collision_worldid_out: wp.array[int],
):
  cid = wp.atomic_add(ncollision_out, 0, 1)  # kernel_analyzer: ignore[atomic]

  if cid >= naconmax_in:
    return

  if type1 > type2:
    pair = wp.vec2i(geom2, geom1)
  else:
    pair = wp.vec2i(geom1, geom2)

  collision_pair_out[cid] = pair
  collision_pairid_out[cid] = pairid
  collision_worldid_out[cid] = worldid


@wp.func
def _geom_surface_velocity(
  # In:
  geom_xpos_val: wp.vec3,
  geom_xmat_val: wp.mat33,
  surfacevel_val: vec6,
  point_val: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  lin_local = wp.vec3(surfacevel_val[0], surfacevel_val[1], surfacevel_val[2])
  ang_local = wp.vec3(surfacevel_val[3], surfacevel_val[4], surfacevel_val[5])

  linear = geom_xmat_val * lin_local
  angular = geom_xmat_val * ang_local

  arm = point_val - geom_xpos_val
  linear = linear + wp.cross(angular, arm)

  return linear, angular


@cache_kernel
def _sap_pair_margin(npair_margin: int, npair_gap: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def sap_pair_margin(
    # Model:
    pair_geom1: wp.array[int],
    pair_geom2: wp.array[int],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    # Out:
    geom_pair_margin_out: wp.array2d[float],
  ):
    batchid, pairid = wp.tid()
    pair_margin_id = batchid % npair_margin if wp.static(npair_margin > 1) else 0
    pair_gap_id = batchid % npair_gap if wp.static(npair_gap > 1) else 0
    margin = pair_margin[pair_margin_id, pairid] + pair_gap[pair_gap_id, pairid]  # kernel_analyzer: ignore
    wp.atomic_max(geom_pair_margin_out, batchid, pair_geom1[pairid], margin)  # kernel_analyzer: ignore[atomic]
    wp.atomic_max(geom_pair_margin_out, batchid, pair_geom2[pairid], margin)  # kernel_analyzer: ignore[atomic]

  return sap_pair_margin


@cache_kernel
def _sap_project(opt_broadphase: int, ngeom_dataid: int, ngeom_pair_margin: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def sap_project(
    # Model:
    ngeom: int,
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_rbound: wp.array2d[float],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    nworld_in: int,
    # In:
    direction_in: wp.vec3,
    geom_pair_margin_in: wp.array2d[float],
    # Out:
    projection_lower_out: wp.array2d[float],
    projection_upper_out: wp.array2d[float],
    sort_index_out: wp.array2d[int],
    segmented_index_out: wp.array[int],
  ):
    worldid, geomid = wp.tid()

    sort_index_out[worldid, geomid] = geomid
    if wp.static(opt_broadphase == BroadphaseType.SAP_SEGMENTED):
      if geomid == 0:
        segmented_index_out[worldid] = worldid * ngeom
        if worldid == nworld_in - 1:
          segmented_index_out[nworld_in] = nworld_in * ngeom

    if wp.static(ngeom_dataid > 1):
      # an inverted interval at the end of the axis: no bounded geom sweeps over a missing mesh
      if _mesh_missing(geom_dataid, geomid, geom_type[geomid], worldid % ngeom_dataid):
        projection_lower_out[worldid, geomid] = MJ_MAXVAL
        projection_upper_out[worldid, geomid] = -MJ_MAXVAL
        return

    xpos = geom_xpos_in[worldid, geomid]
    rbound = geom_rbound[worldid % geom_rbound.shape[0], geomid]

    if rbound == 0.0:
      # geom is a plane
      rbound = MJ_MAXVAL

    margin = geom_margin[worldid % geom_margin.shape[0], geomid] + geom_gap[worldid % geom_gap.shape[0], geomid]
    # explicit contact pairs use the pair margin and gap, which can exceed the geom margins
    pair_margin_id = worldid % ngeom_pair_margin if wp.static(ngeom_pair_margin > 1) else 0
    radius = rbound + wp.max(margin, geom_pair_margin_in[pair_margin_id, geomid])
    center = wp.dot(direction_in, xpos)

    if not wp.isnan(center):
      projection_lower_out[worldid, geomid] = center - radius
      projection_upper_out[worldid, geomid] = center + radius
    else:
      projection_lower_out[worldid, geomid] = MJ_MAXVAL
      projection_upper_out[worldid, geomid] = MJ_MAXVAL

  return sap_project


@cache_kernel
def _sap_broadphase(
  opt_broadphase_filter: int,
  ngeom_aabb: int,
  ngeom_rbound: int,
  ngeom_margin: int,
  ngeom_gap: int,
  npair_margin: int,
  npair_gap: int,
  ngeom_dataid: int,
  enable_sleep: bool = False,
  incremental: bool = False,
):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    ngeom: int,
    geom_type: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_aabb: wp.array3d[wp.vec3],
    geom_rbound: wp.array2d[float],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    nxn_pairid: wp.array[wp.vec2i],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    body_awake_in: wp.array2d[int],
    nworld_in: int,
    naconmax_in: int,
    # In:
    sort_index_in: wp.array2d[int],
    cumulative_sum_in: wp.array[int],
    nsweep_in: int,
    body_awake_prev_in: wp.array2d[int],
    # Data out:
    ncollision_out: wp.array[int],
    # Out:
    collision_pair_out: wp.array[wp.vec2i],
    collision_pairid_out: wp.array[wp.vec2i],
    collision_worldid_out: wp.array[int],
  ):
    worldgeomid = wp.tid()

    nworldgeom = nworld_in * ngeom
    nworkpackages = cumulative_sum_in[nworldgeom - 1]

    while worldgeomid < nworkpackages:
      # binary search to find current and next geom pair indices
      i = sap_binary_search(cumulative_sum_in, worldgeomid, 0, nworldgeom)
      j = i + worldgeomid + 1

      if i > 0:
        j -= cumulative_sum_in[i - 1]

      worldid = i // ngeom
      i = i % ngeom
      j = j % ngeom

      # get geom indices and swap if necessary
      geom1 = sort_index_in[worldid, i]
      geom2 = sort_index_in[worldid, j]

      # find linear index of (geom1, geom2) in upper triangular nxn_pairid
      if geom2 < geom1:
        idx = upper_tri_index(ngeom, geom2, geom1)
      else:
        idx = upper_tri_index(ngeom, geom1, geom2)

      worldgeomid += nsweep_in
      pairid = nxn_pairid[idx]
      if pairid[0] < -1 and pairid[1] < 0:
        continue

      type1 = geom_type[geom1]
      type2 = geom_type[geom2]

      if wp.static(ngeom_dataid > 1):
        dataid_id = worldid % ngeom_dataid
        if _mesh_missing(geom_dataid, geom1, type1, dataid_id) or _mesh_missing(geom_dataid, geom2, type2, dataid_id):
          continue

      if wp.static(enable_sleep):
        b1 = geom_bodyid[geom1]
        b2 = geom_bodyid[geom2]
        s1 = body_awake_in[worldid, b1]
        s2 = body_awake_in[worldid, b2]
        if s1 == SleepState.ASLEEP and s2 == SleepState.ASLEEP:
          continue
        if (s1 == SleepState.ASLEEP and s2 == SleepState.STATIC) or (s2 == SleepState.ASLEEP and s1 == SleepState.STATIC):
          continue

        if wp.static(incremental):
          # only re-emit pairs that pass 1 skipped (see _nxn_broadphase for rationale)
          p1 = body_awake_prev_in[worldid, b1]
          p2 = body_awake_prev_in[worldid, b2]
          skipped_pass1 = (
            (p1 == SleepState.ASLEEP and p2 == SleepState.ASLEEP)
            or (p1 == SleepState.ASLEEP and p2 == SleepState.STATIC)
            or (p2 == SleepState.ASLEEP and p1 == SleepState.STATIC)
          )
          if not skipped_pass1:
            continue

      if (
        wp.static(
          _broadphase_filter(
            opt_broadphase_filter,
            ngeom_aabb,
            ngeom_rbound,
            ngeom_margin,
            ngeom_gap,
            npair_margin,
            npair_gap,
          )
        )(
          geom_aabb,
          geom_rbound,
          geom_margin,
          geom_gap,
          pair_margin,
          pair_gap,
          geom_xpos_in,
          geom_xmat_in,
          geom1,
          geom2,
          pairid[0],
          worldid,
        )
        or pairid[1] >= 0
      ):
        _add_geom_pair(
          naconmax_in,
          geom1,
          geom2,
          type1,
          type2,
          worldid,
          pairid,
          ncollision_out,
          collision_pair_out,
          collision_pairid_out,
          collision_worldid_out,
        )

  return kernel


@cache_kernel
def _segmented_sort(tile_size: int):
  @wp.kernel(module="unique", grid_stride=False)
  def segmented_sort(
    # In:
    projection_lower_in: wp.array2d[float],
    sort_index_in: wp.array2d[int],
    # Out:
    projection_lower_out: wp.array2d[float],
    sort_index_out: wp.array2d[int],
  ):
    worldid = wp.tid()

    # Load input into shared memory
    keys = wp.tile_load(projection_lower_in[worldid], shape=tile_size, storage="shared")
    values = wp.tile_load(sort_index_in[worldid], shape=tile_size, storage="shared")

    # Perform in-place sorting
    wp.tile_sort(keys, values)

    # Store sorted shared memory into output arrays
    wp.tile_store(projection_lower_out[worldid], keys)
    wp.tile_store(sort_index_out[worldid], values)

  return segmented_sort


@event_scope
def sap_broadphase(
  m: Model,
  d: Data,
  ctx: CollisionContext,
  awake_prev: Optional[wp.array] = None,
):
  """Runs broadphase collision detection using a sweep-and-prune (SAP) algorithm.

  This method is more efficient than the N-squared approach for large numbers of
  objects. It works by projecting the bounding spheres of all geoms onto a
  single axis and sorting them. It then sweeps along the axis, only checking
  for overlaps between geoms whose projections are close to each other.

  For each potentially colliding pair identified by the sweep, a more precise
  bounding sphere check is performed. If this check passes, the pair is added
  to the collision arrays in `d` for the narrowphase stage.

  Two sorting strategies are supported, controlled by `m.opt.broadphase`

  - `SAP_TILE`: Uses a tile-based sort.
  - `SAP_SEGMENTED`: Uses a segmented sort.

  Unlike `nxn_broadphase`, SAP cannot be wrapped in a CUDA graph conditional to skip the
  incremental sleeping pass: its sort/scan utilities allocate scratch internally, which is
  not allowed inside a conditional body. The incremental filter in the sweep kernel still
  restricts the emitted pairs to those involving newly-awakened bodies.
  """
  nworldgeom = d.nworld * m.ngeom
  incremental = awake_prev is not None
  awake_prev_in = awake_prev if awake_prev is not None else d.body_awake
  enable_sleep = bool(m.opt.enableflags & EnableBit.SLEEP)

  # TODO(team): direction

  # random fixed direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  projection_lower = wp.empty((d.nworld, m.ngeom, 2), dtype=float)
  projection_upper = wp.empty((d.nworld, m.ngeom), dtype=float)
  sort_index = wp.empty((d.nworld, m.ngeom, 2), dtype=int)
  range_ = wp.empty((d.nworld, m.ngeom), dtype=int)
  cumulative_sum = wp.empty((d.nworld, m.ngeom), dtype=int)
  segmented_index = wp.empty(d.nworld + 1 if m.opt.broadphase == BroadphaseType.SAP_SEGMENTED else 0, dtype=int)

  # largest margin + gap of the explicit contact pairs containing each geom, per world only if the
  # pair margins or gaps are batched
  npair_margin, npair_gap = m.pair_margin.shape[0], m.pair_gap.shape[0]
  ngeom_pair_margin = d.nworld if max(npair_margin, npair_gap) > 1 else 1
  geom_pair_margin = wp.zeros((ngeom_pair_margin, m.ngeom), dtype=float)
  wp.launch(
    _sap_pair_margin(npair_margin, npair_gap),
    dim=(ngeom_pair_margin, m.npair),
    inputs=[m.pair_geom1, m.pair_geom2, m.pair_margin, m.pair_gap],
    outputs=[geom_pair_margin],
  )

  wp.launch(
    kernel=_sap_project(m.opt.broadphase, m.geom_dataid.shape[0], ngeom_pair_margin),
    dim=(d.nworld, m.ngeom),
    inputs=[
      m.ngeom,
      m.geom_type,
      m.geom_dataid,
      m.geom_rbound,
      m.geom_margin,
      m.geom_gap,
      d.geom_xpos,
      d.nworld,
      direction,
      geom_pair_margin,
    ],
    outputs=[
      projection_lower.reshape((-1, m.ngeom)),
      projection_upper,
      sort_index.reshape((-1, m.ngeom)),
      segmented_index,
    ],
  )

  if m.opt.broadphase == BroadphaseType.SAP_TILE:
    wp.launch_tiled(
      kernel=_segmented_sort(m.ngeom),
      dim=d.nworld,
      inputs=[projection_lower.reshape((-1, m.ngeom)), sort_index.reshape((-1, m.ngeom))],
      outputs=[projection_lower.reshape((-1, m.ngeom)), sort_index.reshape((-1, m.ngeom))],
      block_dim=m.block_dim.segmented_sort,
    )
  else:
    wp.utils.segmented_sort_pairs(
      projection_lower.reshape((-1, m.ngeom)), sort_index.reshape((-1, m.ngeom)), nworldgeom, segmented_index
    )

  wp.launch(
    kernel=sap_range,
    dim=(d.nworld, m.ngeom),
    inputs=[m.ngeom, projection_lower.reshape((-1, m.ngeom)), projection_upper, sort_index.reshape((-1, m.ngeom))],
    outputs=[range_],
  )

  # scan is used for load balancing among the threads
  wp.utils.array_scan(range_.reshape(-1), cumulative_sum.reshape(-1), True)

  # estimate number of overlap checks
  # assumes each geom has 5 other geoms (batched over all worlds)
  nsweep = 5 * nworldgeom
  wp.launch(
    kernel=_sap_broadphase(
      m.opt.broadphase_filter,
      m.geom_aabb.shape[0],
      m.geom_rbound.shape[0],
      m.geom_margin.shape[0],
      m.geom_gap.shape[0],
      m.pair_margin.shape[0],
      m.pair_gap.shape[0],
      m.geom_dataid.shape[0],
      enable_sleep,
      incremental,
    ),
    dim=nsweep,
    inputs=[
      m.ngeom,
      m.geom_type,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_aabb,
      m.geom_rbound,
      m.geom_margin,
      m.geom_gap,
      m.pair_margin,
      m.pair_gap,
      m.nxn_pairid,
      d.geom_xpos,
      d.geom_xmat,
      d.body_awake,
      d.nworld,
      d.naconmax,
      sort_index.reshape((-1, m.ngeom)),
      cumulative_sum.reshape(-1),
      nsweep,
      awake_prev_in,
    ],
    outputs=[
      d.ncollision,
      ctx.collision_pair,
      ctx.collision_pairid,
      ctx.collision_worldid,
    ],
  )


@cache_kernel
def _nxn_broadphase(
  opt_broadphase_filter: int,
  ngeom_aabb: int,
  ngeom_rbound: int,
  ngeom_margin: int,
  ngeom_gap: int,
  npair_margin: int,
  npair_gap: int,
  ngeom_dataid: int,
  enable_sleep: bool = False,
  incremental: bool = False,
):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    geom_type: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_aabb: wp.array3d[wp.vec3],
    geom_rbound: wp.array2d[float],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    nxn_geom_pair: wp.array[wp.vec2i],
    nxn_pairid: wp.array[wp.vec2i],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    body_awake_in: wp.array2d[int],
    naconmax_in: int,
    # In:
    body_awake_prev_in: wp.array2d[int],
    # Data out:
    ncollision_out: wp.array[int],
    # Out:
    collision_pair_out: wp.array[wp.vec2i],
    collision_pairid_out: wp.array[wp.vec2i],
    collision_worldid_out: wp.array[int],
  ):
    worldid, elementid = wp.tid()

    geom = nxn_geom_pair[elementid]
    geom1 = geom[0]
    geom2 = geom[1]
    type1 = geom_type[geom1]
    type2 = geom_type[geom2]

    if wp.static(ngeom_dataid > 1):
      dataid_id = worldid % ngeom_dataid
      if _mesh_missing(geom_dataid, geom1, type1, dataid_id) or _mesh_missing(geom_dataid, geom2, type2, dataid_id):
        return

    if wp.static(enable_sleep):
      b1 = geom_bodyid[geom1]
      b2 = geom_bodyid[geom2]
      s1 = body_awake_in[worldid, b1]
      s2 = body_awake_in[worldid, b2]
      if s1 == SleepState.ASLEEP and s2 == SleepState.ASLEEP:
        return
      if (s1 == SleepState.ASLEEP and s2 == SleepState.STATIC) or (s2 == SleepState.ASLEEP and s1 == SleepState.STATIC):
        return

      if wp.static(incremental):
        # On the incremental pass we only want pairs that were *not* already emitted in pass 1.
        # Pass 1 skipped a pair iff both bodies were asleep, or one asleep and one static. Such a
        # pair becomes relevant now only because one of its bodies was newly awakened, so re-emit
        # it. Every other pair was handled in pass 1 and its contact geometry is unchanged.
        p1 = body_awake_prev_in[worldid, b1]
        p2 = body_awake_prev_in[worldid, b2]
        skipped_pass1 = (
          (p1 == SleepState.ASLEEP and p2 == SleepState.ASLEEP)
          or (p1 == SleepState.ASLEEP and p2 == SleepState.STATIC)
          or (p2 == SleepState.ASLEEP and p1 == SleepState.STATIC)
        )
        if not skipped_pass1:
          return

    pairid = nxn_pairid[elementid]
    if (
      wp.static(
        _broadphase_filter(
          opt_broadphase_filter,
          ngeom_aabb,
          ngeom_rbound,
          ngeom_margin,
          ngeom_gap,
          npair_margin,
          npair_gap,
        )
      )(
        geom_aabb,
        geom_rbound,
        geom_margin,
        geom_gap,
        pair_margin,
        pair_gap,
        geom_xpos_in,
        geom_xmat_in,
        geom1,
        geom2,
        pairid[0],
        worldid,
      )
      or pairid[1] >= 0
    ):
      _add_geom_pair(
        naconmax_in,
        geom1,
        geom2,
        type1,
        type2,
        worldid,
        pairid,
        ncollision_out,
        collision_pair_out,
        collision_pairid_out,
        collision_worldid_out,
      )

  return kernel


@wp.kernel
def _any_awake_changed(
  # Data in:
  body_awake_in: wp.array2d[int],
  # In:
  awake_prev_in: wp.array2d[int],
  # Out:
  changed_out: wp.array[int],
):
  worldid, bodyid = wp.tid()
  # benign race: every thread that fires writes the same value
  if awake_prev_in[worldid, bodyid] != body_awake_in[worldid, bodyid]:
    changed_out[0] = 1


@wp.kernel
def _contact_surfacevel(
  # Model:
  geom_surfacevel: wp.array2d[vec6],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_pos_in: wp.array[wp.vec3],
  contact_frame_in: wp.array[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  naconmax_in: int,
  nacon_in: wp.array[int],
  # Data out:
  contact_surfacevel_out: wp.array[wp.vec3],
):
  conid = wp.tid()
  if conid >= wp.min(nacon_in[0], naconmax_in):
    return

  geom = contact_geom_in[conid]
  g0 = geom[0]
  g1 = geom[1]
  if g0 < 0 and g1 < 0:
    contact_surfacevel_out[conid] = wp.vec3(0.0)
    return

  worldid = contact_worldid_in[conid]
  sv_id = worldid % geom_surfacevel.shape[0]
  sv0 = geom_surfacevel[sv_id, g0] if g0 >= 0 else vec6(0.0)
  sv1 = geom_surfacevel[sv_id, g1] if g1 >= 0 else vec6(0.0)

  has_sv0 = sv0[0] != 0.0 or sv0[1] != 0.0 or sv0[2] != 0.0 or sv0[3] != 0.0 or sv0[4] != 0.0 or sv0[5] != 0.0
  has_sv1 = sv1[0] != 0.0 or sv1[1] != 0.0 or sv1[2] != 0.0 or sv1[3] != 0.0 or sv1[4] != 0.0 or sv1[5] != 0.0

  if not (has_sv0 or has_sv1):
    contact_surfacevel_out[conid] = wp.vec3(0.0)
    return

  pos = contact_pos_in[conid]
  svel = wp.vec3(0.0)
  sang = wp.vec3(0.0)

  if has_sv0:
    vw0, ww0 = _geom_surface_velocity(
      geom_xpos_in[worldid, g0],
      geom_xmat_in[worldid, g0],
      sv0,
      pos,
    )
    svel -= vw0
    sang -= ww0

  if has_sv1:
    vw1, ww1 = _geom_surface_velocity(
      geom_xpos_in[worldid, g1],
      geom_xmat_in[worldid, g1],
      sv1,
      pos,
    )
    svel += vw1
    sang += ww1

  frame = contact_frame_in[conid]
  contact_surfacevel_out[conid] = wp.vec3(
    wp.dot(frame[1], svel),
    wp.dot(frame[2], svel),
    wp.dot(frame[0], sang),
  )


@event_scope
def nxn_broadphase(
  m: Model,
  d: Data,
  ctx: CollisionContext,
  awake_prev: Optional[wp.array] = None,
):
  """Runs broadphase collision detection using a brute-force N-squared approach.

  This function iterates through a pre-filtered list of all possible geometry pairs and
  performs a quick bounding sphere check to identify potential collisions.

  For each pair that passes the sphere check, it populates the collision arrays in `d`
  (`d.collision_pair`, `d.collision_pairid`, etc.), which are then consumed by the
  narrowphase.

  The initial list of pairs is filtered at model creation time to exclude pairs based on
  `contype`/`conaffinity`, parent-child relationships, and explicit `<exclude>` tags.

  Passing ``awake_prev`` runs the incremental sleeping pass: only pairs involving a newly-awakened
  body are emitted. When graph conditionals are available the launch is wrapped in one gated on
  whether any body woke since pass 1 (``awake_prev != body_awake``), so the broadphase is skipped
  wholesale on steps where nothing woke; otherwise it runs unconditionally and the per-pair filter
  restricts the emitted pairs.
  """
  enable_sleep = bool(m.opt.enableflags & EnableBit.SLEEP)
  incremental = awake_prev is not None
  awake_prev_in = awake_prev if awake_prev is not None else d.body_awake

  # On the incremental pass, skip the broadphase wholesale via a graph conditional when nothing woke
  # since pass 1. A wake is exactly a body whose state changed between awake_prev and body_awake
  # (nothing sleeps between the passes), so the condition is derived here rather than threaded in.
  cond = None
  if incremental and m.opt.graph_conditional:
    cond = wp.zeros(1, dtype=int)
    wp.launch(_any_awake_changed, dim=(d.nworld, m.nbody), inputs=[d.body_awake, awake_prev], outputs=[cond])

  def _launch():
    kernel = _nxn_broadphase(
      m.opt.broadphase_filter,
      m.geom_aabb.shape[0],
      m.geom_rbound.shape[0],
      m.geom_margin.shape[0],
      m.geom_gap.shape[0],
      m.pair_margin.shape[0],
      m.pair_gap.shape[0],
      m.geom_dataid.shape[0],
      enable_sleep,
      incremental,
    )
    block_size, min_grid_size = wp.get_suggested_block_size(kernel, d.geom_xpos.device)
    wp.launch(
      kernel,
      dim=(d.nworld, m.nxn_geom_pair_filtered.shape[0]),
      block_dim=m.block_dim.nxn_broadphase,
      max_blocks=8 * block_size * min_grid_size // m.block_dim.nxn_broadphase,
      inputs=[
        m.geom_type,
        m.geom_bodyid,
        m.geom_dataid,
        m.geom_aabb,
        m.geom_rbound,
        m.geom_margin,
        m.geom_gap,
        m.pair_margin,
        m.pair_gap,
        m.nxn_geom_pair_filtered,
        m.nxn_pairid_filtered,
        d.geom_xpos,
        d.geom_xmat,
        d.body_awake,
        d.naconmax,
        awake_prev_in,
      ],
      outputs=[
        d.ncollision,
        ctx.collision_pair,
        ctx.collision_pairid,
        ctx.collision_worldid,
      ],
    )

  if cond is not None:
    wp.capture_if(cond, on_true=_launch)
  else:
    _launch()


def _narrowphase(m: Model, d: Data, ctx: CollisionContext):
  collision_table = MJ_COLLISION_TABLE
  if m.opt.disableflags & DisableBit.NATIVECCD:
    collision_table = collision_table.copy()
    collision_table[(GeomType.BOX, GeomType.BOX)] = CollisionType.PRIMITIVE

  convex_pairs = [key for key, value in collision_table.items() if value == CollisionType.CONVEX]
  primitive_pairs = [key for key, value in collision_table.items() if value == CollisionType.PRIMITIVE]

  # TODO(team): we should reject far-away contacts in the narrowphase instead of constraint
  #             partitioning because we can move some pressure of the atomics
  convex_narrowphase(m, d, ctx, convex_pairs)
  primitive_narrowphase(m, d, ctx, primitive_pairs)

  if m.has_sdf_geom:
    sdf_narrowphase(m, d, ctx)


def _sort_contacts(m: Model, d: Data):
  """Deterministic contact sort."""
  # TODO(team): Implementation contact sorting.
  pass


@event_scope
def collision(
  m: Model,
  d: Data,
  awake_prev: Optional[wp.array] = None,
):
  """Runs the full collision detection pipeline.

  This function orchestrates the broadphase and narrowphase collision detection stages. It
  first identifies potential collision pairs using a broadphase algorithm (either N-squared
  or Sweep-and-Prune, based on `m.opt.broadphase`). Then, for each potential pair, it
  performs narrowphase collision detection to compute detailed contact information like
  distance, position, and frame.

  The results are used to populate the `d.contact` array, and the total number of contacts
  is stored in `d.nacon`.  If `d.nacon` is larger than `d.naconmax` then an overflow has
  occurred and the remaining contacts will be skipped.  If this happens, raise the `nconmax`
  parameter in `io.make_data` or `io.put_data`.

  This function will do nothing except zero out arrays if collision detection is disabled
  via `m.opt.disableflags` or if `d.nacon` is 0.

  Passing `awake_prev` (the awake state snapshotted before the post-collision wake) runs the
  incremental sleeping pass: contacts are appended to the existing buffer and only pairs involving
  a newly-awakened body are emitted.
  """
  if d.naconmax == 0 or m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.CONTACT):
    d.nacon.zero_()
    return

  # TODO(team): create context outside collision?
  ctx = create_collision_context(d.naconmax)

  incremental = awake_prev is not None

  # The incremental sleeping pass appends newly-awakened contacts to the pass-1 buffer, so nacon is
  # preserved and only ncollision (the broadphase pair counter) is reset. nxn_broadphase also skips
  # its launch wholesale via a graph conditional when nothing woke; pre-zeroing ncollision here
  # keeps the narrowphase below a no-op in that case. SAP cannot use a conditional (its sort/scan
  # allocate internally), so it relies on the per-pair incremental filter instead.
  d.ncollision.zero_()
  if not incremental:
    d.nacon.zero_()

  if m.opt.broadphase == BroadphaseType.NXN:
    nxn_broadphase(m, d, ctx, awake_prev)
  else:
    sap_broadphase(m, d, ctx, awake_prev)

  _narrowphase(m, d, ctx)

  # Flex collision is not sleeping-aware: pass 1 emits every flex contact regardless of awake state,
  # so the incremental pass has nothing to add (and re-running it would duplicate those contacts).
  # It therefore only runs on the full pass.
  if m.nflex > 0 and not incremental:
    flex_collision(m, d, ctx)

  if m.flg_surfacevel:
    wp.launch(
      _contact_surfacevel,
      dim=d.naconmax,
      inputs=[
        m.geom_surfacevel,
        d.geom_xpos,
        d.geom_xmat,
        d.contact.pos,
        d.contact.frame,
        d.contact.geom,
        d.contact.worldid,
        d.naconmax,
        d.nacon,
      ],
      outputs=[d.contact.surfacevel],
    )

  if m.callback.contactfilter:
    m.callback.contactfilter(m, d)

  if m.opt.deterministic & DeterminismType.CONTACTS:
    _sort_contacts(m, d)
