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

from __future__ import annotations

from typing import Tuple

import mujoco
import numpy as np
import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import math as mjmath
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import DeterminismType
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import RenderContext
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Render splats out to this density response, balancing extent and coverage.
SPLAT_MIN_RESPONSE = 0.01


@event_scope
def refit_bvh(m: Model, d: Data, rc: RenderContext):
  """Refit the dynamic BVH structures in the render context."""
  refit_scene_bvh(m, d, rc)
  if m.nflex:
    refit_flex_bvh(m, d, rc)


@wp.func
def _compute_box_bounds(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
  max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)

  for i in range(2):
    for j in range(2):
      for k in range(2):
        local_corner = wp.vec3(
          size[0] * (2.0 * float(i) - 1.0),
          size[1] * (2.0 * float(j) - 1.0),
          size[2] * (2.0 * float(k) - 1.0),
        )
        world_corner = pos + rot @ local_corner
        min_bound = wp.min(min_bound, world_corner)
        max_bound = wp.max(max_bound, world_corner)

  return min_bound, max_bound


@wp.func
def _compute_sphere_bounds(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  return pos - wp.vec3(radius, radius, radius), pos + wp.vec3(radius, radius, radius)


@wp.func
def _compute_capsule_bounds(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  half_length = size[1]
  z = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
  world_end1 = pos - z * half_length
  world_end2 = pos + z * half_length

  seg_min = wp.min(world_end1, world_end2)
  seg_max = wp.max(world_end1, world_end2)

  inflate = wp.vec3(radius, radius, radius)
  return seg_min - inflate, seg_max + inflate


@wp.func
def _compute_plane_bounds(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  # If plane size is non-positive, treat as infinite plane and use a large default extent
  size_scale = wp.max(size[0], size[1]) * 2.0
  if size[0] <= 0.0 or size[1] <= 0.0:
    size_scale = 1000.0
  min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
  max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)

  for i in range(2):
    for j in range(2):
      local_corner = wp.vec3(
        size_scale * (2.0 * float(i) - 1.0),
        size_scale * (2.0 * float(j) - 1.0),
        0.0,
      )
      world_corner = pos + rot @ local_corner
      min_bound = wp.min(min_bound, world_corner)
      max_bound = wp.max(max_bound, world_corner)

  min_bound = min_bound - wp.vec3(0.01, 0.01, 0.01)
  max_bound = max_bound + wp.vec3(0.01, 0.01, 0.01)

  return min_bound, max_bound


@wp.func
def _compute_ellipsoid_bounds(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  # Half-extent along each world axis equals the norm of the corresponding row of rot*diag(size)
  row0 = wp.vec3(rot[0, 0] * size[0], rot[0, 1] * size[1], rot[0, 2] * size[2])
  row1 = wp.vec3(rot[1, 0] * size[0], rot[1, 1] * size[1], rot[1, 2] * size[2])
  row2 = wp.vec3(rot[2, 0] * size[0], rot[2, 1] * size[1], rot[2, 2] * size[2])
  extent = wp.vec3(wp.length(row0), wp.length(row1), wp.length(row2))
  return pos - extent, pos + extent


@wp.func
def _compute_cylinder_bounds(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  half_height = size[1]

  axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
  axis_abs = wp.vec3(wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2]))

  basis_x = wp.vec3(rot[0, 0], rot[1, 0], rot[2, 0])
  basis_y = wp.vec3(rot[0, 1], rot[1, 1], rot[2, 1])

  radial_x = radius * wp.sqrt(basis_x[0] * basis_x[0] + basis_y[0] * basis_y[0])
  radial_y = radius * wp.sqrt(basis_x[1] * basis_x[1] + basis_y[1] * basis_y[1])
  radial_z = radius * wp.sqrt(basis_x[2] * basis_x[2] + basis_y[2] * basis_y[2])

  extent = wp.vec3(
    radial_x + half_height * axis_abs[0],
    radial_y + half_height * axis_abs[1],
    radial_z + half_height * axis_abs[2],
  )

  return pos - extent, pos + extent


@wp.kernel
def _compute_bvh_bounds(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  bvh_ngeom: int,
  enabled_geom_ids: wp.array[int],
  mesh_bounds_size: wp.array[wp.vec3],
  hfield_bounds_size: wp.array[wp.vec3],
  # Out:
  lower_out: wp.array[wp.vec3],
  upper_out: wp.array[wp.vec3],
  group_out: wp.array[int],
):
  worldid, geom_local_id = wp.tid()
  geom_id = enabled_geom_ids[geom_local_id]

  pos = geom_xpos_in[worldid, geom_id]
  rot = geom_xmat_in[worldid, geom_id]
  size = geom_size[worldid % geom_size.shape[0], geom_id]
  type = geom_type[geom_id]

  # TODO: Investigate branch elimination with static loop unrolling
  if type == GeomType.SPHERE:
    lower_bound, upper_bound = _compute_sphere_bounds(pos, rot, size)
  elif type == GeomType.CAPSULE:
    lower_bound, upper_bound = _compute_capsule_bounds(pos, rot, size)
  elif type == GeomType.PLANE:
    lower_bound, upper_bound = _compute_plane_bounds(pos, rot, size)
  elif type == GeomType.MESH or type == GeomType.SDF:
    did = geom_dataid[worldid % geom_dataid.shape[0], geom_id]
    if did >= 0:
      size = mesh_bounds_size[did]
      lower_bound, upper_bound = _compute_box_bounds(pos, rot, size)
    else:
      lower_bound = pos
      upper_bound = pos
  elif type == GeomType.ELLIPSOID:
    lower_bound, upper_bound = _compute_ellipsoid_bounds(pos, rot, size)
  elif type == GeomType.CYLINDER:
    lower_bound, upper_bound = _compute_cylinder_bounds(pos, rot, size)
  elif type == GeomType.BOX:
    lower_bound, upper_bound = _compute_box_bounds(pos, rot, size)
  elif type == GeomType.HFIELD:
    did = geom_dataid[worldid % geom_dataid.shape[0], geom_id]
    if did >= 0:
      size = hfield_bounds_size[did]
      hfield_center = pos + rot[:, 2] * size[2]
      lower_bound, upper_bound = _compute_box_bounds(hfield_center, rot, size)
    else:
      lower_bound = pos
      upper_bound = pos

  lower_out[worldid * bvh_ngeom + geom_local_id] = lower_bound
  upper_out[worldid * bvh_ngeom + geom_local_id] = upper_bound
  group_out[worldid * bvh_ngeom + geom_local_id] = worldid


@wp.kernel
def compute_bvh_group_roots(
  # In:
  bvh_id: wp.uint64,
  # Out:
  group_root_out: wp.array[int],
):
  tid = wp.tid()
  root = wp.bvh_get_group_root(bvh_id, tid)
  group_root_out[tid] = root


# Warp exposes mesh group-root lookup as a kernel builtin in this version.
@wp.kernel
def compute_mesh_group_roots(
  # In:
  mesh_id: wp.uint64,
  # Out:
  group_root_out: wp.array[int],
):
  tid = wp.tid()
  root = wp.mesh_get_group_root(mesh_id, tid)
  group_root_out[tid] = root


@wp.kernel
def _compute_flex_bvh_bounds(
  # Model:
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_edge: wp.array[wp.vec2i],
  flex_radius: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  flex_geom_flexid: wp.array[int],
  flex_geom_edgeid: wp.array[int],
  bvh_ngeom: int,
  total_bvh_size: int,
  # Out:
  lower_out: wp.array[wp.vec3],
  upper_out: wp.array[wp.vec3],
  group_out: wp.array[int],
):
  worldid, flexlocalid = wp.tid()

  flex_id = flex_geom_flexid[flexlocalid]
  edge_id = flex_geom_edgeid[flexlocalid]
  out_idx = worldid * total_bvh_size + bvh_ngeom + flexlocalid
  radius = flex_radius[flex_id]
  inflate = wp.vec3(radius, radius, radius)

  if edge_id >= 0:  # capsule (1D edge)
    edge = flex_edge[edge_id]
    vert_adr = flex_vertadr[flex_id]
    v0 = flexvert_xpos_in[worldid, vert_adr + edge[0]]
    v1 = flexvert_xpos_in[worldid, vert_adr + edge[1]]
    lower_out[out_idx] = wp.min(v0, v1) - inflate
    upper_out[out_idx] = wp.max(v0, v1) + inflate
  else:  # mesh (2D/3D)
    vert_adr = flex_vertadr[flex_id]
    nvert = flex_vertnum[flex_id]
    min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
    max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)
    for i in range(nvert):
      v = flexvert_xpos_in[worldid, vert_adr + i]
      min_bound = wp.min(min_bound, v)
      max_bound = wp.max(max_bound, v)
    lower_out[out_idx] = min_bound - inflate
    upper_out[out_idx] = max_bound + inflate

  group_out[out_idx] = worldid


def build_scene_bvh(mjm: mujoco.MjModel, mjd: mujoco.MjData, rc: RenderContext, nworld: int):
  """Build a global BVH for all geometries in all worlds."""
  total_bvh_size = rc.bvh_ngeom + rc.bvh_nflexgeom

  geom_type = wp.array(mjm.geom_type, dtype=int)
  geom_dataid = wp.array(np.tile(mjm.geom_dataid, (nworld, 1)), dtype=int)
  geom_size = wp.array(np.tile(mjm.geom_size[np.newaxis, :, :], (nworld, 1, 1)), dtype=wp.vec3)
  geom_xpos = wp.array(np.tile(mjd.geom_xpos[np.newaxis, :, :], (nworld, 1, 1)), dtype=wp.vec3)
  geom_xmat = wp.array(np.tile(mjd.geom_xmat.reshape(mjm.ngeom, 3, 3)[np.newaxis, :, :, :], (nworld, 1, 1, 1)), dtype=wp.mat33)

  flex_vertadr = wp.array(mjm.flex_vertadr, dtype=int)
  flex_vertnum = wp.array(mjm.flex_vertnum, dtype=int)
  flex_edge = wp.array(mjm.flex_edge, dtype=wp.vec2i)
  flex_radius = wp.array(mjm.flex_radius, dtype=float)

  wp.launch(
    kernel=_compute_bvh_bounds,
    dim=(nworld, rc.bvh_ngeom),
    inputs=[
      geom_type,
      geom_dataid,
      geom_size,
      geom_xpos,
      geom_xmat,
      total_bvh_size,
      rc.enabled_geom_ids,
      rc.mesh_bounds_size,
      rc.hfield_bounds_size,
      rc.lower,
      rc.upper,
      rc.group,
    ],
  )

  flexvert_xpos = wp.array(np.tile(mjd.flexvert_xpos[np.newaxis, :, :], (nworld, 1, 1)), dtype=wp.vec3)
  wp.launch(
    kernel=_compute_flex_bvh_bounds,
    dim=(nworld, rc.bvh_nflexgeom),
    inputs=[
      flex_vertadr,
      flex_vertnum,
      flex_edge,
      flex_radius,
      flexvert_xpos,
      rc.flex_geom_flexid,
      rc.flex_geom_edgeid,
      rc.bvh_ngeom,
      total_bvh_size,
      rc.lower,
      rc.upper,
      rc.group,
    ],
  )

  bvh = wp.Bvh(rc.lower, rc.upper, groups=rc.group, constructor="sah")

  # BVH handle must be stored to avoid garbage collection
  rc.bvh = bvh
  rc.bvh_id = bvh.id

  wp.launch(
    kernel=compute_bvh_group_roots,
    dim=nworld,
    inputs=[bvh.id],
    outputs=[rc.group_root],
  )


def refit_scene_bvh(m: Model, d: Data, rc: RenderContext):
  total_bvh_size = rc.bvh_ngeom + rc.bvh_nflexgeom

  wp.launch(
    kernel=_compute_bvh_bounds,
    dim=(d.nworld, rc.bvh_ngeom),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      total_bvh_size,
      rc.enabled_geom_ids,
      rc.mesh_bounds_size,
      rc.hfield_bounds_size,
      rc.lower,
      rc.upper,
      rc.group,
    ],
  )

  if rc.bvh_nflexgeom > 0:
    wp.launch(
      kernel=_compute_flex_bvh_bounds,
      dim=(d.nworld, rc.bvh_nflexgeom),
      inputs=[
        m.flex_vertadr,
        m.flex_vertnum,
        m.flex_edge,
        m.flex_radius,
        d.flexvert_xpos,
        rc.flex_geom_flexid,
        rc.flex_geom_edgeid,
        rc.bvh_ngeom,
        total_bvh_size,
        rc.lower,
        rc.upper,
        rc.group,
      ],
    )

  rc.bvh.refit()


def build_mesh_bvh(
  mjm: mujoco.MjModel,
  meshid: int,
  constructor: str = "sah",
  leaf_size: int = 2,
) -> tuple[wp.Mesh, wp.vec3]:
  """Create a Warp mesh BVH from mesh data."""
  v_start = mjm.mesh_vertadr[meshid]
  v_end = v_start + mjm.mesh_vertnum[meshid]
  points = mjm.mesh_vert[v_start:v_end]

  f_start = mjm.mesh_faceadr[meshid]
  f_end = mjm.mesh_face.shape[0] if (meshid + 1) >= mjm.mesh_faceadr.shape[0] else mjm.mesh_faceadr[meshid + 1]
  indices = mjm.mesh_face[f_start:f_end]
  indices = indices.flatten()
  pmin = np.min(points, axis=0)
  pmax = np.max(points, axis=0)
  half = np.maximum(np.abs(pmin), np.abs(pmax))

  points = wp.array(points, dtype=wp.vec3)
  indices = wp.array(indices, dtype=wp.int32)
  mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor, bvh_leaf_size=leaf_size)

  return mesh, half


def _optimize_hfield_mesh(
  data: np.ndarray,
  nr: int,
  nc: int,
  sx: float,
  sy: float,
  sz_scale: float,
  width: float,
  height: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Greedy meshing for heightfield optimization.

  Merges coplanar adjacent cells into larger rectangles to
  reduce triangle and vertex count.
  """
  points_map = {}
  points_list = []
  indices_list = []

  def get_point_index(r, c):
    if (r, c) in points_map:
      return points_map[(r, c)]

    # Compute vertex position
    x = sx * (float(c) / width - 1.0)
    y = sy * (float(r) / height - 1.0)
    z = float(data[r, c]) * sz_scale

    idx = len(points_list)
    points_list.append([x, y, z])
    points_map[(r, c)] = idx
    return idx

  visited = np.zeros((nr - 1, nc - 1), dtype=bool)

  for r in range(nr - 1):
    for c in range(nc - 1):
      if visited[r, c]:
        continue

      # Check if current cell is planar
      z00 = data[r, c]
      z01 = data[r, c + 1]
      z10 = data[r + 1, c]
      z11 = data[r + 1, c + 1]

      # Approx check for planarity: z00 + z11 == z01 + z10
      is_planar = abs((z00 + z11) - (z01 + z10)) < 1e-5

      if not is_planar:
        # Must emit single cell (2 triangles)
        idx00 = get_point_index(r, c)
        idx01 = get_point_index(r, c + 1)
        idx10 = get_point_index(r + 1, c)
        idx11 = get_point_index(r + 1, c + 1)

        # Tri 1: TL, TR, BR
        indices_list.extend([idx00, idx01, idx11])
        # Tri 2: TL, BR, BL
        indices_list.extend([idx00, idx11, idx10])
        visited[r, c] = True
        continue

      # If planar, try to expand
      slope_x = z01 - z00
      slope_y = z10 - z00
      w = 1
      h = 1

      def fits_plane(rr, cc):
        if rr >= nr - 1 or cc >= nc - 1:
          return False
        # Check planarity of the cell itself
        cz00 = data[rr, cc]
        cz01 = data[rr, cc + 1]
        cz10 = data[rr + 1, cc]
        cz11 = data[rr + 1, cc + 1]
        if abs((cz00 + cz11) - (cz01 + cz10)) >= 1e-5:
          return False

        # Check if it lies on the SAME plane as start cell
        # Expected z at (rr, cc)
        z_pred = z00 + (rr - r) * slope_y + (cc - c) * slope_x
        if abs(cz00 - z_pred) >= 1e-5:
          return False

        # Since cell is planar and one corner matches, slopes must match if connected
        cslope_x = cz01 - cz00
        cslope_y = cz10 - cz00
        if abs(cslope_x - slope_x) >= 1e-5 or abs(cslope_y - slope_y) >= 1e-5:
          return False

        return True

      # Expand width
      while c + w < nc - 1 and not visited[r, c + w] and fits_plane(r, c + w):
        w += 1

      # Expand height
      while r + h < nr - 1:
        # Check entire row
        row_ok = True
        for k in range(w):
          if visited[r + h, c + k] or not fits_plane(r + h, c + k):
            row_ok = False
            break
        if row_ok:
          h += 1
        else:
          break

      # Mark visited
      visited[r : r + h, c : c + w] = True

      # Emit large quad
      idx_tl = get_point_index(r, c)
      idx_tr = get_point_index(r, c + w)
      idx_bl = get_point_index(r + h, c)
      idx_br = get_point_index(r + h, c + w)

      # Tri 1: TL, TR, BR
      indices_list.extend([idx_tl, idx_tr, idx_br])
      # Tri 2: TL, BR, BL
      indices_list.extend([idx_tl, idx_br, idx_bl])

  return np.array(points_list, dtype=np.float32), np.array(indices_list, dtype=np.int32)


def build_hfield_bvh(
  mjm: mujoco.MjModel,
  hfieldid: int,
  constructor: str = "sah",
  leaf_size: int = 2,
) -> tuple[wp.Mesh, wp.vec3]:
  """Create a Warp mesh BVH from heightfield data."""
  nr = mjm.hfield_nrow[hfieldid]
  nc = mjm.hfield_ncol[hfieldid]
  sz = np.asarray(mjm.hfield_size[hfieldid], dtype=np.float32)

  adr = mjm.hfield_adr[hfieldid]
  data = mjm.hfield_data[adr : adr + nr * nc].reshape((nr, nc))

  width = 0.5 * max(nc - 1, 1)
  height = 0.5 * max(nr - 1, 1)

  points, indices = _optimize_hfield_mesh(
    data,
    nr,
    nc,
    sz[0],
    sz[1],
    sz[2],
    width,
    height,
  )
  pmin = np.min(points, axis=0)
  pmax = np.max(points, axis=0)
  half = 0.5 * (pmax - pmin)

  points = wp.array(points, dtype=wp.vec3)
  indices = wp.array(indices, dtype=wp.int32)

  mesh = wp.Mesh(
    points=points,
    indices=indices,
    bvh_constructor=constructor,
    bvh_leaf_size=leaf_size,
  )

  return mesh, half


@cache_kernel
def accumulate_flex_vertex_normals(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    nflex: int,
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_elem: wp.array[int],
    # Data in:
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # Out:
    flexvert_norm_out: wp.array2d[wp.vec3],
  ):
    """Accumulate per-vertex normals by summing adjacent face normals."""
    worldid, elemid = wp.tid()

    for i in range(nflex):
      locid = elemid - flex_elemadr[i]
      if locid >= 0 and locid < flex_elemnum[i]:
        f = i
        break

    if flex_dim[f] == 1 or flex_dim[f] == 3:
      return

    local_elemid = elemid - flex_elemadr[f]
    elem_adr = flex_elemdataadr[f]
    vert_adr = flex_vertadr[f]
    elem_base = elem_adr + local_elemid * 3
    i0 = vert_adr + flex_elem[elem_base + 0]
    i1 = vert_adr + flex_elem[elem_base + 1]
    i2 = vert_adr + flex_elem[elem_base + 2]

    v0 = flexvert_xpos_in[worldid, i0]
    v1 = flexvert_xpos_in[worldid, i1]
    v2 = flexvert_xpos_in[worldid, i2]

    face_nrm = wp.cross(v1 - v0, v2 - v0)
    face_nrm = wp.normalize(face_nrm)
    wp.atomic_add(flexvert_norm_out, worldid, i0, face_nrm)
    wp.atomic_add(flexvert_norm_out, worldid, i1, face_nrm)
    wp.atomic_add(flexvert_norm_out, worldid, i2, face_nrm)

  return kernel


@wp.kernel
def normalize_vertex_normals(
  # Out:
  flexvert_norm_out: wp.array2d[wp.vec3],
):
  """Normalize accumulated vertex normals."""
  worldid, vertid = wp.tid()
  flexvert_norm_out[worldid, vertid] = wp.normalize(flexvert_norm_out[worldid, vertid])


@wp.kernel
def _build_flex_2d_elements(
  # Model:
  flex_elem: wp.array[int],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  flexvert_norm_in: wp.array2d[wp.vec3],
  elem_adr: int,
  vert_adr: int,
  face_offset: int,
  radius: float,
  nfaces: int,
  # Out:
  face_point_out: wp.array[wp.vec3],
  face_index_out: wp.array[int],
  group_out: wp.array[int],
):
  """Create faces from 2D flex elements (triangles).

  Two faces (top/bottom) per element, separated by the radius of the flex element.
  """
  worldid, elemid = wp.tid()

  base = elem_adr + elemid * 3
  i0 = vert_adr + flex_elem[base + 0]
  i1 = vert_adr + flex_elem[base + 1]
  i2 = vert_adr + flex_elem[base + 2]

  v0 = flexvert_xpos_in[worldid, i0]
  v1 = flexvert_xpos_in[worldid, i1]
  v2 = flexvert_xpos_in[worldid, i2]

  n0 = flexvert_norm_in[worldid, i0]
  n1 = flexvert_norm_in[worldid, i1]
  n2 = flexvert_norm_in[worldid, i2]

  p0_pos = v0 + radius * n0
  p1_pos = v1 + radius * n1
  p2_pos = v2 + radius * n2

  p0_neg = v0 - radius * n0
  p1_neg = v1 - radius * n1
  p2_neg = v2 - radius * n2

  world_face_offset = worldid * nfaces

  # First face (top): i0, i1, i2
  face_id0 = world_face_offset + face_offset + 2 * elemid
  base0 = face_id0 * 3
  face_point_out[base0 + 0] = p0_pos
  face_point_out[base0 + 1] = p1_pos
  face_point_out[base0 + 2] = p2_pos

  face_index_out[base0 + 0] = base0 + 0
  face_index_out[base0 + 1] = base0 + 1
  face_index_out[base0 + 2] = base0 + 2

  group_out[face_id0] = worldid

  # Second face (bottom): i0, i2, i1 (opposite winding)
  face_id1 = world_face_offset + face_offset + 2 * elemid + 1
  base1 = face_id1 * 3
  face_point_out[base1 + 0] = p0_neg
  face_point_out[base1 + 1] = p1_neg
  face_point_out[base1 + 2] = p2_neg

  face_index_out[base1 + 0] = base1 + 0
  face_index_out[base1 + 1] = base1 + 2
  face_index_out[base1 + 2] = base1 + 1

  group_out[face_id1] = worldid


@wp.kernel
def _build_flex_2d_sides(
  # Model:
  flex_shell: wp.array[int],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  flexvert_norm_in: wp.array2d[wp.vec3],
  shell_adr: int,
  vert_adr: int,
  face_offset: int,
  radius: float,
  nface: int,
  # Out:
  face_point_out: wp.array[wp.vec3],
  face_index_out: wp.array[int],
  group_out: wp.array[int],
):
  """Create side faces from 2D flex shell fragments.

  For each shell fragment (edge i0 -> i1), we emit two triangles:
    - one using +radius
    - one using -radius (i0/i1 swapped)
  """
  worldid, shellid = wp.tid()

  base = shell_adr + 2 * shellid
  i0 = vert_adr + flex_shell[base + 0]
  i1 = vert_adr + flex_shell[base + 1]

  v0 = flexvert_xpos_in[worldid, i0]
  v1 = flexvert_xpos_in[worldid, i1]

  n0 = flexvert_norm_in[worldid, i0]
  n1 = flexvert_norm_in[worldid, i1]

  neg_radius = -radius

  # First side i0, i1 with +radius
  face_id0 = worldid * nface + face_offset + 2 * shellid
  base0 = face_id0 * 3
  face_point_out[base0 + 0] = v0 + n0 * radius
  face_point_out[base0 + 1] = v1 + n1 * neg_radius
  face_point_out[base0 + 2] = v1 + n1 * radius
  face_index_out[base0 + 0] = base0 + 0
  face_index_out[base0 + 1] = base0 + 1
  face_index_out[base0 + 2] = base0 + 2

  # Second side i1, i0 with -radius
  face_id1 = worldid * nface + face_offset + 2 * shellid + 1
  base1 = face_id1 * 3
  face_point_out[base1 + 0] = v1 + n1 * neg_radius
  face_point_out[base1 + 1] = v0 + n0 * neg_radius
  face_point_out[base1 + 2] = v0 + n0 * radius
  face_index_out[base1 + 0] = base1 + 0
  face_index_out[base1 + 1] = base1 + 1
  face_index_out[base1 + 2] = base1 + 2

  group_out[face_id0] = worldid
  group_out[face_id1] = worldid


@wp.kernel
def _build_flex_3d_shells(
  # Model:
  flex_shell: wp.array[int],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  shell_adr: int,
  vert_adr: int,
  face_offset: int,
  nface: int,
  # Out:
  face_point_out: wp.array[wp.vec3],
  face_index_out: wp.array[int],
  group_out: wp.array[int],
):
  """Create faces from 3D flex shell fragments (triangles).

  Each shell fragment contributes a single triangle whose vertices are taken
  directly from the flex vertex positions (one-sided surface).
  """
  worldid, shellid = wp.tid()

  base = shell_adr + shellid * 3
  i0 = vert_adr + flex_shell[base + 0]
  i1 = vert_adr + flex_shell[base + 1]
  i2 = vert_adr + flex_shell[base + 2]

  face_id = worldid * nface + face_offset + shellid
  base = face_id * 3

  v0 = flexvert_xpos_in[worldid, i0]
  v1 = flexvert_xpos_in[worldid, i1]
  v2 = flexvert_xpos_in[worldid, i2]

  face_point_out[base + 0] = v0
  face_point_out[base + 1] = v1
  face_point_out[base + 2] = v2

  face_index_out[base + 0] = base + 0
  face_index_out[base + 1] = base + 1
  face_index_out[base + 2] = base + 2

  group_out[face_id] = worldid


@wp.kernel
def _update_flex_2d_face_points(
  # Model:
  flex_vertadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_shelldataadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_shell: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  flexvert_norm_in: wp.array2d[wp.vec3],
  flex_id: int,
  nface: int,
  smooth: bool,
  # Out:
  face_point_out: wp.array[wp.vec3],
):
  worldid, workid = wp.tid()

  elem_adr = flex_elemdataadr[flex_id]
  vert_adr = flex_vertadr[flex_id]
  radius = flex_radius[flex_id]
  nelem = flex_elemnum[flex_id]
  world_face_offset = worldid * nface

  if workid < nelem:
    # 2D element faces
    elemid = workid
    ebase = elem_adr + elemid * 3
    i0 = vert_adr + flex_elem[ebase + 0]
    i1 = vert_adr + flex_elem[ebase + 1]
    i2 = vert_adr + flex_elem[ebase + 2]

    v0 = flexvert_xpos_in[worldid, i0]
    v1 = flexvert_xpos_in[worldid, i1]
    v2 = flexvert_xpos_in[worldid, i2]

    # TODO: Use static conditional
    if smooth:
      n0 = flexvert_norm_in[worldid, i0]
      n1 = flexvert_norm_in[worldid, i1]
      n2 = flexvert_norm_in[worldid, i2]
    else:
      face_nrm = wp.cross(v1 - v0, v2 - v0)
      face_nrm = wp.normalize(face_nrm)
      n0 = face_nrm
      n1 = face_nrm
      n2 = face_nrm

    p0_pos = v0 + radius * n0
    p1_pos = v1 + radius * n1
    p2_pos = v2 + radius * n2

    p0_neg = v0 - radius * n0
    p1_neg = v1 - radius * n1
    p2_neg = v2 - radius * n2

    face_id0 = world_face_offset + (2 * elemid)
    base0 = face_id0 * 3
    face_point_out[base0 + 0] = p0_pos
    face_point_out[base0 + 1] = p1_pos
    face_point_out[base0 + 2] = p2_pos

    face_id1 = world_face_offset + (2 * elemid + 1)
    base1 = face_id1 * 3
    face_point_out[base1 + 0] = p0_neg
    face_point_out[base1 + 1] = p1_neg
    face_point_out[base1 + 2] = p2_neg
  else:
    # 2D shell faces
    shell_adr = flex_shelldataadr[flex_id]
    shellid = workid - nelem
    sbase = shell_adr + 2 * shellid
    i0 = vert_adr + flex_shell[sbase + 0]
    i1 = vert_adr + flex_shell[sbase + 1]

    v0 = flexvert_xpos_in[worldid, i0]
    v1 = flexvert_xpos_in[worldid, i1]

    n0 = flexvert_norm_in[worldid, i0]
    n1 = flexvert_norm_in[worldid, i1]

    shell_face_offset = 2 * nelem
    face_id0 = world_face_offset + shell_face_offset + (2 * shellid)
    base0 = face_id0 * 3
    face_point_out[base0 + 0] = v0 + radius * n0
    face_point_out[base0 + 1] = v1 - radius * n1
    face_point_out[base0 + 2] = v1 + radius * n1

    face_id1 = world_face_offset + shell_face_offset + (2 * shellid + 1)
    base1 = face_id1 * 3
    face_point_out[base1 + 0] = v1 - radius * n1
    face_point_out[base1 + 1] = v0 + radius * n0
    face_point_out[base1 + 2] = v0 - radius * n0


@wp.kernel
def _update_flex_3d_face_points(
  # Model:
  flex_vertadr: wp.array[int],
  flex_shelldataadr: wp.array[int],
  flex_shell: wp.array[int],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  flex_id: int,
  nface: int,
  # Out:
  face_point_out: wp.array[wp.vec3],
):
  worldid, shellid = wp.tid()

  shell_adr = flex_shelldataadr[flex_id]
  vert_adr = flex_vertadr[flex_id]

  face_id = worldid * nface + shellid
  fbase = face_id * 3

  sbase = shell_adr + shellid * 3
  i0 = vert_adr + flex_shell[sbase + 0]
  i1 = vert_adr + flex_shell[sbase + 1]
  i2 = vert_adr + flex_shell[sbase + 2]

  face_point_out[fbase + 0] = flexvert_xpos_in[worldid, i0]
  face_point_out[fbase + 1] = flexvert_xpos_in[worldid, i1]
  face_point_out[fbase + 2] = flexvert_xpos_in[worldid, i2]


def build_flex_bvh(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: int,
  flex_id: int,
  constructor: str = "sah",
  leaf_size: int = 2,
  deterministic: bool = False,
) -> tuple[wp.Mesh, wp.array, wp.array, wp.array, int]:
  """Create a Warp mesh BVH for a single 2D or 3D flex."""
  nflexvert = mjm.nflexvert

  flex_dim = wp.array(mjm.flex_dim, dtype=int)
  flex_elemadr = wp.array(mjm.flex_elemadr, dtype=int)
  flex_elemnum = wp.array(mjm.flex_elemnum, dtype=int)
  flex_elem = wp.array(mjm.flex_elem, dtype=int)
  flex_elemdataadr = wp.array(mjm.flex_elemdataadr, dtype=int)
  flex_vertadr = wp.array(mjm.flex_vertadr, dtype=int)
  flexvert_xpos = wp.array(np.tile(mjd.flexvert_xpos[np.newaxis, :, :], (nworld, 1, 1)), dtype=wp.vec3)

  dim = int(mjm.flex_dim[flex_id])
  nelem = int(mjm.flex_elemnum[flex_id])
  nshell = int(mjm.flex_shellnum[flex_id])

  if dim == 2:
    nface = 2 * nelem + 2 * nshell
  else:
    nface = nshell

  face_point = wp.empty(nface * 3 * nworld, dtype=wp.vec3)
  face_index = wp.empty(nface * 3 * nworld, dtype=wp.int32)
  group = wp.empty(nface * nworld, dtype=int)

  flexvert_norm = wp.zeros((nworld, nflexvert), dtype=wp.vec3)
  flex_shell = wp.array(mjm.flex_shell, dtype=int)

  wp.launch(
    kernel=accumulate_flex_vertex_normals(deterministic),
    dim=(nworld, mjm.nflexelem),
    inputs=[mjm.nflex, flex_dim, flex_vertadr, flex_elemadr, flex_elemnum, flex_elemdataadr, flex_elem, flexvert_xpos],
    outputs=[flexvert_norm],
  )

  wp.launch(
    kernel=normalize_vertex_normals,
    dim=(nworld, nflexvert),
    inputs=[flexvert_norm],
  )

  elem_adr = mjm.flex_elemdataadr[flex_id]
  shell_adr = mjm.flex_shelldataadr[flex_id]
  vert_adr = mjm.flex_vertadr[flex_id]

  if dim == 2:
    wp.launch(
      kernel=_build_flex_2d_elements,
      dim=(nworld, nelem),
      inputs=[
        flex_elem,
        flexvert_xpos,
        flexvert_norm,
        elem_adr,
        vert_adr,
        0,  # face_offset
        mjm.flex_radius[flex_id],
        nface,
      ],
      outputs=[face_point, face_index, group],
    )

    wp.launch(
      kernel=_build_flex_2d_sides,
      dim=(nworld, nshell),
      inputs=[
        flex_shell,
        flexvert_xpos,
        flexvert_norm,
        shell_adr,
        vert_adr,
        2 * nelem,  # face_offset
        mjm.flex_radius[flex_id],
        nface,
      ],
      outputs=[face_point, face_index, group],
    )
  elif dim == 3:
    wp.launch(
      kernel=_build_flex_3d_shells,
      dim=(nworld, nshell),
      inputs=[
        flex_shell,
        flexvert_xpos,
        shell_adr,
        vert_adr,
        0,  # face_offset
        nface,
      ],
      outputs=[face_point, face_index, group],
    )

  flex_mesh = wp.Mesh(
    points=face_point,
    indices=face_index,
    groups=group,
    bvh_constructor=constructor,
    bvh_leaf_size=leaf_size,
  )

  group_root = wp.empty(nworld, dtype=int)
  wp.launch(
    kernel=compute_mesh_group_roots,
    dim=nworld,
    inputs=[flex_mesh.id],
    outputs=[group_root],
  )

  return flex_mesh, group_root


def refit_flex_bvh(m: Model, d: Data, rc: RenderContext):
  """Refit per-flex BVHs."""
  flexvert_norm = wp.zeros(d.flexvert_xpos.shape, dtype=wp.vec3)

  wp.launch(
    kernel=accumulate_flex_vertex_normals(bool(m.opt.deterministic & DeterminismType.ATOMICS)),
    dim=(d.nworld, m.nflexelem),
    inputs=[
      m.nflex,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_elem,
      d.flexvert_xpos,
    ],
    outputs=[flexvert_norm],
  )

  wp.launch(
    kernel=normalize_vertex_normals,
    dim=(d.nworld, d.flexvert_xpos.shape[1]),
    inputs=[flexvert_norm],
  )

  for i in range(m.nflex):
    if rc.flex_dim_np[i] == 1:
      continue
    mesh = rc.flex_mesh_registry[i]
    nface = mesh.points.shape[0] // (3 * d.nworld)

    if rc.flex_dim_np[i] == 2:
      wp.launch(
        kernel=_update_flex_2d_face_points,
        dim=(d.nworld, nface // 2),
        inputs=[
          m.flex_vertadr,
          m.flex_elemnum,
          m.flex_elemdataadr,
          m.flex_shelldataadr,
          m.flex_elem,
          m.flex_shell,
          m.flex_radius,
          d.flexvert_xpos,
          flexvert_norm,
          i,
          nface,
          rc.flex_render_smooth,
        ],
        outputs=[mesh.points],
      )
    else:
      wp.launch(
        kernel=_update_flex_3d_face_points,
        dim=(d.nworld, nface),
        inputs=[
          m.flex_vertadr,
          m.flex_shelldataadr,
          m.flex_shell,
          d.flexvert_xpos,
          i,
          nface,
        ],
        outputs=[mesh.points],
      )

    mesh.refit()


@wp.kernel
def _compute_splat_bounds(
  # In:
  splat_position: wp.array[wp.vec3],
  splat_rotation: wp.array[wp.quat],
  splat_scale: wp.array[wp.vec3],
  splat_rgba: wp.array[wp.vec4],
  # Out:
  lower_out: wp.array[wp.vec3],
  upper_out: wp.array[wp.vec3],
):
  tid = wp.tid()
  pos = splat_position[tid]
  rot = splat_rotation[tid]
  scale = splat_scale[tid]
  opacity = wp.max(splat_rgba[tid][3], 1.0e-6)
  # 1e-6 caps the radius of nearly transparent splats,
  # 0.97 keeps it positive below the response cutoff.
  response = wp.clamp(wp.static(SPLAT_MIN_RESPONSE) / opacity, 1.0e-6, 0.97)
  radius = wp.sqrt(-2.0 * wp.log(response))

  lo = wp.vec3(MJ_MAXVAL)
  hi = wp.vec3(-MJ_MAXVAL)
  for x in range(2):
    for y in range(2):
      for z in range(2):
        corner = wp.vec3(
          scale[0] * radius * (2.0 * float(x) - 1.0),
          scale[1] * radius * (2.0 * float(y) - 1.0),
          scale[2] * radius * (2.0 * float(z) - 1.0),
        )
        point = pos + mjmath.rot_vec_quat(corner, rot)
        lo = wp.min(lo, point)
        hi = wp.max(hi, point)

  lower_out[tid] = lo
  upper_out[tid] = hi


def build_splat_bvh(
  splat_position: np.ndarray,
  splat_rotation: np.ndarray,
  splat_scale: np.ndarray,
  splat_rgba: np.ndarray,
  splat_adr: np.ndarray,
  splat_group_id: np.ndarray,
  constructor: str = "sah",
) -> tuple:
  """Creates a splat BVH for a set of splats."""
  count = splat_position.shape[0]
  splat_group = np.empty(count, dtype=np.int32)
  for group_id in range(len(splat_adr) - 1):
    splat_group[splat_adr[group_id] : splat_adr[group_id + 1]] = group_id

  splat_position = wp.array(splat_position, dtype=wp.vec3)
  splat_rotation = wp.array(splat_rotation, dtype=wp.quat)
  splat_scale = wp.array(splat_scale, dtype=wp.vec3)
  splat_rgba = wp.array(splat_rgba, dtype=wp.vec4)

  groups = wp.array(splat_group, dtype=int)
  lower = wp.empty(count, dtype=wp.vec3)
  upper = wp.empty(count, dtype=wp.vec3)
  wp.launch(
    _compute_splat_bounds,
    dim=count,
    inputs=[splat_position, splat_rotation, splat_scale, splat_rgba],
    outputs=[lower, upper],
  )

  splat_bvh = wp.Bvh(lower, upper, groups=groups, constructor=constructor)
  group_root = wp.empty(len(splat_adr) - 1, dtype=int)

  wp.launch(
    kernel=compute_bvh_group_roots,
    dim=group_root.shape[0],
    inputs=[splat_bvh.id],
    outputs=[group_root],
  )

  splat_group_root = wp.array(group_root.numpy()[splat_group_id], dtype=int)
  return splat_position, splat_rotation, splat_scale, splat_rgba, splat_bvh, splat_bvh.id, lower, upper, splat_group_root, count


def refit_splat_bvh(rc: RenderContext):
  """Refits splat BVHs."""
  wp.launch(
    _compute_splat_bounds,
    dim=rc.splat_count,
    inputs=[rc.splat_position, rc.splat_rotation, rc.splat_scale, rc.splat_rgba],
    outputs=[rc.splat_lower, rc.splat_upper],
  )
  rc.splat_bvh.refit()
