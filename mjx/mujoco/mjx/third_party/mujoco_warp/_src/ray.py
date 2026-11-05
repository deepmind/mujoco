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

from typing import Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.math import safe_div
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model
from mujoco.mjx.third_party.mujoco_warp._src.types import RenderContext
from mujoco.mjx.third_party.mujoco_warp._src.types import vec6

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Widen ray prune bounds so a coincident hit survives to the closest-hit tie-break.
RAY_TOL_REL = 1.0e-6
RAY_TOL_ABS = 1.0e-9


@wp.func
def _ray_map(pos: wp.vec3, mat: wp.mat33, pnt: wp.vec3, vec: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:
  """Maps ray to local geom frame coordinates.

  Args:
      pos: position of geom frame
      mat: orientation of geom frame
      pnt: starting point of ray in world coordinates
      vec: direction of ray in world coordinates

  Returns:
      3D point and 3D direction in local geom frame
  """
  matT = wp.transpose(mat)
  lpnt = matT @ (pnt - pos)
  lvec = matT @ vec

  return lpnt, lvec


@wp.func
def _ray_eliminate(
  # Model:
  body_weldid: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_matid: wp.array[int],  # kernel_analyzer: ignore
  geom_group: wp.array[int],
  geom_rgba: wp.array[wp.vec4],  # kernel_analyzer: ignore
  mat_rgba: wp.array[wp.vec4],  # kernel_analyzer: ignore
  # In:
  geomid: int,
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: int,
) -> bool:
  """Eliminate ray."""
  bodyid = geom_bodyid[geomid]
  matid = geom_matid[geomid]  # kernel_analyzer: ignore

  # body exclusion
  if bodyid == bodyexclude:
    return True

  # invisible geom exclusion
  if matid < 0 and geom_rgba[geomid][3] == 0.0:  # kernel_analyzer: ignore
    return True

  # invisible material exclusion
  if matid >= 0:
    if mat_rgba[matid][3] == 0.0:  # kernel_analyzer: ignore
      return True

  # static exclusion
  if not flg_static and body_weldid[bodyid] == 0:
    return True

  # no geomgroup inclusion
  if (
    geomgroup[0] == -1
    and geomgroup[1] == -1
    and geomgroup[2] == -1
    and geomgroup[3] == -1
    and geomgroup[4] == -1
    and geomgroup[5] == -1
  ):
    return False

  # group inclusion/exclusion
  groupid = wp.min(5, wp.max(0, geom_group[geomid]))

  return geomgroup[groupid] == 0


@wp.func
def _ray_quad(a: float, b: float, c: float) -> Tuple[float, wp.vec2]:
  """Compute solutions from quadratic: a*x^2 + 2*b*x + c = 0."""
  det = b * b - a * c
  if det < MJ_MINVAL:
    return -1.0, wp.vec2(-1.0, -1.0)
  det = wp.sqrt(det)

  # compute the two solutions
  den = safe_div(1.0, a)
  x0 = (-b - det) * den
  x1 = (-b + det) * den
  x = wp.vec2(x0, x1)

  # finalize result
  if x0 >= 0.0:
    return x0, x
  elif x1 >= 0.0:
    return x1, x
  else:
    return -1.0, x


@wp.func
def _orthogonal_basis(vec: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:
  """Computes two orthonormal basis vectors b0, b1 perpendicular to unit vector vec.

  Reference:
    Duff et al. (2017), "Building an Orthonormal Basis, Revisited", JCGT 6(1).
    Refines Frisvad (2012), "Building an Orthonormal Basis from a 3D Unit Vector
    Without Normalization", JCGT 1(1), to remove the singularity without branching.
  """
  sign = wp.where(vec[2] >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + vec[2])
  b = vec[0] * vec[1] * a

  b0 = wp.vec3(
    1.0 + sign * vec[0] * vec[0] * a,
    sign * b,
    -sign * vec[0],
  )
  b1 = wp.vec3(
    b,
    sign + vec[1] * vec[1] * a,
    -vec[1],
  )
  return b0, b1


@wp.func
def _ray_triangle(
  v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, pnt: wp.vec3, vec: wp.vec3, b0: wp.vec3, b1: wp.vec3
) -> Tuple[float, wp.vec3]:
  """Returns the distance and normal at which a ray intersects with a triangle."""
  dif0 = v0 - pnt
  dif1 = v1 - pnt
  dif2 = v2 - pnt

  # project difference vectors in normal plane
  planar_00 = wp.dot(dif0, b0)
  planar_01 = wp.dot(dif0, b1)
  planar_10 = wp.dot(dif1, b0)
  planar_11 = wp.dot(dif1, b1)
  planar_20 = wp.dot(dif2, b0)
  planar_21 = wp.dot(dif2, b1)

  # reject if on the same side of any coordinate axis
  if (
    (planar_00 > 0.0 and planar_10 > 0.0 and planar_20 > 0.0)
    or (planar_00 < 0.0 and planar_10 < 0.0 and planar_20 < 0.0)
    or (planar_01 > 0.0 and planar_11 > 0.0 and planar_21 > 0.0)
    or (planar_01 < 0.0 and planar_11 < 0.0 and planar_21 < 0.0)
  ):
    return -1.0, wp.vec3()

  # determine if origin is inside planar projection of triangle
  # A = (p0-p2, p1-p2), b = -p2, solve A*t = b
  A00 = planar_00 - planar_20
  A10 = planar_10 - planar_20
  A01 = planar_01 - planar_21
  A11 = planar_11 - planar_21

  b = wp.vec2(-planar_20, -planar_21)

  det = A00 * A11 - A10 * A01
  if wp.abs(det) < MJ_MINVAL:
    return -1.0, wp.vec3()

  t0 = (A11 * b[0] - A10 * b[1]) / det
  t1 = (-A01 * b[0] + A00 * b[1]) / det

  # check if outside
  if t0 < 0.0 or t1 < 0.0 or t0 + t1 > 1.0:
    return -1.0, wp.vec3()

  # intersect ray with plane of triangle
  dif0 = v0 - v2
  dif1 = v1 - v2
  dif2 = pnt - v2
  nrm = wp.cross(dif0, dif1)  # normal to triangle plane
  denom = wp.dot(vec, nrm)
  if wp.abs(denom) < MJ_MINVAL:
    return -1.0, wp.vec3()

  dist = -wp.dot(dif2, nrm) / denom
  return wp.where(dist >= 0.0, dist, -1.0), wp.normalize(nrm)


@wp.func
def ray_plane(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:
  """Returns the distance and normal at which a ray intersects with a plane."""
  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # z-vec not pointing towards front face: reject
  if lvec[2] > -MJ_MINVAL:
    return -1.0, wp.vec3()

  # intersection with plane
  x = -lpnt[2] / lvec[2]
  if x < 0.0:
    return -1.0, wp.vec3()

  p = wp.vec2(lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1])

  # accept only within rendered rectangle
  if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):
    return x, wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
  else:
    return -1.0, wp.vec3()


@wp.func
def ray_sphere(pos: wp.vec3, dist_sqr: float, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:
  """Returns the distance and normal at which a ray intersects with a sphere."""
  dif = pnt - pos

  a = wp.dot(vec, vec)
  b = wp.dot(vec, dif)
  c = wp.dot(dif, dif) - dist_sqr

  sol, _ = _ray_quad(a, b, c)
  normal = wp.vec3()
  if sol >= 0:
    s = pnt + vec * sol
    normal = wp.normalize(s - pos)
  return sol, normal


@wp.func
def ray_capsule(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:
  """Returns the distance and normal at which a ray intersects with a capsule."""
  # bounding sphere test
  ssz = size[0] + size[1]
  dist_sphere, normal_sphere = ray_sphere(pos, ssz * ssz, pnt, vec)
  if dist_sphere < 0:
    return -1.0, wp.vec3()

  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # init solution
  x = -1.0

  # cylinder round side: (x * lvec + lpnt)' * (x * lvec + lpnt) = size[0] * size[0]
  sq_size0 = size[0] * size[0]
  a = lvec[0] * lvec[0] + lvec[1] * lvec[1]
  b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
  c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - sq_size0

  # solve a * x^2 + 2 * b * x + c = 0
  sol, xx = _ray_quad(a, b, c)
  part = 0  # -1: bottom, 0: cylinder, 1: top

  # make sure round solution is between flat sides (must use local z component)
  # TODO: We should add a test to catch this case.
  if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
    if x < 0.0 or sol < x:
      x = sol

  # top cap
  ldif = wp.vec3(lpnt[0], lpnt[1], lpnt[2] - size[1])
  a += lvec[2] * lvec[2]
  b = wp.dot(lvec, ldif)
  c = wp.dot(ldif, ldif) - sq_size0
  _, xx = _ray_quad(a, b, c)

  # accept only top half of sphere (use local z component)
  for i in range(2):
    if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:
      if x < 0.0 or xx[i] < x:
        x = xx[i]
        part = 1

  # bottom cap
  ldif = wp.vec3(ldif[0], ldif[1], lpnt[2] + size[1])
  b = wp.dot(lvec, ldif)
  c = wp.dot(ldif, ldif) - sq_size0
  _, xx = _ray_quad(a, b, c)

  # accept only bottom half of sphere (use local z component)
  for i in range(2):
    if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:
      if x < 0.0 or xx[i] < x:
        x = xx[i]
        part = -1

  normal = wp.vec3()
  if x >= 0:
    normal[0] = lpnt[0] + lvec[0] * x
    normal[1] = lpnt[1] + lvec[1] * x
    if part == 0:
      normal[2] = 0.0
    else:
      normal[2] = lpnt[2] + lvec[2] * x - size[1] * float(part)

    # normalize, rotate into global frame
    normal = wp.normalize(normal)
    normal = mat @ normal

  return x, normal


@wp.func
def ray_ellipsoid(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:
  """Returns the distance and normal at which a ray intersects with an ellipsoid."""
  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # invert size^2
  s = wp.vec3(safe_div(1.0, size[0] * size[0]), safe_div(1.0, size[1] * size[1]), safe_div(1.0, size[2] * size[2]))

  # (x * lvec + lpnt)' * diag(1 / size^2) * (x * lvec + lpnt) = 1
  slvec = wp.cw_mul(s, lvec)
  a = wp.dot(slvec, lvec)
  b = wp.dot(slvec, lpnt)
  c = wp.dot(wp.cw_mul(s, lpnt), lpnt) - 1.0

  # solve a * x^2 + 2 * b * x + c = 0
  sol, _ = _ray_quad(a, b, c)

  normal = wp.vec3()
  if sol >= 0:
    # surface intersection (local frame)
    l = lpnt + lvec * sol

    # gradient of ellipsoid function
    normal = wp.cw_mul(s, l)
    normal = wp.normalize(normal)
    normal = mat @ normal

  return sol, normal


@wp.func
def ray_cylinder(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:
  """Returns the distance and normal at which a ray intersects with a cylinder."""
  # bounding sphere test
  ssz = size[0] * size[0] + size[1] * size[1]
  dist_sphere, normal_sphere = ray_sphere(pos, ssz, pnt, vec)
  if dist_sphere < 0:
    return -1.0, wp.vec3()

  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # init solution
  x = -1.0
  part = 0  # -1: bottom, 0: cylinder, 1: top

  # flat sides
  if wp.abs(lvec[2]) > MJ_MINVAL:
    for side in range(-1, 2, 2):
      # solution of: lpnt[2] + x * lvec[2] = side * height_size
      sol = (float(side) * size[1] - lpnt[2]) / lvec[2]

      # process if non-negative
      if sol >= 0.0:
        # intersection with horizontal face
        p = wp.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])

        # accept within radius
        if wp.dot(p, p) <= size[0] * size[0]:
          if x < 0.0 or sol < x:
            x = sol
            part = side

  # (x * lvec + lpnt)' * (x * lvec + lpnt) = size[0] * size[0]
  a = lvec[0] * lvec[0] + lvec[1] * lvec[1]
  b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
  c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - size[0] * size[0]

  # solve a * x^2 + 2 * b * x + c = 0
  sol, _ = _ray_quad(a, b, c)

  # make sure round solution is between flat sides
  if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
    if x < 0.0 or sol < x:
      x = sol
      part = 0

  normal = wp.vec3()
  if x >= 0:
    if part == 0:
      normal = lpnt + lvec * x
      normal[2] = 0.0
      normal = wp.normalize(normal)
    else:
      normal = wp.vec3(0.0, 0.0, float(part))

    normal = mat @ normal

  return x, normal


@wp.func
def ray_box(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, vec6, wp.vec3]:
  """Returns distance, per side information, and normal at which a ray intersects with a box."""
  all = vec6(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

  # bounding sphere test
  ssz = wp.dot(size, size)
  dist_sphere, _ = ray_sphere(pos, ssz, pnt, vec)
  if dist_sphere < 0:
    return -1.0, all, wp.vec3()

  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # init solution
  x = float(-1.0)
  face_side = -1
  face_axis = -1

  # loop over axes with non-zero vec
  for i in range(3):
    if wp.abs(lvec[i]) > MJ_MINVAL:
      for side in range(-1, 2, 2):
        # solution of: lpnt[i] + x * lvec[i] = side * size[i]
        sol = (float(side) * size[i] - lpnt[i]) / lvec[i]

        # process if non-negative
        if sol >= 0.0:
          id0 = 1 if i == 0 else 0
          id1 = 1 if i == 2 else 2

          # intersection with face
          p0 = lpnt[id0] + sol * lvec[id0]
          p1 = lpnt[id1] + sol * lvec[id1]

          # accept within rectangle
          if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):
            # update
            if x < 0.0 or sol < x:
              x = sol
              face_axis = i
              face_side = side

            # save in all
            all[2 * i + (side + 1) // 2] = sol

  normal = wp.vec3()
  if x >= 0:
    normal[face_axis] = float(face_side)
    normal = mat @ normal

  return x, all, normal


@wp.func
def ray_hfield(
  # Model:
  geom_dataid: wp.array2d[int],
  hfield_size: wp.array[wp.vec4],
  hfield_nrow: wp.array[int],
  hfield_ncol: wp.array[int],
  hfield_adr: wp.array[int],
  hfield_data: wp.array[float],
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  id: int,
  worldid: int,
) -> Tuple[float, wp.vec3]:
  # hfield id and dimensions
  hid = geom_dataid[worldid % geom_dataid.shape[0], id]
  nrow = hfield_nrow[hid]
  ncol = hfield_ncol[hid]

  size = hfield_size[hid]
  adr = hfield_adr[hid]

  mat_col = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])

  # compute size and pos of base box
  base_scale = size[3] * 0.5
  base_size = wp.vec3(size[0], size[1], base_scale)
  base_pos = pos - mat_col * base_scale

  # compute size and pos of top box
  top_scale = size[2] * 0.5
  top_size = wp.vec3(size[0], size[1], top_scale)
  top_pos = pos + mat_col * top_scale

  # init: intersection with base box
  x, _all_base, normal_base = ray_box(base_pos, mat, base_size, pnt, vec)

  # check top box: done if no intersection
  top_intersect, all, normal_top = ray_box(top_pos, mat, top_size, pnt, vec)

  if top_intersect < 0.0 or nrow < 2 or ncol < 2:
    return x, normal_base

  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # construct basis vectors of normal plane
  b0, b1 = _orthogonal_basis(lvec)

  # find ray segment intersecting top box
  seg = wp.vec2(0.0, top_intersect)
  for i in range(6):
    if all[i] > seg[1]:
      seg[0] = top_intersect
      seg[1] = all[i]

  # project segment endpoints in horizontal plane, discretize
  dx = safe_div(2.0 * size[0], float(ncol - 1))
  dy = safe_div(2.0 * size[1], float(nrow - 1))
  SX = wp.vec2(
    safe_div(lpnt[0] + seg[0] * lvec[0] + size[0], dx),
    safe_div(lpnt[0] + seg[1] * lvec[0] + size[0], dx),
  )
  SY = wp.vec2(
    safe_div(lpnt[1] + seg[0] * lvec[1] + size[1], dy),
    safe_div(lpnt[1] + seg[1] * lvec[1] + size[1], dy),
  )

  # compute ranges, with +1 padding
  cmin = wp.max(0, int(wp.floor(wp.min(SX[0], SX[1])) - 1.0))
  cmax = wp.min(ncol - 1, int(wp.ceil(wp.max(SX[0], SX[1])) + 1.0))
  rmin = wp.max(0, int(wp.floor(wp.min(SY[0], SY[1])) - 1.0))
  rmax = wp.min(nrow - 1, int(wp.ceil(wp.max(SY[0], SY[1])) + 1.0))

  normal_local = wp.vec3()
  if x >= 0:
    normal_local = wp.transpose(mat) @ normal_base

  # check triangles within bounds
  for r in range(rmin, rmax):
    for c in range(cmin, cmax):
      # first triangle
      v0 = wp.vec3(
        dx * float(c) - size[0],
        dy * float(r) - size[1],
        hfield_data[adr + r * ncol + c] * size[2],
      )
      v1 = wp.vec3(
        dx * float(c + 1) - size[0],
        dy * float(r) - size[1],
        hfield_data[adr + r * ncol + (c + 1)] * size[2],
      )
      v2 = wp.vec3(
        dx * float(c + 1) - size[0],
        dy * float(r + 1) - size[1],
        hfield_data[adr + (r + 1) * ncol + (c + 1)] * size[2],
      )
      sol, normal_tri = _ray_triangle(v0, v1, v2, lpnt, lvec, b0, b1)
      if sol >= 0.0 and (x < 0.0 or sol < x):
        x = sol
        normal_local = normal_tri

      # second triangle
      v0 = wp.vec3(dx * float(c) - size[0], dy * float(r) - size[1], hfield_data[adr + r * ncol + c] * size[2])
      v1 = wp.vec3(
        dx * float(c + 1) - size[0], dy * float(r + 1) - size[1], hfield_data[adr + (r + 1) * ncol + (c + 1)] * size[2]
      )
      v2 = wp.vec3(dx * float(c) - size[0], dy * float(r + 1) - size[1], hfield_data[adr + (r + 1) * ncol + c] * size[2])
      sol, normal_tri = _ray_triangle(v0, v1, v2, lpnt, lvec, b0, b1)
      if sol >= 0.0 and (x < 0.0 or sol < x):
        x = sol
        normal_local = normal_tri

  # check viable sides of top box
  for i in range(4):
    if all[i] >= 0.0 and (all[i] < x or x < 0.0):
      # normalized height of intersection point
      z = safe_div(lpnt[2] + all[i] * lvec[2], size[2])

      # rectangle points: y, y0, z0, z1
      # side normal to x-axis
      if i < 2:
        y = safe_div(lpnt[1] + all[i] * lvec[1] + size[1], dy)
        y0 = wp.max(0.0, wp.min(float(nrow - 2), wp.floor(y)))
        if i == 1:
          z0 = hfield_data[adr + int(wp.round(y0 + 0.0)) * ncol + ncol - 1]
          z1 = hfield_data[adr + int(wp.round(y0 + 1.0)) * ncol + ncol - 1]
        else:
          z0 = hfield_data[adr + int(wp.round(y0 + 0.0)) * ncol]
          z1 = hfield_data[adr + int(wp.round(y0 + 1.0)) * ncol]
      # side normal to y-axis
      else:
        y = safe_div(lpnt[0] + all[i] * lvec[0] + size[0], dx)
        y0 = wp.max(0.0, wp.min(float(ncol - 2), wp.floor(y)))
        if i == 3:
          z0 = hfield_data[adr + int(wp.round(y0 + 0.0)) + (nrow - 1) * ncol]
          z1 = hfield_data[adr + int(wp.round(y0 + 1.0)) + (nrow - 1) * ncol]
        else:
          z0 = hfield_data[adr + int(wp.round(y0 + 0.0))]
          z1 = hfield_data[adr + int(wp.round(y0 + 1.0))]

      # check if point is below line segments
      if z < z0 * (y0 + 1.0 - y) + z1 * (y - y0):
        x = all[i]
        normal_local = wp.vec3(float(i == 1) - float(i == 0), float(i == 3) - float(i == 2), 0.0)

  if x >= 0:
    normal_local = mat @ normal_local
  return x, normal_local


@wp.func
def ray_mesh(
  # Model:
  nmeshface: int,
  mesh_vertadr: wp.array[int],
  mesh_faceadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_face: wp.array[wp.vec3i],
  # In:
  data_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> Tuple[float, wp.vec3]:
  """Returns the distance and normal for ray mesh intersections."""
  # bounding box test
  dist_box, _all, _normal = ray_box(pos, mat, size, pnt, vec)
  if dist_box < 0.0:
    return -1.0, wp.vec3()

  pnt, vec = _ray_map(pos, mat, pnt, vec)

  # compute orthogonal basis vectors
  b0, b1 = _orthogonal_basis(vec)

  x = float(-1.0)
  normal = wp.vec3()

  # get mesh vertex data range
  vert_start = mesh_vertadr[data_id]

  # get mesh face and vertex data
  face_start = mesh_faceadr[data_id]

  if data_id + 1 < mesh_faceadr.shape[0]:
    face_end = mesh_faceadr[data_id + 1]
  else:
    face_end = nmeshface

  # iterate through all faces
  for i in range(face_start, face_end):
    # get vertices for this face
    v_idx = mesh_face[i]

    # create triangle struct
    v0 = mesh_vert[vert_start + v_idx.x]
    v1 = mesh_vert[vert_start + v_idx.y]
    v2 = mesh_vert[vert_start + v_idx.z]

    # calculate intersection
    dist, normal_tri = _ray_triangle(v0, v1, v2, pnt, vec, b0, b1)
    if dist >= 0 and (x < 0 or dist < x):
      x = dist
      normal = normal_tri

  normal = mat @ normal

  return x, normal


@wp.func
def ray_mesh_with_bvh(
  # In:
  mesh_bvh_id: wp.array[wp.uint64],
  mesh_geom_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
  cull_backfaces: bool,
) -> Tuple[float, wp.vec3, float, float, int, int]:
  """Returns intersection information for ray mesh intersections.

  Requires wp.Mesh be constructed and their ids to be passed.

  When ``cull_backfaces`` is True, the function rejects exit-face hits in the
  local-space frame. This matches MuJoCo OpenGL rendering's backface culling rule.
  """
  t = float(-1.0)
  u = float(0.0)
  v = float(0.0)
  sign = float(0.0)
  n = wp.vec3(0.0, 0.0, 0.0)
  f = int(-1)

  lpnt, lvec = _ray_map(pos, mat, pnt, vec)
  mesh_bvh = mesh_bvh_id[mesh_geom_id]

  # Culling drops back-facing triangles, not the mesh: like a rasterizer, march
  # past culled hits to the next surface (bounded to keep the kernel in check).
  offset = float(0.0)
  for _ in range(8):
    hit = wp.mesh_query_ray(mesh_bvh, lpnt + lvec * offset, lvec, max_t - offset, t, u, v, sign, n, f)
    if not hit:
      return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1
    if (not cull_backfaces) or wp.dot(lvec, n) < 0.0:
      normal = mat @ n
      normal = wp.normalize(normal)
      return offset + t, normal, u, v, f, mesh_geom_id
    offset += t * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS

  return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1


@wp.func
def ray_mesh_with_bvh_anyhit(
  # In:
  mesh_bvh_id: wp.array[wp.uint64],
  mesh_geom_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
) -> bool:
  """Returns True if there is any hit for ray mesh intersections.

  Requires wp.Mesh be constructed and their ids to be passed. This variant is useful
  for shadow ray casts where the only goal is if there is any ray hit.
  """
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)
  return wp.mesh_query_ray_anyhit(mesh_bvh_id[mesh_geom_id], lpnt, lvec, max_t)


@wp.func
def ray_flex_with_bvh(
  # In:
  flex_bvh_id: wp.array[wp.uint64],
  flexid: int,
  group_root: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
) -> Tuple[float, wp.vec3, float, float, int]:
  """Returns intersection information for flex intersections.

  Requires wp.Mesh be constructed and their ids to be passed. Flex are already in world space.
  """
  t = float(-1.0)
  u = float(0.0)
  v = float(0.0)
  sign = float(0.0)
  n = wp.vec3(0.0, 0.0, 0.0)
  f = int(-1)

  hit = wp.mesh_query_ray(flex_bvh_id[flexid], pnt, vec, max_t, t, u, v, sign, n, f, group_root)

  if hit:
    return t, n, u, v, f

  return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1


@wp.func
def ray_flex_with_bvh_anyhit(
  # In:
  flex_bvh_id: wp.array[wp.uint64],
  flexid: int,
  group_root: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
) -> bool:
  """Returns True if there is any hit for ray flex intersections.

  Requires wp.Mesh be constructed and their ids to be passed. Flex are already in world space.
  """
  return wp.mesh_query_ray_anyhit(flex_bvh_id[flexid], pnt, vec, max_t, group_root)


@wp.func
def ray_geom(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3, geomtype: int) -> Tuple[float, wp.vec3]:
  """Returns distance along ray to intersection with geom and normal at intersection point.

  If no intersection is found, returns -1 and zero vector.
  """
  # TODO(team): static loop unrolling to remove unnecessary branching
  if geomtype == GeomType.PLANE:
    return ray_plane(pos, mat, size, pnt, vec)
  elif geomtype == GeomType.SPHERE:
    return ray_sphere(pos, size[0] * size[0], pnt, vec)
  elif geomtype == GeomType.CAPSULE:
    return ray_capsule(pos, mat, size, pnt, vec)
  elif geomtype == GeomType.ELLIPSOID:
    return ray_ellipsoid(pos, mat, size, pnt, vec)
  elif geomtype == GeomType.CYLINDER:
    return ray_cylinder(pos, mat, size, pnt, vec)
  elif geomtype == GeomType.BOX:
    dist, _, normal = ray_box(pos, mat, size, pnt, vec)
    return dist, normal
  else:
    return -1.0, wp.vec3()


@wp.func
def _ray_geom_mesh(
  # Model:
  nmeshface: int,
  body_weldid: wp.array[int],
  geom_type: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_matid: wp.array2d[int],
  geom_group: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rgba: wp.array2d[wp.vec4],
  mesh_vertadr: wp.array[int],
  mesh_faceadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_face: wp.array[wp.vec3i],
  hfield_size: wp.array[wp.vec4],
  hfield_nrow: wp.array[int],
  hfield_ncol: wp.array[int],
  hfield_adr: wp.array[int],
  hfield_data: wp.array[float],
  mat_rgba: wp.array2d[wp.vec4],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: int,
  geomid: int,
) -> Tuple[float, wp.vec3]:
  if not _ray_eliminate(
    body_weldid,
    geom_bodyid,
    geom_matid[worldid % geom_matid.shape[0]],
    geom_group,
    geom_rgba[worldid % geom_rgba.shape[0]],
    mat_rgba[worldid % mat_rgba.shape[0]],
    geomid,
    geomgroup,
    flg_static,
    bodyexclude,
  ):
    pos = geom_xpos_in[worldid, geomid]
    mat = geom_xmat_in[worldid, geomid]
    type = geom_type[geomid]

    if type == GeomType.MESH:
      return ray_mesh(
        nmeshface,
        mesh_vertadr,
        mesh_faceadr,
        mesh_vert,
        mesh_face,
        geom_dataid[worldid % geom_dataid.shape[0], geomid],
        pos,
        mat,
        geom_size[worldid % geom_size.shape[0], geomid],
        pnt,
        vec,
      )
    elif type == GeomType.HFIELD:
      return ray_hfield(
        geom_dataid,
        hfield_size,
        hfield_nrow,
        hfield_ncol,
        hfield_adr,
        hfield_data,
        pos,
        mat,
        pnt,
        vec,
        geomid,
        worldid,
      )
    else:
      return ray_geom(pos, mat, geom_size[worldid % geom_size.shape[0], geomid], pnt, vec, type)
  else:
    return -1.0, wp.vec3()


@wp.kernel(module="unique")
def _ray(
  # Model:
  ngeom: int,
  nmeshface: int,
  body_weldid: wp.array[int],
  geom_type: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_matid: wp.array2d[int],
  geom_group: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rgba: wp.array2d[wp.vec4],
  mesh_vertadr: wp.array[int],
  mesh_faceadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_face: wp.array[wp.vec3i],
  hfield_size: wp.array[wp.vec4],
  hfield_nrow: wp.array[int],
  hfield_ncol: wp.array[int],
  hfield_adr: wp.array[int],
  hfield_data: wp.array[float],
  mat_rgba: wp.array2d[wp.vec4],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  pnt: wp.array2d[wp.vec3],
  vec: wp.array2d[wp.vec3],
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array[int],
  # Out:
  dist_out: wp.array2d[float],
  geomid_out: wp.array2d[int],
  normal_out: wp.array2d[wp.vec3],
):
  worldid, rayid, tid = wp.tid()

  num_threads = wp.block_dim()

  ray_origin = pnt[worldid % pnt.shape[0], rayid]
  ray_dir = vec[worldid % vec.shape[0], rayid]

  min_dist = float(MJ_MAXVAL)
  min_geomid = int(-1)
  min_normal = wp.vec3()

  upper = ((ngeom + num_threads - 1) // num_threads) * num_threads
  for geomid in range(tid, upper, num_threads):
    if geomid < ngeom:
      dist, normal = _ray_geom_mesh(
        nmeshface,
        body_weldid,
        geom_type,
        geom_bodyid,
        geom_dataid,
        geom_matid,
        geom_group,
        geom_size,
        geom_rgba,
        mesh_vertadr,
        mesh_faceadr,
        mesh_vert,
        mesh_face,
        hfield_size,
        hfield_nrow,
        hfield_ncol,
        hfield_adr,
        hfield_data,
        mat_rgba,
        geom_xpos_in,
        geom_xmat_in,
        worldid,
        ray_origin,
        ray_dir,
        geomgroup,
        flg_static,
        bodyexclude[rayid],
        geomid,
      )
      if dist < 0:
        dist = MJ_MAXVAL
    else:
      dist = MJ_MAXVAL
      normal = wp.vec3()

    tile_dist = wp.tile(dist)
    local_min_geomid = wp.tile_argmin(tile_dist)
    local_min_dist = tile_dist[local_min_geomid[0]]

    tile_geomid = wp.tile(geomid)
    tile_normal = wp.tile(normal, preserve_type=True)

    if local_min_dist < min_dist:
      min_dist = local_min_dist
      min_geomid = tile_geomid[local_min_geomid[0]]
      min_normal = tile_normal[local_min_geomid[0]]

  if min_dist >= MJ_MAXVAL:
    dist_out[worldid, rayid] = -1.0
  else:
    dist_out[worldid, rayid] = min_dist
  geomid_out[worldid, rayid] = min_geomid
  normal_out[worldid, rayid] = min_normal


@wp.func
def _ray_geom_mesh_bvh(
  # Model:
  body_weldid: wp.array[int],
  geom_type: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_matid: wp.array2d[int],
  geom_group: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rgba: wp.array2d[wp.vec4],
  mat_rgba: wp.array2d[wp.vec4],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: int,
  geomid: int,
  mesh_bvh_id: wp.array[wp.uint64],
  hfield_bvh_id: wp.array[wp.uint64],
  min_dist: float,
) -> Tuple[float, wp.vec3]:
  if not _ray_eliminate(
    body_weldid,
    geom_bodyid,
    geom_matid[worldid % geom_matid.shape[0]],
    geom_group,
    geom_rgba[worldid % geom_rgba.shape[0]],
    mat_rgba[worldid % mat_rgba.shape[0]],
    geomid,
    geomgroup,
    flg_static,
    bodyexclude,
  ):
    pos = geom_xpos_in[worldid, geomid]
    mat = geom_xmat_in[worldid, geomid]
    gtype = geom_type[geomid]

    if gtype == GeomType.MESH or gtype == GeomType.HFIELD:
      bvh_ids = mesh_bvh_id if gtype == GeomType.MESH else hfield_bvh_id
      # Public ray API (mjw.ray / mjw.rays) preserves MuJoCo's mj_ray cull rule:
      # rangefinder sensors and user-facing ray casts always cull back-faces.
      t, n, u, v, f, geom_mesh_id = ray_mesh_with_bvh(
        bvh_ids,
        geom_dataid[worldid % geom_dataid.shape[0], geomid],
        pos,
        mat,
        pnt,
        vec,
        min_dist,
        True,
      )
      if t >= 0.0 and t < min_dist:
        return t, n
    else:
      return ray_geom(
        pos,
        mat,
        geom_size[worldid % geom_size.shape[0], geomid],
        pnt,
        vec,
        gtype,
      )

  return -1.0, wp.vec3()


@wp.kernel
def _ray_bvh(
  # Model:
  ngeom: int,
  body_weldid: wp.array[int],
  geom_type: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_matid: wp.array2d[int],
  geom_group: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rgba: wp.array2d[wp.vec4],
  mat_rgba: wp.array2d[wp.vec4],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  pnt: wp.array2d[wp.vec3],
  vec: wp.array2d[wp.vec3],
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array[int],
  bvh_id: wp.uint64,
  group_root: wp.array[int],
  enabled_geom_ids: wp.array[int],
  mesh_bvh_id: wp.array[wp.uint64],
  hfield_bvh_id: wp.array[wp.uint64],
  # Out:
  dist_out: wp.array2d[float],
  geomid_out: wp.array2d[int],
  normal_out: wp.array2d[wp.vec3],
):
  worldid, rayid = wp.tid()

  ray_origin = pnt[worldid % pnt.shape[0], rayid]
  ray_dir = vec[worldid % vec.shape[0], rayid]
  body_exclude = bodyexclude[rayid]

  min_dist = float(MJ_MAXVAL)
  min_geomid = int(-1)
  min_normal = wp.vec3()

  query = wp.bvh_query_ray(bvh_id, ray_origin, ray_dir, group_root[worldid])
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, min_dist):
    bvh_local = bounds_nr - (worldid * ngeom)
    geomid = enabled_geom_ids[bvh_local]

    dist, normal = _ray_geom_mesh_bvh(
      body_weldid,
      geom_type,
      geom_bodyid,
      geom_dataid,
      geom_matid,
      geom_group,
      geom_size,
      geom_rgba,
      mat_rgba,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      ray_origin,
      ray_dir,
      geomgroup,
      flg_static,
      body_exclude,
      geomid,
      mesh_bvh_id,
      hfield_bvh_id,
      min_dist,
    )

    if dist >= 0.0 and dist < min_dist:
      min_dist = dist
      min_geomid = geomid
      min_normal = normal

  if min_dist >= MJ_MAXVAL:
    dist_out[worldid, rayid] = -1.0
  else:
    dist_out[worldid, rayid] = min_dist
  geomid_out[worldid, rayid] = min_geomid
  normal_out[worldid, rayid] = min_normal


def ray(
  m: Model,
  d: Data,
  pnt: wp.array2d[wp.vec3],
  vec: wp.array2d[wp.vec3],
  geomgroup: vec6 | None = None,
  flg_static: bool = True,
  bodyexclude: int = -1,
  rc: RenderContext | None = None,
) -> Tuple[wp.array, wp.array, wp.array]:
  """Returns the distance at which rays intersect with primitive geoms.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    pnt: Ray origin points, shape (1, 1) or (nworld, 1).
    vec: Ray directions, shape (1, 1) or (nworld, 1).
    geomgroup: Group inclusion/exclusion mask.
    flg_static: If True, allows rays to intersect with static geoms.
    bodyexclude: Ignore geoms on specified body id (-1 to disable).
    rc: Optional Render context containing BVH information for BVH accelerated ray
      intersections.

  Returns:
    Distances from ray origins to geom surfaces, IDs of intersected geoms (-1 if none),
    and normals at intersection points.
  """
  assert pnt.shape[0] in (1, d.nworld), f"pnt.shape[0] must be 1 or d.nworld ({d.nworld}), got {pnt.shape[0]}"
  assert pnt.shape == vec.shape, f"pnt.shape {pnt.shape} != vec.shape {vec.shape}"
  assert pnt.shape[1] == 1, (
    f"ray() only supports a single ray per world (shape (*, 1)), got {pnt.shape}; use rays() for multiple rays"
  )

  if geomgroup is None:
    geomgroup = vec6(-1, -1, -1, -1, -1, -1)

  ray_bodyexclude = wp.empty(1, dtype=int)
  ray_bodyexclude.fill_(bodyexclude)
  ray_dist = wp.empty((d.nworld, 1), dtype=float)
  ray_geomid = wp.empty((d.nworld, 1), dtype=int)
  ray_normal = wp.empty((d.nworld, 1), dtype=wp.vec3)

  rays(m, d, pnt, vec, geomgroup, flg_static, ray_bodyexclude, ray_dist, ray_geomid, ray_normal, rc)

  return ray_dist, ray_geomid, ray_normal


def rays(
  m: Model,
  d: Data,
  pnt: wp.array2d[wp.vec3],
  vec: wp.array2d[wp.vec3],
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array[int],
  dist: wp.array2d[float],
  geomid: wp.array2d[int],
  normal: wp.array2d[wp.vec3],
  rc: RenderContext | None = None,
):
  """Ray intersection for multiple worlds and multiple rays.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    pnt: Ray origin points, shape (1, nray) or (nworld, nray).
    vec: Ray directions, shape (1, nray) or (nworld, nray).
    geomgroup: Group inclusion/exclusion mask. Set all elements to -1 to ignore.
    flg_static: If True, allows rays to intersect with static geoms.
    bodyexclude: Per-ray body exclusion array of shape (nray,). Geoms on the
      specified body ids are ignored (-1 to disable for that ray).
    dist: Output array for distances from ray origins to geom surfaces, shape
      (nworld, nray). -1 indicates no intersection.
    geomid: Output array for IDs of intersected geoms, shape (nworld, nray). -1
      indicates no intersection.
    normal: Output array for normals at intersection points, shape (nworld, nray).
    rc: Optional Render context containing BVH information for BVH accelerated ray
      intersections.
  """
  assert pnt.shape[0] in (1, d.nworld), f"pnt.shape[0] must be 1 or d.nworld ({d.nworld}), got {pnt.shape[0]}"
  assert pnt.shape == vec.shape, f"pnt.shape {pnt.shape} != vec.shape {vec.shape}"
  assert bodyexclude.shape[0] == pnt.shape[1], f"bodyexclude length {bodyexclude.shape[0]} != nray {pnt.shape[1]}"
  assert dist.shape == (d.nworld, pnt.shape[1]), f"dist shape {dist.shape} != {(d.nworld, pnt.shape[1])}"
  assert geomid.shape == (d.nworld, pnt.shape[1]), f"geomid shape {geomid.shape} != {(d.nworld, pnt.shape[1])}"
  assert normal.shape == (d.nworld, pnt.shape[1]), f"normal shape {normal.shape} != {(d.nworld, pnt.shape[1])}"
  # TODO: Investigate building rc if none and removing the non-accelerated path
  if rc is None:
    wp.launch_tiled(
      _ray,
      dim=(d.nworld, pnt.shape[1]),
      inputs=[
        m.ngeom,
        m.nmeshface,
        m.body_weldid,
        m.geom_type,
        m.geom_bodyid,
        m.geom_dataid,
        m.geom_matid,
        m.geom_group,
        m.geom_size,
        m.geom_rgba,
        m.mesh_vertadr,
        m.mesh_faceadr,
        m.mesh_vert,
        m.mesh_face,
        m.hfield_size,
        m.hfield_nrow,
        m.hfield_ncol,
        m.hfield_adr,
        m.hfield_data,
        m.mat_rgba,
        d.geom_xpos,
        d.geom_xmat,
        pnt,
        vec,
        geomgroup,
        flg_static,
        bodyexclude,
        dist,
        geomid,
        normal,
      ],
      block_dim=m.block_dim.ray,
    )
  else:
    wp.launch(
      _ray_bvh,
      dim=(d.nworld, pnt.shape[1]),
      inputs=[
        rc.bvh_ngeom,
        m.body_weldid,
        m.geom_type,
        m.geom_bodyid,
        m.geom_dataid,
        m.geom_matid,
        m.geom_group,
        m.geom_size,
        m.geom_rgba,
        m.mat_rgba,
        d.geom_xpos,
        d.geom_xmat,
        pnt,
        vec,
        geomgroup,
        flg_static,
        bodyexclude,
        rc.bvh_id,
        rc.group_root,
        rc.enabled_geom_ids,
        rc.mesh_bvh_id,
        rc.hfield_bvh_id,
      ],
      outputs=[dist, geomid, normal],
    )
