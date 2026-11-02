# Copyright 2023 DeepMind Technologies Limited
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
"""Functions for ray interesection testing."""

from typing import Sequence, Tuple

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import math
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import GeomType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np


def _ray_quad(
    a: jax.Array, b: jax.Array, c: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Returns two solutions for quadratic: a*x^2 + 2*b*x + c = 0."""
  det = b * b - a * c
  det_2 = jp.sqrt(jp.maximum(det, 0))

  x0, x1 = math.safe_div(-b - det_2, a), math.safe_div(-b + det_2, a)
  x0 = jp.where((det < mujoco.mjMINVAL) | (x0 < 0), jp.inf, x0)
  x1 = jp.where((det < mujoco.mjMINVAL) | (x1 < 0), jp.inf, x1)

  return x0, x1


def _ray_plane(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a plane."""
  x = -math.safe_div(pnt[2], vec[2])

  valid = vec[2] <= -mujoco.mjMINVAL  # z-vec pointing towards front face
  valid &= x >= 0
  # only within rendered rectangle
  p = pnt[0:2] + x * vec[0:2]
  valid &= jp.all((size[0:2] <= 0) | (jp.abs(p) <= size[0:2]))

  return jp.where(valid, x, jp.inf)


def _ray_sphere(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a sphere."""
  x0, x1 = _ray_quad(vec @ vec, vec @ pnt, pnt @ pnt - size[0] * size[0])
  x = jp.where(jp.isinf(x0), x1, x0)

  return x


def _ray_capsule(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a capsule."""

  # cylinder round side: (x*lvec+lpnt)'*(x*lvec+lpnt) = size[0]*size[0]
  a = vec[0:2] @ vec[0:2]
  b = vec[0:2] @ pnt[0:2]
  c = pnt[0:2] @ pnt[0:2] - size[0] * size[0]

  # solve a*x^2 + 2*b*x + c = 0
  x0, x1 = _ray_quad(a, b, c)
  x = jp.where(jp.isinf(x0), x1, x0)

  # make sure round solution is between flat sides
  x = jp.where(jp.abs(pnt[2] + x * vec[2]) <= size[1], x, jp.inf)

  # top cap
  dif = pnt - jp.array([0, 0, size[1]])
  x0, x1 = _ray_quad(vec @ vec, vec @ dif, dif @ dif - size[0] * size[0])
  # accept only top half of sphere
  x = jp.where((pnt[2] + x0 * vec[2] >= size[1]) & (x0 < x), x0, x)
  x = jp.where((pnt[2] + x1 * vec[2] >= size[1]) & (x1 < x), x1, x)

  # bottom cap
  dif = pnt + jp.array([0, 0, size[1]])
  x0, x1 = _ray_quad(vec @ vec, vec @ dif, dif @ dif - size[0] * size[0])

  # accept only bottom half of sphere
  x = jp.where((pnt[2] + x0 * vec[2] <= -size[1]) & (x0 < x), x0, x)
  x = jp.where((pnt[2] + x1 * vec[2] <= -size[1]) & (x1 < x), x1, x)

  return x


def _ray_ellipsoid(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with an ellipsoid."""

  # invert size^2
  s = math.safe_div(1, jp.square(size))

  # (x*lvec+lpnt)' * diag(1/size^2) * (x*lvec+lpnt) = 1
  svec = s * vec
  a = svec @ vec
  b = svec @ pnt
  c = (s * pnt) @ pnt - 1

  # solve a*x^2 + 2*b*x + c = 0
  x0, x1 = _ray_quad(a, b, c)
  x = jp.where(jp.isinf(x0), x1, x0)

  return x


def _ray_box(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a box."""

  iface = jp.array([(1, 2), (0, 2), (0, 1), (1, 2), (0, 2), (0, 1)])

  # side +1, -1
  # solution of pnt[i] + x * vec[i] = side * size[i]
  x = jp.concatenate([math.safe_div(size - pnt,  vec), -math.safe_div(size + pnt, vec)])

  # intersection with face
  p0 = pnt[iface[:, 0]] + x * vec[iface[:, 0]]
  p1 = pnt[iface[:, 1]] + x * vec[iface[:, 1]]
  valid = jp.abs(p0) <= size[iface[:, 0]]
  valid &= jp.abs(p1) <= size[iface[:, 1]]
  valid &= x >= 0

  return jp.min(jp.where(valid, x, jp.inf))


def _ray_triangle(
    vert: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
    basis: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a triangle."""
  # project difference vectors in ray normal plane
  planar = jp.dot(vert - pnt, basis)

  # determine if origin is inside planar projection of triangle
  # A = (p0-p2, p1-p2), b = -p2, solve A*t = b
  A = planar[0:2] - planar[2]  # pylint: disable=invalid-name
  b = -planar[2]
  det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]

  t0 = math.safe_div(A[1, 1] * b[0] - A[1, 0] * b[1], det)
  t1 = math.safe_div(-A[0, 1] * b[0] + A[0, 0] * b[1], det)
  valid = (t0 >= 0) & (t1 >= 0) & (t0 + t1 <= 1)

  # intersect ray with plane of triangle
  nrm = jp.cross(vert[0] - vert[2], vert[1] - vert[2])
  dist = math.safe_div(jp.dot(vert[2] - pnt, nrm), jp.dot(vec, nrm))
  valid &= (dist >= 0)
  dist = jp.where(valid, dist, jp.inf)

  return dist


def _ray_mesh(
    m: Model,
    geom_id: np.ndarray,
    unused_size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Returns the best distance and geom_id for ray mesh intersections."""
  data_id = m.geom_dataid[geom_id]

  ray_basis = lambda x: jp.array(math.orthogonals(math.normalize(x))).T
  basis = jax.vmap(ray_basis)(vec)

  faceadr = np.append(m.mesh_faceadr, m.nmeshface)
  vertadr = np.append(m.mesh_vertadr, m.nmeshvert)

  dists, geom_ids = [], []
  for i, id_ in enumerate(data_id):
    face = m.mesh_face[faceadr[id_] : faceadr[id_ + 1]]
    vert = m.mesh_vert[vertadr[id_] : vertadr[id_ + 1]]
    vert = jp.array(vert[face])
    dist = jax.vmap(_ray_triangle, in_axes=(0, None, None, None))(
        vert, pnt[i], vec[i], basis[i]
    )
    dists.append(jp.min(dist))
    geom_ids.append(geom_id[i])

  dists = jp.array(dists)
  min_id = jp.argmin(dists)
  # Grab the best distance amongst all meshes, bypassing the argmin in `ray`.
  # This avoids having to compute the best distance per mesh.
  dist = dists[min_id, None]
  geom_ids_array = jp.array(geom_ids)
  id_ = geom_ids_array[min_id, None]

  return dist, id_


_RAY_FUNC = {
    GeomType.PLANE: _ray_plane,
    GeomType.SPHERE: _ray_sphere,
    GeomType.CAPSULE: _ray_capsule,
    GeomType.ELLIPSOID: _ray_ellipsoid,
    GeomType.BOX: _ray_box,
    GeomType.MESH: _ray_mesh,
}


def ray(
    m: Model,
    d: Data,
    pnt: jax.Array,
    vec: jax.Array,
    geomgroup: Sequence[int] = (),
    flg_static: bool = True,
    bodyexclude: Sequence[int] | int = -1,
) -> Tuple[jax.Array, jax.Array]:
  """Returns the distance and geom id at which a ray intersects with a geom.

  Args:
    m: MJX model
    d: MJX data
    pnt: ray origin point (3,)
    vec: ray direction    (3,)
    geomgroup: group inclusion/exclusion mask, or empty to ignore
    flg_static: if True, allows rays to intersect with static geoms
    bodyexclude: ignore geoms on specified body id or sequence of body ids

  Returns:
    Distance from ray origin to geom surface (or -1.0 for no intersection) and
    id of intersected geom (or -1 for no intersection)
  """

  dists, ids = [], []
  if not isinstance(bodyexclude, Sequence):
    bodyexclude = [bodyexclude]
  geom_filter = flg_static | (m.body_weldid[m.geom_bodyid] != 0)
  # Loop through the body IDs to exclude and update the filter
  for bodyid in bodyexclude:
    geom_filter &= (m.geom_bodyid != bodyid)
  if geomgroup:
    geomgroup = np.array(geomgroup, dtype=bool)  # pyrefly: ignore[bad-assignment]
    geom_filter &= geomgroup[np.clip(m.geom_group, 0, mujoco.mjNGROUP)]  # pyrefly: ignore[bad-index]

  # map ray to local geom frames
  geom_pnts = jax.vmap(lambda x, y: x.T @ (pnt - y))(d.geom_xmat, d.geom_xpos)
  geom_vecs = jax.vmap(lambda x: x.T @ vec)(d.geom_xmat)

  geom_filter_dyn = (m.geom_matid != -1) | (m.geom_rgba[:, 3] != 0)
  geom_filter_dyn &= (m.geom_matid == -1) | (m.mat_rgba[m.geom_matid, 3] != 0)
  for geom_type, fn in _RAY_FUNC.items():
    (id_,) = np.nonzero(geom_filter & (m.geom_type == geom_type))

    if id_.size == 0:
      continue

    args = m.geom_size[id_], geom_pnts[id_], geom_vecs[id_]

    if geom_type == GeomType.MESH:
      dist, id_ = fn(m, id_, *args)  # pyrefly: ignore[bad-argument-count, bad-argument-type]
    else:
      dist = jax.vmap(fn)(*args)  # pyrefly: ignore[bad-argument-type, missing-argument]

    dist = jp.where(geom_filter_dyn[id_], dist, jp.inf)  # pyrefly: ignore[bad-argument-type]
    dists, ids = dists + [dist], ids + [id_]

  if not ids:
    return jp.array(-1.0), jp.array(-1)

  dists = jp.concatenate(dists)
  ids = jp.concatenate(ids)
  min_id = jp.argmin(dists)
  dist = jp.where(jp.isinf(dists[min_id]), -1, dists[min_id])
  id_ = jp.where(jp.isinf(dists[min_id]), -1, ids[min_id])

  return dist, id_


def ray_geom(
    size: jax.Array, pnt: jax.Array, vec: jax.Array, geomtype: GeomType
) -> jax.Array:
  """Returns the distance at which a ray intersects with a primitive geom.

  Args:
    size: geom size (1,), (2,), or (3,)
    pnt: ray origin point (3,)
    vec: ray direction    (3,)
    geomtype: type of geom

  Returns:
    dist: distance from ray origin to geom surface
  """
  return _RAY_FUNC[geomtype](size, pnt, vec)  # pyrefly: ignore[bad-argument-type, bad-return, missing-argument]
