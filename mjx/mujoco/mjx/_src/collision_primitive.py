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
"""Collision primitives."""

from typing import Tuple

import jax
from jax import numpy as jp
from mujoco.mjx._src import math
# pylint: disable=g-importing-member
from mujoco.mjx._src.collision_types import Collision
from mujoco.mjx._src.collision_types import GeomInfo
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member


def collider(ncon: int):
  """Wraps collision functions for use by collision_driver."""

  def wrapper(func):
    def collide(m: Model, d: Data, _, geom: jax.Array) -> Collision:
      g1, g2 = geom.T
      info1 = GeomInfo(d.geom_xpos[g1], d.geom_xmat[g1], m.geom_size[g1])
      info2 = GeomInfo(d.geom_xpos[g2], d.geom_xmat[g2], m.geom_size[g2])
      dist, pos, frame = jax.vmap(func)(info1, info2)
      if ncon > 1:
        return jax.tree_util.tree_map(jp.concatenate, (dist, pos, frame))
      return dist, pos, frame

    collide.ncon = ncon  # pyrefly: ignore[missing-attribute]
    return collide

  return wrapper


def _plane_sphere(
    plane_normal: jax.Array,
    plane_pos: jax.Array,
    sphere_pos: jax.Array,
    sphere_radius: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Returns the distance and contact point between a plane and sphere."""
  dist = jp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@collider(ncon=1)
def plane_sphere(plane: GeomInfo, sphere: GeomInfo) -> Collision:
  """Calculates contact between a plane and a sphere."""
  n = plane.mat[:, 2]
  dist, pos = _plane_sphere(n, plane.pos, sphere.pos, sphere.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=2)
def plane_capsule(plane: GeomInfo, cap: GeomInfo) -> Collision:
  """Calculates two contacts between a capsule and a plane."""
  n, axis = plane.mat[:, 2], cap.mat[:, 2]
  # align contact frames with capsule axis projected onto the plane, as in C;
  # below sqrt(eps) the projection is rounding error, so use make_frame's
  b, b_norm = math.normalize_with_norm(axis - n * jp.dot(n, axis))
  degenerate = b_norm < jp.sqrt(jp.finfo(b_norm.dtype).eps)
  b = jp.where(degenerate, math.orthogonals(n)[0], b)
  frame = jp.array([[n, b, jp.cross(n, b)]])
  segment = axis * cap.size[1]
  collisions = []
  for offset in [segment, -segment]:
    dist, pos = _plane_sphere(n, plane.pos, cap.pos + offset, cap.size[0])
    dist = jp.expand_dims(dist, axis=0)
    pos = jp.expand_dims(pos, axis=0)
    collisions.append((dist, pos, frame))
  return jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *collisions)


@collider(ncon=1)
def plane_ellipsoid(plane: GeomInfo, ellipsoid: GeomInfo) -> Collision:
  """Calculates one contact between an ellipsoid and a plane."""
  n = plane.mat[:, 2]
  size = ellipsoid.size
  sphere_support = -math.normalize((ellipsoid.mat.T @ n) * size)
  pos = ellipsoid.pos + ellipsoid.mat @ (sphere_support * size)
  dist = jp.dot(n, pos - plane.pos)
  pos = pos - n * dist * 0.5
  return dist, pos, math.make_frame(n)


@collider(ncon=3)
def plane_cylinder(plane: GeomInfo, cylinder: GeomInfo) -> Collision:
  """Calculates three contacts between a cylinder and a plane."""
  n = plane.mat[:, 2]
  axis = cylinder.mat[:, 2]

  # make sure axis points towards plane
  prjaxis = jp.dot(n, axis)
  sign = -math.sign(prjaxis)
  axis, prjaxis = axis * sign, prjaxis * sign

  # compute normal distance to cylinder center
  dist0 = jp.dot(cylinder.pos - plane.pos, n)

  # remove component of -normal along axis, compute length
  vec = axis * prjaxis - n
  len_ = math.norm(vec)

  vec = jp.where(
      len_ < 1e-12,
      # disk parallel to plane: pick x-axis of cylinder, scale by radius
      cylinder.mat[:, 0] * cylinder.size[0],
      # general configuration: normalize vector, scale by radius
      math.safe_div(vec, len_) * cylinder.size[0],
  )

  # project vector on normal
  prjvec = jp.dot(vec, n)

  # scale axis by half-length
  axis *= cylinder.size[1]
  prjaxis *= cylinder.size[1]

  # compute sideways vector: vec1
  prjvec1 = -prjvec * 0.5
  vec1 = math.normalize(jp.cross(vec, axis)) * cylinder.size[0]
  vec1 *= jp.sqrt(3.0) * 0.5

  # disk parallel to plane
  d1 = dist0 + prjaxis + prjvec
  d2 = dist0 + prjaxis + prjvec1
  dist = jp.array([d1, d2, d2])
  pos = (
      cylinder.pos
      + axis
      + jp.array([
          vec - n * d1 * 0.5,
          vec1 + vec * -0.5 - n * d2 * 0.5,
          -vec1 + vec * -0.5 - n * d2 * 0.5,
      ])
  )

  # cylinder parallel to plane
  cond = jp.abs(prjaxis) < 1e-3
  d3 = dist0 - prjaxis + prjvec
  dist = jp.where(cond, dist.at[1].set(d3), dist)
  pos = jp.where(
      cond, pos.at[1].set(cylinder.pos + vec - axis - n * d3 * 0.5), pos
  )

  frame = jp.stack([math.make_frame(n)] * 3, axis=0)
  return dist, pos, frame


def _sphere_sphere(
    pos1: jax.Array, radius1: jax.Array, pos2: jax.Array, radius2: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Returns the penetration, contact point, and normal between two spheres."""
  n, dist = math.normalize_with_norm(pos2 - pos1)
  n = jp.where(dist == 0.0, jp.array([1.0, 0.0, 0.0]), n)
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + dist * 0.5)
  return dist, pos, n


@collider(ncon=1)
def sphere_sphere(s1: GeomInfo, s2: GeomInfo) -> Collision:
  """Calculates contact between two spheres."""
  dist, pos, n = _sphere_sphere(s1.pos, s1.size[0], s2.pos, s2.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=1)
def sphere_capsule(sphere: GeomInfo, cap: GeomInfo) -> Collision:
  """Calculates one contact between a sphere and a capsule."""
  axis, length = cap.mat[:, 2], cap.size[1]
  segment = axis * length
  pt = math.closest_segment_point(
      cap.pos - segment, cap.pos + segment, sphere.pos
  )
  dist, pos, n = _sphere_sphere(sphere.pos, sphere.size[0], pt, cap.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=1)
def capsule_capsule(cap1: GeomInfo, cap2: GeomInfo) -> Collision:
  """Calculates one contact between two capsules."""
  axis1, length1 = cap1.mat[:, 2], cap1.size[1]
  axis2, length2 = cap2.mat[:, 2], cap2.size[1]
  seg1, seg2 = axis1 * length1, axis2 * length2
  pt1, pt2 = math.closest_segment_to_segment_points(
      cap1.pos - seg1,
      cap1.pos + seg1,
      cap2.pos - seg2,
      cap2.pos + seg2,
  )
  radius1, radius2 = cap1.size[0], cap2.size[0]
  dist, pos, n = _sphere_sphere(pt1, radius1, pt2, radius2)
  return dist, pos, math.make_frame(n)
