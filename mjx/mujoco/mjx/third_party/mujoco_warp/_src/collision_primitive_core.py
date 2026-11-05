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

from mujoco.mjx.third_party.mujoco_warp._src.math import closest_segment_to_segment_points
from mujoco.mjx.third_party.mujoco_warp._src.math import normalize_with_norm
from mujoco.mjx.third_party.mujoco_warp._src.math import safe_div
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import mat53
from mujoco.mjx.third_party.mujoco_warp._src.types import mat113
from mujoco.mjx.third_party.mujoco_warp._src.types import vec5
from mujoco.mjx.third_party.mujoco_warp._src.types import vec11

wp.set_module_options({"enable_backward": False})


class vec8f(wp.types.vector(length=8, dtype=wp.float32)):
  pass


class mat23f(wp.types.matrix(shape=(2, 3), dtype=wp.float32)):
  pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
  pass


class mat83f(wp.types.matrix(shape=(8, 3), dtype=wp.float32)):
  pass


# core
@wp.func
def plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float) -> Tuple[float, wp.vec3]:
  # TODO(team): docstring
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def sphere_sphere(
  # In:
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Sphere-sphere collision calculation.

  Args:
    pos1: Center position of the first sphere.
    radius1: Radius of the first sphere.
    pos2: Center position of the second sphere.
    radius2: Radius of the second sphere.

  Returns:
    - Distance between sphere surfaces (negative if overlapping).
    - Contact position.
    - Contact normal vector.
  """
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)
  return dist, pos, n


@wp.func
def sphere_capsule(
  # In:
  sphere_pos: wp.vec3,
  sphere_radius: float,
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Core contact geometry calculation for sphere-capsule collision.

  Args:
    sphere_pos: Center position of the sphere.
    sphere_radius: Radius of the sphere.
    capsule_pos: Center position of the capsule.
    capsule_axis: Axis direction of the capsule.
    capsule_radius: Radius of the capsule.
    capsule_half_length: Half length of the capsule.

  Returns:
    - Vector of contact distances.
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  # Project the sphere center directly onto the capsule centerline.
  offset = sphere_pos - capsule_pos
  projection = wp.clamp(
    wp.dot(capsule_axis, offset),
    -capsule_half_length,
    capsule_half_length,
  )
  pt = capsule_pos + capsule_axis * projection

  # Use sphere-sphere collision between sphere and closest point
  return sphere_sphere(sphere_pos, sphere_radius, pt, capsule_radius)


@wp.func
def _clamp_segment_weights(s: float, t: float, s0: float, s1: float) -> wp.vec2:
  """Clamps t to [0, 1] and recomputes s on the first segment (s0 at t = 0, s1 at t = 1)."""
  if t < 0.0:
    return wp.vec2(wp.clamp(s0, 0.0, 1.0), 0.0)
  if t > 1.0:
    return wp.vec2(wp.clamp(s1, 0.0, 1.0), 1.0)
  return wp.vec2(s, t)


@wp.func
def _segment_weights(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> Tuple[int, wp.vec2, wp.vec2]:
  """Barycentric weights (s, t) of the closest points a + s (b - a) and c + t (d - c).

  Port of mjc_segWeights (MuJoCo engine_collision_primitive.c).

  Args:
    a: First endpoint of the first segment.
    b: Second endpoint of the first segment.
    c: First endpoint of the second segment.
    d: Second endpoint of the second segment.

  Returns:
    - Number of closest point pairs: 1, or 2 if the segments are parallel.
    - Weights (s, t) of the first pair.
    - Weights (s, t) of the second pair (parallel segments only).
  """
  ab = b - a
  cd = d - c
  ca = a - c

  dot1 = wp.dot(ab, ab)
  dot2 = wp.dot(cd, cd)
  dot3 = wp.dot(cd, ca)
  dot4 = wp.dot(ab, ca)
  dot5 = wp.dot(ab, cd)

  # degenerate segments (points)
  if dot1 <= MJ_MINVAL and dot2 <= MJ_MINVAL:
    return 1, wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0)
  if dot1 <= MJ_MINVAL:
    return 1, wp.vec2(0.0, wp.clamp(dot3 / dot2, 0.0, 1.0)), wp.vec2(0.0, 0.0)
  if dot2 <= MJ_MINVAL:
    return 1, wp.vec2(wp.clamp(-dot4 / dot1, 0.0, 1.0), 0.0), wp.vec2(0.0, 0.0)

  # projections of c (t = 0) and d (t = 1) onto ab
  s0 = -dot4 / dot1
  s1 = (dot5 - dot4) / dot1

  # determinant via Lagrange's identity to avoid cancellation for parallel segments
  cross = wp.cross(ab, cd)
  det = wp.dot(cross, cross)

  # non-parallel segments (sin^2 of the angle > 1e-8): s via Binet-Cauchy identity
  if det > 1.0e-8 * dot1 * dot2:
    s = wp.clamp(wp.dot(cross, wp.cross(cd, ca)) / det, 0.0, 1.0)
    return 1, _clamp_segment_weights(s, (dot5 * s + dot3) / dot2, s0, s1), wp.vec2(0.0, 0.0)

  # parallel segments: closest points at both ends of the overlap
  w0 = _clamp_segment_weights(0.0, dot3 / dot2, s0, s1)
  w1 = _clamp_segment_weights(1.0, (dot5 + dot3) / dot2, s0, s1)
  return 2, w0, w1


@wp.func
def capsule_capsule(
  # In:
  cap1_pos: wp.vec3,
  cap1_axis: wp.vec3,
  cap1_radius: float,
  cap1_half_length: float,
  cap2_pos: wp.vec3,
  cap2_axis: wp.vec3,
  cap2_radius: float,
  cap2_half_length: float,
) -> Tuple[wp.vec2, mat23f, mat23f]:
  """Core contact geometry calculation for capsule-capsule collision.

  Args:
    cap1_pos: Center position of the first capsule.
    cap1_axis: Axis direction of the first capsule.
    cap1_radius: Radius of the first capsule.
    cap1_half_length: Half length of the first capsule.
    cap2_pos: Center position of the second capsule.
    cap2_axis: Axis direction of the second capsule.
    cap2_radius: Radius of the second capsule.
    cap2_half_length: Half length of the second capsule.

  Returns:
    - Vector of contact distances (wp.inf for invalid contacts).
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  contact_dist = wp.vec2(wp.inf, wp.inf)
  contact_pos = mat23f()
  contact_normal = mat23f()

  # segment endpoints
  axis1 = cap1_axis * cap1_half_length
  axis2 = cap2_axis * cap2_half_length
  p1 = cap1_pos - axis1
  p2 = cap1_pos + axis1
  q1 = cap2_pos - axis2
  q2 = cap2_pos + axis2

  # closest point pairs (2 for parallel axes), sphere-sphere test for each
  n, w0, w1 = _segment_weights(p1, p2, q1, q2)

  dist, pos, normal = sphere_sphere(p1 + w0[0] * (p2 - p1), cap1_radius, q1 + w0[1] * (q2 - q1), cap2_radius)
  contact_dist[0] = dist
  contact_pos[0] = pos
  contact_normal[0] = normal

  if n == 2:
    dist, pos, normal = sphere_sphere(p1 + w1[0] * (p2 - p1), cap1_radius, q1 + w1[1] * (q2 - q1), cap2_radius)
    contact_dist[1] = dist
    contact_pos[1] = pos
    contact_normal[1] = normal

  return contact_dist, contact_pos, contact_normal


@wp.func
def plane_capsule(
  # In:
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
) -> Tuple[wp.vec2, mat23f, wp.mat33]:
  """Core contact geometry calculation for plane-capsule collision.

  Args:
    plane_normal: Normal vector of the plane.
    plane_pos: Position point on the plane.
    capsule_pos: Center position of the capsule.
    capsule_axis: Axis direction of the capsule.
    capsule_radius: Radius of the capsule.
    capsule_half_length: Half length of the capsule.

  Returns:
    - Vector of contact distances.
    - Matrix of contact positions (one per row).
    - Contact frame for both contacts.
  """
  n = plane_normal
  axis = capsule_axis

  # align contact frames with capsule axis
  b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

  if b_norm < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  c = wp.cross(n, b)
  frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])
  segment = axis * capsule_half_length

  # First contact (positive end of capsule)
  dist1, pos1 = plane_sphere(n, plane_pos, capsule_pos + segment, capsule_radius)

  # Second contact (negative end of capsule)
  dist2, pos2 = plane_sphere(n, plane_pos, capsule_pos - segment, capsule_radius)

  dist = wp.vec2(dist1, dist2)
  pos = mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])

  return dist, pos, frame


@wp.func
def plane_ellipsoid(
  # In:
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  ellipsoid_pos: wp.vec3,
  ellipsoid_rot: wp.mat33,
  ellipsoid_size: wp.vec3,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Core contact geometry calculation for plane-ellipsoid collision.

  Args:
    plane_normal: Normal vector of the plane.
    plane_pos: Position point on the plane.
    ellipsoid_pos: Center position of the ellipsoid.
    ellipsoid_rot: Rotation matrix of the ellipsoid.
    ellipsoid_size: Size (radii) of the ellipsoid along each axis.

  Returns:
    - Vector of contact distances.
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  sphere_support = -wp.normalize(wp.cw_mul(wp.transpose(ellipsoid_rot) @ plane_normal, ellipsoid_size))
  pos = ellipsoid_pos + ellipsoid_rot @ wp.cw_mul(sphere_support, ellipsoid_size)
  dist = wp.dot(plane_normal, pos - plane_pos)
  pos = pos - plane_normal * dist * 0.5

  return dist, pos, plane_normal


@wp.func
def plane_box(
  # In:
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
) -> Tuple[vec8f, mat83f, wp.vec3]:
  """Core contact geometry calculation for plane-box collision.

  Args:
    plane_normal: Normal vector of the plane.
    plane_pos: Position point on the plane.
    box_pos: Center position of the box.
    box_rot: Rotation matrix of the box.
    box_size: Half-extents of the box along each axis.
    margin: Collision tolerance.

  Returns:
    - Vector of contact distances (MJ_MAXVAL for unpopulated contacts).
    - Matrix of contact positions (one per row).
    - Contact normal vector.
  """
  center_dist = wp.dot(box_pos - plane_pos, plane_normal)

  dist = vec8f(MJ_MAXVAL)
  pos = mat83f()

  # test all corners, pick bottom 4
  for i in range(8):
    # get corner in local coordinates
    corner = wp.vec3(
      wp.where(i & 1, box_size[0], -box_size[0]),
      wp.where(i & 2, box_size[1], -box_size[1]),
      wp.where(i & 4, box_size[2], -box_size[2]),
    )

    # get corner in global coordinates relative to box center
    corner = box_rot * corner

    # compute distance to plane
    ldist = wp.dot(plane_normal, corner)
    cdist = center_dist + ldist

    dist[i] = cdist
    pos[i] = corner + box_pos - 0.5 * plane_normal * cdist

  return dist, pos, plane_normal


@wp.func
def sphere_cylinder(
  # In:
  sphere_pos: wp.vec3,
  sphere_radius: float,
  cylinder_pos: wp.vec3,
  cylinder_axis: wp.vec3,
  cylinder_radius: float,
  cylinder_half_height: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Core contact geometry calculation for sphere-cylinder collision.

  Args:
    sphere_pos: Center position of the sphere.
    sphere_radius: Radius of the sphere.
    cylinder_pos: Center position of the cylinder.
    cylinder_axis: Axis direction of the cylinder.
    cylinder_radius: Radius of the cylinder.
    cylinder_half_height: Half height of the cylinder.

  Returns:
    - Vector of contact distances.
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  vec = sphere_pos - cylinder_pos
  x = wp.dot(vec, cylinder_axis)

  a_proj = cylinder_axis * x
  p_proj = vec - a_proj
  p_proj_sqr = wp.dot(p_proj, p_proj)

  collide_side = wp.abs(x) < cylinder_half_height
  collide_cap = p_proj_sqr < (cylinder_radius * cylinder_radius)

  if collide_side and collide_cap:
    dist_cap = cylinder_half_height - wp.abs(x)
    dist_radius = cylinder_radius - wp.sqrt(p_proj_sqr)

    if dist_cap < dist_radius:
      collide_side = False
    else:
      collide_cap = False

  # side collision
  if collide_side:
    pos_target = cylinder_pos + a_proj
    return sphere_sphere(sphere_pos, sphere_radius, pos_target, cylinder_radius)
  # cap collision
  elif collide_cap:
    if x > 0.0:
      # top cap
      pos_cap = cylinder_pos + cylinder_axis * cylinder_half_height
      plane_normal = cylinder_axis
    else:
      # bottom cap
      pos_cap = cylinder_pos - cylinder_axis * cylinder_half_height
      plane_normal = -cylinder_axis

    dist, pos = plane_sphere(plane_normal, pos_cap, sphere_pos, sphere_radius)
    return dist, pos, -plane_normal  # flip normal after position calculation
  # corner collision
  else:
    inv_len = safe_div(1.0, wp.sqrt(p_proj_sqr))
    p_proj = p_proj * (cylinder_radius * inv_len)

    cap_offset = cylinder_axis * (wp.sign(x) * cylinder_half_height)
    pos_corner = cylinder_pos + cap_offset + p_proj

    return sphere_sphere(sphere_pos, sphere_radius, pos_corner, 0.0)


@wp.func
def plane_cylinder(
  # In:
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  cylinder_center: wp.vec3,
  cylinder_axis: wp.vec3,
  cylinder_radius: float,
  cylinder_half_height: float,
) -> Tuple[wp.vec4, mat43f, wp.vec3]:
  """Core contact geometry calculation for plane-cylinder collision.

  Args:
    plane_normal: Normal vector of the plane.
    plane_pos: Position point on the plane.
    cylinder_center: Center position of the cylinder.
    cylinder_axis: Axis direction of the cylinder.
    cylinder_radius: Radius of the cylinder.
    cylinder_half_height: Half height of the cylinder.

  Returns:
    - Vector of contact distances.
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  # Initialize output matrices
  contact_dist = wp.vec4(MJ_MAXVAL)
  contact_pos = mat43f()
  contact_count = 0

  n = plane_normal
  axis = cylinder_axis

  # Project, make sure axis points toward plane
  prjaxis = wp.dot(n, axis)
  if prjaxis > 0:
    axis = -axis
    prjaxis = -prjaxis

  # Compute normal distance from plane to cylinder center
  dist0 = wp.dot(cylinder_center - plane_pos, n)

  # Remove component of -normal along cylinder axis
  vec = axis * prjaxis - n
  len_sqr = wp.dot(vec, vec)

  # If vector is nondegenerate, normalize and scale by radius
  # Otherwise use cylinder's x-axis scaled by radius
  vec = wp.where(
    len_sqr >= 1e-12,
    vec * safe_div(cylinder_radius, wp.sqrt(len_sqr)),
    wp.vec3(1.0, 0.0, 0.0) * cylinder_radius,  # Default x-axis when degenerate
  )

  # Project scaled vector on normal
  prjvec = wp.dot(vec, n)

  # Scale cylinder axis by half-length
  axis = axis * cylinder_half_height
  prjaxis = prjaxis * cylinder_half_height

  # First contact point (end cap closer to plane)
  dist1 = dist0 + prjaxis + prjvec
  pos1 = cylinder_center + vec + axis - n * (dist1 * 0.5)
  contact_dist[contact_count] = dist1
  contact_pos[contact_count] = pos1
  contact_count = contact_count + 1

  # Second contact point (end cap farther from plane)
  dist2 = dist0 - prjaxis + prjvec
  pos2 = cylinder_center + vec - axis - n * (dist2 * 0.5)
  contact_dist[contact_count] = dist2
  contact_pos[contact_count] = pos2
  contact_count = contact_count + 1

  # Try triangle contact points on side closer to plane
  prjvec1 = -prjvec * 0.5
  dist3 = dist0 + prjaxis + prjvec1
  # Compute sideways vector scaled by radius*sqrt(3)/2
  vec1 = wp.cross(vec, axis)
  vec1 = wp.normalize(vec1) * (cylinder_radius * wp.sqrt(3.0) * 0.5)

  # Add contact point A - adjust to closest side
  pos3 = cylinder_center + vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
  contact_dist[contact_count] = dist3
  contact_pos[contact_count] = pos3
  contact_count = contact_count + 1

  # Add contact point B - adjust to closest side
  pos4 = cylinder_center - vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
  contact_dist[contact_count] = dist3
  contact_pos[contact_count] = pos4
  contact_count = contact_count + 1

  return contact_dist, contact_pos, n


# A box-box contact manifold is computed in two stages, following MuJoCo's mjc_BoxBox
# (engine_collision_box.c).
#
# Stage 1, separating-axis test: find the axis of maximum separation among the 15 candidate
# directions (3 face normals per box, 9 cross products of edge directions). If the boxes are
# separated by more than margin along any candidate axis there is no contact. Face axes are
# preferred over edge axes on near-ties: a face axis yields a multi-point manifold, which the
# solver strongly prefers over a single edge contact of nearly identical depth.
#
# Stage 2, manifold generation, depends on the kind of winning axis:
#  - face axis: the owner of the face is the reference box. The face of the other (incident)
#    box least aligned with the reference normal is clipped against the four side planes of
#    the reference face (Sutherland-Hodgman). Every clipped vertex within the margin band
#    becomes a contact. Depth is the distance between the surfaces along the reference
#    normal; contact position is midway between the surfaces along the normal, so it lies
#    inside the intersection of the margin-inflated boxes.
#  - edge axis: the contact is at the midpoint of the closest-point pair between the two
#    supporting edge segments, with depth measured along the separating axis.
#
# Every surviving clipped vertex becomes a contact, so a face manifold carries at most
# _BOXBOX_MAXVERT points.

# Rounding scales, in units of float32 epsilon (MuJoCo's mjUSESINGLE values). Supports are
# sums of products of box extents with rotation entries, so their absolute error is
# proportional to the extents: _BOXBOX_SEPEPS multiplies the summed half-sizes. The rest are
# dimensionless.
_BOXBOX_SEPEPS = 1e-6  # slack on the separation tests, times the box scale
_BOXBOX_PAREPS = 1e-7  # sin^2 below which an edge-cross axis is noise
_BOXBOX_SGNEPS = 1e-5  # axis component below which a support corner is ambiguous
_BOXBOX_DUPEPS = 1e-10  # squared relative radius for clip-vertex deduplication

# relative penalty applied to edge-axis separation on near-ties with the best face axis
_BOXBOX_EDGEBIAS = 1e-6

# vertex capacity for face clipping: clipping a convex polygon against a half-plane adds at
# most one vertex, so a 4-gon clipped by 4 half-planes has at most 8 vertices, each of which
# may become a contact; the capacity checks in _box_box_clip guard against rounding
_BOXBOX_MAXVERT = 8


@wp.func
def _box_box_clip(
  # In:
  nin: int,
  poly: mat83f,
  coord: int,
  sign: float,
  limit: float,
) -> Tuple[int, mat83f]:
  """Clips a polygon against the half-plane sign * v[coord] <= limit.

  Vertices are (x, y, z) with z interpolated as an attribute. When every vertex
  is already
  inside, the polygon is returned untouched.

  Args:
    nin: Number of input vertices.
    poly: Input polygon vertices (one per row).
    coord: Clipping coordinate (0 or 1).
    sign: Half-plane orientation (+1 or -1).
    limit: Half-plane offset.

  Returns:
    - Number of output vertices.
    - Output polygon vertices (one per row).
  """
  d = vec8f()
  all_inside = bool(True)
  for k in range(nin):
    d[k] = sign * poly[k, coord] - limit
    if not (d[k] <= 0.0):
      all_inside = False
  if all_inside:
    return nin, poly

  out = mat83f()
  nout = int(0)
  for k in range(nin):
    k1 = wp.where(k + 1 == nin, 0, k + 1)
    dp = d[k]
    dq = d[k1]

    # emit p if inside
    if dp <= 0.0 and nout < _BOXBOX_MAXVERT:
      out[nout] = poly[k]
      nout += 1

    # emit intersection if the edge strictly crosses the plane
    if ((dp < 0.0 and dq > 0.0) or (dp > 0.0 and dq < 0.0)) and nout < _BOXBOX_MAXVERT:
      p = poly[k]
      q = poly[k1]
      t = dp / (dp - dq)
      out[nout] = wp.vec3(
        p[0] + t * (q[0] - p[0]),
        p[1] + t * (q[1] - p[1]),
        p[2] + t * (q[2] - p[2]),
      )
      nout += 1

  return nout, out


@wp.func
def box_box(
  # In:
  box1_pos: wp.vec3,
  box1_rot: wp.mat33,
  box1_size: wp.vec3,
  box2_pos: wp.vec3,
  box2_rot: wp.mat33,
  box2_size: wp.vec3,
  margin: float,
) -> Tuple[int, vec8f, mat83f, wp.vec3]:
  """Core contact geometry calculation for box-box collision.

  Port of MuJoCo's mjc_BoxBox: separating-axis test with face-axis preference,
  followed by
  incident-face clipping (face axis) or a closest-point edge-edge contact (edge
  axis).

  Args:
    box1_pos: Center position of the first box.
    box1_rot: Rotation matrix of the first box.
    box1_size: Half-extents of the first box along each axis.
    box2_pos: Center position of the second box.
    box2_rot: Rotation matrix of the second box.
    box2_size: Half-extents of the second box along each axis.
    margin: Collision tolerance (margin + gap): contacts with dist <= margin are
      reported.

  Returns:
    - Number of contacts.
    - Vector of contact distances (MJ_MAXVAL for unpopulated contacts).
    - Matrix of contact positions (one per row).
    - Contact normal vector, pointing from box1 to box2 (shared by all
    contacts).
  """
  contact_dist = vec8f(MJ_MAXVAL)
  contact_pos = mat83f()
  contact_normal = wp.vec3(0.0)

  # rot: box2 axes in box1 frame (columns); pos21: box2 center in box1 frame;
  # pos12: box1 center in box2 frame
  pos21 = wp.transpose(box1_rot) @ (box2_pos - box1_pos)
  pos12 = wp.transpose(box2_rot) @ (box1_pos - box2_pos)
  rot = wp.transpose(box1_rot) @ box2_rot
  rotabs = wp.matrix_from_rows(wp.abs(rot[0]), wp.abs(rot[1]), wp.abs(rot[2]))

  # ------------------------------ stage 1: separating-axis test

  # the separation tests decide contact against no contact, so they carry rounding slack:
  # without it a box pair that genuinely overlaps by less than the rounding error of its
  # own support evaluation is reported as separated, and the boxes pass through each other
  septol = margin + _BOXBOX_SEPEPS * (box1_size[0] + box1_size[1] + box1_size[2] + box2_size[0] + box2_size[1] + box2_size[2])

  # best separation so far (most positive; negative = penetration), and the winning axis:
  # code 0..2 face of box1, 3..5 face of box2, >= 6 edge pair (i, j) as 6 + 3 * i + j
  sep_best = float(-MJ_MAXVAL)
  code = int(-1)

  # face axes of box1: candidate normal is axis i of box1
  for i in range(3):
    radius2 = rotabs[i, 0] * box2_size[0] + rotabs[i, 1] * box2_size[1] + rotabs[i, 2] * box2_size[2]
    sep = wp.abs(pos21[i]) - box1_size[i] - radius2
    if sep > septol:
      return 0, contact_dist, contact_pos, contact_normal
    if sep > sep_best:
      sep_best = sep
      code = i

  # face axes of box2: candidate normal is axis j of box2
  for j in range(3):
    radius1 = rotabs[0, j] * box1_size[0] + rotabs[1, j] * box1_size[1] + rotabs[2, j] * box1_size[2]
    sep = wp.abs(pos12[j]) - box2_size[j] - radius1
    if sep > septol:
      return 0, contact_dist, contact_pos, contact_normal
    if sep > sep_best:
      sep_best = sep
      code = 3 + j

  sep_face = sep_best
  code_face = code

  # edge-cross axes: candidate direction is axis i of box1 crossed with axis j of box2
  for i in range(3):
    for j in range(3):
      # cross product of e_i with column j of rot, in box1 frame; component i is zero
      i1 = (i + 1) % 3
      i2 = (i + 2) % 3
      ax1 = -rot[i2, j]
      ax2 = rot[i1, j]

      # the cross product of two unit vectors has norm sin(angle); for nearly parallel
      # edges the components above are pure cancellation noise and the direction is
      # meaningless, so require sin(angle) well above rounding; the skipped axes are
      # covered by the face normals, which the cross product converges to as the angle
      # vanishes
      norm2 = ax1 * ax1 + ax2 * ax2
      if norm2 < _BOXBOX_PAREPS:
        continue
      inv = 1.0 / wp.sqrt(norm2)
      ax1 = ax1 * inv
      ax2 = ax2 * inv

      # support radius of box1: component i of axis is zero by construction
      radius1 = box1_size[i1] * wp.abs(ax1) + box1_size[i2] * wp.abs(ax2)

      # support radius of box2: transform axis to box2 frame; component j is zero there,
      # and only components i1, i2 of the axis are nonzero here
      j1 = (j + 1) % 3
      j2 = (j + 2) % 3
      a2_1 = ax1 * rot[i1, j1] + ax2 * rot[i2, j1]
      a2_2 = ax1 * rot[i1, j2] + ax2 * rot[i2, j2]
      radius2 = box2_size[j1] * wp.abs(a2_1) + box2_size[j2] * wp.abs(a2_2)

      sep = wp.abs(ax1 * pos21[i1] + ax2 * pos21[i2]) - radius1 - radius2
      if sep > septol:
        return 0, contact_dist, contact_pos, contact_normal

      # an edge axis must beat the best face axis by a bias-scaled amount: on exact ties
      # the face manifold (multiple points) is strictly better for the solver
      if sep - _BOXBOX_EDGEBIAS * wp.abs(sep) > sep_best and sep > sep_face:
        sep_best = sep
        code = 6 + 3 * i + j

  if code < 0:
    return (
      0,
      contact_dist,
      contact_pos,
      contact_normal,
    )  # cannot happen: some face axis always sets code

  # winning edge pair (edge_i, edge_j) and its unit separating axis in box1 frame, set below
  # if an edge axis wins
  edge_i = int(0)
  edge_j = int(0)
  i1 = int(0)
  i2 = int(0)
  axis = wp.vec3(0.0)

  # a winning edge axis nearly parallel to the best face axis (within ~8 degrees)
  # duplicates it: the face manifold covers the same contact with multiple points, and
  # resting stacks flip between the two codes by rounding noise if the near-tie is
  # allowed to alternate. The face is substituted unless the edge is better by five
  # percent of the face depth. Substituting after the search, rather than filtering during
  # it, prevents a worse non-aliasing edge from stealing the contact that the substitution
  # meant to give to the face.
  if code >= 6:
    edge_i = (code - 6) // 3
    edge_j = (code - 6) % 3
    i1 = (edge_i + 1) % 3
    i2 = (edge_i + 2) % 3
    axis[i1] = -rot[i2, edge_j]
    axis[i2] = rot[i1, edge_j]
    # the edge-axis norm is bounded below by sqrt(_BOXBOX_PAREPS)
    axis = wp.normalize(axis)
    face_dot = float(0.0)
    if code_face < 3:
      face_dot = wp.abs(axis[code_face])
    else:
      f = code_face - 3
      face_dot = wp.abs(axis[0] * rot[0, f] + axis[1] * rot[1, f] + axis[2] * rot[2, f])
    if face_dot > 0.99 and sep_best < sep_face + 0.05 * wp.abs(sep_face) + MJ_MINVAL:
      code = code_face
      sep_best = sep_face

  # ------------------------------ stage 2a: edge-edge contact

  if code >= 6:
    j1 = (edge_j + 1) % 3
    j2 = (edge_j + 2) % 3

    # orient the separating axis from box1 toward box2
    if wp.dot(axis, pos21) < 0.0:
      axis = -axis

    # supporting edges: the box1 edge runs along e_i at a corner selected by the axis
    # signs in (i1, i2); the box2 edge runs along column j at a corner selected by the
    # signs of the axis in box2 coordinates. A near-zero component makes the sign choice
    # meaningless -- both edges support the axis -- and rounding can pick the wrong one,
    # producing witness points on the wrong side of the box. Enumerate both signs for any
    # ambiguous component (at most one per box) and keep the closest witness pair.
    a2 = wp.vec3(
      axis[0] * rot[0, 0] + axis[1] * rot[1, 0] + axis[2] * rot[2, 0],
      axis[0] * rot[0, 1] + axis[1] * rot[1, 1] + axis[2] * rot[2, 1],
      axis[0] * rot[0, 2] + axis[1] * rot[1, 2] + axis[2] * rot[2, 2],
    )
    amb1 = int(-1)
    amb2 = int(-1)
    if wp.abs(axis[i1]) < _BOXBOX_SGNEPS:
      amb1 = i1
    elif wp.abs(axis[i2]) < _BOXBOX_SGNEPS:
      amb1 = i2
    if wp.abs(a2[j1]) < _BOXBOX_SGNEPS:
      amb2 = j1
    elif wp.abs(a2[j2]) < _BOXBOX_SGNEPS:
      amb2 = j2

    d2 = wp.vec3(rot[0, edge_j], rot[1, edge_j], rot[2, edge_j])
    b = d2[edge_i]  # d1 . d2, with d1 = e_i
    denom = 1.0 - b * b

    w1 = wp.vec3(0.0)
    w2 = wp.vec3(0.0)
    best_d2 = float(MJ_MAXVAL)
    for v1 in range(wp.where(amb1 >= 0, 2, 1)):
      for v2 in range(wp.where(amb2 >= 0, 2, 1)):
        # corner of the box1 edge: support along +axis, ambiguous component flipped by v1
        c1 = wp.vec3(0.0)
        c1[i1] = wp.where(axis[i1] >= 0.0, box1_size[i1], -box1_size[i1])
        c1[i2] = wp.where(axis[i2] >= 0.0, box1_size[i2], -box1_size[i2])
        if amb1 >= 0 and v1 != 0:
          c1[amb1] = -c1[amb1]

        # corner of the box2 edge: support along -axis in box2 coordinates
        cc = wp.vec3(0.0)
        cc[j1] = wp.where(a2[j1] >= 0.0, -box2_size[j1], box2_size[j1])
        cc[j2] = wp.where(a2[j2] >= 0.0, -box2_size[j2], box2_size[j2])
        if amb2 >= 0 and v2 != 0:
          cc[amb2] = -cc[amb2]
        c2 = rot @ cc + pos21

        # closest points between the two edge segments (directions are unit vectors)
        e = c2 - c1
        d1e = e[edge_i]  # d1 . e
        d2e = wp.dot(d2, e)
        s = float(0.0)
        if denom >= MJ_MINVAL:
          s = (d1e - b * d2e) / denom

        # clamp into the segments, letting each clamp re-solve the other parameter
        s = wp.clamp(s, -box1_size[edge_i], box1_size[edge_i])
        t = wp.clamp(b * s - d2e, -box2_size[edge_j], box2_size[edge_j])
        s = wp.clamp(d1e + b * t, -box1_size[edge_i], box1_size[edge_i])

        # c1[i] is zero, so the box1 witness is c1 with component i set to s
        p1 = c1
        p1[edge_i] = s
        p2 = c2 + d2 * t
        gap = p2 - p1
        gap2 = wp.dot(gap, gap)
        if gap2 < best_d2:
          best_d2 = gap2
          w1 = p1
          w2 = p2

    # signed distance along the axis
    dist = wp.dot(w2 - w1, axis)
    if dist > septol:
      return 0, contact_dist, contact_pos, contact_normal

    # contact at the midpoint of the witness pair: for penetrating edges this is inside both
    # boxes; in the margin band it is midway between the two surfaces
    mid = 0.5 * (w1 + w2)

    contact_dist[0] = dist
    contact_pos[0] = box1_rot @ mid + box1_pos
    contact_normal = box1_rot @ axis
    return 1, contact_dist, contact_pos, contact_normal

  # ------------------------------ stage 2b: face contact

  # reference box: owner of the winning face; incident box: the other one
  ref1 = code < 3  # is box1 the reference?
  a = wp.where(ref1, code, code - 3)  # face axis of the reference box
  sizeref = wp.where(ref1, box1_size, box2_size)
  sizeinc = wp.where(ref1, box2_size, box1_size)
  posref = wp.where(ref1, box1_pos, box2_pos)
  matref = wp.where(ref1, box1_rot, box2_rot)
  posoi = wp.where(ref1, pos21, pos12)  # incident center in reference frame

  # incident box axes in reference frame: rot maps box2 to box1, transpose maps box1 to box2;
  # rinc[r, c] = component r of incident axis c, in reference frame
  rinc = wp.where(ref1, rot, wp.transpose(rot))

  # face direction: +1 if the incident box lies along +a, else -1
  sgn = wp.where(posoi[a] >= 0.0, 1.0, -1.0)

  # incident face: the face of the incident box most opposed to the reference face normal
  binc = int(0)
  for k in range(1, 3):
    if wp.abs(rinc[a, k]) > wp.abs(rinc[a, binc]):
      binc = k
  tinc = wp.where(sgn * rinc[a, binc] > 0.0, -1.0, 1.0)  # sign making the incident normal oppose

  # corners of the incident face in reference frame, cyclic winding; the in-plane
  # coordinates are (x, y) = the two non-a reference axes, z is the signed distance
  # above the reference face plane (negative = inside the reference box)
  ax = (a + 1) % 3
  ay = (a + 2) % 3
  bu = (binc + 1) % 3
  bv = (binc + 2) % 3

  # face center and in-face half-edge offsets, in the projected (x, y, z) coordinates
  coords = wp.vec3i(ax, ay, a)
  cx = wp.vec3(0.0)
  du = wp.vec3(0.0)
  dv = wp.vec3(0.0)
  for r in range(3):
    c = coords[r]
    cx[r] = posoi[c] + tinc * sizeinc[binc] * rinc[c, binc]
    du[r] = sizeinc[bu] * rinc[c, bu]
    dv[r] = sizeinc[bv] * rinc[c, bv]
  cx[2] = sgn * cx[2] - sizeref[a]
  du[2] = du[2] * sgn
  dv[2] = dv[2] * sgn

  # corner signs (su, sv): (1, 1), (-1, 1), (-1, -1), (1, -1)
  poly = mat83f()
  for k in range(4):
    su = wp.where(k == 0 or k == 3, 1.0, -1.0)
    sv = wp.where(k < 2, 1.0, -1.0)
    poly[k] = cx + su * du + sv * dv

  # clip against the four side planes of the reference face
  nvert = int(4)
  nvert, poly = _box_box_clip(nvert, poly, 0, 1.0, sizeref[ax])
  nvert, poly = _box_box_clip(nvert, poly, 0, -1.0, sizeref[ax])
  nvert, poly = _box_box_clip(nvert, poly, 1, 1.0, sizeref[ay])
  nvert, poly = _box_box_clip(nvert, poly, 1, -1.0, sizeref[ay])

  # accept vertices within the margin band, dropping near-duplicates produced by clipping
  # at polygon corners; duplicate radius is relative to the reference face scale; accepted
  # vertices are staged in contact_pos (reference-frame coordinates)
  naccept = int(0)
  dupe2 = _BOXBOX_DUPEPS * (sizeref[ax] * sizeref[ax] + sizeref[ay] * sizeref[ay])
  for k in range(nvert):
    if poly[k, 2] > margin:
      continue
    dupe = bool(False)
    for q in range(naccept):
      dx = contact_pos[q, 0] - poly[k, 0]
      dy = contact_pos[q, 1] - poly[k, 1]
      if dx * dx + dy * dy < dupe2:
        dupe = True
        break
    if not dupe:
      contact_pos[naccept] = poly[k]
      naccept += 1

  if naccept == 0:
    return 0, contact_dist, contact_pos, contact_normal

  # world normal points from geom1 to geom2: along +sgn * a of the reference frame when box1
  # is the reference, opposite when box2 is
  nsign = wp.where(ref1, sgn, -sgn)
  contact_normal = nsign * wp.vec3(matref[0, a], matref[1, a], matref[2, a])

  for k in range(naccept):
    v = contact_pos[k]

    # contact position: on the clipped incident polygon in (x, y), midway between the
    # reference face plane and the incident surface along the face axis
    posc = wp.vec3(0.0)
    posc[ax] = v[0]
    posc[ay] = v[1]
    posc[a] = sgn * (sizeref[a] + 0.5 * v[2])

    contact_dist[k] = v[2]
    contact_pos[k] = matref @ posc + posref

  return naccept, contact_dist, contact_pos, contact_normal


@wp.func
def sphere_box(
  # In:
  sphere_pos: wp.vec3,
  sphere_radius: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Core contact geometry calculation for sphere-box collision.

  Args:
    sphere_pos: Center position of the sphere.
    sphere_radius: Radius of the sphere.
    box_pos: Center position of the box.
    box_rot: Rotation matrix of the box.
    box_size: Half-extents of the box along each axis.

  Returns:
    - Vector of contact distances.
    - Contact positions.
    - Contact normal vectors.
  """
  center = wp.transpose(box_rot) @ (sphere_pos - box_pos)

  clamped = wp.max(-box_size, wp.min(box_size, center))
  clamped_dir, dist = normalize_with_norm(clamped - center)

  # sphere center inside box
  if dist <= MJ_MINVAL:
    closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])
    k = wp.int32(0)
    for i in range(6):
      face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])
      if closest > face_dist:
        closest = face_dist
        k = i

    nearest = wp.vec3(0.0)
    nearest[k // 2] = wp.where(k % 2, -1.0, 1.0)
    pos = center + nearest * (sphere_radius - closest) / 2.0
    contact_normal = box_rot @ nearest
    contact_distance = -closest - sphere_radius

  else:
    deepest = center + clamped_dir * sphere_radius
    pos = 0.5 * (clamped + deepest)
    contact_normal = box_rot @ clamped_dir
    contact_distance = dist - sphere_radius

  contact_position = box_pos + box_rot @ pos

  return contact_distance, contact_position, contact_normal


@wp.func
def capsule_box(
  # In:
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
) -> Tuple[wp.vec2, mat23f, mat23f]:
  """Core contact geometry calculation for capsule-box collision.

  Args:
    capsule_pos: Center position of the capsule.
    capsule_axis: Axis direction of the capsule.
    capsule_radius: Radius of the capsule.
    capsule_half_length: Half length of the capsule.
    box_pos: Center position of the box.
    box_rot: Rotation matrix of the box.
    box_size: Half-extents of the box along each axis.

  Returns:
    - Vector of contact distances (MJ_MAXVAL for unpopulated contacts).
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  # Based on the mjc implementation
  boxmatT = wp.transpose(box_rot)
  pos = boxmatT @ (capsule_pos - box_pos)
  axis = boxmatT @ capsule_axis
  halfaxis = axis * capsule_half_length  # halfaxis is the capsule direction
  axisdir = wp.int32(halfaxis[0] > 0.0) + 2 * wp.int32(halfaxis[1] > 0.0) + 4 * wp.int32(halfaxis[2] > 0.0)

  # keep track of closest point
  bestdist = wp.float32(1.0e32)
  bestsegmentpos = wp.float32(-12)

  # cltype: encoded collision configuration
  # cltype / 3 == 0 : lower corner is closest to the capsule
  #            == 2 : upper corner is closest to the capsule
  #            == 1 : middle of the edge is closest to the capsule
  # cltype % 3 == 0 : lower corner is closest to the box
  #            == 2 : upper corner is closest to the box
  #            == 1 : middle of the capsule is closest to the box
  cltype = wp.int32(-4)

  # clface: index of the closest face of the box to the capsule
  # -1: no face is closest (edge or corner is closest)
  # 0, 1, 2: index of the axis perpendicular to the closest face
  clface = wp.int32(-12)

  # first: consider cases where a face of the box is closest
  for i in range(-1, 2, 2):
    axisTip = pos + wp.float32(i) * halfaxis
    boxPoint = wp.vec3(axisTip)

    n_out = wp.int32(0)
    ax_out = wp.int32(-1)

    for j in range(3):
      if boxPoint[j] < -box_size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = -box_size[j]
      elif boxPoint[j] > box_size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = box_size[j]

    if n_out > 1:
      continue

    dist = wp.length_sq(boxPoint - axisTip)

    if dist < bestdist:
      bestdist = dist
      bestsegmentpos = wp.float32(i)
      cltype = -2 + i
      clface = ax_out

  # second: consider cases where an edge of the box is closest
  clcorner = wp.int32(-123)  # which corner is the closest
  cledge = wp.int32(-123)  # which axis
  bestboxpos = wp.float32(0.0)

  for i in range(8):
    for j in range(3):
      if (i & (1 << j)) != 0:
        continue

      c2 = wp.int32(-123)

      # box_pt is the starting point (corner) on the box
      box_pt = wp.cw_mul(
        wp.vec3(
          wp.where(i & 1, 1.0, -1.0),
          wp.where(i & 2, 1.0, -1.0),
          wp.where(i & 4, 1.0, -1.0),
        ),
        box_size,
      )
      box_pt[j] = 0.0

      # find closest point between capsule and the edge
      dif = box_pt - pos

      u = -box_size[j] * dif[j]
      v = wp.dot(halfaxis, dif)
      ma = box_size[j] * box_size[j]
      mb = -box_size[j] * halfaxis[j]
      mc = capsule_half_length * capsule_half_length
      det = ma * mc - mb * mb
      if wp.abs(det) < MJ_MINVAL:
        continue

      idet = 1.0 / det
      # sX : X=1 means middle of segment. X=0 or 2 one or the other end

      x1 = wp.float32((mc * u - mb * v) * idet)
      x2 = wp.float32((ma * v - mb * u) * idet)

      s1 = wp.int32(1)
      s2 = wp.int32(1)

      if x1 > 1:
        x1 = 1.0
        s1 = 2
        x2 = safe_div(v - mb, mc)
      elif x1 < -1:
        x1 = -1.0
        s1 = 0
        x2 = safe_div(v + mb, mc)

      x2_over = x2 > 1.0
      if x2_over or x2 < -1.0:
        if x2_over:
          x2 = 1.0
          s2 = 2
          x1 = safe_div(u - mb, ma)
        else:
          x2 = -1.0
          s2 = 0
          x1 = safe_div(u + mb, ma)

        if x1 > 1:
          x1 = 1.0
          s1 = 2
        elif x1 < -1:
          x1 = -1.0
          s1 = 0

      dif -= halfaxis * x2
      dif[j] += box_size[j] * x1

      # encode relative positions of the closest points
      ct = s1 * 3 + s2

      dif_sq = wp.length_sq(dif)
      if dif_sq < bestdist - MJ_MINVAL:
        bestdist = dif_sq
        bestsegmentpos = x2
        bestboxpos = x1
        # ct<6 means closest point on box is at lower end or middle of edge
        c2 = ct // 6

        clcorner = i + (1 << j) * c2  # index of closest box corner
        cledge = j  # axis index of closest box edge
        cltype = ct  # encoded collision configuration

  best = wp.float32(0.0)

  p = wp.vec2(pos.x, pos.y)
  dd = wp.vec2(halfaxis.x, halfaxis.y)
  s = wp.vec2(box_size[0], box_size[1])
  secondpos = wp.float32(-4.0)

  uu = dd.x * s.y
  vv = dd.y * s.x
  w_neg = dd.x * p.y - dd.y * p.x < 0

  best = wp.float32(-1.0)

  ee1 = uu - vv
  ee2 = uu + vv

  if wp.abs(ee1) > best:
    best = wp.abs(ee1)
    c1 = wp.where((ee1 < 0) == w_neg, 0, 3)

  if wp.abs(ee2) > best:
    best = wp.abs(ee2)
    c1 = wp.where((ee2 > 0) == w_neg, 1, 2)

  if cltype == -4:  # invalid type
    return wp.vec2(MJ_MAXVAL), mat23f(), mat23f()

  if cltype >= 0 and cltype // 3 != 1:  # closest to a corner of the box
    c1 = axisdir ^ clcorner
    # Calculate relative orientation between capsule and corner
    # There are two possible configurations:
    # 1. Capsule axis points toward/away from corner
    # 2. Capsule axis aligns with a face or edge
    if c1 != 0 and c1 != 7:  # create second contact point
      if c1 == 1 or c1 == 2 or c1 == 4:
        mul = 1
      else:
        mul = -1
        c1 = 7 - c1

      # "de" and "dp" distance from first closest point on the capsule to both ends of it
      # mul is a direction along the capsule's axis

      if c1 == 1:
        ax = 0
        ax1 = 1
        ax2 = 2
      elif c1 == 2:
        ax = 1
        ax1 = 2
        ax2 = 0
      elif c1 == 4:
        ax = 2
        ax1 = 0
        ax2 = 1

      if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box
        m = 2.0 * safe_div(box_size[ax], wp.abs(halfaxis[ax]))
        secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)
      else:  # second point along a face of the box
        # check for overshoot again
        m = 2.0 * min(
          safe_div(box_size[ax1], wp.abs(halfaxis[ax1])),
          safe_div(box_size[ax2], wp.abs(halfaxis[ax2])),
        )
        secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)
      secondpos *= wp.float32(mul)

  elif cltype >= 0 and cltype // 3 == 1:  # we are on box's edge
    # Calculate relative orientation between capsule and edge
    # Two possible configurations:
    # - T configuration: c1 = 2^n (no additional contacts)
    # - X configuration: c1 != 2^n (potential additional contacts)
    c1 = axisdir ^ clcorner
    c1 &= 7 - (1 << cledge)  # mask out edge axis to determine configuration

    if c1 == 1 or c1 == 2 or c1 == 4:  # create second contact point
      if cledge == 0:
        ax1 = 1
        ax2 = 2
      if cledge == 1:
        ax1 = 2
        ax2 = 0
      if cledge == 2:
        ax1 = 0
        ax2 = 1
      ax = cledge

      # find which face the capsule has a lower angle, and switch the axis
      if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):
        ax1 = ax2
      ax2 = 3 - ax - ax1

      # mul determines direction along capsule axis for second contact point
      if c1 & (1 << ax2):
        mul = 1
        secondpos = 1.0 - bestsegmentpos
      else:
        mul = -1
        secondpos = 1.0 + bestsegmentpos

      # now find out whether we point towards the opposite side or towards one of the sides
      # and also find the farthest point along the capsule that is above the box

      e1 = 2.0 * safe_div(box_size[ax2], wp.abs(halfaxis[ax2]))
      secondpos = min(e1, secondpos)

      if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):
        e2 = 1.0 - bestboxpos
      else:
        e2 = 1.0 + bestboxpos

      e1 = box_size[ax] * safe_div(e2, wp.abs(halfaxis[ax]))

      secondpos = min(e1, secondpos)
      secondpos *= wp.float32(mul)

  elif cltype < 0:
    # similarly we handle the case when one capsule's end is closest to a face of the box
    # and find where is the other end pointing to and clamping to the farthest point
    # of the capsule that's above the box
    # if the closest point is inside the box there's no need for a second point

    if clface != -1:  # create second contact point
      mul = wp.where(cltype == -3, 1, -1)
      secondpos = 2.0

      tmp1 = pos - halfaxis * wp.float32(mul)

      for i in range(3):
        if i != clface:
          ha_r = safe_div(wp.float32(mul), halfaxis[i])
          e1 = (box_size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

          e1 = (-box_size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

      secondpos *= wp.float32(mul)

  # create sphere in original orientation at first contact point
  s1_pos_l = pos + halfaxis * bestsegmentpos
  s1_pos_g = box_rot @ s1_pos_l + box_pos

  # collide with sphere using core function
  dist1, pos1, normal1 = sphere_box(s1_pos_g, capsule_radius, box_pos, box_rot, box_size)

  if secondpos > -3:  # secondpos was modified
    s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)
    s2_pos_g = box_rot @ s2_pos_l + box_pos

    # collide with sphere using core function
    dist2, pos2, normal2 = sphere_box(s2_pos_g, capsule_radius, box_pos, box_rot, box_size)
  else:
    dist2 = MJ_MAXVAL
    pos2 = wp.vec3()
    normal2 = wp.vec3()

  return (
    wp.vec2(dist1, dist2),
    mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]),
    mat23f(normal1[0], normal1[1], normal1[2], normal2[0], normal2[1], normal2[2]),
  )


@wp.func
def _tri_area_sign(p1: wp.vec2, p2: wp.vec2, p3: wp.vec2) -> float:
  """Sign of (signed) area of planar triangle."""
  return wp.sign((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))


@wp.func
def _tri_point_segment(p: wp.vec2, u: wp.vec2, v: wp.vec2) -> wp.vec2:
  """Find nearest point to p within line segment (u, v)."""
  uv = v - u
  up = p - u

  denom = wp.max(MJ_MINVAL, wp.dot(uv, uv))
  a = wp.dot(uv, up) / denom

  if a <= 0.0:
    return u
  elif a >= 1.0:
    return v
  else:
    return u + a * uv


@wp.func
def sphere_triangle(
  # In:
  sphere_pos: wp.vec3,
  sphere_radius: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Core contact geometry calculation for sphere-triangle collision.

  Port of mjraw_SphereTriangle from engine_collision_primitive.c

  Args:
    sphere_pos: Center position of the sphere.
    sphere_radius: Radius of the sphere.
    t1: Triangle vertex positions.
    t2: Triangle vertex positions.
    t3: Triangle vertex positions.
    tri_radius: Triangle (flex element) radius.
    margin: Collision margin for filtering contacts.

  Returns:
    - Contact distance (MJ_MAXVAL if no collision).
    - Contact position.
    - Contact normal vector.
  """
  S = sphere_pos - t1
  A = t2 - t1
  B = t3 - t1

  N = wp.normalize(wp.cross(A, B))

  dstS = wp.dot(N, S)
  if wp.abs(dstS) > margin + sphere_radius + tri_radius:
    return MJ_MAXVAL, wp.vec3(0.0), wp.vec3(0.0)

  P = S - dstS * N

  V1 = wp.normalize(A)
  lenA = wp.length(A)
  V2 = wp.normalize(wp.cross(N, A))

  o = wp.vec2(0.0, 0.0)
  a = wp.vec2(lenA, 0.0)
  b = wp.vec2(wp.dot(V1, B), wp.dot(V2, B))
  p = wp.vec2(wp.dot(V1, P), wp.dot(V2, P))

  sign1 = _tri_area_sign(p, o, a)
  sign2 = _tri_area_sign(p, a, b)
  sign3 = _tri_area_sign(p, b, o)

  X = wp.vec3(0.0)
  if sign1 == sign2 and sign2 == sign3:
    X = P
  else:
    x0 = _tri_point_segment(p, o, a)
    x1 = _tri_point_segment(p, a, b)
    x2 = _tri_point_segment(p, b, o)

    d0 = wp.length(p - x0)
    d1 = wp.length(p - x1)
    d2 = wp.length(p - x2)

    if d0 < d1 and d0 < d2:
      X = x0[0] * V1 + x0[1] * V2
    elif d1 < d2:
      X = x1[0] * V1 + x1[1] * V2
    else:
      X = x2[0] * V1 + x2[1] * V2

  dist, pos, nrm = sphere_sphere(sphere_pos, sphere_radius, t1 + X, tri_radius)
  if dist > margin:
    return MJ_MAXVAL, wp.vec3(0.0), wp.vec3(0.0)

  return dist, pos, nrm


@wp.func
def box_triangle(
  # In:
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
) -> Tuple[vec11, mat113, mat113]:
  """Core contact geometry calculation for box-triangle collision.

  Port of mjraw_BoxTriangle from engine_collision_primitive.c

  Args:
    box_pos: Center position of the box.
    box_rot: Orientation matrix of the box.
    box_size: Half-sizes of the box.
    t1: Triangle vertex positions.
    t2: Triangle vertex positions.
    t3: Triangle vertex positions.
    tri_radius: Triangle (flex element) radius.
    margin: Collision margin for filtering contacts.

  Returns:
    - Vector of contact distances (MJ_MAXVAL for unpopulated contacts): triangle
      vertices 0-2, then box corners 3-10.
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  contact_dist = vec11(MJ_MAXVAL)
  contact_pos = mat113()
  contact_normal = mat113()

  box_rotT = wp.transpose(box_rot)

  for vi in range(3):
    vert = wp.vec3(0.0)
    if vi == 0:
      vert = t1
    elif vi == 1:
      vert = t2
    else:
      vert = t3

    diff = vert - box_pos
    local = box_rotT @ diff

    maxaxis = 0
    maxval = wp.abs(local[0]) - box_size[0]
    for j in range(1, 3):
      val = wp.abs(local[j]) - box_size[j]
      if val > maxval:
        maxval = val
        maxaxis = j

    inside = maxval - tri_radius <= margin
    for j in range(3):
      if wp.abs(local[j]) > box_size[j] + margin + tri_radius:
        inside = False

    if inside:
      nrm_local = wp.vec3(0.0)
      nrm_local[maxaxis] = wp.where(local[maxaxis] > 0.0, 1.0, -1.0)

      nrm_global = box_rot @ nrm_local
      d = maxval - tri_radius
      offset = tri_radius + d * 0.5
      p = vert - nrm_global * offset

      contact_dist[vi] = d
      contact_pos[vi] = p
      contact_normal[vi] = nrm_global

  for i in range(8):
    vec = wp.vec3(
      wp.where(i & 1, box_size[0], -box_size[0]),
      wp.where(i & 2, box_size[1], -box_size[1]),
      wp.where(i & 4, box_size[2], -box_size[2]),
    )
    corner = box_rot @ vec + box_pos

    d, p, n = sphere_triangle(corner, 0.0, t1, t2, t3, tri_radius, margin)
    contact_dist[3 + i] = d
    contact_pos[3 + i] = p
    contact_normal[3 + i] = n

  return contact_dist, contact_pos, contact_normal


@wp.func
def tri_weights(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
  """Barycentric weights of the closest point on triangle abc to p.

  Port of mjc_triWeights from engine_collision_primitive.c.

  Args:
    p: Query point.
    a: Triangle vertex positions.
    b: Triangle vertex positions.
    c: Triangle vertex positions.

  Returns:
    Barycentric weights of the closest point, ordered as (a, b, c).
  """
  # compute barycentric coordinates via Voronoi regions (See Ericson 5.1.5)
  ab = b - a
  ac = c - a
  ap = p - a
  bp = p - b
  cp = p - c

  dot1 = wp.dot(ab, ap)
  dot2 = wp.dot(ac, ap)
  dot3 = wp.dot(ab, bp)
  dot4 = wp.dot(ac, bp)
  dot5 = wp.dot(ab, cp)
  dot6 = wp.dot(ac, cp)

  # vertex region outside a
  if dot1 <= 0.0 and dot2 <= 0.0:
    return wp.vec3(1.0, 0.0, 0.0)

  # vertex region outside b
  if dot3 >= 0.0 and dot4 <= dot3:
    return wp.vec3(0.0, 1.0, 0.0)

  # vertex region outside c
  if dot6 >= 0.0 and dot5 <= dot6:
    return wp.vec3(0.0, 0.0, 1.0)

  # signed areas of pbc, pca, pab scaled by 4 * area(abc)
  va = dot3 * dot6 - dot5 * dot4
  vb = dot5 * dot2 - dot1 * dot6
  vc = dot1 * dot4 - dot3 * dot2

  # edge region ab
  if vc <= 0.0 and dot1 >= 0.0 and dot3 <= 0.0:
    t = dot1 / (dot1 - dot3)
    return wp.vec3(1.0 - t, t, 0.0)

  # edge region ac
  if vb <= 0.0 and dot2 >= 0.0 and dot6 <= 0.0:
    t = dot2 / (dot2 - dot6)
    return wp.vec3(1.0 - t, 0.0, t)

  # edge region bc
  if va <= 0.0 and (dot4 - dot3) >= 0.0 and (dot5 - dot6) >= 0.0:
    t = (dot4 - dot3) / ((dot4 - dot3) + (dot5 - dot6))
    return wp.vec3(0.0, 1.0 - t, t)

  # face region abc
  inv = 1.0 / (va + vb + vc)
  t = vb * inv
  u = vc * inv
  return wp.vec3(1.0 - t - u, t, u)


@wp.func
def capsule_triangle(
  # In:
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
) -> Tuple[vec5, mat53, mat53]:
  """Core contact geometry calculation for capsule-triangle collision.

  Port of mjraw_CapsuleTriangle from engine_collision_primitive.c

  Args:
    capsule_pos: Center position of the capsule.
    capsule_axis: Unit axis direction of the capsule.
    capsule_radius: Radius of the capsule.
    capsule_half_length: Half-length of the capsule cylinder.
    t1: Triangle vertex positions.
    t2: Triangle vertex positions.
    t3: Triangle vertex positions.
    tri_radius: Triangle (flex element) radius.
    margin: Collision margin for filtering contacts.

  Returns:
    - Vector of contact distances (MJ_MAXVAL for unpopulated contacts): capsule
      end caps 0-1, then triangle vertices 2-4.
    - Matrix of contact positions (one per row).
    - Matrix of contact normal vectors (one per row).
  """
  contact_dist = vec5(MJ_MAXVAL)
  contact_pos = mat53()
  contact_normal = mat53()

  p1 = capsule_pos - capsule_axis * capsule_half_length
  p2 = capsule_pos + capsule_axis * capsule_half_length

  d, p, n = sphere_triangle(p1, capsule_radius, t1, t2, t3, tri_radius, margin)
  contact_dist[0] = d
  contact_pos[0] = p
  contact_normal[0] = n

  d, p, n = sphere_triangle(p2, capsule_radius, t1, t2, t3, tri_radius, margin)
  contact_dist[1] = d
  contact_pos[1] = p
  contact_normal[1] = n

  ab = p2 - p1
  ab_len_sq = 4.0 * capsule_half_length * capsule_half_length
  inv_ab_len_sq = 1.0 / wp.max(MJ_MINVAL, ab_len_sq)

  for vi in range(3):
    vert = wp.vec3(0.0)
    if vi == 0:
      vert = t1
    elif vi == 1:
      vert = t2
    else:
      vert = t3

    vec = vert - p1
    t_param = wp.dot(vec, ab) * inv_ab_len_sq

    if t_param > MJ_MINVAL and t_param < 1.0 - MJ_MINVAL:
      closest = p1 + ab * t_param
      d, p, n = sphere_sphere(closest, capsule_radius, vert, tri_radius)
      if d <= margin:
        contact_dist[2 + vi] = d
        contact_pos[2 + vi] = p
        contact_normal[2 + vi] = n

  return contact_dist, contact_pos, contact_normal


@wp.func
def cylinder_triangle(
  # In:
  cylinder_pos: wp.vec3,
  cylinder_axis: wp.vec3,
  cylinder_radius: float,
  cylinder_half_height: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
) -> Tuple[wp.vec2, mat23f, mat23f]:
  """Core contact geometry calculation for cylinder-triangle collision.

  Args:
    cylinder_pos: Center position of the cylinder.
    cylinder_axis: Unit axis direction of the cylinder.
    cylinder_radius: Radius of the cylinder.
    cylinder_half_height: Half-height of the cylinder.
    t1: Triangle vertex positions.
    t2: Triangle vertex positions.
    t3: Triangle vertex positions.
    tri_radius: Triangle (flex element) radius.

  Returns:
    - wp.vec2 of distances for up to 2 contacts (MJ_MAXVAL if no collision).
    - mat23f of contact positions (2 x vec3).
    - mat23f of contact normals (2 x vec3).
  """
  dist1 = MJ_MAXVAL
  dist2 = MJ_MAXVAL
  pos1 = wp.vec3(0.0)
  pos2 = wp.vec3(0.0)
  nrm1 = wp.vec3(0.0)
  nrm2 = wp.vec3(0.0)
  cnt = int(0)

  p1 = cylinder_pos - cylinder_axis * cylinder_half_height
  p2 = cylinder_pos + cylinder_axis * cylinder_half_height

  ab = p2 - p1
  ab_len_sq = 4.0 * cylinder_half_height * cylinder_half_height

  for vi in range(3):
    if cnt >= 2:
      break

    vert = wp.vec3(0.0)
    if vi == 0:
      vert = t1
    elif vi == 1:
      vert = t2
    else:
      vert = t3

    vec = vert - p1
    t_param = wp.dot(vec, ab) / wp.max(MJ_MINVAL, ab_len_sq)

    if t_param > MJ_MINVAL and t_param < 1.0 - MJ_MINVAL:
      closest = p1 + ab * t_param
      diff = vert - closest
      dist_raw = wp.length(diff)

      if dist_raw < cylinder_radius + tri_radius:
        if dist_raw > MJ_MINVAL:
          nrm = diff / dist_raw
          d = dist_raw - cylinder_radius - tri_radius
          p = (closest + vert + nrm * (cylinder_radius - tri_radius)) * 0.5
        else:
          dist_to_side = cylinder_radius
          dist_to_p2 = (1.0 - t_param) * wp.sqrt(ab_len_sq)
          dist_to_p1 = t_param * wp.sqrt(ab_len_sq)

          if dist_to_p2 < dist_to_side and dist_to_p2 < dist_to_p1:
            nrm = cylinder_axis
            d = -dist_to_p2 - tri_radius
            p = vert
          elif dist_to_p1 < dist_to_side:
            nrm = -cylinder_axis
            d = -dist_to_p1 - tri_radius
            p = vert
          else:
            tri_normal = wp.normalize(wp.cross(t2 - t1, t3 - t1))
            nrm = tri_normal
            d = -cylinder_radius - tri_radius
            p = closest

        if cnt == 0:
          dist1 = d
          pos1 = p
          nrm1 = nrm
        else:
          dist2 = d
          pos2 = p
          nrm2 = nrm
        cnt += 1
    elif t_param <= MJ_MINVAL:
      diff = vert - p1
      signed_dist = wp.dot(diff, cylinder_axis)
      perp = diff - cylinder_axis * signed_dist
      perp_len = wp.length(perp)

      if perp_len < cylinder_radius:
        d = -signed_dist - tri_radius
        nrm = -cylinder_axis
        p = vert - nrm * (tri_radius + d * 0.5)
        if cnt == 0:
          dist1 = d
          pos1 = p
          nrm1 = nrm
        else:
          dist2 = d
          pos2 = p
          nrm2 = nrm
        cnt += 1
      elif perp_len < cylinder_radius + tri_radius:
        edge_dir = perp / perp_len
        edge_point = p1 + edge_dir * cylinder_radius
        diff_to_edge = vert - edge_point
        dist_raw = wp.length(diff_to_edge)
        if dist_raw > MJ_MINVAL:
          nrm = diff_to_edge / dist_raw
          d = dist_raw - tri_radius
          p = vert - nrm * (tri_radius + d * 0.5)
          if cnt == 0:
            dist1 = d
            pos1 = p
            nrm1 = nrm
          else:
            dist2 = d
            pos2 = p
            nrm2 = nrm
          cnt += 1
    else:
      diff = vert - p2
      signed_dist = wp.dot(diff, cylinder_axis)
      perp = diff - cylinder_axis * signed_dist
      perp_len = wp.length(perp)

      if perp_len < cylinder_radius:
        d = signed_dist - tri_radius
        nrm = cylinder_axis
        p = vert - nrm * (tri_radius + d * 0.5)
        if cnt == 0:
          dist1 = d
          pos1 = p
          nrm1 = nrm
        else:
          dist2 = d
          pos2 = p
          nrm2 = nrm
        cnt += 1
      elif perp_len < cylinder_radius + tri_radius:
        edge_dir = perp / perp_len
        edge_point = p2 + edge_dir * cylinder_radius
        diff_to_edge = vert - edge_point
        dist_raw = wp.length(diff_to_edge)
        if dist_raw > MJ_MINVAL:
          nrm = diff_to_edge / dist_raw
          d = dist_raw - tri_radius
          p = vert - nrm * (tri_radius + d * 0.5)
          if cnt == 0:
            dist1 = d
            pos1 = p
            nrm1 = nrm
          else:
            dist2 = d
            pos2 = p
            nrm2 = nrm
          cnt += 1

  # edge-vs-segment checks
  for edge_idx in range(3):
    if cnt >= 2:
      break
    u = t1 if edge_idx == 0 else (t2 if edge_idx == 1 else t3)
    v = t2 if edge_idx == 0 else (t3 if edge_idx == 1 else t1)

    closest_axis, closest_edge = closest_segment_to_segment_points(p1, p2, u, v)
    diff = closest_edge - closest_axis
    dist_raw = wp.length(diff)

    if dist_raw < cylinder_radius + tri_radius:
      if dist_raw > MJ_MINVAL:
        nrm = diff / dist_raw
        d = dist_raw - cylinder_radius - tri_radius
        p = (closest_axis + closest_edge + nrm * (cylinder_radius - tri_radius)) * 0.5

        if cnt == 0:
          dist1 = d
          pos1 = p
          nrm1 = nrm
        else:
          dist2 = d
          pos2 = p
          nrm2 = nrm
        cnt += 1

  return (
    wp.vec2(dist1, dist2),
    mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]),
    mat23f(nrm1[0], nrm1[1], nrm1[2], nrm2[0], nrm2[1], nrm2[2]),
  )


@wp.func
def _segment_triangle(a: wp.vec3, b: wp.vec3, t1: wp.vec3, t2: wp.vec3, t3: wp.vec3) -> Tuple[float, wp.vec3, wp.vec3, int]:
  """Closest points between segment ab and triangle t1 t2 t3, which it does not cross.

  Port of segTriangle from engine_collision_flex.c.

  Args:
    a: First endpoint of the segment.
    b: Second endpoint of the segment.
    t1: First triangle vertex.
    t2: Second triangle vertex.
    t3: Third triangle vertex.

  Returns:
    - Squared distance.
    - Closest point on the segment.
    - Closest point on the triangle.
    - Feature of the triangle containing its closest point whose normal is exact: 1 for the
      interior, 2 + i for the interior of edge i (from vertex i to vertex i + 1, from 0) with the
      closest points along the cross product of the segment and the edge, 0 otherwise.
  """
  best = float(MJ_MAXVAL)
  p = a
  q = a
  feature = int(0)

  # segment endpoints : triangle, a single point for a sphere
  point = a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
  for i in range(wp.where(point, 1, 2)):
    end = a if i == 0 else b
    w = tri_weights(end, t1, t2, t3)
    y = w[0] * t1 + w[1] * t2 + w[2] * t3
    dist2 = wp.length_sq(y - end)
    if dist2 < best:
      best = dist2
      p = end
      q = y
      feature = 0
      if w[0] > 0.0 and w[1] > 0.0 and w[2] > 0.0:
        feature = 1

  # segment : triangle edges, which the triangle query covers for a point
  if point:
    return best, p, q, feature
  for i in range(3):
    c = t1 if i == 0 else (t2 if i == 1 else t3)
    e = t2 if i == 0 else (t3 if i == 1 else t1)
    n, w0, w1 = _segment_weights(a, b, c, e)
    for j in range(n):
      ws = w0 if j == 0 else w1
      x = a + ws[0] * (b - a)
      y = c + ws[1] * (e - c)
      dist2 = wp.length_sq(y - x)
      if dist2 < best:
        best = dist2
        p = x
        q = y
        feature = 0
        if n == 1 and ws[0] > 0.0 and ws[0] < 1.0 and ws[1] > 0.0 and ws[1] < 1.0:
          feature = 2 + i

  return best, p, q, feature


@wp.func
def _tet_edge(k: int) -> Tuple[int, int]:
  """Vertex indices of edge k of a tetrahedron: (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)."""
  j0 = wp.where(k < 3, 0, wp.where(k < 5, 1, 2))
  j1 = wp.where(k < 3, k + 1, wp.where(k < 5, k - 1, 3))
  return j0, j1


@wp.func
def _tet_sep_axis(a: wp.vec3, b: wp.vec3, tet: mat43f, fnrm: mat43f, sa: wp.vec4, sb: wp.vec4) -> Tuple[float, wp.vec3, int]:
  """Minimum translation of segment ab out of a tetrahedron over the separating axes.

  Port of tetSepAxis from engine_collision_flex.c. The separating axes are the face normals and the
  cross products of the segment with the edges.

  Args:
    a: First endpoint of the segment.
    b: Second endpoint of the segment.
    tet: Tetrahedron vertices (one per row).
    fnrm: Outward face normals (one per row), face i is opposite vertex i.
    sa: Signed distances of a to the faces.
    sb: Signed distances of b to the faces.

  Returns:
    - Depth, negative if separated (then the axis is the one of maximum separation).
    - Axis from the segment to the tetrahedron.
    - Edge of a cross product axis, or -1 for a face normal.
  """
  ab = b - a
  depth = float(MJ_MAXVAL)
  nrm = wp.vec3(0.0)
  edge = int(-1)

  # face normals, translating out through face i
  for i in range(4):
    depth_i = -wp.min(sa[i], sb[i])
    if depth_i < depth:
      depth = depth_i
      nrm = -fnrm[i]

  # segment x edges, skipping (near) parallel pairs which are covered by the face normals
  for k in range(6):
    j0, j1 = _tet_edge(k)
    e = tet[j1] - tet[j0]
    axis = wp.cross(ab, e)
    norm2 = wp.dot(axis, axis)
    if norm2 <= MJ_MINVAL * wp.dot(ab, ab) * wp.dot(e, e):
      continue
    axis = axis / wp.sqrt(norm2)

    # overlap of the segment and tetrahedron projections, translating along +axis or -axis
    proja = wp.dot(axis, a)
    projb = wp.dot(axis, b)
    smin = wp.min(proja, projb)
    smax = wp.max(proja, projb)
    tmin = float(MJ_MAXVAL)
    tmax = float(-MJ_MAXVAL)
    for i in range(4):
      proj = wp.dot(axis, tet[i])
      tmin = wp.min(tmin, proj)
      tmax = wp.max(tmax, proj)
    if tmax - smin < depth:
      depth = tmax - smin
      nrm = -axis
      edge = k
    if smax - tmin < depth:
      depth = smax - tmin
      nrm = axis
      edge = k

  return depth, nrm, edge


@wp.func
def capsule_tetrahedron(
  # In:
  pa: wp.vec3,
  pb: wp.vec3,
  radius: float,
  v0: wp.vec3,
  v1: wp.vec3,
  v2: wp.vec3,
  v3: wp.vec3,
  tet_radius: float,
) -> Tuple[bool, float, wp.vec3, wp.vec3]:
  """Contact between a capsule (segment pa pb with radius, a sphere if pa = pb) and a tetrahedron.

  Port of capsuleTetrahedron from engine_collision_flex.c, the analytic counterpart of GJK/EPA:
  the closest points if the segment is outside the tetrahedron, otherwise the minimum translation
  over the separating axes (face normals and segment x edge cross products).

  Args:
    pa: First endpoint of the capsule segment.
    pb: Second endpoint of the capsule segment.
    radius: Capsule radius.
    v0: First tetrahedron vertex.
    v1: Second tetrahedron vertex.
    v2: Third tetrahedron vertex.
    v3: Fourth tetrahedron vertex.
    tet_radius: Tetrahedron (flex element) radius.

  Returns:
    - Whether the tetrahedron is nondegenerate (otherwise the distance is MJ_MAXVAL and the
      position and normal are zero).
    - Contact distance.
    - Contact position.
    - Contact normal, from the capsule to the tetrahedron.
  """
  # volume relative to the edge lengths at v0, below which the face orientations are
  # ill-conditioned in single precision; both sides scale as length^3, which does not underflow
  # for small elements
  e1 = v1 - v0
  e2 = v2 - v0
  e3 = v3 - v0
  vol6 = wp.dot(wp.cross(e1, e2), e3)
  if wp.abs(vol6) <= 1.0e-6 * wp.length(e1) * wp.length(e2) * wp.length(e3):
    return False, MJ_MAXVAL, wp.vec3(0.0), wp.vec3(0.0)

  # work relative to v0, limiting rounding at large coordinates
  a = pa - v0
  b = pb - v0
  tet = mat43f()
  tet[0] = wp.vec3(0.0)
  tet[1] = e1
  tet[2] = e2
  tet[3] = e3

  # outward face normals and offsets, face i is opposite vertex i
  fnrm = mat43f()
  foff = wp.vec4()
  for i in range(4):
    va = tet[(i + 1) % 4]
    fn = wp.normalize(wp.cross(tet[(i + 2) % 4] - va, tet[(i + 3) % 4] - va))
    if wp.dot(fn, tet[i] - va) > 0.0:
      fn = -fn
    fnrm[i] = fn
    foff[i] = wp.dot(fn, va)

  # signed distances of a and b to the faces, clip ab to the tetrahedron
  sa = wp.vec4()
  sb = wp.vec4()
  t0 = float(0.0)
  t1 = float(1.0)
  for i in range(4):
    sa[i] = wp.dot(fnrm[i], a) - foff[i]
    sb[i] = wp.dot(fnrm[i], b) - foff[i]
    if sa[i] > 0.0 and sb[i] > 0.0:
      t0 = 1.0
      t1 = 0.0
    elif sa[i] > 0.0:
      t0 = wp.max(t0, sa[i] / (sa[i] - sb[i]))
    elif sb[i] > 0.0:
      t1 = wp.min(t1, sa[i] / (sa[i] - sb[i]))

  ab = b - a
  dist = float(0.0)
  nrm = wp.vec3(0.0)
  p = a

  # separated: closest points between segment and tetrahedron faces, skipping the faces with the
  # segment on their inner side, since the closest point on the tetrahedron lies on a face with p on
  # its outer side (at least one, as the segment is outside)
  if t0 > t1:
    dist2 = float(MJ_MAXVAL)
    q = a
    face = int(0)
    feature = int(0)
    for i in range(4):
      if sa[i] <= 0.0 and sb[i] <= 0.0:
        continue
      di2, pi, qi, fi = _segment_triangle(a, b, tet[(i + 1) % 4], tet[(i + 2) % 4], tet[(i + 3) % 4])
      if di2 < dist2:
        dist2 = di2
        p = pi
        q = qi
        face = i
        feature = fi
    dist = wp.sqrt(dist2)

    if feature == 1:
      # q inside the face: face normal
      nrm = -fnrm[face]
    elif feature > 1:
      # q inside an edge and p inside the segment: cross product, toward the centroid
      j = feature - 2
      c = tet[(face + 1 + j) % 4]
      e = tet[(face + 1 + (j + 1) % 3) % 4]
      nrm = wp.normalize(wp.cross(ab, e - c))
      if wp.dot(nrm, 0.25 * (tet[0] + tet[1] + tet[2] + tet[3]) - q) < 0.0:
        nrm = -nrm
    else:
      # vertex or edge end: direction between the closest points, or the separating axis of
      # maximum separation if they are within rounding (1e-6 of the coordinates) of each other
      scale2 = wp.max(
        wp.max(wp.length_sq(e1), wp.length_sq(e2)), wp.max(wp.length_sq(e3), wp.max(wp.length_sq(a), wp.length_sq(b)))
      )
      if dist2 > 1.0e-12 * scale2:
        nrm = (q - p) / dist
      else:
        unused_depth, nrm, unused_edge = _tet_sep_axis(a, b, tet, fnrm, sa, sb)

  # intersecting: minimum translation over the separating axes
  else:
    depth, nrm, edge = _tet_sep_axis(a, b, tet, fnrm, sa, sb)
    dist = -depth

    # edge axis: closest point on segment to the edge
    if edge >= 0:
      j0, j1 = _tet_edge(edge)
      nseg, wseg0, wseg1 = _segment_weights(a, b, tet[j0], tet[j1])
      p = a + wp.where(nseg == 2, 0.5 * (wseg0[0] + wseg1[0]), wseg0[0]) * ab

    # face axis: deepest endpoint, or the middle of the clipped segment if parallel
    else:
      da = wp.dot(nrm, a)
      db = wp.dot(nrm, b)
      if da > db:
        p = a
      elif db > da:
        p = b
      else:
        p = a + 0.5 * (t0 + t1) * ab

  dist = dist - radius - tet_radius
  return True, dist, v0 + p + nrm * (radius + 0.5 * dist), nrm
