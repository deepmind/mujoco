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

"""Miscellaneous utilities."""

from typing import Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import math
from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MAXVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import ObjType
from mujoco.mjx.third_party.mujoco_warp._src.types import TrnType
from mujoco.mjx.third_party.mujoco_warp._src.types import WrapType
from mujoco.mjx.third_party.mujoco_warp._src.types import vec10


@wp.func
def is_intersect(p1: wp.vec2, p2: wp.vec2, p3: wp.vec2, p4: wp.vec2) -> bool:
  """Check for intersection of two 2D line segments.

  Args:
    p1: 2D point from segment 1
    p2: 2D point from segment 1
    p3: 2D point from segment 2
    p4: 2D point from segment 2

  Returns:
    Intersection status of line segments.
  """
  u = p2 - p1
  v = p4 - p3

  # compute determinant, check
  det = v[1] * u[0] - v[0] * u[1]
  if wp.abs(det) < MJ_MINVAL:
    return False

  # check intersection parameter bounds [0, 1] on each segment without subtracting det
  s = wp.where(det > 0.0, 1.0, -1.0)
  w = p1 - p3
  d = p2 - p4
  a0 = (v[0] * w[1] - v[1] * w[0]) * s
  a1 = (v[0] * d[1] - v[1] * d[0]) * s
  b0 = (u[0] * w[1] - u[1] * w[0]) * s
  b1 = (u[0] * d[1] - u[1] * d[0]) * s

  return a0 >= 0.0 and a1 <= 0.0 and b0 >= 0.0 and b1 <= 0.0


@wp.func
def halton(index: int, base: int) -> float:
  n0 = index
  b = float(base)
  f = float(1.0) / b
  hn = float(0.0)

  while n0 > 0:
    n1 = n0 // base
    r = n0 - n1 * base
    hn += f * float(r)
    f /= b
    n0 = n1

  return hn


@wp.func
def length_circle(p0: wp.vec2, p1: wp.vec2, ind: int, radius: float) -> float:
  """Curve length along circle.

  Args:
    p0: 2D point.
    p1: 2D point.
    ind: input for flip.
    radius: circle radius.

  Returns:
    Curve length.
  """
  # compute angle between 0 and pi
  cross = p0[1] * p1[0] - p0[0] * p1[1]
  angle = wp.atan2(wp.abs(cross), wp.dot(p0, p1))

  # flip if necessary
  if (cross > 0.0 and ind != 0) or (cross < 0.0 and ind == 0):
    angle = 2.0 * wp.pi - angle

  return radius * angle


@wp.func
def _winding_angle(p: wp.vec2, q: wp.vec2) -> float:
  """Signed angle swept from p to q around the origin, in (-pi, pi]."""
  return wp.atan2(p[0] * q[1] - p[1] * q[0], wp.dot(p, q))


@wp.func
def wrap_circle(end: wp.vec4, side: wp.vec2, radius: float) -> Tuple[float, wp.vec2, wp.vec2]:
  """2D circle wrap.

  Args:
    end: Two 2D points.
    side: Optional 2D side point, no side point: wp.vec2(MJ_MAXVAL).
    radius: Circle radius.

  Returns:
    Length of circular wrap or -1.0 if no wrap, pair of 2D wrap points.
  """
  valid_side = wp.norm_l2(side) < MJ_MAXVAL

  end0 = wp.vec2(end[0], end[1])
  end1 = wp.vec2(end[2], end[3])

  sqlen0 = wp.dot(end0, end0)
  sqlen1 = wp.dot(end1, end1)
  sqrad = radius * radius

  # either point inside circle or circle too small: no wrap
  if (sqlen0 < sqrad) or (sqlen1 < sqrad) or (radius < MJ_MINVAL):
    return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  # points too close: no wrap
  dif = end1 - end0
  dd = wp.dot(dif, dif)
  if dd < MJ_MINVAL:
    return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  # find nearest point on line segment to origin: a * dif + d0
  a = -wp.dot(dif, end0) / dd
  a = wp.clamp(a, 0.0, 1.0)

  # check for intersection
  tmp = a * dif + end0
  clear = wp.dot(tmp, tmp) > sqrad

  # side: the path is homotopic to the polyline end0 -> side -> end1, whose winding angle around
  # the center determines the wrap orientation; the segment end0 -> end1 is in the same homotopy
  # class iff its winding angle is the same (they differ by multiples of 2 * pi)
  wside = float(0.0)
  if valid_side:
    wside = _winding_angle(end0, side) + _winding_angle(side, end1)
    if clear and wp.abs(wside - _winding_angle(end0, end1)) < wp.pi:
      return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)
  elif clear:
    return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  sqrt0 = wp.sqrt(sqlen0 - sqrad)
  sqrt1 = wp.sqrt(sqlen1 - sqrad)

  # construct the two solutions, compute goodness
  sol00 = wp.vec2(
    math.safe_div(end[0] * sqrad + radius * end[1] * sqrt0, sqlen0),
    math.safe_div(end[1] * sqrad - radius * end[0] * sqrt0, sqlen0),
  )
  sol01 = wp.vec2(
    math.safe_div(end[2] * sqrad - radius * end[3] * sqrt1, sqlen1),
    math.safe_div(end[3] * sqrad + radius * end[2] * sqrt1, sqlen1),
  )

  sol10 = wp.vec2(
    math.safe_div(end[0] * sqrad - radius * end[1] * sqrt0, sqlen0),
    math.safe_div(end[1] * sqrad + radius * end[0] * sqrt0, sqlen0),
  )
  sol11 = wp.vec2(
    math.safe_div(end[2] * sqrad + radius * end[3] * sqrt1, sqlen1),
    math.safe_div(end[3] * sqrad - radius * end[2] * sqrt1, sqlen1),
  )

  # goodness: shorter path
  tmp0 = sol00 - sol01
  good0 = -wp.dot(tmp0, tmp0)
  tmp1 = sol10 - sol11
  good1 = -wp.dot(tmp1, tmp1)

  # penalize for intersection
  intersect0 = is_intersect(end0, sol00, end1, sol01)
  if intersect0:
    good0 = -10000.0
  intersect1 = is_intersect(end0, sol10, end1, sol11)
  if intersect1:
    good1 = -10000.0

  # select the solution: with side, by orientation (ind=1 is counterclockwise), otherwise the better
  if valid_side:
    ind = wp.where(wside > 0.0, 1, 0)
  else:
    ind = wp.where(good0 > good1, 0, 1)

  if ind == 0:
    intersect = intersect0
    pnt0 = sol00
    pnt1 = sol01
  else:
    intersect = intersect1
    pnt0 = sol10
    pnt1 = sol11

  # check for intersection
  if intersect:
    return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  # side: the path winds like the polyline, so the arc angle is its winding minus that of the
  # tangent segments; unlike length_circle, this is accurate for wraps close to 2 * pi
  if valid_side:
    arc = wside - _winding_angle(end0, pnt0) - _winding_angle(pnt1, end1)
    return radius * wp.abs(arc), pnt0, pnt1

  # return curve length
  return length_circle(pnt0, pnt1, ind, radius), pnt0, pnt1


@wp.func
def wrap_inside(
  # In:
  end: wp.vec4,
  radius: float,
) -> Tuple[float, wp.vec2, wp.vec2]:
  """2D inside wrap.

  Args:
    end: Two 2D points.
    radius: Circle radius.

  Returns:
    0.0 if wrap else -1.0, pair of 2D wrap points.
  """
  # algorithm parameters
  maxiter = int(20)
  tolerance = float(1.0e-6)

  end0 = wp.vec2(end[0], end[1])
  end1 = wp.vec2(end[2], end[3])

  # constants
  len0 = wp.norm_l2(end0)
  len1 = wp.norm_l2(end1)
  dif = end1 - end0
  dd = wp.dot(dif, dif)

  # either point inside circle or circle too small: no wrap
  if (len0 <= radius) or (len1 <= radius) or (radius < MJ_MINVAL) or (len0 < MJ_MINVAL) or (len1 < MJ_MINVAL):
    return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  # segment-circle intersection: no wrap
  if dd > MJ_MINVAL:
    # find nearest point on line segment to origin: d0 + a * dif
    a = -wp.dot(dif, end0) / dd

    # in segment
    if (a > 0.0) and (a < 1.0):
      tmp = end0 + a * dif
      if wp.norm_l2(tmp) <= radius:
        return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  # prepare default in case of numerical failure: average
  pnt = 0.5 * (end0 + end1)
  pnt, _ = math.normalize_with_norm(pnt)
  pnt *= radius

  # compute function parameters: asin(A * z) + asin(B * z) - 2 * asin(z) + G = 0
  A = math.safe_div(radius, len0)
  B = math.safe_div(radius, len1)
  sq_A = A * A
  sq_B = B * B
  cosG = math.safe_div(len0 * len0 + len1 * len1 - dd, 2.0 * len0 * len1)
  if cosG < -1.0 + MJ_MINVAL:
    return -1.0, pnt, pnt
  elif cosG > 1.0 - MJ_MINVAL:
    return 0.0, pnt, pnt
  G = wp.acos(cosG)

  # init: Newton on phi = asin(z) instead of z
  phi = float(wp.static(0.5 * wp.pi))
  z = float(1.0)
  asin_Az = wp.asin(A)
  asin_Bz = wp.asin(B)
  f = asin_Az + asin_Bz - wp.pi + G

  # make sure init is not on the other side
  if f > 0.0:
    return 0.0, pnt, pnt

  # Newton method
  iter = int(0)

  while (iter < maxiter) and (wp.abs(f) > tolerance):
    # derivative with respect to phi
    sq_z = z * z
    df = (
      wp.cos(phi) * (A / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_A)) + B / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_B)))
      - 2.0
    )

    # check sign; SHOULD NOT OCCUR
    if df > -MJ_MINVAL:
      return 0.0, pnt, pnt

    # new point
    phi1 = phi - math.safe_div(f, df)

    # make sure we are moving to the left; SHOULD NOT OCCUR
    if phi1 > phi:
      return 0.0, pnt, pnt

    # update solution
    phi = phi1
    z = wp.sin(phi)
    asin_Az = wp.asin(A * z)
    asin_Bz = wp.asin(B * z)
    f = asin_Az + asin_Bz - 2.0 * phi + G

    # exit if positive: SHOULD NOT OCCUR
    if f > tolerance:
      return 0.0, pnt, pnt

    iter += 1

  # check convergence
  if iter >= maxiter:
    return 0.0, pnt, pnt

  # finalize: rotation by ang from vec = a or b, depending on cross(a, b) sign
  if end[0] * end[3] - end[1] * end[2] > 0.0:
    vec = end0
    ang = phi - asin_Az
  else:
    vec = end1
    ang = phi - asin_Bz

  vec, _ = math.normalize_with_norm(vec)
  pnt = wp.vec2(
    radius * (wp.cos(ang) * vec[0] - wp.sin(ang) * vec[1]),
    radius * (wp.sin(ang) * vec[0] + wp.cos(ang) * vec[1]),
  )

  return 0.0, pnt, pnt


@wp.func
def wrap(
  x0: wp.vec3, x1: wp.vec3, pos: wp.vec3, mat: wp.mat33, radius: float, geomtype: int, side: wp.vec3
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Wrap tendons around spheres and cylinders.

  Args:
    x0: 3D endpoint.
    x1: 3D endpoint.
    pos: Position of geom.
    mat: Orientation of geom.
    radius: Geom radius.
    geomtype: Wrap type (mjtWrap).
    side: 3D position for sidesite, no side point: wp.vec3(MJ_MAXVAL).

  Returns:
    Length of circular wrap else -1.0 if no wrap, pair of 3D wrap points.
  """
  # check object type
  if geomtype != WrapType.SPHERE and geomtype != WrapType.CYLINDER:
    return MJ_MAXVAL, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)

  # map sites to wrap object's local frame
  matT = wp.transpose(mat)
  p0 = matT @ (x0 - pos)
  p1 = matT @ (x1 - pos)

  # too close to origin: return
  if (wp.norm_l2(p0) < MJ_MINVAL) or (wp.norm_l2(p1) < MJ_MINVAL):
    return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)

  # construct 2D frame for circle wrap
  if geomtype == WrapType.SPHERE:
    # 1st axis = p0
    axis0, _ = math.normalize_with_norm(p0)

    # normal to p0-0-p1 plane = cross(p0, p1)
    normal = wp.cross(p0, p1)
    normal, nrm = math.normalize_with_norm(normal)

    # if (p0, p1) parallel: different normal
    if nrm < MJ_MINVAL:
      # find max component of axis0
      axis0_abs = wp.abs(axis0)
      i = int(0)
      if (axis0_abs[1] > axis0_abs[0]) and (axis0_abs[1] > axis0_abs[2]):
        i = 1
      if (axis0_abs[2] > axis0_abs[0]) and (axis0_abs[2] > axis0_abs[1]):
        i = 2

      # init second axis: 0 at i; 1 elsewhere
      axis1 = wp.vec3(1.0)
      axis1[i] = 0.0

      # recompute normal
      normal = wp.cross(axis0, axis1)
      normal, _ = math.normalize_with_norm(normal)

    # 2nd axis = cross(normal, p0)
    axis1 = wp.cross(normal, axis0)
    axis1, _ = math.normalize_with_norm(axis1)
  else:  # WrapType.CYLINDER
    # 1st axis = x
    axis0 = wp.vec3(1.0, 0.0, 0.0)

    # 2nd axis = y
    axis1 = wp.vec3(0.0, 1.0, 0.0)

  # project points in 2D frame: p => end
  end = wp.vec4(
    wp.dot(p0, axis0),
    wp.dot(p0, axis1),
    wp.dot(p1, axis0),
    wp.dot(p1, axis1),
  )

  # handle sidesite
  valid_side = wp.norm_l2(side) < MJ_MAXVAL

  if valid_side:
    # side point: apply same projection as x0, x1
    sidepnt = matT @ (side - pos)

    # side point: project and rescale
    sidepnt_proj = wp.vec2(
      wp.dot(sidepnt, axis0),
      wp.dot(sidepnt, axis1),
    )

    sidepnt_proj, sidenrm = math.normalize_with_norm(sidepnt_proj)
    if sidenrm < MJ_MINVAL:
      sidepnt_proj = wp.vec2(1.0, 0.0)
    sidepnt_proj *= radius
  else:
    sidepnt_proj = wp.vec2(MJ_MAXVAL)

  # apply inside wrap
  if valid_side and wp.norm_l2(sidepnt) < radius:
    wlen, pnt0, pnt1 = wrap_inside(end, radius)
  else:  # apply circle wrap
    wlen, pnt0, pnt1 = wrap_circle(end, sidepnt_proj, radius)

  # no wrap: return
  if wlen < 0.0:
    return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)

  # reconstruct 3D points in local frame: res
  res0 = axis0 * pnt0[0] + axis1 * pnt0[1]
  res1 = axis0 * pnt1[0] + axis1 * pnt1[1]

  # cylinder: correct along z
  if geomtype == WrapType.CYLINDER:
    # set vertical coordinates
    L0 = wp.sqrt((p0[0] - res0[0]) * (p0[0] - res0[0]) + (p0[1] - res0[1]) * (p0[1] - res0[1]))
    L1 = wp.sqrt((p1[0] - res1[0]) * (p1[0] - res1[0]) + (p1[1] - res1[1]) * (p1[1] - res1[1]))
    res0[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0, L0 + wlen + L1)
    res1[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0 + wlen, L0 + wlen + L1)

    # correct wlen for height
    height = wp.abs(res1[2] - res0[2])
    wlen = wp.sqrt(wlen * wlen + height * height)

  # map back to global frame: wpnt
  wpnt0 = mat @ res0 + pos
  wpnt1 = mat @ res1 + pos

  return wlen, wpnt0, wpnt1


@wp.func
def muscle_gain_length(length: float, lmin: float, lmax: float) -> float:
  """Normalized muscle length-gain curve."""
  if (lmin > length) or (length > lmax):
    return 0.0

  # mid-ranges (maximum is at 1.0)
  a = 0.5 * (lmin + 1.0)
  b = 0.5 * (1.0 + lmax)

  if length <= a:
    x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)
    return 0.5 * x * x
  elif length <= 1.0:
    x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)
    return 1.0 - 0.5 * x * x
  elif length <= b:
    x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)
    return 1.0 - 0.5 * x * x
  else:
    x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)
    return 0.5 * x * x


@wp.func
def muscle_gain(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:
  """Muscle active force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)."""
  # unpack parameters
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmin = prm[4]
  lmax = prm[5]
  vmax = prm[6]
  fvmax = prm[8]

  # scale force if negative
  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  # optimum length
  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])

  # normalized length and velocity
  L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)
  V = vel / wp.max(MJ_MINVAL, L0 * vmax)

  # length curve
  FL = muscle_gain_length(L, lmin, lmax)

  # velocity curve
  y = fvmax - 1.0
  if V <= -1.0:
    FV = 0.0
  elif V <= 0.0:
    FV = (V + 1.0) * (V + 1.0)
  elif V <= y:
    FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)
  else:
    FV = fvmax

  # compute FVL and scale, make it negative
  return -force * FL * FV


@wp.func
def muscle_bias(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:
  """Calculates muscle passive force.

  prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax).
  """
  # unpack parameters
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmax = prm[5]
  fpmax = prm[7]

  # scale force if negative
  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  # optimum length
  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])

  # normalized length
  L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)

  # half-quadratic to (L0 + lmax) / 2, linear beyond
  b = 0.5 * (1.0 + lmax)
  if L <= 1.0:
    return 0.0
  elif L <= b:
    x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)
    return -force * fpmax * 0.5 * x * x
  else:
    x = (L - b) / wp.max(MJ_MINVAL, b - 1.0)
    return -force * fpmax * (0.5 + x)


@wp.func
def muscle_gain_length_deriv(length: float, lmin: float, lmax: float) -> float:
  """Derivative of muscle_gain_length w.r.t length."""
  if (lmin > length) or (length > lmax):
    return 0.0

  a = 0.5 * (lmin + 1.0)
  b = 0.5 * (1.0 + lmax)

  if length <= a:
    x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)
    return x / wp.max(MJ_MINVAL, a - lmin)
  elif length <= 1.0:
    x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)
    return x / wp.max(MJ_MINVAL, 1.0 - a)
  elif length <= b:
    x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)
    return -x / wp.max(MJ_MINVAL, b - 1.0)
  else:
    x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)
    return -x / wp.max(MJ_MINVAL, lmax - b)


@wp.func
def muscle_gain_len_deriv(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:
  """Derivative of muscle_gain w.r.t length."""
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmin = prm[4]
  lmax = prm[5]
  vmax = prm[6]
  fvmax = prm[8]

  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])
  L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)
  V = vel / wp.max(MJ_MINVAL, L0 * vmax)

  dFL = muscle_gain_length_deriv(L, lmin, lmax)

  y = fvmax - 1.0
  if V <= -1.0:
    FV = 0.0
  elif V <= 0.0:
    FV = (V + 1.0) * (V + 1.0)
  elif V <= y:
    FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)
  else:
    FV = fvmax

  return -force * dFL * FV / wp.max(MJ_MINVAL, L0)


@wp.func
def muscle_bias_len_deriv(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:
  """Derivative of muscle_bias w.r.t length."""
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmax = prm[5]
  fpmax = prm[7]

  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])
  L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)

  b = 0.5 * (1.0 + lmax)
  if L <= 1.0:
    dFP = 0.0
  elif L <= b:
    x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)
    dFP = x / wp.max(MJ_MINVAL, b - 1.0)
  else:
    dFP = 1.0 / wp.max(MJ_MINVAL, b - 1.0)

  return -force * fpmax * dFP / wp.max(MJ_MINVAL, L0)


@wp.func
def muscle_gain_vel_deriv(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:
  """Derivative of muscle_gain w.r.t velocity."""
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmin = prm[4]
  lmax = prm[5]
  vmax = prm[6]
  fvmax = prm[8]

  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])
  L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)
  V = vel / wp.max(MJ_MINVAL, L0 * vmax)

  FL = muscle_gain_length(L, lmin, lmax)

  y = fvmax - 1.0
  if V <= -1.0:
    dFV = 0.0
  elif V <= 0.0:
    dFV = 2.0 * V + 2.0
  elif V <= y:
    dFV = (-2.0 * V + 2.0 * y) / wp.max(MJ_MINVAL, y)
  else:
    dFV = 0.0

  return -force * FL * dFV / wp.max(MJ_MINVAL, L0 * vmax)


@wp.func
def _sigmoid(x: float) -> float:
  """Sigmoid function over 0 <= x <= 1 using quintic polynomial."""
  if x <= 0.0:
    return 0.0

  if x >= 1.0:
    return 1.0

  # sigmoid f(x) = 6 * x^5 - 15 * x^4 + 10 * x^3
  # solution of f(0) = f'(0) = f''(0) = 0, f(1) = 1, f'(1) = f''(1) = 0
  return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)


@wp.func
def muscle_dynamics_timescale(dctrl: float, tau_act: float, tau_deact: float, smooth_width: float) -> float:
  """Muscle time constant with optional smoothing."""
  # hard switching
  if smooth_width < MJ_MINVAL:
    if dctrl > 0.0:
      return tau_act
    else:
      return tau_deact
  else:  # smooth switching
    # scale by width, center around 0.5 midpoint, rescale to bounds
    return tau_deact + (tau_act - tau_deact) * _sigmoid(dctrl / smooth_width + 0.5)


@wp.func
def muscle_dynamics(control: float, activation: float, prm: vec10) -> float:
  """Muscle activation dynamics, prm = (tau_act, tau_deact, smooth_width)."""
  # clamp control
  ctrlclamp = wp.clamp(control, 0.0, 1.0)

  # clamp activation
  actclamp = wp.clamp(activation, 0.0, 1.0)

  # compute timescales as in Millard et al. (2013) https://doi.org/10.1115/1.4023390
  tau_act = prm[0] * (0.5 + 1.5 * actclamp)  # activation timescale
  tau_deact = prm[1] / (0.5 + 1.5 * actclamp)  # deactivation timescale
  smooth_width = prm[2]  # width of smoothing sigmoid
  dctrl = ctrlclamp - activation  # excess excitation

  tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smooth_width)

  # filter output
  return dctrl / wp.max(MJ_MINVAL, tau)


@wp.func
def dcmotor_slots(dynprm: types.vec10, gainprm: types.vec10) -> types.vec6i:
  """Compute activation slot layout for a DC motor actuator.

  Each DC motor can have up to 5 optional activation states. This function
  determines which states are enabled (based on nonzero parameters) and
  assigns each a contiguous slot offset in the activation array.

  Returns a vec6i where:
    s[0]: slew rate     — enabled when dynprm[7] > 0  (slew rate limit)
    s[1]: integral      — enabled when gainprm[5] > 0 (integral gain ki)
    s[2]: temperature   — enabled when dynprm[2] > 0  (thermal resistance RT)
    s[3]: bristle       — enabled when dynprm[5] > 0  (LuGre stiffness sigma0)
    s[4]: current       — enabled when dynprm[0] > 0  (electrical time const te)
    s[5]: total number of active slots (num_slots)

  Enabled slots hold a contiguous offset (0, 1, 2, ...); disabled slots
  are set to -1.
  """
  s = types.vec6i(-1, -1, -1, -1, -1, 0)
  num_slots = 0
  if dynprm[7] > 0.0:
    s[0] = num_slots
    num_slots += 1
  if gainprm[5] > 0.0:
    s[1] = num_slots
    num_slots += 1
  if dynprm[2] > 0.0:
    s[2] = num_slots
    num_slots += 1
  if dynprm[5] > 0.0:
    s[3] = num_slots
    num_slots += 1
  if dynprm[0] > 0.0:
    s[4] = num_slots
    num_slots += 1
  s[5] = num_slots
  return s


@wp.func
def dcmotor_resistance(
  # Data in:
  act_in: wp.array2d[float],
  # In:
  worldid: int,
  actadr: int,
  dynprm: types.vec10,
  gainprm: types.vec10,
) -> float:
  """Compute DC motor winding resistance at operating temperature."""
  R = gainprm[0]
  if actadr >= 0 and dynprm[2] > 0.0:
    slots = dcmotor_slots(dynprm, gainprm)
    T = act_in[worldid, actadr + slots[2]]
    alpha = gainprm[2]
    T0 = gainprm[3]
    Ta = dynprm[4]
    R *= 1.0 + alpha * (T + Ta - T0)
  return wp.max(MJ_MINVAL, R)


@wp.func
def dcmotor_mech_force(
  # Data in:
  act_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  # In:
  worldid: int,
  actadr: int,
  length: float,
  dynprm: types.vec10,
  gainprm: types.vec10,
  biasprm: types.vec10,
) -> float:
  """Compute DC motor mechanical force (cogging torque and LuGre friction)."""
  mech = 0.0

  # cogging torque
  A = biasprm[0]
  if A != 0.0:
    mech += A * wp.sin(biasprm[1] * length + biasprm[2])

  # LuGre friction
  sigma0 = dynprm[5]
  if actadr >= 0 and sigma0 > 0.0:
    sigma1 = dynprm[6]
    slots = dcmotor_slots(dynprm, gainprm)
    adr = actadr + slots[3]
    mech -= sigma0 * act_in[worldid, adr] + sigma1 * act_dot_in[worldid, adr]

  return mech


@wp.func
def lugre_stribeck(velocity: float, F_C: float, F_S: float, v_S: float) -> float:
  ratio = velocity / wp.max(MJ_MINVAL, v_S)
  return F_C + (F_S - F_C) * wp.exp(-ratio * ratio)


@wp.func
def unpack_servo_inputs(
  # Data in:
  ctrl_in: wp.array2d[float],
  # In:
  worldid: int,
  uadr: int,
  spec: int,
  u_first: float,
) -> wp.vec4:
  out = wp.vec4(0.0, 0.0, 0.0, 0.0)
  adr = uadr
  first_set = False
  if (spec & 1) != 0:  # POS
    out[0] = u_first if not first_set else ctrl_in[worldid, adr]
    first_set = True
    adr += 1
  if (spec & 2) != 0:  # VEL
    out[1] = u_first if not first_set else ctrl_in[worldid, adr]
    first_set = True
    adr += 1
  if (spec & 4) != 0:  # FF
    out[2] = u_first if not first_set else ctrl_in[worldid, adr]
    first_set = True
    adr += 1
  if (spec & 8) != 0:  # VOLTAGE
    out[3] = u_first if not first_set else ctrl_in[worldid, adr]
  return out


@wp.func
def dcmotor_voltage(
  # Data in:
  ctrl_in: wp.array2d[float],
  # In:
  worldid: int,
  uadr: int,
  spec: int,
  u_first: float,
  length: float,
  velocity: float,
  x_I: float,
  gainprm: types.vec10,
) -> float:
  voltage = 0.0
  u4 = unpack_servo_inputs(ctrl_in, worldid, uadr, spec, u_first)

  if (spec & 7) != 0:  # POS (1) | VEL (2) | FF (4)
    kp = gainprm[4]
    ki = gainprm[5]
    kd = gainprm[6]
    torque = kp * (u4[0] - length) + kd * (u4[1] - velocity) + ki * x_I + u4[2]

    R = gainprm[0]
    K = gainprm[1]
    voltage = (R / K) * torque + K * velocity

    Vmax = gainprm[7]
    if Vmax > 0.0:
      voltage = wp.clamp(voltage, -Vmax, Vmax)

    return voltage + u4[3]
  else:
    return u_first


@wp.func
def inside_geom(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, geomtype: int, point: wp.vec3) -> bool:
  """Return True if point is inside primitive geom, False otherwise."""
  # vector from geom to point
  vec = point - pos

  # quick return for spheres, frame rotation not required
  if geomtype == GeomType.SPHERE:
    return wp.dot(vec, vec) < size[0] * size[0]

  # rotate into local frame
  plocal = wp.transpose(mat) @ vec

  # handle other geom types
  if geomtype == GeomType.CAPSULE:
    z = plocal[2]
    z_clamped = wp.clamp(z, -size[1], size[1])
    z_dif = z - z_clamped
    z_dist_sq = z_dif * z_dif
    return plocal[0] * plocal[0] + plocal[1] * plocal[1] + z_dist_sq < size[0] * size[0]
  elif geomtype == GeomType.ELLIPSOID:
    plocalsize = wp.cw_div(plocal, size)
    return wp.dot(plocalsize, plocalsize) < 1.0
  elif geomtype == GeomType.CYLINDER:
    return (wp.abs(plocal[2]) < size[1]) and (plocal[0] * plocal[0] + plocal[1] * plocal[1] < size[0] * size[0])
  elif geomtype == GeomType.BOX:
    return wp.abs(plocal[0]) < size[0] and wp.abs(plocal[1]) < size[1] and wp.abs(plocal[2]) < size[2]
  elif geomtype == GeomType.PLANE:
    return plocal[2] < 0.0

  return False


@wp.func
def _poly_force(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:
  x_val = wp.where(flg_odd == 1, wp.abs(x), x)
  res = linear
  res += poly[0] * x_val
  res += poly[1] * x_val * x_val
  return res


@wp.func
def _poly_force_deriv(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:
  x_val = wp.where(flg_odd == 1, wp.abs(x), x)
  res = linear
  res += 2.0 * poly[0] * x_val
  res += 3.0 * poly[1] * x_val * x_val
  return res


@wp.func
def poly_potential(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:
  x_val = wp.where(flg_odd == 1, wp.abs(x), x)
  x_val2 = x_val * x_val
  x_val3 = x_val2 * x_val
  x_val4 = x_val3 * x_val

  res = 0.5 * linear * x_val2
  res += poly[0] * wp.static(1.0 / 3.0) * x_val3
  res += poly[1] * 0.25 * x_val4
  return res


@wp.func
def _actuator_target(
  # Model:
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  # In:
  objtype: int,
  objid: int,
  actuatorid: int,
) -> bool:
  """Returns whether an actuator's transmission target is a given joint or tendon."""
  if actuator_trnid[actuatorid][0] != objid:
    return False
  trntype = actuator_trntype[actuatorid]
  if objtype == ObjType.JOINT:
    return trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT
  return trntype == TrnType.TENDON


@wp.func
def actuator_target_damping(
  # Model:
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objtype: int,
  objid: int,
  actuatorid: int,
) -> Tuple[float, wp.vec2]:
  """Returns actuator damping and high-order damping contributions to a joint or tendon.

  Contributions are scaled by gear^2. actuatorid is jnt_actuatorid or tendon_actuatorid of objid:
  -1: no actuator, >= 0: single actuator, < -1: scan all actuators.
  """
  damping = float(0.0)
  dpoly = wp.vec2(0.0)

  # single actuator contributes damping
  if actuatorid >= 0:
    gear = actuator_gear[worldid % actuator_gear.shape[0], actuatorid][0]
    damping = actuator_damping[worldid % actuator_damping.shape[0], actuatorid] * gear * gear
    dpoly = actuator_dampingpoly[worldid % actuator_dampingpoly.shape[0], actuatorid] * gear * gear

  # scan all actuators for contributions
  elif actuatorid < -1:
    for k in range(actuator_trntype.shape[0]):
      if not _actuator_target(actuator_trntype, actuator_trnid, objtype, objid, k):
        continue
      gear = actuator_gear[worldid % actuator_gear.shape[0], k][0]
      damping += actuator_damping[worldid % actuator_damping.shape[0], k] * gear * gear
      dpoly += actuator_dampingpoly[worldid % actuator_dampingpoly.shape[0], k] * gear * gear

  return damping, dpoly


@wp.func
def actuator_target_armature(
  # Model:
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array2d[float],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  objtype: int,
  objid: int,
  actuatorid: int,
) -> float:
  """Returns actuator armature contribution to a joint or tendon.

  Contributions are scaled by gear^2. actuatorid is jnt_actuatorid or tendon_actuatorid of objid:
  -1: no actuator, >= 0: single actuator, < -1: scan all actuators.
  """
  armature = float(0.0)

  # single actuator contributes armature
  if actuatorid >= 0:
    gear = actuator_gear[worldid % actuator_gear.shape[0], actuatorid][0]
    armature = actuator_armature[worldid % actuator_armature.shape[0], actuatorid] * gear * gear

  # scan all actuators for contributions
  elif actuatorid < -1:
    for k in range(actuator_trntype.shape[0]):
      if not _actuator_target(actuator_trntype, actuator_trnid, objtype, objid, k):
        continue
      gear = actuator_gear[worldid % actuator_gear.shape[0], k][0]
      armature += actuator_armature[worldid % actuator_armature.shape[0], k] * gear * gear

  return armature


@wp.func
def dof_damping_total(
  # Model:
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  dofid: int,
) -> Tuple[float, wp.vec2]:
  """Returns dof damping and high-order damping, including actuator contributions."""
  jntid = dof_jntid[dofid]
  damping, dpoly = actuator_target_damping(
    actuator_trntype,
    actuator_trnid,
    actuator_damping,
    actuator_dampingpoly,
    actuator_gear,
    worldid,
    ObjType.JOINT,
    jntid,
    jnt_actuatorid[jntid],
  )
  damping += dof_damping[worldid % dof_damping.shape[0], dofid]
  dpoly += dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofid]
  return damping, dpoly


@wp.func
def tendon_damping_total(
  # Model:
  tendon_actuatorid: wp.array[int],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  tenid: int,
) -> Tuple[float, wp.vec2]:
  """Returns tendon damping and high-order damping, including actuator contributions."""
  damping, dpoly = actuator_target_damping(
    actuator_trntype,
    actuator_trnid,
    actuator_damping,
    actuator_dampingpoly,
    actuator_gear,
    worldid,
    ObjType.TENDON,
    tenid,
    tendon_actuatorid[tenid],
  )
  damping += tendon_damping[worldid % tendon_damping.shape[0], tenid]
  dpoly += tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]
  return damping, dpoly


@wp.func
def dof_armature_total(
  # Model:
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_armature: wp.array2d[float],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array2d[float],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  dofid: int,
) -> float:
  """Returns dof armature, including actuator contributions."""
  jntid = dof_jntid[dofid]
  armature = actuator_target_armature(
    actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, ObjType.JOINT, jntid, jnt_actuatorid[jntid]
  )
  return dof_armature[worldid % dof_armature.shape[0], dofid] + armature


@wp.func
def tendon_armature_total(
  # Model:
  tendon_actuatorid: wp.array[int],
  tendon_armature: wp.array2d[float],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array2d[float],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  tenid: int,
) -> float:
  """Returns tendon armature, including actuator contributions."""
  armature = actuator_target_armature(
    actuator_trntype, actuator_trnid, actuator_armature, actuator_gear, worldid, ObjType.TENDON, tenid, tendon_actuatorid[tenid]
  )
  return tendon_armature[worldid % tendon_armature.shape[0], tenid] + armature
