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
"""Some useful math functions."""

from typing import Optional, Tuple, Union

import jax
from jax import numpy as jp

import mujoco


def safe_div(
    num: Union[float, jax.Array], den: Union[float, jax.Array]
) -> Union[float, jax.Array]:
  """Safe division for case where denominator is zero."""
  return num / (den + mujoco.mjMINVAL * (den == 0))


def matmul_unroll(a: jax.Array, b: jax.Array) -> jax.Array:
  """Calculates a @ b via explicit cell value operations.

  This is faster than XLA matmul for small matrices (e.g. 3x3, 4x4).

  Args:
    a: left hand of matmul operand
    b: right hand of matmul operand

  Returns:
    the matrix product of the inputs.
  """
  c = []
  for i in range(a.shape[0]):
    row = []
    for j in range(b.shape[1]):
      s = 0.0
      for k in range(a.shape[1]):
        s += a[i, k] * b[k, j]
      row.append(s)
    c.append(row)

  return jp.array(c)


def norm(
    x: jax.Array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> jax.Array:
  """Calculates a linalg.norm(x) that's safe for gradients at x=0.

  Avoids a poorly defined gradient for jnp.linal.norm(0) see
  https://github.com/jax-ml/jax/issues/3058 for details
  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm

  Returns:
    Norm of the array x.
  """

  is_zero = jp.all(jp.isclose(x, 0.0), axis=axis, keepdims=True)
  # temporarily swap x with ones if is_zero, then swap back
  x = jp.where(is_zero, jp.ones_like(x), x)
  n = jp.linalg.norm(x, axis=axis)
  n = jp.where(jp.squeeze(is_zero, axis=axis), 0.0, n)
  return n


def normalize_with_norm(
    x: jax.Array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> Tuple[jax.Array, jax.Array]:
  """Normalizes an array.

  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm

  Returns:
    A tuple of (normalized array x, the norm).
  """
  n = norm(x, axis=axis)
  denominator = n + 1e-6 * (n == 0.0)
  if axis is not None:
    denominator = jp.expand_dims(denominator, axis)
  x = x / denominator
  return x, n


def normalize(
    x: jax.Array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> jax.Array:
  """Normalizes an array.

  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm

  Returns:
    normalized array x
  """
  return normalize_with_norm(x, axis=axis)[0]


def rotate(vec: jax.Array, quat: jax.Array) -> jax.Array:
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise ValueError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (jp.dot(u, vec) * u) + (s * s - jp.dot(u, u)) * vec
  r = r + 2 * s * jp.cross(u, vec)
  return r


def quat_inv(q: jp.ndarray) -> jp.ndarray:
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * jp.array([1, -1, -1, -1])


def quat_sub(u: jax.Array, v: jax.Array) -> jax.Array:
  """Subtracts two quaternions (u - v) as a 3D velocity."""
  q = quat_mul(quat_inv(v), u)
  axis, angle = quat_to_axis_angle(q)
  return axis * angle


def quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
  """Multiplies two quaternions.

  Args:
    u: (4,) quaternion (w,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return jp.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])


def quat_mul_axis(q: jax.Array, axis: jax.Array) -> jax.Array:
  """Multiplies a quaternion and an axis.

  Args:
    q: (4,) quaternion (w,x,y,z)
    axis: (3,) axis (x,y,z)

  Returns:
    A quaternion q * axis
  """
  return jp.array([
      -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
      q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
      q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
      q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
  ])


# TODO(erikfrey): benchmark this against brax's quat_to_3x3
def quat_to_mat(q: jax.Array) -> jax.Array:
  """Converts a quaternion into a 9-dimensional rotation matrix."""
  q = jp.outer(q, q)

  return jp.array([
      [
          q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
          2 * (q[1, 2] - q[0, 3]),
          2 * (q[1, 3] + q[0, 2]),
      ],
      [
          2 * (q[1, 2] + q[0, 3]),
          q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
          2 * (q[2, 3] - q[0, 1]),
      ],
      [
          2 * (q[1, 3] - q[0, 2]),
          2 * (q[2, 3] + q[0, 1]),
          q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
      ],
  ])


def quat_to_axis_angle(q: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Converts a quaternion into axis and angle."""
  axis, sin_a_2 = normalize_with_norm(q[1:])
  angle = 2 * jp.arctan2(sin_a_2, q[0])
  angle = jp.where(angle > jp.pi, angle - 2 * jp.pi, angle)

  return axis, angle


def axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
  """Provides a quaternion that describes rotating around axis by angle.

  Args:
    axis: (3,) axis (x,y,z)
    angle: () float angle to rotate by

  Returns:
    A quaternion that rotates around axis by angle
  """
  s, c = jp.sin(angle * 0.5), jp.cos(angle * 0.5)
  return jp.insert(axis * s, 0, c)


def quat_integrate(q: jax.Array, v: jax.Array, dt: jax.Array) -> jax.Array:
  """Integrates a quaternion given angular velocity and dt."""
  v, norm_ = normalize_with_norm(v)
  angle = dt * norm_
  q_res = axis_angle_to_quat(v, angle)
  q_res = quat_mul(q, q_res)
  return normalize(q_res)


# minimax polynomial coefficients (in x^2, lowest order first) of the Bortz
# coefficient below |x| = 1.75, see mju_derivExpmap
_BORTZ_COEF_F64 = (
    0.083333333333333384,
    0.0013888888888859755,
    3.3068783094433201e-05,
    8.2671949071520734e-07,
    2.0876900642039142e-08,
    5.2828494303844588e-10,
    1.3455485951057888e-11,
    3.1615515500778219e-13,
    1.230313397310169e-14,
)
_BORTZ_COEF_F32 = (
    0.0833333358,
    0.00138884981,
    3.32480922e-05,
    4.57548566e-07,
    4.23758962e-07,
    -2.48791736e-07,
    8.78585737e-08,
    -1.64186869e-08,
    1.26339661e-09,
)


def deriv_expmap(expmap: jax.Array, vel: jax.Array) -> jax.Array:
  """Time-derivative of expmap in quat = quat0 * exp(expmap).

  Given the rotational velocity vel in the rotated frame, computes
  vel + 1/2 expmap x vel + c expmap x (expmap x vel), with the Bortz
  coefficient c = (1 - (x/2) cot(x/2)) / x^2, x = |expmap|.

  expmap is first replaced with the equivalent rotation vector of angle in
  (-pi, pi], since c is singular at multiples of 2*pi. Below x = 1.75, where
  the closed form loses accuracy to cancellation, c is a minimax polynomial in
  x^2; with the closed form above, the error is below 6 ulp on [-pi, pi].

  Port of mju_derivExpmap (MuJoCo engine_util_spatial.c), using the
  coefficients for the dtype of expmap.

  Args:
    expmap: (3,) rotation vector
    vel: (3,) rotational velocity in the rotated frame

  Returns:
    (3,) time-derivative of expmap
  """
  # reduce the angle to (-pi, pi]; double where keeps gradients finite at 0
  xx = jp.dot(expmap, expmap)
  reduce = xx > jp.pi * jp.pi
  x = jp.sqrt(jp.where(reduce, xx, 1.0))
  scale = (x - 2 * jp.pi * jp.floor(x / (2 * jp.pi) + 0.5)) / x
  r = jp.where(reduce, expmap * scale, expmap)
  xx = jp.dot(r, r)

  # Bortz coefficient: minimax polynomial (Horner form), or closed form
  coef = _BORTZ_COEF_F64 if r.dtype == jp.float64 else _BORTZ_COEF_F32
  poly = jp.zeros_like(xx)
  for k in reversed(coef):
    poly = k + xx * poly
  small = xx < 1.75 * 1.75
  xx_closed = jp.where(small, 4.0, xx)
  x_closed = jp.sqrt(xx_closed)
  closed = (1 - 0.5 * x_closed / jp.tan(0.5 * x_closed)) / xx_closed
  c = jp.where(small, poly, closed)

  rxv = jp.cross(r, vel)
  return vel + 0.5 * rxv + c * jp.cross(r, rxv)


def inert_mul(i: jax.Array, v: jax.Array) -> jax.Array:
  """Multiply inertia by motion, producing force.

  Args:
    i: (10,) inertia (inertia matrix, position, mass)
    v: (6,) spatial motion

  Returns:
    resultant force
  """
  tri_id = jp.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])  # cinert inr order
  inr, pos, mass = i[tri_id], i[6:9], i[9]
  ang = jp.dot(inr, v[:3]) + jp.cross(pos, v[3:])
  vel = mass * v[3:] - jp.cross(pos, v[:3])
  return jp.concatenate((ang, vel))


def sign(x: jax.Array) -> jax.Array:
  """Returns the sign of x in the set {-1, 1}."""
  return jp.where(x < 0, -1, 1)


def transform_motion(vel: jax.Array, offset: jax.Array, rotmat: jax.Array):
  """Transform spatial motion.

  Args:
    vel: (6,) spatial motion (3 angular, 3 linear)
    offset: (3,) translation
    rotmat: (3, 3) rotation

  Returns:
    6d spatial velocity
  """
  # TODO(robotics-simulation): are quaternions faster here
  ang, vel = vel[:3], vel[3:]
  vel = rotmat.T @ (vel - jp.cross(offset, ang))
  ang = rotmat.T @ ang
  return jp.concatenate([ang, vel])


def motion_cross(u, v):
  """Cross product of two motions.

  Args:
    u: (6,) spatial motion
    v: (6,) spatial motion

  Returns:
    resultant spatial motion
  """
  ang = jp.cross(u[:3], v[:3])
  vel = jp.cross(u[3:], v[:3]) + jp.cross(u[:3], v[3:])
  return jp.concatenate((ang, vel))


def motion_cross_force(v, f):
  """Cross product of a motion and force.

  Args:
    v: (6,) spatial motion
    f: (6,) force

  Returns:
    resultant force
  """
  ang = jp.cross(v[:3], f[:3]) + jp.cross(v[3:], f[3:])
  vel = jp.cross(v[:3], f[3:])
  return jp.concatenate((ang, vel))


def orthogonals(a: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Returns orthogonal vectors `b` and `c`, given a vector `a`."""
  y, z = jp.array([0, 1, 0]), jp.array([0, 0, 1])
  b = jp.where((-0.5 < a[1]) & (a[1] < 0.5), y, z)
  b = b - a * a.dot(b)
  # normalize b. however if a is a zero vector, zero b as well.
  b = normalize(b) * jp.any(a)
  return b, jp.cross(a, b)


def make_frame(a: jax.Array) -> jax.Array:
  """Makes a right-handed 3D frame given a direction."""
  a = normalize(a)
  b, c = orthogonals(a)
  return jp.array([a, b, c])


# Geometry.


def closest_segment_point(
    a: jax.Array, b: jax.Array, pt: jax.Array
) -> jax.Array:
  """Returns the closest point on the a-b line segment to a point pt."""
  ab = b - a
  t = jp.dot(pt - a, ab) / (jp.dot(ab, ab) + 1e-6)
  return a + jp.clip(t, 0.0, 1.0) * ab


def closest_segment_point_and_dist(
    a: jax.Array, b: jax.Array, pt: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Returns closest point on the line segment and the distance squared."""
  closest = closest_segment_point(a, b, pt)
  dist = (pt - closest).dot(pt - closest)
  return closest, dist


def closest_segment_to_segment_points(
    a0: jax.Array, a1: jax.Array, b0: jax.Array, b1: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Returns closest points between two line segments."""
  # Gets the closest segment points by first finding the closest points
  # between two lines. Points are then clipped to be on the line segments
  # and edge cases with clipping are handled.
  dir_a, len_a = normalize_with_norm(a1 - a0)
  dir_b, len_b = normalize_with_norm(b1 - b0)

  # Segment mid-points.
  half_len_a = len_a * 0.5
  half_len_b = len_b * 0.5
  a_mid = a0 + dir_a * half_len_a
  b_mid = b0 + dir_b * half_len_b

  # Translation between two segment mid-points.
  trans = a_mid - b_mid

  # Parametrize points on each line as follows:
  #  point_on_a = a_mid + t_a * dir_a
  #  point_on_b = b_mid + t_b * dir_b
  # and analytically minimize the distance between the two points.
  dira_dot_dirb = dir_a.dot(dir_b)
  dira_dot_trans = dir_a.dot(trans)
  dirb_dot_trans = dir_b.dot(trans)
  denom = 1 - dira_dot_dirb * dira_dot_dirb

  orig_t_a = (-dira_dot_trans + dira_dot_dirb * dirb_dot_trans) / (denom + 1e-6)
  orig_t_b = dirb_dot_trans + orig_t_a * dira_dot_dirb
  t_a = jp.clip(orig_t_a, -half_len_a, half_len_a)
  t_b = jp.clip(orig_t_b, -half_len_b, half_len_b)

  best_a = a_mid + dir_a * t_a
  best_b = b_mid + dir_b * t_b

  # Resolve edge cases where both closest points are clipped to the segment
  # endpoints by recalculating the closest segment points for the current
  # clipped points, and then picking the pair of points with smallest
  # distance. An example of this edge case is when lines intersect but line
  # segments don't.
  new_a, d1 = closest_segment_point_and_dist(a0, a1, best_b)
  new_b, d2 = closest_segment_point_and_dist(b0, b1, best_a)
  best_a = jp.where(d1 < d2, new_a, best_a)
  best_b = jp.where(d1 < d2, best_b, new_b)

  return best_a, best_b
