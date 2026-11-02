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

from typing import Any, Tuple

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import types


@wp.func
def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:
  return wp.quat(
    u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
    u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
    u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
    u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  )


@wp.func
def quat_mul_axis(q: wp.quat, axis: wp.vec3f) -> wp.quat:
  """Multiplies a quaternion and an axis."""
  return wp.quat(
    -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
    q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
    q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
    q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
  )


@wp.func
def rot_vec_quat(vec: wp.vec3, quat: wp.quat) -> wp.vec3:
  s, u = quat[0], wp.vec3(quat[1], quat[2], quat[3])
  r = 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec
  r = r + 2.0 * s * wp.cross(u, vec)
  return r


@wp.func
def axis_angle_to_quat(axis: wp.vec3, angle: float) -> wp.quat:
  s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)
  axis = axis * s
  return wp.quat(c, axis[0], axis[1], axis[2])


@wp.func
def quat_to_mat(quat: wp.quat) -> wp.mat33:
  """Converts a quaternion into 3x3 rotation matrix."""
  q00 = quat[0] * quat[0]
  q01 = quat[0] * quat[1]
  q02 = quat[0] * quat[2]
  q03 = quat[0] * quat[3]
  q11 = quat[1] * quat[1]
  q12 = quat[1] * quat[2]
  q13 = quat[1] * quat[3]
  q22 = quat[2] * quat[2]
  q23 = quat[2] * quat[3]
  q33 = quat[3] * quat[3]

  return wp.mat33(
    q00 + q11 - q22 - q33,
    2.0 * (q12 - q03),
    2.0 * (q13 + q02),
    2.0 * (q12 + q03),
    q00 - q11 + q22 - q33,
    2.0 * (q23 - q01),
    2.0 * (q13 - q02),
    2.0 * (q23 + q01),
    q00 - q11 - q22 + q33,
  )


@wp.func
def quat_z2vec(vec: wp.vec3) -> wp.quat:
  """Compute quaternion performing rotation from z-axis to given vector."""
  quat = wp.quat(0.0, 0.0, 0.0, 1.0)

  # normalize vector; if too small, no rotation
  norm = wp.length(vec)
  if norm < types.MJ_MINVAL:
    return quat
  vec = vec / norm

  axis = wp.vec3(-vec[1], vec[0], 0.0)
  a = wp.length(axis)

  # almost parallel
  if a < types.MJ_MINVAL:
    # opposite: 180 deg rotation around x axis
    if vec[2] < 0.0:
      quat = wp.quat(1.0, 0.0, 0.0, 0.0)
    return quat

  # make quaternion from angle and axis
  axis = axis / a
  angle = wp.atan2(a, vec[2])
  quat = axis_angle_to_quat(axis, angle)

  return quat


@wp.func
def quat_inv(quat: wp.quat) -> wp.quat:
  return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])


@wp.func
def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:
  """mju_mulInertVec: multiply 6D vector (rotation, translation) by 6D inertia matrix."""
  return wp.spatial_vector(
    i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
    i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
    i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
    i[8] * v[1] - i[7] * v[2] + i[9] * v[3],
    i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
    i[7] * v[0] - i[6] * v[1] + i[9] * v[5],
  )


@wp.func
def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:
  """Cross product of two motions."""
  u0 = wp.vec3(u[0], u[1], u[2])
  u1 = wp.vec3(u[3], u[4], u[5])
  v0 = wp.vec3(v[0], v[1], v[2])
  v1 = wp.vec3(v[3], v[4], v[5])

  ang = wp.cross(u0, v0)
  vel = wp.cross(u1, v0) + wp.cross(u0, v1)

  return wp.spatial_vector(ang, vel)


@wp.func
def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:
  """Cross product of a motion and a force."""
  v0 = wp.vec3(v[0], v[1], v[2])
  v1 = wp.vec3(v[3], v[4], v[5])
  f0 = wp.vec3(f[0], f[1], f[2])
  f1 = wp.vec3(f[3], f[4], f[5])

  ang = wp.cross(v0, f0) + wp.cross(v1, f1)
  vel = wp.cross(v0, f1)

  return wp.spatial_vector(ang, vel)


@wp.func
def quat_to_vel(quat: wp.quat) -> wp.vec3:
  axis = wp.vec3(quat[1], quat[2], quat[3])
  sin_a_2 = wp.norm_l2(axis)

  if sin_a_2 == 0.0:
    return wp.vec3(0.0)

  speed = 2.0 * wp.atan2(sin_a_2, quat[0])
  # when axis-angle is larger than pi, rotation is in the opposite direction
  if speed > wp.pi:
    speed -= 2.0 * wp.pi

  return axis * speed / sin_a_2


@wp.func
def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:
  """Subtract quaternions, express as 3D velocity: qb*quat(res) = qa."""
  # qdif = neg(qb)*qa
  qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])
  qdif = mul_quat(qneg, qa)

  # convert to 3D velocity
  return quat_to_vel(qdif)


@wp.func
def quat_integrate(q: wp.quat, v: wp.vec3, dt: float) -> wp.quat:
  """Integrates a quaternion given angular velocity and dt."""
  norm_ = wp.length(v)
  v = wp.normalize(v)  # does that need proper zero gradient handling?
  angle = dt * norm_

  q_res = axis_angle_to_quat(v, angle)
  q = wp.normalize(q)
  q_res = mul_quat(q, q_res)

  return wp.normalize(q_res)


@wp.func
def deriv_expmap(expmap: wp.vec3, vel: wp.vec3) -> wp.vec3:
  """Time-derivative of expmap in quat = quat0 * exp(expmap).

  Given the rotational velocity vel in the rotated frame, computes
  vel + 1/2 expmap x vel + c expmap x (expmap x vel), with the Bortz
  coefficient c = (1 - (x/2) cot(x/2)) / x^2, x = |expmap|.

  expmap is first replaced with the equivalent rotation vector of angle in
  (-pi, pi], since c is singular at multiples of 2*pi. Below x = 1.75, where
  the closed form loses accuracy to cancellation, c is a minimax polynomial in
  x^2; with the closed form above, the error is below 6 ulp on [-pi, pi].

  Port of mju_derivExpmap (MuJoCo engine_util_spatial.c), using the
  single-precision (mjUSESINGLE) coefficients of its bortzCoef.
  """
  r = expmap
  xx = wp.dot(r, r)

  # reduce the angle to (-pi, pi]
  if xx > wp.static(wp.pi * wp.pi):
    x = wp.sqrt(xx)
    r = r * ((x - wp.static(2.0 * wp.pi) * wp.floor(x / wp.static(2.0 * wp.pi) + 0.5)) / x)
    xx = wp.dot(r, r)

  # Bortz coefficient: minimax polynomial (Horner form), or closed form
  c = float(0.0)
  if xx < 3.0625:
    c = 1.26339661e-09
    c = -1.64186869e-08 + xx * c
    c = 8.78585737e-08 + xx * c
    c = -2.48791736e-07 + xx * c
    c = 4.23758962e-07 + xx * c
    c = 4.57548566e-07 + xx * c
    c = 3.32480922e-05 + xx * c
    c = 0.00138884981 + xx * c
    c = 0.0833333358 + xx * c
  else:
    x = wp.sqrt(xx)
    c = (1.0 - 0.5 * x / wp.tan(0.5 * x)) / xx

  rxv = wp.cross(r, vel)
  return vel + 0.5 * rxv + c * wp.cross(r, rxv)


@wp.func
def orthogonals(a: wp.vec3):
  y = wp.vec3(0.0, 1.0, 0.0)
  z = wp.vec3(0.0, 0.0, 1.0)
  b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
  b = b - a * wp.dot(a, b)
  b = wp.normalize(b)
  if wp.length(a) == 0.0:
    b = wp.vec3(0.0, 0.0, 0.0)
  c = wp.cross(a, b)

  return b, c


@wp.func
def orthonormal(normal: wp.vec3) -> wp.vec3:
  if wp.abs(normal[0]) < wp.abs(normal[1]) and wp.abs(normal[0]) < wp.abs(normal[2]):
    dir = wp.vec3(1.0 - normal[0] * normal[0], -normal[0] * normal[1], -normal[0] * normal[2])
  elif wp.abs(normal[1]) < wp.abs(normal[2]):
    dir = wp.vec3(-normal[1] * normal[0], 1.0 - normal[1] * normal[1], -normal[1] * normal[2])
  else:
    dir = wp.vec3(-normal[2] * normal[0], -normal[2] * normal[1], 1.0 - normal[2] * normal[2])
  dir, _ = gjk_normalize(dir)
  return dir


@wp.func
def orthonormal_to_z(normal: wp.vec3) -> wp.vec3:
  if wp.abs(normal[0]) < wp.abs(normal[1]):
    dir = wp.vec3(1.0 - normal[0] * normal[0], -normal[0] * normal[1], -normal[0] * normal[2])
  else:
    dir = wp.vec3(-normal[1] * normal[0], 1.0 - normal[1] * normal[1], -normal[1] * normal[2])
  dir, _ = gjk_normalize(dir)
  return dir


@wp.func
def gjk_normalize(a: wp.vec3):
  norm = wp.length(a)
  if norm > 1e-8 and norm < 1e12:
    return a / norm, True
  return a, False


@wp.func
def make_frame(a: wp.vec3):
  a = wp.normalize(a)
  b, c = orthogonals(a)

  # fmt: off
  return wp.mat33(
    a.x, a.y, a.z,
    b.x, b.y, b.z,
    c.x, c.y, c.z
  )
  # fmt: on


@wp.func
def normalize_with_norm(x: Any):
  norm = wp.length(x)
  if norm == 0.0:
    return x, 0.0
  return x / norm, norm


@wp.func
def closest_segment_point(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> wp.vec3:
  """Returns the closest point on the a-b line segment to a point pt."""
  ab = b - a
  length_sq = wp.dot(ab, ab)
  if length_sq == 0.0:
    return a
  t = wp.dot(pt - a, ab) / length_sq
  return a + wp.clamp(t, 0.0, 1.0) * ab


@wp.func
def closest_segment_point_and_dist(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> Tuple[wp.vec3, float]:
  """Returns closest point on the line segment and the distance squared."""
  closest = closest_segment_point(a, b, pt)
  dist = wp.dot((pt - closest), (pt - closest))
  return closest, dist


@wp.func
def closest_segment_to_segment_points(a0: wp.vec3, a1: wp.vec3, b0: wp.vec3, b1: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:
  """Returns closest points between two line segments."""
  dir_a, len_a = normalize_with_norm(a1 - a0)
  dir_b, len_b = normalize_with_norm(b1 - b0)

  half_len_a = len_a * 0.5
  half_len_b = len_b * 0.5
  a_mid = a0 + dir_a * half_len_a
  b_mid = b0 + dir_b * half_len_b

  trans = a_mid - b_mid

  dira_dot_dirb = wp.dot(dir_a, dir_b)
  dira_dot_trans = wp.dot(dir_a, trans)
  dirb_dot_trans = wp.dot(dir_b, trans)
  denom = 1.0 - dira_dot_dirb * dira_dot_dirb

  orig_t_a = (-dira_dot_trans + dira_dot_dirb * dirb_dot_trans) / (denom + 1e-6)
  orig_t_b = dirb_dot_trans + orig_t_a * dira_dot_dirb
  t_a = wp.clamp(orig_t_a, -half_len_a, half_len_a)
  t_b = wp.clamp(orig_t_b, -half_len_b, half_len_b)

  best_a = a_mid + dir_a * t_a
  best_b = b_mid + dir_b * t_b

  new_a, d1 = closest_segment_point_and_dist(a0, a1, best_b)
  new_b, d2 = closest_segment_point_and_dist(b0, b1, best_a)
  if d1 < d2:
    return new_a, best_b
  return best_a, new_b


@wp.func
def safe_div(x: Any, y: Any) -> Any:
  return x / wp.where(y != 0.0, y, types.MJ_MINVAL)


@wp.func
def upper_tri_index(n: int, i: int, j: int) -> int:
  """Returns index of a_ij = a_ji in upper triangular matrix (excluding diagonal)."""
  return (i * (2 * n - i - 3)) // 2 + j - 1


@wp.func
def upper_trid_index(n: int, i: int, j: int) -> int:
  """Returns index of a_ij = a_ji in upper triangular matrix (including diagonal)."""
  if j < i:
    i, j = j, i
  return (i * (2 * n - i - 1)) // 2 + j


@wp.func
def free_bias_vel_blocks(
  # In:
  mass: float,
  R: wp.mat33,
  Iw: wp.mat33,
  s: wp.vec3,
  qvel_rot: wp.vec3,
) -> tuple[wp.mat33, wp.mat33]:
  """3x3 sub-blocks of (d qfrc_bias / d qvel) for a free rigid subtree."""
  w = R @ qvel_rot

  ws = wp.cross(w, s)
  Iww = Iw @ w

  w_dot_s = wp.dot(w, s)
  K = wp.outer(s, w) - wp.identity(n=3, dtype=float) * w_dot_s + wp.skew(ws)

  lin = K @ R

  s_x_K = wp.skew(s) @ K
  w_x_Iw = wp.skew(w) @ Iw
  neg_skew_Iww = wp.mat33(
    0.0,
    Iww[2],
    -Iww[1],
    -Iww[2],
    0.0,
    Iww[0],
    Iww[1],
    -Iww[0],
    0.0,
  )

  C = -mass * s_x_K + w_x_Iw + neg_skew_Iww
  rot = wp.transpose(R) @ C @ R

  return lin, rot


@wp.func
def lu_factor_6x6(
  # In:
  A_in: types.mat66,
) -> tuple[types.mat66, types.vec6i, bool]:
  """LU factorization with partial pivoting for a 6x6 matrix."""
  A = A_in
  pivot = types.vec6i(0)
  for k in range(6):
    maxval = wp.abs(A[k, k])
    maxrow = k
    for i in range(k + 1, 6):
      val = wp.abs(A[i, k])
      if val > maxval:
        maxval = val
        maxrow = i
    if maxval < types.MJ_MINVAL:
      return A, pivot, False
    pivot[k] = maxrow
    if maxrow != k:
      for j in range(6):
        tmp = A[k, j]
        A[k, j] = A[maxrow, j]
        A[maxrow, j] = tmp
    diaginv = 1.0 / A[k, k]
    for i in range(k + 1, 6):
      A[i, k] *= diaginv
      Aik = A[i, k]
      for j in range(k + 1, 6):
        A[i, j] -= Aik * A[k, j]
  return A, pivot, True


@wp.func
def lu_solve_6x6(
  # In:
  A_in: types.mat66,
  pivot_in: types.vec6i,
  b_in: types.vec6,
) -> types.vec6:
  """Solves A * x = b for a 6x6 system using pre-computed LU factorization and pivots."""
  x = b_in
  for i in range(6):
    p = pivot_in[i]
    if p != i:
      tmp = x[i]
      x[i] = x[p]
      x[p] = tmp
    for j in range(i):
      x[i] -= A_in[i, j] * x[j]
  for i in range(5, -1, -1):
    for j in range(i + 1, 6):
      x[i] -= A_in[i, j] * x[j]
    x[i] /= A_in[i, i]
  return x


@wp.func
def eig3(
  # In:
  A_in: wp.mat33,
) -> tuple[wp.vec3, wp.mat33]:
  """Eigenvalue decomposition of a symmetric 3x3 matrix.

  Returns eigenvalues sorted in descending order and eigenvectors as columns of the rotation matrix.
  """
  scale = float(0.0)
  for row in range(3):
    for col in range(3):
      scale = wp.max(scale, wp.abs(A_in[row, col]))

  V = wp.mat33(
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  )
  if scale == 0.0:
    return wp.vec3(0.0, 0.0, 0.0), V

  tol = scale * 2.0e-6
  for _iter in range(12):
    D = wp.transpose(V) @ A_in @ V
    d01 = wp.abs(D[0, 1])
    d02 = wp.abs(D[0, 2])
    d12 = wp.abs(D[1, 2])
    p = int(0)
    q = int(1)
    max_od = d01
    if d02 > max_od:
      p = int(0)
      q = int(2)
      max_od = d02
    if d12 > max_od:
      p = int(1)
      q = int(2)
      max_od = d12
    if max_od <= tol:
      break

    Dpq = D[p, q]
    tau = (D[q, q] - D[p, p]) / (2.0 * Dpq)
    t = 1.0 / (tau + wp.sqrt(1.0 + tau * tau)) if tau >= 0.0 else -1.0 / (-tau + wp.sqrt(1.0 + tau * tau))
    cos_t = 1.0 / wp.sqrt(1.0 + t * t)
    sin_t = t * cos_t
    for i in range(3):
      vip = V[i, p]
      viq = V[i, q]
      V[i, p] = cos_t * vip - sin_t * viq
      V[i, q] = sin_t * vip + cos_t * viq

  D = wp.transpose(V) @ A_in @ V
  eigval = wp.vec3(D[0, 0], D[1, 1], D[2, 2])

  # Sort eigenvalues descending (bubble sort)
  if eigval[0] < eigval[1]:
    t_val = eigval[0]
    eigval[0] = eigval[1]
    eigval[1] = t_val
    for i in range(3):
      tmp_v = V[i, 0]
      V[i, 0] = V[i, 1]
      V[i, 1] = tmp_v
  if eigval[1] < eigval[2]:
    t_val = eigval[1]
    eigval[1] = eigval[2]
    eigval[2] = t_val
    for i in range(3):
      tmp_v = V[i, 1]
      V[i, 1] = V[i, 2]
      V[i, 2] = tmp_v
  if eigval[0] < eigval[1]:
    t_val = eigval[0]
    eigval[0] = eigval[1]
    eigval[1] = t_val
    for i in range(3):
      tmp_v = V[i, 0]
      V[i, 0] = V[i, 1]
      V[i, 1] = tmp_v

  return eigval, V
