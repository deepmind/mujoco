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
"""Tests for math."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
from mujoco.mjx._src import math
import numpy as np


def _get_rand_point(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(1, 3)
  return verts[0, :]


def _get_rand_line_segment(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(2, 3)
  return verts[0, :], verts[1, :]


def _get_rand_unit(seed: int):
  np.random.seed(seed)
  theta = np.random.random(1) * 2 * np.pi
  a = (np.random.random(1) - 0.5) * 2.0
  phi = np.arccos(a)
  x = np.sin(phi) * np.cos(theta)
  y = np.sin(phi) * np.sin(theta)
  z = np.cos(phi)
  return jp.array([x, y, z]).squeeze()


class OrthoganalsTest(parameterized.TestCase):
  """Tests the orthogonals function."""

  @parameterized.parameters(range(30))
  def test_orthogonals(self, i):
    a = _get_rand_unit(i)
    b, c = math.orthogonals(a)
    np.testing.assert_almost_equal(jp.linalg.norm(a), 1)
    np.testing.assert_almost_equal(jp.linalg.norm(b), 1)
    np.testing.assert_almost_equal(jp.linalg.norm(c), 1)
    self.assertAlmostEqual(np.abs(a.dot(b)), 0, 6)
    self.assertAlmostEqual(np.abs(b.dot(c)), 0, 6)
    self.assertAlmostEqual(np.abs(a.dot(c)), 0, 6)


def _minimize(fn, sample_fn, lb, ub, tol, max_iter=20, seed=42):
  """Minimize a function using the cross-entropy method."""
  assert lb.shape == ub.shape, "bounds need to have the same shape"
  np.random.seed(seed)

  i, n = 0, 1_000
  mu = (ub + lb) * 0.5
  sigma = (ub - lb) * 0.5
  size = lb.shape[0]
  val, prev_val = fn(mu), None

  while prev_val is None or np.abs(val - prev_val) > tol:
    params = sample_fn(mu, sigma, n, size, lb, ub)
    vals = np.array([fn(p) for p in params])
    if val < vals.min():  # early exit
      return mu
    idx = vals.argsort()
    best_idx = idx[: int(n * 0.05)]
    mu = params[best_idx].mean(axis=0)
    sigma = params[best_idx].std(axis=0) + 1e-10

    prev_val = val
    val = fn(mu)

    i += 1
    if i == max_iter:
      break

  return mu


def _closest_segment_to_segment_points(a0, a1, b0, b1):
  dir_a = a1 - a0
  len_a = np.sqrt(dir_a.dot(dir_a))
  half_len_a = len_a / 2
  dir_a = dir_a / len_a

  dir_b = b1 - b0
  len_b = np.sqrt(dir_b.dot(dir_b))
  half_len_b = len_b / 2
  dir_b = dir_b / len_b

  a_mid = a0 + dir_a * half_len_a
  b_mid = b0 + dir_b * half_len_b

  # Parametrize both line segments.
  def fn(t):
    best_a = a_mid + dir_a * t[0]
    best_b = b_mid + dir_b * t[1]
    return (best_a - best_b).dot(best_a - best_b)

  def sample_fn(mu, sigma, n, size, lb, ub):
    params = np.random.normal(mu, sigma, size=(n, size))
    params = np.clip(params, lb, ub)
    return params

  lb = np.array([-half_len_a, -half_len_b])
  ub = np.array([half_len_a, half_len_b])
  ta, tb = _minimize(fn, sample_fn, lb, ub, tol=1e-4)
  best_a = a_mid + dir_a * ta
  best_b = b_mid + dir_b * tb
  return best_a, best_b


class ClosestSegmentSegmentPointsTest(parameterized.TestCase):
  """Tests for closest segment-to-segment points."""

  def test_closest_segments_points(self):
    a0 = jp.array([0.73432405, 0.12372768, 0.20272314])
    a1 = jp.array([1.10600128, 0.88555209, 0.65209485])
    b0 = jp.array([0.85599262, 0.61736299, 0.9843583])
    b1 = jp.array([1.84270939, 0.92891793, 1.36343326])
    best_a, best_b = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [1.09063, 0.85404, 0.63351], 5)
    self.assertSequenceAlmostEqual(best_b, [0.99596, 0.66156, 1.03813], 5)

  def test_intersecting_segments(self):
    """Tests segments that intersect."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([-1.0, 0.0, 0.0]), jp.array([1.0, 0.0, 0.0])
    best_a, best_b = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [0.0, 0.0, 0.0], 5)

  def test_intersecting_lines(self):
    """Tests that intersecting lines get clipped."""
    a0, a1 = jp.array([0.2, 0.2, 0.0]), jp.array([1.0, 1.0, 0.0])
    b0, b1 = jp.array([0.2, 0.4, 0.0]), jp.array([1.0, 2.0, 0.0])
    best_a, best_b = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.3, 0.3, 0.0], 2)
    self.assertSequenceAlmostEqual(best_b, [0.2, 0.4, 0.0], 2)

  def test_parallel_segments(self):
    """Tests that parallel segments have closest points at the midpoint."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([1.0, 0.0, -1.0]), jp.array([1.0, 0.0, 1.0])
    best_a, best_b = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 0.0], 5)

  def test_parallel_offset_segments(self):
    """Tests that offset parallel segments are close at segment endpoints."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([1.0, 0.0, 1.0]), jp.array([1.0, 0.0, 3.0])
    best_a, best_b = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 1.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 1.0], 5)

  def test_zero_length_segments(self):
    """Test that zero length segments don't return NaNs."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, -1.0])
    b0, b1 = jp.array([1.0, 0.0, 0.1]), jp.array([1.0, 0.0, 0.1])
    best_a, best_b = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, -1.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 0.1], 5)

  def test_overlapping_segments(self):
    """Tests that perfectly overlapping segments intersect at the midpoints."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    best_a, best_b = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [0.0, 0.0, 0.0], 5)

  params = list(zip(np.repeat(np.arange(10), 10), np.tile(np.arange(10), 10)))

  @parameterized.parameters(*params)
  def test_closest_segment_to_segment_points(self, i, j):
    a0, a1 = _get_rand_line_segment(i)
    b0, b1 = _get_rand_line_segment(j)
    expected = _closest_segment_to_segment_points(a0, a1, b0, b1)
    ans = math.closest_segment_to_segment_points(a0, a1, b0, b1)
    expected_dist = (expected[0] - expected[1]).dot(expected[0] - expected[1])
    test_dist = (ans[0] - ans[1]).dot(ans[0] - ans[1])
    self.assertAlmostEqual(expected_dist, test_dist, 4)


def _rotvec_rate(expmap: np.ndarray, vel: np.ndarray) -> np.ndarray:
  """Float64 reference for math.deriv_expmap, from its definition.

  Computes d/dt log(exp(expmap) * exp(t * vel)) at t = 0 by forward-mode
  autodiff through the quaternion exponential and logarithm, independently of
  the Bortz formula. The logarithm returns angles in (-pi, pi], so for
  |expmap| > pi this is the rate of the equivalent reduced rotation vector.

  Args:
    expmap: (3,) nonzero rotation vector
    vel: (3,) rotational velocity in the rotated frame

  Returns:
    (3,) time-derivative of the rotation vector
  """
  with jax.enable_x64(True):
    angle = np.linalg.norm(expmap)
    q0 = math.axis_angle_to_quat(jp.array(expmap / angle), angle)
    vel = jp.array(vel, dtype=jp.float64)

    def rotvec(t):
      q = math.quat_integrate(q0, vel, t)
      axis, theta = math.quat_to_axis_angle(q)
      return axis * theta

    _, rate = jax.jvp(rotvec, (0.0,), (1.0,))
    return np.asarray(rate, dtype=np.float64)


class DerivExpmapTest(parameterized.TestCase):

  def test_zero_rotation(self):
    """dexp^{-1} at a zero rotation vector is the identity."""
    v = jp.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(math.deriv_expmap(jp.zeros(3), v), v)

  @parameterized.product(
      # below and above the polynomial cutoff (1.75), near pi, and above pi
      angle=(1e-3, 0.5, 1.74, 1.76, 3.0, 4.0, 7.0),
      x64=(False, True),
  )
  def test_reference(self, angle, x64):
    axis = np.array([1.0, 2.0, -2.0]) / 3.0
    vel = np.array([0.3, -1.2, 0.7])
    expected = _rotvec_rate(angle * axis, vel)

    with jax.enable_x64(x64):
      dtype = jp.float64 if x64 else jp.float32
      expmap = jp.array(angle * axis, dtype=dtype)
      res = math.deriv_expmap(expmap, jp.array(vel, dtype=dtype))
      self.assertEqual(res.dtype, dtype)

    tol = 1e-12 if x64 else 1e-6
    np.testing.assert_allclose(res, expected, atol=tol, rtol=tol)


if __name__ == "__main__":
  absltest.main()
