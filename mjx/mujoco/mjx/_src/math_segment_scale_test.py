# Copyright 2026 DeepMind Technologies Limited
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
"""Tests for scale-independent point projection onto line segments."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import numpy as np

from mujoco.mjx._src import math


class SegmentScaleTest(parameterized.TestCase):

  @parameterized.product(
      scale=(1e-5, 1e-3, 1.0, 100.0),
      fraction=(-0.25, 0.25, 0.75, 1.25),
      dtype=(jp.float32, jp.float64),
  )
  def test_projection_matches_analytic_solution(self, scale, fraction, dtype):
    a = jp.array([0.0, 0.0, 0.0], dtype=dtype)
    b = jp.array([scale, 0.0, 0.0], dtype=dtype)
    point = jp.array([fraction * scale, 0.2 * scale, 0.0], dtype=dtype)
    expected = np.array([np.clip(fraction, 0.0, 1.0) * scale, 0.0, 0.0])
    actual = jax.jit(math.closest_segment_point)(a, b, point)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=scale * 1e-8)
    _, distance = math.closest_segment_point_and_dist(a, b, point)
    np.testing.assert_allclose(
        distance, np.sum((np.asarray(point) - expected) ** 2), rtol=1e-6
    )

  @parameterized.parameters(jp.float32, jp.float64)
  def test_interior_projection_derivative(self, dtype):
    a = jp.zeros(3, dtype=dtype)
    b = jp.array([1e-4, 0.0, 0.0], dtype=dtype)
    point = jp.array([0.5e-4, 2e-4, 0.0], dtype=dtype)
    jacobian = jax.jacrev(lambda p: math.closest_segment_point(a, b, p))(point)
    np.testing.assert_allclose(jacobian, np.diag([1.0, 0.0, 0.0]), atol=1e-7)

  def test_degenerate_segment_is_finite(self):
    a = jp.array([1.0, 2.0, 3.0])
    point = jp.array([2.0, 0.0, 1.0])
    np.testing.assert_array_equal(math.closest_segment_point(a, a, point), a)
    for arg in range(3):
      jacobian = jax.jacrev(math.closest_segment_point, argnums=arg)(
          a, a, point
      )
      self.assertTrue(np.isfinite(jacobian).all())


if __name__ == "__main__":
  absltest.main()
