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
"""Tests for quaternion integration at zero angular velocity."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import numpy as np

import mujoco
from mujoco.mjx._src import math


class QuatIntegrateTest(parameterized.TestCase):

  @parameterized.product(dtype=(jp.float32, jp.float64), dt=(0.0, 0.01, 0.5))
  def test_rest_derivative_matches_native_finite_difference(self, dtype, dt):
    q = np.array([0.5, -0.5, 0.5, 0.5])
    q_jax = jp.asarray(q, dtype=dtype)
    velocity = jp.zeros(3, dtype=dtype)
    actual = jax.jit(jax.jacrev(lambda v: math.quat_integrate(q_jax, v, dt)))(
        velocity
    )
    expected = np.empty((4, 3))
    eps = 1e-6
    for i in range(3):
      v = np.eye(3)[i] * eps
      plus, minus = q.copy(), q.copy()
      mujoco.mju_quatIntegrate(plus, v, dt)
      mujoco.mju_quatIntegrate(minus, -v, dt)
      expected[:, i] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

  @parameterized.parameters(0.0, 1e-10, 1e-6, 0.5, 10.0)
  def test_values_match_native_integration(self, speed):
    q = np.array([0.5, -0.5, 0.5, 0.5])
    v = np.array([speed, -0.5 * speed, 0.2 * speed])
    expected = q.copy()
    mujoco.mju_quatIntegrate(expected, v, 0.1)
    actual = math.quat_integrate(jp.asarray(q), jp.asarray(v), 0.1)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(actual), 1.0)

  def test_vmap_rest_and_moving_velocities(self):
    q = jp.array([1.0, 0.0, 0.0, 0.0])
    velocities = jp.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])
    actual = jax.vmap(jax.jacrev(lambda v: math.quat_integrate(q, v, 0.1)))(
        velocities
    )
    self.assertTrue(np.isfinite(actual).all())
    np.testing.assert_allclose(actual[0, 1:, :], np.eye(3) * 0.05, atol=1e-10)


if __name__ == "__main__":
  absltest.main()
