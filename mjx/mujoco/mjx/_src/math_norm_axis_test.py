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
"""Tests for normalization along explicit axes."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import numpy as np

from mujoco.mjx._src import math


class NormAxisTest(parameterized.TestCase):

  @parameterized.product(axis=(0, 1, -1), dtype=(jp.float32, jp.float64))
  def test_mixed_zero_slices_have_finite_gradients(self, axis, dtype):
    x = jp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=dtype)
    if axis == 0:
      x = x.T
    result = jax.jit(lambda value: math.norm(value, axis=axis))(x)
    expected = np.linalg.norm(x, axis=axis)
    np.testing.assert_allclose(result, expected)
    derivative = jax.jacrev(lambda value: math.norm(value, axis=axis).sum())(x)
    expected_derivative = np.array([[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
    if axis == 0:
      expected_derivative = expected_derivative.T
    np.testing.assert_allclose(derivative, expected_derivative, atol=1e-7)

  @parameterized.product(
      axis=(0, 1, -1, (0, 2)), dtype=(jp.float32, jp.float64)
  )
  def test_normalization_preserves_reduced_axes_for_broadcasting(
      self, axis, dtype
  ):
    shape = (2, 3, 4) if isinstance(axis, tuple) else (2, 3)
    x = jp.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
    actual, norm = jax.jit(
        lambda value: math.normalize_with_norm(value, axis=axis)
    )(x)
    expected_norm = np.linalg.norm(x, axis=axis)
    expected = x / np.expand_dims(expected_norm, axis)
    np.testing.assert_allclose(norm, expected_norm)
    np.testing.assert_allclose(actual, expected)
    self.assertEqual(actual.shape, x.shape)

  def test_near_zero_behavior_is_preserved_per_slice(self):
    x = jp.array([[1e-10, 0.0, 0.0], [0.0, 3.0, 4.0]])
    np.testing.assert_array_equal(math.norm(x, axis=1), [0.0, 5.0])


if __name__ == "__main__":
  absltest.main()
