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
"""Tests for the filament renderer."""

from absl.testing import absltest
from mujoco import mjrf
from mujoco.rendering.filament import renderer
import numpy as np


class RendererTest(absltest.TestCase):

  def test_extract_buffer_ndarray(self):
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    buffer = renderer._extract_buffer(pixels)
    width, height = renderer._validate_buffer(
        buffer, mjrf.PixelFormat.PIXEL_FORMAT_RGB8, 'out'
    )
    self.assertEqual((width, height), (8, 4))

  def test_extract_buffer_bytearray(self):
    buffer = bytearray(3)
    self.assertIs(renderer._extract_buffer(buffer), buffer)


if __name__ == '__main__':
  absltest.main()
