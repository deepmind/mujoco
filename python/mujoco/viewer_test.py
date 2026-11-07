# Copyright 2022 DeepMind Technologies Limited
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
"""Minimal tests of mujoco.viewer that do not open a window."""

from absl.testing import absltest
import mujoco
from mujoco import viewer


class _FakeSimulate:
  """Records the overlay texts that a viewer handle passes to simulate."""

  def __init__(self):
    self.texts = None

  def set_texts(self, texts):
    self.texts = texts


class ViewerTest(absltest.TestCase):

  def _make_handle(self, sim):
    return viewer.Handle(
        sim, mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None
    )

  def test_launch_function_exists(self):
    self.assertIsNotNone(viewer.launch)

  def test_set_texts_defaults(self):
    sim = _FakeSimulate()
    self._make_handle(sim).set_texts((None, None, None, None))
    self.assertLen(sim.texts, 1)
    font, gridpos, text1, text2 = sim.texts[0]
    self.assertEqual(font, mujoco.mjtFont.mjFONT_NORMAL)
    self.assertEqual(gridpos, mujoco.mjtGridPos.mjGRID_TOPLEFT)
    self.assertEqual(text1, '')
    self.assertEqual(text2, '')

  def test_set_texts_explicit_values(self):
    sim = _FakeSimulate()
    font, grid = mujoco.mjtFont, mujoco.mjtGridPos
    texts = [
        (font.mjFONT_BIG, grid.mjGRID_TOPRIGHT, 'a', 'b'),
        (font.mjFONT_SHADOW, grid.mjGRID_BOTTOMLEFT, 'c', 'd'),
    ]
    self._make_handle(sim).set_texts(texts)
    self.assertEqual(sim.texts, texts)


if __name__ == '__main__':
  absltest.main()
