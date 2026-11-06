# Copyright 2025 DeepMind Technologies Limited
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

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco.usd import MjcPhysics
from mujoco.usd import test_utils
from mujoco.usd import writer


class TestSink:

  def __init__(self):
    self._num_updates = 0

  def update(self, data: mujoco.MjData, body_paths: list[str]) -> None:
    self._num_updates += 1

  @property
  def num_updates(self) -> int:
    return self._num_updates


class MujocoUsdWriterTest(parameterized.TestCase):

  def test_writer_constructs(self):
    test_data = test_utils.simple_falling_cube()
    writer.UsdWriter(test_data.stage, test_data.spec, test_data.model)

  def test_writer_with_test_sink(self):
    test_data = test_utils.simple_falling_cube()
    usd_writer = writer.UsdWriter(
        test_data.stage, test_data.spec, test_data.model
    )
    sink = TestSink()
    usd_writer.add_sink(sink.update)
    for _ in range(100):
      mujoco.mj_step(test_data.model, test_data.data)
      usd_writer.update(test_data.data)

    self.assertEqual(sink.num_updates, 100)

  def test_mjc_physics_schema(self):
    test_data = test_utils.simple_falling_cube()
    cube_prim = test_data.stage.GetPrimAtPath("/World/Cube")
    collision_api = MjcPhysics.CollisionAPI.Apply(cube_prim)
    collision_api.CreateGroupAttr().Set(3)
    self.assertEqual(collision_api.GetGroupAttr().Get(), 3)
    spec = mujoco.MjSpec.from_file(test_data.stage.GetRootLayer().identifier)
    self.assertEqual(spec.geoms[0].group, 3)


if __name__ == "__main__":
  absltest.main()
