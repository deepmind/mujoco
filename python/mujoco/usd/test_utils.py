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

import dataclasses

import mujoco

from pxr import Usd  # type: ignore
from pxr import UsdGeom  # type: ignore
from pxr import UsdPhysics  # type: ignore


@dataclasses.dataclass
class TestData:
  stage: Usd.Stage
  spec: mujoco.MjSpec
  model: mujoco.MjModel
  data: mujoco.MjData


def simple_falling_cube() -> TestData:
  stage = Usd.Stage.CreateInMemory()

  cube = UsdGeom.Cube.Define(stage, "/World/Cube")

  UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
  UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

  spec = mujoco.MjSpec.from_file(stage.GetRootLayer().identifier)
  model = spec.compile()
  data = mujoco.MjData(model)
  return TestData(stage=stage, spec=spec, model=model, data=data)


def simple_nested_hierarchy() -> TestData:
  """Creates a test scene with a simple nested hierarchy."""
  stage = Usd.Stage.CreateInMemory()

  link0 = UsdGeom.Capsule.Define(stage, "/World/Link0")
  UsdPhysics.RigidBodyAPI.Apply(link0.GetPrim())
  UsdPhysics.CollisionAPI.Apply(link0.GetPrim())

  link1 = UsdGeom.Capsule.Define(stage, "/World/Link0/Link1")
  UsdPhysics.RigidBodyAPI.Apply(link1.GetPrim())
  UsdPhysics.CollisionAPI.Apply(link1.GetPrim())
  UsdGeom.XformCommonAPI(link1).SetTranslate((2.0, 0.0, 0.0))

  link2 = UsdGeom.Capsule.Define(stage, "/World/Link0/Link1/Link2")
  UsdPhysics.RigidBodyAPI.Apply(link2.GetPrim())
  UsdPhysics.CollisionAPI.Apply(link2.GetPrim())
  UsdGeom.XformCommonAPI(link2).SetTranslate((2.0, 0.0, 0.0))

  link3 = UsdGeom.Capsule.Define(stage, "/World/Link0/Link1/Link2/Link3")
  UsdPhysics.RigidBodyAPI.Apply(link3.GetPrim())
  UsdPhysics.CollisionAPI.Apply(link3.GetPrim())
  UsdGeom.XformCommonAPI(link3).SetTranslate((2.0, 0.0, 0.0))

  spec = mujoco.MjSpec.from_file(stage.GetRootLayer().identifier)
  model = spec.compile()
  data = mujoco.MjData(model)
  return TestData(stage=stage, spec=spec, model=model, data=data)
