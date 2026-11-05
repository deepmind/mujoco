# Copyright 2024 DeepMind Technologies Limited
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
"""Light handling for USD exporter."""

from typing import Optional

import numpy as np

# TODO: b/288149332 - Remove once USD Python Binding works well with pytype.
from pxr import Gf
from pxr import Tf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux


class USDSphereLight:
  """Class that handles the sphere lights in the USD scene."""

  def __init__(
      self, stage: Usd.Stage, obj_name: str, radius: Optional[float] = 0.3
  ):
    self.stage = stage

    xform_name = Tf.MakeValidIdentifier(f"Light_Xform_{obj_name}")
    light_name = Tf.MakeValidIdentifier(f"Light_{obj_name}")
    xform_path = f"/World/{xform_name}"
    light_path = f"{xform_path}/{light_name}"
    self.usd_xform = UsdGeom.Xform.Define(stage, xform_path)
    self.usd_light = UsdLux.SphereLight.Define(stage, light_path)
    self.usd_prim = stage.GetPrimAtPath(light_path)

    # we assume in mujoco that all lights are point lights
    self.usd_light.GetRadiusAttr().Set(radius)
    self.usd_light.GetTreatAsPointAttr().Set(False)
    self.usd_light.GetNormalizeAttr().Set(True)

    # defining ops required by update function
    self.translate_op = self.usd_xform.AddTranslateOp()

  def update(
      self, pos: np.ndarray, intensity: int, color: np.ndarray, frame: int
  ):
    """Updates the attributes of a sphere light."""
    self.translate_op.Set(Gf.Vec3d(pos.tolist()), frame)

    if not np.any(pos):
      intensity = 0

    self.usd_light.GetIntensityAttr().Set(intensity)
    self.usd_light.GetColorAttr().Set(Gf.Vec3d(color.tolist()))


class USDDomeLight:
  """Class that handles the dome lights in the USD scene."""

  def __init__(self, stage: Usd.Stage, obj_name: str):
    self.stage = stage

    xform_name = Tf.MakeValidIdentifier(f"Light_Xform_{obj_name}")
    light_name = Tf.MakeValidIdentifier(f"Light_{obj_name}")
    xform_path = f"/World/{xform_name}"
    light_path = f"{xform_path}/{light_name}"
    self.usd_xform = UsdGeom.Xform.Define(stage, xform_path)
    self.usd_light = UsdLux.DomeLight.Define(stage, light_path)
    self.usd_prim = stage.GetPrimAtPath(light_path)

    # we assume in mujoco that all lights are point lights
    self.usd_light.GetNormalizeAttr().Set(True)

  def update(self, intensity: int, color: np.ndarray):
    """Updates the attributes of a dome light."""
    self.usd_light.GetIntensityAttr().Set(intensity)
    self.usd_light.GetExposureAttr().Set(0.0)
    self.usd_light.GetColorAttr().Set(Gf.Vec3d(color.tolist()))
