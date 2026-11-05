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
"""Camera handling for USD exporter."""

import mujoco.usd.utils as utils_module

import numpy as np

# TODO: b/288149332 - Remove once USD Python Binding works well with pytype.
from pxr import Gf
from pxr import Tf
from pxr import Usd
from pxr import UsdGeom


class USDCamera:
  """Class that handles the cameras in the USD scene."""

  def __init__(self, stage: Usd.Stage, obj_name: str):
    self.stage = stage

    xform_name = Tf.MakeValidIdentifier(f"Camera_Xform_{obj_name}")
    camera_name = Tf.MakeValidIdentifier(f"Camera_{obj_name}")
    xform_path = f"/World/{xform_name}"
    camera_path = f"{xform_path}/{camera_name}"
    self.usd_xform = UsdGeom.Xform.Define(stage, xform_path)
    self.usd_camera = UsdGeom.Camera.Define(stage, camera_path)
    self.usd_prim = stage.GetPrimAtPath(camera_path)

    # defining ops required by update function
    self.transform_op = self.usd_xform.AddTransformOp()

    self.usd_camera.CreateFocalLengthAttr().Set(12)
    self.usd_camera.CreateFocusDistanceAttr().Set(400)

    self.usd_camera.GetHorizontalApertureAttr().Set(12)

    self.usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(1e-4, 1e6))

  def update(self, cam_pos: np.ndarray, cam_mat: np.ndarray, frame: int):
    """Updates the position and orientation of the camera in the scene."""
    transformation_mat = utils_module.create_transform_matrix(
        rotation_matrix=cam_mat, translation_vector=cam_pos
    ).T
    self.transform_op.Set(Gf.Matrix4d(transformation_mat.tolist()), frame)
