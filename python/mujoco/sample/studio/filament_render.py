# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple example of rendering a MuJoCo model with Filament."""

from collections.abc import Sequence
import sys

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import model_scene
from mujoco.rendering.filament import renderer
from mujoco.studio import window
import numpy as np


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    raise SystemExit("Usage: filament_render.py <path to model XML file>")

  # create an mjModel and mjData from a path to an xml file
  model = mujoco.MjModel.from_xml_path(argv[1])
  data = mujoco.MjData(model)

  # create a window
  win = window.Window("Filament Render", 512, 512, "opengl_headless")

  # create the filament render context
  ctx = mjrf.Context(
      mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL)
  )

  # create a renderer
  r = renderer.Renderer(ctx)

  # add lights and renderables defined in the model to a scene
  scene = model_scene.ModelScene(ctx, model, r.scene("scn"))

  # setup a camera
  vcam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, vcam)
  vcam.type = int(mujoco.mjtCamera.mjCAMERA_FIXED)
  vcam.fixedcamid = 0

  buf = np.empty((0, 0, 3), dtype=np.uint8)
  while win.NewFrame():
    # simulate one step
    mujoco.mj_step(model, data)

    # update the lights and renderables with the current `data` state
    scene.update(data)

    # ensure output buffer is of right size (in case of window resize)
    w, h = win.GetWidth(), win.GetHeight()
    if buf.shape != (h, w, 3):
      buf = np.empty((h, w, 3), dtype=np.uint8)
      r.target("out", buf)

    # render the scene
    view = renderer.View(
        scene="scn",
        target="out",
        camera=mujoco.mjv_camera2GLCamera(model, data, vcam),
    )
    r.render([view])

    # present the output image to the window
    win.Present(buf)


if __name__ == "__main__":
  main(sys.argv)
