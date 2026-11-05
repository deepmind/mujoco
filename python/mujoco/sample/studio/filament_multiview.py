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
"""Renders multiple views of a MuJoCo model with Filament."""

from collections.abc import Sequence
import sys

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import renderer
from mujoco.studio import window
import numpy as np


def _update_target(
    r: renderer.Renderer, win: window.Window, buf: np.ndarray
) -> np.ndarray:
  w, h = win.GetWidth(), win.GetHeight()
  if buf.shape != (h, w, 3):
    buf = np.empty((h, w, 3), dtype=np.uint8)
    r.target("out", buf)
  return buf


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    raise SystemExit("Usage: filament_multiview.py <path to model XML file>")

  # create an mjModel and mjData from a path to an xml file
  model = mujoco.MjModel.from_xml_path(argv[1])
  data = mujoco.MjData(model)

  modes = [
      mjrf.DrawMode.DRAW_MODE_DEFAULT,
      mjrf.DrawMode.DRAW_MODE_SEGMENTATION_BY_ID,
      mjrf.DrawMode.DRAW_MODE_DEPTH,
  ]
  ncam = max(model.ncam, 1)

  # create a window
  win = window.Window(
      "Filament Multiview", 256 * ncam, 256 * len(modes), "opengl_headless"
  )

  # create the filament render context
  ctx = mjrf.Context(
      mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL)
  )

  # create a renderer
  r = renderer.Renderer(ctx)

  # create GPU objects (textures, meshes, etc.) from the model
  model_resources = mjrf.ModelResources(ctx, model)

  # add model lights and renderables to a scene
  model_lights = mjrf.ModelLights(r.scene("scn"), model_resources)
  model_renderables = mjrf.ModelRenderables(r.scene("scn"), model_resources)

  # define multiple views
  views: dict[str, renderer.View] = {}
  dx = 1.0 / ncam
  dy = 1.0 / len(modes)
  for y, mode in enumerate(modes):
    for x in range(ncam):
      vp = (x * dx, y * dy, dx, dy)
      views[f"v_{x}_{y}"] = renderer.View(
          scene="scn",
          target="out",
          viewport=vp,
          draw_mode=mode,
      )

  buf = np.empty((0, 0, 3), dtype=np.uint8)
  while win.NewFrame():
    # simulate one step
    mujoco.mj_step(model, data)

    # update the lights and renderables
    model_lights.update(data)
    model_renderables.update(data)

    # update the cameras
    for y, _ in enumerate(modes):
      for x in range(ncam):
        vcam = mujoco.MjvCamera()
        if model.ncam == 0:
          vcam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
          mujoco.mjv_defaultFreeCamera(model, vcam)
        else:
          vcam.type = int(mujoco.mjtCamera.mjCAMERA_FIXED)
          vcam.fixedcamid = x
        views[f"v_{x}_{y}"].camera = mujoco.mjv_camera2GLCamera(
            model, data, vcam
        )

    # ensure our output buffer is of the right size
    buf = _update_target(r, win, buf)

    # render the scene
    r.render(list(views.values()))

    # present the output image to the window
    win.Present(buf)


if __name__ == "__main__":
  main(sys.argv)
