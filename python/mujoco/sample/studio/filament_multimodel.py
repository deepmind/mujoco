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
"""Renders multiple MuJoCo models with Filament."""

from collections.abc import Sequence
import dataclasses
import math
import sys

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import renderer
from mujoco.studio import window
import numpy as np


@dataclasses.dataclass
class ModelData:
  name: str
  model: mujoco.MjModel
  data: mujoco.MjData
  objects: mjrf.ModelResources | None = None
  lights: mjrf.ModelLights | None = None
  renderables: mjrf.ModelRenderables | None = None


def _update_target(
    r: renderer.Renderer, win: window.Window, buf: np.ndarray
) -> np.ndarray:
  w, h = win.GetWidth(), win.GetHeight()
  if buf.shape != (h, w, 3):
    buf = np.empty((h, w, 3), dtype=np.uint8)
    r.target("out", buf)
  return buf


def main(argv: Sequence[str]) -> None:
  paths = list(argv[1:])
  if not paths:
    raise SystemExit(
        "Usage: filament_multimodel.py <path to model XML file>..."
    )
  if len(paths) > 16:
    paths = paths[:16]

  models: list[ModelData] = []
  for path in paths:
    # create an mjModel and mjData from a path to an xml file
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    models.append(ModelData(path, model, data))

  nx = math.ceil(math.sqrt(len(paths)))
  ny = math.ceil(len(paths) / nx)

  # create a window
  win = window.Window(
      "Filament Multimodel", nx * 256, ny * 256, "opengl_headless"
  )

  # create the filament render context
  ctx = mjrf.Context(
      mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL)
  )

  # create a renderer
  r = renderer.Renderer(ctx)

  x = 0.0
  y = 0.0
  dx = 1.0 / float(nx)
  dy = 1.0 / float(ny)
  views = []
  for md in models:
    md.objects = mjrf.ModelResources(ctx, md.model)
    md.lights = mjrf.ModelLights(r.scene(md.name), md.objects)
    md.renderables = mjrf.ModelRenderables(r.scene(md.name), md.objects)

    vcam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(md.model, vcam)

    vp = (x, y, dx, dy)
    view = renderer.View(
        scene=md.name,
        camera=mujoco.mjv_camera2GLCamera(md.model, md.data, vcam),
        target="out",
        viewport=vp,
    )
    views.append(view)
    x += dx
    if x >= 1.0:
      x = 0.0
      y += dy

  buf = np.empty((0, 0, 3), dtype=np.uint8)
  while win.NewFrame():
    for md in models:
      mujoco.mj_step(md.model, md.data)
      assert md.lights is not None
      assert md.renderables is not None
      md.lights.update(md.data)
      md.renderables.update(md.data)

    buf = _update_target(r, win, buf)
    r.render(views)
    win.Present(buf)


if __name__ == "__main__":
  main(sys.argv)
