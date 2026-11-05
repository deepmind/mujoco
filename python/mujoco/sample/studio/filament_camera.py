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
"""Demonstrates how to use ImGui inputs to control a free camera in MuJoCo."""

from collections.abc import Sequence
import sys

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import model_scene
from mujoco.rendering.filament import renderer
from mujoco.studio import sim
from mujoco.studio import window
import numpy as np

from mujoco.studio.dear_imgui import dear_imgui as imgui


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    raise SystemExit("Usage: filament_camera.py <path to model XML file>")

  # create an mjModel and mjData from a path to an xml file
  model = mujoco.MjModel.from_xml_path(argv[1])
  data = mujoco.MjData(model)

  step_control = sim.StepControl()

  # create a window
  win = window.Window("Filament Render", 512, 512, "opengl_headless")

  # create the filament render context
  ctx = mjrf.Context(
      mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL)
  )

  # create a renderer
  r = renderer.Renderer(ctx)

  # add lights and renderables defined in the model to a scene
  ms = model_scene.ModelScene(ctx, model, r.scene("scn"))

  # setup a camera
  vcam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, vcam)
  vcam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
  # vcam.fixedcamid = 0

  buf = np.empty((0, 0, 3), dtype=np.uint8)
  while win.NewFrame():
    # simulate one step
    step_control.advance(model, data)
    # mujoco.mj_step(model, data)

    # update the lights and renderables with the current `data` state
    ms.update(data)

    # update the camera based on user inputs
    _update_camera(model, vcam)

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


def _update_camera(model: mujoco.MjModel, vcam: mujoco.MjvCamera):
  """Updates the camera based on user inputs."""
  # pylint: disable=invalid-name
  MOVE_H = int(mujoco.mjtMouse.mjMOUSE_MOVE_H)
  MOVE_H_REL = int(mujoco.mjtMouse.mjMOUSE_MOVE_H_REL)
  MOVE_V = int(mujoco.mjtMouse.mjMOUSE_MOVE_V)
  MOVE_V_REL = int(mujoco.mjtMouse.mjMOUSE_MOVE_V_REL)
  TURN_H = int(mujoco.mjtMouse.mjMOUSE_TURN_H)
  TURN_V = int(mujoco.mjtMouse.mjMOUSE_TURN_V)
  # pylint: enable=invalid-name

  io = imgui.GetIO()
  mouse_dx = io.MouseDelta.x / io.DisplaySize.x
  mouse_dy = io.MouseDelta.y / io.DisplaySize.y
  if mouse_dx != 0.0 or mouse_dy != 0.0:
    if imgui.IsMouseDown(imgui.MouseButton.Left):
      mujoco.mjv_moveCamera(model, TURN_H, mouse_dx, 0.0, vcam)
      mujoco.mjv_moveCamera(model, TURN_V, 0.0, mouse_dy, vcam)
    if imgui.IsMouseDown(imgui.MouseButton.Right) and io.KeyShift:
      mujoco.mjv_moveCamera(model, MOVE_H, mouse_dx, mouse_dy, vcam)
    elif imgui.IsMouseDown(imgui.MouseButton.Right):
      mujoco.mjv_moveCamera(model, MOVE_V, mouse_dx, mouse_dy, vcam)
  if imgui.IsKeyDown(imgui.Key.W):
    mujoco.mjv_moveCamera(model, MOVE_H_REL, 0.0, 1.0, vcam)
  elif imgui.IsKeyDown(imgui.Key.S):
    mujoco.mjv_moveCamera(model, MOVE_H_REL, 0.0, -1.0, vcam)
  if imgui.IsKeyDown(imgui.Key.A):
    mujoco.mjv_moveCamera(model, MOVE_H_REL, -1.0, 0.0, vcam)
  elif imgui.IsKeyDown(imgui.Key.D):
    mujoco.mjv_moveCamera(model, MOVE_H_REL, 1.0, 0.0, vcam)
  if imgui.IsKeyDown(imgui.Key.Q):
    mujoco.mjv_moveCamera(model, MOVE_V_REL, 0.0, 1.0, vcam)
  elif imgui.IsKeyDown(imgui.Key.E):
    mujoco.mjv_moveCamera(model, MOVE_V_REL, 0.0, -1.0, vcam)

if __name__ == "__main__":
  main(sys.argv)
