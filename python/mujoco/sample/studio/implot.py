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
r"""Viewer plugin that draws body-inspection plots using ImGui and ImPlot.

Run with the Studio viewer::

  python -m mujoco.studio.viewer \
      --model=model/humanoid/humanoid.xml \
      --plugins=python/mujoco/sample/studio/implot.py
"""

import math

import mujoco
from mujoco.studio import app as studio_app
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

from mujoco.studio.dear_imgui import dear_imgui as imgui
from mujoco.studio.implot import implot

_N_HISTORY = 100

_PLOT_FLAGS = (
    implot.Flags.NoInputs.value  # Disable pan/zoom mouse interaction.
    | implot.Flags.NoMenus.value  # Disable right-click context menu.
    | implot.Flags.NoBoxSelect.value  # Disable drag-to-select regions.
)

_AXIS_FLAGS = (
    implot.AxisFlags.NoGridLines.value  # Hide background grid lines.
    | implot.AxisFlags.NoTickMarks.value  # Hide small tick marks on the axis.
)


def _setup_plot_flags(plot_size: imgui.Vec2) -> int:
  flags = _PLOT_FLAGS
  if min(plot_size.x, plot_size.y) < 300:
    flags |= implot.Flags.NoTitle.value
  if min(plot_size.x, plot_size.y) < 200:
    flags |= implot.Flags.NoLegend.value
  return flags


def _setup_time_axis(plot_size: imgui.Vec2) -> None:
  flags = _AXIS_FLAGS
  if plot_size.x < 300:
    flags |= implot.AxisFlags.NoTickLabels.value
  implot.SetupAxis(implot.Axis.X1, '', flags)
  implot.SetupAxisLimits(implot.Axis.X1, 0, _N_HISTORY)


def _setup_xpos_axis(centroid: list[np.ndarray], plot_size: imgui.Vec2) -> None:
  flags = _AXIS_FLAGS
  if plot_size.y < 300:
    flags |= implot.AxisFlags.NoTickLabels.value
  implot.SetupAxis(implot.Axis.Y1, '', flags)
  min_y = min(c[1] for c in centroid)
  max_y = max(c[1] for c in centroid)
  margin = max((max_y - min_y) * 0.1, 0.05)
  implot.SetupAxisLimits(
      implot.Axis.Y1,
      min_y - margin,
      max_y + margin,
      cond=implot.Cond.Always,
  )


def _setup_angle_axis(plot_size: imgui.Vec2) -> None:
  flags = _AXIS_FLAGS
  if plot_size.y < 300:
    flags |= implot.AxisFlags.NoTickLabels.value
  implot.SetupAxis(implot.Axis.Y1, '', flags)
  implot.SetupAxisLimits(implot.Axis.Y1, -185.0, 185.0)
  implot.SetupAxisTicks(
      implot.Axis.Y1,
      [-180.0, -90.0, 0.0, 90.0, 180.0],
      ['-180', '-90', '0', '90', '180'],
  )


class BodyInspector(plugin.ViewerPlugin):
  """Handler that draws body-inspection plots using ImGui/ImPlot."""

  def __init__(self) -> None:
    self._app: studio_app.App | None = None
    self._centroid: list[np.ndarray] = [np.zeros(3) for _ in range(_N_HISTORY)]
    self._euler: list[np.ndarray] = [np.zeros(3) for _ in range(_N_HISTORY)]
    self._body_id: int = -1

  @messages.handler
  def on_studio_app_init(self, event: studio_app.InitEvent) -> None:
    """Caches the studio_app App reference on startup."""
    assert isinstance(event.app, studio_app.App)
    self._app = event.app

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    """Resets plotted history and body selection when the model changes."""
    del event
    self._centroid = [np.zeros(3) for _ in range(_N_HISTORY)]
    self._euler = [np.zeros(3) for _ in range(_N_HISTORY)]
    self._body_id = -1

  @messages.handler
  def inspect_body(self, _: messages.BuildGuiEvent) -> None:
    """Renders the body-inspection charts in ImGui/ImPlot."""
    app = self._app
    if app is None:
      return
    if app.viewer.perturb.select > 0:
      self._body_id = app.viewer.perturb.select

    # Display selected body information.
    if self._body_id > 0:
      body_name = mujoco.mj_id2name(
          app.model, int(mujoco.mjtObj.mjOBJ_BODY), self._body_id
      )

      io = imgui.GetIO()
      imgui.SetNextWindowPos(
          imgui.Vec2(io.DisplaySize.x * 0.5, io.DisplaySize.y * 0.5),
          imgui.Cond.FirstUseEver,
          imgui.Vec2(0.5, 0.5),
      )
      imgui.SetNextWindowSize(imgui.Vec2(1200, 600), imgui.Cond.FirstUseEver)

      # Note: The window title uses the special "###" markup to ensure the imgui
      # ID for the window is constant for all body names.  This is needed for
      # the window to retain its state for all bodies.
      window_title = (
          f'Inspect Body {body_name or "(???)"!r} ({self._body_id})###Plot'
      )
      if imgui.Begin(window_title):
        avail = imgui.GetContentRegionAvail()
        wide = avail.x > avail.y

        # Add a small padding factor to prevent scrollbars.
        plot_size = imgui.Vec2(
            avail.x * 0.5 - 4 if wide else avail.x,
            avail.y if wide else avail.y * 0.5 - 4,
        )

        plot_flags = _setup_plot_flags(plot_size)
        if implot.BeginPlot('Centroid vs Time', plot_size, flags=plot_flags):
          _setup_time_axis(plot_size)
          _setup_xpos_axis(self._centroid, plot_size)
          implot.PlotLine(
              'x', range(_N_HISTORY), [c[0] for c in self._centroid]
          )
          implot.PlotLine(
              'y', range(_N_HISTORY), [c[1] for c in self._centroid]
          )
          implot.PlotLine(
              'z', range(_N_HISTORY), [c[2] for c in self._centroid]
          )
          implot.EndPlot()

        if wide:
          imgui.SameLine()

        if implot.BeginPlot('Euler Angle vs Time', plot_size, flags=plot_flags):
          _setup_time_axis(plot_size)
          _setup_angle_axis(plot_size)
          implot.PlotLine(
              'roll', range(_N_HISTORY), [e[0] for e in self._euler]
          )
          implot.PlotLine(
              'pitch', range(_N_HISTORY), [e[1] for e in self._euler]
          )
          implot.PlotLine('yaw', range(_N_HISTORY), [e[2] for e in self._euler])
          implot.EndPlot()
      imgui.End()

    self._centroid.pop(0)
    self._euler.pop(0)
    if self._body_id > 0 and self._body_id < app.model.nbody:
      self._centroid.append(app.data.xpos[self._body_id].copy())
      quat = app.data.xquat[self._body_id]
      mat = np.zeros(9)
      mujoco.mju_quat2Mat(mat, quat)
      # mat is row-major 3x3: R[i,j] = mat[3*i + j].
      roll = math.atan2(mat[7], mat[8])
      pitch = math.atan2(-mat[6], math.sqrt(mat[7] ** 2 + mat[8] ** 2))
      yaw = math.atan2(mat[3], mat[0])
      self._euler.append(np.degrees(np.array([roll, pitch, yaw])))
    else:
      self._centroid.append(np.zeros(3))
      self._euler.append(np.zeros(3))
