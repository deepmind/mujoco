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
r"""Viewer plugin that renders time-delayed semi-transparent ghost geoms.

Run with the Studio viewer::

  python -m mujoco.studio.viewer \
      --model=model/humanoid/humanoid.xml \
      --plugins=python/mujoco/sample/studio/ghost.py
"""

import collections

import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

from mujoco.studio.dear_imgui import dear_imgui as imgui

_PRIMITIVE_GEOM_TYPES = frozenset({
    mujoco.mjtGeom.mjGEOM_PLANE,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_BOX,
})


class GhostRenderer(plugin.ViewerPlugin):
  """Handler that renders time-delayed semi-transparent ghost geoms."""

  def __init__(self) -> None:
    self._delay: float = 0.5
    self._ghost_rgba: list[float] = [0.4, 0.5, 0.9, 0.5]
    self._history: collections.deque[tuple[float, np.ndarray, np.ndarray]] = (
        collections.deque()
    )
    self._last_time: float | None = None

  def _is_fixed_body(self, body_id: int) -> bool:
    """Check if a body is fixed (welded to world, not attached to mocap)."""
    model = self.viewer.model
    is_weld = model.body_weldid[body_id] == 0
    root_id = model.body_rootid[body_id]
    return bool(is_weld and model.body_mocapid[root_id] < 0)

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    """Resets ghost-specific state when the model changes."""
    del event  # Model/data are accessed via self.viewer.
    self._history.clear()
    self._last_time = None

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    """Renders the ghost configuration GUI panel."""
    if imgui.Begin('Ghost', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      changed_delay, delay = imgui.InputFloat(
          'Delay (s)', self._delay, 0.05, 0.5, '%.2f'
      )
      if changed_delay:
        self._delay = max(0.0, delay)

      changed_color, rgba = imgui.ColorEdit4('Color', self._ghost_rgba)
      if changed_color:
        self._ghost_rgba = rgba
    imgui.End()

  @messages.handler
  def on_update(self, _: messages.UpdateEvent) -> None:
    """Updates the time-delayed ghost geoms."""
    model = self.viewer.model
    data = self.viewer.data

    self.viewer.extra_geoms.clear()
    if self._last_time is None or data.time < self._last_time:
      self._history.clear()
    if not self._history or data.time > self._last_time:  # pyrefly: ignore[unsupported-operation]
      self._history.append((
          data.time,
          data.geom_xpos.copy(),
          data.geom_xmat.copy(),
      ))
      self._last_time = data.time

    target_time = data.time - self._delay
    while len(self._history) > 1 and self._history[1][0] <= target_time:
      self._history.popleft()

    _, xpos, xmat = self._history[0]  # pyrefly: ignore[bad-assignment]
    if len(xpos) != model.ngeom or len(xmat) != model.ngeom:
      return

    ghost_rgba = np.array(self._ghost_rgba, dtype=np.float32)
    for i in range(model.ngeom):
      if (
          self._is_fixed_body(model.geom_bodyid[i])
          or model.geom_type[i] not in _PRIMITIVE_GEOM_TYPES
          or model.geom_rgba[i, 3] == 0
          or model.geom_group[i] > 2
      ):
        continue

      geom = mujoco.MjvGeom()
      mujoco.mjv_initGeom(
          geom,
          int(model.geom_type[i]),
          model.geom_size[i],
          xpos[i],
          xmat[i].flatten(),
          ghost_rgba,
      )
      geom.dataid = model.geom_dataid[i]
      geom.objtype = int(mujoco.mjtObj.mjOBJ_GEOM)
      geom.objid = i
      self.viewer.extra_geoms.append(geom)
