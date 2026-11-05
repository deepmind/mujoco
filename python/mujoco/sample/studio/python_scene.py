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
r"""A scene built in Python (`MyScene`) paired with UI (`MySceneGui`).

Run with the Studio viewer::

  python -m mujoco.studio.viewer --hot_reload=all \
      --plugins=python/mujoco/sample/studio/python_scene.py

Interact with the UI, or edit ``on_load_model()`` and save, and the scene is
rebuilt in place with the camera where it was.
"""

import dataclasses

import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin

from mujoco.studio.dear_imgui import dear_imgui as imgui


@dataclasses.dataclass(frozen=True)
class MySceneConfigSnapshot(messages.Snapshot):
  """Describes UI configuration data and transfers it from viewer to sim."""

  height: int


class MyScene(plugin.SimPlugin):
  """Provides the model, applies UI requests, and handles reloading."""

  height: int = 5

  @messages.handler
  def on_load_model(self, event: messages.LoadModelEvent) -> bool:
    # Ignoring event.path keeps this scene when a file is dropped on the viewer;
    # `if event.path: return False` would show the file instead.
    del event
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1])
    for i in range(self.height):
      body = spec.worldbody.add_body(pos=[0, 0, 0.3 + 0.25 * i])
      body.add_freejoint()
      body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=[0.1, 0.1, 0.1],
          rgba=[0.2, 0.5, 0.9, 1],
      )
    self.handle.set_model(spec)
    return True

  @messages.handler
  def on_scene_config(self, snapshot: MySceneConfigSnapshot) -> None:
    self.height = snapshot.height
    self.handle.reload()

  @messages.handler
  def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
    if self in event.plugins:
      self.handle.reload()


class MySceneGui(plugin.ViewerPlugin):
  """Provides UI for tweaking MyScene."""

  height: int = MyScene.height

  @messages.handler
  def build_gui(self, _: messages.BuildGuiEvent) -> None:
    if imgui.Begin('Scene', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      changed, self.height = imgui.SliderInt('Height', self.height, 1, 15)
      if changed:
        self.viewer.send_to_sim(MySceneConfigSnapshot(height=self.height))
    imgui.End()
