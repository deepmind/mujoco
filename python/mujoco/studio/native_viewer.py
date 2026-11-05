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
"""Simulation-agnostic native viewer for MuJoCo models.

See the documentation for app/plugins.py for more details on the architecture
separating the viewer and simulation.
"""

from typing import Any

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import renderer
from mujoco.studio import endpoints
from mujoco.studio import messages
from mujoco.studio import native_viewer_cc as _viewer
from mujoco.studio import ux
from mujoco.studio import viewer_protocol
from mujoco.studio import window
import numpy as np

from mujoco.studio.dear_imgui import dear_imgui as imgui
from mujoco.studio.implot import implot


class NativeViewer(viewer_protocol.Viewer):
  """Simulation-agnostic native viewer for MuJoCo models."""

  def __init__(
      self,
      config: viewer_protocol.ViewerConfig,
      endpoint: endpoints.ViewerEndpoint,
      *,
      model: mujoco.MjModel | None = None,
      plugins: list[Any] | None = None,
      camera: mujoco.MjvCamera | None = None,
      vis_options: mujoco.MjvOption | None = None,
      perturb: mujoco.MjvPerturb | None = None,
      render_flags: ux.RenderFlags | None = None,
      extra_geoms: list[mujoco.MjvGeom] | None = None,
  ) -> None:
    """Initializes the NativeViewer.

    Args:
      config: Viewer window configuration.
      endpoint: The viewer endpoint for communication with the sim side.
      model: Optional initial MjModel. Forwarded to the base Viewer.
      plugins: Optional list of plugin instances.
      camera: Camera parameters. Internal object is created if None.
      vis_options: Visualization options. Internal object is created if None.
      perturb: Perturbation parameters. Internal object is created if None.
      render_flags: Render flags. Internal object is created if None.
      extra_geoms: List of extra geoms. Internal list is created if None.
    """
    super().__init__(
        config,
        endpoint,
        model=model,
        plugins=plugins,
        camera=camera,
        vis_options=vis_options,
        perturb=perturb,
        render_flags=render_flags,
        extra_geoms=extra_geoms,
    )

    gfx_mode = config.gfx or ''

    self._window = window.Window(
        config.title,
        config.width,
        config.height,
        gfx_mode,
    )

    mjrf_ctx = mjrf.Context(
        mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL)
    )
    self._renderer = renderer.Renderer(mjrf_ctx)
    self._model_scene = None
    self._imgui_bridge = _viewer.ImguiBridge(
        mjrf_ctx, self._renderer.scene('ux')
    )

    # Track the python object id of the model currently loaded in the renderer
    # so we can detect when the model changes and re-initialize.
    self._renderer_model_id = id(None)

    ctx = window.GetImGuiContext()
    imgui.SetCurrentContext(ctx)
    ux.set_imgui_context(ctx)
    ux.set_implot_context(window.GetImPlotContext())
    implot.set_imgui_context(ctx)
    implot.set_implot_context(window.GetImPlotContext())

    # Dispatch lifecycle event so handlers can cache the viewer reference.
    try:
      self.dispatch(viewer_protocol.ViewerInitEvent(viewer=self))
    except BaseException:
      self.close()  # Release the C++ viewer on this thread, see close().
      raise

  def _sync_renderer(self, model: mujoco.MjModel) -> None:
    """Re-initializes the renderer if the model object has changed."""
    if id(model) != self._renderer_model_id:
      self._model_scene = None
      self._model_scene = _viewer.ModelScene(
          self._renderer.context, self._renderer.scene('scn'), model
      )
      self._renderer_model_id = id(model)

  def prepare_next_frame(self) -> bool:
    """Advances to the next frame; returns False when the window is closed."""
    if not self._window.NewFrame():
      self.dispatch(messages.ExitEvent())
      return False
    return True

  def sync(self) -> None:
    """Render the scene and present it to the window."""
    width = self._window.GetWidth()
    height = self._window.GetHeight()
    scale = self._window.GetScale()

    pixels = None
    if _viewer.render_to_pixels(self._window.GetGraphicsMode()):
      pixels = np.empty((height, width, 3), dtype=np.uint8)
      self._renderer.target('out', pixels)
    else:
      width = int(width * scale)
      height = int(height * scale)

    self._sync_renderer(self.model)
    self._imgui_bridge.update()
    if self._model_scene:
      self._model_scene.sync(
          width,
          height,
          self.model,
          self.data,
          self.perturb,
          self.camera,
          self.vis_options,
          self.extra_geoms,
      )

    draw_mode = mjrf.DrawMode.DRAW_MODE_DEFAULT
    if self.render_flags.flags[mujoco.mjtRndFlag.mjRND_SEGMENT]:
      if self.render_flags.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR]:
        draw_mode = mjrf.DrawMode.DRAW_MODE_SEGMENTATION_BY_ID
      else:
        draw_mode = mjrf.DrawMode.DRAW_MODE_SEGMENTATION_BY_COLOR
    elif self.render_flags.flags[mujoco.mjtRndFlag.mjRND_DEPTH]:
      draw_mode = mjrf.DrawMode.DRAW_MODE_DEPTH
    elif self.render_flags.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME]:
      draw_mode = mjrf.DrawMode.DRAW_MODE_WIREFRAME
    elif self.vis_options.flags[mujoco.mjtVisFlag.mjVIS_ISLAND]:
      draw_mode = mjrf.DrawMode.DRAW_MODE_ISLANDS

    scene_view = renderer.View(
        scene='scn',
        target='out' if pixels is not None else '',
        camera=mujoco.mjv_camera2GLCamera(self.model, self.data, self.camera),
        draw_mode=draw_mode,
        shadows=self.render_flags.flags[mujoco.mjtRndFlag.mjRND_SHADOW],
        reflections=self.render_flags.flags[mujoco.mjtRndFlag.mjRND_REFLECTION],
        skybox=self.render_flags.flags[mujoco.mjtRndFlag.mjRND_SKYBOX],
        haze=self.render_flags.flags[mujoco.mjtRndFlag.mjRND_HAZE],
        post_processing=self.render_flags.flags[
            mujoco.mjtRndFlag.mjRND_POST_PROCESSING
        ],
        headlight=self.model.vis.headlight.active != 0,
        headlight_color=tuple(self.model.vis.headlight.diffuse),
        headlight_intensity=self._model_scene.get_headlight_intensity()
        if self._model_scene
        else 0.0,
    )

    ux_camera = mjrf.Camera(
        orthographic=True,
        pos=[0.0, 0.0, 1.0],
        forward=[0.0, 0.0, -1.0],
        up=[0.0, 1.0, 0.0],
        frustum_top=0.0,
        frustum_near=0.0,
        frustum_far=1.0,
        frustum_center=width / 2.0,
        frustum_width=width / 2.0,
        frustum_bottom=height,
    )
    ux_view = renderer.View(
        scene='ux',
        target='out' if pixels is not None else '',
        camera=ux_camera,
        shadows=False,
        reflections=False,
        skybox=False,
        haze=False,
        post_processing=False,
    )

    # ensure output buffer is of right size (in case of window resize)
    self._renderer.render([scene_view, ux_view])
    self._window.Present(pixels)

  def close(self) -> None:
    """Close the viewer and explicitly destroy the renderer.

    The C++ Viewer (and its FilamentRenderer) must be destroyed on the same
    thread that created it, because Filament's FEngine::destroy() asserts
    thread affinity.  Without this override the pybind11 prevent object would
    be garbage-collected on the main thread, triggering a SIGABRT.
    """
    try:
      if self._is_running:
        self._is_running = False
        self.dispatch(messages.ExitEvent())
    finally:
      # Destroy the C++ viewer *before* closing the endpoint so that the
      # FilamentRenderer destructor runs on the daemon/viewer thread, even if
      # an ExitEvent handler raised.
      del self._model_scene
      del self._imgui_bridge
      del self._renderer
      del self._window
      super().close()

  def get_drop_file(self) -> str:
    """Returns the path of the file dropped into the window, or empty string."""
    return self._window.GetDropFile()

  def upload_image(
      self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int
  ) -> int:
    """Uploads an image to the backend for GUI rendering.

    The ID can be used in subsequent calls to update the texture data. An empty
    `img` argument will free the texture if it exists. A `tex_id` of 0 will
    create a new texture.

    Args:
      tex_id: The texture ID.
      img: The image data as string or bytes.
      width: Width of the image.
      height: Height of the image.
      bpp: Bytes per pixel.

    Returns:
      The texture ID.
    """
    return self._imgui_bridge.upload_image(tex_id, img, width, height, bpp)
