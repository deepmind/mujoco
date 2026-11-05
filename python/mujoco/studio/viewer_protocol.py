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
"""Base class and configuration for any viewer."""

import abc
import contextlib
import copy
import dataclasses
import enum
import logging
from typing import Any
import mujoco
from mujoco.studio import endpoints
from mujoco.studio import hot_reload
from mujoco.studio import messages
from mujoco.studio import plugin_registry
from mujoco.studio import ux

logger = logging.getLogger(__name__)

GFX_MODES = (
    'opengl',
    'opengl_headless',
    'opengl_software',
    'vulkan',
    'vulkan_software',
    'web',
    'webgl',
)


# -----------------------------------------------------------------------------
# Viewer configuration.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class ViewerConfig:
  """Common configuration for creating a viewer window."""

  title: str = ''
  width: int = 1200
  height: int = 800
  gfx: str = ''  # Graphics mode ('web' launches Web Viewer).
  http_port: int = 0  # Web Viewer port (0 picks first free port >= 8080).
  open_browser: bool = False  # Open Web Viewer URL in a browser tab on launch.
  # What is reloaded when modified. Options are '', 'plugins', 'model', 'all',
  # 'Plugin1,Plugin2,...'. Default is '', meaning nothing.
  hot_reload: str = ''


# -----------------------------------------------------------------------------
# Viewer lifecycle events.
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ViewerInitEvent(messages.Event):
  """Lifecycle event dispatched locally once when a ``Viewer`` initializes.

  ``plugin.ViewerPlugin`` caches ``event.viewer`` as ``self.viewer`` above
  ``CRITICAL`` priority, so ``self.viewer`` is already set when subclass
  handlers run.
  """

  viewer: 'Viewer'


# -----------------------------------------------------------------------------
# Base class for any viewer.
# -----------------------------------------------------------------------------


class Viewer(abc.ABC):
  """Base class for any viewer.

  Owns the communication endpoint, plugin registry and core visualization
  objects. The application is rendered by calling ``sync()``.
  """

  def __init__(
      self,
      config: ViewerConfig,
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
    """Initializes the Viewer.

    Args:
      config: Viewer window configuration.
      endpoint: The viewer endpoint for communication with the sim side.
      model: Optional initial MjModel. If None, an empty model is created from
        an empty MjSpec. The Viewer deep-copies this model and creates its own
        MjData.
      plugins: Optional list of plugin instances for viewer-side processing.
      camera: Camera parameters. Internal object is created if None.
      vis_options: Visualization options. Internal object is created if None.
      perturb: Perturbation parameters. Internal object is created if None.
      render_flags: Render flags. Internal object is created if None.
      extra_geoms: List of extra geoms. Internal list is created if None.
    """
    self.config = config
    self._endpoint = endpoint
    self._is_running = True
    self.exit_reason = ''

    # Viewer-owned model and data.
    if model is None:
      model = mujoco.MjSpec().compile()
    self.model: mujoco.MjModel
    self.data: mujoco.MjData
    self.model_path: str = ''
    self.load_model(model)

    # Visual state.
    self.camera = camera or mujoco.MjvCamera()
    self.cam_speed = 0.001
    self.perturb = perturb or mujoco.MjvPerturb()
    self.vis_options = vis_options or mujoco.MjvOption()
    self.extra_geoms = extra_geoms or []
    if render_flags is not None:
      self.render_flags = render_flags
    else:
      self.render_flags = ux.RenderFlags()
      # Initted to match mujoco/src/engine/engine_vis_init.c
      self.render_flags.flags = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]

    # Plugin infrastructure.
    all_plugins = [self] + list(plugins or [])
    self.plugins = plugin_registry.PluginRegistry(all_plugins)
    if config.hot_reload:
      # Process-wide; also covers the sim's plugins.
      hot_reload.enable(config.hot_reload)

  def close(self) -> None:
    """Closes the viewer, sends an exit event and shuts down the endpoint."""
    # Only attach exit_reason when the exit originated on the viewer side; if
    # _on_exit already handled an ExitEvent from the sim, do not echo the sim's
    # reason back to it.
    reason = self.exit_reason if self._is_running else ''
    exit_event = messages.ExitEvent(reason=reason)
    try:
      if self._is_running:
        self._is_running = False
        self.dispatch(exit_event)
    finally:  # Finish the teardown even if an ExitEvent handler raised.
      try:
        self.send_to_sim(exit_event)
      except Exception:  # pylint: disable=broad-exception-caught
        pass  # Ignore exceptions, the sim may have already closed.
      self._endpoint.close()

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_exit(self, event: messages.ExitEvent) -> None:
    """Stops the viewer loop when the sim side requests an exit."""
    self._is_running = False
    if event.reason:
      self.exit_reason = event.reason

  def is_running(self) -> bool:
    """Returns True while the viewer has not been closed."""
    return self._is_running

  def send_to_sim(self, message: messages.Message) -> None:
    """Sends a message to the simulation process."""
    self._endpoint.send_to_sim(message)

  def send_if_changes(
      self, *make_messages: messages.MakeMessage
  ) -> contextlib.AbstractContextManager[None]:
    """Sends the messages that change during the ``with`` block to the sim.

    Calls each ``make_message()`` on entering and on leaving the block and
    sends those that differ. Nothing is sent if the viewer stops running or
    its model is replaced during the block.
    """
    model = self.model
    return messages.send_if_changes(
        self.send_to_sim,
        *make_messages,
        should_send=lambda: self._is_running and self.model is model,
    )

  def get_sim_events(self) -> list[messages.Event]:
    """Returns all pending events from the simulation."""
    return self._endpoint.get_sim_events()

  def get_sim_snapshots(self) -> list[messages.Snapshot]:
    """Returns all pending latest snapshots from the simulation, one per type."""
    return self._endpoint.get_sim_snapshots()

  def dispatch(self, message: messages.Message) -> None:
    """Dispatches a message to registered handlers in priority order."""
    self.plugins.dispatch(message)

  def load_model(self, model: mujoco.MjModel, model_path: str = '') -> None:
    """Deep-copies a model and creates fresh data for the viewer."""
    self.model_path = model_path
    self.model = copy.deepcopy(model)
    self.data = mujoco.MjData(self.model)
    assert id(self.model) != id(model)
    mujoco.mj_forward(self.model, self.data)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_model(self, event: messages.ModelEvent) -> None:
    """Deep-copies the incoming model so the Viewer owns its data."""
    self.load_model(event.model, event.path)
    self.extra_geoms.clear()
    self.dispatch(
        messages.PostModelEvent(
            model=self.model, data=self.data, path=event.path
        )
    )

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_mjoption(self, event: messages.MjOptionSnapshot) -> None:
    """Applies incoming simulation physics options to the viewer's model."""
    self.model.opt = event.opt

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_state(self, event: messages.StateSnapshot) -> None:
    """Applies incoming simulation state to the viewer's model/data."""
    state_size = mujoco.mj_stateSize(self.model, event.state_sig)
    if len(event.state) == state_size:
      mujoco.mj_setState(self.model, self.data, event.state, event.state_sig)
      mujoco.mj_forward(self.model, self.data)

  @abc.abstractmethod
  def prepare_next_frame(self) -> bool:
    """Advances to the next frame; returns whether one is ready to render."""
    ...

  @abc.abstractmethod
  def sync(self) -> None:
    """Renders the scene using the viewer's current model and data."""
    ...

  @abc.abstractmethod
  def get_drop_file(self) -> str:
    ...

  @abc.abstractmethod
  def upload_image(
      self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int
  ) -> int:
    ...


# -----------------------------------------------------------------------------
# Standalone viewer loop.
# -----------------------------------------------------------------------------


def run_viewer_loop(viewer: Viewer) -> None:
  """Minimal viewer loop: process sim messages, dispatch lifecycle events, sync.

  Runs until the viewer window is closed or an exit event is received.
  On exit, closes the viewer (which sends an ExitEvent to the sim side). The
  viewer is closed even if a plugin handler raises, so its resources are
  released on this thread and the sim side is told to exit before the
  exception propagates.

  An unhandled exception is also logged here, on the viewer thread, as soon as
  it happens: the sim side only re-raises it the next time it touches the
  handle, which may be much later or never (e.g. a notebook cell that has
  already returned).

  Args:
    viewer: A Viewer that owns the endpoint and plugin registry.
  """
  try:
    while True:
      # Apply edited plugin code between frames (no-op unless enabled).
      hot_reload.tick(viewer.plugins, viewer)

      # Get the next frame; this also gets the frame's mouse/keyboard events.
      frame = viewer.prepare_next_frame()

      # Process incoming simulation events.
      for event in viewer.get_sim_events():
        viewer.dispatch(event)

      # Stop the loop if the viewer is not running (endpoint will be closed).
      if not viewer.is_running():
        break

      # Process incoming simulation snapshots.
      for snapshot in viewer.get_sim_snapshots():
        viewer.dispatch(snapshot)

      # Skip rendering when prepare_next_frame returned no active frame.
      # e.g., no browser is connected to the web viewer.
      if frame:
        # Dispatch lifecycle events.
        viewer.dispatch(messages.UpdateEvent())
        viewer.dispatch(messages.BuildGuiEvent())

        # Render the scene.
        viewer.sync()
  except Exception as ex:
    viewer.exit_reason = messages.describe_exception(ex)
    logger.exception(
        'Unhandled exception on the viewer thread; the viewer is closing.'
    )
    raise
  finally:
    viewer.close()
