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
"""Web viewer launcher.

Launches the web viewer without depending on desktop windowing or Filament.
"""

from collections.abc import Iterable
from typing import Any

import mujoco
from mujoco.studio import endpoints
from mujoco.studio import launch_thread
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import viewer_handle
from mujoco.studio import viewer_protocol


def run_web_viewer(
    config: viewer_protocol.ViewerConfig,
    endpoint: endpoints.ViewerEndpoint,
    plugins: list[Any] | None = None,
) -> None:
  """Creates a WebViewer and runs the viewer loop.

  Args:
    config: Configuration specifying the viewer settings.
    endpoint: Endpoint for communicating with the simulation side.
    plugins: Optional list of viewer-side plugin instances.
  """
  from mujoco.studio import web_viewer  # pylint: disable=g-import-not-at-top

  viewer = web_viewer.WebViewer(
      config,
      endpoint,
      plugins=plugins,
  )
  viewer_protocol.run_viewer_loop(viewer)


def launch(
    config: viewer_protocol.ViewerConfig | None = None,
    *,
    plugins: Iterable[plugin.Plugin] | None = None,
) -> viewer_handle.ViewerHandle:
  """Launches the WebViewer in a daemon thread without blocking.

  This target does not depend on Filament or desktop windowing, making it
  safe for headless environments, containers, and remote clusters.

  Args:
    config: Optional ViewerConfig (defaults to ViewerConfig(gfx='web')).
    plugins: Optional viewer and sim plugins.

  Returns:
    A ViewerHandle for interacting with the viewer.
  """
  if config is None:
    config = viewer_protocol.ViewerConfig(gfx='web')
  viewer_plugins, sim_plugins = plugin.split_plugins(plugins)

  def target(endpoint: endpoints.ViewerEndpoint) -> None:
    run_web_viewer(
        config,
        endpoint,
        plugins=viewer_plugins,
    )

  return launch_thread.launch_thread(target, sim_plugins=sim_plugins)


def run(
    config: viewer_protocol.ViewerConfig | None = None,
    *,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    model_path: str | None = None,
    plugins: Iterable[plugin.Plugin] | None = None,
) -> None:
  """Launches the WebViewer and runs the blocking simulation sync loop.

  Args:
    config: Optional ViewerConfig (defaults to ViewerConfig(gfx='web')).
    model: In-memory model to publish instead.
    data: The data for ``model``; a fresh one is created when None.
    model_path: Model file to load.
    plugins: Optional viewer and sim plugins.

  Raises:
    ValueError: If both ``model`` and ``model_path`` are given.
  """
  if model is not None and model_path:
    raise ValueError('Pass either model_path or model, not both.')
  with launch(config, plugins=plugins) as handle:
    if model is not None:
      handle.set_model(model, data=data)
    elif model_path:
      handle.dispatch(messages.LoadModelEvent(path=model_path))
    viewer_handle.run_sim_loop(handle)
