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
"""Unified non-blocking viewer launcher (backward-compatible wrapper).

Launches the viewer in a daemon thread and returns a ``ViewerHandle``
that the calling thread uses to push simulation state into the viewer.
"""

from collections.abc import Iterable

import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import viewer_handle
from mujoco.studio import viewer_protocol


def launch(
    config: viewer_protocol.ViewerConfig,
    *,
    plugins: Iterable[plugin.Plugin] | None = None,
) -> viewer_handle.ViewerHandle:
  """Launches the viewer in a daemon thread without blocking.

  The viewer runs on the rendering thread and renders the scene along with any
  GUI built by registered plugins (such as ``studio_app.App``). The
  caller keeps running and pushes state via ``handle.sync()``.

  Args:
    config: Viewer window configuration.
    plugins: Optional viewer and sim plugins. Include a ``SimPlugin`` that
      handles ``StepEvent`` (e.g. ``step_control.StepControl()``) to advance the
      physics and, at minimum, pace each ``sync`` call.

  Returns:
    A ViewerHandle for interacting with the viewer.
  """
  if config.gfx in ('web', 'webgl'):
    from mujoco.studio import launch_web  # pylint: disable=g-import-not-at-top

    return launch_web.launch(config, plugins=plugins)
  from mujoco.studio import launch_native  # pylint: disable=g-import-not-at-top

  return launch_native.launch(config, plugins=plugins)


def run(
    config: viewer_protocol.ViewerConfig,
    *,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    model_path: str | None = None,
    plugins: Iterable[plugin.Plugin] | None = None,
) -> None:
  """Launches the viewer and runs the blocking simulation sync loop.

  Args:
    config: Viewer window configuration.
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
