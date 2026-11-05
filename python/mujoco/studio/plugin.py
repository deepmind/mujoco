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
"""Base classes for Studio plugins.

A plugin is an object whose methods are decorated with ``messages.handler``;
``plugin_registry.PluginRegistry`` discovers those methods and dispatches
messages to them in priority order. Plugins run on one of two endpoints:

* ``ViewerPlugin`` subclasses run on the viewer side (UI, rendering) and get
  the ``viewer_protocol.Viewer`` as ``self.viewer``.
* ``SimPlugin`` subclasses run on the simulation side (physics, control) and
  get the ``viewer_handle.ViewerHandle`` as ``self.handle``.

The launchers accept both kinds in one ``plugins`` list and route each to its
endpoint (see ``split_plugins``).

Plugins can be wired up in two ways:

* **In code:** import your plugin modules and pass instances in
  ``plugins=[...]`` to ``launch_passive.run`` (or ``launch_web`` /
  ``launch_native``), as ``viewer.py`` does for ``studio_app.App``,
  ``StepControl``, and ``SimHistory``. Plugins can then take constructor
  arguments and use any dependency of the entry point.
* **From files:** ``viewer.py --plugins=a.py,b.py`` calls ``load_plugins``,
  which imports each file and instantiates every ``ViewerPlugin`` and
  ``SimPlugin`` subclass defined in it with ``cls()``.
"""

from collections.abc import Iterable
import importlib
import os
import sys
import types

from mujoco.studio import messages
from mujoco.studio import viewer_handle
from mujoco.studio import viewer_protocol

# The lifecycle events that hand plugins their host, re-exported from the
# modules that dispatch them so plugin authors only need this module.
SimInitEvent = viewer_handle.SimInitEvent
ViewerInitEvent = viewer_protocol.ViewerInitEvent


class SimPlugin:
  """Base class for simulation-side plugins.

  Automatically caches ``self.handle`` (``self._handle``) when ``SimInitEvent``
  is dispatched; stored as a class attribute default so subclasses needn't call
  ``super().__init__()``.
  """

  _handle: viewer_handle.ViewerHandle | None = None

  @property
  def handle(self) -> viewer_handle.ViewerHandle:
    assert self._handle is not None
    return self._handle

  @handle.setter
  def handle(self, value: viewer_handle.ViewerHandle | None) -> None:
    self._handle = value

  # Prefixed so a subclass defining its own ``_on_sim_init`` or ``on_sim_init``
  # doesn't shadow this handler (handlers are discovered by method name), and
  # given priority above CRITICAL so self._handle is populated before all
  # subclass handlers run.
  @messages.handler(priority=messages.Priority.CRITICAL + 1)
  def _sim_plugin_on_sim_init(self, event: SimInitEvent) -> None:
    self._handle = event.handle


class ViewerPlugin:
  """Base class for viewer-side plugins.

  Automatically caches ``self.viewer`` (``self._viewer``) when
  ``ViewerInitEvent`` is dispatched; stored as a class attribute default so
  subclasses needn't call ``super().__init__()``.
  """

  _viewer: viewer_protocol.Viewer | None = None

  @property
  def viewer(self) -> viewer_protocol.Viewer:
    assert self._viewer is not None
    return self._viewer

  @viewer.setter
  def viewer(self, value: viewer_protocol.Viewer | None) -> None:
    self._viewer = value

  # Prefixed so a subclass defining its own ``_on_viewer_init`` or
  # ``on_viewer_init`` doesn't shadow this handler (handlers are discovered by
  # method name), and given priority above CRITICAL so self._viewer is populated
  # before all subclass handlers run.
  @messages.handler(priority=messages.Priority.CRITICAL + 1)
  def _viewer_plugin_on_viewer_init(self, event: ViewerInitEvent) -> None:
    self._viewer = event.viewer


Plugin = SimPlugin | ViewerPlugin


def split_plugins(
    plugins: Iterable[Plugin] | None = None,
) -> tuple[list[ViewerPlugin], list[SimPlugin]]:
  """Splits a sequence of plugins into ``(viewer_plugins, sim_plugins)``.

  Args:
    plugins: Iterable of ``ViewerPlugin`` and ``SimPlugin`` instances.

  Returns:
    A ``(viewer_plugins, sim_plugins)`` tuple preserving input order.

  Raises:
    TypeError: If an item is neither a ``ViewerPlugin`` nor a ``SimPlugin``,
      inherits from both, or is a class rather than an instance.
  """
  viewer_plugins: list[ViewerPlugin] = []
  sim_plugins: list[SimPlugin] = []
  for plugin in plugins or ():
    if isinstance(plugin, ViewerPlugin) and isinstance(plugin, SimPlugin):
      raise TypeError(
          f'Plugin {plugin!r} cannot inherit from both ViewerPlugin and'
          ' SimPlugin; viewer and sim plugins run on separate endpoints.'
      )
    if isinstance(plugin, ViewerPlugin):
      viewer_plugins.append(plugin)
    elif isinstance(plugin, SimPlugin):
      sim_plugins.append(plugin)
    elif isinstance(plugin, type) and issubclass(
        plugin, (ViewerPlugin, SimPlugin)
    ):
      raise TypeError(
          f'Expected a plugin instance, got class {plugin.__qualname__};'
          f' pass {plugin.__qualname__}() instead.'
      )
    else:
      raise TypeError(
          f'Plugin {plugin!r} must subclass ViewerPlugin or SimPlugin.'
      )
  return viewer_plugins, sim_plugins


# Plugin files are imported as submodules of this synthetic package, whose
# ``__path__`` lists their directories. That keeps plugin file names out of the
# top-level namespace and lets the regular import machinery find the files, so
# ``importlib.reload`` (and thus ``hot_reload``) works without any custom code.
PLUGIN_PACKAGE = 'studio_plugins'


def load_plugins(
    paths: Iterable[str | os.PathLike[str]] | None = None,
) -> list[Plugin]:
  """Imports ``.py`` files and instantiates every plugin class defined in them.

  A file ``dir/name.py`` is imported as ``studio_plugins.name``, and ``dir`` is
  appended to ``sys.path`` so the file can import helper files next to it
  (``import helper``). Every ``ViewerPlugin`` or ``SimPlugin`` subclass defined
  in the file (not merely imported into it) is instantiated with ``cls()``, in
  definition order. The file runs inside the current process, so it can only
  import packages present in the running binary's environment.

  Args:
    paths: Paths to ``.py`` files defining plugin classes. Empty entries are
      skipped.

  Returns:
    The plugin instances, in file order then definition order.

  Raises:
    FileNotFoundError: If a path is not a file.
    ImportError: If a file cannot be imported, or another file with the same
      name was loaded already.
    ValueError: If a file defines no ``ViewerPlugin`` or ``SimPlugin`` subclass.
  """
  package = sys.modules.get(PLUGIN_PACKAGE)
  if package is None:
    package = sys.modules[PLUGIN_PACKAGE] = types.ModuleType(PLUGIN_PACKAGE)
    package.__path__ = []
  plugins: list[Plugin] = []
  for raw in paths or ():
    if not os.fspath(raw).strip():
      continue
    path = os.path.abspath(os.fspath(raw).strip())
    if not os.path.isfile(path):
      raise FileNotFoundError(f'Plugin file not found: {path}')
    directory, filename = os.path.split(path)
    for entries in (package.__path__, sys.path):
      if directory not in entries:
        entries.append(directory)
    name = f'{PLUGIN_PACKAGE}.{os.path.splitext(filename)[0]}'
    module = importlib.import_module(name)
    if os.path.realpath(module.__file__ or '') != os.path.realpath(path):
      raise ImportError(
          f'Cannot load {path}: {filename} was already loaded from'
          f' {module.__file__}; rename one of the files.'
      )
    classes: list[type[Plugin]] = []
    for obj in vars(module).values():
      if (
          isinstance(obj, type)
          and issubclass(obj, (ViewerPlugin, SimPlugin))
          and obj.__module__ == name
          and obj not in classes
      ):
        classes.append(obj)
    if not classes:
      raise ValueError(f'{path} defines no ViewerPlugin or SimPlugin subclass.')
    plugins.extend(cls() for cls in classes)
  return plugins
