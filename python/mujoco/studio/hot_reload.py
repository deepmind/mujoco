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
"""Opt-in hot reloading of plugin modules.

Hot reloading is off by default. ``enable()`` turns it on for the whole process
(the viewer calls it with ``ViewerConfig.hot_reload``, i.e. ``--hot_reload``,
which says what to reload: ``all``, ``plugins``, ``model``, or the class names
of the plugins to reload; see ``Options.parse``). The viewer loop calls
``tick()`` once per frame and ``ViewerHandle.sync`` calls it once per step.
When a watched ``.py`` file is saved, ``tick()`` does the following:

1. Reloads the changed module and every watched module that imports it,
   dependencies first.
2. Swaps the class of each affected plugin in place (``plugin.__class__ =
   new_cls``). The instance, its state, and references to it held elsewhere are
   all kept.
3. Rebuilds the registry's handler table and dispatches
   ``HotReloadEvent(plugins=...)`` with the swapped plugins. Once a plugin's
   class has been swapped, exceptions raised by its handlers are caught and
   logged once per handler (rather than terminating the viewer or sim loop), and
   the log suppression resets on the next reload.

Watched modules are the modules of the registry's plugins plus the ``.py``
modules they import from their own directory. Built-in plugins such as
``studio_app.App``, ``StepControl``, and ``SimHistory`` are regular
plugins and are hot-reloaded like any other plugin. Framework modules (the
modules of the registry's host and what they import, ``messages``, ``plugin``,
``plugin_registry``, ``hot_reload_utils``, and this module) are never reloaded;
editing one logs "restart required" (later, when manual restart prompting is
wired into WebViewer, this will also surface a restart button there). Plugins
defined in ``__main__`` (the script being run) cannot be reloaded.

Design goal: a hard boundary between reloadable and fixed code:

- **Reloadable:**
  - The modules of registered plugins (both user plugins and built-in plugins
    such as ``studio_app.App``, ``StepControl``, and ``SimHistory``)
    and their local imports (here).
  - The model file shown and its includes and assets (``ViewerHandle``, via
    ``watch_files``).
- **Fixed:**
  - The framework modules above (``messages``, ``plugin``, ``plugin_registry``,
    ``hot_reload_utils``, ``hot_reload``, and the host viewer/handle modules
    plus their local imports).
  - The C++ binary and ``__main__``.
  - The *set of registered plugin instances*: ``PluginRegistry`` has no late
    registration, ``_register`` scans a registry's plugins once, and a reload
    swaps the class of an existing instance but never creates or registers one.

Keeping the fixed side solid is what makes reloading safe: the reloader only
reasons about code behind instances it already knows, and a bad save can only
break that plugin, never the loop. In practice, the minimal live-coding hook is
one registered viewer plugin and one registered sim plugin whose module you
own; those can grow into anything (new handlers are discovered on the swap),
while adding another instance needs a restart. When adding custom behavior,
prefer registering small separate plugins next to ``studio_app.App`` or
``StepControl`` over subclassing them: handlers are discovered by method name,
so a subclass can accidentally shadow a base-class handler.

State across reloads and ``HotReloadEvent``:

- Declare default attribute values on the class (e.g. ``height: int = 5``).
  Attributes already mutated in ``self.__dict__`` keep their live values across
  reloads (unless the class attribute's default literal in the source file
  changed, in which case the new default takes effect), while any newly added
  class attribute is immediately visible on the existing instance without
  re-running ``__init__``.
- Dispatched ``HotReloadEvent(plugins=...)`` notifications give plugins an
  explicit hook to reconcile state after a live code swap, so any invariant or
  data-shape change introduced while editing can be handled cleanly without
  restarting. Common uses when ``self in event.plugins``:
  - Initialize newly added mutable attributes (buffers, queues, dicts) that
    would normally be created in ``__init__``.
  - Recompute or invalidate cached/derived state (e.g. lookup tables, or cached
    MuJoCo body/joint/geom IDs after changing which names the plugin queries).
  - Migrate or reset live state whose schema or shape changed (e.g. resizing a
    ring buffer or history array, resetting a state machine whose states
    changed, or clearing stale plot series).
  - Trigger an immediate side effect so the edit is visible right away (e.g.
    rebuilding and publishing a procedural ``MjSpec`` scene, or re-sending a UI
    state snapshot).
"""

from collections.abc import Iterable
import dataclasses
import logging
import sys
import threading
import time
import traceback
import types
from typing import Any
import weakref

from mujoco.studio import hot_reload_utils as utils
from mujoco.studio import messages
from mujoco.studio import plugin_registry

FileWatcher = utils.FileWatcher

_HOT_RELOAD_POLL_INTERVAL_S = 0.25


logger = logging.getLogger(__name__)
# INFO is always enabled so status records reach the Studio Console even when
# the root logger is left at its default WARNING level.
logger.setLevel(logging.INFO)


def _log(text: str, level: int = logging.INFO) -> None:
  logger.log(level, '[Studio hot reload] %s', text)


@dataclasses.dataclass(frozen=True)
class Options:
  """What hot reloading covers; ``parse()`` reads it from ``--hot_reload``.

  Attributes:
    model: Whether the model file shown (with its includes and assets) is
      reloaded when saved.
    plugins: Which registered plugins are reloaded when their module is saved:
      all of them (True), none (False), or those whose class has one of the
      given names.
  """

  model: bool = True
  plugins: bool | frozenset[str] = True

  @classmethod
  def parse(cls, text: str) -> 'Options':
    """Parses a ``--hot_reload`` value.

    ``all`` covers plugins and the model, ``plugins`` and ``model`` one of the
    two, and anything else names a plugin class to reload, e.g.
    ``Scene,SceneGui``. Items combine with commas (``model,Scene``).

    Args:
      text: The flag value; empty for nothing.

    Raises:
      ValueError: If an item is neither a keyword nor a valid class name.
    """
    model = all_plugins = False
    names: set[str] = set()
    for item in text.split(','):
      item = item.strip()
      if item == 'all':
        model = all_plugins = True
      elif item == 'plugins':
        all_plugins = True
      elif item == 'model':
        model = True
      elif item.isidentifier():
        names.add(item)
      elif item:
        raise ValueError(
            f'--hot_reload item {item!r} is not all, plugins, model, or a'
            ' plugin class name.'
        )
    return cls(model=model, plugins=all_plugins or frozenset(names) or False)

  def reloads(self, plugin: Any) -> bool:
    """Returns whether ``plugin`` is reloaded when its module is saved."""
    if isinstance(self.plugins, bool):
      return self.plugins
    return type(plugin).__name__ in self.plugins


@dataclasses.dataclass
class _RegistryState:
  """Per-registry bookkeeping."""

  host: Any
  seen_generation: int
  # id() of plugins whose class has been swapped at least once.
  swapped: set[int] = dataclasses.field(default_factory=set)
  # (plugin id, handler name) pairs whose exception has been logged.
  logged: set[tuple[int, str]] = dataclasses.field(default_factory=set)


class Reloader:
  """Watches plugin modules and hot-reloads them into plugin registries.

  One instance serves every registry in the process, since modules are shared.
  Each registry is updated only from its own ``tick()`` call, i.e. on its own
  thread and between frames or steps.
  """

  def __init__(
      self,
      options: Options = Options(),
      poll_interval: float = _HOT_RELOAD_POLL_INTERVAL_S,
  ) -> None:
    self._options = options
    self._poll_interval = poll_interval
    self._next_poll = 0.0
    self._lock = threading.Lock()
    self._files = FileWatcher()
    # A file can be imported under several names (e.g. `helper` and
    # `studio_plugins.helper`); a change must reload all of them.
    self._path_to_modules: dict[str, set[str]] = {}
    # Reloadable modules -> the reloadable modules they import.
    self._deps: dict[str, set[str]] = {}
    self._pinned: set[str] = set()
    self._failed: set[str] = set()
    self._warned: set[str] = set()
    self._registries: weakref.WeakKeyDictionary[Any, _RegistryState] = (
        weakref.WeakKeyDictionary()
    )
    self.generation = 0

  @property
  def options(self) -> Options:
    """What this reloader covers."""
    return self._options

  @property
  def watched_modules(self) -> set[str]:
    """Names of the modules that are reloaded when their source changes."""
    return set(self._deps)

  @property
  def poll_interval(self) -> float:
    """Minimum seconds between two checks of the watched files."""
    return self._poll_interval

  def tick(self, registry: plugin_registry.PluginRegistry, host: Any) -> None:
    """Reloads changed modules and updates ``registry``'s plugins.

    Args:
      registry: The registry to update.
      host: The object that owns ``registry`` (a Viewer or ViewerHandle). Its
        modules are framework code and are never reloaded.
    """
    state = self._registries.get(registry)
    if state is None:
      state = self._register(registry, host)

    now = time.monotonic()
    # Another thread may be reloading; it will bump the generation when done.
    if now >= self._next_poll and self._lock.acquire(blocking=False):
      try:
        self._next_poll = now + self._poll_interval
        self._reload_changed()
      finally:
        self._lock.release()

    if state.seen_generation != self.generation:
      state.seen_generation = self.generation
      self._swap_classes(registry, state)

  # -- Watch set ---------------------------------------------------------------

  def _register(
      self, registry: plugin_registry.PluginRegistry, host: Any
  ) -> _RegistryState:
    """Adds a registry's plugin modules to the watch set."""
    with self._lock:
      for module in (
          sys.modules[__name__],
          utils,
          messages,
          plugin_registry,
      ):
        self._pin(module)
      # The host class and its bases from the same directory (e.g. WebViewer
      # and Viewer) are framework code.
      host_dir = utils.module_dir(type(host).__module__)
      for cls in type(host).__mro__:
        module = sys.modules.get(cls.__module__)
        if module is not None and utils.module_dir(cls.__module__) == host_dir:
          self._pin(module)

      for plugin in registry.plugins:
        if plugin is host or not self._options.reloads(plugin):
          continue
        cls = type(plugin)
        if cls.__module__ == '__main__':
          self._warn_once(
              utils.class_name(cls),
              f'{cls.__qualname__} is defined in the script being run'
              ' (__main__); move it into a module to hot-reload it.',
          )
          continue
        module = sys.modules.get(cls.__module__)
        if module is not None:
          self._watch(module)
        if (
            cls.__module__ not in self._deps
            and cls.__module__ not in self._pinned
        ):
          self._warn_once(
              utils.class_name(cls),
              f'Cannot hot-reload {cls.__qualname__}: module'
              f' {cls.__module__!r} has no .py source file.',
          )

      _log('Watching ' + (', '.join(sorted(self._deps)) or 'nothing'))
      state = _RegistryState(host=host, seen_generation=self.generation)
      self._registries[registry] = state
      return state

  def _watch_file(self, module: types.ModuleType) -> bool:
    path = utils.source_path(module)
    if path is None:
      return False
    self._path_to_modules.setdefault(self._files.watch(path), set()).add(
        module.__name__
    )
    return True

  def _pin(self, module: types.ModuleType) -> None:
    """Marks a module and its local imports as never reloaded."""
    name = module.__name__
    if name in self._pinned:
      return
    self._pinned.add(name)
    self._deps.pop(name, None)
    if self._watch_file(module):
      for dep in utils.local_imports(module):
        self._pin(dep)

  def _watch(self, module: types.ModuleType) -> None:
    """Adds a module and its local imports to the reloadable set."""
    name = module.__name__
    if name in self._deps or name in self._pinned or name == '__main__':
      return
    if not self._watch_file(module):
      return
    self._deps[name] = set()
    self._update_deps(module)

  def _update_deps(self, module: types.ModuleType) -> None:
    deps = set()
    for dep in utils.local_imports(module):
      self._watch(dep)
      if dep.__name__ in self._deps:
        deps.add(dep.__name__)
    self._deps[module.__name__] = deps

  def _warn_once(self, key: str, text: str) -> None:
    if key not in self._warned:
      self._warned.add(key)
      _log(text, logging.WARNING)

  # -- Reloading ---------------------------------------------------------------

  def _reload_changed(self) -> None:
    """Reloads modules whose files changed, plus the modules importing them."""
    changed = set()
    for path in self._files.poll():
      names = self._path_to_modules.get(path, set())
      if names & self._pinned:
        # TODO(matijak): Also surface a restart prompt in the WebViewer.
        _log(
            f'{path} changed; restart required (not hot-reloadable).',
            logging.WARNING,
        )
      changed |= names & self._deps.keys()
    if not changed:
      return

    # Retry previous failures too: related files are often saved one by one.
    failed = set()
    reloaded = []
    for name in utils.reload_order(self._deps, changed | self._failed):
      module = sys.modules.get(name)
      if module is None:
        continue
      if self._deps[name] & failed:
        failed.add(name)  # Retry once its dependency reloads.
        continue
      try:
        utils.reload_module(module)
      except Exception as e:  # pylint: disable=broad-exception-caught
        # A syntax error's location is all that matters, not importlib frames.
        details = (
            traceback.format_exception_only(e)
            if isinstance(e, SyntaxError)
            else traceback.format_exception(e)
        )
        _log(
            f'Reloading {name} failed; keeping the previous code.\n'
            + ''.join(details).rstrip('\n'),
            logging.ERROR,
        )
        failed.add(name)
        continue
      self._update_deps(module)
      reloaded.append(name)

    self._failed = failed
    if reloaded:
      _log('Reloaded ' + ', '.join(reloaded))
      self.generation += 1

  # -- Swapping ----------------------------------------------------------------

  def _swap_classes(
      self, registry: plugin_registry.PluginRegistry, state: _RegistryState
  ) -> None:
    """Points plugins at their reloaded classes and rebuilds handler tables."""
    swapped = []
    for plugin in registry.plugins:
      if plugin is state.host or not self._options.reloads(plugin):
        continue
      old_cls = type(plugin)
      new_cls = utils.current_class(old_cls)
      if new_cls is None:
        cls_name = utils.class_name(old_cls)
        self._warn_once(
            f'missing:{cls_name}@{self.generation}',
            f'Cannot hot-reload {cls_name}: class was renamed or removed'
            ' from its module; restart required.',
        )
        continue
      if new_cls is old_cls:
        continue
      try:
        plugin.__class__ = new_cls
      except TypeError as e:
        cls_name = utils.class_name(old_cls)
        self._warn_once(
            f'{cls_name}@{self.generation}',
            f'Cannot hot-reload {cls_name} ({e}); restart required.',
        )
        continue
      # If a class-level default was edited in the source file, drop the
      # shadowed instance attribute so the new default takes effect.
      for attr in list(vars(plugin)):
        if (
            attr in old_cls.__dict__
            and attr in new_cls.__dict__
            and not utils.same_default(
                old_cls.__dict__[attr], new_cls.__dict__[attr]
            )
        ):
          delattr(plugin, attr)
      swapped.append(plugin)
      state.swapped.add(id(plugin))
    if not swapped:
      return

    state.logged.clear()
    registry.rebuild_handlers(
        wrap=lambda p, b: (
            self._log_exceptions(p, b, state) if id(p) in state.swapped else b
        )
    )

    _log('Updated ' + ', '.join(type(p).__qualname__ for p in swapped))
    registry.dispatch(messages.HotReloadEvent(plugins=tuple(swapped)))

  def _log_exceptions(
      self,
      plugin: Any,
      bound: plugin_registry.Handler,
      state: _RegistryState,
  ) -> plugin_registry.Handler:
    """Wraps a reloaded plugin's handler to log its exceptions once."""
    name = getattr(bound, '__name__', repr(bound))
    key = (id(plugin), name)
    label = f'{type(plugin).__qualname__}.{name}'

    def handler(message: messages.Message) -> Any:
      try:
        return bound(message)
      except Exception:  # pylint: disable=broad-exception-caught
        if key not in state.logged:
          state.logged.add(key)
          _log(
              f'{label} raised (repeats hidden until the next reload):\n'
              + traceback.format_exc().rstrip('\n'),
              logging.ERROR,
          )
        return None

    return handler


# -----------------------------------------------------------------------------
# Process-wide switch.
# -----------------------------------------------------------------------------

_reloader: Reloader | None = None
_enable_lock = threading.Lock()


def enable(what: str = 'all') -> None:
  """Configures hot reloading for this process ('' disables it).

  Calling ``enable`` again updates what is reloaded.

  Args:
    what: What to reload; see ``Options.parse``.
  """
  global _reloader
  options = Options.parse(what)
  with _enable_lock:
    if not (options.model or options.plugins):
      _reloader = None
    elif _reloader is None or _reloader.options != options:
      _reloader = Reloader(options)


def watches_model() -> bool:
  """Returns whether the model file shown is reloaded when saved."""
  reloader = _reloader
  return reloader is not None and reloader.options.model


def watch_files(paths: Iterable[str]) -> FileWatcher | None:
  """Returns a watcher for the files a model is built from, if enabled.

  The watcher polls at the reloader's rate, so polling it every step is cheap.

  Args:
    paths: The files to watch.
  """
  reloader = _reloader
  if reloader is None or not reloader.options.model:
    return None
  return FileWatcher(paths, poll_interval=reloader.poll_interval)


def tick(registry: plugin_registry.PluginRegistry, host: Any) -> None:
  """Hot-reloads changed plugin modules into ``registry``, if enabled.

  Call once per frame or step, never from inside a handler.

  Args:
    registry: The registry to update.
    host: The object that owns ``registry`` (a Viewer or ViewerHandle).
  """
  reloader = _reloader
  if reloader is not None and reloader.options.plugins:
    reloader.tick(registry, host)
