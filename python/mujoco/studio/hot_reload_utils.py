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
"""File-watching, module-graph, and class helpers for hot_reload."""

import collections
from collections.abc import Iterable, Mapping
import contextlib
import dataclasses
import enum
import importlib
import os
import sys
import time
import types
from typing import Any


def _stat(path: str) -> tuple[int, int] | None:
  try:
    st = os.stat(path)
  except OSError:
    return None
  return st.st_mtime_ns, st.st_size


class FileWatcher:
  """Detects file changes by polling ``os.stat``.

  Paths are resolved with ``os.path.realpath`` so that edits to the workspace
  file behind a runfiles symlink are detected.
  """

  def __init__(
      self, paths: Iterable[str] = (), poll_interval: float = 0.0
  ) -> None:
    """Initializes the watcher.

    Args:
      paths: Files to start watching.
      poll_interval: Minimum seconds between two polls that check the files;
        ``poll()`` calls in between return nothing.
    """
    self._stats: dict[str, tuple[int, int] | None] = {}
    self._poll_interval = poll_interval
    self._next_poll = 0.0
    for path in paths:
      self.watch(path)

  def watch(self, path: str) -> str:
    """Starts watching ``path``; returns the resolved path."""
    path = os.path.realpath(path)
    if path not in self._stats:
      self._stats[path] = _stat(path)
    return path

  def poll(self) -> set[str]:
    """Returns the resolved paths that changed since the previous poll."""
    now = time.monotonic()
    if now < self._next_poll:
      return set()
    self._next_poll = now + self._poll_interval
    changed = set()
    for path, old in self._stats.items():
      new = _stat(path)
      if new != old:
        self._stats[path] = new
        changed.add(path)
    return changed


def source_path(module: types.ModuleType) -> str | None:
  """Returns the resolved ``.py`` source of a module, or None if it has none."""
  path = getattr(module, '__file__', None)
  if not isinstance(path, str) or not path.endswith('.py'):
    return None
  path = os.path.realpath(path)
  return path if os.path.isfile(path) else None


def module_dir(name: str) -> str | None:
  """Returns the directory of a loaded module's source, if it has one."""
  module = sys.modules.get(name)
  path = source_path(module) if module is not None else None
  return os.path.dirname(path) if path is not None else None


def local_imports(module: types.ModuleType) -> list[types.ModuleType]:
  """Returns the ``.py`` modules in the same directory that ``module`` uses."""
  # Found through the module's globals: `import m` binds a module, and
  # `from m import f` binds an object whose `__module__` is `m`.
  path = source_path(module)
  if path is None:
    return []
  directory = os.path.dirname(path)
  found: dict[str, types.ModuleType] = {}
  for value in list(vars(module).values()):
    if isinstance(value, types.ModuleType):
      dep = value
    else:
      try:
        name = getattr(value, '__module__', None)
      except Exception:  # pylint: disable=broad-exception-caught
        continue
      dep = sys.modules.get(name) if isinstance(name, str) else None
    if dep is None or dep is module or dep.__name__ in found:
      continue
    dep_path = source_path(dep)
    if dep_path is not None and os.path.dirname(dep_path) == directory:
      found[dep.__name__] = dep
  return list(found.values())


def reload_module(module: types.ModuleType) -> None:
  """Re-executes a module's current source in place."""
  # Bytecode caches are validated by source mtime in whole seconds and size,
  # so two quick edits of equal length would reuse stale bytecode.
  cached = getattr(module, '__cached__', None)
  if cached:
    with contextlib.suppress(OSError):
      os.remove(cached)
  snapshot = dict(vars(module))
  try:
    # importlib.reload (rather than exec) lets absl recognize a reload and
    # accept the module's flag definitions again.
    importlib.reload(module)
  except BaseException:
    vars(module).clear()
    vars(module).update(snapshot)
    raise


def reload_order(
    deps: Mapping[str, set[str]], roots: Iterable[str]
) -> list[str]:
  """Returns ``roots`` and their importers, each after the modules it imports.

  Args:
    deps: Mapping from each watched module name to the watched module names it
      imports.
    roots: Module names whose source changed (or previously failed to reload).
  """
  importers = collections.defaultdict(set)
  for name, imported in deps.items():
    for dep in imported:
      importers[dep].add(name)

  affected = set()
  stack = [name for name in roots if name in deps]
  while stack:
    name = stack.pop()
    if name not in affected:
      affected.add(name)
      stack.extend(importers[name])

  order = []
  visited = set()

  def visit(name: str) -> None:
    if name in visited:
      return
    visited.add(name)
    for dep in sorted(deps[name] & affected):
      visit(dep)
    order.append(name)

  for name in sorted(affected):
    visit(name)
  return order


def current_class(cls: type[Any]) -> type[Any] | None:
  """Returns the class now bound to ``cls``'s module and qualified name."""
  obj: Any = sys.modules.get(cls.__module__)
  for part in cls.__qualname__.split('.'):
    if obj is None or part == '<locals>':
      return None
    obj = getattr(obj, part, None)
  return obj if isinstance(obj, type) else None


def class_name(cls: type[Any]) -> str:
  """Returns the fully qualified ``module.QualName`` of ``cls``."""
  return f'{cls.__module__}.{cls.__qualname__}'


def same_default(a: Any, b: Any) -> bool:
  """Returns whether two class-attribute default values are equal."""
  if a is b:
    return True
  ta, tb = type(a), type(b)
  if ta is not tb:
    if (ta.__module__, ta.__qualname__) != (tb.__module__, tb.__qualname__):
      return False
    if isinstance(a, enum.Enum) and isinstance(b, enum.Enum):
      return a.name == b.name and same_default(a.value, b.value)
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
      return same_default(vars(a), vars(b))
    if hasattr(a, '__dict__') and hasattr(b, '__dict__'):
      return same_default(vars(a), vars(b))
    return repr(a) == repr(b)
  if isinstance(a, (tuple, list)):
    return len(a) == len(b) and all(same_default(x, y) for x, y in zip(a, b))
  if isinstance(a, dict):
    return a.keys() == b.keys() and all(same_default(a[k], b[k]) for k in a)
  try:
    eq = a == b
    return bool(eq.all()) if hasattr(eq, 'all') else bool(eq)
  except Exception:  # pylint: disable=broad-exception-caught
    return True
