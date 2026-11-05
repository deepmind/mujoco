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
"""Discovers plugin ``@handler`` methods and dispatches messages to them."""

import collections
from typing import Any, Callable, NamedTuple
from mujoco.studio import messages

# A bound ``@handler`` method: takes the message, returns True to consume it.
Handler = Callable[..., Any]
# Maps a plugin and one of its handlers to the callable actually registered.
HandlerWrapper = Callable[[Any, Handler], Handler]


class PrioritizedHandler(NamedTuple):
  """A registered handler and its priority; higher priorities run first."""

  priority: int
  handler: Handler


def _discover_handlers(
    plugin: Any,
) -> list[tuple[messages._HandlerInfo, Handler]]:
  """Scan an object for methods marked with handler decorators."""
  handlers = []
  for name in dir(type(plugin)):
    unbound = getattr(type(plugin), name, None)
    if unbound is None:
      continue
    info = getattr(unbound, messages._HANDLER_INFO_ATTR, None)  # pylint: disable=protected-access
    if info is None:
      continue
    bound = getattr(plugin, name)
    handlers.append((info, bound))
  return handlers


def _type_key(cls: type[Any]) -> Any:
  """Returns a reload-stable lookup key for a message class."""
  if '<locals>' in cls.__qualname__:
    return cls
  return (cls.__module__, cls.__qualname__)


class PluginRegistry:
  """Runtime registry of discovered handlers from plugin instances.

  The set of plugins is fixed at construction: there is no late registration.
  ``hot_reload`` relies on this — it scans ``plugins`` once per registry, and a
  reload swaps the class of an existing instance and calls ``rebuild_handlers``,
  but never adds an instance. Plugins therefore need to be registered up front;
  see the hot-reload design goal in ``hot_reload.py``.
  """

  def __init__(self, plugins: list[Any] | None = None) -> None:
    self.plugins: list[Any] = list(plugins or [])
    self.handlers: dict[Any, list[PrioritizedHandler]] = (
        collections.defaultdict(list)
    )
    self.rebuild_handlers()

  def rebuild_handlers(self, wrap: HandlerWrapper | None = None) -> None:
    """Discovers and registers handlers for all plugins in ``self.plugins``.

    Args:
      wrap: Optional ``wrap(plugin, handler)`` returning the callable to
        register in place of ``handler`` (e.g. to log its exceptions).
    """
    handlers = collections.defaultdict(list)
    for obj in self.plugins:
      for info, bound_method in _discover_handlers(obj):
        fn = wrap(obj, bound_method) if wrap is not None else bound_method
        handlers[_type_key(info.message_type)].append(
            PrioritizedHandler(info.priority, fn)
        )
    self.handlers = handlers

  def dispatch(self, message: Any) -> None:
    """Dispatches a message to registered handlers in priority order."""
    # Collect all matching handlers for the given message class and all its
    # ancestors/superclasses.
    message_handlers: list[PrioritizedHandler] = []
    for message_type in type(message).__mro__:
      message_handlers.extend(self.handlers.get(_type_key(message_type), []))

    # Sort by priority; higher values execute first.
    message_handlers.sort(key=lambda item: item.priority, reverse=True)
    for _, handler in message_handlers:
      if handler(message):
        # Handler returned True to consume the message; stop dispatching.
        return
