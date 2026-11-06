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
"""Messages and Channels for Studio."""

from collections.abc import Iterator
import contextlib
import dataclasses
import enum
import inspect
from typing import Any, Callable, Protocol, get_type_hints, runtime_checkable

import mujoco
from mujoco.studio import sim
import numpy as np

# The mjtState signature of KeyframeEvent.state, matching MjModel keyframes.
KEYFRAME_STATE_SIG = (
    mujoco.mjtState.mjSTATE_TIME.value
    | mujoco.mjtState.mjSTATE_QPOS.value
    | mujoco.mjtState.mjSTATE_QVEL.value
    | mujoco.mjtState.mjSTATE_ACT.value
    | mujoco.mjtState.mjSTATE_CTRL.value
    | mujoco.mjtState.mjSTATE_MOCAP_POS.value
    | mujoco.mjtState.mjSTATE_MOCAP_QUAT.value
)

# -----------------------------------------------------------------------------
# Message and Channel types for passing data between the sim and the viewer.
# Concrete messages are defined later in this file, there is one per Event;
# add more if needed in your script.
# Concrete Channel implementations live in the launch_* modules.
# -----------------------------------------------------------------------------


class Message:
  """Base class for data sent between the sim and the viewer.

  Derive from Snapshot or Event, depending on the desired delivery semantics.
  """


class Snapshot(Message):
  """Base class for latest-wins, droppable messages.

  Each new snapshot replaces the previous one, so the receiver always sees the
  most recent value. Intermediate values may be silently dropped, making
  snapshots suitable for frequently updated state where only the latest value
  matters.
  """


class Event(Message):
  """Base class for reliable, ordered, never dropped messages.

  Events are queued in order and every event is delivered exactly once. Use
  events for discrete actions that must not be lost.
  """


@runtime_checkable
class SnapshotChannel(Protocol):
  """Channel to transport Snapshot messages with latest-wins semantics."""

  def put(self, value: Snapshot) -> None:
    """Overwrites the latest snapshot of the same type.

    After calling put() it is the caller's responsibility to ensure it no longer
    holds a reference to the snapshot.

    Args:
      value: The snapshot to put into the channel.
    """
    ...

  def get(self) -> list[Snapshot]:
    """Returns the latest pending snapshots of each type."""
    ...

  def reclaim_unconsumed(
      self, snapshot_type: type[Snapshot]
  ) -> Snapshot | None:
    """Pops and returns an unconsumed pending snapshot of ``snapshot_type``."""
    del snapshot_type
    return None

  def close(self) -> None:
    """Releases resources held by the channel."""
    ...


@runtime_checkable
class EventChannel(Protocol):
  """Channel to transport Event messages in an ordered stream, nothing dropped."""

  def put(self, value: Event) -> None:
    """Puts an event into the channel, appending to the stream."""
    ...

  def get(self) -> list[Event]:
    """Returns all pending events, in order."""
    ...

  def close(self) -> None:
    """Releases resources held by the channel."""
    ...


# Builds a message for send_if_changes, or None to send nothing.
MakeMessage = Callable[[], Message | None]


@contextlib.contextmanager
def send_if_changes(
    send_fn: Callable[[Message], None],
    *make_messages: MakeMessage,
    should_send: Callable[[], bool] | None = None,
) -> Iterator[None]:
  """Sends the messages that change during the ``with`` block.

  Calls each ``make_message()`` on entering and on leaving the block and sends
  those that differ. Pass lambdas so their expressions are re-read on each
  call, e.g. ``lambda: messages.ThreadPoolEvent(nthread=ux_state.nthread)``.

  Args:
    send_fn: Called with each message that changed during the block.
    *make_messages: Callables building the messages to compare; a callable may
      return None to send nothing.
    should_send: Optional predicate evaluated after the block; when it returns
      False nothing is sent (e.g. because the owner stopped running or its model
      was replaced during the block).

  Yields:
    Nothing; the block runs between the two rounds of ``make_message()``
    calls.
  """
  before = [fn() for fn in make_messages]
  yield
  if should_send is not None and not should_send():
    return
  for fn, prev in zip(make_messages, before):
    after = fn()
    if after is not None and after != prev:
      send_fn(after)


# -----------------------------------------------------------------------------
# Concrete Event and Snapshot types.
# -----------------------------------------------------------------------------


def _get_state(
    model: mujoco.MjModel, data: mujoco.MjData, state_sig: int
) -> np.ndarray:
  """Returns the selected components of ``data`` as a flat state array."""
  size = mujoco.mj_stateSize(model, state_sig)
  state = np.empty(size, np.float64)
  mujoco.mj_getState(model, data, state, state_sig)
  return state


@dataclasses.dataclass(frozen=True, eq=False)
class StateSnapshot(Snapshot):
  """A snapshot message that transports a MuJoCo state.

  Attributes:
    state: The state vector, see mj_getState.
    state_sig: The mjtState signature of `state`.
  """

  state: np.ndarray
  state_sig: int

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, StateSnapshot):
      return NotImplemented
    return self.state_sig == other.state_sig and np.array_equal(
        self.state, other.state
    )


@dataclasses.dataclass(frozen=True, eq=False)
class PerturbSnapshot(Snapshot):
  """Carries perturbation forces from the viewer to the simulation.

  Sim-side handlers apply ``mj_setState`` without running ``mj_forward``.
  """

  state: np.ndarray
  state_sig: int

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, PerturbSnapshot):
      return NotImplemented
    return self.state_sig == other.state_sig and np.array_equal(
        self.state, other.state
    )


@dataclasses.dataclass(frozen=True, eq=False)
class StateEvent(Event):
  """A reliable event that transports a MuJoCo state.

  The receiver applies the state with mj_setState using state_sig. Unlike
  StateSnapshot, events are never dropped and different state edits do not
  overwrite each other, so this is the right carrier for state edits such as GUI
  joint/control slider changes.
  """

  state: np.ndarray
  state_sig: int

  @classmethod
  def from_data(
      cls,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      state_sig: int,
  ) -> 'StateEvent | None':
    """Packs the selected components of ``data`` into a ``StateEvent``."""
    if not state_sig:
      return None
    return cls(state=_get_state(model, data, state_sig), state_sig=state_sig)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, StateEvent):
      return NotImplemented
    return self.state_sig == other.state_sig and np.array_equal(
        self.state, other.state
    )


@dataclasses.dataclass(frozen=True)
class ModelEvent(Event):
  """An event that transports a MuJoCo model.

  Attributes:
    model: The compiled MuJoCo model.
    path: The model file that produced ``model``, or empty for a model built in
      Python. Reloading the same ``path`` preserves viewer UI state such as
      camera and pause; changing ``path`` resets it.
  """

  model: mujoco.MjModel
  path: str = ''


@dataclasses.dataclass(frozen=True)
class LoadModelEvent(Event):
  """Asks the sim to load a model file (``path``), or to reload the current model.

  ``ViewerHandle`` is what loads model files: its ``Priority.INTERNAL`` handler
  parses ``path`` and publishes the result with ``set_model``. Sim plugins
  handle the event first (in priority order) and can provide a model built in
  Python instead: such a handler calls ``self.handle.set_model(spec)`` and
  returns ``True``, which consumes the event so that ``ViewerHandle``'s handler
  does not run.

  When ``LoadModelEvent`` is dispatched and how it is handled:

  * **Startup** (``launch_*.run``): dispatched with the ``--model`` file path,
    or ``path=''`` when ``--model`` is omitted. A sim plugin that builds a
    model in Python consumes the event; otherwise ``ViewerHandle`` loads
    ``path`` when it is non-empty.
  * **File drop** on the viewer: sent from the viewer with ``path`` set to the
    dropped file. A sim plugin that checks ``if event.path: return False`` lets
    ``ViewerHandle`` load the dropped file; one that ignores ``event.path``
    keeps its Python scene.
  * **Reload button** or ``ViewerHandle.reload()``: the button sends the same
    event that ``reload()`` dispatches, with ``path`` set to the current
    ``model_path`` (the shown model file, or ``''`` when the shown model was
    built in Python). So a sim plugin that yields to files also yields on
    Reload and ``ViewerHandle`` parses the file again, while a Python scene is
    built again by its plugin.
  * **Model hot reload** (``--hot_reload=all`` or ``=model``): when the shown
    model file, one of its dependencies, or a file passed to
    ``ViewerHandle.set_model(watch=...)`` is saved, ``ViewerHandle.sync`` calls
    ``reload()``, as above. Saving a sim plugin's own code dispatches
    ``HotReloadEvent`` instead; the plugin calls ``reload()`` from its handler
    to rebuild its scene (see the example below).

  A sim plugin that builds a model in Python handles ``LoadModelEvent`` to
  build and publish the model, and ``HotReloadEvent`` to rebuild it when its
  own code is saved::

    class MyScene(plugin.SimPlugin):
      height: int = 5

      @messages.handler
      def on_load_model(self, event: messages.LoadModelEvent) -> bool:
        if event.path:
          return False  # Let ViewerHandle load --model or dropped files.
        spec = mujoco.MjSpec()
        ...  # Build from self.height; edit freely while running.
        self.handle.set_model(spec)
        return True

      @messages.handler
      def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
        if self in event.plugins:
          self.handle.reload()  # What the Reload button does.

  Errors: a model file that fails to load is logged and the previous model
  stays (with no previous model, Reload retries the file); an error raised by a
  plugin handler propagates like any handler's (once the plugin has been
  hot-reloaded it is logged instead).

  Note:
    Pair the sim plugin with a ``ViewerPlugin`` whose controls send it events
    or snapshots; ``sample/python_scene.py`` is a complete scene with UI to
    tweak it.

  Note:
    Models defined in Python are passed via ``--plugins`` (or ``plugins=[...]``
    in code); passing a ``.py`` file via ``--model`` would require a convention
    for what to run in the file, which was avoided by design.

  Attributes:
    path: The model file to load, or empty when there is none (the current model
      was built in Python, or no model is shown yet); see ``ModelEvent.path``.
  """

  path: str = ''


@dataclasses.dataclass(frozen=True)
class PreloadModelEvent(Event):
  """A hint that ``model`` may become the current model soon.

  Sent from the sim to the viewer, for example by a scene playlist that knows
  which models come next. The web viewer serializes the model right away and
  tells connected browsers about it, so they can download it in the background
  and later switch to it without waiting on the network when its ``ModelEvent``
  arrives. Other viewers ignore the hint. The model should not be modified
  between the hint and its ``ModelEvent``; if it is, the viewer simply serves it
  the usual way when it loads.

  Attributes:
    model: The model that may be loaded soon.
  """

  model: mujoco.MjModel


@dataclasses.dataclass(frozen=True)
class PostModelEvent(Event):
  """Lifecycle event dispatched locally after a new model and data are loaded.

  Dispatched on the sim side (by ViewerHandle) and on the viewer side (by
  Viewer) after ModelEvent has replaced model/data and run mj_forward.
  Handlers for this event can rely on model and data (as well as
  handle.model/handle.data or viewer.model/viewer.data) already holding the
  newly initialized simulation state. Never crosses a channel.

  Attributes:
    model: The newly loaded model (sim-side MjModel or viewer's deep copy).
    data: The newly created and forwarded MjData for the model.
    path: The file that produced ``model``, or empty; see ``ModelEvent.path``.
  """

  model: mujoco.MjModel
  data: mujoco.MjData
  path: str = ''


@dataclasses.dataclass(frozen=True)
class SingleStepEvent(Event):
  """Sent from viewer to sim when the user requests a single simulation step."""


@dataclasses.dataclass(frozen=True)
class RequestPauseEvent(Event):
  """Requests a pause state change on the sim or the viewer.

  Dispatched on the sim to ask the active physics plugin to pause or unpause,
  or sent from the sim to the viewer to update the viewer's pause state
  without constructing a full ``StepControlSnapshot``.

  Attributes:
    pause_state: The desired pause state.
  """

  pause_state: sim.PauseState


@dataclasses.dataclass(frozen=True)
class ResetEvent(Event):
  """An event requesting to reset the simulation.

  Attributes:
    key: Optional keyframe index to load. If None or negative, resets to the
      default initial state.
  """

  key: int | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class KeyframeEvent(Event):
  """Carries a keyframe saved in the viewer to the simulation model.

  Attributes:
    key: Index of the keyframe to overwrite.
    state: The keyframe state, which must have signature KEYFRAME_STATE_SIG.
  """

  key: int
  state: np.ndarray

  @classmethod
  def from_model(
      cls, model: mujoco.MjModel, key: int
  ) -> 'KeyframeEvent | None':
    """Packs keyframe ``key`` of ``model`` into a ``KeyframeEvent``."""
    if not 0 <= key < model.nkey:
      return None
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    return cls(
        key=key,
        state=_get_state(model, data, KEYFRAME_STATE_SIG),
    )

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, KeyframeEvent):
      return NotImplemented
    return self.key == other.key and np.array_equal(self.state, other.state)


@dataclasses.dataclass(frozen=True)
class ThreadPoolEvent(Event):
  """An event requesting to resize the simulation's MuJoCo thread pool."""

  nthread: int


@dataclasses.dataclass(frozen=True)
class PreStepEvent(Event):
  """Lifecycle event dispatched by ViewerHandle.sync before StepEvent.

  Dispatched locally to sim-side plugins after incoming viewer messages are
  applied and before ``StepEvent`` advances the simulation. Handlers use this
  phase to prepare for the step (e.g. writing actuator controls, applying
  external forces, or loading a new model via ``handle.set_model``). Never
  crosses a channel.

  Attributes:
    model: The current sim-side model.
    data: The current sim-side data.
  """

  model: mujoco.MjModel
  data: mujoco.MjData


@dataclasses.dataclass(frozen=True)
class StepEvent(Event):
  """Lifecycle event dispatched by ViewerHandle.sync after PreStepEvent.

  Handlers for this event advance the simulation, e.g. the default StepControl
  plugin steps CPU physics with real-time pacing; a custom plugin can step
  differently (e.g. on the GPU). Dispatched locally to sim-side plugins; never
  crosses a channel.

  Attributes:
    model: The sim-side model to step.
    data: The sim-side data to step.
  """

  model: mujoco.MjModel
  data: mujoco.MjData


@dataclasses.dataclass(frozen=True)
class PostStepEvent(Event):
  """Lifecycle event dispatched by ViewerHandle.sync after StepEvent.

  Handlers for this event observe or record the simulation state after stepping
  plugins have advanced physics and before StateSnapshot is broadcast to the
  viewer. Dispatched locally to sim-side plugins; never crosses a channel.

  Attributes:
    model: The current sim-side model.
    data: The current sim-side data.
  """

  model: mujoco.MjModel
  data: mujoco.MjData


@dataclasses.dataclass(frozen=True)
class UpdateEvent(Event):
  """Lifecycle event dispatched on every frame on the viewer side before building GUI."""


@dataclasses.dataclass(frozen=True)
class BuildGuiEvent(Event):
  """Lifecycle event dispatched on every frame on the viewer side to build ImGui elements."""


@dataclasses.dataclass(frozen=True)
class MjOptionSnapshot(Snapshot):
  """Bidirectional snapshot carrying mjOption state between sim and viewer."""

  opt: mujoco.MjOption


@dataclasses.dataclass(frozen=True)
class StepControlSnapshot(Snapshot):
  """Bidirectional snapshot carrying step control state between sim and viewer.

  Attributes:
    pause_state: Whether the simulation is paused.
    speed: The desired speed as a percentage of real time.
    noise_scale: Control noise standard deviation.
    noise_rate: Control noise correlation rate.
    speed_measured: The speed the sim achieves, as a percentage of real time;
      negative when not known or when sent from the viewer.
  """

  pause_state: sim.PauseState = sim.PauseState.UNPAUSED
  speed: float = 100.0
  noise_scale: float = 0.0
  noise_rate: float = 0.0
  speed_measured: float = -1.0

  @classmethod
  def from_step_control(
      cls, step_control: sim.StepControl, speed_measured: float = -1.0
  ) -> 'StepControlSnapshot':
    """Packs the settings of ``step_control`` into a snapshot.

    Args:
      step_control: The step control whose settings to pack.
      speed_measured: The achieved speed to report, if known.
    """
    noise_scale, noise_rate = step_control.get_noise_parameters()
    return cls(
        pause_state=step_control.get_pause_state(),
        speed=step_control.get_speed(),
        noise_scale=noise_scale,
        noise_rate=noise_rate,
        speed_measured=speed_measured,
    )


@dataclasses.dataclass(frozen=True)
class ExitEvent(Event):
  """An event requesting to exit.

  Attributes:
    reason: A one-line summary of the exception that ended the sending side
      (see describe_exception), or empty for a normal exit.
  """

  reason: str = ''


def describe_exception(ex: BaseException | None) -> str:
  """Formats an error exception as a single line.

  For example, `"ValueError: Invalid actuator id 'wrist_roll'"`.
  Returns `''` for `None` and for non-error exits such as `KeyboardInterrupt`
  and `SystemExit`.
  """
  if not isinstance(ex, Exception):
    return ''
  text = f'{type(ex).__name__}: {ex}' if str(ex) else type(ex).__name__
  return ' '.join(text.split())


@dataclasses.dataclass(frozen=True)
class HotReloadEvent(Event):
  """Dispatched after plugins are hot-reloaded (see ``hot_reload.py``).

  The plugin instances are kept and only their class is replaced, so their
  state survives. Handling this event is optional, but it gives plugins a hook
  to reconcile state after a live code swap — for example, initializing newly
  added mutable attributes that ``__init__`` would create, recomputing cached
  IDs or lookup tables, migrating or clearing buffers whose shape changed, or
  triggering an immediate rebuild:

      @messages.handler
      def _on_hot_reload(self, event: messages.HotReloadEvent) -> None:
        if self in event.plugins:
          ...

  Attributes:
    plugins: The plugin instances whose class was replaced.
  """

  plugins: tuple[Any, ...]


# ---------------------------------------------------------------------------
# Message handling decorator.
# ---------------------------------------------------------------------------

_HANDLER_INFO_ATTR = '_studio_handler_info'


@dataclasses.dataclass(frozen=True)
class _HandlerInfo:
  """Metadata stamped on a decorated method."""

  priority: int
  message_type: type[Message]


class Priority(enum.IntEnum):
  """Execution priority levels for message handlers.

  Handlers with higher values execute first. When multiple handlers share the
  same priority, their relative order is undefined.
  """

  INTERNAL = 1  # For built-in handlers.
  LIBRARY = 10  # For library extensions.
  USER = 100  # For user extensions. Default when no priority is specified.
  CRITICAL = 1000  # For handlers that must run before everything else.


def handler(
    fn: Callable[..., Any] | None = None, *, priority: int = Priority.USER
) -> Callable[..., Any]:
  """Decorator to mark a class method as a message handler.

  The class method must accept exactly two arguments: self and a message event.
  e.g., @handler or @handler(priority=...).

  Args:
    fn: The method to stamp with handler metadata.
    priority: The priority of the handler.

  Returns:
    The stamped method.
  """

  def stamp(target: Callable[..., Any]) -> Callable[..., Any]:
    fn_name = getattr(target, '__name__', str(target))
    params = list(inspect.signature(target).parameters.values())
    if len(params) != 2:
      raise TypeError(
          f'{fn_name} must accept exactly two arguments, got {len(params)}'
      )

    message_type = params[1].annotation

    # When `from __future__ import annotations` is enabled or string forward
    # references are used, annotations are stored as strings rather than code
    # objects. In this case, resolve the string into its actual class type.
    if isinstance(message_type, str):
      try:
        hints = get_type_hints(target)
        param_name = params[1].name
        if param_name in hints:
          message_type = hints[param_name]
      except Exception:  # pylint: disable=broad-except
        pass

    # Verify we got a type annotation.
    if message_type is inspect.Parameter.empty:
      raise TypeError(
          f'{fn_name}: second parameter must have a type annotation'
      )

    # Verify that the annotation resolved to an actual Python class and that the
    # class is a valid subclass of Message. This also catches cases where a
    # string forward reference failed to resolve via get_type_hints above.
    if not (
        isinstance(message_type, type) and issubclass(message_type, Message)
    ):
      raise TypeError(
          f'{fn_name}: second parameter type annotation {message_type} is not'
          ' a Message subclass'
      )

    info = _HandlerInfo(priority=priority, message_type=message_type)
    setattr(target, _HANDLER_INFO_ATTR, info)
    return target

  # Used without parens: e.g., @handler
  if fn is not None:
    return stamp(fn)

  # Used with parens: e.g., @handler() or @handler(priority=...)
  return stamp
