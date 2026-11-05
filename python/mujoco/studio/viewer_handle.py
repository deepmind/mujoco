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
"""Viewer handle and event handlers for the simulation side."""

import contextlib
import copy
import dataclasses
import logging
import os
from typing import Any, Callable
import mujoco
from mujoco.studio import endpoints
from mujoco.studio import hot_reload
from mujoco.studio import messages
from mujoco.studio import parser
from mujoco.studio import plugin_registry
import numpy as np

logger = logging.getLogger(__name__)

# Launcher-owned liveness check: returns True while the viewer is still alive.
# This enables the handle to notice a viewer died without sending ExitEvent.
IsAliveFn = Callable[[], bool]

# Launcher-owned shutdown function: waits up to the given timeout (seconds) for
# the viewer to finish after close() has sent the ExitEvent. Waiting lets the
# viewer release its resources before the interpreter tears itself down.
ShutdownFn = Callable[[float], None]

# Launcher-owned error forwarder: raises any unhandled exception that
# terminated the viewer thread.
ForwardErrorFn = Callable[[], None]


@dataclasses.dataclass(frozen=True)
class SimInitEvent(messages.Event):
  """Lifecycle event dispatched locally once when ``ViewerHandle`` initializes.

  ``plugin.SimPlugin`` caches ``event.handle`` as ``self.handle`` above
  ``CRITICAL`` priority, so ``self.handle`` is already set when subclass
  handlers run (raising aborts the launch and shuts down the viewer).
  """

  handle: 'ViewerHandle'


class ViewerHandle:
  """A handle for interacting with a running viewer application from the sim."""

  def __init__(
      self,
      sim_endpoint: endpoints.SimEndpoint,
      *,
      sim_plugins: list[Any] | None = None,
      is_alive_fn: IsAliveFn | None = None,
      shutdown_fn: ShutdownFn | None = None,
      forward_error_fn: ForwardErrorFn | None = None,
  ) -> None:
    """Initializes the ViewerHandle.

    Args:
      sim_endpoint: The endpoint to use for communication with the viewer.
      sim_plugins: Optional list of plugin instances for sim-side processing,
        are classes with methods decorated with ``@handler``.
      is_alive_fn: Optional liveness check; without one the viewer is assumed to
        be running until ``close()`` is called.
      shutdown_fn: Optional launcher-owned shutdown hook, called by ``close()``.
      forward_error_fn: Optional hook that raises any unhandled exception from
        the viewer thread.
    """

    self._sim_endpoint = sim_endpoint
    self._is_running = True
    self._is_alive_fn = is_alive_fn
    self._shutdown_fn = shutdown_fn
    self._forward_error_fn = forward_error_fn
    self.model: mujoco.MjModel | None = None
    self.data: mujoco.MjData | None = None
    self.model_path: str = ''

    # With hot reload on: the files ``path`` is built from. sync() reloads the
    # model when one of them changes.
    self._model_files: hot_reload.FileWatcher | None = None

    # Instantiate handlers from user plugins + framework defaults.
    all_sim_plugins: list[Any] = list(sim_plugins or [])
    all_sim_plugins.append(self)
    self._sim_plugins = plugin_registry.PluginRegistry(all_sim_plugins)

    # Hand the plugins a reference to this handle. Dispatched last, so that a
    # plugin's handler sees a fully constructed handle.
    try:
      self._sim_plugins.dispatch(SimInitEvent(handle=self))
    except BaseException as ex:  # pylint: disable=broad-exception-caught
      # Launchers start the viewer before they construct the handle, so one is
      # already running. Raising out of __init__ means the caller never gets a
      # handle and never enters the ``with`` block, so nothing else will ever
      # call close(): shut the viewer down here or it is left running with no
      # way to stop it.
      self._is_running = False
      self.close(reason=messages.describe_exception(ex))
      raise

  def close(self, reason: str = '') -> None:
    """Signals the viewer to exit and waits for it to shut down.

    Re-raises any unhandled exception that terminated the viewer thread, so a
    viewer-side failure surfaces on the sim side rather than being lost.

    Args:
      reason: Why the viewer is being closed, when this is not a normal exit
        (see ExitEvent.reason). The web viewer shows it in the browser.
    """
    exit_event = messages.ExitEvent(reason=reason)
    try:
      if self._is_running:
        self._is_running = False
        self.dispatch(exit_event)
    finally:  # Finish the teardown even if an ExitEvent handler raised.
      try:
        self.send_to_viewer(exit_event)
      except Exception:  # pylint: disable=broad-exception-caught
        pass  # Ignore exceptions, the viewer may have already closed.
      if self._shutdown_fn is not None:
        self._shutdown_fn(5.0)
        self._shutdown_fn = None
      self._sim_endpoint.close()
      if self._forward_error_fn is not None:
        self._forward_error_fn()

  def __enter__(self) -> 'ViewerHandle':
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: Any,
  ) -> None:
    # An exception leaving the ``with`` block ends the session: tell the viewer
    # which one, so the browser can show it when the page goes dead.
    self.close(reason=messages.describe_exception(exc_val))

  def is_running(self) -> bool:
    """Returns True while the viewer is open.

    Closes the handle once the viewer has gone away, so this is also where an
    unhandled viewer-thread exception is re-raised (see ``close()``).
    """
    if not self._is_running or (
        self._is_alive_fn is not None and not self._is_alive_fn()
    ):
      self.close()
    return self._is_running

  def send_to_viewer(self, message: messages.Message) -> None:
    """Sends an event or snapshot message to the viewer process.

    Args:
      message: The message to send.
    """
    self._sim_endpoint.send_to_viewer(message)

  def send_if_changes(
      self, *make_messages: messages.MakeMessage
  ) -> contextlib.AbstractContextManager[None]:
    """Sends the messages that change during the ``with`` block to the viewer.

    Calls each ``make_message()`` on entering and on leaving the block and
    sends those that differ. Nothing is sent if the handle is closed or its
    model is replaced during the block.
    """
    model = self.model
    return messages.send_if_changes(
        self.send_to_viewer,
        *make_messages,
        should_send=lambda: self._is_running and self.model is model,
    )

  def dispatch(self, message: messages.Message) -> None:
    """Dispatches a message to sim-side handlers in priority order."""
    self._sim_plugins.dispatch(message)

  def set_model(
      self,
      model: mujoco.MjModel,
      path: str = '',
      data: mujoco.MjData | None = None,
  ) -> None:
    """Publishes ``model`` to sim plugins and the viewer as the current model.

    Args:
      model: The new model.
      path: The file that produced ``model``, if any; see ``ModelEvent.path``.
      data: The data for ``model``; a fresh ``MjData`` is created when None.
    """
    path = os.path.abspath(path) if path else ''
    self.model = model
    self.data = mujoco.MjData(model) if data is None else data
    self.model_path = path
    self._watch_model(path)
    mujoco.mj_forward(self.model, self.data)
    event = messages.ModelEvent(model=model, path=path)
    self.dispatch(event)
    self.send_to_viewer(event)

  def reload(self) -> None:
    """Reloads the current model, as the Reload button does."""
    self.dispatch(messages.LoadModelEvent())

  def _watch_model(self, path: str) -> None:
    """With model hot reload on, watches the files ``path`` is built from."""
    self._model_files = None
    if not path or not hot_reload.watches_model():
      return
    self._model_files = hot_reload.watch_files(parser.dependencies(path))

  def sync(
      self,
      model: mujoco.MjModel | None = None,
      data: mujoco.MjData | None = None,
  ) -> tuple[mujoco.MjModel | None, mujoco.MjData | None]:
    """Syncs the simulation with the viewer and returns the updated sim state.

    This method processes incoming messages from the viewer, dispatches
    ``PreStepEvent``, ``StepEvent``, and ``PostStepEvent`` so sim-side plugins
    can prepare, advance, and observe the simulation, and sends the resulting
    simulation state to the viewer as a snapshot.

    ``sync`` itself neither advances physics nor sleeps. When using
    ``run_sim_loop``, include a ``StepEvent`` plugin in ``sim_plugins`` (e.g.
    ``step_control.StepControl()`` for real-time-paced CPU stepping, or your
    own plugin to step differently) that also paces each iteration to avoid
    busy-spinning. Alternatively, a user script can write its own loop around
    ``sync`` and step and pace the simulation directly (see
    ``sample/custom_sim_loop.py``).

    Sim-side edits belong in handlers of those three events: ``sync`` checks
    ``model.opt`` for changes across ``PreStepEvent``, ``StepEvent``, and
    ``PostStepEvent`` and sends an ``MjOptionSnapshot`` to the viewer when it
    was modified.

    Args:
      model: Optional model owned by the caller's loop. If None, keeps
        simulating the model already on the handle (from ``set_model`` or
        ``LoadModelEvent``).
      data: The data for ``model``; ignored when ``model`` is None.

    Returns:
      The current model and data; rebind both, they may be new objects (e.g.
      after the viewer sends a ModelEvent).
    """
    # Read the viewer's queues before dispatching anything: a handler may close
    # the handle, and with it the endpoint (see close()).
    events = self._sim_endpoint.get_viewer_events()
    snapshots = self._sim_endpoint.get_viewer_snapshots()

    if model is not None:
      model_changed = model is not self.model
      self.model, self.data = model, data
      if model_changed:
        # Model passed directly to sync() has no file path to reload from.
        assert self.model is not None and self.data is not None
        self.model_path = ''
        self._model_files = None
        self._sim_endpoint.send_to_viewer(messages.ModelEvent(model=self.model))
        self._sim_plugins.dispatch(
            messages.PostModelEvent(model=self.model, data=self.data)
        )

    # Apply edited plugin code between steps (no-op unless enabled). After the
    # caller's model is applied, so a model rebuilt here is returned to them.
    hot_reload.tick(self._sim_plugins, self)
    if not hot_reload.watches_model():
      self._model_files = None
    elif self._model_files is None and self.model_path:
      self._watch_model(self.model_path)
    if self._model_files is not None and self._model_files.poll():
      self.reload()

    # Process incoming events from the viewer.
    for event in events:
      self.dispatch(event)

    # Process incoming snapshots from the viewer.
    for snapshot in snapshots:
      self.dispatch(snapshot)

    # Nothing left to step for once a handler has closed the handle.
    if self.model is not None and self._is_running:
      assert self.data is not None
      # Prepare, advance, and observe the simulation: dispatched locally to
      # sim-side plugins. Each phase re-reads self.model and self.data so a
      # set_model() call from a PreStepEvent handler (e.g. a hot-reload that
      # finished compiling in the background) is stepped and snapshotted.
      # Sim-side edits to model.opt happen in these handlers, so compare it
      # around the three dispatches and send it to the viewer if it changed.
      with self.send_if_changes(
          lambda: messages.MjOptionSnapshot(opt=copy.deepcopy(self.model.opt))
      ):
        for event_cls in (
            messages.PreStepEvent,
            messages.StepEvent,
            messages.PostStepEvent,
        ):
          self._sim_plugins.dispatch(
              event_cls(model=self.model, data=self.data)
          )
          if not self._is_running:  # A step handler closed the handle.
            return self.model, self.data

      # Send the simulation state to the viewer process as a snapshot.
      integration_sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
      integration_size = mujoco.mj_stateSize(self.model, integration_sig)
      integration_state = np.empty(integration_size, np.float64)
      mujoco.mj_getState(
          self.model,
          self.data,
          integration_state,
          integration_sig,
      )
      self._sim_endpoint.send_to_viewer(
          messages.StateSnapshot(
              state=integration_state, state_sig=integration_sig
          ),
      )

    return self.model, self.data

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_model(self, event: messages.ModelEvent) -> None:
    if event.model is not self.model:
      # Only needed when ModelEvent is dispatched without calling set_model().
      self.model = event.model
      self.data = mujoco.MjData(event.model)
      self.model_path = event.path
      mujoco.mj_forward(self.model, self.data)
    assert self.model is not None and self.data is not None
    self.dispatch(
        messages.PostModelEvent(
            model=self.model, data=self.data, path=event.path
        )
    )

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_load_model(self, event: messages.LoadModelEvent) -> None:
    # Normalize so relative and absolute paths to a file match.
    path = os.path.abspath(event.path) if event.path else self.model_path
    if not path:
      return
    if self.model is None:
      # Record path before loading so Reload can retry a broken initial file.
      self.model_path = path
    try:
      model = parser.load(path)
    except Exception:  # pylint: disable=broad-exception-caught
      if path == self.model_path and self._model_files is None:
        self._watch_model(path)  # So that saving a fix loads it.
      # A broken file must not take down the running viewer.
      logger.exception('Loading %r failed; keeping the previous model.', path)
      return
    self.set_model(model, path=path)

  def _set_state(
      self,
      state: np.ndarray,
      state_sig: int,
      *,
      forward: bool = False,
  ) -> None:
    """Applies a state array to sim data and optionally runs mj_forward."""
    if self.model is not None and self.data is not None:
      state_size = mujoco.mj_stateSize(self.model, state_sig)
      if len(state) == state_size:
        mujoco.mj_setState(self.model, self.data, state, state_sig)
        if forward:
          mujoco.mj_forward(self.model, self.data)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_state(self, event: messages.StateEvent) -> bool:
    self._set_state(event.state, event.state_sig, forward=True)
    return True

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_perturb(self, snapshot: messages.PerturbSnapshot) -> bool:
    self._set_state(snapshot.state, snapshot.state_sig, forward=False)
    return True

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_reset(self, event: messages.ResetEvent) -> None:
    """Resets sim data without consuming the event so plugins observe it."""
    if self.model is not None:
      assert self.data is not None
      if event.key is not None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, event.key)
      else:
        mujoco.mj_resetData(self.model, self.data)
      mujoco.mj_forward(self.model, self.data)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_exit(self, _: messages.ExitEvent) -> None:
    self._is_running = False  # pylint: disable=protected-access

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_mjoption(self, event: messages.MjOptionSnapshot) -> bool:
    if self.model is not None:
      self.model.opt = event.opt
    return True

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_keyframe(self, event: messages.KeyframeEvent) -> bool:
    if self.model is None or not 0 <= event.key < self.model.nkey:
      return True
    # Drop events built for a different model, e.g. sent before a reload.
    sig = messages.KEYFRAME_STATE_SIG
    if len(event.state) != mujoco.mj_stateSize(self.model, sig):
      return True

    data = mujoco.MjData(self.model)
    mujoco.mj_setState(self.model, data, event.state, sig)
    mujoco.mj_setKeyframe(self.model, data, event.key)
    return True

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_threadpool(self, event: messages.ThreadPoolEvent) -> bool:
    if self.data is not None:
      mujoco.mju_threadpool(self.data, event.nthread)
    return True


def run_sim_loop(handle: ViewerHandle) -> None:
  """Runs the standard simulation sync loop until the viewer closes.

  Publish a model before calling this (``handle.set_model`` or a
  ``LoadModelEvent``), or include a sim plugin that does; until one arrives the
  viewer shows an empty scene.

  With this loop, all sim-side behavior—stepping, pacing, and edits to
  ``model``, ``data``, or plugin state—belongs in ``sim_plugins`` handling
  ``PreStepEvent``, ``StepEvent``, and ``PostStepEvent``, which ``sync``
  dispatches inside its ``model.opt`` change detection. For an example of a
  script that owns its loop, stepping and pacing itself, see
  ``sample/custom_sim_loop.py``.
  """
  try:
    while handle.is_running():
      handle.sync()
  except KeyboardInterrupt:
    # Ctrl+C is the documented way to quit; exit cleanly, no traceback.
    print('\nShutting down.', flush=True)
