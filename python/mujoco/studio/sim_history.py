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
"""Simulation history recording and scrubbing plugin.

``SimHistory`` is a sim-side plugin that records a frame after every physics
step, handles scrub requests, and broadcasts buffer metadata via
``SimHistorySnapshot``.

Viewer plugins (e.g. ``studio_app.App``) add scrubbing UI by handling
``SimHistorySnapshot`` and sending one back with the desired index. There is
no need to couple such a viewer plugin to ``SimHistory`` explicitly: the
message is the only contract between them.
"""

import dataclasses
import math

import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import sim
from mujoco.studio import ux


@dataclasses.dataclass(frozen=True)
class SimHistorySnapshot(messages.Snapshot):
  """Bidirectional snapshot carrying history-buffer state.

  The simulation owns an authoritative history buffer (``sim.SimHistory``) that
  records a frame after every physics step.  The viewer keeps a local mirror
  of the buffer's size, index, and head timestamp in ``ux_state`` so its
  scrubber widget can render a timeline, but never stores authoritative frame
  data — the displayed state always comes from the sim via ``StateSnapshot``.

  This single message type is used in both directions on separate directional
  channels, so there is no slot collision in the latest-wins SnapshotChannel:

  Viewer-to-sim (scrub request):
    The viewer sets only ``index`` (the desired history offset) and leaves
    ``size`` and ``head_time`` at their defaults.  The sim loads the requested
    frame into its own model/data, requests a pause, and the result reaches the
    viewer as the next StateSnapshot.

  Sim-to-viewer (buffer description):
    Sent whenever a frame is recorded, history is reset, or a scrub request is
    applied.  ``size``, ``index``, and ``head_time`` are populated so the viewer
    can render the scrubber timeline against the sim's authoritative index
    space.

  Attributes:
    size: Total number of frames stored in the sim's history.  Left at 0 in the
      viewer-to-sim direction (the sim ignores it).
    index: History offset in the sim's index space; 0 is the most recent state,
      negative values go further into the past.
    head_time: Simulation timestamp at the head of the history buffer (index 0).
      Left at 0.0 in the viewer-to-sim direction (the sim ignores it).
  """

  size: int = 0
  index: int = 0
  head_time: float = 0.0

  @classmethod
  def from_ux_state(cls, ux_state: ux.UxState) -> 'SimHistorySnapshot | None':
    """Packs the current scrub index of ``ux_state`` into a snapshot."""
    if ux_state.history_size <= 0:
      return None
    return cls(index=ux_state.history_index)


class SimHistory(plugin.SimPlugin):
  """Sim-side plugin for simulation history recording and scrubbing.

  Owns the authoritative history buffer, records a frame after every physics
  step, handles scrub requests from the viewer, and broadcasts the buffer
  state as a ``SimHistorySnapshot`` whenever the buffer changes or is scrubbed.
  """

  def __init__(self) -> None:
    self.sim_history = sim.SimHistory()
    self._ux_state = ux.UxState()
    self._model: mujoco.MjModel | None = None
    self._last_recorded_time = math.inf

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_sim_init(self, _: plugin.SimInitEvent) -> None:
    self.sim_history = sim.SimHistory()
    self._ux_state = ux.UxState()
    self._model = None
    self._last_recorded_time = math.inf

  def _send_snapshot(self) -> None:
    self.handle.send_to_viewer(
        SimHistorySnapshot(
            size=self.sim_history.size(),
            index=self.sim_history.get_index(),
            head_time=self._ux_state.sim_head_time,
        )
    )

  # -- Event handlers ---------------------------------------------------------

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_post_model(self, event: messages.PostModelEvent) -> None:
    """Marks history to be reset on the next step when a new model is loaded."""
    del event
    self._model = None

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_reset(self, event: messages.ResetEvent) -> None:
    """Marks history to be reset on the next step after physics is reset."""
    del event
    self._model = None

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_post_step(self, event: messages.PostStepEvent) -> None:
    """Records history and broadcasts a SimHistorySnapshot after each step."""
    if self._model is not event.model:
      self._model = event.model
      ux.reset_history(
          self.sim_history, self._ux_state, event.model, event.data
      )
    elif event.data.time != self._last_recorded_time:
      ux.record_history(
          self.sim_history, self._ux_state, event.model, event.data
      )
    else:
      return
    self._last_recorded_time = event.data.time
    self._send_snapshot()

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_scrub(self, snapshot: SimHistorySnapshot) -> bool:
    """Loads a history frame and requests a pause when the viewer scrubs.

    See ``SimHistorySnapshot`` for the request format.
    """
    handle = self.handle
    # A ResetEvent/ModelEvent dispatched earlier in the same sync() clears
    # self._model; the buffer then still holds pre-reset frames that must not
    # overwrite the fresh data. Ignore the scrub; the next PostStepEvent
    # re-initialises the history and broadcasts it.
    if handle.model is not None and self._model is handle.model:
      assert handle.data is not None  # ViewerHandle keeps model/data paired.
      ux.load_history_frame(
          self.sim_history, handle.model, handle.data, snapshot.index
      )
      self._last_recorded_time = handle.data.time
      handle.dispatch(
          messages.RequestPauseEvent(
              pause_state=sim.PauseState.NORMAL_PAUSED,
          )
      )
      self._send_snapshot()
    return True
