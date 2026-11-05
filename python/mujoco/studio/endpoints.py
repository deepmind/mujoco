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
"""Endpoints route messages to the right channel."""

from mujoco.studio import messages as vp


class ViewerEndpoint:
  """Viewer-side endpoint that routes Messages to/from the Sim.

  Outgoing (viewer-to-sim): EventChannel and SnapshotChannel.
  Incoming (sim-to-viewer): EventChannel and SnapshotChannel.

  Each direction uses its own channel instances, so a Snapshot type sent in both
  directions occupies separate per-type slots with no collision.
  """

  def __init__(
      self,
      *,
      s2v_snapshot: vp.SnapshotChannel,
      s2v_events: vp.EventChannel,
      v2s_events: vp.EventChannel,
      v2s_snapshot: vp.SnapshotChannel,
  ):
    self._s2v_snapshot = s2v_snapshot
    self._s2v_events = s2v_events
    self._v2s_events = v2s_events
    self._v2s_snapshot = v2s_snapshot
    self._is_closed = False

  def send_to_sim(self, message: vp.Message) -> None:
    if self._is_closed:
      raise RuntimeError('ViewerEndpoint is closed')
    if isinstance(message, vp.Event):
      self._v2s_events.put(message)
    elif isinstance(message, vp.Snapshot):
      self._v2s_snapshot.put(message)
    else:
      name = type(message).__name__
      raise TypeError(f'expected Event or Snapshot, got {name}')

  def get_sim_events(self) -> list[vp.Event]:
    """Returns all pending events from the simulation."""
    if self._is_closed:
      raise RuntimeError('ViewerEndpoint is closed')
    return self._s2v_events.get()

  def get_sim_snapshots(self) -> list[vp.Snapshot]:
    """Returns all pending latest snapshots from the simulation, one per type."""
    if self._is_closed:
      raise RuntimeError('ViewerEndpoint is closed')
    return self._s2v_snapshot.get()

  def close(self) -> None:
    """Close all channels owned by this endpoint (idempotent)."""
    if not self._is_closed:
      self._is_closed = True
      self._v2s_events.close()
      self._v2s_snapshot.close()
      self._s2v_events.close()
      self._s2v_snapshot.close()


class SimEndpoint:
  """Sim-side endpoint that routes Messages to/from the Viewer.

  Outgoing (sim-to-viewer): EventChannel and SnapshotChannel.
  Incoming (viewer-to-sim): EventChannel and SnapshotChannel.

  Each direction uses its own channel instances, so a Snapshot type sent in both
  directions occupies separate per-type slots with no collision.
  """

  def __init__(
      self,
      *,
      s2v_snapshot: vp.SnapshotChannel,
      s2v_events: vp.EventChannel,
      v2s_events: vp.EventChannel,
      v2s_snapshot: vp.SnapshotChannel,
  ):
    self._s2v_snapshot = s2v_snapshot
    self._s2v_events = s2v_events
    self._v2s_events = v2s_events
    self._v2s_snapshot = v2s_snapshot
    self._is_closed = False

  def send_to_viewer(self, message: vp.Message) -> None:
    if self._is_closed:
      raise RuntimeError('SimEndpoint is closed')
    if isinstance(message, vp.Event):
      self._s2v_events.put(message)
    elif isinstance(message, vp.Snapshot):
      self._s2v_snapshot.put(message)
    else:
      raise TypeError(
          f'expected Snapshot or Event, got {type(message).__name__}'
      )

  def get_viewer_events(self) -> list[vp.Event]:
    """Returns all pending events from the viewer."""
    if self._is_closed:
      raise RuntimeError('SimEndpoint is closed')
    return self._v2s_events.get()

  def get_viewer_snapshots(self) -> list[vp.Snapshot]:
    """Returns all pending latest snapshots from the viewer, one per type."""
    if self._is_closed:
      raise RuntimeError('SimEndpoint is closed')
    return self._v2s_snapshot.get()

  def close(self) -> None:
    """Close all channels owned by this endpoint."""
    if not self._is_closed:
      self._is_closed = True
      self._s2v_snapshot.close()
      self._s2v_events.close()
      self._v2s_events.close()
      self._v2s_snapshot.close()


def make_endpoints(
    *,
    s2v_snapshot: vp.SnapshotChannel,
    s2v_events: vp.EventChannel,
    v2s_events: vp.EventChannel,
    v2s_snapshot: vp.SnapshotChannel,
) -> tuple[ViewerEndpoint, SimEndpoint]:
  """Returns viewer and simulation endpoints."""
  viewer_endpoint = ViewerEndpoint(
      s2v_snapshot=s2v_snapshot,
      s2v_events=s2v_events,
      v2s_events=v2s_events,
      v2s_snapshot=v2s_snapshot,
  )
  sim_endpoint = SimEndpoint(
      s2v_snapshot=s2v_snapshot,
      s2v_events=s2v_events,
      v2s_events=v2s_events,
      v2s_snapshot=v2s_snapshot,
  )
  return viewer_endpoint, sim_endpoint
