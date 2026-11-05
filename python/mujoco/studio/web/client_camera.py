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
"""Hands the controller's camera between the browser and the web viewer.

The controlling browser moves the camera itself (Viewer.client_camera) and
sends it to the viewer, but the viewer still changes it in other ways too:
camera shortcuts, the view gizmo, picking a body to track, loading a model.
Each such viewer-side change starts a new camera generation, streamed to the
browser in the state payload. The browser adopts the payload camera only when
the generation moves on (otherwise it is just an echo of a camera the browser
sent), and tags each camera it sends with the generation it builds on, so the
viewer drops cameras that predate its own latest change instead of letting
them undo it.
"""

import math

import mujoco

# Generations are streamed as uint32 (ViewerInputState.camera_generation).
_GENERATION_MASK = 0xFFFFFFFF


def _camera_fields(camera: mujoco.MjvCamera) -> tuple[float, ...]:
  return (
      camera.type,
      camera.fixedcamid,
      camera.trackbodyid,
      camera.orthographic,
      *camera.lookat,
      camera.distance,
      camera.azimuth,
      camera.elevation,
  )


class ClientCameraSync:
  """Tracks the viewer's camera generation; see the module docstring."""

  def __init__(self):
    self.generation = 0
    # The camera as of the last apply() or update_generation(), against which
    # update_generation() detects the viewer's own changes.
    self._synced: tuple[float, ...] | None = None

  def apply(self, camera: mujoco.MjvCamera, message: bytes) -> bool:
    """Adopts a camera the browser sent, unless it predates `generation`.

    The browser only moves the camera within the mode the viewer streamed to it
    (lookat, distance, azimuth and elevation); switching the mode (type,
    fixedcamid, trackbodyid, orthographic) is the viewer's, which starts a new
    generation. So a message must carry the viewer's current mode: the browser
    never sets camera or body indices, which the viewer would use unchecked.

    Args:
      camera: The viewer's camera, updated in place.
      message: The camera as sent by the browser, a ClientCameraMessage (see
        state_payload.h).

    Returns:
      Whether the camera was adopted; False if the message is malformed (incl.
      non-finite values), changes the camera mode, or predates the viewer's
      latest change (including one not yet streamed).
    """
    if self._synced is not None and _camera_fields(camera) != self._synced:
      return False  # The viewer changed it; update_generation() will bump.
    # Imported here, not at the top: viewer_protocol imports this module, so it
    # must import where the web viewer's native modules are not built (e.g.
    # Windows wheels). Only the web viewer calls apply().
    from mujoco.studio.web import state_payload  # pylint: disable=g-import-not-at-top

    sent = state_payload.parse_client_camera(message)
    if sent is None or sent.generation != self.generation:
      return False
    sent_mode = (sent.type, sent.fixedcamid, sent.trackbodyid, sent.orthographic)
    mode = (
        int(camera.type),
        camera.fixedcamid,
        camera.trackbodyid,
        int(camera.orthographic),
    )
    floats = (*sent.lookat, sent.distance, sent.azimuth, sent.elevation)
    if sent_mode != mode or not all(math.isfinite(v) for v in floats):
      return False
    camera.lookat[:] = sent.lookat
    camera.distance = sent.distance
    camera.azimuth = sent.azimuth
    camera.elevation = sent.elevation
    self._synced = _camera_fields(camera)
    return True

  def update_generation(self, camera: mujoco.MjvCamera) -> int:
    """Returns the generation to stream with `camera`.

    Starts a new generation if the viewer changed the camera since the last
    call (or since it last adopted the browser's).

    Args:
      camera: The viewer's camera, as about to be streamed to the browser.
    """
    fields = _camera_fields(camera)
    if self._synced is not None and fields != self._synced:
      self.generation = (self.generation + 1) & _GENERATION_MASK
    self._synced = fields
    return self.generation
