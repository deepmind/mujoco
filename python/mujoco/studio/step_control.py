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
"""Default sim-side stepping plugin."""

import contextlib
import time

from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import sim

# The achieved speed (the snapshot's ``speed_measured``) drifts a little every
# step. It is reported again only once it has moved this far, in percent of
# real time, from the value last reported.
_MEASURED_SPEED_DEADBAND = 1.0


class StepControl(plugin.SimPlugin):
  """Default sim plugin that steps CPU physics with real-time pacing.

  Pass your own plugin instead to step differently (e.g. a GPU simulation).

  Pacing: this plugin owns real-time pacing. It steps as much sim time as fits
  the wall-clock budget of one loop iteration, then sleeps 1ms per StepEvent.
  The sleep matters because advance() returns at once when no step is due yet
  or while paused, and sync() itself never sleeps: without it, a bare
  ``while handle.is_running(): handle.sync()`` loop spins at 100% CPU and
  starves the viewer threads of the GIL. The 1ms yield matches the physics
  loops of the simulate app and of the native Studio app; advance() catches up
  on the next iteration. It lives in this plugin rather than in advance() so
  that advance() stays non-blocking for callers on render threads (the browser
  build, scripts with their own frame loop). A plugin that replaces this one
  must at minimum pace each iteration the same way.

  Reporting: a ``StepControlSnapshot`` goes to the viewer whenever a setting
  (pause, speed, noise) changes, and whenever the achieved speed has moved by
  ``_MEASURED_SPEED_DEADBAND``.
  """

  def __init__(self) -> None:
    self._model_path: str = ''
    self.step_control = sim.StepControl()
    self._measured_reported = -1.0

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_sim_init(self, _: plugin.SimInitEvent) -> None:
    self._model_path = ''
    self.step_control = sim.StepControl()
    self._measured_reported = -1.0

  def _snapshot(self) -> messages.StepControlSnapshot:
    """Returns the settings, with the achieved speed held within its deadband.

    A negative last-reported value means none has been reported yet.
    """
    measured = self.step_control.get_speed_measured()
    unknown = self._measured_reported < 0.0
    if unknown or abs(measured - self._measured_reported) >= (
        _MEASURED_SPEED_DEADBAND
    ):
      self._measured_reported = measured
    return messages.StepControlSnapshot.from_step_control(
        self.step_control, speed_measured=self._measured_reported
    )

  def _send_if_changes(self) -> contextlib.AbstractContextManager[None]:
    return self.handle.send_if_changes(self._snapshot)

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_post_model(self, event: messages.PostModelEvent) -> None:
    same_scene = bool(event.path and event.path == self._model_path)
    self._model_path = event.path
    if same_scene:
      # Same scene reloaded: keep pause, speed, and noise; only resync pacing.
      self.step_control.force_sync()
    else:
      self.step_control = sim.StepControl()
      self._measured_reported = -1.0

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_step_control(self, event: messages.StepControlSnapshot) -> bool:
    self.step_control.set_pause_state(event.pause_state)
    self.step_control.set_speed(event.speed)
    self.step_control.set_noise_parameters(event.noise_scale, event.noise_rate)
    return True

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_single_step(self, _: messages.SingleStepEvent) -> bool:
    self.step_control.request_single_step()
    return True

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_pause(self, event: messages.RequestPauseEvent) -> bool:
    with self._send_if_changes():
      self.step_control.set_pause_state(event.pause_state)
    return True

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_step(self, event: messages.StepEvent) -> None:
    with self._send_if_changes():
      self.step_control.advance(event.model, event.data)
    # Yield 1ms per StepEvent; see the class docstring ("Pacing") for why.
    time.sleep(0.001)
