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
"""Example of a script that owns its simulation loop without ``run_sim_loop``.

Studio normally steps and paces physics in a sim plugin (``StepControl``), and
the launchers' ``run()`` only loops over ``handle.sync()``. A script that
already has a loop of its own can keep that loop and attach the viewer to it
with ``launch(...)`` instead:

  * ``model, data = handle.sync(model, data)`` hands the handle the model this
    script built and returns the one to keep stepping: the same objects, or new
    ones when the viewer replaced the model (a file dropped on it).
  * Without ``StepControl``, ``sync()`` neither steps nor sleeps. This loop
    steps ``data`` itself, sleeps the rest of each timestep (scaled by the
    speed the viewer asked for) and skips stepping while the viewer is paused.

Here the scene is a stack of boxes that gains a box every few simulated
seconds, to show the loop swapping in a model of its own.
"""

import argparse
import time

import mujoco
from mujoco.studio import app as studio_app
from mujoco.studio import launch_web
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import sim
from mujoco.studio import viewer_protocol


def build(num_boxes: int) -> mujoco.MjModel:
  """Builds ``num_boxes`` boxes stacked above a plane."""
  boxes = ''.join(
      f'<body pos="0 0 {0.3 + 0.25 * i}"><freejoint/>'
      '<geom type="box" size="0.1 0.1 0.1" rgba="0.2 0.5 0.9 1"/></body>'
      for i in range(num_boxes)
  )
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1"/>
        {boxes}
      </worldbody>
    </mujoco>
  """)


class _ViewerControls(plugin.SimPlugin):
  """Mirrors the viewer's pause button and speed setting for the loop below.

  The viewer sends ``StepControlSnapshot`` when either changes; ``StepControl``
  would normally act on it, here the loop in ``main`` does.
  """

  def __init__(self) -> None:
    self.paused = False
    self.speed = 100.0  # Percent of real time.

  @messages.handler
  def on_step_control(self, snapshot: messages.StepControlSnapshot) -> None:
    self.paused = snapshot.pause_state != sim.PauseState.UNPAUSED
    self.speed = snapshot.speed


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--port',
      type=int,
      default=0,
      help='Web Viewer port (0 picks first free port >= 8080).',
  )
  args, _ = parser.parse_known_args()
  config = viewer_protocol.ViewerConfig(
      title='custom_sim_loop', http_port=args.port
  )
  controls = _ViewerControls()
  num_boxes = 1
  model = build(num_boxes)
  data = mujoco.MjData(model)
  own_model = model  # Scene built by this script (vs. a dropped file).

  with launch_web.launch(
      config,
      # No StepControl: loop paces.
      plugins=[studio_app.App(), controls],
  ) as handle:
    while handle.is_running():
      tick = time.perf_counter()

      # Returns the current model and data (new objects if the viewer replaced
      # the model).
      model, data = handle.sync(model, data)
      assert model is not None and data is not None

      if not controls.paused:
        mujoco.mj_step(model, data)

      # Grow the scene now and then; sync() publishes the new model next time.
      if model is own_model and data.time > 4.0:
        num_boxes = num_boxes % 8 + 1
        own_model = model = build(num_boxes)
        data = mujoco.MjData(model)

      # Manually pace at the requested control speed.
      period = model.opt.timestep * 100.0 / controls.speed
      time.sleep(max(0.0, period - (time.perf_counter() - tick)))


if __name__ == '__main__':
  main()
