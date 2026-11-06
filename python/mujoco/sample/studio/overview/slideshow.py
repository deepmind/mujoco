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
r"""Minimal web viewer for the MuJoCo 4.0 feature slideshow.

Runs the interactive slideshow in the browser with mouse controls and a
floating caption card::

  python -m mujoco.sample.studio.overview.slideshow

For hands-free playback or recording::

  python -m mujoco.sample.studio.overview.slideshow \
      --autoplay=8 --orbit=10 --hide_card

For live editing, pass ``--hot_reload=all``::

  python -m mujoco.sample.studio.overview.slideshow --hot_reload=all
"""

import argparse

from mujoco.sample.studio.overview import slideshow_plugins
from mujoco.studio import launch_passive
from mujoco.studio import sim_history
from mujoco.studio import step_control
from mujoco.studio import viewer_protocol


def main() -> None:
  slide_slugs = ', '.join(
      f'{i + 1}:{s.slug}' for i, s in enumerate(slideshow_plugins.SLIDES)
  )
  parser = argparse.ArgumentParser(
      description='Interactive MuJoCo 4.0 feature slideshow viewer.'
  )
  parser.add_argument(
      '--slide',
      type=str,
      default='1',
      help=f'First slide number or name ({slide_slugs}).',
  )
  parser.add_argument(
      '--gfx',
      choices=viewer_protocol.GFX_MODES,
      default='web',
      help='Graphics backend ("web" opens the web viewer).',
  )
  parser.add_argument(
      '--port',
      type=int,
      default=0,
      help='Web viewer HTTP port (0 picks a free port >= 8080).',
  )
  parser.add_argument(
      '--open_browser',
      action='store_true',
      help='Open a browser tab automatically when a display is available.',
  )
  parser.add_argument(
      '--autoplay',
      type=float,
      default=0.0,
      help='Seconds per slide before advancing (0 disables autoplay).',
  )
  parser.add_argument(
      '--orbit',
      type=float,
      default=0.0,
      help='Camera orbit speed in deg/s when idle.',
  )
  parser.add_argument(
      '--hide_card',
      action='store_true',
      help=(
          'Start with the card hidden; move the mouse to the bottom edge to'
          ' show controls.'
      ),
  )
  parser.add_argument(
      '--prefetch',
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          'Compile and download all slides in the background for instant'
          ' switching.'
      ),
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help='Directory containing the MuJoCo XML scenes.',
  )
  parser.add_argument(
      '--width',
      type=int,
      default=1280,
      help='Window width for native graphics modes.',
  )
  parser.add_argument(
      '--height',
      type=int,
      default=800,
      help='Window height for native graphics modes.',
  )
  parser.add_argument(
      '--hot_reload',
      type=str,
      default='',
      help="Live reload mode ('', 'plugins', 'model', or 'all').",
  )
  args, _ = parser.parse_known_args()

  initial_index = slideshow_plugins.resolve_slide_index(args.slide)
  config = viewer_protocol.ViewerConfig(
      title='MuJoCo 4.0 Feature Showcase',
      width=args.width,
      height=args.height,
      gfx=args.gfx,
      http_port=args.port,
      open_browser=args.open_browser,
      hot_reload=args.hot_reload,
  )

  launch_passive.run(
      config,
      plugins=[
          step_control.StepControl(),
          # Flex scenes have large states; keep at least 10 s of scrubbable
          # history by recording every Nth step where needed.
          sim_history.SimHistory(min_seconds=10.0),
          slideshow_plugins.SlideshowSim(
              initial_slide=initial_index,
              autoplay_seconds=args.autoplay,
              model_dir=args.model_dir,
              prefetch=args.prefetch,
          ),
          slideshow_plugins.SlideshowOverlay(
              hide_card=args.hide_card,
              orbit_deg_per_sec=args.orbit,
          ),
      ],
  )


if __name__ == '__main__':
  main()
