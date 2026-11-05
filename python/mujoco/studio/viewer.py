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
"""Interactive Studio GUI viewer for MuJoCo.

Pass ``--plugins=a.py,b.py`` to load ``ViewerPlugin`` and ``SimPlugin``
subclasses from Python files (see ``plugin.load_plugins``). With
``--hot_reload=all``, saving those files, the model file, or its assets reloads
them without restarting (``plugins``, ``model``, or plugin class names narrow
that down).
"""

import argparse
import os

from mujoco.studio import app as studio_app
from mujoco.studio import launch_passive
from mujoco.studio import plugin
from mujoco.studio import viewer_protocol

vp = viewer_protocol


def _parse_plugins(value: str) -> list[str]:
  return [p for p in value.split(',') if p]


def _resolve(path: str) -> str:
  """Resolves a relative path against the directory the command was run from."""
  # Build tools such as Bazel run binaries from their output tree and export
  # the invoking directory as BUILD_WORKING_DIRECTORY; elsewhere it is unset.
  if path:
    path = os.path.join(os.environ.get('BUILD_WORKING_DIRECTORY', ''), path)
  return path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'pos_model', nargs='?', default='', help='Path to model file.'
  )
  parser.add_argument(
      '--model', type=str, default=None, help='Path to model file.'
  )
  parser.add_argument(
      '--plugins',
      type=_parse_plugins,
      default=[],
      help='Comma-separated list of Python plugin files (.py) to load.',
  )
  parser.add_argument(
      '--gfx',
      choices=vp.GFX_MODES,
      default='',
      help='Graphics mode ("web" launches Web Viewer).',
  )
  parser.add_argument(
      '--port',
      type=int,
      default=0,
      help='Web Viewer port (0 picks first free port >= 8080).',
  )
  parser.add_argument(
      '--width', type=int, default=1200, help='Width of the output image.'
  )
  parser.add_argument(
      '--height', type=int, default=800, help='Height of the output image'
  )
  parser.add_argument(
      '--hot_reload',
      type=str,
      default='',
      help=(
          "What is reloaded when modified. Options are '', 'plugins', 'model',"
          " 'all', 'Plugin1,Plugin2,...'. Default is '', meaning nothing."
      ),
  )
  args, _ = parser.parse_known_args()

  config = vp.ViewerConfig(
      width=args.width,
      height=args.height,
      gfx=args.gfx or '',
      http_port=args.port,
      hot_reload=args.hot_reload,
  )

  # Resolve model path, if provided.
  model_path = args.model or args.pos_model
  user_plugins = plugin.load_plugins(map(_resolve, args.plugins))

  launch_passive.run(
      config,
      model_path=_resolve(model_path),
      plugins=[
          studio_app.App(),
          *studio_app.SIM_PLUGINS,
          *user_plugins,
      ],
  )


if __name__ == '__main__':
  main()
