# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import typing
from typing import Any
from mujoco.mjx.warp import render_context
from mujoco.mjx.warp import types

if not typing.TYPE_CHECKING:
  # Runtime.
  warp: Any = None
  mujoco_warp: Any = None
  mjwp_types: Any = None
  mjwp_io: Any = None
  WARP_INSTALLED: bool = False

  # pylint: disable=g-import-not-at-top
  # The warp backend is optional, so any failure to import it must leave MJX
  # usable through the JAX implementation. For example, warp-lang older than
  # the version required by the bundled mujoco_warp raises TypeError during
  # mujoco_warp's initialization.
  try:
    import warp
    WARP_INSTALLED = True
  except Exception as e:  # pylint: disable=broad-exception-caught
    print('Failed to import warp:', e)
    WARP_INSTALLED = False
  try:
    from mujoco.mjx.third_party import mujoco_warp
    from mujoco.mjx.third_party.mujoco_warp._src import types as mjwp_types
    from mujoco.mjx.third_party.mujoco_warp._src import io as mjwp_io
  except Exception as e:  # pylint: disable=broad-exception-caught
    print('Failed to import mujoco_warp:', e)
    pass
  # pylint: enable=g-import-not-at-top
else:
  # Only used for type checking.
  class _WpStub:

    def ScopedDevice(self, device: str):  # pylint: disable=invalid-name
      pass

    def types(self):
      pass

  class _MjwpStub:

    def put_model(self, *args, **kwargs):
      pass

    def make_data(self, *args, **kwargs):
      pass

    def step(self, *args, **kwargs):
      pass

  class _MjwpTypesStub:
    def TileSet(self, *args, **kwargs):  # pylint: disable=invalid-name
      pass

    def BlockDim(self, *args, **kwargs):  # pylint: disable=invalid-name
      pass

  WARP_INSTALLED: bool = True
  warp: Any = _WpStub()
  mujoco_warp: Any = _MjwpStub()
  mjwp_types: Any = _MjwpTypesStub()
