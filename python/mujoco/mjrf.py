# Copyright 2026 DeepMind Technologies Limited
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
"""Convenience wrapper around the MuJoCo Filament C++/python bindings."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mujoco._render_filament import *  # pylint: disable=wildcard-import,g-import-not-at-top
else:
  try:
    _rf = importlib.import_module('mujoco._render_filament')
    globals().update(
        {k: v for k, v in vars(_rf).items() if not k.startswith('_')}
    )
    del _rf
  except ImportError as _err:
    raise ImportError(
        "Filament rendering requires 'mujoco-studio' to be installed. "
        'Install it with:\n'
        '  pip install mujoco-studio\n'
        'or install the full MuJoCo bundle with:\n'
        '  pip install mujoco'
    ) from _err
del importlib, TYPE_CHECKING
