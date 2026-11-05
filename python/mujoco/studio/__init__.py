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
"""MuJoCo Studio package."""

import importlib as _importlib
import importlib.machinery as _machinery
import sys as _sys

try:
  _studio_ext = _importlib.import_module('mujoco.studio._studio')
except ImportError:
  pass
else:
  for _name in ('native_viewer_cc', 'sim', 'ux', 'window'):
    _submod = getattr(_studio_ext, _name, None)
    if _submod is not None:
      _submod.__spec__ = _machinery.ModuleSpec(f'{__name__}.{_name}', None)
      globals()[_name] = _submod
      _sys.modules[f'{__name__}.{_name}'] = _submod
  del _name, _submod, _studio_ext
del _importlib, _machinery, _sys
