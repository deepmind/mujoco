# Copyright 2022 DeepMind Technologies Limited
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
"""Python bindings for MuJoCo."""

import ctypes
import ctypes.util
import os
import platform
import subprocess
from typing import Any, IO, Sequence, TypeAlias, Union
import warnings
import zipfile

# Extend the path to enable multiple directories to contribute to the same
# package. Without this line, the `mujoco-mjx` package would not be able to
# be discovered by import. For more information, see: https://packaging.python.org/guides/packaging-namespace-packages/#pkgutil-style-namespace-packages
# NOTE: As per the Python Packaging User Guide linked above, the preferable way
# of declaring the namespace package is to use the native namespace packages.
# This seems non-trivial at the current state of the project, however.
# For more information, see: https://github.com/google-deepmind/mujoco/issues/2119
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

_SYSTEM = platform.system()
if _SYSTEM == 'Windows':
  _dll_path = os.path.join(os.path.dirname(__file__), 'mujoco.dll')
  if os.path.exists(_dll_path):
    ctypes.WinDLL(_dll_path)
elif _SYSTEM == 'Darwin':
  # sysctl is in /usr/sbin, which is not on minimal PATHs (e.g. the
  # environment Bazel runs tests in), so don't rely on PATH lookup.
  try:
    proc_translated = subprocess.run(
        ['/usr/sbin/sysctl', '-n', 'sysctl.proc_translated'],
        capture_output=True).stdout
    is_rosetta = bool(int(proc_translated))
  except (OSError, ValueError):
    is_rosetta = False
  if is_rosetta and platform.machine() == 'x86_64':
    raise ImportError(
        'You are running an x86_64 build of Python on an Apple Silicon '
        'machine. This is not supported by MuJoCo. Please install and run a '
        'native, arm64 build of Python.')

_EXPECTED_VERSION = '4.0.1'
try:
  from mujoco._version import __version__ as _core_version
  import mujoco._specs
except ModuleNotFoundError as _e:
  if not (_e.name or '').startswith('mujoco'):
    raise
  raise ImportError(
      "The 'mujoco' package is installed, but its core bindings "
      "('mujoco-core') are missing or corrupted (for example, after "
      "uninstalling a previous monolithic 'mujoco' package). "
      "Repair your environment with:\n"
      '  pip install --force-reinstall --no-deps '
      'mujoco mujoco-core mujoco-studio'
  ) from _e
if _core_version != _EXPECTED_VERSION:
  raise ImportError(
      f"Version mismatch between 'mujoco' ({_EXPECTED_VERSION}) and "
      f"'mujoco-core' ({_core_version}). Please upgrade all packages "
      'together:\n'
      f"  pip install --upgrade 'mujoco=={_EXPECTED_VERSION}' "
      f"'mujoco-core=={_EXPECTED_VERSION}' "
      f"'mujoco-studio=={_EXPECTED_VERSION}'"
  )

from mujoco import _specs
from mujoco import _structs
from mujoco._callbacks import *
from mujoco._constants import *
from mujoco._enums import *
from mujoco._errors import *
from mujoco._functions import *
from mujoco._specs import *
from mujoco._structs import *

import numpy as np
MJTNUM_DTYPE = np.float32 if MJTNUM_BYTES == 4 else np.float64  # pylint: disable=undefined-variable

try:
  from mujoco._render import *
except ImportError:
  pass

try:
  from mujoco.rendering.classic.renderer import Renderer  # pyrefly: ignore[missing-import]
except ImportError:
  pass

try:
  from mujoco.rendering.classic.gl_context import *  # pyrefly: ignore[missing-import]
except ImportError:
  pass

MjStruct: TypeAlias = Union[
    _specs.MjsBody,
    _specs.MjsFrame,
    _specs.MjsGeom,
    _specs.MjsJoint,
    _specs.MjsLight,
    _specs.MjsMaterial,
    _specs.MjsSite,
    _specs.MjsMesh,
    _specs.MjsSkin,
    _specs.MjsTexture,
    _specs.MjsText,
    _specs.MjsTuple,
    _specs.MjsCamera,
    _specs.MjsFlex,
    _specs.MjsHField,
    _specs.MjsKey,
    _specs.MjsNumeric,
    _specs.MjsPair,
    _specs.MjsExclude,
    _specs.MjsEquality,
    _specs.MjsTendon,
    _specs.MjsSensor,
    _specs.MjsActuator,
    _specs.MjsPlugin,
]


def to_zip(spec: _specs.MjSpec, file: Union[str, IO[bytes]]) -> None:
  """Converts an MjSpec to a zip file.

  Args:
    spec: The mjSpec to save to a file.
    file: The path to the file to save to or the file object to write to.
  """
  files_to_zip = dict(spec.assets)
  files_to_zip[spec.modelname + '.xml'] = spec.to_xml()
  if isinstance(file, str):
    directory = os.path.dirname(file)
    if directory:
      os.makedirs(directory, exist_ok=True)
    file = open(file, 'wb')
  with zipfile.ZipFile(file, 'w') as zip_file:
    for filename, contents in files_to_zip.items():
      zip_info = zipfile.ZipInfo(filename)
      zip_file.writestr(zip_info, contents)


def from_zip(file: Union[str, IO[bytes]]) -> _specs.MjSpec:
  """Reads a zip file and returns an MjSpec.

  Args:
    file: The path to the file to read from or the file object to read from.
  Returns:
    An MjSpec object.
  """
  assets = {}
  xml_string = None
  if isinstance(file, str):
    file = open(file, 'rb')
  if not zipfile.is_zipfile(file):
    raise ValueError(f'File {file} is not a zip file.')
  with zipfile.ZipFile(file, 'r') as zip_file:
    xml_dir = None
    for zip_info in zip_file.infolist():
      if not zip_info.filename.endswith(os.path.sep):
        with zip_file.open(zip_info.filename) as f:
          if zip_info.filename.endswith('.xml'):
            xml_string = f.read()
            xml_dir = os.path.dirname(zip_info.filename)
          else:
            assets[zip_info.filename] = f.read()

  if not xml_string:
    raise ValueError('No XML file found in zip file.')

  relative_assets = {}
  for key, value in assets.items():
    new_key = os.path.relpath(key, xml_dir)
    relative_assets[new_key] = value
  assets = relative_assets

  return _specs.MjSpec.from_string(xml_string, assets=assets)


def _bind_getattr(elements: Sequence[Any], key: str):
  items = []
  for e in elements:
    val = getattr(e, key)
    if isinstance(val, (str, bytes)):
      items.append(val)
    else:
      try:
        items.extend(val)
      except TypeError:
        items.append(val)
  return items


class _MjBindModel:
  """Wrapper for MjModel that allows binding multiple specs."""

  def __init__(self, elements: Sequence[Any]):
    object.__setattr__(self, 'elements', elements)

  def __getattr__(self, key: str):
    return _bind_getattr(self.elements, key)

  def __setattr__(self, key: str, value: Any):
    try:
      values = list(value)
    except TypeError:
      raise ValueError(
          f'Expected {len(self.elements)} values, got non-iterable'
          f' {type(value).__name__}'
      ) from None
    if len(values) != len(self.elements):
      raise ValueError(
          f'Expected {len(self.elements)} values, got {len(values)}'
      )
    for element, val in zip(self.elements, values):
      setattr(element, key, val)


class _MjBindData:
  """Wrapper for MjData that allows binding multiple specs."""

  def __init__(self, elements: Sequence[Any]):
    object.__setattr__(self, 'elements', elements)

  def __getattr__(self, key: str):
    return _bind_getattr(self.elements, key)

  def __setattr__(self, key: str, value: Any):
    try:
      values = list(value)
    except TypeError:
      raise ValueError(
          f'Expected {len(self.elements)} values, got non-iterable'
          f' {type(value).__name__}'
      ) from None
    if len(values) != len(self.elements):
      raise ValueError(
          f'Expected {len(self.elements)} values, got {len(values)}'
      )
    for element, val in zip(self.elements, values):
      setattr(element, key, val)


def _bind_model(
    model: _structs.MjModel, specs: Union[Sequence[MjStruct], MjStruct]
):
  """Bind a Mujoco spec to a mjModel.

  Args:
    model: The mjModel to bind to.
    specs: The mjSpec elements to use for binding, can be a single element or a
      sequence.
  Returns:
    A MjModelGroupedViews object or a list of the same type.
  """
  if isinstance(specs, Sequence):
    return _MjBindModel([model.bind_scalar(s) for s in specs])
  else:
    return model.bind_scalar(specs)


def _bind_data(
    data: _structs.MjData, specs: Union[Sequence[MjStruct], MjStruct]
):
  """Bind a Mujoco spec to a mjData.

  Args:
    data: The mjData to bind to.
    specs: The mjSpec elements to use for binding, can be a single element or a
      sequence.
  Returns:
    A MjDataGroupedViews object or a list of the same type.
  """
  if isinstance(specs, Sequence):
    return _MjBindData([data.bind_scalar(s) for s in specs])
  else:
    return data.bind_scalar(specs)

_specs.MjSpec.from_zip = from_zip
_specs.MjSpec.to_zip = to_zip
_structs.MjData.bind = _bind_data  # pyrefly: ignore[bad-assignment]
_structs.MjModel.bind = _bind_model  # pyrefly: ignore[bad-assignment]

__version__ = mj_versionString()  # pylint: disable=undefined-variable

HEADERS_DIR = os.path.join(os.path.dirname(__file__), 'include/mujoco')
PLUGINS_DIR = os.path.join(os.path.dirname(__file__), 'plugin')

PLUGIN_HANDLES = []

def _load_all_bundled_plugins():
  if os.path.isdir(PLUGINS_DIR):
    for directory, _, filenames in os.walk(PLUGINS_DIR):
      for filename in filenames:
        if os.path.splitext(filename)[-1] in [".dll", ".dylib", ".so"]:
          lib_path = os.path.join(directory, filename)
          PLUGIN_HANDLES.append(ctypes.CDLL(lib_path))
        elif filename == "__init__.py":
          pass
        else:
          warnings.warn('Ignoring non-library in plugin directory: '
                        f'{os.path.join(directory, filename)}', ImportWarning)
  try:
    import importlib.metadata as _metadata
    for _ep in _metadata.entry_points(group='mujoco.plugins'):
      try:
        _ep.load()()
      except Exception as _ep_err:  # pylint: disable=broad-exception-caught
        warnings.warn(
            f'Failed to load mujoco plugin entry point {_ep.name}: {_ep_err}',
            RuntimeWarning,
        )
  except Exception:  # pylint: disable=broad-exception-caught
    pass

_load_all_bundled_plugins()
