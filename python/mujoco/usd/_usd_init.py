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
"""Internal runtime initialization and plugin registration for mujoco-usd."""

import ctypes
import glob
import importlib
import os
import warnings

import mujoco

from mujoco._version import __version__ as _core_version

try:
  _plug = importlib.import_module('pxr.Plug')
except ImportError:
  _plug = None

__version__ = '4.0.0'

if _core_version != __version__:
  raise ImportError(
      f"Version mismatch between 'mujoco-usd' ({__version__}) and "
      f"'mujoco-core' ({_core_version}). "
      'Please install matching versions:\n'
      f"  pip install 'mujoco-usd=={__version__}' 'mujoco-core=={__version__}'"
  )
del _core_version

_PLUGINS_REGISTERED = False
_DLL_DIRS = []


def register_plugins() -> None:
  """Registers bundled OpenUSD schemas and the MuJoCo USD decoder plugin."""
  global _PLUGINS_REGISTERED
  if _PLUGINS_REGISTERED:
    return
  _PLUGINS_REGISTERED = True

  usd_dir = os.path.dirname(__file__)

  if _plug is not None and getattr(_plug, '__file__', None):
    pxr_dir = os.path.dirname(os.path.dirname(_plug.__file__))
    for usd_libs_dir in (
        os.path.join(os.path.dirname(pxr_dir), 'usd_core.libs'),
        os.path.join(pxr_dir, '.dylibs'),
    ):
      if os.path.isdir(usd_libs_dir):
        for soname, pattern in (
            ('libtbb.so.2', 'libtbb*.so.2'),
            ('libusd_ms.so', 'libusd_ms*.so'),
            ('libtbb.dylib', 'libtbb*.dylib'),
            ('libusd_ms.dylib', 'libusd_ms*.dylib'),
        ):
          matches = sorted(glob.glob(os.path.join(usd_libs_dir, pattern)))
          if matches:
            link_path = os.path.join(usd_dir, soname)
            if not os.path.exists(link_path):
              try:
                if os.path.lexists(link_path):
                  os.unlink(link_path)
                os.symlink(os.path.relpath(matches[0], usd_dir), link_path)
              except OSError:
                pass
            try:
              ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
            except OSError:
              pass

  if hasattr(os, 'add_dll_directory'):
    dll_dirs = [os.path.dirname(mujoco.__file__), usd_dir]
    if _plug is not None and getattr(_plug, '__file__', None):
      dll_dirs.append(os.path.dirname(os.path.dirname(_plug.__file__)))
    for dll_dir in dll_dirs:
      if os.path.isdir(dll_dir):
        try:
          _DLL_DIRS.append(os.add_dll_directory(dll_dir))
        except OSError:
          pass

  for lib_stem in ('mjcPhysics', 'mujoco_usd', 'usdMjcf'):
    for ext in ('.so', '.dylib', '.dll'):
      for prefix in ('lib', ''):
        candidate = os.path.join(usd_dir, f'{prefix}{lib_stem}{ext}')
        if os.path.exists(candidate):
          try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
          except OSError as e:
            warnings.warn(
                f'Failed to preload USD library {candidate}: {e}',
                RuntimeWarning,
            )

  resources_dir = os.path.join(usd_dir, 'mujoco-usd-resources')
  if os.path.isdir(resources_dir) and _plug is not None:
    try:
      _plug.Registry().RegisterPlugins(
          glob.glob(os.path.join(resources_dir, '*', 'plugInfo.json'))
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      warnings.warn(
          f'Failed to register MuJoCo OpenUSD plugins from {resources_dir}:'
          f' {e}',
          RuntimeWarning,
      )

  for ext in ('.so', '.dylib', '.dll'):
    for prefix in ('lib', ''):
      decoder_path = os.path.join(usd_dir, f'{prefix}usd_decoder_plugin{ext}')
      if os.path.exists(decoder_path):
        try:
          mujoco.mj_loadPluginLibrary(decoder_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
          warnings.warn(
              f'Failed to load MuJoCo USD decoder plugin {decoder_path}: {e}',
              RuntimeWarning,
          )


register_plugins()
