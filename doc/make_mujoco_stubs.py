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
"""Generates a pip-installable doc-only mujoco stub package.

This creates a lightweight pure-Python package that provides the mujoco
public API surface (enums, constants) without C extensions, for use by
Sphinx autodoc during documentation builds.

Usage:
  python doc/make_mujoco_stubs.py <output_dir>
"""

import glob
import os
import re
import shutil
import sys
import textwrap
from typing import Dict, Sequence


def _read_version(repo_root: str) -> str:
  """Reads the mujoco version from python/pyproject.toml."""
  with open(os.path.join(repo_root, 'python', 'pyproject.toml')) as f:
    for line in f:
      m = re.fullmatch(r'version\s*=\s*"([^"]+)"', line.strip())
      if m:
        return m.group(1)
  raise RuntimeError('Could not find version in python/pyproject.toml')


def _parse_constants(repo_root: str) -> Dict[str, str]:
  """Parses numeric #define constants from MuJoCo C headers."""
  consts = {}
  for path in glob.glob(os.path.join(repo_root, 'include', 'mujoco', '*.h')):
    with open(path) as f:
      for line in f:
        # Match #define mjCONST 1.23 or 1.23f, capturing only the numeric part.
        # Use (?:\s|$) to ensure we match the end of the token safely.
        m = re.search(
            r'^\s*#define\s+(mj[A-Z]\w*)\s+([\d.eE+\-]+)f?(?:\s|$)', line
        )
        if m:
          consts[m.group(1)] = m.group(2)
  return consts


_PYPROJECT_TEMPLATE = textwrap.dedent("""\
    [build-system]
    requires = ["setuptools"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "mujoco"
    version = "{version}"
    requires-python = ">=3.11"
    dependencies = []

    [tool.setuptools]
    include-package-data = false

    [tool.setuptools.packages.find]
    include = ["mujoco*"]
""")

_INIT_PY_TEMPLATE = textwrap.dedent("""\
    \"\"\"Doc-only stub for MuJoCo. Provides enums and constants for autodoc.\"\"\"
    import enum
    import sys

    __path__ = __import__('pkgutil').extend_path(__path__, __name__)

    from mujoco.introspect.enums import ENUMS

    _mod = sys.modules[__name__]
    for _n, _d in ENUMS.items():
        _cls = enum.IntEnum(_n, list(_d.values.items()))
        setattr(_mod, _n, _cls)
        for _vn, _vv in _d.values.items():
            setattr(_mod, _vn, _vv)

    {constants}

    try:
        from importlib.metadata import version as _v
        __version__ = _v('mujoco')
    except Exception:
        __version__ = '0.0.0'

    def mj_versionString():
        return __version__

    # Stub types for C extensions (MjModel, MjData, mj_* functions, etc.)
    # needed by MJX and mujoco_warp imports during Sphinx autodoc.
    # Each accessed name gets a dynamically created class so that Sphinx
    # renders the real type name instead of "Mock".
    _mock_cache = {{}}

    def _make_mock_meta(mock_name):
        class _MockMeta(type):
            def __getattr__(cls, name):
                return _make_mock(f'{{mock_name}}.{{name}}')
            def __instancecheck__(cls, instance):
                return True
            def __repr__(cls):
                return mock_name
        return _MockMeta

    def _make_mock(qualname):
        if qualname in _mock_cache:
            return _mock_cache[qualname]
        basename = qualname.rsplit('.', 1)[-1]
        meta = _make_mock_meta(qualname)
        cls = meta(basename, (), {{
            '__init__': lambda self, *a, **kw: None,
            '__call__': lambda self, *a, **kw: _make_mock(qualname)(),
            '__getattr__': lambda self, name: _make_mock(f'{{qualname}}.{{name}}'),
            '__class_getitem__': classmethod(lambda cls, item: cls),
            '__iter__': lambda self: iter([]),
            '__bool__': lambda self: False,
            '__repr__': lambda self: qualname,
            '__module__': 'mujoco',
            '__qualname__': basename,
        }})
        _mock_cache[qualname] = cls
        return cls

    def __getattr__(name):
        return _make_mock(name)
""")


def main(argv: Sequence[str]) -> None:
  """Generates the doc-only mujoco stub package."""
  if len(argv) != 2:
    print(f'Usage: {argv[0]} <output_dir>', file=sys.stderr)
    sys.exit(1)

  output_dir = argv[1]
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  version = _read_version(repo_root)
  consts = _parse_constants(repo_root)

  mujoco_dir = os.path.join(output_dir, 'mujoco')
  os.makedirs(mujoco_dir, exist_ok=True)

  # Write pyproject.toml.
  with open(os.path.join(output_dir, 'pyproject.toml'), 'w') as f:
    f.write(_PYPROJECT_TEMPLATE.format(version=version))

  # Write mujoco/__init__.py with constants inlined.
  constants_str = '\n'.join(f'{k} = {v}' for k, v in sorted(consts.items()))
  with open(os.path.join(mujoco_dir, '__init__.py'), 'w') as f:
    f.write(_INIT_PY_TEMPLATE.format(constants=constants_str))

  # Copy introspect/ into the package.
  introspect_src = os.path.join(repo_root, 'python', 'mujoco', 'introspect')
  introspect_dst = os.path.join(mujoco_dir, 'introspect')
  shutil.rmtree(introspect_dst, ignore_errors=True)
  shutil.copytree(introspect_src, introspect_dst)

  print(f'Generated doc-only mujoco {version} package at {output_dir}')


if __name__ == '__main__':
  main(sys.argv)
