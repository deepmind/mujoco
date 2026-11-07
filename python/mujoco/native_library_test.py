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
"""Tests that the installed package can be used to build native modules."""

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile

from absl.testing import absltest
import mujoco


_PACKAGE_DIR = os.path.dirname(os.path.abspath(mujoco.__file__))


def _linker_library_name():
  system = platform.system()
  if system == 'Windows':
    return 'mujoco.lib'
  if system == 'Darwin':
    return 'libmujoco.dylib'
  return 'libmujoco.so'


class NativeLibraryTest(absltest.TestCase):

  def test_linker_library_is_packaged(self):
    self.assertTrue(
        os.path.exists(os.path.join(_PACKAGE_DIR, _linker_library_name())),
        f'{_linker_library_name()} is missing from {_PACKAGE_DIR}; native '
        'modules cannot link against the packaged MuJoCo library',
    )

  def test_headers_are_packaged(self):
    self.assertTrue(
        os.path.exists(
            os.path.join(_PACKAGE_DIR, 'include', 'mujoco', 'mujoco.h')
        )
    )

  @absltest.skipIf(
      platform.system() == 'Windows', 'MSVC is not available on the test PATH'
  )
  def test_build_extension_against_package(self):
    """Compiles and loads a small shared library using `-lmujoco`."""
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
      self.skipTest('no C compiler available')

    with tempfile.TemporaryDirectory() as tmp:
      source = os.path.join(tmp, 'probe.c')
      with open(source, 'w', encoding='utf-8') as f:
        f.write(
            '#include <mujoco/mujoco.h>\n'
            'int probe(void) { return mj_version(); }\n'
        )

      is_apple = platform.system() == 'Darwin'
      flags = ['-dynamiclib'] if is_apple else ['-shared', '-fPIC']
      output = os.path.join(tmp, 'libprobe' + ('.dylib' if is_apple else '.so'))
      subprocess.run(
          [
              cc,
              *flags,
              f'-I{os.path.join(_PACKAGE_DIR, "include")}',
              f'-L{_PACKAGE_DIR}',
              '-lmujoco',
              f'-Wl,-rpath,{_PACKAGE_DIR}',
              source,
              '-o',
              output,
          ],
          check=True,
          capture_output=True,
      )

      probe = ctypes.CDLL(output)
      probe.probe.restype = ctypes.c_int
      self.assertEqual(probe.probe(), mujoco.mj_version())


if __name__ == '__main__':
  absltest.main()
