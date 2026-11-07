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
"""Tests for optional backend imports in mujoco.mjx.warp."""

import subprocess
import sys
import textwrap

from absl.testing import absltest

# Simulates a warp-lang that imports but is incompatible with the bundled
# mujoco_warp, as when warp-lang is older than the version mujoco_warp requires.
# Only the import in mujoco/mjx/warp/__init__.py is failed; warp/types.py
# already tolerates import failures on its own.
_INCOMPATIBLE_WARP_SCRIPT = textwrap.dedent("""
    import builtins

    real_import = builtins.__import__

    def _incompatible_bundled_warp(
        name, globals_=None, locals_=None, fromlist=(), level=0
    ):
      if (
          name == 'mujoco.mjx.third_party'
          and fromlist
          and 'mujoco_warp' in fromlist
          and globals_ is not None
          and globals_.get('__name__') == 'mujoco.mjx.warp'
      ):
        raise TypeError(
            "kernel() got an unexpected keyword argument 'grid_stride'"
        )
      return real_import(name, globals_, locals_, fromlist, level)

    builtins.__import__ = _incompatible_bundled_warp

    import mujoco.mjx
    import mujoco.mjx.warp as mjxw

    assert mjxw.mujoco_warp is None
    print('IMPORT_SUCCEEDED')
""")


class ImportFallbackTest(absltest.TestCase):

  @absltest.skipIf(not sys.executable, 'needs a Python interpreter binary')
  def test_incompatible_mujoco_warp_falls_back_to_jax(self):
    result = subprocess.run(
        [sys.executable, '-c', _INCOMPATIBLE_WARP_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('Failed to import mujoco_warp:', result.stdout)
    self.assertIn("unexpected keyword argument 'grid_stride'", result.stdout)
    self.assertIn('IMPORT_SUCCEEDED', result.stdout)


if __name__ == '__main__':
  absltest.main()
