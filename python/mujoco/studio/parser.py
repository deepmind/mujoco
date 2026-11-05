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
"""Loads and compiles MuJoCo models for the studio viewer."""

import mujoco


def load(filepath: str) -> mujoco.MjModel:
  """Loads and compiles a model file: XML, URDF, MJB, or MJZ/ZIP."""
  if filepath.endswith('.mjb'):
    return mujoco.MjModel.from_binary_path(filepath)
  if filepath.endswith(('.mjz', '.zip')):
    # .mjz archives mount assets into a VFS during parsing; the same VFS must
    # be passed to compile() so the assets remain accessible.
    with mujoco.MjVfs() as vfs:
      spec = mujoco.MjSpec.from_file(filepath, vfs=vfs)
      return spec.compile(vfs=vfs)
  return mujoco.MjSpec.from_file(filepath).compile()


def parse(filepath: str) -> mujoco.MjData:
  """Loads a model file and returns a fresh ``MjData`` for it."""
  return mujoco.MjData(load(filepath))


def dependencies(filepath: str) -> list[str]:
  """Returns the files a model file is built from, including ``filepath``.

  For XML/URDF: the file, its ``<include>`` and ``<model>`` trees and their
  mesh, skin, hfield and texture assets. Other formats: just the file.

  Args:
    filepath: The model file.
  """
  if filepath.endswith(('.mjb', '.mjz', '.zip')):
    return [filepath]
  try:
    return [filepath, *mujoco.mju_getXMLDependencies(filepath)]
  except Exception:  # pylint: disable=broad-exception-caught
    return [filepath]  # Unparsable: watch it, so saving a fix reloads it.
