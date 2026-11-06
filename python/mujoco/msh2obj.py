# Copyright 2023 DeepMind Technologies Limited
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
"""CLI for converting legacy MSH files to Wavefront OBJ files.

Usage:
  python -m mujoco.msh2obj -i <msh_file> -o <obj_file>
"""

import argparse
import dataclasses
import io
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class Msh:
  """MuJoCo legacy binary msh file."""

  vertex_positions: np.ndarray
  vertex_normals: np.ndarray
  vertex_texcoords: np.ndarray
  face_vertex_indices: np.ndarray

  @staticmethod
  def create(file: pathlib.Path) -> "Msh":
    """Create a Msh object from a .msh file."""
    if not file.exists():
      raise FileNotFoundError(f"{file} does not exist.")

    with open(file, "rb") as f:
      header = np.fromfile(f, dtype=np.int32, count=4)
      if header.size != 4:
        raise ValueError(
            f"Invalid MSH header: expected 4 counts, got {header.size}."
        )
      if np.any(header < 0):
        raise ValueError(
            f"Invalid MSH header: counts must be nonnegative, got {header}."
        )
      nvertex, nnormal, ntexcoord, nface = header
      vertex_positions = np.fromfile(f, dtype=np.float32, count=3 * nvertex)
      vertex_normals = np.fromfile(f, dtype=np.float32, count=3 * nnormal)
      vertex_texcoords = np.fromfile(f, dtype=np.float32, count=2 * ntexcoord)
      face_vertex_indices = np.fromfile(f, dtype=np.int32, count=3 * nface)

    if vertex_positions.size != 3 * nvertex:
      raise ValueError(
          f"Invalid number of vertices: {vertex_positions.size} != 3*{nvertex}."
      )
    if vertex_normals.size != 3 * nnormal:
      raise ValueError(
          f"Invalid number of normals: {vertex_normals.size} != 3*{nnormal}."
      )
    if vertex_texcoords.size != 2 * ntexcoord:
      raise ValueError(
          f"Invalid number of texcoords: {vertex_texcoords.size} != "
          f"2*{ntexcoord}."
      )
    if face_vertex_indices.size != 3 * nface:
      raise ValueError(
          f"Invalid number of faces: {face_vertex_indices.size} != 3*{nface}."
      )

    vertex_positions = vertex_positions.reshape(-1, 3)
    vertex_normals = vertex_normals.reshape(-1, 3)
    face_vertex_indices = face_vertex_indices.reshape(-1, 3)

    # Undo vertical flip done by MuJoCo's OBJ loader.
    vertex_texcoords = vertex_texcoords.reshape(-1, 2)
    vertex_texcoords[:, 1] = 1 - vertex_texcoords[:, 1]

    return Msh(
        vertex_positions=vertex_positions,
        vertex_normals=vertex_normals,
        vertex_texcoords=vertex_texcoords,
        face_vertex_indices=face_vertex_indices,
    )


def msh_to_obj(msh_file: pathlib.Path) -> str:
  """Convert a legacy .msh file to the .obj format."""
  msh = Msh.create(msh_file)

  out = io.StringIO()
  for x, y, z in msh.vertex_positions:
    out.write(f"v {x} {y} {z}\n")
  for x, y, z in msh.vertex_normals:
    out.write(f"vn {x} {y} {z}\n")
  for u, v in msh.vertex_texcoords:
    out.write(f"vt {u} {v}\n")

  # Only reference optional vertex attributes that are present in the MSH.
  if msh.vertex_texcoords.size:
    vertex_format = "{0}/{0}/{0}" if msh.vertex_normals.size else "{0}/{0}"
  else:
    vertex_format = "{0}//{0}" if msh.vertex_normals.size else "{0}"
  for face in msh.face_vertex_indices:
    vertices = " ".join(vertex_format.format(i + 1) for i in face)
    out.write(f"f {vertices}\n")

  return out.getvalue()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-i", "--input", type=str, help="Path to the msh file.")
  parser.add_argument("-o", "--output", type=str, help="Path to the obj file.")
  args = parser.parse_args()
  with open(pathlib.Path(args.output), "w") as f:
    f.write(msh_to_obj(pathlib.Path(args.input)))
