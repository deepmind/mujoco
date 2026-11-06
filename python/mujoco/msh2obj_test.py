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
"""Tests for msh2obj.py."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np

import mujoco
from mujoco import msh2obj

_MESH_FIELDS = (
    "mesh_vertadr",
    "mesh_vertnum",
    "mesh_faceadr",
    "mesh_facenum",
    "mesh_bvhadr",
    "mesh_bvhnum",
    "mesh_normaladr",
    "mesh_normalnum",
    "mesh_texcoordadr",
    "mesh_texcoordnum",
    "mesh_graphadr",
    "mesh_vert",
    "mesh_normal",
    "mesh_face",
    "mesh_facenormal",
    "mesh_facetexcoord",
    "mesh_graph",
    "mesh_texcoord",
)

_XML = """
<mujoco>
  <asset>
    <mesh name="abdomen_1_body" file="abdomen_1_body.obj"/>
  </asset>
</mujoco>
"""


class MshTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("vertices_only", False, False, "f 1 3 2"),
      ("with_normals", True, False, "f 1//1 3//3 2//2"),
      ("with_texcoords", False, True, "f 1/1 3/3 2/2"),
      ("with_both", True, True, "f 1/1/1 3/3/3 2/2/2"),
  )
  def test_optional_vertex_attributes(
      self, has_normals, has_texcoords, expected_face
  ):
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    normals = np.array(
        [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    normals[0] /= np.sqrt(3)
    if not has_normals:
      normals = normals[:0]
    texcoords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    if not has_texcoords:
      texcoords = texcoords[:0]
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )
    header = np.array([4, len(normals), len(texcoords), 4], dtype=np.int32)
    with tempfile.TemporaryDirectory() as tmpdir:
      msh_path = epath.Path(tmpdir) / "tetra.msh"
      with msh_path.open("wb") as f:
        for array in (header, vertices, normals, texcoords, faces):
          f.write(array.tobytes())
      msh_bytes = msh_path.read_bytes()
      obj = msh2obj.msh_to_obj(msh_path)

    lines = obj.splitlines()
    self.assertLen(
        [line for line in lines if line.startswith("vn ")], len(normals)
    )
    self.assertLen(
        [line for line in lines if line.startswith("vt ")], len(texcoords)
    )
    self.assertEqual(
        [line for line in lines if line.startswith("f ")][0], expected_face
    )

    for extension, data in (
        ("msh", msh_bytes),
        ("obj", obj.encode()),
    ):
      filename = f"tetra.{extension}"
      model = mujoco.MjModel.from_xml_string(
          '<mujoco><asset><mesh name="tetra"'
          f' file="{filename}"/></asset></mujoco>',
          {filename: data},
      )
      self.assertEqual(model.nmesh, 1)
      self.assertEqual(model.mesh_vertnum[0], 4)
      self.assertEqual(model.mesh_facenum[0], 4)

  @parameterized.parameters(0, 1, 2, 3)
  def test_rejects_truncated_header(self, count) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      msh_path = epath.Path(tmpdir) / "truncated.msh"
      np.zeros(count, dtype=np.int32).tofile(msh_path)

      with self.assertRaisesRegex(
          ValueError,
          rf"Invalid MSH header: expected 4 counts, got {count}\.",
      ):
        msh2obj.Msh.create(msh_path)

  @parameterized.parameters(0, 1, 2, 3)
  def test_rejects_negative_count(self, count_index) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      msh_path = epath.Path(tmpdir) / "negative.msh"
      header = np.zeros(4, dtype=np.int32)
      header[count_index] = -1
      header.tofile(msh_path)

      with self.assertRaisesRegex(
          ValueError, "Invalid MSH header: counts must be nonnegative"
      ):
        msh2obj.Msh.create(msh_path)

  def test_reports_texcoord_count(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      msh_path = epath.Path(tmpdir) / "texcoords.msh"
      np.asarray([0, 0, 1, 0], dtype=np.int32).tofile(msh_path)

      with self.assertRaisesRegex(
          ValueError, r"Invalid number of texcoords: 0 != 2\*1\."
      ):
        msh2obj.Msh.create(msh_path)

  def test_obj_model_matches_msh_model(self) -> None:
    test_path = epath.resource_path("mujoco") / "testdata"

    msh_xml = test_path / "msh.xml"
    msh_model = mujoco.MjModel.from_xml_path(msh_xml.as_posix())

    msh_path = test_path / "abdomen_1_body.msh"
    obj = msh2obj.msh_to_obj(msh_path)

    obj_model = mujoco.MjModel.from_xml_string(
        _XML, {"abdomen_1_body.obj": obj.encode()}
    )

    for field in _MESH_FIELDS:
      np.testing.assert_allclose(
          getattr(msh_model, field),
          getattr(obj_model, field),
          atol=1e-6,
          err_msg=f"Field {field} does not match between msh and obj models.",
      )


if __name__ == "__main__":
  absltest.main()
