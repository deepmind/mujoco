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
"""Tests for the Python resource provider API."""

import itertools

from absl.testing import absltest
import mujoco


BOX_OBJ = b"""\
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

XML_WITH_MESH = b"""\
<mujoco>
  <asset>
    <mesh name="box" file="PREFIX:box.obj"/>
  </asset>
  <worldbody>
    <body>
      <geom type="mesh" mesh="box"/>
    </body>
  </worldbody>
</mujoco>
"""

_PREFIXES = itertools.count()


def _next_prefix():
  return f"pytestasset{next(_PREFIXES)}"


def _make_provider(prefix, files, events=None):
  """Creates a provider that serves `files`, a map from name to bytes."""

  def open_callback(resource):
    if events is not None:
      events.append(("open", resource.name))
    if resource.name not in files:
      return False
    resource.data = files[resource.name]
    return True

  def read_callback(resource):
    if events is not None:
      events.append(("read", resource.name))
    return resource.data

  def close_callback(resource):
    if events is not None:
      events.append(("close", resource.name))

  provider = mujoco.mjpResourceProvider()
  provider.prefix = prefix
  provider.open = open_callback
  provider.read = read_callback
  provider.close = close_callback
  return provider


class ResourceProviderTest(absltest.TestCase):

  def test_register_and_find(self):
    prefix = _next_prefix()
    provider = _make_provider(prefix, {})

    slot = mujoco.mjp_registerProvider(provider)
    self.assertGreater(slot, 0)

    found = mujoco.mjp_findProvider(f"{prefix}:anything.obj")
    self.assertIsNotNone(found)
    self.assertEqual(found.prefix, prefix)
    self.assertIs(found.open, provider.open)
    self.assertIs(found.read, provider.read)
    self.assertIs(found.close, provider.close)

    # A resource name without a prefix, or with an unknown prefix, has no
    # provider.
    self.assertIsNone(mujoco.mjp_findProvider(prefix))
    self.assertIsNone(mujoco.mjp_findProvider("unknownprefix:file.obj"))

  def test_register_rejects_invalid_providers(self):
    # invalid URI scheme
    provider = _make_provider("not a scheme", {})
    with self.assertRaises(ValueError):
      mujoco.mjp_registerProvider(provider)

    provider = _make_provider("1scheme", {})
    with self.assertRaises(ValueError):
      mujoco.mjp_registerProvider(provider)

    # missing required callbacks
    provider = mujoco.mjpResourceProvider()
    provider.prefix = _next_prefix()
    with self.assertRaises(ValueError):
      mujoco.mjp_registerProvider(provider)

    # duplicate prefix
    prefix = _next_prefix()
    mujoco.mjp_registerProvider(_make_provider(prefix, {}))
    with self.assertRaises(ValueError):
      mujoco.mjp_registerProvider(_make_provider(prefix, {}))

  def test_default_provider(self):
    prefix = _next_prefix()
    provider = _make_provider(prefix, {})
    provider.mount = lambda resource: True
    provider.unmount = lambda resource: True
    provider.modified = lambda resource, timestamp: 0
    provider.write = lambda resource, data: len(data)

    mujoco.mjp_defaultProvider(provider)
    self.assertEqual(provider.prefix, "")
    self.assertIsNone(provider.open)
    self.assertIsNone(provider.read)
    self.assertIsNone(provider.close)
    self.assertIsNone(provider.mount)
    self.assertIsNone(provider.unmount)
    self.assertIsNone(provider.modified)
    self.assertIsNone(provider.write)

  def test_load_mesh_from_provider(self):
    prefix = _next_prefix()
    events = []
    provider = _make_provider(prefix, {f"{prefix}:box.obj": BOX_OBJ}, events)
    mujoco.mjp_registerProvider(provider)

    xml = XML_WITH_MESH.replace(b"PREFIX", prefix.encode())
    model = mujoco.MjModel.from_xml_string(xml)

    self.assertEqual(model.nmesh, 1)
    self.assertEqual(model.mesh_vertnum[0], 8)
    self.assertEqual(
        events,
        [
            ("open", f"{prefix}:box.obj"),
            ("read", f"{prefix}:box.obj"),
            ("close", f"{prefix}:box.obj"),
        ],
    )

  def test_resource_data_persists_between_callbacks(self):
    prefix = _next_prefix()
    events = []
    provider = _make_provider(prefix, {f"{prefix}:box.obj": BOX_OBJ}, events)

    def open_callback(resource):
      resource.data = (resource.name, "sentinel")
      return True

    def read_callback(resource):
      self.assertEqual(resource.data, (resource.name, "sentinel"))
      return BOX_OBJ

    def close_callback(resource):
      self.assertEqual(resource.data, (resource.name, "sentinel"))

    provider.open = open_callback
    provider.read = read_callback
    provider.close = close_callback
    mujoco.mjp_registerProvider(provider)

    xml = XML_WITH_MESH.replace(b"PREFIX", prefix.encode())
    mujoco.MjModel.from_xml_string(xml)

  def test_mount_and_unmount(self):
    prefix = _next_prefix()
    events = []
    provider = _make_provider(prefix, {f"{prefix}:box.obj": BOX_OBJ}, events)
    provider.mount = lambda resource: events.append(
        ("mount", resource.name)
    ) or True
    provider.unmount = lambda resource: events.append(
        ("unmount", resource.name)
    ) or True
    mujoco.mjp_registerProvider(provider)

    xml = XML_WITH_MESH.replace(b"PREFIX", prefix.encode())
    mujoco.MjModel.from_xml_string(xml)

    self.assertEqual(events[0], ("mount", prefix))
    self.assertEqual(events[-1], ("unmount", prefix))

  def test_open_failure_fails_load(self):
    prefix = _next_prefix()
    provider = _make_provider(prefix, {})
    mujoco.mjp_registerProvider(provider)

    xml = XML_WITH_MESH.replace(b"PREFIX", prefix.encode())
    with self.assertRaises(ValueError):
      mujoco.MjModel.from_xml_string(xml)


if __name__ == "__main__":
  absltest.main()
