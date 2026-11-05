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
"""Tests for enums.py."""

from absl.testing import absltest

from . import enums


class EnumsTest(absltest.TestCase):

  # "simple" enum that just starts at zero and increment by one.
  def test_mjtJoint(self):  # pylint: disable=invalid-name
    enum_decl = enums.ENUMS['mjtJoint']
    self.assertEqual(enum_decl.name, 'mjtJoint')
    self.assertEqual(enum_decl.declname, 'enum mjtJoint')
    self.assertEqual(
        tuple(enum_decl.values.items()), (('mjJNT_FREE', 0),
                                          ('mjJNT_BALL', 1),
                                          ('mjJNT_SLIDE', 2),
                                          ('mjJNT_HINGE', 3)))

  # all values explicitly specified
  def test_mjtEnableBit(self):  # pylint: disable=invalid-name
    enum_decl = enums.ENUMS['mjtEnableBit']
    self.assertEqual(enum_decl.name, 'mjtEnableBit')
    self.assertEqual(enum_decl.declname, 'enum mjtEnableBit')
    self.assertEqual(
        tuple(enum_decl.values.items()), (('mjENBL_OVERRIDE', 1<<0),
                                          ('mjENBL_ENERGY', 1<<1),
                                          ('mjENBL_FWDINV', 1<<2),
                                          ('mjENBL_INVDISCRETE', 1<<3),
                                          ('mjENBL_SLEEP', 1<<4),
                                          ('mjENBL_DIAGEXACT', 1<<5),
                                          ('mjENBL_IPC', 1<<6),
                                          ('mjENBL_SPECULATIVE', 1<<7),
                                          ('mjNENABLE', 8)))

  # values mostly increment by one with occasional overrides
  def test_mjtGeom(self):  # pylint: disable=invalid-name
    enum_decl = enums.ENUMS['mjtGeom']
    self.assertEqual(enum_decl.name, 'mjtGeom')
    self.assertEqual(enum_decl.declname, 'enum mjtGeom')

    self.assertEqual(enum_decl.values['mjGEOM_PLANE'], 0)
    self.assertEqual(enum_decl.values['mjGEOM_HFIELD'], 1)
    self.assertEqual(enum_decl.values['mjGEOM_SPHERE'], 2)
    # Skip a few...

    self.assertEqual(enum_decl.values['mjGEOM_ARROW'], 100)
    self.assertEqual(enum_decl.values['mjGEOM_ARROW1'], 101)
    self.assertEqual(enum_decl.values['mjGEOM_ARROW2'], 102)
    self.assertEqual(enum_decl.values['mjGEOM_TRIANGLE'], 108)
    # Skip a few...

    self.assertEqual(enum_decl.values['mjGEOM_NONE'], 1001)

if __name__ == '__main__':
  absltest.main()
