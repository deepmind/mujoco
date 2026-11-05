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
"""Tests for structs.py."""

from absl.testing import absltest

from . import ast_nodes
from . import structs
from . import type_parsing


class StructsTest(absltest.TestCase):

  def test_mjData(self):  # pylint: disable=invalid-name
    struct_decl = structs.STRUCTS['mjData']
    self.assertEqual(struct_decl.name, 'mjData')
    self.assertEqual(struct_decl.declname, 'struct mjData_')

    field_names = set()
    for field in struct_decl.fields:
      self.assertNotIn(field.name, field_names)
      field_names.add(field.name)
      if field.name == 'warning':
        self.assertEqual(field.type,
                         type_parsing.parse_type('mjWarningStat[7]'))
        self.assertEqual(field.doc, 'warning statistics (mutable)')
      elif field.name == 'qpos':
        self.assertEqual(field.type, type_parsing.parse_type('mjtNum*'))
        self.assertEqual(field.doc, 'position')
        self.assertEqual(field.array_extent, ('nq',))

    self.assertIn('warning', field_names)
    self.assertIn('qpos', field_names)

  def test_mjVisual(self):  # pylint: disable=invalid-name
    struct_decl = structs.STRUCTS['mjVisual']
    self.assertEqual(struct_decl.name, 'mjVisual')
    self.assertEqual(struct_decl.declname, 'struct mjVisual_')

    outer_fields = set()
    for outer_field in struct_decl.fields:
      self.assertNotIn(outer_field.name, outer_fields)
      outer_fields.add(outer_field.name)
      self.assertIsInstance(outer_field.type, ast_nodes.AnonymousStructDecl)
      inner_fields = set()
      if outer_field.name == 'global':
        for inner_field in outer_field.type.fields:
          self.assertNotIn(inner_field.name, inner_fields)
          inner_fields.add(inner_field.name)
          if inner_field.name == 'ipd':
            self.assertEqual(inner_field.type, type_parsing.parse_type('float'))
            self.assertEqual(
                inner_field.doc, 'inter-pupilary distance for free camera'
            )
          elif inner_field.name == 'offwidth':
            self.assertEqual(inner_field.type, type_parsing.parse_type('int'))
            self.assertEqual(inner_field.doc, 'width of offscreen buffer')
        self.assertIn('ipd', inner_fields)
        self.assertIn('offwidth', inner_fields)
      elif outer_field.name == 'headlight':
        for inner_field in outer_field.type.fields:
          self.assertNotIn(inner_field.name, inner_fields)
          inner_fields.add(inner_field.name)
          if inner_field.name in {'ambient', 'diffuse', 'specular'}:
            self.assertEqual(inner_field.type,
                             type_parsing.parse_type('float[3]'))
            self.assertEqual(inner_field.doc,
                             f'{inner_field.name} rgb (alpha=1)')
          elif inner_field.name == 'active':
            self.assertEqual(inner_field.type, type_parsing.parse_type('int'))
            self.assertEqual(
                inner_field.doc, 'is headlight active (0: no, 1: yes, 2: auto)'
            )
        self.assertIn('ambient', inner_fields)
        self.assertIn('diffuse', inner_fields)
        self.assertIn('specular', inner_fields)
        self.assertIn('active', inner_fields)

    self.assertIn('global', outer_fields)
    self.assertIn('headlight', outer_fields)

  def test_mjuiItem(self):  # pylint: disable=invalid-name
    struct_decl = structs.STRUCTS['mjuiItem']
    self.assertEqual(struct_decl.name, 'mjuiItem')
    self.assertEqual(struct_decl.declname, 'struct mjuiItem_')

    found_anonymous_union = False
    outer_fields = set()
    for outer_field in struct_decl.fields:
      if isinstance(outer_field, ast_nodes.AnonymousUnionDecl):
        self.assertFalse(found_anonymous_union)
        found_anonymous_union = True
        inner_fields = set()
        for inner_field in outer_field.fields:
          self.assertNotIn(inner_field.name, inner_fields)
          inner_fields.add(inner_field.name)
          if inner_field.name == 'single':
            self.assertEqual(inner_field.type,
                             type_parsing.parse_type('struct mjuiItemSingle_'))
            self.assertEqual(inner_field.doc, 'check and button')
          elif inner_field.name == 'multi':
            self.assertEqual(inner_field.type,
                             type_parsing.parse_type('struct mjuiItemMulti_'))
            self.assertEqual(inner_field.doc, 'static, radio and select')
        self.assertIn('single', inner_fields)
        self.assertIn('multi', inner_fields)
      else:
        self.assertNotIn(outer_field.name, outer_fields)
        outer_fields.add(outer_field.name)
        if outer_field.name == 'pdata':
          self.assertEqual(outer_field.type, type_parsing.parse_type('void*'))
          self.assertEqual(outer_field.doc, 'data pointer (type-specific)')

    self.assertTrue(found_anonymous_union)
    self.assertIn('pdata', outer_fields)


if __name__ == '__main__':
  absltest.main()
