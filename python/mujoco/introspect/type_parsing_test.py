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
"""Tests for type_parsing.py."""

from absl.testing import absltest

from . import ast_nodes
from . import type_parsing


class TypeParsingTest(absltest.TestCase):

  def test_parse_complex_type(self):
    parsed_type = type_parsing.parse_type(
        'int unsigned volatile long const long'+
        '(**const(*const restrict*[9])[7])[3][4]')

    expected_type = ast_nodes.ArrayType(
        extents=[9],
        inner_type=ast_nodes.PointerType(
            ast_nodes.PointerType(
                is_const=True,
                is_restrict=True,
                inner_type=ast_nodes.ArrayType(
                    extents=[7],
                    inner_type=ast_nodes.PointerType(
                        is_const=True,
                        inner_type=ast_nodes.PointerType(
                            ast_nodes.ArrayType(
                                extents=(3, 4),
                                inner_type=ast_nodes.ValueType(
                                    'int unsigned long long',
                                    is_const=True, is_volatile=True)
                                )
                            )
                        )
                    )
                )
            )
        )

    self.assertEqual(parsed_type, expected_type)

  def test_duplicate_qualifier(self):
    with self.assertRaises(ValueError):
      type_parsing.parse_type('int const const')

  def test_invalid_pointer_qualifier(self):
    with self.assertRaises(ValueError):
      type_parsing.parse_type('int *const const')

  def test_void_pointer_to_function(self):
    parsed = type_parsing.parse_type('void *(*)(void *)')
    self.assertEqual(parsed, ast_nodes.ValueType('void *(*)(void *)'))

  def test_unbalanced_parens(self):
    with self.assertRaises(ValueError):
      type_parsing.parse_type('int (*)(')

  def test_parse_function_return_type(self):
    parsed = type_parsing.parse_function_return_type('int(int)')
    self.assertEqual(parsed, ast_nodes.ValueType('int'))


if __name__ == '__main__':
  absltest.main()