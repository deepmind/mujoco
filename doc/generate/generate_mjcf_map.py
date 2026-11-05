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
"""Generates the keyword-map header from src/xml/mjcf.schema.

Emits src/xml/generated/mjcf_map.h: one mjMap array and size constant per schema
enum, as C++17 inline variables, so the reader, the writer and the
generated tables share one definition with no extern declarations to
maintain. It is checked in and gated by test/doc/doc_test.py, which
regenerates it from the schema and diffs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
  import resource_loader  # pyrefly: ignore[missing-import]
  import mjcf_schema  # pyrefly: ignore[missing-import]
except ImportError:
  raise

SCHEMA_PATH = str(resource_loader.resolve_path('src/xml/mjcf.schema'))

if resource_loader.is_monorepo():
  _GUARD = 'THIRD_PARTY_MUJOCO_SRC_XML_GENERATED_MJCF_MAP_H_'
  _INCLUDES = '''\
#include "third_party/mujoco/include/mjspec.h"
#include "third_party/mujoco/include/mjtype.h"
#include "third_party/mujoco/src/user/user_flexcomp.h"
#include "third_party/mujoco/src/xml/xml_util.h"'''
else:
  _GUARD = 'MUJOCO_SRC_XML_GENERATED_MJCF_MAP_H_'
  _INCLUDES = '''\
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include "user/user_flexcomp.h"
#include "xml/xml_util.h"'''

_HEADER = f'''\
// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// GENERATED FILE, DO NOT EDIT. Generated from src/xml/mjcf.schema by
// doc/generate/generate_mjcf_map.py; test/doc/doc_test.py checks freshness.
//
// Keyword maps, one per schema enum, shared by the reader, the writer and
// the generated tables. Inline variables: including this header is all a
// translation unit needs.

#ifndef HEADER_GUARD_PLACEHOLDER
#define HEADER_GUARD_PLACEHOLDER

{_INCLUDES}

// clang-format off

// keywords of the built-in bool type (not a schema enum)
inline constexpr mjMap bool_map[] = {{
  {{"false",   0}},
  {{"true",    1}},
}};

'''

_FOOTER = '''\
// clang-format on

#endif  // HEADER_GUARD_PLACEHOLDER
'''


def generate() -> str:
  """Generates the mjcf_map.h content as a string."""
  schema = mjcf_schema.parse_file(SCHEMA_PATH)
  out = []
  for enum in schema.enums.values():
    width = max(len(key) for key in enum.keywords()) + 3
    out.append(f'// enum {enum.name}')
    out.append(f'inline constexpr mjMap {enum.name}_map[] = {{')
    for key, value in enum.items:
      padded = f'"{key}",'.ljust(width + 1)
      out.append(f'  {{{padded} {value}}},')
    out.append('};')
    out.append(f'inline constexpr int {enum.name}_sz = {len(enum.items)};')
    out.append('')
  header = _HEADER.replace('HEADER_GUARD_PLACEHOLDER', _GUARD)
  footer = _FOOTER.replace('HEADER_GUARD_PLACEHOLDER', _GUARD)
  return header + '\n'.join(out) + footer


def main() -> int:
  if len(sys.argv) > 2:
    sys.exit('usage: generate_mjcf_map.py [output.h]')
  text = generate()
  if len(sys.argv) == 2:
    with open(sys.argv[1], 'w', encoding='utf-8') as file:
      file.write(text)
  else:
    sys.stdout.write(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
