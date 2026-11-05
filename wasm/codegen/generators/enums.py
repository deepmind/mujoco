# Copyright 2025 DeepMind Technologies Limited
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

"""Generates Embind bindings for MuJoCo enums."""

from codegen import code_builder
from introspect import ast_nodes


def generate(
    enums: list[ast_nodes.EnumDecl],
) -> list[tuple[str, list[str]]]:
  """Generates all Embind code for the provided enums."""

  builder = code_builder.CodeBuilder()
  for e in sorted(enums, key=lambda e: e.name):
    if e.values:  # Skip empty enums.
      with builder.block(f'enum_<{e.name}>("{e.name}")', braces=False):
        names = list(e.values.keys())
        for name in names[:-1]:
          builder.line(f'.value("{name}", {name})')
        builder.line(f'.value("{names[-1]}", {names[-1]});')

  content = builder.to_string()
  marker = '// {{ ENUM_BINDINGS }}'
  return [(marker, [content])]
