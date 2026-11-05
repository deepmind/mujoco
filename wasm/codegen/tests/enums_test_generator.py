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

"""Generates TypeScript code that tests Mujoco enums."""

import textwrap

from introspect import enums as introspect_enums

from wasm.codegen.generators import common


def generate_typescript_enum_tests():
  """Generates TypeScript code that tests Mujoco enums."""
  output = textwrap.dedent("""\
    import 'jasmine';

    import { MainModule } from "../dist/mujoco"
    import loadMujoco from "../dist/mujoco.js"

    let mujoco: MainModule;

    describe('Enums', () => {
      beforeAll(async () => {
        mujoco = await loadMujoco();
      }, 60_000);
    """)

  for enum_decl in introspect_enums.ENUMS.values():
    enum_name = enum_decl.name
    output += f"""
  it('{enum_name} should exist', () => {{
    expect(mujoco.{enum_name}).toBeDefined();
  }});\n"""

  output += "});\n"
  return output


if __name__ == "__main__":
  ts_test_code = generate_typescript_enum_tests()
  output_file = 'wasm/tests/enums_test.ts'
  common.write_to_file(output_file, ts_test_code)
