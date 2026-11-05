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
"""Tests that the API reference documentation is complete and up to date."""

import os
import re
import sys

_DOC_GEN = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../doc/generate')
)
sys.path.insert(0, _DOC_GEN)
try:
  import resource_loader  # pyrefly: ignore[missing-import]
  import generate_api_header  # pyrefly: ignore[missing-import]
  import generate_default_table  # pyrefly: ignore[missing-import]
  import generate_dmcontrol  # pyrefly: ignore[missing-import]
  import generate_functions  # pyrefly: ignore[missing-import]
  import generate_mjcf_map  # pyrefly: ignore[missing-import]
  import generate_mjcf_table  # pyrefly: ignore[missing-import]
  import generate_read_table  # pyrefly: ignore[missing-import]
  import generate_schema  # pyrefly: ignore[missing-import]
  import generate_units  # pyrefly: ignore[missing-import]
  import generate_xsd  # pyrefly: ignore[missing-import]
  import mjcf_schema  # pyrefly: ignore[missing-import]
except ImportError:
  raise

try:
  from absl.testing import absltest as googletest
except ImportError:
  import unittest as googletest  # pyrefly: ignore[missing-import]

# Functions in headers that are intentionally not in functions.rst.
_FUNCTIONS_TO_SKIP = set()

# Types (STRUCT/ENUM) in headers that are intentionally not in APItypes.rst.
_TYPES_TO_SKIP = set()

# Type names documented in APItypes.rst that aren't STRUCT/ENUM in headers.
# These are typedefs, callbacks, and scalar types.
_EXTRA_DOCUMENTED_TYPES = {
    # scalar typedefs
    'mjtByte',
    'mjtBool',
    'mjtNum',
    'mjtSize',
    # C++ type aliases
    'mjByteVec',
    'mjDoubleVec',
    'mjFloatVec',
    'mjFloatVecVec',
    'mjIntVec',
    'mjIntVecVec',
    'mjString',
    'mjStringVec',
    # function pointer typedefs (callbacks)
    'mjfAct',
    'mjfBatchFunc',
    'mjfCanDecode',
    'mjfCloseResource',
    'mjfCollision',
    'mjfConFilt',
    'mjfDecode',
    'mjfEncode',
    'mjfGeneric',
    'mjfGetResourceDir',
    'mjfItemEnable',
    'mjfLogHandler',
    'mjfOpenResource',
    'mjfReadResource',
    'mjfResourceModified',
    'mjfSensor',
    'mjfTime',
}


def _get_path(*path_parts: str) -> str:
  """Returns absolute path for a repository-relative path."""
  rel_path = '/'.join(path_parts)
  return str(resource_loader.resolve_path(rel_path))


def _check_up_to_date(test_case, rel_path, generated_content):
  """Checks that a generated file matches the checked-in version."""
  path = _get_path(*rel_path.split('/'))
  with open(path, 'r', encoding='utf-8') as file:
    current = file.read()
  if generated_content != current:
    filename = os.path.basename(rel_path)
    test_case.fail(f"The file '{filename}' needs to be updated.")


class DocTest(googletest.TestCase):

  def test_api_header(self):
    """Checks that references.h matches the generated output."""
    source = generate_api_header.generate_reference_header(
        generate_api_header.read_headers()
    )
    _check_up_to_date(self, 'doc/includes/references.h', source)

  def test_mjcf_table(self):
    """Checks that mjcf_table.inc matches the schema-generated output."""
    _check_up_to_date(
        self,
        'src/xml/generated/mjcf_table.inc',
        generate_mjcf_table.generate(),
    )

  def test_default_table(self):
    """Checks that mjcf_default_table.inc matches the schema-generated output."""
    _check_up_to_date(
        self,
        'src/xml/generated/mjcf_default_table.inc',
        generate_default_table.generate(),
    )

  def test_mjcf_map(self):
    """Checks that mjcf_map.h matches the schema-generated output."""
    _check_up_to_date(
        self, 'src/xml/generated/mjcf_map.h', generate_mjcf_map.generate()
    )

  def test_read_table(self):
    """Checks that mjcf_read_table.inc matches the schema-generated output."""
    _check_up_to_date(
        self,
        'src/xml/generated/mjcf_read_table.inc',
        generate_read_table.generate(),
    )

  def test_xsd(self):
    """Checks that mjcf.xsd matches the schema-generated output."""
    _check_up_to_date(
        self,
        'src/xml/generated/mjcf.xsd',
        generate_xsd.generate(),
    )

  def test_dmcontrol_schema(self):
    """Checks that dmcontrol_schema.xml matches the schema-generated output."""
    _check_up_to_date(
        self,
        'src/xml/generated/dmcontrol_schema.xml',
        generate_dmcontrol.generate(),
    )

  def test_read_table_consumed(self):
    """Checks that every generated row array is consumed, and none is stale.

    Elements are auto-included in the read table unless excluded by
    NOT_TABLE_DRIVEN, so a new bound element whose OneX() reader was never
    migrated would get rows that nothing reads; unused inline constexpr
    arrays do not even warn. Conversely, a stale NOT_TABLE_DRIVEN entry
    (e.g. after an element rename) silently stops excluding.
    """
    schema = mjcf_schema.parse_file(_get_path('src/xml/mjcf.schema'))
    sources = ''
    for filename in ('xml_native_reader.cc', 'xml_native_writer.cc'):
      path = _get_path('src/xml', filename)
      with open(path, 'r', encoding='utf-8') as file:
        sources += file.read()
    errors = []
    dispatched = set(generate_read_table.SENSOR_DISPATCH)
    dispatched |= set(
        schema.enums[generate_read_table.ACTUATOR_DISPATCH_ENUM].keywords()
    )
    for name in generate_read_table.table_driven_elements(schema):
      if name in dispatched:
        continue  # consumed through kSensorDispatch / kActuatorDispatch
      array = generate_read_table.array_name(name)
      if array not in sources:
        errors.append(
            f"array '{array}' for element '{name}' is generated but never "
            'consumed: migrate its reader to ReadAttrTable or add the '
            'element to NOT_TABLE_DRIVEN'
        )
    for dispatch in ('kSensorDispatch', 'kActuatorDispatch'):
      if dispatch not in sources:
        errors.append(f"'{dispatch}' is never consumed")
    for _, array in generate_read_table.EMIT_GROUPS.values():
      if array not in sources:
        errors.append(
            f"group array '{array}' is generated but never consumed"
        )
    stale = generate_read_table.NOT_TABLE_DRIVEN - set(schema.elements)
    for name in sorted(stale):
      errors.append(
          f"NOT_TABLE_DRIVEN entry '{name}' does not name a schema element"
      )
    if errors:
      self.fail('read-table coverage:\n' + '\n'.join(errors))

  def test_flag_defaults(self):
    """Checks option flag defaults against the disable/enable bit enums.

    The flag element is unbound (its attributes lower to bits of
    disableflags/enableflags), so SchemaDefaultsTest cannot verify its
    declared defaults. The rule is mechanical: a disable bit is on by
    default, so its flag defaults to 'enable'; an enable bit defaults
    to 'disable'.
    """
    with open(_get_path('include', 'mujoco', 'mjtype.h'),
              encoding='utf-8') as file:
      content = file.read()
    dsbl = set(re.findall(r'mjDSBL_(\w+)', content))
    enbl = set(re.findall(r'mjENBL_(\w+)', content))
    schema = mjcf_schema.parse_file(_get_path('src', 'xml', 'mjcf.schema'))
    errors = []
    for attr in schema.expanded_attrs(schema.elements['flag']):
      name = attr.name.upper()
      if name in dsbl:
        expected = 'enable'
      elif name in enbl:
        expected = 'disable'
      else:
        errors.append(f'flag.{attr.name}: no mjDSBL_/mjENBL_ member')
        continue
      if attr.default != expected:
        errors.append(f'flag.{attr.name}: declared {attr.default!r}, '
                      f'but the bit implies {expected!r}')
    if errors:
      self.fail('flag defaults:\n' + '\n'.join(errors))

  def test_schema_enum_coverage(self):
    """Checks schema enums against the C enums they bind.

    Every schema constant must be a member of the bound C enum, and every C
    member must be a schema keyword, a count sentinel (mjN*), or a documented
    exemption -- so adding a C enum member without updating the schema fails
    here.
    """
    # C members deliberately not exposed as XML keywords
    exempt = {
        'bodysleep': {  # resolved states of 'auto', not settable
            'mjSLEEP_AUTO_ALLOWED',
            'mjSLEEP_AUTO_NEVER',
        },
        'geomtype': {  # rendering-only types and the missing-geom sentinel
            'mjGEOM_ARROW',
            'mjGEOM_ARROW1',
            'mjGEOM_ARROW2',
            'mjGEOM_LINE',
            'mjGEOM_LINEBOX',
            'mjGEOM_FLEX',
            'mjGEOM_SKIN',
            'mjGEOM_LABEL',
            'mjGEOM_TRIANGLE',
            'mjGEOM_NONE',
        },
        'texrole': {  # not settable from XML
            'mjTEXROLE_USER'
        },
    }
    # deliberately partial: keywords are a documented subset of the C enum
    # (inputbit: combinable tokens exclude the whole-attribute keyword
    # mjINPUT_NONE; inputkeyword: the whole-attribute keyword excludes the tokens)
    partial = {'frameobj', 'inputbit', 'inputkeyword'}

    enums_c = {}
    for name in ('mjtype.h', 'mjspec.h'):
      path = _get_path('include', 'mujoco', name)
      with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
      for m in re.finditer(
          r'typedef enum (mjt\w+)\s*\{(.*?)\}\s*\1;', content, re.S
      ):
        enums_c[m.group(1)] = re.findall(
            r'^\s*(mj[A-Z]\w+)', m.group(2), re.M
        )

    schema_path = _get_path('src/xml/mjcf.schema')
    schema = mjcf_schema.parse_file(schema_path)
    errors = []
    for name, enum in schema.enums.items():
      if not enum.ctype:
        continue
      if enum.ctype not in enums_c:
        errors.append(f'  {name}: C enum {enum.ctype} not found in headers')
        continue
      members = set(enums_c[enum.ctype])
      constants = {value for _, value in enum.items}
      for bad in sorted(constants - members):
        errors.append(f'  {name}: {bad} is not a member of {enum.ctype}')
      if name in partial:
        continue
      uncovered = {
          m
          for m in members - constants
          if not re.search(r'^mjN[A-Z]', m)
      } - exempt.get(name, set())
      for miss in sorted(uncovered):
        errors.append(
            f'  {name}: {enum.ctype} member {miss} has no keyword (add it to'
            ' the schema or to the exemptions here)'
        )
    if errors:
      self.fail('schema enum coverage:\n' + '\n'.join(errors))

  def test_schema(self):
    """Checks that XMLschema.rst matches the generated output."""
    _check_up_to_date(self, 'doc/XMLschema.rst', generate_schema.generate())

  def test_units(self):
    """Checks that XMLunits.rst matches the schema-generated output."""
    _check_up_to_date(self, 'doc/XMLunits.rst', generate_units.generate())

  def test_dims(self):
    """Checks that every real-valued attribute declares its dimension."""
    schema = mjcf_schema.parse_file(_get_path('src', 'xml', 'mjcf.schema'))
    missing = [f'{scope}/{attr.name} (line {attr.line})'
               for scope, attr in mjcf_schema.missing_dims(schema)]
    if missing:
      self.fail('attributes without a dim facet:\n  ' + '\n  '.join(missing))

  def test_functions(self):
    """Checks that functions.rst matches the generated output."""
    _check_up_to_date(
        self,
        'doc/APIreference/functions.rst',
        generate_functions.generate(),
    )

  def test_all_functions_included(self):
    """Checks that every public C function has an entry in functions.rst."""
    functions_file = _get_path('doc/APIreference/functions.rst')
    with open(functions_file, 'r', encoding='utf-8') as file:
      content = file.read()

    documented = set(
        re.findall(r'^\.\. _(mj[a-zA-Z0-9_]+):', content, flags=re.MULTILINE)
    )

    api = generate_api_header.read_headers()
    header_funcs = {
        token for token, d in api.items() if d.c_type == 'FUNCTION'
    }

    errors = []
    for token in sorted(header_funcs - documented - _FUNCTIONS_TO_SKIP):
      d = api[token]
      errors.append(f'  undocumented: {token} (section: {d.section!r})')
    for token in sorted(documented - header_funcs):
      errors.append(f'  stale: {token} (in functions.rst but not in headers)')

    if errors:
      msg = 'functions.rst mismatches:\n' + '\n'.join(errors)
      self.fail(msg)

  def test_all_types_included(self):
    """Checks that every public struct and enum has an entry in APItypes.rst."""
    types_file = _get_path('doc/APIreference/APItypes.rst')
    with open(types_file, 'r', encoding='utf-8') as file:
      content = file.read()

    documented = set(
        re.findall(r'^\.\. _(mj[a-zA-Z0-9_]+):', content, flags=re.MULTILINE)
    )

    api = generate_api_header.read_headers()
    header_types = {
        token for token, d in api.items() if d.c_type in ('STRUCT', 'ENUM')
    }

    errors = []
    for token in sorted(header_types - documented - _TYPES_TO_SKIP):
      d = api[token]
      errors.append(
          f'  undocumented: {token} ({d.c_type}, section: {d.section!r})'
      )
    for token in sorted(documented - header_types - _EXTRA_DOCUMENTED_TYPES):
      errors.append(f'  stale: {token} (in APItypes.rst but not in headers)')

    if errors:
      msg = 'APItypes.rst mismatches:\n' + '\n'.join(errors)
      self.fail(msg)

  def test_mjdata_xmacro_fields(self):
    """Checks that every mjData field appears in the mjData X macros."""
    with open(
        _get_path('include', 'mujoco', 'mjdata.h'), encoding='utf-8'
    ) as f:
      header = f.read()
    with open(
        _get_path('include', 'mujoco', 'mjxmacro.h'), encoding='utf-8'
    ) as f:
      xmacro = f.read()

    # fields of struct mjData_, one declaration per line
    body = re.search(
        r'typedef struct mjData_ \{(.*?)\n\} mjData;', header, re.S
    ).group(1)
    fields = []
    for line in body.split('\n'):
      line = line.split('//')[0].strip()
      match = re.match(r'^[\w ]+?[\s\*]+(\w+)(\[[^\]]*\])*;$', line)
      if match:
        fields.append(match.group(1))
    self.assertGreater(len(fields), 100)

    # the second argument of every X entry in the MJDATA_* macros
    xmacro_fields = set()
    for macro in re.finditer(
        r'#define MJDATA_\w*((?:[^\n]*\\\n)*[^\n]*)', xmacro
    ):
      for entry in re.finditer(r'\bX\w*\s*\(\s*[^,]+,\s*(\w+)', macro.group(1)):
        xmacro_fields.add(entry.group(1))

    # buffer and arena are the allocations that the pointer fields index into,
    # signature is the model's compilation signature: not simulation data.
    # threadlock is an internal runtime flag for thread dispatch.
    excluded = {'buffer', 'arena', 'signature', 'threadlock'}
    missing = [
        f for f in fields if f not in xmacro_fields and f not in excluded
    ]
    self.assertEqual(
        missing, [], 'mjData fields missing from the X macros in mjxmacro.h'
    )
    unknown = sorted(xmacro_fields - set(fields))
    self.assertEqual(unknown, [], 'X macro entries that are not mjData fields')

  def test_element_constraints_diamond_inheritance(self):
    con = mjcf_schema.Constraint(
        kind='exclusive', bundles=(('a',), ('b',)), doc=None, line=1
    )
    common_group = mjcf_schema.Group(
        name='common', variant=False, members=[con], doc=None, line=1
    )
    group1 = mjcf_schema.Group(
        name='group1',
        variant=False,
        members=[mjcf_schema.Use(group='common', line=1)],
        doc=None,
        line=1,
    )
    group2 = mjcf_schema.Group(
        name='group2',
        variant=False,
        members=[mjcf_schema.Use(group='common', line=1)],
        doc=None,
        line=1,
    )
    elem = mjcf_schema.Element(
        name='elem',
        spec=None,
        facets={},
        members=[
            mjcf_schema.Use(group='group1', line=1),
            mjcf_schema.Use(group='group2', line=1),
        ],
        doc=None,
        line=1,
    )
    schema = mjcf_schema.Schema(
        enums={},
        groups={'common': common_group, 'group1': group1, 'group2': group2},
        elements={'elem': elem},
        path='<test>',
    )
    cons = generate_mjcf_table._element_constraints(schema, elem)
    self.assertEqual(len(cons), 1)  # pylint: disable=g-generic-assert


if __name__ == '__main__':
  googletest.main()
