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
"""Generates keyword maps and typed attribute-read rows from mjcf.schema.

Emits src/xml/generated/mjcf_read_table.inc: per-element mjXAttr row arrays
consumed by mjXReader::ReadAttrTable, plus the kSensorDispatch tag table and
shared group arrays.

Which elements get rows is determined automatically: every schema element with
a bound spec struct and at least one table-drivable attribute is included,
unless it appears in NOT_TABLE_DRIVEN (elements whose OneX() readers have
custom logic). A coverage check in doc_test verifies that every emitted row
array is consumed by the reader or writer, and that NOT_TABLE_DRIVEN names
only elements that exist in the schema.

Field offsets are emitted as offsetof() expressions, so binding mistakes are
compile errors, and the field's C type (parsed from mjspec.h) selects the row
kind, so mjtNum vs double is decided by the struct, not by the schema.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
  import resource_loader  # pyrefly: ignore[missing-import]
  import mjcf_schema  # pyrefly: ignore[missing-import]
except ImportError:
  raise

SCHEMA_PATH = str(resource_loader.resolve_path('src/xml/mjcf.schema'))
SPEC_H_PATH = str(resource_loader.resolve_path('include/mujoco/mjspec.h'))
MODEL_H_PATH = str(resource_loader.resolve_path('include/mujoco/mjmodel.h'))

# elements with a bound spec whose OneX() reader is NOT table-driven;
# their rows are not emitted. These are the genuinely irregular elements
# whose readers have custom logic that can't be factored into ReadAttrTable.
# If you add a new schema element with a bound spec, either migrate its
# reader to ReadAttrTable (it will be auto-included) or add it here.
NOT_TABLE_DRIVEN = {
    # plugin actuator: shared rows + plugin config
    'actuator_plugin',
    # equality subtypes: read shared equality_base + per-type refs
    'connect', 'weld', 'equality_joint', 'equality_tendon',
    'equality_flex', 'flexstrain',
    # sensors with custom reading logic
    'rangefinder', 'distance', 'user', 'normal', 'fromto',
    'sensor_contact', 'sensor_plugin', 'tactile',
    # other irregulars
    'frame', 'plugin', 'numeric', 'text', 'tuple',
}

# actuator tags whose mechanical rows are collected into kActuatorDispatch,
# keyed by the actuatortype enum; shortcut parameters are reading=custom
ACTUATOR_DISPATCH_ENUM = 'actuatortype'

# sensors whose whole branch derives from the schema (identity constants +
# references); their arrays are also collected into kSensorDispatch
SENSOR_DISPATCH = [
    'touch',
    'accelerometer',
    'velocimeter',
    'gyro',
    'force',
    'torque',
    'magnetometer',
    'camprojection',
    'jointpos',
    'jointvel',
    'tendonpos',
    'tendonvel',
    'actuatorpos',
    'actuatorvel',
    'actuatorfrc',
    'jointactuatorfrc',
    'tendonactuatorfrc',
    'ballquat',
    'ballangvel',
    'jointlimitpos',
    'jointlimitvel',
    'jointlimitfrc',
    'tendonlimitpos',
    'tendonlimitvel',
    'tendonlimitfrc',
    'framepos',
    'framequat',
    'framexaxis',
    'frameyaxis',
    'framezaxis',
    'framelinvel',
    'frameangvel',
    'framelinacc',
    'frameangacc',
    'insidesite',
    'subtreecom',
    'subtreelinvel',
    'subtreeangmom',
    'e_potential',
    'e_kinetic',
    'clock',
]

# shared groups emitted as standalone row arrays, for shared OneX() readers:
# group name -> (bound struct, emitted array name)
EMIT_GROUPS = {
    'equality_base': ('mjsEquality', 'kEqualityBaseAttrs'),
    'sensor_base': ('mjsSensor', 'kSensorBaseAttrs'),
}

# groups whose attributes are hand-read in the OneX() remnant;
# their rows are not emitted
HAND_GROUPS = ['orientation', 'transmission',
               'sensor_base']  # read via kSensorBaseAttrs before dispatch

# mjspec.h array-dimension expressions equivalent to a numeric schema bound;
# the expression is emitted as the row length so it tracks the header
DIM_EQUIV = {'mjNPOLY+1': '3'}

# schema scalar/C-type -> mjXAttr kind
KIND_BY_CTYPE = {'mjString*': 'kString', 'int': 'kInt', 'double': 'kDouble',
                 'mjtNum': 'kNum', 'float': 'kFloat'}

_HEADER = '''\
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
// doc/generate/generate_read_table.py; test/doc/doc_test.py checks freshness.
//
// Typed attribute rows for table-driven reading (mjXReader::ReadAttrTable),
// one array per migrated element. Rows are {attr, kind, len, exact,
// required, nodefault, handwrite, offset}; keyword maps are defined in
// mjcf_map.h.

// clang-format off
'''


def array_name(element_name):
  """Return the row-array name emitted for an element."""
  return f'k{element_name.capitalize()}Attrs'


def _has_table_attrs(schema, element):
  """Check if the element has at least one non-custom table-drivable attr."""
  return any('reading' not in attr.facets
             for attr in schema.expanded_attrs(element))


def table_driven_elements(schema):
  """Return the list of element names whose readers are table-driven.

  An element is table-driven if it has a bound spec struct, at least one
  table-drivable attribute, and is not in NOT_TABLE_DRIVEN.

  Args:
    schema: Parsed schema object.
  """
  result = []
  for name, element in schema.elements.items():
    if not element.spec:
      continue
    if not _has_table_attrs(schema, element):
      continue
    if name in NOT_TABLE_DRIVEN:
      continue
    result.append(name)
  return result


def parse_spec_structs(*paths):
  """Parse struct layouts from C headers.

  Args:
    *paths: Paths to C header files to parse.

  Returns:
    A dict mapping struct name -> {field: (ctype, dim-or-None)}.
    Nested anonymous sub-structs (the mjVisual sections) are exposed as
    'Outer.subname' pseudo-structs.
  """
  structs = {}
  field_re = re.compile(
      r'^\s*([\w<>*]+(?:\s*\*)?)\s+(\w+)(\[([^\]]+)\])?\s*;', re.M)
  for path in paths:
    with open(path, encoding='utf-8') as f:
      text = f.read()
    for m in re.finditer(r'typedef struct (mj\w+)_ \{(.*?)\n\} \1;', text,
                         re.S):
      name, body = m.group(1), m.group(2)
      for sm in re.finditer(r'struct \{(.*?)\} (\w+);', body, re.S):
        sub_fields = {}
        for fm in field_re.finditer(sm.group(1)):
          ctype, fname, _, dim = fm.groups()
          sub_fields[fname] = (ctype.replace(' ', ''), dim)
        structs[f'{name}.{sm.group(2)}'] = sub_fields
      body = re.sub(r'struct \{.*?\} \w+;', '', body, flags=re.S)
      fields = {}
      for fm in field_re.finditer(body):
        ctype, fname, _, dim = fm.groups()
        fields[fname] = (ctype.replace(' ', ''), dim)
      structs[name] = fields
  return structs


def _attr_row(spec, fields, attr, ctx, prefix=''):
  """Return a row tuple for one attribute, or None if it has no row."""
  required = 'true' if attr.facets.get('required') else 'false'
  nodefault = 'true' if attr.facets.get('nodefault') else 'false'
  handwrite = 'true' if attr.facets.get('writing') else 'false'
  if attr.type == 'id' and attr.name == 'name':
    # names are never settable in default classes
    return (f'"{attr.name}"', 'mjXAttr::kName', '1', 'true', required,
            'true', 'false', '-1')
  if attr.type == 'ref' and attr.target == 'default':
    return None  # class selection is handled by the section parsers
  if attr.type == 'file':
    return None  # files need VFS/asset-dir context: hand-read in the remnant
  field = attr.facets.get('field', attr.name)
  entry = fields.get(field)
  if entry is None:
    raise ValueError(f'{ctx}.{attr.name}: no field {spec}.{field}')
  ctype, dim = entry
  if attr.type in ('string', 'file', 'ref', 'id'):
    if ctype == 'mjStringVec*':
      kind, length, exact = 'kStringVec', '1', 'true'
    elif ctype != 'mjString*':
      raise ValueError(f'{ctx}.{attr.name}: bound to non-string '
                       f'field {field} ({ctype})')
    else:
      kind, length, exact = 'kString', '1', 'true'
  elif attr.type == 'enum':
    if not ctype.startswith('mjt') and ctype != 'int':
      raise ValueError(f'{ctx}.{attr.name}: enum bound to '
                       f'field {field} ({ctype})')
    kind = 'kEnumByte' if ctype in ('mjtByte', 'mjtBool') else 'kEnum'
    length, exact = '1', 'true'
  elif attr.type == 'flags':
    if ctype != 'int':
      raise ValueError(f'{ctx}.{attr.name}: flags bound to '
                       f'field {field} ({ctype})')
    kind, length, exact = 'kFlags', '1', 'true'
  elif attr.type == 'bool':
    if ctype in ('mjtBool', 'mjtByte'):
      kind, length, exact = 'kBool', '1', 'true'
    elif ctype == 'int' or ctype.startswith('mjt'):
      # int-typed flag: read through the keyword map (values are 0/1)
      row = [f'"{attr.name}"', 'mjXAttr::kEnum', '1', 'true', required,
             nodefault, handwrite,
             f'(int)offsetof({spec}, {prefix}{field})', 'bool_map', '2']
      return tuple(row)
    else:
      raise ValueError(f'{ctx}.{attr.name}: bool bound to '
                       f'field {field} ({ctype})')
  elif attr.type == 'chars':
    if ctype != 'char' or dim is None:
      raise ValueError(f'{ctx}.{attr.name}: chars bound to '
                       f'field {field} ({ctype})')
    lo, hi = attr.arity.lo, attr.arity.hi
    if str(dim) != str(hi):
      raise ValueError(f'{ctx}.{attr.name}: chars arity {hi} vs '
                       f'field dim {dim}')
    kind, length = 'kChars', str(hi)
    exact = 'true' if lo == hi else 'false'
  elif attr.type in ('double', 'float', 'int') and attr.arity.hi is None:
    vec_kinds = {'mjDoubleVec*': 'kDoubleVec', 'mjFloatVec*': 'kFloatVec',
                 'mjIntVec*': 'kIntVec'}
    kind = vec_kinds.get(ctype)
    if kind is None:
      raise ValueError(f'{ctx}.{attr.name}: unbounded vector '
                       f'bound to field {field} ({ctype})')
    length, exact = '1', 'true'
  elif attr.type in ('double', 'float', 'int'):
    kind = KIND_BY_CTYPE.get(ctype)
    if kind is None or (attr.type == 'int') != (kind == 'kInt'):
      raise ValueError(f'{ctx}.{attr.name}: schema type '
                       f'{attr.type} vs field {field} ({ctype})')
    lo, hi = attr.arity.lo, attr.arity.hi
    length = str(hi)
    declared = dim if dim is not None else '1'
    if str(declared) != length:
      if DIM_EQUIV.get(str(declared)) == length:
        length = str(declared)  # emit the expression: it tracks the header
      else:
        raise ValueError(f'{ctx}.{attr.name}: arity {length} vs '
                         f'field dim {declared}')
    exact = 'true' if lo == hi or (lo == 1 and hi == 1) else 'false'
  else:
    raise ValueError(f'{ctx}.{attr.name}: kind {attr.type} is '
                     'not table-drivable yet')
  offset = f'(int)offsetof({spec}, {prefix}{field})'
  row = [f'"{attr.name}"', f'mjXAttr::{kind}', length, exact,
         required, nodefault, handwrite, offset]
  if kind in ('kEnum', 'kEnumByte', 'kFlags'):
    row += [f'{attr.target}_map', f'{attr.target}_sz']
  return tuple(row)


def rows_for(schema, structs, element_name):
  """Return (struct name, list of row tuples) for one table-driven element."""
  element = schema.elements[element_name]
  if not element.spec:
    raise ValueError(f'{element_name}: no bound spec struct')
  sub = element.facets.get('field')
  key = f'{element.spec}.{sub}' if sub else element.spec
  fields = structs.get(key)
  if fields is None:
    raise ValueError(f'{element_name}: struct {key} not found in headers')
  prefix = f'{sub}.' if sub else ''

  hand_attrs = set()
  for gname in HAND_GROUPS:
    if any(isinstance(m, mjcf_schema.Use) and m.group == gname
           for m in element.members):
      hand_attrs |= {m.name for m in schema.groups[gname].members}

  rows = []
  for const in element.consts():
    entry = fields.get(const.field)
    if entry is None:
      raise ValueError(f'{element_name}: set {const.field}: no field '
                       f'{element.spec}.{const.field}')
    ctype = entry[0]
    if not ctype.startswith('mjt') and ctype != 'int':
      raise ValueError(f'{element_name}: set {const.field}: field is {ctype}')
    rows.append(('nullptr', 'mjXAttr::kConst', '1', 'true', 'false', 'false',
                 'false',
                 f'(int)offsetof({element.spec}, {prefix}{const.field})',
                 'nullptr', '0', const.value))
  for attr in schema.expanded_attrs(element):
    if attr.name in hand_attrs or 'reading' in attr.facets:
      continue  # hand-read in the OneX() remnant
    row = _attr_row(element.spec, fields, attr, element_name,
                    prefix=prefix)
    if row is not None:
      rows.append(row)
  return element.spec, rows


def rows_for_group(schema, structs, group_name, spec):
  """Return list of row tuples for a shared group, bound to the given struct."""
  fields = structs.get(spec)
  if fields is None:
    raise ValueError(f'{group_name}: struct {spec} not in mjspec.h')
  rows = []
  for attr in schema.groups[group_name].members:
    if not isinstance(attr, mjcf_schema.Attr) or 'reading' in attr.facets:
      continue
    row = _attr_row(spec, fields, attr, group_name)
    if row is not None:
      rows.append(row)
  return rows


def generate():
  """Generate the mjcf_read_table.inc content as a string."""
  schema = mjcf_schema.parse_file(SCHEMA_PATH)
  structs = parse_spec_structs(SPEC_H_PATH, MODEL_H_PATH)
  migrated = table_driven_elements(schema)
  out = [_HEADER]
  for name in migrated:
    struct, rows = rows_for(schema, structs, name)
    array = array_name(name)
    out.append(f'// {name} ({struct})')
    out.append(f'inline constexpr mjXAttr {array}[] = {{')
    for row in rows:
      out.append('  {' + ', '.join(row) + '},')
    out.append('};')
    out.append(f'inline constexpr int {array}N = '
               f'sizeof({array}) / sizeof({array}[0]);')
    out.append('')
  out.append('// sensors fully described by the schema: dispatch by tag')
  out.append('struct mjXSensorEntry { const char* tag; const mjXAttr* rows;'
             ' int n; };')
  out.append('inline constexpr mjXSensorEntry kSensorDispatch[] = {')
  for name in SENSOR_DISPATCH:
    array = array_name(name)
    tag = schema.elements[name].xml_name()
    out.append(f'  {{"{tag}", {array}, {array}N}},')
  out.append('};')
  out.append('inline constexpr int kSensorDispatchN = '
             'sizeof(kSensorDispatch) / sizeof(kSensorDispatch[0]);')
  out.append('')
  out.append('// actuator tags and their mechanical rows: dispatch by tag or type')
  out.append('struct mjXActuatorEntry { const char* tag; int type;'
             ' const mjXAttr* rows; int n; };')
  out.append('inline constexpr mjXActuatorEntry kActuatorDispatch[] = {')
  for tag, const in schema.enums[ACTUATOR_DISPATCH_ENUM].items:
    array = array_name(tag)
    out.append(f'  {{"{tag}", {const}, {array}, {array}N}},')
  out.append('};')
  out.append('inline constexpr int kActuatorDispatchN = '
             'sizeof(kActuatorDispatch) / sizeof(kActuatorDispatch[0]);')
  out.append('')
  for gname, (struct, array) in EMIT_GROUPS.items():
    rows = rows_for_group(schema, structs, gname, struct)
    out.append(f'// group {gname} ({struct})')
    out.append(f'inline constexpr mjXAttr {array}[] = {{')
    for row in rows:
      out.append('  {' + ', '.join(row) + '},')
    out.append('};')
    out.append(f'inline constexpr int {array}N = '
               f'sizeof({array}) / sizeof({array}[0]);')
    out.append('')
  out.append('// clang-format on')
  return '\n'.join(out) + '\n'


def main():
  """CLI entry point: generate mjcf_read_table.inc to stdout or a file."""
  if len(sys.argv) > 2:
    sys.exit('usage: generate_read_table.py [output.inc]')
  text = generate()
  if len(sys.argv) == 2:
    with open(sys.argv[1], 'w', encoding='utf-8') as file:
      file.write(text)
  else:
    sys.stdout.write(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
