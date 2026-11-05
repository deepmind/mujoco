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
"""Generates Python bindings for the MuJoCo Filament renderer API.

This file generates a cc.inc file that is included in render_filament.cc.
Details about what it is generating and why can be found in that file. This file
contains minimal comments.
"""

import os
import re
import typing

from absl import app
from codegen import code_builder
from introspect import ast_nodes
from introspect.codegen import ast_processor

import subprocess

# pylint: disable=line-too-long

_CLANG_AST_JSON_PATH = 'mjrfilament.h.json'
_MUJOCO_INCLUDE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'include', 'mujoco'
)

# mjrf structs to be treated as objects.
_OBJECTS = [
    'mjrfContext',
    'mjrfTexture',
    'mjrfMesh',
    'mjrfScene',
    'mjrfLight',
    'mjrfRenderable',
    'mjrfRenderTarget',
]

# Do not autogenerate bindings for the following functions and fields.
_EXCLUDED = [
    # mjrfReadRequest implements a custom `alloc` function.
    'mjrfReadRequest.num_bytes',
    # mjrfTextureData implements a custom `set_image` function.
    'mjrfTextureData.num_bytes',
    # mjrfMeshConfig.attributes is list-assignable, so don't expose the num.
    'mjrfMeshConfig.num_attributes',
    # mjrfMeshData implements a custom `set_vertices` function.
    'mjrfMeshData.num_vertices',
    'mjrfMeshData.interleaved',
    # mjrfMeshData implements a custom `set_indices` function.
    'mjrfMeshData.num_indices',
    'mjrfMeshData.index_type',
    # placeholder field, not intended to be used
    'mjrfSceneParams.unused',
    # material properties for internal use only
    'mjrfMaterial.reflection_texture',
    'mjrfMaterial.reflection_normal',
    'mjrfMaterial.reflection_view_proj',
    'mjrfMaterial.haze_ratio',
    # This getter is implemented with an output pointer, but we want the python
    # implementation to return the value directly.
    'mjrf_getRenderableMaterial',
    # This function takes a pointer+int size in the C API, but we want to pass
    # in a python list/tuple instead.
    'mjrf_render',
    # currently cannot pass a mjModel object with these bindings
    'mjrf_configureSceneFromModel',
]

# These structs are defined outside of the mjrf namespace, but we want to
# include them in the bindings.
_EXTERNAL_STRUCTS = [
    'mjrCamera',
    'mjrRect',
    'mjrVertexAttribute',
]

# These fields are defined outside of the mjrf namespace. They are defined as
# int types, but we want to expose their "true" types in the bindings.
_CAST_AS = {
    'mjrCamera.orthographic': 'bool',
    'mjvGLCamera.orthographic': 'bool',
    'mjrfContextConfig.native_window': 'uint64_t',
    'mjrVertexAttribute.type': 'mjrVertexAttributeType',
    'mjrVertexAttribute.usage': 'mjrVertexAttributeUsage',
}

_RELEASE_GIL = [
    'mjrf_waitForFrame',
    'mjrf_flush',
]


def _get_json_path() -> str:
  if not os.path.exists(_CLANG_AST_JSON_PATH):
    header = os.path.join(_MUJOCO_INCLUDE_DIR, 'mjrfilament.h')
    include_dir = os.path.dirname(_MUJOCO_INCLUDE_DIR)
    cmd = [
        'clang',
        '-ffreestanding',
        f'-I{include_dir}',
        '-Xclang',
        '-ast-dump=json',
        '-fsyntax-only',
        '-fparse-all-comments',
        '-x',
        'c',
        header,
    ]
    with open(_CLANG_AST_JSON_PATH, 'w', encoding='utf-8') as f:
      subprocess.run(cmd, stdout=f, check=True)
  return _CLANG_AST_JSON_PATH


def _get_header_paths() -> list[str]:
  header_dir = _MUJOCO_INCLUDE_DIR
  header_paths = [
      os.path.join(header_dir, p)
      for p in os.listdir(header_dir)
      if p.endswith('.h')
  ]
  return header_paths


def _is_bindable_enum(name: str) -> bool:
  """Returns True if we should generate a binding for the given enum."""
  return name.startswith('mjr')


def _is_bindable_object(name: str) -> bool:
  """Returns True if we should generate a binding for the given object."""
  return name in _OBJECTS


def _is_bindable_struct(name: str):
  """Returns True if we should generate a binding for the given struct."""
  if _is_bindable_object(name):
    return False
  if name.startswith('mjrf'):
    return True
  if name in _EXTERNAL_STRUCTS:
    return True
  return False


def _is_bindable_function(name: str) -> bool:
  """Returns True if we should generate a binding for the given function."""
  if name in _EXCLUDED:
    return False
  if not name.startswith('mjrf'):
    return False
  if '_DEBUG_' in name:
    return False
  # Life-cycle functions are handled explicitly.
  if name.startswith('mjrf_default'):
    return False
  if name.startswith('mjrf_destroy'):
    return False
  if name.startswith('mjrf_create'):
    return False
  return True


def _is_bindable_property(
    struct: ast_nodes.StructDecl, field: ast_nodes.StructFieldDecl
) -> bool:
  """Returns True if we should generate a binding for the given struct field."""
  if field.decltype == 'mjrfCallback':
    return False

  fullname = struct.name + '.' + field.name
  if fullname in _EXCLUDED:
    return False
  if 'void *' in field.decltype and fullname not in _CAST_AS:
    return False
  return True


def _make_enum(code: code_builder.CodeBuilder, enum: ast_nodes.EnumDecl) -> None:
  """Generates a Python enum binding for a MuJoCo enum."""
  ident = _snake_case(enum.name)
  stripped = _try_strip_all_prefixes(enum.name)

  code.line(f'py::enum_<{enum.name}> {ident}(m, "{stripped}", py::module_local());')
  for value in enum.values.keys():
    stripped = _try_strip_all_prefixes(value)
    code.line(f'{ident}.value("{stripped}", {value});')


def _is_param_array(param: ast_nodes.FunctionParameterDecl) -> bool:
  """Returns True if the given function parameter is an array."""
  if isinstance(param.type, ast_nodes.ArrayType):
    array_type = typing.cast(ast_nodes.ArrayType, param.type)
    if isinstance(array_type.inner_type, ast_nodes.ValueType):
      return True
  return False


def _is_param_object(param: ast_nodes.FunctionParameterDecl) -> bool:
  """Returns True if the given function parameter is an object."""
  base_type = param.decltype.replace('const ', '').replace(' *', '')
  if base_type in _OBJECTS:
    return True
  return False


def _get_param_type(param: ast_nodes.FunctionParameterDecl) -> str:
  """Converts a C parameter type into a Python-compatible parameter type."""
  if _is_param_array(param):
    # e.g. convert `float[3]` to `Float3::Ref`
    array_type = typing.cast(ast_nodes.ArrayType, param.type)
    value_type = typing.cast(ast_nodes.ValueType, array_type.inner_type)
    value_type = value_type.name[0].upper() + value_type.name[1:]
    return value_type + str(array_type.extents[0]) + '::Ref&'
  elif _is_param_object(param):
    # e.g. convert `mjrLight*` to `PyLight&`
    base_type = param.decltype.replace('const ', '').replace(' *', '')
    stripped = _try_strip_all_prefixes(base_type)
    return f'Py{stripped}&'
  elif param.decltype == 'mjtBool':
    return 'bool'
  # Assume everything else is compatible.
  return param.decltype


def _get_field_type(field: ast_nodes.StructFieldDecl) -> str:
  """Returns the Python type of a MuJoCo struct field."""
  if field.decltype == 'mjtBool':
    return 'bool'

  # Convert enums into types based on their documentation.
  if field.decltype == 'int' and '[' in field.doc and ']' in field.doc:
    start = field.doc.find('[') + 1
    end = field.doc.find(']')
    return field.doc[start:end].strip()

  # Convert arrays into types based on their documentation.
  if '[' in field.decltype and ']' in field.decltype:
    start = field.decltype.find('[') + 1
    end = field.decltype.find(']')
    if 'float' in field.decltype:
      return 'Float' + field.decltype[start:end].strip()
    elif 'int' in field.decltype:
      return 'Int' + field.decltype[start:end].strip()
    else:
      return 'Vector<' + field.decltype[:start - 1].strip() + '>'

  if '*' in field.decltype:
    obj_type = field.decltype.replace(' *', '').replace('const ', '')
    if obj_type in _OBJECTS:
      stripped = _try_strip_all_prefixes(obj_type)
      return f'Py{stripped}'

  return field.decltype


def _get_init_arg_type(struct_name: str, field: ast_nodes.StructFieldDecl) -> str:
  """Returns the Python type of a MuJoCo struct field."""
  field_type = _get_field_type(field)
  if field_type.startswith('Py'):
    return field_type + '*'
  if 'Float' in field_type:
    field_type = field_type + '::Ref'
  if struct_name + '.' + field.name in _CAST_AS:
    field_type = _CAST_AS[struct_name + '.' + field.name]
  if 'Vector' in field_type:
    return f'std::optional<{field_type.replace('Vector', 'std::vector')}>'
  return f'std::optional<{field_type}>'


def _fwd_param(param: ast_nodes.FunctionParameterDecl) -> str:
  """Converts the python-compatible parameter into an argument for the C API."""
  if _is_param_array(param):
    # e.g. pass `Float3`` data to a function expecting a `const float[3]`.
    return f'{param.name}.data()'
  if _is_param_object(param):
    # e.g. pass `PyLight&` to a function expecting a `mjrLight*`.
    return f'{param.name}.get()'
  return param.name


def _get_function_aliases(func: ast_nodes.FunctionDecl) -> dict[str, str]:
  """Returns a list of aliases for the given function."""
  aliases = {}
  if '[' in func.doc and ']' in func.doc:
    start = func.doc.find('[') + 1
    end = func.doc.find(']')
    args = func.doc[start:end].strip().split(',')
    for arg in args:
      if ':' in arg:
        k, v = arg.split(':')
        aliases[k.strip()] = v.strip()
  return aliases


def _make_struct(code: code_builder.CodeBuilder, struct: ast_nodes.StructDecl) -> None:
  """Generates a Python struct binding for a MuJoCo struct."""
  ident = _snake_case(struct.name)
  stripped = _try_strip_all_prefixes(struct.name)

  fields = []
  has_del = False
  for field in struct.fields:
    if isinstance(field, ast_nodes.StructFieldDecl):
      field = typing.cast(ast_nodes.StructFieldDecl, field)
      if _is_bindable_property(struct, field):
        fields.append(field)
      if field.decltype == 'mjrfCallback':
        has_del = True

  code.line(
      f'py::class_<{struct.name}> {ident}(m, "{stripped}", py::module_local());'
  )

  # Add an initializer that takes all fields as optional arguments. Each
  # argument is then optionally assigned to the corresponding field in the
  # struct.
  optional_args = []
  assignments = []
  for field in fields:
    optional_args.append(f'{_get_init_arg_type(struct.name, field)} {field.name}')

    field_type = _get_field_type(field)
    if 'Vector' in field_type:
      assignments.append(
          f'if ({field.name}) {{\n'
          f'    if ({field.name}->size() > std::size(obj.{field.name})) {{\n'
          f'      throw py::value_error("Too many {field.name}, maximum allowed'
          f' is " + std::to_string(std::size(obj.{field.name})));\n'
          f'    }}\n'
          f'    Setter(obj.{field.name}, *{field.name});\n'
          f'    obj.num_{field.name} = {field.name}->size();\n'
          f'  }}'
      )
    else:
      assignments.append(f'if ({field.name}) Setter(obj.{field.name}, *{field.name});')

  code.line(f'{ident}.def(py::init([](')
  with code.block(braces=False):
    code.line(',\n  '.join(optional_args))
  with code.block(')', ')'):
    code.line(f'{struct.name} obj;')
    code.line('Init(obj);')
    code.line('\n  '.join(assignments))
    code.line('return obj;')
  # code.line(')')
  for field in fields:
    code.line(f', py::arg("{field.name}").none(true) = py::none()')
  code.line(');')

  # If the struct has a callback, it means it has custom memory management
  # requirements. In that case, implement __del__ and free() methods.
  if has_del:
    code.line(f'{ident}.def("__del__", []({struct.name}& obj) {{ Free(obj); }});')
    code.line(f'{ident}.def("free", []({struct.name}& obj) {{ Free(obj); }});')

  # Expose all fields as properties.
  for field in fields:
    stripped = _try_strip_all_prefixes(field.name)
    field_type = _get_field_type(field)
    member_ref = f'&{struct.name}::{field.name}'

    # Vector fields cannot be bound as properties.
    if 'Vector' in field_type:
      continue

    if f'{struct.name}.{field.name}' in _CAST_AS:
      field_type = _CAST_AS[f'{struct.name}.{field.name}']

    code.line(f'DefProperty<{field_type}>({ident}, "{stripped}", {member_ref});')


def _make_object(code: code_builder.CodeBuilder, struct: ast_nodes.StructDecl) -> None:
  """Generates a Python object binding for a MuJoCo struct."""
  ident = _snake_case(struct.name)
  stripped = _try_strip_all_prefixes(struct.name)
  code.line(f'py::class_<Py{stripped}> {ident}(m, "{stripped}");')


def _make_function(code: code_builder.CodeBuilder, func: ast_nodes.FunctionDecl) -> None:
  """Generates a Python method binding for a MuJoCo function."""
  ident = 'm'  # loose functions are bound to the module
  name = func.name

  # If the first parameter to the function is a "object" type, then we'll
  # bind the function as a method. In that case, if the name of the type
  # appears in the function name, we'll remove it. (e.g. mjrf_setLightEnabled
  # becomes Light.set_enabled).
  param0 = func.parameters[0].decltype.replace('const ', '').replace(' *', '')
  if param0 in _OBJECTS:
    ident = _snake_case(param0)
    obj_type = _try_strip_all_prefixes(param0)
    if obj_type in name:
      name = name.replace(obj_type, '')

  name = _snake_case(_try_strip_all_prefixes(name))

  # Some functions have "aliases" documented in their signatures.
  aliases = _get_function_aliases(func)

  # The signature for the bound python function.
  params = []
  for param in func.parameters:
    param_type = _get_param_type(param)
    if param.name in aliases:
      param_type = aliases[param.name]
    params.append(f'{param_type} {param.name}')

  # The arguments to the C function.
  args = []
  for param in func.parameters:
    args.append(_fwd_param(param))

  # The return type of the bound python function.
  return_type = ''
  if 'returns' in aliases:
    return_type = '(' + aliases['returns'] + ')'

  code.line(f'{ident}.def("{name}", [](')
  with code.block(braces=False):
    code.line(',\n  '.join(params))
  with code.block(')'):
    code.line(f'return {return_type}{func.name}({", ".join(args)});')
  for param in func.parameters[1:]:
    code.line(f', py::arg("{param.name}")')
  if func.name in _RELEASE_GIL:
    code.line(', py::call_guard<py::gil_scoped_release>()')
  code.line(');')


def _snake_case(name: str) -> str:
  return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def _try_strip_prefix(name: str, prefix: str) -> str:
  if name.startswith(prefix):
    return name[len(prefix) :]
  return name


def _try_strip_all_prefixes(name: str) -> str:
  stripped = _try_strip_prefix(name, 'mjvGL')
  stripped = _try_strip_prefix(stripped, 'mjrf')
  stripped = _try_strip_prefix(stripped, 'mjr')
  stripped = _try_strip_prefix(stripped, 'mjt')
  stripped = _try_strip_prefix(stripped, 'mj')
  if stripped[0] == '_':
    stripped = stripped[1:]
  return stripped


def main(_) -> None:
  visitor = ast_processor.process(_get_json_path(), _get_header_paths())

  aliases = {}
  for name, qual_type in visitor.type_aliases.items():
    aliases[qual_type] = name

  code = code_builder.CodeBuilder()
  code.license(2026)
  code.line('// AUTO-GENERATED. DO NOT EDIT.')
  code.newline()

  for name, enum in visitor.exported_enums.items():
    alias = aliases.get(name, '')
    if _is_bindable_enum(alias):
      enum.name = alias
    if _is_bindable_enum(enum.name):
      _make_enum(code, enum)
      code.newline()

  for name, struct in visitor.exported_structs.items():
    alias = aliases.get(name, '')
    if _is_bindable_struct(alias):
      struct.name = alias
    if _is_bindable_struct(struct.name):
      _make_struct(code, struct)
      code.newline()

  for name, struct in visitor.exported_structs.items():
    if _is_bindable_object(name):
      _make_object(code, struct)
      code.newline()

  for name, func in visitor.exported_functions.items():
    if not _is_bindable_function(name):
      continue

    _make_function(code, func)
    code.newline()

  print(code.to_string())


if __name__ == '__main__':
  app.run(main)
