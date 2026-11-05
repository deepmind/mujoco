# Copyright 2024 DeepMind Technologies Limited
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
"""Generates the bindings for the MuJoCo specs."""

from collections.abc import Sequence

from absl import app

from introspect import ast_nodes
from introspect import enums
from introspect import structs


SCALAR_TYPES = {'int', 'double', 'float', 'mjtByte', 'mjtBool', 'mjtNum'}

# pylint: disable=bad-whitespace
# key, parent, default, listname, objtype
SPECS = [
    ('mjsBody',     'Body', True,  'bodies',     'mjOBJ_BODY'),
    ('mjsSite',     'Body', True,  'sites',      'mjOBJ_SITE'),
    ('mjsGeom',     'Body', True,  'geoms',      'mjOBJ_GEOM'),
    ('mjsJoint',    'Body', True,  'joints',     'mjOBJ_JOINT'),
    ('mjsCamera',   'Body', True,  'cameras',    'mjOBJ_CAMERA'),
    ('mjsFrame',    'Body', True,  'frames',     'mjOBJ_FRAME'),
    ('mjsLight',    'Body', True,  'lights',     'mjOBJ_LIGHT'),
    ('mjsFlex',     'Spec', False, 'flexes',     'mjOBJ_FLEX'),
    ('mjsMesh',     'Spec', True,  'meshes',     'mjOBJ_MESH'),
    ('mjsSkin',     'Spec', False, 'skins',      'mjOBJ_SKIN'),
    ('mjsHField',   'Spec', False, 'hfields',    'mjOBJ_HFIELD'),
    ('mjsTexture',  'Spec', True,  'textures',   'mjOBJ_TEXTURE'),
    ('mjsMaterial', 'Spec', True,  'materials',  'mjOBJ_MATERIAL'),
    ('mjsPair',     'Spec', True,  'pairs',      'mjOBJ_PAIR'),
    ('mjsEquality', 'Spec', True,  'equalities', 'mjOBJ_EQUALITY'),
    ('mjsTendon',   'Spec', True,  'tendons',    'mjOBJ_TENDON'),
    ('mjsActuator', 'Spec', True,  'actuators',  'mjOBJ_ACTUATOR'),
    ('mjsSensor',   'Spec', False, 'sensors',    'mjOBJ_SENSOR'),
    ('mjsNumeric',  'Spec', False, 'numerics',   'mjOBJ_NUMERIC'),
    ('mjsText',     'Spec', False, 'texts',      'mjOBJ_TEXT'),
    ('mjsTuple',    'Spec', False, 'tuples',     'mjOBJ_TUPLE'),
    ('mjsKey',      'Spec', False, 'keys',       'mjOBJ_KEY'),
    ('mjsExclude',  'Spec', False, 'excludes',   'mjOBJ_EXCLUDE'),
    ('mjsPlugin',   'Spec', False, 'plugins',    'mjOBJ_PLUGIN'),
]
SPECS_ADD = SPECS + [
    ('mjsBody',     'Frame', True, 'bodies',     'mjOBJ_BODY'),
    ('mjsSite',     'Frame', True, 'sites',      'mjOBJ_SITE'),
    ('mjsGeom',     'Frame', True, 'geoms',      'mjOBJ_GEOM'),
    ('mjsJoint',    'Frame', True, 'joints',     'mjOBJ_JOINT'),
    ('mjsCamera',   'Frame', True, 'cameras',    'mjOBJ_CAMERA'),
    ('mjsFrame',    'Frame', True, 'frames',     'mjOBJ_FRAME'),
    ('mjsLight',    'Frame', True, 'lights',     'mjOBJ_LIGHT'),
]
# pylint: enable=bad-whitespace


def _value_binding_code(
    field: ast_nodes.ValueType, classname: str = '', varname: str = ''
) -> str:
  """Creates a string that defines Python bindings for a value type."""
  fulltype = field.name
  if field.name not in SCALAR_TYPES:
    fulltype += '&'
  fullvarname = varname
  rawclassname = classname.replace('mjs', 'raw::Mjs')
  if classname == 'mjSpec':  # raw mjSpec has a wrapper
    rawclassname = classname.replace('mjS', 'MjS')
    fullvarname = 'ptr->' + varname
  if field.name.startswith('mjs'):  # all other mjs are raw structs
    fulltype = field.name.replace('mjs', 'raw::Mjs')
    if (
        field.name == 'mjsPlugin'
        or field.name == 'mjsOrientation'
        or field.name == 'mjsCompiler'
        or field.name == 'mjsAuthored'
    ):
      # plugin, orientation, compiler, authored aren't pointers
      fulltype = fulltype + '&'
    else:
      fulltype = fulltype + '*'
  # non-mjs structs
  rawclassname = rawclassname.replace('mjOption', 'raw::MjOption')
  rawclassname = rawclassname.replace('mjVisual', 'raw::MjVisual')
  rawclassname = rawclassname.replace('mjStatistic', 'raw::MjStatistic')
  fulltype = fulltype.replace('mjOption', 'raw::MjOption')
  fulltype = fulltype.replace('mjVisual', 'raw::MjVisual')
  fulltype = fulltype.replace('mjStatistic', 'raw::MjStatistic')
  element = ''
  is_enum = field.name in enums.ENUMS

  if field.name == 'mjsPlugin':
    setter = f"""[]({rawclassname}& self, {fulltype} {varname}) {{
      if (self.{fullvarname}.name && {varname}.name) *self.{fullvarname}.name = *{varname}.name;
      if (self.{fullvarname}.plugin_name && {varname}.plugin_name) *self.{fullvarname}.plugin_name = *{varname}.plugin_name;
      self.{fullvarname}.active = {varname}.active;
      if (self.{fullvarname}.info && {varname}.info) *self.{fullvarname}.info = *{varname}.info;
    }}"""
  elif is_enum:
    setter = f"""[]({rawclassname}& self, int {varname}) {{
      self.{fullvarname}{element} = static_cast<{field.name}>({varname}){element};
    }}"""
  else:
    setter = f"""[]({rawclassname}& self, {fulltype} {varname}) {{
      self.{fullvarname}{element} = {varname}{element};
    }}"""

  def_property_args = (
      f'"{varname}"',
      f"""[]({rawclassname}& self) -> {field.name if is_enum else fulltype} {{
        return self.{fullvarname};
      }}""",
      setter,
  )

  if field.name not in SCALAR_TYPES and not is_enum:
    def_property_args += ('py::return_value_policy::reference_internal',)

  return f'{classname}.def_property({",".join(def_property_args)});'


def _struct_binding_code(
    field: ast_nodes.AnonymousStructDecl, classname: str = '', varname: str = ''
) -> str:
  """Creates a string that declares Python bindings for an anonymous struct."""
  code = ''
  name = classname + varname.title()
  # explicitly generate for nested fields with arrays
  if any(
      isinstance(f, ast_nodes.StructFieldDecl)
      and isinstance(f.type, ast_nodes.ArrayType)
      for f in field.fields
  ):
    for subfield in field.fields:
      code += _binding_code(subfield, name)  # pyrefly: ignore[bad-argument-type]
  # generate for the struct itself
  field = ast_nodes.ValueType(name=name)  # pyrefly: ignore[bad-assignment]
  code += _value_binding_code(field, classname, varname)  # pyrefly: ignore[bad-argument-type]
  return code


def _array_binding_code(
    field: ast_nodes.ArrayType, classname: str = '', varname: str = ''
) -> str:
  """Creates a string that declares Python bindings for an array type."""
  if len(field.extents) > 1:
    raise NotImplementedError()
  innertype = field.inner_type.decl()
  rawclassname = classname.replace('mjs', 'raw::Mjs')
  rawclassname = rawclassname.replace('mjOption', 'raw::MjOption')
  rawclassname = rawclassname.replace('mjVisual', 'raw::MjVisual')
  rawclassname = rawclassname.replace('mjStatistic', 'raw::MjStatistic')
  fullvarname = varname
  if classname == 'mjSpec':  # raw mjSpec has a wrapper
    rawclassname = classname.replace('mjS', 'MjS')
    fullvarname = 'ptr->' + varname
  if classname == 'mjsFrame' and varname == 'scale':
    return f"""\
  {classname}.def_property(
      "{varname}",
      []({rawclassname}& self) -> MjDouble3 {{
        return MjDouble3(self.{fullvarname});
    }},
      []({rawclassname}& self, py::object {varname}) {{
        try {{
          MjDouble3(self.{fullvarname}) = py::cast<Eigen::Vector3d>({varname});
        }} catch (const py::cast_error&) {{
          try {{
            MjDouble3(self.{fullvarname}).setConstant(py::cast<double>({varname}));
          }} catch (const py::cast_error&) {{
            throw py::value_error("{varname}: expected a scalar or a sequence of size 3");
          }}
        }}
    }}, py::return_value_policy::reference_internal);"""
  if innertype == 'mjtNum':
    innertype = 'MjNum'  # custom Eigen type for mjtNum fields
  elif innertype == 'double':
    innertype = 'MjDouble'  # custom Eigen type for double fields
  elif innertype == 'float':
    innertype = 'MjFloat'  # custom Eigen type
  elif innertype == 'int':
    innertype = 'MjInt'  # custom Eigen type
  elif innertype == 'char':
    # char array special case
    return f"""\
  {classname}.def_property(
    "{varname}",
    []({rawclassname}& self) -> MjTypeVec<char> {{
      return MjTypeVec<char>(self.{fullvarname}, {field.extents[0]});
    }},
    []({rawclassname}& self, py::object rhs) {{
      char buf[{field.extents[0]}] = {{}};
      int i = 0;
      for (auto val : rhs) {{
        if (i >= {field.extents[0]}) {{
          throw py::value_error(
              "{varname} may have at most {field.extents[0]} characters");
        }}
        buf[i++] = py::cast<char>(val);
      }}
      for (int j = 0; j < {field.extents[0]}; ++j) {{
        self.{fullvarname}[j] = buf[j];
      }}
    }}, py::return_value_policy::move);"""
  # all other array types
  return f"""\
  {classname}.def_property(
      "{varname}",
      []({rawclassname}& self) -> {innertype}{field.extents[0]} {{
        return {innertype}{field.extents[0]}(self.{fullvarname});
    }},
      []({rawclassname}& self, {innertype}Ref{field.extents[0]} {varname}) {{
        {innertype}{field.extents[0]}(self.{fullvarname}) = {varname};
    }}, py::return_value_policy::reference_internal);"""


def _ptr_binding_code(
    field: ast_nodes.PointerType, classname: str = '', varname: str = ''
) -> str:
  """Creates a string that declares Python bindings for a pointer type."""
  vartype = field.inner_type.decl()
  rawclassname = classname.replace('mjs', 'raw::Mjs')
  fullvarname = varname
  if classname == 'mjSpec':  # raw mjSpec has a wrapper
    rawclassname = classname.replace('mjS', 'MjS')
    fullvarname = 'ptr->' + varname
  if vartype == 'mjsElement':  # this is ignored by the caller
    return 'mjsElement'
  if vartype.startswith('mjs'):  # for structs, use the value case
    return _value_binding_code(field.inner_type, classname, varname)  # pyrefly: ignore[bad-argument-type]
  elif vartype == 'mjString':  # C++ string -> Python string
    return f"""\
  {classname}.def_property(
      "{varname}",
      []({rawclassname}& self) -> std::string_view {{
        return *self.{fullvarname};
      }},
      []({rawclassname}& self, std::string_view {varname}) {{
        *(self.{fullvarname}) = {varname};
    }});"""
  elif (  # C++ vectors of values -> custom array
      vartype == 'mjDoubleVec'
      or vartype == 'mjFloatVec'
      or vartype == 'mjIntVec'
  ):
    vartype = vartype.replace('mj', '').replace('Vec', '').lower()
    return f"""\
  {classname}.def_property(
    "{varname}",
    []({rawclassname}& self) -> MjTypeVec<{vartype}> {{
        return MjTypeVec<{vartype}>(self.{fullvarname}->data(),
                                    self.{fullvarname}->size());
      }},
    []({rawclassname}& self, py::object rhs) {{
        self.{fullvarname}->clear();
        self.{fullvarname}->reserve(py::len(rhs));
        for (auto val : rhs) {{
          self.{fullvarname}->push_back(py::cast<{vartype}>(val));
      }}
    }}, py::return_value_policy::move);"""
  elif vartype == 'mjByteVec':
    return f"""\
  {classname}.def_property(
    "{varname}",
    []({rawclassname}& self) -> py::bytes {{
      return py::bytes(reinterpret_cast<const char*>(self.{fullvarname}->data()),
                                                     self.{fullvarname}->size());
    }},
    []({rawclassname}& self, py::bytes rhs) {{
      self.{fullvarname}->clear();
      std::string_view rhs_view = py::cast<std::string_view>(rhs);
      self.{fullvarname}->reserve(rhs_view.length());
      for (char val : rhs_view) {{
        self.{fullvarname}->push_back(static_cast<std::byte>(val));
      }}
    }}, py::return_value_policy::move);"""
  elif vartype == 'mjStringVec':
    # Special case for material.textures: must be exactly mjNTEXROLE size
    if classname == 'mjsMaterial' and varname == 'textures':
      return f"""\
  {classname}.def_property(
    "{varname}",
    []({rawclassname}& self) -> MjTypeVec<std::string> {{
        return MjTypeVec<std::string>(self.{fullvarname}->data(),
                                      self.{fullvarname}->size());
      }},
    []({rawclassname}& self, py::object rhs) {{
        if (py::len(rhs) != mjNTEXROLE) {{
          throw pybind11::value_error(
              "material.textures must have exactly " + std::to_string(mjNTEXROLE) +
              " elements, got " + std::to_string(py::len(rhs)) + ". " +
              "Assign a list of " + std::to_string(mjNTEXROLE) + " texture names " +
              "(use empty strings '' for unused slots).");
        }}
        self.{fullvarname}->clear();
        self.{fullvarname}->reserve(mjNTEXROLE);
        for (auto val : rhs) {{
          self.{fullvarname}->push_back(py::cast<std::string>(val));
      }}
    }}, py::return_value_policy::move);"""
    # Default case for other mjStringVec properties
    return f"""\
  {classname}.def_property(
    "{varname}",
    []({rawclassname}& self) -> MjTypeVec<std::string> {{
        return MjTypeVec<std::string>(self.{fullvarname}->data(),
                                      self.{fullvarname}->size());
      }},
    []({rawclassname}& self, py::object rhs) {{
        self.{fullvarname}->clear();
        if (py::isinstance<py::str>(rhs)) {{
          auto s = py::cast<std::string>(rhs);
          if (!s.empty()) {{
            self.{fullvarname}->push_back(std::move(s));
          }}
          return;
        }}
        self.{fullvarname}->reserve(py::len(rhs));
        for (auto val : rhs) {{
          self.{fullvarname}->push_back(py::cast<std::string>(val));
      }}
    }}, py::return_value_policy::move);"""
  elif 'VecVec' in vartype:  # C++ vector of vectors -> Python list of lists
    vartype = vartype.replace('mj', '').replace('VecVec', '').lower()
    return f"""\
  {classname}.def_property(
    "{varname}",
    []({rawclassname}& self) -> py::list {{
        py::list list;
        for (auto inner_vec : *self.{fullvarname}) {{
          py::list inner_list;
          for (auto val : inner_vec) {{
            inner_list.append(val);
          }}
          list.append(inner_list);
        }}
        return list;
      }},
    []({rawclassname}& self, py::object rhs) {{
        self.{fullvarname}->clear();
        self.{fullvarname}->reserve(py::len(rhs));
        for (auto inner_list : rhs) {{
          auto inner_vec = py::cast<std::vector<{vartype}>>(inner_list);
          self.{fullvarname}->push_back(inner_vec);
        }}
    }}, py::return_value_policy::reference_internal);"""

  raise NotImplementedError(
      'Unsupported array type: ' + vartype + ' in ' + classname
  )


def _binding_code(field: ast_nodes.StructFieldDecl, key: str) -> str:
  if isinstance(field.type, ast_nodes.ValueType):
    return _value_binding_code(field.type, key, field.name)
  elif isinstance(field.type, ast_nodes.AnonymousStructDecl):
    return _struct_binding_code(field.type, key, field.name)
  elif isinstance(field.type, ast_nodes.PointerType):
    return _ptr_binding_code(field.type, key, field.name)
  elif isinstance(field.type, ast_nodes.ArrayType):
    return _array_binding_code(field.type, key, field.name)
  return ''


def generate() -> None:
  for key in structs.STRUCTS.keys():
    if (
        key.startswith('mjs')
        or key in ['mjSpec', 'mjOption', 'mjVisual', 'mjStatistic']
    ) and key != 'mjsElement':
      print('\n  // ' + key)
      for field in structs.STRUCTS[key].fields:
        code = _binding_code(field, key)  # pyrefly: ignore[bad-argument-type]
        if code != 'mjsElement':
          print(code)


def generate_add() -> None:
  """Generate add constructors with optional keyword arguments."""
  for key, parent, default, listname, objtype in SPECS_ADD:

    def _field(f: ast_nodes.StructFieldDecl):
      if f.type == ast_nodes.PointerType(
          inner_type=ast_nodes.ValueType(name='mjsElement')
      ):
        return '', '', '', '', '', ''
      elif f.type == ast_nodes.ValueType(name='mjsPlugin'):
        return (
            f'set_plugin(out->{f.name}, plugin);',
            'plugin',
            f.name,
            'MjsPlugin',
            'std::optional<raw::MjsPlugin>& plugin',
            'py::arg("plugin") = py::none()',
        )
      elif f.type == ast_nodes.ValueType(name='mjsOrientation'):
        return (
            (
                f'set_orientation(out->{f.name},'
                f' {"iaxisangle" if f.name == "ialt" else "axisangle"},'
                f' {"ixyaxes" if f.name == "ialt" else "xyaxes"},'
                f' {"izaxis" if f.name == "ialt" else "zaxis"},'
                f' {"ieuler" if f.name == "ialt" else "euler"},'
                f' "{"iaxisangle" if f.name == "ialt" else "axisangle"}",'
                f' "{"ixyaxes" if f.name == "ialt" else "xyaxes"}",'
                f' "{"izaxis" if f.name == "ialt" else "zaxis"}",'
                f' "{"ieuler" if f.name == "ialt" else "euler"}");'
            ),
            'orientation',
            ['iaxisangle', 'ixyaxes', 'izaxis', 'ieuler']
            if f.name == 'ialt'
            else ['axisangle', 'xyaxes', 'zaxis', 'euler'],
            'list[float]',
            [
                f'std::optional<std::vector<double>>& {n}'
                for n in (
                    ['iaxisangle', 'ixyaxes', 'izaxis', 'ieuler']
                    if f.name == 'ialt'
                    else ['axisangle', 'xyaxes', 'zaxis', 'euler']
                )
            ],
            [
                f'py::arg("{n}") = py::none()'
                for n in (
                    ['iaxisangle', 'ixyaxes', 'izaxis', 'ieuler']
                    if f.name == 'ialt'
                    else ['axisangle', 'xyaxes', 'zaxis', 'euler']
                )
            ],
        )
      elif f.type == ast_nodes.PointerType(
          inner_type=ast_nodes.ValueType(name='mjString')
      ):
        return (
            f'set_string(out->{f.name}, {f.name});',
            'string',
            f.name,
            'str',
            f'std::optional<std::string>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )
      elif f.type == ast_nodes.PointerType(
          inner_type=ast_nodes.ValueType(name='mjStringVec')
      ):
        return (
            f'set_str_vec(out->{f.name}, {f.name});',
            'str_vec',
            f.name,
            'list[str]',
            f'std::optional<std::vector<std::string>>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )

      # Handle other vector types
      inner_name = (
          f.type.inner_type.name
          if isinstance(f.type, ast_nodes.PointerType)
          and isinstance(f.type.inner_type, ast_nodes.ValueType)
          else ''
      )

      if inner_name in ('mjIntVec', 'mjByteVec'):
        return (
            f'set_int_vec(out->{f.name}, {f.name});',
            'int_vec',
            f.name,
            'list[int]',
            f'std::optional<std::vector<int>>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )
      elif inner_name in ('mjFloatVec', 'mjDoubleVec'):
        return (
            f'set_vec(out->{f.name}, {f.name});',
            'vec',
            f.name,
            'list[float]',
            f'std::optional<std::vector<double>>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )
      elif inner_name == 'mjIntVecVec':
        return (
            f'set_int_vec_vec(out->{f.name}, {f.name});',
            'int_vec_vec',
            f.name,
            'list[list[int]]',
            f'std::optional<std::vector<std::vector<int>>>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )
      elif inner_name == 'mjFloatVecVec':
        return (
            f'set_float_vec_vec(out->{f.name}, {f.name});',
            'float_vec_vec',
            f.name,
            'list[list[float]]',
            f'std::optional<std::vector<std::vector<double>>>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )

      elif isinstance(f.type, ast_nodes.PointerType):
        return (
            f'set_vec(out->{f.name}, {f.name});',
            'vec',
            f.name,
            'list[float]',
            f'std::optional<std::vector<double>>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )
      elif isinstance(f.type, ast_nodes.ArrayType):
        inner_type = f.type.inner_type.decl()
        if inner_type == 'char':
          return (
              (
                  f'set_char_array(out->{f.name}, {f.name},'
                  f' {f.type.extents[0]}, "{f.name}");'
              ),
              'char_array',
              f.name,
              'str | list[str]',
              f'py::object& {f.name}',
              f'py::arg("{f.name}") = py::none()',
          )
        if key == 'mjsFrame' and f.name == 'scale':
          return (
              (
                  f'set_array_broadcast(out->{f.name}, {f.name},'
                  f' {f.type.extents[0]}, "{f.name}");'
              ),
              'array_broadcast',
              f.name,
              'float | list[float]',
              f'std::optional<py::object>& {f.name}',
              f'py::arg("{f.name}") = py::none()',
          )
        if (
            f.name == 'size'
            and f.type.extents[0] == 3
            or f.name in ('stiffness', 'damping')
        ):
          return (
              (
                  f'set_array_padded(out->{f.name}, {f.name},'
                  f' {f.type.extents[0]}, "{f.name}");'
              ),
              'array_padded',
              f.name,
              'Optional[list[float]]',
              f'std::optional<py::object>& {f.name}',
              f'py::arg("{f.name}") = py::none()',
          )
        return (
            (
                f'set_array(out->{f.name}, {f.name}, {f.type.extents[0]},'
                f' "{f.name}");'
            ),
            'array',
            f.name,
            'list[float]',
            f'std::optional<std::vector<double>>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )
      elif isinstance(f.type, ast_nodes.ValueType):
        type_name = 'float'
        cpp_type = 'double'
        if f.type.name in ('int', 'mjtByte') or f.type.name.startswith('mjt'):
          type_name = 'int'
          cpp_type = 'int'
        return (
            f'set_value(out->{f.name}, {f.name});',
            'value',
            f.name,
            type_name,
            f'std::optional<{cpp_type}>& {f.name}',
            f'py::arg("{f.name}") = py::none()',
        )
      else:
        return '', '', '', '', '', ''

    if key == 'mjsPlugin':
      code_field = ''
      set_types = []
      names = []
      types = []
      cpp_args = []
      py_args = []
    else:
      code_field = 'set_name(out->element, name);'
      set_types = ['name']
      names = ['name']
      types = ['str']
      cpp_args = ['std::optional<std::string>& name']
      py_args = ['py::arg("name") = py::none()']

    for field in structs.STRUCTS[key].fields:
      line, set_type, name, type_name, cpp_arg, py_arg = _field(field)  # pyrefly: ignore[bad-argument-type]
      if line:
        code_field = code_field + '\n        ' + line
        set_types.append(set_type)
        if set_type == 'orientation':
          names.extend(name)
          types.extend([type_name] * len(name))
          cpp_args.extend(cpp_arg)
          py_args.extend(py_arg)
        else:
          names.append(name)
          types.append(type_name)
          cpp_args.append(cpp_arg)
          py_args.append(py_arg)

    # assemble
    elem = key.removeprefix('mjs')
    elemlower = elem.lower()
    titlecase = 'Mjs' + elem

    docstring = f'Add {elemlower} to spec.\n\n      Args:\n'
    for i, name in enumerate(names):
      if i > 0:
        docstring += '\n'
      docstring += f'        {name}: {types[i]}'

    # functions arguments
    args = ', '.join(cpp_args)
    if args:
      args = ', ' + args

    # py::arg definitions
    pyargs = ', '.join(py_args)

    # function definition and call to mjs_add_
    if parent == 'Spec':
      if default:
        code = f"""
          {'mj' + parent}.def("add_{elemlower}", []({'Mj' + parent}& self,
            raw::MjsDefault* default_{args}) -> raw::{titlecase}* {{
            auto out = mjs_add{elem}(self.ptr, default_);
        """
      else:
        code = f"""
          {'mj' + parent}.def("add_{elemlower}", []({'Mj' + parent}& self{args}) -> raw::{titlecase}* {{
            auto out = mjs_add{elem}(self.ptr);
        """
    elif parent == 'Body':
      if key == 'mjsFrame':
        code = f"""
          {'mjs' + parent}.def("add_{elemlower}", []({'raw::Mjs' + parent}& self,
            raw::MjsFrame* parentframe_{args}) -> raw::{titlecase}* {{
            auto out = mjs_add{elem}(&self, parentframe_);
        """
      else:
        code = f"""
          {'mjs' + parent}.def("add_{elemlower}", []({'raw::Mjs' + parent}& self,
            raw::MjsDefault* default_{args}) -> raw::{titlecase}* {{
            auto out = mjs_add{elem}(&self, default_);
        """
    elif parent == 'Frame':
      if key == 'mjsFrame':
        code = f"""
          {'mjs' + parent}.def("add_{elemlower}", []({'raw::Mjs' + parent}& self,
            raw::MjsFrame* parentframe_{args}) -> raw::{titlecase}* {{
            raw::MjsBody* body = mjs_getParent(self.element);
            auto out = mjs_add{elem}(body, &self);
        """
      else:
        code = f"""
          {'mjs' + parent}.def("add_{elemlower}", []({'raw::Mjs' + parent}& self,
            raw::MjsDefault* default_{args}) -> raw::{titlecase}* {{
            raw::MjsBody* body = mjs_getParent(self.element);
            auto out = mjs_add{elem}(body, default_);
            mjs_setFrame(out->element, &self);
        """
    else:
      raise NotImplementedError(f'{parent} parent is not implement.')

    # include helper functions
    if set_types:
      for t in sorted(set(set_types)):
        if t == 'orientation':
          code += """\n
           auto set_orientation = [](raw::MjsOrientation& orientation,
            const std::optional<std::vector<double>>& axisangle,
            const std::optional<std::vector<double>>& xyaxes,
            const std::optional<std::vector<double>>& zaxis,
            const std::optional<std::vector<double>>& euler,
            const char* name_axisangle,
            const char* name_xyaxes,
            const char* name_zaxis,
            const char* name_euler) {
            int nrepresentation = 0;
            nrepresentation += axisangle.has_value();
            nrepresentation += xyaxes.has_value();
            nrepresentation += zaxis.has_value();
            nrepresentation += euler.has_value();

            if (nrepresentation == 0) {
              return;
            } else if (nrepresentation > 1) {
              std::string msg = std::string("Only one of: ") + name_axisangle + ", " + name_xyaxes + ", " + name_zaxis + ", or " + name_euler + " can be set.";
              throw pybind11::value_error(msg);
            }

            auto set_array = [](const std::vector<double>& array, double* des, int size, const char* name) {
              if (array.size() != size) {
                std::string msg = std::string(name) + " should be a list/array of size " + std::to_string(size) + ".";
                throw pybind11::value_error(msg);
              }
              int idx = 0;
              for (auto val : array) {
                des[idx++] = val;
              }
            };
            if (axisangle.has_value()) {
              set_array(axisangle.value(), orientation.axisangle, 4, name_axisangle);
              orientation.type = mjORIENTATION_AXISANGLE;
            } else if (xyaxes.has_value()) {
              set_array(xyaxes.value(), orientation.xyaxes, 6, name_xyaxes);
              orientation.type = mjORIENTATION_XYAXES;
            } else if (zaxis.has_value()) {
              set_array(zaxis.value(), orientation.zaxis, 3, name_zaxis);
              orientation.type = mjORIENTATION_ZAXIS;
            } else if (euler.has_value()) {
              set_array(euler.value(), orientation.euler, 3, name_euler);
              orientation.type = mjORIENTATION_EULER;
            }
          };
          """
        elif t == 'plugin':
          code += """\n
          auto set_plugin = [](raw::MjsPlugin& plugin, const std::optional<raw::MjsPlugin>& input) {
            if (input.has_value()) {
              plugin.name = input->name;
              plugin.plugin_name = input->plugin_name;
              plugin.active = input->active;
              plugin.info = input->info;
            }
          };
          """
        elif t == 'string':
          code += """\n
          auto set_string = [](std::basic_string<char>* des, const std::optional<std::string>& str) {
            if (str.has_value()) {
              *des = str.value();
            }
          };
          """
        elif t == 'str_vec':
          code += """\n
          auto set_str_vec = [](auto&& des, const std::optional<std::vector<std::string>>& vec) {
            if (vec.has_value()) {
              des->clear();
              des->reserve(vec->size());
              for (const auto& val : vec.value()) {
                des->push_back(val);
              }
            }
          };
          """

        elif t == 'int_vec':
          code += """\n
          auto set_int_vec = [](auto&& des, const std::optional<std::vector<int>>& vec) {
            if (vec.has_value()) {
              using T = typename std::decay_t<decltype(*des)>::value_type;
              des->clear();
              des->reserve(vec->size());
              for (auto val : vec.value()) {
                des->push_back(static_cast<T>(val));
              }
            }
          };
          """
        elif t == 'int_vec_vec':
          code += """\n
          auto set_int_vec_vec = [](auto&& des, const std::optional<std::vector<std::vector<int>>>& vec) {
            if (vec.has_value()) {
               des->clear();
               des->reserve(vec->size());
               for (const auto& inner : vec.value()) {
                 using InnerT = typename std::decay_t<decltype(*des)>::value_type;
                 InnerT inner_res;
                 inner_res.reserve(inner.size());
                 using ValT = typename InnerT::value_type;
                 for (auto val : inner) {
                   inner_res.push_back(static_cast<ValT>(val));
                 }
                 des->push_back(inner_res);
               }
            }
          };
          """
        elif t == 'float_vec_vec':
          code += """\n
          auto set_float_vec_vec = [](auto&& des, const std::optional<std::vector<std::vector<double>>>& vec) {
            if (vec.has_value()) {
               des->clear();
               des->reserve(vec->size());
               for (const auto& inner : vec.value()) {
                 using InnerT = typename std::decay_t<decltype(*des)>::value_type;
                 InnerT inner_res;
                 inner_res.reserve(inner.size());
                 using ValT = typename InnerT::value_type;
                 for (auto val : inner) {
                   inner_res.push_back(static_cast<ValT>(val));
                 }
                 des->push_back(inner_res);
               }
            }
          };
          """

        elif t == 'vec':
          code += """\n
          auto set_vec = [](auto&& des, const std::optional<std::vector<double>>& vec) {
            if (vec.has_value()) {
              using T = typename std::decay_t<decltype(*des)>::value_type;
              des->clear();
              des->reserve(vec->size());
              for (auto val : vec.value()) {
                des->push_back(static_cast<T>(val));
              }
            }
          };
          """
        elif t == 'array':
          code += """\n
          auto set_array = [](auto&& des, const std::optional<std::vector<double>>& array, int size, const char* name) {
            if (array.has_value()) {
              if (array->size() != size) {
                std::string msg = std::string(name) + " should be a list/array of size " + std::to_string(size) + ".";
                throw pybind11::value_error(msg);
              }
              int idx = 0;
              for (auto val : array.value()) {
                des[idx++] = val;
              }
            }
          };
          """
        elif t == 'array_broadcast':
          code += """\n
          auto set_array_broadcast = [](auto&& des, const std::optional<py::object>& obj, int size, const char* name) {
            if (obj.has_value() && !obj->is_none()) {
              std::vector<double> array;
              try {
                array = py::cast<std::vector<double>>(*obj);
              } catch (const py::cast_error&) {
                array.assign(size, py::cast<double>(*obj));
              }
              if (array.size() != size) {
                std::string msg = std::string(name) + " should be a scalar or a list/array of size " + std::to_string(size) + ".";
                throw pybind11::value_error(msg);
              }
              for (int i = 0; i < size; i++) {
                des[i] = array[i];
              }
            }
          };
          """
        elif t == 'array_padded':
          code += """\n
          auto set_array_padded = [](auto&& des, const std::optional<py::object>& obj, int size, const char* name) {
            if (obj.has_value() && !obj->is_none()) {
              std::vector<double> array;
              if (py::isinstance<py::int_>(*obj) || py::isinstance<py::float_>(*obj)) {
                array.push_back(py::cast<double>(*obj));
              } else if (py::isinstance<py::str>(*obj)) {
                throw pybind11::type_error(
                    std::string(name) + " should be a numeric scalar or list.");
              } else {
                try {
                  array = py::cast<std::vector<double>>(*obj);
                } catch (const py::cast_error&) {
                  throw pybind11::type_error(
                      std::string(name) + " should be a numeric scalar or list.");
                }
              }
              if (array.empty() || array.size() > static_cast<size_t>(size)) {
                std::string msg = std::string(name) + " should be a list/array of size 1 to " + std::to_string(size) + ".";
                throw pybind11::value_error(msg);
              }
              for (int i = 0; i < size; i++) {
                des[i] = (i < static_cast<int>(array.size())) ? array[i] : 0;
              }
            }
          };
          """
        elif t == 'char_array':
          code += """\n
          auto set_char_array = [](auto&& des, py::object& obj, int size, const char* name) {
            if (obj.is_none()) {
              return;
            }
            std::string chars;
            if (py::isinstance<py::str>(obj)) {
              chars = py::cast<std::string>(obj);
            } else if (py::isinstance<py::list>(obj)) {
              py::list list = py::cast<py::list>(obj);
              chars.reserve(py::len(list));
              for (auto item : list) {
                std::string s = py::cast<std::string>(item);
                if (s.size() != 1) {
                  throw pybind11::value_error(std::string(name) + " list elements must be single characters.");
                }
                chars.push_back(s[0]);
              }
            } else {
              throw pybind11::type_error(std::string(name) + " must be a string or a list of single-character strings.");
            }
            if (chars.size() != size) {
              std::string msg = std::string(name) + " should have length " + std::to_string(size) + ", got " + std::to_string(chars.size()) + ".";
              throw pybind11::value_error(msg);
            }
            int idx = 0;
            for (char val : chars) {
              des[idx++] = val;
            }
          };
          """
        elif t == 'value':
          code += """\n
          auto set_value = [](auto&& des, auto&& val) {
            if (val.has_value()) {
              using T = std::decay_t<decltype(des)>;
              des = static_cast<T>(val.value());
            }
          };
          """
        elif t == 'name':
          code += """\n
          auto set_name = [](raw::MjsElement* el, const std::optional<std::string>& name) {
            if (name.has_value()) {
              if (mjs_setName(el, name->c_str())) {
                throw pybind11::value_error(mjs_getError(mjs_getSpec(el)));
              }
            }
          };
          """
        else:
          raise NotImplementedError(f'Unsupported set type: {t} in {key}')

    code += code_field
    code += f"""\n
        return out;
      }},
      {'py::arg_v("default", nullptr)' + (', ' if pyargs else '') if default else ''}
      {pyargs}{', ' if pyargs else ''}
      R"mydelimiter(
      {docstring}
      )mydelimiter",
      py::return_value_policy::reference_internal);
    """

    code += f"""\n
      mjSpec.def_property_readonly(
        "{listname}",
        [](MjSpec& self) -> py::list {{
          py::list list;
          raw::MjsElement* el = mjs_firstElement(self.ptr, {objtype});
          while (el) {{
            list.append(mjs_as{key[3:]}(el));
            el = mjs_nextElement(self.ptr, el);
          }}
          return list;
        }},
        py::return_value_policy::reference_internal);
    """

    print(code)


def generate_find() -> None:
  """Generate find functions."""
  for key, _, _, _, objtype in SPECS:
    elem = key.removeprefix('mjs')
    elemlower = elem.lower()
    titlecase = 'Mjs' + elem
    code = f"""\n
      mjSpec.def("{elemlower}",
      [](MjSpec& self, std::string& name) -> raw::{titlecase}* {{
        return mjs_as{elem}(
            mjs_findElement(self.ptr, {objtype}, name.c_str()));
      }}, py::return_value_policy::reference_internal);
    """
    print(code)


def generate_signature() -> None:
  """Generate signature functions."""
  for key, _, _, _, _ in SPECS:
    elem = key.removeprefix('mjs')
    titlecase = 'Mjs' + elem
    code = f"""\n
      {key}.def_property_readonly("signature",
      [](raw::{titlecase}& self) -> uint64_t {{
        return mjs_getSpec(self.element)->element->signature;
      }});
    """
    print(code)


def generate_id() -> None:
  """Generate id functions."""
  for key, _, _, _, _ in SPECS:
    if key == 'mjsPlugin':
      continue
    elem = key.removeprefix('mjs')
    titlecase = 'Mjs' + elem
    code = f"""\n
      {key}.def_property_readonly("id",
      [](raw::{titlecase}& self) -> int {{
        return mjs_getId(self.element);
      }});
    """
    print(code)


def generate_name() -> None:
  """Generate name functions."""
  for key, _, _, _, _ in SPECS + [('mjsDefault', '', '', '', '')]:
    if key == 'mjsPlugin':
      continue
    elem = key.removeprefix('mjs')
    titlecase = 'Mjs' + elem
    code = f"""\n
      {key}.def_property("name",
      [](raw::{titlecase}& self) -> std::string* {{
        return mjs_getName(self.element);
      }},
      [](raw::{titlecase}& self, std::string& name) -> void {{
        if (mjs_setName(self.element, name.c_str())) {{
          throw pybind11::value_error(mjs_getError(mjs_getSpec(self.element)));
        }}
      }}, py::return_value_policy::reference_internal);
    """
    print(code)


def generate_compiler() -> None:
  """Generate compiler property for all spec element types."""
  for key, _, _, _, _ in SPECS:
    elem = key.removeprefix('mjs')
    titlecase = 'Mjs' + elem
    code = f"""\n
      {key}.def_property_readonly("compiler",
      [](raw::{titlecase}& self) -> raw::MjsCompiler& {{
        ::mjsCompiler* compiler = mjs_getCompiler(self.element);
        if (!compiler) {{
          throw pybind11::value_error("Element is not attached to a spec.");
        }}
        return *compiler;
      }}, py::return_value_policy::reference_internal);
    """
    print(code)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  generate()
  generate_add()
  generate_find()
  generate_signature()
  generate_id()
  generate_name()
  generate_compiler()


if __name__ == '__main__':
  app.run(main)
