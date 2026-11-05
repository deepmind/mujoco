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

"""Generates Embind bindings for MuJoCo structs."""

import collections
import dataclasses
import math
from typing import Tuple, Union, cast

from codegen import code_builder
from introspect import ast_nodes
from introspect import functions as introspect_functions
from introspect import structs as introspect_structs

from wasm.codegen.generators import common
from wasm.codegen.generators import constants


@dataclasses.dataclass
class WrappedFieldData:
  """Data class for struct field definition and binding."""

  # Line for struct field binding
  binding: str = ""

  # Line for struct field declaration
  declaration: str = ""

  # Used for constructor fields that are class types and need to be initialized
  # with a pointer to the corresponding member in the native struct
  # (e.g., "alt(&ptr_->alt)").
  ptr_initialization: str = ""

  # Statement to reset the inner pointer when copying the field
  ptr_copy_reset: str = ""

  # Whether the field is a primitive or fixed size
  is_primitive_or_fixed_size: bool = False

  # Underlying type of the field. If non-empty, used to determine the order in
  # which structs are written in the bindings.h file.
  typename: str = ""


def _get_property_binding(
    field: ast_nodes.StructFieldDecl,
    struct_wrapper_name: str,
    setter: bool = False,
    reference: bool = False,
) -> str:
  """Builds the C++ code for a simple property binding."""
  f = field
  w = struct_wrapper_name
  setter_txt = f", &{w}::set_{f.name}" if setter else ""
  reference_txt = ", reference()" if reference else ""
  return f'.property("{f.name}", &{w}::{f.name}{setter_txt}{reference_txt})'


def _generate_field_data(
    field: ast_nodes.StructFieldDecl, struct_wrapper_name: str
) -> WrappedFieldData:
  """Generates the C++ definition and binding code for the struct field."""
  f = field
  w = struct_wrapper_name
  s = common.decapitalize(w)

  if f.name in constants.MANUAL_FIELDS.get(w, []):
    # Note: Manually handled MjModel fields are special cased so that a
    # by-reference embind return value policy is used.
    return WrappedFieldData(
        typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
        declaration=f"// {f.name} field is handled manually in template file struct declaration",  # pylint: disable=line-too-long
        binding=_get_property_binding(f, w, reference=(w == "MjModel")),
    )

  if f.name in constants.SKIPPED_FIELDS.get(w, []):
    return WrappedFieldData(
        typename="",
        declaration=f"// {f.name} field is skipped.",
        binding=f"// {f.name} field is skipped.",
    )

  if isinstance(f.type, ast_nodes.ValueType) and (
      f.type.name in constants.PRIMITIVE_TYPES or f.type.name.startswith("mjt")
  ):

    builder = code_builder.CodeBuilder()

    # `mjtSize` is a 64-bit signed integer type and would be represented in
    # JS/TS a `bigint` because it doesn't fit in the number primitive (a 64-bit
    # float). By casting int, we avoid "TS2345: Argument of type 'bigint' is not
    # assignable to parameter of type 'number'" errors.
    if f.type.name == "mjtSize":
      with builder.function(f"int {f.name}() const"):
        builder.line(f"return static_cast<int>(ptr_->{f.name});")
      with builder.function(f"void set_{f.name}(int value)"):
        builder.line(f"ptr_->{f.name} = static_cast<mjtSize>(value);")
    else:
      with builder.function(f"{f.type.name} {f.name}() const"):
        builder.line(f"return ptr_->{f.name};")
      with builder.function(f"void set_{f.name}({f.type.name} value)"):
        builder.line(f"ptr_->{f.name} = value;")

    return WrappedFieldData(
        declaration=builder.to_string(),
        typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
        binding=_get_property_binding(f, w, setter=True),
        is_primitive_or_fixed_size=True,
    )

  elif isinstance(f.type, ast_nodes.ValueType) and f.type.name.startswith("mj"):
    return WrappedFieldData(
        declaration=f"{common.capitalize(f.type.name)} {f.name};",
        typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
        binding=_get_property_binding(f, w, setter=False, reference=True),
        ptr_initialization=f"{f.name}(&ptr_->{f.name})",
        ptr_copy_reset=f"{f.name}.set(&ptr_->{f.name});",
        is_primitive_or_fixed_size=True,
    )

  elif isinstance(f.type, ast_nodes.AnonymousStructDecl):
    anonymous_struct_name = _get_field_struct_type(f, s)
    return WrappedFieldData(
        binding=_get_property_binding(f, w, setter=False, reference=True),
        typename=anonymous_struct_name,  # pyrefly: ignore[bad-argument-type]
        declaration=(
            f"{common.wrapped_struct_name(anonymous_struct_name)} {f.name};"  # pyrefly: ignore[bad-argument-type]
        ),
        ptr_initialization=f"{f.name}(&ptr_->{f.name})",
        ptr_copy_reset=f"{f.name}.set(&ptr_->{f.name});",
        is_primitive_or_fixed_size=True,
    )

  elif isinstance(f.type, ast_nodes.ArrayType):

    inner_type = f.type.inner_type
    size = math.prod(f.type.extents)
    inner_type_name = (
        f.type.inner_type.name
        if isinstance(f.type.inner_type, ast_nodes.ValueType)
        else ""
    )
    if (
        isinstance(inner_type, ast_nodes.ValueType)
        and inner_type.name in constants.PRIMITIVE_TYPES
    ):
      ptr_expr = f"ptr_->{f.name}"
      if inner_type.name == "mjtBool":
        # Embind has no memory_view<bool>, so expose mjtBool as Uint8Array.
        ptr_expr = f"reinterpret_cast<uint8_t*>({ptr_expr})"
      elif len(f.type.extents) > 1:
        # for multi-dimensional arrays, we need to cast the field
        # to a pointer, so embind can correctly interpret the memory
        # view
        ptr_expr = f"reinterpret_cast<{inner_type.name}*>({ptr_expr})"

      builder = code_builder.CodeBuilder()
      if inner_type.name == "mjtSize":
        with builder.function(f"emscripten::val {f.name}() const"):
          builder.line(f"return ToInt32Array({ptr_expr}, {str(size)});")
      else:
        with builder.function(f"emscripten::val {f.name}() const"):
          builder.line(
              "return"
              f" emscripten::val(emscripten::typed_memory_view({str(size)},"
              f" {ptr_expr}));"
          )

      return WrappedFieldData(
          declaration=builder.to_string(),
          typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
          binding=_get_property_binding(f, w),
          is_primitive_or_fixed_size=True,
      )
    elif inner_type_name.startswith("mj"):
      return WrappedFieldData(
          declaration=(
              f"std::vector<{common.capitalize(inner_type_name)}> {f.name};"
          ),
          ptr_initialization=f"{f.name}(&ptr_->{f.name})",
          typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
          binding=_get_property_binding(f, w, reference=True),
      )

  elif isinstance(f.type, ast_nodes.PointerType):
    inner_type_name = (
        f.type.inner_type.name
        if isinstance(f.type.inner_type, ast_nodes.ValueType)
        else ""
    )
    is_dynamically_sized = bool(f.array_extent)

    # Case 1: mjString
    if inner_type_name == "mjString":
      builder = code_builder.CodeBuilder()
      with builder.function(f"mjString {f.name}() const"):
        builder.line(
            f'return (ptr_ && ptr_->{f.name}) ? *(ptr_->{f.name}) : "";'
        )
      with builder.function(f"void set_{f.name}(const mjString& value)"):
        with builder.block(f"if (ptr_ && ptr_->{f.name})"):
          builder.line(f"*(ptr_->{f.name}) = value;")
      return WrappedFieldData(
          declaration=builder.to_string(),
          typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
          binding=_get_property_binding(f, w, setter=True, reference=True),
      )

    # Case 2: mj*Vec
    elif inner_type_name.startswith("mj") and inner_type_name.endswith("Vec"):
      ptr_field_expr_vec = f"*(ptr_->{f.name})"
      vector_type = inner_type_name
      if vector_type == "mjByteVec":
        vector_type = "std::vector<uint8_t>"
        ptr_field_expr_vec = (
            f"*(reinterpret_cast<std::vector<uint8_t>*>(ptr_->{f.name}))"
        )
      builder = code_builder.CodeBuilder()
      with builder.function(f"{vector_type} &{f.name}() const"):
        builder.line(f"return {ptr_field_expr_vec};")
      return WrappedFieldData(
          declaration=builder.to_string(),
          typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
          binding=_get_property_binding(f, w, setter=False, reference=True),
      )

    # Case 2.5: const char* without array_extent (C strings like body, func).
    elif (
        inner_type_name == "char"
        and not is_dynamically_sized
    ):
      builder = code_builder.CodeBuilder()
      if f.type.inner_type.is_const:  # pyrefly: ignore[missing-attribute]
        with builder.function(f"std::string {f.name}() const"):
          builder.line(
              f'return ptr_->{f.name} ? std::string(ptr_->{f.name}) : "";'
          )
      else:
        with builder.function(f"std::string {f.name}() const"):
          builder.line(
              f'return ptr_->{f.name} ? std::string(ptr_->{f.name}) : "";'
          )
        with builder.function(f"void set_{f.name}(const std::string& value)"):
          with builder.block(f"if (ptr_->{f.name})"):
            builder.line(
                f"std::strncpy(ptr_->{f.name}, value.c_str(),"
                f" sizeof(ptr_->{f.name}));"
            )
      return WrappedFieldData(
          declaration=builder.to_string(),
          typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
          binding=_get_property_binding(
              f, w, setter=not f.type.inner_type.is_const, reference=True
          ),
      )

    # Case 3: Non-dynamically sized pointer fields to other structs.
    elif (
        not is_dynamically_sized
        and inner_type_name not in constants.PRIMITIVE_TYPES
        and w not in constants.MANUAL_FIELDS.keys()
    ):
      # These are wrapped as direct members of the wrapper class, initialized
      # with a pointer to the corresponding member in the native struct.
      ptr_field = cast(ast_nodes.PointerType, f.type)
      wrapper_field_name = common.capitalize(
          cast(ast_nodes.ValueType, ptr_field.inner_type).name
      )
      return WrappedFieldData(
          declaration=f"{wrapper_field_name} {f.name};",
          typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
          binding=_get_property_binding(f, w, setter=False, reference=True),
          ptr_initialization=f"{f.name}(ptr_->{f.name})",
      )

    # Case 4: Dynamically sized pointer fields and other pointer types that are
    # exposed as emscripten::typed_memory_view.
    else:
      ptr_field_expr = f"ptr_->{f.name}"
      if inner_type_name == "mjtBool":
        # Embind has no memory_view<bool>, so expose mjtBool as Uint8Array.
        ptr_field_expr = f"reinterpret_cast<uint8_t*>({ptr_field_expr})"
      array_size_str = ""
      if is_dynamically_sized:
        array_size_str = parse_array_extent(f.array_extent, w, f.name)  # pyrefly: ignore[bad-argument-type]
      elif f.name in constants.BYTE_FIELDS.keys():
        # For byte fields, we need to cast the pointer to uint8_t*
        # so embind can correctly interpret the memory view
        ptr_field_expr = f"static_cast<uint8_t*>({ptr_field_expr})"
        # Byte fields lack `array_extent`, so their size is defined in
        # `constants.BYTE_FIELDS` based on the MuJoCo documentation.
        extent = (constants.BYTE_FIELDS[f.name]["size"],)
        array_size_str = parse_array_extent(extent, w, f.name)

      builder = code_builder.CodeBuilder()
      if inner_type_name == "mjtSize":
        # A typed_memory_view over `mjtSize` (int64_t / size_t) would surface as
        # a BigInt64Array in JS, whose elements cannot be mixed with JS numbers.
        # The WASM heap is 32-bit addressed so values always fit an int; return an
        # Int32Array copy, consistent with scalar mjtSize field handling above.
        with builder.function(f"emscripten::val {f.name}() const"):
          builder.line(
              f"return ToInt32Array({ptr_field_expr}, {array_size_str});"
          )
      else:
        with builder.function(f"emscripten::val {f.name}() const"):
          builder.line(
              "return"
              f" emscripten::val(emscripten::typed_memory_view({array_size_str},"
              f" {ptr_field_expr}));"
          )
      return WrappedFieldData(
          declaration=builder.to_string(),
          typename=_get_field_struct_type(f, s),  # pyrefly: ignore[bad-argument-type]
          binding=_get_property_binding(f, w),
      )

  raise RuntimeError(f"Field {f.name} from struct {w} not properly handled")


def _default_function_statement(struct_name: str) -> str:
  """Returns the default function name for the given struct."""
  if struct_name == "mjvGeom":
    f = "mjv_initGeom(ptr_, mjGEOM_NONE, nullptr, nullptr, nullptr, nullptr);"
    return f
  elif struct_name in constants.ANONYMOUS_STRUCTS.keys():
    return ""
  elif struct_name in constants.NO_DEFAULT_CONSTRUCTORS:
    return ""
  elif struct_name.startswith("mjs"):
    return f"mjs_default{struct_name.removeprefix('mjs')}(ptr_);"
  elif struct_name.startswith("mjv"):
    return f"mjv_default{struct_name.removeprefix('mjv')}(ptr_);"
  elif struct_name.startswith("mj"):
    return f"mj_default{struct_name.removeprefix('mj')}(ptr_);"

  return ""


def _delete_ptr_statement(struct_name: str) -> str:
  """Returns the delete function name for the given struct."""
  if struct_name == "mjVFS":
    return "mj_deleteVFS(ptr_);"

  return "delete ptr_;"


def _find_member_inits(
    wrapped_fields: list[WrappedFieldData],
) -> list[WrappedFieldData]:
  """Finds the fields with ptr_initialization in the wrapped fields list."""
  member_inits = []
  for field in wrapped_fields:
    if field.ptr_initialization:
      member_inits.append(field)
  return member_inits


def use_shallow_copy(
    wrapped_fields: list[WrappedFieldData],
) -> bool:
  """Returns true if the struct fields can be shallow copied."""
  for field in wrapped_fields:
    if not field.is_primitive_or_fixed_size:
      return False
  return True


def build_struct_header(
    struct_name: str,
    wrapped_fields: list[WrappedFieldData],
):
  """Builds the C++ header file code for a struct."""
  s = struct_name
  w = common.wrapped_struct_name(s)

  if (
      s not in constants.ANONYMOUS_STRUCTS
      and s not in introspect_structs.STRUCTS
  ):
    raise RuntimeError(f"Struct {s} not found in introspect structs")

  not_mjs = not w.startswith("Mjs")
  member_inits = _find_member_inits(wrapped_fields)
  shallow_copy = not_mjs and use_shallow_copy(wrapped_fields)
  builder = code_builder.CodeBuilder()

  with builder.struct(f"{w}"):
    # destructor
    if not_mjs:
      builder.line(f"~{w}();")

    # default constructor
    if not_mjs and w not in ["MjData", "MjModel"]:
      builder.line(f"{w}();")

    # constructor passing native ptr
    if w != "MjData":
      builder.line(f"explicit {w}({s} *ptr);")
    else:
      # special handling for MjData
      builder.line("MjData(MjModel *m);")
      builder.line("explicit MjData(const MjModel &, const MjData &);")
      builder.line("std::vector<MjContact> contact() const;")

    # copy constructor
    if shallow_copy or w in ["MjSpec", "MjModel"]:
      builder.line(f"{w}(const {w} &);")

    # assignment operator
    if shallow_copy or w == "MjSpec":
      builder.line(f"{w} &operator=(const {w} &);")

    # explicit copy function
    if shallow_copy or w in ["MjSpec", "MjData", "MjModel"]:
      builder.line(f"std::unique_ptr<{w}> copy();")

    # C struct getter/setter
    builder.line(f"{s}* get() const;")
    builder.line(f"void set({s}* ptr);")

    if w == "MjSpec":
      builder.line("emscripten::val timer() const;")

    # field declarations
    for field in wrapped_fields:
      if field.declaration and field not in member_inits:
        for line in field.declaration.splitlines():
          builder.line(line)

    # accessors declarations
    if w == "MjModel":
      builder.line("""
  // Generates functions to return accessor classes.
  #define X_ACCESSOR(NAME, Name, OBJTYPE, accessor_name, nfield)               \\
    MjModel##Name##Accessor accessor_name(const NumberOrString& val) const {   \\
      if (val.isString()) {                                                    \\
        int id = mj_name2id(ptr_, OBJTYPE, val.as<std::string>().c_str());     \\
        if (id == -1) {                                                        \\
          mju_error("%s", KeyErrorMessage(ptr_, OBJTYPE, ptr_->nfield, val.as<std::string>(), #accessor_name).c_str());  \\
        }                                                                      \\
        return MjModel##Name##Accessor(ptr_, id);                              \\
      } else if (val.isNumber()) {                                             \\
        int id = val.as<int>();                                                \\
        if (id < 0 || id >= ptr_->nfield) {                                    \\
          mju_error("%s", IndexErrorMessage(id, ptr_->nfield, #accessor_name).c_str());  \\
        }                                                                      \\
        return MjModel##Name##Accessor(ptr_, id);                              \\
      } else {                                                                 \\
        mju_error(#accessor_name "() argument must be a string or number");    \\
        return MjModel##Name##Accessor(nullptr, 0);                            \\
      }                                                                        \\
    }

    MJMODEL_ACCESSORS
  #undef X_ACCESSOR""".lstrip())
    elif w == "MjData":
      builder.line("""
  // Generates functions to return accessor classes.
  #define X_ACCESSOR(NAME, Name, OBJTYPE, accessor_name, nfield)              \\
    MjData##Name##Accessor accessor_name(const NumberOrString& val) const {   \\
      if (val.isString()) {                                                   \\
        int id = mj_name2id(model, OBJTYPE, val.as<std::string>().c_str());   \\
        if (id == -1) {                                                       \\
          mju_error("%s", KeyErrorMessage(model, OBJTYPE, model->nfield, val.as<std::string>(), #accessor_name).c_str());  \\
        }                                                                     \\
        return MjData##Name##Accessor(ptr_, model, id);                       \\
      } else if (val.isNumber()) {                                            \\
        int id = val.as<int>();                                               \\
        if (id < 0 || id >= model->nfield) {                                  \\
          mju_error("%s", IndexErrorMessage(id, model->nfield, #accessor_name).c_str());  \\
        }                                                                     \\
        return MjData##Name##Accessor(ptr_, model, id);                       \\
      } else {                                                                \\
        mju_error(#accessor_name "() argument must be a string or number");   \\
        return MjData##Name##Accessor(nullptr, nullptr, 0);                   \\
      }                                                                       \\
    }
    MJDATA_ACCESSORS
  #undef X_ACCESSOR""".lstrip())

    # define private struct members
    builder.private()
    builder.line(f"{s}* ptr_;")
    if not_mjs and w not in ["MjData", "MjModel"]:
      builder.line("bool owned_ = false;")

    # define public struct members
    if member_inits:
      builder.public()
      for f in member_inits:
        if f.declaration:
          builder.line(f"{f.declaration}")

      if w == "MjData":
        builder.line("mjModel *model;")

  return builder.to_string() + ";"


def build_struct_source(
    struct_name: str,
    wrapped_fields: list[WrappedFieldData],
):
  """Builds the C++ .cc file code for a struct."""
  s = struct_name
  w = common.wrapped_struct_name(s)
  is_mjs = w.startswith("Mjs")

  member_inits = _find_member_inits(wrapped_fields)
  shallow_copy = use_shallow_copy(wrapped_fields)

  fields_init = ""
  if member_inits:
    fields_init = ", " + ", ".join(
        mjs_field.ptr_initialization for mjs_field in member_inits
    )

  builder = code_builder.CodeBuilder()

  # constructor passing native ptr
  with builder.function(f"{w}::{w}({s} *ptr) : ptr_(ptr){fields_init}"):
    pass

  # destructor
  if not is_mjs:
    with builder.function(f"{w}::~{w}()"):
      with builder.block("if (owned_ && ptr_)"):
        delete_ptr = _delete_ptr_statement(s)
        builder.line(delete_ptr)

  if not is_mjs:
    # default constructor
    with builder.function(f"{w}::{w}() : ptr_(new {s}()){fields_init}"):
      builder.line("owned_ = true;")
      default_func = _default_function_statement(s)
      if default_func:
        builder.line(default_func)

  if shallow_copy and not is_mjs:
    # copy constructor
    with builder.function(f"{w}::{w}(const {w} &other) : {w}()"):
      builder.line("*ptr_ = *other.get();")
      for f in member_inits:
        if f.ptr_copy_reset is not None:
          builder.line(f.ptr_copy_reset)

    # assignment operator
    with builder.function(f"{w}& {w}::operator=(const {w} &other)"):
      with builder.block("if (this == &other)"):
        builder.line("return *this;")
      builder.line("*ptr_ = *other.get();")
      for f in member_inits:
        if f.ptr_copy_reset is not None:
          builder.line(f.ptr_copy_reset)
      builder.line("return *this;")

    # explicit copy function
    with builder.function(f"std::unique_ptr<{w}> {w}::copy()"):
      builder.line(f"return std::make_unique<{w}>(*this);")

  # C struct getter/setter
  with builder.function(f"{s}* {w}::get() const"):
    builder.line("return ptr_;")
  with builder.function(f"void {w}::set({s}* ptr)"):
    builder.line("ptr_ = ptr;")

  return builder.to_string()


def _build_struct_bindings(
    struct_name: str,
    wrapped_fields: list[WrappedFieldData],
):
  """Builds the C++ bindings for a struct."""
  w = common.wrapped_struct_name(struct_name)
  is_mjs = w.startswith("Mjs")

  builder = code_builder.CodeBuilder()
  with builder.block(
      header_line=f'emscripten::class_<{w}>("{w}")', braces=False
  ):
    if w == "MjData":
      builder.line(".constructor<MjModel *>()")
      builder.line(".constructor<const MjModel &, const MjData &>()")
      builder.line("""
    // Binds the functions on MjData that return accessors.
    #define X_ACCESSOR(NAME, Name, OBJTYPE, field_name, nfield) \\
      .function(#field_name, &MjData::field_name)
      MJDATA_ACCESSORS
    #undef X_ACCESSOR""".lstrip())
    elif w == "MjModel":
      builder.line(
          "// mj_loadXML is deprecated and will be removed in a future release"
      )
      builder.line(
          '.class_function("mj_loadXML",'
          " emscripten::select_overload<std::unique_ptr<MjModel>(std::string)>(&mj_loadXML_wrapper_1))"
      )
      builder.line(
          '.class_function("mj_loadXML",'
          " emscripten::select_overload<std::unique_ptr<MjModel>(std::string,"
          " const MjVFS&)>(&mj_loadXML_wrapper_2))"
      )
      f2 = common.wrapped_function_name(
          introspect_functions.FUNCTIONS["mj_loadModel"]
      )
      builder.line(
          "// mj_loadModel is deprecated and will be removed in a future"
          " release"
      )
      builder.line(f'.class_function("mj_loadModel", &{f2})')
      builder.line(f'.class_function("from_binary_path", &{f2})')
      builder.line(
          '.class_function("from_xml_string",'
          " emscripten::select_overload<std::unique_ptr<MjModel>(const"
          " std::string&)>(&from_xml_string_wrapper_1))"
      )
      builder.line(
          '.class_function("from_xml_string",'
          " emscripten::select_overload<std::unique_ptr<MjModel>(const"
          " std::string&, const MjVFS&)>(&from_xml_string_wrapper_2))"
      )
      builder.line(
          '.class_function("from_xml_path",'
          " emscripten::select_overload<std::unique_ptr<MjModel>(std::string)>(&mj_loadXML_wrapper_1))"
      )
      builder.line(
          '.class_function("from_xml_path",'
          " emscripten::select_overload<std::unique_ptr<MjModel>(std::string,"
          " const MjVFS&)>(&mj_loadXML_wrapper_2))"
      )
      builder.line(".constructor<const MjModel &>()")
      builder.line("""
  // Binds the functions on MjModel that return accessors.
  #define X_ACCESSOR(NAME, Name, OBJTYPE, field_name, nfield) \\
    .function(#field_name, &MjModel::field_name)
    MJMODEL_ACCESSORS
  #undef X_ACCESSOR""".lstrip())
    elif w == "MjSpec":
      builder.line(".constructor<const MjSpec &>()")
      builder.line('.property("timer", &MjSpec::timer)')
    elif w == "MjvScene":
      builder.line(".constructor<MjModel *, int>()")
      builder.line(".constructor<>()")
    elif not is_mjs:
      builder.line(".constructor<>()")

    shallow_copy = use_shallow_copy(wrapped_fields)
    if shallow_copy and not is_mjs:
      builder.line(f'.function("copy", &{w}::copy, take_ownership())')

    bindings = sorted(
        [f.binding for f in wrapped_fields if f.binding.startswith(".")]
    )
    for binding in bindings[:-1]:
      builder.line(binding)
    if bindings:
      builder.line(f"{bindings[-1]};")
    elif builder._lines:
      builder._lines[-1] += ";"

  return builder.to_string()


def parse_array_extent(
    extents: Tuple[Union[str, int], ...], wrapper_name: str, field_name: str
) -> str:
  """Parses the array extent of a field, returning a string representing the resolved extents."""
  if not extents:
    return ""
  return " * ".join(
      resolve_extent(extent, wrapper_name, field_name) for extent in extents
  )


def resolve_extent(
    extent: Union[str, int], wrapper_name: str, field_name: str
) -> str:
  """Resolves the extent of an array, handling integers and references to other struct fields.

  Args:
      extent: The extent to resolve, can be an int or a string referencing a
        field.
      wrapper_name: The name of the struct wrapper.
      field_name: The name of the field being processed.

  Returns:
      A string representing the resolved extent, either as a number or a field
      reference.
  """
  if isinstance(extent, int):
    return str(extent)
  # if starts with mj, it's a mujoco constant,
  # so we don't need to get a parent struct ptr
  if extent.startswith("mj"):
    return str(extent)
  if wrapper_name == "MjData" and field_name not in constants.MJDATA_SIZES:
    var_name = "model"
  else:
    var_name = "ptr_"
  return f"{var_name}->{extent}"


def _get_anonymous_struct_field(
    anonymous_structs_key: str,
) -> ast_nodes.StructFieldDecl | None:
  """Looks up the given key in the anonymous_structs dict and generates bindings for its fields."""
  info = constants.ANONYMOUS_STRUCTS[anonymous_structs_key]
  parent_decl = introspect_structs.STRUCTS[info["parent"]]
  target_field = next(
      (
          f
          for f in parent_decl.fields
          if hasattr(f, "name")
          and f.name == info["field_name"]
          and hasattr(f, "type")
          and isinstance(f.type, ast_nodes.AnonymousStructDecl)
      ),
      None,
  )
  return target_field  # pyrefly: ignore[bad-return]


def _get_field_struct_type(
    field: ast_nodes.StructFieldDecl, struct_name: str
) -> str | None:
  """Extracts the base struct name if the field type is a struct or pointer to a struct."""
  s = struct_name
  w = common.wrapped_struct_name(s)
  if isinstance(field.type, ast_nodes.AnonymousStructDecl):
    anonymous_struct_name = ""
    for name, value in constants.ANONYMOUS_STRUCTS.items():
      if value["parent"] == s and value["field_name"] == field.name:
        anonymous_struct_name = name
        break

    if not anonymous_struct_name:
      raise RuntimeError(
          f"Anonymous struct for field {field.name} in {w} not found in"
          " ANONYMOUS_STRUCTS."
      )
    return anonymous_struct_name
  elif isinstance(field.type, ast_nodes.ValueType):
    return field.type.name
  if isinstance(field.type, ast_nodes.PointerType):
    if isinstance(field.type.inner_type, ast_nodes.ValueType):
      return field.type.inner_type.name
  return None


def get_introspect_struct_fields(struct_name: str):
  """Retrieves the fields of a struct from the introspect data."""
  s = struct_name

  if s in introspect_structs.STRUCTS:
    return introspect_structs.STRUCTS[s].fields
  elif s in constants.ANONYMOUS_STRUCTS:
    anonymous_struct = _get_anonymous_struct_field(s)
    if not anonymous_struct or not isinstance(
        anonymous_struct.type, ast_nodes.AnonymousStructDecl
    ):
      raise RuntimeError(f"Anonymous struct not found: {s}")
    return anonymous_struct.type.fields
  else:
    raise RuntimeError(f"Struct not found: {s}")


def sort_structs_by_dependency(
    struct_with_fields: dict[str, list[WrappedFieldData]],
) -> list[str]:
  """Sorts structs based on their field dependencies using topological sort.

    Structs with no dependencies on other structs in the list come first.
  Struct A has a dependency on struct B if struct A has a field where the
  underlying_type is B. Note that this definition is stricter than the C++
  struct dependency criterion where forward declarations can be used to
  eliminate dependencies A and B if A only has a pointer to B.

  Args:
    struct_with_fields: A dictionary mapping struct names to their
      WrappedFieldData.

  Returns:
    A new list of struct names sorted by dependency.

  Raises:
    RuntimeError: If a cyclic dependency is detected.
  """
  adj = collections.defaultdict(list)
  in_degree = collections.defaultdict(int)
  struct_names = struct_with_fields.keys()
  struct_set = set(struct_names)
  sorted_struct_names = sorted(struct_names)

  for s in sorted_struct_names:
    if s == "mjData":
      adj["mjModel"].append("mjData")
      in_degree["mjData"] += 1

    fields = struct_with_fields[s]
    for field in fields:
      field_type_name = field.typename
      if (
          field_type_name
          and field_type_name != s
          and field_type_name in struct_set
      ):
        if s not in adj[field_type_name]:
          adj[field_type_name].append(s)
          in_degree[s] += 1

  queue = collections.deque(
      [name for name in sorted_struct_names if in_degree[name] == 0]
  )
  sorted_list = []

  while queue:
    u = queue.popleft()
    sorted_list.append(u)
    for v in adj[u]:
      in_degree[v] -= 1
      if in_degree[v] == 0:
        queue.append(v)

  if len(sorted_list) == len(struct_names):
    return sorted_list
  else:
    remaining = set(struct_names) - set(sorted_list)
    raise RuntimeError(
        "Cycle detected in struct dependencies, involving: "
        f"{', '.join(sorted(list(remaining)))}"
    )


def generate(struct_to_bind: list[str]) -> list[tuple[str, list[str]]]:
  """Generates C++ header file for binding and wrapping MuJoCo structs."""
  typedefs = []
  for type_name in sorted(constants.ANONYMOUS_STRUCTS):
    s = constants.ANONYMOUS_STRUCTS[type_name]
    typedefs.append(
        f"using {type_name} = decltype(::{s['parent']}::{s['field_name']});"
    )

  wrapped_structs_with_fields: dict[str, list[WrappedFieldData]] = {}
  for s in struct_to_bind:
    fields: list[WrappedFieldData] = []
    introspect_fields = get_introspect_struct_fields(s)
    for field in introspect_fields:
      fields.append(_generate_field_data(field, common.wrapped_struct_name(s)))
    wrapped_structs_with_fields[s] = fields

  dependency_sorted_struct_names = sort_structs_by_dependency(
      wrapped_structs_with_fields
  )

  headers = []
  for s in dependency_sorted_struct_names:
    fields = wrapped_structs_with_fields[s]

    if s not in constants.MANUAL_STRUCTS_HEADERS:
      header = build_struct_header(s, fields)
      headers.append(header + "\n")

  sources = []
  for s in dependency_sorted_struct_names:
    fields = wrapped_structs_with_fields[s]
    if s not in constants.MANUAL_STRUCTS_SOURCES:
      source = build_struct_source(s, fields)
      sources.append(source + "\n")

  bindings = []
  alphabetically_sorted_struct_names = sorted(
      wrapped_structs_with_fields.keys()
  )
  for s in alphabetically_sorted_struct_names:
    fields = wrapped_structs_with_fields[s]
    bindings.append(_build_struct_bindings(s, fields))

  for s in alphabetically_sorted_struct_names:
    w = common.wrapped_struct_name(s)
    if w.startswith("Mjs") or w == "MjSpec":
      bindings.append(f"emscripten::register_optional<{w}>();")

  manual_struct_field_declarations = []
  for s in dependency_sorted_struct_names:
    w = common.wrapped_struct_name(s)
    fields = wrapped_structs_with_fields[s]
    if s in constants.MANUAL_STRUCTS_HEADERS:
      decls: list[str] = []
      for f in sorted(fields, key=lambda f: f.declaration):
        if f.declaration:
          decls.append(f.declaration)
      manual_struct_field_declarations.append(
          (f"// INSERT-GENERATED-{w}-DECLARATION", decls)
      )

  return [
      *manual_struct_field_declarations,
      ("// {{ ANONYMOUS_STRUCT_TYPEDEFS }}", typedefs),
      ("// {{ STRUCTS_HEADER }}", headers),
      ("// {{ STRUCTS_SOURCE }}", sources),
      ("// {{ STRUCTS_BINDINGS }}", bindings),
  ]
