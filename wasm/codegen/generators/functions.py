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

"""Helper functions for processing and generating bindings for MuJoCo functions."""

from typing import Tuple, cast

from codegen import code_builder
from introspect import ast_nodes

from wasm.codegen.generators import common
from wasm.codegen.generators import constants


def param_is_primitive_value(param: ast_nodes.FunctionParameterDecl) -> bool:
  """Checks if param is a primitive value type."""
  if isinstance(param.type, ast_nodes.ValueType):
    return param.type.name in constants.PRIMITIVE_TYPES
  return False


def get_const_qualifier(func: ast_nodes.FunctionDecl) -> str:
  """Returns the const qualifier of func's return type."""
  inner_type = common.get_pointer_return_inner_value_type(func)
  if inner_type and inner_type.is_const:
    return "const "
  return ""


def generate_function_wrapper(func: ast_nodes.FunctionDecl) -> str:
  """Generates C++ code for a wrapper function."""

  bound_check_code = constants.FUNCTION_BOUNDS_CHECKS.get(func.name, "")
  wrapper_parameters = [
      p for p in func.parameters if f"{p.name} = " not in bound_check_code
  ]
  wrapper_params = ", ".join([get_param_string(p) for p in wrapper_parameters])
  ret_type = get_compatible_return_type(func)
  builder = code_builder.CodeBuilder()
  w = common.wrapped_function_name(func)
  with builder.function(f"{ret_type} {w}({wrapper_params})"):

    for p in wrapper_parameters:
      if c_notnullable := get_param_notnullable(p):
        builder.line(f"CHECK_VAL({c_notnullable});")

    for p in wrapper_parameters:
      if c_unpack := get_param_unpack_statement(p):
        builder.line(c_unpack)

    if bound_check_code:
      builder.line(bound_check_code)

    for line in get_compatible_return_code(func).splitlines():
      builder.line(line)

  return builder.to_string()


def get_param_notnullable(
    p: ast_nodes.FunctionParameterDecl,
) -> str:
  """Generates list of param names for checking if they aren't null/undefined."""

  if (
      isinstance(p.type, (ast_nodes.PointerType, ast_nodes.ArrayType))
      and isinstance(p.type.inner_type, ast_nodes.ValueType)
      # We only check for char because others are checked in the unpacker
      # and we don't want to check twice.
      and p.type.inner_type.name == "char"
      and not p.nullable
  ):
    return p.name
  return ""


def get_param_unpack_statement(
    p: ast_nodes.FunctionParameterDecl,
) -> str:
  """Generates C++ statements to unpack JS values for pointer/array parameters."""

  inner_type = common.get_inner_value_type(p)
  if not inner_type:
    return ""

  if inner_type.name not in constants.PRIMITIVE_TYPES:
    return ""

  if inner_type.name == "char":
    # param is Javascript string
    return ""
  if inner_type.is_const:
    # param is Javascript number[]
    if p.nullable:
      return f"UNPACK_NULLABLE_ARRAY({inner_type.name}, {p.name});"
    else:
      return f"UNPACK_ARRAY({inner_type.name}, {p.name});"
  else:
    # param is TypedArray or a WasmBuffer
    if p.nullable:
      return f"UNPACK_NULLABLE_VALUE({inner_type.name}, {p.name});"
    else:
      return f"UNPACK_VALUE({inner_type.name}, {p.name});"


def get_param_string(p: ast_nodes.FunctionParameterDecl) -> str:
  """Generates a list of C++ parameter declarations as strings."""

  if (
      isinstance(p.type, ast_nodes.PointerType)
      and isinstance(p.type.inner_type, ast_nodes.ValueType)
      and p.type.inner_type.name not in constants.PRIMITIVE_TYPES
  ):
    # Pointer to struct parameters
    const_qualifier = "const " if p.type.inner_type.is_const else ""
    return (
        f"{const_qualifier}{common.capitalize(p.type.inner_type.name)}&"
        f" {p.name}"
    )
  elif (
      isinstance(p.type, ast_nodes.ValueType)
      and p.type.name in constants.PRIMITIVE_TYPES
  ):
    # Primitive value parameters
    const_qualifier = "const " if p.type.is_const else ""
    return f"{const_qualifier}{p.type} {p.name}"
  elif (
      isinstance(p.type, (ast_nodes.PointerType, ast_nodes.ArrayType))
      and isinstance(p.type.inner_type, ast_nodes.ValueType)
      and p.type.inner_type.name in constants.PRIMITIVE_TYPES
  ):
    # Pointer to primitive value parameters or arrays
    if p.type.inner_type.name == "char":
      if p.nullable:
        return f"const StringOrNull& {p.name}"
      else:
        return f"const String& {p.name}"
    elif (
        p.type.inner_type.name
        in ["int", "float", "double", "mjtNum", "mjtByte", "mjtBool"]
        and p.type.inner_type.is_const
    ):
      return f"const NumberArray& {p.name}"
    else:
      return f"const val& {p.name}"
  elif common.is_struct_value_type(p.type):
    # Struct by value parameters
    value_type = cast(ast_nodes.ValueType, p.type)
    const_qualifier = "const " if value_type.is_const else "const "
    return (
        f"{const_qualifier}{common.capitalize(value_type.name)}&"
        f" {p.name}"
    )
  else:
    # This case should ideally not be reached if AST is well-formed
    # and types are categorized by the helper booleans correctly.
    raise TypeError(
        "Unable to generate param string. Unhandled parameter type:"
        f" {p.type} for param '{p.name}'"
    )


def get_params_string_maybe_with_conversion(
    ast_params: Tuple[ast_nodes.FunctionParameterDecl, ...],
) -> str:
  """Generates C++ expressions for passing compatible params from JS to MuJoCo C-API functions."""

  native_params = []
  for p in ast_params:
    if inner_type := common.get_inner_value_type(p):
      if inner_type.name in constants.PRIMITIVE_TYPES:
        if inner_type.name == "char":
          const_qualifier = "const " if inner_type.is_const else ""
          native_params.append(
              f"{p.name}.as<{const_qualifier}std::string>().data()"
          )
        else:
          native_params.append(f"{p.name}_.data()")
      else:  # struct
        native_params.append(f"{p.name}.get()")
    elif param_is_primitive_value(p):
      native_params.append(p.name)
    elif common.is_struct_value_type(p.type):
      native_params.append(f"*{p.name}.get()")
    else:
      raise TypeError(
          f"Unhandled parameter type for conversion: {p.type} for param"
          f" '{p.name}'"
      )
  return ", ".join(native_params)


def get_compatible_return_code(func: ast_nodes.FunctionDecl) -> str:
  """Generates embind compatible return value conversion."""
  c_params = get_params_string_maybe_with_conversion(func.parameters)
  c_call = f"{func.name}({c_params})"

  if isinstance(func.return_type, ast_nodes.ValueType):
    if func.return_type.name == "void":
      return f"{c_call};"
    if func.return_type.name in constants.PRIMITIVE_TYPES:
      return f"return {c_call};"
    if common.is_struct_value_type(func.return_type):
      struct_name = func.return_type.name
      w = common.wrapped_struct_name(struct_name)
      builder = code_builder.CodeBuilder()
      builder.line(f"{w} result;")
      builder.line(f"*result.get() = {c_call};")
      builder.line("return result;")
      return builder.to_string()

  if inner_type := common.get_pointer_return_inner_value_type(func):
    if inner_type.name == "char":
      return f"return std::string({c_call});"
    elif inner_type.name == "mjString":
      return f"return *{c_call};"
    elif inner_type.name not in constants.PRIMITIVE_TYPES:
      return get_optional_return_code(func, c_call)

  raise RuntimeError(
      "Failed to calculate return value conversion for function"
      f" {func.name} that returns '{func.return_type}'"
  )


def get_compatible_return_type(func: ast_nodes.FunctionDecl) -> str:
  """Creates embind compatible return type."""

  if inner_type := common.get_pointer_return_inner_value_type(func):
    if inner_type.name in ["char", "mjString"]:
      return "std::string"
    if inner_type.name not in constants.PRIMITIVE_TYPES:
      const_qualifier = get_const_qualifier(func)
      return f"""{const_qualifier}std::optional<{common.capitalize(inner_type.name)}>"""
  if (
      isinstance(func.return_type, ast_nodes.ValueType)
      and func.return_type.name in constants.PRIMITIVE_TYPES
  ):
    return f"{func.return_type.name}"
  if common.is_struct_value_type(func.return_type):
    return_type = cast(ast_nodes.ValueType, func.return_type)
    return common.wrapped_struct_name(return_type.name)
  return "val"


def get_optional_return_code(
    func: ast_nodes.FunctionDecl, c_call: str
) -> str:
  """Generates code to return std::optional of the wrapped struct."""

  const_qualifier = get_const_qualifier(func)
  return_type = cast(ast_nodes.PointerType, func.return_type)
  struct_name = cast(ast_nodes.ValueType, return_type.inner_type).name

  builder = code_builder.CodeBuilder()
  builder.line(f"{const_qualifier}{struct_name}* result = {c_call};")
  with builder.block("if (result == nullptr)"):
    builder.line("return std::nullopt;")
  builder.line(f"return {common.wrapped_struct_name(struct_name)}(result);")

  return builder.to_string()


def is_excluded_function_name(func_name: str) -> bool:
  """Checks if a function name should be excluded from direct binding."""
  return (
      func_name.startswith(("mjr_", "mjui_"))
      or func_name in constants.SKIPPED_FUNCTIONS
  )


def generate(
    functions: list[ast_nodes.FunctionDecl],
) -> list[tuple[str, list[str]]]:
  """Generates Embind bindings for MuJoCo functions."""
  wrapper_functions = []
  for func in sorted(functions, key=lambda f: f.name):
    if common.should_be_wrapped(func):
      if func.name not in constants.MANUAL_WRAPPER_FUNCTIONS:
        wrapper_functions.append(generate_function_wrapper(func))
  wrapper_content = "\n\n".join(wrapper_functions)

  function_bindings = []
  for func in sorted(functions, key=lambda f: f.name):
    if func.name not in constants.MANUAL_WRAPPER_FUNCTIONS:
      w = common.wrapped_function_name(func)
      function_bindings.append(f'function("{func.name}", &{w});')
  bindings_content = "\n".join(function_bindings)

  return [
      ("// {{ WRAPPER_FUNCTIONS }}", [wrapper_content]),
      ("// {{ FUNCTION_BINDINGS }}", [bindings_content]),
  ]
