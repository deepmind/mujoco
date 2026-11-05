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

"""Tests for helper modules."""

from absl.testing import absltest
from codegen import code_builder
from introspect import ast_nodes

from wasm.codegen.generators import common
from wasm.codegen.generators import enums
from wasm.codegen.generators import functions
from wasm.codegen.generators import structs


class CodeBuilderTest(absltest.TestCase):

  def test_code_builder_functionality(self):
    """Test nested indentation blocks."""
    builder = code_builder.CodeBuilder(indent_str="    ")
    builder.line("let a = 1")
    with builder.block("function myFunc()"):
      builder.line("let flag = true")
      with builder.block("while (flag)"):
        builder.line("a++")
        builder.line("flag = a < 10")
      builder.line("return a")
    builder.line("print('Done')")

    expected_lines = [
        "let a = 1",
        "function myFunc() {",
        "    let flag = true",
        "    while (flag) {",
        "        a++",
        "        flag = a < 10",
        "    }",
        "    return a",
        "}",
        "print('Done')",
    ]
    self.assertEqual(builder.to_string(), "\n".join(expected_lines))


class CommonUtilsTest(absltest.TestCase):

  def test_capitalize(self):
    self.assertEqual(common.capitalize(""), "")
    self.assertEqual(common.capitalize("hello"), "Hello")
    self.assertEqual(common.capitalize("1st place"), "1st place")
    self.assertEqual(common.capitalize("!wow"), "!wow")
    self.assertEqual(common.capitalize(" leading space"), " leading space")


class FunctionUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.struct_type = ast_nodes.ValueType("MyStruct")
    self.ptr_to_int = ast_nodes.PointerType(ast_nodes.ValueType("int"))
    self.func_ret_ptr_int = ast_nodes.FunctionDecl(
        "func_pi", ast_nodes.PointerType(ast_nodes.ValueType("int")), [], "doc"
    )
    self.func_ret_ptr_struct = ast_nodes.FunctionDecl(
        "func_ps",
        ast_nodes.PointerType(ast_nodes.ValueType("MyStruct")),
        [],
        "doc",
    )

  def test_param_is_primitive_value(self):
    param_prim_val = ast_nodes.FunctionParameterDecl(
        "prim_v", ast_nodes.ValueType("int")
    )
    param_arr = ast_nodes.FunctionParameterDecl(
        "arr_v", ast_nodes.ArrayType(ast_nodes.ValueType("int"), extents=(10,))
    )

    self.assertTrue(functions.param_is_primitive_value(param_prim_val))
    self.assertFalse(functions.param_is_primitive_value(param_arr))

  def test_should_be_wrapped_with_primitive_ptr_return(self):
    func = ast_nodes.FunctionDecl(
        name="get_data",
        return_type=ast_nodes.PointerType(ast_nodes.ValueType("int")),
        parameters=tuple(),
        doc="Returns int pointer",
    )
    self.assertTrue(common.should_be_wrapped(func))

  def test_generate_function_wrapper_for_simple_func(self):
    func = ast_nodes.FunctionDecl(
        name="mj_defaultLROpt",
        return_type=ast_nodes.ValueType("void"),
        parameters=(
            ast_nodes.FunctionParameterDecl(
                name="opt",
                type=ast_nodes.PointerType(
                    inner_type=ast_nodes.ValueType("mjLROpt"),
                ),
            ),
        ),
        doc="Set default options for length range computation.",
    )
    result = functions.generate_function_wrapper(func)
    self.assertEqual(
        result,
        """void mj_defaultLROpt_wrapper(MjLROpt& opt) {
  mj_defaultLROpt(opt.get());
}""",
    )

  def test_generate_function_wrapper_checking_param(self):
    func = ast_nodes.FunctionDecl(
        name="mj_extractState",
        return_type=ast_nodes.ValueType(name="void"),
        parameters=(
            ast_nodes.FunctionParameterDecl(
                name="m",
                type=ast_nodes.PointerType(
                    inner_type=ast_nodes.ValueType(
                        name="mjModel", is_const=True
                    ),
                ),
            ),
            ast_nodes.FunctionParameterDecl(
                name="src",
                type=ast_nodes.PointerType(
                    inner_type=ast_nodes.ValueType(
                        name="mjtNum", is_const=True
                    ),
                ),
            ),
            ast_nodes.FunctionParameterDecl(
                name="srcsig",
                type=ast_nodes.ValueType(name="unsigned int"),
            ),
            ast_nodes.FunctionParameterDecl(
                name="dst",
                type=ast_nodes.PointerType(
                    inner_type=ast_nodes.ValueType(name="mjtNum"),
                ),
            ),
            ast_nodes.FunctionParameterDecl(
                name="dstsig",
                type=ast_nodes.ValueType(name="unsigned int"),
            ),
        ),
        doc="Extract a subset of components from a state previously obtained via mj_getState.",  # pylint: disable=line-too-long
    )
    result = functions.generate_function_wrapper(func)
    self.assertEqual(
        result,
        """void mj_extractState_wrapper(const MjModel& m, const NumberArray& src, unsigned int srcsig, const val& dst, unsigned int dstsig) {
  UNPACK_ARRAY(mjtNum, src);
  UNPACK_VALUE(mjtNum, dst);
  mj_extractState(m.get(), src_.data(), srcsig, dst_.data(), dstsig);
}""",
    )

  def test_get_params_string_with_struct_ptr(self):
    param = ast_nodes.FunctionParameterDecl(
        name="my_struct",
        type=ast_nodes.PointerType(ast_nodes.ValueType("mystruct")),
    )
    result = functions.get_param_string(param)
    self.assertEqual(result, "Mystruct& my_struct")

  def test_get_params_string_maybe_with_conversion_struct_ptr(self):
    param = ast_nodes.FunctionParameterDecl(
        name="s",
        type=ast_nodes.PointerType(ast_nodes.ValueType("customstruct")),
    )
    result = functions.get_params_string_maybe_with_conversion((param,))
    self.assertEqual(result, "s.get()")

  def test_get_compatible_return_code(self):
    func = ast_nodes.FunctionDecl(
        name="noop",
        return_type=ast_nodes.ValueType("void"),
        parameters=tuple(),
        doc="does nothing",
    )
    result = functions.get_compatible_return_code(func)
    self.assertEqual(result, "noop();")

  def test_get_compatible_return_type(self):
    func = ast_nodes.FunctionDecl(
        name="get_name",
        return_type=ast_nodes.PointerType(ast_nodes.ValueType("char")),
        parameters=tuple(),
        doc="returns name",
    )
    result = functions.get_compatible_return_type(func)
    self.assertEqual(result.strip(), "std::string")

  def test_get_optional_return_code(self):
    func = ast_nodes.FunctionDecl(
        name="get_struct",
        return_type=ast_nodes.PointerType(ast_nodes.ValueType("mystruct")),
        parameters=tuple(),
        doc="returns struct",
    )
    result = functions.get_optional_return_code(func, "get_struct()")
    self.assertIn("mystruct* result = get_struct();", result)
    self.assertIn("return Mystruct(result)", result)

  def test_is_excluded_function_name(self):
    self.assertTrue(functions.is_excluded_function_name("mjr_function"))
    self.assertTrue(functions.is_excluded_function_name("mjui_function"))
    self.assertTrue(functions.is_excluded_function_name("mju_malloc"))
    self.assertTrue(functions.is_excluded_function_name("mj_makeData"))
    self.assertFalse(functions.is_excluded_function_name("mjv_updateScene"))
    self.assertFalse(functions.is_excluded_function_name("mj_normalFunction"))
    self.assertFalse(
        functions.is_excluded_function_name("mju_someOtherFunction")
    )


class StructConstructorCodeBuilderTest(absltest.TestCase):

  def test_constructor_code_with_fields_with_init(self):
    field_with_init = ast_nodes.StructFieldDecl(
        name="element",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="mjsElement"),
        ),
        doc="",
    )
    wrapped_field_data = structs._generate_field_data(
        field_with_init, "MjsTexture"
    )
    self.assertEqual(
        structs.build_struct_source(
            "mjsTexture",
            [wrapped_field_data],
        ),
        """
MjsTexture::MjsTexture(mjsTexture *ptr) : ptr_(ptr), element(ptr_->element) {}
mjsTexture* MjsTexture::get() const {
  return ptr_;
}
void MjsTexture::set(mjsTexture* ptr) {
  ptr_ = ptr;
}
""".strip(),
    )

  def test_constructor_code_with_shallow_copy(self):
    self.assertEqual(
        structs.build_struct_source("mjvLight", []),
        """MjvLight::MjvLight(mjvLight *ptr) : ptr_(ptr) {}
MjvLight::~MjvLight() {
  if (owned_ && ptr_) {
    delete ptr_;
  }
}
MjvLight::MjvLight() : ptr_(new mjvLight()) {
  owned_ = true;
}
MjvLight::MjvLight(const MjvLight &other) : MjvLight() {
  *ptr_ = *other.get();
}
MjvLight& MjvLight::operator=(const MjvLight &other) {
  if (this == &other) {
    return *this;
  }
  *ptr_ = *other.get();
  return *this;
}
std::unique_ptr<MjvLight> MjvLight::copy() {
  return std::make_unique<MjvLight>(*this);
}
mjvLight* MjvLight::get() const {
  return ptr_;
}
void MjvLight::set(mjvLight* ptr) {
  ptr_ = ptr;
}
""".strip(),
    )


def test_build_struct_header_with_nested_wrappers(self):
  self.assertEqual(
      structs.build_struct_header("mjData", []),
      "",
  )


def test_build_struct_header_basic_struct(self):
  self.assertEqual(
      structs.build_struct_header("mjLROpt", []),
      """
struct MjLROpt {
  MjLROpt();
  MjLROpt(const MjLROpt &);
  MjLROpt &operator=(const MjLROpt &);
  explicit MjLROpt(mjLROpt *ptr);
  ~MjLROpt();
  mjLROpt* get() const { return ptr_; }
  void set(mjLROpt* ptr) { ptr_ = ptr; }

 private:
  mjLROpt* ptr_;
  bool owned_ = false;
};
""".strip(),
  )


class StructFieldCodeBuilderTest(absltest.TestCase):

  def test_primitive_type_definition(self):
    field = ast_nodes.StructFieldDecl(
        name="ngeom",
        type=ast_nodes.ValueType(name="int"),
        doc="number of geoms",
    )
    self.assertEqual(
        structs._generate_field_data(field, "ngeom").declaration,
        """
int ngeom() const {
  return ptr_->ngeom;
}
void set_ngeom(int value) {
  ptr_->ngeom = value;
}
""".strip(),
    )

  def test_memory_view_definition(self):
    field = ast_nodes.StructFieldDecl(
        name="geom_rgba",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="float"),
        ),
        doc="rgba when material is omitted",
        array_extent=("ngeom", 4),
    )
    self.assertEqual(
        structs._generate_field_data(field, "geom_rgba").declaration,
        """
emscripten::val geom_rgba() const {
  return emscripten::val(emscripten::typed_memory_view(ptr_->ngeom * 4, ptr_->geom_rgba));
}
""".strip(),
    )

  def test_string_field_definition(self):
    field = ast_nodes.StructFieldDecl(
        name="string_field",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="mjString"),
        ),
        doc="rgba when material is omitted",
    )
    self.assertEqual(
        structs._generate_field_data(field, "MjString").declaration,
        """
mjString string_field() const {
  return (ptr_ && ptr_->string_field) ? *(ptr_->string_field) : "";
}
void set_string_field(const mjString& value) {
  if (ptr_ && ptr_->string_field) {
    *(ptr_->string_field) = value;
  }
}
""".strip(),
    )

  def test_mjvec_pointer_definition(self):
    field = ast_nodes.StructFieldDecl(
        name="vector_field",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="mjDoubleVec"),
        ),
        doc="",
    )
    self.assertEqual(
        structs._generate_field_data(field, "mjDoubleVec").declaration,
        """
mjDoubleVec &vector_field() const {
  return *(ptr_->vector_field);
}""".strip(),
    )

  def test_mjbyte_vec_pointer_definition(self):
    field = ast_nodes.StructFieldDecl(
        name="vector_field",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="mjByteVec"),
        ),
        doc="",
    )
    self.assertEqual(
        structs._generate_field_data(field, "mjByteVec").declaration,
        """
std::vector<uint8_t> &vector_field() const {
  return *(reinterpret_cast<std::vector<uint8_t>*>(ptr_->vector_field));
}""".strip(),
    )

  def test_get_property_binding(self):
    field = ast_nodes.StructFieldDecl(
        name="ngeom",
        type=ast_nodes.ValueType(name="int"),
        doc="number of geoms",
    )
    self.assertEqual(
        structs._get_property_binding(field, "MjModel"),
        '.property("ngeom", &MjModel::ngeom)',
    )

  def test_get_property_binding_with_setter(self):
    field = ast_nodes.StructFieldDecl(
        name="ngeom",
        type=ast_nodes.ValueType(name="int"),
        doc="",
    )
    self.assertEqual(
        structs._get_property_binding(field, "MjModel", True),
        '.property("ngeom", &MjModel::ngeom, &MjModel::set_ngeom)',
    )

  def test_get_property_binding_with_return_value_policy_as_ref(self):
    field = ast_nodes.StructFieldDecl(
        name="ngeom",
        type=ast_nodes.ValueType(name="int"),
        doc="",
    )
    self.assertEqual(
        structs._get_property_binding(
            field,
            "MjModel",
            setter=True,
            reference=True,
        ),
        '.property("ngeom", &MjModel::ngeom, &MjModel::set_ngeom, reference())',
    )


class StructFieldHandlerTest(absltest.TestCase):

  def test_scalar_field(self):
    """Test that a scalar type field is handled correctly."""
    field_scalar = ast_nodes.StructFieldDecl(
        name="ngeom",
        type=ast_nodes.ValueType(name="int"),
        doc="number of geoms",
    )

    wrapped_field_data = structs._generate_field_data(field_scalar, "MjModel")
    self.assertEqual(
        wrapped_field_data.declaration,
        """
int ngeom() const {
  return ptr_->ngeom;
}
void set_ngeom(int value) {
  ptr_->ngeom = value;
}
""".strip(),
    )
    self.assertEqual(
        wrapped_field_data.binding,
        '.property("ngeom", &MjModel::ngeom, &MjModel::set_ngeom)',
    )

  def test_pointer_type_field(self):
    """Test that a pointer type field is handled correctly."""
    field = ast_nodes.StructFieldDecl(
        name="geom_rgba",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="float"),
        ),
        doc="rgba when material is omitted",
        array_extent=("ngeom", 4),
    )
    wrapped_field_data = structs._generate_field_data(field, "MjModel")

    self.assertEqual(
        wrapped_field_data.declaration,
        ("""
emscripten::val geom_rgba() const {
  return emscripten::val(emscripten::typed_memory_view(ptr_->ngeom * 4, ptr_->geom_rgba));
}
""".strip()),
    )

    self.assertEqual(
        wrapped_field_data.binding,
        '.property("geom_rgba", &MjModel::geom_rgba)',
    )

  def test_pointer_type_field_for_byte_type(self):
    """Test that a pointer type field for a byte type is handled correctly."""
    field = ast_nodes.StructFieldDecl(
        name="buffer",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="void"),
        ),
        doc="main buffer; all pointers point in it (nbuffer bytes)",
    )
    wrapped_field_data = structs._generate_field_data(field, "MjData")

    self.assertEqual(
        wrapped_field_data.declaration,
        ("""
emscripten::val buffer() const {
  return emscripten::val(emscripten::typed_memory_view(model->nbuffer, static_cast<uint8_t*>(ptr_->buffer)));
}
""".strip()),
    )
    assert wrapped_field_data.binding == '.property("buffer", &MjData::buffer)'

  def test_pointer_type_field_for_mj_struct(self):
    """Test that a pointer type field for a mj struct is handled correctly."""
    field = ast_nodes.StructFieldDecl(
        name="element",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="mjsElement"),
        ),
        doc="",
    )
    wrapped_field_data = structs._generate_field_data(field, "MjsTexture")
    self.assertEqual(wrapped_field_data.declaration, "MjsElement element;")

    self.assertEqual(
        wrapped_field_data.binding,
        '.property("element", &MjsTexture::element, reference())',
    )
    self.assertEqual(
        wrapped_field_data.ptr_initialization,
        "element(ptr_->element)",
    )

  def test_array_type_field(self):
    """Test that an array type field is handled correctly."""
    field = ast_nodes.StructFieldDecl(
        name="gravity",
        type=ast_nodes.ArrayType(
            inner_type=ast_nodes.ValueType(name="mjtNum"),
            extents=(3,),
        ),
        doc="gravitational acceleration",
    )
    wrapped_field_data = structs._generate_field_data(field, "MjOption")

    self.assertEqual(
        wrapped_field_data.declaration,
        ("""
emscripten::val gravity() const {
  return emscripten::val(emscripten::typed_memory_view(3, ptr_->gravity));
}
""".strip()),
    )
    self.assertEqual(
        wrapped_field_data.binding,
        '.property("gravity", &MjOption::gravity)',
    )

  def test_array_field_with_multi_dimensional_array(self):
    """Test that multi-dimensional arrays are handled correctly."""
    field = ast_nodes.StructFieldDecl(
        name="multi_dim_array",
        type=ast_nodes.ArrayType(
            inner_type=ast_nodes.ValueType(name="float"),
            extents=(3, 4),
        ),
        doc="description",
    )
    wrapped_field_data = structs._generate_field_data(field, "MjModel")
    self.assertEqual(
        wrapped_field_data.declaration,
        """
emscripten::val multi_dim_array() const {
  return emscripten::val(emscripten::typed_memory_view(12, reinterpret_cast<float*>(ptr_->multi_dim_array)));
}
""".strip(),
    )
    self.assertEqual(
        wrapped_field_data.binding,
        '.property("multi_dim_array", &MjModel::multi_dim_array)',
    )

  def test_pointer_type_field_for_mjtbool_type(self):
    """Test that a pointer type field for mjtBool is exposed as uint8_t*."""
    field = ast_nodes.StructFieldDecl(
        name="jnt_limited",
        type=ast_nodes.PointerType(
            inner_type=ast_nodes.ValueType(name="mjtBool"),
        ),
        doc="does joint have limits",
        array_extent=("njnt",),
    )
    wrapped_field_data = structs._generate_field_data(field, "MjModel")
    self.assertEqual(
        wrapped_field_data.declaration,
        """
emscripten::val jnt_limited() const {
  return emscripten::val(emscripten::typed_memory_view(ptr_->njnt, reinterpret_cast<uint8_t*>(ptr_->jnt_limited)));
}
""".strip(),
    )
    self.assertEqual(
        wrapped_field_data.binding,
        '.property("jnt_limited", &MjModel::jnt_limited)',
    )

  def test_array_type_field_for_mjtbool_type(self):
    """Test that an array type field for mjtBool is exposed as uint8_t*."""
    field = ast_nodes.StructFieldDecl(
        name="bool_flags",
        type=ast_nodes.ArrayType(
            inner_type=ast_nodes.ValueType(name="mjtBool"),
            extents=(4,),
        ),
        doc="boolean flags",
    )
    wrapped_field_data = structs._generate_field_data(field, "MjOption")
    self.assertEqual(
        wrapped_field_data.declaration,
        """
emscripten::val bool_flags() const {
  return emscripten::val(emscripten::typed_memory_view(4, reinterpret_cast<uint8_t*>(ptr_->bool_flags)));
}
""".strip(),
    )
    self.assertEqual(
        wrapped_field_data.binding,
        '.property("bool_flags", &MjOption::bool_flags)',
    )

  def test_parse_array_extent(self):
    """Test that parse_array_extent handles various cases correctly."""
    self.assertEqual(
        structs.parse_array_extent((1, 2), "MjModel", "geom_rgba"),
        "1 * 2",
    )
    self.assertEqual(
        structs.parse_array_extent((1, "ngeom"), "MjModel", "geom_rgba"),
        "1 * ptr_->ngeom",
    )
    self.assertEqual(
        structs.parse_array_extent((1, "mjConstant"), "MjModel", "geom_rgba"),
        "1 * mjConstant",
    )

  def test_resolve_extent(self):
    """Test that resolve_extent handles various cases correctly."""
    # for integer just return the number
    self.assertEqual(structs.resolve_extent(1, "MjModel", "geom_rgba"), "1")

    # for string that does not start with mj, it's a member of the struct
    self.assertEqual(
        structs.resolve_extent("ngeom", "MjModel", "geom_rgba"),
        "ptr_->ngeom",
    )

    # when it's MjData and the field is in MJDATA_SIZES, it should use ptr_->
    self.assertEqual(
        structs.resolve_extent("size_value", "MjData", "efc_AR"),
        "ptr_->size_value",
    )
    # when it's MjData and the field is not in MJDATA_SIZES,
    # it should use model->
    self.assertEqual(
        structs.resolve_extent("size_value", "MjData", "data_field"),
        "model->size_value",
    )
    self.assertEqual(
        structs.resolve_extent("mjConstant", "MjModel", "model_field"),
        "mjConstant",
    )


class EnumsGeneratorTest(absltest.TestCase):

  def test_generate_enum_bindings(self):

    expected_code = """
enum_<AnotherEnum>("AnotherEnum")
  .value("ALPHA", ALPHA)
  .value("BETA", BETA);
enum_<TestEnum>("TestEnum")
  .value("FIRST_VAL", FIRST_VAL)
  .value("SECOND_VAL", SECOND_VAL)
  .value("THIRD_VAL", THIRD_VAL);
""".strip()

    markers_and_content = enums.generate([
        ast_nodes.EnumDecl(
            name="TestEnum",
            declname="enum TestEnum_",
            values={"FIRST_VAL": 0, "SECOND_VAL": 1, "THIRD_VAL": 2},
        ),
        ast_nodes.EnumDecl(
            name="AnotherEnum",
            declname="enum AnotherEnum_",
            values={"ALPHA": 100, "BETA": 200},
        ),
        ast_nodes.EnumDecl(
            name="EmptyEnum",
            declname="enum EmptyEnum_",
            values={},
        ),
    ])
    actual_code = "\n\n".join(markers_and_content[0][1])

    self.assertEqual(actual_code, expected_code)


if __name__ == "__main__":
  absltest.main()
