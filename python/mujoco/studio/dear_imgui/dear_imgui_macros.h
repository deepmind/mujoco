// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PYTHON_STUDIO_DEAR_IMGUI_DEAR_IMGUI_MACROS_H_
#define MUJOCO_PYTHON_STUDIO_DEAR_IMGUI_DEAR_IMGUI_MACROS_H_

// WARNING: This file is intended for internal use by dear_imgui libraries ONLY!
//
// The macros defined here use short, generic names (DEF0, ARG_ID, etc.) and
// are NOT #undef'd. Including this header elsewhere may cause naming conflicts.
//
// Define NAMESPACE to be the ImGui library you're binding before including this
// header, e.g. #define NAMESPACE ImGui
//
// ============================================================================
// Quick Reference
// ============================================================================
//
// DEFn(Name, Args...)  // Binds NAMESPACE::Name as Name in Python
// DEFn_AS(CppName, PyName, Args...)  // Binds NAMESPACE::CppName as PyName
// DEFn_F(PyName, Args..., { CppBody })  // Custom implementation
//
// Where 'n' is the number of arguments (0-9).
//
// ============================================================================
// Argument Format
// ============================================================================
//
// Each argument is a tuple: (Type, name, DefaultValue)
//
//   - No default value:   (ImString, label, )  // trailing comma required
//   - With default value: (int, flags, = 0)  // include the '='
//   - Complex defaults:   (const ImVec2&, size, = ImVec2_Zero)
//
// NOTE: Default values cannot contain commas. Use predefined constants like
// ImVec2_Zero, ImVec4_One, etc.
//
// ============================================================================
// Examples (from dear_imgui.cc)
// ============================================================================
//
// Simple binding - NAMESPACE::End() exposed as End():
//   DEF0(End);
//
// Binding with arguments:
//   DEF4(BeginChild,
//        (ImString, str_id, ),
//        (const ImVec2&, size, = ImVec2_Zero),
//        (ImGuiChildFlags, child_flags, = 0),
//        (ImGuiWindowFlags, window_flags, = 0));
//
// Overloaded function - NAMESPACE::BeginChild(ImGuiID) exposed as BeginChildId():
//   DEF4_AS(BeginChild, BeginChildId,
//           (ImGuiID, id, ),
//           (const ImVec2&, size, = ImVec2_Zero),
//           (ImGuiChildFlags, child_flags, = 0),
//           (ImGuiWindowFlags, window_flags, = 0));
//
// Custom implementation using DEFn_F is needed when:
//
//   1. Variadic functions (e.g., Text, TextColored)
//
//      C++ variadic functions (those with "...") cannot be bound directly
//      because the type/count of arguments is unknown at compile time. Use a
//      wrapper that calls the function with a fixed format. Also note that
//      user-controlled format strings are a security risk (format string
//      attacks). Always use "%s":
//
//        DEF1_F(Text, (ImString, txt, ), {
//          return NAMESPACE::Text("%s", txt);
//        });
//
//   2. Output pointer parameters (e.g., Checkbox, SliderFloat)
//
//      Python doesn't have output pointers, so return modified values as a
//      tuple:
//
//        DEF2_F(Checkbox, (ImString, label, ), (bool*, v, ), {
//          auto result = NAMESPACE::Checkbox(label, v);
//          return std::make_tuple(result, *v);
//        });
//
//   3. Type conversions (e.g., Image, ImageButton)
//
//      Some C++ types don't have Python equivalents. For example, ImTextureID
//      is a void* (opaque pointer), which pybind11 can't automatically convert.
//      Accept a Python-friendly type (like long) and cast it:
//
//        DEF2_F(Image, (long, tex_id, ), (const ImVec2&, size, ), {
//          return NAMESPACE::Image(reinterpret_cast<void*>(tex_id), size);
//        });

// ============================================================================
// Internal helper macros (not intended to be called directly by binding code)
// ============================================================================

// Extracts the type and name of an argument tuple.
// Example: ARG_DECL((float, alpha, = 1.0f)) -> float alpha
#define ARG_DECL_X(T_, N_, V_) T_ N_
#define ARG_DECL(A_) ARG_DECL_X A_

// Extracts the identifier of an argument tuple.
// Example: ARG_ID((float, alpha, = 1.0f)) -> alpha
#define ARG_ID_X(T_, N_, V_) N_
#define ARG_ID(A_) ARG_ID_X A_

// Extracts the name of an argument tuple as a quoted string literal.
// Example: ARG_NAME((float, alpha, = 1.0f)) -> "alpha"
#define ARG_NAME_X(T_, N_, V_) #N_
#define ARG_NAME(A_) ARG_NAME_X A_

// Extracts the default value of an argument tuple.
// Example: ARG_DEFVAL((float, alpha, = 1.0f)) -> = 1.0f
#define ARG_DEFVAL_X(T_, N_, V_) V_
#define ARG_DEFVAL(A_) ARG_DEFVAL_X A_

// ============================================================================
// Public macros for binding code
// ============================================================================

//
#define DEF0_F(N, FN)                      \
  m.def(#N, [](                            \
  ) FN                                     \
  );

#define DEF1_F(N, A1, FN)                  \
  m.def(#N, [](                            \
    ARG_DECL(A1)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1)   \
  );

#define DEF2_F(N, A1, A2, FN)              \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2)   \
  );

#define DEF3_F(N, A1, A2, A3, FN)          \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3)   \
  );

#define DEF4_F(N, A1, A2, A3, A4, FN)      \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3),                          \
    ARG_DECL(A4)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3),  \
    py::arg(ARG_NAME(A4)) ARG_DEFVAL(A4)   \
  );

#define DEF5_F(N, A1, A2, A3, A4, A5, FN)  \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3),                          \
    ARG_DECL(A4),                          \
    ARG_DECL(A5)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3),  \
    py::arg(ARG_NAME(A4)) ARG_DEFVAL(A4),  \
    py::arg(ARG_NAME(A5)) ARG_DEFVAL(A5)   \
  );

#define DEF6_F(N, A1, A2, A3, A4, A5, A6, FN)  \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3),                          \
    ARG_DECL(A4),                          \
    ARG_DECL(A5),                          \
    ARG_DECL(A6)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3),  \
    py::arg(ARG_NAME(A4)) ARG_DEFVAL(A4),  \
    py::arg(ARG_NAME(A5)) ARG_DEFVAL(A5),  \
    py::arg(ARG_NAME(A6)) ARG_DEFVAL(A6)   \
  );

#define DEF7_F(N, A1, A2, A3, A4, A5, A6, A7, FN)  \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3),                          \
    ARG_DECL(A4),                          \
    ARG_DECL(A5),                          \
    ARG_DECL(A6),                          \
    ARG_DECL(A7)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3),  \
    py::arg(ARG_NAME(A4)) ARG_DEFVAL(A4),  \
    py::arg(ARG_NAME(A5)) ARG_DEFVAL(A5),  \
    py::arg(ARG_NAME(A6)) ARG_DEFVAL(A6),  \
    py::arg(ARG_NAME(A7)) ARG_DEFVAL(A7)   \
  );

#define DEF8_F(N, A1, A2, A3, A4, A5, A6, A7, A8, FN)  \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3),                          \
    ARG_DECL(A4),                          \
    ARG_DECL(A5),                          \
    ARG_DECL(A6),                          \
    ARG_DECL(A7),                          \
    ARG_DECL(A8)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3),  \
    py::arg(ARG_NAME(A4)) ARG_DEFVAL(A4),  \
    py::arg(ARG_NAME(A5)) ARG_DEFVAL(A5),  \
    py::arg(ARG_NAME(A6)) ARG_DEFVAL(A6),  \
    py::arg(ARG_NAME(A7)) ARG_DEFVAL(A7),  \
    py::arg(ARG_NAME(A8)) ARG_DEFVAL(A8)   \
  );

#define DEF9_F(N, A1, A2, A3, A4, A5, A6, A7, A8, A9, FN)  \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3),                          \
    ARG_DECL(A4),                          \
    ARG_DECL(A5),                          \
    ARG_DECL(A6),                          \
    ARG_DECL(A7),                          \
    ARG_DECL(A8),                          \
    ARG_DECL(A9)                           \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3),  \
    py::arg(ARG_NAME(A4)) ARG_DEFVAL(A4),  \
    py::arg(ARG_NAME(A5)) ARG_DEFVAL(A5),  \
    py::arg(ARG_NAME(A6)) ARG_DEFVAL(A6),  \
    py::arg(ARG_NAME(A7)) ARG_DEFVAL(A7),  \
    py::arg(ARG_NAME(A8)) ARG_DEFVAL(A8),  \
    py::arg(ARG_NAME(A9)) ARG_DEFVAL(A9)   \
  );

#define DEF10_F(N, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, FN)  \
  m.def(#N, [](                            \
    ARG_DECL(A1),                          \
    ARG_DECL(A2),                          \
    ARG_DECL(A3),                          \
    ARG_DECL(A4),                          \
    ARG_DECL(A5),                          \
    ARG_DECL(A6),                          \
    ARG_DECL(A7),                          \
    ARG_DECL(A8),                          \
    ARG_DECL(A9),                          \
    ARG_DECL(A10)                          \
  ) FN,                                    \
    py::arg(ARG_NAME(A1)) ARG_DEFVAL(A1),  \
    py::arg(ARG_NAME(A2)) ARG_DEFVAL(A2),  \
    py::arg(ARG_NAME(A3)) ARG_DEFVAL(A3),  \
    py::arg(ARG_NAME(A4)) ARG_DEFVAL(A4),  \
    py::arg(ARG_NAME(A5)) ARG_DEFVAL(A5),  \
    py::arg(ARG_NAME(A6)) ARG_DEFVAL(A6),  \
    py::arg(ARG_NAME(A7)) ARG_DEFVAL(A7),  \
    py::arg(ARG_NAME(A8)) ARG_DEFVAL(A8),  \
    py::arg(ARG_NAME(A9)) ARG_DEFVAL(A9),  \
    py::arg(ARG_NAME(A10)) ARG_DEFVAL(A10) \
  );

// NOLINTBEGIN(whitespace/line_length)

#define DEF0_AS(N, AS) DEF0_F(AS, { return NAMESPACE::N(); } )
#define DEF1_AS(N, AS, A1) DEF1_F(AS, A1, { return NAMESPACE::N(ARG_ID(A1)); } )
#define DEF2_AS(N, AS, A1, A2) DEF2_F(AS, A1, A2, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2)); } )
#define DEF3_AS(N, AS, A1, A2, A3) DEF3_F(AS, A1, A2, A3, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3)); } )
#define DEF4_AS(N, AS, A1, A2, A3, A4) DEF4_F(AS, A1, A2, A3, A4, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4)); } )
#define DEF5_AS(N, AS, A1, A2, A3, A4, A5) DEF5_F(AS, A1, A2, A3, A4, A5, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5)); } )
#define DEF6_AS(N, AS, A1, A2, A3, A4, A5, A6) DEF6_F(AS, A1, A2, A3, A4, A5, A6, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6)); } )
#define DEF7_AS(N, AS, A1, A2, A3, A4, A5, A6, A7) DEF7_F(AS, A1, A2, A3, A4, A5, A6, A7, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7)); } )
#define DEF8_AS(N, AS, A1, A2, A3, A4, A5, A6, A7, A8) DEF8_F(AS, A1, A2, A3, A4, A5, A6, A7, A8, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7), ARG_ID(A8)); } )
#define DEF9_AS(N, AS, A1, A2, A3, A4, A5, A6, A7, A8, A9) DEF9_F(AS, A1, A2, A3, A4, A5, A6, A7, A8, A9, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7), ARG_ID(A8), ARG_ID(A9)); } )
#define DEF10_AS(N, AS, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) DEF10_F(AS, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7), ARG_ID(A8), ARG_ID(A9), ARG_ID(A10)); } )

#define DEF0(N) DEF0_F(N, { return NAMESPACE::N(); } )
#define DEF1(N, A1) DEF1_F(N, A1, { return NAMESPACE::N(ARG_ID(A1)); } )
#define DEF2(N, A1, A2) DEF2_F(N, A1, A2, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2)); } )
#define DEF3(N, A1, A2, A3) DEF3_F(N, A1, A2, A3, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3)); } )
#define DEF4(N, A1, A2, A3, A4) DEF4_F(N, A1, A2, A3, A4, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4)); } )
#define DEF5(N, A1, A2, A3, A4, A5) DEF5_F(N, A1, A2, A3, A4, A5, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5)); } )
#define DEF6(N, A1, A2, A3, A4, A5, A6) DEF6_F(N, A1, A2, A3, A4, A5, A6, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6)); } )
#define DEF7(N, A1, A2, A3, A4, A5, A6, A7) DEF7_F(N, A1, A2, A3, A4, A5, A6, A7, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7)); } )
#define DEF8(N, A1, A2, A3, A4, A5, A6, A7, A8) DEF8_F(N, A1, A2, A3, A4, A5, A6, A7, A8, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7), ARG_ID(A8)); } )
#define DEF9(N, A1, A2, A3, A4, A5, A6, A7, A8, A9) DEF9_F(N, A1, A2, A3, A4, A5, A6, A7, A8, A9, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7), ARG_ID(A8), ARG_ID(A9)); } )
#define DEF10(N, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) DEF10_F(N, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, { return NAMESPACE::N(ARG_ID(A1), ARG_ID(A2), ARG_ID(A3), ARG_ID(A4), ARG_ID(A5), ARG_ID(A6), ARG_ID(A7), ARG_ID(A8), ARG_ID(A9), ARG_ID(A10)); } )

// NOLINTEND(whitespace/line_length)

#endif  // MUJOCO_PYTHON_STUDIO_DEAR_IMGUI_DEAR_IMGUI_MACROS_H_
