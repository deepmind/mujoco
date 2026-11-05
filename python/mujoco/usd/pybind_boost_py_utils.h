// Copyright 2025 DeepMind Technologies Limited
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

#ifndef MUJOCO_PYTHON_USD_PYBIND_BOOST_PY_UTILS_H_
#define MUJOCO_PYTHON_USD_PYBIND_BOOST_PY_UTILS_H_

#include "pxr/external/boost/python.hpp"  // IWYU pragma: keep
#include <pxr/pxr.h>

PXR_NAMESPACE_USING_DIRECTIVE

#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

/**
 * @brief A generic type caster for types registered with Boost.Python.
 *
 * This template implements the pybind11 `type_caster` interface, enabling
 * seamless use of C++ types exposed via Boost.Python within pybind11-bound
 * functions and classes.
 *
 * @tparam T The C++ type to be converted.
 */
template <typename T>
struct boost_python_converter {
 public:
  // This macro provides required boilerplate for a custom type caster,
  // such as the `value` member and casting operators. The type name
  // is set by the specialization macro below.
  PYBIND11_TYPE_CASTER(T, _(""));

  /**
   * @brief Converts a Python object to a C++ object.
   *
   * This 'load' method is pybind11's entry point for converting a Python
   * object into a C++ function argument. It leverages `boost::python::extract`
   * to perform the check and the conversion.
   *
   * @param src A handle to the source Python object.
   * @return `true` if the conversion is successful, `false` otherwise.
   */
  bool load(handle src, bool /*implicit_conversion*/) {
    namespace bpy = pxr_boost::python;

    // Create a boost::python::object from the raw PyObject*.
    // The `borrowed` policy ensures Boost.Python doesn't try to prematurely
    // manage the object's reference count.
    // NOLINTBEGIN(misc-include-cleaner)
    bpy::object obj(bpy::handle<>(bpy::borrowed(src.ptr())));
    // NOLINTEND(misc-include-cleaner)

    // Use boost::python::extract to check for convertibility and then convert.
    bpy::extract<T> extractor(obj);  // NOLINT(misc-include-cleaner)
    if (extractor.check()) {
      value = extractor();
      return true;
    }

    return false;
  }

  /**
   * @brief Converts a C++ object to a Python object.
   *
   * This `cast` method is pybind11's entry point for converting a C++
   * object (e.g., a return value) into a Python object. It uses the
   * `boost::python::object` constructor for the conversion.
   *
   * @param src The C++ object to convert.
   * @return A handle to the newly created Python object.
   */
  static handle cast(const T& src, return_value_policy /*policy*/,
                     handle /*parent*/) {
    namespace bpy = pxr_boost::python;

    // Create a new Python object from the C++ source.
    bpy::object obj(src);  // NOLINT(misc-include-cleaner)

    // The new Python object's reference count must be incremented before
    // its raw pointer is passed to pybind11, which will then manage it.
    return handle(bpy::incref(obj.ptr()));  // NOLINT(misc-include-cleaner)
  }
};

}  // namespace detail
}  // namespace pybind11

/**
 * @brief A convenience macro to register a C++ type for interop.
 *
 * This macro specializes `pybind11::detail::type_caster` for the given `Type`,
 * delegating all conversion logic to the `boost_python_converter`. This is
 * all that's needed to make a Boost.Python type available in pybind11.
 *
 * @param Type The C++ type to register (e.g., `MyClass`).
 * @param PyName A string literal for the Python type name, used in error
 * messages (e.g., `_("my_module.MyClass")`).
 */
#define REGISTER_BOOST_PYTHON_INTEROP(Type, PyName)                \
  namespace pybind11 {                                             \
  namespace detail {                                               \
  template <>                                                      \
  struct type_caster<Type> : public boost_python_converter<Type> { \
    static constexpr auto name = PyName;                           \
  };                                                               \
  }                                                                \
  }

#endif  // MUJOCO_PYTHON_USD_PYBIND_BOOST_PY_UTILS_H_
