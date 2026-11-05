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

#include <cstdint>
#include <string>
#include <vector>

#include <mujoco/experimental/usd/writer.h>
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>
#include "pybind.h"  // IWYU pragma: keep
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>

namespace py = ::pybind11;

namespace mujoco::usd::python {
namespace {

template <typename T>
T* CastMjPointer(py::handle obj, const char* module_name,
                 const char* type_name) {
  py::object cls = py::module_::import(module_name).attr(type_name);
  if (!py::isinstance(obj, cls)) {
    throw py::type_error(std::string("Expected ") + module_name + "." +
                         type_name);
  }
  return reinterpret_cast<T*>(obj.attr("_address").cast<std::uintptr_t>());
}

class PyUsdWriter {
 public:
  PyUsdWriter(pxr::UsdStagePtr stage, py::handle spec, py::handle model)
      : writer_(stage, CastMjPointer<mjSpec>(spec, "mujoco._specs", "MjSpec"),
                CastMjPointer<mjModel>(model, "mujoco._structs", "MjModel")) {}

  void AddSink(py::function fn) {
    writer_.AddSink([this, fn](const mjData* const /*data*/,
                               std::vector<pxr::SdfPath> body_paths) {
      fn(active_data_, body_paths);
    });
  }

  void Update(py::object data) {
    const auto* raw_data =
        CastMjPointer<const mjData>(data, "mujoco._structs", "MjData");
    py::object prev_data = active_data_;
    active_data_ = data;
    try {
      writer_.Update(raw_data);
      active_data_ = prev_data;
    } catch (...) {
      active_data_ = prev_data;
      throw;
    }
  }

 private:
  mujoco::usd::Writer writer_;
  py::object active_data_ = py::none();
};

}  // namespace

PYBIND11_MODULE(writer, m, pybind11::mod_gil_not_used()) {
  py::module::import("mujoco._usd");
  py::module::import("mujoco._structs");
  py::module::import("mujoco._specs");

  m.doc() = "Writer module for writing pose data back to Usd layers.";

  py::class_<PyUsdWriter>(m, "UsdWriter")
      .def(py::init<pxr::UsdStagePtr, py::handle, py::handle>())
      .def("add_sink", &PyUsdWriter::AddSink)
      .def("update", &PyUsdWriter::Update);
}
}  // namespace mujoco::usd::python
