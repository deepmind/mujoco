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

#include <mujoco/experimental/usd/layer_sink.h>
#include <mujoco/mjdata.h>  // IWYU pragma: keep
#include <mujoco/mujoco.h>
#include "pybind.h"  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>

namespace py = ::pybind11;

namespace mujoco::usd::python {

PYBIND11_MODULE(layer_sink, m, pybind11::mod_gil_not_used()) {
  py::module::import("mujoco._usd");
  py::module::import("mujoco._structs");

  m.doc() =
      "Defines a LayerSink that writes pose data to the current edit target "
      "layer.";

  py::class_<mujoco::usd::LayerSink>(m, "LayerSink")
      .def(py::init<pxr::UsdStagePtr>())
      .def("update", [](mujoco::usd::LayerSink& self, py::handle data,
                        std::vector<pxr::SdfPath> body_paths) {
        py::object cls = py::module_::import("mujoco._structs").attr("MjData");
        if (!py::isinstance(data, cls)) {
          throw py::type_error("Expected mujoco._structs.MjData");
        }
        const auto* raw_data = reinterpret_cast<const mjData*>(
            data.attr("_address").cast<std::uintptr_t>());
        self.Update(raw_data, body_paths);
      });
}
}  // namespace mujoco::usd::python
