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

// Python bindings for the state payload and model bundle wire formats.
//
// WebViewer serializes the /state WebSocket payload with this module each
// frame and the /model bundle on every model change; the browser parses both
// with the same headers.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mujoco/mujoco.h>
#include "structs.h"
#include "state_payload.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Serialize the complete state WebSocket payload (see state_payload.h):
// physics state, render state, extra geoms and viewer input as tagged blocks.
static py::bytes SerializeStatePayload(
    uint32_t model_crc32, int physics_spec, const py::bytes& physics_state,
    const mujoco::python::MjvCameraWrapper& camera,
    const mujoco::python::MjvPerturbWrapper& perturb,
    const mujoco::python::MjvOptionWrapper& vis_options,
    const mujoco::python::MjModelWrapper& model,
    const std::vector<uint8_t>& render_flags, const py::iterable& extra_geoms,
    const mujoco::studio::ViewerInputState& viewer_input) {
  std::vector<mjvGeom> geoms;
  geoms.reserve(py::len_hint(extra_geoms));
  for (py::handle item : extra_geoms) {
    const auto& geom_wrapper =
        py::cast<const mujoco::python::MjvGeomWrapper&>(item);
    if (geom_wrapper.get()) {
      geoms.push_back(*geom_wrapper.get());
    }
  }

  std::string physics = physics_state;
  std::vector<std::byte> buffer;
  {
    py::gil_scoped_release no_gil;
    buffer = mujoco::studio::SerializeStatePayload(
        model_crc32, physics_spec, physics.data(), physics.size(),
        *camera.get(), *perturb.get(), *vis_options.get(), model.get()->opt,
        model.get()->vis, model.get()->stat, render_flags, geoms.data(),
        geoms.size(), viewer_input);
  }
  return py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// Parses a ClientCameraMessage sent by the controlling browser; None if it is
// malformed.
static std::optional<mujoco::studio::ClientCameraMessage> ParseClientCamera(
    const py::bytes& message) {
  const std::string_view data = message;
  mujoco::studio::ClientCameraMessage parsed;
  if (!mujoco::studio::ParseClientCameraMessage(data.data(), data.size(),
                                                &parsed)) {
    return std::nullopt;
  }
  return parsed;
}

// Serializes `camera` as the controlling browser sends it.
static py::bytes SerializeClientCamera(
    const mujoco::python::MjvCameraWrapper& camera, uint32_t generation) {
  const mujoco::studio::ClientCameraMessage message =
      mujoco::studio::MakeClientCameraMessage(*camera.get(), generation);
  return py::bytes(reinterpret_cast<const char*>(&message), sizeof(message));
}

// Upper bound of a serialized payload for a model whose physics state is
// `physics_bytes` long. Used to size the StateServer's shared memory.
static size_t MaxStatePayloadSize(size_t physics_bytes) {
  return mujoco::studio::MaxStatePayloadSize(physics_bytes);
}

// Serializes the /model payload: the MJB plus the file bytes of every
// by-reference texture. Textures whose file cannot be read are reported with a
// UserWarning and left for the browser to render untextured, as the native
// viewer does.
static py::bytes SerializeModelBundle(
    const mujoco::python::MjModelWrapper& model) {
  std::vector<std::byte> buffer;
  std::vector<std::string> unreadable;
  {
    py::gil_scoped_release no_gil;
    buffer = mujoco::studio::SerializeModelBundle(model.get(), &unreadable);
  }
  for (const std::string& what : unreadable) {
    std::string message = "texture left out of the model bundle: " + what;
    if (PyErr_WarnEx(PyExc_UserWarning, message.c_str(), 1) < 0) {
      throw py::error_already_set();
    }
  }
  return py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// Parses a model bundle back into (mjb_bytes, [(path, bytes), ...]). This is
// the parser the browser runs; exposed so tests can round-trip bundles.
static py::tuple ParseModelBundle(const py::bytes& bundle) {
  std::string data = bundle;
  mujoco::studio::ModelBundleView view;
  if (!mujoco::studio::ParseModelBundle(data.data(), data.size(), &view)) {
    throw py::value_error("malformed model bundle");
  }
  py::list resources;
  for (const mujoco::studio::ModelBundleResource& r : view.resources) {
    resources.append(py::make_tuple(
        py::str(std::string(r.path)),
        py::bytes(reinterpret_cast<const char*>(r.data), r.size)));
  }
  return py::make_tuple(
      py::bytes(reinterpret_cast<const char*>(view.mjb), view.mjb_size),
      resources);
}

namespace mujoco::python::studio {

void InitStatePayload(py::module_& m) {
  py::module_::import("mujoco._structs");
  m.doc() = "MuJoCo web viewer state payload serialization";

  using mujoco::studio::ViewerInputState;
  py::class_<ViewerInputState>(m, "ViewerInputState")
      .def(py::init([](bool mouse_focus, int camera_index, bool keyboard_focus,
                       uint32_t camera_generation) {
             return ViewerInputState{
                 .mouse_focus = mouse_focus ? 1 : 0,
                 .camera_index = camera_index,
                 .keyboard_focus = keyboard_focus ? 1 : 0,
                 .camera_generation = camera_generation,
             };
           }),
           py::kw_only(), py::arg("mouse_focus") = false,
           py::arg("camera_index") = 0, py::arg("keyboard_focus") = false,
           py::arg("camera_generation") = 0)
      .def_readwrite("mouse_focus", &ViewerInputState::mouse_focus)
      .def_readwrite("camera_index", &ViewerInputState::camera_index)
      .def_readwrite("keyboard_focus", &ViewerInputState::keyboard_focus)
      .def_readwrite("camera_generation", &ViewerInputState::camera_generation);

  using mujoco::studio::ClientCameraMessage;
  py::class_<ClientCameraMessage>(m, "ClientCameraMessage")
      .def_readonly("generation", &ClientCameraMessage::generation)
      .def_readonly("type", &ClientCameraMessage::type)
      .def_readonly("fixedcamid", &ClientCameraMessage::fixedcamid)
      .def_readonly("trackbodyid", &ClientCameraMessage::trackbodyid)
      .def_readonly("orthographic", &ClientCameraMessage::orthographic)
      .def_property_readonly("lookat",
                             [](const ClientCameraMessage& message) {
                               return py::make_tuple(message.lookat[0],
                                                     message.lookat[1],
                                                     message.lookat[2]);
                             })
      .def_readonly("distance", &ClientCameraMessage::distance)
      .def_readonly("azimuth", &ClientCameraMessage::azimuth)
      .def_readonly("elevation", &ClientCameraMessage::elevation);
  m.def("parse_client_camera", &ParseClientCamera, py::arg("message"));
  m.def("serialize_client_camera", &SerializeClientCamera, py::arg("camera"),
        py::arg("generation"));

  m.def("serialize_state_payload", &SerializeStatePayload,
        py::arg("model_crc32"), py::arg("physics_spec"),
        py::arg("physics_state"), py::arg("camera"), py::arg("perturb"),
        py::arg("vis_options"), py::arg("model"), py::arg("render_flags"),
        py::arg("extra_geoms"), py::arg("viewer_input"));
  m.def("max_state_payload_size", &MaxStatePayloadSize);
  m.def("serialize_model_bundle", &SerializeModelBundle);
  m.def("parse_model_bundle", &ParseModelBundle);
  m.attr("MAX_EXTRA_GEOMS") = mujoco::studio::kMaxExtraGeoms;
}

}  // namespace mujoco::python::studio

#ifndef MUJOCO_STUDIO_COMBINED_MODULE
PYBIND11_MODULE(state_payload, m, pybind11::mod_gil_not_used()) {
  mujoco::python::studio::InitStatePayload(m);
}
#endif
