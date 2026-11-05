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

// Python bindings for MuJoCo platform simulation components.

#include <tuple>

#include <mujoco/mujoco.h>
#include <mujoco/studio/sim/sim_history.h>
#include <mujoco/studio/sim/step_control.h>
#include "structs.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using StepControl = mujoco::studio::StepControl;

namespace mujoco::python::studio {

void InitSim(py::module_& m) {
  py::module_::import("mujoco._structs");
  m.doc() = "MuJoCo platform simulation bindings for Link.";

  py::enum_<StepControl::Status>(m, "StepStatus")
      .value("OK", StepControl::Status::kOk)
      .value("PAUSED", StepControl::Status::kPaused)
      .value("AUTO_RESET", StepControl::Status::kAutoReset)
      .value("DIVERGED", StepControl::Status::kDiverged);

  py::enum_<StepControl::PauseState>(m, "PauseState")
      .value("UNPAUSED", StepControl::PauseState::kUnpaused)
      .value("NORMAL_PAUSED", StepControl::PauseState::kNormalPaused);

  py::class_<StepControl>(m, "StepControl")
      .def(py::init<>())
      .def(
          "advance",
          [](StepControl& self, py::object model_obj, py::object data_obj) {
            mjModel* m = nullptr;
            mjData* d = nullptr;
            if (!model_obj.is_none()) {
              m = py::cast<mujoco::python::MjModelWrapper&>(model_obj).get();
            }
            if (!data_obj.is_none()) {
              d = py::cast<mujoco::python::MjDataWrapper&>(data_obj).get();
            }
            py::gil_scoped_release no_gil;
            return self.Advance(m, d);
          },
          py::arg("model"), py::arg("data"),
          "Step physics forward, respecting speed settings and refresh budget.")
      .def("force_sync", &StepControl::ForceSync,
           "Ensures the next Advance() will synchronize time and step once.")
      .def("get_speed", &StepControl::GetSpeed,
           "Returns the desired simulation speed as a percentage of real time.")
      .def("get_speed_measured", &StepControl::GetSpeedMeasured,
           "Returns the measured simulation speed as a percentage of real "
           "time, or a negative value before the first measurement.")
      .def("set_speed", &StepControl::SetSpeed, py::arg("speed"),
           "Sets the desired speed (clamped to [0.1%, 100%]).")
      .def("set_pause_state", &StepControl::SetPauseState, py::arg("state"),
           "Sets the pause state of the simulation.")
      .def("get_pause_state", &StepControl::GetPauseState,
           "Returns the current pause state.")
      .def("request_single_step", &StepControl::RequestSingleStep,
           "Request a single step if paused.")
      .def("consume_single_step_request",
           &StepControl::ConsumeSingleStepRequest,
           "Returns whether a single step is pending, clearing the request.")
      .def(
          "get_noise_parameters",
          [](const StepControl& self) {
            float noise_scale, noise_rate;
            self.GetNoiseParameters(noise_scale, noise_rate);
            return std::make_tuple(noise_scale, noise_rate);
          },
          "Returns the noise parameters.")
      .def("set_noise_parameters", &StepControl::SetNoiseParameters,
           py::arg("noise_scale"), py::arg("noise_rate"),
           "Sets the noise parameters.");

  using SimHistory = mujoco::studio::SimHistory;
  constexpr int max_history = 2048;
  constexpr int max_bytes = 128 * 1024 * 1024;  // 128 MiB
  py::class_<SimHistory>(m, "SimHistory")
      .def(py::init<>())
      .def("init", &SimHistory::Init, py::arg("state_size"),
           py::arg("max_history") = max_history,
           py::arg("max_bytes") = max_bytes,
           "Clears and initializes the history buffer to hold `state_size` "
           "mjtNum states.")
      .def("get_index", &SimHistory::GetIndex,
           "Returns the current history offset (0 is the most recent state).")
      .def(
          "set_index",
          [](SimHistory& self, int offset) { self.SetIndex(offset); },
          py::arg("offset"),
          "Sets the history offset (0 is the most recent state)."
      )
      .def("size", &SimHistory::Size,
           "Returns the number of recorded states.")
      .def("capacity", &SimHistory::Capacity,
           "Returns the most states the buffer can hold, as set by init().")
      .def("set_size", &SimHistory::SetSize, py::arg("size"),
           "Sets the number of recorded states.");
}

}  // namespace mujoco::python::studio

#ifndef MUJOCO_STUDIO_COMBINED_MODULE
PYBIND11_MODULE(sim, m, pybind11::mod_gil_not_used()) {
  mujoco::python::studio::InitSim(m);
}
#endif
