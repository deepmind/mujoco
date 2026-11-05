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

#include <pybind11/pybind11.h>

namespace mujoco::python::studio {

void InitNativeViewer(pybind11::module_& m);
void InitSim(pybind11::module_& m);
void InitUx(pybind11::module_& m);
void InitWindow(pybind11::module_& m);
#ifdef MUJOCO_STUDIO_WITH_WEB
void InitHeadlessUi(pybind11::module_& m);
void InitStatePayload(pybind11::module_& m);
#endif

}  // namespace mujoco::python::studio

PYBIND11_MODULE(_studio, m, pybind11::mod_gil_not_used()) {
  auto native_viewer_cc = m.def_submodule("native_viewer_cc");
  mujoco::python::studio::InitNativeViewer(native_viewer_cc);

  auto sim = m.def_submodule("sim");
  mujoco::python::studio::InitSim(sim);

  auto ux = m.def_submodule("ux");
  mujoco::python::studio::InitUx(ux);

  auto window = m.def_submodule("window");
  mujoco::python::studio::InitWindow(window);

#ifdef MUJOCO_STUDIO_WITH_WEB
  auto headless_ui = m.def_submodule("headless_ui");
  mujoco::python::studio::InitHeadlessUi(headless_ui);

  auto state_payload = m.def_submodule("state_payload");
  mujoco::python::studio::InitStatePayload(state_payload);
#endif
}
