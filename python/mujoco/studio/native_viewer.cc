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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <mujoco/render/filament/support/imgui_bridge.h>
#include <mujoco/studio/hal/graphics_mode.h>
#include <mujoco/studio/render/model_scene.h>
#include <mujoco/studio/ux/imgui_widgets.h>
#include "render_filament.h"
#include "structs.h"
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

using namespace mujoco;          // NOLINT(build/namespaces)
using namespace mujoco::python;  // NOLINT(build/namespaces)
using namespace mujoco::studio;  // NOLINT(build/namespaces)

namespace {

std::unique_ptr<ImguiBridge> MakeImguiBridge(mujoco::python::PyContext& ctx,
                                             PyScene& scene) {
  return std::make_unique<ImguiBridge>(ctx.get(), scene.get());
}

std::unique_ptr<ModelScene> MakeModelScene(mujoco::python::PyContext& ctx,
                                           PyScene& scene,
                                           MjModelWrapper& model) {
  return std::make_unique<ModelScene>(ctx.get(), scene.get(), model.get(),
                                      DrawTextAt);
}

intptr_t UploadImage(ImguiBridge& imgui_bridge, intptr_t tex_id,
                     const std::string img, int width, int height, int bpp) {
  py::gil_scoped_release no_gil;
  auto bytes = reinterpret_cast<const std::byte*>(img.data());
  const size_t num_bytes = img.size();
  return imgui_bridge.UploadImage(tex_id, {bytes, num_bytes}, width, height);
}

void SyncModelScene(ModelScene& model_scene, int width, int height,
                    const mujoco::python::MjModelWrapper& model,
                    mujoco::python::MjDataWrapper& data,
                    mujoco::python::MjvPerturbWrapper& perturb,
                    mujoco::python::MjvCameraWrapper& camera,
                    mujoco::python::MjvOptionWrapper& vis_options,
                    const py::iterable& extra_geoms) {
  std::vector<mjvGeom> geoms;
  geoms.reserve(py::len_hint(extra_geoms));
  for (py::handle item : extra_geoms) {
    const auto& geom_wrapper =
        py::cast<const mujoco::python::MjvGeomWrapper&>(item);
    if (geom_wrapper.get()) {
      geoms.push_back(*geom_wrapper.get());
    }
  }

  py::gil_scoped_release no_gil;
  model_scene.Sync(model.get(), data.get(), {0, 0, width, height},
                   vis_options.get(), perturb.get(), camera.get(), geoms);
}

bool RenderToPixels(int gfx_mode) {
  return IsHeadless(
    static_cast<mujoco::studio::GraphicsMode>(gfx_mode));
}

}  // namespace

namespace mujoco::python::studio {

void InitNativeViewer(pybind11::module_& m) {
  pybind11::module_::import("mujoco._structs");
  {
    py::class_<ImguiBridge> cls(m, "ImguiBridge", py::module_local());
    cls.def(py::init(&MakeImguiBridge), py::arg("ctx"), py::arg("scene"),
            py::keep_alive<1, 2>(), py::keep_alive<1, 3>());
    cls.def("update", &ImguiBridge::Update);
    cls.def("upload_image", &UploadImage, py::arg("tex_id"), py::arg("img"),
            py::arg("width"), py::arg("height"), py::arg("bpp"));
  }
  {
    py::class_<ModelScene> cls(m, "ModelScene", py::module_local());
    cls.def(py::init(&MakeModelScene), py::arg("ctx"), py::arg("scene"),
            py::arg("model"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>());
    cls.def("sync", &SyncModelScene, py::arg("width"), py::arg("height"),
            py::arg("model"), py::arg("data"), py::arg("perturb"),
            py::arg("camera"), py::arg("vis_options"), py::arg("extra_geoms"));
    cls.def("get_headlight_intensity", &ModelScene::GetHeadlightIntensity);
  }
  m.def("render_to_pixels", &RenderToPixels);
}

}  // namespace mujoco::python::studio

#ifndef MUJOCO_STUDIO_COMBINED_MODULE
PYBIND11_MODULE(native_viewer_cc, m, pybind11::mod_gil_not_used()) {
  mujoco::python::studio::InitNativeViewer(m);
}
#endif
