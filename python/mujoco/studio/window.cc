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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include <imgui.h>
#include <implot.h>
#include <mujoco/studio/hal/graphics_mode.h>
#include <mujoco/studio/hal/window.h>
#include <mujoco/studio/io/resources.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace {

static bool IsCuda() {
#ifdef CUDA
  return true;
#else
  return false;
#endif
}

static bool IsCrd() {
  const char* display = getenv("DISPLAY");
  return display ? strcmp(display, ":20") == 0 : false;
}

static intptr_t GetImGuiContext() {
  return reinterpret_cast<intptr_t>(ImGui::GetCurrentContext());
}

static intptr_t GetImPlotContext() {
  return reinterpret_cast<intptr_t>(ImPlot::GetCurrentContext());
}

class Window {
 public:
  Window(const std::string& title, int width, int height,
         std::string graphics_mode_str) {
    pybind11::gil_scoped_release no_gil;

    mujoco::studio::RegisterProviders();

    mujoco::studio::Window::Config config;
    config.gfx_mode = mujoco::studio::GraphicsModeFromString(
        graphics_mode_str, mujoco::studio::GraphicsMode::FilamentOpenGl);
    window_ = std::make_unique<mujoco::studio::Window>(
        title, width, height, config);
    ImPlot::CreateContext();
  }

  ~Window() {
    pybind11::gil_scoped_release no_gil;
    if (ImPlot::GetCurrentContext() != nullptr) {
      ImPlot::DestroyContext(nullptr);
    }
    window_.reset();
  }

  bool NewFrame() {
    pybind11::gil_scoped_release no_gil;
    const mujoco::studio::Window::Status window_status = window_->NewFrame();
    return window_status == mujoco::studio::Window::Status::kRunning;
  }

  void Present(std::optional<pybind11::buffer> pixels = std::nullopt) {
    std::span<const std::byte> buf = {};
    pybind11::buffer_info info;
    if (pixels.has_value()) {
      info = pixels->request();
      if (!PyBuffer_IsContiguous(info.view(), 'C')) {
        throw pybind11::value_error("Buffer must be C-contiguous.");
      }
      auto data = static_cast<std::byte*>(info.ptr);
      auto size = static_cast<size_t>(info.size * info.itemsize);
      buf = std::span<const std::byte>{data, size};
    }
    pybind11::gil_scoped_release no_gil;
    window_->EndFrame();
    window_->Present(buf);
  }

  int GetWidth() { return window_->GetWidth(); }
  int GetHeight() { return window_->GetHeight(); }
  float GetScale() { return window_->GetScale(); }

  uint64_t GetNativeWindowHandle() {
    return reinterpret_cast<uint64_t>(window_->GetNativeWindowHandle());
  }

  int GetGraphicsMode() {
    return static_cast<int>(window_->GetGraphicsMode());
  }

  std::string GetDropFile() { return window_->GetDropFile(); }

 private:
  std::unique_ptr<mujoco::studio::Window> window_;
};

}  // namespace

namespace mujoco::python::studio {

void InitWindow(pybind11::module_& m) {
  mujoco::studio::RegisterProviders();

  m.def("IsCrd", &IsCrd);
  m.def("IsCuda", &IsCuda);
  m.def("GetImGuiContext", &GetImGuiContext);
  m.def("GetImPlotContext", &GetImPlotContext);

  pybind11::class_<Window> cls(m, "Window");
  cls.def(pybind11::init<const std::string&, int, int, const std::string&>());
  cls.def("NewFrame", &Window::NewFrame);
  cls.def("Present", &Window::Present,
          pybind11::arg("pixels") = pybind11::none());
  cls.def("GetWidth", &Window::GetWidth);
  cls.def("GetHeight", &Window::GetHeight);
  cls.def("GetScale", &Window::GetScale);
  cls.def("GetDropFile", &Window::GetDropFile);
  cls.def("GetGraphicsMode", &Window::GetGraphicsMode);
  cls.def("GetNativeWindowHandle", &Window::GetNativeWindowHandle);
}

}  // namespace mujoco::python::studio

#ifndef MUJOCO_STUDIO_COMBINED_MODULE
PYBIND11_MODULE(window, m, pybind11::mod_gil_not_used()) {
  mujoco::python::studio::InitWindow(m);
}
#endif
