// Copyright 2026 DeepMind Technologies Limited
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

#include "studio/hal/graphics_mode.h"

#include <string_view>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

bool IsFilament(GraphicsMode gfx_mode) {
  return gfx_mode == GraphicsMode::FilamentOpenGl || gfx_mode == GraphicsMode::FilamentVulkan ||
         gfx_mode == GraphicsMode::FilamentVulkanSoftware ||
         gfx_mode == GraphicsMode::FilamentWebGl ||
         gfx_mode == GraphicsMode::FilamentOpenGlHeadless ||
         gfx_mode == GraphicsMode::FilamentOpenGlSoftware;
}

bool IsOpenGl(GraphicsMode gfx_mode) {
  return gfx_mode == GraphicsMode::FilamentOpenGl ||
         gfx_mode == GraphicsMode::FilamentOpenGlHeadless ||
         gfx_mode == GraphicsMode::FilamentOpenGlSoftware;
}

bool IsVulkan(GraphicsMode gfx_mode) {
  return gfx_mode == GraphicsMode::FilamentVulkan ||
         gfx_mode == GraphicsMode::FilamentVulkanSoftware;
}

bool IsWebGl(GraphicsMode gfx_mode) { return gfx_mode == GraphicsMode::FilamentWebGl; }

bool IsHeadless(GraphicsMode gfx_mode) {
  return gfx_mode == GraphicsMode::FilamentOpenGlHeadless ||
         gfx_mode == GraphicsMode::FilamentOpenGlSoftware;
}

bool IsSoftware(GraphicsMode gfx_mode) {
  return gfx_mode == GraphicsMode::FilamentOpenGlSoftware ||
         gfx_mode == GraphicsMode::FilamentVulkanSoftware;
}

GraphicsMode GraphicsModeFromString(std::string_view str, GraphicsMode default_mode) {
  if (str == "opengl") {
    return GraphicsMode::FilamentOpenGl;
  } else if (str == "vulkan") {
    return GraphicsMode::FilamentVulkan;
  } else if (str == "vulkan_software") {
    return GraphicsMode::FilamentVulkanSoftware;
  } else if (str == "web" || str == "webgl") {
    return GraphicsMode::FilamentWebGl;
  } else if (str == "opengl_headless") {
    return GraphicsMode::FilamentOpenGlHeadless;
  } else if (str == "opengl_software") {
    return GraphicsMode::FilamentOpenGlSoftware;
  } else if (str.empty()) {
    return default_mode;
  } else {
    mju_error("Unsupported graphics mode: %s", str.data());
    return default_mode;
  }
}

}  // namespace mujoco::studio
