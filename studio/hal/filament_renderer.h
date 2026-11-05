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

#ifndef MUJOCO_STUDIO_HAL_FILAMENT_RENDERER_H_
#define MUJOCO_STUDIO_HAL_FILAMENT_RENDERER_H_

#include <cstddef>
#include <memory>
#include <span>

#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/imgui_bridge.h"
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/model_decorations.h"
#include "render/filament/support/model_lights.h"
#include "render/filament/support/model_renderables.h"
#include "render/filament/support/model_resources.h"
#include "studio/hal/graphics_mode.h"

namespace mujoco::studio {

// Renders the mujoco simulation and the imgui state using the Filament-based
// MuJoCo renderer.
//
// For OpenGL, two different Filament configurations are available: normal and
// headless. Normal rendering assumes that we will be rendering to an x11
// window and therefore will use a x11-based context. Headless assumes that we
// will be rendering to a texture and will use an EGL context. Vulkan and WebGL
// have no need for such a distinction. If rendering to a window surface (e.g.
// x11), it requires a pointer to the native window to do so.
class FilamentRenderer {
 public:
  FilamentRenderer(void* native_window, GraphicsMode gfx_mode);
  ~FilamentRenderer();

  FilamentRenderer(const FilamentRenderer&) = delete;
  FilamentRenderer& operator=(const FilamentRenderer&) = delete;

  // Initializes the renderer with the given mjModel.
  void Init(const mjModel* model);

  // Synchronizes scene state from the simulation model and data.
  void Sync(const mjModel* model, mjData* data, const mjvPerturb* perturb, mjvCamera* camera,
            const mjvOption* vis_option, int width, int height,
            std::span<mjvGeom> extra_geoms = {});

  // Renders the previously synchronized scene and current UX state.
  void Submit(int width, int height, std::span<std::byte> pixels = {});

  // Populates the given output buffer with RGB888 pixel data. The size of the
  // output buffer must be at least width * height * 3.
  void RenderToTexture(const mjModel* model, mjData* data, mjvCamera* camera, int width, int height,
                       std::byte* output);

  // Uploads an image to the backend for GUI rendering, returning the texture
  // ID for the texture. The ID can be used in subsequent calls to update the
  // texture data. An empty pixels argument will free the texture if it exists.
  // A texture ID of 0 will create a new texture.
  int UploadImage(int tex_id, std::span<const std::byte> pixels, int width, int height);

  // Rendering flags.
  mjtByte* GetRenderFlags() { return render_flags_; }

  // Returns the GPU time of the last rendered frame, in milliseconds.
  double GetGpuFrameMs();

 private:
  // Resets the renderer; no rendering will occur until Init() is called again.
  void Deinit();

  void BuildMainRenderRequest(mjrfRenderRequest* request, const mjVisual& vis,
                              const mjvOption& vis_option, const mjrRect& viewport,
                              const mjrCamera& camera);
  void BuildUxRenderRequest(mjrfRenderRequest* request, const mjrRect& viewport);

  UniquePtr<mjrfContext> filament_context_{nullptr, nullptr};
  UniquePtr<mjrfScene> main_scene_{nullptr, nullptr};
  UniquePtr<mjrfScene> ux_scene_{nullptr, nullptr};
  UniquePtr<mjrfRenderTarget> render_target_{nullptr, nullptr};
  std::unique_ptr<ImguiBridge> imgui_bridge_;
  std::unique_ptr<ModelResources> model_resources_;
  std::unique_ptr<ModelLights> model_lights_;
  std::unique_ptr<ModelRenderables> model_renderables_;
  std::unique_ptr<ModelDecorations> model_decorations_;
  mjrfRenderRequest render_requests_[2];
  mjtByte render_flags_[mjNRNDFLAG];
  int offsamples_ = -1;
  double gpu_frame_ms_ = 0;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_HAL_FILAMENT_RENDERER_H_
