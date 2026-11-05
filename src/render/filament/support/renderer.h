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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_RENDERER_H_
#define MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_RENDERER_H_

#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <mujoco/mjrfilament.h>
#include "render/filament/support/mjrf_ptr.h"

namespace mujoco {

// Coordinates filament context, scenes, render targets, and views in order to
// render a scene.
class Renderer {
 public:
  Renderer(const mjrfContextConfig& config);
  ~Renderer();

  // Strongly-typed handles for different types of resources.
  enum class Kind { Scene, RenderTarget };
  template <Kind T>
  struct Handle {
    int id = -1;
    bool operator==(const Handle& other) const = default;
    bool operator!=(const Handle& other) const = default;
  };

  using SceneHandle = Handle<Kind::Scene>;
  using RenderTargetHandle = Handle<Kind::RenderTarget>;

  static constexpr SceneHandle kInvalidScene = {-1};
  static constexpr RenderTargetHandle kInvalidRenderTarget = {-1};
  static constexpr RenderTargetHandle kWindowRenderTarget = {-2};

  // Returns the scene with the given name. If no such scene exists, creates
  // a new scene with the given name.
  mjrfScene* GetScene(std::string_view name);

  // Returns the scene associated with the given handle, or nullptr if the
  // handle is invalid.
  mjrfScene* GetScene(SceneHandle handle);

  // Returns the handle of the scene with the given name, or kInvalidScene if no
  // such scene exists.
  SceneHandle GetSceneHandle(std::string_view name) const;

  // Adds a new render target with the given name and config. Returns the handle
  // of the new target.
  RenderTargetHandle CreateRenderTarget(std::string_view name,
                                        const mjrfRenderTargetConfig& config);

  // Resizes the specified render target.
  void ResizeRenderTarget(RenderTargetHandle handle, int width, int height);

  // Returns the pixels of the render target with the given handle.
  std::span<const std::byte> GetPixels(RenderTargetHandle handle);

  // Returns the render target with the given name, or kInvalidRenderTarget if
  // no such target exists.
  RenderTargetHandle GetRenderTargetHandle(std::string_view name) const;

  // Represents a single view to render. Views can be rendered in parallel to
  // different viewports in different render targets.
  struct View {
    View() { std::memset(&camera, 0, sizeof(mjrCamera)); }

    // The scene to render.
    SceneHandle scene = kInvalidScene;

    // The render target into which to render the scene. By default, we assume
    // the window is the render target.
    RenderTargetHandle render_target = kWindowRenderTarget;

    // The camera into which to render the scene.
    mjrCamera camera;

    // The area in the render target into which to render the scene.
    float rel_viewport[4] = {0.0f, 0.0f, 1.0f, 1.0f};

    // The draw mode to use for rendering the scene.
    mjrDrawMode draw_mode = mjDRAW_MODE_DEFAULT;

    // Whether to enable various rendering effects.
    bool post_processing = true;
    bool reflections = true;
    bool shadows = true;
    bool skybox = true;
    bool haze = true;
    struct Headlight {
      float color[3] = {0.0f, 0.0f, 0.0f};
      float intensity = 0.0f;
    };
    std::optional<Headlight> headlight;
  };

  // Renders the given views.
  void Render(std::span<const View> views);

  float GetGpuFrameMs() const { return gpu_frame_ms_; }

  // Returns the underlying Filament context.
  mjrfContext* GetContext() { return ctx_.get(); }

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

 private:
  RenderTargetHandle UpdateRenderRequest(mjrfRenderRequest& request,
                                         const View& view);
  void UpdateReadRequest(mjrfReadRequest& request, RenderTargetHandle target);

  UniquePtr<mjrfContext> ctx_;
  std::vector<UniquePtr<mjrfScene>> scenes_;
  std::unordered_map<std::string, SceneHandle> scene_lookup_;

  std::vector<UniquePtr<mjrfRenderTarget>> targets_;
  std::vector<mjrfRenderTargetConfig> target_configs_;
  std::vector<std::vector<std::byte>> target_pixels_;
  std::unordered_map<std::string, RenderTargetHandle> render_target_lookup_;

  int window_width_ = 0;
  int window_height_ = 0;
  float gpu_frame_ms_ = 0;
};
}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_RENDERER_H_
