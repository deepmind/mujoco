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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_FILAMENT_CONTEXT_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_FILAMENT_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <span>

#include <backend/Platform.h>
#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/SwapChain.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include "render/filament/core/material_manager.h"
#include "render/filament/core/object_manager.h"

namespace mujoco {

// Manages the filament::Renderer and provides APIs for rendering scenes.
class FilamentContext : public mjrfContext {
 public:
  explicit FilamentContext(const mjrfContextConfig* config);
  ~FilamentContext();

  FilamentContext(const FilamentContext&) = delete;
  FilamentContext& operator=(const FilamentContext&) = delete;

  // Queues the given render requests for rendering. This function copies the
  // necessary data from the requests into the renderer thread and returns
  // immediately afterwards. The renderer thread will then perform the actual
  // rendering on the GPU. Callers can use WaitForFrame to block until the
  // rendering is complete.
  mjrfFrameHandle Render(
      std::span<const mjrfRenderRequest> render_requests,
      std::span<const mjrfReadRequest> read_requests = {});

  // Blocks until the given frame has completed rendering.
  void WaitForFrame(mjrfFrameHandle frame_handle);

  // Flushes pending commands to the backend and pumps completed callbacks.
  void Flush();

  // Sets the clear color for the renderer.
  void SetClearColor(const filament::math::float4& color);

  // Returns information about the frame.
  void GetFrameStats(mjrfFrameHandle frame, mjrfFrameStats* stats_out) const;

  filament::Engine* GetEngine() const { return engine_; }

  filament::Engine::Backend GetBackend() const { return backend_; }

  ObjectManager* GetObjectManager() const { return object_manager_.get(); }

  MaterialManager* GetMaterialManager() const {
    return material_manager_.get();
  }

  static FilamentContext* downcast(mjrfContext* context) {
    return static_cast<FilamentContext*>(context);
  }
  static const FilamentContext* downcast(const mjrfContext* context) {
    return static_cast<const FilamentContext*>(context);
  }

 private:
  void ValidateSwapChains(std::span<const mjrfRenderRequest> render_requests);

  mjrfContextConfig config_;
  filament::Engine::Backend backend_;
  filament::Engine* engine_ = nullptr;
  filament::Renderer* renderer_ = nullptr;
  filament::SwapChain* window_swap_chain_ = nullptr;
  filament::SwapChain* offscreen_swap_chain_ = nullptr;
  std::unique_ptr<filament::backend::Platform> platform_;
  std::unique_ptr<ObjectManager> object_manager_;
  std::unique_ptr<MaterialManager> material_manager_;
  int window_width_ = 0;
  int window_height_ = 0;
  int offscreen_width_ = 0;
  int offscreen_height_ = 0;
  std::uint64_t frame_counter_ = 0;
  std::uint64_t completed_frame_ = 0;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_FILAMENT_CONTEXT_H_
