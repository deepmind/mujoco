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

#include "render/filament/core/filament_context.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/RenderTarget.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>
#include <filament/View.h>
#include <math/vec4.h>
#include <utils/FixedCapacityVector.h>
#include <utils/compiler.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/filament_platform_factory.h"
#include "render/filament/core/material_manager.h"
#include "render/filament/core/object_manager.h"
#include "render/filament/core/render_target.h"
#include "render/filament/core/scene_view.h"

namespace mujoco {

FilamentContext::FilamentContext(const mjrfContextConfig* config)
    : config_(*config) {
  FilamentPlatformSetup setup = CreateFilamentPlatform(config_);
  backend_ = setup.backend;
  platform_ = std::move(setup.platform);

  filament::Engine::Config engine_config;
  engine_config.commandBufferSizeMB *= 8;
  engine_config.perFrameCommandsSizeMB *= 8;
  engine_config.perRenderPassArenaSizeMB *= 8;
  // A model can define several directional lights, and the headlight is one.
  engine_config.enableMultipleDirectionalLights = true;

  filament::Engine::Builder engine_builder;
  engine_builder.config(&engine_config);
  engine_builder.backend(backend_);
  engine_builder.platform(platform_.get());
  engine_builder.feature("backend.disable_parallel_shader_compile",
                         setup.disable_parallel_shader_compile);
  engine_ = engine_builder.build();

  renderer_ = engine_->createRenderer();
#ifdef __EMSCRIPTEN__
  window_swap_chain_ = engine_->createSwapChain(nullptr);
#else
  if (config_.native_window) {
    window_swap_chain_ = engine_->createSwapChain(config_.native_window);
  }
#endif

  object_manager_ = std::make_unique<ObjectManager>(engine_);
  material_manager_ = std::make_unique<MaterialManager>(object_manager_.get());

  SetClearColor({0.0, 0.0, 0.0, 1.0});
}

FilamentContext::~FilamentContext() {
  engine_->flushAndWait();
  material_manager_.reset();
  object_manager_.reset();
  engine_->destroy(renderer_);
  engine_->destroy(window_swap_chain_);
  engine_->destroy(offscreen_swap_chain_);
  filament::Engine::destroy(engine_);
}

mjrfFrameHandle FilamentContext::Render(
    std::span<const mjrfRenderRequest> requests,
    std::span<const mjrfReadRequest> read_requests) {
  if (read_requests.size() > 1) {
    mju_error("Only one read request is supported for now.");
  }

  if constexpr (UTILS_HAS_THREADING) {
    // Wait until previous frame is completed before requesting a new frame.
    engine_->flushAndWait();
    completed_frame_ = frame_counter_;
  }

  ValidateSwapChains(requests);

  std::unordered_map<mjrfScene*, std::vector<const mjrfRenderRequest*>>
      scene_to_requests;
  for (const mjrfRenderRequest& request : requests) {
    scene_to_requests[request.scene].push_back(&request);
  }

  material_manager_->PrepareToRender();
  for (auto& [scene, requests] : scene_to_requests) {
    SceneView::downcast(scene)->PrepareToRender(requests);
  }
  material_manager_->RemoveUnusedMaterials();

  bool render_began = false;
  mjrfRenderTarget* current_target = nullptr;
  for (const mjrfRenderRequest& request : requests) {
    if (request.target != current_target && render_began) {
      renderer_->endFrame();
      render_began = false;
    }
    current_target = request.target;

    if (current_target == nullptr) {
      if (!read_requests.empty()) {
        mju_error("Cannot read pixels from the window.");
      }

      if (!render_began) {
        render_began = renderer_->beginFrame(window_swap_chain_);
      }
      if (!render_began) {
        break;
      }
      if (render_began) {
        SceneView* scene_view = SceneView::downcast(request.scene);
        scene_view->Render(renderer_, request);
      }
    } else {
      if (read_requests.empty()) {
        mju_error(
            "Rendering to a render target without a read request is "
            "pointless.");
      }

      const mjrfReadRequest& read_request = read_requests[0];
      if (read_request.num_bytes == 0) {
        mju_error("Output buffer size is zero.");
      }

      if (!render_began) {
        render_began = renderer_->beginFrame(offscreen_swap_chain_);
      }
      if (!render_began) {
        break;
      }
      if (render_began) {
        SceneView* scene_view = SceneView::downcast(request.scene);
        scene_view->Render(renderer_, request);
        RenderTarget* render_target = RenderTarget::downcast(request.target);
        render_target->ReadColorPixels(renderer_, (uint8_t*)read_request.output,
                                       read_request.num_bytes);
      }
    }
  }

  if (render_began) {
    renderer_->endFrame();
  }
  if constexpr (!UTILS_HAS_THREADING) {
    engine_->execute();
  }

  const mjrfFrameHandle frame = ++frame_counter_;
  if (!read_requests.empty()) {
    engine_->flushAndWait();
    completed_frame_ = frame;
    if (read_requests[0].read_completed) {
      read_requests[0].read_completed(read_requests[0].user_data);
    }
  }

  return frame;
}

void FilamentContext::WaitForFrame(mjrfFrameHandle frame_handle) {
  if (completed_frame_ < frame_handle) {
    engine_->flushAndWait();
    completed_frame_ = frame_counter_;
  }
}

void FilamentContext::Flush() {
  engine_->flush();
  engine_->pumpMessageQueues();
}

void FilamentContext::SetClearColor(const filament::math::float4& color) {
  filament::Renderer::ClearOptions opts;
  opts.clear = true;
  opts.discard = true;
  opts.clearColor = color;
  renderer_->setClearOptions(opts);
}

void FilamentContext::GetFrameStats(mjrfFrameHandle frame,
                                    mjrfFrameStats* stats_out) const {
  utils::FixedCapacityVector<filament::Renderer::FrameInfo> frame_info =
      renderer_->getFrameInfoHistory(1);
  if (!frame_info.empty()) {
    const int64_t ns = frame_info[0].denoisedGpuFrameDuration;
    stats_out->frame_rate = 1.0e9 / static_cast<double>(ns);
  } else {
    stats_out->frame_rate = 0.0;
  }
}

void FilamentContext::ValidateSwapChains(
    std::span<const mjrfRenderRequest> requests) {
  // Determine the maximum extents of the requests.
  int max_window_width = 0;
  int max_window_height = 0;
  int max_offscreen_width = 0;
  int max_offscreen_height = 0;

  for (const mjrfRenderRequest& request : requests) {
    const int width = request.viewport.width + request.viewport.left;
    const int height = request.viewport.height + request.viewport.bottom;
    if (request.target == nullptr) {
      max_window_width = std::max(max_window_width, width);
      max_window_height = std::max(max_window_height, height);
    } else {
      max_offscreen_width = std::max(max_offscreen_width, width);
      max_offscreen_height = std::max(max_offscreen_height, height);
    }
  }

  if (max_window_width && max_window_height) {
    if (max_window_width != window_width_ ||
        max_window_height != window_height_) {
      engine_->destroy(window_swap_chain_);
      window_width_ = max_window_width;
      window_height_ = max_window_height;
      if (config_.native_window) {
        window_swap_chain_ = engine_->createSwapChain(config_.native_window);
      } else {
        window_swap_chain_ =
            engine_->createSwapChain(window_width_, window_height_);
      }
    }
  }

  if (max_offscreen_width && max_offscreen_height) {
    if (max_offscreen_width != offscreen_width_ ||
        max_offscreen_height != offscreen_height_) {
      engine_->destroy(offscreen_swap_chain_);
      offscreen_width_ = max_offscreen_width;
      offscreen_height_ = max_offscreen_height;
      offscreen_swap_chain_ =
          engine_->createSwapChain(offscreen_width_, offscreen_height_);
    }
  }
}

}  // namespace mujoco
