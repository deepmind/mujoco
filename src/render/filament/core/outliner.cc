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

#include "render/filament/core/outliner.h"

#include <cstdint>
#include <memory>

#include <filament/Camera.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderTarget.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TextureSampler.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <math/vec4.h>
#include <utils/EntityManager.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/object_manager.h"
#include "render/filament/core/render_target.h"
#include "render/filament/core/texture.h"

namespace mujoco {

Outliner::Outliner(ObjectManager* object_mgr, uint8_t layer_mask,
                   uint8_t scene_layer_mask, filament::math::float4 color,
                   float thickness, float occlusion_dim)
    : object_mgr_(object_mgr),
      engine_(object_mgr->GetEngine()),
      layer_mask_(layer_mask),
      scene_layer_mask_(scene_layer_mask),
      color_(color),
      thickness_(thickness),
      occlusion_dim_(occlusion_dim) {}

Outliner::~Outliner() { Reset(); }

void Outliner::Prepare(int width, int height) {
  if (width == width_ && height == height_) {
    return;
  }
  Reset();
  width_ = width;
  height_ = height;
  if (width_ <= 0 || height_ <= 0) {
    return;
  }

  auto& em = utils::EntityManager::get();

  // Sets up a pass with a full-screen quad that renders with a given material.
  auto setup_fullscreen = [&](int pass, ObjectManager::MaterialType type) {
    filament::Material* material = object_mgr_->GetMaterial(type);
    filament::VertexBuffer* quad_vb = object_mgr_->GetQuadVertexBuffer();
    filament::IndexBuffer* quad_ib = object_mgr_->GetQuadIndexBuffer();
    const filament::RenderableManager::PrimitiveType primitive_type =
        filament::RenderableManager::PrimitiveType::TRIANGLES;

    material_instances_[pass] = material->createInstance();
    quads_[pass] = em.create();
    filament::RenderableManager::Builder(1)
        .geometry(0, primitive_type, quad_vb, quad_ib)
        .material(0, material_instances_[pass])
        .culling(false)
        .receiveShadows(false)
        .castShadows(false)
        .build(*engine_, quads_[pass]);
    scenes_[pass] = engine_->createScene();
    scenes_[pass]->addEntity(quads_[pass]);
    views_[pass]->setScene(scenes_[pass]);
  };

  // Sets up a pass to use the given render target as the source and/or output.
  auto bind = [&](int pass, int src, int out) {
    // Set the `source` texture as the color texture from a render target.
    if (src >= 0) {
      const filament::TextureSampler sampler(
          filament::TextureSampler::MinFilter::NEAREST,
          filament::TextureSampler::MagFilter::NEAREST);
      material_instances_[pass]->setParameter(
          "source", targets_[src]->GetColorTexture()->GetFilamentTexture(),
          sampler);
    }
    // Set the output of the view to the given render target.
    if (out >= 0) {
      views_[pass]->setRenderTarget(targets_[out]->GetFilamentRenderTarget());
    }
  };

  // Set up the render targets: one for the scene depth, one for the flattened
  // selection mask and its depth, and two ping-pong targets for chaining the
  // jump flood passes together.
  for (int i = 0; i < kNumTargets; ++i) {
    mjrfRenderTargetConfig config;
    mjrf_defaultRenderTargetConfig(&config);
    config.color_format = mjPIXEL_FORMAT_RGBA8;
    config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
    config.width = width;
    config.height = height;
    targets_[i] = std::make_unique<RenderTarget>(engine_, config);
    targets_[i]->Prepare(width, height);
  }

  // Setup orthographic camera for full-screen quad rendering.
  camera_ = engine_->createCamera(em.create());
  camera_->setProjection(filament::Camera::Projection::ORTHO, -1.0, 1.0, -1.0,
                         1.0, -1.0, 1.0);
  camera_->lookAt({0, 0, 1}, {0, 0, 0}, {0, 1, 0});

  // Setup all the views.
  for (auto& view : views_) {
    view = engine_->createView();
    view->setCamera(camera_);
    view->setViewport({0, 0, (uint32_t)width, (uint32_t)height});
    view->setPostProcessingEnabled(false);
    view->setShadowingEnabled(false);
    view->setMultiSampleAntiAliasingOptions({.enabled = false});
  }

  // In the first pass, we render the scene layers that can occlude the
  // outline; only the resulting depth is used. In the second pass, we render
  // the given scene again, but only the objects marked as outlines. We assume
  // that the objects have already been assigned the kOutlineFlatten material.
  views_[kPassSceneDepth]->setVisibleLayers(0xff, scene_layer_mask_);
  views_[kPassFlatten]->setVisibleLayers(0xff, layer_mask_);

  // All subsequent passes are full-screen post-processing passes.
  setup_fullscreen(kPassJumpFlood1, ObjectManager::kOutlineJumpFlood);
  setup_fullscreen(kPassJumpFlood2, ObjectManager::kOutlineJumpFlood);
  setup_fullscreen(kPassJumpFlood3, ObjectManager::kOutlineJumpFlood);
  setup_fullscreen(kPassJumpFlood4, ObjectManager::kOutlineJumpFlood);
  setup_fullscreen(kPassJumpFlood5, ObjectManager::kOutlineJumpFlood);
  setup_fullscreen(kPassDrawOutline, ObjectManager::kOutlineComposite);

  // Chain the passes together such that the output of a pass is the input to
  // the next pass. The scene passes have no input (we are just rendering the
  // scene) and the last pass has no output (we are just rendering the outline
  // to the externally provided target). The flatten pass has its own target
  // (rather than a ping-pong target) so that its depth attachment survives the
  // jump flood passes and can be sampled by the composite pass.
  bind(kPassSceneDepth, -1, kTargetSceneDepth);
  bind(kPassFlatten, -1, kTargetFlatten);
  bind(kPassJumpFlood1, kTargetFlatten, kTargetPing);
  bind(kPassJumpFlood2, kTargetPing, kTargetPong);
  bind(kPassJumpFlood3, kTargetPong, kTargetPing);
  bind(kPassJumpFlood4, kTargetPing, kTargetPong);
  bind(kPassJumpFlood5, kTargetPong, kTargetPing);
  bind(kPassDrawOutline, kTargetPing, -1);

  // Bind the parameters for each pass. For the jump flood passes, the step
  // parameter determines how far to propagate the outline in each pass.
  material_instances_[kPassJumpFlood1]->setParameter("step", 16.0f);
  material_instances_[kPassJumpFlood2]->setParameter("step", 8.0f);
  material_instances_[kPassJumpFlood3]->setParameter("step", 4.0f);
  material_instances_[kPassJumpFlood4]->setParameter("step", 2.0f);
  material_instances_[kPassJumpFlood5]->setParameter("step", 1.0f);

  // The final pass renders the actual outline onto a render target. It samples
  // the scene depth and the selection depth to dim occluded outline pixels.
  const filament::TextureSampler depth_sampler(
      filament::TextureSampler::MinFilter::NEAREST,
      filament::TextureSampler::MagFilter::NEAREST);
  material_instances_[kPassDrawOutline]->setParameter("color", color_);
  material_instances_[kPassDrawOutline]->setParameter("width", thickness_);
  material_instances_[kPassDrawOutline]->setParameter("dim", occlusion_dim_);
  material_instances_[kPassDrawOutline]->setParameter(
      "scene_depth",
      targets_[kTargetSceneDepth]->GetDepthTexture()->GetFilamentTexture(),
      depth_sampler);
  material_instances_[kPassDrawOutline]->setParameter(
      "selection_depth",
      targets_[kTargetFlatten]->GetDepthTexture()->GetFilamentTexture(),
      depth_sampler);

  // Commit all the material instances to the engine.
  for (auto& material_instance : material_instances_) {
    if (material_instance) {
      material_instance->commit(*engine_);
    }
  }
}

void Outliner::Reset() {
  auto& em = utils::EntityManager::get();

  for (auto& scene : scenes_) {
    if (scene) {
      engine_->destroy(scene);
      scene = nullptr;
    }
  }
  for (auto& quad : quads_) {
    if (quad) {
      engine_->destroy(quad);
      em.destroy(quad);
      quad = utils::Entity();
    }
  }
  for (auto& material_instance : material_instances_) {
    if (material_instance) {
      engine_->destroy(material_instance);
      material_instance = nullptr;
    }
  }
  for (auto& view : views_) {
    if (view) {
      engine_->destroy(view);
      view = nullptr;
    }
  }
  if (camera_) {
    utils::Entity entity = camera_->getEntity();
    engine_->destroyCameraComponent(entity);
    em.destroy(entity);
    camera_ = nullptr;
  }
  for (auto& target : targets_) {
    target.reset();
  }
  width_ = 0;
  height_ = 0;
}

void Outliner::Render(filament::Renderer* renderer, filament::View* view,
                      filament::RenderTarget* render_target) {
  const filament::Viewport viewport = view->getViewport();
  Prepare(viewport.width, viewport.height);

  // The offscreen passes render into targets the size of the viewport, so
  // they cover them from the origin. Only the final pass draws onto the
  // caller's target, where the viewport may be offset (e.g. a scene above a
  // status bar).
  for (auto& pass_view : views_) {
    pass_view->setViewport({0, 0, viewport.width, viewport.height});
  }
  views_[kPassDrawOutline]->setViewport(viewport);

  // Re-render the view's scene to capture the scene depth (used for occlusion
  // dimming) and to create the flattened selection mask.
  auto prev_clear_opts = renderer->getClearOptions();
  renderer->setClearOptions({.clearColor = {0, 0, 0, 0}, .clear = true});
  views_[kPassSceneDepth]->setScene(view->getScene());
  views_[kPassSceneDepth]->setCamera(&view->getCamera());
  renderer->render(views_[kPassSceneDepth]);
  views_[kPassFlatten]->setScene(view->getScene());
  views_[kPassFlatten]->setCamera(&view->getCamera());
  renderer->render(views_[kPassFlatten]);
  renderer->setClearOptions(prev_clear_opts);

  // Run the jump flood steps to expand the selection mask
  for (int i = 0; i < kNumJumpFloodPasses; ++i) {
    renderer->render(views_[kPassJumpFlood1 + i]);
  }

  // Render the final pass as an outline onto the provided render target.
  views_[kPassDrawOutline]->setRenderTarget(render_target);
  renderer->render(views_[kPassDrawOutline]);
  views_[kPassDrawOutline]->setRenderTarget(nullptr);
}
}  // namespace mujoco
