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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_SCENE_VIEW_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_SCENE_VIEW_H_

#include <memory>
#include <span>
#include <unordered_set>

#include <filament/Camera.h>
#include <filament/ColorGrading.h>
#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/color_grading_options.h"
#include "render/filament/core/light.h"
#include "render/filament/core/material_manager.h"
#include "render/filament/core/object_manager.h"
#include "render/filament/core/outliner.h"
#include "render/filament/core/reflection_manager.h"
#include "render/filament/core/renderable.h"
#include "render/filament/core/texture.h"
#include "render/filament/core/types.h"

namespace mujoco {

// Creates and owns the filament Scene and View (and Camera) classes.
//
// The filament Scene is populated with the objects (e.g. lights, renderables,
// skybox, etc.). It manages multiple views to support a variety of draw modes
// (e.g. normal, depth, segmentation, etc.) as well as reflective surfaces.
class SceneView : public mjrfScene {
 public:
  SceneView(ObjectManager* object_mgr, MaterialManager* material_mgr,
            const mjrfSceneParams& params);
  ~SceneView();

  SceneView(const SceneView&) = delete;
  SceneView& operator=(const SceneView&) = delete;

  // Adds/removes entities from the scene.
  void AddToScene(Light* light);
  void RemoveFromScene(Light* light);
  void AddToScene(Renderable* renderable);
  void RemoveFromScene(Renderable* renderable);
  void SetSkybox(const Texture* skybox_texture);

  // Performs necessary preparations in order to render the given requests.
  // Assumes that the Render() function will be called the same number of times
  // and in the same order with the given requests.
  void PrepareToRender(std::span<const mjrfRenderRequest*> requests);

  // Fulfills the given render request using the renderer.
  void Render(filament::Renderer* renderer, const mjrfRenderRequest& request);

  // Returns the filament Engine managing the scene.
  filament::Engine* GetEngine() const { return object_mgr_->GetEngine(); }

  // Returns the underlying filament View that is used for normal rendering.
  // Callers can update rendering settings (e.g. post processing) directly.
  filament::View* GetDefaultRenderView();

  // Helpers for managing the color grading options for the default render view.
  ColorGradingOptions GetColorGradingOptions() const;
  void SetColorGradingOptions(const ColorGradingOptions& opts);

  // Reads filament-specific settings from the mjModel and configures the
  // scene view accordingly.
  void Configure(const mjModel* model);

  // Sets the number of multi-sample anti-aliasing samples; <= 1 disables MSAA.
  void SetMsaa(int samples);

  // Sets the horizon haze ratio and RGBA color for the scene.
  void SetHaze(float ratio, const float rgba[4]);
  float GetHazeRatio() const;
  filament::math::float4 GetHazeColor() const;
  float GetFocusDistance() const { return focus_distance_; }
  void SetFocusDistance(float focus_distance) {
    focus_distance_ = focus_distance;
  }

  static SceneView* downcast(mjrfScene* scene) {
    return static_cast<SceneView*>(scene);
  }
  static const SceneView* downcast(const mjrfScene* scene) {
    return static_cast<const SceneView*>(scene);
  }

 private:
  ObjectManager* object_mgr_ = nullptr;
  MaterialManager* material_mgr_ = nullptr;
  filament::Scene* scene_ = nullptr;
  filament::Camera* camera_ = nullptr;
  filament::ColorGrading* color_grading_ = nullptr;
  ColorGradingOptions color_grading_options_;
  filament::View* main_view_;
  filament::View* depth_segment_view_;

  // Scene objects.
  std::unordered_set<Light*> lights_;
  // The headlight belongs to the render request, not to the scene, so it is
  // owned here and re-aimed at each request's camera.
  std::unique_ptr<Light> headlight_;
  std::unordered_set<Renderable*> renderables_;
  filament::Skybox* skybox_ = nullptr;
  std::unique_ptr<Renderable> haze_;
  float focus_distance_ = 0.0f;
  int numstacks_ = 16;
  int numslices_ = 28;
  int numquads_ = 4;

  // Custom view and camera for reflective surfaces.
  filament::View* reflect_view_ = nullptr;
  filament::Camera* reflect_camera_ = nullptr;

  MaterialManager::MaterialKey outline_material_key_;
  std::unique_ptr<Outliner> outliner_;
  std::unique_ptr<ReflectionManager> reflection_mgr_;
};
}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_SCENE_VIEW_H_
