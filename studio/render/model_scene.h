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

#ifndef MUJOCO_STUDIO_RENDER_MODEL_SCENE_H_
#define MUJOCO_STUDIO_RENDER_MODEL_SCENE_H_

#include <functional>
#include <memory>
#include <span>

#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/model_decorations.h"
#include "render/filament/support/model_lights.h"
#include "render/filament/support/model_renderables.h"
#include "render/filament/support/model_resources.h"

namespace mujoco::studio {

// Populates an mjrfScene with Lights and Renderables from an mjModel.
class ModelScene {
 public:
  using DrawTextAtFn = std::function<void(const char*, float, float, float)>;

  ModelScene(mjrfContext* ctx, mjrfScene* scene, mjModel* model,
             DrawTextAtFn draw_text_at_fn = nullptr);

  // Updates the mjrfScene with the current mjData state of the model.
  void Sync(const mjModel* model, mjData* data, mjrRect viewport, const mjvOption* vis_options,
            const mjvPerturb* perturb, mjvCamera* camera, std::span<const mjvGeom> extra_geoms);

  float GetHeadlightIntensity() { return model_lights_->GetHeadlightIntensity(); }

 private:
  mjrfScene* scene_;
  std::unique_ptr<ModelResources> model_resources_;
  std::unique_ptr<ModelLights> model_lights_;
  std::unique_ptr<ModelRenderables> model_renderables_;
  std::unique_ptr<ModelDecorations> model_decorations_;
  DrawTextAtFn draw_text_at_fn_;
  int offsamples_ = -1;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_RENDER_MODEL_SCENE_H_
