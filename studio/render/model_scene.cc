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

#include "studio/render/model_scene.h"

#include <memory>
#include <span>
#include <utility>

#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/model_decorations.h"
#include "render/filament/support/model_lights.h"
#include "render/filament/support/model_renderables.h"
#include "render/filament/support/model_resources.h"

namespace mujoco::studio {

ModelScene::ModelScene(mjrfContext* ctx, mjrfScene* scene, mjModel* model,
                       DrawTextAtFn draw_text_at_fn)
    : scene_(scene), draw_text_at_fn_(std::move(draw_text_at_fn)) {
  offsamples_ = model->vis.quality.offsamples;

  model_resources_ = std::make_unique<ModelResources>(model, ctx);
  model_lights_ = std::make_unique<ModelLights>(scene, model_resources_.get());
  model_renderables_ = std::make_unique<ModelRenderables>(scene, model_resources_.get());
  model_decorations_ = std::make_unique<ModelDecorations>(ctx, scene, model);

  float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  const int id = mj_name2id(model, mjOBJ_NUMERIC, "filament.clear_color");
  if (id >= 0 && model->numeric_size[id] == 4) {
    const mjtNum* ptr = model->numeric_data + model->numeric_adr[id];
    for (int i = 0; i < 4; ++i) {
      clear_color[i] = static_cast<float>(ptr[i]);
    }
  }
  mjrf_setClearColor(ctx, &clear_color[0]);
  mjrf_configureSceneFromModel(scene, model);
}

void ModelScene::Sync(const mjModel* model, mjData* data, mjrRect viewport,
                      const mjvOption* vis_options, const mjvPerturb* perturb, mjvCamera* camera,
                      std::span<const mjvGeom> extra_geoms) {
  if (model->vis.quality.offsamples != offsamples_) {
    offsamples_ = model->vis.quality.offsamples;
    mjrf_setSceneMsaa(scene_, offsamples_);
  }

  model_lights_->Update(data);

  if (vis_options) {
    model_renderables_->SetOptions(*vis_options);
  }
  if (perturb && perturb->select > 0) {
    model_renderables_->MarkAsSelected(mjOBJ_BODY, perturb->select);
  } else if (perturb && perturb->flexselect >= 0) {
    model_renderables_->MarkAsSelected(mjOBJ_FLEX, perturb->flexselect);
  } else if (perturb && perturb->skinselect >= 0) {
    model_renderables_->MarkAsSelected(mjOBJ_SKIN, perturb->skinselect);
  } else {
    model_renderables_->MarkAsSelected(mjOBJ_UNKNOWN, -1);
  }
  model_renderables_->Update(data);

  model_decorations_->Update(data, vis_options, perturb, camera, viewport, draw_text_at_fn_,
                             extra_geoms);
}

}  // namespace mujoco::studio
