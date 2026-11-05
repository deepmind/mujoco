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

#include "render/filament/support/model_lights.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include <math/TVecHelpers.h>
#include <math/mat4.h>
#include <math/mathfwd.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/filament_util.h"
#include "render/filament/support/model_resources.h"

namespace mujoco {

using filament::math::float3;
using filament::math::float4;
using filament::math::mat3;
using filament::math::mat4;

static UniquePtr<mjrfTexture> CreateFallbackIndirectLightTexture(
    mjrfContext* ctx) {
  const std::string filename = ResolveFilamentAssetPath("ibl.ktx");
  mjResource* resource =
      mju_openResource("", filename.c_str(), nullptr, nullptr, 0);
  if (!resource) {
    mju_error("Failed to open resource: %s", filename.c_str());
  }
  const void* bytes = nullptr;
  const int nbytes = mju_readResource(resource, &bytes);
  if (bytes == nullptr || nbytes <= 0) {
    mju_error("Failed to read resource: %s", filename.c_str());
  }

  mjrfTextureConfig config;
  mjrf_defaultTextureConfig(&config);
  config.width = 1;
  config.height = 1;
  config.sampler_type = mjTEXTURE_CUBE;
  config.format = mjPIXEL_FORMAT_KTX;
  config.color_space = mjCOLORSPACE_AUTO;

  auto texture = CreateTexture(ctx, config);

  mjrfTextureData payload;
  mjrf_defaultTextureData(&payload);
  payload.bytes = bytes;
  payload.num_bytes = nbytes;
  payload.release =
      +[](void* user_data) { mju_closeResource((mjResource*)user_data); };
  payload.user_data = resource;

  mjrf_uploadTexture(texture.get(), &payload);
  return texture;
}

// Sets the color and intensity of the model's i-th light.
//
// Lights that don't specify an intensity share the fallback scene light
// intensity. Filament ranks directional lights by intensity alone to choose the
// one that casts shadows, so lights with the same intensity tie and the choice
// between them is arbitrary. For non-image lights, move the brightness of the
// diffuse color (its largest component) into the intensity: color times
// intensity is unchanged, but the brightest light casts the shadows.
static void SetLightColorAndIntensity(mjrfLight* light, const mjModel* model,
                                      int i, bool use_fallback,
                                      float fallback_intensity) {
  float3 color = ReadFloat3(model->light_diffuse, i);
  float intensity = model->light_intensity[i];
  if (use_fallback) {
    intensity = fallback_intensity;
    const float brightness = std::max({color.r, color.g, color.b});
    if (model->light_type[i] != mjLIGHT_IMAGE && brightness > 0.0f) {
      color /= brightness;
      intensity *= brightness;
    }
  }
  mjrf_setLightColor(light, color.v);
  mjrf_setLightIntensity(light, intensity);
}

ModelLights::ModelLights(mjrfScene* scene, ModelResources* model_resources)
    : scene_(scene), model_resources_(model_resources) {
  const mjModel* model = model_resources->GetModel();
  shadowsize_ = model->vis.quality.shadowsize;
  fallback_head_light_intensity_ =
      ReadElement(model, "filament.fallback.head_light_intensity",
                  fallback_head_light_intensity_);
  fallback_scene_light_intensity_ =
      ReadElement(model, "filament.fallback.scene_light_intensity",
                  fallback_scene_light_intensity_);
  fallback_environment_light_intensity_ =
      ReadElement(model, "filament.fallback.environment_light_intensity",
                  fallback_environment_light_intensity_);
  Prepare();
}

ModelLights::~ModelLights() {
  for (auto& iter : lights_) {
    mjrf_removeSceneLight(scene_, iter.get());
  }
  lights_.clear();
  if (fallback_ibl_) {
    mjrf_removeSceneLight(scene_, fallback_ibl_.get());
  }
  fallback_ibl_.reset();
  if (fallback_directional_) {
    mjrf_removeSceneLight(scene_, fallback_directional_.get());
  }
  fallback_directional_.reset();
}

void ModelLights::Prepare() {
  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();

  // Default to the shadow map resolution in mjVisual, like the classic
  // renderer. Filament clamps shadow map sizes to 2048.
  int default_shadow_map_size = std::min(model->vis.quality.shadowsize, 2048);
  default_shadow_map_size =
      ReadElement(model, "filament.shadows.map_size", default_shadow_map_size);
  shadow_map_size_ = default_shadow_map_size;

  bool has_image_based_light = false;
  bool has_directional_light = false;
  float total_light_intensity = 0.0f;
  for (int i = 0; i < model->nlight; ++i) {
    total_light_intensity += model->light_intensity[i];
    has_directional_light |= model->light_type[i] == mjLIGHT_DIRECTIONAL;

    if (model->light_type[i] == mjLIGHT_IMAGE) {
      mjrfLightParams params;
      mjrf_defaultLightParams(&params);
      params.type = mjLIGHT_IMAGE;
      params.texture = model_resources_->GetTexture(model->light_texid[i]);
      params.intensity = model->light_intensity[i];
      auto light_obj = CreateLight(ctx, params);
      mjrf_addSceneLight(scene_, light_obj.get());
      lights_.emplace_back(std::move(light_obj));
      has_image_based_light = true;
    } else {
      mjrfLightParams params;
      mjrf_defaultLightParams(&params);
      params.color[0] = model->light_diffuse[(3 * i) + 0];
      params.color[1] = model->light_diffuse[(3 * i) + 1];
      params.color[2] = model->light_diffuse[(3 * i) + 2];
      params.type = (mjtLightType)model->light_type[i];
      params.cast_shadow = model->light_castshadow[i];
      params.bulb_radius = model->light_bulbradius[i];
      params.range = model->light_range[i];
      params.intensity = model->light_intensity[i];
      params.shadow_map_size = default_shadow_map_size;
      if (params.type == mjLIGHT_SPOT) {
        params.cutoff = model->light_cutoff[i];
        params.softness = model->light_softness[i];
      }

      auto light_obj = CreateLight(ctx, params);
      mjrf_addSceneLight(scene_, light_obj.get());
      lights_.emplace_back(std::move(light_obj));
    }
  }

  // Workaround for an upstream filament bug, present since 1.74.0
  // (https://github.com/google/filament/issues/10249): under the non-PCF
  // shadow types (VSM/DPCF/PCSS), a scene where a punctual (spot/point) light
  // casts shadows renders fully black unless a directional light is also
  // present. Mere presence suffices: zero intensity, shadows off. Keep
  // filament's directional slot occupied with an invisible light whenever the
  // model has no directional light.
  if (!has_directional_light) {
    mjrfLightParams params;
    mjrf_defaultLightParams(&params);
    params.type = mjLIGHT_DIRECTIONAL;
    params.cast_shadow = 0;
    params.intensity = 0.0f;
    fallback_directional_ = CreateLight(ctx, params);
    mjrf_addSceneLight(scene_, fallback_directional_.get());
  }

  if (!has_image_based_light && total_light_intensity > 0.0f) {
    // Create a black indirect light to ensure that the skybox is
    // oriented to respect mujoco's Z-up convention.
    mjrfLightParams params;
    mjrf_defaultLightParams(&params);
    params.type = mjLIGHT_IMAGE;
    params.intensity = 10.0f;
    fallback_ibl_ = CreateLight(ctx, params);
    mjrf_addSceneLight(scene_, fallback_ibl_.get());
  }

  // There are no "physical" lights in the scene which means we're likely
  // dealing with a "classic renderer" scene. In this case, let's add a
  // default environment light and set the light intensity ourselves.
  if (total_light_intensity == 0.0f) {
    // Create a fallback environment light.
    fallback_ibl_texture_ = CreateFallbackIndirectLightTexture(ctx);

    mjrfLightParams params;
    mjrf_defaultLightParams(&params);
    params.type = mjLIGHT_IMAGE;
    params.texture = fallback_ibl_texture_.get();
    params.intensity = fallback_environment_light_intensity_;
    fallback_ibl_ = CreateLight(ctx, params);
    mjrf_addSceneLight(scene_, fallback_ibl_.get());

    // The headlight is a render request option, not a scene light; publish its
    // intensity for the renderer to pass along.
    headlight_intensity_ = fallback_head_light_intensity_;

    // Distribute the fallback scene light intensity among the lights.
    if (!lights_.empty()) {
      fallback_intensity_ = fallback_scene_light_intensity_ / lights_.size();
      for (int i = 0; i < model->nlight; ++i) {
        if (lights_[i]) {
          SetLightColorAndIntensity(lights_[i].get(), model, i, true,
                                    fallback_intensity_);
        }
      }
    }
  }

  mjrf_setSceneSkybox(scene_, model_resources_->GetSkyboxTexture());
}

void ModelLights::UpdateShadowMapSize() {
  const mjModel* model = model_resources_->GetModel();
  if (model->vis.quality.shadowsize != shadowsize_) {
    shadowsize_ = model->vis.quality.shadowsize;
    shadow_map_size_ = std::min(shadowsize_, 2048);
    for (auto& light : lights_) {
      mjrf_setLightShadowMapSize(light.get(), shadow_map_size_);
    }
  }
}

void ModelLights::Update(const mjData* data) {
  UpdateShadowMapSize();
  if (data == nullptr) {
    return;
  }
  const mjModel* model = model_resources_->GetModel();
  float total_light_intensity = 0.0f;
  for (int i = 0; i < model->nlight; ++i) {
    total_light_intensity += model->light_intensity[i];
  }

  for (int i = 0; i < model->nlight; ++i) {
    mjrfLight* light = lights_[i].get();
    const float3 pos = ReadFloat3(data->light_xpos, i);
    const float3 dir = ReadFloat3(data->light_xdir, i);
    mjrf_setLightPose(light, pos.v, dir.v);

    mjrf_setLightActive(light, model->light_active[i]);
    SetLightColorAndIntensity(light, model, i, total_light_intensity == 0.0f,
                              fallback_intensity_);
    mjrf_setLightRange(light, model->light_range[i]);
    mjrf_setLightCutoff(light, model->light_cutoff[i]);
    mjrf_setLightSoftness(light, model->light_softness[i]);
    mjrf_setLightBulbRadius(light, model->light_bulbradius[i]);
    mjrf_setLightCastShadow(light, model->light_castshadow[i]);
  }
}

mjrfLight* ModelLights::GetLight(int index) {
  if (index < 0 || index >= lights_.size()) {
    return nullptr;
  }
  return lights_[index].get();
}
}  // namespace mujoco
