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

#include "render/filament/core/light.h"

#include <functional>
#include <numbers>
#include <utility>

#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Scene.h>
#include <math/mat3.h>
#include <math/vec3.h>
#include <utils/Entity.h>
#include <utils/EntityManager.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/texture.h"
#include "render/filament/support/filament_util.h"

namespace mujoco {

using filament::math::float3;
using filament::math::mat3f;

static std::pair<float, float> GetSpotLightConeAngles(float angle,
                                                      float softness) {
  // The light delivers its full intensity inside the inner cone, so that
  // illuminance follows E = I/d^2, and falls to zero over the outer
  // softness fraction of the cone angle.
  const float outer = angle * std::numbers::pi / 180.0f;
  const float inner = (1.0f - softness) * outer;
  return {inner, outer};
}

Light::Light(filament::Engine* engine, const mjrfLightParams& params)
    : engine_(engine), params_(params) {
  // Filament treats image-based lights (IBLs) as separate objects (i.e.
  // filament::IndirectLight) and so we need to handle IBLs specially.
  if (params.type == mjLIGHT_IMAGE) {
    filament::IndirectLight::Builder builder;
    if (params.texture) {
      // Allow null textures for fallback lights.
      const Texture* texture = Texture::downcast(params.texture);
      builder.reflections(texture->GetFilamentTexture());
      const Texture::SphericalHarmonics* spherical_harmonics =
          texture->GetSphericalHarmonics();
      if (spherical_harmonics != nullptr) {
        builder.irradiance(3, *spherical_harmonics);
      }
    }
    builder.intensity(params.intensity);
    // Rotate the light to match mujoco's Z-up convention.
    builder.rotation(mat3f::rotation(std::numbers::pi / 2, float3{1, 0, 0}));
    ibl_ = builder.build(*engine_);
    return;
  }

  filament::LightManager::Type type;
  switch (params.type) {
    case mjLIGHT_SPOT:
      // SPOT rather than FOCUSED_SPOT: MJCF intensity is candela, which must
      // not rescale with the cone angle.
      type = filament::LightManager::Type::SPOT;
      break;
    case mjLIGHT_DIRECTIONAL:
      type = filament::LightManager::Type::DIRECTIONAL;
      break;
    case mjLIGHT_POINT:
      type = filament::LightManager::Type::POINT;
      break;
    default:
      mju_error("Unsupported light type: %d", params.type);
      return;
  }

  filament::LightManager::Builder builder(type);
  builder.color(ReadFloat3(params.color));
  builder.intensityCandela(params.intensity);
  builder.castShadows(params.cast_shadow);
  if (type == filament::LightManager::Type::SPOT) {
    auto angles = GetSpotLightConeAngles(params.cutoff,
                                         params.softness);
    builder.spotLightCone(angles.first, angles.second);
  }
  if (type != filament::LightManager::Type::DIRECTIONAL) {
    builder.falloff(params.range);
  }
  filament::LightManager::ShadowOptions opts;
  opts.mapSize = 4096;
  opts.shadowCascades =
      type == filament::LightManager::Type::DIRECTIONAL ? 4 : 1;
  opts.shadowBulbRadius = params.bulb_radius;
  opts.mapSize = params.shadow_map_size;

  builder.shadowOptions(opts);

  entity_ = utils::EntityManager::get().create();
  if (entity_.isNull()) {
    mju_error("Failed to create light entity.");
  }
  builder.build(*engine_, entity_);
}

Light::~Light() noexcept {
  if (remove_from_scene_fn_) {
    auto on_destroy = std::move(remove_from_scene_fn_);
    on_destroy(this);
  }
  if (ibl_) {
    engine_->destroy(ibl_);
  } else {
    utils::EntityManager& em = utils::EntityManager::get();
    if (!entity_.isNull()) {
      engine_->destroy(entity_);
      em.destroy(entity_);
    }
  }
}

void Light::AddToScene(filament::Scene* scene,
                       std::function<void(Light*)> on_destroy) {
  if (assigned_scene_) {
    if (assigned_scene_ != scene) {
      mju_error("Cannot add light to multiple scenes.");
    }
    // Entities are already added to the scene.
    return;
  }
  if (ibl_) {
    scene->setIndirectLight(ibl_);
  } else {
    scene->addEntity(entity_);
  }
  assigned_scene_ = scene;
  remove_from_scene_fn_ = std::move(on_destroy);
}

void Light::RemoveFromScene(filament::Scene* scene) {
  if (assigned_scene_ != scene) {
    mju_error("Attempting to remove light from wrong scene.");
  }
  if (ibl_) {
    scene->setIndirectLight(nullptr);
  } else {
    scene->remove(entity_);
  }
  assigned_scene_ = nullptr;
  remove_from_scene_fn_ = nullptr;
}

void Light::SetTransform(filament::math::float3 position,
                         filament::math::float3 direction) {
  if (!ibl_) {
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    lm.setPosition(li, position);
    lm.setDirection(li, direction);
  }
}

void Light::SetColor(const filament::math::float3& color) {
  if (!ibl_) {
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    lm.setColor(li, color);
  }
  params_.color[0] = color.r;
  params_.color[1] = color.g;
  params_.color[2] = color.b;
}

void Light::SetIntensity(float intensity) {
  if (ibl_) {
    ibl_->setIntensity(intensity);
  } else {
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    lm.setIntensityCandela(li, intensity);
  }
  params_.intensity = intensity;
}

void Light::SetRange(float range) {
  if (params_.range != range && ibl_ == nullptr) {
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    lm.setFalloff(li, range);
  }
  params_.range = range;
}

void Light::SetCutoffAngle(float cutoff) {
  if (params_.cutoff != cutoff && ibl_ == nullptr) {
    auto angles = GetSpotLightConeAngles(cutoff, params_.softness);
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    lm.setSpotLightCone(li, angles.first, angles.second);
  }
  params_.cutoff = cutoff;
}

void Light::SetSoftness(float softness) {
  if (params_.softness != softness && ibl_ == nullptr) {
    auto angles = GetSpotLightConeAngles(params_.cutoff, softness);
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    lm.setSpotLightCone(li, angles.first, angles.second);
  }
  params_.softness = softness;
}

void Light::SetBulbRadius(float radius) {
  if (params_.bulb_radius != radius && ibl_ == nullptr) {
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    filament::LightManager::ShadowOptions opts = lm.getShadowOptions(li);
    opts.shadowBulbRadius = radius;
    lm.setShadowOptions(li, opts);
  }
  params_.bulb_radius = radius;
}

void Light::SetShadowsEnabled(bool enabled) {
  if (params_.cast_shadow != enabled && ibl_ == nullptr) {
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    lm.setShadowCaster(li, enabled);
  }
  params_.cast_shadow = enabled;
}

void Light::SetShadowMapSize(int map_size) {
  if (params_.shadow_map_size != map_size && ibl_ == nullptr) {
    filament::LightManager& lm = engine_->getLightManager();
    const filament::LightManager::Instance li = lm.getInstance(entity_);
    filament::LightManager::ShadowOptions opts = lm.getShadowOptions(li);
    opts.mapSize = map_size;
    lm.setShadowOptions(li, opts);
  }
  params_.shadow_map_size = map_size;
}

void Light::Enable() {
  if (!enabled_) {
    enabled_ = true;
    if (ibl_) {
      ibl_->setIntensity(params_.intensity);
    } else {
      filament::LightManager& lm = engine_->getLightManager();
      const filament::LightManager::Instance li = lm.getInstance(entity_);
      lm.setLightChannel(li, 0, enabled_);
    }
  }
}

void Light::Disable() {
  if (enabled_) {
    enabled_ = false;
    if (ibl_) {
      ibl_->setIntensity(0.f);
    } else {
      filament::LightManager& lm = engine_->getLightManager();
      const filament::LightManager::Instance li = lm.getInstance(entity_);
      lm.setLightChannel(li, 0, enabled_);
    }
  }
}

mjtLightType Light::GetType() const {
  return static_cast<mjtLightType>(params_.type);
}

}  // namespace mujoco
