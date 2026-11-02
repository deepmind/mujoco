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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_LIGHT_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_LIGHT_H_

#include <functional>

#include <filament/Engine.h>
#include <filament/Scene.h>
#include <math/vec3.h>
#include <utils/Entity.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/types.h"

namespace mujoco {

// Wrapper around both a "normal" filament Light Entity and a filament
// IndirectLight.
class Light : public mjrfLight {
 public:
  Light(filament::Engine* engine, const mjrfLightParams& params);
  ~Light() noexcept;

  Light(const Light&) = delete;
  Light& operator=(const Light&) = delete;

  // Adds this light to the filament Scene.
  void AddToScene(filament::Scene* scene,
                  std::function<void(Light*)> on_destroy);

  // Removes this light from the filament Scene.
  void RemoveFromScene(filament::Scene* scene);

  // Updates this light's position and rotation.
  void SetTransform(filament::math::float3 position,
                    filament::math::float3 direction);

  // Sets the color of this light.
  void SetColor(const filament::math::float3& color);

  // Sets the intensity of this light, in candela.
  void SetIntensity(float intensity);

  // Sets the range of this light, in meters.
  void SetRange(float range);

  // Sets the cutoff angle of this light, in degrees.
  void SetCutoffAngle(float cutoff);

  // Sets the softness of this light, in the range [0, 1].
  void SetSoftness(float softness);

  // Sets the radius of this light's bulb, in meters.
  void SetBulbRadius(float radius);

  // Enables/disables shadows for this light.
  void SetShadowsEnabled(bool enabled);

  // Sets the resolution of this light's shadow map, in texels.
  void SetShadowMapSize(int map_size);

  // Returns the type of the light.
  mjtLightType GetType() const;

  // Enables/disables the light in the scene.
  void Enable();
  void Disable();

  static Light* downcast(mjrfLight* light) {
    return static_cast<Light*>(light);
  }
  static const Light* downcast(const mjrfLight* light) {
    return static_cast<const Light*>(light);
  }

 private:
  filament::Engine* engine_ = nullptr;
  filament::IndirectLight* ibl_ = nullptr;
  filament::Scene* assigned_scene_ = nullptr;
  std::function<void(Light*)> remove_from_scene_fn_;
  utils::Entity entity_;
  mjrfLightParams params_;
  bool enabled_ = true;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_LIGHT_H_
