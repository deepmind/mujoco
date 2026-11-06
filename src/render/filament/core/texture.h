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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_TEXTURE_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_TEXTURE_H_

#include <cstdint>
#include <filament/Engine.h>
#include <filament/Texture.h>
#include <math/vec3.h>
#include <mujoco/mjrfilament.h>
#include "render/filament/core/types.h"

// Functions for creating filament textures.
namespace mujoco {

// Wrapper around a filament::Texture.
class Texture : public mjrfTexture {
 public:
  // Flags for internal use.
  struct InternalFlags {
    InternalFlags() : color_attachment(false), depth_attachment(false) {}
    bool color_attachment;
    bool depth_attachment;
  };

  // Creates a texture with the given data.
  Texture(filament::Engine* engine, const mjrfTextureConfig& config,
          InternalFlags flags = InternalFlags());

  ~Texture();

  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;

  // Uploads the given data to the texture.
  void Upload(const mjrfTextureData& data);

  // Returns the width of the texture.
  int GetWidth() const { return config_.width; }

  // Returns the height of the texture.
  int GetHeight() const { return config_.height; }

  // Returns the target of the texture.
  mjrSamplerType GetSamplerType() const;

  // Returns the underlying filament texture.
  filament::Texture* GetFilamentTexture() const { return texture_; }

  // Returns the unique id of the texture.
  uint64_t Id() const { return id_; }

  // Returns any spherical harmonics data associated with the texture.
  using SphericalHarmonics = filament::math::float3[9];
  const SphericalHarmonics* GetSphericalHarmonics() const {
    return has_spherical_harmonics_ ? &spherical_harmonics_ : nullptr;
  }

  static Texture* downcast(mjrfTexture* texture) {
    return static_cast<Texture*>(texture);
  }
  static const Texture* downcast(const mjrfTexture* texture) {
    return static_cast<const Texture*>(texture);
  }

 private:
  uint64_t id_ = 0;
  filament::Engine* engine_ = nullptr;
  filament::Texture* texture_ = nullptr;
  mjrfTextureConfig config_;
  SphericalHarmonics spherical_harmonics_;
  bool has_spherical_harmonics_ = false;
};
}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_TEXTURE_H_
