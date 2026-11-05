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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_OBJECT_MANAGER_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_OBJECT_MANAGER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>

#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/builtins.h"

namespace mujoco {

// Creates and owns various filament objects based on the data in a mjrContext.
class ObjectManager {
 public:
  ObjectManager(filament::Engine* engine);
  ~ObjectManager();

  // The different filament::Materials that are loaded and managed by the
  // ObjectManager.
  enum MaterialType {
    kPbr,
    kPbrTransparent,
    kPbrPacked,
    kPbrPackedTransparent,
    kPbrReflect,
    kPbrPackedReflect,
    kPhong2d,
    kPhong2dFade,
    kPhong2dUv,
    kPhong2dUvFade,
    kPhong2dUvReflect,
    kPhongColor,
    kPhongColorFade,
    kPhongColorReflect,
    kPhongCube,
    kPhongCubeFade,
    kPhongCubeReflect,
    kUnlitSegmentation,
    kDecor,
    kDecorFade,
    kUnlitDepth,
    kUnlitUi,
    kOutlineComposite,
    kOutlineFlatten,
    kOutlineJumpFlood,
    kHaze,
    kNumMaterials,
  };

  // Returns the Material of the given type.
  filament::Material* GetMaterial(MaterialType type) const;

  // Returns the fallback Texture with the given role.
  const filament::Texture* GetFallbackTexture(mjtTextureRole role) const;

  // Returns the buffers for creating a full-screen quad.
  filament::VertexBuffer* GetQuadVertexBuffer() const { return quad_vb_; }
  filament::IndexBuffer* GetQuadIndexBuffer() const { return quad_ib_; }

  // Returns the built-in mesh collection with the given dimensions. For
  // performance reasons, you should consider always using the same dimensions
  // in order to reuse the same meshes.
  Builtins* GetBuiltins(int nstack, int nslice, int nquad);

  // Returns the filament Engine that owns the assets.
  filament::Engine* GetEngine() const { return engine_; }

  ObjectManager(const ObjectManager&) = delete;
  ObjectManager& operator=(const ObjectManager&) = delete;

 private:
  void CreateQuadBuffers();

  filament::Engine* engine_ = nullptr;
  std::array<filament::Material*, kNumMaterials> materials_;
  std::array<filament::Texture*, mjNTEXROLE> fallback_textures_;
  std::unordered_map<std::uint64_t, std::unique_ptr<Builtins>> builtins_;
  filament::Texture* fallback_white_ = nullptr;
  filament::Texture* fallback_black_ = nullptr;
  filament::Texture* fallback_normal_ = nullptr;
  filament::Texture* fallback_orm_ = nullptr;
  filament::VertexBuffer* quad_vb_ = nullptr;
  filament::IndexBuffer* quad_ib_ = nullptr;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_OBJECT_MANAGER_H_
