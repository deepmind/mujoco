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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_MATERIAL_MANAGER_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_MATERIAL_MANAGER_H_

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <filament/Color.h>
#include <filament/Engine.h>
#include <filament/MaterialInstance.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/mesh.h"
#include "render/filament/core/object_manager.h"

namespace mujoco {

// Manages the filament MaterialInstances used by Renderables.
//
// Centralizing the MaterialInstance management allows multiple Renderables to
// share the same MaterialInstance, which can reduce GPU overhead.
// MaterialInstance uniqueness is determined by hashing the parameters (mainly
// the mjrfMaterial) used to create the instance.
//
// Any unused MaterialInstances are destroyed at the end of each frame.
class MaterialManager {
 public:
  using MaterialType = ObjectManager::MaterialType;

  explicit MaterialManager(ObjectManager* object_mgr);
  ~MaterialManager();

  MaterialManager(const MaterialManager&) = delete;
  MaterialManager& operator=(const MaterialManager&) = delete;

  // A key used to uniquely identify a MaterialInstance.
  using MaterialKey = uint64_t;

  // Marks the beginning of a new frame, allowing us to track which
  // MaterialInstances are used during the frame so they can be removed at the
  // end of the frame.
  void PrepareToRender();

  // Removes any unused MaterialInstances.
  void RemoveUnusedMaterials();

  // Returns a MaterialType that best matches the given material data and mesh.
  MaterialType GetMaterialType(const mjrfMaterial& material, const Mesh* mesh);

  // Prepares a MaterialInstance based on the given parameters if one does not
  // already exist. Returns the key associated with the MaterialInstance.
  MaterialKey PrepareMaterialInstance(const mjrfMaterial& material,
                                      mjrDrawMode draw_mode, mjtGeom geom_type,
                                      const Mesh* mesh);

  // Returns the MaterialInstance associated with the given key.
  const filament::MaterialInstance* GetInstance(MaterialKey key);

  // Sets the color space for how BaseColorFactor is interpreted.
  void SetBaseColorType(filament::RgbaType base_color_type);

  ObjectManager* GetObjectManager() const;
  filament::Engine* GetEngine() const;

 private:
  // Updates the material instance using the given parameters and texture
  // data. In some cases where a material needs a texture, but a specific
  // texture is not provided, a default texture from the ObjectManager will be
  // used instead.
  void UpdateMaterialInstance(filament::MaterialInstance* instance,
                              const mjrfMaterial& material);

  ObjectManager* object_mgr_;
  std::unordered_map<MaterialKey, filament::MaterialInstance*> instances_;
  std::unordered_set<MaterialKey> used_keys_;
  filament::RgbaType base_color_type_ = filament::RgbaType::sRGB;
};
}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_MATERIAL_MANAGER_H_
