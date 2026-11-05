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

#ifndef MUJOCO_SRC_EXPERIMENTAL_FILAMENT_COMPAT_SCENE_OBJECTS_H_
#define MUJOCO_SRC_EXPERIMENTAL_FILAMENT_COMPAT_SCENE_OBJECTS_H_

#include <cstdint>
#include <unordered_map>

#include <mujoco/mjmodel.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"

namespace mujoco {

// Creates and owns meshes read from an mjvScene.
class SceneObjects {
 public:
  explicit SceneObjects(mjrfContext* ctx);

  // Creates a skin or flex mesh from the given geom in the mjvScene.
  bool CreateSkinFlexMesh(const mjvScene* scene, const mjModel* model,
                          const mjvGeom& geom);

  // Returns the mesh for the given geom id, as created by CreateSkinFlexMesh.
  const mjrfMesh* GetSkinMesh(int geom_id) const;
  const mjrfMesh* GetFlexMesh(int geom_id, int data_id = -1) const;

  SceneObjects(const SceneObjects&) = delete;
  SceneObjects& operator=(const SceneObjects&) = delete;

 private:
  mjrfContext* ctx_ = nullptr;
  std::unordered_map<int, UniquePtr<mjrfMesh>> skins_;
  std::unordered_map<uint64_t, UniquePtr<mjrfMesh>> flexes_;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_EXPERIMENTAL_FILAMENT_COMPAT_SCENE_OBJECTS_H_
