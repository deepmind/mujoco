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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MODEL_OBJECTS_H_
#define MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MODEL_OBJECTS_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mujoco/mjmodel.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"

namespace mujoco {

// Creates and owns meshes and textures read from an mjModel.
class ModelResources {
 public:
  ModelResources(const mjModel* model, mjrfContext* ctx);

  // Uploads a new mesh from the model with the given id.
  void UploadMesh(const mjModel* model, int id);

  // Uploads a new texture from the model with the given id.
  void UploadTexture(const mjModel* model, int id);

  // Uploads a new height field from the model with the given id.
  void UploadHeightField(const mjModel* model, int id);

  // Returns the mjModel mesh with the given data_id. The data_id is the
  // mesh_id * 2 of the mesh in the mjModel.
  const mjrfMesh* GetMesh(int data_id) const;

  // Returns the mjModel convex hulll mesh with the given data_id. The data_id
  // is the (mesh_id * 2) + 1 of the mesh in the mjModel.
  const mjrfMesh* GetConvexHull(int data_id) const;

  // Returns the mjModel height field mesh with the given id.
  const mjrfMesh* GetHeightField(int hfield_id) const;

  // Returns the mjModel texture with the given id, or nullptr if it has none,
  // e.g. a by-reference texture that could not be loaded.
  const mjrfTexture* GetTexture(int tex_id) const;

  // Returns the skybox texture in the mjModel.
  const mjrfTexture* GetSkyboxTexture() const;

  // Returns the mjrfContext used by the ModelResources.
  mjrfContext* GetContext() const { return ctx_; }

  // Returns the mjModel from which the Model Objects are created.
  const mjModel* GetModel() const { return model_; }

  // Returns the multipliers used for mapping legacy material properties to
  // filament material properties.
  float GetSpecularMultiplier() const { return specular_multiplier_; }
  float GetShininessMultiplier() const { return shininess_multiplier_; }
  float GetEmissiveMultiplier() const { return emissive_multiplier_; }

  ModelResources(const ModelResources&) = delete;
  ModelResources& operator=(const ModelResources&) = delete;

 private:
  // The identity of a by-reference texture: ids with the same key share one GPU
  // texture.
  struct TextureKey {
    TextureKey(const mjModel* model, int id);
    bool operator==(const TextureKey& other) const;

    std::string path;
    std::string content_type;
    int type;
    int nchannel;
    int colorspace;
  };
  struct TextureKeyHash {
    size_t operator()(const TextureKey& key) const;
  };

  // An uploaded texture, null if it could not be loaded, and its key if it is
  // by reference.
  struct TextureEntry {
    std::optional<TextureKey> key;
    std::shared_ptr<mjrfTexture> texture;
  };

  // Removes the id from the ids uploaded with the key.
  void RemoveFromKey(int id, const TextureKey& key);

  // Returns the texture of a by-reference id, loading its file or sharing the
  // texture of another id with the same key.
  std::shared_ptr<mjrfTexture> LoadByReference(const mjModel* model, int id,
                                               const TextureKey& key,
                                               bool reload);

  const mjModel* model_ = nullptr;
  mjrfContext* ctx_ = nullptr;
  std::vector<UniquePtr<mjrfMesh>> meshes_;
  std::vector<UniquePtr<mjrfMesh>> convex_hulls_;
  std::vector<UniquePtr<mjrfMesh>> height_fields_;
  std::vector<TextureEntry> textures_;
  // The ids of the by-reference textures uploaded with each key.
  std::unordered_map<TextureKey, std::unordered_set<int>, TextureKeyHash>
      ids_by_key_;
  float specular_multiplier_ = 0.2f;
  float shininess_multiplier_ = 0.1f;
  float emissive_multiplier_ = 0.3f;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MODEL_OBJECTS_H_
