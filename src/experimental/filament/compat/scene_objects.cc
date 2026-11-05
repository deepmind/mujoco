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

#include "experimental/filament/compat/scene_objects.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include <math/vec2.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"

namespace mujoco {

using filament::math::float2;
using filament::math::float3;
using filament::math::float4;

static uint64_t FlexMeshKey(int geom_id, int data_id) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(geom_id)) << 32) |
         static_cast<uint32_t>(data_id);
}

static int GetFlexFaceCount(const mjvScene* scene, const mjvGeom& geom) {
  return geom.dataid >= 0 ? static_cast<int>(geom.size[2])
                          : scene->flexfaceused[geom.objid];
}

static int GetFlexFaceAdr(const mjvScene* scene, const mjvGeom& geom) {
  return scene->flexfaceadr[geom.objid] +
         (geom.dataid >= 0 ? static_cast<int>(geom.size[1]) : 0);
}

static std::span<const float> GetPositions(const mjModel* model,
                                           const mjvScene* scene,
                                           const mjvGeom& geom) {
  if (geom.type == mjGEOM_FLEX) {
    const int num = 9 * GetFlexFaceCount(scene, geom);
    const int addr = GetFlexFaceAdr(scene, geom);
    const float* ptr = scene->flexface + (9 * addr);
    return {ptr, static_cast<size_t>(num)};
  } else {
    const int num = 3 * scene->skinvertnum[geom.objid];
    const int addr = scene->skinvertadr[geom.objid];
    const float* ptr = scene->skinvert + (3 * addr);
    return {ptr, static_cast<size_t>(num)};
  }
}

static std::span<const float> GetNormals(const mjModel* model,
                                         const mjvScene* scene,
                                         const mjvGeom& geom) {
  if (geom.type == mjGEOM_FLEX) {
    const int num = 9 * GetFlexFaceCount(scene, geom);
    const int addr = GetFlexFaceAdr(scene, geom);
    const float* ptr = scene->flexnormal + (9 * addr);
    return {ptr, static_cast<size_t>(num)};
  } else {
    const int num = 3 * scene->skinvertnum[geom.objid];
    const int addr = scene->skinvertadr[geom.objid];
    const float* ptr = scene->skinnormal + (3 * addr);
    return {ptr, static_cast<size_t>(num)};
  }
}

static std::span<const float> GetUvs(const mjModel* model,
                                     const mjvScene* scene,
                                     const mjvGeom& geom) {
  if (geom.type == mjGEOM_FLEX) {
    if (geom.texcoord && geom.matid >= 0) {
      const int num = 6 * GetFlexFaceCount(scene, geom);
      const int addr = GetFlexFaceAdr(scene, geom);
      const float* ptr = scene->flextexcoord + (6 * addr);
      return {ptr, static_cast<size_t>(num)};
    } else {
      const float* ptr = nullptr;
      return {ptr, 0};
    }
  } else {
    if (model->skin_texcoordadr[geom.objid] >= 0) {
      const int num = 3 * scene->skinvertnum[geom.objid];
      const int addr = model->skin_texcoordadr[geom.objid];
      const float* ptr = model->skin_texcoord + (2 * addr);
      return {ptr, static_cast<size_t>(num)};
    } else {
      const float* ptr = nullptr;
      return {ptr, 0};
    }
  }
}

static std::span<const int> GetIndices(const mjModel* model,
                                       const mjvScene* scene,
                                       const mjvGeom& geom) {
  if (geom.type == mjGEOM_FLEX) {
    const int* ptr = nullptr;
    return {ptr, 0};
  } else {
    const int num = 3 * model->skin_facenum[geom.objid];
    const int* ptr = model->skin_face + 3 * model->skin_faceadr[geom.objid];
    return {ptr, static_cast<size_t>(num)};
  }
}

SceneObjects::SceneObjects(mjrfContext* ctx) : ctx_(ctx) {}

bool SceneObjects::CreateSkinFlexMesh(const mjvScene* scene,
                                      const mjModel* model,
                                      const mjvGeom& geom) {
  auto positions = GetPositions(model, scene, geom);
  if (positions.empty()) {
    return false;
  }

  auto normals = GetNormals(model, scene, geom);
  auto uvs = GetUvs(model, scene, geom);
  auto indices = GetIndices(model, scene, geom);

  int num_indices = indices.size();
  if (num_indices == 0 && geom.type == mjGEOM_FLEX) {
    num_indices = 3 * GetFlexFaceCount(scene, geom);
  }

  mjrfMeshConfig config;
  mjrf_defaultMeshConfig(&config);
  config.num_attributes = uvs.data() ? 3 : 2;
  config.attributes[0].usage = mjVERTEX_ATTRIBUTE_USAGE_POSITION;
  config.attributes[0].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
  config.attributes[1].usage = mjVERTEX_ATTRIBUTE_USAGE_NORMAL;
  config.attributes[1].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
  config.attributes[2].usage = mjVERTEX_ATTRIBUTE_USAGE_UV;
  config.attributes[2].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;
  config.max_vertices = positions.size() / 3;
  config.max_indices = num_indices;
  config.index_type = mjINDEX_TYPE_U32;
  config.primitive_type = mjMESH_PRIMITIVE_TYPE_TRIANGLES;

  mjrfMeshData data;
  mjrf_defaultMeshData(&data);
  data.vertices[0] = positions.data();
  data.vertices[1] = normals.data();
  data.vertices[2] = uvs.data();
  data.num_vertices = positions.size() / 3;
  data.num_indices = num_indices;
  data.indices = indices.data();
  data.compute_bounds = true;
  data.release = nullptr;
  data.user_data = nullptr;

  if (geom.type == mjGEOM_FLEX) {
    const uint64_t key = FlexMeshKey(geom.objid, geom.dataid);
    auto iter = flexes_.find(key);
    if (iter == flexes_.end()) {
      iter = flexes_.insert({key, CreateMesh(ctx_, config)}).first;
    }
    mjrf_uploadMesh(iter->second.get(), &data);
  } else if (geom.type == mjGEOM_SKIN) {
    auto iter = skins_.find(geom.objid);
    if (iter == skins_.end()) {
      iter = skins_.insert({geom.objid, CreateMesh(ctx_, config)}).first;
    }
    mjrf_uploadMesh(iter->second.get(), &data);
  } else {
    mju_error("Unsupported dynamic mesh type: %d", geom.type);
  }
  return true;
}

const mjrfMesh* SceneObjects::GetFlexMesh(int geom_id, int data_id) const {
  if (auto it = flexes_.find(FlexMeshKey(geom_id, data_id));
      it != flexes_.end()) {
    return it->second.get();
  }
  mju_error("Unknown flex mesh %d (dataid %d)", geom_id, data_id);
  return nullptr;
}

const mjrfMesh* SceneObjects::GetSkinMesh(int geom_id) const {
  if (auto it = skins_.find(geom_id); it != skins_.end()) {
    return it->second.get();
  }
  mju_error("Unknown skin mesh %d", geom_id);
  return nullptr;
}
}  // namespace mujoco
