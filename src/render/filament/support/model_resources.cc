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

#include "render/filament/support/model_resources.h"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <math/TVecHelpers.h>
#include <math/vec2.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/filament_util.h"

namespace mujoco {

using filament::math::float2;
using filament::math::float3;
using filament::math::float4;

enum class MeshType {
  kNormal,
  kConvexHull,
  kHeightField,
};

struct MeshBuilder {
  MeshBuilder(int nvertices) : nvertices(nvertices) {
    positions.reserve(nvertices);
    orientations.reserve(nvertices);
    uvs.reserve(nvertices);
  }

  void Append(const float3& position, const float4& orientation,
              const float2& uv) {
    positions.push_back(position);
    orientations.push_back(orientation);
    uvs.push_back(uv);
    bounds_min = min(bounds_min, position);
    bounds_max = max(bounds_max, position);
  }

  int nvertices = 0;
  float3 bounds_min = {FLT_MAX, FLT_MAX, FLT_MAX};
  float3 bounds_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  std::vector<float3> positions;
  std::vector<float4> orientations;
  std::vector<float2> uvs;
};

static bool UseFaceNormal(const float3& face_normal,
                          const float3& mesh_normal) {
  // clang-format off
  return (face_normal[0] * mesh_normal[0] +
          face_normal[1] * mesh_normal[1] +
          face_normal[2] * mesh_normal[2]) < 0.8f;
  // clang-format on
}

// Same object-space projection that phong_2d.mat computes in its shader.
static float2 ProjectUv(const float3& p) {
  return float2(0.5f * p.x, -0.5f * p.y);
}

static void FillConvexHullBuffer(MeshBuilder& builder, const mjModel* model,
                                 int meshid) {
  const int numvert = model->mesh_graph[model->mesh_graphadr[meshid]];
  const int numface = model->mesh_graph[model->mesh_graphadr[meshid] + 1];
  if (builder.nvertices != numface * 3) {
    mju_error("Invalid vertex count (%d vs %d).", builder.nvertices, numface * 3);
    return;
  }

  const int dataadr = model->mesh_graphadr[meshid] + 2;
  const int vertadr = model->mesh_vertadr[meshid];
  const float* vertices = model->mesh_vert + (3 * vertadr);
  const int texcoordadr = model->mesh_texcoordadr[meshid];
  const float* texcoords = texcoordadr >= 0 ? model->mesh_texcoord + (2 * texcoordadr) : nullptr;

  for (int face = 0; face < numface; ++face) {
    const int j = dataadr + (3 * numvert) + (3 * numface) + (3 * face);
    const float3 p1 = ReadFloat3(vertices, model->mesh_graph[j + 0]);
    const float3 p2 = ReadFloat3(vertices, model->mesh_graph[j + 1]);
    const float3 p3 = ReadFloat3(vertices, model->mesh_graph[j + 2]);
    const float4 orientation = CalculateOrientation(p1, p2, p3);
    const float2 uv1 = texcoords ? ReadFloat2(texcoords, model->mesh_graph[j + 0]) : ProjectUv(p1);
    const float2 uv2 = texcoords ? ReadFloat2(texcoords, model->mesh_graph[j + 1]) : ProjectUv(p2);
    const float2 uv3 = texcoords ? ReadFloat2(texcoords, model->mesh_graph[j + 2]) : ProjectUv(p3);
    builder.Append(p1, orientation, uv1);
    builder.Append(p2, orientation, uv2);
    builder.Append(p3, orientation, uv3);
  }
}

static void FillMeshBuffer(MeshBuilder& builder, const mjModel* model, int meshid) {
  const int faceadr = model->mesh_faceadr[meshid];
  const int facenum = model->mesh_facenum[meshid];
  if (builder.nvertices != facenum * 3) {
    mju_error("Invalid vertex count (%d vs %d).", builder.nvertices, facenum * 3);
    return;
  }

  const int vertadr = model->mesh_vertadr[meshid];
  const float* vertices = model->mesh_vert + (3 * vertadr);
  const int normaladr = model->mesh_normaladr[meshid];
  const float* normals = model->mesh_normal + 3 * normaladr;
  const int texcoordadr = model->mesh_texcoordadr[meshid];
  const float* texcoords = texcoordadr >= 0 ? model->mesh_texcoord + (2 * texcoordadr) : nullptr;

  for (int i = 0; i < facenum; ++i) {
    const int face = 3 * (faceadr + i);

    const float3 p1 = ReadFloat3(vertices, model->mesh_face[face + 0]);
    const float3 p2 = ReadFloat3(vertices, model->mesh_face[face + 1]);
    const float3 p3 = ReadFloat3(vertices, model->mesh_face[face + 2]);
    const float3 face_normal = CalculateNormal(p1, p2, p3);
    const float3 n1 = ReadFloat3(normals, model->mesh_facenormal[face + 0]);
    const float3 n2 = ReadFloat3(normals, model->mesh_facenormal[face + 1]);
    const float3 n3 = ReadFloat3(normals, model->mesh_facenormal[face + 2]);
    const float2 uv1 = texcoords ? ReadFloat2(texcoords, model->mesh_facetexcoord[face + 0]) : ProjectUv(p1);
    const float2 uv2 = texcoords ? ReadFloat2(texcoords, model->mesh_facetexcoord[face + 1]) : ProjectUv(p2);
    const float2 uv3 = texcoords ? ReadFloat2(texcoords, model->mesh_facetexcoord[face + 2]) : ProjectUv(p3);

    if (!texcoords && UseFaceNormal(face_normal, n1)) {
      builder.Append(p1, CalculateOrientation(face_normal), uv1);
    } else {
      builder.Append(p1, CalculateOrientation(n1), uv1);
    }

    if (!texcoords && UseFaceNormal(face_normal, n2)) {
      builder.Append(p2, CalculateOrientation(face_normal), uv2);
    } else {
      builder.Append(p2, CalculateOrientation(n2), uv2);
    }

    if (!texcoords && UseFaceNormal(face_normal, n3)) {
      builder.Append(p3, CalculateOrientation(face_normal), uv3);
    } else {
      builder.Append(p3, CalculateOrientation(n3), uv3);
    }
  }
}

static void FillHeightFieldBuffer(MeshBuilder& builder, const mjModel* model,
                                  int hfieldid) {
  auto append_tri = [&](float3 a, float3 b, float3 c) {
    float4 orientation = CalculateOrientation(a, b, c);
    builder.Append(a, orientation, ProjectUv(a));
    builder.Append(b, orientation, ProjectUv(b));
    builder.Append(c, orientation, ProjectUv(c));
  };
  auto append_quad = [&](float3 a, float3 b, float3 c, float3 d) {
    append_tri(a, b, d);
    append_tri(d, b, c);
  };

  const float* data = model->hfield_data + model->hfield_adr[hfieldid];
  const int nrow = model->hfield_nrow[hfieldid];
  const int ncol = model->hfield_ncol[hfieldid];
  const float height = 0.5f * (nrow - 1);
  const float width = 0.5f * (ncol - 1);
  float sz[4];
  for (int i = 0; i < 4; ++i) {
    sz[i] = static_cast<float>(model->hfield_size[4 * hfieldid + i]);
  }

  auto get_pos = [=](int r, int c) {
    const float x = sz[0] * (c / width - 1.0f);
    const float y = sz[1] * (r / height - 1.0f);
    const float z = sz[2] * data[(r * ncol) + c];
    return float3{x, y, z};
  };

  // For each quad defined by 4 points in the height field, we will create 4
  // triangles by introducing a vertex in the middle of the quad.
  //     a---b
  //     |\ /|
  //     | m |
  //     |/ \|
  //     d---c
  for (int row = 0; row < nrow - 1; ++row) {
    for (int col = 0; col < ncol - 1; ++col) {
      const float3 a = get_pos(row, col);
      const float3 b = get_pos(row, col + 1);
      const float3 c = get_pos(row + 1, col + 1);
      const float3 d = get_pos(row + 1, col);

      const float mid_x = (a.x + b.x) * 0.5f;
      const float mid_y = (a.y + d.y) * 0.5f;

      // To determine the height of the middle vertex, we look at the heights
      // of the opposing corners (i.e. {a, c} and {b, d}). Our goal is to avoid
      // creating any odd bumps or valleys in the height field if possible.
      //
      // If one of the two opposing corners are of the same height, then we
      // set the middle vertex such that we're effectively rendering two
      // triangles, preventing an odd bump. Otherwise, we use the higher
      // midpoint between two opposing corners to prevent valleys.
      //     0---0      0---0      6---4
      //     |\  |      |  /|      |\ /|
      //     | 0 |      | 0 |      | 7 |
      //     |  \|      |/  |      |/ \|
      //     2---0      0---2      0---8
      float mid_z = 0;
      if (a.z == c.z && b.z != d.z) {
        mid_z = a.z;
      } else if (a.z != c.z && b.z == d.z) {
        mid_z = b.z;
      } else {
        const float mid_z_ac = (a.z + c.z) * 0.5f;
        const float mid_z_bd = (b.z + d.z) * 0.5f;
        mid_z = std::max(mid_z_ac, mid_z_bd);
      }

      const float3 mid = {mid_x, mid_y, mid_z};
      append_tri(a, b, mid);
      append_tri(b, c, mid);
      append_tri(c, d, mid);
      append_tri(d, a, mid);
    }
  }
  // Build the left edge.
  for (int row = 0; row < nrow - 1; ++row) {
    const float3 a = get_pos(row, 0);
    const float3 b = get_pos(row + 1, 0);
    const float3 c = {b.x, b.y, -sz[3]};
    const float3 d = {a.x, a.y, -sz[3]};
    append_quad(a, b, c, d);
  }
  // Build the right edge.
  for (int row = 0; row < nrow - 1; ++row) {
    const float3 a = get_pos(row + 1, ncol - 1);
    const float3 b = get_pos(row, ncol - 1);
    const float3 c = {b.x, b.y, -sz[3]};
    const float3 d = {a.x, a.y, -sz[3]};
    append_quad(a, b, c, d);
  }
  // Build the front edge.
  for (int col = 0; col < ncol - 1; ++col) {
    const float3 a = get_pos(0, col);
    const float3 b = {a.x, a.y, -sz[3]};
    const float3 d = get_pos(0, col + 1);
    const float3 c = {d.x, d.y, -sz[3]};
    append_quad(a, b, c, d);
  }
  // Build the back edge.
  for (int col = 0; col < ncol - 1; ++col) {
    const float3 a = get_pos(nrow - 1, col + 1);
    const float3 b = {a.x, a.y, -sz[3]};
    const float3 d = get_pos(nrow - 1, col);
    const float3 c = {d.x, d.y, -sz[3]};
    append_quad(a, b, c, d);
  }
  // Build the base. We use the visualization quality as the size rather than
  // the height field dimensions.
  const float base_width = (0.5f * model->vis.quality.numquads);
  const float base_height = (0.5f * model->vis.quality.numquads);
  for (int row = 0; row < model->vis.quality.numquads; ++row) {
    for (int col = 0; col < model->vis.quality.numquads; ++col) {
      const float x0 = sz[0] * ((col + 0) / base_width - 1.0f);
      const float x1 = sz[0] * ((col + 1) / base_width - 1.0f);
      const float y0 = sz[1] * ((row + 0) / base_height - 1.0f);
      const float y1 = sz[1] * ((row + 1) / base_height - 1.0f);
      append_quad({x0, y0, -sz[3]}, {x0, y1, -sz[3]}, {x1, y1, -sz[3]},
                  {x1, y0, -sz[3]});
    }
  }
}

static int CalculateHeightFieldVertexCount(const mjModel* model, int hfieldid) {
  const int nrow = model->hfield_nrow[hfieldid];
  const int ncol = model->hfield_ncol[hfieldid];

  // For details, see the logic in FillHeightFieldBuffer for how many vertices
  // we need. But, in general...

  // We use 4 triangles (i.e. 12 vertices) per quad.
  const int surface_count = 12 * (nrow - 1) * (ncol - 1);
  // We use 1 quad (i.e. 6 vertices) per edge element. We double this because
  // we have two edges per dimension (e.g. left/right and front/back).
  const int edge_count = (12 * (nrow - 1)) + (12 * (ncol - 1));
  // We use 1 quad (i.e. 6 vertices) per base element. We use the visualization
  // quality as the size rather than the height field dimensions.
  const int base_count =
      6 * model->vis.quality.numquads * model->vis.quality.numquads;

  const int total_count = surface_count + edge_count + base_count;
  return total_count;
}

static bool IsValidIndex(const mjModel* model, int id, MeshType mesh_type) {
  switch (mesh_type) {
    case MeshType::kNormal:
      return id >= 0 && id < model->nmesh;
    case MeshType::kConvexHull:
      return id >= 0 && id < model->nmesh;
    case MeshType::kHeightField:
      return id >= 0 && id < model->nhfield;
  }
}

static int GetNumVertices(const mjModel* model, int id, MeshType mesh_type) {
  switch (mesh_type) {
    case MeshType::kNormal:
      return 3 * model->mesh_facenum[id];
    case MeshType::kConvexHull:
      return 3 * model->mesh_graph[model->mesh_graphadr[id] + 1];
    case MeshType::kHeightField:
      return CalculateHeightFieldVertexCount(model, id);
  }
}

static void UpdateMeshData(mjrfMeshConfig* config, mjrfMeshData* data,
                           const mjModel* model, int id, MeshType mesh_type) {
  if (!IsValidIndex(model, id, mesh_type)) {
    mju_error("Invalid index %d for type %d", id, mesh_type);
    return;
  }

  const int num_vertices = GetNumVertices(model, id, mesh_type);

  MeshBuilder* builder = new MeshBuilder(num_vertices);
  data->user_data = builder;
  data->release = [](void* user_data) {
    delete static_cast<MeshBuilder*>(user_data);
  };

  switch (mesh_type) {
    case MeshType::kNormal:
      FillMeshBuffer(*builder, model, id);
      break;
    case MeshType::kConvexHull:
      FillConvexHullBuffer(*builder, model, id);
      break;
    case MeshType::kHeightField:
      FillHeightFieldBuffer(*builder, model, id);
      break;
  }

  config->max_vertices = num_vertices;
  config->max_indices = num_vertices;
  config->primitive_type = mjMESH_PRIMITIVE_TYPE_TRIANGLES;
  config->index_type = num_vertices >= std::numeric_limits<uint16_t>::max()
                           ? mjINDEX_TYPE_U32
                           : mjINDEX_TYPE_U16;
  config->num_attributes = 3;
  config->attributes[0].usage = mjVERTEX_ATTRIBUTE_USAGE_POSITION;
  config->attributes[0].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
  config->attributes[1].usage = mjVERTEX_ATTRIBUTE_USAGE_TANGENTS;
  config->attributes[1].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT4;
  config->attributes[2].usage = mjVERTEX_ATTRIBUTE_USAGE_UV;
  config->attributes[2].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;

  data->num_vertices = num_vertices;
  data->num_indices = num_vertices;
  data->indices = nullptr;
  data->vertices[0] = builder->positions.data();
  data->vertices[1] = builder->orientations.data();
  data->vertices[2] = builder->uvs.data();
  data->bounds_min[0] = builder->bounds_min.x;
  data->bounds_min[1] = builder->bounds_min.y;
  data->bounds_min[2] = builder->bounds_min.z;
  data->bounds_max[0] = builder->bounds_max.x;
  data->bounds_max[1] = builder->bounds_max.y;
  data->bounds_max[2] = builder->bounds_max.z;
}

static mjrPixelFormat GetPixelFormat(int nchannel) {
  switch (nchannel) {
    case 1:
      return mjPIXEL_FORMAT_R8;
    case 3:
      return mjPIXEL_FORMAT_RGB8;
    case 4:
      return mjPIXEL_FORMAT_RGBA8;
    default:
      return mjPIXEL_FORMAT_UNKNOWN;
  }
}

// The identity of a by-reference texture: two ids with the same key would
// upload the same bytes, so they share one GPU texture.
ModelResources::TextureKey::TextureKey(const mjModel* model, int id)
    : path(model->paths + model->tex_pathadr[id]),
      content_type(model->tex_contenttypeadr[id] >= 0
                       ? model->paths + model->tex_contenttypeadr[id]
                       : ""),
      type(model->tex_type[id]),
      nchannel(model->tex_nchannel[id]),
      colorspace(model->tex_colorspace[id]) {}

bool ModelResources::TextureKey::operator==(const TextureKey& other) const {
  return path == other.path && content_type == other.content_type &&
         type == other.type && nchannel == other.nchannel &&
         colorspace == other.colorspace;
}

size_t ModelResources::TextureKeyHash::operator()(const TextureKey& key) const {
  uint64_t seed = 0;
  Combine(seed, &key.path);
  Combine(seed, &key.content_type);
  Combine(seed, &key.type);
  Combine(seed, &key.nchannel);
  Combine(seed, &key.colorspace);
  return seed;
}

// Creates a texture from data laid out as tex_data holds it: nchannel bytes per
// texel, or a KTX file as a single row of bytes, which Filament decodes. The
// data is released with release(user_data) after the upload, if release is not
// null. Returns nullptr, after a warning, if the data cannot be uploaded.
static std::shared_ptr<mjrfTexture> MakeTexture(
    mjrfContext* ctx, int id, int type, int width, int height, int nchannel,
    int colorspace, const void* bytes, void* user_data,
    void (*release)(void*)) {
  mjrfTextureConfig config;
  mjrf_defaultTextureConfig(&config);
  config.width = width;
  config.height = height;
  config.sampler_type = (mjtTexture)type;
  config.color_space = (mjtColorSpace)colorspace;
  config.format = height == 1 && nchannel == 1 ? mjPIXEL_FORMAT_KTX
                                               : GetPixelFormat(nchannel);
  if (config.format == mjPIXEL_FORMAT_UNKNOWN) {
    // skip, like the classic renderer: mju_error here would skip the
    // destructors of the objects being constructed
    mju_warning("texture %d: can't upload %d-channel data, skipping", id,
                nchannel);
    if (release) release(user_data);
    return nullptr;
  }

  mjrfTextureData payload;
  mjrf_defaultTextureData(&payload);
  payload.bytes = bytes;
  payload.num_bytes = (mjtSize)width * height * nchannel;
  payload.user_data = user_data;
  payload.release = release;

  auto texture = CreateTexture(ctx, config);
  mjrf_uploadTexture(texture.get(), &payload);
  return std::shared_ptr<mjrfTexture>(std::move(texture));
}

// Uploads a by-reference texture: mj_loadTexture returns the data as an
// embedded texture holds it, which is released when the upload completes.
// Returns nullptr, after a warning, if the file cannot be loaded.
static std::shared_ptr<mjrfTexture> UploadTextureByReference(
    mjrfContext* ctx, const mjModel* model, int id) {
  mjtByte* data = nullptr;
  int width, height, colorspace;
  char err[1024] = "";
  if (mj_loadTexture(model, id, &data, &width, &height, &colorspace, err,
                     sizeof(err))) {
    mju_warning("texture %d not loaded, left untextured: %s", id, err);
    return nullptr;
  }
  std::shared_ptr<mjrfTexture> result =
      MakeTexture(ctx, id, model->tex_type[id], width, height,
                  model->tex_nchannel[id], colorspace, data, data, mju_free);
  // Flush so this decoded buffer is uploaded and freed (via mju_free) before
  // the next texture is decoded. Without it, every decoded buffer stays alive
  // until scene setup ends, which can exhaust the 32-bit WASM heap.
  mjrf_flush(ctx);
  return result;
}

ModelResources::ModelResources(const mjModel* model, mjrfContext* ctx)
    : model_(model), ctx_(ctx) {
  textures_.reserve(model_->ntex);
  for (int i = 0; i < model_->ntex; ++i) {
    textures_.emplace_back();
    UploadTexture(model_, i);
  }
  meshes_.reserve(model_->nmesh);
  convex_hulls_.reserve(model_->nmesh);
  for (int i = 0; i < model_->nmesh; ++i) {
    meshes_.emplace_back(nullptr, nullptr);
    convex_hulls_.emplace_back(nullptr, nullptr);
    UploadMesh(model_, i);
  }
  height_fields_.reserve(model_->nhfield);
  for (int i = 0; i < model_->nhfield; ++i) {
    height_fields_.emplace_back(nullptr, nullptr);
    UploadHeightField(model_, i);
  }

  specular_multiplier_ = ReadElement(
      model_, "filament.phong.specular_multiplier", specular_multiplier_);
  shininess_multiplier_ = ReadElement(
      model_, "filament.phong.shininess_multiplier", shininess_multiplier_);
  emissive_multiplier_ = ReadElement(
      model_, "filament.phong.emissive_multiplier", emissive_multiplier_);
}

void ModelResources::UploadMesh(const mjModel* model, int id) {
  if (model != model_) {
    mju_error("Model mismatch.");
  }
  if (id < 0 || id >= model->nmesh) {
    mju_error("Invalid mesh index %d", id);
  }
  meshes_[id].reset();
  convex_hulls_[id].reset();

  mjrfMeshConfig config;
  mjrf_defaultMeshConfig(&config);
  mjrfMeshData data;
  mjrf_defaultMeshData(&data);
  UpdateMeshData(&config, &data, model, id, MeshType::kNormal);

  auto mesh = CreateMesh(ctx_, config);
  mjrf_uploadMesh(mesh.get(), &data);
  meshes_[id] = std::move(mesh);

  if (model->mesh_graphadr[id] >= 0) {
    mjrfMeshConfig convex_hull_config;
    mjrf_defaultMeshConfig(&convex_hull_config);
    mjrfMeshData convex_hull_data;
    mjrf_defaultMeshData(&convex_hull_data);
    UpdateMeshData(&convex_hull_config, &convex_hull_data, model, id,
                   MeshType::kConvexHull);

    auto convex_hull = CreateMesh(ctx_, convex_hull_config);
    mjrf_uploadMesh(convex_hull.get(), &convex_hull_data);
    convex_hulls_[id] = std::move(convex_hull);
  }
}

void ModelResources::UploadTexture(const mjModel* model, int id) {
  if (model != model_) {
    mju_error("Model mismatch.");
  }
  if (id < 0 || id >= model->ntex) {
    mju_error("Invalid texture index: %d", id);
  }

  // a by-reference id leaves the ids of its previous key, unless it is
  // re-uploaded with the same key; an embedded texture has no key
  TextureEntry& entry = textures_[id];
  std::optional<TextureKey> key;
  if (model->tex_adr[id] == -1) {
    key.emplace(model, id);
  }
  const bool reload = entry.key && key && *entry.key == *key;
  if (entry.key && !reload) {
    RemoveFromKey(id, *entry.key);
  }
  entry.key = key;

  if (key) {
    entry.texture = LoadByReference(model, id, *key, reload);
  } else {
    entry.texture =
        MakeTexture(ctx_, id, model->tex_type[id], model->tex_width[id],
                    model->tex_height[id], model->tex_nchannel[id],
                    model->tex_colorspace[id],
                    model->tex_data + model->tex_adr[id],
                    // not released: we assume that the model has the same
                    // lifetime as the engine
                    nullptr, nullptr);
  }
}

void ModelResources::RemoveFromKey(int id, const TextureKey& key) {
  auto it = ids_by_key_.find(key);
  if (it != ids_by_key_.end()) {
    it->second.erase(id);
    if (it->second.empty()) {
      ids_by_key_.erase(it);
    }
  }
}

// The file is loaded once per key and shared by the ids with that key, also
// when it fails (a null texture, which warns once); a re-upload reads the file
// again, for all of them.
std::shared_ptr<mjrfTexture> ModelResources::LoadByReference(
    const mjModel* model, int id, const TextureKey& key, bool reload) {
  std::unordered_set<int>& ids = ids_by_key_.try_emplace(key).first->second;
  std::shared_ptr<mjrfTexture> texture;
  if (ids.empty() || reload) {
    texture = UploadTextureByReference(ctx_, model, id);
    for (int other : ids) {
      textures_[other].texture = texture;
    }
  } else {
    texture = textures_[*ids.begin()].texture;
  }
  ids.insert(id);
  return texture;
}

void ModelResources::UploadHeightField(const mjModel* model, int id) {
  if (model != model_) {
    mju_error("Model mismatch.");
  }
  if (id < 0 || id >= model->nhfield) {
    mju_error("Invalid height field index %d", id);
  }

  height_fields_[id].reset();

  mjrfMeshConfig config;
  mjrf_defaultMeshConfig(&config);
  mjrfMeshData data;
  mjrf_defaultMeshData(&data);
  UpdateMeshData(&config, &data, model, id, MeshType::kHeightField);

  auto mesh = CreateMesh(ctx_, config);
  mjrf_uploadMesh(mesh.get(), &data);
  height_fields_[id] = std::move(mesh);
}

const mjrfMesh* ModelResources::GetMesh(int data_id) const {
  if (data_id < 0) {
    return nullptr;
  }
  // As defined by mjv_updateScene:
  //   original mesh: mesh_id * 2
  //   convex hull: (mesh_id * 2) + 1
  const int mesh_id = data_id / 2;
  if (data_id % 2 == 0) {
    return mesh_id < meshes_.size() ? meshes_[mesh_id].get() : nullptr;
  } else {
    return mesh_id < convex_hulls_.size() ? convex_hulls_[mesh_id].get()
                                          : nullptr;
  }
}

const mjrfMesh* ModelResources::GetHeightField(int hfield_id) const {
  if (hfield_id < 0 || hfield_id >= height_fields_.size()) {
    return nullptr;
  }
  return height_fields_[hfield_id].get();
}

const mjrfTexture* ModelResources::GetTexture(int tex_id) const {
  if (tex_id < 0 || tex_id >= textures_.size()) {
    return nullptr;
  }
  return textures_[tex_id].texture.get();
}

const mjrfTexture* ModelResources::GetSkyboxTexture() const {
  for (int i = 0; i < textures_.size(); ++i) {
    if (model_->tex_type[i] == mjTEXTURE_SKYBOX) {
      return textures_[i].texture.get();
    }
  }
  return nullptr;
}

}  // namespace mujoco
