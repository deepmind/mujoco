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

#include "render/filament/core/builtins.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

#include <filament/Engine.h>
#include <math/vec2.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/mesh.h"
#include "render/filament/support/filament_util.h"

namespace mujoco {

using filament::math::float2;
using filament::math::float3;
using filament::math::float4;

static constexpr size_t kNumVerticesPerTriangle = 3;
static constexpr size_t kNumVerticesPerQuad = 4;
static constexpr size_t kNumIndicesPerTriangle = 3;
static constexpr size_t kNumIndicesPerQuad = 6;

static void AppendQuadIndices(std::vector<uint16_t>& vec, uint16_t a,
                              uint16_t b, uint16_t c, uint16_t d) {
  vec.push_back(a);
  vec.push_back(b);
  vec.push_back(c);
  vec.push_back(a);
  vec.push_back(c);
  vec.push_back(d);
}

static std::size_t NumVerticesPerSide(int num_quads_per_axis) {
  return (num_quads_per_axis + 1) * (num_quads_per_axis + 1);
}

static std::size_t NumIndicesPerSide(int num_quads_per_axis) {
  return kNumIndicesPerQuad * num_quads_per_axis * num_quads_per_axis;
}

class BuiltinBuilder {
 public:
  BuiltinBuilder() {}
  virtual ~BuiltinBuilder() = default;

  template <typename T, typename... Args>
  static std::unique_ptr<Mesh> Create(filament::Engine* engine,
                                      Args&&... args) {
    auto builder = new T(std::forward<Args>(args)...);

    mjrfMeshConfig config;
    mjrf_defaultMeshConfig(&config);
    config.max_vertices = builder->positions_.size();
    config.max_indices = builder->indices_.size();
    config.index_type = mjINDEX_TYPE_U16;
    config.primitive_type = builder->primitive_type_;
    config.num_attributes = builder->texcoords_.empty() ? 2 : 3;
    config.attributes[0].usage = mjVERTEX_ATTRIBUTE_USAGE_POSITION;
    config.attributes[0].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
    config.attributes[1].usage = mjVERTEX_ATTRIBUTE_USAGE_TANGENTS;
    config.attributes[1].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT4;
    if (!builder->texcoords_.empty()) {
      config.attributes[2].usage = mjVERTEX_ATTRIBUTE_USAGE_UV;
      config.attributes[2].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;
    }

    mjrfMeshData data;
    mjrf_defaultMeshData(&data);
    data.num_vertices = builder->positions_.size();
    data.vertices[0] = builder->positions_.data();
    data.vertices[1] = builder->orientations_.data();
    if (!builder->texcoords_.empty()) {
      data.vertices[2] = builder->texcoords_.data();
    }
    data.num_indices = builder->indices_.size();
    data.indices = builder->indices_.data();
    data.bounds_min[0] = builder->bounds_min_.x;
    data.bounds_min[1] = builder->bounds_min_.y;
    data.bounds_min[2] = builder->bounds_min_.z;
    data.bounds_max[0] = builder->bounds_max_.x;
    data.bounds_max[1] = builder->bounds_max_.y;
    data.bounds_max[2] = builder->bounds_max_.z;
    data.release = +[](void* user_data) {
      delete static_cast<BuiltinBuilder*>(user_data);
    };
    data.user_data = builder;

    auto mesh = std::make_unique<Mesh>(engine, config);
    mesh->Upload(data);
    return mesh;
  }

 protected:
  void SetBounds(const float3& min, const float3& max) {
    bounds_min_ = min;
    bounds_max_ = max;
  }

  int primitive_type_ = mjMESH_PRIMITIVE_TYPE_TRIANGLES;
  std::vector<float3> positions_;
  std::vector<float4> orientations_;
  std::vector<float2> texcoords_;
  std::vector<uint16_t> indices_;
  float3 bounds_min_ = {0, 0, 0};
  float3 bounds_max_ = {0, 0, 0};
};

class LineBuilder : public BuiltinBuilder {
 public:
  LineBuilder() {
    primitive_type_ = mjMESH_PRIMITIVE_TYPE_LINES;

    positions_.reserve(2);
    positions_.emplace_back(0, 0, 0);
    positions_.emplace_back(0, 0, 1);

    orientations_.resize(positions_.size(), {0, 0, 0, 1});

    indices_.reserve(2);
    indices_.push_back(0);
    indices_.push_back(1);

    SetBounds({0, 0, 0}, {0, 0, 1});
  }
};

class PlaneBuilder : public BuiltinBuilder {
 public:
  explicit PlaneBuilder(int num_quads_per_axis) {
    const int num_vertices = NumVerticesPerSide(num_quads_per_axis);
    positions_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);

    const float delta = 2.0f / num_quads_per_axis;
    for (int x = 0; x <= num_quads_per_axis; ++x) {
      for (int y = 0; y <= num_quads_per_axis; ++y) {
        const float dx = delta * static_cast<float>(x);
        const float dy = delta * static_cast<float>(y);
        positions_.emplace_back(dx - 1.0f, dy - 1.0f, 0);
        texcoords_.emplace_back(0.5f * dx, 1.0f - (0.5f * dy));
      }
    }

    orientations_.resize(positions_.size(), CalculateOrientation({0, 0, 1}));

    const int num_indices = NumIndicesPerSide(num_quads_per_axis);
    indices_.reserve(num_indices);
    for (int x = 0; x < num_quads_per_axis; ++x) {
      for (int y = 0; y < num_quads_per_axis; ++y) {
        const int base_idx = x * (num_quads_per_axis + 1) + y;
        const int i0 = base_idx + 0;
        const int i1 = base_idx + 1;
        const int i2 = base_idx + num_quads_per_axis + 2;
        const int i3 = base_idx + num_quads_per_axis + 1;
        AppendQuadIndices(indices_, i0, i3, i2, i1);
      }
    }

    SetBounds({-1, -1, -0.001}, {1, 1, 0.001});
  }
};

// Two-band open cylinder (3 rings at z = {0, 0.5, 1}, radius 1) encoding
// uv0 = (alpha_weight, band) with alpha_weight = {0, 1, 0}. The vertex shader
// in haze.mat reshapes these rings into an inverted conical frustum spanning
// from the ground disk rim (radius 1 - HazeRatio, z = 0, alpha = 0) through the
// horizon seam (alpha = 1) up to camera eye height (radius 1, z = 1,
// alpha = 0), matching classic OpenGL's mjrHAZE horizon band. Bounds are left
// unset so frustum culling is disabled while haze.mat displaces vertices.
class HazeBuilder : public BuiltinBuilder {
 public:
  explicit HazeBuilder(int num_slices) {
    num_slices = std::max(3, num_slices);
    static constexpr float kBands[3] = {0.0f, 0.5f, 1.0f};
    static constexpr float kWeights[3] = {0.0f, 1.0f, 0.0f};

    const int num_vertices = 3 * num_slices;
    positions_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);

    const float delta_angle =
        2.0f * std::numbers::pi_v<float> / static_cast<float>(num_slices);
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < num_slices; ++j) {
        const float angle = static_cast<float>(j) * delta_angle;
        positions_.emplace_back(std::cos(angle), std::sin(angle), kBands[k]);
        texcoords_.emplace_back(kWeights[k], kBands[k]);
      }
    }

    orientations_.resize(positions_.size(), {0, 0, 0, 1});

    const int num_indices = 2 * num_slices * kNumIndicesPerQuad;
    indices_.reserve(num_indices);
    for (int k = 0; k < 2; ++k) {
      const int lower_base = k * num_slices;
      const int upper_base = (k + 1) * num_slices;
      for (int j = 0; j < num_slices; ++j) {
        const int next = (j + 1) % num_slices;
        AppendQuadIndices(indices_, lower_base + j, lower_base + next,
                          upper_base + next, upper_base + j);
      }
    }
  }
};

class TriangleBuilder : public BuiltinBuilder {
 public:
  TriangleBuilder() {
    positions_.reserve(3);
    positions_.emplace_back(0, 0, 0);
    positions_.emplace_back(1, 0, 0);
    positions_.emplace_back(0, 1, 0);

    texcoords_.reserve(3);
    texcoords_.emplace_back(0, 1);
    texcoords_.emplace_back(1, 1);
    texcoords_.emplace_back(0, 0);

    orientations_.resize(positions_.size(), CalculateOrientation({0, 0, 1}));

    indices_.reserve(3);
    indices_.emplace_back(0);
    indices_.emplace_back(1);
    indices_.emplace_back(2);

    SetBounds({0, 0, -0.001}, {1, 1, 0.001});
  }
};

class LineBoxBuilder : public BuiltinBuilder {
 public:
  explicit LineBoxBuilder() {
    primitive_type_ = mjMESH_PRIMITIVE_TYPE_LINES;

    positions_.reserve(8);
    positions_.emplace_back(-1.0f, -1.0f, -1.0f);
    positions_.emplace_back(1.0f, -1.0f, -1.0f);
    positions_.emplace_back(-1.0f, 1.0f, -1.0f);
    positions_.emplace_back(1.0f, 1.0f, -1.0f);
    positions_.emplace_back(-1.0f, -1.0f, 1.0f);
    positions_.emplace_back(1.0f, -1.0f, 1.0f);
    positions_.emplace_back(-1.0f, 1.0f, 1.0f);
    positions_.emplace_back(1.0f, 1.0f, 1.0f);

    orientations_.resize(positions_.size(), {0, 0, 0, 1});

    indices_.reserve(24);
    indices_.push_back(0);
    indices_.push_back(1);
    indices_.push_back(1);
    indices_.push_back(3);
    indices_.push_back(3);
    indices_.push_back(2);
    indices_.push_back(2);
    indices_.push_back(0);
    // Top square (where z == 1).
    indices_.push_back(4);
    indices_.push_back(5);
    indices_.push_back(5);
    indices_.push_back(7);
    indices_.push_back(7);
    indices_.push_back(6);
    indices_.push_back(6);
    indices_.push_back(4);
    // Connect edges from bottom to top.
    indices_.push_back(2);
    indices_.push_back(6);
    indices_.push_back(3);
    indices_.push_back(7);
    indices_.push_back(0);
    indices_.push_back(4);
    indices_.push_back(1);
    indices_.push_back(5);

    SetBounds({-1, -1, -1}, {1, 1, 1});
  }
};

class BoxBuilder : public BuiltinBuilder {
 public:
  static constexpr int kNumSides = 6;

  explicit BoxBuilder(int num_quads_per_axis)
      : num_quads_per_axis_(num_quads_per_axis) {
    quad_size_ = 2.0f / static_cast<float>(num_quads_per_axis_);

    const int vertices_per_side = NumVerticesPerSide(num_quads_per_axis_);
    const int indices_per_side = NumIndicesPerSide(num_quads_per_axis_);
    const int num_vertices = vertices_per_side * kNumSides;
    const int num_indices = indices_per_side * kNumSides;

    positions_.reserve(num_vertices);
    orientations_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);
    indices_.reserve(num_indices);

    GenerateVerticesForSide({0, 1, 0},
                            [](float2 pt) { return float3{pt.x, 1.0f, pt.y}; });
    GenerateVerticesForSide(
        {0, -1, 0}, [](float2 pt) { return float3{pt.x, -1.0f, pt.y}; });
    GenerateVerticesForSide({1, 0, 0},
                            [](float2 pt) { return float3{1.0f, pt.x, pt.y}; });
    GenerateVerticesForSide(
        {-1, 0, 0}, [](float2 pt) { return float3{-1.0f, pt.x, pt.y}; });
    GenerateVerticesForSide({0, 0, 1},
                            [](float2 pt) { return float3{pt.x, pt.y, 1.0f}; });
    GenerateVerticesForSide(
        {0, 0, -1}, [](float2 pt) { return float3{pt.x, pt.y, -1.0f}; });

    for (int i = 0; i < kNumSides; ++i) {
      for (int x = 0; x < num_quads_per_axis_; ++x) {
        for (int y = 0; y < num_quads_per_axis_; ++y) {
          const int base_idx =
              (i * vertices_per_side) + (x * (num_quads_per_axis_ + 1)) + y;
          const int i0 = base_idx + 0;
          const int i1 = base_idx + 1;
          const int i2 = base_idx + num_quads_per_axis_ + 2;
          const int i3 = base_idx + num_quads_per_axis_ + 1;
          if (i == 2 || i == 1 || i == 4) {
            AppendQuadIndices(indices_, i0, i3, i2, i1);
          } else {
            AppendQuadIndices(indices_, i0, i1, i2, i3);
          }
        }
      }
    }

    SetBounds({-1, -1, -1}, {1, 1, 1});
  }

 private:
  template <typename F>
  void GenerateVerticesForSide(float3 normal, const F& pt_gen) {
    float4 orientation = CalculateOrientation(normal);
    for (int x = 0; x <= num_quads_per_axis_; ++x) {
      for (int y = 0; y <= num_quads_per_axis_; ++y) {
        const float dx = -1.0f + (quad_size_ * static_cast<float>(x));
        const float dy = -1.0f + (quad_size_ * static_cast<float>(y));
        const float3 position = pt_gen({dx, dy});
        positions_.push_back(position);
        orientations_.push_back(orientation);
        const float u = static_cast<float>(x) / num_quads_per_axis_;
        const float v = 1.0f - (static_cast<float>(y) / num_quads_per_axis_);
        texcoords_.emplace_back(u, v);
      }
    }
  }

  int num_quads_per_axis_;
  float quad_size_;
};

class TubeBuilder : public BuiltinBuilder {
 public:
  TubeBuilder(int num_stacks, int num_slices) {
    const int verts_per_ring = num_slices + 1;
    const int num_vertices = verts_per_ring * (num_stacks + 1);
    positions_.reserve(num_vertices);
    orientations_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);

    const float delta_angle = 2.f * std::numbers::pi / (float)num_slices;
    const float delta_stack = 2.f / static_cast<float>(num_stacks);
    for (int i = 0; i <= num_slices; ++i) {
      const int geo_i = i % num_slices;
      const float angle = static_cast<float>(geo_i) * delta_angle;
      const float2 pt{std::cos(angle), std::sin(angle)};
      const float4 orientation = CalculateOrientation({pt.x, pt.y, 0});
      const float u = static_cast<float>(i) / num_slices;
      for (int j = 0; j <= num_stacks; ++j) {
        const float z = -1.0f + (static_cast<float>(j) * delta_stack);
        positions_.emplace_back(pt.x, pt.y, z);
        orientations_.push_back(orientation);
        const float v = 1.0f - ((static_cast<float>(j) * delta_stack) / 2.0f);
        texcoords_.emplace_back(u, v);
      }
    }

    const int num_indices = kNumIndicesPerQuad * num_slices * num_stacks;
    indices_.reserve(num_indices);

    const int num_vertices_in_spine = num_stacks + 1;
    for (int i = 0; i < num_slices; ++i) {
      for (int j = 0; j < num_stacks; ++j) {
        const int base_idx = (i * num_vertices_in_spine) + j;
        const int next_base_idx = ((i + 1) * num_vertices_in_spine) + j;
        const int i0 = base_idx + 0;
        const int i1 = base_idx + 1;
        const int i2 = next_base_idx + 1;
        const int i3 = next_base_idx + 0;
        AppendQuadIndices(indices_, i0, i3, i2, i1);
      }
    }

    SetBounds({-1, -1, -1}, {1, 1, 1});
  }
};

class ConeBuilder : public BuiltinBuilder {
 public:
  ConeBuilder(int num_stacks, int num_slices) {
    const int num_vertices =
        (num_slices * kNumVerticesPerTriangle) +
        ((num_stacks - 1) * num_slices * kNumVerticesPerQuad);
    positions_.reserve(num_vertices);
    orientations_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);

    // Pole: use triangles
    const float delta_angle =
        2.0 * std::numbers::pi / static_cast<float>(num_slices);
    const float delta_radius = 1.0f / static_cast<float>(num_stacks);

    for (int j = 0; j < num_slices; ++j) {
      const float angle1 = (j + 0) * delta_angle;
      const float angle2 = (j + 1) * delta_angle;

      const float u1 = static_cast<float>(j) / num_slices;
      const float u2 = static_cast<float>(j + 1) / num_slices;
      const float v1 = delta_radius;
      const float u_mid = (u1 + u2) / 2.f;

      AppendVert(angle1, delta_radius, u1, v1);
      AppendVert(angle2, delta_radius, u2, v1);

      positions_.emplace_back(0, 0, 1);
      orientations_.emplace_back(CalculateOrientation({0, 0, 1}));
      texcoords_.emplace_back(u_mid, 0.f);
    }

    // The rest: use quads
    for (int i = 1; i < num_stacks; ++i) {
      const float radius1 = delta_radius * (i + 0);
      const float radius2 = delta_radius * (i + 1);

      for (int j = 0; j < num_slices; ++j) {
        const float angle1 = (j + 0) * delta_angle;
        const float angle2 = (j + 1) * delta_angle;
        const float u1 = static_cast<float>(j) / num_slices;
        const float u2 = static_cast<float>(j + 1) / num_slices;
        const float v1 = radius1;
        const float v2 = radius2;
        AppendVert(angle1, radius2, u1, v2);
        AppendVert(angle2, radius2, u2, v2);
        AppendVert(angle2, radius1, u2, v1);
        AppendVert(angle1, radius1, u1, v1);
      }
    }

    const int num_indices =
        (num_slices * kNumIndicesPerTriangle) +
        ((num_stacks - 1) * num_slices * kNumIndicesPerQuad);
    indices_.reserve(num_indices);
    for (int j = 0; j < num_slices * 3; ++j) {
      indices_.push_back(j);
    }

    int quad_idx = num_slices * 3;
    for (int i = 1; i < num_stacks; ++i) {
      for (int j = 0; j < num_slices; ++j) {
        const int i0 = quad_idx + 0;
        const int i1 = quad_idx + 1;
        const int i2 = quad_idx + 2;
        const int i3 = quad_idx + 3;
        quad_idx += 4;
        AppendQuadIndices(indices_, i0, i1, i2, i3);
      }
    }

    SetBounds({-1, -1, 0}, {1, 1, 1});
  }

 private:
  void AppendVert(float theta, float radius, float u, float v) {
    static constexpr float kNormalScale = 0.70710678118f;
    const float cz = std::cos(theta);
    const float sz = std::sin(theta);
    const float3 pt{cz * radius, sz * radius, 1.f - radius};
    const float3 n{cz * kNormalScale, sz * kNormalScale, kNormalScale};
    positions_.push_back(pt);
    orientations_.push_back(CalculateOrientation(n));
    texcoords_.emplace_back(u, v);
  }
};

class DiskBuilder : public BuiltinBuilder {
 public:
  explicit DiskBuilder(int num_slices, int num_rings = 1) {
    num_slices = std::max(3, num_slices);
    num_rings = std::clamp(num_rings, 1, std::max(1, 65534 / num_slices));

    const int num_vertices = 1 + num_rings * num_slices;
    positions_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);
    const float delta_angle =
        2.0f * std::numbers::pi_v<float> / static_cast<float>(num_slices);

    positions_.push_back({0, 0, 0});
    texcoords_.emplace_back(0.5f, 0.5f);
    for (int r = 1; r <= num_rings; ++r) {
      const float radius =
          static_cast<float>(r) / static_cast<float>(num_rings);
      for (int i = 0; i < num_slices; ++i) {
        const float angle = static_cast<float>(i) * delta_angle;
        const float x = radius * std::cos(angle);
        const float y = radius * std::sin(angle);
        positions_.push_back({x, y, 0});
        texcoords_.emplace_back(0.5f + 0.5f * x, 0.5f + 0.5f * y);
      }
    }

    orientations_.resize(positions_.size(), CalculateOrientation({0, 0, 1}));

    const int num_indices =
        num_slices * kNumIndicesPerTriangle +
        (num_rings - 1) * num_slices * kNumIndicesPerQuad;
    indices_.reserve(num_indices);
    for (int i = 0; i < num_slices; ++i) {
      const int next = (i + 1) % num_slices;
      indices_.push_back(0);
      indices_.push_back(1 + i);
      indices_.push_back(1 + next);
    }
    for (int r = 1; r < num_rings; ++r) {
      const int inner_base = 1 + (r - 1) * num_slices;
      const int outer_base = 1 + r * num_slices;
      for (int i = 0; i < num_slices; ++i) {
        const int next = (i + 1) % num_slices;
        AppendQuadIndices(indices_, inner_base + i, outer_base + i,
                          outer_base + next, inner_base + next);
      }
    }

    SetBounds({-1, -1, -0.001}, {1, 1, 0.001});
  }
};

class SphereBuilder : public BuiltinBuilder {
 public:
  SphereBuilder(int num_stacks, int num_slices) {
    // To avoid a UV seam artifact, each latitude ring has num_slices+1
    // vertices: the last vertex is a geometric duplicate of the first but
    // with u=1.0 instead of u=0.0. This prevents the GPU from interpolating
    // backwards from u≈0.97 to u=0.0 across the last quad.
    //
    // Each polar triangle also gets its own pole vertex with u set to the
    // midpoint of the two ring vertices, avoiding the degenerate atan2 at
    // the pole.

    const int verts_per_ring = num_slices + 1;  // extra vertex for u=1 seam
    const int ring_verts = num_stacks * verts_per_ring;
    const int pole_verts = 2 * num_slices;  // one pole vert per polar triangle
    const int num_vertices = ring_verts + pole_verts;
    positions_.reserve(num_vertices);
    orientations_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);

    const float lat_angle_delta =
        std::numbers::pi / static_cast<float>(num_stacks + 1);
    const float lon_angle_delta =
        2.0 * std::numbers::pi / static_cast<float>(num_slices);

    // Latitude ring vertices (with seam column).
    for (int lat = 0; lat < num_stacks; ++lat) {
      const float lat_angle = static_cast<float>(lat + 1) * lat_angle_delta;
      const float cos_lat_angle = std::cos(lat_angle);
      const float sin_lat_angle = std::sin(lat_angle);
      const float z = cos_lat_angle;
      const float v = lat_angle / std::numbers::pi_v<float>;

      for (int lon = 0; lon <= num_slices; ++lon) {
        const float u = static_cast<float>(lon) / num_slices;
        // Wrap the geometry back to lon=0 for the seam column.
        const int geo_lon = lon % num_slices;
        const float lon_angle = static_cast<float>(geo_lon) * lon_angle_delta;

        const float x = sin_lat_angle * std::cos(lon_angle);
        const float y = sin_lat_angle * std::sin(lon_angle);
        AppendVert(x, y, z, u, v);
      }
    }

    // Per-face pole vertices. Each polar triangle gets a unique pole vertex
    // with u set to the midpoint of the two adjacent ring vertices.
    const int north_pole_start = ring_verts;
    for (int lon = 0; lon < num_slices; ++lon) {
      const float u = (static_cast<float>(lon) + 0.5f) / num_slices;
      AppendVert(0, 0, 1, u, 0.0f);
    }
    const int south_pole_start = north_pole_start + num_slices;
    for (int lon = 0; lon < num_slices; ++lon) {
      const float u = (static_cast<float>(lon) + 0.5f) / num_slices;
      AppendVert(0, 0, -1, u, 1.0f);
    }

    // Indices.
    const size_t num_tris_polar_cap = num_slices;
    const size_t num_quads_body = num_slices * (num_stacks - 1);
    const int num_indices = (2 * num_tris_polar_cap * kNumIndicesPerTriangle) +
                            (num_quads_body * kNumIndicesPerQuad);
    indices_.reserve(num_indices);

    // North polar cap — each triangle uses its own pole vertex.
    const uint16_t first_ring_start = 0;
    for (int lon = 0; lon < num_slices; ++lon) {
      indices_.push_back(north_pole_start + lon);
      indices_.push_back(first_ring_start + lon);
      indices_.push_back(first_ring_start + lon + 1);
    }

    // Latitudinal quad strips — no index wrapping needed thanks to seam column.
    for (int lat = 0; lat < num_stacks - 1; lat++) {
      const uint16_t north_start = lat * verts_per_ring;
      const uint16_t south_start = (lat + 1) * verts_per_ring;
      for (int lon = 0; lon < num_slices; ++lon) {
        const int i0 = north_start + lon;
        const int i1 = south_start + lon;
        const int i2 = south_start + lon + 1;
        const int i3 = north_start + lon + 1;
        AppendQuadIndices(indices_, i0, i1, i2, i3);
      }
    }

    // South polar cap.
    const uint16_t last_ring_start = (num_stacks - 1) * verts_per_ring;
    for (int lon = 0; lon < num_slices; ++lon) {
      indices_.push_back(south_pole_start + lon);
      indices_.push_back(last_ring_start + lon + 1);
      indices_.push_back(last_ring_start + lon);
    }

    SetBounds({-1, -1, -1}, {1, 1, 1});
  }

 private:
  void AppendVert(float x, float y, float z, float u, float v) {
    const float3 pt{x, y, z};
    positions_.push_back(pt);
    orientations_.push_back(CalculateOrientation(pt));
    texcoords_.emplace_back(u, v);
  }
};

class DomeBuilder : public BuiltinBuilder {
 public:
  DomeBuilder(int num_stacks, int num_slices, bool flip_u = false,
              bool flip_v = false) {
    // Same seam-fix strategy as SphereBuilder: extra vertex per ring at u=1.0
    // and per-face pole vertices.

    const int verts_per_ring = num_slices + 1;
    const int ring_verts = num_stacks * verts_per_ring;
    const int pole_verts = num_slices;  // one pole vert per polar triangle
    const int num_vertices = ring_verts + pole_verts;
    positions_.reserve(num_vertices);
    orientations_.reserve(num_vertices);
    texcoords_.reserve(num_vertices);

    const float lat_angle_delta =
        0.5 * std::numbers::pi / static_cast<float>(num_stacks);
    const float lon_angle_delta =
        2.0 * std::numbers::pi / static_cast<float>(num_slices);

    // Latitude ring vertices (with seam column).
    for (int lat = 0; lat < num_stacks; ++lat) {
      const float lat_angle = static_cast<float>(lat + 1) * lat_angle_delta;
      const float cos_lat_angle = std::cos(lat_angle);
      const float sin_lat_angle = std::sin(lat_angle);
      const float z = cos_lat_angle;
      const float v_val = 1.0f - (2.0f * lat_angle / std::numbers::pi_v<float>);
      const float v = flip_v ? 1.0f - v_val : v_val;

      for (int lon = 0; lon <= num_slices; ++lon) {
        const float u_val = static_cast<float>(lon) / num_slices;
        const float u = flip_u ? 1.0f - u_val : u_val;
        const int geo_lon = lon % num_slices;
        const float lon_angle = static_cast<float>(geo_lon) * lon_angle_delta;

        const float x = sin_lat_angle * std::cos(lon_angle);
        const float y = sin_lat_angle * std::sin(lon_angle);
        AppendVert(x, y, z, u, v);
      }
    }

    // Per-face pole vertices.
    const int pole_start = ring_verts;
    for (int lon = 0; lon < num_slices; ++lon) {
      const float u_val = (static_cast<float>(lon) + 0.5f) / num_slices;
      const float u = flip_u ? 1.0f - u_val : u_val;
      AppendVert(0, 0, 1, u, flip_v ? 0.0f : 1.0f);
    }

    // Indices.
    const size_t num_tris_polar_cap = num_slices;
    const size_t num_quads_body = num_slices * (num_stacks - 1);
    const int num_indices = (num_tris_polar_cap * kNumIndicesPerTriangle) +
                            (num_quads_body * kNumIndicesPerQuad);
    indices_.reserve(num_indices);

    // Polar cap — each triangle uses its own pole vertex.
    const uint16_t first_ring_start = 0;
    for (int lon = 0; lon < num_slices; ++lon) {
      indices_.push_back(pole_start + lon);
      indices_.push_back(first_ring_start + lon);
      indices_.push_back(first_ring_start + lon + 1);
    }

    // Latitudinal quad strips.
    for (int lat = 0; lat < num_stacks - 1; lat++) {
      const uint16_t north_start = lat * verts_per_ring;
      const uint16_t south_start = (lat + 1) * verts_per_ring;
      for (int lon = 0; lon < num_slices; ++lon) {
        const int i0 = north_start + lon;
        const int i1 = south_start + lon;
        const int i2 = south_start + lon + 1;
        const int i3 = north_start + lon + 1;
        AppendQuadIndices(indices_, i0, i1, i2, i3);
      }
    }

    SetBounds({-1, -1, 0}, {1, 1, 1});
  }

 private:
  void AppendVert(float x, float y, float z, float u, float v) {
    const float3 pt{x, y, z};
    positions_.push_back(pt);
    orientations_.push_back(CalculateOrientation(pt));
    texcoords_.emplace_back(u, v);
  }
};

Builtins::Builtins(filament::Engine* engine, int nstack, int nslice,
                   int nquad) {
  line_ = BuiltinBuilder::Create<LineBuilder>(engine);
  plane_ = BuiltinBuilder::Create<PlaneBuilder>(engine, nquad);
  haze_ = BuiltinBuilder::Create<HazeBuilder>(engine, nslice);
  triangle_ = BuiltinBuilder::Create<TriangleBuilder>(engine);
  box_ = BuiltinBuilder::Create<BoxBuilder>(engine, nquad);
  line_box_ = BuiltinBuilder::Create<LineBoxBuilder>(engine);
  sphere_ = BuiltinBuilder::Create<SphereBuilder>(engine, nstack, nslice);
  tube_ = BuiltinBuilder::Create<TubeBuilder>(engine, nstack, nslice);
  disk_ = BuiltinBuilder::Create<DiskBuilder>(engine, nslice, nquad);
  dome_top_ =
      BuiltinBuilder::Create<DomeBuilder>(engine, nstack, nslice, false, true);
  dome_bottom_ =
      BuiltinBuilder::Create<DomeBuilder>(engine, nstack, nslice, true, false);
  cone_ = BuiltinBuilder::Create<ConeBuilder>(engine, nstack, nslice);
}

const Mesh* Builtins::Line() { return line_.get(); }
const Mesh* Builtins::LineBox() { return line_box_.get(); }
const Mesh* Builtins::Plane() { return plane_.get(); }
const Mesh* Builtins::Haze() { return haze_.get(); }
const Mesh* Builtins::Triangle() { return triangle_.get(); }
const Mesh* Builtins::Box() { return box_.get(); }
const Mesh* Builtins::Sphere() { return sphere_.get(); }
const Mesh* Builtins::Cone() { return cone_.get(); }
const Mesh* Builtins::Disk() { return disk_.get(); }
const Mesh* Builtins::DomeTop() { return dome_top_.get(); }
const Mesh* Builtins::DomeBottom() { return dome_bottom_.get(); }
const Mesh* Builtins::Tube() { return tube_.get(); }

}  // namespace mujoco
