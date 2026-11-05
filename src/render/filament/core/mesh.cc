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

#include "render/filament/core/mesh.h"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

#include <backend/BufferDescriptor.h>
#include <filament/Box.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/VertexBuffer.h>
#include <math/TVecHelpers.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/filament_util.h"

namespace mujoco {

using filament::math::float3;
using filament::math::float4;

static filament::VertexAttribute GetUsage(const mjrVertexAttribute& attrib) {
  switch (attrib.usage) {
    case mjVERTEX_ATTRIBUTE_USAGE_POSITION:
      return filament::VertexAttribute::POSITION;
    case mjVERTEX_ATTRIBUTE_USAGE_NORMAL:
      return filament::VertexAttribute::TANGENTS;
    case mjVERTEX_ATTRIBUTE_USAGE_TANGENTS:
      return filament::VertexAttribute::TANGENTS;
    case mjVERTEX_ATTRIBUTE_USAGE_UV:
      return filament::VertexAttribute::UV0;
    case mjVERTEX_ATTRIBUTE_USAGE_COLOR:
      return filament::VertexAttribute::COLOR;
    default:
      mju_error("Unsupported vertex attribute usage: %d", attrib.usage);
      return filament::VertexAttribute::POSITION;
  }
}

static filament::VertexBuffer::AttributeType GetType(
    const mjrVertexAttribute& attrib) {
  switch (attrib.type) {
    case mjVERTEX_ATTRIBUTE_TYPE_FLOAT2:
      return filament::VertexBuffer::AttributeType::FLOAT2;
    case mjVERTEX_ATTRIBUTE_TYPE_FLOAT3:
      return filament::VertexBuffer::AttributeType::FLOAT3;
    case mjVERTEX_ATTRIBUTE_TYPE_FLOAT4:
      return filament::VertexBuffer::AttributeType::FLOAT4;
    case mjVERTEX_ATTRIBUTE_TYPE_UBYTE4:
      return filament::VertexBuffer::AttributeType::UBYTE4;
    default:
      mju_error("Unsupported vertex attribute type: %d", attrib.type);
      return filament::VertexBuffer::AttributeType::FLOAT3;
  }
}

int VertexAttributeTypeSize(const mjrVertexAttribute& attrib) {
  switch (attrib.type) {
    case mjVERTEX_ATTRIBUTE_TYPE_FLOAT2:
      return sizeof(float) * 2;
    case mjVERTEX_ATTRIBUTE_TYPE_FLOAT3:
      return sizeof(float) * 3;
    case mjVERTEX_ATTRIBUTE_TYPE_FLOAT4:
      return sizeof(float) * 4;
    case mjVERTEX_ATTRIBUTE_TYPE_UBYTE4:
      return sizeof(uint8_t) * 4;
    default:
      mju_error("Unsupported vertex attribute type: %d", attrib.type);
      return 0;
  }
}

// Fills an index buffer with a basic incrementing sequence.
template <typename T>
int FillSequence(std::byte* buffer, std::size_t num_bytes) {
  const T num = num_bytes / sizeof(T);
  T* ptr = reinterpret_cast<T*>(buffer);
  for (T i = 0; i < num; ++i) {
    ptr[i] = i;
  }
  return num;
}

Mesh::Mesh(filament::Engine* engine, const mjrfMeshConfig& config)
    : engine_(engine),
      config_(config),
      shared_state_(std::make_shared<SharedState>()) {
  // Perform some validation on the config.
  const mjrVertexAttribute* positions = nullptr;
  const mjrVertexAttribute* normals = nullptr;
  const mjrVertexAttribute* tangents = nullptr;
  for (int i = 0; i < config_.num_attributes; ++i) {
    if (config_.attributes[i].usage == mjVERTEX_ATTRIBUTE_USAGE_POSITION) {
      positions = &config_.attributes[i];
    } else if (config_.attributes[i].usage == mjVERTEX_ATTRIBUTE_USAGE_NORMAL) {
      normals = &config_.attributes[i];
    } else if (config_.attributes[i].usage == mjVERTEX_ATTRIBUTE_USAGE_TANGENTS) {
      tangents = &config_.attributes[i];
    }
  }
  if (config_.max_vertices == 0) {
    mju_error("Mesh has no vertices.");
  }
  if (!positions) {
    mju_error("Mesh has no positions.");
  }
  if (config_.attributes[0].usage != mjVERTEX_ATTRIBUTE_USAGE_POSITION) {
    mju_error("Positions must be the first attribute.");
  }
  if (normals && tangents) {
    mju_error("Mesh has both normals and tangents.");
  }
  if (normals && config_.interleaved) {
    // We need to build orientations from normals and so we require each
    // attribute to be in a separate buffer.
    mju_error("Cannot support normals with interleaved vertex attributes.");
  }

  InitVertexBuffer();
  InitIndexBuffer();
}

Mesh::~Mesh() {
  ReleaseResources();
  if (index_buffer_) {
    engine_->destroy(index_buffer_);
  }
  if (wireframe_index_buffer_) {
    engine_->destroy(wireframe_index_buffer_);
  }
  if (vertex_buffer_) {
    engine_->destroy(vertex_buffer_);
  }
}

void Mesh::Upload(const mjrfMeshData& data) {
  shared_state_ = std::make_shared<SharedState>();

  // If the user has provided a release callback, then we need to ensure we
  // call is when filament is done with the mesh data.
  if (data.release) {
    shared_state_->callbacks.push_back([=]() { data.release(data.user_data); });
  }

  UpdateVertexBuffer(data);
  UpdateIndexBuffer(data);
  UpdateBounds(data);
}

void Mesh::InitVertexBuffer() {
  filament::VertexBuffer::Builder vb_builder;
  vb_builder.vertexCount(config_.max_vertices);

  if (config_.interleaved) {
    // For an interleaved vertex buffer, we will create a single buffer which
    // contains the data in the order specified by the attributes array,
    // starting from the first attribute's payload.
    vb_builder.bufferCount(1);
    int total_vertex_size = 0;
    for (int i = 0; i < config_.num_attributes; ++i) {
      total_vertex_size += VertexAttributeTypeSize(config_.attributes[i]);
    }

    // We assume the buffer is tightly packed with no padding between
    // attributes. As such, the stride is equal to the total vertex size and
    // each offset is the sum of the sizes of the preceding attributes.
    int offset = 0;
    for (int i = 0; i < config_.num_attributes; ++i) {
      const mjrVertexAttribute& attrib = config_.attributes[i];
      const filament::VertexAttribute usage = GetUsage(attrib);
      filament::VertexBuffer::AttributeType type = GetType(attrib);
      vb_builder.attribute(usage, 0, type, offset, total_vertex_size);
      if (usage == filament::VertexAttribute::COLOR) {
        vb_builder.normalized(usage);
      }
      offset += VertexAttributeTypeSize(attrib);
    }
    vertex_buffer_ = vb_builder.build(*engine_);
  } else {
    // For a non-interleaved vertex buffer, we assign a separate buffer to each
    // attribute.
    vb_builder.bufferCount(config_.num_attributes);
    for (int i = 0; i < config_.num_attributes; ++i) {
      const mjrVertexAttribute& attrib = config_.attributes[i];
      const filament::VertexAttribute usage = GetUsage(attrib);
      filament::VertexBuffer::AttributeType type = GetType(attrib);
      if (attrib.usage == mjVERTEX_ATTRIBUTE_USAGE_NORMAL) {
        // We will replace normals with orientations.
        type = filament::VertexBuffer::AttributeType::FLOAT4;
      }
      vb_builder.attribute(usage, i, type);
      if (usage == filament::VertexAttribute::COLOR) {
        vb_builder.normalized(usage);
      }
    }
    vertex_buffer_ = vb_builder.build(*engine_);
  }
}

void Mesh::InitIndexBuffer() {
  if (config_.max_indices == 0) {
    return;
  }

  filament::IndexBuffer::Builder ib_builder;
  ib_builder.indexCount(config_.max_indices);
  ib_builder.bufferType(config_.index_type == mjINDEX_TYPE_U16
                            ? filament::IndexBuffer::IndexType::USHORT
                            : filament::IndexBuffer::IndexType::UINT);
  index_buffer_ = ib_builder.build(*engine_);
}

void Mesh::UpdateVertexBuffer(const mjrfMeshData& data) {
  if (config_.max_vertices != data.num_vertices) {
    mju_error("Vertex count does not match config.");
  }

  // The filament BufferDescriptor callback for releasing the memory.
  // We pass a heap-allocated shared_ptr to the shared state as the user data.
  auto callback = +[](void* buffer, size_t size, void* user) {
    auto* state_ptr = static_cast<std::shared_ptr<SharedState>*>(user);
    auto state = *state_ptr;
    delete state_ptr;

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->called) {
      for (const auto& cb : state->callbacks) {
        cb();
      }
      state->callbacks.clear();
      state->called = true;
    }
  };

  if (config_.interleaved) {
    int total_vertex_size = 0;
    for (int i = 0; i < config_.num_attributes; ++i) {
      total_vertex_size += VertexAttributeTypeSize(config_.attributes[i]);
    }

    auto* user_data = new std::shared_ptr<SharedState>(shared_state_);
    const void* bytes = data.vertices[0];
    const size_t nbytes = data.num_vertices * total_vertex_size;
    vertex_buffer_->setBufferAt(*engine_, 0,
                                {bytes, nbytes, callback, user_data});
  } else {
    // Assign the individual data buffers.
    for (int i = 0; i < config_.num_attributes; ++i) {
      const mjrVertexAttribute& attrib = config_.attributes[i];
      const void* bytes = data.vertices[i];
      size_t nbytes = data.num_vertices * VertexAttributeTypeSize(attrib);
      if (attrib.usage == mjVERTEX_ATTRIBUTE_USAGE_NORMAL) {
        // Replace normals with orientations.
        nbytes = data.num_vertices * sizeof(float4);
        bytes = BuildOrientationsFromNormals(data.num_vertices, bytes);
      }
      auto* user_data = new std::shared_ptr<SharedState>(shared_state_);
      vertex_buffer_->setBufferAt(*engine_, i,
                                  {bytes, nbytes, callback, user_data});
    }
  }
}

void Mesh::UpdateIndexBuffer(const mjrfMeshData& data) {
  if (data.num_indices == 0) {
    return;
  }
  if (data.num_indices != config_.max_indices) {
    mju_error("Index count does not match config.");
  }

  const int element_size =
      config_.index_type == mjINDEX_TYPE_U16 ? sizeof(uint16_t) : sizeof(uint32_t);
  const int num_bytes = config_.max_indices * element_size;

  // If indices == 0 and num_indices > 0, then the user is specifying that the
  // vertices are provided "in order", i.e. the indices are 0, 1, 2, 3, ...
  // In this case, we need to create the sequence of indices explicitly.
  const void* indices = data.indices;
  if (indices == nullptr) {
    std::byte* sequence = new std::byte[num_bytes];
    shared_state_->callbacks.push_back([=]() { delete[] sequence; });

    if (config_.index_type == mjINDEX_TYPE_U16) {
      FillSequence<uint16_t>(sequence, num_bytes);
    } else {
      FillSequence<uint32_t>(sequence, num_bytes);
    }
    indices = sequence;
  }
  // We don't worry about setting a release callback here because the release
  // callback for the vertex buffer will call release_callbacks_.
  filament::backend::BufferDescriptor desc(indices, num_bytes);
  index_buffer_->setBuffer(*engine_, std::move(desc));

  UpdateWireframeIndexBuffer(indices);
}

template <typename T>
static void FillWireframeIndices(const void* src, int num_indices, T* out) {
  const T* in = static_cast<const T*>(src);
  for (int i = 0; i < num_indices - 2; i += 3) {
    *out++ = in[i + 0];
    *out++ = in[i + 1];
    *out++ = in[i + 1];
    *out++ = in[i + 2];
    *out++ = in[i + 2];
    *out++ = in[i + 0];
  }
}

void Mesh::UpdateWireframeIndexBuffer(const void* src_indices) {
  // Line meshes are their own wireframe. Interior edges shared between two
  // triangles are deliberately not deduplicated so that triangle index ranges
  // map to line index ranges by doubling; see GetWireframeIndexBuffer().
  if (config_.primitive_type != mjMESH_PRIMITIVE_TYPE_TRIANGLES) {
    return;
  }
  const int num_line_indices = 2 * config_.max_indices;
  if (wireframe_index_buffer_ == nullptr) {
    filament::IndexBuffer::Builder ib_builder;
    ib_builder.indexCount(num_line_indices);
    ib_builder.bufferType(config_.index_type == mjINDEX_TYPE_U16
                              ? filament::IndexBuffer::IndexType::USHORT
                              : filament::IndexBuffer::IndexType::UINT);
    wireframe_index_buffer_ = ib_builder.build(*engine_);
  }

  const int element_size = config_.index_type == mjINDEX_TYPE_U16
                               ? sizeof(uint16_t)
                               : sizeof(uint32_t);
  const int num_bytes = num_line_indices * element_size;
  std::byte* lines = new std::byte[num_bytes];
  shared_state_->callbacks.push_back([=]() { delete[] lines; });
  if (config_.index_type == mjINDEX_TYPE_U16) {
    FillWireframeIndices(src_indices, config_.max_indices,
                         reinterpret_cast<uint16_t*>(lines));
  } else {
    FillWireframeIndices(src_indices, config_.max_indices,
                         reinterpret_cast<uint32_t*>(lines));
  }
  filament::backend::BufferDescriptor desc(lines, num_bytes);
  wireframe_index_buffer_->setBuffer(*engine_, std::move(desc));
}

void Mesh::UpdateBounds(const mjrfMeshData& data) {
  float3 bounds_min = ReadFloat3(data.bounds_min);
  float3 bounds_max = ReadFloat3(data.bounds_max);
  if (bounds_min != bounds_max) {
    bounds_.emplace().set(bounds_min, bounds_max);
  } else if (data.compute_bounds) {
    bounds_min = float3(FLT_MAX, FLT_MAX, FLT_MAX);
    bounds_max = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    if (config_.attributes[0].usage != mjVERTEX_ATTRIBUTE_USAGE_POSITION) {
      mju_error("mjrfMeshData has no positions.");
    }
    const float* positions =
        reinterpret_cast<const float*>(data.vertices[0]);

    for (int i = 0; i < data.num_vertices; ++i) {
      const float3 position = ReadFloat3(positions, i);
      bounds_min = min(bounds_min, position);
      bounds_max = max(bounds_max, position);
    }
    bounds_.emplace().set(bounds_min, bounds_max);
  }
}

void Mesh::ReleaseResources() {
  std::lock_guard<std::mutex> lock(shared_state_->mutex);
  if (!shared_state_->called) {
    for (const auto& callback : shared_state_->callbacks) {
      callback();
    }
    shared_state_->callbacks.clear();
    shared_state_->called = true;
  }
}

float4* Mesh::BuildOrientationsFromNormals(int num_vertices,
                                           const void* normals) {
  float4* orientations = new float4[num_vertices];
  shared_state_->callbacks.push_back([=]() { delete[] orientations; });

  const float* normals_ptr = reinterpret_cast<const float*>(normals);
  for (int i = 0; i < num_vertices; ++i) {
    orientations[i] = CalculateOrientation(ReadFloat3(normals_ptr, i));
  }
  return orientations;
}

filament::IndexBuffer* Mesh::GetFilamentIndexBuffer() const {
  return index_buffer_;
}

filament::IndexBuffer* Mesh::GetWireframeIndexBuffer() const {
  return wireframe_index_buffer_;
}

filament::VertexBuffer* Mesh::GetFilamentVertexBuffer() const {
  return vertex_buffer_;
}

filament::RenderableManager::PrimitiveType Mesh::GetPrimitiveType() const {
  return config_.primitive_type == mjMESH_PRIMITIVE_TYPE_TRIANGLES
             ? filament::RenderableManager::PrimitiveType::TRIANGLES
             : filament::RenderableManager::PrimitiveType::LINES;
}

bool Mesh::HasVertexAttribute(mjrVertexAttributeUsage attrib) const {
  for (int i = 0; i < config_.num_attributes; ++i) {
    if (config_.attributes[i].usage == attrib) {
      return true;
    }
  }
  return false;
}

bool Mesh::HasBounds() const { return bounds_.has_value(); }

filament::Box Mesh::GetBounds() const { return bounds_.value(); }
}  // namespace mujoco
