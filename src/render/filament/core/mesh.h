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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_MESH_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_MESH_H_

#include <optional>

#include <filament/Box.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/RenderableManager.h>
#include <filament/VertexBuffer.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include "render/filament/core/releaser.h"
#include "render/filament/core/types.h"

namespace mujoco {

// Owns a filament Vertex and Index buffer representing a geometry mesh.
class Mesh : public mjrfMesh {
 public:
  // Creates a Mesh from the given MeshData.
  Mesh(filament::Engine* engine, const mjrfMeshConfig& config);
  ~Mesh();

  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;

  // Uploads the given MeshData to the mesh.
  void Upload(const mjrfMeshData& data);

  // Returns the filament IndexBuffer for the mesh.
  filament::IndexBuffer* GetFilamentIndexBuffer() const;

  // Returns the filament IndexBuffer of line indices for wireframe rendering,
  // or nullptr for non-triangle meshes. Each triangle contributes its three
  // edges, so a triangle index range [offset, count) maps to the line index
  // range [2*offset, 2*count).
  filament::IndexBuffer* GetWireframeIndexBuffer() const;

  // Returns the filament VertexBuffer for the mesh.
  filament::VertexBuffer* GetFilamentVertexBuffer() const;

  // Returns the primitive type of the mesh.
  filament::RenderableManager::PrimitiveType GetPrimitiveType() const;

  // Returns true if the mesh has the given attribute.
  bool HasVertexAttribute(mjrVertexAttributeUsage attrib) const;

  // Returns whether the mesh has bounds.
  bool HasBounds() const;

  // Returns the bounds of the mesh.
  filament::Box GetBounds() const;

  static Mesh* downcast(mjrfMesh* mesh) { return static_cast<Mesh*>(mesh); }
  static const Mesh* downcast(const mjrfMesh* mesh) {
    return static_cast<const Mesh*>(mesh);
  }

 private:
  void InitVertexBuffer();
  void InitIndexBuffer();

  void UpdateVertexBuffer(const mjrfMeshData& data, Releaser& releaser);
  void UpdateIndexBuffer(const mjrfMeshData& data, Releaser& releaser);
  void UpdateWireframeIndexBuffer(const void* indices, Releaser& releaser);
  void UpdateBounds(const mjrfMeshData& data);

  filament::math::float4* BuildOrientationsFromNormals(int num_vertices,
                                                       const void* normals,
                                                       Releaser& releaser);

  filament::Engine* engine_ = nullptr;
  mjrfMeshConfig config_;

  filament::IndexBuffer* index_buffer_ = nullptr;
  filament::IndexBuffer* wireframe_index_buffer_ = nullptr;
  filament::VertexBuffer* vertex_buffer_ = nullptr;
  std::optional<filament::Box> bounds_;
};
}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_MESH_H_
