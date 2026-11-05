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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_RENDERABLE_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_RENDERABLE_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <span>
#include <vector>

#include <filament/Engine.h>
#include <filament/Scene.h>
#include <math/mat3.h>
#include <math/mat4.h>
#include <math/vec3.h>
#include <utils/Entity.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/material_manager.h"
#include "render/filament/core/mesh.h"
#include "render/filament/core/reflection_manager.h"

namespace mujoco {

enum LayerMask : std::uint8_t {
  // Filament's Skybox occupies layer 0 by default; renderables use the rest.
  kLayerMask_Skybox = 0x01 << 0,
  kLayerMask_Object = 0x01 << 1,
  kLayerMask_Decor = 0x01 << 2,
  kLayerMask_Outline = 0x01 << 3,
  kLayerMask_Haze = 0x01 << 4,
  kLayerMask_All = 0xff,
  kLayerMask_None = 0x00,
};

// Fraction of the camera's far plane at which infinite planes meet the skybox.
inline constexpr float kHorizonFarRatio = 0.70f;

// Returns true if the request's draw mode and settings allow a skybox.
inline bool ShowsSkybox(const mjrfRenderRequest& request) {
  return request.skybox &&
         (request.draw_mode == mjDRAW_MODE_DEFAULT ||
          request.draw_mode == mjDRAW_MODE_DEFAULT_NO_TEXTURES ||
          request.draw_mode == mjDRAW_MODE_ISLANDS);
}

// A Renderable is effectively two things: a mesh and a material.
//
// The mesh describes the surface geometry of the object and the material
// describes how that surface interacts with light (i.e. the color of each point
// on the surface).
class Renderable : public mjrfRenderable {
 public:
  Renderable(filament::Engine* engine, const mjrfRenderableParams& params,
             MaterialManager* material_mgr);
  ~Renderable() noexcept;

  Renderable(const Renderable&) = delete;
  Renderable& operator=(const Renderable&) = delete;

  // Sets the mesh of this renderable. The elem_offset and elem_count parameters
  // can be used to specify a submesh within the mesh. If elem_count is 0,
  // assumes the entire mesh should be appended.
  void SetMesh(const Mesh* mesh, int elem_offset = 0, int elem_count = 0);

  // Sets the mesh of this renderable to a built-in mesh based on the geom type.
  void SetGeomMesh(mjtGeom type, int nstack, int nslice, int nquad);

  // Sets the position and rotation of this renderable.
  void SetTransform(const filament::math::float3& position,
                    const filament::math::mat3f& rotation);

  // Sets the size of this renderable. Note: this is effectively the same as a
  // scale for most renderables. However, for e.g. capsules, the spherical ends
  // are not scaled and remain fixed in size.
  void SetSize(const filament::math::float3& size);

  // Overrides the transform of a single part; reset by the next SetTransform or
  // SetSize.
  void SetPartTransform(int index, const filament::math::mat4f& transform);

  // Returns the current transform of this renderable.
  const filament::math::mat4f& GetTransform() const;

  // Returns true if this renderable is an infinite plane (a square and a
  // horizon disk, see SetPartTransform).
  bool IsInfinitePlane() const { return infinite_plane_; }

  // Sets the layer mask for this renderable. Layer masks can be used to
  // show/hide groups of renderables in scenes. Returns the previous layer mask.
  std::uint8_t SetLayerMask(std::uint8_t mask);

  // Sets the blend order for this renderable. This determines the order in
  // which transparent renderables are blended together. Returns the previous
  // blend order.
  std::uint16_t SetBlendOrder(std::uint16_t blend_order);

  // Sets the coarse draw priority [0..7] for this renderable (defaults to 4).
  // Returns the previous priority.
  std::uint8_t SetPriority(std::uint8_t priority);

  // Enables or disables whether this renderable casts shadows.
  void SetCastShadows(bool cast_shadow);

  // Enables or disables whether this renderable receives shadows.
  void SetReceiveShadows(bool receive_shadow);

  // Adds this renderable to the filament Scene.
  void AddToScene(filament::Scene* scene,
                  std::function<void(Renderable*)> on_destroy);

  // Removes this renderable from the filament Scene.
  void RemoveFromScene(filament::Scene* scene);

  // Updates the parameters and textures of the material for this renderable.
  void UpdateMaterial(const mjrfMaterial& material);

  // Returns this renderable's current material.
  const mjrfMaterial& GetMaterial() const;

  // Prepares the material instances for the given render requests.
  void Prepare(std::span<const mjrfRenderRequest*> requests,
               ReflectionManager* reflection_mgr);

  // Binds the material instance for the given render request.
  void BindMaterialInstance(const mjrfRenderRequest& request);

  void SetMaterialInstance(MaterialManager::MaterialKey key);

  static Renderable* downcast(mjrfRenderable* renderable) {
    return static_cast<Renderable*>(renderable);
  }
  static const Renderable* downcast(const mjrfRenderable* renderable) {
    return static_cast<const Renderable*>(renderable);
  }

 private:
  // For each render request in a batch, we may need to render the object
  // differently. However, we cannot change the material instance during the
  // actual frame rendering (i.e. between beginFrame/endFrame). Instead, we can
  // switch out the material instance entirely for each request. We keep track
  // of which instance, as well as other render state, to use with each render
  // request.
  struct DrawState {
    MaterialManager::MaterialKey material_key = 0;
    bool cast_shadow = true;
    bool receive_shadow = true;
    bool wireframe = false;
    std::uint8_t hidden_parts = 0;  // bitmask of parts to hide
  };

  // In most cases, a Renderable will be composed of a single filament Entity.
  // However, for some built-in geom types (e.g. capsules) we compose the
  // renderable out of multiple Entities.
  struct Part {
    utils::Entity entity;
    const Mesh* mesh = nullptr;
    int elem_offset = 0;
    int elem_count = 0;
  };

  // A tuple of translation, rotation, and size.
  struct Trs {
    filament::math::float3 translation{0.0f, 0.0f, 0.0f};
    filament::math::mat3f rotation;
    filament::math::float3 size{1.0f, 1.0f, 1.0f};
    filament::math::mat4f ToTransform() const {
      return filament::math::mat4f(rotation, translation) *
             filament::math::mat4f::scaling(size);
    }
  };

  // When composing a multi-part renderable, each Entity will have its own
  // transform offset based on the transform of the Renderable itself.
  using GetTransformFn = std::function<filament::math::mat4f(int, const Trs&)>;

  void ClearParts();
  void AppendMesh(const Mesh* mesh);
  void InitPartEntity(Part& part);
  void SetHiddenParts(std::uint8_t hidden_parts);
  void UpdateTransform();
  void ApplyCastShadows(bool cast_shadow);
  void ApplyReceiveShadows(bool receive_shadow);
  void SetWireframe(bool wireframe);

  std::uint8_t GetPartLayerMask(int index) const;

  filament::Engine* GetEngine();

  MaterialManager* material_mgr_;
  mjrfRenderableParams params_;
  mjtGeom geom_type_ = mjGEOM_NONE;
  mjrfMaterial material_;
  std::deque<DrawState> draw_queue_;
  DrawState curr_state_;
  filament::Scene* assigned_scene_ = nullptr;
  std::function<void(Renderable*)> remove_from_scene_fn_;
  std::vector<Part> parts_;
  filament::math::mat4f transform_;
  GetTransformFn get_transform_fn_;
  Trs trs_;
  std::uint8_t layer_mask_ = 0x00;
  std::uint8_t priority_ = 4;
  bool infinite_plane_ = false;
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_RENDERABLE_H_
