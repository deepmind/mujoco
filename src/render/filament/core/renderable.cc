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

#include "render/filament/core/renderable.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <span>
#include <utility>

#include <filament/Engine.h>
#include <filament/Material.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <math/TMatHelpers.h>
#include <math/TVecHelpers.h>
#include <math/mat3.h>
#include <math/mat4.h>
#include <math/vec2.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <utils/EntityManager.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "engine/engine_vis_visualize.h"
#include "render/filament/core/builtins.h"
#include "render/filament/core/material_manager.h"
#include "render/filament/core/mesh.h"
#include "render/filament/core/reflection_manager.h"
#include "render/filament/support/filament_util.h"

namespace mujoco {

using filament::math::float2;
using filament::math::float3;
using filament::math::float4;
using filament::math::mat3f;
using filament::math::mat4f;

// An arbitrary scale factor for arrows.
static constexpr float kArrowScale = 1.f / 6.f;
static constexpr float kArrowHeadSize = 1.75f;

// Planes are either squares or disks. A plane Renderable will store both meshes
// as separate parts, but only one should be visible at a time.
static constexpr std::uint8_t kHideDiskPlane = 0x02;
static constexpr std::uint8_t kHideSquarePlane = 0x01;

Renderable::Renderable(filament::Engine* engine,
                       const mjrfRenderableParams& params,
                       MaterialManager* material_mgr)
    : material_mgr_(material_mgr), params_(params) {
  mjrf_defaultMaterial(&material_);
  curr_state_.cast_shadow = params_.cast_shadow;
  curr_state_.receive_shadow = params_.receive_shadow;
}

Renderable::~Renderable() noexcept {
  if (remove_from_scene_fn_) {
    auto on_destroy = std::move(remove_from_scene_fn_);
    on_destroy(this);
  }
  ClearParts();
}

void Renderable::ClearParts() {
  filament::Engine* engine = GetEngine();
  utils::EntityManager& em = utils::EntityManager::get();
  for (Part& part : parts_) {
    if (assigned_scene_) {
      assigned_scene_->remove(part.entity);
    }
    engine->destroy(part.entity);
    em.destroy(part.entity);
  }
  parts_.clear();
  get_transform_fn_ = nullptr;
  infinite_plane_ = false;
  curr_state_ = DrawState{};
  curr_state_.cast_shadow = params_.cast_shadow;
  curr_state_.receive_shadow = params_.receive_shadow;
}

void Renderable::SetMesh(const Mesh* mesh, int elem_offset, int elem_count) {
  if (mesh == nullptr) {
    mju_error("Cannot set mesh to nullptr.");
  }

  // We use MESH, even though it could be any mesh-like geom type, e.g.
  // heightfields, flex, skin, sdf, etc.
  geom_type_ = mjGEOM_MESH;

  filament::VertexBuffer* vertex_buffer = mesh->GetFilamentVertexBuffer();
  if (vertex_buffer == nullptr) {
    mju_error("Invalid (null) vertex buffer.");
  }
  filament::IndexBuffer* index_buffer = mesh->GetFilamentIndexBuffer();
  if (index_buffer == nullptr) {
    mju_error("Invalid (null) index buffer.");
  }

  if (elem_count == 0) {
    elem_count = index_buffer->getIndexCount() - elem_offset;
  }

  if (parts_.empty()) {
    Part& part = parts_.emplace_back();
    part.mesh = mesh;
    part.elem_offset = elem_offset;
    part.elem_count = elem_count;
    InitPartEntity(part);
  } else if (parts_.size() == 1) {
    Part& part = parts_[0];
    part.mesh = mesh;
    part.elem_offset = elem_offset;
    part.elem_count = elem_count;

    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    auto ri = rm.getInstance(part.entity);
    rm.setGeometryAt(ri, 0, part.mesh->GetPrimitiveType(), vertex_buffer,
                     index_buffer, part.elem_offset, part.elem_count);
    if (part.mesh->HasBounds()) {
      rm.setAxisAlignedBoundingBox(ri, part.mesh->GetBounds());
      rm.setCulling(ri, true);
    } else {
      rm.setCulling(ri, false);
    }
  } else {
    mju_error("Cannot set mesh for renderable with multiple parts.");
  }
  curr_state_.wireframe = false;
}

void Renderable::InitPartEntity(Part& part) {
  part.entity = utils::EntityManager::get().create();
  if (part.entity.isNull()) {
    mju_error("Failed to create entity.");
  }

  filament::VertexBuffer* vertex_buffer = part.mesh->GetFilamentVertexBuffer();
  filament::IndexBuffer* index_buffer = part.mesh->GetFilamentIndexBuffer();

  filament::RenderableManager::Builder builder(1);
  builder.geometry(0, part.mesh->GetPrimitiveType(), vertex_buffer,
                   index_buffer, part.elem_offset, part.elem_count);
  if (part.mesh->HasBounds()) {
    builder.boundingBox(part.mesh->GetBounds());
  } else {
    builder.culling(false);
  }
  builder.castShadows(curr_state_.cast_shadow);
  builder.receiveShadows(curr_state_.receive_shadow);
  builder.blendOrder(0, params_.blend_order);
  builder.priority(priority_);
  builder.screenSpaceContactShadows(true);
  if (layer_mask_ != 0 || curr_state_.hidden_parts != 0) {
    const int part_index = &part - parts_.data();
    builder.layerMask(0xff, GetPartLayerMask(part_index));
  }

  builder.build(*GetEngine(), part.entity);
  if (assigned_scene_) {
    assigned_scene_->addEntity(part.entity);
  }
}

void Renderable::SetTransform(const float3& position, const mat3f& rotation) {
  trs_.translation = position;
  trs_.rotation = rotation;
  UpdateTransform();
}

void Renderable::SetSize(const float3& size) {
  trs_.size = size;
  UpdateTransform();
}

void Renderable::UpdateTransform() {
  if (parts_.empty()) {
    transform_ = trs_.ToTransform();
    return;
  }

  filament::TransformManager& tm = GetEngine()->getTransformManager();
  if (geom_type_ == mjGEOM_PLANE && (trs_.size[0] <= 0 || trs_.size[1] <= 0)) {
    infinite_plane_ = true;
    static constexpr float kInfiniteScale = 0.5f * mjMAXPLANEGRID;
    const mat4f scaling =
        mat4f::scaling(float3{kInfiniteScale, kInfiniteScale, 1.0f});
    const mat4f transform =
        filament::math::mat4f(trs_.rotation, trs_.translation) * scaling;
    for (Part& part : parts_) {
      tm.setTransform(tm.getInstance(part.entity), transform);
    }
  } else if (get_transform_fn_) {
    infinite_plane_ = false;
    for (int i = 0; i < parts_.size(); ++i) {
      const mat4f transform = get_transform_fn_(i, trs_);
      tm.setTransform(tm.getInstance(parts_[i].entity), transform);
    }
  } else {
    infinite_plane_ = false;
    const mat4f transform = trs_.ToTransform();
    for (Part& part : parts_) {
      tm.setTransform(tm.getInstance(part.entity), transform);
    }
  }
  transform_ = tm.getTransform(tm.getInstance(parts_[0].entity));
}

const mat4f& Renderable::GetTransform() const { return transform_; }

void Renderable::SetPartTransform(int index, const mat4f& transform) {
  filament::TransformManager& tm = GetEngine()->getTransformManager();
  tm.setTransform(tm.getInstance(parts_[index].entity), transform);
}

void Renderable::AppendMesh(const Mesh* mesh) {
  if (parts_.size() >= 8) {
    mju_error("Renderables support at most 8 parts.");
  }
  Part& part = parts_.emplace_back();
  part.mesh = Mesh::downcast(mesh);
  part.elem_offset = 0;
  part.elem_count = part.mesh->GetFilamentIndexBuffer()->getIndexCount();
  InitPartEntity(part);
}

void Renderable::AddToScene(filament::Scene* scene,
                            std::function<void(Renderable*)> on_destroy) {
  if (assigned_scene_) {
    if (assigned_scene_ != scene) {
      mju_error("Cannot add renderable to multiple scenes.");
    }
    // Entities are already added to the scene.
    return;
  }
  for (Part& part : parts_) {
    scene->addEntity(part.entity);
  }
  assigned_scene_ = scene;
  remove_from_scene_fn_ = std::move(on_destroy);
}

void Renderable::RemoveFromScene(filament::Scene* scene) {
  if (assigned_scene_ != scene) {
    mju_error("Attempting to remove renderable from wrong scene.");
  }
  SetMaterialInstance(0);
  for (Part& part : parts_) {
    scene->remove(part.entity);
  }
  assigned_scene_ = nullptr;
  remove_from_scene_fn_ = nullptr;
}

void Renderable::UpdateMaterial(const mjrfMaterial& material) {
  uint8_t layer_mask = kLayerMask_Object;
  if (material.haze_ratio >= 0) {
    layer_mask = kLayerMask_Haze;
  } else if (material.decor) {
    layer_mask = kLayerMask_Decor;
  }
  if (material.selected) {
    layer_mask |= kLayerMask_Outline;
  }
  SetLayerMask(layer_mask);
  material_ = material;
}

const mjrfMaterial& Renderable::GetMaterial() const { return material_; }

void Renderable::Prepare(std::span<const mjrfRenderRequest*> requests,
                         ReflectionManager* reflection_mgr) {
  const bool scene_has_skybox =
      assigned_scene_ && assigned_scene_->getSkybox() != nullptr;
  // Ensure the renderable has no material instances set from the previous
  // frame. This will allow us to recycle material instances if needed.
  SetMaterialInstance(0);
  // We assume BindMaterialInstance will be called with the same requests in
  // the same order. As such, we'll just store the draw state in a deque rather
  // than trying to perform any kind of matching with the requests.
  draw_queue_.clear();

  for (const mjrfRenderRequest* request : requests) {
    mjrfMaterial material = material_;
    // Selected materials are handled in a separate pass, so clear the value
    // here to take advantage of shared materials.
    material.selected = false;

    DrawState draw_state;
    draw_state.cast_shadow = material.decor ? false : params_.cast_shadow;
    draw_state.receive_shadow = material.decor ? false : params_.receive_shadow;
    draw_state.wireframe = (request->draw_mode == mjDRAW_MODE_WIREFRAME);

    if (request->draw_mode == mjDRAW_MODE_SEGMENTATION_BY_COLOR) {
      constexpr double phi1 = 1.61803398874989484820;  // Cached Phi(1).
      constexpr double coef1 = 1.0 / phi1;
      const double index = static_cast<double>(material.segmentation_id);
      const double sample = std::fmod(0.5 + coef1 * index, 1.0);
      material.segmentation_id = 0x01000000 * sample;
    } else if (request->draw_mode != mjDRAW_MODE_SEGMENTATION_BY_ID) {
      // Clear out the segmentation color in order to take advantage of shared
      // materials.
      material.segmentation_id = 0;
    }

    if (request->draw_mode == mjDRAW_MODE_ISLANDS &&
        material.sleep_state != mjS_STATIC) {
      float hue = 1.0f;
      float saturation = 0.0f;
      float value = 0.7f;
      if (material.island_id >= 0) {
        const double h = static_cast<double>(material.island_id) + 1.0;
        hue = mju_Halton(h, 7);
        saturation = 0.5 + 0.5 * mju_Halton(h, 3);
        value = 0.6 + 0.4 * mju_Halton(h, 5);
      }
      if (material.sleep_state == mjS_ASLEEP) {
        value *= 0.6;
        saturation *= 0.7;
      }
      mjv_hsv2rgb(material.color, hue, saturation, value);
      material.color[3] = 1;

      // Remove textures so they don't interfere with the island color.
      material.color_texture = nullptr;
      material.metallic_texture = nullptr;
      material.roughness_texture = nullptr;
      material.occlusion_texture = nullptr;
      material.orm_texture = nullptr;
    }

    if (request->draw_mode == mjDRAW_MODE_DEFAULT_NO_TEXTURES) {
      material.color_texture = nullptr;
      material.metallic_texture = nullptr;
      material.roughness_texture = nullptr;
      material.occlusion_texture = nullptr;
      material.orm_texture = nullptr;
    }

    if (geom_type_ == mjGEOM_PLANE) {
      const float3 camera_pos = ReadFloat3(request->camera.pos);
      const float3 plane_normal = normalize(trs_.rotation[2]);
      const float elevation = dot(camera_pos - trs_.translation, plane_normal);
      if (elevation < 0.0f) {
        material.color[3] *= 0.3f;
        draw_state.receive_shadow = false;
      }

      // By default, assume we want a square plane.
      draw_state.hidden_parts = kHideDiskPlane;

      // When rendered with a skybox, infinite planes are drawn as disks.
      if (infinite_plane_ && scene_has_skybox && ShowsSkybox(*request)) {
        draw_state.hidden_parts = kHideSquarePlane;

        static constexpr float kBasePlaneScale = 0.5f * mjMAXPLANEGRID;
        const float3 plane_center = camera_pos - elevation * plane_normal;
        const float3 shift = plane_center - trs_.translation;
        const float dx = dot(shift, trs_.rotation[0]);
        const float dy = dot(shift, trs_.rotation[1]);
        const float plane_scale =
            kHorizonFarRatio * request->camera.frustum_far;
        const float ratio = plane_scale / kBasePlaneScale;
        const float base_scale_x = material.uv_scale[0];
        const float base_scale_y = material.uv_scale[1];
        // DiskBuilder uses v = 0.5 + 0.5*y whereas PlaneBuilder uses
        // v = 0.5 - 0.5*y; flip V scale/offset so textures match.
        material.uv_scale[0] = base_scale_x * ratio;
        material.uv_scale[1] = -base_scale_y * ratio;
        material.uv_offset[0] +=
            0.5f * base_scale_x * (1.0f - ratio + dx / kBasePlaneScale);
        material.uv_offset[1] +=
            0.5f * base_scale_y * (1.0f + ratio - dy / kBasePlaneScale);
        if (!material.decor) {
          material.uv_offset[0] = std::fmod(material.uv_offset[0], 1.0f);
          material.uv_offset[1] = std::fmod(material.uv_offset[1], 1.0f);
        }
      }
    }

    const Mesh* mesh = !parts_.empty() ? parts_[0].mesh : nullptr;
    const auto draw_mode = static_cast<mjrDrawMode>(request->draw_mode);
    const bool reflective =
        request->draw_mode == mjDRAW_MODE_DEFAULT &&
        request->reflections &&
        (geom_type_ == mjGEOM_PLANE || geom_type_ == mjGEOM_BOX) &&
        material.reflectance > 0.0;
    if (reflective) {
      // The mirror plane normal is the geom's local +Z (transform_[2], whose
      // scale carries the geom size -- normalize it). The reflect shader uses
      // it to apply the reflection only on the front face, so box mirrors don't
      // show the reflection on their back/side faces. The shader gates on
      // getWorldGeometricNormalVector(), which is expressed in filament's Y-up
      // frame, so rotate this normal out of mujoco's Z-up frame to match
      const float3 refl_normal =
          geom_type_ == mjGEOM_PLANE
              ? float3(0.0f, 0.0f, 0.0f)
              : ToFilamentFrame(normalize(transform_[2].xyz));
      reflection_mgr->Register(this, mesh, material, geom_type_, refl_normal,
                               request);

      // Bind the non-reflective material by default so secondary reflection
      // bounces are suppressed during reflection passes.
      material.reflectance = 0.0f;
    }

    draw_state.material_key = material_mgr_->PrepareMaterialInstance(
        material, draw_mode, geom_type_, mesh);
    draw_queue_.push_back(draw_state);
  }
}

void Renderable::BindMaterialInstance(const mjrfRenderRequest& request) {
  if (draw_queue_.empty()) {
    mju_error("No material instances to bind.");
  }

  const DrawState& state = draw_queue_.front();
  SetHiddenParts(state.hidden_parts);
  ApplyCastShadows(state.cast_shadow);
  ApplyReceiveShadows(state.receive_shadow);
  SetWireframe(state.wireframe);
  SetMaterialInstance(state.material_key);
  curr_state_ = state;
  draw_queue_.pop_front();
}

void Renderable::SetMaterialInstance(MaterialManager::MaterialKey key) {
  if (key != curr_state_.material_key) {
    curr_state_.material_key = key;
    const filament::MaterialInstance* instance =
        material_mgr_->GetInstance(key);
    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (Part& part : parts_) {
      filament::RenderableManager::Instance ri = rm.getInstance(part.entity);
      rm.setMaterialInstanceAt(ri, 0, instance);
    }
  }
}

std::uint8_t Renderable::SetLayerMask(std::uint8_t mask) {
  std::uint8_t prev = layer_mask_;
  if (mask != layer_mask_) {
    layer_mask_ = mask;

    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (int i = 0; i < parts_.size(); ++i) {
      rm.setLayerMask(rm.getInstance(parts_[i].entity), 0xff,
                      GetPartLayerMask(i));
    }
  }
  return prev;
}

std::uint8_t Renderable::GetPartLayerMask(int index) const {
  return (curr_state_.hidden_parts >> index) & 1 ? 0 : layer_mask_;
}

void Renderable::SetHiddenParts(std::uint8_t hidden_parts) {
  if (hidden_parts != curr_state_.hidden_parts) {
    curr_state_.hidden_parts = hidden_parts;
    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (int i = 0; i < parts_.size(); ++i) {
      rm.setLayerMask(rm.getInstance(parts_[i].entity), 0xff,
                      GetPartLayerMask(i));
    }
  }
}

std::uint16_t Renderable::SetBlendOrder(std::uint16_t blend_order) {
  std::uint16_t prev = params_.blend_order;
  if (blend_order != params_.blend_order) {
    params_.blend_order = blend_order;

    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (Part& part : parts_) {
      rm.setBlendOrderAt(rm.getInstance(part.entity), 0, params_.blend_order);
    }
  }
  return prev;
}

std::uint8_t Renderable::SetPriority(std::uint8_t priority) {
  std::uint8_t prev = priority_;
  if (priority != priority_) {
    priority_ = priority;

    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (Part& part : parts_) {
      rm.setPriority(rm.getInstance(part.entity), priority_);
    }
  }
  return prev;
}

void Renderable::SetCastShadows(bool cast_shadow) {
  params_.cast_shadow = cast_shadow;
}

void Renderable::SetReceiveShadows(bool receive_shadow) {
  params_.receive_shadow = receive_shadow;
}

void Renderable::ApplyCastShadows(bool cast_shadow) {
  if (curr_state_.cast_shadow != cast_shadow) {
    curr_state_.cast_shadow = cast_shadow;

    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (Part& part : parts_) {
      rm.setCastShadows(rm.getInstance(part.entity), curr_state_.cast_shadow);
    }
  }
}

void Renderable::ApplyReceiveShadows(bool receive_shadow) {
  if (curr_state_.receive_shadow != receive_shadow) {
    curr_state_.receive_shadow = receive_shadow;

    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (Part& part : parts_) {
      rm.setReceiveShadows(rm.getInstance(part.entity),
                           curr_state_.receive_shadow);
    }
  }
}

void Renderable::SetWireframe(bool wireframe) {
  static constexpr auto kWireframeType =
      filament::RenderableManager::PrimitiveType::LINES;

  if (wireframe != curr_state_.wireframe) {
    curr_state_.wireframe = wireframe;

    filament::RenderableManager& rm = GetEngine()->getRenderableManager();
    for (Part& part : parts_) {
      filament::VertexBuffer* vertex_buffer =
          part.mesh->GetFilamentVertexBuffer();
      filament::IndexBuffer* wireframe_index_buffer =
          part.mesh->GetWireframeIndexBuffer();
      if (wireframe && wireframe_index_buffer) {
        // Triangle meshes have a dedicated line index buffer enumerating the
        // three edges of each triangle, addressed by doubling the range.
        rm.setGeometryAt(rm.getInstance(part.entity), 0, kWireframeType,
                         vertex_buffer, wireframe_index_buffer,
                         2 * part.elem_offset, 2 * part.elem_count);
      } else {
        // Line meshes are drawn as-is in both modes.
        rm.setGeometryAt(rm.getInstance(part.entity), 0,
                         wireframe ? kWireframeType
                                   : part.mesh->GetPrimitiveType(),
                         vertex_buffer, part.mesh->GetFilamentIndexBuffer(),
                         part.elem_offset, part.elem_count);
      }
    }
  }
}

void Renderable::SetGeomMesh(mjtGeom type, int nstack, int nslice, int nquad) {
  ClearParts();
  Builtins* builtins =
      material_mgr_->GetObjectManager()->GetBuiltins(nstack, nslice, nquad);
  geom_type_ = type;

  switch (type) {
    case mjGEOM_PLANE:
      AppendMesh(builtins->Plane());
      AppendMesh(builtins->Disk());
      SetHiddenParts(kHideDiskPlane);
      break;
    case mjGEOM_SPHERE:
      AppendMesh(builtins->Sphere());
      break;
    case mjGEOM_ELLIPSOID:
      AppendMesh(builtins->Sphere());
      break;
    case mjGEOM_BOX:
      AppendMesh(builtins->Box());
      break;
    case mjGEOM_CAPSULE:
      // Capsules are a tube with two domes at the ends.
      AppendMesh(builtins->Tube());
      AppendMesh(builtins->DomeTop());
      AppendMesh(builtins->DomeBottom());

      get_transform_fn_ = [](int index, const Trs& trs) {
        // We apply an inverse scale to the domes to counteract the capsule's
        // overall scale so that the domes remain spherical in shape.
        const float xz_size = 0.5f * (trs.size.x + trs.size.y);
        if (index == 0) {
          return trs.ToTransform();
        } else if (index == 1) {
          // Move the first dome to the top of the capsule.
          mat4f top = mat4f(trs.rotation, trs.translation);
          top *= mat4f::translation(float3{0, 0, trs.size.z});
          top *= mat4f::scaling(float3{trs.size.x, trs.size.y, xz_size});
          return top;
        } else if (index == 2) {
          // Move the second dome to the bottom of the capsule and rotate it 180
          // degrees so that it's facing the right way.
          mat4f bottom = mat4f(trs.rotation, trs.translation);
          bottom *= mat4f::translation(float3{0, 0, -trs.size.z});
          bottom *= mat4f::rotation(std::numbers::pi, float3{1, 0, 0});
          bottom *= mat4f::scaling(float3{trs.size.x, trs.size.y, xz_size});
          return bottom;
        } else {
          mju_error("Invalid index for capsule geom: %d (expected [0,2])",
                    index);
          return trs.ToTransform();
        }
      };
      break;
    case mjGEOM_CYLINDER:
      // Cylinders are a tube with two disks at the ends.
      AppendMesh(builtins->Tube());
      AppendMesh(builtins->Disk());
      AppendMesh(builtins->Disk());

      get_transform_fn_ = [](int index, const Trs& trs) {
        if (index == 0) {
          return trs.ToTransform();
        } else if (index == 1) {
          // Move the first disk to the top of the cylinder.
          mat4f top = mat4f(trs.rotation, trs.translation);
          top *= mat4f::translation(float3{0, 0, trs.size.z});
          top *= mat4f::scaling(trs.size);
          return top;
        } else if (index == 2) {
          // Move the second disk to the bottom of the cylinder. Rotate the disk
          // 180 degrees so that the normals point outwards.
          mat4f bottom = mat4f(trs.rotation, trs.translation);
          bottom *= mat4f::translation(float3{0, 0, -trs.size.z});
          bottom *= mat4f::rotation(std::numbers::pi, float3{1, 0, 0});
          bottom *= mat4f::scaling(trs.size);
          return bottom;
        } else {
          mju_error("Invalid index for cylinder geom: %d (expected [0,2])",
                    index);
          return trs.ToTransform();
        }
      };
      break;
    case mjGEOM_ARROW:
      AppendMesh(builtins->Tube());
      AppendMesh(builtins->Cone());
      AppendMesh(builtins->Disk());
      AppendMesh(builtins->Disk());

      get_transform_fn_ = [](int index, const Trs& trs) {
        mat4f base = mat4f(trs.rotation, trs.translation);
        base *= mat4f::scaling(float3{1, 1, kArrowScale});
        base *= mat4f::translation(float3{0, 0, trs.size.z});
        if (index == 0) {
          return base * mat4f::scaling(trs.size);
        } else if (index == 1) {
          mat4f top = base;
          top *= mat4f::translation(float3{0, 0, trs.size.z});
          top *= mat4f::scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
          return top * mat4f::scaling(trs.size);
        } else if (index == 2) {
          mat4f top_disk = base;
          top_disk *= mat4f::translation(float3{0, 0, trs.size.z});
          top_disk *= mat4f::rotation(std::numbers::pi, float3{1, 0, 0});
          top_disk *=
              mat4f::scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
          return top_disk * mat4f::scaling(trs.size);
        } else if (index == 3) {
          mat4f bottom = base;
          bottom *= mat4f::translation(float3{0, 0, -trs.size.z});
          bottom *= mat4f::rotation(std::numbers::pi, float3{1, 0, 0});
          return bottom * mat4f::scaling(trs.size);
        } else {
          mju_error("Invalid index for arrow geom: %d (expected [0,3])", index);
          return trs.ToTransform();
        }
      };
      break;
    case mjGEOM_ARROW1:
      AppendMesh(builtins->Tube());
      AppendMesh(builtins->Cone());
      AppendMesh(builtins->Disk());

      get_transform_fn_ = [](int index, const Trs& trs) {
        mat4f base = mat4f(trs.rotation, trs.translation);
        base *= mat4f::scaling(float3{1, 1, kArrowScale});
        base *= mat4f::translation(float3{0, 0, trs.size.z});
        if (index == 0) {
          return base * mat4f::scaling(trs.size);
        } else if (index == 1) {
          mat4f top = base;
          top *= mat4f::translation(float3{0, 0, trs.size.z});
          return top * mat4f::scaling(trs.size);
        } else if (index == 2) {
          mat4f bottom = base;
          bottom *= mat4f::translation(float3{0, 0, -trs.size.z});
          bottom *= mat4f::rotation(std::numbers::pi, float3{1, 0, 0});
          return bottom * mat4f::scaling(trs.size);
        } else {
          mju_error("Invalid index for arrow1 geom: %d (expected [0,2])",
                    index);
          return trs.ToTransform();
        }
      };
      break;
    case mjGEOM_ARROW2:
      AppendMesh(builtins->Tube());
      AppendMesh(builtins->Cone());
      AppendMesh(builtins->Cone());
      AppendMesh(builtins->Disk());
      AppendMesh(builtins->Disk());

      get_transform_fn_ = [](int index, const Trs& trs) {
        mat4f base = mat4f(trs.rotation, trs.translation);
        base *= mat4f::scaling(float3{1, 1, kArrowScale});
        base *= mat4f::translation(float3{0, 0, trs.size.z});
        if (index == 0) {
          return base * mat4f::scaling(trs.size);
        } else if (index == 1) {
          mat4f top = base;
          top *= mat4f::translation(float3{0, 0, trs.size.z});
          top *= mat4f::scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
          return top * mat4f::scaling(trs.size);
        } else if (index == 2) {
          mat4f bottom = base;
          bottom *= mat4f::translation(float3{0, 0, -trs.size.z});
          bottom *= mat4f::rotation(std::numbers::pi, float3{1, 0, 0});
          bottom *=
              mat4f::scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
          return bottom * mat4f::scaling(trs.size);
        } else if (index == 3) {
          mat4f top_disk = base;
          top_disk *= mat4f::translation(float3{0, 0, trs.size.z});
          top_disk *= mat4f::rotation(std::numbers::pi, float3{1, 0, 0});
          top_disk *=
              mat4f::scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
          return top_disk * mat4f::scaling(trs.size);
        } else if (index == 4) {
          mat4f bottom_disk = base;
          bottom_disk *= mat4f::translation(float3{0, 0, -trs.size.z});
          return bottom_disk * mat4f::scaling(trs.size);
        } else {
          mju_error("Invalid index for arrow2 geom: %d (expected [0,4])",
                    index);
          return trs.ToTransform();
        }
      };
      break;
    case mjGEOM_LINE:
      AppendMesh(builtins->Line());
      break;
    case mjGEOM_LINEBOX:
      AppendMesh(builtins->LineBox());
      break;
    case mjGEOM_TRIANGLE:
      AppendMesh(builtins->Triangle());
      break;
    default:
      mju_error("Unsupported geom type: %d", type);
      break;
  }
  UpdateTransform();
}

filament::Engine* Renderable::GetEngine() { return material_mgr_->GetEngine(); }

}  // namespace mujoco
