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

#include "render/filament/core/material_manager.h"

#include <cstddef>
#include <cstdint>
#include <functional>

#include <filament/Color.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/TextureSampler.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/mesh.h"
#include "render/filament/core/object_manager.h"
#include "render/filament/core/texture.h"
#include "render/filament/support/filament_util.h"

namespace mujoco {

static void Combine(uint64_t& seed, const mjrfTexture* texture) {
  if (texture) {
    const uint64_t id = Texture::downcast(texture)->Id();
    Combine(seed, &id);
  }
}

template <typename T>
static uint64_t hash(const T& obj) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Only trivially copyable types are hashable.");
  int num_bytes = sizeof(T);
  const std::byte* ptr = reinterpret_cast<const std::byte*>(&obj);
  uint64_t seed = 0;
  while (num_bytes >= sizeof(uint64_t)) {
    Combine(seed, reinterpret_cast<const uint64_t*>(ptr));
    ptr += sizeof(uint64_t);
    num_bytes -= sizeof(uint64_t);
  }
  while (num_bytes >= sizeof(uint32_t)) {
    Combine(seed, reinterpret_cast<const uint32_t*>(ptr));
    ptr += sizeof(uint32_t);
    num_bytes -= sizeof(uint32_t);
  }
  while (num_bytes >= sizeof(uint16_t)) {
    Combine(seed, reinterpret_cast<const uint16_t*>(ptr));
    ptr += sizeof(uint16_t);
    num_bytes -= sizeof(uint16_t);
  }
  while (num_bytes >= sizeof(uint8_t)) {
    Combine(seed, reinterpret_cast<const uint8_t*>(ptr));
    ptr += sizeof(uint8_t);
    num_bytes -= sizeof(uint8_t);
  }
  return seed;
}

MaterialManager::MaterialManager(ObjectManager* object_mgr)
    : object_mgr_(object_mgr) {}

MaterialManager::~MaterialManager() {
  for (auto& [key, instance] : instances_) {
    object_mgr_->GetEngine()->destroy(instance);
  }
}

void MaterialManager::PrepareToRender() { used_keys_.clear(); }

void MaterialManager::RemoveUnusedMaterials() {
  if (instances_.size() == used_keys_.size()) {
    return;
  }
  for (auto it = instances_.begin(); it != instances_.end();) {
    if (!used_keys_.contains(it->first)) {
      object_mgr_->GetEngine()->destroy(it->second);
      it = instances_.erase(it);
    } else {
      ++it;
    }
  }
}

static MaterialManager::MaterialKey BuildMaterialKey(
    MaterialManager::MaterialType material_type, mjtGeom geom_type,
    const mjrfMaterial& material) {
  // Normally, hashing the struct by memory would be a problem because of
  // padding and other uninitialized data. However, we do a memset(0) on the
  // entire structure in mjr_defaultMaterial so this should be safe.
  uint64_t key = hash(material);
  Combine(key, &geom_type);
  Combine(key, &material_type);
  Combine(key, material.color_texture);
  Combine(key, material.opacity_texture);
  Combine(key, material.normal_texture);
  Combine(key, material.orm_texture);
  Combine(key, material.metallic_texture);
  Combine(key, material.roughness_texture);
  Combine(key, material.occlusion_texture);
  Combine(key, material.emissive_texture);
  Combine(key, material.reflection_texture);
  return key;
}

MaterialManager::MaterialType MaterialManager::GetMaterialType(
    const mjrfMaterial& material, const Mesh* mesh) {
  if (material.haze_ratio >= 0) {
    return ObjectManager::kHaze;
  } else if (material.decor) {
    if (material.color_texture) {
      return ObjectManager::kUnlitUi;
    } else if (material.color[3] < 1.0f) {
      return ObjectManager::kDecorFade;
    } else {
      return ObjectManager::kDecor;
    }
  } else if (material.orm_texture) {
    if (material.opacity_texture || material.color[3] < 1.0f) {
      return ObjectManager::kPbrPackedTransparent;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPbrPackedReflect;
    } else {
      return ObjectManager::kPbrPacked;
    }
  } else if (material.metallic_texture) {
    if (material.opacity_texture || material.color[3] < 1.0f) {
      return ObjectManager::kPbrTransparent;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPbrReflect;
    } else {
      return ObjectManager::kPbr;
    }
  } else if (material.roughness_texture) {
    if (material.opacity_texture || material.color[3] < 1.0f) {
      return ObjectManager::kPbrTransparent;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPbrReflect;
    } else {
      return ObjectManager::kPbr;
    }
  } else if (material.metallic >= 0) {
    if (material.opacity_texture || material.color[3] < 1.0f) {
      return ObjectManager::kPbrTransparent;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPbrReflect;
    } else {
      return ObjectManager::kPbr;
    }
  } else if (material.roughness >= 0) {
    if (material.opacity_texture || material.color[3] < 1.0f) {
      return ObjectManager::kPbrTransparent;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPbrReflect;
    } else {
      return ObjectManager::kPbr;
    }
  }

  // Check to see if we're dealing with a mesh with texture coordinates.
  // `data_id` is the id of the mesh in model (i.e. the geom has mesh
  // geometry) and `mesh_texcoordadr` stores the address of the mesh uvs if
  // it has them.
  const Texture* color_texture = Texture::downcast(material.color_texture);
  const bool has_texcoords =
      mesh ? mesh->HasVertexAttribute(mjVERTEX_ATTRIBUTE_USAGE_UV) : false;

  if (color_texture == nullptr) {
    if (material.color[3] < 1.0f) {
      return ObjectManager::kPhongColorFade;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPhongColorReflect;
    } else {
      return ObjectManager::kPhongColor;
    }
  } else if (color_texture->GetSamplerType() == mjTEXTURE_CUBE) {
    if (material.color[3] < 1.0f) {
      return ObjectManager::kPhongCubeFade;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPhongCubeReflect;
    } else {
      return ObjectManager::kPhongCube;
    }
  } else if (has_texcoords) {
    if (material.color[3] < 1.0f) {
      return ObjectManager::kPhong2dUvFade;
    } else if (material.reflectance > 0) {
      return ObjectManager::kPhong2dUvReflect;
    } else {
      return ObjectManager::kPhong2dUv;
    }
  } else {
    // Reached only by a mesh without texture coordinates. Meshes are not planar
    // reflectors, so reflectance has no reflect variant here.
    if (material.color[3] < 1.0f) {
      return ObjectManager::kPhong2dFade;
    } else {
      return ObjectManager::kPhong2d;
    }
  }
}

MaterialManager::MaterialKey MaterialManager::PrepareMaterialInstance(
    const mjrfMaterial& material, mjrDrawMode draw_mode, mjtGeom geom_type,
    const Mesh* mesh) {
  ObjectManager::MaterialType type;
  if (material.selected) {
    type = ObjectManager::kOutlineFlatten;
  } else if (draw_mode == mjDRAW_MODE_DEPTH) {
    type = ObjectManager::kUnlitDepth;
  } else if (draw_mode == mjDRAW_MODE_SEGMENTATION_BY_ID) {
    type = ObjectManager::kUnlitSegmentation;
  } else if (draw_mode == mjDRAW_MODE_SEGMENTATION_BY_COLOR) {
    type = ObjectManager::kUnlitSegmentation;
  } else {
    type = GetMaterialType(material, mesh);
  }

  const MaterialKey key = BuildMaterialKey(type, geom_type, material);

  auto it = instances_.find(key);
  if (it == instances_.end()) {
    filament::MaterialInstance* instance =
        object_mgr_->GetMaterial(type)->createInstance();
    UpdateMaterialInstance(instance, material);
    if (geom_type == mjGEOM_PLANE || geom_type == mjGEOM_TRIANGLE) {
      instance->setCullingMode(filament::MaterialInstance::CullingMode::NONE);
    }
    instances_[key] = instance;
  }
  used_keys_.insert(key);
  return key;
}

const filament::MaterialInstance* MaterialManager::GetInstance(MaterialKey key) {
  if (key == 0) {
    return GetEngine()->getDefaultMaterial()->getDefaultInstance();
  }

  auto it = instances_.find(key);
  if (it == instances_.end()) {
    return nullptr;
  }
  return it->second;
}

void MaterialManager::SetBaseColorType(filament::RgbaType base_color_type) {
  base_color_type_ = base_color_type;
}

void MaterialManager::UpdateMaterialInstance(
    filament::MaterialInstance* instance, const mjrfMaterial& material) {
  if (material.scissor[2] != 0 && material.scissor[3] != 0) {
    instance->setScissor(material.scissor[0], material.scissor[1],
                         material.scissor[2], material.scissor[3]);
  }

  const filament::Material* fmaterial = instance->getMaterial();
  if (fmaterial->hasParameter("BaseColorFactor")) {
    instance->setParameter("BaseColorFactor", base_color_type_,
                           ReadFloat4(material.color));
  }
  if (fmaterial->hasParameter("HazeColor")) {
    instance->setParameter("HazeColor", filament::RgbaType::LINEAR,
                           ReadFloat4(material.color));
  }
  if (fmaterial->hasParameter("HazeRatio")) {
    instance->setParameter("HazeRatio", material.haze_ratio);
  }
  if (fmaterial->hasParameter("SegmentationColor")) {
    const uint8_t r = material.segmentation_id >> 0;
    const uint8_t g = material.segmentation_id >> 8;
    const uint8_t b = material.segmentation_id >> 16;
    const filament::math::float4 color{static_cast<float>(r) / 255.0f,
                                       static_cast<float>(g) / 255.0f,
                                       static_cast<float>(b) / 255.0f, 1.0f};
    instance->setParameter("SegmentationColor", filament::RgbaType::LINEAR,
                           color);
  }
  if (fmaterial->hasParameter("EmissiveFactor")) {
    const float emissive =
        material.emissive > 0
            ? material.emissive
            : (material.emissive_texture != nullptr ? 1.0f : 0.0f);
    instance->setParameter("EmissiveFactor", emissive);
  }
  if (fmaterial->hasParameter("SpecularFactor")) {
    instance->setParameter("SpecularFactor", material.specular);
  }
  if (fmaterial->hasParameter("GlossinessFactor")) {
    instance->setParameter("GlossinessFactor", material.glossiness);
  }
  if (fmaterial->hasParameter("MetallicFactor")) {
    instance->setParameter("MetallicFactor",
                           material.metallic >= 0 ? material.metallic : 1.0f);
  }
  if (fmaterial->hasParameter("RoughnessFactor")) {
    instance->setParameter("RoughnessFactor",
                           material.roughness >= 0 ? material.roughness : 1.0f);
  }
  if (fmaterial->hasParameter("UvScale")) {
    instance->setParameter("UvScale", ReadFloat3(material.uv_scale));
  }
  if (fmaterial->hasParameter("UvOffset")) {
    instance->setParameter("UvOffset", ReadFloat3(material.uv_offset));
  }
  if (fmaterial->hasParameter("Reflectance")) {
    instance->setParameter("Reflectance", material.reflectance);
  }
  if (fmaterial->hasParameter("ReflectionNormal")) {
    instance->setParameter("ReflectionNormal",
                           ReadFloat3(material.reflection_normal));
  }
  if (fmaterial->hasParameter("ReflectionViewProj")) {
    instance->setParameter("ReflectionViewProj",
                           ReadMat4(material.reflection_view_proj));
  }

  // Use CLAMP_TO_EDGE for UI/ImGui textures so opposite borders do not bleed
  // across UV 0/1 boundaries under bilinear magnification.
  const auto wrap_mode = material.decor
                             ? filament::TextureSampler::WrapMode::CLAMP_TO_EDGE
                             : filament::TextureSampler::WrapMode::REPEAT;
  // UI textures are drawn at their native size, and Dear ImGui rasterizes
  // glyphs with horizontal oversampling (2 atlas texels per pixel), so a
  // mipmapped minification filter would sample text from a half-resolution
  // mip level. Sample them without mipmaps, like Dear ImGui's own backends.
  const auto min_filter =
      material.decor
          ? filament::TextureSampler::MinFilter::LINEAR
          : filament::TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR;
  filament::TextureSampler sampler;
  sampler.setWrapModeR(wrap_mode);
  sampler.setWrapModeS(wrap_mode);
  sampler.setWrapModeT(wrap_mode);
  sampler.setMagFilter(filament::TextureSampler::MagFilter::LINEAR);
  sampler.setMinFilter(min_filter);

  auto TrySetTexture = [&](const char* name, const mjrfTexture* texture,
                           mjtTextureRole role) {
    if (fmaterial->hasParameter(name)) {
      if (texture != nullptr) {
        instance->setParameter(
            name, Texture::downcast(texture)->GetFilamentTexture(), sampler);
      } else {
        instance->setParameter(name, object_mgr_->GetFallbackTexture(role),
                               sampler);
      }
    }
  };

  TrySetTexture("BaseColor", material.color_texture, mjTEXROLE_RGB);
  TrySetTexture("Opacity", material.opacity_texture, mjTEXROLE_OPACITY);
  TrySetTexture("Normal", material.normal_texture, mjTEXROLE_NORMAL);
  TrySetTexture("Metallic", material.metallic_texture, mjTEXROLE_METALLIC);
  TrySetTexture("Roughness", material.roughness_texture, mjTEXROLE_ROUGHNESS);
  TrySetTexture("Occlusion", material.occlusion_texture, mjTEXROLE_OCCLUSION);
  TrySetTexture("ORM", material.orm_texture, mjTEXROLE_ORM);
  TrySetTexture("Emissive", material.emissive_texture, mjTEXROLE_EMISSIVE);
  TrySetTexture("Reflection", material.reflection_texture, mjTEXROLE_USER);
}

ObjectManager* MaterialManager::GetObjectManager() const { return object_mgr_; }

filament::Engine* MaterialManager::GetEngine() const {
  return object_mgr_->GetEngine();
}

}  // namespace mujoco
