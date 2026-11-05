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

#include "render/filament/core/object_manager.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/Material.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>
#include <math/vec2.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/builtins.h"
#include "render/filament/support/filament_util.h"
#include "user/user_resource.h"

namespace mujoco {

static filament::Material* LoadMaterial(filament::Engine* engine,
                                        std::string_view filename) {
  const std::string path = ResolveFilamentAssetPath(std::string(filename));
  mjResource* resource =
      mju_openResource("", path.c_str(), nullptr, nullptr, 0);
  if (!resource) {
    mju_error("Failed to open filament asset '%.*s' at '%s'",
              static_cast<int>(filename.size()), filename.data(), path.c_str());
    return nullptr;
  }
  void* payload = nullptr;
  int size = mju_readResource(resource, const_cast<const void**>(&payload));
  if (size <= 0 || !payload) {
    mju_error("Failed to read filament asset '%.*s' (size=%d)",
              static_cast<int>(filename.size()), filename.data(), size);
    mju_closeResource(resource);
    return nullptr;
  }
  filament::Material::Builder material_builder;
  material_builder.package(payload, size);
  filament::Material* material = material_builder.build(*engine);
  mju_closeResource(resource);
  return material;
};

ObjectManager::ObjectManager(filament::Engine* engine) : engine_(engine) {
  materials_[kPbr] = LoadMaterial(engine, "pbr.filamat");
  materials_[kPbrPacked] = LoadMaterial(engine, "pbr_packed.filamat");
  materials_[kPbrTransparent] = LoadMaterial(engine, "pbr_transparent.filamat");
  materials_[kPbrPackedTransparent] =
      LoadMaterial(engine, "pbr_packed_transparent.filamat");
  materials_[kPbrReflect] = LoadMaterial(engine, "pbr_reflect.filamat");
  materials_[kPbrPackedReflect] =
      LoadMaterial(engine, "pbr_packed_reflect.filamat");
  materials_[kPhong2d] = LoadMaterial(engine, "phong_2d.filamat");
  materials_[kPhong2dFade] = LoadMaterial(engine, "phong_2d_fade.filamat");
  materials_[kPhong2dUv] = LoadMaterial(engine, "phong_2d_uv.filamat");
  materials_[kPhong2dUvFade] = LoadMaterial(engine, "phong_2d_uv_fade.filamat");
  materials_[kPhong2dUvReflect] =
      LoadMaterial(engine, "phong_2d_uv_reflect.filamat");
  materials_[kPhongColor] = LoadMaterial(engine, "phong_color.filamat");
  materials_[kPhongColorFade] =
      LoadMaterial(engine, "phong_color_fade.filamat");
  materials_[kPhongColorReflect] =
      LoadMaterial(engine, "phong_color_reflect.filamat");
  materials_[kPhongCube] = LoadMaterial(engine, "phong_cube.filamat");
  materials_[kPhongCubeFade] = LoadMaterial(engine, "phong_cube_fade.filamat");
  materials_[kPhongCubeReflect] =
      LoadMaterial(engine, "phong_cube_reflect.filamat");
  materials_[kUnlitSegmentation] =
      LoadMaterial(engine, "unlit_segmentation.filamat");
  materials_[kDecor] = LoadMaterial(engine, "decor.filamat");
  materials_[kDecorFade] = LoadMaterial(engine, "decor_fade.filamat");
  materials_[kUnlitDepth] = LoadMaterial(engine, "unlit_depth.filamat");
  materials_[kUnlitUi] = LoadMaterial(engine, "unlit_ui.filamat");
  materials_[kOutlineComposite] =
      LoadMaterial(engine, "outline_composite.filamat");
  materials_[kOutlineFlatten] = LoadMaterial(engine, "outline_flatten.filamat");
  materials_[kOutlineJumpFlood] =
      LoadMaterial(engine, "outline_jumpflood.filamat");
  materials_[kHaze] = LoadMaterial(engine, "haze.filamat");

  static uint8_t black_rgb[3] = {0, 0, 0};
  static uint8_t white_rgb[3] = {255, 255, 255};
  static uint8_t normal_data[3] = {128, 128, 255};
  static uint8_t orm_data[3] = {0, 255, 0};

  auto CreateFallbackTexture = [this](uint8_t color[3]) {
    filament::Texture::Builder builder;
    builder.width(1);
    builder.height(1);
    builder.format(filament::Texture::InternalFormat::RGB8);
    builder.sampler(filament::Texture::Sampler::SAMPLER_2D);
    filament::Texture* texture = builder.build(*engine_);
    const filament::Texture::Type type = filament::Texture::Type::UBYTE;
    const filament::Texture::Format format = filament::Texture::Format::RGB;
    texture->setImage(*engine_, 0, {color, 3, format, type});
    return texture;
  };

  fallback_black_ = CreateFallbackTexture(black_rgb);
  fallback_white_ = CreateFallbackTexture(white_rgb);
  fallback_normal_ = CreateFallbackTexture(normal_data);
  fallback_orm_ = CreateFallbackTexture(orm_data);

  fallback_textures_[mjTEXROLE_USER] = fallback_black_;
  fallback_textures_[mjTEXROLE_RGB] = fallback_white_;
  fallback_textures_[mjTEXROLE_OPACITY] = fallback_white_;
  fallback_textures_[mjTEXROLE_OCCLUSION] = fallback_white_;
  fallback_textures_[mjTEXROLE_ROUGHNESS] = fallback_white_;
  fallback_textures_[mjTEXROLE_METALLIC] = fallback_white_;
  fallback_textures_[mjTEXROLE_NORMAL] = fallback_normal_;
  fallback_textures_[mjTEXROLE_EMISSIVE] = fallback_white_;
  fallback_textures_[mjTEXROLE_ORM] = fallback_orm_;

  CreateQuadBuffers();
}

ObjectManager::~ObjectManager() {
  engine_->destroy(quad_vb_);
  engine_->destroy(quad_ib_);
  engine_->destroy(fallback_black_);
  engine_->destroy(fallback_white_);
  engine_->destroy(fallback_normal_);
  engine_->destroy(fallback_orm_);
  for (auto& iter : materials_) {
    engine_->destroy(iter);
  }
}

filament::Material* ObjectManager::GetMaterial(MaterialType type) const {
  if (type < 0 || type >= kNumMaterials) {
    mju_error("Invalid material type: %d", type);
  }
  return materials_[type];
}

Builtins* ObjectManager::GetBuiltins(int nstack, int nslice, int nquad) {
  // Assumes nstack, nslice, and nquad are non-negative and less than 2^20.
  std::uint64_t key = (static_cast<uint64_t>(nstack) << 20) |
                      (static_cast<uint64_t>(nslice) << 40) |
                      static_cast<uint64_t>(nquad);

  auto iter = builtins_.find(key);
  if (iter == builtins_.end()) {
    auto builtins = std::make_unique<Builtins>(engine_, nstack, nslice, nquad);
    Builtins* ptr = builtins.get();
    builtins_[key] = std::move(builtins);
    return ptr;
  }
  return iter->second.get();
}

const filament::Texture* ObjectManager::GetFallbackTexture(
    mjtTextureRole role) const {
  if (role < 0 || role >= mjNTEXROLE) {
    mju_error("Invalid texture role: %d", role);
  }
  return fallback_textures_[role];
}

void ObjectManager::CreateQuadBuffers() {
  // Define a single triangle that completely covers the viewport. (A single
  // large triangle is more efficient than a two-triangle quad because it
  // avoids the diagonal edge where pixels might be rasterized twice.)
  static const filament::math::float2 kVertices[3] = {
      {-1.0f, -1.0f}, {3.0f, -1.0f}, {-1.0f, 3.0f}};

  static const uint16_t kIndices[3] = {0, 1, 2};

  filament::VertexBuffer::Builder vb_builder;
  vb_builder.bufferCount(1);
  vb_builder.vertexCount(3);
  vb_builder.attribute(filament::VertexAttribute::POSITION, 0,
                       filament::VertexBuffer::AttributeType::FLOAT2, 0,
                       sizeof(filament::math::float2));
  quad_vb_ = vb_builder.build(*engine_);
  quad_vb_->setBufferAt(*engine_, 0, {kVertices, sizeof(kVertices)});

  filament::IndexBuffer::Builder ib_builder;
  ib_builder.indexCount(3);
  ib_builder.bufferType(filament::IndexBuffer::IndexType::USHORT);
  quad_ib_ = ib_builder.build(*engine_);
  quad_ib_->setBuffer(*engine_, {kIndices, sizeof(kIndices)});
}

}  // namespace mujoco
