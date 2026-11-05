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

#include "render/filament/support/imgui_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <imgui.h>
#include <math/mat3.h>
#include <math/vec3.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"

namespace mujoco {

using filament::math::float3;
using filament::math::mat3f;

ImguiBridge::ImguiBridge(mjrfContext* ctx, mjrfScene* scene)
    : ctx_(ctx), scene_(scene) {
  mjrfSceneParams params;
  mjrf_defaultSceneParams(&params);
}

ImguiBridge::~ImguiBridge() {
  PrepareRenderables(0);

  // Destroy all textures tracked by ImGui.
  if (ImGui::GetCurrentContext()) {
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
      if (tex->Status != ImTextureStatus_Destroyed) {
        DestroyTexture(tex);
      }
    }
  }
}

uintptr_t ImguiBridge::UploadImage(uintptr_t tex_id,
                                   std::span<const std::byte> pixels, int width,
                                   int height) {
  if (pixels.empty()) {
    // If the pixels are nullptr, we destroy the texture. This is checked before
    // the bpp validation below: a destruction request carries no pixel format.
    if (tex_id != 0) {
      textures_.erase(tex_id);
    }
    return 0;
  }

  if (width <= 0 || height <= 0 ||
      (pixels.size() != static_cast<size_t>(width * height * 3) &&
       pixels.size() != static_cast<size_t>(width * height * 4))) {
    mju_error(
        "Image size (%zu bytes) does not match %dx%d RGB or RGBA",
        pixels.size(), width, height);
  }
  const int bpp = pixels.size() / (width * height);

  // Assign a new texture ID.
  if (tex_id == 0) {
    tex_id = next_tex_id_++;
  }

  mjrfTexture* texture = GetTexture(tex_id);

  // If the texture does not exist or the dimensions have changed, we create a
  // new texture.
  if (texture == nullptr || mjrf_getTextureWidth(texture) != width ||
      mjrf_getTextureHeight(texture) != height) {
    mjrfTextureConfig config;
    mjrf_defaultTextureConfig(&config);
    config.width = width;
    config.height = height;
    config.sampler_type = mjTEXTURE_2D;
    config.format = bpp == 4 ? mjPIXEL_FORMAT_RGBA8 : mjPIXEL_FORMAT_RGB8;
    config.color_space = mjCOLORSPACE_LINEAR;
    UniquePtr<mjrfTexture> new_texture = ::mujoco::CreateTexture(ctx_, config);
    texture = new_texture.get();
    textures_.insert_or_assign(tex_id, std::move(new_texture));
  }

  // Create a copy of the image to pass it to filament as we don't know the
  // lifetime of the data.
  const size_t num_bytes = pixels.size();
  std::byte* bytes = new std::byte[num_bytes];
  std::memcpy(bytes, pixels.data(), num_bytes);
  const auto callback =
      +[](void* user) { delete[] reinterpret_cast<std::byte*>(user); };

  mjrfTextureData texture_data;
  mjrf_defaultTextureData(&texture_data);
  texture_data.bytes = bytes;
  texture_data.num_bytes = num_bytes;
  texture_data.user_data = bytes;
  texture_data.release = callback;
  mjrf_uploadTexture(texture, &texture_data);
  return tex_id;
}

void ImguiBridge::CreateTexture(ImTextureData* data) {
  if (data->Format != ImTextureFormat_RGBA32) {
    mju_error("Unsupported texture format.");
  }

  mjrfTextureConfig config;
  mjrf_defaultTextureConfig(&config);
  config.width = data->Width;
  config.height = data->Height;
  config.sampler_type = mjTEXTURE_2D;
  config.format = mjPIXEL_FORMAT_RGBA8;
  config.color_space = mjCOLORSPACE_LINEAR;

  const uintptr_t tex_id = next_tex_id_++;
  textures_.insert_or_assign(tex_id, ::mujoco::CreateTexture(ctx_, config));
  data->SetTexID((ImTextureID)tex_id);
  UpdateTexture(data);
}

void ImguiBridge::UpdateTexture(ImTextureData* data) {
  auto iter = textures_.find(data->TexID);
  if (iter == textures_.end()) {
    mju_error("Texture not found: %llu", data->TexID);
  }

  mjrfTextureData texture_data;
  mjrf_defaultTextureData(&texture_data);
  texture_data.bytes = data->GetPixels();
  texture_data.num_bytes = data->Width * data->Height * 4;
  texture_data.user_data = nullptr;
  texture_data.release = nullptr;
  mjrf_uploadTexture(iter->second.get(), &texture_data);
  data->SetStatus(ImTextureStatus_OK);
}

void ImguiBridge::DestroyTexture(ImTextureData* data) {
  auto iter = textures_.find(data->TexID);
  if (iter != textures_.end()) {
    textures_.erase(data->TexID);
    data->SetTexID(ImTextureID_Invalid);
    data->SetStatus(ImTextureStatus_Destroyed);
  }
}

mjrfTexture* ImguiBridge::GetTexture(uintptr_t tex_id) const {
  auto iter = textures_.find(tex_id);
  if (iter == textures_.end()) {
    return nullptr;
  }
  return iter->second.get();
}

void ImguiBridge::Update() {
  if (!ImGui::GetCurrentContext()) {
    PrepareRenderables(0);
    return;
  }

  // Prepare the imgui draw commands. We must call this function even if we do
  // not plan on rendering anything to ensure imgui state is updated.
  ImGui::Render();

  ImGuiIO& io = ImGui::GetIO();
  const ImVec2& size = io.DisplaySize;
  const ImVec2& scale = io.DisplayFramebufferScale;
  ImDrawData* commands = ImGui::GetDrawData();
  if (!commands || size.x == 0 || size.y == 0) {
    PrepareRenderables(0);
    return;
  }

  // 2 floats for position, 2 floats for uv, 4 bytes for color.
  constexpr size_t kExpectedVertexSize =
      sizeof(float) * 4 + sizeof(uint8_t) * 4;

  int num_elements = 0;
  for (int n = 0; n < commands->CmdLists.Size; ++n) {
    const ImDrawList* cmds = commands->CmdLists[n];
    if (kExpectedVertexSize != sizeof(cmds->VtxBuffer.Data[0])) {
      mju_error("Invalid vertex buffer size.");
    }
    if (sizeof(uint16_t) != sizeof(cmds->IdxBuffer.Data[0])) {
      mju_error("Invalid index buffer size.");
    }
    num_elements += cmds->CmdBuffer.size();
  }

  if (commands->Textures != nullptr) {
    for (ImTextureData* tex : *commands->Textures) {
      if (tex->Status == ImTextureStatus_WantCreate) {
        CreateTexture(tex);
      } else if (tex->Status == ImTextureStatus_WantUpdates) {
        UpdateTexture(tex);
      } else if (tex->Status == ImTextureStatus_WantDestroy &&
                 tex->UnusedFrames >= 3) {
        DestroyTexture(tex);
      }
    }
  }

  PrepareRenderables(num_elements);
  if (num_elements == 0) {
    return;
  }

  meshes_.clear();
  int renderable_index = 0;
  for (int n = 0; n < commands->CmdLists.Size; ++n) {
    const ImDrawList* cmds = commands->CmdLists[n];

    mjrfMeshConfig config;
    mjrf_defaultMeshConfig(&config);
    config.num_attributes = 3;
    config.attributes[0].usage = mjVERTEX_ATTRIBUTE_USAGE_POSITION;
    config.attributes[0].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;
    config.attributes[1].usage = mjVERTEX_ATTRIBUTE_USAGE_UV;
    config.attributes[1].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;
    config.attributes[2].usage = mjVERTEX_ATTRIBUTE_USAGE_COLOR;
    config.attributes[2].type = mjVERTEX_ATTRIBUTE_TYPE_UBYTE4;
    config.max_vertices = cmds->VtxBuffer.Size;
    config.max_indices = cmds->IdxBuffer.Size;
    config.interleaved = true;
    config.index_type = mjINDEX_TYPE_U16;
    config.primitive_type = mjMESH_PRIMITIVE_TYPE_TRIANGLES;
    meshes_.push_back(CreateMesh(ctx_, config));
    mjrfMesh* mesh = meshes_.back().get();

    mjrfMeshData data;
    mjrf_defaultMeshData(&data);
    data.num_vertices = cmds->VtxBuffer.Size;
    data.vertices[0] = cmds->VtxBuffer.Data;
    data.vertices[1] = cmds->VtxBuffer.Data + sizeof(float) * 2;
    data.vertices[2] = cmds->VtxBuffer.Data + sizeof(float) * 4;
    data.num_indices = cmds->IdxBuffer.Size;
    data.indices = cmds->IdxBuffer.Data;
    mjrf_uploadMesh(mesh, &data);

    for (const ImDrawCmd& command : cmds->CmdBuffer) {
      // The framebuffer's size. The scale is the framebuffer size over the
      // display size, so the product is a whole number up to float error;
      // round it, since truncating a 875.99994 to 875 would shift every
      // scissor (measured from the bottom edge) down a row and clip the top
      // row of every window.
      const int width = static_cast<int>(std::lround(size.x * scale.x));
      const int height = static_cast<int>(std::lround(size.y * scale.y));

      UniquePtr<mjrfRenderable>& renderable = renderables_[renderable_index];
      mjrf_setRenderableMesh(renderable.get(), mesh, command.IdxOffset,
                             command.ElemCount);

      mjrfMaterial material;
      mjrf_defaultMaterial(&material);
      material.color_texture = GetTexture(command.GetTexID());

      material.decor = true;
      // Scale clip rects to physical pixels here instead of using
      // ScaleClipRects, which previously caused flickering and clipping bugs:
      // it mutates draw data in place, so draw lists that are re-rendered
      // across frames get scaled repeatedly (this happens in the web viewer,
      // where the same draw lists are rendered until a new network frame
      // arrives). We also clamp to the viewport because clip rects may extend
      // slightly outside it (modal dialogs by design; fractional DPI rounding
      // for bottom or right-docked windows), and filament rejects out-of-window
      // scissor rects.
      const float clip_x0 = std::clamp(command.ClipRect.x * scale.x, 0.0f,
                                       static_cast<float>(width));
      const float clip_y0 = std::clamp(command.ClipRect.y * scale.y, 0.0f,
                                       static_cast<float>(height));
      const float clip_x1 = std::clamp(command.ClipRect.z * scale.x, clip_x0,
                                       static_cast<float>(width));
      const float clip_y1 = std::clamp(command.ClipRect.w * scale.y, clip_y0,
                                       static_cast<float>(height));
      // Snap the rect outward to whole pixels. At a fractional DPI scale the
      // clip edges fall between pixels, and filament truncates the scissor
      // to integers: measured from the bottom (the GL origin), that drops a
      // row along the top edge of every window, showing whatever is behind
      // the UI through a one-pixel seam (e.g. the 3D scene above and below
      // the menu bar). Growing the rect is harmless: the geometry's own
      // anti-aliased edges end where they should.
      const float x0 = std::floor(clip_x0);
      const float y0 = std::floor(clip_y0);
      const float x1 = std::ceil(clip_x1);
      const float y1 = std::ceil(clip_y1);
      material.scissor[0] = x0;
      material.scissor[1] = height - y1;
      material.scissor[2] = x1 - x0;
      material.scissor[3] = y1 - y0;
      mjrf_setRenderableMaterial(renderable.get(), &material);

      const float size[] = {scale.x, scale.y, 1.0f};
      mjrf_setRenderableSize(renderable.get(), size);

      ++renderable_index;
    }
  }
}

void ImguiBridge::PrepareRenderables(int count) {
  while (renderables_.size() < count) {
    mjrfRenderableParams params;
    mjrf_defaultRenderableParams(&params);
    params.cast_shadow = false;
    params.receive_shadow = false;
    params.blend_order = static_cast<std::uint16_t>(renderables_.size() + 1);
    auto& renderable =
        renderables_.emplace_back(CreateRenderable(ctx_, params));
    mjrf_addSceneRenderable(scene_, renderable.get());
  }
  while (renderables_.size() > count) {
    mjrf_removeSceneRenderable(scene_, renderables_.back().get());
    renderables_.pop_back();
  }
}
}  // namespace mujoco
