// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Wire-format implementation for the web viewer's state payload and model
// bundle. Compiled into BOTH sides of the wire: the state_payload pybind
// module and the wasm web_client. It must therefore stay free of python- or
// browser-specific dependencies.

#include "state_payload.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::studio {
namespace {

// Appends raw bytes to the payload buffer.
void AppendBytes(std::vector<std::byte>& buffer, const void* data,
                 size_t size) {
  const std::byte* bytes = static_cast<const std::byte*>(data);
  buffer.insert(buffer.end(), bytes, bytes + size);
}

// Appends a complete [u32 tag][u32 size][payload] block.
void AppendStateBlock(std::vector<std::byte>& buffer, uint32_t tag,
                      const void* data, size_t size) {
  StateBlockHeader block_header{tag, static_cast<uint32_t>(size)};
  AppendBytes(buffer, &block_header, sizeof(block_header));
  AppendBytes(buffer, data, size);
}

// Serializes the render state (exactly kRenderStateSize bytes) into `ptr`.
// Must copy the same fields in the same order as ParseRenderState.
void SerializeRenderStateInto(std::byte* ptr, const mjvCamera& camera,
                              const mjvPerturb& perturb,
                              const mjvOption& vis_options, const mjOption& opt,
                              const mjVisual& vis, const mjStatistic& stat,
                              const std::vector<uint8_t>& render_flags) {
  memcpy(ptr, &camera, sizeof(mjvCamera));
  ptr += sizeof(mjvCamera);

  memcpy(ptr, &perturb, sizeof(mjvPerturb));
  ptr += sizeof(mjvPerturb);

  memcpy(ptr, &vis_options, sizeof(mjvOption));
  ptr += sizeof(mjvOption);

  memcpy(ptr, &opt, sizeof(mjOption));
  ptr += sizeof(mjOption);

  memcpy(ptr, &vis, sizeof(mjVisual));
  ptr += sizeof(mjVisual);

  memcpy(ptr, &stat, sizeof(mjStatistic));
  ptr += sizeof(mjStatistic);

  // Pack render flags (mjNRNDFLAG bytes).
  memset(ptr, 0, mjNRNDFLAG);
  for (size_t i = 0; i < mjNRNDFLAG && i < render_flags.size(); ++i) {
    ptr[i] = static_cast<std::byte>(render_flags[i]);
  }
}

}  // namespace

// Parses a render state block produced by SerializeStatePayload.
// Must copy the same fields in the same order as SerializeRenderStateInto.
void ParseRenderState(const std::byte* data, RenderStateView* out) {
  const std::byte* ptr = data;

  memcpy(&out->camera, ptr, sizeof(mjvCamera));
  ptr += sizeof(mjvCamera);

  memcpy(&out->perturb, ptr, sizeof(mjvPerturb));
  ptr += sizeof(mjvPerturb);

  memcpy(&out->vis_options, ptr, sizeof(mjvOption));
  ptr += sizeof(mjvOption);

  memcpy(&out->opt, ptr, sizeof(mjOption));
  ptr += sizeof(mjOption);

  memcpy(&out->vis, ptr, sizeof(mjVisual));
  ptr += sizeof(mjVisual);

  memcpy(&out->stat, ptr, sizeof(mjStatistic));
  ptr += sizeof(mjStatistic);

  memcpy(out->render_flags, ptr, mjNRNDFLAG);
}

size_t MaxStatePayloadSize(size_t physics_bytes) {
  return sizeof(StatePayloadHeader) + 3 * sizeof(StateBlockHeader) +
         (sizeof(int32_t) + physics_bytes) + kRenderStateSize +
         kMaxExtraGeoms * sizeof(mjvGeom);
}

std::vector<std::byte> SerializeStatePayload(
    uint32_t model_crc32, int32_t physics_spec, const void* physics,
    size_t physics_bytes, const mjvCamera& camera, const mjvPerturb& perturb,
    const mjvOption& vis_options, const mjOption& opt, const mjVisual& vis,
    const mjStatistic& stat, const std::vector<uint8_t>& render_flags,
    const mjvGeom* extra_geoms, size_t extra_geom_count) {
  extra_geom_count =
      extra_geom_count > kMaxExtraGeoms ? kMaxExtraGeoms : extra_geom_count;
  std::vector<std::byte> buffer;
  buffer.reserve(MaxStatePayloadSize(physics_bytes));

  StatePayloadHeader header;
  header.nblocks = extra_geom_count > 0 ? 3 : 2;
  header.model_crc32 = model_crc32;
  AppendBytes(buffer, &header, sizeof(header));

  // Physics state: [i32 spec][mjtNum values...].
  StateBlockHeader physics_header{
      kTagPhysicsState, static_cast<uint32_t>(sizeof(int32_t) + physics_bytes)};
  AppendBytes(buffer, &physics_header, sizeof(physics_header));
  AppendBytes(buffer, &physics_spec, sizeof(int32_t));
  AppendBytes(buffer, physics, physics_bytes);

  // Render state, serialized into place.
  StateBlockHeader render_header{kTagRenderState,
                                 static_cast<uint32_t>(kRenderStateSize)};
  AppendBytes(buffer, &render_header, sizeof(render_header));
  const size_t render_offset = buffer.size();
  buffer.resize(render_offset + kRenderStateSize);
  SerializeRenderStateInto(buffer.data() + render_offset, camera, perturb,
                           vis_options, opt, vis, stat, render_flags);

  // Extra geoms (only when present).
  if (extra_geom_count > 0) {
    AppendStateBlock(buffer, kTagExtraGeoms, extra_geoms,
                     extra_geom_count * sizeof(mjvGeom));
  }

  return buffer;
}

bool ParseStatePayload(const void* data, size_t size, StatePayloadView* out) {
  const std::byte* bytes = static_cast<const std::byte*>(data);
  if (size < sizeof(StatePayloadHeader)) return false;

  StatePayloadHeader header;
  memcpy(&header, bytes, sizeof(header));
  if (header.magic != kStatePayloadMagic) return false;
  if (header.version != kStatePayloadVersion) return false;
  out->model_crc32 = header.model_crc32;

  size_t offset = sizeof(StatePayloadHeader);
  for (uint16_t i = 0; i < header.nblocks; ++i) {
    if (offset + sizeof(StateBlockHeader) > size) return false;
    StateBlockHeader block;
    memcpy(&block, bytes + offset, sizeof(block));
    offset += sizeof(StateBlockHeader);
    if (offset + block.size > size) return false;
    const std::byte* payload = bytes + offset;

    switch (block.tag) {
      case kTagPhysicsState:
        if (block.size < sizeof(int32_t)) return false;
        // The state is an array of mjtNum; a block that is not a whole number
        // of them is corrupt (see StatePayloadView::physics_bytes).
        if ((block.size - sizeof(int32_t)) % sizeof(mjtNum) != 0) return false;
        memcpy(&out->physics_spec, payload, sizeof(int32_t));
        out->physics = payload + sizeof(int32_t);
        out->physics_bytes = block.size - sizeof(int32_t);
        break;
      case kTagRenderState:
        if (block.size != kRenderStateSize) return false;
        out->render_state = payload;
        break;
      case kTagExtraGeoms:
        if (block.size % sizeof(mjvGeom) != 0) return false;
        out->extra_geoms = payload;
        out->extra_geom_count = block.size / sizeof(mjvGeom);
        break;
      default:
        break;  // Unknown tag: skip.
    }
    offset += block.size;
  }
  return true;
}

std::vector<std::byte> SerializeModelBundle(
    const mjModel* m, std::vector<std::string>* unreadable) {
  std::vector<std::byte> buffer;

  // Header, patched once the resource count is known.
  ModelBundleHeader header;
  header.mjb_size = mj_sizeModel(m);
  AppendBytes(buffer, &header, sizeof(header));

  // The MJB, saved straight into the bundle.
  const size_t mjb_offset = buffer.size();
  buffer.resize(mjb_offset + header.mjb_size);
  mj_saveModel(m, nullptr, buffer.data() + mjb_offset, header.mjb_size);

  // One entry per distinct by-reference texture path. The VFS keeps any .mjz
  // archive mounted across the reads of this bundle and is deleted with it.
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  std::vector<std::string_view> seen;
  for (int i = 0; i < m->ntex; ++i) {
    if (m->tex_adr[i] != -1 || m->tex_pathadr[i] < 0) continue;
    std::string_view path(m->paths + m->tex_pathadr[i]);
    if (path.empty()) continue;
    if (std::find(seen.begin(), seen.end(), path) != seen.end()) continue;
    seen.push_back(path);

    char error[1024] = "";
    mjResource* resource = mju_openResource("", std::string(path).c_str(),
                                            &vfs, error, sizeof(error));
    const void* data = nullptr;
    int size = resource ? mju_readResource(resource, &data) : -1;
    if (size < 0) {
      if (unreadable) {
        unreadable->push_back(std::string(path) + ": " +
                              (resource ? "read failed" : error));
      }
      if (resource) mju_closeResource(resource);
      continue;
    }

    const uint32_t path_size = static_cast<uint32_t>(path.size());
    const uint64_t data_size = static_cast<uint64_t>(size);
    AppendBytes(buffer, &path_size, sizeof(path_size));
    AppendBytes(buffer, &data_size, sizeof(data_size));
    AppendBytes(buffer, path.data(), path.size());
    AppendBytes(buffer, data, size);
    mju_closeResource(resource);
    ++header.nresources;
  }
  mj_deleteVFS(&vfs);

  memcpy(buffer.data(), &header, sizeof(header));
  return buffer;
}

bool ParseModelBundle(const void* data, size_t size, ModelBundleView* out) {
  const std::byte* bytes = static_cast<const std::byte*>(data);
  if (size < sizeof(ModelBundleHeader)) return false;

  ModelBundleHeader header;
  memcpy(&header, bytes, sizeof(header));
  if (header.magic != kModelBundleMagic) return false;
  if (header.version != kModelBundleVersion) return false;

  size_t offset = sizeof(ModelBundleHeader);
  if (header.mjb_size > size - offset) return false;
  out->mjb = bytes + offset;
  out->mjb_size = header.mjb_size;
  offset += header.mjb_size;

  out->resources.clear();
  out->resources.reserve(header.nresources);
  for (uint32_t i = 0; i < header.nresources; ++i) {
    uint32_t path_size;
    uint64_t data_size;
    if (size - offset < sizeof(path_size) + sizeof(data_size)) return false;
    memcpy(&path_size, bytes + offset, sizeof(path_size));
    offset += sizeof(path_size);
    memcpy(&data_size, bytes + offset, sizeof(data_size));
    offset += sizeof(data_size);
    if (path_size > size - offset) return false;
    ModelBundleResource resource;
    resource.path = std::string_view(
        reinterpret_cast<const char*>(bytes + offset), path_size);
    offset += path_size;
    if (data_size > size - offset) return false;
    resource.data = bytes + offset;
    resource.size = data_size;
    offset += data_size;
    out->resources.push_back(resource);
  }
  return true;
}

}  // namespace mujoco::studio
