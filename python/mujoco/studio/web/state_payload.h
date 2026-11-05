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

// This file defines the serialization format for the web viewer's browser
// client render payload containing the data needed so that the browser can
// render the scene using the following call:
//
// Render(model, data, perturb, camera, vis_options, width, height, extra_geoms)
//
// The arguments come from the Python process:
//
//   * model       : fetched once over HTTP as /model; its runtime-mutable
//                   parts (opt/vis/stat) are re-sent in the render state block.
//   * data        : streamed as the physics state vector (mjSTATE_INTEGRATION);
//                   the browser recomputes the rest via mj_setState/mj_forward.
//   * width/height: the browser canvas size.
//   * extra_geoms : optional variable-size kTagExtraGeoms block.
//   * ...         : the rest of the arguments are sent as a fixed-size block
//
// The payload (SerializeStatePayload) is a sequence of tagged blocks:
//
//   [StatePayloadHeader][u32 tag][u32 size][payload]...
//
// The payload is serialized by Python, sent over the /state WebSocket, and
// parsed by the browser.
//
// TODO(matijak): Try shrinking the physics block: float32 (or quantized) values
// instead of doubles, and/or delta-encoding against the client's last-acked
// payload. The /state ack (web_server.py) tells the server which snapshot each
// client last applied, which is the baseline that delta compression needs. For
// 100humanoids.xml the payload is ~181 KB of doubles and dominates slow links.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_STATE_PAYLOAD_H_
#define MUJOCO_PYTHON_STUDIO_WEB_STATE_PAYLOAD_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// "MJWS" as little-endian bytes. This magic constant identifies the
// StateServer WebSocket payload header and helps detect malformed or
// misrouted messages.
constexpr uint32_t kStatePayloadMagic =
    'M' | ('J' << 8) | ('W' << 16) | ('S' << 24);
constexpr uint16_t kStatePayloadVersion = 1;

struct StatePayloadHeader {
  uint32_t magic = kStatePayloadMagic;
  uint16_t version = kStatePayloadVersion;
  uint16_t nblocks = 0;
  // CRC32 of the model's MJB bytes. When this changes, the browser must
  // refetch /model before applying any further state.
  uint32_t model_crc32 = 0;
};
static_assert(sizeof(StatePayloadHeader) == 12);

// Block tags. Readers must skip unknown tags.
enum StateBlockTag : uint32_t {
  kTagPhysicsState = 1,  // [i32 mjtState spec signature][mjtNum values...]
  kTagRenderState = 2,   // fixed-size block of kRenderStateSize bytes
  kTagExtraGeoms = 3,    // n x mjvGeom (n = size / sizeof(mjvGeom))
};

struct StateBlockHeader {
  uint32_t tag = 0;
  uint32_t size = 0;
};
static_assert(sizeof(StateBlockHeader) == 8);

// Fixed byte size of the render state block appended after physics state.
// These are plain C structs of int/float/double members whose total size is
// fixed, independent of the model and generally negligible compared to the size
// of the physics state
constexpr size_t kRenderStateSize =
    sizeof(mjvCamera) + sizeof(mjvPerturb) + sizeof(mjvOption) +
    sizeof(mjOption) + sizeof(mjVisual) + sizeof(mjStatistic) + mjNRNDFLAG;

// Maximum number of extra geoms serialized per frame. Bounds the shared
// memory buffer the StateServer allocates; WebViewer truncates longer lists.
constexpr uint32_t kMaxExtraGeoms = 1024;

// Upper bound of a serialized payload, used to size the StateServer's shared
// memory buffer. `physics_bytes` is mj_stateSize(...) * sizeof(mjtNum).
size_t MaxStatePayloadSize(size_t physics_bytes);

// Serialize the complete state payload sent over the state WebSocket.
std::vector<std::byte> SerializeStatePayload(
    uint32_t model_crc32, int32_t physics_spec, const void* physics,
    size_t physics_bytes, const mjvCamera& camera, const mjvPerturb& perturb,
    const mjvOption& vis_options, const mjOption& opt, const mjVisual& vis,
    const mjStatistic& stat, const std::vector<uint8_t>& render_flags,
    const mjvGeom* extra_geoms, size_t extra_geom_count);

// Parsed view into a serialized payload. Pointers alias the input buffer and
// are NOT guaranteed to be aligned; so you must memcpy the data out before use.
struct StatePayloadView {
  uint32_t model_crc32 = 0;
  int32_t physics_spec = 0;
  const std::byte* physics = nullptr;
  size_t physics_bytes = 0;  // Always a multiple of sizeof(mjtNum).
  const std::byte* render_state = nullptr;  // kRenderStateSize bytes when non-null
  const std::byte* extra_geoms = nullptr;   // extra_geom_count * sizeof(mjvGeom)
  size_t extra_geom_count = 0;
};

// Parses a payload produced by SerializeStatePayload. Returns false if the
// buffer is malformed (bad magic/version or out-of-bounds block). Blocks
// with unknown tags are skipped.
bool ParseStatePayload(const void* data, size_t size, StatePayloadView* out);

// A decoded render state block.
struct RenderStateView {
  mjvCamera camera;
  mjvPerturb perturb;
  mjvOption vis_options;
  mjOption opt;
  mjVisual vis;
  mjStatistic stat;
  uint8_t render_flags[mjNRNDFLAG];
};

// Decodes a render state block produced by the serializer. `data` must hold
// kRenderStateSize bytes (e.g. StatePayloadView::render_state).
void ParseRenderState(const std::byte* data, RenderStateView* out);

// ---------------------------------------------------------------------------
// Model bundle: the payload served as /model.
//
// A model compiled with mjPASSTHROUGH_TEXTURES stores no texture pixels; its
// MJB only carries the path of every by-reference texture (tex_adr == -1,
// path at m->paths + tex_pathadr). Those paths point at the server's
// filesystem (or into a .mjz archive), which the browser cannot open. The
// bundle therefore packs the MJB together with the encoded file bytes of every
// such texture, so one download (through the existing chunked, retried
// /model transport) carries everything the browser needs:
//
//   [ModelBundleHeader][mjb bytes]
//   nresources x { [u32 path_size][u64 data_size][path bytes][data bytes] }
//
// Paths are the exact strings found in m->paths, so the browser can match
// them without any out-of-band listing. A model without by-reference textures
// is a bundle with nresources == 0.

// "MJBD" as little-endian bytes.
constexpr uint32_t kModelBundleMagic =
    'M' | ('J' << 8) | ('B' << 16) | ('D' << 24);
constexpr uint32_t kModelBundleVersion = 1;

struct ModelBundleHeader {
  uint32_t magic = kModelBundleMagic;
  uint32_t version = kModelBundleVersion;
  uint64_t mjb_size = 0;
  uint32_t nresources = 0;
  uint32_t reserved = 0;
};
static_assert(sizeof(ModelBundleHeader) == 24);

// Serializes `m` (mj_saveModel) plus the encoded bytes of each by-reference
// texture, read with mju_openResource so local files, .mjz archives and
// registered resource providers all work. Archives are mounted into a VFS
// that lives only for the duration of this call. Textures whose file cannot
// be read are left out of the bundle (the browser then renders them untextured
// with a warning, like the native viewer) and their paths are appended to
// `unreadable`, together with the error from mju_openResource.
std::vector<std::byte> SerializeModelBundle(
    const mjModel* m, std::vector<std::string>* unreadable = nullptr);

// One by-reference texture file in a parsed bundle. Pointers alias the input
// buffer.
struct ModelBundleResource {
  std::string_view path;  // as found in mjModel::paths
  const std::byte* data = nullptr;
  size_t size = 0;
};

// Parsed view into a serialized bundle. Pointers alias the input buffer.
struct ModelBundleView {
  const std::byte* mjb = nullptr;
  size_t mjb_size = 0;
  std::vector<ModelBundleResource> resources;
};

// Parses a bundle produced by SerializeModelBundle. Returns false if the
// buffer is malformed.
bool ParseModelBundle(const void* data, size_t size, ModelBundleView* out);

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_STATE_PAYLOAD_H_
