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

#include "ui_codec.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <imgui.h>

namespace mujoco::studio::netui {
namespace {

constexpr size_t kChunkBytes = sizeof(uint64_t);

constexpr uint64_t AlignUp8(uint64_t bytes) {
  return (bytes + kChunkBytes - 1) & ~(uint64_t{kChunkBytes - 1});
}

// Per-list wire header (16 bytes) preceding each draw list's payload in a
// kDrawFrame message.
struct WireListHeader {
  uint64_t group_id = 0;
  uint32_t unpacked_bytes = 0;
  uint32_t payload_bytes = 0;
};

// Fixed header at the start of each SerializedList::bytes buffer, followed
// by 8-byte-aligned sections for vertices (Vert), indices (uint16_t), and
// draw commands (WireDrawCommand).
struct SerializedListHeader {
  float ref_pos[2] = {};
  uint32_t vert_count = 0;
  uint32_t idx_count = 0;
  uint32_t command_count = 0;
};
constexpr size_t kListHeaderBytes = AlignUp8(sizeof(SerializedListHeader));

// Fixed 36-byte wire representation of a DrawCommand (with no tail padding).
struct WireDrawCommand {
  uint64_t tex_id;
  uint32_t vtx_offset;
  uint32_t idx_offset;
  uint32_t elem_count;
  float clip_rect[4];
};
constexpr size_t kWireDrawCommandBytes = 36;

// Layout for version 1.
static_assert(std::endian::native == std::endian::little);
static_assert(kCodecVersion == 1);
static_assert(sizeof(MessageHeader) == 8);
static_assert(offsetof(Hello, header) == 0);
static_assert(sizeof(Hello) == 8);
static_assert(offsetof(Input, header) == 0);
static_assert(sizeof(Input) == 328);
static_assert(ImGuiKey_NamedKey_COUNT <= static_cast<int>(Input::kMaxKeys),
              "Increase Input::kMaxKeys for updated Dear ImGui key count");
static_assert(ImGuiMouseButton_COUNT <= 8,
              "mouse_down_mask assumes at most 8 mouse buttons");
static_assert(offsetof(Texture, header) == 0);
static_assert(sizeof(Texture) == 32);
static_assert(offsetof(Clipboard, header) == 0);
static_assert(sizeof(Clipboard) == 8);
static_assert(offsetof(DrawFrame, header) == 0);
static_assert(sizeof(DrawFrame) == 40);
static_assert(sizeof(Vert) == 12);
static_assert(std::is_trivially_copyable_v<Vert>);
static_assert(sizeof(WireListHeader) == 16);
static_assert(sizeof(SerializedListHeader) == 20);
static_assert(sizeof(WireDrawCommand) >= kWireDrawCommandBytes);
static_assert(offsetof(WireDrawCommand, tex_id) == 0);
static_assert(offsetof(WireDrawCommand, vtx_offset) == 8);
static_assert(offsetof(WireDrawCommand, idx_offset) == 12);
static_assert(offsetof(WireDrawCommand, elem_count) == 16);
static_assert(offsetof(WireDrawCommand, clip_rect) == 20);
static_assert(std::is_trivially_copyable_v<WireDrawCommand>);

bool AllFinite(const float* values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

// Quantizes `value` in [min_val, max_val] to a 16-bit unsigned integer,
// clamping out-of-range and NaN inputs before scaling.
uint16_t QuantizeNormalized(float value, float min_val, float max_val) {
  const float normalized = (value - min_val) / (max_val - min_val);
  // Reject NaN and <= 0 before scaling to [0, 65535].
  if (!(normalized > 0.0f)) return 0;
  if (normalized >= 1.0f) return 0xFFFFu;
  return static_cast<uint16_t>(normalized * 65535.0f + 0.5f);
}

// Quantizes a vertex position offset to the nearest Vert::kPosStep step above
// Vert::kPosMin, clamping out-of-range and NaN inputs.
uint16_t QuantizePos(float offset) {
  const float steps = (offset - Vert::kPosMin) / Vert::kPosStep;
  // Reject NaN and <= 0 before rounding to [0, 65535].
  if (!(steps > 0.0f)) return 0;
  if (steps >= 65535.0f) return 0xFFFFu;
  return static_cast<uint16_t>(steps + 0.5f);
}

// Computes a non-zero 64-bit hash of the Dear ImGui window owner name so each
// window's draw list can be matched across frames.
uint64_t HashOwnerName(const char* name) {
  if (name == nullptr) return 1;
  const uint64_t hash =
      std::hash<std::string_view>()(std::string_view(name));
  return hash == 0 ? 1 : hash;
}

// Finds the serialized draw list with the same `group_id` in `prev_frame`,
// checking `index_hint` first since window order rarely changes between frames.
const Bytes* FindMatchingListBytes(const SerializedFrame* prev_frame,
                                   uint64_t group_id, size_t index_hint) {
  if (prev_frame == nullptr) return nullptr;
  if (index_hint < prev_frame->lists.size() &&
      prev_frame->lists[index_hint].group_id == group_id) {
    return &prev_frame->lists[index_hint].bytes;
  }
  for (const SerializedList& list : prev_frame->lists) {
    if (list.group_id == group_id) return &list.bytes;
  }
  return nullptr;
}

// RLE delta compression compares draw lists in 8-byte (uint64_t) chunks rather
// than byte-by-byte: comparing 64 bits at a time is faster and avoids
// splitting runs on matching single zero bytes inside floats and integers.
// Each RLE block begins with an 8-byte header packing (same_chunks,
// diff_chunks), followed by `diff_chunks` raw 8-byte chunks from `curr`.
uint64_t ReadChunk(const std::byte* data, size_t chunk_index) {
  uint64_t chunk = 0;
  std::memcpy(&chunk, data + chunk_index * kChunkBytes, sizeof(chunk));
  return chunk;
}

void AppendBytes(const std::byte* data, size_t size, Bytes* out) {
  out->insert(out->end(), data, data + size);
}

// Appends an 8-byte RLE block header packing (same_chunks, diff_chunks).
void AppendBlockHeader(uint32_t same_chunks, uint32_t diff_chunks, Bytes* out) {
  const uint32_t parts[2] = {same_chunks, diff_chunks};
  AppendBytes(reinterpret_cast<const std::byte*>(parts), sizeof(parts), out);
}

// Unpacks an 8-byte RLE block header into `*same_chunks` and `*diff_chunks`.
void UnpackBlockHeader(const std::byte* data, uint32_t* same_chunks,
                       uint32_t* diff_chunks) {
  uint32_t parts[2] = {};
  std::memcpy(parts, data, sizeof(parts));
  *same_chunks = parts[0];
  *diff_chunks = parts[1];
}

// Appends RLE delta blocks encoding `curr` against `prev` (which may be null
// when a window is new in this frame) onto `out` and returns the number of
// payload bytes appended.
uint32_t CompressDelta(const Bytes* prev, const Bytes& curr, Bytes* out) {
  const size_t start_out = out->size();
  const size_t prev_chunks = prev != nullptr ? prev->size() / kChunkBytes : 0;
  const size_t curr_chunks = curr.size() / kChunkBytes;
  const size_t common_chunks = std::min(prev_chunks, curr_chunks);
  size_t n = 0;

  // Walk the overlapping prefix of `prev` and `curr`, emitting alternating
  // runs of identical and differing 8-byte chunks.
  while (n < common_chunks) {
    const size_t same_start = n;
    while (n < common_chunks &&
           ReadChunk(prev->data(), n) == ReadChunk(curr.data(), n)) {
      ++n;
    }
    const uint32_t same_count = static_cast<uint32_t>(n - same_start);

    const size_t diff_start = n;
    while (n < common_chunks &&
           ReadChunk(prev->data(), n) != ReadChunk(curr.data(), n)) {
      ++n;
    }
    // If the diff run reaches the end of `prev` and `curr` is longer, fold the
    // remaining tail of `curr` into the same diff run instead of emitting an
    // extra block header.
    if (n == common_chunks && common_chunks < curr_chunks) {
      n = curr_chunks;
    }
    const uint32_t diff_count = static_cast<uint32_t>(n - diff_start);

    AppendBlockHeader(same_count, diff_count, out);
    if (diff_count > 0) {
      AppendBytes(curr.data() + diff_start * kChunkBytes,
                  static_cast<size_t>(diff_count) * kChunkBytes, out);
    }
  }

  // If `curr` is longer than `prev` and the entire overlapping prefix matched,
  // emit one final diff block carrying the remaining tail bytes.
  if (n < curr_chunks) {
    const uint32_t tail_count = static_cast<uint32_t>(curr_chunks - n);
    AppendBlockHeader(0, tail_count, out);
    AppendBytes(curr.data() + n * kChunkBytes,
                static_cast<size_t>(tail_count) * kChunkBytes, out);
  }

  return static_cast<uint32_t>(out->size() - start_out);
}

// Reconstructs `target_bytes` of serialized list data into `out` by applying
// the RLE delta stream in `payload` against `prev`. Returns false if any block
// header or offset is out of bounds or fails to consume the exact payload.
bool DecompressDelta(const Bytes* prev, BytesView payload, size_t target_bytes,
                     Bytes* out) {
  if ((payload.size() % kChunkBytes) != 0 ||
      (target_bytes % kChunkBytes) != 0) {
    return false;
  }
  const size_t prev_chunks = prev != nullptr ? prev->size() / kChunkBytes : 0;
  const size_t payload_chunks = payload.size() / kChunkBytes;
  const size_t target_chunks = target_bytes / kChunkBytes;
  out->assign(target_bytes, std::byte{0});

  size_t out_chunk = 0;
  size_t in_chunk = 0;
  while (out_chunk < target_chunks) {
    // Read the next RLE block header (same_count, diff_count) and reject
    // empty `(0, 0)` blocks that would otherwise loop without making progress.
    if (in_chunk >= payload_chunks) return false;
    uint32_t same_count = 0;
    uint32_t diff_count = 0;
    UnpackBlockHeader(payload.data() + in_chunk * kChunkBytes, &same_count,
                      &diff_count);
    ++in_chunk;
    if (same_count == 0 && diff_count == 0) return false;

    // Validate that the block fits within the target buffer, the reference
    // list `prev`, and the remaining compressed payload.
    const uint64_t total_step =
        static_cast<uint64_t>(same_count) + static_cast<uint64_t>(diff_count);
    if (total_step > target_chunks - out_chunk) return false;
    if (same_count > 0 &&
        static_cast<uint64_t>(out_chunk) + same_count > prev_chunks) {
      return false;
    }
    if (diff_count > payload_chunks - in_chunk) return false;

    // Copy unchanged chunks from `prev`, then copy modified chunks from the
    // wire payload.
    if (same_count > 0) {
      std::memcpy(out->data() + out_chunk * kChunkBytes,
                  prev->data() + out_chunk * kChunkBytes,
                  static_cast<size_t>(same_count) * kChunkBytes);
      out_chunk += same_count;
    }
    if (diff_count > 0) {
      std::memcpy(out->data() + out_chunk * kChunkBytes,
                  payload.data() + in_chunk * kChunkBytes,
                  static_cast<size_t>(diff_count) * kChunkBytes);
      out_chunk += diff_count;
      in_chunk += diff_count;
    }
  }
  return in_chunk == payload_chunks;
}

// Unpacks a serialized draw-list buffer (`src.bytes`) into vertex, index, and
// draw-command vectors in `dst`, returning false if the header counts do not
// match the buffer's byte length, a float is not finite, or a draw command
// would read outside the list's indices or vertices.
bool UnpackSerializedList(const SerializedList& src, DrawList* dst) {
  if (src.bytes.size() < kListHeaderBytes) return false;
  SerializedListHeader prefix;
  std::memcpy(&prefix, src.bytes.data(), sizeof(prefix));
  // The 4 padding bytes between SerializedListHeader (20B) and kListHeaderBytes
  // (24B) must be zero.
  uint32_t header_padding = 0;
  std::memcpy(&header_padding, src.bytes.data() + sizeof(prefix),
              sizeof(header_padding));
  if (header_padding != 0) return false;
  if (!AllFinite(prefix.ref_pos, 2)) return false;

  // Compute the 8-byte-aligned section sizes and verify that their sum matches
  // the serialized buffer size.
  const uint64_t vert_bytes =
      static_cast<uint64_t>(prefix.vert_count) * sizeof(Vert);
  const uint64_t idx_bytes =
      static_cast<uint64_t>(prefix.idx_count) * sizeof(uint16_t);
  const uint64_t cmd_bytes =
      static_cast<uint64_t>(prefix.command_count) * kWireDrawCommandBytes;
  const uint64_t vert_aligned = AlignUp8(vert_bytes);
  const uint64_t idx_aligned = AlignUp8(idx_bytes);
  const uint64_t cmd_aligned = AlignUp8(cmd_bytes);
  if (kListHeaderBytes + vert_aligned + idx_aligned + cmd_aligned !=
      src.bytes.size()) {
    return false;
  }

  // Copy vertices, indices, and draw commands out of their aligned sections.
  dst->group_id = src.group_id;
  dst->ref_pos[0] = prefix.ref_pos[0];
  dst->ref_pos[1] = prefix.ref_pos[1];
  dst->verts.resize(prefix.vert_count);
  dst->indices.resize(prefix.idx_count);
  dst->commands.resize(prefix.command_count);

  const std::byte* cursor = src.bytes.data() + kListHeaderBytes;
  if (vert_bytes > 0) {
    std::memcpy(dst->verts.data(), cursor, static_cast<size_t>(vert_bytes));
  }
  cursor += vert_aligned;
  if (idx_bytes > 0) {
    std::memcpy(dst->indices.data(), cursor, static_cast<size_t>(idx_bytes));
  }
  cursor += idx_aligned;
  for (uint32_t i = 0; i < prefix.command_count; ++i) {
    WireDrawCommand wire_cmd = {};
    const size_t offset = static_cast<size_t>(i) * kWireDrawCommandBytes;
    std::memcpy(&wire_cmd, cursor + offset, kWireDrawCommandBytes);

    // Reject commands a renderer could not draw safely: each one draws whole
    // triangles from its own range of indices, and every index it reads,
    // offset by `vtx_offset`, must name a vertex of this list.
    if (!AllFinite(wire_cmd.clip_rect, 4) || wire_cmd.elem_count % 3 != 0 ||
        static_cast<uint64_t>(wire_cmd.idx_offset) + wire_cmd.elem_count >
            prefix.idx_count) {
      return false;
    }
    if (wire_cmd.elem_count > 0) {
      const auto first = dst->indices.begin() + wire_cmd.idx_offset;
      const uint16_t max_index =
          *std::max_element(first, first + wire_cmd.elem_count);
      if (static_cast<uint64_t>(wire_cmd.vtx_offset) + max_index >=
          prefix.vert_count) {
        return false;
      }
    }

    DrawCommand& cmd = dst->commands[i];
    cmd.tex_id = wire_cmd.tex_id;
    cmd.vtx_offset = wire_cmd.vtx_offset;
    cmd.idx_offset = wire_cmd.idx_offset;
    cmd.elem_count = wire_cmd.elem_count;
    std::memcpy(cmd.clip_rect, wire_cmd.clip_rect, sizeof(cmd.clip_rect));
  }
  return true;
}

bool HasValidInputFields(const Input& input) {
  constexpr uint16_t kKnownFlags =
      Input::kRequestKeyframe | Input::kMacOsBehaviors;
  return (input.header.flags & ~kKnownFlags) == 0 &&
         input.char_count <= Input::kMaxChars &&
         AllFinite(&input.framebuffer_scale, 1) &&
         AllFinite(input.mouse_pos, 2) && AllFinite(input.mouse_wheel, 2);
}

bool HasValidTexturePayload(const Texture& hdr, size_t payload_bytes) {
  switch (hdr.action) {
    case Texture::Action::kDestroy:
      return payload_bytes == 0;
    case Texture::Action::kCreate:
    case Texture::Action::kUpdate:
      if (hdr.width == 0 || hdr.height == 0) return false;
      switch (hdr.format) {
        case Texture::Format::kAlpha8:
        case Texture::Format::kRgba8: {
          // Computed in 64 bits: size_t is 32 bits in WebAssembly.
          const uint64_t bytes_per_pixel =
              hdr.format == Texture::Format::kAlpha8 ? 1 : 4;
          return static_cast<uint64_t>(hdr.width) * hdr.height *
                     bytes_per_pixel ==
                 payload_bytes;
        }
        case Texture::Format::kCustom:
          return payload_bytes > 0;
        default:
          return false;
      }
    default:
      return false;
  }
}

}  // namespace

void SerializeDrawData(const ImDrawData* draw_data, int mouse_cursor,
                       SerializedFrame* out) {
  // 1. Reset `out` and record frame-level cursor and display geometry.
  out->Clear();
  out->mouse_cursor = static_cast<int32_t>(mouse_cursor);
  if (draw_data == nullptr) return;

  out->display_pos[0] = draw_data->DisplayPos.x;
  out->display_pos[1] = draw_data->DisplayPos.y;
  out->display_size[0] = draw_data->DisplaySize.x;
  out->display_size[1] = draw_data->DisplaySize.y;

  std::vector<uint64_t> used_group_ids;
  used_group_ids.reserve(static_cast<size_t>(draw_data->CmdLists.Size));
  out->lists.reserve(static_cast<size_t>(draw_data->CmdLists.Size));

  // 2. Serialize each Dear ImGui draw list into its own 8-byte-aligned buffer.
  for (int n = 0; n < draw_data->CmdLists.Size; ++n) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    if (cmd_list == nullptr) continue;

    // Count draw commands without user callbacks (callbacks hold host function
    // pointers that cannot be sent over the wire).
    uint32_t command_count = 0;
    for (int c = 0; c < cmd_list->CmdBuffer.Size; ++c) {
      if (cmd_list->CmdBuffer[c].UserCallback == nullptr) {
        ++command_count;
      }
    }

    // Derive a deterministic, unique `group_id` from the owning window's name
    // so the encoder and decoder can pair this draw list with its counterpart
    // in the previous frame during RLE delta compression. This relies on
    // ImDrawList::_OwnerName, which imgui.h documents as an internal debugging
    // field. If a Dear ImGui update ever stops setting it, every list hashes
    // the same and the loop below numbers them in draw order, so lists are
    // still matched (by position) and only compression suffers.
    uint64_t group_id = HashOwnerName(cmd_list->_OwnerName);
    while (std::find(used_group_ids.begin(), used_group_ids.end(), group_id) !=
           used_group_ids.end()) {
      ++group_id;
      if (group_id == 0) group_id = 1;
    }
    used_group_ids.push_back(group_id);

    // Choose the first vertex position as the local origin (`ref_pos`) and
    // size the 8-byte-aligned sections for the list header, vertices, indices,
    // and draw commands.
    const uint32_t vert_count =
        static_cast<uint32_t>(cmd_list->VtxBuffer.Size);
    const uint32_t idx_count = static_cast<uint32_t>(cmd_list->IdxBuffer.Size);
    const float ref_x = vert_count > 0 ? cmd_list->VtxBuffer[0].pos.x : 0.0f;
    const float ref_y = vert_count > 0 ? cmd_list->VtxBuffer[0].pos.y : 0.0f;

    const size_t vert_aligned =
        static_cast<size_t>(AlignUp8(uint64_t{vert_count} * sizeof(Vert)));
    const size_t idx_aligned =
        static_cast<size_t>(AlignUp8(uint64_t{idx_count} * sizeof(uint16_t)));
    const size_t cmd_aligned = static_cast<size_t>(
        AlignUp8(uint64_t{command_count} * kWireDrawCommandBytes));

    SerializedList serialized;
    serialized.group_id = group_id;
    serialized.bytes.assign(
        kListHeaderBytes + vert_aligned + idx_aligned + cmd_aligned,
        std::byte{0});

    // Write the 20-byte list header at the start of the buffer (leaving the
    // trailing 4 alignment bytes zeroed).
    SerializedListHeader prefix;
    prefix.ref_pos[0] = ref_x;
    prefix.ref_pos[1] = ref_y;
    prefix.vert_count = vert_count;
    prefix.idx_count = idx_count;
    prefix.command_count = command_count;
    std::memcpy(serialized.bytes.data(), &prefix, sizeof(prefix));

    // Quantize vertex positions (relative to `ref_pos` so dragging a window
    // keeps relative vertex coordinates unchanged) and UVs into 16-bit ints.
    std::byte* vert_dst = serialized.bytes.data() + kListHeaderBytes;
    for (uint32_t i = 0; i < vert_count; ++i) {
      const ImDrawVert& src_v = cmd_list->VtxBuffer[static_cast<int>(i)];
      Vert vert;
      vert.color = src_v.col;
      vert.pos[0] = QuantizePos(src_v.pos.x - ref_x);
      vert.pos[1] = QuantizePos(src_v.pos.y - ref_y);
      vert.uv[0] = QuantizeNormalized(src_v.uv.x, Vert::kUvMin, Vert::kUvMax);
      vert.uv[1] = QuantizeNormalized(src_v.uv.y, Vert::kUvMin, Vert::kUvMax);
      std::memcpy(vert_dst + static_cast<size_t>(i) * sizeof(Vert), &vert,
                  sizeof(Vert));
    }

    // Copy 16-bit triangle indices into the next 8-byte-aligned section.
    std::byte* idx_dst = vert_dst + vert_aligned;
    if (idx_count > 0) {
      if constexpr (sizeof(ImDrawIdx) == sizeof(uint16_t)) {
        std::memcpy(idx_dst, cmd_list->IdxBuffer.Data,
                    static_cast<size_t>(idx_count) * sizeof(uint16_t));
      } else {
        for (uint32_t i = 0; i < idx_count; ++i) {
          const auto index =
              static_cast<uint16_t>(cmd_list->IdxBuffer[static_cast<int>(i)]);
          std::memcpy(idx_dst + static_cast<size_t>(i) * sizeof(uint16_t),
                      &index, sizeof(index));
        }
      }
    }

    // Pack the non-callback draw commands into the final 8-byte-aligned
    // section.
    std::byte* cmd_dst = idx_dst + idx_aligned;
    uint32_t out_c = 0;
    for (int c = 0; c < cmd_list->CmdBuffer.Size; ++c) {
      const ImDrawCmd& src_c = cmd_list->CmdBuffer[c];
      if (src_c.UserCallback != nullptr) continue;
      WireDrawCommand wire_cmd = {};
      wire_cmd.tex_id = static_cast<uint64_t>(src_c.GetTexID());
      wire_cmd.vtx_offset = src_c.VtxOffset;
      wire_cmd.idx_offset = src_c.IdxOffset;
      wire_cmd.elem_count = src_c.ElemCount;
      wire_cmd.clip_rect[0] = src_c.ClipRect.x;
      wire_cmd.clip_rect[1] = src_c.ClipRect.y;
      wire_cmd.clip_rect[2] = src_c.ClipRect.z;
      wire_cmd.clip_rect[3] = src_c.ClipRect.w;
      std::memcpy(
          cmd_dst + static_cast<size_t>(out_c++) * kWireDrawCommandBytes,
          &wire_cmd, kWireDrawCommandBytes);
    }

    out->lists.push_back(std::move(serialized));
  }
}

Bytes EncodeDrawFrame(const SerializedFrame& frame, uint64_t frame_index,
                      const SerializedFrame* prev_frame) {
  const bool compress = (prev_frame != nullptr);

  // 1. Give up on frames the decoder would reject for their unpacked size
  //    (which also keeps every list size below 2^32), and reserve space for the
  //    outer frame header and per-list headers/payloads.
  uint64_t unpacked_bytes_total = sizeof(DrawFrame);
  size_t reserve_bytes =
      sizeof(DrawFrame) + frame.lists.size() * sizeof(WireListHeader);
  for (const SerializedList& list : frame.lists) {
    unpacked_bytes_total += list.bytes.size();
    reserve_bytes += compress ? 2 * kChunkBytes : list.bytes.size();
  }
  if (unpacked_bytes_total > kMaxMessageBytes) return {};
  Bytes out;
  out.reserve(reserve_bytes);
  out.resize(sizeof(DrawFrame), std::byte{0});

  // 2. Append each draw list preceded by its 16-byte WireListHeader, either
  //    RLE-delta-compressed against the matching list in `prev_frame` or raw.
  for (size_t i = 0; i < frame.lists.size(); ++i) {
    const SerializedList& list = frame.lists[i];
    const size_t list_hdr_offset = out.size();
    out.resize(list_hdr_offset + sizeof(WireListHeader), std::byte{0});

    uint32_t payload_bytes = 0;
    if (compress) {
      const Bytes* prev_bytes =
          FindMatchingListBytes(prev_frame, list.group_id, i);
      payload_bytes = CompressDelta(prev_bytes, list.bytes, &out);
    } else {
      payload_bytes = static_cast<uint32_t>(list.bytes.size());
      AppendBytes(list.bytes.data(), list.bytes.size(), &out);
    }

    WireListHeader list_hdr;
    list_hdr.group_id = list.group_id;
    list_hdr.unpacked_bytes = static_cast<uint32_t>(list.bytes.size());
    list_hdr.payload_bytes = payload_bytes;
    std::memcpy(out.data() + list_hdr_offset, &list_hdr, sizeof(list_hdr));
  }

  // RLE block headers can make a compressed frame larger than its lists.
  if (out.size() > kMaxMessageBytes) return {};

  // 3. Populate the outer DrawFrame header with the final byte size and
  //    metadata.
  DrawFrame hdr;
  hdr.header.size = static_cast<uint32_t>(out.size());
  hdr.header.flags = compress ? DrawFrame::kCompressed : 0;
  hdr.header.kind = MessageKind::kDrawFrame;
  hdr.header.version = kCodecVersion;
  hdr.frame_index = frame_index;
  hdr.mouse_cursor = frame.mouse_cursor;
  hdr.list_count = static_cast<uint32_t>(frame.lists.size());
  hdr.display_pos[0] = frame.display_pos[0];
  hdr.display_pos[1] = frame.display_pos[1];
  hdr.display_size[0] = frame.display_size[0];
  hdr.display_size[1] = frame.display_size[1];
  std::memcpy(out.data(), &hdr, sizeof(hdr));

  return out;
}

bool DecodeDrawFrame(BytesView message, SerializedFrame* prev_frame,
                     uint64_t* prev_frame_index, DecodedFrame* out,
                     bool* request_keyframe) {
  // 1. Validate output pointers and overall message size/alignment.
  if (request_keyframe != nullptr) *request_keyframe = false;
  if (prev_frame == nullptr || prev_frame_index == nullptr || out == nullptr) {
    return false;
  }
  if (message.size() < sizeof(DrawFrame) ||
      message.size() > kMaxMessageBytes ||
      (message.size() % kChunkBytes) != 0) {
    return false;
  }

  // 2. Parse and validate the outer DrawFrame header and sequence continuity.
  DrawFrame hdr;
  std::memcpy(&hdr, message.data(), sizeof(hdr));
  if (hdr.header.kind != MessageKind::kDrawFrame ||
      hdr.header.version != kCodecVersion ||
      hdr.header.size != message.size() ||
      (hdr.header.flags & ~DrawFrame::kCompressed) != 0) {
    return false;
  }

  const bool compressed = (hdr.header.flags & DrawFrame::kCompressed) != 0;
  if (compressed) {
    if (*prev_frame_index == 0 || hdr.frame_index != *prev_frame_index + 1) {
      if (request_keyframe != nullptr) *request_keyframe = true;
      return false;
    }
  }

  // Every list needs at least its WireListHeader, so a larger count cannot be
  // valid, and checking it here keeps a tiny message from making the decoder
  // allocate per-list storage for millions of lists.
  if (hdr.list_count >
          (message.size() - sizeof(DrawFrame)) / sizeof(WireListHeader) ||
      !AllFinite(hdr.display_pos, 2) || !AllFinite(hdr.display_size, 2)) {
    if (compressed && request_keyframe != nullptr) *request_keyframe = true;
    return false;
  }

  // 3. Decode into temporary `next_serialized` and `next_out` structures so a
  //    malformed or truncated frame leaves `*prev_frame` and `*out` untouched.
  SerializedFrame next_serialized;
  next_serialized.mouse_cursor = hdr.mouse_cursor;
  next_serialized.display_pos[0] = hdr.display_pos[0];
  next_serialized.display_pos[1] = hdr.display_pos[1];
  next_serialized.display_size[0] = hdr.display_size[0];
  next_serialized.display_size[1] = hdr.display_size[1];
  next_serialized.lists.reserve(hdr.list_count);

  DecodedFrame next_out;
  next_out.mouse_cursor = hdr.mouse_cursor;
  next_out.display_pos[0] = hdr.display_pos[0];
  next_out.display_pos[1] = hdr.display_pos[1];
  next_out.display_size[0] = hdr.display_size[0];
  next_out.display_size[1] = hdr.display_size[1];
  next_out.lists.resize(hdr.list_count);

  // 4. Reconstruct and unpack each draw list, enforcing a cumulative unpacked
  //    byte cap (`kMaxMessageBytes`) across all lists in the frame.
  size_t byte_pos = sizeof(DrawFrame);
  uint64_t unpacked_bytes_total = sizeof(DrawFrame);
  for (uint32_t i = 0; i < hdr.list_count; ++i) {
    if (message.size() - byte_pos < sizeof(WireListHeader)) {
      if (compressed && request_keyframe != nullptr) *request_keyframe = true;
      return false;
    }
    WireListHeader list_hdr;
    std::memcpy(&list_hdr, message.data() + byte_pos, sizeof(list_hdr));
    byte_pos += sizeof(WireListHeader);

    if (list_hdr.unpacked_bytes < kListHeaderBytes ||
        (list_hdr.unpacked_bytes % kChunkBytes) != 0 ||
        (list_hdr.payload_bytes % kChunkBytes) != 0 ||
        unpacked_bytes_total + list_hdr.unpacked_bytes > kMaxMessageBytes ||
        list_hdr.payload_bytes > message.size() - byte_pos) {
      if (compressed && request_keyframe != nullptr) *request_keyframe = true;
      return false;
    }
    unpacked_bytes_total += list_hdr.unpacked_bytes;

    SerializedList serialized_list;
    serialized_list.group_id = list_hdr.group_id;
    const BytesView payload =
        message.subspan(byte_pos, list_hdr.payload_bytes);
    if (compressed) {
      const Bytes* prev_bytes =
          FindMatchingListBytes(prev_frame, list_hdr.group_id, i);
      if (!DecompressDelta(prev_bytes, payload, list_hdr.unpacked_bytes,
                           &serialized_list.bytes)) {
        if (request_keyframe != nullptr) *request_keyframe = true;
        return false;
      }
    } else {
      if (list_hdr.payload_bytes != list_hdr.unpacked_bytes) return false;
      serialized_list.bytes.assign(payload.begin(), payload.end());
    }
    byte_pos += list_hdr.payload_bytes;

    if (!UnpackSerializedList(serialized_list, &next_out.lists[i])) {
      if (compressed && request_keyframe != nullptr) *request_keyframe = true;
      return false;
    }
    next_serialized.lists.push_back(std::move(serialized_list));
  }

  // 5. Reject messages with unconsumed trailing bytes, then commit the decoded
  //    frame and updated reference state.
  if (byte_pos != message.size()) {
    if (compressed && request_keyframe != nullptr) *request_keyframe = true;
    return false;
  }

  *prev_frame = std::move(next_serialized);
  *prev_frame_index = hdr.frame_index;
  *out = std::move(next_out);
  return true;
}

Bytes EncodeInput(const Input& input) {
  if (!HasValidInputFields(input)) return {};
  Input wire = input;
  wire.header.size = sizeof(Input);
  wire.header.kind = MessageKind::kInput;
  wire.header.version = kCodecVersion;
  Bytes out(sizeof(wire));
  std::memcpy(out.data(), &wire, sizeof(wire));
  return out;
}

bool DecodeInput(BytesView message, Input* out) {
  if (out == nullptr || message.size() != sizeof(Input)) return false;
  Input input;
  std::memcpy(&input, message.data(), sizeof(input));
  if (input.header.kind != MessageKind::kInput ||
      input.header.version != kCodecVersion ||
      input.header.size != sizeof(Input) || !HasValidInputFields(input)) {
    return false;
  }
  *out = input;
  return true;
}

Bytes EncodeTexture(const Texture& texture, BytesView payload) {
  if (payload.size() > kMaxMessageBytes - sizeof(Texture) ||
      texture.header.flags != 0 ||
      !HasValidTexturePayload(texture, payload.size())) {
    return {};
  }
  Texture hdr = texture;
  hdr.header.size = static_cast<uint32_t>(sizeof(Texture) + payload.size());
  hdr.header.flags = 0;
  hdr.header.kind = MessageKind::kTexture;
  hdr.header.version = kCodecVersion;
  Bytes out(sizeof(hdr) + payload.size());
  std::memcpy(out.data(), &hdr, sizeof(hdr));
  if (!payload.empty()) {
    std::memcpy(out.data() + sizeof(hdr), payload.data(), payload.size());
  }
  return out;
}

bool DecodeTexture(BytesView message, Texture* out, BytesView* out_payload) {
  if (out == nullptr || out_payload == nullptr ||
      message.size() < sizeof(Texture) || message.size() > kMaxMessageBytes) {
    return false;
  }
  Texture hdr;
  std::memcpy(&hdr, message.data(), sizeof(hdr));
  if (hdr.header.kind != MessageKind::kTexture ||
      hdr.header.version != kCodecVersion ||
      hdr.header.size != message.size() || hdr.header.flags != 0) {
    return false;
  }
  const BytesView payload = message.subspan(sizeof(Texture));
  if (!HasValidTexturePayload(hdr, payload.size())) return false;
  *out = hdr;
  *out_payload = payload;
  return true;
}

Bytes EncodeClipboard(std::string_view utf8, bool open_url) {
  constexpr size_t kMaxPayload = kMaxMessageBytes - sizeof(Clipboard);
  if (utf8.size() > kMaxPayload) {
    size_t end = kMaxPayload;
    while (end > 0 && (static_cast<unsigned char>(utf8[end]) & 0xC0) == 0x80) {
      --end;
    }
    utf8 = utf8.substr(0, end);
  }
  Clipboard hdr;
  hdr.header.size = static_cast<uint32_t>(sizeof(Clipboard) + utf8.size());
  hdr.header.flags = open_url ? Clipboard::kOpenUrl : 0;
  Bytes out(sizeof(hdr) + utf8.size());
  std::memcpy(out.data(), &hdr, sizeof(hdr));
  if (!utf8.empty()) {
    std::memcpy(out.data() + sizeof(hdr), utf8.data(), utf8.size());
  }
  return out;
}

bool DecodeClipboard(BytesView message, std::string* out_utf8, bool* open_url) {
  if (out_utf8 == nullptr || message.size() < sizeof(Clipboard) ||
      message.size() > kMaxMessageBytes) {
    return false;
  }
  Clipboard hdr;
  std::memcpy(&hdr, message.data(), sizeof(hdr));
  if (hdr.header.kind != MessageKind::kClipboard ||
      hdr.header.version != kCodecVersion ||
      hdr.header.size != message.size() ||
      (hdr.header.flags & ~Clipboard::kOpenUrl) != 0) {
    return false;
  }
  const auto* payload =
      reinterpret_cast<const char*>(message.data() + sizeof(hdr));
  out_utf8->assign(payload, message.size() - sizeof(hdr));
  if (open_url != nullptr) {
    *open_url = (hdr.header.flags & Clipboard::kOpenUrl) != 0;
  }
  return true;
}

bool IsOpenableUrl(std::string_view url) {
  const size_t colon = url.find(':');
  if (colon == std::string_view::npos ||
      url.substr(colon).substr(0, 3) != "://") {
    return false;
  }
  std::string scheme(url.substr(0, colon));
  std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return scheme == "http" || scheme == "https";
}

}  // namespace mujoco::studio::netui
