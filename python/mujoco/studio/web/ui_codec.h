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

// Binary wire protocol and run-length encoded (RLE) delta-compressed
// draw-frame codec for streaming a headless Dear ImGui UI over the Studio web
// viewer's /ui WebSocket.
//
// Every binary WebSocket frame carries one message starting with a
// fixed-layout little-endian MessageHeader. Wire structs contain no pointers
// or ABI-dependent padding, so 32-bit WebAssembly browsers and 64-bit Python
// hosts share the exact same byte layout.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_UI_CODEC_H_
#define MUJOCO_PYTHON_STUDIO_WEB_UI_CODEC_H_

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>

namespace mujoco::studio::netui {

using Bytes = std::vector<std::byte>;
using BytesView = std::span<const std::byte>;

inline constexpr uint8_t kCodecVersion = 1;

// Upper bound (16 MiB) on any single wire message accepted by decoders.
inline constexpr size_t kMaxMessageBytes = 16u * 1024u * 1024u;

enum class MessageKind : uint8_t {
  kInvalid = 0,
  kHello = 1,
  kInput = 2,
  kTexture = 3,
  kDrawFrame = 4,
  kClipboard = 5,
};

struct MessageHeader {
  uint32_t size = 0;
  uint16_t flags = 0;
  MessageKind kind = MessageKind::kInvalid;
  uint8_t version = kCodecVersion;
};

// Exchanged first in both directions when opening a /ui session.
struct Hello {
  MessageHeader header{sizeof(Hello), 0, MessageKind::kHello, kCodecVersion};
};

// Sent by the browser every frame to report viewport size, DPI scale, mouse,
// keyboard, and Unicode character input.
struct Input {
  // MessageHeader::flags bits.
  static constexpr uint16_t kRequestKeyframe = 1u << 0;
  static constexpr uint16_t kMacOsBehaviors = 1u << 1;

  // Fixed capacities sized above Dear ImGui's named key count so the wire ABI
  // stays stable across Dear ImGui updates.
  static constexpr size_t kMaxKeys = 256;
  static constexpr size_t kMaxChars = 64;

  MessageHeader header{sizeof(Input), 0, MessageKind::kInput, kCodecVersion};
  uint64_t keys_down[kMaxKeys / 64] = {};
  uint32_t chars[kMaxChars] = {};
  float framebuffer_scale = 1.0f;
  float mouse_pos[2] = {-FLT_MAX, -FLT_MAX};
  float mouse_wheel[2] = {0.0f, 0.0f};  // [0] = vertical, [1] = horizontal.
  uint32_t char_count = 0;
  uint32_t mouse_down_mask = 0;
  uint16_t screen_size[2] = {};

  bool IsKeyDown(int key_offset) const {
    if (key_offset < 0 || key_offset >= static_cast<int>(kMaxKeys)) {
      return false;
    }
    return (keys_down[key_offset / 64] &
            (uint64_t{1} << (key_offset % 64))) != 0;
  }

  void SetKeyDown(int key_offset, bool down = true) {
    if (key_offset < 0 || key_offset >= static_cast<int>(kMaxKeys)) {
      return;
    }
    const uint64_t bit = uint64_t{1} << (key_offset % 64);
    if (down) {
      keys_down[key_offset / 64] |= bit;
    } else {
      keys_down[key_offset / 64] &= ~bit;
    }
  }

  bool IsImGuiKeyDown(ImGuiKey key) const {
    return IsKeyDown(static_cast<int>(key) - ImGuiKey_NamedKey_BEGIN);
  }

  void SetImGuiKeyDown(ImGuiKey key, bool down = true) {
    SetKeyDown(static_cast<int>(key) - ImGuiKey_NamedKey_BEGIN, down);
  }
};

// Followed immediately on the wire by `header.size - sizeof(Texture)` bytes of
// payload: `width * height * bytes_per_pixel` raw pixels for kAlpha8 and
// kRgba8, encoded image bytes (such as QOI or WebP) for kCustom, and 0 bytes
// when `action == Action::kDestroy`.
struct Texture {
  enum class Action : uint32_t {
    kCreate = 0,
    kUpdate = 1,
    kDestroy = 2,
  };

  enum class Format : uint32_t {
    kAlpha8 = 0,
    kRgba8 = 1,
    kCustom = 2,
  };

  MessageHeader header{sizeof(Texture), 0, MessageKind::kTexture,
                       kCodecVersion};
  uint64_t tex_id = 0;
  Action action = Action::kCreate;
  Format format = Format::kRgba8;
  uint16_t x = 0;
  uint16_t y = 0;
  uint16_t width = 0;
  uint16_t height = 0;
};

// Followed immediately on the wire by `header.size - sizeof(Clipboard)` bytes
// of UTF-8 text.
struct Clipboard {
  // MessageHeader::flags bits.
  //
  // The text is a URL for the browser to open in a new tab rather than text to
  // copy. The browser must open only URLs that pass IsOpenableUrl(): opening a
  // `javascript:` URL would run script in the page.
  static constexpr uint16_t kOpenUrl = 1u << 0;

  MessageHeader header{sizeof(Clipboard), 0, MessageKind::kClipboard,
                       kCodecVersion};
};

// Fixed prefix of every kDrawFrame message on the wire.
struct DrawFrame {
  // MessageHeader::flags bits.
  static constexpr uint16_t kCompressed = 1u << 0;

  MessageHeader header{sizeof(DrawFrame), 0, MessageKind::kDrawFrame,
                       kCodecVersion};
  uint64_t frame_index = 0;
  int32_t mouse_cursor = 0;
  uint32_t list_count = 0;
  float display_pos[2] = {};
  float display_size[2] = {};
};

// Quantized vertex: position relative to the draw list's `ref_pos` in exact
// 0.25 px steps over [-8192, +8191.75] (so an offset of 0 round-trips exactly)
// and UV over [0, 1] (1/65535 precision).
struct Vert {
  static constexpr float kPosMin = -8192.0f;
  static constexpr float kPosStep = 0.25f;
  static constexpr float kUvMin = 0.0f;
  static constexpr float kUvMax = 1.0f;

  uint32_t color = 0;
  uint16_t pos[2] = {};
  uint16_t uv[2] = {};

  ImVec2 UnpackPos(const float ref_pos[2]) const {
    return ImVec2(static_cast<float>(pos[0]) * kPosStep + kPosMin + ref_pos[0],
                  static_cast<float>(pos[1]) * kPosStep + kPosMin + ref_pos[1]);
  }

  ImVec2 UnpackUv() const {
    constexpr float kScale = (kUvMax - kUvMin) / 65535.0f;
    return ImVec2(static_cast<float>(uv[0]) * kScale + kUvMin,
                  static_cast<float>(uv[1]) * kScale + kUvMin);
  }
};

struct DrawCommand {
  uint64_t tex_id = 0;
  uint32_t vtx_offset = 0;
  uint32_t idx_offset = 0;
  uint32_t elem_count = 0;
  float clip_rect[4] = {};
};

struct DrawList {
  uint64_t group_id = 0;
  float ref_pos[2] = {};
  std::vector<Vert> verts;
  std::vector<uint16_t> indices;
  std::vector<DrawCommand> commands;
};

struct DecodedFrame {
  int32_t mouse_cursor = 0;
  float display_pos[2] = {};
  float display_size[2] = {};
  std::vector<DrawList> lists;
};

// Canonical byte representation of a single window's draw list (padded to an
// 8-byte boundary), retained across frames on both encoder and decoder as the
// reference for per-window RLE delta compression.
struct SerializedList {
  uint64_t group_id = 0;
  Bytes bytes;
};

struct SerializedFrame {
  int32_t mouse_cursor = 0;
  float display_pos[2] = {};
  float display_size[2] = {};
  std::vector<SerializedList> lists;

  void Clear() {
    mouse_cursor = 0;
    display_pos[0] = 0.0f;
    display_pos[1] = 0.0f;
    display_size[0] = 0.0f;
    display_size[1] = 0.0f;
    lists.clear();
  }
};

// Converts Dear ImGui's rendered draw lists into `out` (one 8-byte-aligned
// byte buffer per window). Safe to call when `draw_data` is null or empty.
void SerializeDrawData(const ImDrawData* draw_data, int mouse_cursor,
                       SerializedFrame* out);

// Encodes `frame` into a single kDrawFrame wire message with `frame_index`.
// When `prev_frame` is non-null, RLE-delta-compresses each draw list against
// the matching `group_id` list in `*prev_frame` and sets
// `DrawFrame::kCompressed`. Returns an empty Bytes if the message, or the
// frame's unpacked draw lists, would exceed kMaxMessageBytes, since the
// decoder would reject it.
Bytes EncodeDrawFrame(const SerializedFrame& frame, uint64_t frame_index,
                      const SerializedFrame* prev_frame = nullptr);

// Decodes a kDrawFrame wire message into `*out`, updating `*prev_frame` and
// `*prev_frame_index` on success. When `prev_frame_index` holds 0, no previous
// reference frame is considered present. Sets `*request_keyframe = true` when
// a compressed frame is missing its predecessor (`*prev_frame_index + 1`) or
// fails decompression. A decoded frame is safe to render as is: every float is
// finite and every draw command's indices, offset by its `vtx_offset`, lie
// within its draw list's vertices.
bool DecodeDrawFrame(BytesView message, SerializedFrame* prev_frame,
                     uint64_t* prev_frame_index, DecodedFrame* out,
                     bool* request_keyframe);

// Encodes `input` into a kInput wire message. Returns an empty Bytes if
// `input` has unknown flags, too many characters, or a non-finite float.
Bytes EncodeInput(const Input& input);

// Decodes a kInput wire message into `*out`. Returns false if the message is
// not exactly one Input, has unknown flags or too many characters, or holds a
// non-finite float.
bool DecodeInput(BytesView message, Input* out);

// Encodes `texture` and `payload` into a kTexture wire message. Returns an
// empty Bytes if `texture` has non-zero flags, an unknown action or format, a
// payload size that does not match the action and format (see Texture), or
// exceeds kMaxMessageBytes.
Bytes EncodeTexture(const Texture& texture, BytesView payload = {});

// Decodes a kTexture wire message into its header `*out` and the payload that
// follows it, `*out_payload` (a view into `message`). Returns false if the
// header is malformed, the action or format is unknown, or the payload size
// does not match the action and format (see Texture).
bool DecodeTexture(BytesView message, Texture* out, BytesView* out_payload);

// Encodes UTF-8 clipboard text, or a URL to open when `open_url` is true, into
// a kClipboard wire message, truncated on a codepoint boundary if needed to fit
// within kMaxMessageBytes.
Bytes EncodeClipboard(std::string_view utf8, bool open_url = false);

// Decodes a kClipboard wire message into `*out_utf8`, and sets `*open_url`
// (when non-null) to whether the text is a URL to open. Returns false if the
// header is malformed, has unknown flags, is truncated, or exceeds
// kMaxMessageBytes.
bool DecodeClipboard(BytesView message, std::string* out_utf8,
                     bool* open_url = nullptr);

// True if `url` is safe for the browser to open from a Clipboard::kOpenUrl
// message: an `http://` or `https://` URL (scheme compared
// case-insensitively).
bool IsOpenableUrl(std::string_view url);

}  // namespace mujoco::studio::netui

#endif  // MUJOCO_PYTHON_STUDIO_WEB_UI_CODEC_H_
