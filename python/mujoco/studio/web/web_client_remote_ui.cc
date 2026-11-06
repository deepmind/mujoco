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

#include "web_client_remote_ui.h"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <imgui.h>
#include "studio/ux/imgui_widgets.h"
#include "logging.h"
#include "ui_codec.h"
#include "web_client_websocket.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/em_js.h>

// Opens url in a new tab. Needs a recent user gesture, which the click that
// made the server send the URL provides; the popup blocker refuses it
// otherwise, which is logged to the browser console.
EM_JS(void, OpenBrowserUrl, (const char* url), {
  const str = UTF8ToString(url);
  const opened = window.open(str, "_blank");
  if (opened) {
    opened.opener = null;
  } else {
    console.warn("Opening URL was blocked by the browser:", str);
  }
});
#else
void OpenBrowserUrl(const char* url) {}
#endif  // defined(__EMSCRIPTEN__)

namespace mujoco::studio {
namespace {

// Sent as a text message after receiving draw frames, followed by the index of
// the newest (netui::DrawFrame::frame_index).
constexpr std::string_view kMsgUiAckPrefix = "ui_ack=";

// Converts `value` to the integer type T, clamped to T's range. Casting an
// out-of-range float is undefined (and can trap in WebAssembly).
template <typename T>
T ClampToInt(float value) {
  constexpr float kMin = static_cast<float>(std::numeric_limits<T>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<T>::max());
  if (!(value > kMin)) return std::numeric_limits<T>::min();  // Also NaN.
  if (value >= kMax) return std::numeric_limits<T>::max();
  return static_cast<T>(value);
}

// Reverses Dear ImGui's macOS Ctrl <-> Super swap so Input carries physical
// keys on the wire and the host swaps them at most once.
ImGuiKey PhysicalKey(ImGuiKey key, bool macos_behaviors) {
  if (!macos_behaviors) return key;
  switch (key) {
    case ImGuiKey_LeftCtrl:
      return ImGuiKey_LeftSuper;
    case ImGuiKey_LeftSuper:
      return ImGuiKey_LeftCtrl;
    case ImGuiKey_RightCtrl:
      return ImGuiKey_RightSuper;
    case ImGuiKey_RightSuper:
      return ImGuiKey_RightCtrl;
    case ImGuiKey_ReservedForModCtrl:
      return ImGuiKey_ReservedForModSuper;
    case ImGuiKey_ReservedForModSuper:
      return ImGuiKey_ReservedForModCtrl;
    default:
      return key;
  }
}

// Logs details when a draw command references a texture ID with no GPU handle.
void LogUnmappedTexture(
    RemoteUi::TextureId tex_id, size_t map_size, uint32_t draw_idx,
    const std::unordered_map<RemoteUi::TextureId, uintptr_t>& tex_map) {
  VLOG(1, "DrawFrame: UNMAPPED tex_id=%lu, map_size=%zu, draw#=%u",
       static_cast<unsigned long>(tex_id), map_size, draw_idx);
  std::string keys_str;
  for (const auto& kv : tex_map) {
    keys_str += " " + std::to_string(kv.first) +
                "(fil=" + std::to_string(kv.second) + ")";
  }
  VLOG(1, "  map keys:%s", keys_str.c_str());
}

}  // namespace

void RemoteUi::ResetSessionState() {
  if (callbacks_.GpuReady()) {
    for (const auto& [tex_id, handle] : texture_map_) {
      if (handle != 0) {
        callbacks_.UploadTexture(handle, nullptr, 0, 0);
      }
    }
  }
  texture_map_.clear();
  texture_cpu_.clear();
  remote_draw_data_.Clear();
  remote_draw_lists_.clear();
  pending_input_chars_.clear();
  frame_to_ack_.reset();
  last_serialized_frame_.Clear();
  last_frame_index_ = 0;
  last_decoded_frame_.reset();
  handshake_sent_ = false;
  handshake_accepted_ = false;
  mouse_cursor_ = ImGuiMouseCursor_Arrow;
}

void RemoteUi::Connect(std::unique_ptr<WebSocket> socket) {
  ResetSessionState();
  socket_ = std::move(socket);
  last_state_ = ReadyState::kDisconnected;
  request_keyframe_ = true;
}

RemoteUi::ReadyState RemoteUi::ConnectionState() const {
  return socket_ ? socket_->GetReadyState() : ReadyState::kDisconnected;
}

int RemoteUi::CloseCode() const { return socket_ ? socket_->CloseCode() : 0; }

void RemoteUi::ReceiveAndProcessCommands(int frame) {
  if (!socket_) return;

  const ReadyState state = ConnectionState();
  if (last_state_ != state) {
    LOG(Info, "WebSocket status changed: '%s' -> '%s' (frame %d)",
        ReadyStateName(last_state_), ReadyStateName(state), frame);
    last_state_ = state;
  }

  if (state == ReadyState::kOpen && !handshake_sent_) {
    netui::Hello hello;
    if (socket_->SendBinary(std::as_bytes(std::span(&hello, 1)))) {
      LOG(Info, "Sent the Hello handshake (version=%d)",
          static_cast<int>(hello.header.version));
      handshake_sent_ = true;
      request_keyframe_ = true;
    }
  }

  // Everything received, in order: a compressed draw frame needs the one
  // before it, and the draw frames need the textures sent before them.
  for (const WebSocket::Bytes& message : socket_->TakeMessages()) {
    ProcessMessage(message);
  }
  // Flow control: the server keeps at most one draw frame in flight, and sends
  // the next (newest) one once this page acks it by its index. Without it, a
  // slow link would buffer seconds of stale frames. Sent from the frame loop,
  // so the UI streams no faster than this page draws it.
  if (frame_to_ack_) {
    const std::string ack =
        std::string(kMsgUiAckPrefix) + std::to_string(*frame_to_ack_);
    socket_->SendText(ack.c_str());
    frame_to_ack_.reset();
  }
  VLOG(1, "Frame %d: status='%s', cmds=%d, draws=%d, textures=%d", frame,
       ReadyStateName(state), total_cmds_received_, draw_frames_received_,
       textures_received_);
}

void RemoteUi::ProcessMessage(netui::BytesView bytes) {
  bytes_accum_ += bytes.size();
  if (bytes.size() < sizeof(netui::MessageHeader) ||
      bytes.size() > netui::kMaxMessageBytes) {
    LOG(Warning, "Dropping a %zu-byte UI message outside valid bounds",
        bytes.size());
    return;
  }
  netui::MessageHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (header.size != bytes.size() || header.version != netui::kCodecVersion) {
    LOG(Warning, "Dropping a %zu-byte UI message that is not one valid command",
        bytes.size());
    return;
  }
  ++total_cmds_received_;

  switch (header.kind) {
    case netui::MessageKind::kDrawFrame:
      ++draw_frames_received_;
      VLOG(2, "Received DrawFrame #%d (size=%u)", draw_frames_received_,
           header.size);
      if (bytes.size() >= sizeof(netui::DrawFrame)) {
        netui::DrawFrame frame_hdr;
        std::memcpy(&frame_hdr, bytes.data(), sizeof(frame_hdr));
        frame_to_ack_ = frame_hdr.frame_index;
      }
      ProcessDrawFrameMessage(bytes);
      break;
    case netui::MessageKind::kTexture:
      ++textures_received_;
      VLOG(2, "Received Texture #%d (size=%u)", textures_received_,
           header.size);
      ProcessTextureMessage(bytes);
      break;
    case netui::MessageKind::kClipboard:
      VLOG(2, "Received Clipboard (size=%u)", header.size);
      ProcessClipboardMessage(bytes);
      break;
    case netui::MessageKind::kHello:
      if (bytes.size() == sizeof(netui::Hello)) {
        netui::Hello hello;
        std::memcpy(&hello, bytes.data(), sizeof(hello));
        handshake_accepted_ = true;
        LOG(Info, "Connected to Studio UI host (version=%d)",
            static_cast<int>(hello.header.version));
      }
      break;
    default:
      VLOG(2, "Ignoring command kind=%d, size=%u",
           static_cast<int>(header.kind), header.size);
      break;
  }
}

// The headless server's ImGui called SetClipboardText, e.g. for the Console's
// Copy All, and HeadlessUiSession forwarded the text here. A message with
// Clipboard::kOpenUrl carries a URL to open instead (see OpenBrowserUrl), which
// must be http or https: opening a javascript: URL would run it in this page.
void RemoteUi::ProcessClipboardMessage(netui::BytesView message) {
  std::string text;
  bool open_url = false;
  if (!netui::DecodeClipboard(message, &text, &open_url)) {
    LOG(Warning, "ProcessClipboardMessage: dropping invalid clipboard message");
    return;
  }
  if (open_url) {
    if (!netui::IsOpenableUrl(text)) {
      LOG(Warning, "ProcessClipboardMessage: refusing to open a non-http URL");
      return;
    }
    OpenBrowserUrl(text.c_str());
    return;
  }
  MaybeSaveToClipboard(text);
}

void RemoteUi::UpdateTextures() {
  // Upload only textures with no GPU handle yet; the rest survive a model swap
  // along with the Filament context. In particular, do not reset the ImGui font
  // atlas here: that trips an assert in ImFontAtlasUpdateNewFrame.
  bool uploaded_any = false;
  for (auto& [tex_id, entry] : texture_cpu_) {
    uintptr_t& local_tex = texture_map_[tex_id];
    if (local_tex == 0 && !entry.pixels.empty()) {
      local_tex = callbacks_.UploadTexture(local_tex, entry.pixels.data(),
                                           entry.width, entry.height);
      LOG(Info, "UpdateTextures: uploaded tex_id=%lu -> filament=%lu",
          static_cast<unsigned long>(tex_id),
          static_cast<unsigned long>(local_tex));
      if (local_tex != 0) {
        uploaded_any = true;
      }
    }
  }
  if (uploaded_any && last_decoded_frame_.has_value()) {
    RebuildDrawData();
  }
}

void RemoteUi::ProcessTextureMessage(netui::BytesView message) {
  netui::Texture hdr;
  netui::BytesView payload;
  if (!netui::DecodeTexture(message, &hdr, &payload)) {
    LOG(Warning, "ProcessTextureMessage: dropping invalid texture command");
    return;
  }

  const TextureId tex_id = hdr.tex_id;
  if (hdr.action == netui::Texture::Action::kDestroy) {
    auto it = texture_map_.find(tex_id);
    if (it != texture_map_.end()) {
      if (it->second != 0 && callbacks_.GpuReady()) {
        // Pass nullptr pixels to destroy the GPU texture.
        callbacks_.UploadTexture(it->second, nullptr, 0, 0);
      }
      texture_map_.erase(it);
    }
    texture_cpu_.erase(tex_id);
    return;
  }

  const bool is_a8 = hdr.format == netui::Texture::Format::kAlpha8;
  if (!is_a8 && hdr.format != netui::Texture::Format::kRgba8) {
    LOG(Warning, "ProcessTextureMessage: dropping unknown texture format %d",
        static_cast<int>(hdr.format));
    return;
  }

  // DecodeTexture() checked that the payload holds exactly the patch's pixels.
  // Count them in 64 bits (size_t is 32 bits in WebAssembly) and cap the RGBA
  // footprint before allocating.
  const uint32_t patch_w = hdr.width;
  const uint32_t patch_h = hdr.height;
  const uint64_t pixel_count64 = static_cast<uint64_t>(patch_w) * patch_h;
  if (pixel_count64 * 4 > netui::kMaxMessageBytes) {
    LOG(Warning, "ProcessTextureMessage: dropping oversize texture payload");
    return;
  }

  const size_t pixel_count = static_cast<size_t>(pixel_count64);
  const std::byte* pixels = payload.data();

  // Keep a CPU-side RGBA mirror of each remote texture so partial updates can
  // be patched in before re-uploading the full texture to Filament (which does
  // not support sub-rect GPU texture updates).
  TextureEntry* entry = nullptr;
  if (hdr.action == netui::Texture::Action::kCreate) {
    entry = &texture_cpu_[tex_id];
    entry->width = patch_w;
    entry->height = patch_h;
    entry->pixels.resize(pixel_count * 4);
    if (is_a8) {
      for (size_t p = 0; p < pixel_count; ++p) {
        entry->pixels[p * 4 + 0] = std::byte{0xFF};
        entry->pixels[p * 4 + 1] = std::byte{0xFF};
        entry->pixels[p * 4 + 2] = std::byte{0xFF};
        entry->pixels[p * 4 + 3] = pixels[p];
      }
    } else {
      std::memcpy(entry->pixels.data(), pixels, pixel_count * 4);
    }
  } else {
    auto cpu_it = texture_cpu_.find(tex_id);
    if (cpu_it == texture_cpu_.end() || cpu_it->second.pixels.empty()) {
      LOG(Warning,
          "ProcessTextureMessage: partial update for tex_id=%lu "
          "but no CPU mirror exists, skipping",
          static_cast<unsigned long>(tex_id));
      return;
    }
    entry = &cpu_it->second;
    const uint32_t off_x = hdr.x;
    const uint32_t off_y = hdr.y;
    if (static_cast<uint64_t>(off_x) + patch_w > entry->width ||
        static_cast<uint64_t>(off_y) + patch_h > entry->height) {
      LOG(Warning,
          "ProcessTextureMessage: patch %ux%u at (%u,%u) exceeds mirror %ux%u "
          "for tex_id=%lu; dropping",
          patch_w, patch_h, off_x, off_y, entry->width, entry->height,
          static_cast<unsigned long>(tex_id));
      return;
    }
    for (uint32_t row = 0; row < patch_h; ++row) {
      std::byte* dst =
          &entry->pixels[(static_cast<size_t>(off_y + row) * entry->width +
                          off_x) *
                         4];
      if (is_a8) {
        const std::byte* src = &pixels[static_cast<size_t>(row) * patch_w];
        for (uint32_t col = 0; col < patch_w; ++col) {
          dst[col * 4 + 0] = std::byte{0xFF};
          dst[col * 4 + 1] = std::byte{0xFF};
          dst[col * 4 + 2] = std::byte{0xFF};
          dst[col * 4 + 3] = src[col];
        }
      } else {
        const std::byte* src = &pixels[static_cast<size_t>(row) * patch_w * 4];
        std::memcpy(dst, src, static_cast<size_t>(patch_w) * 4);
      }
    }
  }

  uintptr_t& local_tex = texture_map_[tex_id];
  if (callbacks_.GpuReady()) {
    const bool was_unmapped = (local_tex == 0);
    local_tex = callbacks_.UploadTexture(local_tex, entry->pixels.data(),
                                         entry->width, entry->height);
    VLOG(1, "uploadGuiImage for client_tex_id=%lu -> filament=%lu",
         static_cast<unsigned long>(tex_id),
         static_cast<unsigned long>(local_tex));
    if (was_unmapped && local_tex != 0 && last_decoded_frame_.has_value()) {
      RebuildDrawData();
    }
  } else {
    VLOG(1, "ProcessTextureMessage: buffered tex_id=%lu, deferring GPU upload",
         static_cast<unsigned long>(tex_id));
  }
}

void RemoteUi::ProcessDrawFrameMessage(netui::BytesView message) {
  netui::DecodedFrame frame;
  bool need_keyframe = false;
  if (!netui::DecodeDrawFrame(message, &last_serialized_frame_,
                              &last_frame_index_, &frame, &need_keyframe)) {
    if (need_keyframe) {
      request_keyframe_ = true;
    } else {
      LOG(Warning, "ProcessDrawFrameMessage: dropping invalid draw frame");
    }
    return;
  }

  if (frame.mouse_cursor >= ImGuiMouseCursor_None &&
      frame.mouse_cursor < ImGuiMouseCursor_COUNT) {
    mouse_cursor_ = static_cast<ImGuiMouseCursor>(frame.mouse_cursor);
  } else {
    mouse_cursor_ = ImGuiMouseCursor_Arrow;
  }

  last_decoded_frame_ = std::move(frame);
  RebuildDrawData();
}

void RemoteUi::RebuildDrawData() {
  if (!last_decoded_frame_.has_value()) return;
  const netui::DecodedFrame& frame = *last_decoded_frame_;
  uint64_t total_vtx = 0;
  uint64_t total_idx = 0;
  uint64_t total_draws = 0;
  for (const netui::DrawList& src_list : frame.lists) {
    total_vtx += src_list.verts.size();
    total_idx += src_list.indices.size();
    total_draws += src_list.commands.size();
  }

  if (!remote_draw_data_.Valid) {
    LOG(Info,
        "First remote draw frame applied (%zu windows, %llu draws, %llu vtx)",
        frame.lists.size(), static_cast<unsigned long long>(total_draws),
        static_cast<unsigned long long>(total_vtx));
  }
  ImDrawData* draw_data = &remote_draw_data_;
  draw_data->Valid = true;
  draw_data->TotalVtxCount = static_cast<int>(total_vtx);
  draw_data->TotalIdxCount = static_cast<int>(total_idx);
  draw_data->DisplayPos = ImVec2(frame.display_pos[0], frame.display_pos[1]);
  draw_data->DisplaySize = ImVec2(frame.display_size[0], frame.display_size[1]);
  draw_data->FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;

  // One ImDrawList per remote window, keeping each window's own 0-based vertex
  // indices and reusing allocated buffers from frame to frame.
  const size_t list_count = frame.lists.size();
  remote_draw_lists_.resize(list_count);
  draw_data->CmdLists.resize(0);

  for (size_t i = 0; i < list_count; ++i) {
    const netui::DrawList& src_list = frame.lists[i];
    std::unique_ptr<ImDrawList>& list = remote_draw_lists_[i];
    if (!list) list = std::make_unique<ImDrawList>(nullptr);
    ImDrawList* cmd_list = list.get();
    draw_data->CmdLists.push_back(cmd_list);

    cmd_list->IdxBuffer.resize(static_cast<int>(src_list.indices.size()));
    cmd_list->VtxBuffer.resize(static_cast<int>(src_list.verts.size()));
    // Omit skipped draws from CmdBuffer rather than leaving zeroed ImDrawCmd
    // entries (Filament's Renderable::SetMesh treats elem_count == 0 as "draw
    // the entire remainder of the index buffer" and binds ObjectManager::kDecor
    // when color_texture is nullptr).
    cmd_list->CmdBuffer.resize(0);
    cmd_list->CmdBuffer.reserve(static_cast<int>(src_list.commands.size()));
    cmd_list->Flags = ImDrawListFlags_AllowVtxOffset |
                      ImDrawListFlags_AntiAliasedLines |
                      ImDrawListFlags_AntiAliasedFill |
                      ImDrawListFlags_AntiAliasedLinesUseTex;

    if (!src_list.indices.empty()) {
      if constexpr (sizeof(ImDrawIdx) == sizeof(uint16_t)) {
        std::memcpy(cmd_list->IdxBuffer.Data, src_list.indices.data(),
                    src_list.indices.size() * sizeof(ImDrawIdx));
      } else {
        for (size_t idx = 0; idx < src_list.indices.size(); ++idx) {
          cmd_list->IdxBuffer[static_cast<int>(idx)] =
              static_cast<ImDrawIdx>(src_list.indices[idx]);
        }
      }
    }

    ImDrawVert* vertex_dst = cmd_list->VtxBuffer.Data;
    for (size_t v = 0; v < src_list.verts.size(); ++v) {
      vertex_dst[v].pos = src_list.verts[v].UnpackPos(src_list.ref_pos);
      vertex_dst[v].uv = src_list.verts[v].UnpackUv();
      vertex_dst[v].col = src_list.verts[v].color;
    }

    for (size_t d = 0; d < src_list.commands.size(); ++d) {
      const netui::DrawCommand& src_cmd = src_list.commands[d];
      if (src_cmd.elem_count == 0 ||
          static_cast<uint64_t>(src_cmd.idx_offset) + src_cmd.elem_count >
              src_list.indices.size() ||
          src_cmd.vtx_offset > src_list.verts.size()) {
        continue;
      }

      auto it = texture_map_.find(src_cmd.tex_id);
      if (it == texture_map_.end() || it->second == 0) {
        LogUnmappedTexture(src_cmd.tex_id, texture_map_.size(),
                           static_cast<uint32_t>(d), texture_map_);
        continue;
      }

      const float cx =
          std::max(0.0f, std::min(screen_area_[0], src_cmd.clip_rect[0]));
      const float cy =
          std::max(0.0f, std::min(screen_area_[1], src_cmd.clip_rect[1]));
      const float cz =
          std::max(cx, std::min(screen_area_[0], src_cmd.clip_rect[2]));
      const float cw =
          std::max(cy, std::min(screen_area_[1], src_cmd.clip_rect[3]));

      ImDrawCmd cmd;
      cmd.ClipRect = ImVec4(cx, cy, cz, cw);
      cmd.VtxOffset = src_cmd.vtx_offset;
      cmd.IdxOffset = src_cmd.idx_offset;
      cmd.ElemCount = src_cmd.elem_count;
      cmd.UserCallback = nullptr;
      cmd.UserCallbackData = nullptr;
      cmd.TexRef._TexData = nullptr;
      cmd.TexRef._TexID = static_cast<ImTextureID>(it->second);
      cmd_list->CmdBuffer.push_back(cmd);
    }

    cmd_list->_VtxWritePtr =
        cmd_list->VtxBuffer.Data + cmd_list->VtxBuffer.Size;
    cmd_list->_IdxWritePtr =
        cmd_list->IdxBuffer.Data + cmd_list->IdxBuffer.Size;
    const unsigned int last_vtx_offset =
        cmd_list->CmdBuffer.empty() ? 0u : cmd_list->CmdBuffer.back().VtxOffset;
    const unsigned int vtx_since_offset =
        static_cast<unsigned int>(cmd_list->VtxBuffer.Size) >= last_vtx_offset
            ? static_cast<unsigned int>(cmd_list->VtxBuffer.Size) -
                  last_vtx_offset
            : 0u;
    cmd_list->_VtxCurrentIdx =
        vtx_since_offset < (1u << 16) ? vtx_since_offset : 0u;
  }
}

// Must run before ImGui::EndFrame() (which ImGui::Render() calls implicitly),
// because EndFrame() clears io.InputQueueCharacters.
void RemoteUi::CaptureAndSendInput() {
  if (!socket_ || !handshake_sent_) return;
  if (ConnectionState() != ReadyState::kOpen) return;

  const ImGuiIO& io = ImGui::GetIO();
  const bool local_wants_mouse = io.WantCaptureMouse;
  const bool local_wants_keyboard = io.WantCaptureKeyboard;

  if (!local_wants_keyboard) {
    for (int i = 0; i < io.InputQueueCharacters.Size; ++i) {
      pending_input_chars_.push_back(
          static_cast<uint32_t>(io.InputQueueCharacters[i]));
    }
  }
  if (!local_wants_mouse) {
    mouse_wheel_pos_[0] += io.MouseWheel;
    mouse_wheel_pos_[1] += io.MouseWheelH;
  }

  netui::Input input;
  // The remote client lays out for the screen area (the canvas minus the
  // local role bar), not the whole display.
  input.screen_size[0] = ClampToInt<uint16_t>(screen_area_[0]);
  input.screen_size[1] = ClampToInt<uint16_t>(screen_area_[1]);
  if (input.screen_size[0] != last_screen_size_[0] ||
      input.screen_size[1] != last_screen_size_[1]) {
    LOG(Info, "Screen size sent to client changed: %ux%u -> %ux%u",
        last_screen_size_[0], last_screen_size_[1], input.screen_size[0],
        input.screen_size[1]);
    last_screen_size_[0] = input.screen_size[0];
    last_screen_size_[1] = input.screen_size[1];
  }
  input.framebuffer_scale =
      io.DisplayFramebufferScale.x > 0.0f ? io.DisplayFramebufferScale.x : 1.0f;
  input.header.flags =
      (request_keyframe_ ? netui::Input::kRequestKeyframe : 0) |
      (io.ConfigMacOSXBehaviors ? netui::Input::kMacOsBehaviors : 0);

  // Wheel fields are lifetime running totals so freezing accumulation while a
  // local overlay captures the mouse avoids spurious deltas on release.
  input.mouse_wheel[0] = mouse_wheel_pos_[0];
  input.mouse_wheel[1] = mouse_wheel_pos_[1];
  if (!local_wants_mouse) {
    input.mouse_pos[0] = io.MousePos.x;
    input.mouse_pos[1] = io.MousePos.y;
  } else {
    input.mouse_pos[0] = -FLT_MAX;
    input.mouse_pos[1] = -FLT_MAX;
  }

  input.mouse_down_mask = 0;
  if (!local_wants_mouse) {
    for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
      if (ImGui::IsMouseDown(button)) {
        input.mouse_down_mask |= 1u << button;
      }
    }
  }

  if (!local_wants_keyboard) {
    for (int i = 0; i < ImGuiKey_NamedKey_COUNT; ++i) {
      const auto key = static_cast<ImGuiKey>(ImGuiKey_NamedKey_BEGIN + i);
      if (ImGui::IsKeyDown(key)) {
        input.SetImGuiKeyDown(PhysicalKey(key, io.ConfigMacOSXBehaviors), true);
      }
    }

    const size_t count =
        std::min<size_t>(netui::Input::kMaxChars, pending_input_chars_.size());
    if (count > 0) {
      std::memcpy(input.chars, pending_input_chars_.data(),
                  count * sizeof(uint32_t));
      input.char_count = static_cast<uint32_t>(count);
      pending_input_chars_.erase(
          pending_input_chars_.begin(),
          pending_input_chars_.begin() + static_cast<ptrdiff_t>(count));
    }
  } else {
    pending_input_chars_.clear();
  }

  if (socket_->SendBinary(std::as_bytes(std::span(&input, 1))) &&
      (input.header.flags & netui::Input::kRequestKeyframe) != 0) {
    request_keyframe_ = false;
  }
}

void RemoteUi::Shutdown() {
  ResetSessionState();
  socket_.reset();
  last_state_ = ReadyState::kDisconnected;
}

}  // namespace mujoco::studio
