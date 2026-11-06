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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <imgui.h>
#include "studio/ux/imgui_widgets.h"
#include <NetImgui_CmdPackets.h>
#include "google/logging.h"
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

using ::NetImgui::Internal::ArrayCount;
using ::NetImgui::Internal::CmdClipboard;
using ::NetImgui::Internal::CmdDrawFrame;
using ::NetImgui::Internal::CmdHeader;
using ::NetImgui::Internal::CmdInput;
using ::NetImgui::Internal::CmdTexture;
using ::NetImgui::Internal::CmdVersion;
using ::NetImgui::Internal::DecompressCmdDrawFrame;
using ::NetImgui::Internal::ImguiDraw;
using ::NetImgui::Internal::ImguiDrawGroup;
using ::NetImgui::Internal::ImguiVert;
using ::NetImgui::Internal::netImguiDelete;
using ::NetImgui::Internal::netImguiDeleteSafe;
using ::NetImgui::Internal::netImguiSizedNew;
using ::NetImgui::Internal::PendingCom;
using ::NetImgui::Internal::StringCopy;

// Delta-compress the GUI stream (relayed to the client via CmdInput).
constexpr bool kUseCompression = true;

// Sent as a text message after receiving draw frames, followed by the index of
// the newest (CmdDrawFrame::mFrameIndex).
constexpr char kMsgUiAckPrefix[] = "ui_ack=";

// Clipboard text with this prefix is a URL for this browser to open in a new
// tab rather than something to copy: the headless server has no browser of its
// own, and the clipboard is the only text channel NetImgui offers it.
constexpr char kOpenUrlPrefix[] = "\x01OPEN_URL:";

// Converts `value` to the integer type T, clamped to T's range. Casting an
// out-of-range float is undefined (and can trap in WebAssembly), e.g. Dear
// ImGui's "no mouse" position, -FLT_MAX.
template <typename T>
T ClampToInt(float value) {
  constexpr float kMin = static_cast<float>(std::numeric_limits<T>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<T>::max());
  if (!(value > kMin)) return std::numeric_limits<T>::min();  // Also NaN.
  if (value >= kMax) return std::numeric_limits<T>::max();
  return static_cast<T>(value);
}

// Logs details when a draw command references a texture ID with no GPU handle.
void LogUnmappedTexture(
    RemoteUi::ClientTextureID client_tex_id, size_t map_size, uint32_t draw_idx,
    const std::unordered_map<RemoteUi::ClientTextureID, uintptr_t>& tex_map) {
  VLOG(1, "DrawFrame: UNMAPPED client_tex_id=%lu, map_size=%zu, draw#=%u",
       static_cast<unsigned long>(client_tex_id), map_size, draw_idx);
  std::string keys_str = "";
  for (auto& kv : tex_map) {
    keys_str += " " + std::to_string(kv.first) +
                "(fil=" + std::to_string(kv.second) + ")";
  }
  VLOG(1, "  map keys:%s", keys_str.c_str());
}

}  // namespace

void RemoteUi::Connect(std::unique_ptr<WebSocket> socket) {
  socket_ = std::move(socket);
  handshake_sent_ = false;
  last_state_ = ReadyState::kDisconnected;
  frame_to_ack_.reset();
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
    CmdVersion cmd_version;
    StringCopy(cmd_version.mClientName, "MuJoCo Web Viewer");
    if (socket_->SendBinary(std::as_bytes(std::span(&cmd_version, 1)))) {
      LOG(Info, "Sent the CmdVersion handshake (version=%d, wchar_size=%d)",
          static_cast<int>(cmd_version.mVersion), cmd_version.mWCharSize);
      handshake_sent_ = true;
      // Fresh connection: the client may resume delta compression against
      // a frame from a previous session; ask for an uncompressed keyframe.
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
    socket_->SendText(
        (kMsgUiAckPrefix + std::to_string(*frame_to_ack_)).c_str());
    frame_to_ack_.reset();
  }
  VLOG(1, "Frame %d: status='%s', cmds=%d, draws=%d, textures=%d", frame,
       ReadyStateName(state), total_cmds_received_, draw_frames_received_,
       textures_received_);
}

void RemoteUi::ProcessMessage(WebSocket::BytesView message) {
  bytes_accum_ += message.size();
  CmdHeader header(CmdHeader::eCommands::Count, 0);
  if (message.size() >= sizeof(header)) {
    std::memcpy(static_cast<void*>(&header), message.data(), sizeof(header));
  }
  if (message.size() < sizeof(header) || header.mSize != message.size()) {
    LOG(Warning, "Dropping a %zu-byte UI message that is not one command",
        message.size());
    return;
  }
  ++total_cmds_received_;

  // Commands are read in place, which needs NetImgui's 8-byte alignment.
  auto* command =
      reinterpret_cast<CmdHeader*>(netImguiSizedNew<uint8_t>(message.size()));
  std::memcpy(static_cast<void*>(command), message.data(), message.size());
  switch (header.mType) {
    case CmdHeader::eCommands::DrawFrame:
      ++draw_frames_received_;
      VLOG(2, "Received DrawFrame #%d (size=%u)", draw_frames_received_,
           header.mSize);
      if (header.mSize >= sizeof(CmdDrawFrame)) {
        frame_to_ack_ = reinterpret_cast<CmdDrawFrame*>(command)->mFrameIndex;
      }
      ProcessCmdDrawFrame(reinterpret_cast<CmdDrawFrame*>(command));
      return;  // ProcessCmdDrawFrame owns the command now.
    case CmdHeader::eCommands::Texture:
      ++textures_received_;
      VLOG(2, "Received Texture #%d (size=%u)", textures_received_,
           header.mSize);
      ProcessCmdTexture(reinterpret_cast<CmdTexture*>(command));
      break;
    case CmdHeader::eCommands::Clipboard:
      VLOG(2, "Received Clipboard cmd (size=%u)", header.mSize);
      ProcessCmdClipboard(reinterpret_cast<CmdClipboard*>(command));
      break;
    case CmdHeader::eCommands::Version:
      if (header.mSize == sizeof(CmdVersion)) {
        const auto* version = reinterpret_cast<const CmdVersion*>(command);
        LOG(Info, "Connected to '%.*s' (version=%d)",
            static_cast<int>(sizeof(version->mClientName)),
            version->mClientName, static_cast<int>(version->mVersion));
      }
      break;
    default:
      VLOG(2, "Ignoring command type=%d, size=%u",
           static_cast<int>(header.mType), header.mSize);
      break;
  }
  netImguiDelete(command);
}

// The headless server's ImGui called SetClipboardText, e.g. for the Console's
// Copy All, and NetImgui forwarded the text here. MaybeSaveToClipboard copies
// it, so the caller can free the command afterwards. Text prefixed with
// kOpenUrlPrefix is a URL to open instead (see OpenBrowserUrl).
void RemoteUi::ProcessCmdClipboard(CmdClipboard* cmd_clipboard) {
  if (!cmd_clipboard) return;
  if (!cmd_clipboard->ToPointers()) {
    LOG(Warning, "ProcessCmdClipboard: dropping invalid clipboard command");
    return;
  }
  const char* text = cmd_clipboard->mContentUTF8.Get();
  if (text == nullptr) return;
  constexpr size_t kPrefixLength = sizeof(kOpenUrlPrefix) - 1;
  if (std::strncmp(text, kOpenUrlPrefix, kPrefixLength) == 0) {
    OpenBrowserUrl(text + kPrefixLength);
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
  if (uploaded_any && last_uncompressed_frame_ != nullptr) {
    RebuildDrawData();
  }
}

void RemoteUi::ProcessCmdTexture(CmdTexture* cmd_texture) {
  if (!cmd_texture) return;
  if (cmd_texture->mSize < sizeof(CmdTexture)) {
    LOG(Warning, "ProcessCmdTexture: dropping truncated texture command");
    return;
  }

  const ClientTextureID tex_id = cmd_texture->mTextureClientID;
  if (cmd_texture->mStatus == CmdTexture::eType::Destroy) {
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

  // The command must carry the pixels it describes: one byte each for the
  // font atlas (alpha only), four for color images.
  const bool is_a8 = cmd_texture->mFormat == NetImgui::eTexFormat::kTexFmtA8;
  if (!is_a8 && cmd_texture->mFormat != NetImgui::eTexFormat::kTexFmtRGBA8) {
    LOG(Warning, "ProcessCmdTexture: dropping texture of unknown format %d",
        static_cast<int>(cmd_texture->mFormat));
    return;
  }
  // Width, height and size come off the wire. Count the pixels in 64 bits
  // (size_t is 32 bits in the browser), and cap the count at NetImgui's largest
  // command before narrowing it, so that the RGBA copy below fits in size_t.
  const uint32_t patch_w = cmd_texture->mWidth;
  const uint32_t patch_h = cmd_texture->mHeight;
  const uint64_t pixel_count64 = static_cast<uint64_t>(patch_w) * patch_h;
  const uint64_t pixels_size = pixel_count64 * (is_a8 ? 1 : 4);
  if (pixel_count64 == 0 || pixel_count64 > PendingCom::kMaxCmdSize ||
      !cmd_texture->mpTextureData.ToPointer(
          cmd_texture,
          reinterpret_cast<const uint8_t*>(cmd_texture) + cmd_texture->mSize,
          pixels_size)) {
    LOG(Warning, "ProcessCmdTexture: dropping invalid texture command");
    return;
  }
  const size_t pixel_count = static_cast<size_t>(pixel_count64);
  const uint8_t* pixels = cmd_texture->mpTextureData.Get();

  VLOG(
      1,
      "ProcessCmdTexture: client_tex_id=%lu, status=%d, size=%ux%u, format=%d, "
      "offset=%u,%u, map_size=%zu",
      static_cast<unsigned long>(tex_id),
      static_cast<int>(cmd_texture->mStatus), patch_w, patch_h,
      static_cast<int>(cmd_texture->mFormat),
      static_cast<uint32_t>(cmd_texture->mOffsetX),
      static_cast<uint32_t>(cmd_texture->mOffsetY), texture_map_.size());

  // Convert incoming pixels to RGBA (Filament requires RGBA).
  // For A8 font atlas data, expand each byte to (255, 255, 255, alpha).
  WebSocket::Bytes rgba_pixels(pixel_count * 4);
  if (is_a8) {
    for (size_t p = 0; p < pixel_count; ++p) {
      rgba_pixels[p * 4 + 0] = std::byte{0xFF};
      rgba_pixels[p * 4 + 1] = std::byte{0xFF};
      rgba_pixels[p * 4 + 2] = std::byte{0xFF};
      rgba_pixels[p * 4 + 3] = static_cast<std::byte>(pixels[p]);
    }
  } else {
    std::memcpy(rgba_pixels.data(), pixels, pixel_count * 4);
  }

  uintptr_t& local_tex = texture_map_[tex_id];
  // Keep a CPU-side RGBA mirror of each remote texture so partial updates from
  // NetImgui can be patched in before re-uploading the full texture to
  // Filament (which does not support sub-rect GPU texture updates).
  TextureEntry& entry = texture_cpu_[tex_id];

  if (cmd_texture->mStatus == CmdTexture::eType::Create) {
    // Full texture creation — store the CPU-side mirror.
    entry.width = patch_w;
    entry.height = patch_h;
    entry.pixels = std::move(rgba_pixels);
  } else {
    // Partial update — patch the sub-region into the existing CPU mirror.
    // If no CPU mirror exists (e.g. we missed the Create), skip.
    if (entry.pixels.empty()) {
      LOG(Warning,
          "ProcessCmdTexture: partial update for tex_id=%lu "
          "but no CPU mirror exists, skipping",
          static_cast<unsigned long>(tex_id));
      return;
    }
    const uint32_t off_x = cmd_texture->mOffsetX;
    const uint32_t off_y = cmd_texture->mOffsetY;
    // The patch rectangle is wire-supplied; reject one that would write past
    // the stored mirror (e.g. dims desynced from a stale entry after a
    // reconnect) rather than corrupting the heap.
    if (static_cast<uint64_t>(off_x) + patch_w > entry.width ||
        static_cast<uint64_t>(off_y) + patch_h > entry.height) {
      LOG(Warning,
          "ProcessCmdTexture: patch %ux%u at (%u,%u) exceeds mirror %ux%u "
          "for tex_id=%lu; dropping",
          patch_w, patch_h, off_x, off_y, entry.width, entry.height,
          static_cast<unsigned long>(tex_id));
      return;
    }
    for (uint32_t row = 0; row < patch_h; ++row) {
      size_t dst_offset =
          (static_cast<size_t>(off_y + row) * entry.width + off_x) * 4;
      size_t src_offset = static_cast<size_t>(row) * patch_w * 4;
      std::memcpy(&entry.pixels[dst_offset], &rgba_pixels[src_offset],
                  static_cast<size_t>(patch_w) * 4);
    }
  }

  // Upload the full CPU-side texture to the GPU if the context is ready.
  // If it isn't yet (model still loading), the texture stays in texture_cpu_
  // and will be flushed by UpdateTextures() later.
  if (callbacks_.GpuReady()) {
    const bool was_unmapped = (local_tex == 0);
    local_tex = callbacks_.UploadTexture(local_tex, entry.pixels.data(),
                                         entry.width, entry.height);
    VLOG(1, "uploadGuiImage for client_tex_id=%lu -> filament=%lu",
         static_cast<unsigned long>(tex_id),
         static_cast<unsigned long>(local_tex));
    if (was_unmapped && local_tex != 0 && last_uncompressed_frame_ != nullptr) {
      RebuildDrawData();
    }
  } else {
    VLOG(1, "ProcessCmdTexture: buffered tex_id=%lu, deferring GPU upload",
         static_cast<unsigned long>(tex_id));
  }
}

// Ported from ProcessCmdDrawFrame in the vendored
// netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp, which merges
// all windows into one draw list; this builds one list per window instead.
void RemoteUi::ProcessCmdDrawFrame(CmdDrawFrame* cmd_draw_frame) {
  if (!cmd_draw_frame) return;

  if (!cmd_draw_frame->ToPointers()) {
    LOG(Warning, "ProcessCmdDrawFrame: dropping invalid draw frame");
    netImguiDeleteSafe(cmd_draw_frame);
    return;
  }

  if (cmd_draw_frame->mCompressed) {
    if (last_uncompressed_frame_ != nullptr &&
        (last_uncompressed_frame_->mFrameIndex + 1) ==
            cmd_draw_frame->mFrameIndex) {
      CmdDrawFrame* uncompressed_frame = DecompressCmdDrawFrame(
          last_uncompressed_frame_.get(), cmd_draw_frame);
      netImguiDeleteSafe(cmd_draw_frame);
      cmd_draw_frame = uncompressed_frame;
      if (!cmd_draw_frame) {  // Invalid compressed data, get a fresh keyframe.
        request_keyframe_ = true;
        return;
      }
    } else {
      // Missing previous / reference frame data. Ignore this delta-encoded
      // drawframe and ask the client for a fresh uncompressed keyframe.
      request_keyframe_ = true;
      netImguiDeleteSafe(cmd_draw_frame);
      return;
    }
  }

  // Release previous cached frame and store current for the next delta
  // decompression.
  last_uncompressed_frame_.reset(cmd_draw_frame);
  RebuildDrawData();
}

void RemoteUi::RebuildDrawData() {
  CmdDrawFrame* cmd_draw_frame = last_uncompressed_frame_.get();
  if (!cmd_draw_frame) return;

  if (!remote_draw_data_.Valid) {
    LOG(Info, "First remote draw frame applied (%u windows, %u draws, %u vtx)",
        cmd_draw_frame->mDrawGroupCount, cmd_draw_frame->mTotalDrawCount,
        cmd_draw_frame->mTotalVerticeCount);
  }
  ImDrawData* draw_data = &remote_draw_data_;
  draw_data->Valid = true;
  draw_data->TotalVtxCount =
      static_cast<int>(cmd_draw_frame->mTotalVerticeCount);
  draw_data->TotalIdxCount =
      static_cast<int>(cmd_draw_frame->mTotalIndiceCount);

  draw_data->DisplayPos.x = cmd_draw_frame->mDisplayArea[0];
  draw_data->DisplayPos.y = cmd_draw_frame->mDisplayArea[1];
  draw_data->DisplaySize.x =
      cmd_draw_frame->mDisplayArea[2] - cmd_draw_frame->mDisplayArea[0];
  draw_data->DisplaySize.y =
      cmd_draw_frame->mDisplayArea[3] - cmd_draw_frame->mDisplayArea[1];
  draw_data->FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;

  // One ImDrawList per draw group (remote window), the way Dear ImGui produced
  // them: each window keeps its own 0-based vertex indices, so they are copied
  // as-is and no window shares the 16-bit index range with another. The lists
  // only carry finished geometry, so they need no ImDrawListSharedData. They
  // are reused from frame to frame, which keeps their buffers allocated, and
  // freed as windows close.
  const uint32_t group_count = cmd_draw_frame->mDrawGroupCount;
  remote_draw_lists_.resize(group_count);
  draw_data->CmdLists.resize(0);

  constexpr float kPosRangeMin = static_cast<float>(ImguiVert::kPosRange_Min);
  constexpr float kPosRangeMax = static_cast<float>(ImguiVert::kPosRange_Max);
  constexpr float kUVRangeMin = static_cast<float>(ImguiVert::kUvRange_Min);
  constexpr float kUVRangeMax = static_cast<float>(ImguiVert::kUvRange_Max);

  for (uint32_t i(0); i < group_count; ++i) {
    const ImguiDrawGroup& draw_group = cmd_draw_frame->mpDrawGroups[i];
    std::unique_ptr<ImDrawList>& list = remote_draw_lists_[i];
    if (!list) list = std::make_unique<ImDrawList>(nullptr);
    ImDrawList* cmd_list = list.get();
    draw_data->CmdLists.push_back(cmd_list);

    cmd_list->IdxBuffer.resize(draw_group.mIndiceCount);
    cmd_list->VtxBuffer.resize(draw_group.mVerticeCount);
    // Omit skipped draws from CmdBuffer rather than leaving zeroed ImDrawCmd
    // entries (Filament's Renderable::SetMesh treats elem_count == 0 as "draw
    // the entire remainder of the index buffer" and binds ObjectManager::kDecor
    // when color_texture is nullptr).
    cmd_list->CmdBuffer.resize(0);
    cmd_list->CmdBuffer.reserve(draw_group.mDrawCount);
    cmd_list->Flags = ImDrawListFlags_AllowVtxOffset |
                      ImDrawListFlags_AntiAliasedLines |
                      ImDrawListFlags_AntiAliasedFill |
                      ImDrawListFlags_AntiAliasedLinesUseTex;

    // Indices
    const uint16_t* indices =
        reinterpret_cast<const uint16_t*>(draw_group.mpIndices.Get());
    if (draw_group.mIndiceCount > 0) {
      if (draw_group.mBytePerIndex == sizeof(ImDrawIdx)) {
        memcpy(cmd_list->IdxBuffer.Data, indices,
               draw_group.mIndiceCount * sizeof(ImDrawIdx));
      } else {
        for (uint32_t index_idx(0); index_idx < draw_group.mIndiceCount;
             ++index_idx) {
          cmd_list->IdxBuffer[index_idx] =
              static_cast<ImDrawIdx>(indices[index_idx]);
        }
      }
    }

    // Vertices — unpack quantized positions and UVs.
    const ImguiVert* vertex_src = draw_group.mpVertices.Get();
    ImDrawVert* vertex_dst = cmd_list->VtxBuffer.Data;
    for (uint32_t vtx_idx(0); vtx_idx < draw_group.mVerticeCount; ++vtx_idx) {
      vertex_dst[vtx_idx].pos.x =
          (static_cast<float>(vertex_src[vtx_idx].mPos[0]) *
           (kPosRangeMax - kPosRangeMin)) /
              static_cast<float>(0xFFFF) +
          kPosRangeMin + draw_group.mReferenceCoord[0];
      vertex_dst[vtx_idx].pos.y =
          (static_cast<float>(vertex_src[vtx_idx].mPos[1]) *
           (kPosRangeMax - kPosRangeMin)) /
              static_cast<float>(0xFFFF) +
          kPosRangeMin + draw_group.mReferenceCoord[1];
      vertex_dst[vtx_idx].uv.x =
          (static_cast<float>(vertex_src[vtx_idx].mUV[0]) *
           (kUVRangeMax - kUVRangeMin)) /
              static_cast<float>(0xFFFF) +
          kUVRangeMin;
      vertex_dst[vtx_idx].uv.y =
          (static_cast<float>(vertex_src[vtx_idx].mUV[1]) *
           (kUVRangeMax - kUVRangeMin)) /
              static_cast<float>(0xFFFF) +
          kUVRangeMin;
      vertex_dst[vtx_idx].col = vertex_src[vtx_idx].mColor;
    }

    // Draw commands.
    const ImguiDraw* draw_src = draw_group.mpDraws.Get();
    for (uint32_t draw_idx(0); draw_idx < draw_group.mDrawCount; ++draw_idx) {
      uint32_t idx_off = draw_src[draw_idx].mIdxOffset;
      uint32_t elem_count = draw_src[draw_idx].mIdxCount;

      // The draw range comes off the wire: skip draws that are empty or would
      // index past the buffer.
      if (elem_count == 0 ||
          static_cast<uint64_t>(idx_off) + elem_count >
              static_cast<uint64_t>(cmd_list->IdxBuffer.Size)) {
        continue;
      }

      // Map remote ClientTextureID -> local GL handle.
      ClientTextureID client_tex_id = draw_src[draw_idx].mClientTexId;
      auto it = texture_map_.find(client_tex_id);
      if (it == texture_map_.end() || it->second == 0) {
        LogUnmappedTexture(client_tex_id, texture_map_.size(), draw_idx,
                           texture_map_);
        continue;
      }

      float cx = std::max(
          0.f, std::min(screen_area_[0], draw_src[draw_idx].mClipRect[0]));
      float cy = std::max(
          0.f, std::min(screen_area_[1], draw_src[draw_idx].mClipRect[1]));
      float cz = std::max(
          cx, std::min(screen_area_[0], draw_src[draw_idx].mClipRect[2]));
      float cw = std::max(
          cy, std::min(screen_area_[1], draw_src[draw_idx].mClipRect[3]));

      ImDrawCmd cmd;
      cmd.ClipRect.x = cx;
      cmd.ClipRect.y = cy;
      cmd.ClipRect.z = cz;
      cmd.ClipRect.w = cw;
      cmd.VtxOffset = 0;
      cmd.IdxOffset = idx_off;
      cmd.ElemCount = elem_count;
      cmd.UserCallback = nullptr;
      cmd.UserCallbackData = nullptr;
      cmd.TexRef._TexData = nullptr;
      cmd.TexRef._TexID = static_cast<ImTextureID>(it->second);
      cmd_list->CmdBuffer.push_back(cmd);
    }

    // ImGui's AddDrawList() asserts that these point at the end of the buffers,
    // as they would after its own PrimReserve() API.
    cmd_list->_VtxWritePtr =
        cmd_list->VtxBuffer.Data + cmd_list->VtxBuffer.Size;
    cmd_list->_IdxWritePtr =
        cmd_list->IdxBuffer.Data + cmd_list->IdxBuffer.Size;
    cmd_list->_VtxCurrentIdx = cmd_list->VtxBuffer.Size;
  }
}

// Keep in sync with CaptureImguiInput in the vendored
// netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp. Must run
// before ImGui::EndFrame() (which ImGui::Render() calls implicitly), because
// EndFrame() clears io.InputQueueCharacters.
void RemoteUi::CaptureAndSendInput() {
  // The handshake goes first (see ReceiveAndProcessCommands()).
  if (!socket_ || !handshake_sent_) return;

  if (ConnectionState() != ReadyState::kOpen) return;

  // Capture input from Dear ImGui.
  const ImGuiIO& io = ImGui::GetIO();

  // When the local UI wants to capture mouse or keyboard, suppress the
  // corresponding input from being forwarded to the remote client. This
  // prevents clicks/keys meant for the local status overlay (or any other
  // local window) from leaking through to the UI server.
  const bool local_wants_mouse = io.WantCaptureMouse;
  const bool local_wants_keyboard = io.WantCaptureKeyboard;

  {
    // Only accumulate characters when the local UI is NOT capturing keyboard.
    if (!local_wants_keyboard) {
      const size_t initial_size = pending_input_chars_.size();
      const size_t added_char = io.InputQueueCharacters.size();
      if (added_char) {
        pending_input_chars_.resize(initial_size + added_char);
        memcpy(&pending_input_chars_[initial_size],
               io.InputQueueCharacters.Data, added_char * sizeof(ImWchar));
      }
    }

    // Gate scroll ACCUMULATION on local capture, the totals themselves are
    // still sent every frame. This implements the "FREEZING" mentioned in the
    // CmdInput wheel comment below.
    if (!local_wants_mouse) {
      mouse_wheel_pos_[0] += io.MouseWheel;
      mouse_wheel_pos_[1] += io.MouseWheelH;
    }
  }

  CmdInput cmd_input;
  // The remote client lays out for the screen area (the canvas minus the
  // local role bar), not the whole display.
  cmd_input.mScreenSize[0] = ClampToInt<uint16_t>(screen_area_[0]);
  cmd_input.mScreenSize[1] = ClampToInt<uint16_t>(screen_area_[1]);
  // An unstable screen size forces the remote UI to relayout every frame
  // (visible as UI flicker), so make changes loud.
  if (cmd_input.mScreenSize[0] != last_screen_size_[0] ||
      cmd_input.mScreenSize[1] != last_screen_size_[1]) {
    LOG(Info, "Screen size sent to client changed: %ux%u -> %ux%u",
        last_screen_size_[0], last_screen_size_[1], cmd_input.mScreenSize[0],
        cmd_input.mScreenSize[1]);
    last_screen_size_[0] = cmd_input.mScreenSize[0];
    last_screen_size_[1] = cmd_input.mScreenSize[1];
  }
  cmd_input.mFontDPIScaling = 1.f;
  cmd_input.mDesiredFps = 60.0f;
  cmd_input.mCompressionUse = kUseCompression;
  cmd_input.mCompressionSkip = request_keyframe_;

  // NetImgui wheel fields are lifetime running totals, not per-frame deltas:
  // the receiving side derives each frame's scroll as the difference between
  // the current total and the previous total. So scroll is suppressed during
  // local UI capture by FREEZING the totals (accumulation above is gated on
  // !local_wants_mouse), while still sending them every frame. Sending 0
  // instead would rewind the receiver's baseline and produce two huge spurious
  // deltas: -total when capture starts (wild zoom out as the local window
  // expands) and +total when it ends (snap back on collapse). Note the
  // deliberate asymmetry with the mouse position below, which IS per-frame
  // absolute and can simply be parked while captured.
  cmd_input.mMouseWheelVert = mouse_wheel_pos_[0];
  cmd_input.mMouseWheelHoriz = mouse_wheel_pos_[1];
  if (!local_wants_mouse) {
    cmd_input.mMousePos[0] = ClampToInt<int16_t>(io.MousePos.x);
    cmd_input.mMousePos[1] = ClampToInt<int16_t>(io.MousePos.y);
  } else {
    // Park the mouse off-screen so the remote side doesn't think we're
    // hovering over anything.
    cmd_input.mMousePos[0] = -1;
    cmd_input.mMousePos[1] = -1;
  }

  // Mouse button inputs. This static_assert detects when a Dear ImGui update
  // changes ImGuiMouseButton, which requires updating NetImgui's enum copy.
  static_assert(
      static_cast<int>(CmdInput::NetImguiMouseButton::ImGuiMouseButton_COUNT) ==
          static_cast<int>(ImGuiMouseButton_::ImGuiMouseButton_COUNT),
      "Update the NetImgui enum to match the updated Dear ImGui enum");
  cmd_input.mMouseDownMask = 0;
  if (!local_wants_mouse) {
    cmd_input.mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Left)
            ? 1 << CmdInput::ImGuiMouseButton_Left
            : 0;
    cmd_input.mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Right)
            ? 1 << CmdInput::ImGuiMouseButton_Right
            : 0;
    cmd_input.mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Middle)
            ? 1 << CmdInput::ImGuiMouseButton_Middle
            : 0;
    cmd_input.mMouseDownMask |=
        ImGui::IsMouseDown(3) ? 1 << CmdInput::ImGuiMouseButton_Extra1 : 0;
    cmd_input.mMouseDownMask |=
        ImGui::IsMouseDown(4) ? 1 << CmdInput::ImGuiMouseButton_Extra2 : 0;
  }

// Keyboard inputs. These static_asserts detect when a Dear ImGui update
// changes ImGuiKey, which requires updating NetImgui's enum copy to match.
#define EnumKeynameTest(KEYNAME)                                               \
  static_cast<int>(CmdInput::NetImguiKeys::KEYNAME) ==                         \
      static_cast<int>(ImGuiKey::KEYNAME - ImGuiKey::ImGuiKey_NamedKey_BEGIN), \
      "Update the NetImgui enum to match the updated Dear ImGui enum"
  static_assert(
      CmdInput::NetImguiKeys::ImGuiKey_COUNT ==
          (ImGuiKey_NamedKey_END - ImGuiKey_NamedKey_BEGIN),
      "Update the NetImgui enum to match the updated Dear ImGui enum");
  static_assert(EnumKeynameTest(ImGuiKey_Tab));
  static_assert(EnumKeynameTest(ImGuiKey_Escape));
  static_assert(EnumKeynameTest(ImGuiKey_RightSuper));
  static_assert(EnumKeynameTest(ImGuiKey_Apostrophe));
  static_assert(EnumKeynameTest(ImGuiKey_Keypad0));
  static_assert(EnumKeynameTest(ImGuiKey_CapsLock));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModCtrl));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModShift));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModAlt));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModSuper));
#undef EnumKeynameTest

  // Save every keydown status to out bitmask — only when local UI is not
  // capturing keyboard.
  if (!local_wants_keyboard) {
    uint64_t value_mask(0);
    for (uint32_t i(0); i < ImGuiKey::ImGuiKey_NamedKey_COUNT; ++i) {
      value_mask |=
          ImGui::IsKeyDown(static_cast<ImGuiKey>(ImGuiKey_NamedKey_BEGIN + i))
              ? 0x0000000000000001ull << (i % 64)
              : 0;
      if (((i % 64) == 63) || i == (ImGuiKey::ImGuiKey_NamedKey_COUNT - 1)) {
        cmd_input.mInputDownMask[i / 64] = value_mask;
        value_mask = 0;
      }
    }
  }
  // When local UI captures keyboard, mInputDownMask and mInputAnalog stay
  // zero-initialized from the CmdInput constructor.

  // Copy waiting characters inputs — only when local UI is not capturing
  // keyboard. When captured, pending chars are discarded to avoid buffering
  // stale input that would replay when focus returns to remote.
  if (!local_wants_keyboard) {
    size_t added_key_count = std::min<size_t>(
        ArrayCount(cmd_input.mKeyChars) - cmd_input.mKeyCharCount,
        pending_input_chars_.size());
    if (added_key_count) {
      memcpy(&cmd_input.mKeyChars[cmd_input.mKeyCharCount],
             &pending_input_chars_[0], added_key_count * sizeof(ImWchar));
      cmd_input.mKeyCharCount += static_cast<uint16_t>(added_key_count);
      size_t char_remain_count = pending_input_chars_.size() - added_key_count;
      if (char_remain_count > 0) {
        memcpy(&pending_input_chars_[0], &pending_input_chars_[added_key_count],
               char_remain_count * sizeof(ImWchar));
      }
      pending_input_chars_.resize(char_remain_count);
    }
    if (cmd_input.mKeyCharCount > 0) {
      VLOG(1, "[web_client_remote_ui.cc] Queued %u characters to send\n",
           cmd_input.mKeyCharCount);
    }
  } else {
    // Discard any pending characters that were accumulated while local UI
    // had keyboard focus.
    pending_input_chars_.clear();
  }

  if (socket_->SendBinary(std::as_bytes(std::span(&cmd_input, 1))) &&
      cmd_input.mCompressionSkip) {
    request_keyframe_ = false;
  }
}

void RemoteUi::Shutdown() {
  remote_draw_data_.Clear();
  remote_draw_lists_.clear();
  last_uncompressed_frame_.reset();
  socket_.reset();
}

}  // namespace mujoco::studio
