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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <imgui.h>
#include "studio/ux/imgui_widgets.h"
#include <NetImgui_CmdPackets.h>
#include <NetImgui_Network.h>
#include "google/logging.h"
#include "google/network_status.h"

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

using namespace NetImgui::Internal;

namespace {

// Delta-compress the GUI stream (relayed to the client via CmdInput).
constexpr bool kUseCompression = true;

// Clipboard text with this prefix is a URL for this browser to open in a new
// tab rather than something to copy: the headless server has no browser of its
// own, and the clipboard is the only text channel NetImgui offers it.
constexpr char kOpenUrlPrefix[] = "\x01OPEN_URL:";

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

void LogCmdReceived(CmdHeader::eCommands cmd_type, uint32_t cmd_size,
                    int draw_frames, int textures,
                    const PendingCom& pending_receive) {
  if (cmd_type == CmdHeader::eCommands::Version) {
    if (cmd_size < sizeof(CmdVersion)) {
      LOG(Warning, "Received truncated CmdVersion (size=%u)", cmd_size);
      return;
    }
    const CmdVersion* ver =
        reinterpret_cast<const CmdVersion*>(pending_receive.pCommand);
    LOG(Info,
        "Received CmdVersion from client: name='%s', version=%d, wchar_size=%d",
        ver->mClientName, static_cast<int>(ver->mVersion), ver->mWCharSize);
  } else if (cmd_type == CmdHeader::eCommands::DrawFrame) {
    VLOG(2, "Received DrawFrame #%d (size=%u)", draw_frames, cmd_size);
  } else if (cmd_type == CmdHeader::eCommands::Texture) {
    VLOG(2, "Received Texture #%d (size=%u)", textures, cmd_size);
  } else if (cmd_type == CmdHeader::eCommands::Background) {
    VLOG(2, "Received Background cmd (size=%u)", cmd_size);
  } else if (cmd_type == CmdHeader::eCommands::Clipboard) {
    VLOG(2, "Received Clipboard cmd (size=%u)", cmd_size);
  } else if (cmd_type != CmdHeader::eCommands::Count &&
             cmd_type != CmdHeader::eCommands::Input) {
    VLOG(2, "Received UNKNOWN cmd: type=%d, size=%u",
         static_cast<int>(cmd_type), cmd_size);
  }
}

}  // namespace

void RemoteUi::Connect(const std::string& url) {
  if (socket_) {
    Network::Disconnect(socket_);
    socket_ = nullptr;
  }

  // The port argument is unused when a full URL is passed.
  LOG(Info, "Connecting to WebSocket at %s", url.c_str());
  socket_ = Network::Connect(url.c_str(), 0);
  LOG(Info, "Network::Connect returned: socket=%p",
      static_cast<void*>(socket_));
}

RemoteUi::ReadyState RemoteUi::ConnectionState() const {
  return Network::GetReadyState(socket_);
}

int RemoteUi::CloseCode() const { return Network::GetCloseCode(socket_); }

void RemoteUi::ReceiveAndProcessCommands(int frame) {
  if (!socket_) return;

  const ReadyState state = ConnectionState();
  if (last_state_ != state) {
    LOG(Info, "WebSocket status changed: '%s' -> '%s' (frame %d)",
        Network::ReadyStateName(last_state_), Network::ReadyStateName(state),
        frame);
    last_state_ = state;
  }
  VLOG(1,
       "Frame %d: status='%s', handshake=%s, cmds=%d, draws=%d, textures=%d, "
       "has_draw_data=%s",
       frame, Network::ReadyStateName(state),
       handshake_sent_ ? "sent" : "not_sent", total_cmds_received_,
       draw_frames_received_, textures_received_,
       remote_draw_data_.Valid ? "yes" : "no");

  if (state == ReadyState::kOpen) {
    if (!handshake_sent_) {
      CmdVersion cmd_version;
      StringCopy(cmd_version.mClientName, "MuJoCo Web Viewer");
      LOG(Info,
          "Sending CmdVersion handshake: size=%u, type=%d, version=%d, "
          "wchar_size=%d, name='%s'",
          cmd_version.mSize, static_cast<int>(cmd_version.mType),
          static_cast<int>(cmd_version.mVersion), cmd_version.mWCharSize,
          cmd_version.mClientName);

      PendingCom pending_send;
      pending_send.pCommand = &cmd_version;
      pending_send.SizeCurrent = 0;

      int send_attempts = 0;
      while (!pending_send.IsDone() && !pending_send.IsError()) {
        const size_t before = pending_send.SizeCurrent;
        Network::DataSend(socket_, pending_send);
        send_attempts++;
        if (pending_send.SizeCurrent == before) {
          // No progress: the socket already reads OPEN (from emscripten's
          // ready state) but the send backend is not ready yet, because the
          // open callback that sets mConnected has not run. DataSend then
          // returns with neither progress nor error, so this loop would spin
          // the browser's main thread forever. Stop and retry on a later
          // frame instead (handshake_sent_ stays false below).
          break;
        }
      }
      LOG(Info,
          "CmdVersion send: done=%s, error=%s, attempts=%d, bytes_sent=%zu",
          pending_send.IsDone() ? "true" : "false",
          pending_send.IsError() ? "true" : "false", send_attempts,
          static_cast<size_t>(pending_send.SizeCurrent));
      if (pending_send.IsDone()) {
        handshake_sent_ = true;
        // Fresh connection: the client may resume delta compression against
        // a frame from a previous session; ask for an uncompressed keyframe.
        request_keyframe_ = true;
      }
    }
  } else {
    if (handshake_sent_) {
      LOG(Info, "Resetting handshake (status='%s')",
          Network::ReadyStateName(state));
    }
    handshake_sent_ = false;
  }

  // Network receive — drain all available data in one frame.
  const bool is_connected = (state == ReadyState::kOpen);

  // If the connection has closed, discard any buffered data immediately
  // rather than churning through stale commands for several seconds.
  if (was_connected_ && !is_connected) {
    LOG(Info, "Connection lost (status='%s'). Discarding buffered data.",
        Network::ReadyStateName(state));
    // Reset any in-progress receive.
    if (pending_receive_.bAutoFree)
      netImguiDeleteSafe(pending_receive_.pCommand);
    pending_receive_ = PendingCom();
    was_connected_ = false;
  }
  if (is_connected) was_connected_ = true;

  int max_commands_per_frame = 64;
  int cmds_this_frame = 0;
  bool had_pending_data = Network::DataReceivePending(socket_);
  if (had_pending_data) {
    VLOG(1, "Frame %d: data pending on socket", frame);
  }

  while (is_connected && max_commands_per_frame-- > 0) {
    if (pending_receive_.IsReady()) {
      cmd_pending_read_ = CmdPendingRead();
      pending_receive_.pCommand = &cmd_pending_read_;
      pending_receive_.SizeMax = sizeof(cmd_pending_read_);
      pending_receive_.bAutoFree = false;
    }

    if (!Network::DataReceivePending(socket_)) break;

    Network::DataReceive(socket_, pending_receive_);

    // Received the header of a command bigger than the header: allocate it.
    if (pending_receive_.IsBiggerBufferNeeded()) {
      VLOG(2, "Allocating %u bytes for incoming cmd type=%d",
           pending_receive_.pCommand->mSize,
           static_cast<int>(pending_receive_.pCommand->mType));
      CmdPendingRead* cmd_header = reinterpret_cast<CmdPendingRead*>(
          netImguiSizedNew<uint8_t>(pending_receive_.pCommand->mSize));
      *cmd_header = cmd_pending_read_;
      pending_receive_.pCommand = cmd_header;
      pending_receive_.SizeMax = cmd_header->mSize;
      pending_receive_.bAutoFree = true;
    }

    if (!pending_receive_.IsDone()) {
      if (pending_receive_.IsError()) {
        LOG(Error, "Receive ERROR: cmd_size=%u, got=%zu, type=%d",
            pending_receive_.pCommand->mSize,
            static_cast<size_t>(pending_receive_.SizeCurrent),
            static_cast<int>(pending_receive_.pCommand->mType));
        if (pending_receive_.bAutoFree)
          netImguiDeleteSafe(pending_receive_.pCommand);
        pending_receive_ = PendingCom();
      }
      continue;
    }

    // Command fully received — dispatch.
    bytes_accum_ += pending_receive_.pCommand->mSize;
    cmds_this_frame++;
    total_cmds_received_++;
    CmdHeader::eCommands cmd_type = pending_receive_.pCommand->mType;
    LogCmdReceived(cmd_type, pending_receive_.pCommand->mSize,
                   draw_frames_received_, textures_received_, pending_receive_);

    if (cmd_type == CmdHeader::eCommands::Count) {
      // CmdPendingRead sentinel — skip silently.
    } else if (cmd_type == CmdHeader::eCommands::DrawFrame) {
      draw_frames_received_++;
      ProcessCmdDrawFrame(
          reinterpret_cast<CmdDrawFrame*>(pending_receive_.pCommand));
    } else if (cmd_type == CmdHeader::eCommands::Texture) {
      textures_received_++;
      ProcessCmdTexture(
          reinterpret_cast<CmdTexture*>(pending_receive_.pCommand));
    } else if (cmd_type == CmdHeader::eCommands::Clipboard) {
      ProcessCmdClipboard(
          reinterpret_cast<CmdClipboard*>(pending_receive_.pCommand));
    }

    if (pending_receive_.bAutoFree)
      netImguiDeleteSafe(pending_receive_.pCommand);
    pending_receive_ = PendingCom();
  }

  VLOG(2, "Frame %d: processed %d commands", frame, cmds_this_frame);
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
  for (auto& [tex_id, entry] : texture_cpu_) {
    uintptr_t& local_tex = texture_map_[tex_id];
    if (local_tex == 0 && !entry.pixels.empty()) {
      local_tex = callbacks_.UploadTexture(
          local_tex, reinterpret_cast<const std::byte*>(entry.pixels.data()),
          entry.width, entry.height);
      LOG(Info, "UpdateTextures: uploaded tex_id=%lu -> filament=%lu",
          static_cast<unsigned long>(tex_id),
          static_cast<unsigned long>(local_tex));
    }
  }
}

void RemoteUi::ProcessCmdTexture(CmdTexture* cmd_texture) {
  if (!cmd_texture) return;

  if (!cmd_texture->ToPointers()) {
    LOG(Warning, "ProcessCmdTexture: dropping invalid texture command");
    return;
  }

  ClientTextureID tex_id = cmd_texture->mTextureClientID;
  VLOG(
      1,
      "ProcessCmdTexture: client_tex_id=%lu, status=%d, size=%ux%u, format=%d, "
      "offset=%u,%u, map_size=%zu",
      static_cast<unsigned long>(tex_id),
      static_cast<int>(cmd_texture->mStatus),
      static_cast<uint32_t>(cmd_texture->mWidth),
      static_cast<uint32_t>(cmd_texture->mHeight),
      static_cast<int>(cmd_texture->mFormat),
      static_cast<uint32_t>(cmd_texture->mOffsetX),
      static_cast<uint32_t>(cmd_texture->mOffsetY), texture_map_.size());

  uintptr_t& local_tex = texture_map_[tex_id];

  if (cmd_texture->mStatus == CmdTexture::eType::Destroy) {
    if (local_tex != 0 && callbacks_.GpuReady()) {
      // Pass nullptr pixels to destroy the GPU texture.
      callbacks_.UploadTexture(local_tex, nullptr, 0, 0);
    }
    local_tex = 0;
    texture_cpu_.erase(tex_id);
    return;
  }

  uint8_t* pixels = cmd_texture->mpTextureData.Get();
  if (!pixels) return;

  // All of width/height/size come off the wire, so compute in size_t (no
  // 32-bit overflow) and verify the command actually carries the pixel
  // bytes before reading them — a short or corrupt command must not make
  // the memcpy/expand below read past the command buffer.
  const uint32_t patch_w = cmd_texture->mWidth;
  const uint32_t patch_h = cmd_texture->mHeight;
  const size_t pixel_count = static_cast<size_t>(patch_w) * patch_h;
  const size_t data_size = cmd_texture->mSize >= sizeof(CmdTexture)
                               ? cmd_texture->mSize - sizeof(CmdTexture)
                               : 0;
  const size_t expected_rgba = pixel_count * 4;

  // Detect actual format by data size, not format tag (which can be wrong).
  const bool is_a8 = (cmd_texture->mFormat == 1) ||
                     (data_size == pixel_count && data_size != expected_rgba);
  const size_t needed = is_a8 ? pixel_count : expected_rgba;
  if (pixel_count == 0 || data_size < needed) {
    LOG(Warning,
        "ProcessCmdTexture: tex_id=%lu declares %ux%u but carries only %zu "
        "bytes; dropping",
        static_cast<unsigned long>(tex_id), patch_w, patch_h, data_size);
    return;
  }

  // Convert incoming pixels to RGBA (Filament requires RGBA).
  // For A8 font atlas data, expand each byte to (255, 255, 255, alpha).
  std::vector<uint8_t> rgba_pixels(pixel_count * 4);
  if (is_a8) {
    for (size_t p = 0; p < pixel_count; ++p) {
      rgba_pixels[p * 4 + 0] = 255;
      rgba_pixels[p * 4 + 1] = 255;
      rgba_pixels[p * 4 + 2] = 255;
      rgba_pixels[p * 4 + 3] = pixels[p];
    }
  } else {
    memcpy(rgba_pixels.data(), pixels, expected_rgba);
  }

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
    if (static_cast<size_t>(off_x) + patch_w > entry.width ||
        static_cast<size_t>(off_y) + patch_h > entry.height) {
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
      memcpy(&entry.pixels[dst_offset], &rgba_pixels[src_offset],
             static_cast<size_t>(patch_w) * 4);
    }
  }

  // Upload the full CPU-side texture to the GPU if the context is ready.
  // If it isn't yet (model still loading), the texture stays in texture_cpu_
  // and will be flushed by UpdateTextures() later.
  if (callbacks_.GpuReady()) {
    local_tex = callbacks_.UploadTexture(
        local_tex, reinterpret_cast<const std::byte*>(entry.pixels.data()),
        entry.width, entry.height);
    VLOG(1, "uploadGuiImage for client_tex_id=%lu -> filament=%lu",
         static_cast<unsigned long>(tex_id),
         static_cast<unsigned long>(local_tex));
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

  // Take ownership to prevent pending_receive_ from deleting it prematurely.
  pending_receive_.bAutoFree = false;
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
    // Start from zeroed commands (ImVector::resize() doesn't construct), so
    // the draws skipped below stay empty.
    cmd_list->CmdBuffer.resize(0);
    cmd_list->CmdBuffer.resize(draw_group.mDrawCount, ImDrawCmd());
    cmd_list->Flags =
        ImDrawListFlags_AllowVtxOffset | ImDrawListFlags_AntiAliasedLines |
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
    ImDrawCmd* command_dst = cmd_list->CmdBuffer.Data;
    for (uint32_t draw_idx(0); draw_idx < draw_group.mDrawCount; ++draw_idx) {
      uint32_t idx_off = draw_src[draw_idx].mIdxOffset;
      uint32_t elem_count = draw_src[draw_idx].mIdxCount;

      // The draw range comes off the wire: skip (leave empty) draws that
      // would index past the buffer.
      if (static_cast<uint64_t>(idx_off) + elem_count >
          static_cast<uint64_t>(cmd_list->IdxBuffer.Size)) {
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

      command_dst[draw_idx].ClipRect.x = cx;
      command_dst[draw_idx].ClipRect.y = cy;
      command_dst[draw_idx].ClipRect.z = cz;
      command_dst[draw_idx].ClipRect.w = cw;
      command_dst[draw_idx].VtxOffset = 0;
      command_dst[draw_idx].IdxOffset = idx_off;
      command_dst[draw_idx].ElemCount = elem_count;
      command_dst[draw_idx].UserCallback = nullptr;
      command_dst[draw_idx].UserCallbackData = nullptr;

      // Map remote ClientTextureID -> local GL handle.
      ClientTextureID client_tex_id = draw_src[draw_idx].mClientTexId;
      auto it = texture_map_.find(client_tex_id);
      if (it != texture_map_.end() && it->second != 0) {
        command_dst[draw_idx].TexRef._TexData = nullptr;
        command_dst[draw_idx].TexRef._TexID =
            static_cast<ImTextureID>(it->second);
      } else {
        LogUnmappedTexture(client_tex_id, texture_map_.size(), draw_idx,
                           texture_map_);
        // Skip draw commands with unmapped textures.
        command_dst[draw_idx].ElemCount = 0;
      }
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
// netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp
void RemoteUi::CaptureAndSendInput() {
  if (!socket_) return;

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
  cmd_input.mScreenSize[0] = static_cast<uint16_t>(screen_area_[0]);
  cmd_input.mScreenSize[1] = static_cast<uint16_t>(screen_area_[1]);
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
    cmd_input.mMousePos[0] = static_cast<int16_t>(io.MousePos.x);
    cmd_input.mMousePos[1] = static_cast<int16_t>(io.MousePos.y);
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

  PendingCom pending_send;
  pending_send.pCommand = &cmd_input;
  pending_send.SizeCurrent = 0;
  Network::DataSend(socket_, pending_send);
  if (pending_send.IsDone() && cmd_input.mCompressionSkip) {
    request_keyframe_ = false;
  }
}

void RemoteUi::Shutdown() {
  remote_draw_data_.Clear();
  remote_draw_lists_.clear();
  last_uncompressed_frame_.reset();
  if (socket_) {
    Network::Disconnect(socket_);
    socket_ = nullptr;
  }
}

}  // namespace mujoco::studio
