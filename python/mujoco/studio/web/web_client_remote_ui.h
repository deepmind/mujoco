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

// The NetImgui client side of the web viewer. Receives the headless Studio UI
// (draw frames/textures) over the /ui WebSocket and replies with browser input.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_
#define MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <NetImgui_Api.h>
#include <NetImgui_CmdPackets.h>
#include <NetImgui_Network.h>
#include "google/network_status.h"

namespace mujoco::studio {

// Deleter for objects allocated by NetImgui.
struct NetImguiDeleter {
  template <typename T>
  void operator()(T* ptr) const {
    NetImgui::Internal::netImguiDelete(ptr);
  }
};

// The browser side of the streamed ImGui UI: connects to the UI WebSocket,
// receives NetImgui draw frames and textures from the Python side, assembles
// them into ImGui draw data for rendering, and sends the browser's input back.
class RemoteUi {
 public:
  using SocketInfo = NetImgui::Internal::Network::SocketInfo;
  using ReadyState = NetImgui::Internal::Network::ReadyState;
  using ClientTextureID = NetImgui::Internal::ClientTextureID;

  // The renderer-facing callbacks of the link; the app implements them once.
  // The link reaches the GPU only through it, so the protocol logic stays
  // renderer-agnostic.
  class Callbacks {
   public:
    virtual ~Callbacks() = default;

    // Uploads a full RGBA texture and returns the GPU handle; nullptr
    // pixels destroy the texture backing `current`.
    virtual uintptr_t UploadTexture(uintptr_t current, const std::byte* rgba,
                                    uint32_t width, uint32_t height) = 0;

    // True once the GPU context can accept texture uploads.
    virtual bool GpuReady() = 0;
  };

  explicit RemoteUi(Callbacks& callbacks) : callbacks_(callbacks) {}
  ~RemoteUi() { Shutdown(); }

  // (Re)connects to the UI WebSocket. Disconnects any existing socket first.
  void Connect(const std::string& url);

  bool HasSocket() const { return socket_ != nullptr; }

  // Current socket state; browser WebSockets connect and close
  // asynchronously, so this can differ from HasSocket() (see
  // google/network_status.h).
  ReadyState ConnectionState() const;

  // The WebSocket close code once the socket has closed, else 0.
  int CloseCode() const;

  // The part of the canvas the remote UI may use, in logical pixels: the
  // whole window minus the browser-local role bar along the bottom. Sent to
  // the remote client as its screen size (so Studio lays out its dock space
  // and status bar inside it) and used to clamp remote clip rects. Set each
  // frame before ReceiveAndProcessCommands() and CaptureAndSendInput().
  void SetScreenArea(float width, float height) {
    screen_area_[0] = width;
    screen_area_[1] = height;
  }

  // Sends the CmdVersion handshake when the socket is newly open, then drains
  // all incoming commands (draw frames, textures, clipboard) for this frame.
  void ReceiveAndProcessCommands(int frame);

  // Captures ImGui input and sends it to the remote client. Must run before
  // ImGui::EndFrame(), because EndFrame() clears the io.InputQueueCharacters
  // queue this reads (note that ImGui::Render() calls EndFrame() implicitly).
  void CaptureAndSendInput();

  // Invalidates cached GPU texture IDs and re-uploads all CPU-buffered
  // textures (font atlas, streamed UI images) after a Filament context reset.
  void UpdateTextures();

  void Shutdown();

  // The latest assembled remote draw data, or nullptr before the first frame.
  ImDrawData* RemoteDrawData() {
    return remote_draw_data_.Valid ? &remote_draw_data_ : nullptr;
  }

  // Returns the bytes received since the last call and resets the counter.
  uint64_t ConsumeByteCount() {
    uint64_t bytes = bytes_accum_;
    bytes_accum_ = 0;
    return bytes;
  }

  void ProcessCmdDrawFrame(NetImgui::Internal::CmdDrawFrame* cmd);
  void ProcessCmdTexture(NetImgui::Internal::CmdTexture* cmd);
  // Puts text the remote client copied on this browser's clipboard, or opens
  // it in a new tab if it is a URL marked as one to open.
  void ProcessCmdClipboard(NetImgui::Internal::CmdClipboard* cmd);

 private:
  Callbacks& callbacks_;

  float screen_area_[2] = {0.0f, 0.0f};

  // --- Connection-scoped state, reset on every (re)connect. ----------------

  SocketInfo* socket_ = nullptr;
  bool handshake_sent_ = false;
  bool was_connected_ = false;
  ReadyState last_state_ = ReadyState::kDisconnected;
  NetImgui::Internal::PendingCom pending_receive_;
  NetImgui::Internal::CmdPendingRead cmd_pending_read_;

  // When true, the next CmdInput asks the client to send one uncompressed
  // draw frame (CmdInput::mCompressionSkip). Set on (re)connect and whenever
  // a delta frame arrives whose reference frame we don't have — without it,
  // a browser joining mid-stream drops every delta-compressed frame forever
  // and the remote UI never appears. Mirrors mbCompressionSkipOncePending in
  // NetImguiServer RemoteClient.
  bool request_keyframe_ = true;

  std::unique_ptr<NetImgui::Internal::CmdDrawFrame, NetImguiDeleter>
      last_uncompressed_frame_;

  // --- Session state. -------------------------------------------------------

  // The latest remote frame as Dear ImGui draw data, with one ImDrawList per
  // remote window (an ImDrawData only points at lists owned elsewhere). The
  // lists are reused from frame to frame while their windows stay open.
  ImDrawData remote_draw_data_;
  std::vector<std::unique_ptr<ImDrawList>> remote_draw_lists_;
  std::unordered_map<ClientTextureID, uintptr_t> texture_map_;

  // CPU-side texture cache so partial updates from NetImgui can be patched into
  // these buffers before re-uploading the full texture (partial updates into
  // GPU textures are not supported by Filament).
  struct TextureEntry {
    std::vector<uint8_t> pixels;  // Full RGBA pixel data
    uint32_t width = 0;
    uint32_t height = 0;
  };
  std::unordered_map<ClientTextureID, TextureEntry> texture_cpu_;

  // Persistent input state for character accumulation across frames.
  std::vector<ImWchar> pending_input_chars_;
  float mouse_wheel_pos_[2] = {0.0f, 0.0f};
  uint16_t last_screen_size_[2] = {0, 0};

  // --- Telemetry. -----------------------------------------------------------

  uint64_t bytes_accum_ = 0;
  int total_cmds_received_ = 0;
  int draw_frames_received_ = 0;
  int textures_received_ = 0;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_
