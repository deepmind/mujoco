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

// The browser side of the web viewer's UI stream. Receives the headless Studio
// UI (draw frames and textures) over the /ui WebSocket and replies with browser
// input. Every binary WebSocket message carries one Studio UI wire command.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_
#define MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include "ui_codec.h"
#include "web_client_websocket.h"

namespace mujoco::studio {

// Talks to the Python side over the /ui WebSocket, assembles incoming draw
// frames and textures into Dear ImGui draw data for rendering, and sends the
// browser's input back.
class RemoteUi {
 public:
  using ReadyState = WebSocket::ReadyState;
  using TextureId = uint64_t;

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

  // Starts talking over `socket` (see OpenWebSocket), a new connection to the
  // UI WebSocket, closing any previous one.
  void Connect(std::unique_ptr<WebSocket> socket);

  bool HasSocket() const { return socket_ != nullptr; }

  // Current socket state; browser WebSockets connect and close
  // asynchronously, so this can differ from HasSocket().
  ReadyState ConnectionState() const;

  // True once the server has replied with a compatible Hello handshake.
  bool IsSessionAccepted() const { return handshake_accepted_; }

  // The WebSocket close code once the socket has closed, else 0.
  int CloseCode() const;

  // The mouse cursor requested by the latest remote draw frame.
  ImGuiMouseCursor MouseCursor() const { return mouse_cursor_; }

  // The part of the canvas the remote UI may use, in logical pixels: the
  // whole window minus the browser-local role bar along the bottom. Sent to
  // the remote client as its screen size (so Studio lays out its dock space
  // and status bar inside it) and used to clamp remote clip rects. Set each
  // frame before ReceiveAndProcessCommands() and CaptureAndSendInput().
  void SetScreenArea(float width, float height) {
    screen_area_[0] = width;
    screen_area_[1] = height;
  }

  // Sends the Hello handshake when the socket is newly open, then processes
  // all commands (draw frames, textures, clipboard) received since the last
  // call.
  void ReceiveAndProcessCommands(int frame);

  // Captures ImGui input and sends it to the remote client. Call before
  // ImGui::EndFrame() or ImGui::Render().
  void CaptureAndSendInput();

  // Uploads any CPU-buffered textures without GPU handles yet once the GPU
  // context becomes ready.
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

  // Processes one wire DrawFrame, Texture, or Clipboard message directly.
  void ProcessDrawFrameMessage(netui::BytesView message);
  void ProcessTextureMessage(netui::BytesView message);
  // Puts text the remote client copied on this browser's clipboard, or opens
  // it in a new tab if it is a URL marked as one to open.
  void ProcessClipboardMessage(netui::BytesView message);

 private:
  // Processes one received WebSocket binary message.
  void ProcessMessage(netui::BytesView message);

  // Destroys GPU textures and clears per-session draw/texture/input state.
  void ResetSessionState();

  void RebuildDrawData();

  Callbacks& callbacks_;

  float screen_area_[2] = {0.0f, 0.0f};

  // --- Connection-scoped state, reset on every (re)connect. ----------------

  netui::SerializedFrame last_serialized_frame_;
  std::optional<netui::DecodedFrame> last_decoded_frame_;
  // The index of the newest draw frame received since the last ack.
  std::optional<uint64_t> frame_to_ack_;
  uint64_t last_frame_index_ = 0;
  std::unique_ptr<WebSocket> socket_;
  ReadyState last_state_ = ReadyState::kDisconnected;
  ImGuiMouseCursor mouse_cursor_ = ImGuiMouseCursor_Arrow;
  bool handshake_sent_ = false;
  bool handshake_accepted_ = false;
  // When true, the next Input requests an uncompressed keyframe.
  bool request_keyframe_ = true;

  // --- Session state. -------------------------------------------------------

  ImDrawData remote_draw_data_;
  std::vector<std::unique_ptr<ImDrawList>> remote_draw_lists_;
  std::unordered_map<TextureId, uintptr_t> texture_map_;

  // CPU-side RGBA mirror of each remote texture for applying sub-rect updates.
  struct TextureEntry {
    netui::Bytes pixels;
    uint32_t width = 0;
    uint32_t height = 0;
  };
  std::unordered_map<TextureId, TextureEntry> texture_cpu_;

  // Persistent input state for character accumulation across frames.
  std::vector<uint32_t> pending_input_chars_;
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
