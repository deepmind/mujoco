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

// The session: this page's relationship with the Python-side viewer.
//
// Owns the /state WebSocket (simulation payloads in, control messages out),
// the session wire protocol (roster, grants, heartbeats, acks), the role
// state machine (claiming -> controlling / spectating), and the
// model-change / page-reload / close-code policies. Everything outside the
// session goes through the Callbacks interface: applying a payload to the
// scene, and driving the remote UI stream on role transitions. The session
// also implements SessionActions, so the role window's intents land here
// directly.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_SESSION_H_
#define MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_SESSION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "role_transitions.h"
#include "state_payload.h"
#include "web_client_websocket.h"

namespace mujoco::studio {

// Custom WebSocket close codes in range 4xxx.
inline constexpr int kWsCloseControllerTaken = 4001;  // /ui: other tab controls
inline constexpr int kWsCloseSessionFull = 4002;  // /state: spectator limit hit
inline constexpr int kWsCloseInactive = 4003;     // /state: hidden tab kicked
inline constexpr int kWsCloseNotController = 4004;  // /drop: controller only
inline constexpr int kWsCloseViewerError = 4005;    // /state: viewer error

// Read-only snapshot of the session, passed to the local UI each frame.
struct SessionView {
  uint64_t gui_bytes_per_sec = 0;
  uint64_t sim_bytes_per_sec = 0;
  // Staleness of the state stream, for the role bar's stall indicator;
  // negative before the first payload.
  double seconds_since_last_payload = -1.0;
  size_t bytes_downloaded = 0;
  size_t total_bytes = 0;
  SessionRole role = SessionRole::kClaiming;
  int viewers = 0;
  int queue_pos = 0;  // 1-based position in the control queue; 0 = unqueued.
  int queue_len = 0;
  int max_spectators = 0;
  int camera_mode = 0;  // [SpectatorCamMode].
  int fps_cap = 60;     // 30, 60, 120, or 0 (Uncapped).
  int retry_count = 0;
  bool have_remote_frame = false;
  bool is_downloading = true;
};

// User intent reported by the role window.
class SessionActions {
 public:
  virtual ~SessionActions() = default;
  virtual void RequestControl() = 0;
  virtual void LeaveQueue() = 0;
  virtual void StealControl() = 0;
  virtual void ReleaseControl() = 0;
  virtual void SetCameraMode(int mode) = 0;      // mode is [SpectatorCamMode].
  virtual void SetMaxSpectators(int count) = 0;  // Already clamped by the UI.
  virtual void SetFpsCap(int fps_cap) = 0;  // 30, 60, 120, or 0 (Uncapped).
};

// Server membership broadcast sent as a text frame on /state whenever viewers,
// roles, or control queue positions change.
struct Roster {
  int viewers = 0;
  int queue_pos = 0;  // 1-based position in the control queue; 0 = unqueued.
  int queue_len = 0;
  int max_spectators = 8;  // Runtime spectator limit.
  bool spectator = false;  // The server's view: true = not the controller.
};

// Parses a roster line; returns false when text is not a roster.
bool ParseRoster(const char* text, Roster* roster);

// The remote UI stream's connection state, reported to the role state
// machine by the app once per frame.
enum class RemoteUiState {
  kNoSocket = 0,   // No connection attempt exists.
  kConnecting,     // In flight (or closing); the machine waits.
  kOpen,           // The claim succeeded: this page controls.
  kClosedOrError,  // Rejected or dropped; the machine retries or settles.
};

class Session : public SessionActions {
 public:
  // Everything the session needs from the rest of the application.
  class Callbacks {
   public:
    virtual ~Callbacks() = default;
    // Payloads are dropped until this returns true (model loaded).
    virtual bool ReadyForPayload() = 0;
    // Applies a parsed payload to the application.
    virtual void OnPayload(const StatePayloadView& view) = 0;
    // Server swapped models: fetch (or take from the page's cache) and load
    // the bundle whose crc32 the state payload now carries, in place.
    virtual void OnModelChanged(uint32_t model_crc32) = 0;
    // Role transition: claim the controller slot.
    virtual void ConnectRemoteUi() = 0;
    // Role transition: drop the stream when spectating
    virtual void ShutdownRemoteUi() = 0;
    // Spectator camera mode change.
    virtual void SetCameraMode(int mode) = 0;
    // Client framerate cap change (30, 60, 120, or 0 for Uncapped).
    virtual void SetFpsCap(int fps_cap) {}
  };

  using ClockFn = double (*)();

  explicit Session(Callbacks& callbacks, ClockFn clock = DefaultClock)
      : callbacks_(callbacks), clock_(clock) {}

  // Starts talking over `socket` (see OpenWebSocket), a new connection to the
  // /state WebSocket, closing any previous one.
  void Connect(std::unique_ptr<WebSocket> socket);

  // Records the CRC32 of the model this page actually loaded.
  // Must match zlib.crc32 (used to compute model_crc32 in web_viewer.py).
  void SetModelCrc32(uint32_t crc) { model_crc32_ = crc; }

  // True while a connect attempt exists; used to pace reconnects.
  bool HasSocket() const { return socket_ != nullptr; }

  // True only while the WebSocket is actually open.
  bool Connected() const {
    return socket_ && socket_->GetReadyState() == WebSocket::ReadyState::kOpen;
  }

  // Close code from the server deliberately ending this connection (4000-4999),
  // or 0 if none. Cleared on the next received message.
  int ServerCloseCode() const { return server_close_code_; }

  // Reason text sent with ServerCloseCode(), or empty when ServerCloseCode() is
  // 0.
  const std::string& ServerCloseReason() const { return server_close_reason_; }

  // Returns the bytes received since the last call and resets the counter.
  uint64_t ConsumeByteCount() {
    uint64_t bytes = bytes_accum_;
    bytes_accum_ = 0;
    return bytes;
  }

  // Wall-clock seconds of the last received message, or 0 before the first one.
  double LastMessageTime() const { return last_message_time_; }

  SessionRole Role() const { return role_; }

  // Fills the role and roster fields of the view.
  void FillView(SessionView* view) const;

  // Drains incoming /state frames, tracks socket close codes, and sends the
  // ~30s liveness heartbeat; call once per frame.
  void Update();

  // Feeds the role state machine the remote UI stream's connection state; call
  // once per frame.
  void HandleRemoteUiState(RemoteUiState state, int close_code);

  // Parses one WebSocket message and applies the model-change/reload policy.
  void HandleMessage(std::span<const std::byte> message);

  // SessionActions (used to implement the role window UI).
  void RequestControl() override;
  void LeaveQueue() override;
  void StealControl() override;
  void ReleaseControl() override;
  void SetCameraMode(int mode) override;
  void SetMaxSpectators(int count) override;
  void SetFpsCap(int fps_cap) override;

  // Remembers this page's theme name ("light", "dark" or "classic") and sends
  // it to the server, which applies it to the remote UI if this page controls.
  // Sent again whenever this page gains control, so the remote UI matches the
  // page's theme from the first frame.
  void SetTheme(const char* theme);

  // Sends the controller's locally moved camera to the viewer as a
  // ClientCameraMessage (state_payload.h). The viewer adopts it unless it
  // changed the camera itself since `generation`
  // (ViewerInputState::camera_generation; see client_camera.py).
  void SendCamera(const mjvCamera& camera, uint32_t generation);

 private:
  static double DefaultClock();

  // Sends a session message (control requests, acks, activity reports) to the
  // server as a text frame. Dropped silently while not connected.
  void SendText(const char* text);

  // Sends theme_ (see SetTheme).
  void SendTheme();

  // Updates the role and mirrors it into JS (Module.isSpectator), which gates
  // controller-only page behavior (model drag-and-drop upload).
  void SetRole(SessionRole role);

  // Routes a session text frame: roster updates, control grants, and the
  // list of model bundles the page may prefetch (handed to the page's JS).
  void OnSessionText(const char* text);

  Callbacks& callbacks_;
  ClockFn clock_;

  std::unique_ptr<WebSocket> socket_;
  std::string server_close_reason_;
  std::string theme_;  // See SetTheme.

  uint64_t bytes_accum_ = 0;
  double last_message_time_ = 0;
  double last_ui_retry_time_ = 0;
  double last_heartbeat_time_ = 0;

  // Role state machine and roster.
  Roster roster_;
  SessionRole role_ = SessionRole::kClaiming;

  // CRC32 of the model this page loaded. When the payload's CRC changes, the
  // Python side has swapped models so we trigger an in-place reload that
  // refetches /model and reinitializes the scene.
  std::optional<uint32_t> model_crc32_;

  int server_close_code_ = 0;
  // Consecutive rejected /ui claims; the page eventually stops claiming and
  // settles into spectating.
  int ui_reject_count_ = 0;
  RemoteUiState remote_ui_state_ = RemoteUiState::kNoSocket;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_SESSION_H_
