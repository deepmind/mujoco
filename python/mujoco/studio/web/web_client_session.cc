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

#include "web_client_session.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#include <ctime>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "logging.h"
#include "state_payload.h"
#include "web_client_websocket.h"

namespace mujoco::studio {
namespace {

// Session-channel messages sent/received as text frames on the state WebSocket.
constexpr char kMsgRequestControl[] = "request_control";
constexpr char kMsgLeaveQueue[] = "leave_queue";
constexpr char kMsgForceControl[] = "force_control";
constexpr char kMsgHeartbeat[] = "heartbeat";
constexpr char kMsgStateAck[] = "state_ack";
constexpr char kMsgGrant[] = "grant";
constexpr char kMsgMaxSpectatorsPrefix[] = "max_spectators=";
// The model bundles the page may prefetch, "prefetch=<crc32>:<bytes>,...";
// handled by the page's JS (index.html), which owns the model cache.
constexpr char kMsgPrefetchPrefix[] = "prefetch=";
constexpr char kMsgThemePrefix[] = "theme=";

// Liveness heartbeat period. A hidden tab's rendering loop stops, so Update()
// and the heartbeat stop with it. The server kicks spectators (and releases a
// controller with a waiting queue) on silence.
constexpr double kHeartbeatSec = 30.0;

// Minimum time between /ui claim retries (and between role machine steps).
constexpr double kUiRetrySec = 1.0;

// After this many consecutive rejected claims the page stops claiming and
// settles into spectating.
constexpr int kMaxUiRejects = 3;

}  // namespace

double Session::DefaultClock() {
#ifdef __EMSCRIPTEN__
  return emscripten_get_now() / 1000.0;
#else
  struct timespec ts{};
  timespec_get(&ts, TIME_UTC);
  return static_cast<double>(ts.tv_sec) +
         static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

void Session::Connect(std::unique_ptr<WebSocket> socket) {
  socket_ = std::move(socket);
}

void Session::SendText(const char* text) {
  if (socket_) {
    socket_->SendText(text);
  }
}

void Session::SetRole(SessionRole role) {
  role_ = role;
#ifdef __EMSCRIPTEN__
  // Warning: clang-format splits `!==` into `!= =`, so avoid that syntax!
  EM_ASM(
      { Module.isSpectator = !!$0; },
      role == SessionRole::kControlling ? 0 : 1);
#endif
}

bool ParseRoster(const char* text, Roster* roster) {
  Roster parsed;
  char role[16] = {0};
  if (sscanf(text,
             "viewers=%d;role=%15[^;];queue_pos=%d;queue_len=%d;"
             "max_spectators=%d",
             &parsed.viewers, role, &parsed.queue_pos, &parsed.queue_len,
             &parsed.max_spectators) != 5) {
    return false;
  }
  parsed.spectator = strcmp(role, "spectator") == 0;
  *roster = parsed;
  return true;
}

void Session::OnSessionText(const char* text) {
  Roster roster;
  if (ParseRoster(text, &roster)) {
    if (roster.viewers != roster_.viewers ||
        roster.queue_pos != roster_.queue_pos ||
        roster.queue_len != roster_.queue_len) {
      LOG(Info, "Session roster: %s", text);
    }
    roster_ = roster;
    // The roster is authoritative about this page's role. Settling on it
    // (rather than after several rejected /ui retries) makes the SPECTATING
    // banner appear within the first roster (~200ms). Only while claiming with
    // /ui closed: an in-flight claim must not be aborted, and an established
    // controller is never demoted here (the kWsCloseControllerTaken close path
    // handles that).
    if (role_ == SessionRole::kClaiming && roster.spectator) {
      if (remote_ui_state_ == RemoteUiState::kNoSocket ||
          remote_ui_state_ == RemoteUiState::kClosedOrError) {
        LOG(Info, "Roster says spectator; settling");
        SetRole(SessionRole::kSpectating);
        callbacks_.ShutdownRemoteUi();
      }
    }
  } else if (strcmp(text, kMsgGrant) == 0) {
    // The controller slot is reserved for this page. The role flips to
    // kControlling when the claim's socket opens.
    LOG(Info, "Control granted; claiming the controller slot");
    SetRole(SessionRole::kClaiming);
    ui_reject_count_ = 0;
    callbacks_.ConnectRemoteUi();
    // Mark the claim in flight now: the roster broadcast right after the grant
    // arrives before the next frame refreshes remote_ui_state_, and the settle
    // rule above must not shut down the fresh claim.
    remote_ui_state_ = RemoteUiState::kConnecting;
  } else if (strncmp(text, kMsgPrefetchPrefix, strlen(kMsgPrefetchPrefix)) ==
             0) {
#ifdef __EMSCRIPTEN__
    // The page's JS owns the model cache and the background downloads.
    EM_ASM(
        {
          if (Module.onPrefetchHint) Module.onPrefetchHint(UTF8ToString($0));
        },
        text);
#endif
  }
}

void Session::FillView(SessionView* view) const {
  view->role = role_;
  view->viewers = roster_.viewers;
  view->queue_pos = roster_.queue_pos;
  view->queue_len = roster_.queue_len;
  view->max_spectators = roster_.max_spectators;
}

void Session::Update() {
  if (socket_) {
    // server_close_code_ is NOT cleared on socket open: a rejected connection
    // also opens briefly before the server's close frame arrives. It clears on
    // the first received message, which proves the server accepted us.
    for (const WebSocket::Frame& frame : socket_->TakeFrames()) {
      server_close_code_ = 0;
      server_close_reason_.clear();
      if (frame.is_text) {
        OnSessionText(frame.text.c_str());
      } else {
        HandleMessage(frame.binary);
        // Flow control: the server keeps at most one state payload in flight
        // and sends the next (freshest) one only after this ack. Without it, a
        // slow link buffers seconds of stale payloads in the socket and the
        // whole viewer lags by that queue.
        SendText(kMsgStateAck);
      }
    }

    const WebSocket::ReadyState state = socket_->GetReadyState();
    if (state == WebSocket::ReadyState::kClosed ||
        state == WebSocket::ReadyState::kError) {
      const int code = socket_->CloseCode();
      LOG(Info, "State WebSocket closed (code=%d)", code);
      // Codes 4xxx are deliberate server-side closes (e.g. kWsCloseSessionFull)
      // whose reason text is surfaced in the disconnect notice.
      if (code >= 4000 && code <= 4999) {
        server_close_code_ = code;
        server_close_reason_ = socket_->CloseReason();
        LOG(Info,
            "Server ended this connection (code=%d, reason=\"%s\"); retrying "
            "slowly.",
            code, server_close_reason_.c_str());
      }
      socket_.reset();
    }
  }

  const double now = clock_();
  if (Connected() && now - last_heartbeat_time_ >= kHeartbeatSec) {
    last_heartbeat_time_ = now;
    SendText(kMsgHeartbeat);
  }
}

void Session::HandleRemoteUiState(RemoteUiState state, int close_code) {
  remote_ui_state_ = state;
  // No stream to manage, nothing left to claim (settled spectator), or the
  // session itself is down — the /state policies rule then.
  if (state == RemoteUiState::kNoSocket || role_ == SessionRole::kSpectating ||
      server_close_code_ != 0) {
    return;
  }
  if (state == RemoteUiState::kOpen) {
    ui_reject_count_ = 0;
    if (role_ != SessionRole::kControlling) {
      SetRole(SessionRole::kControlling);  // The claim succeeded.
      SendTheme();  // The remote UI takes this page's theme.
    }
    return;
  }
  if (state == RemoteUiState::kClosedOrError) {
    // A kWsCloseControllerTaken close while kControlling means another page
    // took the slot (Steal Control): settle instantly. A rejected claim
    // (kClaiming) retries a few times first, because a reloading controller
    // briefly races its own slot.
    if (close_code == kWsCloseControllerTaken &&
        role_ == SessionRole::kControlling) {
      LOG(Info, "Controller slot taken; spectating");
      SetRole(SessionRole::kSpectating);
      // Also drops the last received UI frame: a page forced out of the
      // controller slot must not keep showing a frozen Studio UI.
      callbacks_.ShutdownRemoteUi();
      return;
    }
    const double now = clock_();
    if (now - last_ui_retry_time_ < kUiRetrySec) {
      return;
    }
    last_ui_retry_time_ = now;
    if (close_code == kWsCloseControllerTaken &&
        ++ui_reject_count_ >= kMaxUiRejects) {
      LOG(Info, "Controller slot taken; spectating");
      SetRole(SessionRole::kSpectating);
      callbacks_.ShutdownRemoteUi();
    } else {
      LOG(Info, "UI WebSocket closed; reconnecting...");
      callbacks_.ConnectRemoteUi();
      remote_ui_state_ = RemoteUiState::kConnecting;  // As on the grant path.
    }
  }
}

void Session::HandleMessage(std::span<const std::byte> message) {
  last_message_time_ = clock_();
  if (!callbacks_.ReadyForPayload()) {
    return;
  }
  bytes_accum_ += message.size();

  StatePayloadView view;
  if (!ParseStatePayload(message.data(), message.size(), &view)) {
    LOG(Error, "Malformed state payload (%zu bytes); dropping", message.size());
    return;
  }

  if (!model_crc32_.has_value()) {
    model_crc32_ = view.model_crc32;
  } else if (view.model_crc32 != *model_crc32_) {
    LOG(Info, "Model changed on the Python side (ident %u -> %u); reloading",
        *model_crc32_, view.model_crc32);
    // Set the new CRC so subsequent payloads don't re-trigger while the new
    // model is being fetched.
    model_crc32_ = view.model_crc32;
    callbacks_.OnModelChanged(view.model_crc32);
    // Fall through: this payload already carries the new model's first state
    // and camera; OnPayload buffers the state while the download runs (or
    // applies it right away when the page had the bundle cached).
  }

  callbacks_.OnPayload(view);
}

void Session::RequestControl() { SendText(kMsgRequestControl); }

void Session::LeaveQueue() { SendText(kMsgLeaveQueue); }

void Session::StealControl() { SendText(kMsgForceControl); }

void Session::ReleaseControl() {
  // Become a spectator; the server grants the slot down the queue.
  callbacks_.ShutdownRemoteUi();
  SetRole(SessionRole::kSpectating);
}

void Session::SetCameraMode(int mode) { callbacks_.SetCameraMode(mode); }

void Session::SetFpsCap(int fps_cap) { callbacks_.SetFpsCap(fps_cap); }

void Session::SetMaxSpectators(int count) {
  char msg[48];
  snprintf(msg, sizeof(msg), "%s%d", kMsgMaxSpectatorsPrefix, count);
  SendText(msg);
}

void Session::SetTheme(const char* theme) {
  theme_ = theme;
  SendTheme();
}

void Session::SendTheme() {
  if (theme_.empty()) return;
  SendText((std::string(kMsgThemePrefix) + theme_).c_str());
}

void Session::SendCamera(const mjvCamera& camera, uint32_t generation) {
  if (socket_) {
    ClientCameraMessage message = MakeClientCameraMessage(camera, generation);
    socket_->SendBinary(std::as_bytes(std::span(&message, 1)));
  }
}
}  // namespace mujoco::studio
