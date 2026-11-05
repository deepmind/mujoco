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

#include <emscripten.h>

#include <cstdio>
#include <cstring>

#include "google/logging.h"

namespace mujoco::studio {

// Session-channel messages sent/received as text frames on the state WebSocket.
constexpr char kMsgRequestControl[] = "request_control";
constexpr char kMsgLeaveQueue[] = "leave_queue";
constexpr char kMsgForceControl[] = "force_control";
constexpr char kMsgHeartbeat[] = "heartbeat";
constexpr char kMsgStateAck[] = "state_ack";
constexpr char kMsgGrant[] = "grant";
constexpr char kMsgMaxSpectatorsPrefix[] = "max_spectators=";

// Liveness heartbeat period. A hidden tab's rendering loop stops, so Update()
// and the heartbeat stop with it. The server kicks spectators (and releases a
// controller with a waiting queue) on silence.
constexpr double kHeartbeatSec = 30.0;

// Minimum time between /ui claim retries (and between role machine steps).
constexpr double kUiRetrySec = 1.0;

// After this many consecutive rejected claims the page stops claiming and
// settles into spectating.
constexpr int kMaxUiRejects = 3;

EM_BOOL Session::OnWsMessage(int event_type,
                             const EmscriptenWebSocketMessageEvent* event,
                             void* user_data) {
  auto* session = static_cast<Session*>(user_data);
  session->server_close_code_ = 0;  // Accepted; hide the disconnect notice.
  session->server_close_reason_.clear();
  if (event->isText) {
    // Text frames carry session metadata; emscripten null-terminates them.
    session->OnSessionText(reinterpret_cast<const char*>(event->data));
  } else {
    session->HandleMessage(event->data, event->numBytes);
    // Flow control: the server keeps at most one state payload in flight and
    // sends the next (freshest) one only after this ack. Without it, a slow
    // link buffers seconds of stale payloads in the socket and the whole viewer
    // lags by that queue.
    session->SendText(kMsgStateAck);
  }
  return EM_TRUE;
}

EM_BOOL Session::OnWsOpen(int event_type,
                          const EmscriptenWebSocketOpenEvent* event,
                          void* user_data) {
  auto* session = static_cast<Session*>(user_data);
  session->connected_ = true;
  // server_close_code_ is NOT cleared here: a rejected connection also fires
  // open before the server's closing code arrives. It clears on the first
  // received message, which proves the server accepted us.
  LOG(Info, "State WebSocket connected");
  return EM_TRUE;
}

EM_BOOL Session::OnWsError(int event_type,
                           const EmscriptenWebSocketErrorEvent* event,
                           void* user_data) {
  LOG(Error, "State WebSocket error");
  return EM_TRUE;
}

EM_BOOL Session::OnWsClose(int event_type,
                           const EmscriptenWebSocketCloseEvent* event,
                           void* user_data) {
  auto* session = static_cast<Session*>(user_data);
  LOG(Info, "State WebSocket closed (code=%d)", event->code);
  session->connected_ = false;

  // Codes 4xxx are deliberate server-side closes (e.g. kWsCloseSessionFull).
  // These conditions pass, so the GUI shows a notice while the reconnect loop
  // retries at a slower pace. The reason is shown too: for kWsCloseViewerError
  // it names the exception that stopped the viewer.
  if (event->code >= 4000 && event->code <= 4999) {
    session->server_close_code_ = event->code;
    session->server_close_reason_ = event->reason;  // Null-terminated.
    LOG(Info,
        "Server ended this connection (code=%d, reason=\"%s\"); retrying "
        "slowly.",
        event->code, event->reason);
  }

  // Free the handle; without this, every closed socket (including each failed
  // reconnect) leaks its handle and callback registrations in Emscripten's
  // socket table. Session (this) is a stable global, so the user_data of any
  // already-queued event stays valid; detaching the callbacks first stops them
  // firing on the freed handle.
  session->CloseSocket();

  return EM_TRUE;
}

void Session::CloseSocket() {
  if (socket_ <= 0) return;
  emscripten_websocket_set_onopen_callback(socket_, nullptr, nullptr);
  emscripten_websocket_set_onmessage_callback(socket_, nullptr, nullptr);
  emscripten_websocket_set_onerror_callback(socket_, nullptr, nullptr);
  emscripten_websocket_set_onclose_callback(socket_, nullptr, nullptr);
  emscripten_websocket_delete(socket_);
  socket_ = 0;
  connected_ = false;
}

void Session::Connect(const std::string& url) {
  // Drop any lingering socket first so a reconnect cannot leak the old one.
  CloseSocket();

  EmscriptenWebSocketCreateAttributes attr;
  emscripten_websocket_init_create_attributes(&attr);
  attr.url = url.c_str();
  attr.protocols = nullptr;
  attr.createOnMainThread = EM_TRUE;

  socket_ = emscripten_websocket_new(&attr);
  if (socket_ <= 0) {
    LOG(Error, "Failed to create state WebSocket");
    return;
  }
  emscripten_websocket_set_onopen_callback(socket_, this, OnWsOpen);
  emscripten_websocket_set_onmessage_callback(socket_, this, OnWsMessage);
  emscripten_websocket_set_onerror_callback(socket_, this, OnWsError);
  emscripten_websocket_set_onclose_callback(socket_, this, OnWsClose);
  LOG(Info, "State WebSocket connecting to %s", url.c_str());
}

void Session::SendText(const char* text) {
  if (socket_ && connected_) {
    emscripten_websocket_send_utf8_text(socket_, text);
  }
}

void Session::SetRole(SessionRole role) {
  role_ = role;
  // Warning: clang-format splits `!==` into `!= =`, so avoid that syntax!
  EM_ASM(
      { Module.isSpectator = !!$0; },
      role == SessionRole::kControlling ? 0 : 1);
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
  const double now = emscripten_get_now() / 1000.0;
  if (now - last_heartbeat_time_ >= kHeartbeatSec) {
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
  const double now = emscripten_get_now() / 1000.0;
  if (now - last_ui_retry_time_ < kUiRetrySec) {
    return;
  }
  if (state == RemoteUiState::kOpen) {
    ui_reject_count_ = 0;
    if (role_ != SessionRole::kControlling) {
      SetRole(SessionRole::kControlling);  // The claim succeeded.
    }
  } else if (state == RemoteUiState::kClosedOrError) {
    last_ui_retry_time_ = now;
    // A kWsCloseControllerTaken close while kControlling means another page
    // took the slot (Steal Control): settle instantly. A rejected claim
    // (kClaiming) retries a few times first, because a reloading controller
    // briefly races its own slot.
    const bool ousted = close_code == kWsCloseControllerTaken &&
                        role_ == SessionRole::kControlling;
    if (ousted || (close_code == kWsCloseControllerTaken &&
                   ++ui_reject_count_ >= kMaxUiRejects)) {
      LOG(Info, "Controller slot taken; spectating");
      SetRole(SessionRole::kSpectating);
      // Also drops the last received UI frame: a page forced out of the
      // controller slot must not keep showing a frozen Studio UI.
      callbacks_.ShutdownRemoteUi();
    } else {
      LOG(Info, "UI WebSocket closed; reconnecting...");
      callbacks_.ConnectRemoteUi();
      remote_ui_state_ = RemoteUiState::kConnecting;  // As on the grant path.
    }
  }
}

void Session::HandleMessage(const uint8_t* data, uint32_t num_bytes) {
  last_message_time_ = emscripten_get_now() / 1000.0;
  if (!callbacks_.ReadyForPayload()) {
    return;
  }
  bytes_accum_ += num_bytes;

  StatePayloadView view;
  if (!ParseStatePayload(data, num_bytes, &view)) {
    LOG(Error, "Malformed state payload (%u bytes); dropping", num_bytes);
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
    callbacks_.OnModelChanged();
    // Fall through: this payload already carries the new model's first state
    // and camera; OnPayload buffers the state while the download runs.
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

void Session::SetMaxSpectators(int count) {
  char msg[48];
  snprintf(msg, sizeof(msg), "%s%d", kMsgMaxSpectatorsPrefix, count);
  SendText(msg);
}

}  // namespace mujoco::studio
