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

// WASM networking backend for NetImgui.
//
// Implements the NetImgui::Internal::Network interface using Emscripten
// WebSockets. This file is compiled only under the Emscripten toolchain and
// provides the browser-side transport layer for netimgui draw data.
#include "NetImgui_Shared.h"

#ifndef __EMSCRIPTEN__
#error "This file must be compiled with emscripten."
#endif

#include <emscripten/console.h>
#include <emscripten/websocket.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#include "NetImgui_CmdPackets.h"
#include "google/logging.h"
#include "google/network_status.h"

namespace NetImgui {
namespace Internal {
namespace Network {

struct SocketInfo {
  EMSCRIPTEN_WEBSOCKET_T mSocket = 0;
  std::atomic<bool> mConnected{false};
  std::atomic<bool> mError{false};
  std::atomic<bool> mClosed{false};
  std::atomic<int> mCloseCode{0};

  std::vector<uint8_t> mBuffer;
  size_t mBufferReadOffset = 0;
  std::mutex mBufferMutex;
  int mSendSizeMax =
      1024 * 1024;  // Interface compatibility with other backends
};

// --- WebSocket event callbacks ---

static EM_BOOL OnWebSocketOpen(int /*event_type*/,
                               const EmscriptenWebSocketOpenEvent* /*event*/,
                               void* user_data) {
  auto* socket = static_cast<SocketInfo*>(user_data);
  if (socket) {
    socket->mConnected = true;
  }
  return EM_TRUE;
}

static EM_BOOL OnWebSocketMessage(int /*event_type*/,
                                  const EmscriptenWebSocketMessageEvent* event,
                                  void* user_data) {
  auto* socket = static_cast<SocketInfo*>(user_data);
  if (socket && !event->isText) {
    std::lock_guard<std::mutex> lock(socket->mBufferMutex);
    // DataReceive resets the buffer as soon as it has consumed all of it, so a
    // non-zero read offset always has unread bytes after it. Compact when the
    // consumed prefix is large, so a long-lived connection does not keep
    // growing the buffer.
    if (socket->mBufferReadOffset >= 65536 &&
        socket->mBufferReadOffset >= socket->mBuffer.size() / 2) {
      socket->mBuffer.erase(
          socket->mBuffer.begin(),
          socket->mBuffer.begin() + socket->mBufferReadOffset);
      socket->mBufferReadOffset = 0;
    }
    size_t old_size = socket->mBuffer.size() - socket->mBufferReadOffset;
    socket->mBuffer.insert(socket->mBuffer.end(), event->data,
                           event->data + event->numBytes);
    static int msg_count = 0;
    ++msg_count;
    VLOG(1, "onmessage #%d: %d bytes, buffer: %zu -> %zu", msg_count,
         event->numBytes, old_size,
         socket->mBuffer.size() - socket->mBufferReadOffset);
  } else if (socket && event->isText) {
    VLOG(1, "onmessage: TEXT frame (%d bytes), IGNORED", event->numBytes);
  }
  return EM_TRUE;
}

static EM_BOOL OnWebSocketClose(int /*event_type*/,
                                const EmscriptenWebSocketCloseEvent* event,
                                void* user_data) {
  auto* socket = static_cast<SocketInfo*>(user_data);
  if (socket) {
    socket->mClosed = true;
    socket->mCloseCode = event->code;
  }
  return EM_TRUE;
}

static EM_BOOL OnWebSocketError(int /*event_type*/,
                                const EmscriptenWebSocketErrorEvent* /*event*/,
                                void* user_data) {
  auto* socket = static_cast<SocketInfo*>(user_data);
  if (socket) {
    socket->mError = true;
  }
  return EM_TRUE;
}

// --- Network interface implementation ---

bool Startup() { return emscripten_websocket_is_supported(); }

void Shutdown() {}

SocketInfo* Connect(const char* server_host, uint32_t /*server_port*/) {
  if (!emscripten_websocket_is_supported()) return nullptr;

  SocketInfo* socket_info = netImguiNew<SocketInfo>();

  EmscriptenWebSocketCreateAttributes attr;
  emscripten_websocket_init_create_attributes(&attr);

  // The web viewer always passes a complete WebSocket URL (e.g.
  // "ws://host:8080/ui", built from the page origin by WsUrl), so it is used
  // verbatim and server_port is ignored — a browser reaches the viewer's
  // paths on the page's own shared port, not a dedicated NetImgui port.
  const std::string url(server_host);
  attr.url = url.c_str();
  attr.createOnMainThread = EM_TRUE;
  LOG(Info, "Connecting WebSocket to: %s", url.c_str());

  socket_info->mSocket = emscripten_websocket_new(&attr);
  if (socket_info->mSocket <= 0) {
    netImguiDelete(socket_info);
    return nullptr;
  }

  emscripten_websocket_set_onopen_callback(socket_info->mSocket, socket_info,
                                           OnWebSocketOpen);
  emscripten_websocket_set_onmessage_callback(socket_info->mSocket, socket_info,
                                              OnWebSocketMessage);
  emscripten_websocket_set_onclose_callback(socket_info->mSocket, socket_info,
                                            OnWebSocketClose);
  emscripten_websocket_set_onerror_callback(socket_info->mSocket, socket_info,
                                            OnWebSocketError);

  return socket_info;
}

// Abandoned sockets are not freed immediately: with -pthread, websocket
// events are queued across threads, so an already-queued close/error event
// can still dereference the SocketInfo after emscripten_websocket_delete().
// When the freed block was recycled for the next socket, such a late event
// stamped a stale mClosed flag onto a healthy connection, which the
// reconnect logic then tore down — a self-sustaining reconnect loop. Keep
// abandoned sockets in a small ring and free them several disconnects
// later, when any queued events are long gone.
static SocketInfo* s_socket_graveyard[8] = {};
static int s_socket_graveyard_idx = 0;

void Disconnect(SocketInfo* client_socket) {
  if (!client_socket) return;
  client_socket->mClosed = true;
  // Detach this socket from future events; queued events may already hold
  // the pointer (the graveyard above covers those).
  emscripten_websocket_set_onopen_callback(client_socket->mSocket, nullptr,
                                           OnWebSocketOpen);
  emscripten_websocket_set_onmessage_callback(client_socket->mSocket, nullptr,
                                              OnWebSocketMessage);
  emscripten_websocket_set_onclose_callback(client_socket->mSocket, nullptr,
                                            OnWebSocketClose);
  emscripten_websocket_set_onerror_callback(client_socket->mSocket, nullptr,
                                            OnWebSocketError);
  emscripten_websocket_close(client_socket->mSocket, 1000,
                             "Normal Disconnection");
  emscripten_websocket_delete(client_socket->mSocket);
  {
    // Release the receive buffer now; only the flags must stay valid.
    std::lock_guard<std::mutex> lock(client_socket->mBufferMutex);
    client_socket->mBuffer.clear();
    client_socket->mBufferReadOffset = 0;
    client_socket->mBuffer.shrink_to_fit();
  }
  if (s_socket_graveyard[s_socket_graveyard_idx]) {
    netImguiDelete(s_socket_graveyard[s_socket_graveyard_idx]);
  }
  s_socket_graveyard[s_socket_graveyard_idx] = client_socket;
  s_socket_graveyard_idx = (s_socket_graveyard_idx + 1) % 8;
}

bool DataReceivePending(SocketInfo* client_socket) {
  if (!client_socket) return false;

  if (client_socket->mError || client_socket->mClosed) {
    // Connection is dead — flush any buffered data so we stop processing
    // stale commands that arrived before the close.
    std::lock_guard<std::mutex> lock(client_socket->mBufferMutex);
    if (client_socket->mBufferReadOffset < client_socket->mBuffer.size()) {
      LOG(Warning, "Connection closed/error. Discarding %zu buffered bytes.",
          client_socket->mBuffer.size() - client_socket->mBufferReadOffset);
      client_socket->mBuffer.clear();
      client_socket->mBufferReadOffset = 0;
    }
    return false;
  }

  std::lock_guard<std::mutex> lock(client_socket->mBufferMutex);
  return client_socket->mBufferReadOffset < client_socket->mBuffer.size();
}

void DataReceive(SocketInfo* client_socket, PendingCom& pending_rcv) {
  if (!client_socket || !pending_rcv.pCommand) {
    pending_rcv.bError = true;
    return;
  }

  if (!client_socket->mConnected) {
    pending_rcv.bError = false;  // Not ready yet, caller will retry.
    return;
  }

  // The header is received first; the command size it carries comes off the
  // wire and is validated against the destination buffer.
  size_t bytes_to_read = pending_rcv.GetSizeToReceive();
  if (bytes_to_read == 0) return;  // Fully received, or invalid size (error)

  std::lock_guard<std::mutex> lock(client_socket->mBufferMutex);

  size_t available =
      client_socket->mBuffer.size() - client_socket->mBufferReadOffset;
  VLOG(1, "DataReceive: want=%zu, have=%zu, cmd_size=%u, progress=%zu",
       bytes_to_read, available, pending_rcv.pCommand->mSize,
       pending_rcv.SizeCurrent);

  if (available == 0) {
    if (client_socket->mError || client_socket->mClosed) {
      pending_rcv.bError = true;
    }
    return;
  }

  size_t bytes_to_consume = std::min(bytes_to_read, available);
  if (bytes_to_consume > 0) {
    memcpy(reinterpret_cast<uint8_t*>(pending_rcv.pCommand) +
               pending_rcv.SizeCurrent,
           client_socket->mBuffer.data() + client_socket->mBufferReadOffset,
           bytes_to_consume);
    client_socket->mBufferReadOffset += bytes_to_consume;
    if (client_socket->mBufferReadOffset == client_socket->mBuffer.size()) {
      client_socket->mBuffer.clear();
      client_socket->mBufferReadOffset = 0;
    }
    pending_rcv.SizeCurrent += bytes_to_consume;
    pending_rcv.bError = false;
  }
}

void DataSend(SocketInfo* client_socket, PendingCom& pending_send) {
  if (!client_socket || client_socket->mClosed || client_socket->mError ||
      !pending_send.pCommand) {
    pending_send.bError = true;
    return;
  }

  if (!client_socket->mConnected) {
    pending_send.bError = false;  // Not ready yet, caller will retry.
    return;
  }

  size_t bytes_remaining =
      pending_send.pCommand->mSize - pending_send.SizeCurrent;
  if (bytes_remaining == 0) return;

  EMSCRIPTEN_RESULT result = emscripten_websocket_send_binary(
      client_socket->mSocket,
      reinterpret_cast<uint8_t*>(pending_send.pCommand) +
          pending_send.SizeCurrent,
      bytes_remaining);

  if (result == EMSCRIPTEN_RESULT_SUCCESS) {
    pending_send.SizeCurrent += bytes_remaining;
    pending_send.bError = false;
  } else {
    pending_send.bError = true;
  }
}

SocketInfo* ListenStart(uint32_t /*listen_port*/) {
  return nullptr;  // Browsers cannot open listening ports.
}

SocketInfo* ListenConnect(SocketInfo* /*listen_socket*/) { return nullptr; }

int GetCloseCode(SocketInfo* client_socket) {
  return client_socket ? client_socket->mCloseCode.load() : 0;
}

ReadyState GetReadyState(SocketInfo* client_socket) {
  if (!client_socket) return ReadyState::kDisconnected;
  if (client_socket->mError) return ReadyState::kError;
  if (client_socket->mClosed) return ReadyState::kClosed;

  uint16_t ready_state = 0;
  emscripten_websocket_get_ready_state(client_socket->mSocket, &ready_state);
  switch (ready_state) {
    case 0:
      return ReadyState::kConnecting;
    case 1:
      return ReadyState::kOpen;
    case 2:
      return ReadyState::kClosing;
    case 3:
      return ReadyState::kClosed;
  }
  return ReadyState::kError;
}

}  // namespace Network
}  // namespace Internal
}  // namespace NetImgui
