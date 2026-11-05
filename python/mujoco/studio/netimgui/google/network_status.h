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

// Connection-state query for the WASM WebSocket network backend.
//
// Stock NetImgui has no such query and does not need one: its Connect()
// implementations block until the connection is established, so a non-null
// SocketInfo is always usable and later failures surface as DataSend /
// DataReceive errors inside the client thread. In the browser,
// emscripten_websocket_new() returns a socket handle immediately while the
// connection completes (or fails) asynchronously, and no client state
// machine is watching it. Callers poll this state to hold back traffic
// until the socket is actually open and to detect closure for reconnecting.

#ifndef THIRD_PARTY_NETIMGUI_GOOGLE_NETWORK_STATUS_H_
#define THIRD_PARTY_NETIMGUI_GOOGLE_NETWORK_STATUS_H_

namespace NetImgui {
namespace Internal {
namespace Network {

struct SocketInfo;

enum class ReadyState {
  kDisconnected,  // Null socket.
  kConnecting,
  kOpen,
  kClosing,
  kClosed,
  kError,
};

// Implemented in NetImgui_NetworkWASM.cpp (Emscripten builds only).
ReadyState GetReadyState(SocketInfo* client_socket);

// The WebSocket close code once the socket has closed, else 0. Lets callers
// distinguish a deliberate server-side rejection (e.g. 4001 = driver slot
// taken) from an ordinary drop. Implemented in NetImgui_NetworkWASM.cpp.
int GetCloseCode(SocketInfo* client_socket);

// Human-readable state name, for status overlays and logs.
inline const char* ReadyStateName(ReadyState state) {
  switch (state) {
    case ReadyState::kDisconnected:
      return "Disconnected";
    case ReadyState::kConnecting:
      return "Connecting";
    case ReadyState::kOpen:
      return "Open";
    case ReadyState::kClosing:
      return "Closing";
    case ReadyState::kClosed:
      return "Closed";
    case ReadyState::kError:
      return "Error";
  }
  return "Unknown";
}

}  // namespace Network
}  // namespace Internal
}  // namespace NetImgui

#endif  // THIRD_PARTY_NETIMGUI_GOOGLE_NETWORK_STATUS_H_
