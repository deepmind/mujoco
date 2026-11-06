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

// A browser WebSocket as the web client's /ui and /state streams use it: whole
// binary or text frames each way, read from the app's frame loop. The browser
// connects and closes WebSockets asynchronously; GetReadyState() tells where it
// stands.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_WEBSOCKET_H_
#define MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_WEBSOCKET_H_

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mujoco::studio {

class WebSocket {
 public:
  using Bytes = std::vector<std::byte>;
  using BytesView = std::span<const std::byte>;

  enum class ReadyState {
    kDisconnected,  // No socket.
    kConnecting,
    kOpen,
    kClosed,
    kError,
  };

  struct Frame {
    Bytes binary;
    std::string text;
    bool is_text = false;
  };

  // Closes the socket.
  virtual ~WebSocket() = default;

  virtual ReadyState GetReadyState() const = 0;

  // The close code the server sent, once the socket has closed; else 0.
  virtual int CloseCode() const = 0;

  // The close reason text sent by the server once the socket has closed.
  virtual std::string CloseReason() const { return ""; }

  // Sends one binary message. Returns false if the socket is not open.
  virtual bool SendBinary(BytesView data) = 0;

  // Sends one text message. Returns false if the socket is not open.
  virtual bool SendText(const char* text) = 0;

  // Returns the binary messages received since the last call, oldest first.
  virtual std::vector<Bytes> TakeMessages() = 0;

  // Returns all frames (text and binary) received since the last call, oldest
  // first. Default implementation wraps TakeMessages() as binary frames.
  virtual std::vector<Frame> TakeFrames() {
    std::vector<Frame> frames;
    for (Bytes& message : TakeMessages()) {
      Frame frame;
      frame.binary = std::move(message);
      frames.push_back(std::move(frame));
    }
    return frames;
  }
};

// Human-readable state name, for logs.
inline const char* ReadyStateName(WebSocket::ReadyState state) {
  switch (state) {
    case WebSocket::ReadyState::kDisconnected:
      return "Disconnected";
    case WebSocket::ReadyState::kConnecting:
      return "Connecting";
    case WebSocket::ReadyState::kOpen:
      return "Open";
    case WebSocket::ReadyState::kClosed:
      return "Closed";
    case WebSocket::ReadyState::kError:
      return "Error";
  }
  return "Unknown";
}

// Opens a WebSocket to `url`, e.g. "ws://localhost:8080/ui" (browser builds
// only, see web_client_websocket.cc).
std::unique_ptr<WebSocket> OpenWebSocket(const std::string& url);

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_WEBSOCKET_H_
