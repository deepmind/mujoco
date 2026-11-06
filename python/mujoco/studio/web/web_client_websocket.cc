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

#include "web_client_websocket.h"

#include <emscripten/websocket.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"

#if !defined(__EMSCRIPTEN__)
#error "web_client_websocket.cc is only supported for Emscripten builds"
#endif

namespace mujoco::studio {
namespace {

// An Emscripten WebSocket. Its callbacks run on the browser's main thread,
// which also runs the app's frame loop, from the socket's JavaScript event
// handlers. They only record what happened; the app reads it each frame.
class BrowserWebSocket final : public WebSocket {
 public:
  explicit BrowserWebSocket(const std::string& url) {
    EmscriptenWebSocketCreateAttributes attributes;
    emscripten_websocket_init_create_attributes(&attributes);
    attributes.url = url.c_str();
    attributes.protocols = nullptr;
    attributes.createOnMainThread = EM_TRUE;
    socket_ = emscripten_websocket_new(&attributes);
    if (socket_ <= 0) {
      LOG(Error, "Failed to create a WebSocket to %s", url.c_str());
      socket_ = 0;
      state_ = ReadyState::kError;
      return;
    }
    emscripten_websocket_set_onopen_callback(socket_, this, OnOpen);
    emscripten_websocket_set_onmessage_callback(socket_, this, OnMessage);
    emscripten_websocket_set_onerror_callback(socket_, this, OnError);
    emscripten_websocket_set_onclose_callback(socket_, this, OnClose);
    LOG(Info, "WebSocket connecting to %s", url.c_str());
  }

  ~BrowserWebSocket() override {
    if (socket_ == 0) return;
    // Deleting the socket also removes its event handlers, so no callback can
    // reach this object afterwards.
    emscripten_websocket_close(socket_, 1000, nullptr);
    emscripten_websocket_delete(socket_);
  }

  BrowserWebSocket(const BrowserWebSocket&) = delete;
  BrowserWebSocket& operator=(const BrowserWebSocket&) = delete;

  ReadyState GetReadyState() const override { return state_; }

  int CloseCode() const override { return close_code_; }

  std::string CloseReason() const override { return close_reason_; }

  bool SendBinary(BytesView data) override {
    if (state_ != ReadyState::kOpen) return false;
    return emscripten_websocket_send_binary(
               socket_, const_cast<std::byte*>(data.data()),
               static_cast<uint32_t>(data.size())) == EMSCRIPTEN_RESULT_SUCCESS;
  }

  bool SendText(const char* text) override {
    if (state_ != ReadyState::kOpen) return false;
    return emscripten_websocket_send_utf8_text(socket_, text) ==
           EMSCRIPTEN_RESULT_SUCCESS;
  }

  std::vector<Frame> TakeFrames() override {
    std::vector<Frame> frames;
    frames.swap(received_);
    return frames;
  }

  std::vector<Bytes> TakeMessages() override {
    std::vector<Bytes> messages;
    for (Frame& frame : TakeFrames()) {
      if (!frame.is_text) {
        messages.push_back(std::move(frame.binary));
      }
    }
    return messages;
  }

 private:
  static EM_BOOL OnOpen(int event_type,
                        const EmscriptenWebSocketOpenEvent* event,
                        void* user_data) {
    static_cast<BrowserWebSocket*>(user_data)->state_ = ReadyState::kOpen;
    return EM_TRUE;
  }

  static EM_BOOL OnMessage(int event_type,
                           const EmscriptenWebSocketMessageEvent* event,
                           void* user_data) {
    // Emscripten frees the event buffer when this callback returns.
    auto* self = static_cast<BrowserWebSocket*>(user_data);
    Frame frame;
    if (event->isText) {
      // Emscripten's numBytes includes the trailing NUL for UTF-8 text.
      const size_t len =
          event->numBytes > 0 ? static_cast<size_t>(event->numBytes - 1) : 0;
      frame.is_text = true;
      frame.text.assign(reinterpret_cast<const char*>(event->data), len);
    } else {
      const auto* bytes = reinterpret_cast<const std::byte*>(event->data);
      frame.binary.assign(bytes, bytes + static_cast<size_t>(event->numBytes));
    }
    self->received_.push_back(std::move(frame));
    return EM_TRUE;
  }

  static EM_BOOL OnError(int event_type,
                         const EmscriptenWebSocketErrorEvent* event,
                         void* user_data) {
    static_cast<BrowserWebSocket*>(user_data)->state_ = ReadyState::kError;
    return EM_TRUE;
  }

  static EM_BOOL OnClose(int event_type,
                         const EmscriptenWebSocketCloseEvent* event,
                         void* user_data) {
    auto* self = static_cast<BrowserWebSocket*>(user_data);
    // Browsers report a failed connection as an error, then a close.
    if (self->state_ != ReadyState::kError) self->state_ = ReadyState::kClosed;
    self->close_code_ = event->code;
    self->close_reason_ = event->reason;
    return EM_TRUE;
  }

  std::vector<Frame> received_;
  std::string close_reason_;
  EMSCRIPTEN_WEBSOCKET_T socket_ = 0;
  ReadyState state_ = ReadyState::kConnecting;
  int close_code_ = 0;
};

}  // namespace

std::unique_ptr<WebSocket> OpenWebSocket(const std::string& url) {
  return std::make_unique<BrowserWebSocket>(url);
}

}  // namespace mujoco::studio
