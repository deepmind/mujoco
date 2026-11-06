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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "state_payload.h"
#include "web_client_websocket.h"

namespace mujoco::studio {
namespace {

using ReadyState = WebSocket::ReadyState;

double g_fake_now = 100.0;
double FakeClock() { return g_fake_now; }

class FakeStateWebSocket : public WebSocket {
 public:
  explicit FakeStateWebSocket(bool* destroyed = nullptr)
      : destroyed_(destroyed) {}
  ~FakeStateWebSocket() override {
    if (destroyed_ != nullptr) *destroyed_ = true;
  }

  ReadyState GetReadyState() const override { return state; }
  int CloseCode() const override { return close_code; }
  std::string CloseReason() const override { return close_reason; }

  bool SendBinary(BytesView data) override {
    if (state != ReadyState::kOpen) return false;
    sent_binary.emplace_back(data.begin(), data.end());
    return true;
  }

  bool SendText(const char* text) override {
    if (state != ReadyState::kOpen) return false;
    sent_text.emplace_back(text);
    return true;
  }

  std::vector<Bytes> TakeMessages() override { return {}; }

  std::vector<Frame> TakeFrames() override {
    std::vector<Frame> out;
    out.swap(frames);
    return out;
  }

  void PushText(std::string text) {
    Frame frame;
    frame.is_text = true;
    frame.text = std::move(text);
    frames.push_back(std::move(frame));
  }

  void PushBinary(const Bytes& payload) {
    Frame frame;
    frame.binary = payload;
    frames.push_back(std::move(frame));
  }

  std::vector<Bytes> sent_binary;
  std::vector<std::string> sent_text;
  std::vector<Frame> frames;
  std::string close_reason;
  ReadyState state = ReadyState::kOpen;
  int close_code = 0;

 private:
  bool* destroyed_ = nullptr;
};

class FakeSessionCallbacks : public Session::Callbacks {
 public:
  bool ReadyForPayload() override { return ready_for_payload; }

  void OnPayload(const StatePayloadView& view) override {
    ++payload_count;
    last_payload_crc = view.model_crc32;
  }

  void OnModelChanged(uint32_t model_crc32) override {
    ++model_changed_count;
    last_model_changed_crc = model_crc32;
  }
  void ConnectRemoteUi() override { ++connect_ui_calls; }
  void ShutdownRemoteUi() override { ++shutdown_ui_calls; }
  void SetCameraMode(int mode) override { last_camera_mode = mode; }

  int payload_count = 0;
  uint32_t last_payload_crc = 0;
  int model_changed_count = 0;
  uint32_t last_model_changed_crc = 0;
  int connect_ui_calls = 0;
  int shutdown_ui_calls = 0;
  int last_camera_mode = -1;
  bool ready_for_payload = true;
};

std::vector<std::byte> MakeStatePayload(uint32_t model_crc32) {
  const mjtNum physics[2] = {1.25, -2.5};
  mjvCamera camera{};
  mjvPerturb perturb{};
  mjvOption vis_options{};
  mjOption opt{};
  mjVisual vis{};
  mjStatistic stat{};
  std::vector<uint8_t> render_flags(mjNRNDFLAG, 0);
  return SerializeStatePayload(model_crc32, mjSTATE_INTEGRATION, physics,
                               sizeof(physics), camera, perturb, vis_options,
                               opt, vis, stat, render_flags, nullptr, 0,
                               ViewerInputState{});
}

class WebClientSessionTest : public ::testing::Test {
 protected:
  WebClientSessionTest() : session_(callbacks_, &FakeClock) {
    g_fake_now = 100.0;
  }

  FakeStateWebSocket* ConnectSocket(bool* destroyed = nullptr) {
    auto socket = std::make_unique<FakeStateWebSocket>(destroyed);
    FakeStateWebSocket* raw = socket.get();
    session_.Connect(std::move(socket));
    return raw;
  }

  FakeSessionCallbacks callbacks_;
  Session session_;
};

TEST_F(WebClientSessionTest, PromotesImmediatelyWhenRemoteUiOpensAfterRetry) {
  // First rejected attempt triggers a reconnect at t = 100.0s.
  session_.HandleRemoteUiState(RemoteUiState::kClosedOrError,
                               kWsCloseControllerTaken);
  EXPECT_EQ(callbacks_.connect_ui_calls, 1);
  EXPECT_EQ(session_.Role(), SessionRole::kClaiming);

  // Even 50ms later (well inside the 1.0s retry cooldown), an accepted /ui
  // stream promotes the page to kControlling immediately.
  g_fake_now = 100.05;
  session_.HandleRemoteUiState(RemoteUiState::kOpen, 0);
  EXPECT_EQ(session_.Role(), SessionRole::kControlling);
}

TEST_F(WebClientSessionTest, PacesClaimRetriesAndOustsControllerInstantly) {
  // Rejected claims retry at most once per second and settle on the 3rd reject.
  session_.HandleRemoteUiState(RemoteUiState::kClosedOrError,
                               kWsCloseControllerTaken);
  EXPECT_EQ(callbacks_.connect_ui_calls, 1);
  EXPECT_EQ(session_.Role(), SessionRole::kClaiming);

  g_fake_now = 100.4;
  session_.HandleRemoteUiState(RemoteUiState::kClosedOrError,
                               kWsCloseControllerTaken);
  EXPECT_EQ(callbacks_.connect_ui_calls, 1);  // Still pacing.

  g_fake_now = 101.0;
  session_.HandleRemoteUiState(RemoteUiState::kClosedOrError,
                               kWsCloseControllerTaken);
  EXPECT_EQ(callbacks_.connect_ui_calls, 2);
  EXPECT_EQ(session_.Role(), SessionRole::kClaiming);

  g_fake_now = 102.0;
  session_.HandleRemoteUiState(RemoteUiState::kClosedOrError,
                               kWsCloseControllerTaken);
  EXPECT_EQ(session_.Role(), SessionRole::kSpectating);
  EXPECT_EQ(callbacks_.shutdown_ui_calls, 1);

  // Once controlling, a kWsCloseControllerTaken close settles immediately
  // without waiting for retry pacing.
  FakeStateWebSocket* ws = ConnectSocket();
  ws->PushText("grant");
  session_.Update();
  ASSERT_EQ(session_.Role(), SessionRole::kClaiming);
  session_.HandleRemoteUiState(RemoteUiState::kOpen, 0);
  ASSERT_EQ(session_.Role(), SessionRole::kControlling);

  g_fake_now = 102.1;
  session_.HandleRemoteUiState(RemoteUiState::kClosedOrError,
                               kWsCloseControllerTaken);
  EXPECT_EQ(session_.Role(), SessionRole::kSpectating);
  EXPECT_EQ(callbacks_.shutdown_ui_calls, 2);
}

TEST_F(WebClientSessionTest, RosterSettlesClaimingAndGrantProtectsFreshClaim) {
  FakeStateWebSocket* ws = ConnectSocket();

  // While /ui has no socket, a spectator roster settles kClaiming immediately.
  ws->PushText(
      "viewers=2;role=spectator;queue_pos=1;queue_len=2;max_spectators=8");
  session_.Update();
  EXPECT_EQ(session_.Role(), SessionRole::kSpectating);
  EXPECT_EQ(callbacks_.shutdown_ui_calls, 1);

  SessionView view;
  session_.FillView(&view);
  EXPECT_EQ(view.role, SessionRole::kSpectating);
  EXPECT_EQ(view.viewers, 2);
  EXPECT_EQ(view.queue_pos, 1);
  EXPECT_EQ(view.queue_len, 2);
  EXPECT_EQ(view.max_spectators, 8);

  // Receiving a grant starts a fresh claim and marks /ui as connecting, so the
  // trailing spectator roster right after the grant does not abort it.
  ws->PushText("grant");
  ws->PushText(
      "viewers=2;role=spectator;queue_pos=0;queue_len=1;max_spectators=8");
  session_.Update();
  EXPECT_EQ(callbacks_.connect_ui_calls, 1);
  EXPECT_EQ(session_.Role(), SessionRole::kClaiming);
  EXPECT_EQ(callbacks_.shutdown_ui_calls, 1);
}

TEST_F(WebClientSessionTest, HandlesStatePayloadsAcksAndModelReloads) {
  FakeStateWebSocket* ws = ConnectSocket();

  const std::vector<std::byte> payload1 = MakeStatePayload(/*model_crc32=*/111);
  ws->PushBinary(payload1);
  g_fake_now = 105.0;
  session_.Update();

  EXPECT_EQ(callbacks_.payload_count, 1);
  EXPECT_EQ(callbacks_.last_payload_crc, 111u);
  EXPECT_EQ(callbacks_.model_changed_count, 0);
  EXPECT_DOUBLE_EQ(session_.LastMessageTime(), 105.0);
  EXPECT_EQ(session_.ConsumeByteCount(), payload1.size());
  EXPECT_EQ(session_.ConsumeByteCount(), 0u);
  // Update() at t = 105.0 sends both "state_ack" and the initial "heartbeat".
  ASSERT_EQ(ws->sent_text.size(), 2u);
  EXPECT_EQ(ws->sent_text[0], "state_ack");
  EXPECT_EQ(ws->sent_text[1], "heartbeat");

  // Changing model_crc32 triggers OnModelChanged() once and still delivers the
  // new model's initial state payload.
  ws->sent_text.clear();
  ws->PushBinary(MakeStatePayload(/*model_crc32=*/222));
  ws->PushBinary(MakeStatePayload(/*model_crc32=*/222));
  g_fake_now = 110.0;
  session_.Update();
  EXPECT_EQ(callbacks_.model_changed_count, 1);
  EXPECT_EQ(callbacks_.last_model_changed_crc, 222u);
  EXPECT_EQ(callbacks_.payload_count, 3);
  EXPECT_EQ(callbacks_.last_payload_crc, 222u);
  EXPECT_EQ(ws->sent_text,
            (std::vector<std::string>{"state_ack", "state_ack"}));
}

TEST_F(WebClientSessionTest,
       RecordsServerCloseReasonAndSendsPeriodicHeartbeat) {
  bool first_destroyed = false;
  FakeStateWebSocket* ws1 = ConnectSocket(&first_destroyed);
  session_.Update();  // Sends initial heartbeat at t = 100.0.
  ASSERT_EQ(ws1->sent_text, std::vector<std::string>{"heartbeat"});
  ws1->sent_text.clear();

  // Another heartbeat is sent once 30s have elapsed.
  g_fake_now = 129.9;
  session_.Update();
  EXPECT_TRUE(ws1->sent_text.empty());
  g_fake_now = 130.0;
  session_.Update();
  EXPECT_EQ(ws1->sent_text, std::vector<std::string>{"heartbeat"});

  // A 4xxx close records code + reason and releases the closed socket.
  ws1->state = ReadyState::kClosed;
  ws1->close_code = kWsCloseViewerError;
  ws1->close_reason = "RuntimeError: boom";
  session_.Update();
  EXPECT_TRUE(first_destroyed);
  EXPECT_FALSE(session_.HasSocket());
  EXPECT_EQ(session_.ServerCloseCode(), kWsCloseViewerError);
  EXPECT_EQ(session_.ServerCloseReason(), "RuntimeError: boom");

  // Merely opening a new socket keeps the close notice until a frame arrives.
  FakeStateWebSocket* ws2 = ConnectSocket();
  session_.Update();
  EXPECT_EQ(session_.ServerCloseCode(), kWsCloseViewerError);

  ws2->PushText(
      "viewers=1;role=controller;queue_pos=0;queue_len=0;max_spectators=8");
  session_.Update();
  EXPECT_EQ(session_.ServerCloseCode(), 0);
  EXPECT_TRUE(session_.ServerCloseReason().empty());
}

}  // namespace
}  // namespace mujoco::studio
