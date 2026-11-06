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

#include "web_client_remote_ui.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include <imgui.h>
#include <mujoco/mujoco.h>
#include "state_payload.h"
#include "ui_codec.h"
#include "web_client_websocket.h"

namespace mujoco::studio {
namespace {

using ReadyState = WebSocket::ReadyState;
using TextureId = RemoteUi::TextureId;

class FakeWebSocket : public WebSocket {
 public:
  ReadyState GetReadyState() const override { return state; }
  int CloseCode() const override { return close_code; }
  bool SendBinary(BytesView data) override {
    if (state != ReadyState::kOpen) return false;
    sent.emplace_back(data.begin(), data.end());
    return true;
  }
  bool SendText(const char* text) override {
    if (state != ReadyState::kOpen) return false;
    sent_text.emplace_back(text);
    return true;
  }
  std::vector<Bytes> TakeMessages() override {
    std::vector<Bytes> messages;
    messages.swap(received);
    return messages;
  }

  std::vector<Bytes> sent;             // Binary messages, oldest first.
  std::vector<std::string> sent_text;  // Text messages, oldest first.
  std::vector<Bytes> received;         // Taken by the next TakeMessages().
  ReadyState state = ReadyState::kOpen;
  int close_code = 0;
};

template <typename T>
T CommandIn(netui::BytesView message) {
  T command;
  CHECK_EQ(message.size(), sizeof(command));
  std::memcpy(&command, message.data(), sizeof(command));
  return command;
}

class FakeCallbacks : public RemoteUi::Callbacks {
 public:
  struct UploadedTexture {
    std::vector<uint8_t> rgba;
    uint32_t width = 0;
    uint32_t height = 0;
    bool destroyed = false;
  };

  uintptr_t UploadTexture(uintptr_t current, const std::byte* rgba,
                          uint32_t width, uint32_t height) override {
    upload_calls++;
    if (rgba == nullptr) {
      if (current != 0) {
        textures[current].destroyed = true;
        textures[current].rgba.clear();
      }
      return 0;
    }
    uintptr_t handle = current != 0 ? current : next_handle++;
    UploadedTexture& tex = textures[handle];
    tex.width = width;
    tex.height = height;
    tex.destroyed = false;
    const auto* bytes = reinterpret_cast<const uint8_t*>(rgba);
    tex.rgba.assign(bytes, bytes + static_cast<size_t>(width) * height * 4);
    return handle;
  }

  bool GpuReady() override { return gpu_ready; }

  std::unordered_map<uintptr_t, UploadedTexture> textures;
  uintptr_t next_handle = 100;
  int upload_calls = 0;
  bool gpu_ready = true;
};

class RemoteUiTest : public ::testing::Test {
 protected:
  RemoteUiTest() : context_(ImGui::CreateContext()), remote_ui_(callbacks_) {
    ImGui::SetCurrentContext(context_);
    remote_ui_.SetScreenArea(1920.0f, 1080.0f);
  }

  ~RemoteUiTest() override {
    remote_ui_.Shutdown();
    ImGui::DestroyContext(context_);
  }

  FakeWebSocket* ConnectFakeSocket() {
    auto socket = std::make_unique<FakeWebSocket>();
    FakeWebSocket* fake = socket.get();
    remote_ui_.Connect(std::move(socket));
    return fake;
  }

  static netui::Bytes MakeTextureMsg(TextureId tex_id,
                                     netui::Texture::Action action,
                                     netui::Texture::Format format,
                                     uint16_t width, uint16_t height,
                                     const std::vector<uint8_t>& pixels,
                                     uint16_t offset_x = 0,
                                     uint16_t offset_y = 0) {
    netui::Bytes msg(sizeof(netui::Texture) + pixels.size());
    netui::Texture hdr;
    hdr.header.size = static_cast<uint32_t>(msg.size());
    hdr.tex_id = tex_id;
    hdr.action = action;
    hdr.format = format;
    hdr.x = offset_x;
    hdr.y = offset_y;
    hdr.width = width;
    hdr.height = height;
    std::memcpy(msg.data(), &hdr, sizeof(hdr));
    if (!pixels.empty()) {
      std::memcpy(msg.data() + sizeof(hdr), pixels.data(), pixels.size());
    }
    return msg;
  }

  static netui::SerializedFrame RenderSerializedFrame(
      const char* text, TextureId tex_id = 1, bool second_window = false) {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(640, 480);
    unsigned char* font_pixels = nullptr;
    int font_w = 0, font_h = 0;
    io.Fonts->GetTexDataAsAlpha8(&font_pixels, &font_w, &font_h);
    io.Fonts->SetTexID(static_cast<ImTextureID>(tex_id));

    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(250, 150));
    ImGui::Begin("Window1");
    ImGui::TextUnformatted(text);
    ImGui::Button("Action");
    ImGui::End();

    if (second_window) {
      ImGui::SetNextWindowPos(ImVec2(300, 10));
      ImGui::SetNextWindowSize(ImVec2(250, 150));
      ImGui::Begin("Window2");
      ImGui::TextUnformatted("Second window");
      ImGui::End();
    }

    ImGui::Render();
    netui::SerializedFrame serialized;
    netui::SerializeDrawData(ImGui::GetDrawData(), ImGuiMouseCursor_Arrow,
                             &serialized);
    CHECK(!serialized.lists.empty());
    return serialized;
  }

  void RegisterDummyTexture(TextureId tex_id) {
    std::vector<uint8_t> pixels(4 * 4 * 4, 255);
    remote_ui_.ProcessTextureMessage(
        MakeTextureMsg(tex_id, netui::Texture::Action::kCreate,
                       netui::Texture::Format::kRgba8, 4, 4, pixels));
  }

  ImGuiContext* context_ = nullptr;
  FakeCallbacks callbacks_;
  RemoteUi remote_ui_;
};

TEST_F(RemoteUiTest, CreatesUpdatesAndDestroysRgba8Texture) {
  std::vector<uint8_t> initial_pixels(4 * 4 * 4);
  for (size_t i = 0; i < 16; ++i) {
    initial_pixels[i * 4 + 0] = 10;
    initial_pixels[i * 4 + 1] = 20;
    initial_pixels[i * 4 + 2] = 30;
    initial_pixels[i * 4 + 3] = 255;
  }
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(42, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kRgba8, 4, 4, initial_pixels));

  ASSERT_EQ(callbacks_.upload_calls, 1);
  ASSERT_TRUE(callbacks_.textures.count(100));
  EXPECT_EQ(callbacks_.textures[100].width, 4u);
  EXPECT_EQ(callbacks_.textures[100].height, 4u);
  EXPECT_EQ(callbacks_.textures[100].rgba, initial_pixels);

  // Patch a 2x2 sub-region at offset (1, 2).
  std::vector<uint8_t> patch_pixels(2 * 2 * 4);
  for (size_t i = 0; i < 4; ++i) {
    patch_pixels[i * 4 + 0] = 200;
    patch_pixels[i * 4 + 1] = 100;
    patch_pixels[i * 4 + 2] = 50;
    patch_pixels[i * 4 + 3] = 128;
  }
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(42, netui::Texture::Action::kUpdate,
                     netui::Texture::Format::kRgba8, 2, 2, patch_pixels,
                     /*offset_x=*/1, /*offset_y=*/2));

  ASSERT_EQ(callbacks_.upload_calls, 2);
  const auto& updated = callbacks_.textures[100].rgba;
  for (uint32_t y = 0; y < 4; ++y) {
    for (uint32_t x = 0; x < 4; ++x) {
      const size_t idx = (y * 4 + x) * 4;
      const bool in_patch = (x >= 1 && x < 3 && y >= 2 && y < 4);
      EXPECT_EQ(updated[idx + 0], in_patch ? 200 : 10);
      EXPECT_EQ(updated[idx + 1], in_patch ? 100 : 20);
      EXPECT_EQ(updated[idx + 2], in_patch ? 50 : 30);
      EXPECT_EQ(updated[idx + 3], in_patch ? 128 : 255);
    }
  }

  // Destroy the texture.
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(42, netui::Texture::Action::kDestroy,
                     netui::Texture::Format::kRgba8, 0, 0, /*pixels=*/{}));

  EXPECT_EQ(callbacks_.upload_calls, 3);
  EXPECT_TRUE(callbacks_.textures[100].destroyed);
}

TEST_F(RemoteUiTest, ExpandsA8FontTextureToRgba8) {
  std::vector<uint8_t> a8_pixels = {0, 64, 128, 255};
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(7, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kAlpha8, 2, 2, a8_pixels));

  ASSERT_EQ(callbacks_.upload_calls, 1);
  const auto& rgba = callbacks_.textures[100].rgba;
  ASSERT_EQ(rgba.size(), 16u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(rgba[i * 4 + 0], 255);
    EXPECT_EQ(rgba[i * 4 + 1], 255);
    EXPECT_EQ(rgba[i * 4 + 2], 255);
    EXPECT_EQ(rgba[i * 4 + 3], a8_pixels[i]);
  }
}

TEST_F(RemoteUiTest, DefersTextureUploadUntilGpuReady) {
  callbacks_.gpu_ready = false;
  std::vector<uint8_t> pixels(2 * 2 * 4, 99);
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(11, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kRgba8, 2, 2, pixels));

  EXPECT_EQ(callbacks_.upload_calls, 0);

  // A draw frame arriving before GpuReady() omits draws whose textures are not
  // uploaded yet (never leaving ElemCount=0 / TexID=0 commands in CmdBuffer).
  const netui::SerializedFrame frame =
      RenderSerializedFrame("DeferredFrame", /*tex_id=*/11);
  remote_ui_.ProcessDrawFrameMessage(
      netui::EncodeDrawFrame(frame, /*frame_index=*/1));
  ImDrawData* draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  ASSERT_EQ(draw_data->CmdLists.Size, 1);
  EXPECT_EQ(draw_data->CmdLists[0]->CmdBuffer.Size, 0);

  // Once the GPU becomes ready and UpdateTextures() uploads the deferred
  // texture, the cached draw frame's draw commands are rebuilt with the newly
  // uploaded GPU texture handle without waiting for another network frame.
  callbacks_.gpu_ready = true;
  remote_ui_.UpdateTextures();
  EXPECT_EQ(callbacks_.upload_calls, 1);
  EXPECT_EQ(callbacks_.textures[100].rgba, pixels);
  draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  ASSERT_EQ(draw_data->CmdLists.Size, 1);
  ASSERT_GT(draw_data->CmdLists[0]->CmdBuffer.Size, 0);
  EXPECT_GT(draw_data->CmdLists[0]->CmdBuffer[0].ElemCount, 0u);
  EXPECT_EQ(draw_data->CmdLists[0]->CmdBuffer[0].TexRef._TexID,
            static_cast<ImTextureID>(100));

  // Calling UpdateTextures() again does not re-upload already-uploaded entries.
  remote_ui_.UpdateTextures();
  EXPECT_EQ(callbacks_.upload_calls, 1);
}

TEST_F(RemoteUiTest, RejectsTruncatedPixelsAndOutOfBoundsTexturePatch) {
  std::vector<uint8_t> valid_pixels(4 * 4 * 4, 42);
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(9, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kRgba8, 4, 4, valid_pixels));
  ASSERT_EQ(callbacks_.upload_calls, 1);

  // Truncated pixel buffer: declares 4x4 RGBA8 (64 bytes) but only carries 8.
  std::vector<uint8_t> short_pixels(8, 1);
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(10, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kRgba8, 4, 4, short_pixels));
  EXPECT_EQ(callbacks_.upload_calls, 1);

  // A pixel format this viewer does not know.
  remote_ui_.ProcessTextureMessage(MakeTextureMsg(
      11, netui::Texture::Action::kCreate,
      static_cast<netui::Texture::Format>(99), 4, 4, valid_pixels));
  EXPECT_EQ(callbacks_.upload_calls, 1);

  // Out-of-bounds patch rectangle: 2x2 at offset (3, 3) exceeds 4x4 mirror.
  std::vector<uint8_t> patch_pixels(2 * 2 * 4, 99);
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(9, netui::Texture::Action::kUpdate,
                     netui::Texture::Format::kRgba8, 2, 2, patch_pixels,
                     /*offset_x=*/3, /*offset_y=*/3));
  EXPECT_EQ(callbacks_.upload_calls, 1);
  EXPECT_EQ(callbacks_.textures[100].rgba, valid_pixels);
}

TEST_F(RemoteUiTest, ProcessesUncompressedAndDeltaCompressedDrawFrames) {
  RegisterDummyTexture(/*tex_id=*/1);

  const netui::SerializedFrame frame1 = RenderSerializedFrame("Hello Frame 1");
  const netui::SerializedFrame frame2 = RenderSerializedFrame("Hello Frame 2");
  const netui::Bytes msg1 = netui::EncodeDrawFrame(frame1, /*frame_index=*/1);
  const netui::Bytes msg2 =
      netui::EncodeDrawFrame(frame2, /*frame_index=*/2, &frame1);

  netui::SerializedFrame tmp_prev;
  uint64_t tmp_idx = 0;
  netui::DecodedFrame decoded2;
  bool req = false;
  ASSERT_TRUE(netui::DecodeDrawFrame(netui::EncodeDrawFrame(frame2, 1),
                                     &tmp_prev, &tmp_idx, &decoded2, &req));
  const size_t expected_vtx_count = decoded2.lists[0].verts.size();
  const size_t expected_idx_count = decoded2.lists[0].indices.size();

  remote_ui_.ProcessDrawFrameMessage(msg1);

  ImDrawData* draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  EXPECT_TRUE(draw_data->Valid);
  EXPECT_GT(draw_data->TotalVtxCount, 0);
  ASSERT_EQ(draw_data->CmdLists.Size, 1);
  EXPECT_GT(draw_data->CmdLists[0]->CmdBuffer.Size, 0);
  EXPECT_EQ(draw_data->CmdLists[0]->CmdBuffer[0].TexRef._TexID,
            static_cast<ImTextureID>(100));

  // Delta-compressed frame 2 decompresses against frame 1.
  remote_ui_.ProcessDrawFrameMessage(msg2);
  draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  EXPECT_EQ(static_cast<size_t>(draw_data->TotalVtxCount), expected_vtx_count);
  EXPECT_EQ(static_cast<size_t>(draw_data->TotalIdxCount), expected_idx_count);
}

TEST_F(RemoteUiTest, ProcessesADrawFrameWithoutDraws) {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(640, 480);
  unsigned char* font_pixels = nullptr;
  int font_w = 0, font_h = 0;
  io.Fonts->GetTexDataAsAlpha8(&font_pixels, &font_w, &font_h);
  io.Fonts->SetTexID(static_cast<ImTextureID>(1));
  ImGui::NewFrame();  // Draws nothing.
  ImGui::Render();

  netui::SerializedFrame empty;
  netui::SerializeDrawData(ImGui::GetDrawData(), ImGuiMouseCursor_Arrow,
                           &empty);
  ASSERT_TRUE(empty.lists.empty());

  remote_ui_.ProcessDrawFrameMessage(
      netui::EncodeDrawFrame(empty, /*frame_index=*/1));
  ImDrawData* draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  EXPECT_EQ(draw_data->CmdLists.Size, 0);
}

TEST_F(RemoteUiTest,
       BuildsOneDrawListPerWindowClampsClipRectAndSkipsUnmappedTextures) {
  RegisterDummyTexture(/*tex_id=*/1);
  remote_ui_.SetScreenArea(120.0f, 80.0f);

  const netui::SerializedFrame frame =
      RenderSerializedFrame("First", /*tex_id=*/1, /*second_window=*/true);
  ASSERT_GE(frame.lists.size(), 2u);
  const netui::Bytes wire = netui::EncodeDrawFrame(frame, /*frame_index=*/1);

  netui::SerializedFrame tmp_prev;
  uint64_t tmp_idx = 0;
  netui::DecodedFrame expected;
  bool req = false;
  ASSERT_TRUE(
      netui::DecodeDrawFrame(wire, &tmp_prev, &tmp_idx, &expected, &req));

  remote_ui_.ProcessDrawFrameMessage(wire);

  ImDrawData* draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  ASSERT_EQ(static_cast<size_t>(draw_data->CmdLists.Size),
            expected.lists.size());
  ImDrawList* list0 = draw_data->CmdLists[0];
  ImDrawList* list1 = draw_data->CmdLists[1];
  EXPECT_EQ(static_cast<size_t>(list0->VtxBuffer.Size),
            expected.lists[0].verts.size());
  EXPECT_EQ(static_cast<size_t>(list1->VtxBuffer.Size),
            expected.lists[1].verts.size());
  EXPECT_EQ(static_cast<size_t>(list1->IdxBuffer.Size),
            expected.lists[1].indices.size());
  EXPECT_EQ(list1->IdxBuffer[0],
            static_cast<ImDrawIdx>(expected.lists[1].indices[0]));
  for (int i = 0; i < draw_data->CmdLists.Size; ++i) {
    for (const ImDrawCmd& cmd : draw_data->CmdLists[i]->CmdBuffer) {
      EXPECT_EQ(cmd.VtxOffset, 0u);
      EXPECT_LE(cmd.ClipRect.z, 120.0f);
      EXPECT_LE(cmd.ClipRect.w, 80.0f);
    }
  }

  // A draw frame referencing an unmapped texture ID omits the draw command from
  // CmdBuffer (rather than leaving an ImDrawCmd with ElemCount=0 and _TexID=0,
  // which Filament's Renderable::SetMesh treats as "draw the entire index
  // buffer" with ObjectManager::kDecor), and reuses the existing ImDrawList.
  const netui::SerializedFrame unmapped =
      RenderSerializedFrame("Unmapped", /*tex_id=*/999);
  remote_ui_.ProcessDrawFrameMessage(
      netui::EncodeDrawFrame(unmapped, /*frame_index=*/2));
  draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  ASSERT_EQ(draw_data->CmdLists.Size, 1);
  EXPECT_EQ(draw_data->CmdLists[0], list0);
  EXPECT_EQ(draw_data->CmdLists[0]->CmdBuffer.Size, 0);
}

TEST_F(RemoteUiTest, SocketHandshakeReceiveAndInputRoundTrip) {
  FakeWebSocket* socket = ConnectFakeSocket();
  ASSERT_TRUE(remote_ui_.HasSocket());

  // Nothing is sent before the socket opens.
  socket->state = ReadyState::kConnecting;
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);
  remote_ui_.CaptureAndSendInput();
  EXPECT_TRUE(socket->sent.empty());

  // Then RemoteUi sends its Hello handshake, as one message.
  socket->state = ReadyState::kOpen;
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/2);
  ASSERT_EQ(socket->sent.size(), 1u);
  const netui::Hello handshake = CommandIn<netui::Hello>(socket->sent[0]);
  EXPECT_EQ(handshake.header.kind, netui::MessageKind::kHello);

  // A texture and a draw frame arrive, one command per message.
  std::vector<uint8_t> pixels(4 * 4 * 4, 200);
  const netui::Bytes tex_msg =
      MakeTextureMsg(1, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kRgba8, 4, 4, pixels);
  const netui::Bytes draw_msg = netui::EncodeDrawFrame(
      RenderSerializedFrame("Streamed over socket", 1), 1);
  const uint64_t expected_bytes = tex_msg.size() + draw_msg.size();
  socket->received.push_back(tex_msg);
  socket->received.push_back(draw_msg);

  remote_ui_.ReceiveAndProcessCommands(/*frame=*/3);
  ASSERT_NE(remote_ui_.RemoteDrawData(), nullptr);
  EXPECT_EQ(callbacks_.upload_calls, 1);
  EXPECT_EQ(remote_ui_.ConsumeByteCount(), expected_bytes);
  EXPECT_EQ(remote_ui_.ConsumeByteCount(), 0u);

  // CaptureAndSendInput sends an Input message back. The screen size it
  // reports is the screen area (the canvas minus the local role bar), not the
  // display size.
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(1280, 720);
  remote_ui_.SetScreenArea(1280.0f, 688.0f);
  io.MousePos = ImVec2(100.5f, 200.25f);
  io.WantCaptureMouse = false;
  io.WantCaptureKeyboard = false;
  io.InputQueueCharacters.push_back('a');

  remote_ui_.CaptureAndSendInput();
  ASSERT_EQ(socket->sent.size(), 2u);
  const netui::Input input_cmd = CommandIn<netui::Input>(socket->sent[1]);
  EXPECT_EQ(input_cmd.header.kind, netui::MessageKind::kInput);
  EXPECT_EQ(input_cmd.screen_size[0], 1280u);
  EXPECT_EQ(input_cmd.screen_size[1], 688u);
  EXPECT_EQ(input_cmd.mouse_pos[0], 100.5f);
  EXPECT_EQ(input_cmd.mouse_pos[1], 200.25f);
  EXPECT_NE(input_cmd.header.flags & netui::Input::kRequestKeyframe, 0u);
  EXPECT_EQ(input_cmd.char_count, 1u);
  EXPECT_EQ(input_cmd.chars[0], static_cast<uint32_t>('a'));

  // Once the keyframe request has been sent, subsequent inputs clear
  // Input::kRequestKeyframe unless a delta frame is missing its reference.
  io.InputQueueCharacters.resize(0);
  remote_ui_.CaptureAndSendInput();
  ASSERT_EQ(socket->sent.size(), 3u);
  EXPECT_EQ(CommandIn<netui::Input>(socket->sent[2]).header.flags &
                netui::Input::kRequestKeyframe,
            0u);

  // Receiving a delta frame with a missing reference frame re-arms
  // Input::kRequestKeyframe on the next Input.
  const netui::SerializedFrame prev_ref = RenderSerializedFrame("Prev", 1);
  const netui::SerializedFrame gap_frame = RenderSerializedFrame("Gap", 1);
  remote_ui_.ProcessDrawFrameMessage(
      netui::EncodeDrawFrame(gap_frame, /*frame_index=*/99, &prev_ref));
  remote_ui_.CaptureAndSendInput();
  ASSERT_EQ(socket->sent.size(), 4u);
  EXPECT_NE(CommandIn<netui::Input>(socket->sent[3]).header.flags &
                netui::Input::kRequestKeyframe,
            0u);
}

TEST_F(RemoteUiTest, DropsMessagesThatAreNotOneCommand) {
  FakeWebSocket* socket = ConnectFakeSocket();
  std::vector<uint8_t> pixels(2 * 2 * 4, 7);
  const netui::Bytes message =
      MakeTextureMsg(5, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kRgba8, 2, 2, pixels);

  socket->received.emplace_back(message.begin(),
                                message.begin() + 4);  // Shorter than a header.
  socket->received.emplace_back(message.begin(), message.end() - 1);
  netui::Bytes doubled = message;
  doubled.insert(doubled.end(), message.begin(), message.end());
  socket->received.push_back(std::move(doubled));  // Two commands.
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);
  EXPECT_EQ(callbacks_.upload_calls, 0);

  socket->received.push_back(message);
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/2);
  EXPECT_EQ(callbacks_.upload_calls, 1);
}

TEST_F(RemoteUiTest, StartsOverOnEachConnection) {
  FakeWebSocket* first = ConnectFakeSocket();
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);
  remote_ui_.CaptureAndSendInput();
  remote_ui_.CaptureAndSendInput();
  ASSERT_EQ(first->sent.size(), 3u);
  EXPECT_EQ(CommandIn<netui::Input>(first->sent[2]).header.flags &
                netui::Input::kRequestKeyframe,
            0u);

  // The server closes the connection, e.g. as another page took control.
  first->state = ReadyState::kClosed;
  first->close_code = 4001;
  EXPECT_EQ(remote_ui_.ConnectionState(), ReadyState::kClosed);
  EXPECT_EQ(remote_ui_.CloseCode(), 4001);

  // A new connection sends the handshake again, then asks for a full frame.
  FakeWebSocket* second = ConnectFakeSocket();
  EXPECT_EQ(remote_ui_.ConnectionState(), ReadyState::kOpen);
  EXPECT_EQ(remote_ui_.CloseCode(), 0);
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/2);
  remote_ui_.CaptureAndSendInput();
  ASSERT_EQ(second->sent.size(), 2u);
  EXPECT_EQ(CommandIn<netui::Hello>(second->sent[0]).header.kind,
            netui::MessageKind::kHello);
  EXPECT_NE(CommandIn<netui::Input>(second->sent[1]).header.flags &
                netui::Input::kRequestKeyframe,
            0u);

  remote_ui_.Shutdown();
  EXPECT_FALSE(remote_ui_.HasSocket());
  EXPECT_EQ(remote_ui_.ConnectionState(), ReadyState::kDisconnected);
}

TEST_F(RemoteUiTest, AcksEachBatchOfMessagesHoldingADrawFrame) {
  FakeWebSocket* socket = ConnectFakeSocket();
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);
  EXPECT_TRUE(socket->sent_text.empty());

  // Textures alone are not acked: the server waits for the frame's ack.
  std::vector<uint8_t> pixels(4 * 4 * 4, 200);
  socket->received.push_back(MakeTextureMsg(1, netui::Texture::Action::kCreate,
                                            netui::Texture::Format::kRgba8, 4,
                                            4, pixels));
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/2);
  EXPECT_TRUE(socket->sent_text.empty());

  const netui::SerializedFrame serialized = RenderSerializedFrame("Acked", 1);
  socket->received.push_back(
      netui::EncodeDrawFrame(serialized, /*frame_index=*/1));
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/3);
  EXPECT_EQ(socket->sent_text, std::vector<std::string>{"ui_ack=1"});

  // Nothing new, nothing to ack.
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/4);
  EXPECT_EQ(socket->sent_text.size(), 1u);

  // Frames received together get one ack, for the newest: the server only
  // waits for that one.
  for (uint64_t frame_index : {2, 3}) {
    socket->received.push_back(netui::EncodeDrawFrame(serialized, frame_index));
  }
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/5);
  EXPECT_EQ(socket->sent_text,
            (std::vector<std::string>{"ui_ack=1", "ui_ack=3"}));
}

TEST_F(RemoteUiTest, ClampsScreenSizeAndPreservesFloatMousePosition) {
  FakeWebSocket* socket = ConnectFakeSocket();
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);

  // Out-of-range values (such as Dear ImGui's "no mouse" position, -FLT_MAX),
  // which do not fit the wire's integers.
  remote_ui_.SetScreenArea(-1.0f, 1e9f);
  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(-FLT_MAX, FLT_MAX);
  remote_ui_.CaptureAndSendInput();

  ASSERT_EQ(socket->sent.size(), 2u);
  const netui::Input input = CommandIn<netui::Input>(socket->sent[1]);
  EXPECT_EQ(input.screen_size[0], 0u);
  EXPECT_EQ(input.screen_size[1], 65535u);
  EXPECT_EQ(input.mouse_pos[0], -FLT_MAX);
  EXPECT_EQ(input.mouse_pos[1], FLT_MAX);
}

TEST_F(RemoteUiTest, CopiesClipboardMessagesAndInterceptsOpenUrls) {
  FakeWebSocket* socket = ConnectFakeSocket();
  ImGui::SetClipboardText("");

  socket->received.push_back(netui::EncodeClipboard("camera_xml"));
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);
  EXPECT_STREQ(ImGui::GetClipboardText(), "camera_xml");
  EXPECT_TRUE(socket->sent_text.empty());  // Clipboard messages are not acked.

  // URLs marked with Clipboard::kOpenUrl open in the browser (or, for other
  // schemes, are refused) instead of overwriting the clipboard.
  socket->received.push_back(
      netui::EncodeClipboard("https://mujoco.org", /*open_url=*/true));
  socket->received.push_back(
      netui::EncodeClipboard("javascript:alert(1)", /*open_url=*/true));
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/2);
  EXPECT_STREQ(ImGui::GetClipboardText(), "camera_xml");
}

TEST_F(RemoteUiTest, StateQueueSynchronizesFullPayloadAndBoundsLatency) {
  StateQueue queue;
  EXPECT_TRUE(queue.Empty());

  auto make_payload = [](mjtNum q0, mjtNum q1, int spec, double azimuth,
                         int ngeom) {
    const mjtNum physics[] = {q0, q1};
    mjvCamera cam{};
    cam.azimuth = azimuth;
    mjvPerturb pert{};
    pert.select = spec;
    mjvOption vopt{};
    mjOption opt{};
    mjVisual vis{};
    mjStatistic stat{};
    std::vector<uint8_t> rflags(mjNRNDFLAG, 0);
    std::vector<mjvGeom> geoms(ngeom);
    for (int i = 0; i < ngeom; ++i) {
      geoms[i].objid = 100 + i;
    }
    // The viewer input rides along too; tag it with the azimuth to tell the
    // payloads apart.
    const ViewerInputState viewer_input{.camera_generation =
                                            static_cast<uint32_t>(azimuth)};
    return SerializeStatePayload(
        /*model_crc32=*/42u, spec, physics, sizeof(physics), cam, pert, vopt,
        opt, vis, stat, rflags, geoms.data(), geoms.size(), viewer_input);
  };

  const std::vector<std::byte> b1 = make_payload(1.0, 10.0, 7, 15.0, 1);
  const std::vector<std::byte> b2 = make_payload(2.0, 20.0, 7, 30.0, 2);
  const std::vector<std::byte> b3 = make_payload(3.0, 30.0, 3, 45.0, 0);
  const std::vector<std::byte> b4 = make_payload(4.0, 40.0, 4, 60.0, 3);

  StatePayloadView v1, v2, v3, v4;
  ASSERT_TRUE(ParseStatePayload(b1.data(), b1.size(), &v1));
  ASSERT_TRUE(ParseStatePayload(b2.data(), b2.size(), &v2));
  ASSERT_TRUE(ParseStatePayload(b3.data(), b3.size(), &v3));
  ASSERT_TRUE(ParseStatePayload(b4.data(), b4.size(), &v4));

  // Burst of 2 packets in one VSync window followed by 0 in the next:
  // both frames get distinct consecutive physics, camera, and extra_geoms
  // states without lag between physics and render state.
  queue.Push(v1);
  queue.Push(v2);
  EXPECT_EQ(queue.Size(), 2u);

  StateQueue::Entry out;
  ASSERT_TRUE(queue.Pop(&out));
  EXPECT_EQ(out.sig, 7);
  EXPECT_THAT(out.state, testing::ElementsAre(1.0, 10.0));
  ASSERT_TRUE(out.has_render_state);
  EXPECT_DOUBLE_EQ(out.render_state.camera.azimuth, 15.0);
  EXPECT_EQ(out.viewer_input.camera_generation, 15u);
  ASSERT_EQ(out.extra_geoms.size(), 1u);
  EXPECT_EQ(out.extra_geoms[0].objid, 100);

  ASSERT_TRUE(queue.Pop(&out));
  EXPECT_EQ(out.sig, 7);
  EXPECT_THAT(out.state, testing::ElementsAre(2.0, 20.0));
  ASSERT_TRUE(out.has_render_state);
  EXPECT_DOUBLE_EQ(out.render_state.camera.azimuth, 30.0);
  EXPECT_EQ(out.viewer_input.camera_generation, 30u);
  ASSERT_EQ(out.extra_geoms.size(), 2u);
  EXPECT_EQ(out.extra_geoms[1].objid, 101);
  EXPECT_FALSE(queue.Pop(&out));

  // Overflowing past kMaxDepth (3) drops the oldest unconsumed snapshot.
  v1.physics_spec = 1;
  v2.physics_spec = 2;
  queue.Push(v1);
  queue.Push(v2);
  queue.Push(v3);
  queue.Push(v4);
  EXPECT_EQ(queue.Size(), 3u);

  ASSERT_TRUE(queue.Pop(&out));
  EXPECT_EQ(out.sig, 2);
  EXPECT_THAT(out.state, testing::ElementsAre(2.0, 20.0));
  EXPECT_DOUBLE_EQ(out.render_state.camera.azimuth, 30.0);
  ASSERT_TRUE(queue.Pop(&out));
  EXPECT_EQ(out.sig, 3);
  EXPECT_THAT(out.state, testing::ElementsAre(3.0, 30.0));
  EXPECT_DOUBLE_EQ(out.render_state.camera.azimuth, 45.0);
  EXPECT_TRUE(out.extra_geoms.empty());
  ASSERT_TRUE(queue.Pop(&out));
  EXPECT_EQ(out.sig, 4);
  EXPECT_THAT(out.state, testing::ElementsAre(4.0, 40.0));
  EXPECT_DOUBLE_EQ(out.render_state.camera.azimuth, 60.0);
  EXPECT_EQ(out.extra_geoms.size(), 3u);
  EXPECT_TRUE(queue.Empty());
}

TEST_F(RemoteUiTest, SessionAcceptanceMouseCursorAndTextureCleanupOnShutdown) {
  FakeWebSocket* socket = ConnectFakeSocket();
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);
  EXPECT_FALSE(remote_ui_.IsSessionAccepted());

  netui::Hello server_hello;
  const auto hello_bytes = std::as_bytes(std::span(&server_hello, 1));
  socket->received.emplace_back(hello_bytes.begin(), hello_bytes.end());
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/2);
  EXPECT_TRUE(remote_ui_.IsSessionAccepted());

  RegisterDummyTexture(/*tex_id=*/1);
  ASSERT_TRUE(callbacks_.textures.count(100));
  EXPECT_FALSE(callbacks_.textures[100].destroyed);

  netui::SerializedFrame frame = RenderSerializedFrame("Cursor", /*tex_id=*/1);
  frame.mouse_cursor = ImGuiMouseCursor_TextInput;
  remote_ui_.ProcessDrawFrameMessage(
      netui::EncodeDrawFrame(frame, /*frame_index=*/1));
  EXPECT_EQ(remote_ui_.MouseCursor(), ImGuiMouseCursor_TextInput);

  // Out-of-range cursor values fall back to Arrow.
  frame.mouse_cursor = 999;
  remote_ui_.ProcessDrawFrameMessage(
      netui::EncodeDrawFrame(frame, /*frame_index=*/2));
  EXPECT_EQ(remote_ui_.MouseCursor(), ImGuiMouseCursor_Arrow);

  frame.mouse_cursor = ImGuiMouseCursor_ResizeEW;
  remote_ui_.ProcessDrawFrameMessage(
      netui::EncodeDrawFrame(frame, /*frame_index=*/3));
  ASSERT_EQ(remote_ui_.MouseCursor(), ImGuiMouseCursor_ResizeEW);

  // Shutdown destroys live GPU textures and clears session state.
  remote_ui_.Shutdown();
  EXPECT_FALSE(remote_ui_.IsSessionAccepted());
  EXPECT_EQ(remote_ui_.MouseCursor(), ImGuiMouseCursor_Arrow);
  EXPECT_EQ(remote_ui_.RemoteDrawData(), nullptr);
  EXPECT_TRUE(callbacks_.textures[100].destroyed);
}

TEST_F(RemoteUiTest, ExpandsPartialA8TextureUpdateInPlace) {
  std::vector<uint8_t> initial_a8(4 * 4, 10);
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(77, netui::Texture::Action::kCreate,
                     netui::Texture::Format::kAlpha8, 4, 4, initial_a8));
  ASSERT_EQ(callbacks_.upload_calls, 1);

  const std::vector<uint8_t> patch_a8 = {100, 110, 120, 130};
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(77, netui::Texture::Action::kUpdate,
                     netui::Texture::Format::kAlpha8, 2, 2, patch_a8,
                     /*offset_x=*/1, /*offset_y=*/1));
  ASSERT_EQ(callbacks_.upload_calls, 2);

  const auto& rgba = callbacks_.textures[100].rgba;
  ASSERT_EQ(rgba.size(), 4u * 4u * 4u);
  for (uint32_t y = 0; y < 4; ++y) {
    for (uint32_t x = 0; x < 4; ++x) {
      const size_t idx = (y * 4 + x) * 4;
      EXPECT_EQ(rgba[idx + 0], 255);
      EXPECT_EQ(rgba[idx + 1], 255);
      EXPECT_EQ(rgba[idx + 2], 255);
      if (x >= 1 && x < 3 && y >= 1 && y < 3) {
        EXPECT_EQ(rgba[idx + 3], patch_a8[(y - 1) * 2 + (x - 1)]);
      } else {
        EXPECT_EQ(rgba[idx + 3], 10);
      }
    }
  }

  // Updating an unknown texture ID is ignored without uploading.
  remote_ui_.ProcessTextureMessage(
      MakeTextureMsg(999, netui::Texture::Action::kUpdate,
                     netui::Texture::Format::kAlpha8, 2, 2, patch_a8,
                     /*offset_x=*/0, /*offset_y=*/0));
  EXPECT_EQ(callbacks_.upload_calls, 2);
}

TEST_F(RemoteUiTest, PreservesPhysicalModifierKeysUnderMacOsBehaviors) {
  FakeWebSocket* socket = ConnectFakeSocket();
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);

  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(800, 600);
  io.DisplayFramebufferScale = ImVec2(2.0f, 2.0f);
  io.ConfigInputTrickleEventQueue = false;
  io.ConfigMacOSXBehaviors = true;
  unsigned char* font_pixels = nullptr;
  int font_w = 0, font_h = 0;
  io.Fonts->GetTexDataAsAlpha8(&font_pixels, &font_w, &font_h);
  io.Fonts->SetTexID(static_cast<ImTextureID>(1));

  io.AddKeyEvent(ImGuiKey_LeftSuper, true);
  io.AddKeyEvent(ImGuiMod_Super, true);
  ImGui::NewFrame();
  // Locally, Dear ImGui swaps Super to Ctrl when ConfigMacOSXBehaviors is set.
  EXPECT_TRUE(io.KeyCtrl);
  EXPECT_TRUE(ImGui::IsKeyDown(ImGuiKey_LeftCtrl));

  remote_ui_.CaptureAndSendInput();
  ImGui::EndFrame();

  ASSERT_EQ(socket->sent.size(), 2u);
  const netui::Input sent = CommandIn<netui::Input>(socket->sent[1]);
  EXPECT_FLOAT_EQ(sent.framebuffer_scale, 2.0f);
  EXPECT_NE(sent.header.flags & netui::Input::kMacOsBehaviors, 0u);
  EXPECT_TRUE(sent.IsImGuiKeyDown(ImGuiKey_LeftSuper));
  EXPECT_TRUE(sent.IsImGuiKeyDown(ImGuiKey_ReservedForModSuper));
  EXPECT_FALSE(sent.IsImGuiKeyDown(ImGuiKey_LeftCtrl));
  EXPECT_FALSE(sent.IsImGuiKeyDown(ImGuiKey_ReservedForModCtrl));
}
}  // namespace
}  // namespace mujoco::studio
