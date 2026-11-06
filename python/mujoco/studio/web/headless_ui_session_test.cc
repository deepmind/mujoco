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

#include "headless_ui_session.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <imgui.h>
#include <imgui_internal.h>
#include "ui_codec.h"

namespace mujoco::studio {
namespace {

template <typename T>
netui::Bytes WireBytes(const T& command) {
  const auto* raw = reinterpret_cast<const std::byte*>(&command);
  return netui::Bytes(raw, raw + sizeof(command));
}

template <typename T>
T HeaderOf(netui::BytesView message) {
  T header;
  std::memcpy(&header, message.data(), sizeof(header));
  return header;
}

netui::MessageKind KindOf(netui::BytesView message) {
  return HeaderOf<netui::MessageHeader>(message).kind;
}

bool HoldsItsPixels(netui::BytesView message) {
  if (message.size() < sizeof(netui::Texture)) return false;
  const netui::Texture tex = HeaderOf<netui::Texture>(message);
  if (tex.header.size != message.size()) return false;
  const size_t bpp = tex.format == netui::Texture::Format::kAlpha8 ? 1 : 4;
  const size_t pixels_size = tex.action == netui::Texture::Action::kDestroy
                                 ? 0
                                 : bpp * tex.width * tex.height;
  return message.size() == sizeof(netui::Texture) + pixels_size;
}

const uint8_t* TexturePixels(netui::BytesView message) {
  return reinterpret_cast<const uint8_t*>(message.data() +
                                          sizeof(netui::Texture));
}

void ExpectSameDrawData(const netui::DecodedFrame& actual,
                        const netui::SerializedFrame& expected_serialized) {
  netui::SerializedFrame dummy_prev;
  uint64_t dummy_idx = 0;
  netui::DecodedFrame expected;
  bool request_keyframe = false;
  ASSERT_TRUE(netui::DecodeDrawFrame(
      netui::EncodeDrawFrame(expected_serialized, 1), &dummy_prev, &dummy_idx,
      &expected, &request_keyframe));
  ASSERT_EQ(actual.lists.size(), expected.lists.size());
  for (size_t g = 0; g < expected.lists.size(); ++g) {
    const netui::DrawList& a = actual.lists[g];
    const netui::DrawList& e = expected.lists[g];
    ASSERT_EQ(a.verts.size(), e.verts.size());
    ASSERT_EQ(a.indices, e.indices);
    ASSERT_EQ(a.commands.size(), e.commands.size());
    EXPECT_EQ(std::memcmp(a.verts.data(), e.verts.data(),
                          e.verts.size() * sizeof(netui::Vert)),
              0);
    for (size_t c = 0; c < e.commands.size(); ++c) {
      EXPECT_EQ(a.commands[c].tex_id, e.commands[c].tex_id);
      EXPECT_EQ(a.commands[c].vtx_offset, e.commands[c].vtx_offset);
      EXPECT_EQ(a.commands[c].idx_offset, e.commands[c].idx_offset);
      EXPECT_EQ(a.commands[c].elem_count, e.commands[c].elem_count);
      EXPECT_EQ(std::memcmp(a.commands[c].clip_rect, e.commands[c].clip_rect,
                            sizeof(e.commands[c].clip_rect)),
                0);
    }
  }
}

class HeadlessUiSessionTest : public ::testing::Test {
 protected:
  HeadlessUiSessionTest() : context_(ImGui::CreateContext()) {
    ImGui::SetCurrentContext(context_);
    ImGuiIO& io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(640, 480);
    io.DeltaTime = 1.0f / 60.0f;
    // Process every queued input event in the frame it arrives in, so the
    // expectations below don't depend on Dear ImGui spreading them out.
    io.ConfigInputTrickleEventQueue = false;

    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    platform_io.Platform_ClipboardUserData = &session_;
    platform_io.Platform_SetClipboardTextFn = [](ImGuiContext*,
                                                 const char* text) {
      auto* session = static_cast<HeadlessUiSession*>(
          ImGui::GetPlatformIO().Platform_ClipboardUserData);
      if (session && text) {
        session->QueueClipboard(text);
      }
    };
  }

  ~HeadlessUiSessionTest() override { ImGui::DestroyContext(context_); }

  void StartSession() {
    const netui::Bytes handshake = WireBytes(netui::Hello());
    netui::Bytes reply;
    std::string error;
    ASSERT_TRUE(session_.StartSession(handshake, &reply, &error)) << error;
  }

  // Starts a session and sends the browser's first input, as the browser does
  // right after its handshake: frames are captured for its screen from then on.
  void Connect() {
    StartSession();
    netui::Input input;
    input.screen_size[0] = 640;
    input.screen_size[1] = 480;
    ASSERT_TRUE(PushInput(input));
  }

  // Runs one UI frame through the session, drawing `text` in a window, and
  // optionally an image. Dear ImGui adds glyphs to the font texture as text
  // first uses them, so tests that count messages reuse the same letters.
  void RunFrame(const char* text, ImTextureData* image = nullptr) {
    BeginFrame();
    EndFrame(text, image);
  }

  void BeginFrame() {
    session_.ApplyInput();
    ImGui::NewFrame();
  }

  void EndFrame(const char* text, ImTextureData* image = nullptr) {
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(300, 200));
    ImGui::Begin("Window");
    ImGui::TextUnformatted(text);
    ImGui::Button("Button");
    if (image != nullptr) {
      ImGui::Image(image->GetTexRef(), ImVec2(16, 16));
    }
    ImGui::End();
    ImGui::Render();
    session_.CaptureFrame(ImGuiMouseCursor_Arrow);
  }

  static netui::SerializedFrame CurrentDrawData() {
    netui::SerializedFrame frame;
    netui::SerializeDrawData(ImGui::GetDrawData(), ImGuiMouseCursor_Arrow,
                             &frame);
    return frame;
  }

  bool PushInput(const netui::Input& input) {
    const netui::Bytes message = WireBytes(input);
    return session_.PushInput(message);
  }

  ImGuiContext* context_ = nullptr;
  HeadlessUiSession session_;
};

TEST_F(HeadlessUiSessionTest, StartSessionChecksTheHandshake) {
  netui::Bytes reply;
  std::string error;

  const netui::Bytes full_handshake = WireBytes(netui::Hello());
  EXPECT_FALSE(session_.StartSession(netui::BytesView(full_handshake).first(4),
                                     &reply, &error));

  netui::Hello other_version;
  other_version.header.version = netui::kCodecVersion + 1;
  const netui::Bytes other = WireBytes(other_version);
  EXPECT_FALSE(session_.StartSession(other, &reply, &error));
  EXPECT_NE(error.find("protocol"), std::string::npos);
  EXPECT_FALSE(session_.IsActive());

  ASSERT_TRUE(session_.StartSession(full_handshake, &reply, &error));
  EXPECT_TRUE(session_.IsActive());
  ASSERT_EQ(reply.size(), sizeof(netui::Hello));
  const netui::Hello hello = HeaderOf<netui::Hello>(reply);
  EXPECT_EQ(hello.header.kind, netui::MessageKind::kHello);
}

TEST_F(HeadlessUiSessionTest, SendsNothingWithoutASession) {
  RunFrame("Hello");
  EXPECT_TRUE(session_.TakeMessages().empty());
  EXPECT_FALSE(PushInput(netui::Input()));
}

TEST_F(HeadlessUiSessionTest, HasInputOnceTheSessionsBrowserSendsInput) {
  StartSession();
  EXPECT_FALSE(session_.HasInput());
  ASSERT_TRUE(PushInput(netui::Input()));
  EXPECT_TRUE(session_.HasInput());

  session_.EndSession();
  EXPECT_FALSE(session_.HasInput());
  StartSession();
  EXPECT_FALSE(session_.HasInput());  // The new browser has not sent any yet.
}

TEST_F(HeadlessUiSessionTest, FirstFrameSendsTexturesThenAFullFrame) {
  Connect();
  RunFrame("Hello");

  std::vector<netui::Bytes> messages = session_.TakeMessages();
  ASSERT_GE(messages.size(), 2u);
  std::vector<uint64_t> texture_ids;
  for (size_t i = 0; i + 1 < messages.size(); ++i) {
    ASSERT_EQ(KindOf(messages[i]), netui::MessageKind::kTexture);
    const netui::Texture texture = HeaderOf<netui::Texture>(messages[i]);
    EXPECT_EQ(texture.header.size, messages[i].size());
    EXPECT_EQ(texture.action, netui::Texture::Action::kCreate);
    EXPECT_TRUE(HoldsItsPixels(messages[i]));
    texture_ids.push_back(texture.tex_id);
  }

  ASSERT_EQ(KindOf(messages.back()), netui::MessageKind::kDrawFrame);
  const netui::DrawFrame hdr = HeaderOf<netui::DrawFrame>(messages.back());
  EXPECT_EQ(hdr.header.size, messages.back().size());
  EXPECT_EQ(hdr.header.flags & netui::DrawFrame::kCompressed, 0u);
  EXPECT_EQ(hdr.frame_index, 1u);

  netui::SerializedFrame decoder_prev;
  uint64_t decoder_idx = 0;
  netui::DecodedFrame frame;
  bool request_keyframe = false;
  ASSERT_TRUE(netui::DecodeDrawFrame(messages.back(), &decoder_prev,
                                     &decoder_idx, &frame, &request_keyframe));
  ExpectSameDrawData(frame, CurrentDrawData());
  // The draw calls use the textures sent before the frame.
  for (const netui::DrawList& list : frame.lists) {
    for (const netui::DrawCommand& cmd : list.commands) {
      EXPECT_NE(std::find(texture_ids.begin(), texture_ids.end(), cmd.tex_id),
                texture_ids.end());
    }
  }

  // Nothing new to send until the next frame.
  EXPECT_TRUE(session_.TakeMessages().empty());
}

TEST_F(HeadlessUiSessionTest, CompressesAgainstTheLastFrameTaken) {
  Connect();
  RunFrame("First");
  std::vector<netui::Bytes> first = session_.TakeMessages();
  netui::SerializedFrame decoder_prev;
  uint64_t decoder_idx = 0;
  netui::DecodedFrame decoded;
  bool request_keyframe = false;
  ASSERT_TRUE(netui::DecodeDrawFrame(first.back(), &decoder_prev, &decoder_idx,
                                     &decoded, &request_keyframe));

  // A frame captured but replaced before being taken is never sent.
  RunFrame("Frist");
  RunFrame("Fisrt");
  const netui::SerializedFrame expected = CurrentDrawData();
  std::vector<netui::Bytes> second = session_.TakeMessages();
  ASSERT_EQ(second.size(), 1u);
  const netui::DrawFrame packed_hdr = HeaderOf<netui::DrawFrame>(second.back());
  EXPECT_NE(packed_hdr.header.flags & netui::DrawFrame::kCompressed, 0u);
  EXPECT_EQ(packed_hdr.frame_index, 2u);

  ASSERT_TRUE(netui::DecodeDrawFrame(second.back(), &decoder_prev, &decoder_idx,
                                     &decoded, &request_keyframe));
  ExpectSameDrawData(decoded, expected);
}

// The browser acks a frame by echoing its index.
TEST_F(HeadlessUiSessionTest, LastFrameIndexIsThatOfTheLastFrameTaken) {
  EXPECT_EQ(session_.LastFrameIndex(), 0u);
  Connect();
  RunFrame("First");
  EXPECT_EQ(session_.LastFrameIndex(), 0u);  // Not taken yet.
  std::vector<netui::Bytes> messages = session_.TakeMessages();
  EXPECT_EQ(HeaderOf<netui::DrawFrame>(messages.back()).frame_index, 1u);
  EXPECT_EQ(session_.LastFrameIndex(), 1u);

  RunFrame("Frist");
  messages = session_.TakeMessages();
  EXPECT_EQ(HeaderOf<netui::DrawFrame>(messages.back()).frame_index, 2u);
  EXPECT_EQ(session_.LastFrameIndex(), 2u);
  EXPECT_TRUE(session_.TakeMessages().empty());  // Nothing new, no change.
  EXPECT_EQ(session_.LastFrameIndex(), 2u);
}

TEST_F(HeadlessUiSessionTest, SendsAFullFrameWhenTheBrowserAsks) {
  Connect();
  RunFrame("First");
  session_.TakeMessages();

  netui::Input input;
  input.header.flags = netui::Input::kRequestKeyframe;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Frist");
  std::vector<netui::Bytes> messages = session_.TakeMessages();
  EXPECT_EQ(HeaderOf<netui::DrawFrame>(messages.back()).header.flags &
                netui::DrawFrame::kCompressed,
            0u);

  input.header.flags = 0;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Fisrt");
  messages = session_.TakeMessages();
  EXPECT_NE(HeaderOf<netui::DrawFrame>(messages.back()).header.flags &
                netui::DrawFrame::kCompressed,
            0u);
}

TEST_F(HeadlessUiSessionTest, AppliesBrowserInput) {
  StartSession();
  RunFrame("Hello");

  netui::Input input;
  input.screen_size[0] = 800;
  input.screen_size[1] = 600;
  input.mouse_pos[0] = 50.0f;
  input.mouse_pos[1] = 60.0f;
  input.mouse_down_mask = 1u << ImGuiMouseButton_Left;
  input.SetImGuiKeyDown(ImGuiKey_A);
  input.SetImGuiKeyDown(ImGuiKey_ReservedForModCtrl);
  input.chars[0] = 'h';
  input.chars[1] = 'i';
  input.char_count = 2;
  input.mouse_wheel[0] = 5.0f;  // A running total: no scroll yet.
  ASSERT_TRUE(PushInput(input));

  BeginFrame();
  ImGuiIO& io = ImGui::GetIO();
  EXPECT_EQ(io.DisplaySize.x, 800.0f);
  EXPECT_EQ(io.DisplaySize.y, 600.0f);
  EXPECT_EQ(io.MousePos.x, 50.0f);
  EXPECT_EQ(io.MousePos.y, 60.0f);
  EXPECT_TRUE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_TRUE(ImGui::IsKeyDown(ImGuiKey_A));
  EXPECT_TRUE(io.KeyCtrl);
  ASSERT_EQ(io.InputQueueCharacters.Size, 2);
  EXPECT_EQ(io.InputQueueCharacters[0], 'h');
  EXPECT_EQ(io.InputQueueCharacters[1], 'i');
  EXPECT_EQ(io.MouseWheel, 0.0f);
  EndFrame("Hello");

  // The wheel total grows by 2: a scroll of 2. Releases are events too.
  input.mouse_down_mask = 0;
  input.SetImGuiKeyDown(ImGuiKey_A, false);
  input.char_count = 0;
  input.mouse_wheel[0] = 7.0f;
  ASSERT_TRUE(PushInput(input));
  BeginFrame();
  EXPECT_EQ(io.MouseWheel, 2.0f);
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
  EndFrame("Hello");
}

TEST_F(HeadlessUiSessionTest, ANewSessionReleasesInputAndResendsTextures) {
  StartSession();
  netui::Input input;
  input.mouse_down_mask = 1u << ImGuiMouseButton_Left;
  input.SetImGuiKeyDown(ImGuiKey_A);
  ASSERT_TRUE(PushInput(input));
  RunFrame("Hello");
  EXPECT_TRUE(ImGui::IsKeyDown(ImGuiKey_A));
  std::vector<netui::Bytes> first = session_.TakeMessages();
  const size_t first_texture_count = first.size() - 1;
  ASSERT_GT(first_texture_count, 0u);

  RunFrame("Hello");
  EXPECT_EQ(session_.TakeMessages().size(), 1u);  // Just the frame.

  session_.EndSession();
  EXPECT_FALSE(session_.IsActive());
  Connect();
  BeginFrame();
  EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EndFrame("Hello");

  std::vector<netui::Bytes> again = session_.TakeMessages();
  ASSERT_EQ(again.size(), first_texture_count + 1);
  for (size_t i = 0; i < first_texture_count; ++i) {
    EXPECT_EQ(again[i], first[i]);  // The same textures, created again.
  }
  EXPECT_EQ(HeaderOf<netui::DrawFrame>(again.back()).header.flags &
                netui::DrawFrame::kCompressed,
            0u);
}

TEST_F(HeadlessUiSessionTest, SendsNoFrameBeforeTheBrowsersFirstInput) {
  StartSession();
  RunFrame("Hello");  // Not laid out for the browser's screen.
  EXPECT_TRUE(session_.TakeMessages().empty());

  netui::Input input;
  input.screen_size[0] = 800;
  input.screen_size[1] = 600;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Hello");
  std::vector<netui::Bytes> messages = session_.TakeMessages();
  ASSERT_GE(messages.size(), 2u);
  EXPECT_EQ(KindOf(messages.front()), netui::MessageKind::kTexture);
  ASSERT_EQ(KindOf(messages.back()), netui::MessageKind::kDrawFrame);
  const netui::DrawFrame first = HeaderOf<netui::DrawFrame>(messages.back());
  EXPECT_EQ(first.header.flags & netui::DrawFrame::kCompressed, 0u);
  EXPECT_EQ(first.frame_index, 1u);
  EXPECT_EQ(first.display_size[0], 800.0f);
  EXPECT_EQ(first.display_size[1], 600.0f);
}

TEST_F(HeadlessUiSessionTest, DropsAFrameLaidOutForThePreviousBrowser) {
  Connect();  // A 640x480 screen.
  RunFrame("Hello");
  const size_t texture_count = session_.TakeMessages().size() - 1;

  // Another browser takes control while a frame is laid out for this one.
  BeginFrame();
  StartSession();
  EndFrame("Hello");
  EXPECT_TRUE(session_.TakeMessages().empty());

  // Its first frame waits for its input, so it is laid out for its screen.
  netui::Input input;
  input.screen_size[0] = 800;
  input.screen_size[1] = 600;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Hello");
  std::vector<netui::Bytes> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), texture_count + 1);  // The textures again.
  ASSERT_EQ(KindOf(messages.back()), netui::MessageKind::kDrawFrame);
  const netui::DrawFrame first = HeaderOf<netui::DrawFrame>(messages.back());
  EXPECT_EQ(first.header.flags & netui::DrawFrame::kCompressed, 0u);
  EXPECT_EQ(first.display_size[0], 800.0f);
  EXPECT_EQ(first.display_size[1], 600.0f);
}

TEST_F(HeadlessUiSessionTest, CreatesUpdatesAndDestroysUserTextures) {
  Connect();
  RunFrame("Hello");
  session_.TakeMessages();

  auto image = std::make_unique<ImTextureData>();
  image->Create(ImTextureFormat_RGBA32, 2, 2);
  std::memset(image->GetPixels(), 0x40, image->GetSizeInBytes());
  ImGui::RegisterUserTexture(image.get());
  RunFrame("Hello", image.get());

  std::vector<netui::Bytes> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  const netui::Texture created = HeaderOf<netui::Texture>(messages[0]);
  EXPECT_EQ(created.action, netui::Texture::Action::kCreate);
  EXPECT_EQ(created.width, 2u);
  EXPECT_EQ(created.height, 2u);
  ASSERT_TRUE(HoldsItsPixels(messages[0]));
  EXPECT_EQ(TexturePixels(messages[0])[0], 0x40);
  const uint64_t image_id = created.tex_id;

  // Change one pixel: only that rectangle is sent.
  static_cast<uint8_t*>(image->GetPixelsAt(1, 1))[0] = 0x80;
  image->UpdateRect = {1, 1, 1, 1};
  image->SetStatus(ImTextureStatus_WantUpdates);
  RunFrame("Hello", image.get());
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  const netui::Texture updated = HeaderOf<netui::Texture>(messages[0]);
  EXPECT_EQ(updated.action, netui::Texture::Action::kUpdate);
  EXPECT_EQ(updated.tex_id, image_id);
  EXPECT_EQ(updated.x, 1u);
  EXPECT_EQ(updated.y, 1u);
  EXPECT_EQ(updated.width, 1u);
  ASSERT_TRUE(HoldsItsPixels(messages[0]));
  EXPECT_EQ(TexturePixels(messages[0])[0], 0x80);

  // A texture that disappears is destroyed in the browser too.
  ImGui::UnregisterUserTexture(image.get());
  RunFrame("Hello");
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  const netui::Texture destroyed = HeaderOf<netui::Texture>(messages[0]);
  EXPECT_EQ(destroyed.action, netui::Texture::Action::kDestroy);
  EXPECT_EQ(destroyed.tex_id, image_id);
  EXPECT_TRUE(HoldsItsPixels(messages[0]));
}

TEST_F(HeadlessUiSessionTest, SendsOnlyTheNewestCopyOfATexture) {
  Connect();
  auto image = std::make_unique<ImTextureData>();
  image->Create(ImTextureFormat_RGBA32, 2, 2);
  std::memset(image->GetPixels(), 0x40, image->GetSizeInBytes());
  ImGui::RegisterUserTexture(image.get());
  RunFrame("Hello", image.get());
  session_.TakeMessages();

  // Uploaded whole every frame (like an image from Python) while the server
  // waits for the browser: only the newest copy is sent, as a Create that
  // replaces the browser's.
  for (int value : {0x50, 0x60, 0x70}) {
    std::memset(image->GetPixels(), value, image->GetSizeInBytes());
    image->UpdateRect = {0, 0, 2, 2};
    image->SetStatus(ImTextureStatus_WantUpdates);
    RunFrame("Hello", image.get());
  }
  std::vector<netui::Bytes> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  const netui::Texture uploaded = HeaderOf<netui::Texture>(messages[0]);
  EXPECT_EQ(uploaded.action, netui::Texture::Action::kCreate);
  EXPECT_EQ(uploaded.width, 2u);
  EXPECT_EQ(uploaded.height, 2u);
  ASSERT_TRUE(HoldsItsPixels(messages[0]));
  EXPECT_EQ(TexturePixels(messages[0])[0], 0x70);
  const uint64_t image_id = uploaded.tex_id;
  EXPECT_EQ(KindOf(messages[1]), netui::MessageKind::kDrawFrame);

  // Destroyed before its last upload is sent: only the Destroy is.
  std::memset(image->GetPixels(), 0x80, image->GetSizeInBytes());
  image->UpdateRect = {0, 0, 2, 2};
  image->SetStatus(ImTextureStatus_WantUpdates);
  RunFrame("Hello", image.get());
  ImGui::UnregisterUserTexture(image.get());
  RunFrame("Hello");
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  const netui::Texture destroyed = HeaderOf<netui::Texture>(messages[0]);
  EXPECT_EQ(destroyed.action, netui::Texture::Action::kDestroy);
  EXPECT_EQ(destroyed.tex_id, image_id);
}

TEST_F(HeadlessUiSessionTest, RejectsMalformedInput) {
  StartSession();
  const netui::Bytes input = WireBytes(netui::Input());
  EXPECT_FALSE(
      session_.PushInput(netui::BytesView(input).first(input.size() - 1)));
  EXPECT_FALSE(session_.PushInput(netui::BytesView()));

  netui::Input not_input;
  not_input.header.kind = netui::MessageKind::kDrawFrame;
  const netui::Bytes wrong_type = WireBytes(not_input);
  EXPECT_FALSE(session_.PushInput(wrong_type));

  netui::Input too_many_chars;
  too_many_chars.char_count = netui::Input::kMaxChars + 1;
  const netui::Bytes overflow = WireBytes(too_many_chars);
  EXPECT_FALSE(session_.PushInput(overflow));

  EXPECT_TRUE(session_.PushInput(input));
}

TEST_F(HeadlessUiSessionTest, ForwardsClipboardTextWithNextFrame) {
  Connect();
  RunFrame("Hello");
  session_.TakeMessages();

  // SetClipboardText during a frame queues a Clipboard message sent before the
  // frame, and survives if another frame is captured before TakeMessages().
  BeginFrame();
  ImGui::SetClipboardText("copied camera");
  EndFrame("Hello");
  RunFrame("Hello");

  std::vector<netui::Bytes> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  ASSERT_EQ(KindOf(messages[0]), netui::MessageKind::kClipboard);
  std::string copied;
  bool open_url = true;
  ASSERT_TRUE(netui::DecodeClipboard(messages[0], &copied, &open_url));
  EXPECT_EQ(copied, "copied camera");
  EXPECT_FALSE(open_url);
  EXPECT_EQ(KindOf(messages[1]), netui::MessageKind::kDrawFrame);

  // Text with the open-URL prefix goes out as a URL to open, without the
  // prefix.
  BeginFrame();
  ImGui::SetClipboardText("\x01OPEN_URL:https://mujoco.org");
  EndFrame("Hello");
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  ASSERT_TRUE(netui::DecodeClipboard(messages[0], &copied, &open_url));
  EXPECT_EQ(copied, "https://mujoco.org");
  EXPECT_TRUE(open_url);

  // If control changes hands mid-frame, clipboard text from the old session's
  // frame is dropped instead of leaking to the new browser.
  BeginFrame();
  Connect();
  ImGui::SetClipboardText("stale secret");
  EndFrame("Hello");
  RunFrame("Hello");
  messages = session_.TakeMessages();
  for (const netui::Bytes& msg : messages) {
    EXPECT_NE(KindOf(msg), netui::MessageKind::kClipboard);
  }
}

TEST_F(HeadlessUiSessionTest,
       KeyframeRequestAfterCaptureFrameTakesEffectImmediately) {
  Connect();
  RunFrame("First");
  session_.TakeMessages();

  // CaptureFrame() runs on the UI thread before the browser's keyframe request
  // arrives on the server thread; TakeMessages() encodes lazily so the request
  // still produces an uncompressed frame immediately.
  RunFrame("Frist");
  netui::Input input;
  input.header.flags = netui::Input::kRequestKeyframe;
  ASSERT_TRUE(PushInput(input));

  std::vector<netui::Bytes> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(HeaderOf<netui::DrawFrame>(messages.back()).header.flags &
                netui::DrawFrame::kCompressed,
            0u);

  // Subsequent frames resume delta compression once the keyframe was sent.
  RunFrame("Fisrt");
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_NE(HeaderOf<netui::DrawFrame>(messages.back()).header.flags &
                netui::DrawFrame::kCompressed,
            0u);
}

TEST_F(HeadlessUiSessionTest, AppliesFramebufferScaleAndMacOsModifierSwap) {
  StartSession();

  netui::Input input;
  input.screen_size[0] = 800;
  input.screen_size[1] = 600;
  input.framebuffer_scale = 2.0f;
  ASSERT_TRUE(PushInput(input));
  BeginFrame();
  ImGuiIO& io = ImGui::GetIO();
  EXPECT_FLOAT_EQ(io.DisplayFramebufferScale.x, 2.0f);
  EXPECT_FLOAT_EQ(io.DisplayFramebufferScale.y, 2.0f);
  EndFrame("HiDPI");

  // Out-of-range scales clamp to [0.5, 4.0].
  input.framebuffer_scale = 10.0f;
  ASSERT_TRUE(PushInput(input));
  BeginFrame();
  EXPECT_FLOAT_EQ(io.DisplayFramebufferScale.x, 4.0f);
  EndFrame("HiDPI");

  input.framebuffer_scale = 0.1f;
  ASSERT_TRUE(PushInput(input));
  BeginFrame();
  EXPECT_FLOAT_EQ(io.DisplayFramebufferScale.x, 0.5f);
  EndFrame("HiDPI");

  // Physical Super from a macOS browser maps to Dear ImGui Ctrl once.
  netui::Input mac_input;
  mac_input.screen_size[0] = 800;
  mac_input.screen_size[1] = 600;
  mac_input.header.flags = netui::Input::kMacOsBehaviors;
  mac_input.SetImGuiKeyDown(ImGuiKey_LeftSuper);
  mac_input.SetImGuiKeyDown(ImGuiKey_ReservedForModSuper);
  ASSERT_TRUE(PushInput(mac_input));
  BeginFrame();
  EXPECT_TRUE(io.ConfigMacOSXBehaviors);
  EXPECT_TRUE(io.KeyCtrl);
  EXPECT_TRUE(ImGui::IsKeyDown(ImGuiKey_LeftCtrl));
  EndFrame("Mac");

  // Switching macOS behavior off releases held modifiers before flipping mode.
  netui::Input linux_input;
  linux_input.screen_size[0] = 800;
  linux_input.screen_size[1] = 600;
  ASSERT_TRUE(PushInput(linux_input));
  BeginFrame();
  EXPECT_FALSE(io.ConfigMacOSXBehaviors);
  EXPECT_FALSE(io.KeyCtrl);
  EXPECT_FALSE(io.KeySuper);
  EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_LeftCtrl));
  EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_LeftSuper));
  EndFrame("Linux");
}
}  // namespace
}  // namespace mujoco::studio
