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
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <imgui.h>
#include <NetImgui_Api.h>
#include <NetImgui_CmdPackets.h>

namespace mujoco::studio {
namespace {

using ::NetImgui::Internal::CmdClipboard;
using ::NetImgui::Internal::CmdDrawFrame;
using ::NetImgui::Internal::CmdHeader;
using ::NetImgui::Internal::CmdInput;
using ::NetImgui::Internal::CmdTexture;
using ::NetImgui::Internal::CmdVersion;
using ::NetImgui::Internal::ConvertToCmdDrawFrame;
using ::NetImgui::Internal::DecompressCmdDrawFrame;
using ::NetImgui::Internal::DivUp;
using ::NetImgui::Internal::ImguiDraw;
using ::NetImgui::Internal::ImguiDrawGroup;
using ::NetImgui::Internal::ImguiVert;
using ::NetImgui::Internal::netImguiDelete;

struct FrameDeleter {
  void operator()(CmdDrawFrame* frame) const { netImguiDelete(frame); }
};
using FramePtr = std::unique_ptr<CmdDrawFrame, FrameDeleter>;

template <typename T>
std::string Bytes(const T& command) {
  return std::string(reinterpret_cast<const char*>(&command), sizeof(command));
}

std::span<const std::byte> AsBytes(std::string_view message) {
  return std::as_bytes(std::span(message));
}

CmdHeader::eCommands TypeOf(const std::string& message) {
  CmdHeader header(CmdHeader::eCommands::Count, 0);
  std::memcpy(static_cast<void*>(&header), message.data(), sizeof(header));
  return header.mType;
}

// Whether a received texture command holds the pixels it describes, as the
// browser checks (CmdTexture::ToPointers() needs NetImgui's client runtime).
// Turns its pixel offset into a pointer.
bool HoldsItsPixels(CmdTexture* texture) {
  const uint64_t bytes_per_pixel =
      texture->mFormat == NetImgui::eTexFormat::kTexFmtA8 ? 1 : 4;
  const uint64_t pixels_size =
      texture->mStatus == CmdTexture::eType::Destroy
          ? 0
          : bytes_per_pixel * texture->mWidth * texture->mHeight;
  return texture->mpTextureData.ToPointer(
      texture, reinterpret_cast<const uint8_t*>(texture) + texture->mSize,
      pixels_size);
}

// A received message, copied into 8-byte aligned memory like the browser does.
class Received {
 public:
  explicit Received(const std::string& message)
      : storage_(DivUp<size_t>(message.size(), sizeof(uint64_t))) {
    std::memcpy(storage_.data(), message.data(), message.size());
  }
  template <typename T>
  T* As() {
    return reinterpret_cast<T*>(storage_.data());
  }

 private:
  std::vector<uint64_t> storage_;
};

// Expects two (uncompressed, pointer form) frames to hold the same draw data.
void ExpectSameDrawData(const CmdDrawFrame& actual,
                        const CmdDrawFrame& expected) {
  ASSERT_EQ(actual.mDrawGroupCount, expected.mDrawGroupCount);
  for (uint32_t g = 0; g < expected.mDrawGroupCount; ++g) {
    const ImguiDrawGroup& a = actual.mpDrawGroups[g];
    const ImguiDrawGroup& e = expected.mpDrawGroups[g];
    ASSERT_EQ(a.mVerticeCount, e.mVerticeCount);
    ASSERT_EQ(a.mIndiceCount, e.mIndiceCount);
    ASSERT_EQ(a.mDrawCount, e.mDrawCount);
    ASSERT_EQ(a.mBytePerIndex, e.mBytePerIndex);
    EXPECT_EQ(std::memcmp(a.mpVertices.Get(), e.mpVertices.Get(),
                          e.mVerticeCount * sizeof(ImguiVert)),
              0);
    EXPECT_EQ(std::memcmp(a.mpIndices.Get(), e.mpIndices.Get(),
                          e.mIndiceCount * e.mBytePerIndex),
              0);
    EXPECT_EQ(std::memcmp(a.mpDraws.Get(), e.mpDraws.Get(),
                          e.mDrawCount * sizeof(ImguiDraw)),
              0);
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
    const std::string handshake = Bytes(CmdVersion());
    std::string reply;
    std::string error;
    ASSERT_TRUE(session_.StartSession(AsBytes(handshake), &reply, &error))
        << error;
  }

  // Starts a session and sends the browser's first input, as the browser does
  // right after its handshake: frames are captured for its screen from then on.
  void Connect() {
    StartSession();
    CmdInput input;
    input.mScreenSize[0] = 640;
    input.mScreenSize[1] = 480;
    input.mCompressionUse = true;
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

  // The draw data of the frame just run, as an uncompressed frame.
  static FramePtr CurrentDrawData() {
    return FramePtr(
        ConvertToCmdDrawFrame(ImGui::GetDrawData(), ImGuiMouseCursor_Arrow));
  }

  bool PushInput(const CmdInput& input) {
    const std::string message = Bytes(input);
    return session_.PushInput(AsBytes(message));
  }

  ImGuiContext* context_ = nullptr;
  HeadlessUiSession session_{"Test UI"};
};

TEST_F(HeadlessUiSessionTest, StartSessionChecksTheHandshake) {
  std::string reply;
  std::string error;

  const std::string short_handshake = Bytes(CmdVersion()).substr(0, 8);
  EXPECT_FALSE(
      session_.StartSession(AsBytes(short_handshake), &reply, &error));

  CmdVersion other_version;
  other_version.mVersion = static_cast<CmdVersion::eVersion>(
      static_cast<int>(other_version.mVersion) + 1);
  const std::string other = Bytes(other_version);
  EXPECT_FALSE(session_.StartSession(AsBytes(other), &reply, &error));
  EXPECT_NE(error.find("protocol"), std::string::npos);
  EXPECT_FALSE(session_.IsActive());

  const std::string handshake = Bytes(CmdVersion());
  ASSERT_TRUE(session_.StartSession(AsBytes(handshake), &reply, &error));
  EXPECT_TRUE(session_.IsActive());
  ASSERT_EQ(reply.size(), sizeof(CmdVersion));
  Received received(reply);
  EXPECT_EQ(received.As<CmdVersion>()->mType, CmdHeader::eCommands::Version);
  EXPECT_STREQ(received.As<CmdVersion>()->mClientName, "Test UI");
}

TEST_F(HeadlessUiSessionTest, TruncatesLongClientNameOnUtf8Boundary) {
  auto reply_name = [](const std::string& name) {
    HeadlessUiSession session(name);
    const std::string handshake = Bytes(CmdVersion());
    std::string reply;
    std::string error;
    if (!session.StartSession(AsBytes(handshake), &reply, &error)) {
      return "StartSession failed: " + error;
    }
    Received received(reply);
    return std::string(received.As<CmdVersion>()->mClientName);
  };
  // The name field holds 63 bytes; "\xC3\x9C" is one 2-byte character.
  const std::string fits = std::string(61, 'a') + "\xC3\x9C";
  EXPECT_EQ(reply_name(fits), fits);
  const std::string too_long = std::string(62, 'a') + "\xC3\x9C";
  EXPECT_EQ(reply_name(too_long), std::string(62, 'a'));
}

TEST_F(HeadlessUiSessionTest, SendsNothingWithoutASession) {
  RunFrame("Hello");
  EXPECT_TRUE(session_.TakeMessages().empty());
  EXPECT_FALSE(PushInput(CmdInput()));
}

TEST_F(HeadlessUiSessionTest, HasInputOnceTheSessionsBrowserSendsInput) {
  StartSession();
  EXPECT_FALSE(session_.HasInput());
  ASSERT_TRUE(PushInput(CmdInput()));
  EXPECT_TRUE(session_.HasInput());

  session_.EndSession();
  EXPECT_FALSE(session_.HasInput());
  StartSession();
  EXPECT_FALSE(session_.HasInput());  // The new browser has not sent any yet.
}

TEST_F(HeadlessUiSessionTest, FirstFrameSendsTexturesThenAFullFrame) {
  Connect();
  RunFrame("Hello");

  std::vector<std::string> messages = session_.TakeMessages();
  ASSERT_GE(messages.size(), 2u);
  std::vector<uint64_t> texture_ids;
  for (size_t i = 0; i + 1 < messages.size(); ++i) {
    ASSERT_EQ(TypeOf(messages[i]), CmdHeader::eCommands::Texture);
    Received received(messages[i]);
    CmdTexture* texture = received.As<CmdTexture>();
    EXPECT_EQ(texture->mSize, messages[i].size());
    EXPECT_EQ(texture->mStatus, CmdTexture::eType::Create);
    EXPECT_TRUE(HoldsItsPixels(texture));
    texture_ids.push_back(texture->mTextureClientID);
  }

  ASSERT_EQ(TypeOf(messages.back()), CmdHeader::eCommands::DrawFrame);
  Received received(messages.back());
  CmdDrawFrame* frame = received.As<CmdDrawFrame>();
  EXPECT_EQ(frame->mSize, messages.back().size());
  EXPECT_FALSE(frame->mCompressed);
  EXPECT_EQ(frame->mFrameIndex, 1u);
  ASSERT_TRUE(frame->ToPointers());
  ExpectSameDrawData(*frame, *CurrentDrawData());
  // The draw calls use the textures sent before the frame.
  for (uint32_t g = 0; g < frame->mDrawGroupCount; ++g) {
    const ImguiDrawGroup& group = frame->mpDrawGroups[g];
    for (uint32_t d = 0; d < group.mDrawCount; ++d) {
      EXPECT_NE(std::find(texture_ids.begin(), texture_ids.end(),
                          group.mpDraws[d].mClientTexId),
                texture_ids.end());
    }
  }

  // Nothing new to send until the next frame.
  EXPECT_TRUE(session_.TakeMessages().empty());
}

TEST_F(HeadlessUiSessionTest, CompressesAgainstTheLastFrameTaken) {
  Connect();
  RunFrame("First");
  std::vector<std::string> first = session_.TakeMessages();
  Received first_frame(first.back());
  ASSERT_TRUE(first_frame.As<CmdDrawFrame>()->ToPointers());

  // A frame captured but replaced before being taken is never sent.
  RunFrame("Frist");
  RunFrame("Fisrt");
  FramePtr expected = CurrentDrawData();
  std::vector<std::string> second = session_.TakeMessages();
  ASSERT_EQ(second.size(), 1u);
  Received second_frame(second.back());
  CmdDrawFrame* packed = second_frame.As<CmdDrawFrame>();
  EXPECT_TRUE(packed->mCompressed);
  EXPECT_EQ(packed->mFrameIndex, 2u);
  ASSERT_TRUE(packed->ToPointers());

  FramePtr unpacked(
      DecompressCmdDrawFrame(first_frame.As<CmdDrawFrame>(), packed));
  ASSERT_NE(unpacked, nullptr);
  ExpectSameDrawData(*unpacked, *expected);
}

// The browser acks a frame by echoing its index.
TEST_F(HeadlessUiSessionTest, LastFrameIndexIsThatOfTheLastFrameTaken) {
  EXPECT_EQ(session_.LastFrameIndex(), 0u);
  Connect();
  RunFrame("First");
  EXPECT_EQ(session_.LastFrameIndex(), 0u);  // Not taken yet.
  std::vector<std::string> messages = session_.TakeMessages();
  Received first(messages.back());
  EXPECT_EQ(first.As<CmdDrawFrame>()->mFrameIndex, 1u);
  EXPECT_EQ(session_.LastFrameIndex(), 1u);

  RunFrame("Frist");
  messages = session_.TakeMessages();
  Received second(messages.back());
  EXPECT_EQ(second.As<CmdDrawFrame>()->mFrameIndex, 2u);
  EXPECT_EQ(session_.LastFrameIndex(), 2u);
  EXPECT_TRUE(session_.TakeMessages().empty());  // Nothing new, no change.
  EXPECT_EQ(session_.LastFrameIndex(), 2u);
}

TEST_F(HeadlessUiSessionTest, SendsAFullFrameWhenTheBrowserAsks) {
  Connect();
  RunFrame("First");
  session_.TakeMessages();

  CmdInput input;
  input.mCompressionUse = true;
  input.mCompressionSkip = true;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Frist");
  std::vector<std::string> messages = session_.TakeMessages();
  Received frame(messages.back());
  EXPECT_FALSE(frame.As<CmdDrawFrame>()->mCompressed);

  input.mCompressionSkip = false;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Fisrt");
  messages = session_.TakeMessages();
  Received next(messages.back());
  EXPECT_TRUE(next.As<CmdDrawFrame>()->mCompressed);
}

TEST_F(HeadlessUiSessionTest, AppliesBrowserInput) {
  StartSession();
  RunFrame("Hello");

  CmdInput input;
  input.mScreenSize[0] = 800;
  input.mScreenSize[1] = 600;
  input.mMousePos[0] = 50;
  input.mMousePos[1] = 60;
  input.mMouseDownMask = 1 << CmdInput::ImGuiMouseButton_Left;
  input.mInputDownMask[CmdInput::ImGuiKey_A / 64] |=
      uint64_t{1} << (CmdInput::ImGuiKey_A % 64);
  input.mInputDownMask[CmdInput::ImGuiKey_ReservedForModCtrl / 64] |=
      uint64_t{1} << (CmdInput::ImGuiKey_ReservedForModCtrl % 64);
  input.mKeyChars[0] = 'h';
  input.mKeyChars[1] = 'i';
  input.mKeyCharCount = 2;
  input.mMouseWheelVert = 5.0f;  // A running total: no scroll yet.
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
  input.mMouseDownMask = 0;
  input.mInputDownMask[CmdInput::ImGuiKey_A / 64] = 0;
  input.mKeyCharCount = 0;
  input.mMouseWheelVert = 7.0f;
  ASSERT_TRUE(PushInput(input));
  BeginFrame();
  EXPECT_EQ(io.MouseWheel, 2.0f);
  EXPECT_FALSE(ImGui::IsMouseDown(ImGuiMouseButton_Left));
  EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
  EndFrame("Hello");
}

TEST_F(HeadlessUiSessionTest, ANewSessionReleasesInputAndResendsTextures) {
  StartSession();
  CmdInput input;
  input.mMouseDownMask = 1 << CmdInput::ImGuiMouseButton_Left;
  input.mInputDownMask[CmdInput::ImGuiKey_A / 64] |=
      uint64_t{1} << (CmdInput::ImGuiKey_A % 64);
  ASSERT_TRUE(PushInput(input));
  RunFrame("Hello");
  EXPECT_TRUE(ImGui::IsKeyDown(ImGuiKey_A));
  std::vector<std::string> first = session_.TakeMessages();
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

  std::vector<std::string> again = session_.TakeMessages();
  ASSERT_EQ(again.size(), first_texture_count + 1);
  for (size_t i = 0; i < first_texture_count; ++i) {
    EXPECT_EQ(again[i], first[i]);  // The same textures, created again.
  }
  Received frame(again.back());
  EXPECT_FALSE(frame.As<CmdDrawFrame>()->mCompressed);
}

TEST_F(HeadlessUiSessionTest, SendsNoFrameBeforeTheBrowsersFirstInput) {
  StartSession();
  RunFrame("Hello");  // Not laid out for the browser's screen.
  EXPECT_TRUE(session_.TakeMessages().empty());

  CmdInput input;
  input.mScreenSize[0] = 800;
  input.mScreenSize[1] = 600;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Hello");
  std::vector<std::string> messages = session_.TakeMessages();
  ASSERT_GE(messages.size(), 2u);
  EXPECT_EQ(TypeOf(messages.front()), CmdHeader::eCommands::Texture);
  ASSERT_EQ(TypeOf(messages.back()), CmdHeader::eCommands::DrawFrame);
  Received frame(messages.back());
  const CmdDrawFrame* first = frame.As<CmdDrawFrame>();
  EXPECT_FALSE(first->mCompressed);
  EXPECT_EQ(first->mFrameIndex, 1u);
  EXPECT_EQ(first->mDisplayArea[2] - first->mDisplayArea[0], 800.0f);
  EXPECT_EQ(first->mDisplayArea[3] - first->mDisplayArea[1], 600.0f);
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
  CmdInput input;
  input.mScreenSize[0] = 800;
  input.mScreenSize[1] = 600;
  ASSERT_TRUE(PushInput(input));
  RunFrame("Hello");
  std::vector<std::string> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), texture_count + 1);  // The textures again.
  ASSERT_EQ(TypeOf(messages.back()), CmdHeader::eCommands::DrawFrame);
  Received frame(messages.back());
  const CmdDrawFrame* first = frame.As<CmdDrawFrame>();
  EXPECT_FALSE(first->mCompressed);
  EXPECT_EQ(first->mDisplayArea[2] - first->mDisplayArea[0], 800.0f);
  EXPECT_EQ(first->mDisplayArea[3] - first->mDisplayArea[1], 600.0f);
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

  std::vector<std::string> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  Received create(messages[0]);
  CmdTexture* created = create.As<CmdTexture>();
  EXPECT_EQ(created->mStatus, CmdTexture::eType::Create);
  EXPECT_EQ(created->mWidth, 2u);
  EXPECT_EQ(created->mHeight, 2u);
  ASSERT_TRUE(HoldsItsPixels(created));
  EXPECT_EQ(created->mpTextureData.Get()[0], 0x40);
  const uint64_t image_id = created->mTextureClientID;

  // Change one pixel: only that rectangle is sent.
  static_cast<uint8_t*>(image->GetPixelsAt(1, 1))[0] = 0x80;
  image->UpdateRect = {1, 1, 1, 1};
  image->SetStatus(ImTextureStatus_WantUpdates);
  RunFrame("Hello", image.get());
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  Received update(messages[0]);
  CmdTexture* updated = update.As<CmdTexture>();
  EXPECT_EQ(updated->mStatus, CmdTexture::eType::Update);
  EXPECT_EQ(updated->mTextureClientID, image_id);
  EXPECT_EQ(updated->mOffsetX, 1u);
  EXPECT_EQ(updated->mOffsetY, 1u);
  EXPECT_EQ(updated->mWidth, 1u);
  ASSERT_TRUE(HoldsItsPixels(updated));
  EXPECT_EQ(updated->mpTextureData.Get()[0], 0x80);

  // A texture that disappears is destroyed in the browser too.
  ImGui::UnregisterUserTexture(image.get());
  RunFrame("Hello");
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  Received destroy(messages[0]);
  EXPECT_EQ(destroy.As<CmdTexture>()->mStatus, CmdTexture::eType::Destroy);
  EXPECT_EQ(destroy.As<CmdTexture>()->mTextureClientID, image_id);
  EXPECT_TRUE(HoldsItsPixels(destroy.As<CmdTexture>()));
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
  std::vector<std::string> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  Received upload(messages[0]);
  CmdTexture* uploaded = upload.As<CmdTexture>();
  EXPECT_EQ(uploaded->mStatus, CmdTexture::eType::Create);
  EXPECT_EQ(uploaded->mWidth, 2u);
  EXPECT_EQ(uploaded->mHeight, 2u);
  ASSERT_TRUE(HoldsItsPixels(uploaded));
  EXPECT_EQ(uploaded->mpTextureData.Get()[0], 0x70);
  const uint64_t image_id = uploaded->mTextureClientID;
  EXPECT_EQ(TypeOf(messages[1]), CmdHeader::eCommands::DrawFrame);

  // Destroyed before its last upload is sent: only the Destroy is.
  std::memset(image->GetPixels(), 0x80, image->GetSizeInBytes());
  image->UpdateRect = {0, 0, 2, 2};
  image->SetStatus(ImTextureStatus_WantUpdates);
  RunFrame("Hello", image.get());
  ImGui::UnregisterUserTexture(image.get());
  RunFrame("Hello");
  messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  Received destroy(messages[0]);
  EXPECT_EQ(destroy.As<CmdTexture>()->mStatus, CmdTexture::eType::Destroy);
  EXPECT_EQ(destroy.As<CmdTexture>()->mTextureClientID, image_id);
}

TEST_F(HeadlessUiSessionTest, RejectsMalformedInput) {
  StartSession();
  const std::string input = Bytes(CmdInput());
  EXPECT_FALSE(
      session_.PushInput(AsBytes(input).first(input.size() - 1)));
  EXPECT_FALSE(session_.PushInput(std::span<const std::byte>()));

  CmdInput not_input;
  not_input.mType = CmdHeader::eCommands::DrawFrame;
  const std::string wrong_type = Bytes(not_input);
  EXPECT_FALSE(session_.PushInput(AsBytes(wrong_type)));

  CmdInput too_many_chars;
  too_many_chars.mKeyCharCount = 257;
  const std::string overflow = Bytes(too_many_chars);
  EXPECT_FALSE(session_.PushInput(AsBytes(overflow)));

  EXPECT_TRUE(session_.PushInput(AsBytes(input)));
}

TEST_F(HeadlessUiSessionTest, ForwardsClipboardTextWithNextFrame) {
  Connect();
  RunFrame("Hello");
  session_.TakeMessages();

  // SetClipboardText during a frame queues a CmdClipboard sent before the
  // frame, and survives if another frame is captured before TakeMessages().
  BeginFrame();
  ImGui::SetClipboardText("copied camera");
  EndFrame("Hello");
  RunFrame("Hello");

  std::vector<std::string> messages = session_.TakeMessages();
  ASSERT_EQ(messages.size(), 2u);
  ASSERT_EQ(TypeOf(messages[0]), CmdHeader::eCommands::Clipboard);
  Received clip(messages[0]);
  CmdClipboard* cmd = clip.As<CmdClipboard>();
  ASSERT_TRUE(cmd->ToPointers());
  EXPECT_STREQ(cmd->mContentUTF8.Get(), "copied camera");
  EXPECT_EQ(TypeOf(messages[1]), CmdHeader::eCommands::DrawFrame);

  // If control changes hands mid-frame, clipboard text from the old session's
  // frame is dropped instead of leaking to the new browser.
  BeginFrame();
  Connect();
  ImGui::SetClipboardText("stale secret");
  EndFrame("Hello");
  RunFrame("Hello");
  messages = session_.TakeMessages();
  for (const std::string& msg : messages) {
    EXPECT_NE(TypeOf(msg), CmdHeader::eCommands::Clipboard);
  }
}

}  // namespace
}  // namespace mujoco::studio
