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

#include <dlfcn.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include <imgui.h>
#include <mujoco/mujoco.h>
#include <NetImgui_Api.h>
#include <NetImgui_CmdPackets.h>
#include <NetImgui_Network.h>
#include "google/network_status.h"
#include "state_payload.h"

// Host shims for browser-specific WebSocket symbols used by RemoteUi.
namespace mujoco::studio {
void PrimeWebSocketUrl(const char* url) {}
}  // namespace mujoco::studio

namespace NetImgui::Internal::Network {
namespace {
ReadyState g_test_ready_state = ReadyState::kOpen;
int g_test_close_code = 0;
}  // namespace

ReadyState GetReadyState(SocketInfo* socket) {
  return socket ? g_test_ready_state : ReadyState::kDisconnected;
}

int GetCloseCode(SocketInfo* socket) { return socket ? g_test_close_code : 0; }
}  // namespace NetImgui::Internal::Network

// RemoteUi::Connect(url) calls Network::Connect(url.c_str(), 0) because on WASM
// the port is embedded in the WebSocket URL. On POSIX, interposing getaddrinfo
// lets "127.0.0.1:<port>" resolve to the loopback port so RemoteUi can be
// tested end-to-end over a real POSIX loopback socket.
extern "C" int getaddrinfo(const char* node, const char* service,
                           const addrinfo* hints, addrinfo** res) {
  using GetAddrInfoFn =
      int (*)(const char*, const char*, const addrinfo*, addrinfo**);
  static const auto real_getaddrinfo =
      reinterpret_cast<GetAddrInfoFn>(dlsym(RTLD_NEXT, "getaddrinfo"));
  if (node != nullptr && service != nullptr && std::strcmp(service, "0") == 0) {
    const char* colon = std::strchr(node, ':');
    if (colon != nullptr) {
      std::string host(node, colon - node);
      return real_getaddrinfo(host.c_str(), colon + 1, hints, res);
    }
  }
  return real_getaddrinfo(node, service, hints, res);
}

namespace mujoco::studio {
namespace {

using ::NetImgui::eTexFormat;
using ::NetImgui::Internal::ClientTextureID;
using ::NetImgui::Internal::CmdDrawFrame;
using ::NetImgui::Internal::CmdHeader;
using ::NetImgui::Internal::CmdInput;
using ::NetImgui::Internal::CmdTexture;
using ::NetImgui::Internal::CmdVersion;
using ::NetImgui::Internal::CompressCmdDrawFrame;
using ::NetImgui::Internal::ConvertToCmdDrawFrame;
using ::NetImgui::Internal::ImguiDraw;
using ::NetImgui::Internal::netImguiDelete;
using ::NetImgui::Internal::netImguiSizedNew;
using ::NetImgui::Internal::Network::ReadyState;

class FakeCallbacks : public RemoteUi::Callbacks {
 public:
  struct UploadedTexture {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;
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

  bool gpu_ready = true;
  int upload_calls = 0;
  uintptr_t next_handle = 100;
  std::unordered_map<uintptr_t, UploadedTexture> textures;
};

class RemoteUiTest : public ::testing::Test {
 protected:
  RemoteUiTest() : context_(ImGui::CreateContext()), remote_ui_(callbacks_) {
    ImGui::SetCurrentContext(context_);
    NetImgui::Internal::Network::g_test_ready_state = ReadyState::kOpen;
    NetImgui::Internal::Network::g_test_close_code = 0;
    remote_ui_.SetScreenArea(1920.0f, 1080.0f);
  }

  ~RemoteUiTest() override {
    remote_ui_.Shutdown();
    ImGui::DestroyContext(context_);
  }

  static CmdTexture* MakeTextureCmd(ClientTextureID tex_id,
                                    CmdTexture::eType status, eTexFormat format,
                                    uint16_t width, uint16_t height,
                                    const std::vector<uint8_t>& pixels,
                                    uint16_t offset_x = 0,
                                    uint16_t offset_y = 0) {
    const size_t total_size = sizeof(CmdTexture) + pixels.size();
    auto* cmd = netImguiSizedNew<CmdTexture>(total_size);
    cmd->mSize = static_cast<uint32_t>(total_size);
    cmd->mTextureClientID = tex_id;
    cmd->mStatus = status;
    cmd->mFormat = static_cast<uint8_t>(format);
    cmd->mWidth = width;
    cmd->mHeight = height;
    cmd->mOffsetX = offset_x;
    cmd->mOffsetY = offset_y;
    uint8_t* dst = reinterpret_cast<uint8_t*>(&cmd[1]);
    if (!pixels.empty()) {
      std::memcpy(dst, pixels.data(), pixels.size());
    }
    cmd->mpTextureData.SetPtr(dst);
    cmd->mpTextureData.ToOffset();
    return cmd;
  }

  static CmdDrawFrame* MakeDrawFrame(const char* text, uint64_t frame_index,
                                     ClientTextureID tex_id = 1,
                                     bool second_window = false) {
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
    CmdDrawFrame* frame =
        ConvertToCmdDrawFrame(ImGui::GetDrawData(), ImGuiMouseCursor_Arrow);
    CHECK_NE(frame, nullptr);
    CHECK_GT(frame->mDrawGroupCount, 0u);
    CHECK_GT(frame->mTotalVerticeCount, 0u);
    frame->mFrameIndex = frame_index;
    for (uint32_t g = 0; g < frame->mDrawGroupCount; ++g) {
      ImguiDraw* draws = frame->mpDrawGroups[g].mpDraws.Get();
      for (uint32_t d = 0; d < frame->mpDrawGroups[g].mDrawCount; ++d) {
        draws[d].mClientTexId = tex_id;
      }
    }
    return frame;
  }

  void RegisterDummyTexture(ClientTextureID tex_id) {
    std::vector<uint8_t> pixels(4 * 4 * 4, 255);
    CmdTexture* cmd = MakeTextureCmd(tex_id, CmdTexture::eType::Create,
                                     eTexFormat::kTexFmtRGBA8, 4, 4, pixels);
    remote_ui_.ProcessCmdTexture(cmd);
    netImguiDelete(cmd);
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
  CmdTexture* create_cmd =
      MakeTextureCmd(42, CmdTexture::eType::Create, eTexFormat::kTexFmtRGBA8, 4,
                     4, initial_pixels);
  remote_ui_.ProcessCmdTexture(create_cmd);
  netImguiDelete(create_cmd);

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
  CmdTexture* update_cmd =
      MakeTextureCmd(42, CmdTexture::eType::Update, eTexFormat::kTexFmtRGBA8, 2,
                     2, patch_pixels, /*offset_x=*/1, /*offset_y=*/2);
  remote_ui_.ProcessCmdTexture(update_cmd);
  netImguiDelete(update_cmd);

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
  CmdTexture* destroy_cmd =
      MakeTextureCmd(42, CmdTexture::eType::Destroy, eTexFormat::kTexFmtRGBA8,
                     0, 0, /*pixels=*/{});
  remote_ui_.ProcessCmdTexture(destroy_cmd);
  netImguiDelete(destroy_cmd);

  EXPECT_EQ(callbacks_.upload_calls, 3);
  EXPECT_TRUE(callbacks_.textures[100].destroyed);
}

TEST_F(RemoteUiTest, ExpandsA8FontTextureToRgba8) {
  std::vector<uint8_t> a8_pixels = {0, 64, 128, 255};
  CmdTexture* cmd = MakeTextureCmd(7, CmdTexture::eType::Create,
                                   eTexFormat::kTexFmtA8, 2, 2, a8_pixels);
  remote_ui_.ProcessCmdTexture(cmd);
  netImguiDelete(cmd);

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
  CmdTexture* cmd = MakeTextureCmd(11, CmdTexture::eType::Create,
                                   eTexFormat::kTexFmtRGBA8, 2, 2, pixels);
  remote_ui_.ProcessCmdTexture(cmd);
  netImguiDelete(cmd);

  EXPECT_EQ(callbacks_.upload_calls, 0);

  // A draw frame arriving before GpuReady() omits draws whose textures are not
  // uploaded yet (never leaving ElemCount=0 / TexID=0 commands in CmdBuffer).
  CmdDrawFrame* frame =
      MakeDrawFrame("DeferredFrame", /*frame_index=*/1, /*tex_id=*/11);
  frame->ToOffsets();
  remote_ui_.ProcessCmdDrawFrame(frame);
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
  CmdTexture* create_cmd =
      MakeTextureCmd(9, CmdTexture::eType::Create, eTexFormat::kTexFmtRGBA8, 4,
                     4, valid_pixels);
  remote_ui_.ProcessCmdTexture(create_cmd);
  netImguiDelete(create_cmd);
  ASSERT_EQ(callbacks_.upload_calls, 1);

  // Truncated pixel buffer: declares 4x4 RGBA8 (64 bytes) but only carries 8.
  std::vector<uint8_t> short_pixels(8, 1);
  CmdTexture* short_cmd =
      MakeTextureCmd(10, CmdTexture::eType::Create, eTexFormat::kTexFmtRGBA8, 4,
                     4, short_pixels);
  remote_ui_.ProcessCmdTexture(short_cmd);
  netImguiDelete(short_cmd);
  EXPECT_EQ(callbacks_.upload_calls, 1);

  // Out-of-bounds patch rectangle: 2x2 at offset (3, 3) exceeds 4x4 mirror.
  std::vector<uint8_t> patch_pixels(2 * 2 * 4, 99);
  CmdTexture* oob_patch =
      MakeTextureCmd(9, CmdTexture::eType::Update, eTexFormat::kTexFmtRGBA8, 2,
                     2, patch_pixels, /*offset_x=*/3, /*offset_y=*/3);
  remote_ui_.ProcessCmdTexture(oob_patch);
  netImguiDelete(oob_patch);
  EXPECT_EQ(callbacks_.upload_calls, 1);
  EXPECT_EQ(callbacks_.textures[100].rgba, valid_pixels);
}

TEST_F(RemoteUiTest, ProcessesUncompressedAndDeltaCompressedDrawFrames) {
  RegisterDummyTexture(/*tex_id=*/1);

  CmdDrawFrame* frame1 = MakeDrawFrame("Hello Frame 1", /*frame_index=*/1);
  CmdDrawFrame* frame2 = MakeDrawFrame("Hello Frame 2", /*frame_index=*/2);
  CmdDrawFrame* compressed_frame2 = CompressCmdDrawFrame(frame1, frame2);
  compressed_frame2->mFrameIndex = 2;
  compressed_frame2->ToOffsets();
  const uint32_t expected_vtx_count = frame2->mTotalVerticeCount;
  const uint32_t expected_idx_count = frame2->mTotalIndiceCount;
  netImguiDelete(frame2);

  frame1->ToOffsets();
  remote_ui_.ProcessCmdDrawFrame(frame1);

  ImDrawData* draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  EXPECT_TRUE(draw_data->Valid);
  EXPECT_GT(draw_data->TotalVtxCount, 0);
  ASSERT_EQ(draw_data->CmdLists.Size, 1);
  EXPECT_GT(draw_data->CmdLists[0]->CmdBuffer.Size, 0);
  EXPECT_EQ(draw_data->CmdLists[0]->CmdBuffer[0].TexRef._TexID,
            static_cast<ImTextureID>(100));

  // Delta-compressed frame 2 decompresses against frame 1.
  remote_ui_.ProcessCmdDrawFrame(compressed_frame2);
  draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  EXPECT_EQ(static_cast<uint32_t>(draw_data->TotalVtxCount),
            expected_vtx_count);
  EXPECT_EQ(static_cast<uint32_t>(draw_data->TotalIdxCount),
            expected_idx_count);
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
  CmdDrawFrame* frame =
      ConvertToCmdDrawFrame(ImGui::GetDrawData(), ImGuiMouseCursor_Arrow);
  ASSERT_NE(frame, nullptr);
  ASSERT_EQ(frame->mTotalDrawCount, 0u);
  frame->mFrameIndex = 1;
  frame->ToOffsets();

  remote_ui_.ProcessCmdDrawFrame(frame);
  ImDrawData* draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  EXPECT_EQ(draw_data->CmdLists.Size, 0);
}

TEST_F(RemoteUiTest,
       BuildsOneDrawListPerWindowClampsClipRectAndSkipsUnmappedTextures) {
  RegisterDummyTexture(/*tex_id=*/1);
  remote_ui_.SetScreenArea(120.0f, 80.0f);

  // Two windows produce two draw groups; each gets its own ImDrawList with its
  // 0-based indices copied unchanged.
  CmdDrawFrame* frame = MakeDrawFrame("First", /*frame_index=*/1,
                                      /*tex_id=*/1, /*second_window=*/true);
  ASSERT_GE(frame->mDrawGroupCount, 2u);
  const uint32_t group_count = frame->mDrawGroupCount;
  const uint32_t group0_vtx = frame->mpDrawGroups[0].mVerticeCount;
  const uint32_t group1_vtx = frame->mpDrawGroups[1].mVerticeCount;
  const uint32_t group1_idx = frame->mpDrawGroups[1].mIndiceCount;
  ASSERT_GT(group0_vtx, 0u);
  ASSERT_GT(group1_idx, 0u);
  const uint16_t raw_first_idx_group1 = reinterpret_cast<const uint16_t*>(
      frame->mpDrawGroups[1].mpIndices.Get())[0];

  frame->ToOffsets();
  remote_ui_.ProcessCmdDrawFrame(frame);

  ImDrawData* draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  ASSERT_EQ(draw_data->CmdLists.Size, static_cast<int>(group_count));
  ImDrawList* list0 = draw_data->CmdLists[0];
  ImDrawList* list1 = draw_data->CmdLists[1];
  EXPECT_EQ(static_cast<uint32_t>(list0->VtxBuffer.Size), group0_vtx);
  EXPECT_EQ(static_cast<uint32_t>(list1->VtxBuffer.Size), group1_vtx);
  EXPECT_EQ(static_cast<uint32_t>(list1->IdxBuffer.Size), group1_idx);
  EXPECT_EQ(list1->IdxBuffer[0], static_cast<ImDrawIdx>(raw_first_idx_group1));
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
  CmdDrawFrame* unmapped =
      MakeDrawFrame("Unmapped", /*frame_index=*/2, /*tex_id=*/999);
  unmapped->ToOffsets();
  remote_ui_.ProcessCmdDrawFrame(unmapped);
  draw_data = remote_ui_.RemoteDrawData();
  ASSERT_NE(draw_data, nullptr);
  ASSERT_EQ(draw_data->CmdLists.Size, 1);
  EXPECT_EQ(draw_data->CmdLists[0], list0);
  EXPECT_EQ(draw_data->CmdLists[0]->CmdBuffer.Size, 0);
}

TEST_F(RemoteUiTest, SocketHandshakeReceiveAndInputRoundTrip) {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(listen_fd, 0);
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ASSERT_EQ(bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)),
            0);
  ASSERT_EQ(listen(listen_fd, 1), 0);
  socklen_t addr_len = sizeof(addr);
  ASSERT_EQ(
      getsockname(listen_fd, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
  const uint16_t port = ntohs(addr.sin_port);

  remote_ui_.Connect("127.0.0.1:" + std::to_string(port));
  ASSERT_TRUE(remote_ui_.HasSocket());

  int peer_fd = accept(listen_fd, nullptr, nullptr);
  ASSERT_GE(peer_fd, 0);
  close(listen_fd);

  // Frame 1: RemoteUi sends its CmdVersion handshake over the socket.
  remote_ui_.ReceiveAndProcessCommands(/*frame=*/1);
  CmdVersion handshake;
  ASSERT_EQ(recv(peer_fd, &handshake, sizeof(handshake), MSG_WAITALL),
            static_cast<ssize_t>(sizeof(handshake)));
  EXPECT_EQ(handshake.mType, CmdHeader::eCommands::Version);
  EXPECT_STREQ(handshake.mClientName, "MuJoCo Web Viewer");

  // Stream a texture and a draw frame from the peer to RemoteUi.
  std::vector<uint8_t> pixels(4 * 4 * 4, 200);
  CmdTexture* tex_cmd = MakeTextureCmd(1, CmdTexture::eType::Create,
                                       eTexFormat::kTexFmtRGBA8, 4, 4, pixels);
  CmdDrawFrame* draw_cmd = MakeDrawFrame("Streamed over socket", 1, 1);
  draw_cmd->ToOffsets();
  const uint32_t expected_bytes = tex_cmd->mSize + draw_cmd->mSize;

  ASSERT_EQ(send(peer_fd, tex_cmd, tex_cmd->mSize, 0),
            static_cast<ssize_t>(tex_cmd->mSize));
  ASSERT_EQ(send(peer_fd, draw_cmd, draw_cmd->mSize, 0),
            static_cast<ssize_t>(draw_cmd->mSize));
  netImguiDelete(tex_cmd);
  netImguiDelete(draw_cmd);

  for (int i = 0; i < 10 && remote_ui_.RemoteDrawData() == nullptr; ++i) {
    remote_ui_.ReceiveAndProcessCommands(/*frame=*/2 + i);
  }
  ASSERT_NE(remote_ui_.RemoteDrawData(), nullptr);
  EXPECT_EQ(remote_ui_.ConsumeByteCount(), expected_bytes);
  EXPECT_EQ(remote_ui_.ConsumeByteCount(), 0u);

  // CaptureAndSendInput sends a CmdInput packet back to the peer. The screen
  // size it reports is the screen area (the canvas minus the local role
  // bar), not the display size.
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(1280, 720);
  remote_ui_.SetScreenArea(1280.0f, 688.0f);
  io.MousePos = ImVec2(100, 200);
  io.WantCaptureMouse = false;
  io.WantCaptureKeyboard = false;
  io.InputQueueCharacters.push_back('a');

  remote_ui_.CaptureAndSendInput();
  CmdInput input_cmd;
  ASSERT_EQ(recv(peer_fd, &input_cmd, sizeof(input_cmd), MSG_WAITALL),
            static_cast<ssize_t>(sizeof(input_cmd)));
  EXPECT_EQ(input_cmd.mType, CmdHeader::eCommands::Input);
  EXPECT_EQ(input_cmd.mScreenSize[0], 1280u);
  EXPECT_EQ(input_cmd.mScreenSize[1], 688u);
  EXPECT_EQ(input_cmd.mMousePos[0], 100);
  EXPECT_EQ(input_cmd.mMousePos[1], 200);
  EXPECT_TRUE(input_cmd.mCompressionUse);
  EXPECT_TRUE(input_cmd.mCompressionSkip);
  EXPECT_EQ(input_cmd.mKeyCharCount, 1u);
  EXPECT_EQ(input_cmd.mKeyChars[0], static_cast<uint16_t>('a'));

  // Once the keyframe request has been sent, subsequent inputs clear
  // mCompressionSkip unless a delta frame is missing its reference frame.
  io.InputQueueCharacters.resize(0);
  remote_ui_.CaptureAndSendInput();
  ASSERT_EQ(recv(peer_fd, &input_cmd, sizeof(input_cmd), MSG_WAITALL),
            static_cast<ssize_t>(sizeof(input_cmd)));
  EXPECT_FALSE(input_cmd.mCompressionSkip);

  // Receiving a delta frame with a missing reference frame re-arms
  // mCompressionSkip on the next CmdInput.
  CmdDrawFrame* gap_frame = MakeDrawFrame("Gap", /*frame_index=*/99);
  gap_frame->mCompressed = true;
  gap_frame->ToOffsets();
  remote_ui_.ProcessCmdDrawFrame(gap_frame);
  remote_ui_.CaptureAndSendInput();
  ASSERT_EQ(recv(peer_fd, &input_cmd, sizeof(input_cmd), MSG_WAITALL),
            static_cast<ssize_t>(sizeof(input_cmd)));
  EXPECT_TRUE(input_cmd.mCompressionSkip);

  close(peer_fd);
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

}  // namespace
}  // namespace mujoco::studio
