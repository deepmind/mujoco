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

// Unit and adversarial bounds-checking tests for Studio's /ui wire codec.

#include "ui_codec.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <imgui.h>

namespace mujoco::studio::netui {
namespace {

// Byte layout of a SerializedList buffer (see ui_codec.cc): a 24-byte header
// holding ref_pos[2], vert_count, idx_count, and command_count, then 8-byte
// aligned vertex, index, and 36-byte draw command sections.
constexpr size_t kListHeaderBytes = 24;
constexpr size_t kVertCountOffset = 8;
constexpr size_t kIdxCountOffset = 12;
constexpr size_t kCommandBytes = 36;
constexpr size_t kCmdVtxOffset = 8;
constexpr size_t kCmdIdxOffset = 12;
constexpr size_t kCmdElemCount = 16;
constexpr size_t kCmdClipRect = 20;

size_t AlignUp8(size_t bytes) { return (bytes + 7) & ~size_t{7}; }

template <typename T>
T Peek(const SerializedList& list, size_t offset) {
  T value;
  std::memcpy(&value, list.bytes.data() + offset, sizeof(T));
  return value;
}

template <typename T>
void Poke(SerializedList* list, size_t offset, T value) {
  std::memcpy(list->bytes.data() + offset, &value, sizeof(T));
}

size_t IndexOffset(const SerializedList& list, size_t index) {
  const auto vert_count = Peek<uint32_t>(list, kVertCountOffset);
  return kListHeaderBytes + AlignUp8(vert_count * sizeof(Vert)) +
         index * sizeof(uint16_t);
}

size_t CommandOffset(const SerializedList& list, size_t command) {
  const auto idx_count = Peek<uint32_t>(list, kIdxCountOffset);
  return IndexOffset(list, 0) + AlignUp8(idx_count * sizeof(uint16_t)) +
         command * kCommandBytes;
}

// True if `frame` encodes into a keyframe that a fresh decoder accepts.
bool DecodesAsKeyframe(const SerializedFrame& frame) {
  SerializedFrame prev;
  uint64_t prev_idx = 0;
  DecodedFrame out;
  bool request_keyframe = false;
  return DecodeDrawFrame(EncodeDrawFrame(frame, /*frame_index=*/1), &prev,
                         &prev_idx, &out, &request_keyframe);
}

void ExpectSameSerialized(const SerializedFrame& a, const SerializedFrame& b) {
  EXPECT_EQ(a.mouse_cursor, b.mouse_cursor);
  EXPECT_EQ(std::memcmp(a.display_size, b.display_size, sizeof(a.display_size)),
            0);
  ASSERT_EQ(a.lists.size(), b.lists.size());
  for (size_t i = 0; i < a.lists.size(); ++i) {
    EXPECT_EQ(a.lists[i].group_id, b.lists[i].group_id);
    EXPECT_EQ(a.lists[i].bytes, b.lists[i].bytes);
  }
}

void ExpectSameDecoded(const DecodedFrame& a, const DecodedFrame& b) {
  EXPECT_EQ(a.mouse_cursor, b.mouse_cursor);
  EXPECT_EQ(std::memcmp(a.display_size, b.display_size, sizeof(a.display_size)),
            0);
  ASSERT_EQ(a.lists.size(), b.lists.size());
  for (size_t i = 0; i < a.lists.size(); ++i) {
    const DrawList& la = a.lists[i];
    const DrawList& lb = b.lists[i];
    EXPECT_EQ(la.group_id, lb.group_id);
    ASSERT_EQ(la.verts.size(), lb.verts.size());
    EXPECT_EQ(std::memcmp(la.verts.data(), lb.verts.data(),
                          la.verts.size() * sizeof(Vert)),
              0);
    EXPECT_EQ(la.indices, lb.indices);
    ASSERT_EQ(la.commands.size(), lb.commands.size());
    for (size_t c = 0; c < la.commands.size(); ++c) {
      EXPECT_EQ(la.commands[c].tex_id, lb.commands[c].tex_id);
      EXPECT_EQ(la.commands[c].vtx_offset, lb.commands[c].vtx_offset);
      EXPECT_EQ(la.commands[c].idx_offset, lb.commands[c].idx_offset);
      EXPECT_EQ(la.commands[c].elem_count, lb.commands[c].elem_count);
      EXPECT_EQ(std::memcmp(la.commands[c].clip_rect, lb.commands[c].clip_rect,
                            sizeof(la.commands[c].clip_rect)),
                0);
    }
  }
}

class UiCodecTest : public ::testing::Test {
 protected:
  UiCodecTest() : context_(ImGui::CreateContext()) {
    ImGui::SetCurrentContext(context_);
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1280, 720);
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int w = 0, h = 0;
    io.Fonts->GetTexDataAsAlpha8(&pixels, &w, &h);
    io.Fonts->SetTexID(static_cast<ImTextureID>(7));
  }

  ~UiCodecTest() override { ImGui::DestroyContext(context_); }

  static SerializedFrame RenderTwoWindows(
      const char* win1_text, const char* win2_text,
      ImGuiMouseCursor cursor = ImGuiMouseCursor_Arrow) {
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20, 30));
    ImGui::SetNextWindowSize(ImVec2(300, 180));
    ImGui::Begin("Inspector");
    ImGui::TextUnformatted(win1_text);
    ImGui::Button("Apply");
    ImGui::End();

    if (win2_text != nullptr) {
      ImGui::SetNextWindowPos(ImVec2(400, 30));
      ImGui::SetNextWindowSize(ImVec2(320, 220));
      ImGui::Begin("Scene Tree");
      ImGui::TextUnformatted(win2_text);
      ImGui::Button("Reset");
      ImGui::End();
    }

    ImGui::Render();
    SerializedFrame serialized;
    SerializeDrawData(ImGui::GetDrawData(), cursor, &serialized);
    return serialized;
  }

  ImGuiContext* context_ = nullptr;
};

TEST_F(UiCodecTest, EmptyFrameRoundTripsKeyframeAndDelta) {
  ImGui::NewFrame();
  ImGui::Render();

  SerializedFrame empty;
  SerializeDrawData(ImGui::GetDrawData(), ImGuiMouseCursor_Hand, &empty);
  EXPECT_TRUE(empty.lists.empty());

  const Bytes keyframe = EncodeDrawFrame(empty, /*frame_index=*/1);
  SerializedFrame decoder_prev;
  uint64_t decoder_index = 0;
  DecodedFrame decoded;
  bool request_keyframe = false;

  ASSERT_TRUE(DecodeDrawFrame(keyframe, &decoder_prev, &decoder_index, &decoded,
                              &request_keyframe));
  EXPECT_FALSE(request_keyframe);
  EXPECT_EQ(decoder_index, 1u);
  EXPECT_EQ(decoded.mouse_cursor, ImGuiMouseCursor_Hand);
  EXPECT_EQ(decoded.display_size[0], 1280.0f);
  EXPECT_EQ(decoded.display_size[1], 720.0f);
  EXPECT_TRUE(decoded.lists.empty());

  const Bytes delta =
      EncodeDrawFrame(empty, /*frame_index=*/2, /*prev_frame=*/&empty);
  ASSERT_TRUE(DecodeDrawFrame(delta, &decoder_prev, &decoder_index, &decoded,
                              &request_keyframe));
  EXPECT_FALSE(request_keyframe);
  EXPECT_EQ(decoder_index, 2u);
  EXPECT_TRUE(decoded.lists.empty());
}

TEST_F(UiCodecTest,
       PerWindowDeltaCompressesUnchangedWindowsEvenWhenSizesShift) {
  // Warm up one frame so Dear ImGui finishes initial window appearing state.
  RenderTwoWindows("Warmup", "Unchanged hierarchy with several widgets");

  // Frame 1 has two windows. Frame 2 changes the vertex count of Window 1
  // ("Short" -> a much longer string) while Window 2 is untouched.
  const SerializedFrame frame1 =
      RenderTwoWindows("Short", "Unchanged hierarchy with several widgets");
  const SerializedFrame frame2 = RenderTwoWindows(
      "Much longer inspector text that adds dozens of vertices!",
      "Unchanged hierarchy with several widgets", ImGuiMouseCursor_TextInput);
  ASSERT_EQ(frame1.lists.size(), 2u);
  ASSERT_EQ(frame2.lists.size(), 2u);
  ASSERT_NE(frame1.lists[0].bytes.size(), frame2.lists[0].bytes.size());
  ASSERT_EQ(frame1.lists[1].bytes, frame2.lists[1].bytes);

  const Bytes msg1 = EncodeDrawFrame(frame1, /*frame_index=*/1);
  const Bytes msg2_full = EncodeDrawFrame(frame2, /*frame_index=*/2);
  const Bytes msg2_delta = EncodeDrawFrame(frame2, /*frame_index=*/2, &frame1);

  // Delta frame is substantially smaller than the keyframe because Window 2
  // compresses to a single 8-byte RLE block.
  EXPECT_LT(msg2_delta.size(), msg2_full.size());

  SerializedFrame decoder_prev;
  uint64_t decoder_index = 0;
  DecodedFrame out1, out2_from_delta, out2_from_full;
  bool request_keyframe = false;
  ASSERT_TRUE(DecodeDrawFrame(msg1, &decoder_prev, &decoder_index, &out1,
                              &request_keyframe));
  ASSERT_TRUE(DecodeDrawFrame(msg2_delta, &decoder_prev, &decoder_index,
                              &out2_from_delta, &request_keyframe));
  EXPECT_FALSE(request_keyframe);

  SerializedFrame fresh_prev;
  uint64_t fresh_index = 0;
  ASSERT_TRUE(DecodeDrawFrame(msg2_full, &fresh_prev, &fresh_index,
                              &out2_from_full, &request_keyframe));

  EXPECT_EQ(out2_from_delta.mouse_cursor, ImGuiMouseCursor_TextInput);
  ExpectSameDecoded(out2_from_delta, out2_from_full);
}

TEST_F(UiCodecTest, ClampsExtremeAndNanVertexCoordsWithoutUndefinedBehavior) {
  ImDrawList list(ImGui::GetDrawListSharedData());
  ImDrawVert v0, v1, v2;
  v0.pos = ImVec2(100.0f, 200.0f);
  v0.uv = ImVec2(0.25f, 0.75f);
  v0.col = 0xFF112233u;

  v1.pos = ImVec2(1e9f, -1e9f);  // Far outside [-8192, +8192].
  v1.uv = ImVec2(-5.0f, 9.0f);   // Outside [0, 1].
  v1.col = 0xFF445566u;

  v2.pos = ImVec2(std::numeric_limits<float>::quiet_NaN(), 250.5f);
  v2.uv = ImVec2(std::numeric_limits<float>::quiet_NaN(), 0.5f);
  v2.col = 0xFF778899u;

  list.VtxBuffer.push_back(v0);
  list.VtxBuffer.push_back(v1);
  list.VtxBuffer.push_back(v2);
  list.IdxBuffer.push_back(0);
  list.IdxBuffer.push_back(1);
  list.IdxBuffer.push_back(2);
  list.Flags = ImDrawListFlags_AllowVtxOffset;
  list._VtxWritePtr = list.VtxBuffer.Data + list.VtxBuffer.Size;
  list._IdxWritePtr = list.IdxBuffer.Data + list.IdxBuffer.Size;
  list._VtxCurrentIdx = 3;
  ImDrawCmd cmd;
  cmd.ElemCount = 3;
  cmd.VtxOffset = 0;
  cmd.ClipRect = ImVec4(0, 0, 500, 400);
  cmd.TexRef._TexID = static_cast<ImTextureID>(42);
  list.CmdBuffer.push_back(cmd);

  ImDrawData draw_data;
  draw_data.Valid = true;
  draw_data.DisplayPos = ImVec2(0, 0);
  draw_data.DisplaySize = ImVec2(500, 400);
  draw_data.AddDrawList(&list);

  SerializedFrame serialized;
  SerializeDrawData(&draw_data, ImGuiMouseCursor_Arrow, &serialized);
  const Bytes wire = EncodeDrawFrame(serialized, /*frame_index=*/1);

  SerializedFrame prev;
  uint64_t prev_idx = 0;
  DecodedFrame decoded;
  bool request_keyframe = false;
  ASSERT_TRUE(
      DecodeDrawFrame(wire, &prev, &prev_idx, &decoded, &request_keyframe));
  ASSERT_EQ(decoded.lists.size(), 1u);
  const DrawList& out_list = decoded.lists[0];
  ASSERT_EQ(out_list.verts.size(), 3u);
  ASSERT_EQ(out_list.commands.size(), 1u);
  EXPECT_EQ(out_list.commands[0].vtx_offset, 0u);
  EXPECT_EQ(out_list.commands[0].tex_id, 42u);

  const ImVec2 p0 = out_list.verts[0].UnpackPos(out_list.ref_pos);
  const ImVec2 uv0 = out_list.verts[0].UnpackUv();
  EXPECT_NEAR(p0.x, 100.0f, 0.25f);
  EXPECT_NEAR(p0.y, 200.0f, 0.25f);
  EXPECT_NEAR(uv0.x, 0.25f, 1e-4f);
  EXPECT_NEAR(uv0.y, 0.75f, 1e-4f);

  // Clamped extremes and NaNs stay finite and bounded.
  EXPECT_EQ(out_list.verts[1].pos[0], 0xFFFFu);
  EXPECT_EQ(out_list.verts[1].pos[1], 0u);
  EXPECT_EQ(out_list.verts[1].uv[0], 0u);
  EXPECT_EQ(out_list.verts[1].uv[1], 0xFFFFu);
  EXPECT_EQ(out_list.verts[2].pos[0], 0u);
  EXPECT_EQ(out_list.verts[2].uv[0], 0u);
}

TEST_F(UiCodecTest, MissingReferenceFrameRequestsKeyframeWithoutMutatingState) {
  const SerializedFrame frame1 = RenderTwoWindows("One", nullptr);
  const SerializedFrame frame2 = RenderTwoWindows("Two", nullptr);
  const SerializedFrame frame3 = RenderTwoWindows("Three", nullptr);

  const Bytes msg1 = EncodeDrawFrame(frame1, 1);
  const Bytes msg2_delta = EncodeDrawFrame(frame2, 2, &frame1);
  const Bytes msg3_delta = EncodeDrawFrame(frame3, 3, &frame2);

  SerializedFrame prev;
  uint64_t prev_idx = 0;
  DecodedFrame out;
  bool request_keyframe = false;

  // Delta frame before any keyframe -> rejected + requests keyframe.
  EXPECT_FALSE(DecodeDrawFrame(msg2_delta, &prev, &prev_idx, &out,
                               &request_keyframe));
  EXPECT_TRUE(request_keyframe);
  EXPECT_EQ(prev_idx, 0u);

  // Keyframe 1 succeeds, then skipping frame 2 to frame 3 requests keyframe and
  // preserves frame 1 as the last valid reference.
  ASSERT_TRUE(
      DecodeDrawFrame(msg1, &prev, &prev_idx, &out, &request_keyframe));
  EXPECT_FALSE(request_keyframe);
  EXPECT_EQ(prev_idx, 1u);
  const SerializedFrame prev_before = prev;
  const DecodedFrame out_before = out;

  EXPECT_FALSE(DecodeDrawFrame(msg3_delta, &prev, &prev_idx, &out,
                               &request_keyframe));
  EXPECT_TRUE(request_keyframe);
  EXPECT_EQ(prev_idx, 1u);
  ExpectSameSerialized(prev, prev_before);
  ExpectSameDecoded(out, out_before);

  // Frame 2 still decodes cleanly against preserved frame 1!
  EXPECT_TRUE(DecodeDrawFrame(msg2_delta, &prev, &prev_idx, &out,
                              &request_keyframe));
  EXPECT_FALSE(request_keyframe);
  EXPECT_EQ(prev_idx, 2u);
}

TEST_F(UiCodecTest, RejectsTruncatedMisalignedAndCorruptedWireFrames) {
  const SerializedFrame frame1 = RenderTwoWindows("Alpha", "Beta");
  const SerializedFrame frame2 = RenderTwoWindows("Alpha!", "Beta");
  const Bytes keyframe = EncodeDrawFrame(frame1, 1);
  const Bytes delta = EncodeDrawFrame(frame2, 2, &frame1);
  const BytesView keyframe_view(keyframe);

  SerializedFrame prev;
  uint64_t prev_idx = 0;
  DecodedFrame out;
  bool request_keyframe = false;

  // Truncated and non-8-byte-aligned messages.
  EXPECT_FALSE(DecodeDrawFrame(keyframe_view.first(4), &prev, &prev_idx, &out,
                               &request_keyframe));
  EXPECT_FALSE(DecodeDrawFrame(keyframe_view.first(keyframe.size() - 1), &prev,
                               &prev_idx, &out, &request_keyframe));
  EXPECT_FALSE(DecodeDrawFrame(keyframe_view.first(keyframe.size() - 8), &prev,
                               &prev_idx, &out, &request_keyframe));

  // Header size mismatch (trailing 8 bytes appended, header.size updated so
  // only trailing-byte validation catches it).
  Bytes trailing = keyframe;
  trailing.resize(trailing.size() + 8, std::byte{0});
  DrawFrame hdr;
  std::memcpy(&hdr, trailing.data(), sizeof(hdr));
  hdr.header.size = static_cast<uint32_t>(trailing.size());
  std::memcpy(trailing.data(), &hdr, sizeof(hdr));
  EXPECT_FALSE(
      DecodeDrawFrame(trailing, &prev, &prev_idx, &out, &request_keyframe));

  // Seed valid reference frame 1, then corrupt the RLE block header in `delta`.
  ASSERT_TRUE(
      DecodeDrawFrame(keyframe, &prev, &prev_idx, &out, &request_keyframe));
  const SerializedFrame prev_before = prev;
  const DecodedFrame out_before = out;

  // Offset 40 (DrawFrame) + 16 (first WireListHeader) = first RLE block.
  constexpr size_t kFirstRleBlockOffset = sizeof(DrawFrame) + 16;
  ASSERT_GT(delta.size(), kFirstRleBlockOffset + sizeof(uint64_t));

  // 1) Huge `same_chunks` overflowing `prev` and `unpacked_bytes`.
  {
    Bytes bad = delta;
    const uint32_t bad_block[2] = {0x7FFFFFFFu, 0u};
    std::memcpy(bad.data() + kFirstRleBlockOffset, bad_block, 8);
    EXPECT_FALSE(
        DecodeDrawFrame(bad, &prev, &prev_idx, &out, &request_keyframe));
    EXPECT_TRUE(request_keyframe);
    EXPECT_EQ(prev_idx, 1u);
    ExpectSameSerialized(prev, prev_before);
    ExpectSameDecoded(out, out_before);
  }

  // 2) Huge `diff_chunks` exceeding remaining packed payload.
  {
    Bytes bad = delta;
    const uint32_t bad_block[2] = {0u, 0x7FFFFFFFu};
    std::memcpy(bad.data() + kFirstRleBlockOffset, bad_block, 8);
    EXPECT_FALSE(
        DecodeDrawFrame(bad, &prev, &prev_idx, &out, &request_keyframe));
    EXPECT_TRUE(request_keyframe);
  }

  // 3) Zero-progress `(0, 0)` RLE block must be rejected immediately.
  {
    Bytes bad = delta;
    const uint32_t zero_block[2] = {0u, 0u};
    std::memcpy(bad.data() + kFirstRleBlockOffset, zero_block, 8);
    EXPECT_FALSE(
        DecodeDrawFrame(bad, &prev, &prev_idx, &out, &request_keyframe));
    EXPECT_TRUE(request_keyframe);
    EXPECT_EQ(prev_idx, 1u);
    ExpectSameSerialized(prev, prev_before);
    ExpectSameDecoded(out, out_before);
  }

  // 4) Corrupted `vert_count` inside a keyframe's SerializedListHeader so
  // section sizes disagree with `list.bytes.size()`.
  {
    Bytes bad = keyframe;
    constexpr size_t kWireVertCountOffset =
        sizeof(DrawFrame) + 16 + kVertCountOffset;
    const uint32_t bogus_verts = 999999u;
    std::memcpy(bad.data() + kWireVertCountOffset, &bogus_verts, 4);
    SerializedFrame fresh_prev;
    uint64_t fresh_idx = 0;
    EXPECT_FALSE(DecodeDrawFrame(bad, &fresh_prev, &fresh_idx, &out,
                                 &request_keyframe));
  }
}

TEST_F(UiCodecTest,
       DeltaFramesHandleWindowsAppearingDisappearingAndReordering) {
  RenderTwoWindows("Warmup", "Warmup");
  const SerializedFrame both = RenderTwoWindows("Alpha", "Beta");
  const SerializedFrame only_first = RenderTwoWindows("Alpha", nullptr);
  const SerializedFrame both_again = RenderTwoWindows("Alpha 2", "Beta");
  ASSERT_EQ(both.lists.size(), 2u);
  ASSERT_EQ(only_first.lists.size(), 1u);
  ASSERT_EQ(both_again.lists.size(), 2u);
  // The same windows in the opposite order, so matching lists by position
  // fails and the decoder must search by group_id.
  SerializedFrame reordered = both_again;
  std::swap(reordered.lists[0], reordered.lists[1]);

  const SerializedFrame* frames[] = {&both, &only_first, &both_again,
                                     &reordered};
  SerializedFrame prev;
  uint64_t prev_idx = 0;
  bool request_keyframe = false;
  for (uint64_t i = 0; i < std::size(frames); ++i) {
    SCOPED_TRACE(i);
    const SerializedFrame* reference = i > 0 ? frames[i - 1] : nullptr;
    const Bytes message = EncodeDrawFrame(*frames[i], i + 1, reference);
    DecodedFrame from_delta;
    ASSERT_TRUE(DecodeDrawFrame(message, &prev, &prev_idx, &from_delta,
                                &request_keyframe));
    EXPECT_FALSE(request_keyframe);

    SerializedFrame fresh_prev;
    uint64_t fresh_idx = 0;
    DecodedFrame from_keyframe;
    ASSERT_TRUE(DecodeDrawFrame(EncodeDrawFrame(*frames[i], 1), &fresh_prev,
                                &fresh_idx, &from_keyframe, &request_keyframe));
    ExpectSameDecoded(from_delta, from_keyframe);
  }

  // Reordering alone changes nothing, so the delta is far smaller.
  EXPECT_LT(EncodeDrawFrame(reordered, 5, &both_again).size(),
            EncodeDrawFrame(reordered, 5).size() / 4);
}

TEST_F(UiCodecTest, RejectsDrawFrameHeadersThatAreNotOneValidFrame) {
  const SerializedFrame frame = RenderTwoWindows("Alpha", "Beta");
  const Bytes keyframe = EncodeDrawFrame(frame, 1);
  SerializedFrame prev;
  uint64_t prev_idx = 0;
  DecodedFrame out;
  bool request_keyframe = false;
  ASSERT_TRUE(
      DecodeDrawFrame(keyframe, &prev, &prev_idx, &out, &request_keyframe));
  const SerializedFrame prev_before = prev;
  const DecodedFrame out_before = out;

  auto expect_rejected = [&](auto mutate) {
    Bytes bad = keyframe;
    DrawFrame hdr;
    std::memcpy(&hdr, bad.data(), sizeof(hdr));
    mutate(hdr, bad);
    std::memcpy(bad.data(), &hdr, sizeof(hdr));
    EXPECT_FALSE(
        DecodeDrawFrame(bad, &prev, &prev_idx, &out, &request_keyframe));
    EXPECT_EQ(prev_idx, 1u);
    ExpectSameSerialized(prev, prev_before);
    ExpectSameDecoded(out, out_before);
  };
  expect_rejected(
      [](DrawFrame& hdr, Bytes&) { hdr.header.kind = MessageKind::kTexture; });
  expect_rejected([](DrawFrame& hdr, Bytes&) {
    hdr.header.version = kCodecVersion + 1;
  });
  expect_rejected([](DrawFrame& hdr, Bytes&) { hdr.header.flags |= 1u << 1; });
  expect_rejected([](DrawFrame& hdr, Bytes&) { hdr.list_count = 0xFFFFFFFFu; });
  expect_rejected([](DrawFrame& hdr, Bytes&) { hdr.list_count += 1; });
  expect_rejected([](DrawFrame& hdr, Bytes&) {
    hdr.display_size[0] = std::numeric_limits<float>::infinity();
  });
  expect_rejected([](DrawFrame& hdr, Bytes&) {
    hdr.display_pos[1] = std::numeric_limits<float>::quiet_NaN();
  });
  // A keyframe list whose payload is not its unpacked bytes.
  expect_rejected([](DrawFrame&, Bytes& bad) {
    constexpr size_t kPayloadBytesOffset = sizeof(DrawFrame) + 12;
    uint32_t payload_bytes = 0;
    std::memcpy(&payload_bytes, bad.data() + kPayloadBytesOffset, 4);
    payload_bytes -= 8;
    std::memcpy(bad.data() + kPayloadBytesOffset, &payload_bytes, 4);
  });
}

TEST_F(UiCodecTest, RejectsDrawCommandsOutsideTheirList) {
  const SerializedFrame frame = RenderTwoWindows("Alpha", nullptr);
  ASSERT_EQ(frame.lists.size(), 1u);
  ASSERT_TRUE(DecodesAsKeyframe(frame));
  const SerializedList& list = frame.lists[0];
  const size_t cmd = CommandOffset(list, 0);
  const auto vert_count = Peek<uint32_t>(list, kVertCountOffset);
  const auto idx_count = Peek<uint32_t>(list, kIdxCountOffset);
  const auto idx_offset = Peek<uint32_t>(list, cmd + kCmdIdxOffset);
  const auto elem_count = Peek<uint32_t>(list, cmd + kCmdElemCount);
  ASSERT_GT(elem_count, 0u);
  ASSERT_LT(vert_count, 0xFFFFu);

  auto expect_rejected = [&](auto mutate) {
    SerializedFrame bad = frame;
    mutate(&bad);
    EXPECT_FALSE(DecodesAsKeyframe(bad));
  };
  // Index range past the end of the index buffer.
  expect_rejected([&](SerializedFrame* bad) {
    Poke<uint32_t>(&bad->lists[0], cmd + kCmdIdxOffset,
                   idx_count - elem_count + 3);
  });
  expect_rejected([&](SerializedFrame* bad) {
    Poke<uint32_t>(&bad->lists[0], cmd + kCmdIdxOffset, 0xFFFFFFFFu);
  });
  // Not whole triangles.
  expect_rejected([&](SerializedFrame* bad) {
    Poke<uint32_t>(&bad->lists[0], cmd + kCmdElemCount, elem_count - 1);
  });
  // Indices offset past the last vertex.
  expect_rejected([&](SerializedFrame* bad) {
    Poke<uint32_t>(&bad->lists[0], cmd + kCmdVtxOffset, vert_count);
  });
  expect_rejected([&](SerializedFrame* bad) {
    Poke<uint16_t>(&bad->lists[0], IndexOffset(list, idx_offset),
                   static_cast<uint16_t>(vert_count));
  });
  // Non-finite floats.
  expect_rejected([&](SerializedFrame* bad) {
    Poke<float>(&bad->lists[0], cmd + kCmdClipRect + 4,
                std::numeric_limits<float>::quiet_NaN());
  });
  expect_rejected([&](SerializedFrame* bad) {
    Poke<float>(&bad->lists[0], 0, std::numeric_limits<float>::infinity());
  });
}

TEST_F(UiCodecTest, QuantizesPositionsToExactQuarterPixelSteps) {
  ImDrawList list(ImGui::GetDrawListSharedData());
  ImDrawVert v0, v1, v2;
  v0.pos = ImVec2(100.0f, 200.0f);  // Becomes ref_pos.
  v1.pos = ImVec2(100.0f + 8000.1f, 200.0f - 8000.3f);
  v2.pos = ImVec2(100.0f + 3.6f, 200.0f - 0.1f);
  for (const ImDrawVert& v : {v0, v1, v2}) list.VtxBuffer.push_back(v);
  list.IdxBuffer.push_back(0);
  list.IdxBuffer.push_back(1);
  list.IdxBuffer.push_back(2);
  list.Flags = ImDrawListFlags_AllowVtxOffset;
  list._VtxWritePtr = list.VtxBuffer.Data + list.VtxBuffer.Size;
  list._IdxWritePtr = list.IdxBuffer.Data + list.IdxBuffer.Size;
  list._VtxCurrentIdx = 3;
  ImDrawCmd cmd;
  cmd.ElemCount = 3;
  cmd.ClipRect = ImVec4(0, 0, 500, 400);
  list.CmdBuffer.push_back(cmd);
  ImDrawData draw_data;
  draw_data.Valid = true;
  draw_data.DisplaySize = ImVec2(500, 400);
  draw_data.AddDrawList(&list);

  SerializedFrame serialized;
  SerializeDrawData(&draw_data, ImGuiMouseCursor_Arrow, &serialized);
  SerializedFrame prev;
  uint64_t prev_idx = 0;
  DecodedFrame decoded;
  bool request_keyframe = false;
  ASSERT_TRUE(DecodeDrawFrame(EncodeDrawFrame(serialized, 1), &prev, &prev_idx,
                              &decoded, &request_keyframe));
  ASSERT_EQ(decoded.lists.size(), 1u);
  const DrawList& out = decoded.lists[0];
  ASSERT_EQ(out.verts.size(), 3u);

  // An offset of 0 round-trips exactly, and the error elsewhere is at most
  // half a step, even far from ref_pos.
  const ImVec2 p0 = out.verts[0].UnpackPos(out.ref_pos);
  EXPECT_EQ(p0.x, 100.0f);
  EXPECT_EQ(p0.y, 200.0f);
  for (int i = 1; i < 3; ++i) {
    const ImVec2 p = out.verts[i].UnpackPos(out.ref_pos);
    const ImVec2 expected = list.VtxBuffer[i].pos;
    EXPECT_LE(std::abs(p.x - expected.x), Vert::kPosStep / 2 + 1e-3f);
    EXPECT_LE(std::abs(p.y - expected.y), Vert::kPosStep / 2 + 1e-3f);
  }
}

TEST(EncodeDrawFrameTest, ReturnsEmptyForFramesTheDecoderWouldReject) {
  // Two lists whose unpacked bytes add up to more than kMaxMessageBytes.
  SerializedFrame too_big;
  too_big.lists.push_back({1, Bytes(kMaxMessageBytes / 2)});
  too_big.lists.push_back({2, Bytes(kMaxMessageBytes / 2)});
  EXPECT_TRUE(EncodeDrawFrame(too_big, 1).empty());
  EXPECT_TRUE(EncodeDrawFrame(too_big, 2, &too_big).empty());

  // One list that exactly fills a keyframe, but whose RLE block header makes a
  // compressed frame against an empty reference one chunk too large.
  constexpr size_t kListBytes =
      kMaxMessageBytes - sizeof(DrawFrame) - 16 /* WireListHeader */;
  SerializedFrame just_fits;
  just_fits.lists.push_back({1, Bytes(kListBytes)});
  EXPECT_EQ(EncodeDrawFrame(just_fits, 1).size(), kMaxMessageBytes);
  const SerializedFrame empty;
  EXPECT_TRUE(EncodeDrawFrame(just_fits, 2, &empty).empty());
}

TEST(InputTest, EncodeAndDecodeInputAcceptOneValidInputOnly) {
  Input input;
  input.header.flags = Input::kRequestKeyframe | Input::kMacOsBehaviors;
  input.chars[0] = 'a';
  input.char_count = 1;
  input.mouse_pos[0] = 10.0f;
  input.mouse_pos[1] = 20.0f;
  const auto bytes = [](const Input& in) {
    return std::as_bytes(std::span(&in, 1));
  };
  const Bytes encoded = EncodeInput(input);
  ASSERT_EQ(encoded.size(), sizeof(Input));
  Input decoded;
  ASSERT_TRUE(DecodeInput(encoded, &decoded));
  EXPECT_EQ(decoded.char_count, 1u);
  EXPECT_EQ(decoded.mouse_pos[1], 20.0f);
  // The default "no mouse" position (-FLT_MAX) is finite and accepted.
  EXPECT_TRUE(DecodeInput(EncodeInput(Input()), &decoded));

  EXPECT_FALSE(DecodeInput(bytes(input), nullptr));
  EXPECT_FALSE(DecodeInput(bytes(input).first(sizeof(Input) - 1), &decoded));
  auto expect_rejected = [&](auto mutate) {
    Input bad = input;
    mutate(bad);
    EXPECT_FALSE(DecodeInput(bytes(bad), &decoded));
  };
  expect_rejected([](Input& in) { in.header.kind = MessageKind::kTexture; });
  expect_rejected([](Input& in) { in.header.version = kCodecVersion + 1; });
  expect_rejected([](Input& in) { in.header.size -= 1; });
  expect_rejected([](Input& in) {
    in.header.flags |= 1u << 2;
    EXPECT_TRUE(EncodeInput(in).empty());
  });
  expect_rejected([](Input& in) {
    in.char_count = Input::kMaxChars + 1;
    EXPECT_TRUE(EncodeInput(in).empty());
  });
  expect_rejected([](Input& in) {
    in.framebuffer_scale = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(EncodeInput(in).empty());
  });
  expect_rejected([](Input& in) {
    in.mouse_pos[0] = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(EncodeInput(in).empty());
  });
  expect_rejected([](Input& in) {
    in.mouse_wheel[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(EncodeInput(in).empty());
  });
}

Bytes TextureMessage(Texture::Action action, Texture::Format format,
                     uint16_t width, uint16_t height, size_t payload_bytes) {
  Texture hdr;
  hdr.tex_id = 3;
  hdr.action = action;
  hdr.format = format;
  hdr.width = width;
  hdr.height = height;
  const Bytes payload(payload_bytes, std::byte{0x5A});
  if (const Bytes encoded = EncodeTexture(hdr, payload); !encoded.empty()) {
    return encoded;
  }
  hdr.header.size = static_cast<uint32_t>(sizeof(Texture) + payload_bytes);
  Bytes message(sizeof(Texture) + payload_bytes, std::byte{0x5A});
  std::memcpy(message.data(), &hdr, sizeof(hdr));
  return message;
}

TEST(TextureTest, EncodeAndDecodeTextureCheckPayloadSizeForActionAndFormat) {
  using Action = Texture::Action;
  using Format = Texture::Format;
  Texture hdr;
  BytesView payload;

  ASSERT_TRUE(DecodeTexture(TextureMessage(Action::kCreate, Format::kRgba8, 4,
                                           2, 4 * 2 * 4),
                            &hdr, &payload));
  EXPECT_EQ(hdr.tex_id, 3u);
  EXPECT_EQ(payload.size(), 32u);
  EXPECT_TRUE(DecodeTexture(
      TextureMessage(Action::kUpdate, Format::kAlpha8, 3, 5, 15), &hdr,
      &payload));
  EXPECT_TRUE(DecodeTexture(
      TextureMessage(Action::kCreate, Format::kCustom, 640, 480, 100), &hdr,
      &payload));
  EXPECT_TRUE(DecodeTexture(
      TextureMessage(Action::kDestroy, Format::kRgba8, 0, 0, 0), &hdr,
      &payload));

  // EncodeTexture rejects non-zero flags and invalid action/format/size
  // combinations before writing to the wire.
  Texture bad_tex;
  bad_tex.width = 2;
  bad_tex.height = 2;
  EXPECT_TRUE(EncodeTexture(bad_tex, Bytes(15)).empty());
  bad_tex.header.flags = 1;
  EXPECT_TRUE(EncodeTexture(bad_tex, Bytes(16)).empty());

  // Payload sizes that disagree with the dimensions, including a 65535 x 65535
  // RGBA texture whose byte count overflows a 32-bit size_t.
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(Action::kCreate, Format::kRgba8, 4, 2, 31), &hdr,
      &payload));
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(Action::kCreate, Format::kRgba8, 0xFFFF, 0xFFFF, 4), &hdr,
      &payload));
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(Action::kCreate, Format::kRgba8, 0, 2, 0), &hdr,
      &payload));
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(Action::kCreate, Format::kCustom, 640, 480, 0), &hdr,
      &payload));
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(Action::kDestroy, Format::kRgba8, 0, 0, 8), &hdr,
      &payload));
  // Unknown action and format.
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(static_cast<Action>(7), Format::kRgba8, 1, 1, 4), &hdr,
      &payload));
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(Action::kCreate, static_cast<Format>(7), 1, 1, 4), &hdr,
      &payload));
  // Malformed headers.
  Bytes bad_flags = TextureMessage(Action::kCreate, Format::kRgba8, 1, 1, 4);
  bad_flags[4] = std::byte{1};  // MessageHeader::flags.
  EXPECT_FALSE(DecodeTexture(bad_flags, &hdr, &payload));
  Bytes bad_size = TextureMessage(Action::kCreate, Format::kRgba8, 1, 1, 4);
  bad_size.push_back(std::byte{0});
  EXPECT_FALSE(DecodeTexture(bad_size, &hdr, &payload));
  EXPECT_FALSE(DecodeTexture(
      TextureMessage(Action::kCreate, Format::kRgba8, 1, 1, 4), nullptr,
      &payload));
}

TEST(InputTest, KeyBitmaskHelpersHandleBoundsAndRoundTrip) {
  Input input;
  EXPECT_FALSE(input.IsImGuiKeyDown(ImGuiKey_A));
  input.SetImGuiKeyDown(ImGuiKey_A, true);
  input.SetImGuiKeyDown(ImGuiKey_ReservedForModCtrl, true);
  EXPECT_TRUE(input.IsImGuiKeyDown(ImGuiKey_A));
  EXPECT_TRUE(input.IsImGuiKeyDown(ImGuiKey_ReservedForModCtrl));
  input.SetImGuiKeyDown(ImGuiKey_A, false);
  EXPECT_FALSE(input.IsImGuiKeyDown(ImGuiKey_A));

  // Out-of-range offsets are safely ignored.
  input.SetKeyDown(-1, true);
  input.SetKeyDown(static_cast<int>(Input::kMaxKeys), true);
  EXPECT_FALSE(input.IsKeyDown(-1));
  EXPECT_FALSE(input.IsKeyDown(static_cast<int>(Input::kMaxKeys)));
}

TEST(ClipboardTest, RoundTripsUtf8AndRejectsMalformedMessages) {
  for (std::string_view sample :
       {std::string_view(""), std::string_view("pos=\"0 0 1\""),
        std::string_view("MuJoCo — 60 Hz"),
        std::string_view("\x01OPEN_URL:https://mujoco.org")}) {
    const Bytes wire = EncodeClipboard(sample);
    std::string decoded = "untouched";
    ASSERT_TRUE(DecodeClipboard(wire, &decoded));
    EXPECT_EQ(decoded, sample);
  }

  std::string out;
  EXPECT_FALSE(DecodeClipboard(std::as_bytes(std::span("short", 5)), &out));
  EXPECT_FALSE(DecodeClipboard(EncodeClipboard("ok"), nullptr));

  // Wrong message kind.
  Hello hello;
  EXPECT_FALSE(DecodeClipboard(std::as_bytes(std::span(&hello, 1)), &out));

  // Mismatched header size.
  Bytes bad_size = EncodeClipboard("abc");
  bad_size.push_back(std::byte{'x'});
  EXPECT_FALSE(DecodeClipboard(bad_size, &out));

  // Unknown flag bit.
  Bytes bad_flags = EncodeClipboard("abc");
  bad_flags[4] = std::byte{1u << 1};  // MessageHeader::flags.
  EXPECT_FALSE(DecodeClipboard(bad_flags, &out));
}

TEST(ClipboardTest, OpenUrlFlagRoundTripsAndKeepsTextPure) {
  bool open_url = true;
  std::string out;
  // Text that happens to start with the old in-band marker is still text.
  ASSERT_TRUE(DecodeClipboard(EncodeClipboard("\x01OPEN_URL:not a url"), &out,
                              &open_url));
  EXPECT_FALSE(open_url);
  EXPECT_EQ(out, "\x01OPEN_URL:not a url");

  ASSERT_TRUE(DecodeClipboard(
      EncodeClipboard("https://mujoco.org", /*open_url=*/true), &out,
      &open_url));
  EXPECT_TRUE(open_url);
  EXPECT_EQ(out, "https://mujoco.org");
}

TEST(ClipboardTest, TruncatesOversizeTextOnACodepointBoundary) {
  constexpr size_t kMaxText = kMaxMessageBytes - sizeof(Clipboard);
  // A 3-byte euro sign straddling the size limit is dropped whole.
  const std::string text = std::string(kMaxText - 1, 'a') + "\xE2\x82\xAC";
  const Bytes wire = EncodeClipboard(text);
  EXPECT_EQ(wire.size(), kMaxMessageBytes - 1);
  std::string out;
  ASSERT_TRUE(DecodeClipboard(wire, &out));
  EXPECT_EQ(out, text.substr(0, kMaxText - 1));
}

TEST(ClipboardTest, IsOpenableUrlAllowsOnlyHttpAndHttps) {
  EXPECT_TRUE(IsOpenableUrl("https://mujoco.org"));
  EXPECT_TRUE(IsOpenableUrl("http://localhost:8000/path?q=1"));
  EXPECT_TRUE(IsOpenableUrl("HTTPS://MUJOCO.ORG"));
  EXPECT_FALSE(IsOpenableUrl("javascript:alert(1)"));
  EXPECT_FALSE(IsOpenableUrl("javascript://%0Aalert(1)"));
  EXPECT_FALSE(IsOpenableUrl("data:text/html,<script>alert(1)</script>"));
  EXPECT_FALSE(IsOpenableUrl("file:///etc/passwd"));
  EXPECT_FALSE(IsOpenableUrl("https:mujoco.org"));
  EXPECT_FALSE(IsOpenableUrl(" https://mujoco.org"));
  EXPECT_FALSE(IsOpenableUrl("mujoco.org"));
  EXPECT_FALSE(IsOpenableUrl(""));
}

}  // namespace
}  // namespace mujoco::studio::netui
