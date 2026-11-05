// Google modifications:
//  - ImguiDrawGroup::ToPointers() validates the offsets and sizes received
//    from the network against the draw frame command buffer.
//  - DecompressCmdDrawFrame() returns null for invalid compressed data.
#pragma once

#include "NetImgui_Shared.h"

namespace NetImgui {
namespace Internal {

struct ImguiVert {
  // Note: If updating this, increase 'CmdVersion::eVersion'
  enum Constants {
    kUvRange_Min = 0,
    kUvRange_Max = 1,
    kPosRange_Min = -8192,
    kPosRange_Max = 8192
  };
  uint16_t mPos[2];
  uint16_t mUV[2];
  uint32_t mColor;
};

struct ImguiDraw {
  ClientTextureID
      mClientTexId;     // TextureID used by client to identify the texture
  uint32_t mIdxCount;   // Drawcall number of indices (3 indices per triangles)
  uint32_t mVtxOffset;  // Drawcall start position in vertices buffer
                        // (considered index 0)
  uint32_t mIdxOffset;  // Drawcall start position in indices buffer
  float mClipRect[4];
  uint8_t PADDING[4] = {};
};

// Each DearImgui window has its own vertex/index buffers with multiple
// drawcalls
struct alignas(8) ImguiDrawGroup {
  static constexpr uint32_t kInvalidDrawGroup = 0xFFFFFFFF;
  uint64_t mGroupID = 0;  // Unique ID to recognize DrawGroup between 2 frames
  uint32_t mVerticeCount = 0;
  uint32_t mIndiceCount = 0;
  uint32_t mDrawCount = 0;
  uint32_t mDrawGroupIdxPrev =
      kInvalidDrawGroup;      // Group index in previous DrawFrame
                              // (kInvalidDrawGroup when not using delta
                              // compression)
  uint8_t mBytePerIndex = 2;  // 2, 4 bytes
  uint8_t PADDING[7] = {};
  float mReferenceCoord[2] = {};  // Reference position for the encoded vertices
                                  // offsets (1st vertice top/left position)
  OffsetPointer<uint8_t> mpIndices;
  OffsetPointer<ImguiVert> mpVertices;
  OffsetPointer<ImguiDraw> mpDraws;
  inline bool ToPointers(const void* pBufferStart, const void* pBufferEnd,
                         bool bCompressed);
  inline void ToOffsets();
};

struct CmdDrawFrame* ConvertToCmdDrawFrame(const ImDrawData* pDearImguiData,
                                           ImGuiMouseCursor cursor);
struct CmdDrawFrame* CompressCmdDrawFrame(const CmdDrawFrame* pDrawFramePrev,
                                          const CmdDrawFrame* pDrawFrameNew);
struct CmdDrawFrame* DecompressCmdDrawFrame(
    const CmdDrawFrame* pDrawFramePrev, const CmdDrawFrame* pDrawFramePacked);

}  // namespace Internal
}  // namespace NetImgui
