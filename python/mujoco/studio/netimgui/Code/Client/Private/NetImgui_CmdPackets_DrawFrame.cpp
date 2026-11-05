// Google modifications:
//  - Guard SetAndIncreaseDataPointer when dataCount is 0, and zero-initialize
//    ImguiDraw entries in ImGui_ExtractDraws before delta compression.
//  - DecompressData/DecompressCmdDrawFrame validate the compressed data
//    received from the network against the input, output and previous frame
//    buffers.
#include "NetImgui_Shared.h"

#if NETIMGUI_ENABLED
#include "NetImgui_CmdPackets.h"
#include "NetImgui_WarningDisable.h"

namespace NetImgui {
namespace Internal {

template <typename TType>
inline void SetAndIncreaseDataPointer(OffsetPointer<TType>& dataPointer,
                                      uint32_t dataSize,
                                      ComDataType*& pDataOutput) {
  dataPointer.SetComDataPtr(pDataOutput);
  const size_t dataCount = DivUp<size_t>(dataSize, ComDataSize);
  if (dataCount > 0) {
    pDataOutput[dataCount - 1] = 0;
  }
  pDataOutput += dataCount;
}

//=============================================================================
// Safely convert a pointer to a int value, even if int storage size > pointer
//=============================================================================
template <typename TInt, typename TPointer>
TInt PointerCast(TPointer* pointer) {
  union CastHelperUnion {
    TInt ValueInt;
    TPointer* ValuePointer;
  };
  CastHelperUnion helperObject = {};
  helperObject.ValuePointer = pointer;
  return helperObject.ValueInt;
}

//=================================================================================================
//
//=================================================================================================
inline void ImGui_ExtractIndices(const ImDrawList& cmdList,
                                 ImguiDrawGroup& drawGroupOut,
                                 ComDataType*& pDataOutput) {
  bool is16Bit =
      sizeof(ImDrawIdx) == 2 ||
      cmdList.VtxBuffer.size() <=
          0xFFFF;  // When Dear Imgui is compiled with ImDrawIdx = uint16, we
                   // know for certain that there won't be any drawcall with
                   // index > 65k, even if Vertex buffer is bigger than 65k.
  drawGroupOut.mBytePerIndex = is16Bit ? 2 : 4;
  drawGroupOut.mIndiceCount = static_cast<uint32_t>(cmdList.IdxBuffer.size());
  uint32_t sizeNeeded = drawGroupOut.mIndiceCount * drawGroupOut.mBytePerIndex;
  SetAndIncreaseDataPointer(drawGroupOut.mpIndices, sizeNeeded, pDataOutput);

  // No conversion needed, straight copy
  if (drawGroupOut.mBytePerIndex == sizeof(ImDrawIdx)) {
    memcpy(drawGroupOut.mpIndices.Get(), &cmdList.IdxBuffer.front(),
           sizeNeeded);
  }
  // From 32bits to 16bits
  else if (is16Bit) {
    for (int i(0); i < static_cast<int>(drawGroupOut.mIndiceCount); ++i)
      reinterpret_cast<uint16_t*>(drawGroupOut.mpIndices.Get())[i] =
          static_cast<uint16_t>(cmdList.IdxBuffer[i]);
  }
  // From 16bits to 32bits
  else {
    for (int i(0); i < static_cast<int>(drawGroupOut.mIndiceCount); ++i)
      reinterpret_cast<uint32_t*>(drawGroupOut.mpIndices.Get())[i] =
          static_cast<uint32_t>(cmdList.IdxBuffer[i]);
  }
}

//=================================================================================================
//
//=================================================================================================
inline void ImGui_ExtractVertices(const ImDrawList& cmdList,
                                  ImguiDrawGroup& drawGroupOut,
                                  ComDataType*& pDataOutput) {
  drawGroupOut.mVerticeCount = static_cast<uint32_t>(cmdList.VtxBuffer.size());
  drawGroupOut.mReferenceCoord[0] =
      drawGroupOut.mVerticeCount > 0 ? cmdList.VtxBuffer[0].pos.x : 0.f;
  drawGroupOut.mReferenceCoord[1] =
      drawGroupOut.mVerticeCount > 0 ? cmdList.VtxBuffer[0].pos.y : 0.f;
  SetAndIncreaseDataPointer(drawGroupOut.mpVertices,
                            drawGroupOut.mVerticeCount * sizeof(ImguiVert),
                            pDataOutput);
  ImguiVert* pVertices = drawGroupOut.mpVertices.Get();
  for (int i(0); i < static_cast<int>(drawGroupOut.mVerticeCount); ++i) {
    const auto& Vtx = cmdList.VtxBuffer[i];
    pVertices[i].mColor = Vtx.col;
    pVertices[i].mUV[0] = static_cast<uint16_t>(
        (Vtx.uv.x - static_cast<float>(ImguiVert::kUvRange_Min) +
         0.5f / 65535.f) *
        0xFFFF / (ImguiVert::kUvRange_Max - ImguiVert::kUvRange_Min));
    pVertices[i].mUV[1] = static_cast<uint16_t>(
        (Vtx.uv.y - static_cast<float>(ImguiVert::kUvRange_Min) +
         0.5f / 65535.f) *
        0xFFFF / (ImguiVert::kUvRange_Max - ImguiVert::kUvRange_Min));
    pVertices[i].mPos[0] = static_cast<uint16_t>(
        (Vtx.pos.x - drawGroupOut.mReferenceCoord[0] -
         static_cast<float>(ImguiVert::kPosRange_Min)) *
        0xFFFF / (ImguiVert::kPosRange_Max - ImguiVert::kPosRange_Min));
    pVertices[i].mPos[1] = static_cast<uint16_t>(
        (Vtx.pos.y - drawGroupOut.mReferenceCoord[1] -
         static_cast<float>(ImguiVert::kPosRange_Min)) *
        0xFFFF / (ImguiVert::kPosRange_Max - ImguiVert::kPosRange_Min));
  }
}

//=================================================================================================
//
//=================================================================================================
inline void ImGui_ExtractDraws(const ImDrawList& cmdList,
                               ImguiDrawGroup& drawGroupOut,
                               ComDataType*& pDataOutput) {
  int maxDrawCount = static_cast<int>(cmdList.CmdBuffer.size());
  uint32_t drawCount = 0;
  ImguiDraw* pOutDraws = reinterpret_cast<ImguiDraw*>(pDataOutput);
  for (int cmd_i = 0; cmd_i < maxDrawCount; ++cmd_i) {
    const ImDrawCmd* pCmd = &cmdList.CmdBuffer[cmd_i];
    if (pCmd->UserCallback == nullptr) {
      pOutDraws[drawCount] = ImguiDraw();
#if IMGUI_VERSION_NUM >= 17100
      pOutDraws[drawCount].mVtxOffset = pCmd->VtxOffset;
      pOutDraws[drawCount].mIdxOffset = pCmd->IdxOffset;
#else
      pOutDraws[drawCount].mVtxOffset = 0;
      pOutDraws[drawCount].mIdxOffset = 0;
#endif

#if NETIMGUI_IMGUI_TEXTURES_ENABLED
      ClientTextureID texClientID = ConvertToClientTexID(pCmd->TexRef);
#else
      ClientTextureID texClientID = ConvertToClientTexID(pCmd->TextureId);
#endif
      pOutDraws[drawCount].mClientTexId = texClientID;
      pOutDraws[drawCount].mIdxCount = pCmd->ElemCount;
      pOutDraws[drawCount].mClipRect[0] = pCmd->ClipRect.x;
      pOutDraws[drawCount].mClipRect[1] = pCmd->ClipRect.y;
      pOutDraws[drawCount].mClipRect[2] = pCmd->ClipRect.z;
      pOutDraws[drawCount].mClipRect[3] = pCmd->ClipRect.w;
      ++drawCount;
    }
  }
  drawGroupOut.mDrawCount = drawCount;
  static_assert(sizeof(ImguiDraw) % ComDataSize == 0,
                "Need to support zero-ing the pending bytes, when not a size "
                "multiple of DataComType");
  drawGroupOut.mpDraws.SetComDataPtr(pDataOutput);
  pDataOutput += drawGroupOut.mDrawCount * sizeof(ImguiDraw) / ComDataSize;
}

//=================================================================================================
// Delta comress data.
// Take 2 data stream and output a stream with only the data difference from
// each other
//=================================================================================================
void CompressData(const ComDataType* pDataPrev, size_t dataSizePrev,
                  const ComDataType* pDataNew, size_t dataSizeNew,
                  ComDataType*& pCommandMemoryInOut) {
  static_assert(sizeof(uint32_t) * 2 <= ComDataSize,
                "Need to adjust compression algorithm pointer calculation");
  const size_t elemCountPrev =
      static_cast<size_t>(DivUp(dataSizePrev, sizeof(uint64_t)));
  const size_t elemCountNew =
      static_cast<size_t>(DivUp(dataSizeNew, sizeof(uint64_t)));
  const size_t elemCount =
      elemCountPrev < elemCountNew ? elemCountPrev : elemCountNew;
  size_t n = 0;

  if (pDataPrev) {
    while (n < elemCount) {
      uint32_t* pBlockInfo = reinterpret_cast<uint32_t*>(
          pCommandMemoryInOut++);  // Add a new block info to output

      // Find number of elements with same value as last frame
      size_t startN = n;
      while (n < elemCount && pDataPrev[n] == pDataNew[n]) ++n;
      pBlockInfo[0] = static_cast<uint32_t>(n - startN);

      // Find number of elements with different value as last frame, and save
      // new value
      while (n < elemCount && pDataPrev[n] != pDataNew[n]) {
        *pCommandMemoryInOut = pDataNew[n++];
        ++pCommandMemoryInOut;
      }
      pBlockInfo[1] =
          static_cast<uint32_t>(pCommandMemoryInOut -
                                reinterpret_cast<ComDataType*>(pBlockInfo)) -
          1;
    }
  }

  // New frame has more element than previous frame, add the remaining entries
  if (elemCount < elemCountNew) {
    uint32_t* pBlockInfo = reinterpret_cast<uint32_t*>(
        pCommandMemoryInOut++);  // Add a new block info to output
    while (n < elemCountNew) {
      *pCommandMemoryInOut = pDataNew[n++];
      ++pCommandMemoryInOut;
    }
    pBlockInfo[0] = 0;
    pBlockInfo[1] =
        static_cast<uint32_t>(pCommandMemoryInOut -
                              reinterpret_cast<uint64_t*>(pBlockInfo)) -
        1;
  }
}

//=================================================================================================
// Unpack a delta data compressed stream
// Returns false when the compressed data (received from the network) is
// invalid: it must stay inside [pDataPack, pDataPackEnd), generate exactly
// 'dataUnpackSize' bytes of output, and only keep unchanged the elements copied
// from the previous data.
//=================================================================================================
bool DecompressData(const ComDataType* pDataPrev, size_t dataSizePrev,
                    const ComDataType* pDataPack,
                    const ComDataType* pDataPackEnd, size_t dataUnpackSize,
                    ComDataType*& pCommandMemoryInOut) {
  const size_t elemCountPrev = pDataPrev ? DivUp(dataSizePrev, ComDataSize) : 0;
  const size_t elemCountUnpack = DivUp(dataUnpackSize, ComDataSize);
  const size_t elemCountCopy =
      elemCountPrev < elemCountUnpack ? elemCountPrev : elemCountUnpack;
  const uint64_t* pCommandMemoryCopyEnd = &pCommandMemoryInOut[elemCountCopy];
  uint64_t* pCommandMemoryEnd = &pCommandMemoryInOut[elemCountUnpack];
  if (pDataPrev) {
    memcpy(pCommandMemoryInOut, pDataPrev, elemCountCopy * ComDataSize);
  }
  while (pCommandMemoryInOut < pCommandMemoryEnd) {
    if (pDataPack >= pDataPackEnd) {
      return false;
    }
    const uint32_t* pBlockInfo = reinterpret_cast<const uint32_t*>(
        pDataPack++);  // Add a new block info to output
    const uint64_t elemCountSame = pBlockInfo[0];
    const uint64_t elemCountNew = pBlockInfo[1];
    // Unchanged elements are the ones copied from the previous data.
    const uint64_t elemCountSameMax =
        pCommandMemoryInOut < pCommandMemoryCopyEnd
            ? static_cast<uint64_t>(pCommandMemoryCopyEnd - pCommandMemoryInOut)
            : 0;
    if (elemCountSame + elemCountNew >
            static_cast<uint64_t>(pCommandMemoryEnd - pCommandMemoryInOut) ||
        elemCountNew > static_cast<uint64_t>(pDataPackEnd - pDataPack) ||
        elemCountSame > elemCountSameMax) {
      return false;
    }
    pCommandMemoryInOut += elemCountSame;
    memcpy(pCommandMemoryInOut, pDataPack, elemCountNew * sizeof(uint64_t));
    pCommandMemoryInOut += elemCountNew;
    pDataPack += elemCountNew;
  }
  return true;
}

//=================================================================================================
// Take a regular NetImgui DrawFrame command and create a new compressed command
// It uses a basic delta compression method that works really well with Imgui
// data
//	- Most of the drawing data do not change between 2 frames
//  - Even if 1 window content changes, the others windows probably won't be
//  changing at all
//  - This means that for each window, we can only send the data that changed
//  - This requires little cpu usage and generate good results
//    - In 'SampleBasic' with 3 windows open (Main Window, ImGui Demo, ImGui
//    Metric) at 30fps
//		- Compression Off: 1650KB/sec of transfert
//      - Compression On : 12KB/sec of transfert (130x less data)
//=================================================================================================
CmdDrawFrame* CompressCmdDrawFrame(const CmdDrawFrame* pDrawFramePrev,
                                   const CmdDrawFrame* pDrawFrameNew) {
  //-----------------------------------------------------------------------------------------
  // Allocate memory for the new compressed command
  //-----------------------------------------------------------------------------------------
  // Allocate memory for worst case scenario (no compression possible)
  // New DrawFrame size + 2 'compression block info' per data stream
  size_t neededDataCount =
      DivUp<size_t>(pDrawFrameNew->mSize, ComDataSize) +
      6 * static_cast<size_t>(pDrawFrameNew->mDrawGroupCount);
  CmdDrawFrame* pDrawFramePacked =
      netImguiSizedNew<CmdDrawFrame>(neededDataCount * ComDataSize);
  *pDrawFramePacked = *pDrawFrameNew;
  pDrawFramePacked->mCompressed = true;

  ComDataType* pDataOutput =
      reinterpret_cast<ComDataType*>(&pDrawFramePacked[1]);
  SetAndIncreaseDataPointer(
      pDrawFramePacked->mpDrawGroups,
      pDrawFramePacked->mDrawGroupCount * sizeof(ImguiDrawGroup), pDataOutput);

  //-----------------------------------------------------------------------------------------
  // Copy draw data (vertices, indices, drawcall info, ...)
  //-----------------------------------------------------------------------------------------
  const uint32_t groupCountPrev = pDrawFramePrev->mDrawGroupCount;
  for (uint32_t n = 0; n < pDrawFramePacked->mDrawGroupCount; n++) {
    // Look for the same drawgroup in previous frame
    // Can usually avoid a search by checking same index in previous frame
    // (drawgroup ordering shouldn't change often)
    const ImguiDrawGroup& drawGroupNew = pDrawFrameNew->mpDrawGroups[n];
    ImguiDrawGroup& drawGroup = pDrawFramePacked->mpDrawGroups[n];
    drawGroup = drawGroupNew;
    drawGroup.mDrawGroupIdxPrev =
        (n < groupCountPrev &&
         drawGroup.mGroupID == pDrawFramePrev->mpDrawGroups[n].mGroupID)
            ? n
            : ImguiDrawGroup::kInvalidDrawGroup;
    for (uint32_t j(0);
         j < groupCountPrev &&
         drawGroup.mDrawGroupIdxPrev == ImguiDrawGroup::kInvalidDrawGroup;
         ++j) {
      drawGroup.mDrawGroupIdxPrev =
          (drawGroup.mGroupID == pDrawFramePrev->mpDrawGroups[j].mGroupID)
              ? j
              : ImguiDrawGroup::kInvalidDrawGroup;
    }

    // Delta compress the 3 data streams
    const uint64_t *pVerticePrev(nullptr), *pIndicePrev(nullptr),
        *pDrawsPrev(nullptr);
    size_t verticeSizePrev(0), indiceSizePrev(0), drawSizePrev(0);
    if (drawGroup.mDrawGroupIdxPrev < pDrawFramePrev->mDrawGroupCount) {
      const ImguiDrawGroup& drawGroupPrev =
          pDrawFramePrev->mpDrawGroups[drawGroup.mDrawGroupIdxPrev];
      pVerticePrev =
          reinterpret_cast<const uint64_t*>(drawGroupPrev.mpVertices.Get());
      pIndicePrev =
          reinterpret_cast<const uint64_t*>(drawGroupPrev.mpIndices.Get());
      pDrawsPrev =
          reinterpret_cast<const uint64_t*>(drawGroupPrev.mpDraws.Get());
      verticeSizePrev = drawGroupPrev.mVerticeCount * sizeof(ImguiVert);
      indiceSizePrev = drawGroupPrev.mIndiceCount *
                       static_cast<size_t>(drawGroupPrev.mBytePerIndex);
      drawSizePrev = drawGroupPrev.mDrawCount * sizeof(ImguiDraw);
    }

    drawGroup.mpIndices.SetComDataPtr(pDataOutput);
    CompressData(pIndicePrev, indiceSizePrev,
                 drawGroupNew.mpIndices.GetComData(),
                 drawGroupNew.mIndiceCount *
                     static_cast<size_t>(drawGroupNew.mBytePerIndex),
                 pDataOutput);

    drawGroup.mpVertices.SetComDataPtr(pDataOutput);
    CompressData(pVerticePrev, verticeSizePrev,
                 drawGroupNew.mpVertices.GetComData(),
                 drawGroupNew.mVerticeCount * sizeof(ImguiVert), pDataOutput);

    drawGroup.mpDraws.SetComDataPtr(pDataOutput);
    CompressData(pDrawsPrev, drawSizePrev, drawGroupNew.mpDraws.GetComData(),
                 drawGroupNew.mDrawCount * sizeof(ImguiDraw), pDataOutput);
  }

  // Adjust data transfert amount to memory that has been actually needed
  pDrawFramePacked->mSize =
      static_cast<uint32_t>(
          (pDataOutput - reinterpret_cast<ComDataType*>(pDrawFramePacked))) *
      static_cast<uint32_t>(sizeof(uint64_t));
  return pDrawFramePacked;
}

//=================================================================================================
// Returns null when the compressed command (received from the network) is
// invalid.
//=================================================================================================
CmdDrawFrame* DecompressCmdDrawFrame(const CmdDrawFrame* pDrawFramePrev,
                                     const CmdDrawFrame* pDrawFramePacked) {
  //-----------------------------------------------------------------------------------------
  // Find memory needed for the uncompressed command from its draw groups
  // (instead of trusting the 'mUncompressedSize' received)
  //-----------------------------------------------------------------------------------------
  uint64_t neededDataCount = DivUp<uint64_t>(sizeof(CmdDrawFrame), ComDataSize);
  neededDataCount += DivUp<uint64_t>(
      static_cast<uint64_t>(pDrawFramePacked->mDrawGroupCount) *
          sizeof(ImguiDrawGroup),
      ComDataSize);
  for (uint32_t n = 0; n < pDrawFramePacked->mDrawGroupCount; n++) {
    const ImguiDrawGroup& drawGroupPack = pDrawFramePacked->mpDrawGroups[n];
    neededDataCount +=
        DivUp<uint64_t>(static_cast<uint64_t>(drawGroupPack.mIndiceCount) *
                            drawGroupPack.mBytePerIndex,
                        ComDataSize);
    neededDataCount +=
        DivUp<uint64_t>(static_cast<uint64_t>(drawGroupPack.mVerticeCount) *
                            sizeof(ImguiVert),
                        ComDataSize);
    neededDataCount += DivUp<uint64_t>(
        static_cast<uint64_t>(drawGroupPack.mDrawCount) * sizeof(ImguiDraw),
        ComDataSize);
  }
  if (neededDataCount * ComDataSize > PendingCom::kMaxCmdSize) {
    return nullptr;
  }
  const ComDataType* pDataPackEnd =
      reinterpret_cast<const ComDataType*>(pDrawFramePacked) +
      pDrawFramePacked->mSize / ComDataSize;

  //-----------------------------------------------------------------------------------------
  // Allocate memory for the new uncompressed compressed command
  //-----------------------------------------------------------------------------------------
  CmdDrawFrame* pDrawFrameNew = netImguiSizedNew<CmdDrawFrame>(
      static_cast<size_t>(neededDataCount * ComDataSize));
  *pDrawFrameNew = *pDrawFramePacked;
  pDrawFrameNew->mSize = pDrawFrameNew->mUncompressedSize =
      static_cast<uint32_t>(neededDataCount * ComDataSize);
  pDrawFrameNew->mCompressed = false;
  ComDataType* pDataOutput = reinterpret_cast<ComDataType*>(&pDrawFrameNew[1]);
  SetAndIncreaseDataPointer(
      pDrawFrameNew->mpDrawGroups,
      pDrawFrameNew->mDrawGroupCount * sizeof(ImguiDrawGroup), pDataOutput);

  for (uint32_t n = 0; n < pDrawFrameNew->mDrawGroupCount; n++) {
    const ImguiDrawGroup& drawGroupPack = pDrawFramePacked->mpDrawGroups[n];
    ImguiDrawGroup& drawGroup = pDrawFrameNew->mpDrawGroups[n];
    drawGroup = drawGroupPack;

    // Uncompress the 3 data streams
    const ComDataType* pVerticePrev = nullptr;
    const ComDataType* pIndicePrev = nullptr;
    const ComDataType* pDrawsPrev = nullptr;
    size_t verticeSizePrev(0), indiceSizePrev(0), drawSizePrev(0);
    if (drawGroup.mDrawGroupIdxPrev < pDrawFramePrev->mDrawGroupCount) {
      const ImguiDrawGroup& drawGroupPrev =
          pDrawFramePrev->mpDrawGroups[drawGroup.mDrawGroupIdxPrev];
      pVerticePrev =
          reinterpret_cast<const ComDataType*>(drawGroupPrev.mpVertices.Get());
      pIndicePrev =
          reinterpret_cast<const ComDataType*>(drawGroupPrev.mpIndices.Get());
      pDrawsPrev =
          reinterpret_cast<const ComDataType*>(drawGroupPrev.mpDraws.Get());
      verticeSizePrev = drawGroupPrev.mVerticeCount * sizeof(ImguiVert);
      indiceSizePrev = drawGroupPrev.mIndiceCount *
                       static_cast<size_t>(drawGroupPrev.mBytePerIndex);
      drawSizePrev = drawGroupPrev.mDrawCount * sizeof(ImguiDraw);
    }

    drawGroup.mpIndices.SetComDataPtr(pDataOutput);
    bool bValid = DecompressData(
        pIndicePrev, indiceSizePrev, drawGroupPack.mpIndices.GetComData(),
        pDataPackEnd,
        drawGroupPack.mIndiceCount *
            static_cast<size_t>(drawGroupPack.mBytePerIndex),
        pDataOutput);

    drawGroup.mpVertices.SetComDataPtr(pDataOutput);
    bValid = bValid && DecompressData(pVerticePrev, verticeSizePrev,
                                      drawGroupPack.mpVertices.GetComData(),
                                      pDataPackEnd,
                                      drawGroupPack.mVerticeCount *
                                          sizeof(ImguiVert),
                                      pDataOutput);

    drawGroup.mpDraws.SetComDataPtr(pDataOutput);
    bValid = bValid && DecompressData(pDrawsPrev, drawSizePrev,
                                      drawGroupPack.mpDraws.GetComData(),
                                      pDataPackEnd,
                                      drawGroupPack.mDrawCount *
                                          sizeof(ImguiDraw),
                                      pDataOutput);
    if (!bValid) {
      netImguiDeleteSafe(pDrawFrameNew);
      return nullptr;
    }
  }
  return pDrawFrameNew;
}

//=================================================================================================
// Take a regular Dear Imgui Draw Data, and convert it to a NetImgui DrawFrame
// Command It involves saving each window draw group vertex/indices/draw buffers
// and packing their data a little bit, to reduce the bandwidth usage
//=================================================================================================
CmdDrawFrame* ConvertToCmdDrawFrame(const ImDrawData* pDearImguiData,
                                    ImGuiMouseCursor mouseCursor) {
  //-----------------------------------------------------------------------------------------
  // Find memory needed for entire DrawFrame Command
  //-----------------------------------------------------------------------------------------
  static_assert(sizeof(CmdDrawFrame) % ComDataSize == 0,
                "Make sure Command Data is aligned to com data type size");
  size_t neededDataCount = DivUp(sizeof(CmdDrawFrame), ComDataSize);
  neededDataCount += DivUp(static_cast<size_t>(pDearImguiData->CmdListsCount) *
                               sizeof(ImguiDrawGroup),
                           ComDataSize);
  for (int n = 0; n < pDearImguiData->CmdListsCount; n++) {
    const ImDrawList* pCmdList = pDearImguiData->CmdLists[n];
    bool is16Bit = pCmdList->VtxBuffer.size() <= 0xFFFF;
    neededDataCount += DivUp(
        static_cast<size_t>(pCmdList->VtxBuffer.size()) * sizeof(ImguiVert),
        ComDataSize);
    neededDataCount += DivUp(
        static_cast<size_t>(pCmdList->IdxBuffer.size()) * (is16Bit ? 2 : 4),
        ComDataSize);
    neededDataCount += DivUp(
        static_cast<size_t>(pCmdList->CmdBuffer.size()) * sizeof(ImguiDraw),
        ComDataSize);
  }

  //-----------------------------------------------------------------------------------------
  // Allocate Data and initialize general frame information
  //-----------------------------------------------------------------------------------------
  CmdDrawFrame* pDrawFrame =
      netImguiSizedNew<CmdDrawFrame>(neededDataCount * ComDataSize);
  ComDataType* pDataOutput = reinterpret_cast<ComDataType*>(&pDrawFrame[1]);
  pDrawFrame->mMouseCursor = static_cast<uint32_t>(mouseCursor);
  pDrawFrame->mDisplayArea[0] = pDearImguiData->DisplayPos.x;
  pDrawFrame->mDisplayArea[1] = pDearImguiData->DisplayPos.y;
  pDrawFrame->mDisplayArea[2] =
      pDearImguiData->DisplayPos.x + pDearImguiData->DisplaySize.x;
  pDrawFrame->mDisplayArea[3] =
      pDearImguiData->DisplayPos.y + pDearImguiData->DisplaySize.y;
  pDrawFrame->mDrawGroupCount =
      static_cast<uint32_t>(pDearImguiData->CmdListsCount);
  SetAndIncreaseDataPointer(pDrawFrame->mpDrawGroups,
                            static_cast<uint32_t>(pDrawFrame->mDrawGroupCount *
                                                  sizeof(ImguiDrawGroup)),
                            pDataOutput);

  //-----------------------------------------------------------------------------------------
  // Copy draw data (vertices, indices, drawcall info, ...)
  //-----------------------------------------------------------------------------------------
  for (size_t n = 0; n < pDrawFrame->mDrawGroupCount; n++) {
    ImguiDrawGroup& drawGroup = pDrawFrame->mpDrawGroups[n];
    const ImDrawList* pCmdList = pDearImguiData->CmdLists[static_cast<int>(n)];
    drawGroup = ImguiDrawGroup();
    drawGroup.mGroupID = PointerCast<uint64_t>(
        pCmdList->_OwnerName);  // Use the name string pointer as a unique ID
                                // (seems to remain the same between frame)
    ImGui_ExtractIndices(*pCmdList, drawGroup, pDataOutput);
    ImGui_ExtractVertices(*pCmdList, drawGroup, pDataOutput);
    ImGui_ExtractDraws(*pCmdList, drawGroup, pDataOutput);
    pDrawFrame->mTotalVerticeCount += drawGroup.mVerticeCount;
    pDrawFrame->mTotalIndiceCount += drawGroup.mIndiceCount;
    pDrawFrame->mTotalDrawCount += drawGroup.mDrawCount;
  }

  pDrawFrame->mSize =
      static_cast<uint32_t>(pDataOutput -
                            reinterpret_cast<const ComDataType*>(pDrawFrame)) *
      ComDataSize;
  pDrawFrame->mUncompressedSize =
      pDrawFrame->mSize;  // No compression with this item, so same value
  return pDrawFrame;
}

}  // namespace Internal
}  // namespace NetImgui

#include "NetImgui_WarningReenable.h"
#endif  // #if NETIMGUI_ENABLED
