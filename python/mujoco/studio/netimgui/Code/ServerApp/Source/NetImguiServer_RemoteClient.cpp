// Google modifications:
//  - Request a new uncompressed frame when a received compressed frame is
//    invalid (DecompressCmdDrawFrame returns null).
#include "NetImguiServer_RemoteClient.h"

#include <Private/NetImgui_CmdPackets.h>

#include <algorithm>

#include "NetImguiServer_App.h"
#include "NetImguiServer_Config.h"
#include "NetImguiServer_UI.h"

namespace NetImguiServer {
namespace RemoteClient {

static Client* gpClients =
    nullptr;  // Table of all potentially connected clients to this server
static uint32_t gClientCountMax = 0;

NetImguiImDrawData::NetImguiImDrawData() : mCommandList(nullptr) {
  CmdListsCount = 1;  // All draws collapsed in same CmdList
  CmdLists.push_back(&mCommandList);
}

Client::Client()
    : mPendingTextureReadIndex(0),
      mPendingTextureWriteIndex(0),
      mbIsFree(true),
      mbCompressionSkipOncePending(false),
      mbDisconnectPending(false),
      mClientConfigID(NetImguiServer::Config::Client::kInvalidRuntimeID) {}

Client::~Client() { Uninitialize(); }

void Client::ReceiveDrawFrame(NetImgui::Internal::CmdDrawFrame* pFrameData) {
  if (pFrameData->mCompressed) {
    if (mpFrameDrawPrev != nullptr &&
        (mpFrameDrawPrev->mFrameIndex + 1) == pFrameData->mFrameIndex) {
      NetImgui::Internal::CmdDrawFrame* pUncompressedFrame =
          NetImgui::Internal::DecompressCmdDrawFrame(mpFrameDrawPrev,
                                                     pFrameData);
      netImguiDeleteSafe(pFrameData);
      pFrameData = pUncompressedFrame;
      if (pFrameData == nullptr) {
        mbCompressionSkipOncePending = true;  // Invalid data, get a new frame
      }
    }
    // Missing previous frame data
    // ignore this drawframe and request a new uncompressed one to be able to
    // resume display
    else {
      mbCompressionSkipOncePending = true;
      netImguiDeleteSafe(pFrameData);
    }
  }

  netImguiDeleteSafe(mpFrameDrawPrev);
  if (pFrameData) {
    // Convert DrawFrame command to Dear Imgui DrawData,
    // and make it available for main thread to use in rendering
    ProcessCmdDrawFrame(pFrameData);

    // Update framerate
    constexpr float kHysteresis = 0.025f;  // Between 0 to 1.0
    auto elapsedTime = std::chrono::steady_clock::now() - mLastDrawFrame;
    float elapsedMs =
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime)
                .count()) /
        1000.f;
    mStatsDrawElapsedMs =
        mStatsDrawElapsedMs * (1.f - kHysteresis) + elapsedMs * kHysteresis;
    mLastDrawFrame = std::chrono::steady_clock::now();
  }
}

void Client::ReceiveTexture(NetImgui::Internal::CmdTexture* pTextureCmd) {
  if (pTextureCmd) {
    // Wait for a free spot in the ring buffer
    while (mPendingTextureWriteIndex - mPendingTextureReadIndex >=
           IM_ARRAYSIZE(mpPendingTextures)) {
      std::this_thread::yield();
    }

    uint64_t writeIndex =
        mPendingTextureWriteIndex % IM_ARRAYSIZE(mpPendingTextures);
    mpPendingTextures[writeIndex] = pTextureCmd;
    mPendingTextureWriteIndex = mPendingTextureWriteIndex + 1;
  }
}

//=================================================================================================
// Process pending texture commands received from Client
//=================================================================================================
void Client::ProcessPendingTextureCmds() {
  while (mPendingTextureReadIndex != mPendingTextureWriteIndex) {
    uint64_t readIndex =
        mPendingTextureReadIndex % IM_ARRAYSIZE(mpPendingTextures);
    NetImgui::Internal::CmdTexture* pTextureCmd = mpPendingTextures[readIndex];
    auto texIt = mTextureTable.find(pTextureCmd->mTextureClientID);
    NetImguiServer::App::ServerTexture* serverTex =
        texIt != mTextureTable.end() ? texIt->second : nullptr;

    bool isCreate =
        pTextureCmd->mStatus == NetImgui::Internal::CmdTexture::eType::Create &&
        pTextureCmd->mFormat != NetImgui::eTexFormat::kTexFmt_Invalid;
    bool isUpdate =
        pTextureCmd->mStatus == NetImgui::Internal::CmdTexture::eType::Update &&
        pTextureCmd->mFormat != NetImgui::eTexFormat::kTexFmtCustom;
    uint32_t texDataSize =
        pTextureCmd->mSize - sizeof(NetImgui::Internal::CmdTexture);

    // Delete a texture on request or when creating new one with same
    // ClientTextureID
    if (!isUpdate && serverTex) {
      serverTex->MarkForDelete();
      mTextureTable.erase(texIt);
    }

    // Add a texture
    if (isCreate) {
      serverTex = NetImguiServer::App::CreateTexture(*pTextureCmd, texDataSize);
      if (serverTex) {
        serverTex->mOwnerClientIndex = static_cast<int32_t>(mClientIndex);
        mTextureTable.insert({pTextureCmd->mTextureClientID, serverTex});
      }
    }
    // Update a Texture
    else if (isUpdate && serverTex &&
             serverTex->mTexData.Status !=
                 ImTextureStatus::ImTextureStatus_WantDestroy &&
             serverTex->mTexData.Status !=
                 ImTextureStatus::ImTextureStatus_Destroyed) {
      auto TexFormat = static_cast<NetImgui::eTexFormat>(pTextureCmd->mFormat);
      if (serverTex->mTexData.Width <
              (int)(pTextureCmd->mWidth + pTextureCmd->mOffsetX) ||
          serverTex->mTexData.Height <
              (int)(pTextureCmd->mHeight + pTextureCmd->mOffsetY)) {
        // Update bigger than texture should not happen, but left here as a
        // precaution to avoid memory corruption. Could happen if there's an
        // error with client/server re-using a texture ID somehow
      } else {
        size_t SrcLineBytes = NetImgui::GetTexture_BytePerLine(
            TexFormat, static_cast<uint32_t>(pTextureCmd->mWidth));
        size_t DstLineBytes = NetImgui::GetTexture_BytePerLine(
            TexFormat, static_cast<uint32_t>(serverTex->mTexData.Width));
        size_t OffsetBytes =
            (pTextureCmd->mOffsetY * DstLineBytes) +
            NetImgui::GetTexture_BytePerLine(TexFormat, pTextureCmd->mOffsetX);
        const uint8_t* pDataSrc = pTextureCmd->mpTextureData.Get();
        uint8_t* pDataDst =
            reinterpret_cast<uint8_t*>(serverTex->mTexData.GetPixels());
        for (uint64_t y(0); y < pTextureCmd->mHeight; ++y) {
          memcpy(&pDataDst[OffsetBytes + (y * DstLineBytes)],
                 &pDataSrc[y * SrcLineBytes], SrcLineBytes);
        }

        // No need to queue if status is _WantCreate
        if (serverTex->mTexData.Status != ImTextureStatus_WantCreate) {
          // Following code mostly copied from
          // ImFontAtlasTextureBlockQueueUpload
          ImTextureData* tex = &serverTex->mTexData;
          ImTextureRect req = {(unsigned short)pTextureCmd->mOffsetX,
                               (unsigned short)pTextureCmd->mOffsetY,
                               (unsigned short)pTextureCmd->mWidth,
                               (unsigned short)pTextureCmd->mHeight};
          int new_x1 = ImMax(tex->UpdateRect.w == 0
                                 ? 0
                                 : tex->UpdateRect.x + tex->UpdateRect.w,
                             req.x + req.w);
          int new_y1 = ImMax(tex->UpdateRect.h == 0
                                 ? 0
                                 : tex->UpdateRect.y + tex->UpdateRect.h,
                             req.y + req.h);
          tex->UpdateRect.x = ImMin(tex->UpdateRect.x, req.x);
          tex->UpdateRect.y = ImMin(tex->UpdateRect.y, req.y);
          tex->UpdateRect.w = (unsigned short)(new_x1 - tex->UpdateRect.x);
          tex->UpdateRect.h = (unsigned short)(new_y1 - tex->UpdateRect.y);
          tex->UsedRect.x = ImMin(tex->UsedRect.x, req.x);
          tex->UsedRect.y = ImMin(tex->UsedRect.y, req.y);
          tex->UsedRect.w =
              (unsigned short)(ImMax(tex->UsedRect.x + tex->UsedRect.w,
                                     req.x + req.w) -
                               tex->UsedRect.x);
          tex->UsedRect.h =
              (unsigned short)(ImMax(tex->UsedRect.y + tex->UsedRect.h,
                                     req.y + req.h) -
                               tex->UsedRect.y);
          tex->Status = ImTextureStatus_WantUpdates;
          tex->Updates.push_back(req);
        }
      }
    }
    NetImgui::Internal::netImguiDeleteSafe(pTextureCmd);
    mPendingTextureReadIndex = mPendingTextureReadIndex + 1;
  }
}

void Client::Initialize() {
  mConnectedTime = std::chrono::steady_clock::now();
  mLastUpdateTime = std::chrono::steady_clock::now() - std::chrono::hours(1);
  mLastDrawFrame = std::chrono::steady_clock::now();
  mLastIncomingComTime = std::chrono::steady_clock::now();
  mLastDrawFrameIndex = 0;
  mStatsIndex = 0;
  mStatsRcvdBps = 0;
  mStatsSentBps = 0;
  mStatsDrawElapsedMs = 0.f;
  mStatsDataRcvd = 0;
  mStatsDataSent = 0;
  mStatsDataRcvdPrev = 0;
  mStatsDataSentPrev = 0;
  mbIsReleased = false;
  mStatsTime = std::chrono::steady_clock::now();
  mBGSettings =
      NetImgui::Internal::CmdBackground();  // Assign background default value,
                                            // until we receive first update
                                            // from client
  mPendingRcv = NetImgui::Internal::PendingCom();
  mPendingSend = NetImgui::Internal::PendingCom();
  NetImgui::Internal::netImguiDeleteSafe(mpImguiDrawData);
  NetImgui::Internal::netImguiDeleteSafe(mpFrameDrawPrev);
}

void Client::Uninitialize() {
  NetImguiServer::App::HAL_DestroyRenderTarget(mpHAL_AreaRT, mHAL_AreaTexture);

  for (auto serverTexIt : mTextureTable) {
    if (serverTexIt.second) {
      serverTexIt.second->MarkForDelete();
    }
  }
  mTextureTable.clear();

  mPendingImguiDrawDataIn.Free();
  mPendingBackgroundIn.Free();
  mPendingInputOut.Free();
  mPendingClipboardOut.Free();

  NetImgui::Internal::netImguiDeleteSafe(mpImguiDrawData);
  NetImgui::Internal::netImguiDeleteSafe(mpFrameDrawPrev);
  if (mpBGContext) {
    ImGui::DestroyContext(mpBGContext);
    mpBGContext = nullptr;
  }

  mInfoName[0] = 0;
  mClientIndex = 0;
  mClientConfigID = NetImguiServer::Config::Client::kInvalidRuntimeID;
  mbCompressionSkipOncePending = false;
  mbDisconnectPending = false;
  mbIsConnected = false;
  mbIsFree = true;
  mBGNeedUpdate = true;
}

// Used on communication thread to let main thread know this client resources
// should be deleted
void Client::Release() { mbIsReleased = true; }

bool Client::Startup(uint32_t clientCountMax) {
  gClientCountMax = clientCountMax;
  gpClients = new Client[clientCountMax];
  return gpClients != nullptr;
}

void Client::Shutdown() {
  gClientCountMax = 0;
  if (gpClients) {
    delete[] gpClients;
    gpClients = nullptr;
  }
}

uint32_t Client::GetCountMax() { return gClientCountMax; }

Client& Client::Get(uint32_t index) {
  bool bValid = gpClients && index < gClientCountMax;
  static Client sInvalidClient;
  assert(bValid);
  return bValid ? gpClients[index] : sInvalidClient;
}

uint32_t Client::GetFreeIndex() {
  for (uint32_t i(0); i < gClientCountMax; ++i) {
    if (gpClients[i].mbIsFree.exchange(false) == true) return i;
  }
  return kInvalidClient;
}

//=================================================================================================
// Get the current Dear Imgui drawdata to use for this client rendering content
//=================================================================================================
NetImguiImDrawData* Client::GetImguiDrawData(ImTextureID EmtpyTextureID) {
  // DrawData's textures should now have been created, safe to use it
  if (mpPendingDrawData) {
    NetImgui::Internal::netImguiDeleteSafe(mpImguiDrawData);
    mpImguiDrawData = mpPendingDrawData;
    mLastDrawFrameIndex = mpImguiDrawData->mFrameIndex;
    mpPendingDrawData = nullptr;
  }

  // Check if a new frame has been added. If yes, then take ownership of it.
  NetImguiImDrawData* pPendingDrawData = mPendingImguiDrawDataIn.Release();
  if (pPendingDrawData) {
    bool bHasPendingTextureUpdate(false);

    // When a new drawdata is available, need to convert the textureid from
    // NetImgui Id to the backend renderer format (texture view pointer). Done
    // here (in main thread) instead of when first received on the (com thread),
    // since 'mvTextures' can only be safely accessed on (main thread).
    for (int i(0); i < pPendingDrawData->CmdListsCount; ++i) {
      ImDrawList* pCmdList = pPendingDrawData->CmdLists[i];
      for (int drawIdx(0), drawCount(pCmdList->CmdBuffer.size());
           drawIdx < drawCount; ++drawIdx) {
        uint64_t clientTexUserID = pCmdList->CmdBuffer[drawIdx].TexRef._TexID;
        auto texIt = mTextureTable.find(clientTexUserID);
        ImTextureRef serverTexRef = EmtpyTextureID;
        if (texIt != mTextureTable.end() && texIt->second) {
          ImTextureData* texData = &texIt->second->mTexData;
          serverTexRef = texData->GetTexRef();
          texIt->second->mLastFrameUsed =
              pPendingDrawData->mFrameIndex;  // Needed to know when it is safe
                                              // to release the texture resource
          bHasPendingTextureUpdate |=
              texData->Status != ImTextureStatus::ImTextureStatus_OK;
        }
        pCmdList->CmdBuffer[drawIdx].TexRef = serverTexRef;
      }
    }

    // DrawData contains textures with pending updates,
    // wait 1 frame to display it
    if (bHasPendingTextureUpdate) {
      mpPendingDrawData = pPendingDrawData;
    }
    // New valid DrawData, use it for display
    else {
      NetImgui::Internal::netImguiDeleteSafe(mpImguiDrawData);
      mpImguiDrawData = pPendingDrawData;
      mLastDrawFrameIndex = mpImguiDrawData->mFrameIndex;
    }
  }

  return mpImguiDrawData;
}

//=================================================================================================
// Create a new Dear Imgui DrawData ready to be submitted for rendering
//=================================================================================================
void Client::ProcessCmdDrawFrame(
    NetImgui::Internal::CmdDrawFrame* pCmdDrawFrame) {
  constexpr float kPosRangeMin =
      static_cast<float>(NetImgui::Internal::ImguiVert::kPosRange_Min);
  constexpr float kPosRangeMax =
      static_cast<float>(NetImgui::Internal::ImguiVert::kPosRange_Max);
  constexpr float kUVRangeMin =
      static_cast<float>(NetImgui::Internal::ImguiVert::kUvRange_Min);
  constexpr float kUVRangeMax =
      static_cast<float>(NetImgui::Internal::ImguiVert::kUvRange_Max);

  if (!pCmdDrawFrame) {
    return;
  }
  mMouseCursor = static_cast<ImGuiMouseCursor>(pCmdDrawFrame->mMouseCursor);

  NetImguiImDrawData* pDrawData =
      NetImgui::Internal::netImguiNew<NetImguiImDrawData>();
  pDrawData->mFrameIndex = pCmdDrawFrame->mFrameIndex;
  pDrawData->Valid = true;
  pDrawData->TotalVtxCount =
      static_cast<int>(pCmdDrawFrame->mTotalVerticeCount);
  pDrawData->TotalIdxCount = static_cast<int>(pCmdDrawFrame->mTotalIndiceCount);
  pDrawData->DisplayPos.x = pCmdDrawFrame->mDisplayArea[0];
  pDrawData->DisplayPos.y = pCmdDrawFrame->mDisplayArea[1];
  pDrawData->DisplaySize.x =
      pCmdDrawFrame->mDisplayArea[2] - pCmdDrawFrame->mDisplayArea[0];
  pDrawData->DisplaySize.y =
      pCmdDrawFrame->mDisplayArea[3] - pCmdDrawFrame->mDisplayArea[1];
  pDrawData->FramebufferScale =
      ImVec2(1, 1);  //! @sammyfreg Currently untested, so force set to 1
  pDrawData->OwnerViewport = nullptr;

  ImDrawList* pCmdList = pDrawData->CmdLists[0];
  pCmdList->IdxBuffer.resize(pCmdDrawFrame->mTotalIndiceCount);
  pCmdList->VtxBuffer.resize(pCmdDrawFrame->mTotalVerticeCount);
  pCmdList->CmdBuffer.resize(pCmdDrawFrame->mTotalDrawCount);
  pCmdList->Flags =
      ImDrawListFlags_AllowVtxOffset | ImDrawListFlags_AntiAliasedLines |
      ImDrawListFlags_AntiAliasedFill | ImDrawListFlags_AntiAliasedLinesUseTex;

  if (pCmdDrawFrame->mTotalDrawCount != 0) {
    uint32_t indexOffset(0), vertexOffset(0);
    ImDrawIdx* pIndexDst = &pCmdList->IdxBuffer[0];
    ImDrawVert* pVertexDst = &pCmdList->VtxBuffer[0];
    ImDrawCmd* pCommandDst = &pCmdList->CmdBuffer[0];

    for (uint32_t i(0); i < pCmdDrawFrame->mDrawGroupCount; ++i) {
      const NetImgui::Internal::ImguiDrawGroup& drawGroup =
          pCmdDrawFrame->mpDrawGroups[i];

      // Copy/Convert Indices from network command to Dear ImGui indices format
      const uint16_t* pIndices =
          reinterpret_cast<const uint16_t*>(drawGroup.mpIndices.Get());
      if (drawGroup.mBytePerIndex == sizeof(ImDrawIdx)) {
        memcpy(pIndexDst, pIndices, drawGroup.mIndiceCount * sizeof(ImDrawIdx));
      } else {
        for (uint32_t indexIdx(0); indexIdx < drawGroup.mIndiceCount;
             ++indexIdx) {
          pIndexDst[indexIdx] = static_cast<ImDrawIdx>(pIndices[indexIdx]);
        }
      }

      // Convert the Vertices from network command to Dear Imgui Format
      const NetImgui::Internal::ImguiVert* pVertexSrc =
          drawGroup.mpVertices.Get();
      for (uint32_t vtxIdx(0); vtxIdx < drawGroup.mVerticeCount; ++vtxIdx) {
        pVertexDst[vtxIdx].pos.x =
            (static_cast<float>(pVertexSrc[vtxIdx].mPos[0]) *
             (kPosRangeMax - kPosRangeMin)) /
                static_cast<float>(0xFFFF) +
            kPosRangeMin + drawGroup.mReferenceCoord[0];
        pVertexDst[vtxIdx].pos.y =
            (static_cast<float>(pVertexSrc[vtxIdx].mPos[1]) *
             (kPosRangeMax - kPosRangeMin)) /
                static_cast<float>(0xFFFF) +
            kPosRangeMin + drawGroup.mReferenceCoord[1];
        pVertexDst[vtxIdx].uv.x =
            (static_cast<float>(pVertexSrc[vtxIdx].mUV[0]) *
             (kUVRangeMax - kUVRangeMin)) /
                static_cast<float>(0xFFFF) +
            kUVRangeMin;
        pVertexDst[vtxIdx].uv.y =
            (static_cast<float>(pVertexSrc[vtxIdx].mUV[1]) *
             (kUVRangeMax - kUVRangeMin)) /
                static_cast<float>(0xFFFF) +
            kUVRangeMin;
        pVertexDst[vtxIdx].col = pVertexSrc[vtxIdx].mColor;
      }

      // Convert the Draws from network command to Dear Imgui Format
      const NetImgui::Internal::ImguiDraw* pDrawSrc = drawGroup.mpDraws.Get();
      for (uint32_t drawIdx(0); drawIdx < drawGroup.mDrawCount; ++drawIdx) {
        pCommandDst[drawIdx].ClipRect.x = pDrawSrc[drawIdx].mClipRect[0];
        pCommandDst[drawIdx].ClipRect.y = pDrawSrc[drawIdx].mClipRect[1];
        pCommandDst[drawIdx].ClipRect.z = pDrawSrc[drawIdx].mClipRect[2];
        pCommandDst[drawIdx].ClipRect.w = pDrawSrc[drawIdx].mClipRect[3];
        pCommandDst[drawIdx].VtxOffset =
            pDrawSrc[drawIdx].mVtxOffset + vertexOffset;
        pCommandDst[drawIdx].IdxOffset =
            pDrawSrc[drawIdx].mIdxOffset + indexOffset;
        pCommandDst[drawIdx].ElemCount = pDrawSrc[drawIdx].mIdxCount;
        pCommandDst[drawIdx].UserCallback = nullptr;
        pCommandDst[drawIdx].UserCallbackData = nullptr;
        pCommandDst[drawIdx].TexRef._TexID =
            NetImgui::Internal::ConvertFromClientTexID(
                pDrawSrc[drawIdx].mClientTexId);
      }

      pIndexDst += drawGroup.mIndiceCount;
      pVertexDst += drawGroup.mVerticeCount;
      pCommandDst += drawGroup.mDrawCount;
      indexOffset += drawGroup.mIndiceCount;
      vertexOffset += drawGroup.mVerticeCount;
    }
  }
  mpFrameDrawPrev = pCmdDrawFrame;
  mPendingImguiDrawDataIn.Assign(pDrawData);
}

//=================================================================================================
// Note: Caller must take ownership of item and delete the object
//=================================================================================================
NetImgui::Internal::CmdInput* Client::TakePendingInput() {
  return mPendingInputOut.Release();
}

NetImgui::Internal::CmdClipboard* Client::TakePendingClipboard() {
  return mPendingClipboardOut.Release();
}

//=================================================================================================
// Capture current received Dear ImGui input, and forward it to the active
// client Note:	Even if a client is not focused, we are still sending it the
// mouse position,
//			so it can update its UI.
// Note:	Sending an input command, will trigger a redraw on the client,
//			which we receive on the server afterward
//=================================================================================================
void Client::CaptureImguiInput() {
  // Capture input from Dear ImGui (when this Client is in focus)
  const ImGuiIO& io = ImGui::GetIO();
  if (mbIsVisible) {
    if (ImGui::IsWindowFocused()) {
      const size_t initialSize = mPendingInputChars.size();
      const size_t addedChar = io.InputQueueCharacters.size();
      if (addedChar) {
        mPendingInputChars.resize(initialSize + addedChar);
        memcpy(&mPendingInputChars[initialSize], io.InputQueueCharacters.Data,
               addedChar * sizeof(ImWchar));
      }

      mMouseWheelPos[0] += io.MouseWheel;
      mMouseWheelPos[1] += io.MouseWheelH;
    }

    // Update persistent mouse status
    if (ImGui::IsMousePosValid(&io.MousePos)) {
      mMousePos[0] = io.MousePos.x - ImGui::GetCursorScreenPos().x;
      mMousePos[1] = io.MousePos.y - ImGui::GetCursorScreenPos().y;
    }
  }

  // This method is tied to the Server VSync setting, which might not match our
  // client desired refresh setting When client refresh drops too much, take
  // into consideration the lenght of the Server frame, to evaluate if we should
  // update or not
  bool wasActive = mbIsActive;
  mbIsActive = ImGui::IsWindowFocused();
  float clientFPS = !mbIsVisible ? 0.f
                    : !mbIsActive
                        ? NetImguiServer::Config::Server::sRefreshFPSInactive
                        : NetImguiServer::Config::Server::sRefreshFPSActive;
  float elapsedMs =
      static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - mLastUpdateTime)
                             .count()) /
      1000.f;
  bool bRefresh = (wasActive != mbIsActive) || elapsedMs > 1000.f / 60.f;

  if (!bRefresh) {
    return;
  }

  // Try to re-acquire unsent input command, or create a new one if none pending
  NetImgui::Internal::CmdInput* pNewInput = TakePendingInput();
  pNewInput =
      pNewInput
          ? pNewInput
          : NetImgui::Internal::netImguiNew<NetImgui::Internal::CmdInput>();

  // Create new Input command to send to client
  NetImguiServer::Config::Client config;
  NetImguiServer::Config::Client::GetConfigByID(mClientConfigID, config);
  pNewInput->mScreenSize[0] =
      static_cast<uint16_t>(ImGui::GetContentRegionAvail().x);
  pNewInput->mScreenSize[1] =
      static_cast<uint16_t>(ImGui::GetContentRegionAvail().y);
  pNewInput->mMousePos[0] = static_cast<int16_t>(mMousePos[0]);
  pNewInput->mMousePos[1] = static_cast<int16_t>(mMousePos[1]);
  pNewInput->mMouseWheelVert = mMouseWheelPos[0];
  pNewInput->mMouseWheelHoriz = mMouseWheelPos[1];
  pNewInput->mCompressionUse =
      NetImguiServer::Config::Server::sCompressionEnable;
  pNewInput->mCompressionSkip = mbCompressionSkipOncePending;
  pNewInput->mFontDPIScaling = 1.f;
  pNewInput->mDesiredFps = clientFPS;

  if (config.mDPIScaleEnabled) {
    float scale = ImGui::GetMainViewport()->DpiScale;
    scale = scale > 1.f ? scale : 1.f;
    pNewInput->mFontDPIScaling =
        1.f + (scale - 1.f) * NetImguiServer::Config::Server::sDPIScaleRatio;
  }

  mbCompressionSkipOncePending = false;

  if ((mbIsVisible && mbIsActive) && ImGui::IsWindowFocused()) {
    // Mouse Buttons Inputs
    // If Dear ImGui Update this enum, must also adjust our enum copy
    static_assert(
        static_cast<int>(NetImgui::Internal::CmdInput::NetImguiMouseButton::
                             ImGuiMouseButton_COUNT) ==
            static_cast<int>(ImGuiMouseButton_::ImGuiMouseButton_COUNT),
        "Update the NetImgui enum to match the updated Dear ImGui enum");
    pNewInput->mMouseDownMask = 0;
    pNewInput->mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Left)
            ? 1 << NetImgui::Internal::CmdInput::ImGuiMouseButton_Left
            : 0;
    pNewInput->mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Right)
            ? 1 << NetImgui::Internal::CmdInput::ImGuiMouseButton_Right
            : 0;
    pNewInput->mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Middle)
            ? 1 << NetImgui::Internal::CmdInput::ImGuiMouseButton_Middle
            : 0;
    pNewInput->mMouseDownMask |=
        ImGui::IsMouseDown(3)
            ? 1 << NetImgui::Internal::CmdInput::ImGuiMouseButton_Extra1
            : 0;
    pNewInput->mMouseDownMask |=
        ImGui::IsMouseDown(4)
            ? 1 << NetImgui::Internal::CmdInput::ImGuiMouseButton_Extra2
            : 0;

// Keyboard / Gamepads Inputs
// If Dear ImGui Update their enum, must also adjust our enum copy,
// so adding a few check to detect a change
#define EnumKeynameTest(KEYNAME)                                               \
  static_cast<int>(NetImgui::Internal::CmdInput::NetImguiKeys::KEYNAME) ==     \
      static_cast<int>(ImGuiKey::KEYNAME - ImGuiKey::ImGuiKey_NamedKey_BEGIN), \
      "Update the NetImgui enum to match the updated Dear ImGui enum"
    static_assert(
        NetImgui::Internal::CmdInput::NetImguiKeys::ImGuiKey_COUNT ==
            (ImGuiKey_NamedKey_END - ImGuiKey_NamedKey_BEGIN),
        "Update the NetImgui enum to match the updated Dear ImGui enum");
    static_assert(EnumKeynameTest(ImGuiKey_Tab));
    static_assert(EnumKeynameTest(ImGuiKey_Escape));
    static_assert(EnumKeynameTest(ImGuiKey_RightSuper));
    static_assert(EnumKeynameTest(ImGuiKey_Apostrophe));
    static_assert(EnumKeynameTest(ImGuiKey_Keypad0));
    static_assert(EnumKeynameTest(ImGuiKey_CapsLock));
    static_assert(EnumKeynameTest(ImGuiKey_GamepadStart));
    static_assert(EnumKeynameTest(ImGuiKey_GamepadLStickUp));
    static_assert(EnumKeynameTest(ImGuiKey_ReservedForModCtrl));
    static_assert(EnumKeynameTest(ImGuiKey_ReservedForModShift));
    static_assert(EnumKeynameTest(ImGuiKey_ReservedForModAlt));
    static_assert(EnumKeynameTest(ImGuiKey_ReservedForModSuper));
    static_assert(EnumKeynameTest(ImGuiKey_GamepadStart));
    static_assert(EnumKeynameTest(ImGuiKey_GamepadR3));
    static_assert(EnumKeynameTest(ImGuiKey_GamepadLStickUp));
    static_assert(EnumKeynameTest(ImGuiKey_GamepadRStickRight));

    // Save every keydown status to out bitmask
    uint64_t valueMask(0);
    for (uint32_t i(0); i < ImGuiKey::ImGuiKey_NamedKey_COUNT; ++i) {
      valueMask |=
          ImGui::IsKeyDown(static_cast<ImGuiKey>(ImGuiKey_NamedKey_BEGIN + i))
              ? 0x0000000000000001ull << (i % 64)
              : 0;
      if (((i % 64) == 63) || i == (ImGuiKey::ImGuiKey_NamedKey_COUNT - 1)) {
        pNewInput->mInputDownMask[i / 64] = valueMask;
        valueMask = 0;
      }
    }
    // Save analog keys (gamepad)
    for (uint32_t i(0); i < NetImgui::Internal::CmdInput::kAnalog_Count; ++i) {
      pNewInput->mInputAnalog[i] =
          ImGui::GetIO()
              .KeysData[NetImgui::Internal::CmdInput::kAnalog_First + i]
              .AnalogValue;
    }
  }

  // Copy waiting characters inputs
  size_t addedKeyCount =
      std::min<size_t>(NetImgui::Internal::ArrayCount(pNewInput->mKeyChars) -
                           pNewInput->mKeyCharCount,
                       mPendingInputChars.size());
  if (addedKeyCount) {
    memcpy(&pNewInput->mKeyChars[pNewInput->mKeyCharCount],
           &mPendingInputChars[0], addedKeyCount * sizeof(ImWchar));
    pNewInput->mKeyCharCount += static_cast<uint16_t>(addedKeyCount);
    size_t charRemainCount = mPendingInputChars.size() - addedKeyCount;
    if (charRemainCount > 0) {
      memcpy(&mPendingInputChars[0], &mPendingInputChars[addedKeyCount],
             mPendingInputChars.size() - addedKeyCount);
    }
    mPendingInputChars.resize(charRemainCount);
  }

  mPendingInputOut.Assign(pNewInput);
  mLastUpdateTime = std::chrono::steady_clock::now();
}

}  // namespace RemoteClient
}  // namespace NetImguiServer
