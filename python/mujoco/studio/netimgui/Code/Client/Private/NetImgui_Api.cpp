// Google modifications:
//  - Adding missing headers.
//  - Added logging for network events.
//  - Use atomic loads/stores for shared ClientInfo fields.
#include "NetImgui_Shared.h"
#include "NetImgui_WarningDisable.h"

#if NETIMGUI_ENABLED
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <algorithm>
#include <thread>

#include "NetImgui_Client.h"
#include "NetImgui_CmdPackets.h"
#include "NetImgui_Network.h"
#include "google/logging.h"

using namespace NetImgui::Internal;

namespace NetImgui {

static Client::ClientInfo* gpClientInfo = nullptr;

bool ProcessInputData(Client::ClientInfo& client);

//=================================================================================================
void DefaultStartCommunicationThread(void ComFunctPtr(void*), void* pClient)
//=================================================================================================
{
// Visual Studio 2017 generate this warning on std::thread, avoid the warning
// preventing build
#if defined(_MSC_VER) && (_MSC_VER < 1920)
#pragma warning(push)
#pragma warning(disable \
                : 4625)  // 'std::_LaunchPad<_Target>' : copy constructor was
                         // implicitly defined as deleted
#pragma warning(disable : 4626)  // 'std::_LaunchPad<_Target>' : assignment
                                 // operator was implicitly defined as deleted
#endif

  std::thread(ComFunctPtr, pClient).detach();

#if defined(_MSC_VER) && (_MSC_VER < 1920)
#pragma warning(pop)
#endif
}

//=================================================================================================
bool ConnectToApp(const char* clientName, const char* ServerHost,
                  uint32_t serverPort, ThreadFunctPtr threadFunction,
                  FontCreateFuncPtr FontCreateFunction)
//=================================================================================================
{
  if (!gpClientInfo) return false;

  Client::ClientInfo& client = *gpClientInfo;
  Disconnect();

  while (client.IsActive()) std::this_thread::yield();

  client.ContextRestore();      // Restore context setting override, after a
                                // disconnect
  client.ContextRemoveHooks();  // Remove hooks callback only when completely
                                // disconnected

  StringCopy(
      client.mName,
      (clientName == nullptr || clientName[0] == 0 ? "Unnamed" : clientName));
  client.mpSocketPending = Network::Connect(ServerHost, serverPort);
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
  IM_UNUSED(FontCreateFunction);
#else
  client.mFontCreationFunction = FontCreateFunction;
#endif
  if (client.mpSocketPending.load() != nullptr) {
    client.ContextInitialize();
    threadFunction = threadFunction == nullptr ? DefaultStartCommunicationThread
                                               : threadFunction;
    threadFunction(Client::CommunicationsConnect, &client);
  }

  return client.IsActive();
}

//=================================================================================================
bool ConnectFromApp(const char* clientName, uint32_t serverPort,
                    ThreadFunctPtr threadFunction,
                    FontCreateFuncPtr FontCreateFunction)
//=================================================================================================
{
  if (!gpClientInfo) return false;

  Client::ClientInfo& client = *gpClientInfo;
  Disconnect();

  while (client.IsActive()) std::this_thread::yield();

  client.ContextRestore();      // Restore context setting override, after a
                                // disconnect
  client.ContextRemoveHooks();  // Remove hooks callback only when completly
                                // disconnected

  StringCopy(
      client.mName,
      (clientName == nullptr || clientName[0] == 0 ? "Unnamed" : clientName));
  client.mpSocketPending = Network::ListenStart(serverPort);
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
  IM_UNUSED(FontCreateFunction);
#else
  client.mFontCreationFunction = FontCreateFunction;
#endif
  client.mThreadFunction = (threadFunction == nullptr)
                               ? DefaultStartCommunicationThread
                               : threadFunction;
  if (client.mpSocketPending.load() != nullptr) {
    client.ContextInitialize();
    client.mSocketListenPort = serverPort;
    client.mThreadFunction(Client::CommunicationsHost, &client);
  }

  return client.IsActive();
}

//=================================================================================================
void Disconnect(void)
//=================================================================================================
{
  if (!gpClientInfo) return;

  // Attempt fake connection on local socket waiting for a Server connection,
  // so the blocking operation can terminate and release the communication
  // thread
  Client::ClientInfo& client = *gpClientInfo;
  client.mbDisconnectPending = true;
  client.mbDisconnectListen = true;
  if (client.mpSocketListen.load() != nullptr &&
      client.mSocketListenPort != 0) {
    Network::SocketInfo* pFakeSocket =
        Network::Connect("127.0.0.1", client.mSocketListenPort);
    client.mSocketListenPort = 0;
    client.mbDisconnectPending = true;

    if (pFakeSocket) {
      Network::Disconnect(pFakeSocket);
    }
  }

  // Wait for connection attempt to complete and fail
  while (client.mbComInitActive || client.mbClientThreadActive);

  // If fake connection to exit Listening failed, force disconnect socket
  // directly even though it might potentially cause a race condition
  Network::SocketInfo* pListenSocket = client.mpSocketListen.exchange(nullptr);
  if (pListenSocket) {
    Network::Disconnect(pListenSocket);
  }

  Network::SocketInfo* pPendingSocket =
      client.mpSocketPending.exchange(nullptr);
  if (pPendingSocket) {
    Network::Disconnect(pPendingSocket);
  }
}

//=================================================================================================
bool IsConnected(void)
//=================================================================================================
{
  if (!gpClientInfo) return false;

  Client::ClientInfo& client = *gpClientInfo;

  // If disconnected in middle of a remote frame drawing,
  // want to behave like it is still connected to finish frame properly
  return client.IsConnected() || IsDrawingRemote();
}

//=================================================================================================
bool IsConnectionPending(void)
//=================================================================================================
{
  if (!gpClientInfo) return false;

  Client::ClientInfo& client = *gpClientInfo;
  return client.IsConnectPending();
}

//=================================================================================================
bool IsDrawing(void)
//=================================================================================================
{
  if (!gpClientInfo) return false;

  Client::ClientInfo& client = *gpClientInfo;
  return client.mbIsDrawing;
}

//=================================================================================================
bool IsDrawingRemote(void)
//=================================================================================================
{
  if (!gpClientInfo) return false;

  Client::ClientInfo& client = *gpClientInfo;
  return IsDrawing() && client.mbIsRemoteDrawing;
}

//=================================================================================================
bool NewFrame(bool bSupportFrameSkip)
//=================================================================================================
{
  if (!gpClientInfo || gpClientInfo->mbIsDrawing) return false;

  Client::ClientInfo& client = *gpClientInfo;
  ScopedBool scopedInside(client.mbInsideNewEnd, true);

  // ImGui Newframe handled by remote connection settings
  if (NetImgui::IsConnected()) {
    ImGui::SetCurrentContext(client.mpContext);

    // Save current context settings and override settings to fit our netImgui
    // usage
    if (!client.IsContextOverriden()) {
      client.ContextOverride();
    }

    auto elapsedCheck =
        std::chrono::steady_clock::now() - client.mLastOutgoingDrawCheckTime;
    auto elapsedDraw =
        std::chrono::steady_clock::now() - client.mLastOutgoingDrawTime;
    auto elapsedCheckMs =
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsedCheck)
                .count()) /
        1000.f;
    auto elapsedDrawMs =
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsedDraw)
                .count()) /
        1000.f;
    client.mLastOutgoingDrawCheckTime = std::chrono::steady_clock::now();

    // Update input and see if remote netImgui expect a new frame
    client.mbValidDrawFrame = false;

    // Take into account delay until next method call, for more precise fps
    const float desiredFps = client.mDesiredFps.load();
    bool shouldDraw =
        desiredFps > 0.f &&
        (elapsedDrawMs + elapsedCheckMs / 2.f) > (1000.f / desiredFps);

    VLOG(2,
         "NewFrame: elapsedDrawMs=%.2f, elapsedCheckMs=%.2f, shouldDraw=%s, "
         "mDesiredFps=%.1f (target=%.1f)",
         elapsedDrawMs, elapsedCheckMs, shouldDraw ? "Y" : "N",
         desiredFps, 1000.f / desiredFps);

    if (shouldDraw) {
      VLOG(2, "NewFrame: shouldDraw=Y, triggering draw!");

      client.mLastOutgoingDrawTime = std::chrono::steady_clock::now();
      client.mbValidDrawFrame = true;
      client.mSavedDisplaySize = ImGui::GetIO().DisplaySize;

#if NETIMGUI_IMGUI_TEXTURES_ENABLED
      client.mFontSavedScaling = ImGui::GetStyle().FontScaleDpi;
      ImGui::GetStyle().FontScaleDpi = client.mFontServerScale;
#else
      // We are about to start drawing for remote context, check for font data
      // update
      const ImFontAtlas* pFonts = ImGui::GetIO().Fonts;
      if (pFonts->TexPixelsAlpha8 &&
          (pFonts->TexPixelsAlpha8 != client.mpFontTextureData ||
           client.mFontTextureID != ConvertToClientTexID(pFonts->TexID))) {
        uint8_t* pPixelData(nullptr);
        int width(0), height(0);
        ImGui::GetIO().Fonts->GetTexDataAsAlpha8(&pPixelData, &width, &height);
        SendDataTexture(pFonts->TexID, pPixelData, static_cast<uint16_t>(width),
                        static_cast<uint16_t>(height), eTexFormat::kTexFmtA8);
      }
      // No font texture has been sent to the netImgui server, you can either
      // 1. Leave font data available in ImGui (not call ImGui::ClearTexData)
      // for netImgui to auto send it
      // 2. Manually call 'NetImgui::SendDataTexture' with font texture data
      assert(client.mbFontUploaded);
#endif
    }

    ProcessInputData(client);

    // Update current active content with our time
    ImGui::GetIO().DeltaTime =
        std::max<float>(1.f / 1000.f, elapsedCheckMs / 1000.f);

    // NetImgui isn't waiting for a new frame, try to skip drawing when caller
    // supports it
    if (!client.mbValidDrawFrame && bSupportFrameSkip) {
      return false;
    }
  }
  // Regular Imgui NewFrame
  else {
    // Restore context setting override, after a disconnect
    client.ContextRestore();

    // Remove hooks callback only when completly disconnected
    if (!client.IsConnectPending()) {
      client.ContextRemoveHooks();
    }
  }

  // A new frame is expected, update the current time of the drawing context,
  // and let Imgui know to prepare a new drawing frame
  client.mbIsRemoteDrawing = NetImgui::IsConnected();
  client.mbIsDrawing = true;

  // Reset Dear ImGui managed Textures status if not handled by backend, to not
  // re-process the same elements (active on 1.92+)
  client.TextureTrackingClear();

  // This function can be called from a 'NewFrame' ImGui hook, we should not
  // start a new frame again
  if (!client.mbInsideHook) {
    ImGui::NewFrame();
  }

  return true;
}

//=================================================================================================
void EndFrame(void)
//=================================================================================================
{
  if (!gpClientInfo) return;

  Client::ClientInfo& client = *gpClientInfo;
  ScopedBool scopedInside(client.mbInsideNewEnd, true);

  if (client.mbIsDrawing) {
    // Must be fetched before 'Render'
    ImGuiMouseCursor Cursor = ImGui::GetMouseCursor();

    // This function can be called from a 'EndFrame' ImGui hook, in which case
    // no need to call this again
    if (!client.mbInsideHook) {
      ImGui::Render();
    }

    // Detect all DearImgui ImGui managed Textures waiting for updates before
    // backend process them (active on 1.92+)
    client.TextureTrackingUpdate();

    // Prepare the Dear Imgui DrawData for later transmission to Server
    ImDrawData* imDrawData = ImGui::GetDrawData();
    client.ProcessDrawData(imDrawData, Cursor);

    // Detect change to background settings by user, and forward them to server
    if (client.mBGSetting != client.mBGSettingSent) {
      CmdBackground* pCmdBackground = netImguiNew<CmdBackground>();
      *pCmdBackground = client.mBGSetting;
      client.mBGSettingSent = client.mBGSetting;
      client.mPendingBackgroundOut.Assign(pCmdBackground);
    }

    if (client.mbIsRemoteDrawing) {
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
      // Clear the ImGui Draw Data while leaving the texture updates intact
      // This allows the backend to process the texture updates/status normally
      // (usually done in Render backend implementation) while not displaying
      // anything.
      if (imDrawData) {
        imDrawData->CmdLists.resize(0);
        imDrawData->CmdListsCount = 0;
        imDrawData->TotalIdxCount = 0;
        imDrawData->TotalVtxCount = 0;
      }
      ImGui::GetStyle().FontScaleDpi = client.mFontSavedScaling;
#endif
      // Restore display size, so we never lose original setting
      // that may get updated after initial connection
      ImGui::GetIO().DisplaySize = client.mSavedDisplaySize;
    }
  }

  VLOG(2, "EndFrame: mbIsDrawing was %s, mbIsRemoteDrawing was %s",
       client.mbIsDrawing ? "Y" : "N", client.mbIsRemoteDrawing ? "Y" : "N");

  client.mbIsRemoteDrawing = false;
  client.mbIsDrawing = false;
  client.mbValidDrawFrame = false;
}

//=================================================================================================
ImGuiContext* GetContext()
//=================================================================================================
{
  if (!gpClientInfo) return nullptr;

  Client::ClientInfo& client = *gpClientInfo;
  return client.mpContext;
}

//=================================================================================================
void SendDataTexture(ImTextureID textureId, void* pData, uint16_t width,
                     uint16_t height, eTexFormat format, uint32_t dataSize)
//=================================================================================================
{
  if (!gpClientInfo) return;
  Client::ClientInfo& client = *gpClientInfo;
  ClientTextureID clientTexID = ConvertToClientTexID(textureId);

  // Add/Update a texture
  if (pData != nullptr) {
    CmdTexture* pCmdTexture =
        client.TextureCmdAllocate(clientTexID, width, height, format, dataSize);
    if (pCmdTexture) {
      memcpy(pCmdTexture->mpTextureData.Get(), pData, dataSize);
      pCmdTexture->mpTextureData.ToOffset();
      client.TextureTrackingAdd(*pCmdTexture);

      // Detects when user is sending the font texture
#if !NETIMGUI_IMGUI_TEXTURES_ENABLED
      ScopedImguiContext scopedCtx(
          client.mpContext ? client.mpContext : ImGui::GetCurrentContext());
      if (ImGui::GetIO().Fonts && ImGui::GetIO().Fonts->TexID == textureId) {
        client.mbFontUploaded |= true;
        client.mpFontTextureData = ImGui::GetIO().Fonts->TexPixelsAlpha8;
        client.mFontTextureID = clientTexID;
      }
#endif
    }
  }
  // Texture to remove
  else {
    client.TextureTrackingRem(clientTexID);
  }
}

#if NETIMGUI_IMGUI_TEXTURES_ENABLED
//=================================================================================================
void SendDataTexture(const ImTextureRef& textureRef, void* pData,
                     uint16_t width, uint16_t height, eTexFormat format,
                     uint32_t dataSize)
//=================================================================================================
{
  if (!gpClientInfo) return;
  Client::ClientInfo& client = *gpClientInfo;
  ClientTextureID clientTexID = ConvertToClientTexID(textureRef);

  // Add/Update a texture
  if (pData != nullptr) {
    CmdTexture* pCmdTexture =
        client.TextureCmdAllocate(clientTexID, width, height, format, dataSize);
    if (pCmdTexture) {
      memcpy(pCmdTexture->mpTextureData.Get(), pData, dataSize);
      pCmdTexture->mpTextureData.ToOffset();
      client.TextureTrackingAdd(*pCmdTexture);
    }
  }
  // Texture to remove
  else {
    client.TextureTrackingRem(clientTexID);
  }
}
#endif  // NETIMGUI_IMGUI_TEXTURES_ENABLED

//=================================================================================================
void SetBackground(const ImVec4& bgColor)
//=================================================================================================
{
  if (!gpClientInfo) return;

  Client::ClientInfo& client = *gpClientInfo;
  client.mBGSetting = NetImgui::Internal::CmdBackground();
  client.mBGSetting.mClearColor[0] = bgColor.x;
  client.mBGSetting.mClearColor[1] = bgColor.y;
  client.mBGSetting.mClearColor[2] = bgColor.z;
  client.mBGSetting.mClearColor[3] = bgColor.w;
}

//=================================================================================================
void SetBackground(const ImVec4& bgColor, const ImVec4& textureTint)
//=================================================================================================
{
  if (!gpClientInfo) return;

  Client::ClientInfo& client = *gpClientInfo;
  client.mBGSetting.mClearColor[0] = bgColor.x;
  client.mBGSetting.mClearColor[1] = bgColor.y;
  client.mBGSetting.mClearColor[2] = bgColor.z;
  client.mBGSetting.mClearColor[3] = bgColor.w;
  client.mBGSetting.mTextureTint[0] = textureTint.x;
  client.mBGSetting.mTextureTint[1] = textureTint.y;
  client.mBGSetting.mTextureTint[2] = textureTint.z;
  client.mBGSetting.mTextureTint[3] = textureTint.w;
  client.mBGSetting.mTextureId =
      NetImgui::Internal::CmdBackground::kDefaultTexture;
}

//=================================================================================================
void SetBackground(const ImVec4& bgColor, const ImVec4& textureTint,
                   ImTextureID bgTextureID)
//=================================================================================================
{
  if (!gpClientInfo) return;

  Client::ClientInfo& client = *gpClientInfo;
  client.mBGSetting.mClearColor[0] = bgColor.x;
  client.mBGSetting.mClearColor[1] = bgColor.y;
  client.mBGSetting.mClearColor[2] = bgColor.z;
  client.mBGSetting.mClearColor[3] = bgColor.w;
  client.mBGSetting.mTextureTint[0] = textureTint.x;
  client.mBGSetting.mTextureTint[1] = textureTint.y;
  client.mBGSetting.mTextureTint[2] = textureTint.z;
  client.mBGSetting.mTextureTint[3] = textureTint.w;

  uint64_t texId64(0);
  reinterpret_cast<ImTextureID*>(&texId64)[0] = bgTextureID;
  client.mBGSetting.mTextureId = texId64;
}

#if NETIMGUI_IMGUI_TEXTURES_ENABLED
//=================================================================================================
void SetBackground(const ImVec4& bgColor, const ImVec4& textureTint,
                   const ImTextureRef& bgTextureRef)
//=================================================================================================
{
  SetBackground(bgColor, textureTint, ConvertToClientTexID(bgTextureRef));
}
#endif

//=================================================================================================
void SetCompressionMode(eCompressionMode eMode)
//=================================================================================================
{
  if (!gpClientInfo) return;

  Client::ClientInfo& client = *gpClientInfo;
  client.mClientCompressionMode = static_cast<uint8_t>(eMode);
}
//=================================================================================================
eCompressionMode GetCompressionMode()
//=================================================================================================
{
  if (!gpClientInfo) return eCompressionMode::kUseServerSetting;

  Client::ClientInfo& client = *gpClientInfo;
  return static_cast<eCompressionMode>(client.mClientCompressionMode);
}

//=================================================================================================
bool Startup(void)
//=================================================================================================
{
  if (!gpClientInfo) {
    gpClientInfo = netImguiNew<Client::ClientInfo>();
  }

  return Network::Startup();
}

//=================================================================================================
void Shutdown()
//=================================================================================================
{
  if (!gpClientInfo) return;

  Disconnect();
  while (gpClientInfo->IsActive()) std::this_thread::yield();
  Network::Shutdown();

  netImguiDeleteSafe(gpClientInfo);
}

//=================================================================================================
ImGuiContext* CloneContext(ImGuiContext* pSourceContext)
//=================================================================================================
{
  // Create a context duplicate
  ScopedImguiContext scopedSourceCtx(pSourceContext);
  ImGuiContext* pContextClone = ImGui::CreateContext(ImGui::GetIO().Fonts);
  ImGuiIO& sourceIO = ImGui::GetIO();
  ImGuiStyle& sourceStyle = ImGui::GetStyle();
  {
    ScopedImguiContext scopedCloneCtx(pContextClone);
    ImGuiIO& newIO = ImGui::GetIO();
    ImGuiStyle& newStyle = ImGui::GetStyle();

    // Import the style/options settings of current context, into this one
    memcpy(&newStyle, &sourceStyle, sizeof(newStyle));
    memcpy(&newIO, &sourceIO, sizeof(newIO));
    // memcpy(newIO.KeyMap, sourceIO.KeyMap, sizeof(newIO.KeyMap));
    newIO.InputQueueCharacters.Data = nullptr;
    newIO.InputQueueCharacters.Size = 0;
    newIO.InputQueueCharacters.Capacity = 0;
  }
  return pContextClone;
}

//=================================================================================================
uint8_t GetTexture_BitsPerPixel(eTexFormat eFormat)
//=================================================================================================
{
  switch (eFormat) {
    case eTexFormat::kTexFmtA8:
      return 8 * 1;
    case eTexFormat::kTexFmtRGBA8:
      return 8 * 4;
    case eTexFormat::kTexFmtCustom:
      return 0;
    case eTexFormat::kTexFmt_Invalid:
      return 0;
  }
  return 0;
}

//=================================================================================================
uint32_t GetTexture_BytePerLine(eTexFormat eFormat, uint32_t pixelWidth)
//=================================================================================================
{
  uint32_t bitsPerPixel =
      static_cast<uint32_t>(GetTexture_BitsPerPixel(eFormat));
  return pixelWidth * bitsPerPixel / 8;
  // Note: If adding support to BC compression format, have to take into account
  // 4x4 size alignment
}

//=================================================================================================
uint32_t GetTexture_BytePerImage(eTexFormat eFormat, uint32_t pixelWidth,
                                 uint32_t pixelHeight)
//=================================================================================================
{
  return GetTexture_BytePerLine(eFormat, pixelWidth) * pixelHeight;
  // Note: If adding support to BC compression format, have to take into account
  // 4x4 size alignement
}

static inline void AddKeyEvent(const Client::ClientInfo& client,
                               const CmdInput* pCmdInput,
                               CmdInput::NetImguiKeys netimguiKey,
                               ImGuiKey imguiKey) {
  uint32_t valIndex = netimguiKey / 64;
  uint64_t valMask = 0x0000000000000001ull << (netimguiKey % 64);
#if IMGUI_VERSION_NUM < 18700
  IM_UNUSED(client);
  ImGui::GetIO().KeysDown[imguiKey] =
      (pCmdInput->mInputDownMask[valIndex] & valMask) != 0;
#else
  bool bChanged = (pCmdInput->mInputDownMask[valIndex] ^
                   client.mPreviousInputState.mInputDownMask[valIndex]) &
                  valMask;
  if (bChanged) {
    ImGui::GetIO().AddKeyEvent(imguiKey,
                               pCmdInput->mInputDownMask[valIndex] & valMask);
  }
#endif
}

static inline void AddKeyAnalogEvent(const Client::ClientInfo& client,
                                     const CmdInput* pCmdInput,
                                     CmdInput::NetImguiKeys netimguiKey,
                                     ImGuiKey imguiKey) {
  uint32_t valIndex = netimguiKey / 64;
  uint64_t valMask = 0x0000000000000001ull << (netimguiKey % 64);
  assert(CmdInput::kAnalog_First <= static_cast<uint32_t>(netimguiKey) &&
         static_cast<uint32_t>(netimguiKey) <= CmdInput::kAnalog_Last);
#if IMGUI_VERSION_NUM < 18700
  IM_UNUSED(client);
  IM_UNUSED(pCmdInput);
  IM_UNUSED(netimguiKey);
  IM_UNUSED(imguiKey);
#else
  int indexAnalog = netimguiKey - CmdInput::kAnalog_First;
  indexAnalog = indexAnalog >= static_cast<int>(CmdInput::kAnalog_Count)
                    ? CmdInput::kAnalog_Count - 1
                    : indexAnalog;
  float analogValue = pCmdInput->mInputAnalog[indexAnalog];
  bool bChanged = (pCmdInput->mInputDownMask[valIndex] ^
                   client.mPreviousInputState.mInputDownMask[valIndex]) &
                  valMask;
  bChanged |= abs(client.mPreviousInputState.mInputAnalog[indexAnalog] -
                  analogValue) > 0.001f;
  if (bChanged) {
    ImGui::GetIO().AddKeyAnalogEvent(
        imguiKey, pCmdInput->mInputDownMask[valIndex] & valMask, analogValue);
  }
#endif
}

//=================================================================================================
bool ProcessInputData(Client::ClientInfo& client)
//=================================================================================================
{
  // Update the current clipboard data received from Server
  CmdClipboard* pCmdClipboardNew = client.mPendingClipboardIn.Release();
  if (pCmdClipboardNew) {
    netImguiDeleteSafe(client.mpCmdClipboard);
    client.mpCmdClipboard = pCmdClipboardNew;
  }

  // Update the keyboard/mouse/gamepad inputs
  CmdInput* pCmdInputNew = client.mPendingInputIn.Release();
  bool hasNewInput = pCmdInputNew != nullptr;
  CmdInput* pCmdInput = hasNewInput ? pCmdInputNew : client.mpCmdInputPending;
  ImGuiIO& io = ImGui::GetIO();

  if (pCmdInput) {
    const float wheelY = pCmdInput->mMouseWheelVert -
                         client.mPreviousInputState.mMouseWheelVertPrev;
    const float wheelX = pCmdInput->mMouseWheelHoriz -
                         client.mPreviousInputState.mMouseWheelHorizPrev;
    io.DisplaySize =
        ImVec2(pCmdInput->mScreenSize[0], pCmdInput->mScreenSize[1]);
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
    client.mFontServerScale = pCmdInput->mFontDPIScaling;
#else
    // User assigned a function callback handling FontScaling,
    // use it to request a Font update on DPI scaling change on the server
    if (gpClientInfo->mFontCreationFunction != nullptr) {
      io.FontGlobalScale = 1.f;
      if (abs(gpClientInfo->mFontServerScale - pCmdInput->mFontDPIScaling) >
          0.01f) {
        gpClientInfo->mFontCreationFunction(gpClientInfo->mFontSavedScaling,
                                            pCmdInput->mFontDPIScaling);
        client.mFontServerScale = pCmdInput->mFontDPIScaling;
      }
    }
    // Client doesn't support regenerating the font at new DPI
    // Use FontGlobalScale to affect rendering size, resulting in blurrier
    // result
    else {
      io.FontGlobalScale = pCmdInput->mFontDPIScaling;
    }
#endif

#if IMGUI_VERSION_NUM < 18700
    io.MousePos = ImVec2(pCmdInput->mMousePos[0], pCmdInput->mMousePos[1]);
    io.MouseWheel = wheelY;
    io.MouseWheelH = wheelX;
    for (uint32_t i(0);
         i < CmdInput::NetImguiMouseButton::ImGuiMouseButton_COUNT; ++i) {
      io.MouseDown[i] =
          (pCmdInput->mMouseDownMask & (0x0000000000000001ull << i)) != 0;
    }

#define AddInputDown(KEYNAME) \
  AddKeyEvent(client, pCmdInput, CmdInput::KEYNAME, ImGuiKey_::KEYNAME);
    AddInputDown(ImGuiKey_Tab) AddInputDown(ImGuiKey_LeftArrow) AddInputDown(
        ImGuiKey_RightArrow) AddInputDown(ImGuiKey_UpArrow)
        AddInputDown(ImGuiKey_DownArrow) AddInputDown(ImGuiKey_PageUp)
            AddInputDown(ImGuiKey_PageDown) AddInputDown(ImGuiKey_Home)
                AddInputDown(ImGuiKey_End) AddInputDown(ImGuiKey_Insert)
                    AddInputDown(ImGuiKey_Delete) AddInputDown(
                        ImGuiKey_Backspace) AddInputDown(ImGuiKey_Space)
                        AddInputDown(ImGuiKey_Enter)
                            AddInputDown(ImGuiKey_Escape) AddInputDown(
                                ImGuiKey_A)  // for text edit CTRL+A: select all
        AddInputDown(ImGuiKey_C)             // for text edit CTRL+C: copy
        AddInputDown(ImGuiKey_V)             // for text edit CTRL+V: paste
        AddInputDown(ImGuiKey_X)             // for text edit CTRL+X: cut
        AddInputDown(ImGuiKey_Y)             // for text edit CTRL+Y: redo
        AddInputDown(ImGuiKey_Z)             // for text edit CTRL+Z: undo

        io.KeyShift = pCmdInput->IsKeyDown(
        CmdInput::NetImguiKeys::ImGuiKey_ReservedForModShift);
    io.KeyCtrl = pCmdInput->IsKeyDown(
        CmdInput::NetImguiKeys::ImGuiKey_ReservedForModCtrl);
    io.KeyAlt = pCmdInput->IsKeyDown(
        CmdInput::NetImguiKeys::ImGuiKey_ReservedForModAlt);
    io.KeySuper = pCmdInput->IsKeyDown(
        CmdInput::NetImguiKeys::ImGuiKey_ReservedForModSuper);
#else
#if IMGUI_VERSION_NUM < 18837
#define ImGuiKey ImGuiKey_
#endif
// At the moment All Dear Imgui version share the same ImGuiKey_ enum (with a
// 512 value offset), but could change in the future, so convert from our own
// enum version, to Dear ImGui.
#define AddInputDown(KEYNAME) \
  AddKeyEvent(client, pCmdInput, CmdInput::KEYNAME, ImGuiKey::KEYNAME);
#define AddAnalogInputDown(KEYNAME) \
  AddKeyAnalogEvent(client, pCmdInput, CmdInput::KEYNAME, ImGuiKey::KEYNAME);
    AddInputDown(ImGuiKey_Tab) AddInputDown(ImGuiKey_LeftArrow) AddInputDown(
        ImGuiKey_RightArrow) AddInputDown(ImGuiKey_UpArrow) AddInputDown(ImGuiKey_DownArrow)
        AddInputDown(ImGuiKey_PageUp) AddInputDown(ImGuiKey_PageDown) AddInputDown(
            ImGuiKey_Home) AddInputDown(ImGuiKey_End) AddInputDown(ImGuiKey_Insert)
            AddInputDown(ImGuiKey_Delete) AddInputDown(ImGuiKey_Backspace) AddInputDown(
                ImGuiKey_Space) AddInputDown(ImGuiKey_Enter) AddInputDown(ImGuiKey_Escape)

                AddInputDown(ImGuiKey_LeftCtrl) AddInputDown(ImGuiKey_LeftShift) AddInputDown(
                    ImGuiKey_LeftAlt) AddInputDown(ImGuiKey_LeftSuper) AddInputDown(ImGuiKey_RightCtrl)
                    AddInputDown(ImGuiKey_RightShift) AddInputDown(ImGuiKey_RightAlt) AddInputDown(
                        ImGuiKey_RightSuper) AddInputDown(ImGuiKey_Menu) AddInputDown(ImGuiKey_0)
                        AddInputDown(ImGuiKey_1) AddInputDown(ImGuiKey_2) AddInputDown(ImGuiKey_3) AddInputDown(
                            ImGuiKey_4) AddInputDown(ImGuiKey_5) AddInputDown(ImGuiKey_6)
                            AddInputDown(ImGuiKey_7) AddInputDown(ImGuiKey_8) AddInputDown(
                                ImGuiKey_9) AddInputDown(ImGuiKey_A) AddInputDown(ImGuiKey_B)
                                AddInputDown(ImGuiKey_C) AddInputDown(ImGuiKey_D) AddInputDown(
                                    ImGuiKey_E) AddInputDown(ImGuiKey_F) AddInputDown(ImGuiKey_G)
                                    AddInputDown(ImGuiKey_H) AddInputDown(ImGuiKey_I) AddInputDown(
                                        ImGuiKey_J) AddInputDown(ImGuiKey_K) AddInputDown(ImGuiKey_L)
                                        AddInputDown(ImGuiKey_M) AddInputDown(ImGuiKey_N) AddInputDown(
                                            ImGuiKey_O) AddInputDown(ImGuiKey_P) AddInputDown(ImGuiKey_Q)
                                            AddInputDown(ImGuiKey_R) AddInputDown(ImGuiKey_S) AddInputDown(
                                                ImGuiKey_T) AddInputDown(ImGuiKey_U) AddInputDown(ImGuiKey_V)
                                                AddInputDown(ImGuiKey_W) AddInputDown(ImGuiKey_X) AddInputDown(
                                                    ImGuiKey_Y) AddInputDown(ImGuiKey_Z)
                                                    AddInputDown(ImGuiKey_F1) AddInputDown(
                                                        ImGuiKey_F2) AddInputDown(ImGuiKey_F3)
                                                        AddInputDown(ImGuiKey_F4) AddInputDown(
                                                            ImGuiKey_F5) AddInputDown(ImGuiKey_F6)
                                                            AddInputDown(ImGuiKey_F7) AddInputDown(
                                                                ImGuiKey_F8) AddInputDown(ImGuiKey_F9)
                                                                AddInputDown(ImGuiKey_F10) AddInputDown(
                                                                    ImGuiKey_F11) AddInputDown(ImGuiKey_F12)

                                                                    AddInputDown(ImGuiKey_Apostrophe) AddInputDown(
                                                                        ImGuiKey_Comma) AddInputDown(ImGuiKey_Minus)
                                                                        AddInputDown(ImGuiKey_Period) AddInputDown(
                                                                            ImGuiKey_Slash) AddInputDown(ImGuiKey_Semicolon)
                                                                            AddInputDown(
                                                                                ImGuiKey_Equal)
                                                                                AddInputDown(
                                                                                    ImGuiKey_LeftBracket)
                                                                                    AddInputDown(
                                                                                        ImGuiKey_Backslash)
                                                                                        AddInputDown(ImGuiKey_RightBracket) AddInputDown(ImGuiKey_GraveAccent) AddInputDown(ImGuiKey_CapsLock) AddInputDown(ImGuiKey_ScrollLock) AddInputDown(ImGuiKey_NumLock) AddInputDown(ImGuiKey_PrintScreen) AddInputDown(ImGuiKey_Pause) AddInputDown(ImGuiKey_Keypad0) AddInputDown(ImGuiKey_Keypad1) AddInputDown(ImGuiKey_Keypad2) AddInputDown(ImGuiKey_Keypad3) AddInputDown(ImGuiKey_Keypad4) AddInputDown(ImGuiKey_Keypad5) AddInputDown(ImGuiKey_Keypad6) AddInputDown(ImGuiKey_Keypad7)
                                                                                            AddInputDown(ImGuiKey_Keypad8) AddInputDown(ImGuiKey_Keypad9) AddInputDown(ImGuiKey_KeypadDecimal) AddInputDown(ImGuiKey_KeypadDivide) AddInputDown(ImGuiKey_KeypadMultiply) AddInputDown(ImGuiKey_KeypadSubtract) AddInputDown(ImGuiKey_KeypadAdd) AddInputDown(ImGuiKey_KeypadEnter)
                                                                                                AddInputDown(
                                                                                                    ImGuiKey_KeypadEqual)

#if IMGUI_VERSION_NUM >= 19000
                                                                                                    AddInputDown(
                                                                                                        ImGuiKey_F13)
                                                                                                        AddInputDown(
                                                                                                            ImGuiKey_F14)
                                                                                                            AddInputDown(
                                                                                                                ImGuiKey_F15) AddInputDown(ImGuiKey_F16) AddInputDown(ImGuiKey_F17) AddInputDown(ImGuiKey_F18) AddInputDown(ImGuiKey_F19) AddInputDown(ImGuiKey_F20) AddInputDown(ImGuiKey_F21) AddInputDown(ImGuiKey_F22) AddInputDown(ImGuiKey_F23) AddInputDown(ImGuiKey_F24)
                                                                                                                AddInputDown(
                                                                                                                    ImGuiKey_AppBack)
                                                                                                                    AddInputDown(
                                                                                                                        ImGuiKey_AppForward)
#endif
        // Gamepad
        AddInputDown(ImGuiKey_GamepadStart) AddInputDown(
            ImGuiKey_GamepadBack) AddInputDown(ImGuiKey_GamepadFaceUp)
            AddInputDown(ImGuiKey_GamepadFaceDown) AddInputDown(
                ImGuiKey_GamepadFaceLeft) AddInputDown(ImGuiKey_GamepadFaceRight)
                AddInputDown(ImGuiKey_GamepadDpadUp) AddInputDown(
                    ImGuiKey_GamepadDpadDown) AddInputDown(ImGuiKey_GamepadDpadLeft)
                    AddInputDown(ImGuiKey_GamepadDpadRight) AddInputDown(
                        ImGuiKey_GamepadL1) AddInputDown(ImGuiKey_GamepadR1)
                        AddInputDown(ImGuiKey_GamepadL2) AddInputDown(
                            ImGuiKey_GamepadR2) AddInputDown(ImGuiKey_GamepadL3)
                            AddInputDown(ImGuiKey_GamepadR3) AddAnalogInputDown(
                                ImGuiKey_GamepadLStickUp)
                                AddAnalogInputDown(ImGuiKey_GamepadLStickDown) AddAnalogInputDown(
                                    ImGuiKey_GamepadLStickLeft)
                                    AddAnalogInputDown(
                                        ImGuiKey_GamepadLStickRight)
                                        AddAnalogInputDown(
                                            ImGuiKey_GamepadRStickUp)
                                            AddAnalogInputDown(
                                                ImGuiKey_GamepadRStickDown)
                                                AddAnalogInputDown(
                                                    ImGuiKey_GamepadRStickLeft)
                                                    AddAnalogInputDown(
                                                        ImGuiKey_GamepadRStickRight)

#undef AddInputDown
#undef AddAnalogInputDown
#if IMGUI_VERSION_NUM < 18837
#undef ImGuiKey
#endif

#if IMGUI_VERSION_NUM < 18837
#define ImGuiMod_Ctrl ImGuiKey_ModCtrl
#define ImGuiMod_Shift ImGuiKey_ModShift
#define ImGuiMod_Alt ImGuiKey_ModAlt
#define ImGuiMod_Super ImGuiKey_ModSuper
#endif
                                                        io.AddKeyEvent(
                                                            ImGuiMod_Ctrl,
                                                            pCmdInput->IsKeyDown(
                                                                CmdInput::NetImguiKeys::
                                                                    ImGuiKey_ReservedForModCtrl));
    io.AddKeyEvent(ImGuiMod_Shift,
                   pCmdInput->IsKeyDown(
                       CmdInput::NetImguiKeys::ImGuiKey_ReservedForModShift));
    io.AddKeyEvent(ImGuiMod_Alt,
                   pCmdInput->IsKeyDown(
                       CmdInput::NetImguiKeys::ImGuiKey_ReservedForModAlt));
    io.AddKeyEvent(ImGuiMod_Super,
                   pCmdInput->IsKeyDown(
                       CmdInput::NetImguiKeys::ImGuiKey_ReservedForModSuper));

    // Mouse
    io.AddMouseWheelEvent(wheelX, wheelY);
    io.AddMousePosEvent(pCmdInput->mMousePos[0], pCmdInput->mMousePos[1]);
    for (int i(0); i < CmdInput::NetImguiMouseButton::ImGuiMouseButton_COUNT;
         ++i) {
      uint64_t valMask = 0x0000000000000001ull << i;
      if ((pCmdInput->mMouseDownMask ^
           client.mPreviousInputState.mMouseDownMask) &
          valMask) {
        io.AddMouseButtonEvent(i, pCmdInput->mMouseDownMask & valMask);
      }
    }
#endif
    uint16_t character;
    io.InputQueueCharacters.resize(0);
    while (client.mPendingKeyIn.ReadData(&character)) {
      ImWchar ConvertedKey = static_cast<ImWchar>(character);
      io.AddInputCharacter(ConvertedKey);
    }

    static_assert(sizeof(client.mPreviousInputState.mInputDownMask) ==
                      sizeof(pCmdInput->mInputDownMask),
                  "Array size should match");
    static_assert(sizeof(client.mPreviousInputState.mInputAnalog) ==
                      sizeof(pCmdInput->mInputAnalog),
                  "Array size should match");
    memcpy(client.mPreviousInputState.mInputDownMask, pCmdInput->mInputDownMask,
           sizeof(client.mPreviousInputState.mInputDownMask));
    memcpy(client.mPreviousInputState.mInputAnalog, pCmdInput->mInputAnalog,
           sizeof(client.mPreviousInputState.mInputAnalog));
    client.mPreviousInputState.mMouseDownMask = pCmdInput->mMouseDownMask;
    client.mPreviousInputState.mMouseWheelVertPrev = pCmdInput->mMouseWheelVert;
    client.mPreviousInputState.mMouseWheelHorizPrev =
        pCmdInput->mMouseWheelHoriz;
    client.mServerCompressionEnabled = pCmdInput->mCompressionUse;
    if (pCmdInput->mCompressionSkip) {
      client.mServerCompressionSkip = true;
    }
  }

  if (hasNewInput) {
    netImguiDeleteSafe(client.mpCmdInputPending);
    client.mpCmdInputPending = pCmdInputNew;
  }
  return hasNewInput;
}

}  // namespace NetImgui

#endif  // NETIMGUI_ENABLED

#include "NetImgui_WarningReenable.h"
