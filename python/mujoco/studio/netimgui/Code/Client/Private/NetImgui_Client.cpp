// Google modifications:
//  - Replaced yield() with sleep_for(1ms) in handshake wait loop to reduce
//    CPU usage, and added a 5-second timeout to prevent infinite hangs when
//    the server is unreachable.
//  - Added cmdTexture.mpNext = nullptr in TexturePendingServerAdd to
//    terminate the linked list when appending a new texture command.
//  - Added logging for network events.
//  - Free outgoing CmdBackground commands and any in-flight auto-freed
//    commands on disconnect.
//  - Synchronize mPendingTextures and CmdHeader::mSent accesses under
//    mPendingTexturesLock, and use atomics for ClientInfo fields shared
//    between the main thread and communication thread.
//  - Receive loop and handshake validate the command size received from the
//    network (PendingCom::SizeMax / IsBiggerBufferNeeded).
//  - Incoming commands are validated (size, offsets) before being used.
#include "NetImgui_Shared.h"
#include "google/logging.h"

#if NETIMGUI_ENABLED
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <chrono>
#include <thread>

#include "NetImgui_Client.h"
#include "NetImgui_CmdPackets.h"
#include "NetImgui_Network.h"
#include "NetImgui_WarningDisable.h"

namespace NetImgui {
namespace Internal {
namespace Client {

//=================================================================================================
// SAVED IMGUI CONTEXT
// Because we overwrite some Imgui context IO values, we save them before makign
// any change and restore them after detecting a disconnection
//=================================================================================================
void SavedImguiContext::Save(ImGuiContext* copyFrom) {
  ScopedImguiContext scopedContext(copyFrom);
  ImGuiIO& sourceIO = ImGui::GetIO();
  mSavedContext = true;
  mConfigFlags = sourceIO.ConfigFlags;
  mBackendFlags = sourceIO.BackendFlags;
  mBackendPlatformName = sourceIO.BackendPlatformName;
  mBackendRendererName = sourceIO.BackendRendererName;
  mDrawMouse = sourceIO.MouseDrawCursor;
#if !NETIMGUI_IMGUI_TEXTURES_ENABLED
  mFontGeneratedSize =
      sourceIO.Fonts->Fonts.Size > 0
          ? sourceIO.Fonts->Fonts[0]->FontSize
          : 13.f;  // Save size to restore the font to original size
  mFontGlobalScale = sourceIO.FontGlobalScale;
#endif

#if IMGUI_VERSION_NUM < 18700
  memcpy(mKeyMap, sourceIO.KeyMap, sizeof(mKeyMap));
#endif
#if IMGUI_VERSION_NUM < 19110
  mGetClipboardTextFn = sourceIO.GetClipboardTextFn;
  mSetClipboardTextFn = sourceIO.SetClipboardTextFn;
  mClipboardUserData = sourceIO.ClipboardUserData;
#else
  ImGuiPlatformIO& plaformIO = ImGui::GetPlatformIO();
  mGetClipboardTextFn = plaformIO.Platform_GetClipboardTextFn;
  mSetClipboardTextFn = plaformIO.Platform_SetClipboardTextFn;
  mClipboardUserData = plaformIO.Platform_ClipboardUserData;
#endif
}

void SavedImguiContext::Restore(ImGuiContext* copyTo) {
  ScopedImguiContext scopedContext(copyTo);
  ImGuiIO& destIO = ImGui::GetIO();
  mSavedContext = false;
  destIO.ConfigFlags = mConfigFlags;
  destIO.BackendFlags = mBackendFlags;
  destIO.BackendPlatformName = mBackendPlatformName;
  destIO.BackendRendererName = mBackendRendererName;
  destIO.MouseDrawCursor = mDrawMouse;
#if !NETIMGUI_IMGUI_TEXTURES_ENABLED
  destIO.FontGlobalScale = mFontGlobalScale;
#endif
#if IMGUI_VERSION_NUM < 18700
  memcpy(destIO.KeyMap, mKeyMap, sizeof(destIO.KeyMap));
#endif
#if IMGUI_VERSION_NUM < 19110
  destIO.GetClipboardTextFn = mGetClipboardTextFn;
  destIO.SetClipboardTextFn = mSetClipboardTextFn;
  destIO.ClipboardUserData = mClipboardUserData;
#else
  ImGuiPlatformIO& plaformIO = ImGui::GetPlatformIO();
  plaformIO.Platform_GetClipboardTextFn = mGetClipboardTextFn;
  plaformIO.Platform_SetClipboardTextFn = mSetClipboardTextFn;
  plaformIO.Platform_ClipboardUserData = mClipboardUserData;
#endif
}

//=================================================================================================
// GET CLIPBOARD
// Content received from the Server
//=================================================================================================
#if IMGUI_VERSION_NUM < 19110
static const char* GetClipboardTextFn_NetImguiImpl(void* user_data_ctx) {
  const ClientInfo* pClient =
      reinterpret_cast<const ClientInfo*>(user_data_ctx);
  return pClient && pClient->mpCmdClipboard
             ? pClient->mpCmdClipboard->mContentUTF8.Get()
             : nullptr;
}
#else
static const char* GetClipboardTextFn_NetImguiImpl(ImGuiContext* ctx) {
  ScopedImguiContext scopedContext(ctx);
  const ClientInfo* pClient = reinterpret_cast<const ClientInfo*>(
      ImGui::GetPlatformIO().Platform_ClipboardUserData);
  return pClient && pClient->mpCmdClipboard
             ? pClient->mpCmdClipboard->mContentUTF8.Get()
             : nullptr;
}
#endif

//=================================================================================================
// SET CLIPBOARD
//=================================================================================================
#if IMGUI_VERSION_NUM < 19110
static void SetClipboardTextFn_NetImguiImpl(void* user_data_ctx,
                                            const char* text) {
  if (user_data_ctx) {
    ClientInfo* pClient = reinterpret_cast<ClientInfo*>(user_data_ctx);
    CmdClipboard* pClipboardOut = CmdClipboard::Create(text);
    pClient->mPendingClipboardOut.Assign(pClipboardOut);
  }
}
#else
static void SetClipboardTextFn_NetImguiImpl(ImGuiContext* ctx,
                                            const char* text) {
  ScopedImguiContext scopedContext(ctx);
  ClientInfo* pClient = reinterpret_cast<ClientInfo*>(
      ImGui::GetPlatformIO().Platform_ClipboardUserData);
  CmdClipboard* pClipboardOut = CmdClipboard::Create(text);
  pClient->mPendingClipboardOut.Assign(pClipboardOut);
}
#endif

//=================================================================================================
// INCOM: INPUT
// Receive new keyboard/mouse/screen resolution input to pass on to dearImgui
//=================================================================================================
void Communications_Incoming_Input(ClientInfo& client) {
  auto pCmdInput = static_cast<CmdInput*>(client.mPendingRcv.pCommand);
  if (pCmdInput->mSize < sizeof(CmdInput)) {
    client.mPendingRcv.bError = true;  // Invalid command received
    return;
  }
  client.mPendingRcv.bAutoFree = false;  // Taking ownership of the data
  client.mDesiredFps =
      pCmdInput->mDesiredFps > 0.f ? pCmdInput->mDesiredFps : 0.f;
  size_t keyCount(pCmdInput->mKeyCharCount);
  if (keyCount > ArrayCount(pCmdInput->mKeyChars)) {
    keyCount = ArrayCount(pCmdInput->mKeyChars);
  }
  client.mPendingKeyIn.AddData(pCmdInput->mKeyChars, keyCount);

  VLOG(2, "CmdInput: mDesiredFps=%.1f, screen=%dx%d, mouse=%d,%d",
       pCmdInput->mDesiredFps, pCmdInput->mScreenSize[0],
       pCmdInput->mScreenSize[1], pCmdInput->mMousePos[0],
       pCmdInput->mMousePos[1]);

  client.mPendingInputIn.Assign(pCmdInput);
}

//=================================================================================================
// INCOM: CLIPBOARD
// Receive server new clipboard content, updating internal cache
//=================================================================================================
void Communications_Incoming_Clipboard(ClientInfo& client) {
  auto pCmdClipboard = static_cast<CmdClipboard*>(client.mPendingRcv.pCommand);
  if (!pCmdClipboard->ToPointers()) {
    client.mPendingRcv.bError = true;  // Invalid command received
    return;
  }
  client.mPendingRcv.bAutoFree = false;  // Taking ownership of the data
  client.mPendingClipboardIn.Assign(pCmdClipboard);
}

//=================================================================================================
// OUTCOM: TEXTURE
// Transmit the next pending texture command
//=================================================================================================
void Communications_Outgoing_Textures(ClientInfo& client) {
  std::lock_guard<std::mutex> guard(client.mPendingTexturesLock);
  if (client.mPendingTextures) {
    // Get oldest texture command to sent to server
    CmdTexture* pPendingTexture = client.mPendingTextures;
    client.mPendingTextures = client.mPendingTextures->mpNext;
    pPendingTexture->mpNext = nullptr;
    client.mPendingSend.pCommand = pPendingTexture;
    client.mPendingSend.bAutoFree =
        false;  // free handled by main update thread
  }
}

//=================================================================================================
// OUTCOM: BACKGROUND
// Transmit the current client background settings
//=================================================================================================
void Communications_Outgoing_Background(ClientInfo& client) {
  CmdBackground* pPendingBackground = client.mPendingBackgroundOut.Release();
  if (pPendingBackground) {
    client.mPendingSend.pCommand = pPendingBackground;
    client.mPendingSend.bAutoFree = true;
  }
}

//=================================================================================================
// OUTCOM: FRAME
// Transmit a new dearImgui frame to render
//=================================================================================================
void Communications_Outgoing_Frame(ClientInfo& client) {
  CmdDrawFrame* pPendingDraw = client.mPendingFrameOut.Release();
  if (pPendingDraw) {
    pPendingDraw->mFrameIndex = client.mFrameIndex++;
    //---------------------------------------------------------------------
    // Apply delta compression to DrawCommand, when requested
    if (pPendingDraw->mCompressed) {
      // Create a new Compressed DrawFrame Command
      if (client.mpCmdDrawLast && !client.mServerCompressionSkip) {
        client.mpCmdDrawLast->ToPointers();
        CmdDrawFrame* pDrawCompressed =
            CompressCmdDrawFrame(client.mpCmdDrawLast, pPendingDraw);
        netImguiDeleteSafe(client.mpCmdDrawLast);
        client.mpCmdDrawLast = pPendingDraw;  // Keep original new command for
                                              // next frame delta compression
        pPendingDraw =
            pDrawCompressed;  // Request compressed copy to be sent to server
      }
      // Save DrawCmd for next frame delta compression
      else {
        pPendingDraw->mCompressed = false;
        client.mpCmdDrawLast = pPendingDraw;
      }
    }
    client.mServerCompressionSkip = false;

    //---------------------------------------------------------------------
    // Ready to send command to server
    pPendingDraw->ToOffsets();
    client.mPendingSend.pCommand = pPendingDraw;
    client.mPendingSend.bAutoFree = client.mpCmdDrawLast != pPendingDraw;
  }
}

//=================================================================================================
// OUTCOM: Clipboard
// Send client 'Copy' clipboard content to Server
//=================================================================================================
void Communications_Outgoing_Clipboard(ClientInfo& client) {
  CmdClipboard* pPendingClipboard = client.mPendingClipboardOut.Release();
  if (pPendingClipboard) {
    pPendingClipboard->ToOffsets();
    client.mPendingSend.pCommand = pPendingClipboard;
    client.mPendingSend.bAutoFree = true;
  }
}

//=================================================================================================
// INCOMING COMMUNICATIONS
//=================================================================================================
void Communications_Incoming(ClientInfo& client) {
  if (Network::DataReceivePending(client.mpSocketComs)) {
    //-----------------------------------------------------------------------------------------
    // 1. Ready to receive new command, starts the process by reading Header
    //-----------------------------------------------------------------------------------------
    if (client.mPendingRcv.IsReady()) {
      client.mCmdPendingRead = CmdPendingRead();
      client.mPendingRcv.pCommand = &client.mCmdPendingRead;
      client.mPendingRcv.SizeMax = sizeof(client.mCmdPendingRead);
      client.mPendingRcv.bAutoFree = false;
    }

    //-----------------------------------------------------------------------------------------
    // 2. Read incoming command from server
    //-----------------------------------------------------------------------------------------
    if (client.mPendingRcv.IsPending()) {
      Network::DataReceive(client.mpSocketComs, client.mPendingRcv);

      // Detected a new command bigger than header, allocate memory for it
      if (client.mPendingRcv.IsBiggerBufferNeeded()) {
        CmdPendingRead* pCmdHeader = reinterpret_cast<CmdPendingRead*>(
            netImguiSizedNew<uint8_t>(client.mPendingRcv.pCommand->mSize));
        *pCmdHeader = client.mCmdPendingRead;
        client.mPendingRcv.pCommand = pCmdHeader;
        client.mPendingRcv.SizeMax = pCmdHeader->mSize;
        client.mPendingRcv.bAutoFree = true;
      }
    }

    //-----------------------------------------------------------------------------------------
    // 3. Command fully received from Server, process it
    //-----------------------------------------------------------------------------------------
    if (client.mPendingRcv.IsDone()) {
      if (!client.mPendingRcv.IsError()) {
        switch (client.mPendingRcv.pCommand->mType) {
          case CmdHeader::eCommands::Input:
            Communications_Incoming_Input(client);
            break;
          case CmdHeader::eCommands::Clipboard:
            Communications_Incoming_Clipboard(client);
            break;
          // Commands not received in main loop, by Client
          case CmdHeader::eCommands::Version:
          case CmdHeader::eCommands::Texture:
          case CmdHeader::eCommands::DrawFrame:
          case CmdHeader::eCommands::Background:
          case CmdHeader::eCommands::Count:
            break;
        }
      }

      // Cleanup after read completion
      if (client.mPendingRcv.IsError()) {
        // Routine on any disconnect (server restart, browser handover), so
        // logged as Info rather than Error.
        VLOG(1,
             "Communications_Incoming: receive error, triggering disconnect");
        client.mbDisconnectPending = true;
      }
      if (client.mPendingRcv.bAutoFree) {
        netImguiDeleteSafe(client.mPendingRcv.pCommand);
      }
      client.mPendingRcv = PendingCom();
    }
  }
  // Prevent high CPU usage when waiting for new data
  else {
    // std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::microseconds(250));
  }
}

//=================================================================================================
// OUTGOING COMMUNICATIONS
//=================================================================================================
void Communications_Outgoing(ClientInfo& client) {
  //---------------------------------------------------------------------------------------------
  // Try finishing sending a pending command to Server
  //---------------------------------------------------------------------------------------------
  if (client.mPendingSend.IsPending()) {
    Network::DataSend(client.mpSocketComs, client.mPendingSend);

    // Free allocated memory for command
    if (client.mPendingSend.IsDone()) {
      VLOG(2, "Outgoing: send DONE type=%d, size=%u, error=%s",
           static_cast<int>(client.mPendingSend.pCommand->mType),
           client.mPendingSend.pCommand->mSize,
           client.mPendingSend.IsError() ? "Y" : "N");
      {
        std::lock_guard<std::mutex> guard(client.mPendingTexturesLock);
        client.mPendingSend.pCommand->mSent = true;
      }
      if (client.mPendingSend.IsError()) {
        client.mbDisconnectPending = true;
      }
      if (client.mPendingSend.bAutoFree) {
        netImguiDeleteSafe(client.mPendingSend.pCommand);
      }
      client.mPendingSend = PendingCom();
    }
  }
  //---------------------------------------------------------------------------------------------
  // Initiate sending next command to Server, when none are in flight
  // Note: The cmd order is important, textures must always be sent before
  // DrawFrame
  // @sammyfreg todo Could add a frame number awareness to avoid sending only
  // textures with no stop
  //---------------------------------------------------------------------------------------------
  constexpr CmdHeader::eCommands kCommandsOrder[] = {
      CmdHeader::eCommands::Texture, CmdHeader::eCommands::Background,
      CmdHeader::eCommands::Clipboard, CmdHeader::eCommands::DrawFrame};
  constexpr uint32_t kCommandCounts = IM_ARRAYSIZE(kCommandsOrder);

  uint32_t Index(0);
  while (client.mPendingSend.IsReady() && Index < kCommandCounts) {
    CmdHeader::eCommands NextCmd = kCommandsOrder[Index++];
    switch (NextCmd) {
      case CmdHeader::eCommands::Texture:
        Communications_Outgoing_Textures(client);
        break;
      case CmdHeader::eCommands::Background:
        Communications_Outgoing_Background(client);
        break;
      case CmdHeader::eCommands::Clipboard:
        Communications_Outgoing_Clipboard(client);
        break;
      case CmdHeader::eCommands::DrawFrame:
        Communications_Outgoing_Frame(client);
        break;
      // Commands not sent in main loop, by Client
      case CmdHeader::eCommands::Input:
      case CmdHeader::eCommands::Version:
      case CmdHeader::eCommands::Count:
        break;
    }
  }

  // Log new send if one was started
  VLOG(2, "Outgoing: ready=%s, pending=%s, frameOut=%s",
       client.mPendingSend.IsReady() ? "Y" : "N",
       client.mPendingSend.IsPending() ? "Y" : "N",
       client.mPendingFrameOut.IsNull() ? "empty" : "HAS_DATA");
  if (client.mPendingSend.IsPending()) {
    VLOG(2, "Outgoing: started type=%d, size=%u",
         static_cast<int>(client.mPendingSend.pCommand->mType),
         client.mPendingSend.pCommand->mSize);
  }
}

//=================================================================================================
// COMMUNICATIONS INITIALIZE
// Initialize a new connection to a RemoteImgui server
//=================================================================================================
bool Communications_Initialize(ClientInfo& client) {
  CmdVersion cmdVersionSend, cmdVersionRcv;
  PendingCom PendingRcv, PendingSend;

  client.mbComInitActive = true;

  //---------------------------------------------------------------------
  // Handshake confirming connection validity
  //---------------------------------------------------------------------
  PendingRcv.pCommand = reinterpret_cast<CmdPendingRead*>(&cmdVersionRcv);
  PendingRcv.SizeMax = sizeof(cmdVersionRcv);
  while (!PendingRcv.IsDone() &&
         cmdVersionRcv.mType == CmdHeader::eCommands::Version) {
    auto start_time = std::chrono::steady_clock::now();
    while (!client.mbDisconnectPending &&
           !Network::DataReceivePending(client.mpSocketPending)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(
          1));  // Idle until we receive the remote data
      if (std::chrono::steady_clock::now() - start_time >
          std::chrono::seconds(5)) {
        LOG(Error, "Communications_Initialize: Timeout waiting for data");
        client.mbDisconnectPending = true;
        break;
      }
    }
    if (client.mbDisconnectPending) break;
    Network::DataReceive(client.mpSocketPending, PendingRcv);
  }

  bool bForceConnect =
      client.mServerForceConnectEnabled &&
      (cmdVersionRcv.mFlags &
       static_cast<uint8_t>(CmdVersion::eFlags::ConnectForce)) != 0;
  bool bCanConnect = !PendingRcv.IsError() &&
                     cmdVersionRcv.mSize == sizeof(CmdVersion) &&
                     cmdVersionRcv.mType == cmdVersionSend.mType &&
                     cmdVersionRcv.mVersion == cmdVersionSend.mVersion &&
                     cmdVersionRcv.mWCharSize == cmdVersionSend.mWCharSize &&
                     (!client.IsConnected() || bForceConnect);

  StringCopy(cmdVersionSend.mClientName, client.mName);
  cmdVersionSend.mFlags =
      client.IsConnected() && !bCanConnect
          ? static_cast<uint8_t>(CmdVersion::eFlags::IsConnected)
          : 0;
  cmdVersionSend.mFlags |=
      client.IsConnected() && !client.mServerForceConnectEnabled
          ? static_cast<uint8_t>(CmdVersion::eFlags::IsUnavailable)
          : 0;

  PendingSend.pCommand = reinterpret_cast<CmdPendingRead*>(&cmdVersionSend);
  while (!PendingSend.IsDone()) {
    Network::DataSend(client.mpSocketPending, PendingSend);
  }

  //---------------------------------------------------------------------
  // Connection established, init client
  //---------------------------------------------------------------------
  if (bCanConnect && !PendingSend.IsError() &&
      (!client.IsConnected() || bForceConnect)) {
    Network::SocketInfo* pNewComSocket =
        client.mpSocketPending.exchange(nullptr);

    // If we detect an active connection with Server and 'ForceConnect was
    // requested, close it first
    if (client.IsConnected()) {
      client.mbDisconnectPending = true;
      while (client.IsConnected());
    }

    client.mpSocketComs = pNewComSocket;  // Take ownerhip of socket
    client.mBGSettingSent.mTextureId =
        client.mBGSetting.mTextureId -
        1u;  // Force sending the Background settings (by making different than
             // current settings)
    client.mFrameIndex = 0;
    client.mClientTextureIDNext = 0;
    client.mServerForceConnectEnabled =
        (cmdVersionRcv.mFlags &
         static_cast<uint8_t>(CmdVersion::eFlags::ConnectExclusive)) == 0;
    client.mPendingRcv = PendingCom();
    client.mPendingSend = PendingCom();
  }

  // Disconnect pending socket if init failed
  Network::SocketInfo* SocketPending = client.mpSocketPending.exchange(nullptr);
  bool bValidConnection = SocketPending == nullptr;
  if (SocketPending) {
    NetImgui::Internal::Network::Disconnect(SocketPending);
  }

  client.mbComInitActive = false;
  return bValidConnection;
}

//=================================================================================================
// COMMUNICATIONS MAIN LOOP
//=================================================================================================
void Communications_Loop(void* pClientVoid) {
  IM_ASSERT(pClientVoid != nullptr);
  ClientInfo* pClient = reinterpret_cast<ClientInfo*>(pClientVoid);
  pClient->mbDisconnectPending = false;
  pClient->mbClientThreadActive = true;

  VLOG(1, "Communications_Loop: entering loop");
  while (!pClient->mbDisconnectPending) {
    Communications_Outgoing(*pClient);
    Communications_Incoming(*pClient);
    VLOG(2, "Communications_Loop spin, disconnectPending=%s",
         pClient->mbDisconnectPending ? "Y" : "N");
  }
  VLOG(1, "Communications_Loop: exiting loop");

  if (pClient->mPendingRcv.bAutoFree) {
    netImguiDeleteSafe(pClient->mPendingRcv.pCommand);
  }
  pClient->mPendingRcv = PendingCom();
  if (pClient->mPendingSend.bAutoFree) {
    netImguiDeleteSafe(pClient->mPendingSend.pCommand);
  }
  pClient->mPendingSend = PendingCom();

  Network::SocketInfo* pSocket = pClient->mpSocketComs.exchange(nullptr);
  if (pSocket) {
    NetImgui::Internal::Network::Disconnect(pSocket);
  }

  pClient->mbClientThreadActive = false;
}

//=================================================================================================
// COMMUNICATIONS CONNECT THREAD : Reach out and connect to a NetImGuiServer
//=================================================================================================
void CommunicationsConnect(void* pClientVoid) {
  IM_ASSERT(pClientVoid != nullptr);
  ClientInfo* pClient = reinterpret_cast<ClientInfo*>(pClientVoid);
  if (Communications_Initialize(*pClient)) {
    Communications_Loop(pClientVoid);
  }
}

//=================================================================================================
// COMMUNICATIONS HOST THREAD : Waiting NetImGuiServer reaching out to us.
//								Launch a new com
//loop when connection is established
//=================================================================================================
void CommunicationsHost(void* pClientVoid) {
  ClientInfo* pClient = reinterpret_cast<ClientInfo*>(pClientVoid);
  pClient->mbListenThreadActive = true;
  pClient->mbDisconnectListen = false;
  pClient->mpSocketListen = pClient->mpSocketPending.exchange(nullptr);

  while (pClient->mpSocketListen.load() != nullptr &&
         !pClient->mbDisconnectListen) {
    pClient->mpSocketPending = Network::ListenConnect(pClient->mpSocketListen);
    if (pClient->mpSocketPending.load() != nullptr &&
        Communications_Initialize(*pClient)) {
      pClient->mThreadFunction(Client::Communications_Loop, pClient);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(
        16));  // Prevents this thread from taking entire core, waiting on
               // server connection requests
  }

  Network::SocketInfo* pSocket = pClient->mpSocketListen.exchange(nullptr);
  NetImgui::Internal::Network::Disconnect(pSocket);
  pClient->mbListenThreadActive = false;
}

//=================================================================================================
// Support of the Dear ImGui hooks
// (automatic call to NetImgui::BeginFrame()/EndFrame() on
// ImGui::BeginFrame()/Imgui::Render()
//=================================================================================================
#if NETIMGUI_IMGUI_CALLBACK_ENABLED
void HookBeginFrame(ImGuiContext*, ImGuiContextHook* hook) {
  Client::ClientInfo& client =
      *reinterpret_cast<Client::ClientInfo*>(hook->UserData);
  if (!client.mbInsideNewEnd) {
    ScopedBool scopedInside(client.mbInsideHook, true);
    NetImgui::NewFrame(false);
  }
}

void HookEndFrame(ImGuiContext*, ImGuiContextHook* hook) {
  Client::ClientInfo& client =
      *reinterpret_cast<Client::ClientInfo*>(hook->UserData);
  if (!client.mbInsideNewEnd) {
    ScopedBool scopedInside(client.mbInsideHook, true);
    NetImgui::EndFrame();
  }
}

#endif  // NETIMGUI_IMGUI_CALLBACK_ENABLED

//=================================================================================================
// CLIENT INFO Constructor
//=================================================================================================
ClientInfo::ClientInfo()
    : mpSocketPending(nullptr),
      mpSocketComs(nullptr),
      mpSocketListen(nullptr),
      mbClientThreadActive(false),
      mbListenThreadActive(false),
      mbComInitActive(false) {}

//=================================================================================================
// CLIENT INFO Constructor
//=================================================================================================
ClientInfo::~ClientInfo() {
  ContextRemoveHooks();

  // Free all tracked textures
  for (auto cmdTexture : mTrackedTextures) {
    netImguiDelete(cmdTexture);
  }
  mTrackedTextures.clear();

  netImguiDeleteSafe(mpCmdInputPending);
  netImguiDeleteSafe(mpCmdDrawLast);
  netImguiDeleteSafe(mpCmdClipboard);
}

//=================================================================================================
// Initialize the associated ImguiContext
//=================================================================================================
void ClientInfo::ContextInitialize() {
  mpContext = ImGui::GetCurrentContext();
#if NETIMGUI_IMGUI_CALLBACK_ENABLED
  ImGuiContextHook hookNewframe, hookEndframe;
  ContextRemoveHooks();
  hookNewframe.HookId = 0;
  hookNewframe.Type = ImGuiContextHookType_NewFramePre;
  hookNewframe.Callback = HookBeginFrame;
  hookNewframe.UserData = this;
  mhImguiHookNewframe = ImGui::AddContextHook(mpContext, &hookNewframe);
  hookEndframe.HookId = 0;
  hookEndframe.Type = ImGuiContextHookType_RenderPost;
  hookEndframe.Callback = HookEndFrame;
  hookEndframe.UserData = this;
  mhImguiHookEndframe = ImGui::AddContextHook(mpContext, &hookEndframe);
#endif
#if !NETIMGUI_IMGUI_TEXTURES_ENABLED
  mbFontUploaded = false;
#endif
}

//=================================================================================================
// Take over a Dear ImGui context for use with NetImgui
//=================================================================================================
void ClientInfo::ContextOverride() {
  ScopedImguiContext scopedSourceCtx(mpContext);

  // Keep a copy of original settings of this context
  mSavedContextValues.Save(mpContext);
  mLastOutgoingDrawCheckTime = std::chrono::steady_clock::now();

  // Override some settings
  // Note: Make sure every setting overwritten here, are handled in
  // 'SavedImguiContext::Save(...)'
  {
    ImGuiIO& newIO = ImGui::GetIO();
    newIO.MouseDrawCursor = false;
    newIO.BackendPlatformName = "NetImgui";
    newIO.BackendRendererName = "DirectX11";
#if IMGUI_VERSION_NUM < 18700
    for (uint32_t i(0); i < ImGuiKey_COUNT; ++i) {
      newIO.KeyMap[i] = i;
    }
#endif
#if IMGUI_VERSION_NUM < 19110
    newIO.GetClipboardTextFn = GetClipboardTextFn_NetImguiImpl;
    newIO.SetClipboardTextFn = SetClipboardTextFn_NetImguiImpl;
    newIO.ClipboardUserData = this;
#else
    ImGuiPlatformIO& platformIO = ImGui::GetPlatformIO();
    platformIO.Platform_GetClipboardTextFn = GetClipboardTextFn_NetImguiImpl;
    platformIO.Platform_SetClipboardTextFn = SetClipboardTextFn_NetImguiImpl;
    platformIO.Platform_ClipboardUserData = this;
#endif
#if defined(IMGUI_HAS_VIEWPORT)
    newIO.ConfigFlags &=
        ~(ImGuiConfigFlags_ViewportsEnable);  // Viewport unsupported at the
                                              // moment
#endif

    // Resend every tracked textures
    for (auto textureCmd : mTrackedTextures) {
      if (textureCmd->mStatus == CmdTexture::eType::Create) {
        TexturePendingServerAdd(*textureCmd);
      }
    }

#if NETIMGUI_IMGUI_TEXTURES_ENABLED
    mFontSavedScaling = ImGui::GetStyle().FontScaleDpi;
    newIO.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    TextureTrackingUpdate(
        true);  // Force resend all Dear Imgui managed textures
#else
    if (mFontCreationFunction != nullptr) {
      newIO.FontGlobalScale = 1;
      mFontServerScale = -1;
    }
#endif
  }
}

//=================================================================================================
// Restore a Dear ImGui context to initial state before we modified it
//=================================================================================================
void ClientInfo::ContextRestore() {
  // Note: only happens if context overriden is same as current one, to prevent
  // trying to restore to a deleted context
  if (IsContextOverriden() && ImGui::GetCurrentContext() == mpContext) {
#ifdef IMGUI_HAS_VIEWPORT
    ImGui::UpdatePlatformWindows();  // Prevents issue with mismatched frame
                                     // tracking, when restoring enabled
                                     // viewport feature
#endif
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
    // Remove all auto managed texture from our tracking,
    // leaving only the manually managed textures.
    // Auto-Managed will automatically get re-added on reconnect.
    ImVector<ImTextureData*>& Textures = ImGui::GetPlatformIO().Textures;
    ImTextureRef ClientTextureRef;
    for (auto TexData : Textures) {
      ImTextureStatus TexStatus =
          TexData != nullptr ? TexData->Status : ImTextureStatus_Destroyed;
      if (TexStatus != ImTextureStatus_Destroyed &&
          TexStatus != ImTextureStatus_WantDestroy) {
        ClientTextureRef._TexData = TexData;
        TextureTrackingRem(ClientTextureRef.GetTexID());
      }
    }
    ImGui::GetStyle().FontScaleDpi = mFontSavedScaling;
#else
    if (mFontCreationFunction && ImGui::GetIO().Fonts &&
        ImGui::GetIO().Fonts->Fonts.size() > 0) {
      float noScaleSize =
          ImGui::GetIO().Fonts->Fonts[0]->FontSize / mFontServerScale;
      float originalScale =
          mSavedContextValues.mFontGeneratedSize / noScaleSize;
      mFontCreationFunction(mFontSavedScaling, originalScale);
    }
#endif
    mSavedContextValues.Restore(mpContext);
  }
}

//=================================================================================================
// Remove callback hooks, once we detect a disconnection
//=================================================================================================
void ClientInfo::ContextRemoveHooks() {
#if NETIMGUI_IMGUI_CALLBACK_ENABLED
  if (mpContext && mhImguiHookNewframe != 0) {
    ImGui::RemoveContextHook(mpContext, mhImguiHookNewframe);
    ImGui::RemoveContextHook(mpContext, mhImguiHookEndframe);
    mhImguiHookNewframe = mhImguiHookNewframe = 0;
  }
#endif
}

//=================================================================================================
// If backend doesn't support Dear ImGui managed texture (used by font) as
// indicated with 'ImGuiBackendFlags_RendererHasTextures', we still want to
// handle it with our RemoteServer so we take it upon ourselves to clear the
// list of texture updates. Otherwise, we'd be constantly re applying the same
// updates
//=================================================================================================
void ClientInfo::TextureTrackingClear() {
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
  if (IsConnected() && (mSavedContextValues.mBackendFlags &
                        ImGuiBackendFlags_RendererHasTextures) == 0) {
    // This is basically a stripped version of 'ImGui_ImplDX11_UpdateTexture'
    // where we only care about the TexID and status values
    ImVector<ImTextureData*>& Textures = ImGui::GetPlatformIO().Textures;
    for (auto TexData : Textures) {
      if (TexData->Status == ImTextureStatus_WantCreate) {
        IM_ASSERT(TexData->TexID == ImTextureID_Invalid &&
                  TexData->BackendUserData == nullptr);
        static ImTextureID sUniqueID(1);
        TexData->SetTexID(static_cast<ImTextureID>(sUniqueID++));
        TexData->SetStatus(ImTextureStatus_OK);
      } else if (TexData->Status == ImTextureStatus_WantUpdates) {
        TexData->SetStatus(ImTextureStatus_OK);
      } else if (TexData->Status == ImTextureStatus_WantDestroy &&
                 TexData->UnusedFrames > 0) {
        TexData->SetTexID(ImTextureID_Invalid);
        TexData->SetStatus(ImTextureStatus_Destroyed);
      }
    }
  }

  // Remove all Dear ImGui Managed textures on disconnect,
  // they will be resent on reconnect
  if (!IsConnected() && mDearImguiTextureCount > 0) {
    for (auto pCmdTexture : mTrackedTextures) {
      if (pCmdTexture->mIsDearImGuiManaged) {
        mbTrackedTexturesPending |=
            TextureTrackingRem(pCmdTexture->mTextureClientID);
      }
    }
  }
#endif
}

//=================================================================================================
// Process backend texture updates and forward it to NetImguiServer
// Note: This is mostly used by the Font Atlas Texture added in Dear ImGui 1.92+
//=================================================================================================
void ClientInfo::TextureTrackingUpdate(bool bResendAll) {
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
  if (IsConnected()) {
    ImVector<ImTextureData*>& Textures = ImGui::GetPlatformIO().Textures;
    ImTextureRef ClientTextureRef;

    //------------------------------------------------------------------------
    // Find and send all Dear ImGui managed textures creation/update to
    // remote NetImgui Server
    //------------------------------------------------------------------------
    for (auto TexData : Textures) {
      ClientTextureRef._TexData = TexData;
      ImTextureStatus TexStatus =
          TexData != nullptr ? TexData->Status : ImTextureStatus_OK;
      eTexFormat TexFormat =
          NetImgui::Internal::ConvertTextureFormat(TexData->Format);
      uint32_t dataSize = 0;
      uint16_t w = static_cast<uint16_t>(TexData->Width);
      uint16_t h = static_cast<uint16_t>(TexData->Height);

      if ((TexStatus == ImTextureStatus_WantCreate) ||
          (bResendAll && TexStatus != ImTextureStatus_Destroyed &&
           TexStatus != ImTextureStatus_WantDestroy)) {
        // @sammyfreg todo 	UserID and mClientTextureIDNext used for dear
        // imgui managed's textures
        //					could potentially collide when
        //generating ClientTextureID. Needs something more robust.
        TexData->UniqueID = ++mClientTextureIDNext;
        CmdTexture* pCmdTexture = TextureCmdAllocate(
            ConvertToClientTexID(ClientTextureRef), w, h, TexFormat, dataSize);
        if (pCmdTexture) {
          memcpy(pCmdTexture->mpTextureData.Get(), TexData->GetPixels(),
                 dataSize);
          pCmdTexture->mUpdatable = true;
          pCmdTexture->mpTextureData.ToOffset();
          pCmdTexture->mIsDearImGuiManaged = true;
          TextureTrackingAdd(
              *pCmdTexture);  // Add texture to our list and request it to be
                              // sent over to Server
          mbTrackedTexturesPending = true;
        }
      } else if (TexStatus == ImTextureStatus_WantUpdates) {
        uint16_t dstW = static_cast<uint16_t>(TexData->UpdateRect.w);
        uint16_t dstH = static_cast<uint16_t>(TexData->UpdateRect.h);
        CmdTexture* pCmdTexture =
            TextureCmdAllocate(ConvertToClientTexID(ClientTextureRef), dstW,
                               dstH, TexFormat, dataSize);
        if (pCmdTexture) {
          size_t SrcLineBytes = static_cast<size_t>(TexData->GetPitch());
          size_t DstLineBytes = static_cast<size_t>(TexData->BytesPerPixel *
                                                    TexData->UpdateRect.w);
          size_t OffsetBytes = static_cast<size_t>(
              TexData->GetPitch() * TexData->UpdateRect.y +
              TexData->BytesPerPixel * TexData->UpdateRect.x);
          const uint8_t* pDataSrc =
              reinterpret_cast<uint8_t*>(TexData->GetPixels());
          uint8_t* pDataDst = pCmdTexture->mpTextureData.Get();
          for (uint64_t y(0); y < TexData->UpdateRect.h; ++y) {
            memcpy(&pDataDst[y * DstLineBytes],
                   &pDataSrc[OffsetBytes + (y * SrcLineBytes)], DstLineBytes);
          }
          pCmdTexture->mStatus = CmdTexture::eType::Update;
          pCmdTexture->mOffsetX = TexData->UpdateRect.x;
          pCmdTexture->mOffsetY = TexData->UpdateRect.y;
          pCmdTexture->mUpdatable = true;
          pCmdTexture->mIsDearImGuiManaged = true;
          pCmdTexture->mpTextureData.ToOffset();
          TexturePendingServerAdd(
              *pCmdTexture);  // Request texture to be sent over to Server
          mbTrackedTexturesPending = true;
        }
      }
    }

    //------------------------------------------------------------------------
    // Detects Textures changes that have not been tracked
    // (caused by internal font atlas changes)
    // This is not ideal with large texture count but should be ok for most
    // cases. Using a unordered_map for 'mTrackedTextures' could help for
    // high count, but adds a std dependency
    //------------------------------------------------------------------------
    if (mDearImguiTextureCount != Textures.Size) {
      for (auto pCmdTexture : mTrackedTextures) {
        if (pCmdTexture->mIsDearImGuiManaged) {
          bool bFound(false);
          for (int i(0); !bFound && i < Textures.Size; ++i) {
            ClientTextureRef._TexData = Textures[i];
            bFound = pCmdTexture->mTextureClientID ==
                     ConvertToClientTexID(ClientTextureRef);
          }
          if (!bFound) {
            mbTrackedTexturesPending |=
                TextureTrackingRem(pCmdTexture->mTextureClientID);
          }
        }
      }
    }
  }
#endif

  //----------------------------------------
  if (mbTrackedTexturesPending) {
    std::lock_guard<std::mutex> guard(mPendingTexturesLock);
    int TrackedCount(mTrackedTextures.Size);
    mbTrackedTexturesPending = false;
    for (int i(0); i < TrackedCount; ++i) {
      mbTrackedTexturesPending |=
          !mTrackedTextures[i]->mSent &&
          mTrackedTextures[i]->mStatus != CmdTexture::eType::Create;
      // As soon as we detect a command to be un-needed, release it
      if (mTrackedTextures[i]->mSent &&
          mTrackedTextures[i]->mStatus != CmdTexture::eType::Create) {
        netImguiDelete(mTrackedTextures[i]);
        mTrackedTextures[i] =
            mTrackedTextures[--TrackedCount];  // Erase swap with last element
        --i;  // re-process same index after its entry was swapped with last
              // valid element
      }
    }
    mTrackedTextures.resize(TrackedCount);
  }
}

CmdTexture* ClientInfo::TextureCmdAllocate(ClientTextureID clientTexID,
                                           uint16_t width, uint16_t height,
                                           eTexFormat format,
                                           uint32_t& dataSizeInOut) {
  if (format != eTexFormat::kTexFmtCustom) {
    dataSizeInOut = GetTexture_BytePerImage(format, width, height);
  }

  uint32_t SizeNeeded = dataSizeInOut + sizeof(CmdTexture);
  CmdTexture* pCmdTexture = netImguiSizedNew<CmdTexture>(SizeNeeded);
  if (pCmdTexture) {
    pCmdTexture->mpTextureData.SetPtr(
        reinterpret_cast<uint8_t*>(&pCmdTexture[1]));
    pCmdTexture->mStatus = CmdTexture::eType::Create;
    pCmdTexture->mSize = SizeNeeded;
    pCmdTexture->mWidth = width;
    pCmdTexture->mHeight = height;
    pCmdTexture->mTextureClientID = clientTexID;
    pCmdTexture->mFormat = static_cast<uint8_t>(format);
    pCmdTexture->mUpdatable = false;
  }
  return pCmdTexture;
}

// Start tracking this client texture
bool ClientInfo::TextureTrackingAdd(CmdTexture& cmdTexture) {
  if (cmdTexture.mStatus != CmdTexture::eType::Create) {
    return false;
  }

  TextureTrackingRem(cmdTexture.mTextureClientID);
  mTrackedTextures.push_back(
      &cmdTexture);  // Add the new entry needing to be tracked
  TexturePendingServerAdd(
      cmdTexture);  // Request texture to be sent over to Server
  mDearImguiTextureCount += cmdTexture.mIsDearImGuiManaged ? 1 : 0;
  return true;
}

bool ClientInfo::TextureTrackingRem(ClientTextureID clientTextureID) {
  // If texture has been sent to server, re-purpose existing command
  // as a 'destroy' and re-send it to server
  for (int i(0); i < mTrackedTextures.Size; ++i) {
    CmdTexture* pCmdTexture = mTrackedTextures[i];
    if (pCmdTexture && pCmdTexture->mTextureClientID == clientTextureID &&
        pCmdTexture->mStatus == CmdTexture::eType::Create) {
      {
        std::lock_guard<std::mutex> guard(mPendingTexturesLock);
        pCmdTexture->mSent = false;
        pCmdTexture->mStatus =
            CmdTexture::eType::Destroy;  // Re-purpose create cmd to destroy the
                                         // texture
      }
      TexturePendingServerAdd(*pCmdTexture);

      // Remove item from our list
      mDearImguiTextureCount -= pCmdTexture->mIsDearImGuiManaged ? 1 : 0;
      mTrackedTextures[i] = mTrackedTextures.back();
      mTrackedTextures.pop_back();
      return true;
    }
  }
  return false;
}

void ClientInfo::TexturePendingServerAdd(CmdTexture& cmdTexture) {
  std::lock_guard<std::mutex> guard(mPendingTexturesLock);
  if (IsConnected()) {
    // Find last added entry
    CmdTexture** ppNextTexture = &mPendingTextures;
    CmdTexture* pendingTexture = mPendingTextures;
    while (pendingTexture != nullptr) {
      // Remove all unprocessed texture commands with same id
      // (only need the latest action for create/destroy, but can have multiple
      // update queued)
      if (cmdTexture.mStatus != CmdTexture::eType::Update &&
          cmdTexture.mSent == false &&
          cmdTexture.mTextureClientID == pendingTexture->mTextureClientID) {
        // Mark as sent and un-needed (which gets it removed from tracking array
        // and deleted later)
        pendingTexture->mSent = true;
        pendingTexture->mStatus = CmdTexture::eType::Destroy;
        *ppNextTexture = pendingTexture->mpNext;
      }
      ppNextTexture = &pendingTexture->mpNext;
      pendingTexture = pendingTexture->mpNext;
    }

    // Add as last element and ready to be sent
    cmdTexture.mSent = false;
    cmdTexture.mpNext = nullptr;
    *ppNextTexture = &cmdTexture;
  }
}

//=================================================================================================
// Create a new Draw Command from Dear Imgui draw data.
// 1. New ImGui frame has been completed, create a new draw command from draw
// data (Main Thread)
// 2. We see a pending Draw Command, take ownership of it and send it to Server
// (Com thread)
//=================================================================================================
void ClientInfo::ProcessDrawData(const ImDrawData* pDearImguiData,
                                 ImGuiMouseCursor mouseCursor) {
  if (!mbValidDrawFrame) return;

  VLOG(2, "ProcessDrawData: %s (mbValidDrawFrame=%s, mDesiredFps=%.1f)",
       mbValidDrawFrame ? "SENDING" : "SKIPPED",
       mbValidDrawFrame ? "true" : "false", mDesiredFps.load());

  CmdDrawFrame* pDrawFrameNew =
      ConvertToCmdDrawFrame(pDearImguiData, mouseCursor);
  pDrawFrameNew->mCompressed =
      mClientCompressionMode == eCompressionMode::kForceEnable ||
      (mClientCompressionMode == eCompressionMode::kUseServerSetting &&
       mServerCompressionEnabled);
  mPendingFrameOut.Assign(pDrawFrameNew);
}

}  // namespace Client
}  // namespace Internal
}  // namespace NetImgui

#include "NetImgui_WarningReenable.h"
#endif  // #if NETIMGUI_ENABLED
