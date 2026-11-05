// Google modifications:
//  - Use atomics for ClientInfo fields shared between the main thread and
//    communication thread.
#pragma once

#include <mutex>

#include "NetImgui_CmdPackets.h"
#include "NetImgui_Shared.h"

//=============================================================================
// Forward Declares
//=============================================================================
namespace NetImgui {
namespace Internal {
namespace Network {
struct SocketInfo;
}
}  // namespace Internal
}  // namespace NetImgui

namespace NetImgui {
namespace Internal {
namespace Client {

//=============================================================================
// Keeps a list of ImGui context values NetImgui overrides (to restore)
//=============================================================================
struct SavedImguiContext {
  void Save(ImGuiContext* copyFrom);
  void Restore(ImGuiContext* copyTo);
  const char* mBackendPlatformName = nullptr;
  const char* mBackendRendererName = nullptr;
  void* mImeWindowHandle = nullptr;
  ImGuiBackendFlags mBackendFlags = 0;
  ImGuiConfigFlags mConfigFlags = 0;
  bool mDrawMouse = false;
  bool mSavedContext = false;
  char mPadding1[2] = {};
  void* mClipboardUserData = nullptr;
#if IMGUI_VERSION_NUM < 19110
  const char* (*mGetClipboardTextFn)(void*) = nullptr;
  void (*mSetClipboardTextFn)(void*, const char*) = nullptr;
#else
  const char* (*mGetClipboardTextFn)(ImGuiContext*) = nullptr;
  void (*mSetClipboardTextFn)(ImGuiContext*, const char*) = nullptr;
#endif
#if IMGUI_VERSION_NUM < 18700
  int mKeyMap[ImGuiKey_COUNT] = {};
  char mPadding2[8 - (sizeof(mKeyMap) % 8)] = {};
#endif
#if !NETIMGUI_IMGUI_TEXTURES_ENABLED
  float mFontGlobalScale = 1.f;
  float mFontGeneratedSize = 0.f;
#endif
};

//=============================================================================
// Keep all Client infos needed for communication with server
//=============================================================================
struct ClientInfo {
  using BufferKeys = Ringbuffer<uint16_t, 1024>;
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  struct InputState {
    uint64_t mInputDownMask[(CmdInput::ImGuiKey_COUNT + 63) / 64] = {};
    float mInputAnalog[CmdInput::kAnalog_Count] = {};
    uint64_t mMouseDownMask = 0;
    float mMouseWheelVertPrev = 0.f;
    float mMouseWheelHorizPrev = 0.f;
  };
  ClientInfo();
  ~ClientInfo();
  void ContextInitialize();
  void ContextOverride();
  void ContextRestore();
  void ContextRemoveHooks();
  inline bool IsContextOverriden() const;
  inline bool IsConnected() const;
  inline bool IsConnectPending() const;
  inline bool IsActive() const;

  bool TextureTrackingAdd(CmdTexture& cmdTexture);
  bool TextureTrackingRem(ClientTextureID cmdTexture);
  void TextureTrackingClear();
  void TextureTrackingUpdate(
      bool bResendAll = false);  // Process Backend ImGui textures
  CmdTexture* TextureCmdAllocate(ClientTextureID clientTexID, uint16_t width,
                                 uint16_t height, eTexFormat format,
                                 uint32_t& dataSizeInOut);

  void TexturePendingServerAdd(
      CmdTexture& cmdTexture);  // Add CmdTexture to list of command waiting for
                                // send off to Server

  void ProcessDrawData(const ImDrawData* pDearImguiData,
                       ImGuiMouseCursor mouseCursor);

  std::atomic<Network::SocketInfo*>
      mpSocketPending;  // Hold socket info until communication is established
  std::atomic<Network::SocketInfo*>
      mpSocketComs;  // Socket used for communications with server
  std::atomic<Network::SocketInfo*>
      mpSocketListen;  // Socket used to wait for communication request from
                       // server
  std::atomic_bool mbDisconnectPending;  // Terminate Client/Server coms
  std::atomic_bool
      mbDisconnectListen;          // Terminate waiting connection from Server
  uint32_t mSocketListenPort = 0;  // Socket Port number used to wait for
                                   // communication request from server
  char mName[64] = {};
  uint64_t mFrameIndex =
      0;  // Incremented every time we send a DrawFrame Command
  std::mutex mPendingTexturesLock;  // Lock to prevent thread contention on the
                                    // list of texure cmd waiting to be sent to
                                    // the NetImgui Server
  CmdTexture* mPendingTextures =
      nullptr;  // List of texture commands waiting to be send to Sever (single
                // linked list with oldest item at the head)
  ImVector<CmdTexture*>
      mTrackedTextures;  // List of texture commands to create textures used by
                         // this client (note for large texture count, should be
                         // replace with a unordered_map for fast operations)
  ExchangePtr<CmdDrawFrame> mPendingFrameOut;
  ExchangePtr<CmdBackground> mPendingBackgroundOut;
  ExchangePtr<CmdInput> mPendingInputIn;
  ExchangePtr<CmdClipboard>
      mPendingClipboardIn;  // Clipboard content received from Server and
                            // waiting to be taken by client
  ExchangePtr<CmdClipboard>
      mPendingClipboardOut;  // Clipboard content copied on Client and waiting
                             // to be sent to Server
  ImGuiContext* mpContext =
      nullptr;  // Context that the remote drawing should use (the one active
                // when connection request happened)
  PendingCom mPendingRcv;          // Data being currently received from Server
  PendingCom mPendingSend;         // Data being currently sent to Server
  CmdPendingRead mCmdPendingRead;  // Used to get info on the next incoming
                                   // command from Server
  CmdInput* mpCmdInputPending = nullptr;   // Last Input Command from server,
                                           // waiting to be processed by client
  CmdClipboard* mpCmdClipboard = nullptr;  // Last received clipboad command
  CmdDrawFrame* mpCmdDrawLast =
      nullptr;  // Last sent Draw Command. Used by data compression, to generate
                // delta between previous and current frame
  CmdBackground
      mBGSetting;  // Current value assigned to background appearance by user
  CmdBackground mBGSettingSent;  // Last sent value to remote server
  BufferKeys
      mPendingKeyIn;  // Keys pressed received. Results of 2 CmdInputs are
                      // concatenated if received before being processed
  TimePoint mLastOutgoingDrawCheckTime;  // When we last checked if we have a
                                         // pending draw command to send
  TimePoint mLastOutgoingDrawTime;  // When we last sent an updated draw command
                                    // to the server
  ImVec2 mSavedDisplaySize = {
      0, 0};  // Save original display size on 'NewFrame' and restore it on
              // 'EndFrame' (making sure size is still valid after a disconnect)
  SavedImguiContext mSavedContextValues;  // Oiginal ImGui context values that
                                          // will be restored on disconnect
  std::atomic_bool mbClientThreadActive;  // True when connected and
                                          // communicating with Server
  std::atomic_bool mbListenThreadActive;  // True when listening from connection
                                          // request from Server
  std::atomic_bool
      mbComInitActive;  // True when attempting to initialize a new connection
  bool mbTrackedTexturesPending =
      false;  // True if there are some pending tracked textures waiting to be
              // removed
  bool mbIsDrawing =
      false;  // We are inside a 'NetImgui::NewFrame' / 'NetImgui::EndFrame'
              // (even if not for a remote draw)
  bool mbIsRemoteDrawing =
      false;  // True if the rendering it meant for the remote netImgui server
  bool mbRestorePending =
      false;  // Original context has had some settings overridden, original
              // values stored in mRestoreXXX
  bool mbInsideHook = false;  // Currently inside ImGui hook callback
  bool mbInsideNewEnd =
      false;  // Currently inside NetImgui::NewFrame() or NetImgui::EndFrame()
              // (prevents recusrive hook call)
  bool mbValidDrawFrame = false;  // If we should forward the drawdata to the
                                  // server at the end of ImGui::Render()
  uint8_t mClientCompressionMode = eCompressionMode::kUseServerSetting;
  bool mServerCompressionEnabled =
      false;  // If Server would like compression to be enabled
              // (mClientCompressionMode value can override this value)
  std::atomic_bool mServerCompressionSkip{
      false};  // Force ignore compression setting for 1 frame
  std::atomic_bool mServerForceConnectEnabled{
      true};  // If another NetImguiServer can take connection away from the one
              // currently active
  ThreadFunctPtr mThreadFunction =
      nullptr;                    // Function to use when laucnhing new threads
  float mFontSavedScaling = 0.f;  // Original Font scaling before our override
                                  // between NewFrame / EndFrame
  float mFontServerScale =
      1.f;                   // Desired Font DPI Scaling by the NetImgui Server
  std::atomic<float> mDesiredFps{
      30.f};  // How often we should update the remote drawing. Received from
              // server
  InputState mPreviousInputState;  // Keeping track of last keyboard/mouse state
  ImGuiID mhImguiHookNewframe = 0;
  ImGuiID mhImguiHookEndframe = 0;
  int mClientTextureIDNext =
      0;  // Next available ID to assign to new Dear ImGui managed textures
  int mDearImguiTextureCount =
      0;  // Keep track of number of Dear ImGui managed texture added (to detect
          // when DearImgui released some)
#if !NETIMGUI_IMGUI_TEXTURES_ENABLED
  const void* mpFontTextureData =
      nullptr;  // Last font texture data send to server (used to detect if font
                // was changed)
  uint64_t mFontTextureID =
      0;  // Used to detect textureID change [Before ImGui 1.92, old Font Atlas]
  FontCreateFuncPtr mFontCreationFunction =
      nullptr;  // Method to call to generate the remote ImGui font. By default,
                // re-use the local font, but this doesn't handle native DPI
                // scaling on remote server. //NOTE: Unused by Dear imGui 1.92+
  bool mbFontUploaded = false;  // Auto detect if font was sent to server
#endif

  // Prevent warnings about implicitly created copy
 protected:
  ClientInfo(const ClientInfo&) = delete;
  ClientInfo(const ClientInfo&&) = delete;
  void operator=(const ClientInfo&) = delete;
};

//=============================================================================
// Main communication loop threads that are run in separate threads
//=============================================================================
void CommunicationsConnect(void* pClientVoid);
void CommunicationsHost(void* pClientVoid);

}  // namespace Client
}  // namespace Internal
}  // namespace NetImgui

#include "NetImgui_Client.inl"
