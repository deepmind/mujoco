// Google modifications:
//  - Added #ifndef guards around HAL_API_PLATFORM_* defines so they can be
//    overridden from BUILD-level defines (-D flags).
#pragma once

#include <Private/NetImgui_Network.h>
#include <Private/NetImgui_Shared.h>

//=============================================================================================
// SELECT RENDERING/OS API HERE
//=============================================================================================
#ifndef HAL_API_PLATFORM_WIN32_DX11
#define HAL_API_PLATFORM_WIN32_DX11 1
#endif
#ifndef HAL_API_PLATFORM_GLFW_GL3
#define HAL_API_PLATFORM_GLFW_GL3 0  // Note: Currently doesn't work on VS2026
#endif
#ifndef HAL_API_PLATFORM_SOKOL
#define HAL_API_PLATFORM_SOKOL \
  0  // Sokol Lib needs to be updated to latest Dear ImGui 1.92 support for this
     // to work
#endif
#define HAL_API_RENDERTARGET_INVERT_Y \
  (HAL_API_PLATFORM_GLFW_GL3 ||       \
   HAL_API_PLATFORM_SOKOL)  // Invert client render target Y axis (since OpenGL
                            // start texture UV from BottomLeft instead of
                            // DirectX TopLeft)
//=============================================================================================

// Forward declare
namespace NetImguiServer {
namespace RemoteClient {
struct Client;
}
}  // namespace NetImguiServer
namespace NetImgui {
namespace Internal {
struct CmdTexture;
}
}  // namespace NetImgui

namespace NetImguiServer {
namespace App {
struct WindowPlacement {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  bool isMaximized = false;
};

//=============================================================================================
// Code specific to 'NetImgui Server' application and needed inside platform
// specific code
//=============================================================================================
// Additional initialisation needed by 'NetImGui Server' and not part of default
// ImGui sample code
bool Startup(const char* CmdLine);
// Prepare for shutdown of application
void Shutdown();
// Receive rendering request of each Remote client and output it to their own
// RenderTarget
void UpdateClientDraw();
// Save server window placement (to restore it on next start)
void UpdateWindowPlacement(int x, int y, int w, int h, bool isMaximized);
// Get last server window placement
WindowPlacement GetWindowPlacement();
// Add a new remote client config to our list (to attempt connexion)
bool AddTransientClientConfigFromString(const char* string);
// Initialize the font atlas used by the Serve
void LoadFonts();

// Descriptor of each textures by Server. Format always RGBA8
struct ServerTexture {
  inline ServerTexture() {
    mTexData.Status = ImTextureStatus_Destroyed;
    mTexData.RefCount = 1;
  }
  inline bool IsValid() const {
    return mTexData.Status != ImTextureStatus_WantCreate &&
           mTexData.Status != ImTextureStatus_Destroyed;
  }
  inline void MarkForDelete() {
    mTexData.WantDestroyNextFrame = true;
    mTexData.UnusedFrames = 0;
  }
  ImTextureData mTexData;          // Struct used by backend for texture support
  ImTextureID mClientTexID;        // Client UserID associated with this texture
  uint64_t mCustomData = 0u;       // Memory available to custom command
  uint64_t mLastFrameUsed = 0u;    // Last draw frame this texture was used
                                   // (needed for resources release)
  int32_t mOwnerClientIndex = -1;  // Client that created this texture (if any)
  uint8_t mIsCustom = 0u;  // Format handled by custom version of NetImguiServer
                           // modified by library user
  uint8_t mIsUpdatable = 0u;  // True when textures can be updated (font)
  uint8_t mPadding[6] = {};
};

//=============================================================================================
// Handling of texture data
//=============================================================================================
ServerTexture* CreateTexture(const NetImgui::Internal::CmdTexture& cmdTexture,
                             uint32_t customDataSize);

// Library users can implement their own texture format (on client/server).
// Useful for vidoe streaming, new format, etc.
bool CreateTexture_Custom(ServerTexture& serverTexture,
                          const NetImgui::Internal::CmdTexture& cmdTexture,
                          uint32_t customDataSize);
bool DestroyTexture_Custom(ServerTexture& serverTexture,
                           const NetImgui::Internal::CmdTexture& cmdTexture,
                           uint32_t customDataSize);

//=============================================================================================
// Note (H)ardware (A)bstraction (L)ayer
//		When porting the 'NetImgui Server' application to other
//platform, 		theses are the few functions needed to be supported by each specific
//API that 		are not already supported by de 'Dear ImGui' provided backends
//=============================================================================================
// Additional initialisation that are platform specific
bool HAL_Startup(const char* CmdLine);
// Prepare for shutdown of application, with platform specific code
void HAL_Shutdown();
// Receive a platform specific socket, and return us with info on the connection
bool HAL_GetSocketInfo(NetImgui::Internal::Network::SocketInfo* pClientSocket,
                       char* pOutHostname, size_t HostNameLen, int& outPort);
// Provide the current user setting folder (used to save the shared config file)
const char* HAL_GetUserSettingFolder();
// Return true when new content should be retrieved from Clipboard (avoid
// constantly reading/converting content)
bool HAL_GetClipboardUpdated();
// Receive a ImDrawData drawlist and render it to backbuffer
void HAL_RenderDrawData(ImDrawData* pDrawData);
// Receive a ImDrawData drawlist and request Dear ImGui's backend to output it
// into a texture
void HAL_RenderDrawData(RemoteClient::Client& client, ImDrawData* pDrawData);
// Allocate a RenderTarget that each client will use to output their ImGui
// drawing into.
bool HAL_CreateRenderTarget(uint16_t Width, uint16_t Height, void*& pOutRT,
                            ImTextureData& OutTexture);
// Free a RenderTarget resource
void HAL_DestroyRenderTarget(void*& pOutRT, ImTextureData& OutTexture);
}  // namespace App
}  // namespace NetImguiServer
