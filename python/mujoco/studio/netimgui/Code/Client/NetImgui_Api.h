#pragma once

//=================================================================================================
//! @Name		: NetImgui
//=================================================================================================
//! @author		: Sammy Fatnassi
//! @date		: 2026/01/04
//!	@version	: v1.13.0
//! @Details	: For integration info :
//! https://github.com/sammyfreg/netImgui/wiki
//=================================================================================================
#define NETIMGUI_VERSION "1.13.0"  // Release of v 1.13
#define NETIMGUI_VERSION_NUM 11300

//-------------------------------------------------------------------------------------------------
// Deactivate a few warnings to allow Imgui header include
// without generating warnings in maximum level '-Wall'
//-------------------------------------------------------------------------------------------------
#if defined(__clang__)
#pragma clang diagnostic push
// ImGui.h warnings(s)
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored \
    "-Wreserved-identifier"  // Enum values using '__' or member starting with
                             // '_' in imgui.h
// NetImgui_Api.h Warning(s)
#pragma clang diagnostic ignored \
    "-Wzero-as-null-pointer-constant"  // Not using nullptr in case this file is
                                       // used in pre C++11
#elif defined(_MSC_VER)
#pragma warning(push)
// ImGui.h warnings(s)
#pragma warning( \
    disable : 4514)  // 'xxx': unreferenced inline function has been removed
#pragma warning(disable : 4710)  // 'xxx': function not inlined
#pragma warning( \
    disable      \
    : 4820)  // 'xxx': 'yyy' bytes padding added after data member 'zzz'
#pragma warning(disable : 5045)  // Compiler will insert Spectre mitigation for
                                 // memory load if /Qspectre switch specified
#endif

//=================================================================================================
// Include the user config file. It should contain the include for :
// 'imgui.h' : always
// 'imgui_internal.h' when 'NETIMGUI_INTERNAL_INCLUDE' is defined
//=================================================================================================
#ifdef NETIMGUI_IMPLEMENTATION
#define NETIMGUI_INTERNAL_INCLUDE
#include "NetImgui_Config.h"
#undef NETIMGUI_INTERNAL_INCLUDE
#else
#include "NetImgui_Config.h"
#endif

//-------------------------------------------------------------------------------------------------
// If 'NETIMGUI_ENABLED' hasn't been defined yet (in project settings or
// NetImgui_Config.h') we define this library as 'Disabled'
//-------------------------------------------------------------------------------------------------
#ifndef NETIMGUI_ENABLED
#define NETIMGUI_ENABLED 0
#endif

//-------------------------------------------------------------------------------------------------
// NetImgui needs to detect Dear ImGui to be active, otherwise we disable it
// When including this header, make sure imgui.h is included first
// (either always included in NetImgui_config.h or have it included after
// Imgui.h in your cpp)
//-------------------------------------------------------------------------------------------------
#if !defined(IMGUI_VERSION)
#undef NETIMGUI_ENABLED
#define NETIMGUI_ENABLED 0
#endif

//-------------------------------------------------------------------------------------------------
// Control support for native Dear ImGui texture backend support
// At the moment, meant to always be active on recent Dear Imgui version (1.92+)
//-------------------------------------------------------------------------------------------------
#ifndef NETIMGUI_IMGUI_TEXTURES_ENABLED
#ifdef IMGUI_HAS_TEXTURES
#define NETIMGUI_IMGUI_TEXTURES_ENABLED 1
#else
#define NETIMGUI_IMGUI_TEXTURES_ENABLED 0
#endif
#endif

#if NETIMGUI_ENABLED

#include <stdint.h>

//=================================================================================================
// Default Build settings defines values
// Assign default values when not set in user NetImgui_Config.h
//=================================================================================================
//-------------------------------------------------------------------------------------------------
// Prepended to functions signature, for dll export/import
//-------------------------------------------------------------------------------------------------
#ifndef NETIMGUI_API
#define NETIMGUI_API \
  IMGUI_API  // Use same value as defined by Dear ImGui by default
#endif

//-------------------------------------------------------------------------------------------------
// Enable TCP socket 'reuse port' option when opening it as a 'listener'.
// Note:	Can help when unable to open a socket because it wasn't properly
// released after a crash.
//-------------------------------------------------------------------------------------------------
#ifndef NETIMGUI_FORCE_TCP_LISTEN_BINDING
#define NETIMGUI_FORCE_TCP_LISTEN_BINDING \
  0  // Doesn't seem to be needed on Window/Linux
#endif

//-------------------------------------------------------------------------------------------------
// Enable Dear ImGui Callbacks support for BeginFrame/Render automatic
// interception. Note:	Avoid having to replace ImGui::BeginFrame/ImGui::Render
// with in library user code, by
//			'NetImgui::NewFrame/NetImgui::EndFrame'. But prevent
//benefit of skipping frame draw 			when unneeded, that 'NetImgui::NewFrame' can
//provide. 			For more info, consult 'SampleNewFrame.cpp'. 			Needs Dear ImGui 1.81+
//-------------------------------------------------------------------------------------------------
#ifndef NETIMGUI_IMGUI_CALLBACK_ENABLED
#define NETIMGUI_IMGUI_CALLBACK_ENABLED \
  (IMGUI_VERSION_NUM >= 18100)  // Not supported pre Dear ImGui 1.81
#endif

namespace NetImgui {

//=================================================================================================
// List of texture format supported
//=================================================================================================
enum eTexFormat {
  // Match Dear Imgui 1.92 'ImTextureFormat'
  kTexFmtRGBA8,
  kTexFmtA8,

  // Support of 'user defined' texture format.
  // Implementation must be added on both client and Server code.
  // Search for TEXTURE_CUSTOM_SAMPLE for example implementation.
  kTexFmtCustom,

  //
  kTexFmt_Count,
  kTexFmt_Invalid = kTexFmt_Count
};

//=================================================================================================
// Data Compression wanted status
//=================================================================================================
enum eCompressionMode {
  kForceDisable,     // Disable data compression for communications
  kForceEnable,      // Enable data compression for communications
  kUseServerSetting  // Use Server setting for compression (default)
};

//-------------------------------------------------------------------------------------------------
// Function typedefs
//-------------------------------------------------------------------------------------------------
typedef void (*ThreadFunctPtr)(void threadedFunction(void* pClientInfo),
                               void* pClientInfo);
typedef void (*FontCreateFuncPtr)(float PreviousDPIScale, float NewDPIScale);

//=================================================================================================
// Initialize the Network Library
//=================================================================================================
NETIMGUI_API bool Startup(void);

//=================================================================================================
// Free Resources
// Wait until all communication threads have terminated before returning
//=================================================================================================
NETIMGUI_API void Shutdown();

//=================================================================================================
// Establish a connection between the NetImgui server application and this
// client.
//
// Can connect with NetImgui Server application by either reaching it directly
// using 'ConnectToApp' or waiting for Server to reach us after Client called
// 'ConnectFromApp'.
//
// Note:	Start a new communication thread using std::Thread by default,
// but can receive custom
//			thread start function instead (Look at ClientExample
//'CustomCommunicationThread').
//-------------------------------------------------------------------------------------------------
// clientName			: Client name displayed in the Server's clients
// list serverHost			: NetImgui Server Application address
// (Ex1: 127.0.0.2, Ex2: localhost) serverPort			: PortID of the
// NetImgui Server application to connect to clientPort			: PortID
// this Client should wait for connection from Server application threadFunction
// : User provided function to launch new networking thread.
//						  Use
//'DefaultStartCommunicationThread' by default (uses 'std::thread').
// fontCreateFunction	: User provided function to call when the Server expect
// an update of
//						  the font atlas, because of a
//monitor DPI change. When left to nullptr, 						  uses 'ImGuiIO.FontGlobalScale'
//instead to increase text size, 						  with blurier results. 						  NOTE: Not used by Dear
//ImGui 1.92+, unneeded with font update support.
//=================================================================================================
NETIMGUI_API bool ConnectToApp(const char* clientName, const char* serverHost,
                               uint32_t serverPort = kDefaultServerPort,
                               ThreadFunctPtr threadFunction = 0,
                               FontCreateFuncPtr FontCreateFunction = 0);
NETIMGUI_API bool ConnectFromApp(const char* clientName,
                                 uint32_t clientPort = kDefaultClientPort,
                                 ThreadFunctPtr threadFunction = 0,
                                 FontCreateFuncPtr fontCreateFunction = 0);

//=================================================================================================
// Request a disconnect from the NetImguiServer application
//=================================================================================================
NETIMGUI_API void Disconnect(void);

//=================================================================================================
// True if connected to the NetImguiServer application
//=================================================================================================
NETIMGUI_API bool IsConnected(void);

//=================================================================================================
// True if connection request is waiting to be completed. For example, while
// waiting for Server to reach ud after having called 'ConnectFromApp()'
//=================================================================================================
NETIMGUI_API bool IsConnectionPending(void);

//=================================================================================================
// True when Dear ImGui is currently expecting draw commands
// This means that we are between NewFrame() and EndFrame()
//=================================================================================================
NETIMGUI_API bool IsDrawing(void);

//=================================================================================================
// True when we are currently drawing on the NetImguiServer application
// Means that we are between NewFrame() and EndFrame() of drawing for remote
// application
//=================================================================================================
NETIMGUI_API bool IsDrawingRemote(void);

//=================================================================================================
// Send an updated texture used by imgui, to the NetImguiServer application
// Note: To remove a texture, set pData to nullptr
// Note: User needs to provide a valid 'dataSize' when using format
// 'kTexFmtCustom',
//		 can be ignored otherwise
// Note: Can now rely on native Dear ImGui managed texture support to let the
// system handle their
//		 creation/update/destruction automatically, without needing to
//call this function 		 (since Dear ImGui 1.92+. See 'SampleTextures').
//=================================================================================================
NETIMGUI_API void SendDataTexture(ImTextureID textureId, void* pData,
                                  uint16_t width, uint16_t height,
                                  eTexFormat format, uint32_t dataSize = 0);
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
NETIMGUI_API void SendDataTexture(const ImTextureRef& textureRef, void* pData,
                                  uint16_t width, uint16_t height,
                                  eTexFormat format, uint32_t dataSize = 0);
#endif

//=================================================================================================
// Start a new Imgui Frame and wait for Draws commands, using ImContext that was
// active on connect. Returns true if we are awaiting a new ImGui frame.
//
// All ImGui drawing should be skipped when return is false.
//
// Note: This code can be used instead, to know if you should be drawing or not
// :
//			'if( !NetImgui::IsDrawing() )'
//
// Note: If your code cannot handle skipping a ImGui frame, leave
// 'bSupportFrameSkip=false',
//		 and this function will always call 'ImGui::NewFrame()'
//internally and return true
//
// Note: With Dear ImGui 1.81+, you can keep using the
// ImGui::BeginFrame()/Imgui::Render()
//		 without having to use NetImgui::NewFrame()/NetImgui::EndFrame()
//		 (unless wanting to support frame skip)
//=================================================================================================
NETIMGUI_API bool NewFrame(bool bSupportFrameSkip = false);

//=================================================================================================
// Process all receives draws, send them to remote connection and restore the
// ImGui Context
//=================================================================================================
NETIMGUI_API void EndFrame(void);

//=================================================================================================
// Return the context associated to this remote connection. Null when not
// connected.
//=================================================================================================
NETIMGUI_API ImGuiContext* GetContext();

//=================================================================================================
// Set the remote client background color and texture
// Note: If no TextureID is specified, will use the default server texture
//=================================================================================================
NETIMGUI_API void SetBackground(const ImVec4& bgColor);
NETIMGUI_API void SetBackground(const ImVec4& bgColor,
                                const ImVec4& textureTint);
NETIMGUI_API void SetBackground(const ImVec4& bgColor,
                                const ImVec4& textureTint,
                                ImTextureID bgTextureID);
#if NETIMGUI_IMGUI_TEXTURES_ENABLED
NETIMGUI_API void SetBackground(const ImVec4& bgColor,
                                const ImVec4& textureTint,
                                const ImTextureRef& bgTextureRef);
#endif

//=================================================================================================
// Control the data compression for communications between Client/Server
//=================================================================================================
NETIMGUI_API void SetCompressionMode(eCompressionMode eMode);
NETIMGUI_API eCompressionMode GetCompressionMode();

//=================================================================================================
// Helper functions
//=================================================================================================
NETIMGUI_API uint8_t GetTexture_BitsPerPixel(eTexFormat eFormat);
NETIMGUI_API uint32_t GetTexture_BytePerLine(eTexFormat eFormat,
                                             uint32_t pixelWidth);
NETIMGUI_API uint32_t GetTexture_BytePerImage(eTexFormat eFormat,
                                              uint32_t pixelWidth,
                                              uint32_t pixelHeight);
}  // namespace NetImgui

//=================================================================================================
// Optional single include compiling option
// Note: User wanting to avoid adding the few NetImgui sources files to their
// project,
//		 can instead define 'NETIMGUI_IMPLEMENTATION' *once* before
//including 'NetImgui_Api.h' 		 to pull all the needed cpp files alongside for
//compilation
//=================================================================================================
#if defined(NETIMGUI_IMPLEMENTATION)
#include "Private/NetImgui_Api.cpp"
#include "Private/NetImgui_Client.cpp"
#include "Private/NetImgui_CmdPackets_DrawFrame.cpp"
#include "Private/NetImgui_NetworkPosix.cpp"
#include "Private/NetImgui_NetworkUE4.cpp"
#include "Private/NetImgui_NetworkWin32.cpp"
#endif

#endif  // NETIMGUI_ENABLED

//-------------------------------------------------------------------------------------------------
// Re-Enable the Deactivated warnings
//-------------------------------------------------------------------------------------------------
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
