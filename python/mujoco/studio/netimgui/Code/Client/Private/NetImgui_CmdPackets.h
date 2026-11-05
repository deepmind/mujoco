// Google modifications:
//  - Zero-initialize CmdInput with memset in its constructor, and check
//    sizeof(CmdInput) with static_assert.
//  - PendingCom: added SizeMax/kMaxCmdSize, GetSizeToReceive() and
//    IsBiggerBufferNeeded() so the command size received from the network can
//    never overflow the receive buffer.
//  - ToPointers() of commands received from the network return false when
//    their offsets or sizes are invalid (added CmdTexture::ToPointers()).
#pragma once

#include <cstring>

#include "NetImgui_CmdPackets_DrawFrame.h"
#include "NetImgui_Shared.h"

namespace NetImgui {
namespace Internal {

// Note: If updating any of these commands data structure, increase
// 'CmdVersion::eVersion'
struct alignas(8) CmdHeader {
  enum class eCommands : uint8_t {
    Version,
    Texture,
    Input,
    DrawFrame,
    Background,
    Clipboard,
    Count
  };
  CmdHeader(eCommands CmdType, uint16_t Size) : mSize(Size), mType(CmdType) {}
  uint32_t mSize = 0;
  eCommands mType = eCommands::Count;
  uint8_t mSent =
      false;  // True when command finished being sent to client or server
  uint8_t mPadding[2] = {};
};

// Used as step 1 of 2 of reading incoming transmission between Client/Server,
// to get header whose size we know
struct alignas(8) CmdPendingRead : public CmdHeader {
  CmdPendingRead() : CmdHeader(eCommands::Count, sizeof(CmdPendingRead)) {}
};

struct alignas(8) CmdVersion : public CmdHeader {
  enum class eVersion : uint32_t {
    Initial = 1,
    NewTextureFormat = 2,
    ImguiVersionInfo =
        3,  // Added Dear Imgui/ NetImgui version info to 'CmdVersion'
    ServerRefactor = 4,  // Change to 'CmdInput' and 'CmdVersion' store size of
                         // 'ImWchar' to make sure they are compatible
    BackgroundCmd = 5,   // Added new command to control background appearance
    ClientName =
        6,  // Increase maximum allowed client name that a program can set
    DataCompression = 7,   // Adding support for data compression between
                           // client/server. Simple low cost delta compressor
                           // (only send difference from previous frame)
    DataCompression2 = 8,  // Improvement to data compression (save corner
                           // position and use SoA for vertices data)
    VertexUVRange = 9,     // Changed vertices UV value range to [0,1] for
                           // increased precision on large font texture
    Imgui_1_87 = 10,       // Added Dear ImGui Input refactor
    OffetPointer =
        11,  // Updated the handling of OffsetPoint. Moved flag bit from last
             // bit to first bit. Addresses and data are always at least 4 bytes
             // aligned, so should never conflict with potential address space
    CustomTexture = 12,  // Added a 'custom' texture format to let user
                         // potentially handle their how format
    DPIScale = 13,       // Server now handle monitor DPI
    Clipboard = 14,      // Added clipboard support between server/client
    ForceReconnect =
        15,  // Server can now take over the connection from another server
    UpdatedComs = 16,      // Faster protocol by removing blocking coms
    RemDisconnect = 17,    // Removed Disconnect command
    ManagedTextures = 18,  // Adding support for Dear Imgui Managed Textures
                           // (introduced in 1.92))
    // Insert new version here

    //--------------------------------
    _count,
    _current = _count - 1
  };
  enum class eFlags : uint8_t {
    IsUnavailable = 0x01,  // Client telling Server it cannot be used
    IsConnected =
        0x02,  // Client telling Server there's already a valid connection (can
               // potentially be taken over if !IsUnavailable)
    ConnectForce = 0x04,      // Server telling Client it want to take over
                              // connection if there's already one
    ConnectExclusive = 0x08,  // Server telling Client that once connected,
                              // others servers should be denied access
  };
  CmdVersion() : CmdHeader(CmdHeader::eCommands::Version, sizeof(CmdVersion)) {}
  char mClientName[64] = {};
  char mImguiVerName[16] = {IMGUI_VERSION};
  char mNetImguiVerName[16] = {NETIMGUI_VERSION};
  eVersion mVersion = eVersion::_current;
  uint32_t mImguiVerID = IMGUI_VERSION_NUM;
  uint32_t mNetImguiVerID = NETIMGUI_VERSION_NUM;
  uint8_t mWCharSize = static_cast<uint8_t>(sizeof(ImWchar));
  uint8_t mFlags = 0;
  uint8_t PADDING[2] = {};
};

struct alignas(8) CmdInput : public CmdHeader {
  // Identify a mouse button.
  // Those values are guaranteed to be stable and we frequently use 0/1
  // directly. Named enums provided for convenience.
  enum NetImguiMouseButton {
    ImGuiMouseButton_Left = 0,
    ImGuiMouseButton_Right = 1,
    ImGuiMouseButton_Middle = 2,
    ImGuiMouseButton_Extra1 = 3,  // Additional entry
    ImGuiMouseButton_Extra2 = 4,  // Additional entry
    ImGuiMouseButton_COUNT = 5
  };

  // Copy of Dear ImGui key enum
  // We keep our own internal version, to make sure Client key is the same as
  // Server Key (since they can have different Imgui version)
  enum NetImguiKeys {
    // Keyboard
    ImGuiKey_Tab,
    ImGuiKey_LeftArrow,
    ImGuiKey_RightArrow,
    ImGuiKey_UpArrow,
    ImGuiKey_DownArrow,
    ImGuiKey_PageUp,
    ImGuiKey_PageDown,
    ImGuiKey_Home,
    ImGuiKey_End,
    ImGuiKey_Insert,
    ImGuiKey_Delete,
    ImGuiKey_Backspace,
    ImGuiKey_Space,
    ImGuiKey_Enter,
    ImGuiKey_Escape,
    ImGuiKey_LeftCtrl,
    ImGuiKey_LeftShift,
    ImGuiKey_LeftAlt,
    ImGuiKey_LeftSuper,
    ImGuiKey_RightCtrl,
    ImGuiKey_RightShift,
    ImGuiKey_RightAlt,
    ImGuiKey_RightSuper,
    ImGuiKey_Menu,
    ImGuiKey_0,
    ImGuiKey_1,
    ImGuiKey_2,
    ImGuiKey_3,
    ImGuiKey_4,
    ImGuiKey_5,
    ImGuiKey_6,
    ImGuiKey_7,
    ImGuiKey_8,
    ImGuiKey_9,
    ImGuiKey_A,
    ImGuiKey_B,
    ImGuiKey_C,
    ImGuiKey_D,
    ImGuiKey_E,
    ImGuiKey_F,
    ImGuiKey_G,
    ImGuiKey_H,
    ImGuiKey_I,
    ImGuiKey_J,
    ImGuiKey_K,
    ImGuiKey_L,
    ImGuiKey_M,
    ImGuiKey_N,
    ImGuiKey_O,
    ImGuiKey_P,
    ImGuiKey_Q,
    ImGuiKey_R,
    ImGuiKey_S,
    ImGuiKey_T,
    ImGuiKey_U,
    ImGuiKey_V,
    ImGuiKey_W,
    ImGuiKey_X,
    ImGuiKey_Y,
    ImGuiKey_Z,
    ImGuiKey_F1,
    ImGuiKey_F2,
    ImGuiKey_F3,
    ImGuiKey_F4,
    ImGuiKey_F5,
    ImGuiKey_F6,
    ImGuiKey_F7,
    ImGuiKey_F8,
    ImGuiKey_F9,
    ImGuiKey_F10,
    ImGuiKey_F11,
    ImGuiKey_F12,
    ImGuiKey_F13,
    ImGuiKey_F14,
    ImGuiKey_F15,
    ImGuiKey_F16,
    ImGuiKey_F17,
    ImGuiKey_F18,
    ImGuiKey_F19,
    ImGuiKey_F20,
    ImGuiKey_F21,
    ImGuiKey_F22,
    ImGuiKey_F23,
    ImGuiKey_F24,
    ImGuiKey_Apostrophe,    // '
    ImGuiKey_Comma,         // ,
    ImGuiKey_Minus,         // -
    ImGuiKey_Period,        // .
    ImGuiKey_Slash,         // /
    ImGuiKey_Semicolon,     // ;
    ImGuiKey_Equal,         // =
    ImGuiKey_LeftBracket,   // [
    ImGuiKey_Backslash,     // \ (this text inhibit multiline comment caused by
                            // backslash)
    ImGuiKey_RightBracket,  // ]
    ImGuiKey_GraveAccent,   // `
    ImGuiKey_CapsLock,
    ImGuiKey_ScrollLock,
    ImGuiKey_NumLock,
    ImGuiKey_PrintScreen,
    ImGuiKey_Pause,
    ImGuiKey_Keypad0,
    ImGuiKey_Keypad1,
    ImGuiKey_Keypad2,
    ImGuiKey_Keypad3,
    ImGuiKey_Keypad4,
    ImGuiKey_Keypad5,
    ImGuiKey_Keypad6,
    ImGuiKey_Keypad7,
    ImGuiKey_Keypad8,
    ImGuiKey_Keypad9,
    ImGuiKey_KeypadDecimal,
    ImGuiKey_KeypadDivide,
    ImGuiKey_KeypadMultiply,
    ImGuiKey_KeypadSubtract,
    ImGuiKey_KeypadAdd,
    ImGuiKey_KeypadEnter,
    ImGuiKey_KeypadEqual,
    ImGuiKey_AppBack,  // Available on some keyboard/mouses. Often referred as
                       // "Browser Back"
    ImGuiKey_AppForward,
    ImGuiKey_Oem102,  // Non-US backslash.

    //                              // XBOX        | SWITCH  | PLAYSTA. | ->
    //                              ACTION
    ImGuiKey_GamepadStart,     // Menu        | +       | Options  |
    ImGuiKey_GamepadBack,      // View        | -       | Share    |
    ImGuiKey_GamepadFaceLeft,  // X           | Y       | Square   | Tap: Toggle
                               // Menu. Hold: Windowing mode (Focus/Move/Resize
                               // windows)
    ImGuiKey_GamepadFaceRight,  // B           | A       | Circle   | Cancel /
                                // Close / Exit
    ImGuiKey_GamepadFaceUp,  // Y           | X       | Triangle | Text Input /
                             // On-screen Keyboard
    ImGuiKey_GamepadFaceDown,   // A           | B       | Cross    | Activate /
                                // Open / Toggle / Tweak
    ImGuiKey_GamepadDpadLeft,   // D-pad Left  | "       | "        | Move /
                                // Tweak / Resize Window (in Windowing mode)
    ImGuiKey_GamepadDpadRight,  // D-pad Right | "       | "        | Move /
                                // Tweak / Resize Window (in Windowing mode)
    ImGuiKey_GamepadDpadUp,  // D-pad Up    | "       | "        | Move / Tweak
                             // / Resize Window (in Windowing mode)
    ImGuiKey_GamepadDpadDown,  // D-pad Down  | "       | "        | Move /
                               // Tweak / Resize Window (in Windowing mode)
    ImGuiKey_GamepadL1,  // L Bumper    | L       | L1       | Tweak Slower /
                         // Focus Previous (in Windowing mode)
    ImGuiKey_GamepadR1,  // R Bumper    | R       | R1       | Tweak Faster /
                         // Focus Next (in Windowing mode)
    ImGuiKey_GamepadL2,  // L Trigger   | ZL      | L2       | [Analog]
    ImGuiKey_GamepadR2,  // R Trigger   | ZR      | R2       | [Analog]
    ImGuiKey_GamepadL3,  // L Stick     | L3      | L3       |
    ImGuiKey_GamepadR3,  // R Stick     | R3      | R3       |
    ImGuiKey_GamepadLStickLeft,   //             |         |          | [Analog]
                                  //             Move Window (in Windowing mode)
    ImGuiKey_GamepadLStickRight,  //             |         |          | [Analog]
                                  //             Move Window (in Windowing mode)
    ImGuiKey_GamepadLStickUp,     //             |         |          | [Analog]
                                  //             Move Window (in Windowing mode)
    ImGuiKey_GamepadLStickDown,   //             |         |          | [Analog]
                                  //             Move Window (in Windowing mode)
    ImGuiKey_GamepadRStickLeft,   //             |         |          | [Analog]
    ImGuiKey_GamepadRStickRight,  //             |         |          | [Analog]
    ImGuiKey_GamepadRStickUp,     //             |         |          | [Analog]
    ImGuiKey_GamepadRStickDown,   //             |         |          | [Analog]

    // Mouse Buttons (auto-submitted from AddMouseButtonEvent() calls)
    // - This is mirroring the data also written to io.MouseDown[],
    // io.MouseWheel, in a format allowing them to be accessed via standard key
    // API.
    ImGuiKey_MouseLeft,
    ImGuiKey_MouseRight,
    ImGuiKey_MouseMiddle,
    ImGuiKey_MouseX1,
    ImGuiKey_MouseX2,
    ImGuiKey_MouseWheelX,
    ImGuiKey_MouseWheelY,

    // Keyboard Modifiers (explicitly submitted by backend via AddKeyEvent()
    // calls)
    // - This is mirroring the data also written to io.KeyCtrl, io.KeyShift,
    // io.KeyAlt, io.KeySuper, in a format allowing
    //   them to be accessed via standard key API, allowing calls such as
    //   IsKeyPressed(), IsKeyReleased(), querying duration etc.
    // - Code polling every keys (e.g. an interface to detect a key press for
    // input mapping) might want to ignore those
    //   and prefer using the real keys (e.g. ImGuiKey_LeftCtrl,
    //   ImGuiKey_RightCtrl instead of ImGuiKey_ModCtrl).
    // - In theory the value of keyboard modifiers should be roughly equivalent
    // to a logical or of the equivalent left/right keys.
    //   In practice: it's complicated; mods are often provided from different
    //   sources. Keyboard layout, IME, sticky keys and backends tend to
    //   interfere and break that equivalence. The safer decision is to relay
    //   that ambiguity down to the end-user...
    ImGuiKey_ReservedForModCtrl,
    ImGuiKey_ReservedForModShift,
    ImGuiKey_ReservedForModAlt,
    ImGuiKey_ReservedForModSuper,

    // End of list
    ImGuiKey_COUNT,  // No valid ImGuiKey is ever greater than this value
  };

  static constexpr uint32_t kAnalog_First = ImGuiKey_GamepadLStickLeft;
  static constexpr uint32_t kAnalog_Last = ImGuiKey_GamepadRStickDown;
  static constexpr uint32_t kAnalog_Count = kAnalog_Last - kAnalog_First + 1;

  CmdInput() : CmdHeader(CmdHeader::eCommands::Input, sizeof(CmdInput)) {
    memset(static_cast<void*>(this), 0, sizeof(*this));
    mSize = sizeof(CmdInput);
    mType = CmdHeader::eCommands::Input;
    mFontDPIScaling = 1.f;
    mDesiredFps = 30.f;
  }
  uint16_t mScreenSize[2];
  int16_t mMousePos[2];
  float mMouseWheelVert;
  float mMouseWheelHoriz;
  uint16_t mKeyChars[256];  // Input characters
  uint16_t mKeyCharCount;   // Number of valid input characters
  bool mCompressionUse;     // Server would like client to compress the
                            // communication data
  bool mCompressionSkip;  // Server forcing next client's frame data to be
                          // uncompressed
  float mFontDPIScaling;  // Font scaling request by Server accounting for
                          // monitor DPI
  float mDesiredFps;      // Requested redraw speed
  uint64_t mMouseDownMask;
  uint64_t mInputDownMask[(ImGuiKey_COUNT + 63) / 64];
  float mInputAnalog[kAnalog_Count];
  inline bool IsKeyDown(NetImguiKeys netimguiKey) const;
};
static_assert(sizeof(CmdInput) == 616,
              "CmdInput is sent over the network as is");

struct alignas(8) CmdTexture : public CmdHeader {
  enum class eType : uint8_t { Create, Update, Destroy };
  CmdTexture() : CmdHeader(CmdHeader::eCommands::Texture, sizeof(CmdTexture)) {}
  ClientTextureID mTextureClientID = 0;
  eType mStatus = eType::Create;
  uint8_t mFormat = eTexFormat::kTexFmt_Invalid;  // eTexFormat
  uint8_t mUpdatable = false;  // Set to true on Create, for updatable textures
  uint8_t mIsDearImGuiManaged =
      false;  // True if this is not an user created/managed texture
  uint16_t mWidth =
      0;  // Either the texture width on create, or the update area width
  uint16_t mHeight =
      0;  // Either the texture height on create, or the update area height
  uint16_t mOffsetX = 0;  // Used by partial update
  uint16_t mOffsetY = 0;  // Used by partial update
  uint8_t PADDING[4] = {};
  alignas(8) CmdTexture* mpNext =
      nullptr;  // Used for single linked list of pending textures (alignas
                // needed to keep class size the same between win32/x64)
  OffsetPointer<uint8_t> mpTextureData;
  inline bool ToPointers();
};

struct alignas(8) CmdDrawFrame : public CmdHeader {
  CmdDrawFrame()
      : CmdHeader(CmdHeader::eCommands::DrawFrame, sizeof(CmdDrawFrame)) {}
  uint64_t mFrameIndex = 0;
  uint32_t mMouseCursor = 0;  // ImGuiMouseCursor value
  float mDisplayArea[4] = {};
  uint32_t mIndiceByteSize = 0;
  uint32_t mDrawGroupCount = 0;
  uint32_t mTotalVerticeCount = 0;
  uint32_t mTotalIndiceCount = 0;
  uint32_t mTotalDrawCount = 0;
  uint32_t mUncompressedSize = 0;
  uint8_t mCompressed = false;
  uint8_t PADDING[3] = {};
  OffsetPointer<ImguiDrawGroup> mpDrawGroups;
  inline bool ToPointers();
  inline void ToOffsets();
};

struct alignas(8) CmdBackground : public CmdHeader {
  CmdBackground()
      : CmdHeader(CmdHeader::eCommands::Background, sizeof(CmdBackground)) {}
  static constexpr uint64_t kDefaultTexture = ~0u;
  float mClearColor[4] = {0.2f, 0.2f, 0.2f, 1.f};  // Background color
  float mTextureTint[4] = {1.f, 1.f, 1.f,
                           0.5f};         // Tint/alpha applied to texture
  uint64_t mTextureId = kDefaultTexture;  // Texture rendered in background, use
                                          // server texture by default
  inline bool operator==(const CmdBackground& cmp) const;
  inline bool operator!=(const CmdBackground& cmp) const;
};

struct alignas(8) CmdClipboard : public CmdHeader {
  CmdClipboard()
      : CmdHeader(CmdHeader::eCommands::Clipboard, sizeof(CmdClipboard)) {}
  size_t mByteSize = 0;
  OffsetPointer<char> mContentUTF8;
  inline bool ToPointers();
  inline void ToOffsets();
  inline static CmdClipboard* Create(const char* clipboard);
};

//=============================================================================
// Keeping track of partial incoming/outgoing transmissions
//=============================================================================
struct PendingCom {
  // Commands claiming a bigger size are rejected (protects against bad data)
  static constexpr uint32_t kMaxCmdSize = 256u * 1024u * 1024u;
  size_t SizeCurrent = 0;  // Amount of data sent or received so far
  size_t SizeMax = 0;      // Capacity of 'pCommand' buffer (when receiving)
  bool bAutoFree = false;  // Need to free data buffer at the end of processing
  bool bError = false;     // If an error occurs during coms
  CmdHeader* pCommand =
      nullptr;  // Where to store incoming data or read to send data
  inline bool IsError() const { return bError; }
  inline bool IsDone() const {
    return IsError() || (pCommand && SizeCurrent >= sizeof(CmdHeader) &&
                         pCommand->mSize == SizeCurrent);
  }
  inline bool IsReady() const { return !IsError() && pCommand == nullptr; }
  inline bool IsPending() const {
    return !IsError() && !IsDone() && !IsReady();
  }
  // Receiving a command: the header is received first (into a CmdHeader sized
  // buffer), then the caller allocates a buffer for the command size it holds.
  // Both methods validate that size, since it comes from the network.
  //
  // Amount of data left to receive. Flags an error when the command size is
  // smaller than the data already received or exceeds the buffer 'SizeMax'.
  size_t GetSizeToReceive() {
    const size_t sizeExpected = SizeCurrent < sizeof(CmdHeader)
                                    ? sizeof(CmdHeader)
                                    : pCommand->mSize;
    bError |= sizeExpected < SizeCurrent || sizeExpected > SizeMax;
    return bError ? 0 : sizeExpected - SizeCurrent;
  }
  // True once the header of a command bigger than the header buffer has been
  // received: the caller must allocate 'pCommand->mSize' bytes for it (and
  // update 'SizeMax'). Flags an error when that size is invalid.
  bool IsBiggerBufferNeeded() {
    const bool bHeaderDone =
        SizeCurrent == sizeof(CmdHeader) && SizeMax == sizeof(CmdHeader);
    bError |= bHeaderDone && (pCommand->mSize < sizeof(CmdHeader) ||
                              pCommand->mSize > kMaxCmdSize);
    return !bError && bHeaderDone && pCommand->mSize > sizeof(CmdHeader);
  }
};

}  // namespace Internal
}  // namespace NetImgui

#include "NetImgui_CmdPackets.inl"
