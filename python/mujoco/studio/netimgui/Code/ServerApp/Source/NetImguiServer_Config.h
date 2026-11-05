// Google modifications:
// - Added #include <chrono> for std::chrono::steady_clock::time_point

#pragma once

#include <chrono>
#include <stdint.h>

namespace NetImguiServer {
namespace Config {

//=================================================================================================
// Client Configs are used by this server to reach remote netImgui Clients.
//
// Note:For multihreading safety, we are always working with copies of the
// original data.
//
// Note:It is also possible for a remote netImgui client to connect to Server
// directly,
//		in which case they don't need to have an associated config.
//=================================================================================================
class Client {
 public:
  using RuntimeID = uint32_t;
  using TimeStamp = std::chrono::steady_clock::time_point;
  static constexpr RuntimeID kInvalidRuntimeID = static_cast<RuntimeID>(0);

  enum class eVersion : uint32_t {
    Initial = 1,          // First version save file deployed
    Refresh = 2,          // Added refresh rate support
    DPIScale = 3,         // Added DPI scaling
    BlockTakeOver = 4,    // Added Takeover Block
    WindowPlacement = 5,  // Added saving of main window location and size
    _Count,
    _Latest = _Count - 1
  };
  enum class eConfigType : uint8_t {
    Pending,   // New config, will try saving it in the local config
    Local,     // Config fetched from local config file, in the current working
               // directory
    Local2nd,  // Config fetched from a second local config file, in the current
               // working directory. Used when 'Local' file is read only
    Shared,    // Config fetched from the shared user folder
    Transient,  // Config created from connection request (command line, OS
                // pipes), cannot be saved
  };
  enum class eStatus : uint8_t {
    Disconnected,  // No connection detected on client
    Connecting,    // This server is connecting to client
    Connected,     // This server is connect to client
    Available,     // Client already taken, but this server can take over
    ErrorBusy,     // Client already taken
    ErrorVer,      // Server/Client network api mismatch
  };

  // Config settings
  char mClientName[128];    //!< Client display name
  char mHostName[128];      //!< Client IP or remote host address to attempt
                            //!< connection at
  uint32_t mHostPort;       //!< Client Port to attempt connection at
  RuntimeID mRuntimeID;     //!< Unique RuntimeID used to find this Config
  eConfigType mConfigType;  //!< Type of the configuration
  bool mDPIScaleEnabled;    //!< Enable support of Font DPI scaling requests by
                            //!< Server
  bool mBlockTakeover;  //!< If another NetImguiServer is allowed to forcefully
                        //!< disconnect this client to connect to it
  bool mReadOnly;       //!< Config comes from read only file, can't be modified
  bool mConnectAuto;    //!< Try automatically connecting to client

  // Transient values used while running
  mutable bool
      mConnectRequest;  //!< Attempt connecting to Client, after user request
  mutable bool mConnectForce;      //!< Attempt connecting to Client, after user
                                   //!< request, even if already connected
  mutable eStatus mConnectStatus;  //!< Connection status of associated client
  mutable TimeStamp mConnectLastTime;  //!< Last connection attempt time (avoid
                                       //!< quickly retrying same client)

  // Access methods
 public:
  Client();

  inline bool IsReadOnly() const { return mReadOnly; };
  inline bool IsTransient() const {
    return mConfigType == eConfigType::Transient;
  };
  inline bool IsConnected() const {
    return mConnectStatus == eStatus::Connected;
  }
  inline bool IsConnecting() const {
    return mConnectStatus == eStatus::Connecting || mConnectRequest ||
           mConnectForce;
  }
  inline bool IsConnectReady() const {
    bool bAvailable = mConnectStatus != eStatus::Connected &&
                      mConnectStatus != eStatus::Connecting;
    return bAvailable && (mConnectRequest || mConnectForce);
  }
  inline bool IsAutoConnectReady() const {
    bool bAvailable = mConnectStatus != eStatus::Connected &&
                      mConnectStatus != eStatus::Connecting;
    auto elapsedTime = std::chrono::steady_clock::now() - mConnectLastTime;
    int durationSec = static_cast<int>(
        std::chrono::duration_cast<std::chrono::seconds>(elapsedTime).count());
    bool elapseOk =
        durationSec > (mConnectStatus == eStatus::Disconnected ? 5 : 30);
    return bAvailable && mConnectAuto && elapseOk;
  }

  // Add/Edit/Remove config
  static void SetConfig(
      const Client& config);  //!< Add or replace a client configuration info
  static void DelConfig(uint32_t configID);  //!< Remove a client configuration
  static bool GetConfigByID(
      uint32_t configID,
      Client& outConfig);  //!< Find client configuration with this id (return
                           //!< true if found)
  static bool GetConfigByIndex(
      uint32_t index,
      Client& outConfig);  //!< Find client configuration at the x position
  static uint32_t GetConfigCount();

  // Set property value directly (without having to copy entire structure)
  static bool GetProperty_BlockTakeover(uint32_t configID);
  static void SetProperty_Status(uint32_t configID, eStatus Status);
  static void SetProperty_ConnectAuto(uint32_t configID, bool value);
  static void SetProperty_ConnectRequest(uint32_t configID, bool value,
                                         bool force);

  // Client Config list management
  static void SaveAll();
  static void LoadAll();
  static void Clear();

 protected:
  static void SaveConfigFile(eConfigType fileConfigType,
                             bool writeServerSettings);
  static void LoadConfigFile(eConfigType fileConfigType);
  inline bool ShouldSave(eConfigType fileConfigType) const;
};

struct Server {
  static uint32_t sPort;  //!< Port that Server should use for connection.
                          //!< (Note: not really a 'Client' setting, but easier
                          //!< to just bundle the value here for the moment)
  static float sRefreshFPSActive;    //!< Refresh rate of active Window
  static float sRefreshFPSInactive;  //!< Refresh rate of inactive Window
  static float
      sDPIScaleRatio;  //!< Ratio of DPI scale applied to Font size (helps with
                       //!< high resolution monitor, default 1.0)
  static bool sCompressionEnable;  //!< Ask the clients to compress their data
                                   //!< before transmission
  static float sFontSize;          //!< Font size used for Server UI
  static int sWindowPlacement[4];  //!< Main window position and size
                                   //!< (x,y,width,height)
  static bool sWindowMaximized;
};

}  // namespace Config
}  // namespace NetImguiServer
