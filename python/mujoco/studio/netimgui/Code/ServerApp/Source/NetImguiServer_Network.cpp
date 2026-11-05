// Google modifications:
//  - Receive loop and handshake validate the command size received from the
//    network (PendingCom::SizeMax / IsBiggerBufferNeeded).
//  - Incoming commands are validated (size, offsets) before being used.
#include "NetImguiServer_Network.h"

#include <Private/NetImgui_CmdPackets.h>
#include <Private/NetImgui_Network.h>

#include <thread>

#include "NetImguiServer_App.h"
#include "NetImguiServer_Config.h"
#include "NetImguiServer_RemoteClient.h"
#include "Private/NetImgui_WarningDisableStd.h"

using namespace NetImgui::Internal;

namespace NetImguiServer {
namespace Network {
using atomic_SocketInfo = std::atomic<::Network::SocketInfo*>;
static bool gbShutdown(false);  // Set to true when NetImguiServer exiting
static atomic_SocketInfo gListenSocket(
    nullptr);  // Need global access to kill socket on shutdown
static std::atomic_uint32_t gActiveClientThreadCount(
    0);  // How many active client connection currently running
static std::atomic_bool gActiveThreadConnectOut(
    false);  // True while Server is still trying to connect to new clients
static std::atomic_bool gActiveThreadConnectIn(
    false);  // True while Server is still trying to receive connection from new
             // clients
static std::atomic_uint64_t gStatsDataSent(0);
static std::atomic_uint64_t gStatsDataRcvd(0);

//=================================================================================================
// (IN) COMMAND TEXTURE
//=================================================================================================
void Communications_Incoming_CmdTexture(RemoteClient::Client& Client) {
  auto pCmdTexture = reinterpret_cast<NetImgui::Internal::CmdTexture*>(
      Client.mPendingRcv.pCommand);
  if (!pCmdTexture->ToPointers()) {
    Client.mPendingRcv.bError = true;  // Invalid command received
    return;
  }
  Client.mPendingRcv.bAutoFree = false;  // Taking ownership of the data

  // For debug tracking
  uint32_t idx =
      Client.mTextureHistoryIndex % IM_ARRAYSIZE(Client.mTextureHistory);
  Client.mTextureHistory[idx].UpdateId = Client.mTextureHistoryIndex++;
  Client.mTextureHistory[idx].Frame = Client.mLastDrawFrameIndex;
  Client.mTextureHistory[idx].ClientId = pCmdTexture->mTextureClientID;
  Client.mTextureHistory[idx].Format = pCmdTexture->mFormat;
  Client.mTextureHistory[idx].isCreate =
      pCmdTexture->mStatus == CmdTexture::eType::Create;
  Client.mTextureHistory[idx].isDestroy =
      pCmdTexture->mStatus == CmdTexture::eType::Destroy;
  Client.mTextureHistory[idx].isUpdate =
      pCmdTexture->mStatus == CmdTexture::eType::Update;
  Client.mTextureHistory[idx].isDearImguiManaged =
      pCmdTexture->mIsDearImGuiManaged != 0;
  Client.mTextureHistory[idx].x = pCmdTexture->mOffsetX;
  Client.mTextureHistory[idx].y = pCmdTexture->mOffsetY;
  Client.mTextureHistory[idx].w = pCmdTexture->mWidth;
  Client.mTextureHistory[idx].h = pCmdTexture->mHeight;
  Client.ReceiveTexture(pCmdTexture);
}

//=================================================================================================
// (IN) COMMAND BACKGROUND
//=================================================================================================
void Communications_Incoming_CmdBackground(RemoteClient::Client& Client) {
  auto pCmdBackground = reinterpret_cast<NetImgui::Internal::CmdBackground*>(
      Client.mPendingRcv.pCommand);
  if (pCmdBackground->mSize < sizeof(NetImgui::Internal::CmdBackground)) {
    Client.mPendingRcv.bError = true;  // Invalid command received
    return;
  }
  Client.mPendingRcv.bAutoFree = false;  // Taking ownership of the data
  Client.mPendingBackgroundIn.Assign(pCmdBackground);
}

//=================================================================================================
// (IN) COMMAND DRAW FRAME
//=================================================================================================
void Communications_Incoming_CmdDrawFrame(RemoteClient::Client& Client) {
  auto pCmdDraw = reinterpret_cast<NetImgui::Internal::CmdDrawFrame*>(
      Client.mPendingRcv.pCommand);
  if (!pCmdDraw->ToPointers()) {
    Client.mPendingRcv.bError = true;  // Invalid command received
    return;
  }
  Client.mPendingRcv.bAutoFree = false;  // Taking ownership of the data
  Client.ReceiveDrawFrame(pCmdDraw);
}

//=================================================================================================
// (IN) COMMAND CLIPBOARD
//=================================================================================================
void Communications_Incoming_CmdClipboard(RemoteClient::Client& Client) {
  auto pCmdClipboard = reinterpret_cast<NetImgui::Internal::CmdClipboard*>(
      Client.mPendingRcv.pCommand);
  if (!pCmdClipboard->ToPointers()) {
    Client.mPendingRcv.bError = true;  // Invalid command received
    return;
  }
  Client.mPendingRcv.bAutoFree = false;  // Taking ownership of the data
  Client.mPendingClipboardIn.Assign(pCmdClipboard);
}

//=================================================================================================
// Receive every commands sent by remote client and process them
//=================================================================================================
void Communications_Incoming(RemoteClient::Client& Client) {
  if (::Network::DataReceivePending(Client.mpSocket)) {
    //-----------------------------------------------------------------------------------------
    // 1. Ready to receive new command, starts the process by reading Header
    //-----------------------------------------------------------------------------------------
    if (Client.mPendingRcv.IsReady()) {
      Client.mCmdPendingRead = NetImgui::Internal::CmdPendingRead();
      Client.mPendingRcv.pCommand = &Client.mCmdPendingRead;
      Client.mPendingRcv.SizeMax = sizeof(Client.mCmdPendingRead);
      Client.mPendingRcv.bAutoFree = false;
    }

    //-----------------------------------------------------------------------------------------
    // 2. Read incoming command from server
    //-----------------------------------------------------------------------------------------
    if (Client.mPendingRcv.IsPending()) {
      ::Network::DataReceive(Client.mpSocket, Client.mPendingRcv);

      // Detected a new command bigger than header, allocate memory for it
      if (Client.mPendingRcv.IsBiggerBufferNeeded()) {
        CmdPendingRead* pCmdHeader =
            reinterpret_cast<NetImgui::Internal::CmdPendingRead*>(
                netImguiSizedNew<uint8_t>(Client.mPendingRcv.pCommand->mSize));
        *pCmdHeader = Client.mCmdPendingRead;
        Client.mPendingRcv.pCommand = pCmdHeader;
        Client.mPendingRcv.SizeMax = pCmdHeader->mSize;
        Client.mPendingRcv.bAutoFree = true;
      }
    }

    //-----------------------------------------------------------------------------------------
    // 3. Command fully received from Server, process it
    //-----------------------------------------------------------------------------------------
    if (Client.mPendingRcv.IsDone()) {
      if (!Client.mPendingRcv.IsError()) {
        Client.mStatsDataRcvd += Client.mPendingRcv.pCommand->mSize;
        Client.mLastIncomingComTime = std::chrono::steady_clock::now();
        switch (Client.mPendingRcv.pCommand->mType) {
          case NetImgui::Internal::CmdHeader::eCommands::Texture:
            Communications_Incoming_CmdTexture(Client);
            break;
          case NetImgui::Internal::CmdHeader::eCommands::Background:
            Communications_Incoming_CmdBackground(Client);
            break;
          case NetImgui::Internal::CmdHeader::eCommands::DrawFrame:
            Communications_Incoming_CmdDrawFrame(Client);
            break;
          case NetImgui::Internal::CmdHeader::eCommands::Clipboard:
            Communications_Incoming_CmdClipboard(Client);
            break;
            // Commands not received in main loop, by Server
          case NetImgui::Internal::CmdHeader::eCommands::Version:
          case NetImgui::Internal::CmdHeader::eCommands::Input:
          case NetImgui::Internal::CmdHeader::eCommands::Count:
            break;
        }
      }

      // Reset pending read
      if (Client.mPendingRcv.IsError()) {
        Client.mbDisconnectPending = true;
      }
      if (Client.mPendingRcv.bAutoFree) {
        netImguiDeleteSafe(Client.mPendingRcv.pCommand);
      }
      Client.mPendingRcv = PendingCom();
    }
  }
  // Prevent high CPU usage when waiting for new data
  else {
    // std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::microseconds(250));
  }
}

//=================================================================================================
// Send the updates to RemoteClient
// Ends with a Ping Command (signal a end of commands)
//=================================================================================================
void Communications_Outgoing(RemoteClient::Client& Client) {
  //---------------------------------------------------------------------------------------------
  // Try finishing sending a pending command to Server
  //---------------------------------------------------------------------------------------------
  if (Client.mPendingSend.IsPending()) {
    ::Network::DataSend(Client.mpSocket, Client.mPendingSend);

    // Free allocated memory for command
    if (Client.mPendingSend.IsDone()) {
      if (Client.mPendingSend.IsError()) {
        Client.mbDisconnectPending = true;
      }
      Client.mStatsDataSent += Client.mPendingSend.pCommand->mSize;
      if (Client.mPendingSend.bAutoFree) {
        netImguiDeleteSafe(Client.mPendingSend.pCommand);
      }
      Client.mPendingSend = PendingCom();
    }
  }

  if (Client.mPendingSend.IsReady()) {
    NetImgui::Internal::CmdClipboard* pClipboardCmd =
        Client.TakePendingClipboard();
    if (pClipboardCmd) {
      pClipboardCmd->ToOffsets();
      Client.mPendingSend.pCommand = pClipboardCmd;
      Client.mPendingSend.bAutoFree = true;
    }
  }

  if (Client.mPendingSend.IsReady()) {
    NetImgui::Internal::CmdInput* pInputCmd = Client.TakePendingInput();
    Client.mPendingSend.pCommand = pInputCmd;
    Client.mPendingSend.bAutoFree = true;
  }
}

//=================================================================================================
// Update communications stats of a client, after a frame
//=================================================================================================
void Communications_UpdateClientStats(RemoteClient::Client& Client) {
  // Update data transfer stats
  auto elapsedTime = std::chrono::steady_clock::now() - Client.mStatsTime;
  if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime)
          .count() >= 250) {
    constexpr uint64_t kHysteresis = 10;  // out of 100
    uint64_t newDataRcvd = Client.mStatsDataRcvd - Client.mStatsDataRcvdPrev;
    uint64_t newDataSent = Client.mStatsDataSent - Client.mStatsDataSentPrev;
    uint64_t tmMicrosS =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime)
            .count();
    uint32_t newDataRcvdBps =
        static_cast<uint32_t>(newDataRcvd * 1000000u / tmMicrosS);
    uint32_t newDataSentBps =
        static_cast<uint32_t>(newDataSent * 1000000u / tmMicrosS);
    Client.mStatsRcvdBps = (Client.mStatsRcvdBps * (100u - kHysteresis) +
                            newDataRcvdBps * kHysteresis) /
                           100u;
    Client.mStatsSentBps = (Client.mStatsSentBps * (100u - kHysteresis) +
                            newDataSentBps * kHysteresis) /
                           100u;
    gStatsDataRcvd += newDataRcvd;
    gStatsDataSent += newDataSent;

    Client.mStatsTime = std::chrono::steady_clock::now();
    Client.mStatsDataRcvdPrev = Client.mStatsDataRcvd;
    Client.mStatsDataSentPrev = Client.mStatsDataSent;
  }
}

//=================================================================================================
// Keep sending/receiving commands to Remote Client, until disconnection occurs
//=================================================================================================
void Communications_ClientExchangeLoop(RemoteClient::Client* pClient) {
  gActiveClientThreadCount++;

  NetImguiServer::Config::Client::SetProperty_Status(
      pClient->mClientConfigID,
      NetImguiServer::Config::Client::eStatus::Connected);
  pClient->mbDisconnectPending = false;
  pClient->mbIsConnected = true;
  while (!gbShutdown && !pClient->mbDisconnectPending) {
    Communications_Outgoing(*pClient);
    Communications_Incoming(*pClient);
    Communications_UpdateClientStats(*pClient);
  }
  NetImguiServer::Config::Client::SetProperty_Status(
      pClient->mClientConfigID,
      NetImguiServer::Config::Client::eStatus::Disconnected);
  NetImgui::Internal::Network::Disconnect(pClient->mpSocket);
  pClient->Release();
  gActiveClientThreadCount--;
}

//=================================================================================================
// Establish connection with Remote Client
// Makes sure that Server/Client are compatible
//=================================================================================================
bool Communications_InitializeClient(
    NetImgui::Internal::Network::SocketInfo* pClientSocket,
    RemoteClient::Client* pClient, bool ConnectForce) {
  NetImgui::Internal::CmdVersion cmdVersionSend, cmdVersionRcv;
  NetImgui::Internal::PendingCom PendingRcv, PendingSend;

  //---------------------------------------------------------------------
  // Handshake confirming connection validity
  //---------------------------------------------------------------------
  NetImgui::Internal::StringCopy(cmdVersionSend.mClientName, "Server");
  const bool ConnectExclusive =
      NetImguiServer::Config::Client::GetProperty_BlockTakeover(
          pClient->mClientConfigID);
  cmdVersionSend.mFlags |=
      ConnectExclusive
          ? static_cast<uint8_t>(
                NetImgui::Internal::CmdVersion::eFlags::ConnectExclusive)
          : 0;
  cmdVersionSend.mFlags |=
      ConnectForce ? static_cast<uint8_t>(
                         NetImgui::Internal::CmdVersion::eFlags::ConnectForce)
                   : 0;
  PendingSend.pCommand = reinterpret_cast<CmdPendingRead*>(&cmdVersionSend);
  while (!gbShutdown && !PendingSend.IsDone()) {
    ::Network::DataSend(pClientSocket, PendingSend);
  }

  if (!PendingSend.IsError()) {
    PendingRcv.pCommand = reinterpret_cast<CmdPendingRead*>(&cmdVersionRcv);
    PendingRcv.SizeMax = sizeof(cmdVersionRcv);
    while (!PendingRcv.IsDone() &&
           cmdVersionRcv.mType == CmdHeader::eCommands::Version) {
      while (!gbShutdown && !::Network::DataReceivePending(pClientSocket)) {
        std::this_thread::yield();  // Idle until we receive the remote data
      }
      ::Network::DataReceive(pClientSocket, PendingRcv);
    }

    if (!gbShutdown && !PendingRcv.IsError()) {
      //---------------------------------------------------------------------
      // Connection accepted, initialize client
      //---------------------------------------------------------------------
      if (cmdVersionRcv.mSize != sizeof(NetImgui::Internal::CmdVersion) ||
          cmdVersionRcv.mType !=
              NetImgui::Internal::CmdHeader::eCommands::Version ||
          cmdVersionRcv.mVersion !=
              NetImgui::Internal::CmdVersion::eVersion::_current) {
        NetImguiServer::Config::Client::SetProperty_Status(
            pClient->mClientConfigID,
            NetImguiServer::Config::Client::eStatus::ErrorVer);
        return false;
      } else if (cmdVersionRcv.mFlags &
                 static_cast<uint8_t>(
                     NetImgui::Internal::CmdVersion::eFlags::IsConnected)) {
        bool bAvailable =
            (cmdVersionRcv.mFlags &
             static_cast<uint8_t>(
                 NetImgui::Internal::CmdVersion::eFlags::IsUnavailable)) == 0;
        NetImguiServer::Config::Client::SetProperty_Status(
            pClient->mClientConfigID,
            bAvailable ? NetImguiServer::Config::Client::eStatus::Available
                       : NetImguiServer::Config::Client::eStatus::ErrorBusy);
        return false;
      }

      pClient->Initialize();
      pClient->mInfoImguiVerID = cmdVersionRcv.mImguiVerID;
      pClient->mInfoNetImguiVerID = cmdVersionRcv.mNetImguiVerID;
      pClient->mPendingRcv = PendingCom();
      pClient->mPendingSend = PendingCom();
      NetImgui::Internal::StringCopy(pClient->mInfoName,
                                     cmdVersionRcv.mClientName);
      NetImgui::Internal::StringCopy(pClient->mInfoImguiVerName,
                                     cmdVersionRcv.mImguiVerName);
      NetImgui::Internal::StringCopy(pClient->mInfoNetImguiVerName,
                                     cmdVersionRcv.mNetImguiVerName);

      NetImguiServer::Config::Client clientConfig;
      if (NetImguiServer::Config::Client::GetConfigByID(
              pClient->mClientConfigID, clientConfig)) {
        NetImgui::Internal::StringFormat(
            pClient->mWindowID, "%s (%s)##%i", pClient->mInfoName,
            clientConfig.mClientName,
            static_cast<int>(pClient->mClientIndex));  // Using ClientIndex as a
                                                       // window unique ID
      } else {
        NetImgui::Internal::StringFormat(
            pClient->mWindowID, "%s##%i", pClient->mInfoName,
            static_cast<int>(pClient->mClientIndex));  // Using ClientIndex as a
                                                       // window unique ID
      }
      return true;
    }
  }

  NetImguiServer::Config::Client::SetProperty_Status(
      pClient->mClientConfigID,
      NetImguiServer::Config::Client::eStatus::Disconnected);
  return false;
}

//=================================================================================================
// New connection Init request.
// Start new communication thread if handshake sucessfull
//=================================================================================================
void NetworkConnectionNew(
    NetImgui::Internal::Network::SocketInfo* pClientSocket,
    RemoteClient::Client* pNewClient, bool ConnectForce) {
  const char* zErrorMsg(nullptr);
  if (pNewClient == nullptr) {
    zErrorMsg = "Too many connection on server already";
  } else {
    NetImguiServer::Config::Client::SetProperty_Status(
        pNewClient->mClientConfigID,
        NetImguiServer::Config::Client::eStatus::Connecting);
    if (zErrorMsg == nullptr && !gbShutdown &&
        Communications_InitializeClient(pClientSocket, pNewClient,
                                        ConnectForce) == false) {
      zErrorMsg = "Initialization failed. Wrong communication version?";
    }
  }

  if (zErrorMsg == nullptr && !gbShutdown) {
    pNewClient->mpSocket = pClientSocket;
    std::thread(Communications_ClientExchangeLoop, pNewClient).detach();
  } else {
    NetImgui::Internal::Network::Disconnect(pClientSocket);
    if (!gbShutdown) {
      if (pNewClient) {
        pNewClient->mbIsFree = true;
        printf("Error connecting to client '%s:%i' (%s)\n",
               pNewClient->mConnectHost, pNewClient->mConnectPort, zErrorMsg);
      } else {
        printf("Error connecting to client (%s)\n", zErrorMsg);
      }
    }
  }
}

//=================================================================================================
// Thread waiting on new Client Connection request
//=================================================================================================
void NetworkConnectRequest_Receive() {
  uint32_t serverPort(0);

  gActiveThreadConnectIn = true;
  while (!gbShutdown) {
    // Open (and update when needed) listening socket
    if (gListenSocket == nullptr ||
        serverPort != NetImguiServer::Config::Server::sPort) {
      serverPort = NetImguiServer::Config::Server::sPort;
      gListenSocket = NetImgui::Internal::Network::ListenStart(serverPort);
      if (gListenSocket.load() == nullptr) {
        printf("Failed to start connection listen on port : %i", serverPort);
        std::this_thread::sleep_for(std::chrono::milliseconds(
            500));  // Reduce Server listening socket open attempt frequency
      }
    }

    // Detect connection request from Clients
    if (gListenSocket.load() != nullptr) {
      NetImgui::Internal::Network::SocketInfo* pClientSocket =
          NetImgui::Internal::Network::ListenConnect(gListenSocket.load());
      if (pClientSocket) {
        uint32_t freeIndex = RemoteClient::Client::GetFreeIndex();
        if (freeIndex != RemoteClient::Client::kInvalidClient) {
          RemoteClient::Client& newClient =
              RemoteClient::Client::Get(freeIndex);
          newClient.mClientConfigID =
              NetImguiServer::Config::Client::kInvalidRuntimeID;
          newClient.mClientIndex = freeIndex;
          NetImguiServer::App::HAL_GetSocketInfo(
              pClientSocket, newClient.mConnectHost,
              sizeof(newClient.mConnectHost), newClient.mConnectPort);
          NetworkConnectionNew(pClientSocket, &newClient, false);
        } else {
          NetImgui::Internal::Network::Disconnect(pClientSocket);
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }
  NetImgui::Internal::Network::SocketInfo* socketDisconnect =
      gListenSocket.exchange(nullptr);
  NetImgui::Internal::Network::Disconnect(socketDisconnect);
  gActiveThreadConnectIn = false;
}

//=================================================================================================
// Thread trying to reach out new Clients with a connection
//=================================================================================================
void NetworkConnectRequest_Send() {
  uint64_t loopIndex(0);
  NetImguiServer::Config::Client clientConfig;
  gActiveThreadConnectOut = true;
  while (!gbShutdown) {
    uint32_t clientConfigID(NetImguiServer::Config::Client::kInvalidRuntimeID);
    NetImgui::Internal::Network::SocketInfo* pClientSocket = nullptr;

    // Find next client configuration to attempt connection to
    bool ConnectForce = false;
    uint64_t configCount =
        static_cast<uint64_t>(NetImguiServer::Config::Client::GetConfigCount());
    uint32_t configIdx =
        configCount ? static_cast<uint32_t>(loopIndex++ % configCount) : 0;
    if (NetImguiServer::Config::Client::GetConfigByIndex(configIdx,
                                                         clientConfig)) {
      ConnectForce = clientConfig.mConnectForce;
      if ((clientConfig.IsConnectReady() ||
           clientConfig.IsAutoConnectReady()) &&
          clientConfig.mHostPort != NetImguiServer::Config::Server::sPort) {
        NetImguiServer::Config::Client::SetProperty_ConnectRequest(
            clientConfig.mRuntimeID, false,
            false);  // Reset the Connection request, we are processing it
        NetImguiServer::Config::Client::SetProperty_Status(
            clientConfig.mRuntimeID,
            NetImguiServer::Config::Client::eStatus::Disconnected);
        clientConfigID =
            clientConfig.mRuntimeID;  // Keep track of ClientConfig we are
                                      // attempting to connect to
        pClientSocket = NetImgui::Internal::Network::Connect(
            clientConfig.mHostName, clientConfig.mHostPort);
      }
    }

    // Connection successful, find an available client slot
    if (pClientSocket) {
      uint32_t freeIndex = RemoteClient::Client::GetFreeIndex();
      if (freeIndex != RemoteClient::Client::kInvalidClient) {
        RemoteClient::Client& newClient = RemoteClient::Client::Get(freeIndex);
        if (NetImguiServer::Config::Client::GetConfigByID(clientConfigID,
                                                          clientConfig)) {
          NetImgui::Internal::StringCopy(newClient.mInfoName,
                                         clientConfig.mClientName);
          newClient.mConnectPort = clientConfig.mHostPort;
          newClient.mClientConfigID = clientConfigID;
          newClient.mClientIndex = freeIndex;
        }
        NetImguiServer::App::HAL_GetSocketInfo(
            pClientSocket, newClient.mConnectHost,
            sizeof(newClient.mConnectHost), newClient.mConnectPort);
        NetworkConnectionNew(pClientSocket, &newClient, ConnectForce);
      } else {
        NetImgui::Internal::Network::Disconnect(pClientSocket);
        NetImguiServer::Config::Client::SetProperty_Status(
            clientConfigID,
            NetImguiServer::Config::Client::eStatus::Disconnected);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(
        500));  // There's already a wait time in Connect attempt, so no need to
                // sleep for too long here
  }
  gActiveThreadConnectOut = false;
}

//=================================================================================================
// Initialize Networking and start listening thread
//=================================================================================================
bool Startup() {
  // Relying on shared network implementation for Winsock Init
  if (!NetImgui::Internal::Network::Startup()) {
    return false;
  }

  gbShutdown = false;
  gActiveClientThreadCount = 0;
  std::thread(NetworkConnectRequest_Receive).detach();
  std::thread(NetworkConnectRequest_Send).detach();
  return true;
}

//=================================================================================================
// Send signal to terminate all communications and wait until all client have
// been released
//=================================================================================================
void Shutdown() {
  gbShutdown = true;
  NetImgui::Internal::Network::SocketInfo* socketDisconnect =
      gListenSocket.exchange(nullptr);
  NetImgui::Internal::Network::Disconnect(socketDisconnect);
  while (gActiveClientThreadCount > 0 || gActiveThreadConnectIn ||
         gActiveThreadConnectOut) {
    std::this_thread::yield();
  }

  NetImgui::Internal::Network::Shutdown();
}

//=================================================================================================
// True when we are listening for client's connection requests
//=================================================================================================
bool IsWaitingForConnection() { return gListenSocket.load() != nullptr; }

//=================================================================================================
// Total amount of data sent to clients since start
//=================================================================================================
uint64_t GetStatsDataSent() { return gStatsDataSent; }

//=================================================================================================
// Total amount of data received from clients since start
//=================================================================================================
uint64_t GetStatsDataRcvd() { return gStatsDataRcvd; }

}  // namespace Network
}  // namespace NetImguiServer
