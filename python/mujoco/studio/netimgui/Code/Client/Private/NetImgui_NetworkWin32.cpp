// Google modifications:
//  - DataReceive validates the command size with PendingCom::GetSizeToReceive.
#include "NetImgui_Shared.h"

#if NETIMGUI_ENABLED && NETIMGUI_WINSOCKET_ENABLED
#include <WS2tcpip.h>
#include <WinSock2.h>

#include "NetImgui_WarningDisableStd.h"

#if defined(_MSC_VER)
#pragma comment(lib, "ws2_32")
#endif

#include "NetImgui_CmdPackets.h"

namespace NetImgui {
namespace Internal {
namespace Network {
//=================================================================================================
// Wrapper around native socket object and init some socket options
//=================================================================================================
struct SocketInfo {
  SocketInfo(SOCKET socket) : mSocket(socket) {
    u_long kNonBlocking = true;
    ioctlsocket(mSocket, static_cast<long>(FIONBIO), &kNonBlocking);

    constexpr DWORD kComsNoDelay = 1;
    setsockopt(mSocket, SOL_SOCKET, TCP_NODELAY,
               reinterpret_cast<const char*>(&kComsNoDelay),
               sizeof(kComsNoDelay));

    const int kComsSendBuffer = 2 * mSendSizeMax;
    setsockopt(mSocket, SOL_SOCKET, SO_SNDBUF,
               reinterpret_cast<const char*>(&kComsSendBuffer),
               sizeof(kComsSendBuffer));

    // constexpr int	kComsRcvBuffer = 1014*1024;
    // setsockopt(mSocket, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const
    // char*>(&kComsRcvBuffer), sizeof(kComsRcvBuffer));
  }

  SOCKET mSocket;
  int mSendSizeMax =
      1024 *
      1024;  // Limit tx data to avoid socket error on large amount (texture)
};

bool Startup() {
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;

  return true;
}

void Shutdown() { WSACleanup(); }

//=================================================================================================
// Try establishing a connection to a remote client at given address
//=================================================================================================
SocketInfo* Connect(const char* ServerHost, uint32_t ServerPort) {
  const timeval kConnectTimeout = {
      1, 0};  // Waiting 1 seconds before failing connection attempt
  u_long kNonBlocking = true;

  SOCKET ClientSocket = socket(AF_INET, SOCK_STREAM, 0);
  if (ClientSocket == INVALID_SOCKET) return nullptr;

  char zPortName[32] = {};
  addrinfo* pResults = nullptr;
  SocketInfo* pSocketInfo = nullptr;
  NetImgui::Internal::StringFormat(zPortName, "%i", ServerPort);
  getaddrinfo(ServerHost, zPortName, nullptr, &pResults);
  addrinfo* pResultCur = pResults;
  fd_set SocketSet;

  ioctlsocket(ClientSocket, static_cast<long>(FIONBIO), &kNonBlocking);
  while (pResultCur && !pSocketInfo) {
    int Result = connect(ClientSocket, pResultCur->ai_addr,
                         static_cast<int>(pResultCur->ai_addrlen));
    bool Connected = Result != SOCKET_ERROR;

    // Not connected yet, wait some time before bailing out
    if (Result == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK) {
      FD_ZERO(&SocketSet);
      FD_SET(ClientSocket, &SocketSet);
      Result = select(0, nullptr, &SocketSet, nullptr, &kConnectTimeout);
      Connected =
          Result == 1;  // when 1 socket ready for write, otherwise it's -1 or 0
    }

    if (Connected) {
      pSocketInfo = netImguiNew<SocketInfo>(ClientSocket);
    }

    pResultCur = pResultCur->ai_next;
  }
  freeaddrinfo(pResults);
  if (!pSocketInfo) {
    closesocket(ClientSocket);
  }
  return pSocketInfo;
}

//=================================================================================================
// Start waiting for connection request on this socket
//=================================================================================================
SocketInfo* ListenStart(uint32_t ListenPort) {
  SOCKET ListenSocket = INVALID_SOCKET;
  if ((ListenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) !=
      INVALID_SOCKET) {
    sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(static_cast<USHORT>(ListenPort));

#if NETIMGUI_FORCE_TCP_LISTEN_BINDING
    constexpr BOOL ReUseAdrValue(true);
    setsockopt(ListenSocket, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&ReUseAdrValue),
               sizeof(ReUseAdrValue));
#endif
    if (bind(ListenSocket, reinterpret_cast<sockaddr*>(&server),
             sizeof(server)) != SOCKET_ERROR &&
        listen(ListenSocket, 0) != SOCKET_ERROR) {
      u_long kIsNonBlocking = false;
      ioctlsocket(ListenSocket, static_cast<long>(FIONBIO), &kIsNonBlocking);
      return netImguiNew<SocketInfo>(ListenSocket);
    }
    closesocket(ListenSocket);
  }
  return nullptr;
}

//=================================================================================================
// Establish a new connection to a remote request
//=================================================================================================
SocketInfo* ListenConnect(SocketInfo* ListenSocket) {
  if (ListenSocket) {
    sockaddr ClientAddress;
    int Size(sizeof(ClientAddress));
    SOCKET ClientSocket = accept(ListenSocket->mSocket, &ClientAddress, &Size);
    if (ClientSocket != INVALID_SOCKET) {
      return netImguiNew<SocketInfo>(ClientSocket);
    }
  }
  return nullptr;
}

//=================================================================================================
// Close a connection and free allocated object
//=================================================================================================
void Disconnect(SocketInfo* pClientSocket) {
  if (pClientSocket) {
    shutdown(pClientSocket->mSocket, SD_BOTH);
    closesocket(pClientSocket->mSocket);
    netImguiDelete(pClientSocket);
  }
}

//=================================================================================================
// Return true if data has been received, or there's a connection error
//=================================================================================================
bool DataReceivePending(SocketInfo* pClientSocket) {
  const timeval kConnectTimeout = {0, 0};  // No wait time
  if (pClientSocket) {
    fd_set fdSetRead;
    fd_set fdSetErr;
    FD_ZERO(&fdSetRead);
    FD_ZERO(&fdSetErr);
    FD_SET(pClientSocket->mSocket, &fdSetRead);
    FD_SET(pClientSocket->mSocket, &fdSetErr);

    // Note: return true if data ready or connection error (to exit parent loop
    // waiting on data)
    int result = select(0, &fdSetRead, nullptr, &fdSetErr, &kConnectTimeout);
    return result != 0;
  }
  return true;
}

//=================================================================================================
// Receive as much as possible a command and keep track of transfer status
//=================================================================================================
void DataReceive(SocketInfo* pClientSocket,
                 NetImgui::Internal::PendingCom& PendingComRcv) {
  // Invalid command
  if (!pClientSocket || !PendingComRcv.pCommand || !pClientSocket->mSocket) {
    PendingComRcv.bError = true;
    return;
  }

  size_t BytesToRead = PendingComRcv.GetSizeToReceive();
  if (BytesToRead == 0) {
    return;  // Already fully received, or invalid size (error flagged)
  }

  // Receive data from remote connection
  int resultRcv = recv(pClientSocket->mSocket,
                       &reinterpret_cast<char*>(
                           PendingComRcv.pCommand)[PendingComRcv.SizeCurrent],
                       static_cast<int>(BytesToRead), 0);

  // Note: 'DataReceive' is called after pending data has been confirm.
  //		 0 received data means connection lost
  if (resultRcv != SOCKET_ERROR) {
    PendingComRcv.SizeCurrent += static_cast<size_t>(resultRcv);
    PendingComRcv.bError |=
        resultRcv <= 0;  // Error if no data read since DataReceivePending()
                         // said there was some available
  }
  // Connection error, abort transmission
  else if (WSAGetLastError() != WSAEWOULDBLOCK) {
    PendingComRcv.bError = true;
  }
}

//=================================================================================================
// Receive as much as possible a command and keep track of transfer status
//=================================================================================================
void DataSend(SocketInfo* pClientSocket,
              NetImgui::Internal::PendingCom& PendingComSend) {
  // Invalid command
  if (!pClientSocket || !PendingComSend.pCommand || !pClientSocket->mSocket) {
    PendingComSend.bError = true;
    return;
  }

  // Send data to remote connection
  int sizeToSend = static_cast<int>(PendingComSend.pCommand->mSize -
                                    PendingComSend.SizeCurrent);
  sizeToSend = sizeToSend > pClientSocket->mSendSizeMax
                   ? pClientSocket->mSendSizeMax
                   : sizeToSend;
  int resultSent =
      send(pClientSocket->mSocket,
           &reinterpret_cast<char*>(
               PendingComSend.pCommand)[PendingComSend.SizeCurrent],
           sizeToSend, 0);

  if (resultSent != SOCKET_ERROR) {
    PendingComSend.SizeCurrent += static_cast<size_t>(resultSent);
  }
  // Connection error, abort transmission
  else if (WSAGetLastError() != WSAEWOULDBLOCK) {
    PendingComSend.bError = true;
  }
}

}  // namespace Network
}  // namespace Internal
}  // namespace NetImgui

#include "NetImgui_WarningReenable.h"
#else

// Prevents Linker warning LNK4221 in Visual Studio (This object file does not
// define any previously undefined public symbols, so it will not be used by any
// link operation that consumes this library)
extern int sSuppresstLNK4221_NetImgui_NetworkWin23;
int sSuppresstLNK4221_NetImgui_NetworkWin23(0);

#endif  // #if NETIMGUI_ENABLED && NETIMGUI_WINSOCKET_ENABLED
