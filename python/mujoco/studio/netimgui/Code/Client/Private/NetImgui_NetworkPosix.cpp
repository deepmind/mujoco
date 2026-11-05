// Google modifications:
//  - Replaced select() with poll() to avoid FD_SETSIZE (1024) limit and
//    a SIGSEGV caused by Linux select() writing to a const timeval placed
//    in read-only memory by the compiler.
//  - Added logging for network events.
//  - DataReceive validates the command size with PendingCom::GetSizeToReceive.
#include "NetImgui_Shared.h"
#include "google/logging.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4221)
#endif

#if NETIMGUI_ENABLED && NETIMGUI_POSIX_SOCKETS_ENABLED
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>  // Required for TCP_NODELAY
#include <poll.h>         // Preferred over select() (no FD_SETSIZE limit)
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <string>  // Required for std::string / std::to_string with GCC
#include <time.h>
#include <unistd.h>

#include <cstdint>

#include "NetImgui_CmdPackets.h"

// NOTE: Removed static_assert(0) as requested changes are implemented below

namespace NetImgui {
namespace Internal {
namespace Network {

//=================================================================================================
// Wrapper around native socket object and init some socket options
//=================================================================================================
struct SocketInfo {
  SocketInfo(int socket) : mSocket(socket) {
    if (mSocket != -1) {
      // Set Non-Blocking
      int flags = fcntl(mSocket, F_GETFL, 0);
      fcntl(mSocket, F_SETFL, flags | O_NONBLOCK);

      // Set TCP No Delay
      int flag = 1;
      setsockopt(mSocket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int));

      // Optional: Set Send Buffer Size (Mirroring Win32's attempt)
      // int kComsSendBuffer = 2 * mSendSizeMax;
      // setsockopt(mSocket, SOL_SOCKET, SO_SNDBUF, (char*)&kComsSendBuffer,
      // sizeof(kComsSendBuffer));
    }
  }

  int mSocket = -1;
  int mSendSizeMax = 1024 * 1024;  // Limit tx data to avoid socket error on
                                   // large amount (texture) [cite: 258]
};

bool Startup() {
  // No specific startup needed for POSIX sockets like WSAStartup in Winsock
  return true;
}

void Shutdown() {
  // No specific cleanup needed for POSIX sockets like WSACleanup in Winsock
}

//=================================================================================================
// Try establishing a connection to a remote client at given address
// (Non-Blocking)
//=================================================================================================
SocketInfo* Connect(const char* ServerHost, uint32_t ServerPort) {
  int ClientSocket = -1;
  addrinfo hints, *pResults = nullptr, *pResultCur = nullptr;
  SocketInfo* pSocketInfo = nullptr;
  char zPortName[32];
  sprintf(zPortName, "%u", ServerPort);

  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;  // Allow IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(ServerHost, zPortName, &hints, &pResults) != 0) {
    return nullptr;  // Failed to resolve host
  }

  for (pResultCur = pResults; pResultCur != nullptr && pSocketInfo == nullptr;
       pResultCur = pResultCur->ai_next) {
    ClientSocket = socket(pResultCur->ai_family, pResultCur->ai_socktype,
                          pResultCur->ai_protocol);
    if (ClientSocket == -1) {
      continue;  // Failed to create socket for this address
    }

    // Set non-blocking *before* connect for non-blocking connect
    int flags = fcntl(ClientSocket, F_GETFL, 0);
    fcntl(ClientSocket, F_SETFL, flags | O_NONBLOCK);

    int Result =
        connect(ClientSocket, pResultCur->ai_addr, pResultCur->ai_addrlen);
    bool Connected = (Result == 0);

    if (Result == -1 && errno == EINPROGRESS) {
      // Connection attempt is in progress, use poll to wait
      struct pollfd pfd = {ClientSocket, POLLOUT, 0};
      Result = poll(&pfd, 1, 1000);  // 1 second timeout

      if (Result > 0) {
        // Select indicated socket is writable, check for connection errors
        int optVal;
        socklen_t optLen = sizeof(optVal);
        if (getsockopt(ClientSocket, SOL_SOCKET, SO_ERROR, &optVal, &optLen) ==
                0 &&
            optVal == 0) {
          Connected = true;
        } else {
          // Connection failed
          Connected = false;
        }
      } else {
        // Select timed out or error
        Connected = false;
      }
    } else if (Result == -1) {
      // Immediate connection error
      Connected = false;
    }

    if (Connected) {
      pSocketInfo = netImguiNew<SocketInfo>(ClientSocket);
      // Socket is already non-blocking from the SocketInfo constructor
    } else if (ClientSocket != -1) {
      close(ClientSocket);  // Close socket if connection failed
      ClientSocket = -1;
    }
  }

  freeaddrinfo(pResults);

  if (!pSocketInfo && ClientSocket != -1) {
    close(ClientSocket);  // Clean up socket if loop finished without success
  }

  return pSocketInfo;
}

//=================================================================================================
// Start waiting for connection request on this socket
//=================================================================================================
SocketInfo* ListenStart(uint32_t ListenPort) {
  addrinfo hints, *addrInfo;
  int ListenSocket = -1;

  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_INET;  // Typically listen on IPv4 for simplicity, or
                              // AF_UNSPEC for both
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;  // Use my IP

  std::string portStr = std::to_string(ListenPort);
  if (getaddrinfo(nullptr, portStr.c_str(), &hints, &addrInfo) != 0) {
    return nullptr;
  }

  ListenSocket =
      socket(addrInfo->ai_family, addrInfo->ai_socktype, addrInfo->ai_protocol);
  if (ListenSocket != -1) {
#if NETIMGUI_FORCE_TCP_LISTEN_BINDING
    int flag = 1;
    setsockopt(ListenSocket, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
#ifdef SO_REUSEPORT  // SO_REUSEPORT might not be available on all POSIX systems
    setsockopt(ListenSocket, SOL_SOCKET, SO_REUSEPORT, &flag, sizeof(flag));
#endif
#endif

    if (bind(ListenSocket, addrInfo->ai_addr, addrInfo->ai_addrlen) != -1 &&
        listen(ListenSocket, SOMAXCONN) != -1)  // Use SOMAXCONN for backlog
    {
      // Keep listening socket blocking for accept() simplicity,
      // the *accepted* socket will be non-blocking via SocketInfo ctor
      int flags = fcntl(ListenSocket, F_GETFL, 0);
      fcntl(ListenSocket, F_SETFL, flags & (~O_NONBLOCK));  // Ensure blocking

      freeaddrinfo(addrInfo);
      // Note: We create SocketInfo wrapper here just for consistency,
      // but it doesn't set options on the *listening* socket.
      return netImguiNew<SocketInfo>(ListenSocket);
    }
    close(ListenSocket);
  }

  freeaddrinfo(addrInfo);
  return nullptr;
}

//=================================================================================================
// Accept a new connection (blocking call on ListenSocket)
//=================================================================================================
SocketInfo* ListenConnect(SocketInfo* pListenSocket) {
  if (pListenSocket && pListenSocket->mSocket != -1) {
    sockaddr_storage ClientAddress;
    socklen_t Size = sizeof(ClientAddress);
    // ListenSocket should be blocking (set in ListenStart)
    int ServerSocket =
        accept(pListenSocket->mSocket, (sockaddr*)&ClientAddress, &Size);
    if (ServerSocket != -1) {
      VLOG(1, "TCP connection accepted (fd=%d)", ServerSocket);
      // Create SocketInfo wrapper, which sets the new socket to non-blocking
      return netImguiNew<SocketInfo>(ServerSocket);
    }
  }
  return nullptr;
}

//=================================================================================================
// Close a connection and free allocated object
//=================================================================================================
void Disconnect(SocketInfo* pClientSocket) {
  if (pClientSocket && pClientSocket->mSocket != -1) {
    // Set SO_LINGER option to force close and discard pending data
    // to ensure the socket is closed immediately and exits the CLOSE_WAIT state
    // more reliably
    struct linger sl;
    sl.l_onoff = 1;   // Enable linger
    sl.l_linger = 0;  // Set timeout to 0 seconds (force RST)
    setsockopt(pClientSocket->mSocket, SOL_SOCKET, SO_LINGER, &sl, sizeof(sl));

    shutdown(pClientSocket->mSocket, SHUT_RDWR);
    close(pClientSocket->mSocket);
    pClientSocket->mSocket = -1;  // Mark as closed
  }
  netImguiDelete(pClientSocket);
}

//=================================================================================================
// Return true if data has been received, or there's a connection error
//=================================================================================================
bool DataReceivePending(SocketInfo* pClientSocket) {
  if (!pClientSocket || pClientSocket->mSocket == -1) {
    return true;  // Error condition
  }

  // Use poll() instead of select() to avoid FD_SETSIZE limitation.
  struct pollfd pfd = {pClientSocket->mSocket, POLLIN, 0};
  int result = poll(&pfd, 1, 0);  // 0ms timeout = non-blocking check

  // poll() syscall error (e.g. EINTR) — treat as "no data" and retry next
  // frame rather than speculatively calling recv() on unknown socket state.
  if (result < 0) {
    return false;
  }

  // No events on the socket.
  if (result == 0) {
    return false;
  }

  // result > 0: the socket has events.
  if (pfd.revents & POLLERR) {
    // Report as data-pending so the caller's recv() surfaces the error
    // through its own error-handling path. Info, not Error: this is routine
    // when the peer tears the connection down (e.g. viewer shutdown).
    VLOG(1, "DataReceivePending: POLLERR revents=%d", pfd.revents);
    return true;
  }

  if (pfd.revents & POLLHUP) {
    // Peer hung up. The persistent TCP proxy keeps connections alive across
    // browser refreshes, so a transient POLLHUP does not mean the session is
    // over — ignore it.
    VLOG(1, "DataReceivePending: POLLHUP revents=%d", pfd.revents);
  }

  // Only signal data-pending when the kernel has bytes ready to read.
  return (pfd.revents & POLLIN) != 0;
}

//=================================================================================================
// Receive as much as possible into PendingCom buffer (Non-Blocking)
//=================================================================================================
void DataReceive(SocketInfo* pClientSocket,
                 NetImgui::Internal::PendingCom& PendingComRcv) {
  // Invalid command or socket state
  if (!pClientSocket || pClientSocket->mSocket == -1 ||
      !PendingComRcv.pCommand) {
    PendingComRcv.bError = true;
    return;
  }

  size_t BytesToRead = PendingComRcv.GetSizeToReceive();
  if (BytesToRead == 0) {
    return;  // Already fully received, or invalid size (error flagged)
  }

  // Receive data from remote connection (non-blocking)
  ssize_t resultRcv =
      recv(pClientSocket->mSocket,
           &reinterpret_cast<uint8_t*>(
               PendingComRcv.pCommand)[PendingComRcv.SizeCurrent],
           BytesToRead,
           0);  // No flags, non-blocking behavior comes from socket setting

  if (resultRcv > 0) {
    // Successfully received some data
    PendingComRcv.SizeCurrent += static_cast<size_t>(resultRcv);
    PendingComRcv.bError = false;  // Reset error flag on successful read
  } else if (resultRcv == 0) {
    // Connection closed gracefully by peer
    PendingComRcv.bError = true;
    VLOG(1, "DataReceive: Connection closed gracefully");
  } else {  // resultRcv < 0
    // Error occurred
    if (errno == EWOULDBLOCK || errno == EAGAIN) {
      // Not an error, just no data available right now on non-blocking socket
      PendingComRcv.bError = false;
    } else {
      // Actual socket error
      PendingComRcv.bError = true;
    }
  }
}

//=================================================================================================
// Send as much as possible from PendingCom buffer (Non-Blocking)
//=================================================================================================
void DataSend(SocketInfo* pClientSocket,
              NetImgui::Internal::PendingCom& PendingComSend) {
  // Invalid command or socket state
  if (!pClientSocket || pClientSocket->mSocket == -1 ||
      !PendingComSend.pCommand) {
    PendingComSend.bError = true;
    return;
  }

  size_t BytesRemaining =
      PendingComSend.pCommand->mSize - PendingComSend.SizeCurrent;
  if (BytesRemaining == 0) {
    return;  // Already fully sent
  }

  // Limit send size per call [cite: 281]
  size_t BytesToSend =
      BytesRemaining > static_cast<size_t>(pClientSocket->mSendSizeMax)
          ? static_cast<size_t>(pClientSocket->mSendSizeMax)
          : BytesRemaining;

  // Send data to remote connection (non-blocking)
  ssize_t resultSent =
      send(pClientSocket->mSocket,
           &reinterpret_cast<const uint8_t*>(
               PendingComSend.pCommand)[PendingComSend.SizeCurrent],
           BytesToSend,
           MSG_NOSIGNAL);  // Use MSG_NOSIGNAL to prevent SIGPIPE on Linux if
                           // connection is broken

  if (resultSent > 0) {
    // Successfully sent some data
    PendingComSend.SizeCurrent += static_cast<size_t>(resultSent);
    PendingComSend.bError = false;  // Reset error flag on successful send
  } else if (resultSent == 0) {
    // This shouldn't typically happen with TCP unless BytesToSend was 0
    PendingComSend.bError = false;  // Treat as non-error for now
  } else {                          // resultSent < 0
    // Error occurred
    if (errno == EWOULDBLOCK || errno == EAGAIN) {
      // Not an error, socket buffer is full, try again later
      PendingComSend.bError = false;
    } else {
      // Actual socket error (e.g., EPIPE if connection broken and MSG_NOSIGNAL
      // not used/supported)
      PendingComSend.bError = true;
    }
  }
}

}  // namespace Network
}  // namespace Internal
}  // namespace NetImgui
#else

// Prevents Linker warning LNK4221 in Visual Studio (This object file does not
// define any previously undefined public symbols, so it will not be used by any
// link operation that consumes this library)
extern int sSuppresstLNK4221_NetImgui_NetworkPosix;
int sSuppresstLNK4221_NetImgui_NetworkPosix(0);

#endif  // #if NETIMGUI_ENABLED && NETIMGUI_POSIX_SOCKETS_ENABLED
