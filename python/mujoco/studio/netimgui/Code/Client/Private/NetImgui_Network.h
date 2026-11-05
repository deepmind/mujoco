// Google modifications:
// - Added #include <cstdint> for uint32_t (not implicitly available in Google)
#pragma once

#include <cstdint>
namespace NetImgui {
namespace Internal {
struct PendingCom;
}
}  // namespace NetImgui

namespace NetImgui {
namespace Internal {
namespace Network {

struct SocketInfo;

bool Startup(void);
void Shutdown(void);

SocketInfo* Connect(
    const char* ServerHost,
    uint32_t ServerPort);  // Communication Socket expected to be blocking
SocketInfo* ListenConnect(
    SocketInfo* ListenSocket);  // Communication Socket expected to be blocking
SocketInfo* ListenStart(
    uint32_t ListenPort);  // Listening Socket expected to be non blocking
void Disconnect(SocketInfo* pClientSocket);

bool DataReceivePending(
    SocketInfo* pClientSocket);  // True if some new data if waiting to be
                                 // processed from remote connection
void DataReceive(
    SocketInfo* pClientSocket,
    PendingCom& PendingComRcv);  // Try reading X amount of bytes from remote
                                 // connection, but can fall short.
void DataSend(
    SocketInfo* pClientSocket,
    PendingCom& PendingComSend);  // Try sending X amount of bytes to remote
                                  // connection, but can fall short.

}  // namespace Network
}  // namespace Internal
}  // namespace NetImgui
