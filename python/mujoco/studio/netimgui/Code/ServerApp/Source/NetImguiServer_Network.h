// Google modifications:
//  - Added #include <cstdint> for uint32_t (not implicitly available in
//  Google).
#pragma once

#include <cstdint>

namespace NetImguiServer {
namespace Network {
// Initialize application's networking
bool Startup();

// Shutdown application's networking
void Shutdown();

// True if this application is actively waiting for clients to reach it
// Note: False when unable to open listen socket (probably because port number
// alreay in use)
bool IsWaitingForConnection();

// Total amount of data sent to clients since start
uint64_t GetStatsDataSent();

// Total amount of data received from clients since start
uint64_t GetStatsDataRcvd();

}  // namespace Network
}  // namespace NetImguiServer
