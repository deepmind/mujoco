// Google modifications:
// - Added #include <cstdint> for uint32_t (not implicitly available in Google)

#pragma once

#include <cstdint>
#include <imgui.h>

namespace NetImguiServer {
namespace App {
struct ServerTexture;
}
}  // namespace NetImguiServer

namespace NetImguiServer {
namespace UI {
constexpr uint32_t kWindowDPIDefault = 96;
bool Startup();
void Shutdown();
ImVec4 DrawImguiContent();
void DrawCenteredBackground(const App::ServerTexture* Texture,
                            const ImVec4& tint = ImVec4(1.f, 1.f, 1.f, 1.f));
float GetDisplayFPS();
const App::ServerTexture* GetBackgroundTexture();
}  // namespace UI
}  // namespace NetImguiServer
