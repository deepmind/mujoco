// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The web viewer's headless Studio UI.
//
// This pybind11 module provides a HeadlessUi class that:
// 1. Creates a headless ImGui context (no window, no renderer).
// 2. Connects as a netimgui client, streaming ImGui draw data to a remote
//    viewer (the browser's web_client, bridged through web_server.py).
// 3. Receives input events from the remote viewer and injects them into
//    the ImGui context.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <mujoco/studio/ux/fonts.h>
#include <NetImgui_Api.h>
#include "google/logging.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// Headless ImGui + netimgui viewer for the MuJoCo web viewer.
//
// This class manages a headless ImGui context that streams its draw data
// via the netimgui protocol to a remote viewer. Unlike the NativeViewer's
// Viewer class (native_viewer.cc), this does NOT create a window, initialize
// a renderer, or handle mouse/keyboard input directly. All rendering and
// input handling happens on the remote client (web_client.cc in the browser).
//
// The lifecycle follows the SampleNoBackend pattern:
//   Client_Startup()   — create context, load fonts, init NetImgui
//   Client_Connect()   — manage connection state (called each frame)
//   Client_Shutdown()  — release resources

// Initialize the Dear ImGui Context and the NetImgui library.
// Based on SampleNoBackend::Client_Startup() from
// netimgui/Code/Sample/SampleNoBackend/SampleNoBackend.cpp
static bool Client_Startup(ImGuiContext*& context,
                           ImPlotContext*& implot_context,
                           const std::string& assets_dir) {
  IMGUI_CHECKVERSION();
  context = ImGui::CreateContext();
  ImGui::SetCurrentContext(context);
  implot_context = ImPlot::CreateContext();
  ImPlot::SetCurrentContext(implot_context);

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
  io.IniFilename = nullptr;
  io.ConfigDpiScaleFonts = true;
  io.ConfigDpiScaleViewports = true;
  io.DisplaySize = ImVec2(1400, 720);

  // Disable ImGui auto-repeat in headless mode since the browser handles key
  // repeat at the OS level, avoiding the frame amplification issue where many
  // frames with identical key-down states trigger unwanted repeats.
  io.KeyRepeatDelay = 9999.0f;

  // Initialize the main viewport's DPI scale. Without this, the headless
  // context leaves DpiScale at 0.0f (no platform backend sets it), which
  // triggers an assertion in ImGui::SetCurrentViewport() when
  // BeginMainMenuBar() is called.
  ImGuiViewport* main_vp = ImGui::GetMainViewport();
  if (main_vp) {
    main_vp->DpiScale = 1.0f;
  }

  ImGui::StyleColorsLight();

  // The Studio font set shared with the native viewer and the browser client.
  mujoco::studio::AddStudioFonts([&assets_dir](std::string_view filename) {
    return mujoco::studio::LoadFontAsset(assets_dir, filename);
  });

  if (!NetImgui::Startup()) {
    LOG(Error, "NetImgui::Startup() failed");
    return false;
  }

  return true;
}

// Release resources.
// Based on SampleNoBackend::Client_Shutdown() from
// netimgui/Code/Sample/SampleNoBackend/SampleNoBackend.cpp
static void Client_Shutdown(ImGuiContext*& context,
                            ImPlotContext*& implot_context) {
  NetImgui::Shutdown();
  if (implot_context) {
    ImPlot::DestroyContext(implot_context);
    implot_context = nullptr;
  }
  if (context) {
    ImGui::DestroyContext(context);
    context = nullptr;
  }
}

// Manage connection to the netimgui proxy.
// Based on SampleNoBackend::Client_Connect() from
// netimgui/Code/Sample/SampleNoBackend/SampleNoBackend.cpp
static void Client_Connect(const char* title, int port) {
  bool connected = NetImgui::IsConnected();
  bool pending = NetImgui::IsConnectionPending();

  if (!connected && !pending) {
    static std::chrono::steady_clock::time_point last_reconnect_time =
        std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    if (now - last_reconnect_time > std::chrono::seconds(1)) {
      last_reconnect_time = now;
      VLOG(1, "Retrying ConnectToApp...");
      NetImgui::ConnectToApp(title, "127.0.0.1", port);
    }
  }

  static bool last_connected = false;
  if (connected != last_connected) {
    VLOG(1, "Status change: Connected=%s", connected ? "true" : "false");
    last_connected = connected;
  }
}

class HeadlessUi {
 public:
  HeadlessUi(const std::string& title, int port, const std::string& assets_dir)
      : title_(title), port_(port) {
    if (!Client_Startup(context_, implot_context_, assets_dir)) {
      return;
    }

    // Python imgui.Image() calls pass bare integer ImTextureIDs
    // (leaving ImDrawCmd::TexRef._TexData == nullptr). After ImGui::EndFrame()
    // finalizes draw lists, resolve each draw command's TexRef to its
    // registered ImTextureData so NetImgui::EndFrame() can track and transmit
    // texture updates to the remote client.
    ImGuiContextHook hook;
    hook.Type = ImGuiContextHookType_EndFramePost;
    hook.UserData = this;
    hook.Callback = [](ImGuiContext* ctx, ImGuiContextHook* h) {
      auto* self = static_cast<HeadlessUi*>(h->UserData);
      for (ImGuiWindow* window : ctx->Windows) {
        if (!window->DrawList) continue;
        for (ImDrawCmd& cmd : window->DrawList->CmdBuffer) {
          if (cmd.TexRef._TexData == nullptr) {
            auto it =
                self->textures_.find(static_cast<uintptr_t>(cmd.TexRef._TexID));
            if (it != self->textures_.end() && it->second) {
              cmd.TexRef = it->second->GetTexRef();
            }
          }
        }
      }
    };
    ImGui::AddContextHook(context_, &hook);

    VLOG(1, "Calling ConnectToApp('%s', '127.0.0.1', %d)", title_.c_str(),
         port_);
    bool connect_result =
        NetImgui::ConnectToApp(title_.c_str(), "127.0.0.1", port_);
    VLOG(1, "ConnectToApp returned: %s", connect_result ? "true" : "false");
    VLOG(1, "IsConnected: %s, IsConnectionPending: %s",
         NetImgui::IsConnected() ? "true" : "false",
         NetImgui::IsConnectionPending() ? "true" : "false");
  }

  ~HeadlessUi() {
    if (context_) {
      ImGui::SetCurrentContext(context_);
      for (auto& [id, tex] : textures_) {
        if (tex) {
          ImGui::UnregisterUserTexture(tex.get());
        }
      }
    }
    textures_.clear();
    Client_Shutdown(context_, implot_context_);
  }

  bool NewFrame() {
    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);
    static int frame_count = 0;
    frame_count++;

    // Returns true with an active ImGui frame once a browser is connected and
    // ready. Returns false (no frame) while no browser is connected so the
    // caller's loop keeps running and can drain messages (e.g., to respond to
    // an ExitEvent for shutdown).
    std::chrono::steady_clock::time_point last_signal_check =
        std::chrono::steady_clock::now();
    while (true) {
      // While no browser is connected, periodically check for
      // Python signals so Ctrl+C interrupts the wait instead of hanging.
      const std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now();
      if (now - last_signal_check > std::chrono::milliseconds(200)) {
        last_signal_check = now;
        py::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0) {
          throw py::error_already_set();
        }
      }

      Client_Connect(title_.c_str(), port_);

      if (!NetImgui::IsConnected()) {
        // Not connected; return false so the caller's loop is not blocked.
        is_drawing_remote_ = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return false;
      }

      // Limit frame rate to 60 FPS (~16.6 ms per frame). Unlike native desktop
      // viewers whose frame loops are throttled by hardware display VSync,
      // headless rendering has no VSync and would otherwise spin unthrottled
      // at 1000+ FPS, starving the simulation thread of CPU cycles and GIL
      // access. Note: this sleep occurs while GIL is released, allowing the
      // simulation thread full uninterrupted execution.
      //
      // Schedule each frame from the previous frame's deadline rather than from
      // the time the sleep returned, so sleep overshoot does not accumulate and
      // slow the frame rate below 60 FPS. If more than a frame behind (e.g.
      // after a reconnect), resync to avoid bursting to catch up.
      constexpr auto kFramePeriod =
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              kTargetFrameDuration);
      const auto frame_now = std::chrono::steady_clock::now();
      auto deadline = last_frame_time_ + kFramePeriod;
      if (deadline + kFramePeriod < frame_now) {
        deadline = frame_now;
      }
      std::this_thread::sleep_until(deadline);
      last_frame_time_ = deadline;

      bool new_frame_result = NetImgui::NewFrame(false);
      is_drawing_remote_ = new_frame_result;
      if (!new_frame_result) {
        // Connected but NetImgui not ready for a draw — wait and retry.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      break;
    }

    VLOG(1, "Frame %d: DrawingRemote=%s", frame_count,
         is_drawing_remote_ ? "Y" : "N");
    return true;
  }

  void EndFrame() {
    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);
    static int end_frame_count = 0;
    end_frame_count++;
    if (!is_drawing_remote_) {
      // No frame was started — nothing to end.
      return;
    }
    VLOG(1, "EndFrame %d: sending remote draw data", end_frame_count);
    NetImgui::EndFrame();
  }

  // Uploads an RGB/RGBA image to the browser over the NetImgui texture
  // channel so handlers can display it with imgui.Image(). Returns the
  // texture id to use (allocates one when tex_id == 0). This is the UI-link
  // counterpart of the native renderer's UploadImage.
  uintptr_t UploadImage(uintptr_t tex_id, const py::bytes& pixels, int width,
                        int height, int bpp) {
    if (tex_id == 0) {
      tex_id = next_tex_id_++;
    }
    std::string data = pixels;
    if (width <= 0 || height <= 0 ||
        data.size() < static_cast<size_t>(width) * height * bpp) {
      LOG(Error, "UploadImage: bad dimensions %dx%dx%d for %zu bytes", width,
          height, bpp, data.size());
      return tex_id;
    }

    // NetImgui transfers RGBA8; expand RGB if needed.
    std::vector<uint8_t> rgba;
    const void* upload_data = data.data();
    if (bpp == 3) {
      rgba.resize(static_cast<size_t>(width) * height * 4);
      for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
        rgba[p * 4 + 0] = data[p * 3 + 0];
        rgba[p * 4 + 1] = data[p * 3 + 1];
        rgba[p * 4 + 2] = data[p * 3 + 2];
        rgba[p * 4 + 3] = 255;
      }
      upload_data = rgba.data();
    } else if (bpp != 4) {
      LOG(Error, "UploadImage: unsupported bpp %d (expected 3 or 4)", bpp);
      return tex_id;
    }

    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);

    std::unique_ptr<ImTextureData>& tex = textures_[tex_id];
    const size_t rgba_bytes = static_cast<size_t>(width) * height * 4;
    if (tex && tex->Width == width && tex->Height == height) {
      std::memcpy(tex->GetPixels(), upload_data, rgba_bytes);
      tex->UpdateRect = {0, 0, static_cast<uint16_t>(width),
                         static_cast<uint16_t>(height)};
      if (tex->Status != ImTextureStatus_WantCreate) {
        tex->SetStatus(ImTextureStatus_WantUpdates);
      }
      return tex_id;
    }

    if (tex) {
      ImGui::UnregisterUserTexture(tex.get());
      textures_.erase(tex_id);
      tex_id = next_tex_id_++;
    }
    auto new_tex = std::make_unique<ImTextureData>();
    new_tex->Create(ImTextureFormat_RGBA32, width, height);
    new_tex->SetStatus(ImTextureStatus_WantCreate);
    std::memcpy(new_tex->GetPixels(), upload_data, rgba_bytes);
    ImGui::RegisterUserTexture(new_tex.get());
    textures_[tex_id] = std::move(new_tex);
    return tex_id;
  }

  intptr_t GetContext() const { return reinterpret_cast<intptr_t>(context_); }

  // The ImPlot context created alongside the ImGui context. Python must pass
  // this to ux.set_implot_context: extension modules each hold their own copy
  // of the ImPlot globals, so the context pointer has to be shared explicitly
  // (same pattern as get_context/set_imgui_context).
  intptr_t GetImPlotContext() const {
    return reinterpret_cast<intptr_t>(implot_context_);
  }

 private:
  std::string title_;
  int port_;
  ImGuiContext* context_ = nullptr;
  ImPlotContext* implot_context_ = nullptr;
  bool is_drawing_remote_ = false;
  std::unordered_map<uintptr_t, std::unique_ptr<ImTextureData>> textures_;
  std::chrono::steady_clock::time_point last_frame_time_ =
      std::chrono::steady_clock::now();
  static constexpr auto kTargetFrameDuration =
      std::chrono::duration<double>(1.0 / 60.0);
  // User texture ids start well above the ids ImGui's managed texture system
  // (font atlas) hands out, so the two can never collide in the browser's
  // texture map.
  uintptr_t next_tex_id_ = 0x10000;
};

}  // namespace

namespace mujoco::python::studio {

void InitHeadlessUi(py::module_& m) {
  m.doc() = "MuJoCo web viewer headless Studio UI, streamed via NetImgui";

  py::class_<HeadlessUi>(m, "HeadlessUi")
      .def(py::init<const std::string&, int, const std::string&>(),
           py::arg("title"), py::arg("port") = 8888, py::arg("assets_dir") = "")
      .def("new_frame", &HeadlessUi::NewFrame,
           "Starts a headless ImGui frame, returning True once the frame is "
           "active. When a browser is viewing the page (via the URL printed "
           "at startup), this returns at the browser's requested frame rate. "
           "When no browser is viewing, it returns False (no frame) after a "
           "short wait, so the caller's loop keeps running and can shut down.")
      .def("end_frame", &HeadlessUi::EndFrame)
      .def("get_context", &HeadlessUi::GetContext)
      .def("get_implot_context", &HeadlessUi::GetImPlotContext)
      .def("upload_image", &HeadlessUi::UploadImage);
}

}  // namespace mujoco::python::studio

#ifndef MUJOCO_STUDIO_COMBINED_MODULE
PYBIND11_MODULE(headless_ui, m, pybind11::mod_gil_not_used()) {
  mujoco::python::studio::InitHeadlessUi(m);
}
#endif
