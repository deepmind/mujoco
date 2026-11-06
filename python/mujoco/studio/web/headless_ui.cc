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
// 2. Turns each frame into Studio UI wire messages for the browser's
//    web_client, which web_server.py sends over the /ui WebSocket (see
//    headless_ui_session.h).
// 3. Applies the input the browser sends back to the ImGui context.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <mujoco/studio/ux/fonts.h>
#include "headless_ui_session.h"
#include "logging.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// A headless ImGui context whose frames are streamed to the browser.
//
// Unlike the native viewer, this does not create a window, initialize a
// renderer, or read the mouse and keyboard. All rendering and input handling
// happens in the browser (web_client.cc); this class turns frames into
// messages and browser input into ImGui input, through a
// mujoco::studio::HeadlessUiSession.
class HeadlessUi {
 public:
  HeadlessUi(const std::string& title, const std::string& assets_dir) {
    IMGUI_CHECKVERSION();
    context_ = ImGui::CreateContext();
    ImGui::SetCurrentContext(context_);
    implot_context_ = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(implot_context_);

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // HeadlessUiSession plays the renderer backend's part for textures: it
    // sends them to the browser, which renders them.
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.BackendPlatformName = "mujoco_studio_headless";
    io.BackendRendererName = "mujoco_studio_web";
    io.IniFilename = nullptr;
    io.ConfigDpiScaleFonts = true;
    io.ConfigDpiScaleViewports = true;
    io.DisplaySize = ImVec2(1400, 720);

    // Initialize the main viewport's DPI scale. Without this, the headless
    // context leaves DpiScale at 0.0f (no platform backend sets it), which
    // triggers an assertion in ImGui::SetCurrentViewport() when
    // BeginMainMenuBar() is called.
    ImGuiViewport* main_vp = ImGui::GetMainViewport();
    if (main_vp) {
      main_vp->DpiScale = 1.0f;
    }

    // Forward ImGui::SetClipboardText (used by Studio's Copy buttons and
    // open-URL commands) to the active browser session over /ui.
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    platform_io.Platform_ClipboardUserData = &session_;
    platform_io.Platform_SetClipboardTextFn = [](ImGuiContext* ctx,
                                                 const char* text) {
      if (!ctx || !text) return;
      auto* session = static_cast<mujoco::studio::HeadlessUiSession*>(
          ctx->PlatformIO.Platform_ClipboardUserData);
      if (session) {
        session->QueueClipboard(text);
      }
    };

    ImGui::StyleColorsLight();

    // The Studio font set shared with the native viewer and the browser client.
    mujoco::studio::AddStudioFonts([&assets_dir](std::string_view filename) {
      return mujoco::studio::LoadFontAsset(assets_dir, filename);
    });

    // Python imgui.Image() calls pass bare integer ImTextureIDs
    // (leaving ImDrawCmd::TexRef._TexData == nullptr). After ImGui::EndFrame()
    // finalizes draw lists, resolve each draw command's TexRef to its
    // registered ImTextureData, so the browser receives the texture and the
    // draw command refers to it.
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
  }

  ~HeadlessUi() {
    ImGui::SetCurrentContext(context_);
    for (auto& [id, tex] : textures_) {
      if (tex) {
        ImGui::UnregisterUserTexture(tex.get());
      }
    }
    textures_.clear();
    ImPlot::DestroyContext(implot_context_);
    ImGui::DestroyContext(context_);
  }

  // Starts a headless ImGui frame and returns true, once a browser controls the
  // UI and has sent its first input (its screen size). Returns false (no frame)
  // after a short wait otherwise, so the caller's loop keeps running and can
  // drain messages (e.g., to respond to an ExitEvent for shutdown).
  bool NewFrame() {
    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);
    if (!session_.HasInput()) {
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
    auto deadline = frame_deadline_ + kFramePeriod;
    if (deadline + kFramePeriod < frame_now) {
      deadline = frame_now;
    }
    std::this_thread::sleep_until(deadline);
    frame_deadline_ = deadline;

    const auto now = std::chrono::steady_clock::now();
    // Clamped: a session can start after a long idle time, and ImGui expects
    // a positive time step.
    const float delta_time =
        std::chrono::duration<float>(now - last_frame_time_).count();
    ImGui::GetIO().DeltaTime = std::clamp(delta_time, 1e-3f, 0.1f);
    last_frame_time_ = now;

    session_.ApplyInput();
    ImGui::NewFrame();
    is_drawing_ = true;
    return true;
  }

  // Ends the frame started by NewFrame() and queues it for the browser.
  void EndFrame() {
    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);
    if (!is_drawing_) {
      return;  // No frame was started.
    }
    is_drawing_ = false;
    const ImGuiMouseCursor cursor = ImGui::GetMouseCursor();
    ImGui::Render();
    session_.CaptureFrame(cursor);
  }

  // Uploads an RGB/RGBA image to the browser over the UI stream so handlers can
  // display it with imgui.Image(). Returns the texture id to use (allocates one
  // when tex_id == 0). This is the UI stream's counterpart of the native
  // renderer's UploadImage.
  uintptr_t UploadImage(uintptr_t tex_id, const py::bytes& pixels, int width,
                        int height, int bpp) {
    if (tex_id == 0) {
      tex_id = next_tex_id_++;
    }
    const std::string_view data(pixels);
    if (width <= 0 || height <= 0 ||
        data.size() < static_cast<size_t>(width) * height * bpp) {
      LOG(Error, "UploadImage: bad dimensions %dx%dx%d for %zu bytes", width,
          height, bpp, data.size());
      return tex_id;
    }

    // Textures are sent as RGBA8; expand RGB if needed.
    std::vector<std::byte> rgba;
    const std::byte* upload_data = std::as_bytes(std::span(data)).data();
    if (bpp == 3) {
      rgba.resize(static_cast<size_t>(width) * height * 4);
      for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
        rgba[p * 4 + 0] = static_cast<std::byte>(data[p * 3 + 0]);
        rgba[p * 4 + 1] = static_cast<std::byte>(data[p * 3 + 1]);
        rgba[p * 4 + 2] = static_cast<std::byte>(data[p * 3 + 2]);
        rgba[p * 4 + 3] = std::byte{0xFF};
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

  // ---- The UI stream, called from the web server's thread. -----------------

  py::bytes StartSession(const py::bytes& handshake) {
    const std::string_view data(handshake);
    mujoco::studio::netui::Bytes reply;
    std::string error;
    bool started = false;
    {
      py::gil_scoped_release no_gil;
      started =
          session_.StartSession(std::as_bytes(std::span(data)), &reply, &error);
    }
    if (!started) {
      throw py::value_error(error);
    }
    return py::bytes(reinterpret_cast<const char*>(reply.data()), reply.size());
  }

  void EndSession() {
    py::gil_scoped_release no_gil;
    session_.EndSession();
  }

  bool PushInput(const py::bytes& message) {
    const std::string_view data(message);
    py::gil_scoped_release no_gil;
    return session_.PushInput(std::as_bytes(std::span(data)));
  }

  py::list TakeMessages() {
    std::vector<mujoco::studio::netui::Bytes> messages;
    {
      py::gil_scoped_release no_gil;
      messages = session_.TakeMessages();
    }
    py::list result;
    for (const mujoco::studio::netui::Bytes& message : messages) {
      result.append(py::bytes(reinterpret_cast<const char*>(message.data()),
                              message.size()));
    }
    return result;
  }

  uint64_t LastFrameIndex() const {
    py::gil_scoped_release no_gil;
    return session_.LastFrameIndex();
  }

 private:
  static constexpr auto kTargetFrameDuration =
      std::chrono::duration<double>(1.0 / 60.0);

  mujoco::studio::HeadlessUiSession session_;
  std::unordered_map<uintptr_t, std::unique_ptr<ImTextureData>> textures_;
  ImGuiContext* context_ = nullptr;
  ImPlotContext* implot_context_ = nullptr;
  std::chrono::steady_clock::time_point frame_deadline_ =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last_frame_time_ =
      std::chrono::steady_clock::now();
  // User texture ids start well above the ids ImGui hands out for its own
  // textures (font atlas), so Python handles can never be mistaken for them.
  uintptr_t next_tex_id_ = 0x10000;
  bool is_drawing_ = false;
};

}  // namespace

namespace mujoco::python::studio {

void InitHeadlessUi(py::module_& m) {
  m.doc() = "MuJoCo web viewer headless Studio UI, streamed to the browser";

  py::class_<HeadlessUi>(m, "HeadlessUi")
      .def(py::init<const std::string&, const std::string&>(), py::arg("title"),
           py::arg("assets_dir") = "")
      .def("new_frame", &HeadlessUi::NewFrame,
           "Starts a headless ImGui frame, returning True once the frame is "
           "active. While a browser controls the UI, this returns at up to 60 "
           "frames per second (from its first input on). While no browser "
           "does, it returns False (no frame) after a short wait, so the "
           "caller's loop keeps running and can shut down.")
      .def("end_frame", &HeadlessUi::EndFrame,
           "Ends the frame and queues it for the browser (see take_messages).")
      .def("get_context", &HeadlessUi::GetContext)
      .def("get_implot_context", &HeadlessUi::GetImPlotContext)
      .def("upload_image", &HeadlessUi::UploadImage)
      .def("start_session", &HeadlessUi::StartSession, py::arg("handshake"),
           "Starts streaming to a newly connected browser. Takes the browser's "
           "first /ui message (its Hello handshake) and returns the "
           "reply to send back. Raises ValueError for an incompatible "
           "browser. The first frame of a session re-sends everything.")
      .def("end_session", &HeadlessUi::EndSession,
           "Stops streaming; new_frame() returns False until the next "
           "start_session().")
      .def("push_input", &HeadlessUi::PushInput, py::arg("message"),
           "Queues an Input message from the browser for the next "
           "frame. Returns False, ignoring the message, if it is not one.")
      .def("take_messages", &HeadlessUi::TakeMessages,
           "Returns the messages to send to the browser now, one Studio UI "
           "wire command each: none until a new frame is ready, then the "
           "texture changes since the last frame returned followed by the "
           "newest frame, compressed against that last frame.")
      .def("last_frame_index", &HeadlessUi::LastFrameIndex,
           "Returns the index of the last frame take_messages() returned (0 "
           "before the first). The browser acks a frame by echoing its index.");
}

}  // namespace mujoco::python::studio

#ifndef MUJOCO_STUDIO_COMBINED_MODULE
PYBIND11_MODULE(headless_ui, m, pybind11::mod_gil_not_used()) {
  mujoco::python::studio::InitHeadlessUi(m);
}
#endif
