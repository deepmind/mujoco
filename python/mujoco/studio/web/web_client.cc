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

// This file implements the Web Viewer Browser Client. The Studio UI wire codec
// streams remote ImGui draw data over /ui, and the Filament renderer draws
// both the UI and the 3D scene.

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/fetch.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <implot.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/imgui_bridge.h"
#include "render/filament/support/renderer.h"
#include "studio/hal/graphics_mode.h"
#include "studio/hal/window.h"
#include "studio/render/model_scene.h"
#include "studio/sim/model_holder.h"
#include "studio/ux/camera_control.h"
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/interaction.h"
#include "studio/ux/theme.h"
#include "logging.h"
#include "state_payload.h"
#include "web_client_local_ui.h"
#include "web_client_remote_ui.h"
#include "web_client_session.h"
#include "web_client_websocket.h"

#if !defined(__EMSCRIPTEN__)
#error "web_client.cc is only supported for Emscripten builds"
#endif

// No-op wrapper for glGetError(), installed via -Wl,--wrap=glGetError.
// Filament's GL backend calls glGetError() hundreds of times per frame
// through GLUtils::checkGLError/assertGLError. In WebGL each call forces a
// synchronous GPU pipeline flush across the JS-WASM bridge (~0.2ms each).
// This stub eliminates that cost entirely.
extern "C" GLenum __wrap_glGetError(void) { return GL_NO_ERROR; }

using ::mujoco::studio::DisconnectNotice;
using ::mujoco::studio::GuiTheme;
using ::mujoco::studio::OpenWebSocket;
using ::mujoco::studio::RemoteUi;
using ::mujoco::studio::RemoteUiState;
using ::mujoco::studio::RoleWindow;
using ::mujoco::studio::Session;
using ::mujoco::studio::SessionRole;
using ::mujoco::studio::SessionView;
using ::mujoco::studio::StatePayloadView;

// How a spectating page drives its camera. The free modes control the local
// camera directly; kSpecCamFollow mirrors the controller's camera from the
// state broadcast.
enum SpectatorCamMode {
  kSpecCamTumble = 0,  // Orbit around the lookat point with the mouse.
  kSpecCamWasd,        // Fly camera: WASD/QE moves, mouse drag turns.
  kSpecCamFollow,      // Follow the controller's camera.
};

// Byte-rate telemetry shown in the role window.
struct Telemetry {
  double last_rate_time = 0;
  uint64_t gui_bytes_per_sec = 0;
  uint64_t sim_bytes_per_sec = 0;
};

struct ModelDownloadStatus {
  size_t bytes_downloaded = 0;
  size_t total_bytes = 0;
  int retry_count = 0;
  bool is_downloading = true;
};

// The implementation of every interface needed by the session and remote UI.
class AppCallbacks final : public RemoteUi::Callbacks,
                           public Session::Callbacks {
 public:
  // RemoteUi::Callbacks
  uintptr_t UploadTexture(uintptr_t current, const std::byte* rgba,
                          uint32_t width, uint32_t height) override;
  bool GpuReady() override;

  // Session::Callbacks
  bool ReadyForPayload() override;
  void OnPayload(const StatePayloadView& view) override;
  void OnModelChanged(uint32_t model_crc32) override;
  void ConnectRemoteUi() override;
  void ShutdownRemoteUi() override;
  void SetCameraMode(int mode) override;
  void SetFpsCap(int fps_cap) override;
};

struct App {
  std::unique_ptr<mujoco::studio::Window> window;
  std::unique_ptr<mujoco::Renderer> renderer;
  std::unique_ptr<mujoco::studio::ModelHolder> model_holder;
  // Replacement for model_holder->model()->paths when the model has
  // by-reference textures: the original paths followed by the "asset:" URIs
  // they were rewritten to (see RegisterBundledTextures). Must outlive the
  // model, which points into it.
  std::vector<char> model_paths;
  std::unique_ptr<mujoco::studio::ModelScene> model_scene;
  std::unique_ptr<mujoco::ImguiBridge> imgui_bridge;
  mjvPerturb perturb;
  mjvCamera camera;
  mjvOption vis_options;
  mjtByte render_flags[mjNRNDFLAG];

  // Local camera control (MoveLocalCamera): shared by spectators in a free
  // camera mode and the controller, which maps input with the viewer's input
  // state from the latest payload and sends the moved camera back.
  int spectator_cam_mode = kSpecCamTumble;
  mujoco::studio::ViewerInputState viewer_input;
  bool view_mouse_focus = false;  // The 3D view has the mouse (drag) focus.
  float cam_speed = mujoco::studio::kMinFlySpeed;  // WASD fly speed.
  // The viewer's camera generation (ViewerInputState::camera_generation) as of
  // the last payload camera adopted; see client_camera.py. Invalid until this
  // page is the controller and has adopted one.
  uint32_t camera_generation = 0;
  bool camera_generation_valid = false;

  // The part of the canvas above the role bar, in logical pixels (x, y,
  // width, height): the 3D scene renders into it, the remote UI lays out in
  // it, 3D labels project into it. Refreshed every frame from the bar's
  // height, which follows the font scale.
  ImVec4 scene_rect = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

  Telemetry telemetry;

  // Main loop frame counters (MainLoopImpl): reconnect pacing for the state
  // WebSocket. (The session paces its own /ui claims by time.)
  int frame_count = 0;
  int last_state_retry_frame = 0;

  // Client framerate cap in Hz (0 for Uncapped; defaults to 60).
  int fps_cap = 60;
  double next_frame_time_ms = 0.0;

  // FIFO jitter buffer for state payloads received from the Python simulation
  // via WebSocket, plus a reusable scratch entry for popping on each frame.
  mujoco::studio::StateQueue state_queue;
  mujoco::studio::StateQueue::Entry scratch_state;

  // User-injected geoms received with the state payload.
  std::vector<mjvGeom> extra_geoms;

  ModelDownloadStatus download_status;

  AppCallbacks callbacks;
  RemoteUi remote_ui{callbacks};
  Session session{callbacks};
  DisconnectNotice disconnect_notice;
  RoleWindow role_window;
  GuiTheme theme = GuiTheme::kDark;
};
App g_app;

// WebSocket base URL matching the page origin, e.g. "ws://host:8080" or
// "wss://tunnel.example.com" behind an HTTPS tunnel. All viewer WebSockets
// are served as paths (/ui, /state) on the same host and port as the page
// itself, so exposing or tunneling that single port exposes the whole
// viewer.
std::string GetWsBaseUrl() {
  char* base = emscripten_run_script_string(
      "(window.location.protocol === 'https:' ? 'wss://' : 'ws://') + "
      "window.location.host");
  std::string url = base != nullptr ? base : "";
  if (url == "ws://" || url == "wss://" || url.empty()) {
    // No host in the page URL (e.g. file://) — assume a local server.
    return "ws://localhost:8080";
  }
  return url;
}

// Stable per-tab id sent with every WebSocket connect (?sid=...), letting the
// server tie this page's /ui and /state connections together. It lives in
// sessionStorage so it survives page reloads: on a model change every page
// reloads and the server restarts with the controller slot reserved for the
// previous controller's sid, a reload-stable id is what lets the controller
// keep control.
std::string GetSessionId() {
  static const std::string sid = emscripten_run_script_string(
      "(function() {"
      "  Module.session_id = Module.session_id ||"
      "      sessionStorage.getItem('mjwv_sid') ||"
      "      (Date.now().toString(36) + Math.random().toString(36).slice(2));"
      "  sessionStorage.setItem('mjwv_sid', Module.session_id);"
      "  return Module.session_id;"
      "})()");
  return sid;
}

std::string WsUrl(const char* path) {
  return GetWsBaseUrl() + path + "?sid=" + GetSessionId();
}

// Returns true if the Filament rendering context is initialized and ready for
// GPU texture uploads. The Renderer object is created in main() and is always
// non-null, but the Filament context is only initialized when Renderer::Init()
// is called from SetupScene after the async model fetch completes.
bool IsFilamentReady() {
  return g_app.renderer && g_app.model_holder && g_app.model_holder->ok();
}

// Enqueues a parsed state payload to be applied on the next rendered frame.
// Called via AppCallbacks::OnPayload after Session validates model CRC and
// payload readiness.
void ApplyStatePayload(const StatePayloadView& view) {
  // Physics state. While a new model is downloading, buffer it unconditionally
  // (the current model's state size is the old model's); MainLoopImpl checks
  // the size against the model actually in use right before mj_setState.
  if (view.physics != nullptr && !g_app.download_status.is_downloading) {
    mjModel* model = g_app.model_holder->model();
    const size_t expected_bytes =
        mj_stateSize(model, view.physics_spec) * sizeof(mjtNum);
    if (view.physics_bytes != expected_bytes) {
      LOG(Warning, "Physics state size mismatch (%zu != %zu); dropping",
          view.physics_bytes, expected_bytes);
      return;
    }
  }
  g_app.state_queue.Push(view);
}

void SetSpectatorCameraMode(int mode) {
  if (mode == g_app.spectator_cam_mode) {
    return;
  }
  g_app.spectator_cam_mode = mode;
  if (!g_app.model_holder || !g_app.model_holder->ok()) {
    return;
  }
  const mjModel* model = g_app.model_holder->model();
  if (mode == kSpecCamTumble) {
    mujoco::studio::SetCamera(model, &g_app.camera,
                              mujoco::studio::kTumbleCameraIdx);
  } else if (mode == kSpecCamWasd) {
    mujoco::studio::SetCamera(model, &g_app.camera,
                              mujoco::studio::kFreeCameraIdx);
  }
  // kSpecCamFollow: the next state payload restores the controller's camera.
}

// Moves g_app.camera from this frame's mouse drags/wheel and WASD/QE keys with
// the viewer's own camera controls (ux/camera_control.h), and returns whether
// the camera moved. `viewer_mouse_focus` and `viewer_keyboard_focus` tell
// whether the viewer's UI (for the controller) leaves the mouse/keyboard focus
// to the 3D view; this page's own local UI is checked here as well.
bool MoveLocalCamera(const mjModel* model, bool wasd, bool viewer_mouse_focus,
                     bool viewer_keyboard_focus) {
  mujoco::studio::CameraInput input = mujoco::studio::ReadImGuiCameraInput();

  // Mouse motion is normalised by the scene rect (the canvas above the role
  // bar), the area the camera actually renders into, not the whole display.
  const ImVec4& rect = g_app.scene_rect;
  if (rect.z > 0 && rect.w > 0) {
    input.mouse_dx = ImGui::GetIO().MouseDelta.x / rect.z;
    input.mouse_dy = ImGui::GetIO().MouseDelta.y / rect.w;
  } else {
    input.mouse_focus = false;
  }

  // The viewer's mouse focus arrives a round trip late, so whether the view
  // has the mouse focus for a drag is decided as it starts and kept until it
  // ends, as ImGui does for its own UI.
  const bool any_button =
      ((input.left_button | input.right_button | input.middle_button) &
       mujoco::studio::kButtonPressed) != 0;
  if (!any_button || ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
      ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
      ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
    g_app.view_mouse_focus = viewer_mouse_focus && input.mouse_focus;
  }
  input.mouse_focus = g_app.view_mouse_focus;
  input.keyboard_focus = viewer_keyboard_focus && input.keyboard_focus;

  bool moved =
      mujoco::studio::MoveCameraWithMouse(model, &g_app.camera, wasd, input);
  if (wasd) {
    moved |= mujoco::studio::MoveCameraWithKeys(model, &g_app.camera, input,
                                                &g_app.cam_speed);
  }
  return moved;
}

// Local camera control for a spectating page in a free camera mode. The
// controller moves its camera in HandleControllerCameraInput instead.
void HandleSpectatorCameraInput() {
  if (g_app.spectator_cam_mode == kSpecCamFollow) {
    return;
  }
  if (!g_app.model_holder || !g_app.model_holder->ok() ||
      g_app.download_status.is_downloading) {
    return;
  }
  const mjModel* model = g_app.model_holder->model();
  const bool wasd = g_app.spectator_cam_mode == kSpecCamWasd;

  // The camera can be fixed on entry (SetupScene honours the model's
  // vis.global.cameraid, and Follow Controller mirrors whatever camera the
  // controller uses), and mjv_moveCamera ignores fixed cameras. Coerce to a
  // free camera so the free modes always respond to input.
  if (g_app.camera.type == mjCAMERA_FIXED) {
    mjv_defaultFreeCamera(model, &g_app.camera);
    if (wasd) {
      mujoco::studio::SetCamera(model, &g_app.camera,
                                mujoco::studio::kFreeCameraIdx);
    }
  }

  MoveLocalCamera(model, wasd, /*viewer_mouse_focus=*/true,
                  /*viewer_keyboard_focus=*/true);
}

// Local camera control for the controlling page. Drags and scrolls over the 3D
// view, and WASD/QE keys in the free camera, move the camera here, at the
// page's frame rate rather than once per network round trip, with the same
// camera controls as the viewer, and the camera is then sent to the viewer.
// The viewer still receives the input (for picking, perturbation and
// shortcuts) but leaves the camera alone (see Viewer.client_camera).
void HandleControllerCameraInput() {
  if (!g_app.model_holder || !g_app.model_holder->ok() ||
      g_app.download_status.is_downloading) {
    return;
  }
  // Until the first payload as controller arrives, the camera (and generation)
  // is still the spectator's, and the viewer's would replace any move anyway.
  if (!g_app.camera_generation_valid) {
    return;
  }
  const mjModel* model = g_app.model_holder->model();
  const mujoco::studio::ViewerInputState& input = g_app.viewer_input;
  const bool wasd = input.camera_index == mujoco::studio::kFreeCameraIdx;
  if (MoveLocalCamera(model, wasd, input.mouse_focus != 0,
                      input.keyboard_focus != 0)) {
    g_app.session.SendCamera(g_app.camera, g_app.camera_generation);
  }
}

// Styles the browser's local ImGui context (the role bar and notices) with
// one of the Studio themes. Picked with the role bar's theme button and
// remembered in localStorage so it survives page reloads (every model change
// reloads the page). The remote UI follows this page's theme while it
// controls (see Session::SetTheme).
constexpr char kThemeStorageKey[] = "mjwv_theme";

// The theme's key in localStorage: "light", "dark" or "classic".
const char* ThemeStorageValue(GuiTheme theme) {
  switch (theme) {
    case GuiTheme::kLight:
      return "light";
    case GuiTheme::kClassic:
      return "classic";
    case GuiTheme::kDark:
      return "dark";
  }
  return "dark";
}

// The theme stored by a previous page load, or dark.
GuiTheme LoadTheme() {
  const std::string script = std::string("(function() { try { return ") +
                             "localStorage.getItem('" + kThemeStorageKey +
                             "') || ''; } catch (e) { return ''; } })()";
  const char* name = emscripten_run_script_string(script.c_str());
  const std::string stored = name != nullptr ? name : "";
  if (stored == "light") return GuiTheme::kLight;
  if (stored == "classic") return GuiTheme::kClassic;
  return GuiTheme::kDark;
}

void ApplyTheme(GuiTheme theme) {
  g_app.theme = theme;
  // SetupTheme sizes paddings from FontScaleDpi, which Window set for the
  // display's DPI. The font scale itself stays 1: browser zoom scales the
  // whole page, bar included.
  mujoco::studio::SetupTheme(theme);
  const std::string script = std::string("try { localStorage.setItem('") +
                             kThemeStorageKey + "', '" +
                             ThemeStorageValue(theme) + "'); } catch (e) {}";
  emscripten_run_script(script.c_str());
  // The remote UI follows this page's theme while it controls.
  g_app.session.SetTheme(ThemeStorageValue(theme));
}

// --- AppCallbacks method bodies (see the class declaration above App). ---

uintptr_t AppCallbacks::UploadTexture(uintptr_t current, const std::byte* rgba,
                                      uint32_t width, uint32_t height) {
  // All GUI textures are RGBA8. When rgba is nullptr this is a destruction
  // request and the format is ignored.
  const size_t num_bytes = width * height * 4;
  if (rgba) {
    return g_app.imgui_bridge->UploadImage(current, {rgba, num_bytes}, width,
                                           height);
  } else {
    return g_app.imgui_bridge->UploadImage(current, {}, width, height);
  }
}

bool AppCallbacks::GpuReady() { return IsFilamentReady(); }

bool AppCallbacks::ReadyForPayload() {
  return g_app.model_holder && g_app.model_holder->ok();
}

void AppCallbacks::OnPayload(const StatePayloadView& view) {
  ApplyStatePayload(view);
}

void AppCallbacks::OnModelChanged(uint32_t model_crc32) {
  // Mark downloading immediately so the main loop stops applying state updates
  // to the old model while the new one downloads asynchronously. The page
  // loads the bundle from its cache when it was prefetched (synchronously, so
  // the payload that announced the change already applies to the new model),
  // and fetches it by crc32 otherwise. The crc32 crosses as a signed int; the
  // page restores the unsigned value.
  g_app.download_status.is_downloading = true;
  g_app.state_queue.Clear();
  EM_ASM({ reloadModel($0); }, model_crc32);
}

void AppCallbacks::ConnectRemoteUi() {
  g_app.remote_ui.Connect(OpenWebSocket(WsUrl("/ui")));
}

void AppCallbacks::ShutdownRemoteUi() { g_app.remote_ui.Shutdown(); }

void AppCallbacks::SetCameraMode(int mode) { SetSpectatorCameraMode(mode); }

void AppCallbacks::SetFpsCap(int fps_cap) {
  fps_cap = std::max(0, fps_cap);
  g_app.fps_cap = fps_cap;
  g_app.next_frame_time_ms = 0.0;
  const std::string script = "try { localStorage.setItem('mjwv_fps_cap', '" +
                             std::to_string(fps_cap) + "'); } catch (e) {}";
  emscripten_run_script(script.c_str());
}

void BuildBrowserGui() {
  // Refresh the byte-rate telemetry the role bar shows.
  const double now = ImGui::GetTime();
  if (now - g_app.telemetry.last_rate_time >= 1.0) {
    g_app.telemetry.gui_bytes_per_sec = g_app.remote_ui.ConsumeByteCount();
    g_app.telemetry.sim_bytes_per_sec = g_app.session.ConsumeByteCount();
    g_app.telemetry.last_rate_time = now;
  }

  const double last_msg = g_app.session.LastMessageTime();
  const double stale_sec =
      last_msg > 0 ? emscripten_get_now() / 1000.0 - last_msg : -1.0;
  g_app.disconnect_notice.Draw(g_app.session.ServerCloseCode(),
                               g_app.session.ServerCloseReason().c_str(),
                               stale_sec, g_app.download_status.is_downloading,
                               g_app.role_window.BarHeight());

  SessionView view;
  g_app.session.FillView(&view);
  view.gui_bytes_per_sec = g_app.telemetry.gui_bytes_per_sec;
  view.sim_bytes_per_sec = g_app.telemetry.sim_bytes_per_sec;
  view.have_remote_frame = g_app.remote_ui.RemoteDrawData() != nullptr;
  view.seconds_since_last_payload = stale_sec;
  view.camera_mode = g_app.spectator_cam_mode;
  view.fps_cap = g_app.fps_cap;
  view.is_downloading = g_app.download_status.is_downloading;
  view.bytes_downloaded = g_app.download_status.bytes_downloaded;
  view.total_bytes = g_app.download_status.total_bytes;
  view.retry_count = g_app.download_status.retry_count;
  // The session is the SessionActions implementation: the role window's
  // intents land there directly.
  GuiTheme theme = g_app.theme;
  g_app.role_window.Draw(view, g_app.session, &theme);
  if (theme != g_app.theme) {
    ApplyTheme(theme);
  }
}

// The canvas's backing store in device pixels: the render target, the
// scene viewport and the UI's orthographic camera all have to be exactly
// this size. SDL sizes it to floor(css size * devicePixelRatio) on each
// axis, so with a fractional ratio the height is not the CSS height times
// the width's scale: that lands one row short for about a quarter of all
// window sizes, which left the top row of the canvas unrendered (a dark
// line above the menu bar) and squashed the UI pass by a row so its bottom
// edges landed between pixels.
static int FramebufferWidth() { return g_app.window->GetDrawableWidth(); }
static int FramebufferHeight() { return g_app.window->GetDrawableHeight(); }

// The scene's viewport in framebuffer pixels (GL convention: origin at the
// bottom-left): the window minus the role bar along the bottom. The bar's
// inset is floored so that at a fractional DPI scale the scene reaches at
// most one pixel under the opaque bar rather than leaving a gap above it.
static mjrRect SceneViewport(int width, int height) {
  const float scale_y = static_cast<float>(height) /
                        static_cast<float>(g_app.window->GetHeight());
  const int bar_px =
      static_cast<int>(std::floor(g_app.role_window.BarHeight() * scale_y));
  const int scene_height = std::max(1, height - bar_px);
  return {0, height - scene_height, width, scene_height};
}

static mujoco::Renderer::View GetSceneView() {
  const mjModel* model = g_app.model_holder->model();
  const mjData* data = g_app.model_holder->data();
  const int width = FramebufferWidth();
  const int height = FramebufferHeight();

  mujoco::Renderer::View view;
  view.scene = g_app.renderer->GetSceneHandle("studio");
  view.camera = mjv_camera2GLCamera(model, data, &g_app.camera);
  // The renderer truncates rel_viewport * target size to whole pixels; the
  // half pixel keeps float error from landing one pixel short of the bar.
  const mjrRect viewport = SceneViewport(width, height);
  view.rel_viewport[1] = (viewport.bottom + 0.5f) / height;
  view.rel_viewport[3] = (viewport.height + 0.5f) / height;
  if (g_app.render_flags[mjRND_SEGMENT]) {
    if (g_app.render_flags[mjRND_IDCOLOR]) {
      view.draw_mode = mjDRAW_MODE_SEGMENTATION_BY_ID;
    } else {
      view.draw_mode = mjDRAW_MODE_SEGMENTATION_BY_COLOR;
    }
  } else if (g_app.render_flags[mjRND_DEPTH]) {
    view.draw_mode = mjDRAW_MODE_DEPTH;
  } else if (g_app.render_flags[mjRND_WIREFRAME]) {
    view.draw_mode = mjDRAW_MODE_WIREFRAME;
  } else if (g_app.vis_options.flags[mjVIS_ISLAND]) {
    view.draw_mode = mjDRAW_MODE_ISLANDS;
  } else {
    view.draw_mode = mjDRAW_MODE_DEFAULT;
  }
  view.shadows = g_app.render_flags[mjRND_SHADOW];
  view.reflections = g_app.render_flags[mjRND_REFLECTION];
  view.skybox = g_app.render_flags[mjRND_SKYBOX];
  view.haze = g_app.render_flags[mjRND_HAZE];
  view.post_processing = g_app.render_flags[mjRND_POST_PROCESSING];
  if (model->vis.headlight.active) {
    view.headlight = mujoco::Renderer::View::Headlight();
    view.headlight->color[0] = model->vis.headlight.diffuse[0];
    view.headlight->color[1] = model->vis.headlight.diffuse[1];
    view.headlight->color[2] = model->vis.headlight.diffuse[2];
    view.headlight->intensity = g_app.model_scene->GetHeadlightIntensity();
  }
  return view;
}

static mujoco::Renderer::View GetUxView() {
  const int width = FramebufferWidth();
  const int height = FramebufferHeight();

  mujoco::Renderer::View view;
  view.scene = g_app.renderer->GetSceneHandle("ux");
  view.shadows = false;
  view.reflections = false;
  view.skybox = false;
  view.haze = false;
  view.post_processing = false;
  view.camera.orthographic = true;
  view.camera.pos[0] = 0.0f;
  view.camera.pos[1] = 0.0f;
  view.camera.pos[2] = 1.0f;
  view.camera.forward[0] = 0.0f;
  view.camera.forward[1] = 0.0f;
  view.camera.forward[2] = -1.0f;
  view.camera.up[0] = 0.0f;
  view.camera.up[1] = 1.0f;
  view.camera.up[2] = 0.0f;
  view.camera.frustum_top = 0.0f;
  view.camera.frustum_near = 0.0f;
  view.camera.frustum_far = 1.0f;
  view.camera.frustum_center = width / 2.0f;
  view.camera.frustum_width = width / 2.0f;
  view.camera.frustum_bottom = height;
  return view;
}

//=================================================================================================
// Main loop — called once per frame.
//=================================================================================================
void MainLoopImpl();

void MainLoop() {
  // An exception escaping the requestAnimationFrame callback kills the main
  // loop silently causing the canvas to freeze on the last rendered frame and
  // input capture to stop with only an opaque "Uncaught <ptr>" in the console.
  // Here we catch, log, and stop explicitly instead.
  try {
    MainLoopImpl();
  } catch (const std::exception& e) {
    LOG(Error, "FATAL: uncaught exception in MainLoop: %s", e.what());
    emscripten_cancel_main_loop();
  } catch (...) {
    LOG(Error, "FATAL: uncaught non-std exception in MainLoop");
    emscripten_cancel_main_loop();
  }
}

void MainLoopImpl() {
  if (g_app.fps_cap > 0) {
    const double now_ms = emscripten_get_now();
    const double period_ms = 1000.0 / static_cast<double>(g_app.fps_cap);
    // Allow 2ms of VSync timestamp jitter so a 60 FPS cap on a 60Hz display
    // never skips a VSync, while cleanly throttling on 120Hz+ displays.
    if (g_app.next_frame_time_ms > 0.0 &&
        now_ms + 2.0 < g_app.next_frame_time_ms) {
      return;
    }
    if (g_app.next_frame_time_ms <= 0.0 ||
        now_ms - g_app.next_frame_time_ms > period_ms) {
      g_app.next_frame_time_ms = now_ms + period_ms;
    } else {
      g_app.next_frame_time_ms += period_ms;
    }
  }

  g_app.frame_count++;

  // Session upkeep (the liveness heartbeat).
  g_app.session.Update();

  // Reconnect the state WebSocket if it dropped. Receiving a payload with a
  // different model identity then triggers a page reload (Session). Deliberate
  // server closes (session full, inactivity, the viewer stopped with an error)
  // are transient; retry them too, using a gentler pace.
  const int state_retry_interval =
      g_app.session.ServerCloseCode() != 0 ? 300 : 60;
  if (!g_app.session.HasSocket() && g_app.model_holder &&
      g_app.model_holder->ok() && !g_app.download_status.is_downloading &&
      g_app.frame_count - g_app.last_state_retry_frame > state_retry_interval) {
    g_app.last_state_retry_frame = g_app.frame_count;
    LOG(Info, "State WebSocket down; reconnecting...");
    g_app.session.Connect(OpenWebSocket(WsUrl("/state")));
  }

  // Pass the remote UI (/ui) WebSocket state to the session state machine.
  // The session uses this to manage attempts to claim the single controller
  // slot, handle retry pacing if a claim is rejected or dropped, and drive role
  // transitions (Controlling vs Spectating).
  RemoteUiState ui_state = RemoteUiState::kNoSocket;
  if (g_app.remote_ui.HasSocket()) {
    switch (g_app.remote_ui.ConnectionState()) {
      case RemoteUi::ReadyState::kOpen:
        ui_state = g_app.remote_ui.IsSessionAccepted()
                       ? RemoteUiState::kOpen
                       : RemoteUiState::kConnecting;
        break;
      case RemoteUi::ReadyState::kClosed:
      case RemoteUi::ReadyState::kError:
        ui_state = RemoteUiState::kClosedOrError;
        break;
      default:
        ui_state = RemoteUiState::kConnecting;
        break;
    }
  }
  g_app.session.HandleRemoteUiState(ui_state, g_app.remote_ui.CloseCode());

  // Process incoming UI data BEFORE the ImGui frame: textures and draw
  // frames must be ready before we start the local ImGui frame and render.
  // The role bar's height (it follows the font scale) sets this frame's
  // scene rect: the remote UI lays out in it and its draw data is clipped
  // to it, so the streamed Studio UI never runs under the bar.
  if (g_app.window) {
    const float bar_h = g_app.role_window.BarHeight();
    g_app.scene_rect = ImVec4(
        0.0f, 0.0f, static_cast<float>(g_app.window->GetWidth()),
        std::max(0.0f, static_cast<float>(g_app.window->GetHeight()) - bar_h));
    g_app.remote_ui.SetScreenArea(g_app.scene_rect.z, g_app.scene_rect.w);
  }
  g_app.remote_ui.ReceiveAndProcessCommands(g_app.frame_count);

  // Event loop and ImGui NewFrame via window abstraction.
  mujoco::studio::Window::Status status = g_app.window->NewFrame();
  if (status == mujoco::studio::Window::kQuitting) {
    // NewFrame() started an ImGui frame; end it before bailing out.
    ImGui::EndFrame();
    emscripten_cancel_main_loop();
    return;
  }

  // For the controller, the camera moves here (HandleControllerCameraInput) and
  // is sent to the headless viewer, while all other scene interaction
  // (perturbation, picking) is handled by the viewer: CaptureAndSendInput()
  // forwards this frame's input over /ui, and the resulting state streams back
  // over the state WebSocket (see ApplyStatePayload). Spectators have no input
  // channel; in a free camera mode they drive their local camera directly.
  g_app.remote_ui.CaptureAndSendInput();
  if (g_app.session.Role() == SessionRole::kControlling) {
    if (!ImGui::GetIO().WantCaptureMouse) {
      ImGui::SetMouseCursor(g_app.remote_ui.MouseCursor());
    }
    HandleControllerCameraInput();
  } else {
    g_app.camera_generation_valid = false;
    if (g_app.session.Role() == SessionRole::kSpectating) {
      HandleSpectatorCameraInput();
    }
  }

  BuildBrowserGui();

  // Sync the 3D scene before ImGui::Render() because ModelScene::Sync() calls
  // DrawTextAt, which emits 3D label text into the active ImGui frame via
  // ImGui::Begin("labels", ...).
  bool can_submit = false;
  if (g_app.renderer && g_app.model_holder && g_app.model_holder->ok()) {
    // Pop one queued state payload per rendered frame so network micro-bursts
    // play out smoothly across consecutive VSync intervals with physics state,
    // camera, perturbation, options, and extra geoms kept in sync.
    if (!g_app.download_status.is_downloading &&
        g_app.state_queue.Pop(&g_app.scratch_state)) {
      mjModel* model = g_app.model_holder->model();
      mjData* data = g_app.model_holder->data();
      if (g_app.scratch_state.has_render_state) {
        const mujoco::studio::RenderStateView& rs =
            g_app.scratch_state.render_state;
        // Spectators in a free camera mode keep their own local camera, and
        // those in Follow Controller mirror the controller's camera. The
        // controller moves its camera itself and sends it to the viewer, so the
        // payload camera is mostly an echo of one it sent; it adopts it only
        // when the viewer changed the camera itself (a new generation, see
        // client_camera.py), e.g. from a camera shortcut or the view gizmo.
        bool adopt_camera;
        if (g_app.session.Role() == SessionRole::kControlling) {
          const uint32_t generation =
              g_app.scratch_state.viewer_input.camera_generation;
          adopt_camera = !g_app.camera_generation_valid ||
                         generation != g_app.camera_generation;
          g_app.camera_generation = generation;
          g_app.camera_generation_valid = true;
        } else {
          adopt_camera = g_app.session.Role() != SessionRole::kSpectating ||
                         g_app.spectator_cam_mode == kSpecCamFollow;
          // Adopt again once controlling.
          g_app.camera_generation_valid = false;
        }
        if (adopt_camera) {
          g_app.camera = rs.camera;
        }
        g_app.perturb = rs.perturb;
        g_app.vis_options = rs.vis_options;
        model->opt = rs.opt;
        model->vis = rs.vis;
        model->stat = rs.stat;
        memcpy(g_app.render_flags, rs.render_flags, mjNRNDFLAG);
      }
      if (!g_app.scratch_state.state.empty()) {
        const size_t expected_size =
            static_cast<size_t>(mj_stateSize(model, g_app.scratch_state.sig));
        if (g_app.scratch_state.state.size() == expected_size) {
          mj_setState(model, data, g_app.scratch_state.state.data(),
                      g_app.scratch_state.sig);
          mj_forward(model, data);
        }
      }
      g_app.extra_geoms.swap(g_app.scratch_state.extra_geoms);
      g_app.viewer_input = g_app.scratch_state.viewer_input;
    }

    const int width = FramebufferWidth();
    const int height = FramebufferHeight();
    if (width > 0 && height > 0) {
      g_app.renderer->ResizeRenderTarget(mujoco::Renderer::kWindowRenderTarget,
                                         width, height);

      if (g_app.model_scene) {
        g_app.model_scene->Sync(
            g_app.model_holder->model(), g_app.model_holder->data(),
            SceneViewport(width, height), &g_app.vis_options, &g_app.perturb,
            &g_app.camera, std::span(g_app.extra_geoms));
      }
      can_submit = true;
    }
  }

  // Finalize the local ImGui draw data. ImguiBridge::Update() will call
  // ImGui::Render() again, but that is a no-op once the frame has already been
  // rendered.
  ImGui::Render();

  // Inject remote draw lists into the local ImDrawData so that Filament's
  // ImguiBridge renders them alongside the local UI. 3D scene labels render
  // behind remote UI, and browser-local UI (role/disconnect overlays) renders
  // in the foreground. Note that ImGui::GetDrawData() is only valid after
  // ImGui::Render() and until the next call to ImGui::NewFrame().
  ImDrawData* remote_draw_data = g_app.remote_ui.RemoteDrawData();
  if (remote_draw_data && remote_draw_data->Valid &&
      g_app.session.Role() == SessionRole::kControlling) {
    ImDrawData* local_draw_data = ImGui::GetDrawData();
    if (local_draw_data) {
      // Save local draw lists.
      ImVector<ImDrawList*> local_lists;
      local_lists.reserve(local_draw_data->CmdLists.Size);
      for (int i = 0; i < local_draw_data->CmdLists.Size; ++i) {
        local_lists.push_back(local_draw_data->CmdLists[i]);
      }

      // Clear and rebuild: 3D scene labels first, then remote, then local UI.
      local_draw_data->CmdLists.resize(0);
      local_draw_data->TotalVtxCount = 0;
      local_draw_data->TotalIdxCount = 0;

      auto is_label_list = [](const ImDrawList* list) {
        return list->_OwnerName != nullptr &&
               std::strncmp(list->_OwnerName, "labels", 6) == 0;
      };

      // 3D scene labels (background).
      for (int i = 0; i < local_lists.Size; ++i) {
        if (is_label_list(local_lists[i])) {
          local_draw_data->AddDrawList(local_lists[i]);
        }
      }

      // Remote draw lists.
      for (int i = 0; i < remote_draw_data->CmdLists.Size; ++i) {
        local_draw_data->AddDrawList(remote_draw_data->CmdLists[i]);
      }

      // Local browser overlay draw lists (foreground).
      for (int i = 0; i < local_lists.Size; ++i) {
        if (!is_label_list(local_lists[i])) {
          local_draw_data->AddDrawList(local_lists[i]);
        }
      }
    }
  }

  // Render the scene and all ImGui UI (local and remote).
  if (can_submit) {
    g_app.imgui_bridge->Update();

    mujoco::Renderer::View views[2] = {GetSceneView(), GetUxView()};
    g_app.renderer->Render(views);
  }

  g_app.window->Present();
}

void SetupScene(mjModel* m) {
  g_app.model_scene.reset();
  if (m) {
    // 3D labels project into the scene rect, not the whole window: the
    // scene renders above the role bar.
    g_app.model_scene = std::make_unique<mujoco::studio::ModelScene>(
        g_app.renderer->GetContext(), g_app.renderer->GetScene("studio"), m,
        [](const char* text, float x, float y, float z) {
          mujoco::studio::DrawTextAtInRect(text, x, y, z, g_app.scene_rect);
        });
  }

  // Upload any CPU-buffered streamed UI textures without GPU handles yet.
  g_app.remote_ui.UpdateTextures();

  // Only initialize default camera/perturb/vis_options before the state
  // WebSocket connects; on model reloads, keep the current camera and visual
  // state streamed from the Python viewer so the view does not snap to the
  // default camera for a frame.
  if (!g_app.session.HasSocket()) {
    mjv_defaultPerturb(&g_app.perturb);
    mjv_defaultCamera(&g_app.camera);
    mjv_defaultOption(&g_app.vis_options);

    const int model_cam = m->vis.global.cameraid;
    if (model_cam >= 0 && model_cam < m->ncam) {
      mujoco::studio::SetCamera(m, &g_app.camera, model_cam);
    } else {
      mjv_defaultFreeCamera(m, &g_app.camera);
    }
  }
}

// Standard CRC-32 (ISO-HDLC, poly 0xEDB88320), matching Python's zlib.crc32
// so the fetched model's checksum can be compared against the crc the Python
// side stamps into each state payload.
uint32_t Crc32(const uint8_t* data, size_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

// Holds assets that the resource providers below resolve without a
// filesystem: Filament assets and local ImGui fonts pushed in by the page via
// the registerAsset() binding (owned copies), and the by-reference textures of
// the current model bundle (non-owning views into the download buffer, see
// RegisterBundledTextures).
class AssetRegistry {
 public:
  static AssetRegistry& Instance() {
    static AssetRegistry instance;
    return instance;
  }

  // Registers an owned copy of `contents`.
  void RegisterAsset(std::string filename, std::string contents) {
    std::string& owned = owned_[filename];
    owned = std::move(contents);
    assets_[std::move(filename)] = owned;
  }

  // Registers a view of `contents`, which the caller must keep alive until the
  // asset is cleared again.
  void RegisterAssetView(std::string filename, std::string_view contents) {
    assets_[std::move(filename)] = contents;
  }

  // Removes every asset whose name starts with `prefix`.
  void ClearPrefix(std::string_view prefix) {
    std::erase_if(assets_, [prefix](const auto& kv) {
      return std::string_view(kv.first).starts_with(prefix);
    });
    std::erase_if(owned_, [prefix](const auto& kv) {
      return std::string_view(kv.first).starts_with(prefix);
    });
  }

  // Looks up "<provider>:<filename>"; empty if unknown.
  std::string_view Get(std::string_view filename) const {
    filename = filename.substr(filename.find_first_of(':') + 1);
    auto it = assets_.find(std::string(filename));
    return it != assets_.end() ? it->second : std::string_view();
  }

 private:
  std::unordered_map<std::string, std::string_view> assets_;
  std::unordered_map<std::string, std::string> owned_;
};

// Prefix of the AssetRegistry names of the current model's textures.
constexpr std::string_view kBundledTexturePrefix = "tex_";

// Makes the by-reference textures shipped in `bundle` loadable by `m`.
//
// mj_loadTexture() (called for every by-reference texture when SetupScene()
// builds the scene) opens m->paths + tex_pathadr[i] with no VFS, so the only
// hook the browser has is a resource provider selected by URI prefix. Each
// texture's path is therefore rewritten to "asset:tex_<i><ext>" and its bytes
// registered under that name. The names are deliberately flat: the VFS would
// try to mount any ".mjz/" segment of the original path as an archive. The
// extension is kept so the content type can still be derived from it.
//
// The rewritten paths live in g_app.model_paths. Only m->paths is redirected
// there; m->npaths is left alone so the model stays consistent with its own
// buffer for mj_sizeModel/mj_copyModel/mj_saveModel, which would all still
// see the original paths.
static void RegisterBundledTextures(mjModel* m,
                                    const mujoco::studio::ModelBundleView& b) {
  std::vector<char>& paths = g_app.model_paths;
  paths.assign(m->paths, m->paths + m->npaths);
  bool rewritten = false;
  for (int i = 0; i < m->ntex; ++i) {
    if (m->tex_adr[i] != -1 || m->tex_pathadr[i] < 0) continue;
    std::string_view path(m->paths + m->tex_pathadr[i]);
    auto it = std::find_if(b.resources.begin(), b.resources.end(),
                           [path](const auto& r) { return r.path == path; });
    if (it == b.resources.end()) {
      LOG(Warning, "Texture %d (%.*s) is not in the model bundle", i,
          static_cast<int>(path.size()), path.data());
      continue;
    }
    size_t dot = path.find_last_of('.');
    size_t slash = path.find_last_of("/\\");
    std::string_view ext =
        dot != std::string_view::npos &&
                (slash == std::string_view::npos || dot > slash)
            ? path.substr(dot)
            : std::string_view();
    std::string name = std::string(kBundledTexturePrefix) + std::to_string(i) +
                       std::string(ext);
    AssetRegistry::Instance().RegisterAssetView(
        name,
        std::string_view(reinterpret_cast<const char*>(it->data), it->size));
    const std::string uri = "asset:" + name;
    m->tex_pathadr[i] = static_cast<int>(paths.size());
    paths.insert(paths.end(), uri.begin(), uri.end());
    paths.push_back('\0');
    rewritten = true;
  }
  if (rewritten) m->paths = paths.data();
}

// Parses a model bundle and installs its model; returns the bundle's crc32
// (the identity the state payloads carry), or 0 on failure.
uint32_t ParseModelBufferImpl(const char* data, size_t size) {
  LOG(Info, "Fetched model bundle, size: %zu", size);
  mujoco::studio::ModelBundleView bundle;
  if (!mujoco::studio::ParseModelBundle(data, size, &bundle)) {
    LOG(Error, "Failed to load model: malformed model bundle");
    return 0;
  }
  LOG(Info, "Model bundle: %zu byte MJB, %zu texture file(s)", bundle.mjb_size,
      bundle.resources.size());
  g_app.model_holder = mujoco::studio::ModelHolder::FromBuffer(
      std::span<const std::byte>(bundle.mjb, bundle.mjb_size),
      "application/mjb", "model.mjb");
  if (g_app.model_holder && g_app.model_holder->ok()) {
    mjModel* m = g_app.model_holder->model();
    if (m && m->name_pluginadr) {
      for (int i = 0; i < m->nplugin; ++i) {
        if (m->name_pluginadr[i] < 0) continue;
        std::string_view iname(m->names + m->name_pluginadr[i]);
        if (iname.empty()) continue;
        for (int s = 0; s < mjp_pluginCount(); ++s) {
          const mjpPlugin* p = mjp_getPluginAtSlot(s);
          if (!p || !p->name) continue;
          std::string_view pname(p->name);
          if (pname == iname ||
              (pname.size() > iname.size() && pname.ends_with(iname) &&
               pname[pname.size() - iname.size() - 1] == '.')) {
            m->plugin[i] = s;
            break;
          }
        }
      }
    }
    if (m) RegisterBundledTextures(m, bundle);
    const uint32_t crc = Crc32(reinterpret_cast<const uint8_t*>(data), size);
    g_app.session.SetModelCrc32(crc);
    LOG(Info, "Model parsed successfully!");
    return crc;
  }
  LOG(Error, "Failed to load model: %s",
      g_app.model_holder ? g_app.model_holder->error().data()
                         : "Unknown error");
  return 0;
}

void UpdateModelDownloadProgress(size_t bytes_downloaded, size_t total_bytes,
                                 int retry_count) {
  g_app.download_status.is_downloading = true;
  g_app.download_status.bytes_downloaded = bytes_downloaded;
  g_app.download_status.total_bytes = total_bytes;
  g_app.download_status.retry_count = retry_count;
}

void FinishModelLoad() {
  mj_forward(g_app.model_holder->model(), g_app.model_holder->data());
  if (g_app.window) {
    SetupScene(g_app.model_holder->model());
  }
  // Connect the state WebSocket to receive simulation state from Python.
  if (!g_app.session.HasSocket()) {
    g_app.session.Connect(OpenWebSocket(WsUrl("/state")));
  }
  g_app.download_status.is_downloading = false;
  LOG(Info, "Model loaded and scene initialized successfully!");
}

// Allocates a buffer in WASM linear memory for zero-copy streaming chunk
// downloads. The caller owns the returned pointer and must free it via
// FreeModelBuffer when done.
uintptr_t AllocModelBuffer(size_t size) {
  char* ptr = new (std::nothrow) char[size];
  if (!ptr) {
    LOG(Error, "Failed to allocate WASM model buffer of size %zu", size);
    return 0;
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

// Frees a buffer previously returned by AllocModelBuffer.
void FreeModelBuffer(uintptr_t ptr_val) {
  if (ptr_val) {
    delete[] reinterpret_cast<char*>(ptr_val);
  }
}

// Parses a model bundle from a buffer and rebuilds the scene from it. The
// buffer is the page's to free once this returns. Returns the bundle's crc32,
// under which the page caches the bundle, or 0 if it failed to load.
uint32_t ParseModelBuffer(uintptr_t ptr_val, size_t size) {
  const uint32_t crc =
      ParseModelBufferImpl(reinterpret_cast<char*>(ptr_val), size);
  if (crc != 0) {
    FinishModelLoad();
  }
  // SetupScene() has uploaded every bundled texture to the GPU, and nothing
  // reads the encoded bytes again; drop the views before the buffer they
  // point into is freed. A re-upload would need the bundle fetched again.
  AssetRegistry::Instance().ClearPrefix(kBundledTexturePrefix);
  return crc;
}

// Exposed to JS (see EMSCRIPTEN_BINDINGS): the page calls this once per asset.
void RegisterAsset(std::string filename, emscripten::val contents) {
  std::string data;
  if (contents.typeOf().as<std::string>() == "string") {
    data = contents.as<std::string>();
  } else if (contents.instanceof(emscripten::val::global("Uint8Array")) ||
             contents.hasOwnProperty("length")) {
    size_t len = contents["length"].as<size_t>();
    data.resize(len);
    if (len > 0) {
      emscripten::val memory_view =
          emscripten::val(emscripten::typed_memory_view(len, data.data()));
      memory_view.call<void>("set", contents);
    }
  } else if (contents.instanceof(emscripten::val::global("ArrayBuffer"))) {
    emscripten::val u8 = emscripten::val::global("Uint8Array").new_(contents);
    size_t len = u8["length"].as<size_t>();
    data.resize(len);
    if (len > 0) {
      emscripten::val memory_view =
          emscripten::val(emscripten::typed_memory_view(len, data.data()));
      memory_view.call<void>("set", u8);
    }
  }
  AssetRegistry::Instance().RegisterAsset(std::move(filename), std::move(data));
}

// Registers resource providers so that "filament:", "font:" and "asset:"
// requests resolve from the AssetRegistry: the first two are populated by
// registerAsset(), the last by RegisterBundledTextures().
static void RegisterAssetProviders() {
  mjpResourceProvider resource_provider;
  mjp_defaultProvider(&resource_provider);

  resource_provider.open = [](mjResource* resource) {
    AssetRegistry& r = AssetRegistry::Instance();
    return static_cast<int>(r.Get(resource->name).size());
  };
  resource_provider.read = [](mjResource* resource, const void** buffer) {
    AssetRegistry& r = AssetRegistry::Instance();
    std::string_view contents = r.Get(resource->name);
    *buffer = contents.data();
    return static_cast<int>(contents.size());
  };
  resource_provider.close = [](mjResource* resource) {};
  for (const char* prefix : {"filament", "font", "asset"}) {
    resource_provider.prefix = prefix;
    mjp_registerProvider(&resource_provider);
  }
}

// Starts the viewer once the page has registered every asset. Exposed to JS
// and called from index.html after the fetches complete.
void StartApp() {
  mujoco::studio::Window::Config config;
  config.gfx_mode = mujoco::studio::GraphicsMode::FilamentWebGl;
  // Load the Studio UI fonts for the browser's local ImGui (the role window);
  // the "font:" resource provider above resolves them from the AssetRegistry.
  // (The streamed/remote UI carries its own font atlas separately.)
  config.load_fonts = true;

  g_app.window = std::make_unique<mujoco::studio::Window>("MuJoCo Web Viewer",
                                                          1400, 720, config);
  ImPlot::CreateContext();  // Needed if the server app uses ImPlot.
  // The local UI's theme from the first frame: the one this browser last
  // chose, or dark.
  ApplyTheme(LoadTheme());

  for (int i = 0; i < mjNRNDFLAG; i++) {
    g_app.render_flags[i] = (mjRNDSTRING[i][1][0] == '1');
  }

  mjrfContextConfig renderer_config;
  mjrf_defaultContextConfig(&renderer_config);
  renderer_config.native_window = g_app.window->GetNativeWindowHandle();
  renderer_config.graphics_api = mjGRAPHICS_API_OPENGL;
  g_app.renderer = std::make_unique<mujoco::Renderer>(renderer_config);
  g_app.imgui_bridge = std::make_unique<mujoco::ImguiBridge>(
      g_app.renderer->GetContext(), g_app.renderer->GetScene("ux"));

  // Initialize an empty dummy scene so Filament and ImGui are ready to render
  // the "DOWNLOADING..." progress bar while /model downloads asynchronously
  g_app.model_holder = mujoco::studio::ModelHolder::FromSpec(mj_makeSpec());
  if (g_app.model_holder && g_app.model_holder->ok()) {
    SetupScene(g_app.model_holder->model());
  }

  SDL_StartTextInput();

  const char* saved_fps_str = emscripten_run_script_string(
      "(function() { try { return localStorage.getItem('mjwv_fps_cap') || ''; "
      "} catch (e) { return ''; } })()");
  int saved_fps_cap = 60;
  if (saved_fps_str != nullptr && saved_fps_str[0] != '\0') {
    char* end = nullptr;
    const long parsed = std::strtol(saved_fps_str, &end, 10);
    if (end != saved_fps_str && *end == '\0' && parsed >= 0) {
      saved_fps_cap = static_cast<int>(parsed);
    }
  }
  g_app.callbacks.SetFpsCap(saved_fps_cap);

  // The UI stream is a path on the page's own host and port.
  g_app.remote_ui.Connect(OpenWebSocket(WsUrl("/ui")));

  emscripten_set_main_loop(MainLoop, 0, 0);
}

int main(int argc, char** argv) {
  RegisterAssetProviders();
  return 0;
}

EMSCRIPTEN_BINDINGS(web_client_bindings) {
  // Registers staged static files fetched by index.html into the C++ in-memory
  // AssetRegistry before startApp() runs.
  emscripten::function("registerAsset", &RegisterAsset);

  // Allocates a buffer in WASM linear memory for zero-copy streaming model
  // chunk downloads. The caller must free it via freeModelBuffer when done.
  emscripten::function("allocModelBuffer", &AllocModelBuffer);

  // Frees a buffer previously allocated by allocModelBuffer.
  emscripten::function("freeModelBuffer", &FreeModelBuffer);

  // Parses a completed model bundle buffer (MJB plus by-reference textures)
  // and reinitializes the Filament scene. Returns the bundle's crc32 (0 on
  // failure), the key the page caches the bundle under.
  emscripten::function("parseModelBuffer", &ParseModelBuffer);

  // Initializes the Filament window, ImGui context, WebSocket connections, and
  // starts the Emscripten main simulation loop.
  emscripten::function("startApp", &StartApp);

  // Updates the model download progress in C++ so the ImGui role window can
  // display a real-time progress bar and status while downloading.
  emscripten::function("updateModelDownloadProgress",
                       &UpdateModelDownloadProgress);
}
