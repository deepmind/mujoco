// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "studio/app.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include "webp/encode.h"
#include "webp/types.h"
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/imgui_editor.h"
#include "render/filament/support/imgui_bridge.h"
#include "render/filament/support/renderer.h"
#include "studio/hal/graphics_mode.h"
#include "studio/hal/window.h"
#include "studio/render/model_scene.h"
#include "studio/sim/model_holder.h"
#include "studio/sim/step_control.h"
#include "studio/ux/camera_control.h"
#include "studio/ux/console.h"
#include "studio/ux/file_dialog.h"
#include "studio/ux/gui.h"
#include "studio/ux/gui_helpers.h"
#include "studio/ux/gui_spec.h"
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/interaction.h"
#include "studio/ux/picture_gui.h"
#include "studio/ux/plugin.h"

namespace mujoco::studio {

using PauseState = StepControl::PauseState;

static void ToggleFlag(mjtByte& flag) { flag = flag ? 0 : 1; }

static void ToggleWindow(bool& window) {
  window = !window;
  ImGui::GetIO().WantSaveIniSettings = true;
}

static void SelectParentPerturb(const mjModel* model, mjvPerturb& perturb) {
  if (perturb.select > 0) {
    perturb.select = model->body_parentid[perturb.select];
    perturb.flexselect = -1;
    perturb.skinselect = -1;
    if (perturb.select <= 0) {
      perturb.active = 0;
    }
  }
  // TODO: update selected element!
}

static std::string CheckPathForFile(const std::filesystem::path& path, std::string_view filename) {
  std::filesystem::path resolved = path / filename;
  if (std::filesystem::exists(resolved)) {
    return resolved.string();
  }
  resolved += ".xml";
  if (std::filesystem::exists(resolved)) {
    return resolved.string();
  }
  return "";
}

// Attempts to find a file with the given name by recursively searching the
// given search paths.
static std::string ResolveFile(std::string_view filename,
                               const std::vector<std::string>& search_paths) {
  if (std::filesystem::exists(filename)) {
    return std::string(filename);
  }

  std::string resolved;
  for (const std::string& path : search_paths) {
    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
      continue;
    }

    resolved = CheckPathForFile(std::filesystem::path(path), filename);
    if (!resolved.empty()) {
      return resolved;
    }

    std::vector<std::filesystem::path> entries;
    for (const auto& it : std::filesystem::recursive_directory_iterator(path)) {
      entries.push_back(it.path());
    }
    std::sort(entries.begin(), entries.end());
    for (const auto& entry : entries) {
      resolved = CheckPathForFile(entry, filename);
      if (!resolved.empty()) {
        return resolved;
      }
    }
  }
  return std::string(filename);
}

// Exports the given image (assumed to be RGB888) to a webp file.
static void SaveToWebp(int width, int height, const std::byte* data, const std::string& filename) {
  uint8_t* webp = nullptr;
  const size_t size = WebPEncodeLosslessRGB(reinterpret_cast<const uint8_t*>(data), width, height,
                                            width * 3, &webp);
  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<const char*>(webp), size);
  file.close();
  WebPFree(webp);
}

static constexpr const char* ICON_RELOAD_MODEL = ICON_FA_REFRESH;
static constexpr const char* ICON_RESET_MODEL = ICON_FA_UNDO;
static constexpr const char* ICON_UNDO_SPEC = ICON_FA_UNDO;
static constexpr const char* ICON_REDO_SPEC = ICON_FA_REPEAT;

App::App(Config config)
    : app_title_(std::move(config.title)), ini_path_(std::move(config.ini_path)) {
  Console::Init();
  SwitchGraphicsMode(config.width, config.height, config.gfx_mode);

  if (config.initial_theme.has_value()) {
    ui_.theme = *config.initial_theme;
  }

  ImPlot::CreateContext();
  mj_defaultVFS(&vfs_);
  mjv_defaultPerturb(&perturb_);
  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&vis_options_);
  for (int i = 0; i < mjNRNDFLAG; i++) {
    render_flags_[i] = (mjRNDSTRING[i][1][0] == '1');
  }
  std::memset(&plugin_scene_, 0, sizeof(mjvScene));
  mjv_makeScene(nullptr, &plugin_scene_, 2000);

  profiler_.Clear();

  step_control_.SetPreStepCallback([this](const mjModel* m, mjData* d) { PreStep(m, d); });
  step_control_.SetPostStepCallback([this](const mjModel* m, mjData* d) { PostStep(m, d); });

  LoadSettings();

#ifndef __EMSCRIPTEN__
  physics_thread_ = std::thread(&App::PhysicsThreadLoop, this);
#endif
}

App::~App() {
#ifndef __EMSCRIPTEN__
  stop_physics_thread_.store(true);
  if (physics_thread_.joinable()) {
    physics_thread_.join();
  }
#endif
  // ImGui's autosave is on a timer and covers only state it tracks, so recent
  // changes and plugin visibility would be lost. window_ owns the ImGui context
  // and outlives this body.
  SaveSettings();
  if (ImPlot::GetCurrentContext() != nullptr) {
    ImPlot::DestroyContext(nullptr);
  }
  mjv_freeScene(&plugin_scene_);
  model_holder_.reset();
  mj_deleteVFS(&vfs_);
}

#ifndef __EMSCRIPTEN__
void App::PhysicsThreadLoop() {
  while (!stop_physics_thread_.load()) {
    {
      std::unique_lock<std::mutex> lock(physics_mutex_);
      if (tmp_.file_dialog == UiTempState::FileDialog_None) {
        UpdatePhysics();
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
#endif

void App::SwitchGraphicsMode(int width, int height, GraphicsMode mode) {
  model_scene_.reset();
  imgui_bridge_.reset();
  renderer_.reset();
  window_.reset();

  Window::Config window_config;
  window_config.gfx_mode = mode;
  window_ = std::make_unique<Window>(app_title_, width, height, window_config);

  mjrfContextConfig renderer_config;
  mjrf_defaultContextConfig(&renderer_config);
  renderer_config.native_window = window_->GetNativeWindowHandle();
  renderer_config.software_rendering = IsSoftware(mode);
  renderer_config.graphics_api =
      IsOpenGl(mode) || IsWebGl(mode) ? mjGRAPHICS_API_OPENGL : mjGRAPHICS_API_VULKAN;
  renderer_ = std::make_unique<Renderer>(renderer_config);

  // Define two render targets: one for rendering to the window, and one for
  // RenderToTexture.
  mjrfRenderTargetConfig config;
  mjrf_defaultRenderTargetConfig(&config);
  config.color_format = mjPIXEL_FORMAT_RGB8;
  config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  renderer_->CreateRenderTarget("pixels", config);
  renderer_->CreateRenderTarget("tmp", config);

  imgui_bridge_ = std::make_unique<ImguiBridge>(renderer_->GetContext(), renderer_->GetScene("ux"));

  // TODO: Figure out why this breaks on some platforms.
  // LoadSettings();
  if (ui_.window_width > 0 && ui_.window_height > 0) {
    window_->Resize(ui_.window_width, ui_.window_height);
  }
}

void App::RequestModelLoad(std::string model_file) { pending_load_ = std::move(model_file); }

void App::RequestModelReload() { pending_reload_ = true; }

void App::InitEmptyModel() {
  std::unique_lock<std::mutex> lock(physics_mutex_);
  BuildModel(EmptyModel{});
}

void App::LoadModelFromFile(const std::string& filepath) {
  std::unique_lock<std::mutex> lock(physics_mutex_);
  BuildModel(FileModel{filepath});
}

void App::LoadModelFromBuffer(std::span<const std::byte> buffer, std::string_view content_type,
                              std::string_view filename) {
  std::unique_lock<std::mutex> lock(physics_mutex_);
  BuildModel(BufferModel{.buffer = buffer, .content_type = content_type, .name = filename});
}

void App::LoadKeyframe(std::string_view keyframe) {
  std::unique_lock<std::mutex> lock(physics_mutex_);
  if (!has_model() || !has_data() || model()->nkey <= 0) {
    return;
  }
  const mjModel* m = model();
  int id = mj_name2id(m, mjOBJ_KEY, std::string(keyframe).c_str());
  if (id >= 0 && id < m->nkey) {
    tmp_.key_idx = id;
    ResetPhysics();
    return;
  }

  if (!keyframe.empty()) {
    int parsed = -1;
    auto [ptr, ec] = std::from_chars(keyframe.data(), keyframe.data() + keyframe.size(), parsed);
    if (ec == std::errc() && ptr == keyframe.data() + keyframe.size() && parsed >= 0 &&
        parsed < m->nkey) {
      tmp_.key_idx = parsed;
      ResetPhysics();
      return;
    }
  }
}

App::KeyframeSelection App::CaptureKeyframeSelection(bool is_reload) const {
  KeyframeSelection saved;
  saved.is_reload = is_reload;
  saved.key_idx = tmp_.key_idx;
  if (is_reload && has_model()) {
    const mjModel* m = model_holder_->model();
    if (tmp_.key_idx >= 0 && tmp_.key_idx < m->nkey) {
      if (const char* name = mj_id2name(m, mjOBJ_KEY, tmp_.key_idx)) {
        saved.key_name = name;
      }
    }
    saved.all_old_names.reserve(m->nkey);
    for (int k = 0; k < m->nkey; ++k) {
      if (const char* name = mj_id2name(m, mjOBJ_KEY, k)) {
        saved.all_old_names.push_back(name);
      } else {
        saved.all_old_names.push_back("");
      }
    }
  }
  return saved;
}

void App::RestoreKeyframeSelection(const KeyframeSelection& saved) {
  if (!saved.is_reload || !has_model() || saved.key_idx < 0) {
    tmp_.key_idx = -1;
    return;
  }
  const mjModel* m = model();
  if (!saved.key_name.empty()) {
    int id = mj_name2id(m, mjOBJ_KEY, saved.key_name.c_str());
    if (id >= 0) {
      tmp_.key_idx = id;
      return;
    }
    if (saved.key_idx >= 0 && saved.key_idx < m->nkey &&
        mj_id2name(m, mjOBJ_KEY, saved.key_idx) == nullptr) {
      tmp_.key_idx = saved.key_idx;
      return;
    }
    tmp_.key_idx = -1;
    return;
  } else {
    if (saved.key_idx >= 0 && saved.key_idx < m->nkey) {
      const char* new_name = mj_id2name(m, mjOBJ_KEY, saved.key_idx);
      if (new_name == nullptr) {
        tmp_.key_idx = saved.key_idx;
        return;
      }
      bool shifted_from_other = false;
      for (size_t i = 0; i < saved.all_old_names.size(); ++i) {
        if (static_cast<int>(i) != saved.key_idx && saved.all_old_names[i] == new_name) {
          shifted_from_other = true;
          break;
        }
      }
      if (!shifted_from_other) {
        tmp_.key_idx = saved.key_idx;
        return;
      }
    }
    tmp_.key_idx = -1;
  }
}

void App::UpdateFilePaths(const std::string& resolved_path) {
  if (!resolved_path.empty()) {
    const std::filesystem::path path(resolved_path);
    const std::string base_path = path.parent_path().string() + "/";
    const std::string model_name = path.stem().string();
    tmp_.last_path[UiTempState::FileDialog_Load] = base_path + model_name + ".xml";
    tmp_.last_path[UiTempState::FileDialog_SaveMjb] = base_path + model_name + "_saved.mjb";
    tmp_.last_path[UiTempState::FileDialog_SaveXml] = base_path + model_name + "_saved.xml";
    tmp_.last_path[UiTempState::FileDialog_PrintModel] = base_path + model_name + "_MJMODEL.TXT";
    tmp_.last_path[UiTempState::FileDialog_PrintData] = base_path + model_name + "_MJDATA.TXT";
    tmp_.last_path[UiTempState::FileDialog_SaveScreenshot] = base_path + "screenshot.webp";
  }
}

void App::SetLoadError(std::string error) {
  BuildModel(EmptyModel{});
  RecordLoadError(error);
}

void App::RecordLoadError(std::string_view error) {
  load_error_ = error;
  if (!load_error_.empty()) {
    Console::Get().Log(mjLOG_ERROR, load_error_);
  }
}

void App::ResetPhysics() {
  if (has_model() && has_data()) {
    if (tmp_.key_idx >= model()->nkey || tmp_.key_idx < -1) {
      tmp_.key_idx = -1;
    }
    mj_resetDataKeyframe(model(), data(), tmp_.key_idx);
    mj_forward(model(), data());
  }
  ResetHistory(sim_history_, timeline_, has_model() ? model() : nullptr,
               has_data() ? data() : nullptr);
  timer_prev_ = {};
  profiler_.ResetTimers();
}

void App::UpdatePhysics() {
  if (!has_model()) {
    return;
  }

  if (tmp_.update_threadpool) {
    mju_threadpool(data(), ui_.nthread);
    tmp_.update_threadpool = false;
  }

  bool plugin_stepped = false;
  ForEachPlugin<ModelPlugin>([&](auto* plugin) {
    if (plugin->do_update) {
      if (plugin->do_update(plugin, model(), data())) {
        plugin_stepped = true;
      }
    }
  });

  bool stepped = plugin_stepped;
  if (!plugin_stepped) {
    if (step_control_.GetPauseState() != PauseState::kNormalPaused) {
      mju_zero(data()->xfrc_applied, 6 * model()->nbody);
      mjv_applyPerturbPose(model(), data(), &perturb_, 0);
      mjv_applyPerturbForce(model(), data(), &perturb_);
    } else {
      mjv_applyPerturbPose(model(), data(), &perturb_, 1);
    }

    StepControl::Status status = step_control_.Advance(model(), data());
    if (status == StepControl::Status::kPaused) {
      profiler_.Update(model(), data());
    } else if (status == StepControl::Status::kOk) {
      stepped = true;
    } else if (status == StepControl::Status::kAutoReset) {
      ResetPhysics();
    } else if (status == StepControl::Status::kDiverged) {
      // The engine reported the divergence with mju_warning, which adds it to
      // the Console.
      stepped = true;
    }
  }

  if (stepped) {
    profiler_.Update(model(), data());
    if (plugin_stepped) {
      RecordHistoryFrame(sim_history_, timeline_, model(), data());
    }
  }

  plugin_scene_.ngeom = 0;
  ForEachPlugin<ScenePlugin>([&](auto* plugin) {
    if (plugin->enhance_scene) {
      plugin->enhance_scene(plugin, model(), data(), &plugin_scene_);
    }
  });
}

void App::PreStep(const mjModel* m, mjData* d) {
  ForEachPlugin<ModelPlugin>([&](auto* plugin) {
    if (plugin->pre_step) {
      plugin->pre_step(plugin, m, d);
    }
  });
}

void App::PostStep(const mjModel* m, mjData* d) {
  ForEachPlugin<ModelPlugin>([&](auto* plugin) {
    if (plugin->post_step) {
      plugin->post_step(plugin, m, d);
    }
  });

  RecordHistoryFrame(sim_history_, timeline_, m, d);
}

void App::LoadHistory(int offset) {
  LoadHistoryFrame(sim_history_, model(), data(), offset);
  timeline_.SetHistoryIndex(sim_history_.GetIndex());
  step_control_.SetPauseState(StepControl::PauseState::kNormalPaused);
}

void App::ApplyTimelineScrub() {
  // The GUI only edits timeline_.history_index; compare it against the
  // buffer's own cursor rather than a pre-GUI snapshot. A reset triggered from
  // inside the GUI (Reset button, keyframe) re-initialises both to index 0, so
  // this avoids a redundant LoadHistory(0), which would also force a pause.
  if (timeline_.history_index != sim_history_.GetIndex()) {
    LoadHistory(timeline_.history_index);
  }
}

bool App::Update() {
  const auto now = std::chrono::steady_clock::now();
  if (last_frame_time_.time_since_epoch().count() != 0) {
    const double dt = std::chrono::duration<double>(now - last_frame_time_).count();
    const double fps = dt > 0 ? 1.0 / dt : 0;
    fps_ = fps_ > 0 ? 0.9 * fps_ + 0.1 * fps : fps;
  }
  last_frame_time_ = now;

  const Window::Status status = window_->NewFrame();

  std::unique_lock<std::mutex> lock(physics_mutex_);

  // Execute any operations that could not be performed during the actual
  // BuildGui() flow.
  if (pending_op_) {
    pending_op_();
    pending_op_ = nullptr;
  }

  HandleWindowEvents();
  HandleMouseEvents();
  HandleKeyboardEvents();

  ProcessPendingLoads();

  PauseState current_pause = step_control_.GetPauseState();
  if (current_pause != last_pause_state_) {
    load_error_ = "";
    last_pause_state_ = current_pause;
  }

#ifdef __EMSCRIPTEN__
  // Only update the simulation if a popup window is not open. Note that the
  // simulation itself will only update if it is not paused.
  if (tmp_.file_dialog == UiTempState::FileDialog_None) {
    UpdatePhysics();
  }
#endif

  return status == Window::Status::kRunning && !tmp_.should_exit;
}

void App::Render() {
  const float width = window_->GetWidth();
  const float height = window_->GetHeight();
  const float scale = window_->GetScale();
  const mjrRect viewport = {0, 0, static_cast<int>(width * scale),
                            static_cast<int>(height * scale)};

  if (model_scene_) {
    std::unique_lock<std::mutex> lock(physics_mutex_);
    model_scene_->Sync(model(), data(), viewport, &vis_options_, &perturb_, &camera_,
                       {plugin_scene_.geoms, static_cast<size_t>(plugin_scene_.ngeom)});
  }
  imgui_bridge_->Update();

  Renderer::View views[2] = {GetSceneView(), GetUxView()};
  renderer_->ResizeRenderTarget(Renderer::kWindowRenderTarget, viewport.width, viewport.height);

  auto target = renderer_->GetRenderTargetHandle("pixels");
  if (IsHeadless(window_->GetGraphicsMode())) {
    renderer_->ResizeRenderTarget(target, viewport.width, viewport.height);
    views[0].render_target = target;
    views[1].render_target = target;
  }

  renderer_->Render(views);

  window_->EndFrame();
  window_->Present(renderer_->GetPixels(target));
}

std::span<const std::byte> App::RenderToTexture(mjvCamera* camera, int width, int height) {
  auto target = renderer_->GetRenderTargetHandle("tmp");
  renderer_->ResizeRenderTarget(target, width, height);

  Renderer::View views[1] = {GetSceneView()};
  views[0].camera = mjv_camera2GLCamera(model(), data(), camera);
  views[0].render_target = target;
  renderer_->Render(views);
  return renderer_->GetPixels(target);
}

Renderer::View App::GetSceneView() {
  Renderer::View view;
  view.scene = renderer_->GetSceneHandle("studio");
  view.camera = mjv_camera2GLCamera(model(), data(), &camera_);
  if (render_flags_[mjRND_SEGMENT]) {
    if (render_flags_[mjRND_IDCOLOR]) {
      view.draw_mode = mjDRAW_MODE_SEGMENTATION_BY_ID;
    } else {
      view.draw_mode = mjDRAW_MODE_SEGMENTATION_BY_COLOR;
    }
  } else if (render_flags_[mjRND_DEPTH]) {
    view.draw_mode = mjDRAW_MODE_DEPTH;
  } else if (render_flags_[mjRND_WIREFRAME]) {
    view.draw_mode = mjDRAW_MODE_WIREFRAME;
  } else if (vis_options_.flags[mjVIS_ISLAND]) {
    view.draw_mode = mjDRAW_MODE_ISLANDS;
  } else {
    view.draw_mode = mjDRAW_MODE_DEFAULT;
  }
  view.haze = render_flags_[mjRND_HAZE];
  view.skybox = render_flags_[mjRND_SKYBOX];
  view.shadows = render_flags_[mjRND_SHADOW];
  view.reflections = render_flags_[mjRND_REFLECTION];
  view.post_processing = render_flags_[mjRND_POST_PROCESSING];
  if (model()->vis.headlight.active) {
    view.headlight = Renderer::View::Headlight();
    view.headlight->color[0] = model()->vis.headlight.diffuse[0];
    view.headlight->color[1] = model()->vis.headlight.diffuse[1];
    view.headlight->color[2] = model()->vis.headlight.diffuse[2];
    view.headlight->intensity = model_scene_->GetHeadlightIntensity();
  }
  return view;
}

Renderer::View App::GetUxView() {
  const int width = window_->GetWidth() * window_->GetScale();
  const int height = window_->GetHeight() * window_->GetScale();

  Renderer::View view;
  view.scene = renderer_->GetSceneHandle("ux");
  view.shadows = false;
  view.reflections = false;
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

void App::BuildModel(const BuildModelInfo& info) {
  if (std::holds_alternative<RecompileFromSpec>(info)) {
    const KeyframeSelection saved_key = CaptureKeyframeSelection(/*is_reload=*/true);
    auto tmp_holder = spec_editor_.Compile(&vfs_);
    if (tmp_holder->ok()) {
      preserve_camera_on_load_ = true;
      model_holder_ = std::move(tmp_holder);
      tmp_.curr_element = nullptr;
      RestoreKeyframeSelection(saved_key);
      OnModelLoaded("", model_kind_);
    } else {
      RecordLoadError(tmp_holder->error());
    }
  } else if (std::holds_alternative<RecompileModel>(info)) {
    const KeyframeSelection saved_key = CaptureKeyframeSelection(/*is_reload=*/true);
    tmp_.curr_element = nullptr;
    model_holder_->Recompile();
    if (!model_holder_->ok()) {
      SetLoadError(std::string(model_holder_->error()));
      return;
    }

    RestoreKeyframeSelection(saved_key);
    InitModelScene();
    ResetHistory(sim_history_, timeline_, has_model() ? model() : nullptr,
                 has_data() ? data() : nullptr);
    timer_prev_ = {};
    profiler_.ResetTimers();
  } else if (std::holds_alternative<EmptyModel>(info)) {
    model_holder_.reset();
    tmp_.curr_element = nullptr;
    mj_deleteVFS(&vfs_);
    mj_defaultVFS(&vfs_);
    model_holder_ = ModelHolder::FromSpec(mj_makeSpec(), &vfs_);
    tmp_.key_idx = -1;
    last_buffer_.clear();
    last_content_type_.clear();
    OnModelLoaded("", kEmptyModel);
    spec_editor_.Reset(*spec());
  } else if (std::holds_alternative<FileModel>(info)) {
    auto& load = std::get<FileModel>(info);
    const std::string resolved_file = ResolveFile(load.filepath, search_paths_);
    const KeyframeSelection saved_key = CaptureKeyframeSelection(preserve_camera_on_load_);
    model_holder_.reset();
    tmp_.curr_element = nullptr;
    mj_deleteVFS(&vfs_);
    mj_defaultVFS(&vfs_);
    model_holder_ = ModelHolder::FromFile(resolved_file, &vfs_);
    if (model_holder_->ok()) {
      last_buffer_.clear();
      last_content_type_.clear();
      RestoreKeyframeSelection(saved_key);
      OnModelLoaded(load.filepath, kModelFromFile);
      if (spec()) {
        spec_editor_.Reset(*spec());
      } else {
        spec_editor_.Reset();
      }
      UpdateFilePaths(resolved_file);
      if (model() && model()->names) {
        // Assumes the first string in the model is the name of the model
        // itself.
        window_->SetTitle(app_title_ + " : " + std::string(model()->names));
      } else {
        window_->SetTitle(app_title_ + " : " +
                          std::filesystem::path(load.filepath).stem().string());
      }
    } else {
      SetLoadError(std::string(model_holder_->error()));
      // Keep track of the attempted load in case the user fixes the error and
      // tries to reload the same file again.
      model_path_ = resolved_file;
    }
  } else if (std::holds_alternative<BufferModel>(info)) {
    auto& load = std::get<BufferModel>(info);
    const KeyframeSelection saved_key = CaptureKeyframeSelection(preserve_camera_on_load_);
    model_holder_.reset();
    tmp_.curr_element = nullptr;
    mj_deleteVFS(&vfs_);
    mj_defaultVFS(&vfs_);
    model_holder_ = ModelHolder::FromBuffer(load.buffer, load.content_type, load.name, &vfs_);
    if (model_holder_->ok()) {
      last_buffer_.assign(load.buffer.begin(), load.buffer.end());
      last_content_type_ = std::string(load.content_type);
      RestoreKeyframeSelection(saved_key);
      OnModelLoaded(load.name, kModelFromFile);
    } else {
      SetLoadError(std::string(model_holder_->error()));
    }
    if (spec()) {
      spec_editor_.Reset(*spec());
    } else {
      spec_editor_.Reset();
    }
  }
}

void App::OnModelLoaded(std::string_view filename, ModelKind model_kind) {
  load_error_ = "";
  tmp_.curr_element = nullptr;

  timer_prev_ = {};
  profiler_.Clear();

  if (!model_holder_->warning().empty()) {
    load_error_ = model_holder_->warning();
  }
  model_path_ = std::string(filename);

  if (model_kind_ == kEmptyModel) {
    step_control_.SetPauseState(PauseState::kUnpaused);
  }
  model_kind_ = model_kind;
  if (model_kind_ == kEmptyModel || !load_error_.empty()) {
    step_control_.SetPauseState(PauseState::kNormalPaused);
  }

  // Reset/reinitialize everything that depends on the new mjModel.
  mjModel* model = model_holder_->model();
  InitModelScene();
  ResetPhysics();

  if (!preserve_camera_on_load_) {
    const int model_cam = model->vis.global.cameraid;
    if (model_cam >= 0 && model_cam < model->ncam) {
      tmp_.camera_idx = SetCamera(model, &camera_, model_cam);
    } else {
      mjv_defaultFreeCamera(model, &camera_);
    }
  }
  preserve_camera_on_load_ = false;

  // Initialize the speed based on the model's default real-time setting.
  float min_error = FLT_MAX;
  const float desired = mju_log(100 * model->vis.global.realtime);
  for (int i = 0; i < kPercentRealTime.size(); ++i) {
    const float speed = std::stof(kPercentRealTime[i]);
    const float error = mju_abs(mju_log(speed) - desired);
    if (error < min_error) {
      min_error = error;
      SetSpeedIndex(i);
    }
  }

  ForEachPlugin<ModelPlugin>([&](auto* plugin) {
    if (plugin->post_model_loaded) {
      plugin->post_model_loaded(plugin, model, model_path_.c_str());
    }
  });
  tmp_.update_threadpool = true;
  last_pause_state_ = step_control_.GetPauseState();
}

void App::InitModelScene() {
  // Ensure all entities in the scene are removed before populating it again.
  model_scene_.reset();
  if (has_model()) {
    mjrfContext* ctx = renderer_->GetContext();
    mjrfScene* scene = renderer_->GetScene("studio");
    model_scene_ = std::make_unique<ModelScene>(ctx, scene, model(), DrawTextAt);
  }
}

void App::ProcessPendingLoads() {
  if (pending_reload_) {
    pending_reload_ = false;
    if (model_kind_ == kModelFromFile || (model_kind_ == kEmptyModel && !model_path_.empty())) {
      pending_load_ = model_path_;
      preserve_camera_on_load_ = true;
    }
  }

  // Check to see if we need to load a new model.
  if (pending_load_.has_value()) {
    std::string load_data = std::move(pending_load_.value());
    pending_load_.reset();

    if (load_data.empty()) {
      BuildModel(EmptyModel{});
    } else if (!last_buffer_.empty() && load_data == model_path_) {
      BuildModel(BufferModel{last_buffer_, last_content_type_, load_data});
    } else {
      BuildModel(FileModel(load_data));
    }
  }

  if (recompile_spec_) {
    recompile_spec_ = false;
    BuildModel(RecompileFromSpec{});
  }

  // Allow plugins to edit the spec as well.
  ForEachPlugin<SpecEditorPlugin>([&](auto* plugin) {
    if (plugin->pre_compile) {
      if (plugin->pre_compile(plugin, spec(), model(), data(), &camera_)) {
        BuildModel(RecompileModel{});
        if (plugin->post_compile) {
          plugin->post_compile(plugin, spec(), model(), data());
        }
      };
    }
  });

  // Check plugins to see if we need to load a new model.
  ForEachPlugin<ModelPlugin>([&](auto* plugin) {
    if (plugin->get_model_to_load) {
      char model_name[1000] = "";
      char content_type[1000] = "";
      int size = 0;
      const char* buf = plugin->get_model_to_load(plugin, &size, content_type, sizeof(content_type),
                                                  model_name, sizeof(model_name));
      if (buf && buf == model_name) {
        BuildModel(FileModel(model_name));
      } else if (buf && size) {
        const std::byte* bytes = reinterpret_cast<const std::byte*>(buf);
        BuildModel(BufferModel{{bytes, bytes + size}, content_type, model_name});
      }
    }
  });
}

void App::HandleWindowEvents() {
  const std::string drop_file = window_->GetDropFile();
  if (!drop_file.empty()) {
    RequestModelLoad(drop_file);
  }
}

void App::HandleMouseEvents() {
  auto& io = ImGui::GetIO();
  if (io.WantCaptureMouse) {
    return;
  }

  if (!has_model() || !has_data()) {
    return;
  }

  // Normalize mouse positions and movement to display size.
  const float mouse_x = io.MousePos.x / io.DisplaySize.x;
  const float mouse_y = io.MousePos.y / io.DisplaySize.y;
  const float mouse_dx = io.MouseDelta.x / io.DisplaySize.x;
  const float mouse_dy = io.MouseDelta.y / io.DisplaySize.y;
  const bool is_mouse_moving = (mouse_dx != 0.0f || mouse_dy != 0.0f);
  const bool is_any_mouse_down = ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                                 ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                                 ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  const bool is_mouse_dragging = is_mouse_moving && is_any_mouse_down;

  // If no mouse buttons are down, end any active perturbations.
  if (!is_any_mouse_down) {
    perturb_.active = 0;
  }

  // Handle perturbation mouse actions.
  if (is_mouse_dragging && io.KeyCtrl) {
    if (perturb_.select > 0) {
      mjtMouse action = mjMOUSE_NONE;
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (io.KeyAlt) {
          action = io.KeyShift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        } else {
          action = io.KeyShift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        }
      } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        action = io.KeyShift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
      } else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        action = mjMOUSE_ZOOM;
      }
      const mjtPertBit active =
          (action == mjMOUSE_MOVE_V || action == mjMOUSE_MOVE_H) ? mjPERT_TRANSLATE : mjPERT_ROTATE;
      if (active != perturb_.active) {
        InitPerturb(model(), data(), &camera_, &perturb_, active);
      }
      MovePerturb(model(), data(), &camera_, &perturb_, action, mouse_dx, mouse_dy);
    }
  }

  // Camera drags and wheel (Ctrl drags are left to the perturbation above).
  MoveCameraWithMouse(model(), &camera_, tmp_.camera_idx == kFreeCameraIdx, ReadImGuiCameraInput());

  // Left double click.
  if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    PickResult picked =
        Pick(model(), data(), &camera_, mouse_x, mouse_y, window_->GetAspectRatio(), &vis_options_);
    if (picked.body >= 0) {
      perturb_.select = picked.body;
      perturb_.flexselect = picked.flex;
      perturb_.skinselect = picked.skin;

      // Select the corresponding element in the spec.
      tmp_.curr_element = nullptr;
      if (has_spec()) {
        mjsElement* element = mjs_firstElement(spec(), mjOBJ_BODY);
        while (element) {
          if (mjs_getId(element) == picked.body) {
            tmp_.curr_element = element;
            break;
          }
          element = mjs_nextElement(spec(), element);
        }
      }

      // Compute the local position of the selected object in the world.
      mjtNum tmp[3];
      mju_sub3(tmp, picked.point, data()->xpos + 3 * picked.body);
      mju_mulMatTVec(perturb_.localpos, data()->xmat + 9 * picked.body, tmp, 3, 3);
    } else {
      mjv_defaultPerturb(&perturb_);
    }
  }

  // Right double click.
  if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right)) {
    PickResult picked =
        Pick(model(), data(), &camera_, mouse_x, mouse_y, window_->GetAspectRatio(), &vis_options_);
    mju_copy3(camera_.lookat, picked.point);
    if (picked.body > 0 && io.KeyCtrl) {
      // Switch camera to tracking mode and track the selected body.
      camera_.type = mjCAMERA_TRACKING;
      camera_.trackbodyid = picked.body;
      camera_.fixedcamid = -1;
      tmp_.camera_idx = kTrackingCameraIdx;
    }
  }
}

void App::HandleKeyboardEvents() {
  if (ImGui::GetIO().WantCaptureKeyboard) {
    return;
  }

  constexpr auto ImGuiMod_CtrlShift = ImGuiMod_Ctrl | ImGuiMod_Shift;

  bool is_freecam_wasd = tmp_.camera_idx == kFreeCameraIdx;

  // Menu shortcuts.
  if (ImGui_IsChordJustPressed(ImGuiKey_O | ImGuiMod_Ctrl)) {
    tmp_.file_dialog = UiTempState::FileDialog_Load;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_S | ImGuiMod_CtrlShift)) {
    tmp_.file_dialog = UiTempState::FileDialog_SaveMjb;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_S | ImGuiMod_Ctrl)) {
    tmp_.file_dialog = UiTempState::FileDialog_SaveXml;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_M | ImGuiMod_Ctrl)) {
    tmp_.file_dialog = UiTempState::FileDialog_PrintModel;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_D | ImGuiMod_Ctrl)) {
    tmp_.file_dialog = UiTempState::FileDialog_PrintData;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_P | ImGuiMod_Ctrl)) {
    tmp_.file_dialog = UiTempState::FileDialog_SaveScreenshot;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_C | ImGuiMod_Ctrl)) {
    std::string keyframe = KeyframeToString(model(), data(), false);
    MaybeSaveToClipboard(keyframe);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_L | ImGuiMod_Ctrl)) {
    RequestModelReload();
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Q | ImGuiMod_Ctrl)) {
    tmp_.should_exit = true;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_A | ImGuiMod_Ctrl)) {
    const int cam_id = model()->vis.global.cameraid;
    if (cam_id >= 0 && cam_id < model()->ncam) {
      tmp_.camera_idx = SetCamera(model(), &camera_, cam_id);
    } else {
      mjv_defaultFreeCamera(model(), &camera_);
    }
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Tab | ImGuiMod_Shift)) {
    tmp_.inspector_panel = !tmp_.inspector_panel;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Tab)) {
    tmp_.options_panel = !tmp_.options_panel;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Minus | ImGuiMod_Ctrl)) {
    float old_scale = ui_.font_scale;
    ui_.font_scale = std::clamp(ui_.font_scale - 0.1f, 0.5f, 3.0f);
    RescaleDock(ui_.font_scale / old_scale);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Equal | ImGuiMod_Ctrl)) {
    float old_scale = ui_.font_scale;
    ui_.font_scale = std::clamp(ui_.font_scale + 0.1f, 0.5f, 3.0f);
    RescaleDock(ui_.font_scale / old_scale);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Minus)) {
    SetSpeedIndex(tmp_.speed_index + 1);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Equal)) {
    SetSpeedIndex(tmp_.speed_index - 1);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_LeftArrow)) {
    if (step_control_.GetPauseState() == PauseState::kNormalPaused) {
      LoadHistory(sim_history_.GetIndex() - 1);
    }
  } else if (ImGui_IsChordJustPressed(ImGuiKey_RightArrow)) {
    if (step_control_.GetPauseState() == PauseState::kNormalPaused) {
      if (sim_history_.GetIndex() == 0) {
        step_control_.RequestSingleStep();
      } else {
        LoadHistory(sim_history_.GetIndex() + 1);
      }
    }
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Space)) {
    if (step_control_.GetPauseState() == PauseState::kUnpaused) {
      step_control_.SetPauseState(PauseState::kNormalPaused);
    } else {
      step_control_.SetPauseState(PauseState::kUnpaused);
    }
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Backspace)) {
    ResetPhysics();
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Delete)) {
    // TODO: SpecDeleteElement(tmp_.curr_element);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_PageUp)) {
    SelectParentPerturb(model(), perturb_);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_F1)) {
    ToggleWindow(tmp_.help);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_F2)) {
    ToggleWindow(tmp_.info);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_F3)) {
    // Cycle through profiler views.
    if (!tmp_.profiler) {
      tmp_.profiler = true;
      tmp_.profiler_show_iter = false;
    } else if (!tmp_.profiler_show_iter) {
      tmp_.profiler_show_iter = true;
    } else {
      tmp_.profiler = false;
    }
  } else if (ImGui_IsChordJustPressed(ImGuiKey_F6)) {
    vis_options_.frame = (vis_options_.frame + 1) % mjNFRAME;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_F7)) {
    vis_options_.label = (vis_options_.label + 1) % mjNLABEL;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_F11)) {
    tmp_.full_screen = !tmp_.full_screen;
  } else if (ImGui_IsChordJustPressed(ImGuiKey_H)) {
    ToggleFlag(vis_options_.flags[mjVIS_CONVEXHULL]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_X)) {
    ToggleFlag(vis_options_.flags[mjVIS_TEXTURE]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_J)) {
    ToggleFlag(vis_options_.flags[mjVIS_JOINT]);
  } else if (!is_freecam_wasd && ImGui_IsChordJustPressed(ImGuiKey_Q)) {
    ToggleFlag(vis_options_.flags[mjVIS_CAMERA]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_U)) {
    ToggleFlag(vis_options_.flags[mjVIS_ACTUATOR]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Comma)) {
    ToggleFlag(vis_options_.flags[mjVIS_ACTIVATION]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Z)) {
    ToggleFlag(vis_options_.flags[mjVIS_LIGHT]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_V)) {
    ToggleFlag(vis_options_.flags[mjVIS_TENDON]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Y)) {
    ToggleFlag(vis_options_.flags[mjVIS_RANGEFINDER]);
  } else if (!is_freecam_wasd && ImGui_IsChordJustPressed(ImGuiKey_E)) {
    ToggleFlag(vis_options_.flags[mjVIS_CONSTRAINT]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_I)) {
    ToggleFlag(vis_options_.flags[mjVIS_INERTIA]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Apostrophe)) {
    ToggleFlag(vis_options_.flags[mjVIS_SCLINERTIA]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_B)) {
    ToggleFlag(vis_options_.flags[mjVIS_PERTFORCE]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_O)) {
    ToggleFlag(vis_options_.flags[mjVIS_PERTOBJ]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_C)) {
    ToggleFlag(vis_options_.flags[mjVIS_CONTACTPOINT]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_N)) {
    ToggleFlag(vis_options_.flags[mjVIS_ISLAND]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_F)) {
    ToggleFlag(vis_options_.flags[mjVIS_CONTACTFORCE]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_P)) {
    ToggleFlag(vis_options_.flags[mjVIS_CONTACTSPLIT]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_T)) {
    ToggleFlag(vis_options_.flags[mjVIS_TRANSPARENT]);
  } else if (!is_freecam_wasd && ImGui_IsChordJustPressed(ImGuiKey_A)) {
    ToggleFlag(vis_options_.flags[mjVIS_AUTOCONNECT]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_M)) {
    ToggleFlag(vis_options_.flags[mjVIS_COM]);
  } else if (!is_freecam_wasd && ImGui_IsChordJustPressed(ImGuiKey_D)) {
    ToggleFlag(vis_options_.flags[mjVIS_STATIC]);
  } else if (!is_freecam_wasd && ImGui_IsChordJustPressed(ImGuiKey_W)) {
    ToggleFlag(render_flags_[mjRND_WIREFRAME]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Semicolon)) {
    ToggleFlag(vis_options_.flags[mjVIS_SKIN]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_GraveAccent)) {
    ToggleFlag(vis_options_.flags[mjVIS_BODYBVH]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_Backslash)) {
    ToggleFlag(vis_options_.flags[mjVIS_MESHBVH]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_0 | ImGuiMod_Shift)) {
    ToggleFlag(vis_options_.sitegroup[0]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_1 | ImGuiMod_Shift)) {
    ToggleFlag(vis_options_.sitegroup[1]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_2 | ImGuiMod_Shift)) {
    ToggleFlag(vis_options_.sitegroup[2]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_3 | ImGuiMod_Shift)) {
    ToggleFlag(vis_options_.sitegroup[3]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_4 | ImGuiMod_Shift)) {
    ToggleFlag(vis_options_.sitegroup[4]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_5 | ImGuiMod_Shift)) {
    ToggleFlag(vis_options_.sitegroup[5]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_0)) {
    ToggleFlag(vis_options_.geomgroup[0]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_1)) {
    ToggleFlag(vis_options_.geomgroup[1]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_2)) {
    ToggleFlag(vis_options_.geomgroup[2]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_3)) {
    ToggleFlag(vis_options_.geomgroup[3]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_4)) {
    ToggleFlag(vis_options_.geomgroup[4]);
  } else if (ImGui_IsChordJustPressed(ImGuiKey_5)) {
    ToggleFlag(vis_options_.geomgroup[5]);
  } else if (has_model() && ImGui_IsChordJustPressed(ImGuiKey_Escape)) {
    tmp_.camera_idx = SetCamera(model(), &camera_, kTumbleCameraIdx);
  } else if (has_model() && ImGui_IsChordJustPressed(ImGuiKey_LeftBracket)) {
    tmp_.camera_idx = SetCamera(model(), &camera_, tmp_.camera_idx - 1);
  } else if (has_model() && ImGui_IsChordJustPressed(ImGuiKey_RightBracket)) {
    tmp_.camera_idx = SetCamera(model(), &camera_, tmp_.camera_idx + 1);
  } else if (!is_freecam_wasd ||
             !MoveCameraWithKeys(model(), &camera_, ReadImGuiCameraInput(), &tmp_.cam_speed)) {
    // Unless WASD/QE flew the fly camera, offer the keys to plugins.
    ForEachPlugin<KeyHandlerPlugin>([&](auto* plugin) {
      if (plugin->key_chord && plugin->on_key_pressed) {
        if (ImGui_IsChordJustPressed(plugin->key_chord)) {
          plugin->on_key_pressed(plugin);
        }
      }
    });
  }
}

void App::LoadSettings() {
  if (!ini_path_.empty()) {
    std::string settings = LoadText(ini_path_);
    if (!settings.empty()) {
      ui_.FromDict(ReadIniSection(settings, "[Studio][UX]"));
      ImGui::LoadIniSettingsFromMemory(settings.data(), settings.size());

      KeyValues plugin_names = ReadIniSection(settings, "[Studio][Plugins]");
      ForEachPlugin<GuiPlugin>([&](auto* plugin) {
        auto it = plugin_names.find(plugin->name);
        if (it != plugin_names.end()) {
          plugin->active = std::stoi(it->second) != 0;
        }
      });

      // Applied later: the owning windows do not exist until first submitted.
      window_state_storage_ = ReadIniSection(settings, "[Studio][WindowStateStorage]");
    }
  }
}

// Key for one entry in state storage; the window name locates its owner on
// load.
static std::string WindowStateStorageKey(const char* window_name, ImGuiID id) {
  char id_str[16];
  std::snprintf(id_str, sizeof(id_str), "%08X", id);
  return std::string(window_name) + "/" + id_str;
}

void App::ApplyWindowStateStorage() {
  for (auto it = window_state_storage_.begin(); it != window_state_storage_.end();) {
    const std::string::size_type sep = it->first.rfind('/');
    if (sep == std::string::npos) {
      it = window_state_storage_.erase(it);
      continue;
    }
    // Child windows are "<parent>/<child>_<id>": id follows the last '/'.
    ImGuiWindow* window = ImGui::FindWindowByName(it->first.substr(0, sep).c_str());
    if (window == nullptr) {
      ++it;  // Window not submitted yet; try again next frame.
      continue;
    }
    // Windows that opt out of saved settings may keep non-boolean state in their storage.
    if (window->Flags & ImGuiWindowFlags_NoSavedSettings) {
      it = window_state_storage_.erase(it);
      continue;
    }
    const ImGuiID id = std::strtoul(it->first.c_str() + sep + 1, nullptr, 16);
    window->StateStorage.SetInt(id, std::stoi(it->second) != 0);
    it = window_state_storage_.erase(it);
  }
}

void App::SaveSettings() {
  if (!ini_path_.empty()) {
    std::string settings = ImGui::SaveIniSettingsToMemory();
    if (window_) {
      ui_.window_width = window_->GetWidth();
      ui_.window_height = window_->GetHeight();
    }
    AppendIniSection(settings, "[Studio][UX]", ui_.ToDict());

    KeyValues plugin_names;
    ForEachPlugin<GuiPlugin>(
        [&](auto* plugin) { plugin_names[plugin->name] = std::to_string((int)plugin->active); });
    AppendIniSection(settings, "[Studio][Plugins]", plugin_names);

    // Pending entries first, so state for windows never opened this session is
    // not dropped, then let the live windows override. Merge into a local copy:
    // window_state_storage_ must stay pending-only, or a stale value could
    // re-apply over a newer toggle.
    KeyValues window_state_storage = window_state_storage_;
    ImGuiContext& g = *ImGui::GetCurrentContext();
    for (ImGuiWindow* window : g.Windows) {
      if (window->Flags & ImGuiWindowFlags_NoSavedSettings) {
        continue;
      }
      for (const ImGuiStoragePair& pair : window->StateStorage.Data) {
        window_state_storage[WindowStateStorageKey(window->Name, pair.key)] =
            std::to_string(pair.val_i);
      }
    }
    AppendIniSection(settings, "[Studio][WindowStateStorage]", window_state_storage);

    SaveText(settings, ini_path_);
  }
}

void App::SetSpeedIndex(int idx) {
  ::mujoco::studio::SetSpeedIndex(&step_control_, tmp_.speed_index, idx);
}

void App::BuildGui() {
  std::unique_lock<std::mutex> lock(physics_mutex_);
  if (tmp_.full_screen) {
    return;
  }

  if (!tmp_.style_editor) {
    SetupTheme(ui_.theme);
  }

  ImGui::GetIO().FontGlobalScale = ui_.font_scale;

  const ImVec4 workspace_rect = ConfigureDockingLayout(tmp_.toolbar, tmp_.status_bar);

  // Place charts in bottom right corner of the workspace.
  const ImVec2 chart_size(250, 500);
  const ImVec2 chart_pos(workspace_rect.x + workspace_rect.z - chart_size.x,
                         workspace_rect.y + workspace_rect.w - chart_size.y);

  MainMenuGui();

  if (tmp_.toolbar) {
    if (ImGui::Begin("ToolBar")) {
      ToolBarGui();
    }
    ImGui::End();
  }

  if (tmp_.status_bar) {
    ScopedStyle style;
    style.Var(ImGuiStyleVar_CellPadding, ImVec2(0, 0));
    style.Var(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
    style.Var(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    if (ImGui::Begin("StatusBar")) {
      StatusBarGui();
    }
    ImGui::End();
  }

  if (tmp_.options_panel) {
    if (ImGui::Begin("Options", &tmp_.options_panel)) {
      ModelOptionsGui();
    }
    ImGui::End();
  }

  if (tmp_.inspector_panel) {
    if (ImGui::Begin("Explorer", &tmp_.inspector_panel)) {
      SpecExplorerGui();
    }
    ImGui::End();

    if (ImGui::Begin("Inspector", &tmp_.inspector_panel)) {
      DataInspectorGui();
    }
    ImGui::End();
  }

  if (tmp_.editor_panel) {
    if (ImGui::Begin("Editor (\xEF\x83\x83)", &tmp_.editor_panel)) {
      SpecEditorGui();
    }
    ImGui::End();
  }

  if (tmp_.profiler) {
    if (ImGui::Begin("Profiler", &tmp_.profiler,
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
      ProfilerGui(model(), data(), &profiler_, tmp_.profiler_show_iter);
    }
    ImGui::End();
  }

  if (tmp_.console) {
    if (ImGui::Begin("Console", &tmp_.console)) {
      Console::Get().WindowGui();
    } else if (ImGui::IsWindowCollapsed()) {
      ImGui::SetWindowCollapsed(false);
      tmp_.console = false;
    }
    ImGui::End();
  }

  if (tmp_.picture_in_picture) {
    if (ImGui::Begin("Picture-in-Picture", &tmp_.picture_in_picture)) {
      auto render_cb = [this](const PipState& pip, int width, int height) {
        mjtByte* flags = render_flags_;
        const int prev_depth = flags[mjRND_DEPTH];
        const int prev_segment = flags[mjRND_SEGMENT];
        flags[mjRND_DEPTH] = (pip.mode == PipState::Depth) ? 1 : 0;
        flags[mjRND_SEGMENT] = (pip.mode == PipState::Segmentation) ? 1 : 0;

        mjvCamera camera;
        mjv_defaultCamera(&camera);
        camera.type = mjCAMERA_FIXED;
        camera.fixedcamid = pip.camera;

        auto output = RenderToTexture(&camera, width, height);
        int id = imgui_bridge_->UploadImage(pip.texture, output, width, height);

        flags[mjRND_DEPTH] = prev_depth;
        flags[mjRND_SEGMENT] = prev_segment;
        return id;
      };

      PipGui(model(), window_->GetAspectRatio(), render_cb, &tmp_.pips);
    }
    ImGui::End();
  }

  if (tmp_.help) {
    ScopedStyle style;
    style.Var(ImGuiStyleVar_Alpha, 0.8f);
    ImGui::SetNextWindowPos(ImVec2(workspace_rect.x, workspace_rect.y), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Appearing);
    if (ImGui::Begin("Help", &tmp_.help, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2.0f, 5.0f));
      HelpGui();
      ImGui::PopStyleVar();
    }
    ImGui::End();
  }

  if (!load_error_.empty()) {
    const float scale = ImGui::GetWindowDpiScale();
    const float max_line_width = ImGui::CalcTextSize(load_error_.c_str()).x;
    const float padding = 30.0f * scale;
    const float max_workspace_width = std::max(0.0f, workspace_rect.z - 20.0f);
    const float min_target_width = std::min(350.0f * scale, max_workspace_width);
    const float target_width =
        std::clamp(max_line_width + padding, min_target_width, max_workspace_width);

    if (BeginOverlay("WarningOverlay", OverlayPos::kBottom, workspace_rect, target_width, 0.8f)) {
      const bool is_dark = ImGui::GetStyle().Colors[ImGuiCol_WindowBg].x < 0.5f;
      if (!model_holder_->warning().empty()) {
        const ImVec4 warning_color =
            is_dark ? ImVec4(1.0f, 0.8f, 0.2f, 1.0f) : ImVec4(0.7f, 0.45f, 0.0f, 1.0f);
        ImGui::TextColored(warning_color, "Warning:");
      } else {
        const ImVec4 error_color =
            is_dark ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f) : ImVec4(0.8f, 0.1f, 0.1f, 1.0f);
        ImGui::TextColored(error_color, "Compiler Error:");
      }
      ImGui::Separator();
      if (max_line_width + padding > max_workspace_width) {
        ImGui::PushTextWrapPos(max_workspace_width - padding);
        ImGui::TextUnformatted(load_error_.c_str());
        ImGui::PopTextWrapPos();
      } else {
        ImGui::TextUnformatted(load_error_.c_str());
      }
    }
    EndOverlay();
  }

  if (tmp_.info) {
    const float scale = ImGui::GetWindowDpiScale();
    if (BeginOverlay("Info", OverlayPos::kBottomLeft, workspace_rect, 180.0f * scale)) {
      InfoGui(model(), data(), step_control_.GetPauseState() == PauseState::kNormalPaused, fps_,
              renderer_->GetGpuFrameMs(), timer_prev_.data());
    }
    EndOverlay();
  }
  if (has_data()) {
    std::copy_n(data()->timer, mjNTIMER, timer_prev_.begin());
  }

  // pause overlay
  if (tmp_.info && has_model() && step_control_.GetPauseState() == PauseState::kNormalPaused) {
    TextOverlay("Pause", OverlayPos::kTop, workspace_rect, "PAUSE", ImVec4(0, 0, 0, 0), 3.0f);
  }

  // realtime factor overlay
  if (has_model() && step_control_.GetPauseState() == PauseState::kUnpaused) {
    const float desired = step_control_.GetSpeed();
    const float measured = step_control_.GetSpeedMeasured();
    if (desired != 100.0f || mju_abs(measured - desired) > 0.1f * desired) {
      char rtlabel[30];
      bool misaligned = mju_abs(measured - desired) > 0.1f * desired;
      if (misaligned) {
        std::snprintf(rtlabel, sizeof(rtlabel), "%g%% (%-4.1f%%)", desired, measured);
      } else {
        std::snprintf(rtlabel, sizeof(rtlabel), "%g%%", desired);
      }
      TextOverlay("Realtime", OverlayPos::kTopLeft, workspace_rect, rtlabel, ImVec4(0, 0, 0, 0),
                  2.0f);
    }
  }

  // XYZ view gizmo.
  if (tmp_.view_gizmo && has_model() && has_data()) {
    ViewGizmoGui(model(), data(), camera_, tmp_.camera_idx, tmp_.view_gizmo_animation,
                 workspace_rect);
  }

  // Display a drag-and-drop message if no model is loaded.
  if (model_kind_ == kEmptyModel) {
#ifndef __EMSCRIPTEN__
    const char* text = "Load model file or drag-and-drop model file here.";

    const float width = window_->GetWidth() * ImGui::GetWindowDpiScale();
    const float height = window_->GetHeight() * ImGui::GetWindowDpiScale();
    const ImVec2 text_size = ImGui::CalcTextSize(text);

    ImGui::SetNextWindowPos(ImVec2((width - text_size.x) / 2, height / 2), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(text_size.x + 10, text_size.y + 10), ImGuiCond_Always);
    const ImGuiWindowFlags kOverlayFlags = ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoBackground |
                                           ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoDocking;
    if (ImGui::Begin("##Overlay", 0, kOverlayFlags)) {
      ScopedStyle style;
      style.Color(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
      ImGui::Text("%s", text);
    }
    ImGui::End();
#endif  // !__EMSCRIPTEN__
  }

  FileDialogGui();

  if (tmp_.imgui_demo) {
    ImGui::ShowDemoWindow();
  }
  if (tmp_.implot_demo) {
    ImPlot::ShowDemoWindow();
  }
  if (tmp_.style_editor) {
    if (ImGui::Begin("Style Editor", &tmp_.style_editor)) {
      ImGui::ShowStyleEditor();
    }
    ImGui::End();
  }

  mjrfScene* scene = renderer_->GetScene("studio");
  if (scene) {
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("Plugins")) {
        if (ImGui::MenuItem("Filament", "", tmp_.filament_editor)) {
          tmp_.filament_editor = !tmp_.filament_editor;
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
    if (tmp_.filament_editor) {
      if (ImGui::Begin("Filament", &tmp_.filament_editor)) {
        DrawSceneEditor(scene);
      }
      ImGui::End();
    }
  }
  ForEachPlugin<GuiPlugin>([](auto* plugin) {
    if (!plugin->update) {
      return;
    }
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("Plugins")) {
        if (ImGui::MenuItem(plugin->name, "", plugin->active)) {
          plugin->active = !plugin->active;
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
    if (plugin->active) {
      if (ImGui::Begin(plugin->name, &plugin->active)) {
        plugin->update(plugin);
      }
      ImGui::End();
    }
  });

  // This frame's windows are submitted, so pending state storage can be
  // restored. Must precede the save below, which would otherwise write the
  // defaults.
  ApplyWindowStateStorage();

  ImGuiIO& io = ImGui::GetIO();
  if (io.WantSaveIniSettings) {
    SaveSettings();
    io.WantSaveIniSettings = false;
  }
}

void App::ModelOptionsGui() {
  const float min_width = GetExpectedLabelWidth();
  const ImGuiChildFlags child_flags =
      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysAutoResize;
  const ImGuiTreeNodeFlags node_flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;

  const SimulationGuiContext sim_ctx = {
      .model = model(),
      .data = data(),
      .step_control = &step_control_,
      .timeline = &timeline_,
      .speed_index = &tmp_.speed_index,
      .key_idx = &tmp_.key_idx,
      .nthread = &ui_.nthread,
      .update_threadpool = &tmp_.update_threadpool,
      .reset = [this] { ResetPhysics(); },
      .reload = [this] { RequestModelReload(); },
      .align =
          [this] {
            const int cam_id = model()->vis.global.cameraid;
            if (cam_id >= 0 && cam_id < model()->ncam) {
              tmp_.camera_idx = SetCamera(model(), &camera_, cam_id);
            } else {
              mjv_defaultFreeCamera(model(), &camera_);
            }
          },
      .save_keyframe = [this] { mj_setKeyframe(model(), data(), tmp_.key_idx); },
  };
  SimulationGui(sim_ctx);
  ApplyTimelineScrub();

  ImGui::BeginChild("PhysicsGui", {0, 0}, child_flags);
  if (SectionHeader("Physics", node_flags, 0.65f)) {
    PhysicsGui(model(), spec(), min_width);
    ImGui::TreePop();
  }
  ImGui::EndChild();

  ImGui::BeginChild("RenderingGui", {0, 0}, child_flags);
  if (SectionHeader("Rendering", node_flags, 0.65f)) {
    ImGui::PushID("RenderingSection");

    // Compute combo width so labels align on the right.
    char frame_label_tmp[32];
    std::snprintf(frame_label_tmp, sizeof(frame_label_tmp), " %s  Camera", ICON_FA_CAMERA);
    const float combo_w =
        -ImGui::CalcTextSize(frame_label_tmp).x - ImGui::GetStyle().ItemInnerSpacing.x;

    // Camera selector.
    {
      std::string cam_name = GetCameraName(model(), camera_, tmp_.camera_idx);

      char camera_text[32];
      std::snprintf(camera_text, sizeof(camera_text), "%s  Camera", ICON_FA_CAMERA);

      ImGui::SetNextItemWidth(combo_w);
      if (ImGui::BeginCombo(camera_text, cam_name.c_str())) {
        auto select = [&](int idx) {
          std::string name = GetCameraName(model(), camera_, idx);
          if (ImGui::Selectable(name.c_str(), (tmp_.camera_idx == idx))) {
            tmp_.camera_idx = SetCamera(model(), &camera_, idx);
          }
        };
        select(kTumbleCameraIdx);
        select(kFreeCameraIdx);
        select(kTrackingCameraIdx);
        for (int c = 0; c < model()->ncam; c++) {
          select(c);
        }
        ImGui::EndCombo();
      }
    }

    // Label selector.
    {
      char label_text[32];
      std::snprintf(label_text, sizeof(label_text), "%s  Label", ICON_FA_COMMENT);
      ImGui::SetNextItemWidth(combo_w);
      if (ImGui::BeginCombo(label_text, mjLABELSTRING[vis_options_.label])) {
        for (int n = 0; n < mjNLABEL; n++) {
          if (ImGui::Selectable(mjLABELSTRING[n], (vis_options_.label == n))) {
            vis_options_.label = n;
          }
        }
        ImGui::EndCombo();
      }
    }

    // Frame selector.
    {
      char frame_text[32];
      std::snprintf(frame_text, sizeof(frame_text), "%s  Frame", ICON_FA_ARROWS);
      ImGui::SetNextItemWidth(combo_w);
      if (ImGui::BeginCombo(frame_text, mjFRAMESTRING[vis_options_.frame])) {
        for (int n = 0; n < mjNFRAME; n++) {
          if (ImGui::Selectable(mjFRAMESTRING[n], (vis_options_.frame == n))) {
            vis_options_.frame = n;
          }
        }
        ImGui::EndCombo();
      }
    }

    // Copy camera button.
    {
      char copy_cam_label[48];
      std::snprintf(copy_cam_label, sizeof(copy_cam_label), "%s  Copy Camera", ICON_FA_COPY);
      if (ImGui::Button(copy_cam_label)) {
        std::string camera_string = CameraToString(data(), &camera_);
        MaybeSaveToClipboard(camera_string);
      }
      ImGui::SetItemTooltip("%s", "Copy the current camera pose as an MJCF XML string");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::PopID();

    RenderingGui(model(), &vis_options_, render_flags_, min_width);
    ImGui::TreePop();
  }
  ImGui::EndChild();

  ImGui::BeginChild("GroupsGui", {0, 0}, child_flags);
  if (SectionHeader("Visibility Groups", node_flags, 0.65f)) {
    GroupsGui(model(), &vis_options_, min_width);
    ImGui::TreePop();
  }
  ImGui::EndChild();

  ImGui::BeginChild("VisualizationGui", {0, 0}, child_flags);
  if (SectionHeader("Visualization", node_flags, 0.65f)) {
    VisualizationGui(model(), &vis_options_, &camera_, min_width);
    ImGui::TreePop();
  }
  ImGui::EndChild();
}

void App::DataInspectorGui() {
  if (!has_data()) {
    ImGui::Text("No mjData loaded.");
    return;
  }

  const float min_width = GetExpectedLabelWidth();
  const ImGuiChildFlags child_flags =
      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysAutoResize;
  const ImGuiTreeNodeFlags node_flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;

  ImGui::BeginChild("JointsGui", {0, 0}, child_flags);
  if (SectionHeader("Joints", node_flags, 0.65f)) {
    JointsGui(model(), data(), &vis_options_);
    ImGui::TreePop();
  }
  ImGui::EndChild();

  ImGui::BeginChild("ControlsGui", {0, 0}, child_flags);
  if (SectionHeader("Controls", node_flags, 0.65f)) {
    NoiseGui(&step_control_);
    ImGui::Separator();

    ControlsGui(model(), data(), &vis_options_);
    ImGui::TreePop();
  }
  ImGui::EndChild();

  ImGui::BeginChild("SensorGui", {0, 0}, child_flags);
  if (SectionHeader("Sensor", node_flags, 0.65f)) {
    SensorGui(model(), data());
    ImGui::TreePop();
  }
  ImGui::EndChild();

  ImGui::BeginChild("WatchGui", {0, 0}, child_flags);
  if (SectionHeader("Watch", node_flags, 0.65f)) {
    WatchGui(model(), data(), tmp_.watch_field, sizeof(tmp_.watch_field), tmp_.watch_index);
    ImGui::TreePop();
  }
  ImGui::EndChild();

  ImGui::BeginChild("StateGui", {0, 0}, child_flags);
  if (SectionHeader("State", node_flags, 0.65f)) {
    StateGui(model(), data(), tmp_.state, tmp_.state_sig, min_width);
    ImGui::TreePop();
  }
  ImGui::EndChild();
}

void App::SpecExplorerGui() {
  if (!has_spec()) {
    ImGui::Text("No mjSpec loaded.");
    return;
  }

  // Initialize the split height.
  const ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.y > 60.0f) {
    if (tmp_.explorer_split < 0) {
      tmp_.explorer_split = region.y * 0.7f;
    }
    tmp_.explorer_split = std::clamp(tmp_.explorer_split, 20.0f, region.y - 40.0f);
  }

  mjsElement* element = tmp_.curr_element;
  bool open = element != nullptr;
  if (ImGui_BeginHSplit("SpecExplorerTree", &tmp_.explorer_split, &open)) {
    SpecTreeGui(&element, spec());

    if (element != tmp_.curr_element) {
      tmp_.curr_element = element;

      // A different element was selected, so select it for perturb.
      if (element) {
        open = true;
        const int element_id = mjs_getId(element);
        if (element->elemtype == mjOBJ_BODY && perturb_.select != element_id) {
          mjv_defaultPerturb(&perturb_);
          perturb_.select = element_id;
        }
      }
    }

    if (ImGui_HSplit("SpecExplorerProperties", &tmp_.explorer_split, &open)) {
      ImGui::Text("%s", mju_type2Str(tmp_.curr_element->elemtype));
      ImGui::SameLine();
      ImGui::Text("(%d)", mjs_getId(tmp_.curr_element));
      ImGui::SameLine(ImGui::GetContentRegionAvail().x - 100.0f);

      auto mode_button = [&](const char* label, const char* tooltip, SpecPropertiesMode mode) {
        ScopedStyle style;
        style.Var(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        if (tmp_.spec_prop_mode == mode) {
          style.Color(ImGuiCol_Button, ImGuiCol_ButtonActive);
        }
        if (ImGui::Button(label, ImVec2(24.0f, 20.0f))) {
          tmp_.spec_prop_mode = mode;
        }
        ImGui::SetItemTooltip("%s", tooltip);
      };

      mode_button("S", "Spec", SpecPropertiesMode::kSpec);
      ImGui::SameLine();
      mode_button("M", "Model", SpecPropertiesMode::kModel);
      ImGui::SameLine();
      mode_button("D", "Data", SpecPropertiesMode::kData);

      ImGui::Separator();

      if (tmp_.spec_prop_mode == SpecPropertiesMode::kSpec) {
        ElementSpecGui(element);
      } else if (tmp_.spec_prop_mode == SpecPropertiesMode::kModel) {
        ElementModelGui(model(), tmp_.curr_element);
      } else {
        ElementDataGui(data(), tmp_.curr_element);
      }
    }
    if (!open) {
      tmp_.curr_element = nullptr;
    }
  }
  ImGui_EndHSplit(open);
}

void App::SpecEditorGui() {
  if (ImGui::BeginChild("SpecEditor", ImVec2(-1, 36))) {
    if (ImGui::BeginTable("##SpecEditorHeader", 3)) {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 50.0f);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 30.0f);

      ImGui::TableNextColumn();
      ImGui::BeginDisabled(!spec_editor_.CanUndo());
      if (ImGui::Button(ICON_UNDO_SPEC)) {
        spec_editor_.Undo();
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(!spec_editor_.CanRedo());
      if (ImGui::Button(ICON_REDO_SPEC)) {
        spec_editor_.Redo();
      }
      ImGui::EndDisabled();

      ImGui::TableNextColumn();
      bool is_dark = ImGui::GetStyle().Colors[ImGuiCol_WindowBg].x < 0.5f;
      ImColor compile_green = is_dark ? ImColor(40, 125, 60, 255) : ImColor(40, 180, 40, 255);
      ImGui::PushStyleColor(ImGuiCol_Button, compile_green.Value);
      if (ImGui::Button("Compile and Reload", ImVec2(-1, 0))) {
        recompile_spec_ = true;
      }
      ImGui::PopStyleColor();

      ImGui::TableNextColumn();
      if (ImGui::Button(" + ")) {
        ImGui::OpenPopupOnItemClick("SpecAddElement", 0);
      }
      if (ImGui::BeginPopupContextItem("SpecAddElement")) {
        auto option = [&](const char* label, mjtObj type) {
          if (ImGui::Selectable(label)) {
            mjsElement* element = spec_editor_.AddElement(type);
            spec_editor_.SetActiveElement(element);
          }
        };

        option("Actuator", mjOBJ_ACTUATOR);
        option("Equality", mjOBJ_EQUALITY);
        option("Exclude", mjOBJ_EXCLUDE);
        option("Flex", mjOBJ_FLEX);
        option("Height Field", mjOBJ_HFIELD);
        option("Key", mjOBJ_KEY);
        option("Material", mjOBJ_MATERIAL);
        option("Mesh", mjOBJ_MESH);
        option("Numeric", mjOBJ_NUMERIC);
        option("Pair", mjOBJ_PAIR);
        option("Sensor", mjOBJ_SENSOR);
        option("Skin", mjOBJ_SKIN);
        option("Tendon", mjOBJ_TENDON);
        option("Text", mjOBJ_TEXT);
        option("Texture", mjOBJ_TEXTURE);
        option("Tuple", mjOBJ_TUPLE);
        ImGui::EndPopup();
      }

      ImGui::EndTable();
    }
  }
  ImGui::EndChild();

  // Initialize the split height.
  const ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.y > 60.0f) {
    if (tmp_.editor_split < 0) {
      tmp_.editor_split = region.y * 0.7f;
    }
    tmp_.editor_split = std::clamp(tmp_.editor_split, 20.0f, region.y - 40.0f);
  }

  mjsElement* element = spec_editor_.GetActiveElement();

  bool open = element != nullptr;
  if (ImGui_BeginHSplit("SpecEditorTree", &tmp_.editor_split, &open)) {
    SpecTreeGui(&element, spec_editor_.GetActiveSpec(), &spec_editor_);
    open = element != nullptr;
    spec_editor_.SetActiveElement(element);

    if (ImGui_HSplit("SpecEditorProperties", &tmp_.editor_split, &open)) {
      ImGui::Text("%s", mju_type2Str(element->elemtype));
      ImGui::SameLine();
      ImGui::Text("(%d)", mjs_getId(element));
      ImGui::Separator();

      ElementSpecGui(element, &spec_editor_);
    }
    ImGui_EndHSplit(open);
    if (!open) {
      spec_editor_.SetActiveElement(nullptr);
    }
  }
}

void App::HelpGui() {
  // With ConfigMacOSXBehaviors, imgui swaps the Cmd and Ctrl keys, so
  // ImGuiMod_Ctrl chords and io.KeyCtrl match the Command key on macOS.
  const char* mod = ImGui::GetIO().ConfigMacOSXBehaviors ? "Cmd" : "Ctrl";

  const float pad = ImGui::GetStyle().ItemSpacing.x;
  const float indent = pad * 2;
  const float col0 = ImGui::CalcTextSize("Toggle Fullscreen").x + pad;
  const float col1 = ImGui::CalcTextSize("Ctrl+R-Dblclick").x + pad + indent;
  const float col2 = ImGui::CalcTextSize("Center of Mass").x + pad + indent;
  const float col3 = ImGui::CalcTextSize("M").x + pad + indent;
  ImGui::Dummy(ImVec2(col0 + col1 + col2 + col3, 0));
  ImGui::Columns(4);
  ImGui::SetColumnWidth(0, col0);
  ImGui::SetColumnWidth(1, col1);
  ImGui::SetColumnWidth(2, col2);
  ImGui::SetColumnWidth(3, col3);

  ImGui::SeparatorText("Windows & UI");
  ImGui::Text("Help");
  ImGui::Text("Info");
  ImGui::Text("Profiler");
  ImGui::Text("Toggle Fullscreen");
  ImGui::Text("Toggle Left UI");
  ImGui::Text("Toggle Right UI");
  ImGui::Text("Font Bigger");
  ImGui::Text("Font Smaller");
  ImGui::SeparatorText("Simulation");
  ImGui::Text("Toggle Pause");
  ImGui::Text("Reset Sim");
  ImGui::Text("Step Back");
  ImGui::Text("Step Forward");
  ImGui::Text("Speed Up");
  ImGui::Text("Speed Down");
  ImGui::Text("Copy Keyframe");
  ImGui::SeparatorText("Camera");
  ImGui::Text("Tumble Camera");
  ImGui::Text("Prev Camera");
  ImGui::Text("Next Camera");
  ImGui::Text("Align Camera");
  ImGui::Text("Orbit Camera");
  ImGui::Text("Pan Camera");
  ImGui::Text("Zoom Camera");
  ImGui::Text("Center Camera");
  ImGui::Text("Track Camera");
  ImGui::SeparatorText("Selection");
  ImGui::Text("Select");
  ImGui::Text("Select Parent");
  ImGui::Text("Rotate Object");
  ImGui::Text("Move Object");
  ImGui::SeparatorText("Visualization");
  ImGui::Text("Cycle Frames");
  ImGui::Text("Cycle Labels");
  ImGui::Text("Geom Group");
  ImGui::Text("Site Group");

  ImGui::NextColumn();
  ImGui::Indent(indent);
  ImGui::SeparatorText("");
  ImGui::Text("F1");
  ImGui::Text("F2");
  ImGui::Text("F3");
  ImGui::Text("F11");
  ImGui::Text("Tab");
  ImGui::Text("Sh+Tab");
  ImGui::Text("%s+=", mod);
  ImGui::Text("%s+-", mod);
  ImGui::SeparatorText("");
  ImGui::Text("Spc");
  ImGui::Text("Bksp");
  ImGui::Text("Left");
  ImGui::Text("Right");
  ImGui::Text("+");
  ImGui::Text("-");
  ImGui::Text("%s+C", mod);
  ImGui::SeparatorText("");
  ImGui::Text("Esc");
  ImGui::Text("[");
  ImGui::Text("]");
  ImGui::Text("%s+A", mod);
  ImGui::Text("L-Drag");
  ImGui::Text("[Sh+]R-Drag");
  ImGui::Text("Scroll");
  ImGui::Text("R-Dblclick");
  ImGui::Text("%s+R-Dblclick", mod);
  ImGui::SeparatorText("");
  ImGui::Text("Dblclick");
  ImGui::Text("PgUp");
  ImGui::Text("%s+L-Drag", mod);
  ImGui::Text("%s+R-Drag", mod);
  ImGui::SeparatorText("");
  ImGui::Text("F6");
  ImGui::Text("F7");
  ImGui::Text("0-5");
  ImGui::Text("Sh+0-5");

  ImGui::NextColumn();
  ImGui::Indent(indent);
  ImGui::SeparatorText("Flags");
  ImGui::Text("Activation");
  ImGui::Text("Actuator");
  ImGui::Text("Auto Connect");
  ImGui::Text("Body Tree");
  ImGui::Text("Camera");
  ImGui::Text("Center of Mass");
  ImGui::Text("Constraint");
  ImGui::Text("Contact Force");
  ImGui::Text("Contact Point");
  ImGui::Text("Contact Split");
  ImGui::Text("Convex Hull");
  ImGui::Text("Inertia");
  ImGui::Text("Island");
  ImGui::Text("Joint");
  ImGui::Text("Light");
  ImGui::Text("Mesh Tree");
  ImGui::Text("Perturb Force");
  ImGui::Text("Perturb Object");
  ImGui::Text("Range Finder");
  ImGui::Text("Scale Inertia");
  ImGui::Text("Skin");
  ImGui::Text("Static Body");
  ImGui::Text("Tendon");
  ImGui::Text("Texture");
  ImGui::Text("Transparent");
  ImGui::Text("Wireframe");

  ImGui::NextColumn();
  ImGui::Indent(indent);
  ImGui::SeparatorText("");
  ImGui::Text(",");
  ImGui::Text("U");
  ImGui::Text("A");
  ImGui::Text("`");
  ImGui::Text("Q");
  ImGui::Text("M");
  ImGui::Text("E");
  ImGui::Text("F");
  ImGui::Text("C");
  ImGui::Text("P");
  ImGui::Text("H");
  ImGui::Text("I");
  ImGui::Text("N");
  ImGui::Text("J");
  ImGui::Text("Z");
  ImGui::Text("\\");
  ImGui::Text("B");
  ImGui::Text("O");
  ImGui::Text("Y");
  ImGui::Text("'");
  ImGui::Text(";");
  ImGui::Text("D");
  ImGui::Text("V");
  ImGui::Text("X");
  ImGui::Text("T");
  ImGui::Text("W");

  ImGui::Columns();
}

void App::ToolBarGui() {
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0, 0));
  if (ImGui::BeginTable("##ToolBarTable", 2)) {
    ScopedStyle style;
    style.Var(ImGuiStyleVar_ItemSpacing,
              ImVec2(ImGui::GetStyle().ItemSpacing.x * 2.0f, ImGui::GetStyle().ItemSpacing.y));

    const float label_width = GetExpectedLabelWidth();
    const float copy_btn_width = ImGui::GetFrameHeight();
    const float sp = ImGui::GetStyle().ItemSpacing.x;
    const float right_width = copy_btn_width + label_width + sp + label_width + sp + label_width;
    const float separator_width = ImGui::GetFrameHeight() * .6f;

    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, right_width);

    ImGui::TableNextColumn();
    ImGui::Dummy(ImVec2(ImGui::GetStyle().WindowPadding.x, 0.0f));
    ImGui::SameLine(0.0f, 0.0f);

    const float btn_size = ImGui::GetFrameHeight();
    const ImVec2 square_size(btn_size, btn_size);

    // Reload button.
    {
      style.Var(ImGuiStyleVar_FrameRounding, 2.0f);
      if (ImGui::Button(ICON_RELOAD_MODEL, square_size)) {
        RequestModelReload();
      }
      ImGui::SetItemTooltip("%s", "Reload");
    }

    // Reset button.
    ImGui::SameLine(0, 0.5 * separator_width);
    if (ImGui::Button(ICON_RESET_MODEL, square_size)) {
      ResetPhysics();
    }
    ImGui::SetItemTooltip("%s", "Reset");

    // Combined (Pause, Play) widget and Speed selection.
    ImGui::SameLine(0, separator_width);
    StepControlGui(&step_control_, tmp_.speed_index);

    ImGui::SameLine(0, separator_width);
    TimelineScrubberGui(model(), data(), step_control_, timeline_);
    ApplyTimelineScrub();

    ImGui::TableNextColumn();

    CameraSelectionGui(model(), data(), camera_, tmp_.camera_idx);

    ImGui::SameLine();
    LabelSelectionGui(&vis_options_);

    ImGui::SameLine();
    FrameSelectionGui(&vis_options_);

    ImGui::EndTable();
  }
  ImGui::PopStyleVar();
}

void App::StatusBarGui() {
  if (ImGui::BeginTable("##StatusBarTable", 2)) {
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableNextColumn();

    if (!has_model()) {
      ImGui::Text("No model loaded");
    } else if (step_control_.GetPauseState() == PauseState::kNormalPaused) {
      ImGui::Text("Paused");
    } else {
      ImGui::Text("Running");
    }

    ImGui::TableNextColumn();
    Console::Get().StatusGui(&tmp_.console);

    ImGui::EndTable();
  }
}

void App::MainMenuGui() {
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12.0f * ImGui::GetStyle().FontScaleDpi,
                                                        ImGui::GetStyle().ItemSpacing.y));
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
#ifndef __EMSCRIPTEN__
      if (ImGui::MenuItem("Open Model File", "Ctrl+O")) {
        tmp_.file_dialog = UiTempState::FileDialog_Load;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Save XML", "Ctrl+S")) {
        tmp_.file_dialog = UiTempState::FileDialog_SaveXml;
      }
      if (ImGui::MenuItem("Save MJB", "Ctrl+Shift+S")) {
        tmp_.file_dialog = UiTempState::FileDialog_SaveMjb;
      }
      if (ImGui::MenuItem("Save Screenshot", "Ctrl+P")) {
        tmp_.file_dialog = UiTempState::FileDialog_SaveScreenshot;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Print Model", "Ctrl+M")) {
        tmp_.file_dialog = UiTempState::FileDialog_PrintModel;
      }
      if (ImGui::MenuItem("Print Data", "Ctrl+D")) {
        tmp_.file_dialog = UiTempState::FileDialog_PrintData;
      }
      ImGui::Separator();
#endif  // !__EMSCRIPTEN__
      if (ImGui::MenuItem("Unload", "Ctrl+U")) {
        BuildModel(EmptyModel{});
      }
#ifndef __EMSCRIPTEN__
      ImGui::Separator();
      if (ImGui::MenuItem("Quit", "Ctrl+Q")) {
        tmp_.should_exit = true;
      }
#endif  // !__EMSCRIPTEN__
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Simulation")) {
      if (ImGui::MenuItem("Pause", "Space",
                          step_control_.GetPauseState() == PauseState::kNormalPaused)) {
        if (step_control_.GetPauseState() != PauseState::kNormalPaused) {
          step_control_.SetPauseState(PauseState::kNormalPaused);
        } else {
          step_control_.SetPauseState(PauseState::kUnpaused);
        }
      }
      if (ImGui::MenuItem("Reset", "Backspace")) {
        ResetPhysics();
      }
      if (ImGui::MenuItem("Reload", "Ctrl+L")) {
        RequestModelReload();
      }
      if (model()->nkey > 0) {
        ImGui::Separator();
        if (ImGui::BeginMenu("Keyframes")) {
          if (tmp_.key_idx >= model()->nkey || tmp_.key_idx < -1) {
            tmp_.key_idx = -1;
          }
          std::string key_name = GetKeyframeName(model(), tmp_.key_idx);
          ImGui::SetNextItemWidth(200);
          if (ImGui::BeginCombo("##Key", key_name.c_str())) {
            if (ImGui::Selectable("\xE2\x80\x94", (tmp_.key_idx == -1))) {
              tmp_.key_idx = -1;
              ResetPhysics();
            }
            for (int k = 0; k < model()->nkey; k++) {
              std::string item_name = GetKeyframeName(model(), k);
              ImGui::PushID(k);
              if (ImGui::Selectable(item_name.c_str(), (tmp_.key_idx == k))) {
                tmp_.key_idx = k;
                ResetPhysics();
              }
              ImGui::PopID();
            }
            ImGui::EndCombo();
          }
          ImGui::BeginDisabled(tmp_.key_idx < 0 || tmp_.key_idx >= model()->nkey);
          if (ImGui::MenuItem("Save")) {
            mj_setKeyframe(model(), data(), tmp_.key_idx);
          }
          ImGui::EndDisabled();
          if (ImGui::MenuItem("Copy")) {
            std::string str = KeyframeToString(model(), data(), false);
            MaybeSaveToClipboard(str);
          }
          ImGui::EndMenu();
        }
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Save Config")) {
        SaveSettings();
      }
      if (ImGui::MenuItem("Reset Config")) {
        ResetConfig(ini_path_);
        LoadSettings();
      }
      ImGui::Separator();

      if (ImGui::MenuItem("Options", "Tab", tmp_.options_panel)) {
        tmp_.options_panel = !tmp_.options_panel;
      }
      if (ImGui::MenuItem("Inspector", "Shift+Tab", tmp_.inspector_panel)) {
        tmp_.inspector_panel = !tmp_.inspector_panel;
      }
      if (ImGui::MenuItem("Editor (\xEF\x83\x83)", nullptr, tmp_.editor_panel)) {
        tmp_.editor_panel = !tmp_.editor_panel;
        if (tmp_.editor_panel) {
          tmp_.inspector_panel = true;
        }
      }
      if (ImGui::MenuItem("Toolbar", nullptr, tmp_.toolbar)) {
        tmp_.toolbar = !tmp_.toolbar;
      }
      if (ImGui::MenuItem("View Gizmo", nullptr, tmp_.view_gizmo)) {
        tmp_.view_gizmo = !tmp_.view_gizmo;
      }
      if (ImGui::MenuItem("Status Bar", nullptr, tmp_.status_bar)) {
        tmp_.status_bar = !tmp_.status_bar;
      }
      if (ImGui::MenuItem("Full Screen", "F11", tmp_.full_screen)) {
        tmp_.full_screen = !tmp_.full_screen;
      }
      ImGui::Separator();

      if (ImGui::MenuItem("Info", "F2", tmp_.info)) {
        ToggleWindow(tmp_.info);
      }
      if (ImGui::MenuItem("Profiler", "F3", tmp_.profiler)) {
        if (!tmp_.profiler) {
          tmp_.profiler = true;
          tmp_.profiler_show_iter = false;
        } else if (!tmp_.profiler_show_iter) {
          tmp_.profiler_show_iter = true;
        } else {
          tmp_.profiler = false;
        }
      }
      if (ImGui::MenuItem("Console", nullptr, tmp_.console)) {
        ToggleWindow(tmp_.console);
      }
      ImGui::Separator();

      if (ImGui::MenuItem("Picture-in-Picture", nullptr, tmp_.picture_in_picture)) {
        tmp_.picture_in_picture = !tmp_.picture_in_picture;
      }
      ImGui::Separator();

      ThemeMenuGui(&ui_.theme);

#ifdef __linux__
      if (ImGui::BeginMenu("Graphics Mode (Experimental)")) {
        std::optional<GraphicsMode> mode;
        if (ImGui::MenuItem("Filament OpenGL", nullptr,
                            window_->GetGraphicsMode() == GraphicsMode::FilamentOpenGl)) {
          mode = GraphicsMode::FilamentOpenGl;
        }
        if (ImGui::MenuItem("Filament OpenGL Headless", nullptr,
                            window_->GetGraphicsMode() == GraphicsMode::FilamentOpenGlHeadless)) {
          mode = GraphicsMode::FilamentOpenGlHeadless;
        }
        if (ImGui::MenuItem("Filament OpenGL Software", nullptr,
                            window_->GetGraphicsMode() == GraphicsMode::FilamentOpenGlSoftware)) {
          mode = GraphicsMode::FilamentOpenGlSoftware;
        }
        if (ImGui::MenuItem("Filament Vulkan", nullptr,
                            window_->GetGraphicsMode() == GraphicsMode::FilamentVulkan)) {
          mode = GraphicsMode::FilamentVulkan;
        }
        if (ImGui::MenuItem("Filament Vulkan Software", nullptr,
                            window_->GetGraphicsMode() == GraphicsMode::FilamentVulkanSoftware)) {
          mode = GraphicsMode::FilamentVulkanSoftware;
        }
        if (mode.has_value()) {
          pending_op_ = [=, this]() {
            const int width = window_->GetWidth();
            const int height = window_->GetHeight();
            SwitchGraphicsMode(width, height, *mode);
            InitModelScene();
          };
        }
        ImGui::EndMenu();
      }
#endif  // __linux__

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Plugins")) {
      // Placeholder menu item that will be populated by plugins later on. We
      // do this now in so that the menu is present at the right place.
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
      if (ImGui::MenuItem("Help", "F1", tmp_.help)) {
        ToggleWindow(tmp_.help);
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Style Editor", "", tmp_.style_editor)) {
        tmp_.style_editor = !tmp_.style_editor;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("ImGui Demo")) {
        tmp_.imgui_demo = !tmp_.imgui_demo;
      }
      if (ImGui::MenuItem("ImPlot Demo")) {
        tmp_.implot_demo = !tmp_.implot_demo;
      }
      ImGui::Separator();
      std::string version = "Version " + std::string(mj_versionString());
      ImGui::MenuItem(version.c_str());
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
  ImGui::PopStyleVar();
}

void App::FileDialogGui() {
  if (tmp_.file_dialog == UiTempState::FileDialog_None) {
    return;
  }

  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("FileDialog", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
    const DialogResult res = tmp_.file_dialog == UiTempState::FileDialog_Load
                                 ? OpenFileDialog(tmp_.filename)
                                 : SaveFileDialog(tmp_.filename);

    if (res.status == DialogResult::kAccepted) {
      tmp_.last_path[tmp_.file_dialog] = res.path;
      switch (tmp_.file_dialog) {
        case UiTempState::FileDialog_Load:
          RequestModelLoad(res.path);
          break;
        case UiTempState::FileDialog_SaveXml:
          mj_saveLastXML(res.path.c_str(), model(), nullptr, 0);
          break;
        case UiTempState::FileDialog_SaveMjb:
          mj_saveModel(model(), res.path.c_str(), nullptr, 0);
          break;
        case UiTempState::FileDialog_SaveScreenshot: {
          const int width = window_->GetWidth();
          const int height = window_->GetHeight();
          auto buffer = RenderToTexture(&camera_, width, height);
          SaveToWebp(width, height, buffer.data(), res.path);
          break;
        }
        case UiTempState::FileDialog_PrintModel:
          mj_printModel(model(), res.path.c_str());
          break;
        case UiTempState::FileDialog_PrintData:
          mj_printData(model(), data(), res.path.c_str());
          break;
        default:
          break;
      }
    }

    if (res.status != DialogResult::kError) {
      strncpy(tmp_.filename, res.path.c_str(), res.path.size());
      tmp_.filename[res.path.size()] = 0;
    }
    if (res.status != DialogResult::kPending) {
      tmp_.file_dialog = UiTempState::FileDialog_None;
      ImGui::CloseCurrentPopup();
      window_->EnableWindowResizing();
    }
    ImGui::EndPopup();
  }

  if (!ImGui::IsPopupOpen("FileDialog") && tmp_.file_dialog != UiTempState::FileDialog_None) {
    const std::string& src = tmp_.last_path[tmp_.file_dialog];
    strncpy(tmp_.filename, src.c_str(), src.size());
    tmp_.filename[src.size()] = 0;
    ImGui::OpenPopup("FileDialog");
    window_->DisableWindowResizing();
  }
}

App::UiState::Dict App::UiState::ToDict() const {
  return {
      {"theme", std::to_string(static_cast<int>(theme))},
      {"font_scale", std::to_string(font_scale)},
      {"window_width", std::to_string(window_width)},
      {"window_height", std::to_string(window_height)},
      {"nthread", std::to_string(nthread)},
  };
}

void App::UiState::FromDict(const Dict& dict) {
  *this = UiState();
  theme = ReadIniValue(dict, "theme", theme);
  window_width = ReadIniValue(dict, "window_width", window_width);
  window_height = ReadIniValue(dict, "window_height", window_height);
  font_scale = ReadIniValue(dict, "font_scale", font_scale);
  nthread = ReadIniValue(dict, "nthread", nthread);
}
}  // namespace mujoco::studio
