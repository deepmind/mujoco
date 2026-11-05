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

#ifndef MUJOCO_STUDIO_APP_H_
#define MUJOCO_STUDIO_APP_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>
#include "render/filament/support/imgui_bridge.h"
#include "render/filament/support/renderer.h"
#include "studio/hal/graphics_mode.h"
#include "studio/hal/window.h"
#include "studio/render/model_scene.h"
#include "studio/sim/model_holder.h"
#include "studio/sim/sim_history.h"
#include "studio/sim/sim_profiler.h"
#include "studio/sim/step_control.h"
#include "studio/ux/gui.h"
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/interaction.h"
#include "studio/ux/picture_gui.h"
#include "studio/ux/spec_editor.h"

namespace mujoco::studio {

struct AppTestPeer;

// Owns, updates, and renders a MuJoCo simulation.
class App {
 public:
  // Configuration/initialization options for the application.
  struct Config {
    // The original width and height of the window.
    int width = 0;
    int height = 0;

    // The path to the ini file containing the user settings.
    std::string ini_path;

    // The graphics configuration used for initializing the window.
    GraphicsMode gfx_mode = GraphicsMode::FilamentVulkan;

    // The initial GUI theme. If set, overrides the default (kLight).
    std::optional<GuiTheme> initial_theme;

    // The application title shown in the window title bar.
    std::string title = "MuJoCo Studio";
  };

  explicit App(Config config);
  ~App();

  // Loads an empty mjModel.
  void InitEmptyModel();

  // Loads an mjModel from the given file. This extension should be one of:
  // .xml, .mjb, or .mjz.
  void LoadModelFromFile(const std::string& filepath);

  // Loads an mjModel from the given memory buffer. The content_type should be
  // one of: "text/xml", "application/mjb", or "application/mjz". For zip files,
  // a name is required in order to uniquely identify the model within the
  // archive.
  void LoadModelFromBuffer(std::span<const std::byte> buffer, std::string_view content_type,
                           std::string_view name);

  // Selects and loads a keyframe by name or numerical index. If invalid,
  // silently ignores it.
  void LoadKeyframe(std::string_view keyframe);

  // Processes window events and advances the state of the simulation.
  bool Update();

  // Builds the GUI. We do this after Sync() to ensure we have the latest data
  // for building the GUI.
  void BuildGui();

  // Renders everything (e.g. the simulation and the GUI).
  void Render();

 private:
  friend struct AppTestPeer;
  // The kind of model that is currently loaded.
  enum ModelKind {
    kEmptyModel,
    kModelFromFile,
    kModelFromBuffer,
  };

  // Information for building a new empty model.
  struct EmptyModel {};

  // Information needed for building a model from a file.
  struct FileModel {
    std::string_view filepath;
  };

  // Information needed for building a model from a memory buffer.
  struct BufferModel {
    std::span<const std::byte> buffer;
    std::string_view content_type;
    std::string_view name;
  };

  // Information needed for recompiling the model.
  struct RecompileModel {};

  // Information needed for recompiling the model from the spec editor.
  struct RecompileFromSpec {};

  // The different ways in which the model can be built.
  using BuildModelInfo =
      std::variant<EmptyModel, FileModel, BufferModel, RecompileModel, RecompileFromSpec>;

  enum class SpecPropertiesMode {
    kSpec,
    kModel,
    kData,
  };

  // UI state that is persisted across application runs
  struct UiState {
    GuiTheme theme = GuiTheme::kDark;
    float font_scale = 1.0f;
    int window_width = 0;
    int window_height = 0;
    int nthread = 0;

    using Dict = std::unordered_map<std::string, std::string>;
    Dict ToDict() const;
    void FromDict(const Dict& dict);
  };

  // UI state that is transient and only needed while the application runs
  struct UiTempState {
    bool should_exit = false;
    bool update_threadpool = false;

    // Windows.
    bool help = false;
    bool info = false;
    bool profiler = false;
    bool profiler_show_iter = false;
    bool picture_in_picture = false;
    bool options_panel = true;
    bool toolbar = false;
    bool view_gizmo = true;
    bool status_bar = true;
    bool console = false;
    bool inspector_panel = true;
    bool editor_panel = false;
    bool full_screen = false;
    bool style_editor = false;
    bool filament_editor = false;
    bool imgui_demo = false;
    bool implot_demo = false;
    float editor_split = -1;
    float explorer_split = -1;

    int camera_idx = kTumbleCameraIdx;
    int key_idx = -1;

    // View gizmo camera animation.
    ViewGizmoAnimation view_gizmo_animation;

    // Controls.
    int speed_index = 0;
    float cam_speed = 0.0f;

    // Spec editing.
    SpecPropertiesMode spec_prop_mode = SpecPropertiesMode::kSpec;
    mjsElement* curr_element = nullptr;

    // Watch.
    char watch_field[1000] = "qpos";
    int watch_index = 0;

    // State.
    int state_sig = 0;
    std::vector<mjtNum> state;

    // Picture-in-Picture.
    std::vector<PipState> pips;

    // File dialogs.
    enum FileDialog {
      FileDialog_None,
      FileDialog_Load,
      FileDialog_SaveXml,
      FileDialog_SaveMjb,
      FileDialog_PrintModel,
      FileDialog_PrintData,
      FileDialog_SaveScreenshot,
      NumFileDialogs,
    };
    FileDialog file_dialog = FileDialog_None;
    std::string last_path[NumFileDialogs];
    char filename[1000] = "";
  };

  // Requests that the model be loaded from the given file at the next update.
  void RequestModelLoad(std::string model_file);

  // Requests that the currently loaded model be reloaded at the next update.
  void RequestModelReload();

  // (Re)builds the model based on the given configuration.
  void BuildModel(const BuildModelInfo& info);

  // Updates the currently loaded model to the given model. If model is null,
  // then compile the spec to a model.
  void OnModelLoaded(std::string_view filename, ModelKind model_kind);

  // Populates the scene with the recently loaded model.
  void InitModelScene();

  // Returns the Renderer::View for the main simulation.
  Renderer::View GetSceneView();

  // Returns the Renderer::View for the UX.
  Renderer::View GetUxView();

  struct KeyframeSelection {
    bool is_reload = false;
    int key_idx = -1;
    std::string key_name;
    std::vector<std::string> all_old_names;
  };
  KeyframeSelection CaptureKeyframeSelection(bool is_reload) const;
  void RestoreKeyframeSelection(const KeyframeSelection& saved);

  void SwitchGraphicsMode(int width, int height, GraphicsMode mode);

  void SetLoadError(std::string error);
  void RecordLoadError(std::string_view error);
  void UpdateFilePaths(const std::string& resolved_path);

  void ResetPhysics();
  void UpdatePhysics();
  void PreStep(const mjModel* m, mjData* d);
  void PostStep(const mjModel* m, mjData* d);

  void LoadSettings();
  void SaveSettings();
  void ApplyWindowStateStorage();

  void LoadHistory(int offset);
  // Loads the frame selected in the timeline GUI if it differs from the frame
  // the history buffer is currently at. Call after rendering a scrubber.
  void ApplyTimelineScrub();

  void SetSpeedIndex(int idx);

  void HandleWindowEvents();
  void HandleMouseEvents();
  void HandleKeyboardEvents();

  void ProcessPendingLoads();

  void MainMenuGui();
  void ToolBarGui();
  void StatusBarGui();
  void HelpGui();
  void FileDialogGui();
  void ModelOptionsGui();
  void DataInspectorGui();
  void SpecExplorerGui();
  void SpecEditorGui();

  mjSpec* spec() { return model_holder_->spec(); }
  mjModel* model() { return model_holder_->model(); }
  mjData* data() { return model_holder_->data(); }
  bool has_spec() const { return model_holder_ && model_holder_->spec(); }
  bool has_model() const { return model_holder_ && model_holder_->model(); }
  bool has_data() const { return model_holder_ && model_holder_->data(); }

  std::span<const std::byte> RenderToTexture(mjvCamera* camera, int width, int height);

  std::unique_ptr<Window> window_;
  std::unique_ptr<Renderer> renderer_;
  std::unique_ptr<ModelHolder> model_holder_;
  std::unique_ptr<ModelScene> model_scene_;
  std::unique_ptr<ImguiBridge> imgui_bridge_;

  mutable std::mutex physics_mutex_;
#ifndef __EMSCRIPTEN__
  void PhysicsThreadLoop();
  std::thread physics_thread_;
  std::atomic<bool> stop_physics_thread_{false};
#endif

  std::string app_title_;
  std::string ini_path_;

  ModelKind model_kind_ = kEmptyModel;
  std::string model_path_;
  std::string load_error_;

  // Cached previous state.
  std::vector<std::byte> last_buffer_;
  std::string last_content_type_;
  StepControl::PauseState last_pause_state_ = StepControl::PauseState::kNormalPaused;
  std::chrono::steady_clock::time_point last_frame_time_;
  double fps_ = 0;

  // Pending operations.
  std::optional<std::string> pending_load_;
  bool pending_reload_ = false;
  bool recompile_spec_ = false;
  std::function<void()> pending_op_;
  bool preserve_camera_on_load_ = false;

  // Studio components.
  StepControl step_control_;
  SimProfiler profiler_;
  SimHistory sim_history_;
  SimulationTimelineState timeline_;
  SpecEditor spec_editor_;
  // Window state storage (e.g. collapsing header open/closed state), keyed
  // "<window name>/<id>", which ImGui does not serialize. Entries stay
  // pending until their window is first created.
  KeyValues window_state_storage_;

  mjVFS vfs_;
  mjvCamera camera_;
  mjvPerturb perturb_;
  mjvOption vis_options_;
  mjvScene plugin_scene_;
  mjtByte render_flags_[mjNRNDFLAG];

  std::vector<std::string> search_paths_;

  std::array<mjTimerStat, mjNTIMER> timer_prev_ = {};
  UiState ui_;
  UiTempState tmp_;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_APP_H_
