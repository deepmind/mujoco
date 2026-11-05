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

#ifndef MUJOCO_STUDIO_UX_GUI_H_
#define MUJOCO_STUDIO_UX_GUI_H_

// A collection of functions for building ImGui panels for common MuJoCo
// visualization and manipulation UX. These functions are primarily used by
// Studio, but are available for other applications.
//
// Like most ImGui functions, the actual "storage" for the GUI state is managed
// by the caller. In most cases, this is already stored in mjModel, mjData,
// mjvOption, etc. But, some functions take additional arguments as needed.

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <imgui.h>
#include <mujoco/mujoco.h>
#include "studio/sim/sim_history.h"
#include "studio/sim/sim_profiler.h"
#include "studio/sim/step_control.h"
#include "studio/ux/theme.h"

namespace mujoco::studio {

// Render a section header with custom, visually-balanced smaller
// expand/collapse arrow.
bool SectionHeader(const char* label, ImGuiTreeNodeFlags flags = 0, float arrow_scale = 0.55f);

// Returns the default INI settings file path (~/.mujoco.ini).
std::string GetDefaultIniPath();

// Loads the GUI theme from an INI file (defaults to ~/.mujoco.ini if empty).
GuiTheme LoadTheme(const std::string& ini_path = "", GuiTheme def_theme = GuiTheme::kLight);

// Loads settings from an INI file into ImGui and returns the loaded theme.
GuiTheme LoadSettings(const std::string& ini_path = "", GuiTheme def_theme = GuiTheme::kLight);

// Saves settings (including ImGui state and theme) to an INI file.
void SaveSettings(GuiTheme theme, const std::string& ini_path = "");

// Resets configuration INI file.
void ResetConfig(const std::string& ini_path = "");

// Rescales all dock node widths by the given ratio.
void RescaleDock(float ratio);

// Configures the ImGui docking module to the standard layout used by Studio.
// This includes the following named sections:
//   "ToolBar": fixed size bar spanning the top of the window; for placing
//       buttons and other controls that are always needed.
//   "StatusBar": fixed size bar spanning the bottom of the window; for
//       placing information and controls that are always needed.
//   "Options": resizable section of the left; designed for GUI elements that
//       are used to configure the simulation (e.g. PhysicsGui).
//   "Inspector": resizable section of the right; designed for inspecting
//       or manipulating mjData elements (e.g. ControlsGui).
//   "Explorer": secondary tab connected to the Inspector; designed for
//       displaying the tree of mjSpec elements.
//   "Properties": resizable section below the explorer; designed for displaying
//       properties of mjSpec elements (e.g. BodyPropertiesGui); hidden by
//       default.
//
// Returns the size and position of the remaining workspace area which can then
// be used to place additional elements (e.g. floating charts).
ImVec4 ConfigureDockingLayout(bool show_toolbar = true, bool show_status_bar = false);

// logarithmically spaced real-time slow-down coefficients (percent)
// clang-format off
static constexpr std::array<const char*, 31> kPercentRealTime = {
"100.0 ", " 80.0 ", " 63.0 ", " 50.0 ", " 40.0 ", " 32.0 ", " 25.0 ", " 20.0 ", " 16.0 ", " 13.0 ",
" 10.0 ", "  8.0 ", "  6.3 ", "  5.0 ", "  4.0 ", "  3.2 ", "  2.5 ", "  2.0 ", "  1.6 ", "  1.3 ",
"  1.0 ", "  0.8 ", "  0.63", "  0.5 ", "  0.4 ", "  0.32", "  0.25", "  0.2 ", "  0.16", "  0.13",
"  0.1 ",
};
// clang-format on

// UX for controlling the simulation stepping. `speed_index` is an index into
// kPercentRealTime, an array of available speeds (indices in range [0, 30] map
// to real-time percentages in range [100%, 0.1%]).
void StepControlGui(StepControl* step_control, int& speed_index);

// Sets the simulation speed index and updates the StepControl object.
void SetSpeedIndex(StepControl* step_control, int& speed_index, int request_idx);

// Loads history frame `index` into `data`. A no-op if the requested frame is
// empty.
void LoadHistoryFrame(SimHistory& history, const mjModel* model, mjData* data, int index);

// Per-session state for the timeline scrubber widget: the simulation time at
// the head of the history buffer, the history buffer size and current offset,
// the (monotonically-growing) label box widths, and the current drag.
struct SimulationTimelineState {
  double sim_head_time = 0.0;
  int history_size = 0;
  int history_index = 0;
  float lh_width = 0.0f;
  float rh_width = 0.0f;
  bool scrubber_active = false;
  float scrubber_grab_offset = 0.0f;

  void SetHistorySize(int size) {
    history_size = std::max(0, size);
    history_index = history_size > 0 ? std::clamp(history_index, 1 - history_size, 0) : 0;
  }

  void SetHistoryIndex(int index) {
    history_index = history_size > 0 ? std::clamp(index, 1 - history_size, 0) : 0;
  }
};

// Appends the state in `data` to `history` and moves the head of the timeline
// to that state. Call whenever the simulation has advanced.
void RecordHistoryFrame(SimHistory& history, SimulationTimelineState& timeline,
                        const mjModel* model, const mjData* data);

// Clears `history`, sizes it for `model`, resets the scrubber and records the
// state in `data` as the first frame. Call on model load and after a reset.
void ResetHistory(SimHistory& history, SimulationTimelineState& timeline, const mjModel* model,
                  const mjData* data);

// The timeline scrubber row: a spine with a draggable knob that scrubs through
// the simulation history. Used by the Simulation panel and the toolbar.
void TimelineScrubberGui(const mjModel* model, mjData* data, StepControl& step_control,
                         SimulationTimelineState& timeline);

// Everything the Simulation panel reads or drives. All pointers are owned by
// the caller and edited in place; the callbacks perform application actions the
// panel cannot do on its own.
struct SimulationGuiContext {
  const mjModel* model = nullptr;
  mjData* data = nullptr;
  StepControl* step_control = nullptr;
  SimulationTimelineState* timeline = nullptr;
  int* speed_index = nullptr;
  int* key_idx = nullptr;
  int* nthread = nullptr;
  bool* update_threadpool = nullptr;
  std::function<void()> reset;          // reset the physics state
  std::function<void()> reload;         // reload the model
  std::function<void()> align;          // recenter the camera on the model
  std::function<void()> save_keyframe;  // save active state to key_idx
};

// The Simulation panel: reset / reload / align, a pause-run toggle, the speed
// slider, the history scrubber, keyframe controls and the thread count. This is
// a reusable view over the platform simulation objects; it holds no application
// state of its own.
void SimulationGui(const SimulationGuiContext& ctx);

// UX for selecting the GUI theme.
bool ThemeSelectGui(GuiTheme* theme, const ImVec2& size = ImVec2(0, 0));

// UX for selecting the GUI theme from a menu item.
bool ThemeMenuGui(GuiTheme* theme);

// UX for selecting the visualization label option.
bool LabelSelectionGui(mjvOption* opts);

// UX for selecting the visualization frame option.
bool FrameSelectionGui(mjvOption* opts);

// Get the display name for a keyframe given its index.
std::string GetKeyframeName(const mjModel* model, int index);

// Get the display name for a camera given its index.
std::string GetCameraName(const mjModel* model, const mjvCamera& camera, int index);

// UX for selecting the camera.
bool CameraSelectionGui(const mjModel* model, mjData* data, mjvCamera& camera, int& index);

// Transient camera animation state of ViewGizmoGui, owned by the caller.
struct ViewGizmoAnimation {
  double start_time = -1.0;  // negative when idle
  double start_azimuth = 0;
  double start_elevation = 0;
  double target_azimuth = 0;
  double target_elevation = 0;
};

// XYZ axes in the top-right corner of the 3D view (`workspace_rect`, from
// ConfigureDockingLayout) that turn with the camera. Clicking an axis animates
// the camera to look along it, or from the other side if it already does;
// dragging orbits the camera. A fixed camera is first swapped for the tumble
// camera, updating `camera_index`. The animation progress is kept in
// `animation`. Returns true if the camera changed.
bool ViewGizmoGui(const mjModel* model, const mjData* data, mjvCamera& camera, int& camera_index,
                  ViewGizmoAnimation& animation, const ImVec4& workspace_rect);

// UX for controlling the physics simulation parameters (e.g. integrator,
// solver, etc.) in mjOption.
void PhysicsGui(mjModel* model, mjSpec* spec = nullptr, float min_width = 150.0f);

// UX for enabling/disabling visualization groups in mjvOption.
void GroupsGui(const mjModel* model, mjvOption* vis_options, float min_width);

// UX for enabling/disabling rendering (mjtRndFlag) and visualization
// (mjtVisFlag) flags. We combine these into a single function because the sets
// of flags are closely related.
void RenderingGui(const mjModel* model, mjvOption* vis_options, mjtByte* render_flags,
                  float min_width);

// UX for controlling the mjvOption and mjvCamera settings used for visualizing
// scenes (mjvScene).
void VisualizationGui(mjModel* model, mjvOption* vis_options, mjvCamera* camera, float min_width);

// UX for visualizing actuator controls data in mjData.
void ControlsGui(const mjModel* model, mjData* data, const mjvOption* vis_options);

// UX for visualizing joint data in mjData.
void JointsGui(const mjModel* model, const mjData* data, const mjvOption* vis_options);

// UX for visualizing sensor data in mjData.
void SensorGui(const mjModel* model, const mjData* data);

// UX for visualizing the data as returned from mj_getState(). We use a
// user-supplied vector here to avoid allocating memory every frame.
void StateGui(const mjModel* model, mjData* data, std::vector<mjtNum>& state, int& state_sig,
              float min_width);

// UX for visualizing a named field from mjData. `field_name` and `field_index`
// are used to index into the data buffer.
void WatchGui(const mjModel* model, const mjData* data, char* field_name, int field_len,
              int& field_index);

// UX for controlling noise parameters which can then be applied to the
// simulation via StepControl::SetNoiseParameters / StepControl::InjectNoise.
void NoiseGui(StepControl* step_control);

// UX for the solver convergence chart.
void ConvergenceGui(const mjModel* model, mjData* data, ImVec2 plot_size = ImVec2(-1, 0));

// UX for the solver counts chart.
void CountsGui(const mjModel* model, mjData* data, ImVec2 plot_size = ImVec2(-1, 0));

// UX for Profiler panel combining Solver and Performance metrics.
void ProfilerGui(const mjModel* model, mjData* data, SimProfiler* profiler, bool show_iter);

// UX for displaying basic simulation information. Note that the pause state and
// FPS and GPU frame time need to be tracked by the caller and passed here to be
// displayed.
void InfoGui(const mjModel* model, const mjData* data, bool paused, float fps, float gpu_ms,
             const mjTimerStat* timer_prev = nullptr);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_GUI_H_
