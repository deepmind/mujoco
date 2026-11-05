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

#include "studio/ux/gui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include "studio/sim/sim_history.h"
#include "studio/sim/sim_profiler.h"
#include "studio/sim/step_control.h"
#include "studio/ux/gui_helpers.h"
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/interaction.h"

namespace mujoco::studio {
namespace {
struct SpeedStatus {
  bool misaligned;
  float measured;
};

// Reports whether the measured step speed has drifted from the desired speed
// enough for the toolbar to surface it, plus the measured value itself.
static SpeedStatus IsSpeedMisaligned(const StepControl& step_control) {
  static bool misaligned = false;

  const float desired = step_control.GetSpeed();
  const float measured = step_control.GetSpeedMeasured();
  const float deviation = std::abs(measured - desired);

  // Hysteresis: a raw threshold flips at exactly the 10% band edge, and
  // measurement noise straddling it would toggle the preview text (and with it
  // the combo width) every few frames causing layout flicker. Require the state
  // to leave a wider band before switching back: become misaligned past 12%,
  // return to aligned only under 8%, and follow an extreme deviation (>20%)
  // immediately.
  const bool misaligned_now = deviation > 0.1f * desired;
  misaligned = misaligned ? (deviation > 0.08f * desired) : (deviation > 0.12f * desired);
  if (misaligned_now != misaligned && deviation > 0.2f * desired) {
    misaligned = misaligned_now;  // far outside both bands: follow immediately
  }

  return {misaligned, measured};
}
}  // namespace

static ImVec2 GetFlexElementSize(int num_cols) {
  const float width = (GetStableAvailWidth() / num_cols) - ImGui::GetStyle().FramePadding.x * 2;
  return ImVec2(width, 0);
}

bool SectionHeader(const char* label, ImGuiTreeNodeFlags flags, float arrow_scale) {
  const bool is_framed = (flags & ImGuiTreeNodeFlags_Framed) != 0;

  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  if (window->SkipItems) return false;

  const ImGuiStyle& style = g.Style;
  const ImGuiID id = window->GetID(label);

  // Control the bar height via FramePadding.y.
  // Main (framed): a bit of padding. Sub (unframed): less padding to be
  // visually thinner.
  const float pad_y = (is_framed ? 3.0f : 2.0f) * style.FontScaleDpi;
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(style.FramePadding.x, pad_y));

  // Push transparent text color to hide the default arrow and label rendered by
  // TreeNodeBehavior.
  ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 0, 0, 0));

  // Record cursor position before the widget so we can compute the arrow
  // position.
  const ImVec2 cursor_before = ImGui::GetCursorScreenPos();

  // Use TreeNodeBehavior (the same function CollapsingHeader uses internally).
  // This gives us proper click-to-toggle and open/close state management.
  // We do NOT add _Leaf here so toggling works, and we don't add
  // _NoTreePushOnOpen so the callers' TreePop() still works.
  //
  // For unframed nodes, add _FramePadding so TreeNodeBehavior uses
  // FramePadding.y for layout (making arrow Y position predictable = cursor.y +
  // pad_y). Also add _SpanAvailWidth for consistent full-width hit testing.
  ImGuiTreeNodeFlags effective_flags = flags;
  if (!is_framed) {
    effective_flags |= ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth;
  }
  bool is_open = ImGui::TreeNodeBehavior(id, effective_flags, label);

  // Restore style state before manual rendering (so we get the correct text
  // color).
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();

  // --- Render custom arrow and text manually ---
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImU32 text_col = ImGui::GetColorU32(ImGuiCol_Text);

  // Match position offsets used by TreeNodeBehavior.
  // Arrow coordinate: (cursor_before.x + FramePadding.x, cursor_before.y +
  // pad_y)
  const float pad_x = style.FramePadding.x;
  const float arrow_x = cursor_before.x + pad_x;
  const float arrow_y = cursor_before.y + pad_y;
  const float full_arrow_size = g.FontSize;

  // Draw the custom smaller arrow, centered within the same region.
  const float custom_size = full_arrow_size * arrow_scale;
  const float offset = (full_arrow_size - custom_size) * 0.5f;
  const ImGuiDir arrow_dir = is_open ? ImGuiDir_Down : ImGuiDir_Right;
  ImGui::RenderArrow(dl, ImVec2(arrow_x + offset, arrow_y + offset), text_col, arrow_dir,
                     arrow_scale);

  // Render text manually.
  // text_offset_x = FontSize + padding.x * (display_frame ? 3 : 2)
  const float text_offset_x = full_arrow_size + pad_x * (is_framed ? 3.0f : 2.0f);
  ImGui::RenderText(ImVec2(cursor_before.x + text_offset_x, cursor_before.y + pad_y), label);

  return is_open;
}

std::string GetDefaultIniPath() {
  const char* home = std::getenv("HOME");
  if (!home) {
    home = std::getenv("USERPROFILE");
  }
  return std::string(home ? home : ".") + "/.mujoco.ini";
}

GuiTheme LoadTheme(const std::string& ini_path, GuiTheme def_theme) {
  const std::string path = ini_path.empty() ? GetDefaultIniPath() : ini_path;
  const std::string settings = LoadText(path);
  if (settings.empty()) {
    return def_theme;
  }
  KeyValues ux_section = ReadIniSection(settings, "[Studio][UX]");
  return ReadIniValue(ux_section, "theme", def_theme);
}

GuiTheme LoadSettings(const std::string& ini_path, GuiTheme def_theme) {
  const std::string path = ini_path.empty() ? GetDefaultIniPath() : ini_path;
  const std::string settings = LoadText(path);
  if (settings.empty()) {
    return def_theme;
  }
  KeyValues ux_section = ReadIniSection(settings, "[Studio][UX]");
  GuiTheme theme = ReadIniValue(ux_section, "theme", def_theme);
  if (ImGui::GetCurrentContext() != nullptr) {
    ImGui::LoadIniSettingsFromMemory(settings.data(), settings.size());
  }
  return theme;
}

void SaveSettings(GuiTheme theme, const std::string& ini_path) {
  const std::string path = ini_path.empty() ? GetDefaultIniPath() : ini_path;
  const std::string existing = LoadText(path);
  std::string settings;
  if (ImGui::GetCurrentContext() != nullptr) {
    settings = ImGui::SaveIniSettingsToMemory();
  }

  // Preserve existing [Studio][UX] keys while updating theme.
  KeyValues ux_dict = ReadIniSection(existing, "[Studio][UX]");
  ux_dict["theme"] = std::to_string(static_cast<int>(theme));
  AppendIniSection(settings, "[Studio][UX]", ux_dict);

  // Preserve other Studio custom sections (e.g. [Studio][Plugins],
  // [Studio][WindowStateStorage]).
  std::istringstream f(existing);
  std::string line;
  std::vector<std::string> custom_sections;
  while (std::getline(f, line)) {
    if (!line.empty() && line.front() == '[' && line.back() == ']') {
      if (line.rfind("[Studio]", 0) == 0 && line != "[Studio][UX]") {
        if (std::find(custom_sections.begin(), custom_sections.end(), line) ==
            custom_sections.end()) {
          custom_sections.push_back(line);
        }
      }
    }
  }
  for (const auto& sec : custom_sections) {
    KeyValues kv = ReadIniSection(existing, sec);
    AppendIniSection(settings, sec, kv);
  }

  SaveText(settings, path);
}

void ResetConfig(const std::string& ini_path) {
  const std::string path = ini_path.empty() ? GetDefaultIniPath() : ini_path;
  SaveText("\n\n", path);
}

void RescaleDock(float ratio) {
  if (ratio == 1) return;
  ImGuiID root = ImGui::GetID("Root");
  ImGuiDockNode* root_node = ImGui::DockBuilderGetNode(root);
  if (root_node) {
    struct ScaleNodes {
      static void Apply(ImGuiDockNode* node, float r) {
        node->SizeRef.x *= r;
        if (node->ChildNodes[0]) Apply(node->ChildNodes[0], r);
        if (node->ChildNodes[1]) Apply(node->ChildNodes[1], r);
      }
    };
    ScaleNodes::Apply(root_node, ratio);
  }
}

ImVec4 ConfigureDockingLayout(bool show_toolbar, bool show_status_bar) {
  ImGuiViewport* viewport = ImGui::GetMainViewport();
  const float scale = ImGui::GetWindowDpiScale();
  const float font_scale = ImGui::GetIO().FontGlobalScale;

  const float kOptionsRelWidth = 0.15f;
  const float kInspectorRelWidth = 0.22f;
  const float kPropertiesRelHeight = 0.3f;
  const float kToolsBarHeight = show_toolbar ? 36.f * scale * font_scale : 0.0f;
  const float status_bar_height = show_status_bar ? kStatusBarHeight * scale * font_scale : 0.0f;

  // Lay out from the absolute viewport origin with an explicitly computed
  // menu bar height, NOT from WorkPos/WorkSize: the work area only reflects
  // the menu bar's reservation from the previous frame, so any hiccup in
  // that reservation shifts the whole dock layout by a menu-bar height for
  // a frame and every docked window re-clips (visible as UI-wide flicker).
  // The menu bar's height equals the frame height by definition.
  const float menu_bar_height = ImGui::GetFrameHeight();
  const ImVec2 toolbar_pos{viewport->Pos.x, viewport->Pos.y + menu_bar_height};
  const ImVec2 dockspace_pos{viewport->Pos.x, viewport->Pos.y + menu_bar_height + kToolsBarHeight};
  const ImVec2 dockspace_size{
      viewport->Size.x, viewport->Size.y - menu_bar_height - kToolsBarHeight - status_bar_height};

  ImGuiID root = ImGui::GetID("Root");
  const bool first_time = (ImGui::DockBuilderGetNode(root) == nullptr);

  if (first_time) {
    ImGui::DockBuilderRemoveNode(root);
    ImGui::DockBuilderAddNode(root, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(root, dockspace_size);

    // Slice up the main dock space.
    ImGuiID main = root;

    ImGuiID options = 0;
    ImGui::DockBuilderSplitNode(main, ImGuiDir_Left, kOptionsRelWidth, &options, &main);

    ImGuiID inspector = 0;
    ImGui::DockBuilderSplitNode(main, ImGuiDir_Right, kInspectorRelWidth, &inspector, &main);

    ImGuiID properties = 0;
    ImGui::DockBuilderSplitNode(inspector, ImGuiDir_Down, kPropertiesRelHeight, &properties,
                                &inspector);

    ImGuiID console = 0;
    ImGui::DockBuilderSplitNode(main, ImGuiDir_Down, 0.25f, &console, &main);

    ImGuiID profiler = 0;
    ImGui::DockBuilderSplitNode(main, ImGuiDir_Right, 0.42f, &profiler, &main);

    ImGui::DockBuilderDockWindow("Dockspace", main);
    ImGui::DockBuilderDockWindow("Options", options);
    ImGui::DockBuilderDockWindow("Explorer", inspector);
    ImGui::DockBuilderDockWindow("Editor", inspector);
    ImGui::DockBuilderDockWindow("Inspector", inspector);
    ImGui::DockBuilderDockWindow("Properties", properties);
    ImGui::DockBuilderDockWindow("Console", console);
    ImGui::DockBuilderDockWindow("Profiler", profiler);
    ImGui::DockBuilderFinish(root);
  }

  // Create a dummy window filling the entire workspace in which we can perform
  // docking.
  ImGui::SetNextWindowPos(dockspace_pos);
  ImGui::SetNextWindowSize(dockspace_size);
  ImGui::SetNextWindowViewport(viewport->ID);

  const ImGuiWindowFlags kWorkspaceFlags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking |
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
      ImGuiWindowFlags_NoBackground;

  const ImGuiWindowFlags kFixedFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoDocking;

  // Main workspace area in which we can dock other windows.
  {
    ScopedStyle style;
    style.Var(ImGuiStyleVar_WindowRounding, 0.0f);
    style.Var(ImGuiStyleVar_WindowBorderSize, 0.0f);
    style.Var(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Dockspace", nullptr, kWorkspaceFlags);
    const ImGuiDockNodeFlags kDockSpaceFlags =
        ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingOverCentralNode;
    ImGui::DockSpace(root, ImVec2(0.0f, 0.0f), kDockSpaceFlags);
    ImGui::End();
  }

  // Toolbar is fixed at the top.
  if (show_toolbar) {
    ScopedStyle style;
    style.Var(ImGuiStyleVar_WindowBorderSize, 1.0f);
    style.Var(ImGuiStyleVar_WindowRounding, 0.0f);
    style.Var(ImGuiStyleVar_WindowMinSize, ImVec2(1, 1));
    const float toolbar_vpad =
        std::max(0.f, (36.f * scale * font_scale - ImGui::GetFrameHeight()) * 0.5f);
    style.Var(ImGuiStyleVar_WindowPadding, ImVec2(4 * scale, toolbar_vpad));
    ImGui::SetNextWindowPos(toolbar_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, 36.f * scale * font_scale), ImGuiCond_Always);
    ImGui::Begin("ToolBar", nullptr, kFixedFlags);
    ImGui::End();
  }

  // StatusBar is fixed at the bottom.
  if (show_status_bar) {
    ScopedStyle style;
    style.Var(ImGuiStyleVar_WindowBorderSize, 1.0f);
    style.Var(ImGuiStyleVar_WindowRounding, 0.0f);
    style.Var(ImGuiStyleVar_WindowMinSize, ImVec2(1, 1));
    ImGui::SetNextWindowPos(ImVec2(0, viewport->Size.y - status_bar_height), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, status_bar_height), ImGuiCond_Always);
    ImGui::Begin("StatusBar", nullptr, kFixedFlags);
    ImGui::End();
  }

  ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(root);
  if (central) {
    return ImVec4(central->Pos.x, central->Pos.y, central->Size.x, central->Size.y);
  }
  const int settings_width = dockspace_size.x * kOptionsRelWidth;
  const int inspector_width = dockspace_size.x * kInspectorRelWidth;
  const float workspace_x = dockspace_pos.x + settings_width;
  const float workspace_y = dockspace_pos.y;
  const float workspace_w = dockspace_size.x - settings_width - inspector_width;
  const float workspace_h = dockspace_size.y;
  return ImVec4(workspace_x, workspace_y, workspace_w, workspace_h);
}

void StepControlGui(StepControl* step_control, int& speed_index) {
  ScopedStyle style;

  bool is_dark = ImGui::GetStyle().Colors[ImGuiCol_WindowBg].x < 0.5f;
  const ImColor yellow = is_dark ? ImColor(158, 115, 18, 255) : ImColor(255, 215, 0, 255);
  const ImColor green = is_dark ? ImColor(40, 125, 60, 255) : ImColor(40, 180, 40, 255);

  auto make_button = [&](const char* icon, StepControl::PauseState target_state, ImColor color,
                         ImDrawFlags corners, const char* tooltip = "", float hover_alpha = 1.f,
                         float width_scale = 1.f) {
    ImVec2 size(0, 0);
    if (width_scale != 1.f) {
      const ImGuiStyle& s = ImGui::GetStyle();
      const float w = ImGui::CalcTextSize(icon).x + s.FramePadding.x * 2;
      size.x = w * width_scale;
    }
    bool active = step_control->GetPauseState() == target_state;
    if (ImGui_ColorButtonEx(icon, active, color, corners, size, hover_alpha)) {
      step_control->SetPauseState(target_state);
    }
    if (!std::string_view(tooltip).empty()) {
      ImGui::SetItemTooltip("%s", tooltip);
    }
  };

  make_button(ICON_FA_PAUSE, StepControl::PauseState::kNormalPaused, yellow,
              ImDrawFlags_RoundCornersLeft, "Pause", .3f, 1.6f);
  ImGui::SameLine(0.f, 0.f);
  make_button(ICON_FA_PLAY, StepControl::PauseState::kUnpaused, green,
              ImDrawFlags_RoundCornersRight, "", .3f, 1.6f);

  // Speed selection.
  style.Reset();
  ImGui::SameLine(0, ImGui::GetFrameHeight() * .6f);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x +
                                                             5.f * ImGui::GetStyle().FontScaleDpi,
                                                         ImGui::GetStyle().FramePadding.y));

  const auto [misaligned, measured] = IsSpeedMisaligned(*step_control);

  char speed_preview[64];
  if (misaligned) {
    snprintf(speed_preview, sizeof(speed_preview), "%s %s (%-4.1f%%)", ICON_FA_TACHOMETER,
             kPercentRealTime[speed_index], measured);
  } else {
    snprintf(speed_preview, sizeof(speed_preview), "%s %s", ICON_FA_TACHOMETER,
             kPercentRealTime[speed_index]);
  }

  // Size the combo for the worst-case text so toggling the measured-speed
  // suffix (or its digit count changing) never shifts the toolbar layout.
  char speed_sizing[64];
  snprintf(speed_sizing, sizeof(speed_sizing), "%s %s (-99.9%%)", ICON_FA_TACHOMETER,
           kPercentRealTime[speed_index]);
  ImGui::SetNextItemWidth(ImGui::CalcTextSize(speed_sizing).x +
                          ImGui::GetStyle().FramePadding.x * 2.f);
  if (ImGui::BeginCombo("##Speed", speed_preview, ImGuiComboFlags_NoArrowButton)) {
    for (int n = 0; n < kPercentRealTime.size(); n++) {
      if (ImGui::Selectable(kPercentRealTime[n], (speed_index == n))) {
        SetSpeedIndex(step_control, speed_index, n);
      }
    }
    ImGui::EndCombo();
  }

  ImGui::PopStyleVar();
  if (misaligned) {
    ImGui::SetItemTooltip("%s", "Desired Speed (Measured Speed)");
  } else {
    ImGui::SetItemTooltip("%s", "Desired Speed");
  }
}

void SetSpeedIndex(StepControl* step_control, int& speed_index, int request_idx) {
  if (!step_control || request_idx == speed_index || kPercentRealTime.empty()) {
    return;
  }

  speed_index = std::clamp<int>(request_idx, 0, kPercentRealTime.size() - 1);
  float speed = std::stof(kPercentRealTime[speed_index]);
  step_control->SetSpeed(speed);
}

void LoadHistoryFrame(SimHistory& history, const mjModel* model, mjData* data, int index) {
  if (model == nullptr || data == nullptr) {
    return;
  }
  std::span<mjtNum> state = history.SetIndex(index);
  if (!state.empty()) {
    mj_setState(model, data, state.data(), mjSTATE_INTEGRATION);
    mj_forward(model, data);
  }
}

void RecordHistoryFrame(SimHistory& history, SimulationTimelineState& timeline,
                        const mjModel* model, const mjData* data) {
  if (model == nullptr || data == nullptr) {
    return;
  }
  std::span<mjtNum> state = history.AddToHistory();
  if (!state.empty()) {
    mj_getState(model, data, state.data(), mjSTATE_INTEGRATION);
    timeline.sim_head_time = data->time;
    timeline.SetHistorySize(history.Size());
    timeline.SetHistoryIndex(history.GetIndex());
  }
}

void ResetHistory(SimHistory& history, SimulationTimelineState& timeline, const mjModel* model,
                  const mjData* data) {
  if (model == nullptr) {
    return;
  }
  history.Init(mj_stateSize(model, mjSTATE_INTEGRATION));
  timeline = SimulationTimelineState();
  RecordHistoryFrame(history, timeline, model, data);
}

static std::string FormatTimelineTime(double time_in_s) {
  if (time_in_s < 0.0) time_in_s = 0.0;
  char buf[64];
  if (time_in_s == 0.0) {
    return "0.00s";
  } else if (time_in_s < 1e-3) {
    // Microseconds range.
    const double t_us = time_in_s * 1e6;
    if (t_us >= 100.0) {
      std::snprintf(buf, sizeof(buf), "%.0f\xC2\xB5s", t_us);
    } else if (t_us >= 10.0) {
      std::snprintf(buf, sizeof(buf), "%.1f\xC2\xB5s", t_us);
    } else {
      std::snprintf(buf, sizeof(buf), "%.2f\xC2\xB5s", t_us);
    }
  } else if (time_in_s < 1.0) {
    // Milliseconds range.
    const double t_ms = time_in_s * 1e3;
    if (t_ms >= 100.0) {
      std::snprintf(buf, sizeof(buf), "%.0fms", t_ms);
    } else if (t_ms >= 10.0) {
      std::snprintf(buf, sizeof(buf), "%.1fms", t_ms);
    } else {
      std::snprintf(buf, sizeof(buf), "%.2fms", t_ms);
    }
  } else {
    // Seconds range.
    if (time_in_s >= 100.0) {
      std::snprintf(buf, sizeof(buf), "%.0fs", time_in_s);
    } else if (time_in_s >= 10.0) {
      std::snprintf(buf, sizeof(buf), "%.1fs", time_in_s);
    } else {
      std::snprintf(buf, sizeof(buf), "%.2fs", time_in_s);
    }
  }
  return buf;
}

void TimelineScrubberGui(const mjModel* model, mjData* data, StepControl& step_control,
                         SimulationTimelineState& timeline) {
  // Timeline scrubber: spine + sliding knob widget.
  const double max_time = timeline.sim_head_time;
  const int hist_size = timeline.history_size;
  const int hist_min = 1 - hist_size;  // most negative index (oldest)
  const int hist_max = 0;              // index 0 = most recent
  int current_index = timeline.history_index;
  const bool locked = (hist_size <= 1);

  // Compute current timestamp for LH label.
  const double timestep = (model != nullptr) ? model->opt.timestep : 0.002;
  double curr_time = (data != nullptr) ? data->time : (max_time + current_index * timestep);
  if (curr_time < 0.0) curr_time = 0.0;

  const int curr_step = ((model != nullptr) && model->opt.timestep > 0)
                            ? static_cast<int>(std::round(curr_time / model->opt.timestep))
                            : 0;
  const int max_step = ((model != nullptr) && model->opt.timestep > 0)
                           ? static_cast<int>(std::round(max_time / model->opt.timestep))
                           : 0;

  // Both LH and RH labels aim to be at 3 digits, switching units
  // independently.
  std::string lh_top_str = FormatTimelineTime(curr_time);
  std::string rh_top_str = FormatTimelineTime(max_time);
  std::string lh_bot_str = "(" + std::to_string(curr_step) + ")";
  std::string rh_bot_str = "(" + std::to_string(max_step) + ")";
  const char* lh_top_label = lh_top_str.c_str();
  const char* rh_top_label = rh_top_str.c_str();
  const char* lh_bot_label = lh_bot_str.c_str();
  const char* rh_bot_label = rh_bot_str.c_str();

  // Use slightly smaller text for timeline labels (e.g. 90% scale).
  ImGui::SetWindowFontScale(0.9f);

  const ImVec2 cursor = ImGui::GetCursorScreenPos();
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  const float base_label_w =
      std::max(ImGui::CalcTextSize("88.8 ms").x, ImGui::CalcTextSize("88.8 \xC2\xB5s").x);
  const float curr_lh_w = std::max(
      {base_label_w, ImGui::CalcTextSize(lh_top_label).x, ImGui::CalcTextSize(lh_bot_label).x});
  if (curr_lh_w > timeline.lh_width) {
    timeline.lh_width = curr_lh_w;
  }
  const float curr_rh_w = std::max(
      {base_label_w, ImGui::CalcTextSize(rh_top_label).x, ImGui::CalcTextSize(rh_bot_label).x});
  if (curr_rh_w > timeline.rh_width) {
    timeline.rh_width = curr_rh_w;
  }
  const float lh_box_w = timeline.lh_width;
  const float rh_box_w = timeline.rh_width;
  const float track_w =
      std::max(10.0f, ImGui::GetContentRegionAvail().x - lh_box_w - rh_box_w - spacing * 2.0f);

  const float spine_h = 3.0f;
  const float knob_w = 12.0f;
  const float knob_h = 27.0f;  // 1.5x taller than 18.0f
  const float text_line_h = ImGui::GetTextLineHeight();
  const float custom_line_spacing = 1.0f;  // Decreased spacing between the two lines
  const float text_2lines_h = text_line_h * 2.0f + custom_line_spacing;
  const float total_h = std::max(knob_h, text_2lines_h);

  // Vertical center of the track row.
  const float row_center_y = cursor.y + total_h * 0.5f;
  const float track_x0 = cursor.x + lh_box_w + spacing;
  const float spine_y0 = row_center_y - spine_h * 0.5f;
  const float spine_y1 = row_center_y + spine_h * 0.5f;
  const float spine_x0 = track_x0;
  const float spine_x1 = track_x0 + track_w;

  // Compute knob position [0,1] along the spine.
  float t = 1.0f;  // Default: pinned to right end.
  if (!locked) {
    t = static_cast<float>(current_index - hist_min) / static_cast<float>(hist_max - hist_min);
  }
  const float knob_cx = spine_x0 + t * (track_w - knob_w) + knob_w * 0.5f;

  // Invisible interaction button over the full track area.
  ImGui::SetCursorScreenPos(ImVec2(track_x0, row_center_y - total_h * 0.5f));
  ImGui::InvisibleButton("##Scrubber", ImVec2(track_w, total_h));
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();

  // Handle dragging.
  if (!locked && (active || (hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left)))) {
    const float mouse_x = ImGui::GetIO().MousePos.x;
    if (!timeline.scrubber_active) {
      timeline.scrubber_active = true;
      if (std::abs(mouse_x - knob_cx) <= knob_w) {
        timeline.scrubber_grab_offset = mouse_x - knob_cx;
      } else {
        timeline.scrubber_grab_offset = 0.0f;
      }
    }

    const float effective_mouse_x = mouse_x - timeline.scrubber_grab_offset;
    const float raw_t =
        (effective_mouse_x - spine_x0 - knob_w * 0.5f) / std::max(1.0f, track_w - knob_w);
    const float clamped_t = std::clamp(raw_t, 0.0f, 1.0f);

    int new_index = current_index;
    const float snap_eps =
        std::max(0.02f, std::min(0.05f, 8.0f / std::max(1.0f, track_w - knob_w)));
    if (clamped_t <= snap_eps || effective_mouse_x <= spine_x0 + knob_w * 0.5f + 6.0f) {
      new_index = hist_min;
    } else if (clamped_t >= 1.0f - snap_eps ||
               effective_mouse_x >= spine_x1 - knob_w * 0.5f - 6.0f) {
      new_index = hist_max;
    } else {
      const float inner_t = (clamped_t - snap_eps) / (1.0f - 2.0f * snap_eps);
      new_index = hist_min + static_cast<int>(std::round(inner_t * (hist_max - hist_min)));
    }

    if (new_index != current_index) {
      timeline.SetHistoryIndex(new_index);
      step_control.SetPauseState(StepControl::PauseState::kNormalPaused);
      current_index = timeline.history_index;
      t = static_cast<float>(current_index - hist_min) / static_cast<float>(hist_max - hist_min);
      curr_time = max_time + current_index * timestep;
      if (curr_time < 0.0) curr_time = 0.0;
      lh_top_str = FormatTimelineTime(curr_time);
      lh_top_label = lh_top_str.c_str();
      const int updated_curr_step =
          ((model != nullptr) && model->opt.timestep > 0)
              ? static_cast<int>(std::round(curr_time / model->opt.timestep))
              : 0;
      lh_bot_str = "(" + std::to_string(updated_curr_step) + ")";
      lh_bot_label = lh_bot_str.c_str();
      const float updated_lh_w = std::max(
          {base_label_w, ImGui::CalcTextSize(lh_top_label).x, ImGui::CalcTextSize(lh_bot_label).x});
      if (updated_lh_w > timeline.lh_width) {
        timeline.lh_width = updated_lh_w;
      }
    }
  } else {
    timeline.scrubber_active = false;
  }

  // Draw LH labels (two lines, both right-aligned).
  const float text_y0 = row_center_y - text_2lines_h * 0.5f;
  const float text_y1 = text_y0 + text_line_h + custom_line_spacing;

  const float lh_top_x = cursor.x + lh_box_w - ImGui::CalcTextSize(lh_top_label).x;
  ImGui::SetCursorScreenPos(ImVec2(lh_top_x, text_y0));
  ImGui::TextUnformatted(lh_top_label);
  ImGui::SetItemTooltip("%s", "Current simulation time");

  const float lh_bot_x = cursor.x + lh_box_w - ImGui::CalcTextSize(lh_bot_label).x;
  ImGui::SetCursorScreenPos(ImVec2(lh_bot_x, text_y1));
  ImGui::TextUnformatted(lh_bot_label);
  ImGui::SetItemTooltip("%s", "Current simulation step");

  // Draw RH labels (two lines, both left-aligned).
  const float rh_x0 = track_x0 + track_w + spacing;
  ImGui::SetCursorScreenPos(ImVec2(rh_x0, text_y0));
  ImGui::TextUnformatted(rh_top_label);
  ImGui::SetItemTooltip("%s", "Latest recorded time");

  ImGui::SetCursorScreenPos(ImVec2(rh_x0, text_y1));
  ImGui::TextUnformatted(rh_bot_label);
  ImGui::SetItemTooltip("%s", "Latest recorded step");

  // Restore window font scale.
  ImGui::SetWindowFontScale(1.0f);

  // Draw spine and knob on the draw list.
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImGuiStyle& style = ImGui::GetStyle();

  // Spine.
  const ImVec4 scrollbar_bg = ImGui::GetStyle().Colors[ImGuiCol_ScrollbarBg];
  ImGuiCol bg_col_idx;
  if (active) {
    bg_col_idx = ImGuiCol_FrameBgActive;
  } else if (hovered) {
    bg_col_idx = ImGuiCol_FrameBgHovered;
  } else {
    bg_col_idx = (scrollbar_bg.w > 0.01f) ? ImGuiCol_ScrollbarBg : ImGuiCol_FrameBg;
  }
  const ImU32 spine_col = ImGui::GetColorU32(bg_col_idx);
  dl->AddRectFilled(ImVec2(spine_x0, spine_y0), ImVec2(spine_x1, spine_y1), spine_col,
                    spine_h * 0.5f);

  // Knob rectangle.
  const float knob_x0 = knob_cx - knob_w * 0.5f;
  const float knob_x1 = knob_cx + knob_w * 0.5f;
  const float knob_y0 = row_center_y - knob_h * 0.5f;
  const float knob_y1 = row_center_y + knob_h * 0.5f;

  ImU32 knob_col;
  if (active) {
    knob_col = ImGui::GetColorU32(ImGuiCol_SliderGrabActive);
  } else if (hovered) {
    knob_col = ImGui::GetColorU32(ImGuiCol_SliderGrab);
  } else {
    knob_col = ImGui::GetColorU32(ImGuiCol_SliderGrab);
  }
  dl->AddRectFilled(ImVec2(knob_x0, knob_y0), ImVec2(knob_x1, knob_y1), knob_col,
                    style.GrabRounding);
  // Knob border.
  dl->AddRect(ImVec2(knob_x0, knob_y0), ImVec2(knob_x1, knob_y1),
              ImGui::GetColorU32(ImGuiCol_Border), style.GrabRounding);

  // Advance layout cursor past this row.
  ImGui::SetCursorScreenPos(ImVec2(cursor.x, cursor.y + total_h));
  ImGui::Dummy(ImVec2(0.0f, 0.0f));
}

void SimulationGui(const SimulationGuiContext& ctx) {
  const ImGuiChildFlags child_flags =
      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysAutoResize;
  const ImGuiTreeNodeFlags node_flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;

  ImGui::BeginChild("SimulationGui", {0, 0}, child_flags);
  if (SectionHeader("Simulation", node_flags | ImGuiTreeNodeFlags_DefaultOpen, 0.65f)) {
    ImGui::PushID("SimSection");

    const float slider_w =
        -ImGui::CalcTextSize(" Keyframe").x - ImGui::GetStyle().ItemInnerSpacing.x;

    bool is_dark = ImGui::GetStyle().Colors[ImGuiCol_WindowBg].x < 0.5f;
    const ImColor green = is_dark ? ImColor(40, 125, 60, 255) : ImColor(40, 180, 40, 255);
    const ImColor yellow = is_dark ? ImColor(158, 115, 18, 255) : ImColor(255, 215, 0, 255);

    // Reset / Reload / Align buttons.
    {
      char reset_label[32];
      std::snprintf(reset_label, sizeof(reset_label), "%s  Reset", ICON_FA_UNDO);
      char reload_label[32];
      std::snprintf(reload_label, sizeof(reload_label), "%s  Reload", ICON_FA_REFRESH);
      char align_label[32];
      std::snprintf(align_label, sizeof(align_label), "%s  Align", ICON_FA_CROSSHAIRS);

      const float avail = ImGui::GetContentRegionAvail().x;
      const float spacing = ImGui::GetStyle().ItemSpacing.x;
      const float btn_w = (avail - spacing * 2) / 3.0f;

      if (ImGui::Button(reset_label, ImVec2(btn_w, 0))) {
        ctx.reset();
      }
      ImGui::SameLine();
      if (ImGui::Button(reload_label, ImVec2(btn_w, 0))) {
        ctx.reload();
      }
      ImGui::SameLine();
      if (ImGui::Button(align_label, ImVec2(btn_w, 0))) {
        ctx.align();
      }
    }

    // Pause / Run toggle.
    {
      ImGui::Spacing();
      char pause_label[32];
      std::snprintf(pause_label, sizeof(pause_label), "%s  Pause", ICON_FA_PAUSE);
      char run_label[32];
      std::snprintf(run_label, sizeof(run_label), "%s  Run", ICON_FA_PLAY);

      bool paused = ctx.step_control->GetPauseState() != StepControl::PauseState::kUnpaused;
      bool running = ctx.step_control->GetPauseState() == StepControl::PauseState::kUnpaused;

      const float avail = ImGui::GetContentRegionAvail().x;
      const float half = avail * 0.5f;
      const float h = ImGui::GetFrameHeight() * 1.4f;

      ImGui::SetWindowFontScale(1.3f);
      if (ImGui_ColorButtonEx(pause_label, paused, yellow, ImDrawFlags_RoundCornersLeft,
                              ImVec2(half, h))) {
        ctx.step_control->SetPauseState(StepControl::PauseState::kNormalPaused);
      }
      ImGui::SameLine(0.f, 0.f);
      if (ImGui_ColorButtonEx(run_label, running, green, ImDrawFlags_RoundCornersRight,
                              ImVec2(half, h))) {
        ctx.step_control->SetPauseState(StepControl::PauseState::kUnpaused);
      }
      ImGui::SetWindowFontScale(1.0f);
    }

    // Speed slider.
    {
      const int max_idx = kPercentRealTime.size() - 1;
      int slider_val = max_idx - (*ctx.speed_index);
      float speed_pct = std::stof(kPercentRealTime[(*ctx.speed_index)]);

      char fmt[64];
      const float desired = ctx.step_control->GetSpeed();
      const float measured = ctx.step_control->GetSpeedMeasured();
      bool misaligned = std::abs(measured - desired) > 0.1f * desired;
      if (misaligned) {
        std::snprintf(fmt, sizeof(fmt), "%.1f%%%% (%.1f%%%%)", speed_pct, measured);
      } else {
        std::snprintf(fmt, sizeof(fmt), "%.1f%%%%", speed_pct);
      }

      ImGui::SetNextItemWidth(slider_w);
      if (ImGui::SliderInt("Speed", &slider_val, 0, max_idx, fmt)) {
        SetSpeedIndex(ctx.step_control, *ctx.speed_index, max_idx - slider_val);
      }
      if (misaligned) {
        ImGui::SetItemTooltip("%s", "Desired Speed (Measured Speed)");
      } else {
        ImGui::SetItemTooltip("%s", "Percent of real-time");
      }
    }

    // History controls (Frame Scrubber).
    if (ctx.timeline != nullptr && ctx.timeline->history_size > 0) {
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();
      char prev_label[32];
      std::snprintf(prev_label, sizeof(prev_label), "%s Step Back", ICON_FA_CARET_LEFT);
      char next_label[32];
      std::snprintf(next_label, sizeof(next_label), "%s Step Fwd", ICON_FA_CARET_RIGHT);

      const float avail = ImGui::GetContentRegionAvail().x;
      const float spacing = ImGui::GetStyle().ItemSpacing.x;
      const float btn_w = (avail - spacing) / 2.0f;

      if (ImGui::Button(prev_label, ImVec2(btn_w, 0))) {
        ctx.timeline->SetHistoryIndex(ctx.timeline->history_index - 1);
        ctx.step_control->SetPauseState(StepControl::PauseState::kNormalPaused);
      }
      ImGui::SetItemTooltip("%s", "Load previous frame from history");
      ImGui::SameLine();
      if (ImGui::Button(next_label, ImVec2(btn_w, 0))) {
        if (ctx.timeline->history_index == 0) {
          ctx.step_control->RequestSingleStep();
        } else {
          ctx.timeline->SetHistoryIndex(ctx.timeline->history_index + 1);
          ctx.step_control->SetPauseState(StepControl::PauseState::kNormalPaused);
        }
      }
      ImGui::SetItemTooltip("%s", "Load next frame from history / Single step");

      // Timeline scrubber.
      TimelineScrubberGui(ctx.model, ctx.data, *ctx.step_control, *ctx.timeline);
    }

    // Keyframe controls.
    if (ctx.model->nkey > 0) {
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();
      {
        if (*ctx.key_idx >= ctx.model->nkey || *ctx.key_idx < -1) {
          *ctx.key_idx = -1;
        }
        std::string key_name = GetKeyframeName(ctx.model, *ctx.key_idx);
        ImGui::SetNextItemWidth(slider_w);
        if (ImGui::BeginCombo("Keyframe", key_name.c_str())) {
          if (ImGui::Selectable("\xE2\x80\x94", (*ctx.key_idx == -1))) {
            *ctx.key_idx = -1;
            ctx.reset();
          }
          for (int k = 0; k < ctx.model->nkey; k++) {
            std::string item_name = GetKeyframeName(ctx.model, k);
            ImGui::PushID(k);
            if (ImGui::Selectable(item_name.c_str(), (*ctx.key_idx == k))) {
              *ctx.key_idx = k;
              ctx.reset();
            }
            ImGui::PopID();
          }
          ImGui::EndCombo();
        }
      }

      // Keyframe buttons.
      {
        char save_label[32];
        std::snprintf(save_label, sizeof(save_label), "%s Save keyframe", ICON_FA_UPLOAD);
        char copy_label[32];
        std::snprintf(copy_label, sizeof(copy_label), "%s Copy keyframe", ICON_FA_COPY);

        const float avail = ImGui::GetContentRegionAvail().x;
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float btn_w = (avail - spacing) / 2.0f;

        ImGui::BeginDisabled(*ctx.key_idx < 0 || *ctx.key_idx >= ctx.model->nkey);
        if (ImGui::Button(save_label, ImVec2(btn_w, 0))) {
          ctx.save_keyframe();
        }
        ImGui::SetItemTooltip("%s", "Save active state to selected keyframe");
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button(copy_label, ImVec2(btn_w, 0))) {
          std::string str = KeyframeToString(ctx.model, ctx.data, false);
          MaybeSaveToClipboard(str);
        }
        ImGui::SetItemTooltip("%s", "Copy active state to clipboard as MJCF XML");
      }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Thread control.
    ImGui::SetNextItemWidth(slider_w);
    ImGui::BeginDisabled(std::thread::hardware_concurrency() <= 1);
    if (ImGui::SliderInt("Threads", &(*ctx.nthread), 0, 8, "%d worker threads")) {
      (*ctx.update_threadpool) = true;
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("%s", "Number of worker threads in threadpool");
    ImGui::Spacing();

    ImGui::PopID();
    ImGui::TreePop();
  }
  ImGui::EndChild();
}

bool ThemeSelectGui(GuiTheme* theme, const ImVec2& size) {
  static constexpr const char* ICON_DARKMODE = ICON_FA_CIRCLE;
  static constexpr const char* ICON_LIGHTMODE = ICON_FA_CIRCLE_O;
  static constexpr const char* ICON_CLASSICMODE = ICON_FA_ADJUST;
  const char* theme_icons[] = {ICON_LIGHTMODE, ICON_DARKMODE, ICON_CLASSICMODE};

  int theme_idx = static_cast<int>(*theme);
  if (ImGui::Button(theme_icons[theme_idx], size)) {
    theme_idx = (theme_idx + 1) % IM_ARRAYSIZE(theme_icons);
    *theme = static_cast<GuiTheme>(theme_idx);
    return true;
  }
  ImGui::SetItemTooltip("%s", "Toggle theme");

  return false;
}

bool ThemeMenuGui(GuiTheme* theme) {
  bool changed = false;
  if (ImGui::BeginMenu("Theme")) {
    if (ImGui::MenuItem("Light", nullptr, *theme == GuiTheme::kLight)) {
      *theme = GuiTheme::kLight;
      SetupTheme(*theme);
      ImGui::GetIO().WantSaveIniSettings = true;
      changed = true;
    }
    if (ImGui::MenuItem("Dark", nullptr, *theme == GuiTheme::kDark)) {
      *theme = GuiTheme::kDark;
      SetupTheme(*theme);
      ImGui::GetIO().WantSaveIniSettings = true;
      changed = true;
    }
    if (ImGui::MenuItem("Classic", nullptr, *theme == GuiTheme::kClassic)) {
      *theme = GuiTheme::kClassic;
      SetupTheme(*theme);
      ImGui::GetIO().WantSaveIniSettings = true;
      changed = true;
    }
    ImGui::EndMenu();
  }
  return changed;
}

bool LabelSelectionGui(mjvOption* opts) {
  static constexpr const char* ICON_LABEL = ICON_FA_COMMENT;
  static constexpr const char* kLabelNames[] = {
      "None",      "Body",    "Joint",    "Geom",       "Site",  "Camera",
      "Light",     "Tendon",  "Actuator", "Constraint", "Flex",  "Skin",
      "Selection", "Sel Pnt", "Contact",  "Force",      "Island"};

  bool changed = false;
  const std::string label_preview = opts->label == 0
                                        ? std::string(ICON_LABEL) + " Label"
                                        : std::string(ICON_LABEL) + " " + kLabelNames[opts->label];
  ImGui::SetNextItemWidth(GetExpectedLabelWidth());
  if (ImGui::BeginCombo("##Label", label_preview.c_str(), ImGuiComboFlags_NoArrowButton)) {
    for (int n = 0; n < IM_ARRAYSIZE(kLabelNames); n++) {
      if (ImGui::Selectable(kLabelNames[n], (opts->label == n))) {
        changed = true;
        opts->label = n;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SetItemTooltip("%s", "Label");
  return changed;
}

bool FrameSelectionGui(mjvOption* opts) {
  static constexpr const char* ICON_FRAME = ICON_FA_ARROWS;
  static constexpr const char* kFrameNames[] = {"None",   "Body",  "Geom",    "Site",
                                                "Camera", "Light", "Contact", "World"};

  bool changed = false;
  const std::string frame_preview = opts->frame == 0
                                        ? std::string(ICON_FRAME) + " Frame"
                                        : std::string(ICON_FRAME) + " " + kFrameNames[opts->frame];
  ImGui::SetNextItemWidth(GetExpectedLabelWidth());
  if (ImGui::BeginCombo("##Frame", frame_preview.c_str(), ImGuiComboFlags_NoArrowButton)) {
    for (int n = 0; n < IM_ARRAYSIZE(kFrameNames); n++) {
      if (ImGui::Selectable(kFrameNames[n], (opts->frame == n))) {
        opts->frame = n;
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SetItemTooltip("%s", "Frame");
  return changed;
}

std::string GetKeyframeName(const mjModel* model, int index) {
  if (index < 0) {
    return "\xE2\x80\x94";
  }
  if (model && index < model->nkey) {
    if (const char* key_name = mj_id2name(model, mjOBJ_KEY, index)) {
      return key_name;
    }
  }
  return "Key " + std::to_string(index);
}

std::string GetCameraName(const mjModel* model, const mjvCamera& camera, int index) {
  static constexpr char kCameraTumbleName[] = "Free: tumble";
  static constexpr char kCameraWasdName[] = "Free: wasd";
  static constexpr char kCameraUnnamedName[] = "Unnamed";

  if (index == kTumbleCameraIdx) {
    return kCameraTumbleName;
  } else if (index == kFreeCameraIdx) {
    return kCameraWasdName;
  } else if (index == kTrackingCameraIdx) {
    return "Tracking (" + std::to_string(camera.trackbodyid) + ")";
  } else if (const char* cam_name = mj_id2name(model, mjOBJ_CAMERA, index)) {
    return cam_name;
  } else {
    return kCameraUnnamedName;
  }
}

bool CameraSelectionGui(const mjModel* model, mjData* data, mjvCamera& camera, int& index) {
  static constexpr const char* ICON_CAMERA = ICON_FA_CAMERA;
  static constexpr const char* ICON_COPY_CAMERA = ICON_FA_COPY;

  // Copy camera button.
  const float btn_size = ImGui::GetFrameHeight();
  const ImVec2 square_size(btn_size, btn_size);
  if (ImGui::Button(ICON_COPY_CAMERA, square_size)) {
    std::string camera_string = CameraToString(data, &camera);
    MaybeSaveToClipboard(camera_string);
  }
  ImGui::SetItemTooltip("%s", "Copy Camera");
  ImGui::SameLine(0, 0);

  ImGui::SetNextItemWidth(GetExpectedLabelWidth());

  auto select = [&](int type, int idx) {
    if (ImGui::Selectable(GetCameraName(model, camera, type).c_str(), (type == idx))) {
      return true;
    }
    return false;
  };

  bool changed = false;

  const std::string preview = std::string(ICON_CAMERA) + " " + GetCameraName(model, camera, index);
  if (ImGui::BeginCombo("##Camera", preview.c_str(), ImGuiComboFlags_NoArrowButton)) {
    if (select(kTumbleCameraIdx, index)) {
      index = SetCamera(model, &camera, kTumbleCameraIdx);
      changed = true;
    }
    if (select(kFreeCameraIdx, index)) {
      index = SetCamera(model, &camera, kFreeCameraIdx);
      changed = true;
    }
    if (select(kTrackingCameraIdx, index)) {
      index = SetCamera(model, &camera, kTrackingCameraIdx);
      changed = true;
    }
    for (int cam = 0; cam < model->ncam; cam++) {
      if (select(cam, index)) {
        index = SetCamera(model, &camera, cam);
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SetItemTooltip("%s", "Camera");
  return changed;
}

bool ViewGizmoGui(const mjModel* model, const mjData* data, mjvCamera& camera, int& camera_index,
                  ViewGizmoAnimation& animation, const ImVec4& workspace_rect) {
  if (model == nullptr || data == nullptr) {
    return false;
  }
  // Axis colors and labels.
  static constexpr ImU32 kAxisColors[3] = {IM_COL32(245, 63, 82, 255), IM_COL32(139, 220, 0, 255),
                                           IM_COL32(44, 143, 255, 255)};
  static constexpr const char* kAxisLabels[3] = {"X", "Y", "Z"};
  static constexpr const char* kNegativeAxisLabels[3] = {"-X", "-Y", "-Z"};
  constexpr double kAnimationSec = 0.25;
  constexpr double kDragDegPerPixel = 0.4;

  const float scale = ImGui::GetWindowDpiScale();
  const float axis_length = 40.0f * scale;
  const float bubble_radius = 9.0f * scale;
  const float half_extent = axis_length + bubble_radius;
  const float margin = 8.0f * scale;
  const ImVec2 size(2.0f * half_extent, 2.0f * half_extent);

  ImGui::SetNextWindowPos(
      ImVec2(workspace_rect.x + workspace_rect.z - margin, workspace_rect.y + margin),
      ImGuiCond_Always, ImVec2(1.0f, 0.0f));
  ImGui::SetNextWindowSize(size, ImGuiCond_Always);
  constexpr ImGuiWindowFlags kFlags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoScrollWithMouse;
  ScopedStyle style;
  style.Var(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  style.Var(ImGuiStyleVar_WindowBorderSize, 0.0f);
  style.Var(ImGuiStyleVar_WindowMinSize, ImVec2(1, 1));
  bool changed = false;
  if (!ImGui::Begin("##ViewGizmo", nullptr, kFlags)) {
    ImGui::End();
    return false;
  }

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 center(origin.x + half_extent, origin.y + half_extent);
  ImGui::InvisibleButton("##Gizmo", size);
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  const bool released = ImGui::IsItemDeactivated();

  // Project the world axes onto the view (x right, y down, depth away).
  mjtNum forward[3], up[3], right[3];
  mjv_cameraFrame(nullptr, forward, up, right, data, &camera);
  struct Bubble {
    int axis;
    int sign;
    ImVec2 pos;
    float depth;
  };
  Bubble bubbles[6];
  for (int i = 0; i < 6; ++i) {
    const int axis = i / 2;
    const int sign = (i % 2 == 0) ? 1 : -1;
    bubbles[i] = {axis, sign,
                  ImVec2(center.x + sign * axis_length * static_cast<float>(right[axis]),
                         center.y - sign * axis_length * static_cast<float>(up[axis])),
                  sign * static_cast<float>(forward[axis])};
  }
  std::sort(std::begin(bubbles), std::end(bubbles),
            [](const Bubble& a, const Bubble& b) { return a.depth > b.depth; });

  // The front-most bubble under the mouse.
  int hot = -1;
  if (hovered || active) {
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    for (int i = 5; i >= 0; --i) {
      const float dx = mouse.x - bubbles[i].pos.x;
      const float dy = mouse.y - bubbles[i].pos.y;
      if (dx * dx + dy * dy <= bubble_radius * bubble_radius) {
        hot = i;
        break;
      }
    }
  }

  const double now = ImGui::GetTime();

  // Fixed cameras ignore azimuth/elevation, so switch to the tumble camera.
  auto release_fixed_camera = [&] {
    if (camera.type == mjCAMERA_FIXED) {
      camera_index = SetCamera(model, &camera, kTumbleCameraIdx);
      mjv_defaultFreeCamera(model, &camera);
    }
  };

  // Dragging orbits the camera.
  const float drag_threshold = ImGui::GetIO().MouseDragThreshold;
  const bool dragged = ImGui::GetIO().MouseDragMaxDistanceSqr[0] > drag_threshold * drag_threshold;
  if (active && dragged) {
    const ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (delta.x != 0.0f || delta.y != 0.0f) {
      release_fixed_camera();
      camera.azimuth -= delta.x * kDragDegPerPixel;
      camera.elevation = std::clamp(camera.elevation - delta.y * kDragDegPerPixel, -90.0, 90.0);
      animation.start_time = -1.0;  // A drag cancels the animation.
      changed = true;
    }
  }

  // Clicking an axis looks along it, or from the other side if it already does.
  if (released && !dragged && hot >= 0) {
    double f[3] = {0, 0, 0};  // The new forward direction.
    f[bubbles[hot].axis] = -bubbles[hot].sign;
    if (forward[0] * f[0] + forward[1] * f[1] + forward[2] * f[2] > 0.9999) {
      f[bubbles[hot].axis] = -f[bubbles[hot].axis];
    }
    release_fixed_camera();
    const double el = std::asin(std::clamp(f[2], -1.0, 1.0)) * 180.0 / mjPI;
    // Looking straight down (up), keep +Y pointing up the screen.
    const double az =
        std::abs(f[2]) > 0.5 ? (f[2] < 0 ? 90.0 : -90.0) : std::atan2(f[1], f[0]) * 180.0 / mjPI;
    animation = {.start_time = now,
                 .start_azimuth = camera.azimuth,
                 .start_elevation = camera.elevation,
                 .target_azimuth = az,
                 .target_elevation = el};
  }

  // Animate to the target, the short way around in azimuth.
  if (animation.start_time >= 0.0) {
    const double u = std::clamp((now - animation.start_time) / kAnimationSec, 0.0, 1.0);
    const double ease = u * u * (3.0 - 2.0 * u);  // Smoothstep.
    const double daz = std::remainder(animation.target_azimuth - animation.start_azimuth, 360.0);
    camera.azimuth = animation.start_azimuth + daz * ease;
    camera.elevation =
        animation.start_elevation + (animation.target_elevation - animation.start_elevation) * ease;
    if (u >= 1.0) {
      animation.start_time = -1.0;
    }
    changed = true;
  }

  // Draw back to front, over a faint disc while hovered.
  ImDrawList* dl = ImGui::GetWindowDrawList();
  if (hovered || active) {
    dl->AddCircleFilled(center, half_extent, ImGui::GetColorU32(ImGuiCol_Text, 0.12f), 48);
  }
  const ImU32 label_color = IM_COL32(20, 20, 20, 255);
  for (int i = 0; i < 6; ++i) {
    const Bubble& b = bubbles[i];
    const ImU32 color = kAxisColors[b.axis];
    const bool is_hot = (i == hot);
    if (b.sign > 0) {
      dl->AddLine(center, b.pos, color, 2.0f * scale);
      dl->AddCircleFilled(b.pos, bubble_radius, color, 24);
      const ImVec2 text = ImGui::CalcTextSize(kAxisLabels[b.axis]);
      dl->AddText(ImVec2(b.pos.x - text.x * 0.5f, b.pos.y - text.y * 0.5f), label_color,
                  kAxisLabels[b.axis]);
    } else {
      dl->AddCircleFilled(b.pos, bubble_radius, (color & ~IM_COL32_A_MASK) | IM_COL32(0, 0, 0, 80),
                          24);
      dl->AddCircle(b.pos, bubble_radius, color, 24, 1.5f * scale);
      if (is_hot) {
        const char* label = kNegativeAxisLabels[b.axis];
        const ImVec2 text = ImGui::CalcTextSize(label);
        dl->AddText(ImVec2(b.pos.x - text.x * 0.5f, b.pos.y - text.y * 0.5f),
                    ImGui::GetColorU32(ImGuiCol_Text), label);
      }
    }
    if (is_hot) {
      dl->AddCircle(b.pos, bubble_radius + 1.0f * scale, IM_COL32_WHITE, 24, 1.5f * scale);
    }
  }
  if (hot >= 0) {
    ImGui::SetTooltip("View from %s%s", bubbles[hot].sign > 0 ? "+" : "-",
                      kAxisLabels[bubbles[hot].axis]);
  }

  ImGui::End();
  return changed;
}

void SensorGui(const mjModel* model, const mjData* data) {
  if (model->nsensor == 0) {
    return;
  }

  ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.1f, 0.1f));
  if (ImPlot::BeginPlot("Sensors", ImVec2(-1, 0), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);

    ImPlot::SetupAxis(ImAxis_X1, "sensor", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, 5, ImPlotCond_Once);

    ImPlot::SetupAxis(ImAxis_Y1, "value", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel);
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.1f");
    ImPlot::SetupAxisLimits(ImAxis_Y1, -100, 100, ImPlotCond_Once);
    ImPlot::SetupFinish();

    // The values to be plotted.
    std::vector<ImPlotPoint> sensor_values;

    // The x-value of the bar to be plotted. Multiple bars will belong to the
    // same sensor (i.e. the sensor_dim), but each group of bars will be appear
    // in sequence along the x-axis.
    float x_value = 0.f;

    // The index of the sensor being plotted, based on sensor_type.
    int sensor_index = 0;

    // Function that plots the current group of sensor bars.
    auto plot_lines = [](int sensor_idx, const ImPlotPoint* values, int count) {
      constexpr float bar_weight = 5.0f;
      std::string sensor_label = "Sensor " + std::to_string(sensor_idx);
      ImPlot::PlotLine(sensor_label.c_str(), &values->x, &values->y, count,
                       ImPlotSpec(ImPlotProp_Flags, ImPlotLineFlags_Segments, ImPlotProp_Stride,
                                  2 * sizeof(double), ImPlotProp_LineWeight, bar_weight));
    };

    for (int n = 0; n < model->nsensor; n++) {
      if (n > 0 && model->sensor_type[n] != model->sensor_type[n - 1]) {
        plot_lines(sensor_index, sensor_values.data(), sensor_values.size());
        sensor_values.clear();
        ++sensor_index;
      }

      const int adr = model->sensor_adr[n];
      const int dim = model->sensor_dim[n];
      const mjtNum cutoff = (model->sensor_cutoff[n] > 0 ? model->sensor_cutoff[n] : 1);
      for (int i = 0; i < dim; ++i) {
        sensor_values.push_back({x_value, 0});
        sensor_values.push_back({x_value, data->sensordata[adr + i] / cutoff});
        x_value += 1.f;
      }
    }

    // Ensure the last group of sensors is plotted.
    plot_lines(sensor_index, sensor_values.data(), sensor_values.size());
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleVar();
}

void StateGui(const mjModel* model, mjData* data, std::vector<mjtNum>& state, int& state_sig,
              float min_width) {
  const float available_width = GetStableAvailWidth() - ImGui::GetTreeNodeToLabelSpacing();
  const int num_cols = std::clamp(static_cast<int>(std::floor(available_width / min_width)), 1, 4);
  const ImVec2 size = GetFlexElementSize(num_cols);

  ImGui::Unindent(0.5f * ImGui::GetTreeNodeToLabelSpacing());
  // State component names and tooltips.
  static constexpr const char* name_and_tooltip[mjNSTATE][2] = {
      {"TIME", "Time"},
      {"QPOS", "Position"},
      {"QVEL", "Velocity"},
      {"ACT", "Actuator activation"},
      {"HISTORY", "History buffers (control, sensor)"},
      {"WARMSTART", "Acceleration used for warmstart"},
      {"CTRL", "Control"},
      {"QFRC_APPLIED", "Applied generalized force"},
      {"XFRC_APPLIED", "Applied Cartesian force/torque"},
      {"EQ_ACTIVE", "Enable/disable constraints"},
      {"MOCAP_POS", "Positions of mocap bodies"},
      {"MOCAP_QUAT", "Orientations of mocap bodies"},
      {"USERDATA", "User data"},
      {"PLUGIN", "Plugin state"},
      {"PLASTIC", "Plastic state"},
  };
  static_assert(
      [] {
        for (const auto& row : name_and_tooltip) {
          if (row[0] == nullptr || row[1] == nullptr) return false;
        }
        return true;
      }(),
      "name_and_tooltip is missing a row for an mjtState component");
  static_assert(mjSTATE_INTEGRATION == (1 << mjNSTATE) - 1,
                "mjNSTATE does not match the number of mjtState components");

  int prev_state_sig = state_sig;

  // State component checkboxes.
  if (ImGui::BeginTable("##StateSignature", num_cols)) {
    for (int i = 0; i < mjNSTATE; ++i) {
      ImGui::TableNextColumn();
      bool checked = state_sig & (1 << i);
      ImGui::Checkbox(name_and_tooltip[i][0], &checked);
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", name_and_tooltip[i][1]);
      }
      state_sig = checked ? (state_sig | (1 << i)) : (state_sig & ~(1 << i));
    }
    ImGui::EndTable();
  }

  // Buttons to select commonly used state signatures.
  if (ImGui::BeginTable("##CommonSignatures", num_cols)) {
    ImGui::TableNextColumn();
    if (ImGui::Button("Physics", size)) {
      state_sig = (state_sig == mjSTATE_PHYSICS) ? 0 : mjSTATE_PHYSICS;
    }
    ImGui::TableNextColumn();
    if (ImGui::Button("Full Physics", size)) {
      state_sig = (state_sig == mjSTATE_FULLPHYSICS) ? 0 : mjSTATE_FULLPHYSICS;
    }
    ImGui::TableNextColumn();
    if (ImGui::Button("User", size)) {
      state_sig = (state_sig == mjSTATE_USER) ? 0 : mjSTATE_USER;
    }
    ImGui::TableNextColumn();
    if (ImGui::Button("Integration", size)) {
      state_sig = (state_sig == mjSTATE_INTEGRATION) ? 0 : mjSTATE_INTEGRATION;
    }
    ImGui::EndTable();
  }

  if (state_sig != prev_state_sig) {
    const int size = mj_stateSize(model, state_sig);
    state.resize(size);
  }

  if (state.empty()) {
    // The state size is 0, let the user know why.
    ImGui::Separator();
    ImGui::BeginDisabled();
    ImGui::TextWrapped(state_sig == 0 ? "No state components are selected."
                                      : "Selected state components do not exist in the model.");
    ImGui::EndDisabled();
  } else {
    mj_getState(model, data, state.data(), state_sig);
    bool changed = false;

    if (ImGui::BeginTable("State", 3,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter |
                              ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY,
                          ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 20))) {
      ImGui::TableSetupColumn("Index");
      ImGui::TableSetupColumn("Name");
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableHeadersRow();

      ImGuiListClipper clipper;
      clipper.Begin(state.size());
      while (clipper.Step()) {
        int global = 0;
        for (int i = 0; i < mjNSTATE; ++i) {
          if (state_sig & (1 << i)) {
            for (int local = 0; local < mj_stateSize(model, (1 << i)); ++local, ++global) {
              if (global < clipper.DisplayStart) {
                continue;
              }
              if (global >= clipper.DisplayEnd) {
                break;
              }
              ImGui::TableNextRow();

              ImGui::TableNextColumn();
              ImGui::Text("%d", global);

              ImGui::TableNextColumn();
              ImGui::Text("%s[%d]", name_and_tooltip[i][0], local);

              ImGui::TableNextColumn();
              float value = state[global];
              ImGui::PushItemWidth(-std::numeric_limits<float>::min());
              ImGui::PushID(global);
              if (ImGui::DragFloat("##value", &value, 0.01f, 0, 0, "%.3f")) {
                changed = true;
              }
              ImGui::PopID();
              ImGui::PopItemWidth();
              state[global] = value;
            }
          }
        }
      }
      ImGui::EndTable();
    }

    if (changed) {
      mj_setState(model, data, state.data(), state_sig);
    }
  }

  ImGui::Indent(0.5f * ImGui::GetTreeNodeToLabelSpacing());
}

void WatchGui(const mjModel* model, const mjData* data, char* field_name, int field_len,
              int& field_index) {
  const float item_width = ImGui::GetWindowWidth() * .6f;
  ImGui::PushItemWidth(item_width);

  ImGui::InputText("Field", field_name, field_len);
  ImGui::InputInt("Index", &field_index);
  const mjtNum* value = static_cast<const mjtNum*>(GetValue(model, data, field_name, field_index));

  ScopedStyle style;
  style.Color(ImGuiCol_FrameBg, ImGui::GetStyle().Colors[ImGuiCol_WindowBg]);

  if (value) {
    char buf[100];
    int size = std::snprintf(buf, sizeof(buf), "%0.3f", *value);
    ImGui::InputText("Value", buf, size, ImGuiInputTextFlags_ReadOnly);
  } else {
    ImGui::BeginDisabled();
    style.Color(ImGuiCol_Text, ImColor(255, 0, 0, 255));
    char buf[] = "Invalid field/index!";
    ImGui::InputText("Value", buf, sizeof(buf), ImGuiInputTextFlags_ReadOnly);
    ImGui::EndDisabled();
  }

  ImGui::PopItemWidth();
}

void PhysicsGui(mjModel* model, mjSpec* spec, float min_width) {
  if (!model && !spec) {
    return;
  }

  const float available_width = GetStableAvailWidth() - ImGui::GetTreeNodeToLabelSpacing();
  const int num_cols = std::clamp(static_cast<int>(std::floor(available_width / min_width)), 1, 6);

  const float item_width = ImGui::GetWindowWidth() * .6f;
  ImGui::PushItemWidth(item_width);

  auto& opt = spec ? spec->option : model->opt;

  const char* opts0[] = {"Euler", "RK4", "implicit", "implicitfast", "discrete"};
  ImGui::Combo("Integrator", &opt.integrator, opts0, IM_ARRAYSIZE(opts0));

  const char* opts1[] = {"Pyramidal", "Elliptic"};
  ImGui::Combo("Cone", &opt.cone, opts1, IM_ARRAYSIZE(opts1));

  const char* opts2[] = {"Dense", "Sparse", "Auto"};
  ImGui::Combo("Jacobian", &opt.jacobian, opts2, IM_ARRAYSIZE(opts2));

  const char* opts3[] = {"PGS", "CG", "Newton"};
  ImGui::Combo("Solver", &opt.solver, opts3, IM_ARRAYSIZE(opts3));

  if (SectionHeader("Algorithmic Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui_LogStepper("Timestep", &opt.timestep, {0, 1});
    ImGui_Input("Iterations", &opt.iterations, {0, 1000, 1, 10});
    ImGui_LogStepper("Tolerance", &opt.tolerance, {.min = 0.0, .max = 1.0, .zero_below = 1e-10});
    ImGui_Input("LS Iter", &opt.ls_iterations, {0, 100, 1, 0.1});
    ImGui_LogStepper("LS Tol", &opt.ls_tolerance, {.min = 0.0, .max = 0.1, .zero_below = 1e-4});
    ImGui_Input("Noslip Iter", &opt.noslip_iterations, {0, 1000, 1, 100});
    ImGui_LogStepper("Noslip Tol", &opt.noslip_tolerance,
                     {.min = 0.0, .max = 1.0, .zero_below = 1e-8});
    ImGui_Input("CCD Iter", &opt.ccd_iterations, {0, 1000, 1, 100});
    ImGui_LogStepper("CCD Tol", &opt.ccd_tolerance, {.min = 0.0, .max = 1.0, .zero_below = 1e-8});
    ImGui_LogStepper("Sleep Tol", &opt.sleep_tolerance,
                     {.min = 0.0, .max = 1.0, .zero_below = 1e-6});
    ImGui_Input("SDF Iter", &opt.sdf_iterations, {1, 20, 1, 10});
    ImGui_Input("SDF Init", &opt.sdf_initpoints, {1, 100, 1, 10});
    ImGui::TreePop();
  }

  if (SectionHeader("Physical Parameters")) {
    // Viscous posing mode toggle: disables gravity and passive springs, adds
    // viscosity. All changes are directly visible in the parameters below.
    {
      constexpr mjtNum kPosingViscosity = 10;
      bool active = (opt.disableflags & mjDSBL_GRAVITY) && (opt.disableflags & mjDSBL_SPRING) &&
                    (opt.viscosity >= kPosingViscosity);
      if (ImGui_ButtonToggle("Viscous posing mode", &active)) {
        if (active) {
          opt.disableflags |= mjDSBL_GRAVITY;
          opt.disableflags |= mjDSBL_SPRING;
          opt.viscosity += kPosingViscosity;
        } else {
          opt.disableflags &= ~mjDSBL_GRAVITY;
          opt.disableflags &= ~mjDSBL_SPRING;
          opt.viscosity = std::max<mjtNum>(0, opt.viscosity - kPosingViscosity);
        }
      }
      ImGui::SetItemTooltip(
          "Disable gravity and passive springs,\n"
          "add viscosity for easier posing.");
    }
    ImGui::Spacing();

    ImGui_InputN("Gravity", opt.gravity, 3);
    ImGui_InputN("Wind", opt.wind, 3);
    ImGui_InputN("Magnetic", opt.magnetic, 3);
    ImGui_Input("Density", &opt.density, {.min = 0.0});
    ImGui_Input("Viscosity", &opt.viscosity, {.min = 0.0});
    ImGui_LogStepper("Imp Ratio", &opt.impratio, {.min = 0.0});
    ImGui::TreePop();
  };

  if (SectionHeader("Flags", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::BeginTable("##PhysicsFlagsTable", num_cols)) {
      const ImVec2 size = GetFlexElementSize(num_cols);
      for (int i = 0; i < mjNDISABLE; ++i) {
        ImGui::TableNextColumn();
        int flipped = ~opt.disableflags;
        ImGui_BitToggle(mjDISABLESTRING[i], &flipped, 1 << i, size);
        opt.disableflags = ~flipped;
      }
      for (int i = 0; i < mjNENABLE; ++i) {
        ImGui::TableNextColumn();
        ImGui_BitToggle(mjENABLESTRING[i], &opt.enableflags, 1 << i, size);
      }
      ImGui::EndTable();
    }
    ImGui::TreePop();
  }

  if (SectionHeader("Contact Override")) {
    ImGui_Input("Margin", &opt.o_margin, {.min = 0.0});
    ImGui_InputN("Sol Imp", opt.o_solimp, 5, {.format = "%0.3f"});
    ImGui_InputN("Sol Ref", opt.o_solref, 2, {.format = "%0.3f"});
    ImGui_InputN("Friction", opt.o_friction, 5, {.format = "%.3f"});
    ImGui::TreePop();
  }

  if (SectionHeader("Actuator Groups")) {
    if (ImGui::BeginTable("##ActuatorGroupsTable", num_cols)) {
      const ImVec2 size = GetFlexElementSize(num_cols);
      for (int i = 0; i < 6; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "Act Group %d", i);
        ImGui::TableNextColumn();
        int flipped = ~opt.disableactuator;
        ImGui_BitToggle(label, &flipped, 1 << i, size);
        opt.disableactuator = ~flipped;
      }
      ImGui::EndTable();
    }
    ImGui::TreePop();
  }

  ImGui::PopItemWidth();

  if (spec && model) {
    model->opt = spec->option;
  }
}

void VisualizationGui(mjModel* model, mjvOption* vis_options, mjvCamera* camera, float min_width) {
  auto& vis = model->vis;
  auto& stat = model->stat;

  const float item_width = ImGui::GetWindowWidth() * .6f;
  ImGui::PushItemWidth(item_width);

  ImGui::SliderInt("Tree depth", &vis_options->bvh_depth, 0, 20);
  ImGui::SliderInt("Flex layer", &vis_options->flex_layer, 0, 10);

  if (SectionHeader("Headlight")) {
    const char* opts[] = {"Off", "On", "Auto"};
    ImGui::Combo("Active", &vis.headlight.active, opts, IM_ARRAYSIZE(opts));
    ImGui::ColorEdit3("Ambient", vis.headlight.ambient);
    ImGui::ColorEdit3("Diffuse", vis.headlight.diffuse);
    ImGui::ColorEdit3("Specular", vis.headlight.specular);
    ImGui::TreePop();
  }
  if (SectionHeader("Free Camera")) {
    if (ImGui_SwitchToggle("Orthographic", &camera->orthographic)) {
      vis.global.orthographic = camera->orthographic;
    }
    ImGui_Input("FOV", &vis.global.fovy, {.format = "%0.2f"});
    ImGui_InputN("Center", stat.center, 3, {.format = "%0.2f"});
    ImGui_Input("Azimuth", &vis.global.azimuth, {.format = "%0.2f"});
    ImGui_Input("Elevation", &vis.global.elevation, {.format = "%0.2f"});
    if (ImGui::Button("Align")) {
      mjv_defaultFreeCamera(model, camera);
    }
    ImGui::TreePop();
  }
  if (SectionHeader("Global")) {
    ImGui_Input("Extent", &stat.extent);
    const char* opts[] = {"Box", "Ellipsoid"};
    ImGui::SliderInt("Inertia", &vis.global.ellipsoidinertia, 0, 1,
                     opts[vis.global.ellipsoidinertia]);
    ImGui_ButtonToggle("BVH active", &vis.global.bvactive);
    ImGui::TreePop();
  }
  if (SectionHeader("Quality")) {
    ImGui_LogStepper("Shadow size", &vis.quality.shadowsize, {.min = 8});
    ImGui_Input("Offsamples", &vis.quality.offsamples, {.min = 0});
    ImGui_Input("Slices", &vis.quality.numslices, {.min = 1});
    ImGui_Input("Stacks", &vis.quality.numstacks, {.min = 1});
    ImGui_Input("Quads", &vis.quality.numquads, {.min = 1});
    ImGui::TreePop();
  }
  if (SectionHeader("Mapping")) {
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
    ImGui_Input("Stiffness", &vis.map.stiffness);
    ImGui_Input("Rot stiffness", &vis.map.stiffnessrot);
    ImGui_Input("Force", &vis.map.force);
    ImGui_Input("Torque", &vis.map.torque);
    ImGui_Input("Alpha", &vis.map.alpha);
    ImGui_Input("Fog start", &vis.map.fogstart);
    ImGui_Input("Fog end", &vis.map.fogend);
    ImGui_Input("Z near", &vis.map.znear);
    ImGui_Input("Z far", &vis.map.zfar);
    ImGui_Input("Haze", &vis.map.haze);
    ImGui_Input("Shadow clip", &vis.map.shadowclip);
    ImGui_Input("Shadow scale", &vis.map.shadowscale);
    ImGui::PopItemWidth();
    ImGui::TreePop();
  }
  if (SectionHeader("Scale")) {
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
    ImGui_Input("All (meansize)", &stat.meansize, {.format = "%0.3f"});
    ImGui_Input("Force width", &vis.scale.forcewidth);
    ImGui_Input("Contact width", &vis.scale.contactwidth);
    ImGui_Input("Contact height", &vis.scale.contactheight);
    ImGui_Input("Connect", &vis.scale.connect);
    ImGui_Input("Com", &vis.scale.com);
    ImGui_Input("Camera", &vis.scale.camera);
    ImGui_Input("Light", &vis.scale.light);
    ImGui_Input("Select point", &vis.scale.selectpoint);
    ImGui_Input("Joint length", &vis.scale.jointlength);
    ImGui_Input("Joint width", &vis.scale.jointwidth);
    ImGui_Input("Actuator length", &vis.scale.actuatorlength);
    ImGui_Input("Actuator width", &vis.scale.actuatorwidth);
    ImGui_Input("Frame length", &vis.scale.framelength);
    ImGui_Input("Frame width", &vis.scale.framewidth);
    ImGui_Input("Constraint", &vis.scale.constraint);
    ImGui_Input("Slider-crank", &vis.scale.slidercrank);
    ImGui::PopItemWidth();
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Colors")) {
    ImGui::ColorEdit4("Fog", vis.rgba.fog);
    ImGui::ColorEdit4("Haze", vis.rgba.haze);
    ImGui::ColorEdit4("Force", vis.rgba.force);
    ImGui::ColorEdit4("Inertia", vis.rgba.inertia);
    ImGui::ColorEdit4("Joint", vis.rgba.joint);
    ImGui::ColorEdit4("Actuator", vis.rgba.actuator);
    ImGui::ColorEdit4("Act. Negative", vis.rgba.actuatornegative);
    ImGui::ColorEdit4("Act. Positive", vis.rgba.actuatorpositive);
    ImGui::ColorEdit4("Center of Mass", vis.rgba.com);
    ImGui::ColorEdit4("Camera", vis.rgba.camera);
    ImGui::ColorEdit4("Light", vis.rgba.light);
    ImGui::ColorEdit4("Select Point", vis.rgba.selectpoint);
    ImGui::ColorEdit4("Auto Connect", vis.rgba.connect);
    ImGui::ColorEdit4("Contact Point", vis.rgba.contactpoint);
    ImGui::ColorEdit4("Contact Force", vis.rgba.contactforce);
    ImGui::ColorEdit4("Contact Friction", vis.rgba.contactfriction);
    ImGui::ColorEdit4("Contact Torque", vis.rgba.contacttorque);
    ImGui::ColorEdit4("Contact Gap", vis.rgba.contactgap);
    ImGui::ColorEdit4("Range Finder", vis.rgba.rangefinder);
    ImGui::ColorEdit4("Constraint", vis.rgba.constraint);
    ImGui::ColorEdit4("Slider Crank", vis.rgba.slidercrank);
    ImGui::ColorEdit4("Crank Broken", vis.rgba.crankbroken);
    ImGui::ColorEdit4("Frustum", vis.rgba.frustum);
    ImGui::ColorEdit4("Bounding Vol.", vis.rgba.bv);
    ImGui::ColorEdit4("BV Active", vis.rgba.bvactive);
    ImGui::TreePop();
  }

  ImGui::PopItemWidth();
}

void RenderingGui(const mjModel* model, mjvOption* vis_options, mjtByte* render_flags,
                  float min_width) {
  const float available_width = GetStableAvailWidth() - ImGui::GetTreeNodeToLabelSpacing();
  const int num_cols = std::clamp(static_cast<int>(std::floor(available_width / min_width)), 1, 6);

  if (ImGui::TreeNodeEx("Model Elements", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::BeginTable("##ModelElementsTable", num_cols)) {
      const ImVec2 size = GetFlexElementSize(num_cols);
      for (int i = 0; i < mjNVISFLAG; ++i) {
        ImGui::TableNextColumn();
        ImGui_ButtonToggle(mjVISSTRING[i][0], &vis_options->flags[i], size);
      }
      ImGui::EndTable();
    }
    ImGui::TreePop();
  }

  if (ImGui::TreeNodeEx("Render Flags", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::BeginTable("##RenderFlagsTable", num_cols)) {
      const ImVec2 size = GetFlexElementSize(num_cols);
      for (int i = 0; i < mjNRNDFLAG; ++i) {
        ImGui::TableNextColumn();
        ImGui_ButtonToggle(mjRNDSTRING[i][0], &render_flags[i], size);
      }
      ImGui::EndTable();
    }
    ImGui::TreePop();
  }
}

void GroupsGui(const mjModel* model, mjvOption* vis_options, float min_width) {
  const float available_width = GetStableAvailWidth();
  // We limit the number of columns to 1, 2, 3, or 6 depending on how much
  // space the window has available.
  int num_cols = std::clamp(static_cast<int>(std::floor(available_width / min_width)), 1, 6);
  if (num_cols == 4 || num_cols == 5) {
    num_cols = 3;
  }

  auto GroupGui = [&](const char* name, mjtByte* group) {
    if (ImGui::TreeNodeEx(name, ImGuiTreeNodeFlags_DefaultOpen)) {
      char label[64];
      std::snprintf(label, sizeof(label), "##%s", name);
      if (ImGui::BeginTable(label, num_cols)) {
        const ImVec2 size = GetFlexElementSize(num_cols);
        for (int i = 0; i < 6; ++i) {
          ImGui::TableNextColumn();
          std::snprintf(label, sizeof(label), "%s %d", name, i);
          ImGui_ButtonToggle(label, &group[i], size);
        }

        ImGui::EndTable();
      }
      ImGui::TreePop();
    }
  };

  GroupGui("Geoms", vis_options->geomgroup);
  GroupGui("Sites", vis_options->sitegroup);
  GroupGui("Joints", vis_options->jointgroup);
  GroupGui("Tendons", vis_options->tendongroup);
  GroupGui("Actuators", vis_options->actuatorgroup);
  GroupGui("Flexes", vis_options->flexgroup);
  GroupGui("Skins", vis_options->skingroup);
}

void NoiseGui(StepControl* step_control) {
  float noise_scale, noise_rate;
  step_control->GetNoiseParameters(noise_scale, noise_rate);
  const float item_width = ImGui::GetWindowWidth() * .6f;
  ImGui::PushItemWidth(item_width);
  ImGui::SliderFloat("Noise scale", &noise_scale, 0, 1);
  ImGui::SliderFloat("Noise rate", &noise_rate, 0, 4);
  ImGui::PopItemWidth();
  step_control->SetNoiseParameters(noise_scale, noise_rate);
}

// Slider row with a clipped label and a right-aligned reset button; the
// button is disabled while the value is at its reset target.
static void SliderRowWithReset(const char* name, mjtNum* value, mjtNum min, mjtNum max,
                               mjtNum reset) {
  const float btn = ImGui::GetFrameHeight();
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  const float usable_end = ImGui::GetWindowContentRegionMax().x - btn - spacing;
  const float usable = usable_end - ImGui::GetCursorPosX();
  ImGui::SetNextItemWidth(usable * 0.5f);
  char hidden[104];
  std::snprintf(hidden, sizeof(hidden), "##%s", name);
  ImGui_Slider(hidden, value, min, max);
  ImGui::SameLine();
  const ImVec2 clip_min = ImGui::GetCursorScreenPos();
  const ImVec2 clip_max(ImGui::GetWindowPos().x + usable_end, clip_min.y + ImGui::GetFrameHeight());
  ImGui::PushClipRect(clip_min, clip_max, true);
  ImGui::TextUnformatted(name);
  ImGui::PopClipRect();
  ImGui::BeginDisabled(*value == reset);
  if (ImGui_ResetButton(name)) {
    *value = reset;
  }
  ImGui::EndDisabled();
}

void JointsGui(const mjModel* model, const mjData* data, const mjvOption* vis_options) {
  char name[100];
  for (int i = 0; i < model->njnt; ++i) {
    if (model->jnt_type[i] != mjJNT_HINGE && model->jnt_type[i] != mjJNT_SLIDE) {
      continue;
    }
    const int group = std::clamp(model->jnt_group[i], 0, mjNGROUP - 1);
    if (!vis_options->jointgroup[group]) {
      continue;
    }

    const char* jnt_name = mj_id2name(model, mjOBJ_JOINT, i);
    if (jnt_name) {
      std::snprintf(name, sizeof(name), "%s", jnt_name);
    } else {
      std::snprintf(name, sizeof(name), "joint %d", i);
    }

    double min = -1.0;
    double max = 1.0;
    if (model->jnt_limited[i]) {
      min = model->jnt_range[2 * i + 0];
      max = model->jnt_range[2 * i + 1];
    } else if (model->jnt_type[i] == mjJNT_SLIDE) {
      min = -1.0;
      max = 1.0;
    } else {
      min = -3.1416;
      max = 3.1416;
    }

    const int data_adr = model->jnt_qposadr[i];
    SliderRowWithReset(name, &data->qpos[data_adr], min, max, model->qpos0[data_adr]);
  }
}

void ControlsGui(const mjModel* model, mjData* data, const mjvOption* vis_options) {
  if (ImGui::Button("Clear All")) {
    mj_resetCtrl(model, data);
  }

  char name[100];
  for (int i = 0; i < model->nactuator; i++) {
    int group = std::clamp(model->actuator_group[i], 0, mjNGROUP - 1);
    if (!vis_options->actuatorgroup[group]) {
      continue;
    }
    if (group >= 0 && group <= 30 && model->opt.disableactuator & (1 << group)) {
      continue;
    }

    // one slider per control; multi-input actuators suffix the input name
    const char* act_name = mj_id2name(model, mjOBJ_ACTUATOR, i);
    int ctrlnum = model->actuator_ctrlnum[i];
    for (int k = 0; k < ctrlnum; k++) {
      int j = model->actuator_ctrladr[i] + k;
      if (act_name && ctrlnum > 1) {
        const char* input_name = mj_actuatorInputName(model, i, k);
        if (input_name) {
          std::snprintf(name, sizeof(name), "%s/%s", act_name, input_name);
        } else {
          std::snprintf(name, sizeof(name), "%s/%d", act_name, k);
        }
      } else if (act_name) {
        std::snprintf(name, sizeof(name), "%s", act_name);
      } else {
        std::snprintf(name, sizeof(name), "control %d", j);
      }

      double min = -1.0;
      double max = 1.0;
      // a defined ctrlrange sets the slider range, even when ctrl is not
      // clamped
      if (model->actuator_ctrlrange[2 * j] < model->actuator_ctrlrange[2 * j + 1]) {
        min = model->actuator_ctrlrange[2 * j + 0];
        max = model->actuator_ctrlrange[2 * j + 1];
      }
      SliderRowWithReset(name, &data->ctrl[j], min, max, mju_clip(0.0, min, max));
    }
  }
}

static int GetPlotXLimit(const mjData* data) {
  int max_niter = 0;
  const int nisland0 = data->nefc ? mjMAX(1, mjMIN(data->nisland, mjNISLAND)) : 0;
  for (int k = 0; k < nisland0; k++) {
    max_niter = mjMAX(max_niter, data->solver_niter[k]);
  }
  return max_niter <= 10 ? 10 : ((max_niter + 59) / 60) * 60;
}

void ConvergenceGui(const mjModel* model, mjData* data, ImVec2 plot_size) {
  ScopedStyle style;
  style.Font(ScopedFont::kMono);

  int xlim = GetPlotXLimit(data);
  ImPlotFlags flags = ImPlot_SetupPlotFlags(plot_size) | ImPlotFlags_NoMouseText;
  if (ImPlot::BeginPlot("Convergence vs iter", plot_size, flags)) {
    ImPlot::SetupAxis(ImAxis_X1, "", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, xlim, ImPlotCond_Always);
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.0e");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 1e-15, 1e0, ImPlotCond_Always);
    ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    const double ticks[] = {1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0};
    ImPlot::SetupAxisTicks(ImAxis_Y1, ticks, 6);
    ImPlot::SetupLegend(ImPlotLocation_NorthEast);
    ImPlot::SetupFinish();
    ImPlotSpec spec(ImPlotProp_LineWeight, 2.0f);

    const int nisland = data->nefc ? mjMAX(1, mjMIN(data->nisland, mjNISLAND)) : 0;
    for (int k = 0; k < nisland; k++) {
      mjSolverStat* stats = data->solver + k * mjNSOLVER;
      const int npoints = mjMIN(mjMIN(data->solver_niter[k], mjNSOLVER), mjMAXLINEPNT);

      ImPlot::PlotLineG(
          "improvement",
          +[](int i, void* user_data) {
            const mjSolverStat* stats = static_cast<const mjSolverStat*>(user_data);
            const float x = static_cast<float>(i);
            const float y = mju_max(mjMINVAL, stats[i].improvement);
            return ImPlotPoint{x, y};
          },
          stats, npoints, spec);

      if (model->opt.solver == mjSOL_PGS) {
        continue;
      }

      ImPlot::PlotLineG(
          "gradient",
          +[](int i, void* user_data) {
            const mjSolverStat* stats = static_cast<const mjSolverStat*>(user_data);
            const float x = static_cast<float>(i);
            const float y = mju_max(mjMINVAL, stats[i].gradient);
            return ImPlotPoint{x, y};
          },
          stats, npoints, spec);

      ImPlot::PlotLineG(
          "lineslope",
          +[](int i, void* user_data) {
            const mjSolverStat* stats = static_cast<const mjSolverStat*>(user_data);
            const float x = static_cast<float>(i);
            const float y = mju_max(mjMINVAL, stats[i].lineslope);
            return ImPlotPoint{x, y};
          },
          stats, npoints, spec);
    }

    ImPlot::EndPlot();
  }
}

void CountsGui(const mjModel* model, mjData* data, ImVec2 plot_size) {
  ScopedStyle style;
  style.Font(ScopedFont::kMono);

  int xlim = GetPlotXLimit(data);
  ImPlotFlags flags = ImPlot_SetupPlotFlags(plot_size) | ImPlotFlags_NoMouseText;
  if (ImPlot::BeginPlot("Counts vs iter", plot_size, flags)) {
    ImPlot::SetupAxis(ImAxis_X1, "", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, xlim, ImPlotCond_Always);
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.0f");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 80, ImPlotCond_Always);
    ImPlot::SetupLegend(ImPlotLocation_NorthEast);
    ImPlot::SetupFinish();
    ImPlotSpec spec(ImPlotProp_LineWeight, 2.0f);

    const int nisland = data->nefc ? mjMAX(1, mjMIN(data->nisland, mjNISLAND)) : 0;
    for (int k = 0; k < nisland; k++) {
      const int npoints = mjMIN(mjMIN(data->solver_niter[k], mjNSOLVER), mjMAXLINEPNT);

      mjSolverStat* stats = data->solver + k * mjNSOLVER;

      int nefc = nisland == 1 ? data->nefc : data->island_nefc[k];

      ImPlot::PlotLineG(
          "total",
          +[](int i, void* user_data) {
            const float x = static_cast<float>(i);
            const float y = *(static_cast<int*>(user_data));
            return ImPlotPoint{x, y};
          },
          &nefc, npoints, spec);

      ImPlot::PlotLineG(
          "active",
          +[](int i, void* user_data) {
            const mjSolverStat* stats = static_cast<const mjSolverStat*>(user_data);
            const float x = static_cast<float>(i);
            const float y = stats[i].nactive;
            return ImPlotPoint{x, y};
          },
          stats, npoints, spec);

      ImPlot::PlotLineG(
          "changed",
          +[](int i, void* user_data) {
            const mjSolverStat* stats = static_cast<const mjSolverStat*>(user_data);
            const float x = static_cast<float>(i);
            const float y = stats[i].nchange;
            return ImPlotPoint{x, y};
          },
          stats, npoints, spec);

      if (model->opt.solver == mjSOL_PGS) {
        continue;
      }

      ImPlot::PlotLineG(
          "evals",
          +[](int i, void* user_data) {
            const mjSolverStat* stats = static_cast<const mjSolverStat*>(user_data);
            const float x = static_cast<float>(i);
            const float y = stats[i].neval;
            return ImPlotPoint{x, y};
          },
          stats, npoints, spec);

      if (model->opt.solver == mjSOL_CG) {
        continue;
      }

      ImPlot::PlotLineG(
          "updates",
          +[](int i, void* user_data) {
            const mjSolverStat* stats = static_cast<const mjSolverStat*>(user_data);
            const float x = static_cast<float>(i);
            const float y = stats[i].nupdate;
            return ImPlotPoint{x, y};
          },
          stats, npoints, spec);
    }

    ImPlot::EndPlot();
  }
}

void InfoGui(const mjModel* model, const mjData* data, bool paused, float fps, float gpu_ms,
             const mjTimerStat* timer_prev) {
  const int num_islands = std::clamp(data->nisland, 1, mjNISLAND);

  // compute solver error (maximum over islands)
  mjtNum solver_err = 0;
  int solver_iter = 0;
  for (int i = 0; i < num_islands; i++) {
    solver_iter += data->solver_niter[i];

    mjtNum solerr_i = 0;
    if (data->solver_niter[i]) {
      const int ind = mjMIN(data->solver_niter[i], mjNSOLVER) - 1;
      const mjSolverStat* stat = data->solver + i * mjNSOLVER + ind;
      solerr_i = mju_min(stat->improvement, stat->gradient);
      if (solerr_i == 0) {
        solerr_i = mju_max(stat->improvement, stat->gradient);
      }
    }
    solver_err = mju_max(solver_err, solerr_i);
  }
  solver_err = mju_log10(mju_max(mjMINVAL, solver_err));

  auto type = paused ? mjTIMER_FORWARD : mjTIMER_STEP;
  mjtNum duration = data->timer[type].duration;
  int number = data->timer[type].number;
  if (timer_prev) {
    duration -= timer_prev[type].duration;
    number -= timer_prev[type].number;
  }
  auto cpu = duration / mjMAX(1, number);
  auto mempct = 100 * data->maxuse_arena / (double)(data->narena);
  auto memlimit = mju_writeNumBytes(data->narena);

  ImGui::Columns(2);
  ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.4f);
  ImGui::SetColumnWidth(1, ImGui::GetWindowWidth() * 0.6f);

  ImGui::Text("Time");
  ImGui::Text("Size");
  ImGui::Text("CPU");
  ImGui::Text("Solver");
  ImGui::Text("FPS");
  ImGui::Text("GPU");
  ImGui::Text("Memory");
  if (model->opt.enableflags & mjENBL_ENERGY) {
    ImGui::Text("Energy");
  }
  if (model->opt.enableflags & mjENBL_FWDINV) {
    ImGui::Text("FwdInv");
  }
  if (!(model->opt.disableflags & mjDSBL_ISLAND)) {
    ImGui::Text("Islands");
  }

  ImGui::NextColumn();
  ImGui::Text("%-9.3f", data->time);
  ImGui::Text("%d (%d con)", data->nefc, data->ncon);
  ImGui::Text("%.3f", cpu);
  ImGui::Text("%.1f (%d it)", solver_err, solver_iter);
  ImGui::Text("%0.1f", fps);
  ImGui::Text("%.2f ms", gpu_ms);
  ImGui::Text("%.1f%% of %s", mempct, memlimit);
  if (model->opt.enableflags & mjENBL_ENERGY) {
    ImGui::Text("%.3f", data->energy[0] + data->energy[1]);
  }
  if (model->opt.enableflags & mjENBL_FWDINV) {
    ImGui::Text("%.1f %.1f", mju_log10(mju_max(mjMINVAL, data->solver_fwdinv[0])),
                mju_log10(mju_max(mjMINVAL, data->solver_fwdinv[1])));
  }
  if (!(model->opt.disableflags & mjDSBL_ISLAND)) {
    ImGui::Text("%d", data->nisland);
  }
  ImGui::Columns();
}

void ProfilerGui(const mjModel* model, mjData* data, SimProfiler* profiler, bool show_iter) {
  ImVec2 avail = ImGui::GetContentRegionAvail();
  const float pad = ImGui::GetStyle().ItemSpacing.x;
  const float aspect = avail.y > 0 ? avail.x / avail.y : 1.0f;

  ImVec2 plot_size;
  int cols;

  int current_col = 0;
  auto advance = [&]() {
    current_col++;
    if (current_col < cols) {
      ImGui::SameLine();
    } else {
      current_col = 0;
    }
  };

  if (!show_iter) {
    if (aspect < 0.8f) {
      plot_size.x = avail.x;
      plot_size.y = (avail.y - pad) * 0.5f;
      cols = 1;
    } else {
      plot_size.x = (avail.x - pad) * 0.5f;
      plot_size.y = avail.y;
      cols = 2;
    }
    profiler->DimensionsGraph(plot_size);
    advance();
    profiler->CpuTimeGraph(plot_size);
  } else {
    if (aspect < 0.8f) {
      plot_size.x = avail.x;
      plot_size.y = (avail.y - pad * 3.0f) * 0.25f;
      cols = 1;
    } else if (aspect < 1.8f) {
      plot_size.x = (avail.x - pad) * 0.5f;
      plot_size.y = (avail.y - pad) * 0.5f;
      cols = 2;
    } else {
      plot_size.x = (avail.x - pad * 3.0f) * 0.25f;
      plot_size.y = avail.y;
      cols = 4;
    }

    if (cols == 2) {
      // In 2x2 layout, vertically stack charts with the same x-axis.
      CountsGui(model, data, plot_size);
      advance();
      profiler->DimensionsGraph(plot_size);
      advance();
      ConvergenceGui(model, data, plot_size);
      advance();
      profiler->CpuTimeGraph(plot_size);
    } else {
      CountsGui(model, data, plot_size);
      advance();
      ConvergenceGui(model, data, plot_size);
      advance();
      profiler->DimensionsGraph(plot_size);
      advance();
      profiler->CpuTimeGraph(plot_size);
    }
  }
}

}  // namespace mujoco::studio
