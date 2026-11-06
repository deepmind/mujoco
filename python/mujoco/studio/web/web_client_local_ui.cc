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

#include "web_client_local_ui.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include <imgui.h>
#include "studio/ux/imgui_widgets.h"
#include "logging.h"
#include "web_client_session.h"

namespace mujoco::studio {
namespace {

// Semantic colours: grey = normal, green = you control, amber = attention,
// red = destructive to someone else. The light theme gets darker accents:
// the dark theme's read as pastel on its near-white background.
struct Palette {
  ImVec4 neutral;
  ImVec4 controlling;
  ImVec4 attention;
  ImVec4 danger;
};
const Palette kDarkPalette = {
    ImVec4(0.62f, 0.62f, 0.62f, 1.0f), ImVec4(0.30f, 0.85f, 0.40f, 1.0f),
    ImVec4(1.00f, 0.68f, 0.15f, 1.0f), ImVec4(0.90f, 0.30f, 0.30f, 1.0f)};
const Palette kLightPalette = {
    ImVec4(0.45f, 0.47f, 0.50f, 1.0f), ImVec4(0.12f, 0.55f, 0.25f, 1.0f),
    ImVec4(0.78f, 0.46f, 0.00f, 1.0f), ImVec4(0.75f, 0.15f, 0.15f, 1.0f)};

const Palette& PaletteFor(GuiTheme theme) {
  return theme == GuiTheme::kLight ? kLightPalette : kDarkPalette;
}

// The widest button label; every bar button is sized to it so the cluster
// keeps its width across roles.
constexpr char kButtonWidthText[] = "Request control";
// Space between Steal control and the polite action next to it.
constexpr float kStealGap = 8.0f;

// Draws one DISCONNECTED window centred in the canvas above bottom_inset;
// empty lines are skipped.
void DrawDisconnectWindow(const char* window_id,
                          std::initializer_list<const char*> lines,
                          float bottom_inset) {
  const ImGuiIO& io = ImGui::GetIO();
  ImGui::SetNextWindowPos(
      ImVec2(io.DisplaySize.x * 0.5f, (io.DisplaySize.y - bottom_inset) * 0.5f),
      ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::Begin(window_id, nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize);
  CenteredBanner("DISCONNECTED", ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
  for (const char* line : lines) {
    if (line[0] != '\0') CenteredLine(line, nullptr);
  }
  ImGui::End();
}

// A coloured status dot, vertically centred on the current frame line.
void StatusDot(const ImVec4& color) {
  constexpr float kRadius = 4.0f;
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const float h = ImGui::GetFrameHeight();
  ImGui::GetWindowDrawList()->AddCircleFilled(
      ImVec2(pos.x + kRadius + 2.0f, pos.y + h * 0.5f), kRadius,
      ImGui::GetColorU32(color));
  ImGui::Dummy(ImVec2(kRadius * 2.0f + 4.0f, h));
}

// The " · " between bar tokens.
void DotSeparator() {
  ImGui::SameLine(0.0f, 6.0f);
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("\xC2\xB7");
  ImGui::SameLine(0.0f, 6.0f);
}

// A button tinted with color (nullptr for the default button colours).
bool TintedButton(const char* label, const ImVec4* color, const ImVec2& size) {
  if (color != nullptr) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImVec4(color->x, color->y, color->z, 0.35f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4(color->x, color->y, color->z, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4(color->x, color->y, color->z, 0.8f));
  }
  const bool clicked = ImGui::Button(label, size);
  if (color != nullptr) {
    ImGui::PopStyleColor(3);
  }
  return clicked;
}

}  // namespace

void DisconnectNotice::Draw(int server_close_code,
                            const char* server_close_reason,
                            double seconds_since_last_payload,
                            bool is_downloading, float bottom_inset) {
  if (server_close_code != 0) {
    const char* what = "Disconnected by the viewer.";
    const char* detail = "";
    if (server_close_code == kWsCloseSessionFull) {
      what = "Session is full: too many viewers connected.";
    } else if (server_close_code == kWsCloseInactive) {
      what = "Disconnected after inactivity.";
    } else if (server_close_code == kWsCloseViewerError) {
      what = "The viewer stopped with an error (see terminal for traceback):";
      detail = server_close_reason;  // E.g. "RuntimeError: broken handler".
    }
    DrawDisconnectWindow("##disconnected_by_server",
                         {what, detail, "Retrying; reconnects automatically."},
                         bottom_inset);
  }

  // The silence notice keys on state-stream staleness rather than socket
  // state: a killed, suspended (Ctrl+Z), or unreachable server all go
  // silent, but only a killed one closes its sockets. Clears itself when
  // traffic resumes. Suppressed before the first payload (negative
  // staleness), after a deliberate server-side close (its own notice
  // above), and while a new model is downloading.
  const bool link_stale =
      seconds_since_last_payload > kServerSilenceNoticeSec &&
      server_close_code == 0 && !is_downloading;
  if (!link_stale) {
    logged_ = false;
    return;
  }
  if (!logged_) {
    LOG(Info, "Showing server-not-reachable notice");
    logged_ = true;
  }
  DrawDisconnectWindow(
      "##disconnected",
      {"Viewer server is not reachable: the Python script may",
       "have stopped. This page reconnects automatically if the",
       "viewer comes back."},
      bottom_inset);
}

void RoleWindow::Draw(const SessionView& view, SessionActions& actions,
                      GuiTheme* theme) {
  const double now = ImGui::GetTime();
  theme_ = *theme;
  UpdateTransitions(view, now);
  DrawBar(view, actions, theme);
  DrawToast(view, now);
}

// Hands the role to the transition detector and starts the toast it picks.
void RoleWindow::UpdateTransitions(const SessionView& view, double now) {
  if (const char* toast = transitions_.Update(view.role, now)) {
    toast_text_ = toast;
    toast_at_ = now;
  }
}

float RoleWindow::BarHeight() const {
  const float scale =
      ImGui::GetWindowDpiScale() * ImGui::GetIO().FontGlobalScale;
  return std::floor(kStatusBarHeight * scale);
}

void RoleWindow::DrawBar(const SessionView& view, SessionActions& actions,
                         GuiTheme* theme) {
  const ImGuiIO& io = ImGui::GetIO();
  const float bar_h = BarHeight();
  ImGui::SetNextWindowPos(ImVec2(0.0f, io.DisplaySize.y - bar_h),
                          ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, bar_h), ImGuiCond_Always);

  // Push FramePadding first: GetFrameHeight() depends on it, and the
  // vertical window padding must centre the frames actually drawn.
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 2.0f));
  const float border = 1.0f;
  const float vpad = std::floor(
      std::max(0.0f, (bar_h - 2.0f * border - ImGui::GetFrameHeight()) * 0.5f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, border);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1.0f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, vpad));

  // Opaque whatever the theme's window alpha: the scene is rendered right
  // up to the bar (and, at a fractional DPI scale, a pixel under it) and
  // must not show through.
  ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
  bg.w = 1.0f;
  ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);

  ImGui::Begin("##RoleBar", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                   ImGuiWindowFlags_NoDocking |
                   ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);
  if (view.is_downloading) {
    DrawDownloading(view);
  } else {
    DrawRoleChip(view);
    if (view.role == SessionRole::kSpectating) {
      DotSeparator();
      DrawCameraToken(view, actions);
    } else if (view.role == SessionRole::kControlling) {
      DotSeparator();
      DrawCapacityToken(view, actions);
    }
    DotSeparator();
    DrawDataRate(view);
    DotSeparator();
    DrawFpsCapToken(view, actions);

    // The right-aligned cluster of buttons: placed with last frame's width,
    // measured again as it draws. Never starts left of the chip's tokens.
    const float right_edge =
        ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x;
    const float start_x =
        std::max(ImGui::GetCursorPosX() + 14.0f, right_edge - cluster_width_);
    ImGui::SameLine(start_x, 0.0f);
    const float origin = ImGui::GetCursorScreenPos().x;
    DrawActions(view, actions, theme);
    cluster_width_ = ImGui::GetItemRectMax().x - origin;
  }
  ImGui::End();

  ImGui::PopStyleColor();
  ImGui::PopStyleVar(5);
}

void RoleWindow::DrawDownloading(const SessionView& view) {
  const bool done_downloading =
      view.total_bytes > 0 && view.bytes_downloaded >= view.total_bytes;
  StatusDot(PaletteFor(theme_).neutral);
  ImGui::SameLine(0.0f, 2.0f);
  ImGui::AlignTextToFramePadding();
  if (done_downloading) {
    ImGui::TextColored(PaletteFor(theme_).neutral, "Parsing model...");
    return;
  }
  ImGui::TextUnformatted("Downloading model");
  ImGui::SameLine(0.0f, 10.0f);
  const float progress = view.total_bytes > 0
                             ? static_cast<float>(view.bytes_downloaded) /
                                   static_cast<float>(view.total_bytes)
                             : 0.0f;
  char buf[32];
  snprintf(buf, sizeof(buf), "%.0f%%", progress * 100.0f);
  ImGui::ProgressBar(progress, ImVec2(220.0f, 0.0f), buf);
  if (view.retry_count > 0) {
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(PaletteFor(theme_).attention, "Retrying chunk (%d)...",
                       view.retry_count);
  }
}

// Left: me. The dot answers "am I controlling?" (green = yes, grey =
// spectating, amber = spectating but waiting in the queue).
void RoleWindow::DrawRoleChip(const SessionView& view) {
  const char* word = "Connecting...";
  ImVec4 color = PaletteFor(theme_).neutral;
  if (view.role == SessionRole::kControlling) {
    word = "Controlling";
    color = PaletteFor(theme_).controlling;
  } else if (view.role == SessionRole::kSpectating) {
    word = "Spectating";
    if (view.queue_pos > 0) color = PaletteFor(theme_).attention;
  }
  StatusDot(color);
  ImGui::SameLine(0.0f, 2.0f);
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(color, "%s", word);
}

// The spectator's camera: a combo drawn as plain text (no arrow; the hover
// highlight is affordance enough) followed by the word "camera", so it
// reads "Controller camera". Sized to its widest option so nothing shifts
// on selection. The indices are [SpectatorCamMode].
// TODO(matijak): Use the studio camera-selection UI here in future, so
// a spectator can also pick any camera defined in the model.
void RoleWindow::DrawCameraToken(const SessionView& view,
                                 SessionActions& actions) {
  static constexpr const char* kModes[] = {"Tumble", "WASD", "Controller"};
  constexpr int kNumModes = 3;
  float widest = 0.0f;
  for (const char* mode : kModes) {
    widest = std::max(widest, ImGui::CalcTextSize(mode).x);
  }
  ImGui::SetNextItemWidth(widest + 2.0f * ImGui::GetStyle().FramePadding.x);
  const int current = std::clamp(view.camera_mode, 0, kNumModes - 1);
  if (ImGui::BeginCombo("##camera", kModes[current],
                        ImGuiComboFlags_NoArrowButton)) {
    for (int i = 0; i < kNumModes; ++i) {
      if (ImGui::Selectable(kModes[i], i == current)) {
        actions.SetCameraMode(i);
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine(0.0f, 4.0f);
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("camera");
}

// The controller's "3/[16] spectators": the count, then the limit as an
// always-visible input box. Amber once the session is full.
void RoleWindow::DrawCapacityToken(const SessionView& view,
                                   SessionActions& actions) {
  const int spectators = std::max(0, view.viewers - 1);
  // Local edits win for a grace period (the roster round trip takes a
  // few frames and typing takes longer); afterwards the server's value
  // is the truth.
  if (max_spectators_edit_ < 0 ||
      ImGui::GetTime() - max_spectators_edit_time_ > kEditGraceSec) {
    max_spectators_edit_ = view.max_spectators;
  }
  const bool full = spectators >= max_spectators_edit_;
  const ImVec4 color = full ? PaletteFor(theme_).attention
                            : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);

  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(color, "%d/", spectators);
  ImGui::SameLine(0.0f, 2.0f);

  // Two digits (the limit is clamped to 32) plus frame padding.
  ImGui::SetNextItemWidth(ImGui::CalcTextSize("00").x +
                          2.0f * ImGui::GetStyle().FramePadding.x + 2.0f);
  if (ImGui::InputInt("##max_spectators", &max_spectators_edit_, 0, 0,
                      ImGuiInputTextFlags_AutoSelectAll)) {
    max_spectators_edit_time_ = ImGui::GetTime();
    max_spectators_edit_ = std::clamp(max_spectators_edit_, 0, 32);
    actions.SetMaxSpectators(max_spectators_edit_);
  }
  if (ImGui::IsItemActive()) {
    max_spectators_edit_time_ = ImGui::GetTime();
  }
  ImGui::SetItemTooltip(
      "Maximum number of spectators; only you can change it.");

  ImGui::SameLine(0.0f, 4.0f);
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(color, "spectators");
}

void RoleWindow::DrawFpsCapToken(const SessionView& view,
                                 SessionActions& actions) {
  static constexpr const char* kLabels[] = {"30 FPS", "60 FPS", "120 FPS",
                                            "Uncapped"};
  static constexpr int kValues[] = {30, 60, 120, 0};
  constexpr int kNumOptions = 4;
  float widest = 0.0f;
  for (const char* label : kLabels) {
    widest = std::max(widest, ImGui::CalcTextSize(label).x);
  }
  ImGui::SetNextItemWidth(widest + 2.0f * ImGui::GetStyle().FramePadding.x);
  int current = 1;  // Default: 60 FPS
  for (int i = 0; i < kNumOptions; ++i) {
    if (view.fps_cap == kValues[i]) {
      current = i;
      break;
    }
  }
  if (ImGui::BeginCombo("##fps_cap", kLabels[current],
                        ImGuiComboFlags_NoArrowButton)) {
    for (int i = 0; i < kNumOptions; ++i) {
      if (ImGui::Selectable(kLabels[i], i == current)) {
        actions.SetFpsCap(kValues[i]);
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SetItemTooltip("Client framerate cap");
}

void RoleWindow::DrawActions(const SessionView& view, SessionActions& actions,
                             GuiTheme* theme) {
  const ImVec2 size(ImGui::CalcTextSize(kButtonWidthText).x +
                        2.0f * ImGui::GetStyle().FramePadding.x + 8.0f,
                    0.0f);
  if (view.role == SessionRole::kSpectating) {
    // The queue is told in the tooltips: the amber dot is the only inline
    // cue that this page is in it.
    char queue_line[96];
    if (TintedButton("Steal control", &PaletteFor(theme_).danger, size)) {
      transitions_.ExpectToast("You took control", ImGui::GetTime());
      actions.StealControl();
    }
    // Slower tooltip: it reads as a warning, not a label.
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip |
                             ImGuiHoveredFlags_DelayNormal)) {
      if (view.queue_pos > 0) {
        snprintf(queue_line, sizeof(queue_line), "\nYou are #%d in the queue.",
                 view.queue_pos);
      } else if (view.queue_len > 0) {
        snprintf(queue_line, sizeof(queue_line), "\n%d spectator%s waiting.",
                 view.queue_len, view.queue_len == 1 ? " is" : "s are");
      } else {
        queue_line[0] = '\0';
      }
      ImGui::SetTooltip(
          "Take control immediately\nfrom the current controller.%s",
          queue_line);
    }
    ImGui::SameLine(0.0f, kStealGap);
    if (view.queue_pos == 0) {
      if (TintedButton("Request control", nullptr, size)) {
        // A steal the server ignored is moot now.
        transitions_.ClearExpectedToast();
        actions.RequestControl();
      }
      if (view.queue_len == 0) {
        snprintf(queue_line, sizeof(queue_line), "No one is waiting.");
      } else {
        snprintf(queue_line, sizeof(queue_line),
                 "%d spectator%s already waiting.", view.queue_len,
                 view.queue_len == 1 ? " is" : "s are");
      }
      ImGui::SetItemTooltip("Join the queue for control.\n%s", queue_line);
    } else {
      if (TintedButton("Cancel request", nullptr, size)) {
        transitions_.ClearExpectedToast();
        actions.LeaveQueue();
      }
      if (view.queue_pos == 1) {
        snprintf(queue_line, sizeof(queue_line),
                 "You're next — waiting for\nthe controller to release.");
      } else {
        snprintf(queue_line, sizeof(queue_line),
                 "%d ahead of you (the controller\nand %d queued spectator%s).",
                 view.queue_pos, view.queue_pos - 1,
                 view.queue_pos - 1 == 1 ? "" : "s");
      }
      ImGui::SetItemTooltip("Leave the queue.\n%s", queue_line);
    }
  } else if (view.role == SessionRole::kControlling) {
    // The queue sits right next to the button that serves it, growing
    // leftwards (the cluster is right-aligned as a whole).
    if (view.queue_len > 0) {
      ImGui::AlignTextToFramePadding();
      ImGui::TextColored(PaletteFor(theme_).attention, "%d waiting",
                         view.queue_len);
      ImGui::SetItemTooltip("Spectators waiting for control.");
      ImGui::SameLine(0.0f, 10.0f);
    }
    if (TintedButton("Release control", nullptr, size)) {
      transitions_.ExpectToast(view.queue_len > 0
                                   ? "Control passed to the next in queue"
                                   : "You released control",
                               ImGui::GetTime());
      actions.ReleaseControl();
    }
  } else {
    ImGui::Dummy(size);  // Connecting: keep the cluster's width.
  }
  ImGui::SameLine(0.0f, kStealGap);
  DrawThemeButton(theme);
}

// A paint-brush button that drops down the list of themes (the same combo
// drawn as plain text as the camera token).
void RoleWindow::DrawThemeButton(GuiTheme* theme) {
  static constexpr GuiTheme kThemes[] = {GuiTheme::kLight, GuiTheme::kDark,
                                         GuiTheme::kClassic};
  ImGui::SetNextItemWidth(ImGui::CalcTextSize(ICON_FA_PAINT_BRUSH).x +
                          2.0f * ImGui::GetStyle().FramePadding.x);
  if (ImGui::BeginCombo("##theme", ICON_FA_PAINT_BRUSH,
                        ImGuiComboFlags_NoArrowButton)) {
    for (GuiTheme item : kThemes) {
      if (ImGui::Selectable(ThemeName(item), item == *theme)) {
        *theme = item;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SetItemTooltip("Theme: %s", ThemeName(theme_));
}

// The download rate, after the role's own token: dim, at its natural
// width, and amber only when the state stream has gone quiet.
void RoleWindow::DrawDataRate(const SessionView& view) {
  const uint64_t sim_kib = view.sim_bytes_per_sec / 1024;
  const uint64_t gui_kib = view.gui_bytes_per_sec / 1024;
  ImGui::AlignTextToFramePadding();
  if (view.seconds_since_last_payload > kStallAfterSec) {
    ImGui::TextColored(PaletteFor(theme_).attention, "%s stalled %.0fs",
                       ICON_FA_EXCLAMATION_TRIANGLE,
                       view.seconds_since_last_payload);
  } else {
    ImGui::TextDisabled("%" PRIu64 " KiB/s", sim_kib + gui_kib);
  }
  ImGui::SetItemTooltip("Sim data rate: %" PRIu64
                        " KiB/s\n"
                        "GUI data rate: %" PRIu64 " KiB/s",
                        sim_kib, gui_kib);
}

// The transition moment: a small rounded toast at the top centre of the
// canvas that drops in, holds, and fades out. Takes no input and no
// focus so it never gets in the way of the scene or the remote UI.
void RoleWindow::DrawToast(const SessionView& view, double now) {
  const double age = now - toast_at_;
  if (toast_text_ == nullptr || age > kToastSec) {
    return;
  }
  // Drop in over the first 0.18s (eased), fade out over the last 20%.
  const double drop_linear = std::min(1.0, age / 0.18);
  const float drop = static_cast<float>(1.0 - std::pow(1.0 - drop_linear, 3.0));
  const double hold_until = kToastSec * 0.8;
  const float alpha =
      age < hold_until
          ? 1.0f
          : static_cast<float>(1.0 - (age - hold_until) / (kToastSec * 0.2));
  const ImGuiIO& io = ImGui::GetIO();
  const float y = ImGui::GetFrameHeight() + 10.0f + 26.0f * drop;
  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, y), ImGuiCond_Always,
                          ImVec2(0.5f, 0.0f));
  ImGui::SetNextWindowBgAlpha(0.92f * alpha);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 6.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
  ImGui::Begin(
      "##RoleToast", nullptr,
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
          ImGuiWindowFlags_NoSavedSettings |
          ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
          ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoDocking);
  const Palette& palette = PaletteFor(theme_);
  StatusDot(view.role == SessionRole::kControlling ? palette.controlling
                                                   : palette.neutral);
  ImGui::SameLine(0.0f, 4.0f);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(toast_text_);
  ImGui::End();
  ImGui::PopStyleVar(3);
}

}  // namespace mujoco::studio
