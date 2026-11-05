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

// The web viewer's local (browser-drawn) UI: the role bar and the
// DISCONNECTED notices. Deliberately decoupled from the app: it reads a
// SessionView snapshot and reports user intent through the SessionActions
// interface (both defined in web_client_session.h), so drawing depends
// only on ImGui and the Studio theme enum.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_LOCAL_UI_H_
#define MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_LOCAL_UI_H_

#include <imgui.h>
#include "studio/ux/theme.h"
#include "role_transitions.h"
#include "web_client_session.h"

namespace mujoco::studio {

// The screen-centered DISCONNECTED notices: a big red banner over white
// explanation lines, shown for deliberate server closes (session full,
// inactivity, the viewer stopped with an error) and for server silence. The
// two never show at the same time: a close code suppresses the silence notice.
class DisconnectNotice {
 public:
  // server_close_reason is the close frame's reason text (shown for
  // kWsCloseViewerError); seconds_since_last_payload is negative before the
  // first payload. The notices centre in the canvas above bottom_inset (the
  // role bar's height), not in the whole window.
  void Draw(int server_close_code, const char* server_close_reason,
            double seconds_since_last_payload, bool is_downloading,
            float bottom_inset);

 private:
  // How long the state stream (~60Hz while the Python side is alive) may go
  // silent before the notice appears. A model-change restart resumes
  // traffic in about a second and should not flash the notice.
  static constexpr double kServerSilenceNoticeSec = 3.0;

  // True while the server-not-reachable notice is up (log once per outage).
  bool logged_ = false;
};

// The role bar: a full-width strip along the bottom of the canvas, the
// height of Studio's status bar; the scene and the streamed UI are laid out
// above it (see BarHeight). The role chip (dot + word) sits flush left,
// followed by the role's own token (the spectator's camera mode, or the
// controller's spectator count and limit) and the download rate; the
// action buttons form one right-aligned cluster, where the controller also
// sees how many are waiting, next to Release control. A queued spectator's
// only inline cue is the amber dot; the queue details are in the buttons'
// tooltips. Role transitions raise a short toast at the top of the canvas.
// The cluster ends with a theme picker for the local UI's theme; the bar
// only shows and changes it, the app applies and stores it. Owns its
// presentation state: the transition detector, the toast timer, the
// measured cluster width, and the spectator-limit edit grace.
class RoleWindow {
 public:
  void Draw(const SessionView& view, SessionActions& actions, GuiTheme* theme);

  // The bar's height in logical pixels for the current DPI and font scale,
  // floored to whole pixels so the scene rect above it meets it exactly.
  // Valid outside an ImGui frame; call it each frame, the font scale can
  // change at any time.
  float BarHeight() const;

 private:
  static constexpr double kToastSec = 2.5;
  // State-stream silence before the rate token turns into a stall warning
  // (matches DisconnectNotice::kServerSilenceNoticeSec).
  static constexpr double kStallAfterSec = 3.0;
  // Max spectators edit grace: the local value wins over the roster until
  // the edit is this old (see the InputInt).
  static constexpr double kEditGraceSec = 1.5;

  void UpdateTransitions(const SessionView& view, double now);
  void DrawBar(const SessionView& view, SessionActions& actions,
               GuiTheme* theme);
  void DrawDownloading(const SessionView& view);
  void DrawRoleChip(const SessionView& view);
  void DrawCameraToken(const SessionView& view, SessionActions& actions);
  void DrawCapacityToken(const SessionView& view, SessionActions& actions);
  void DrawActions(const SessionView& view, SessionActions& actions,
                   GuiTheme* theme);
  void DrawThemeButton(GuiTheme* theme);
  void DrawDataRate(const SessionView& view);
  void DrawFpsCapToken(const SessionView& view, SessionActions& actions);
  void DrawToast(const SessionView& view, double now);

  // The theme being drawn this frame.
  GuiTheme theme_ = GuiTheme::kDark;
  // Role changes and the toast each one deserves; a button click registers
  // the wording it expects (see RoleTransitions).
  RoleTransitions transitions_;
  const char* toast_text_ = nullptr;
  double toast_at_ = -1e9;
  // The right-aligned cluster is measured as it draws; last frame's width
  // places it this frame (the usual ImGui right-alignment trick).
  float cluster_width_ = 0.0f;
  int max_spectators_edit_ = -1;
  double max_spectators_edit_time_ = -1e9;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_WEB_CLIENT_LOCAL_UI_H_
