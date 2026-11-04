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

// The page's role in the collaborative session, and the logic that turns
// role changes into the role bar's transition toasts. Kept free of ImGui
// and Emscripten so it can be unit-tested on the host.

#ifndef MUJOCO_PYTHON_STUDIO_WEB_ROLE_TRANSITIONS_H_
#define MUJOCO_PYTHON_STUDIO_WEB_ROLE_TRANSITIONS_H_

namespace mujoco::studio {

// The page's role in the collaborative session. Every page starts by
// claiming the controller slot; the claim either succeeds (kControlling)
// or the page settles into spectating. A control grant puts a spectator
// back into kClaiming while it reconnects to /ui.
enum class SessionRole {
  kClaiming = 0,  // /ui claim in flight; the role is not yet resolved.
  kControlling,   // This page holds the open /ui connection.
  kSpectating,    // Another page controls; scene + local role window only.
};

// Detects role transitions and picks the toast to show for each one.
//
// kClaiming is transient (a grant or a steal passes through it on the way
// to kControlling), so only settled roles count, and the very first settled
// role (page load) is not a transition. A click on one of the page's own
// buttons registers the wording it expects (ExpectToast); if the role then
// flips, that wording is used, otherwise the flip is someone else's doing
// (a queue grant, or another viewer stealing control) and a generic toast
// is picked. An expected toast that the server never acts on expires after
// kExpectSec, so a much later grant is not mislabelled as the page's own
// doing.
class RoleTransitions {
 public:
  // How long a registered expectation stays valid without the role flipping.
  static constexpr double kExpectSec = 5.0;

  // Registers the toast a local click wants once the role actually flips.
  void ExpectToast(const char* text, double now);
  // Forgets a registered expectation (a click that makes an earlier one moot).
  void ClearExpectedToast();

  // Feeds the current role. Returns the toast text if this call completed a
  // transition, nullptr otherwise. The returned pointer is a string literal
  // (or the one passed to ExpectToast), valid for the program's lifetime.
  const char* Update(SessionRole role, double now);

  SessionRole settled_role() const { return settled_role_; }

 private:
  // The last settled role (kClaiming is never recorded; it doubles as "no
  // role seen yet").
  SessionRole settled_role_ = SessionRole::kClaiming;
  const char* expected_toast_ = nullptr;
  double expected_at_ = -1e9;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_STUDIO_WEB_ROLE_TRANSITIONS_H_
